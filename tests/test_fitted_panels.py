"""Fitted-structure panels: the classic pictures of hierarchies, regressions and shrinkage priors.

The panels read only saved draws and fit-time data, so these tests feed them prior draws
relabelled as a posterior (no sampler needed) and check the numbers against direct formulas.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr
from structures import STRUCTURE_MODELS

import bayesdag
from bayesdag.adapters import fitted
from bayesdag.convert import to_ir


def _as_posterior(model, draws=150):
    with model:
        pri = pm.sample_prior_predictive(draws=draws, random_seed=3)
    groups = {
        "posterior": pri["prior"].to_dataset(),
        "observed_data": pri["observed_data"].to_dataset(),
    }
    if "constant_data" in pri:
        groups["constant_data"] = pri["constant_data"].to_dataset()
    return xr.DataTree.from_dict(groups)


def _setup(name):
    m = STRUCTURE_MODELS[name]()
    ir = to_ir(m)
    return m, ir, _as_posterior(m)


def _d(kind, m, ir, idata, block_id):
    return fitted.compute(kind, m, ir, idata, ir.block(block_id))


def test_eight_schools_raw_estimates_are_the_data():
    """y_obs ~ N(theta, sigma): nothing else is in the predictor, so each school's raw estimate
    is its own y and its standard error is sigma."""
    m, ir, idata = _setup("eight_schools_noncentered")
    d = _d("shrinkage", m, ir, idata, "block:hierarchy:theta")
    np.testing.assert_allclose(d["raw"], [28, 8, -3, 7, -1, 1, 18, 12])
    np.testing.assert_allclose(d["se"], [15, 10, 16, 11, 9, 11, 10, 18])
    assert d["x_kind"] == "se"
    theta = idata["posterior"]["theta"].values.reshape(-1, 8)
    np.testing.assert_allclose(d["post"], theta.mean(axis=0))
    assert d["pop"] == pytest.approx(float(idata["posterior"]["mu"].mean()))


def test_raw_estimates_subtract_the_rest_of_the_predictor():
    """Radon: y ~ N(a[county] + b*floor, sigma). The raw estimate of a_j is the county mean of
    y - E[b]*floor — the no-pooling estimate with the slope held at the fit."""
    m, ir, idata = _setup("radon_noncentered")
    d = _d("shrinkage", m, ir, idata, "block:hierarchy:a")
    y = np.asarray(m.rvs_to_values[m["y"]].eval())
    x = m["floor"].get_value()
    idx = m["county_idx"].get_value()
    b = float(idata["posterior"]["b"].mean())
    resid = y - b * x
    want = [resid[idx == j].mean() for j in range(8) if (idx == j).any()]
    np.testing.assert_allclose(d["raw"], want, rtol=1e-6)
    assert d["x_kind"] == "n"


def test_fit_uses_saved_group_codes_after_mutation_and_refuses_missing_snapshot():
    m, ir, idata = _setup("radon_noncentered")
    before = _d("shrinkage", m, ir, idata, "block:hierarchy:a")
    with m:
        pm.set_data({"county_idx": np.zeros_like(m["county_idx"].get_value())})
    after = _d("shrinkage", m, ir, idata, "block:hierarchy:a")
    np.testing.assert_allclose(after["raw"], before["raw"])
    del idata["constant_data"]
    with pytest.raises(fitted.Unavailable, match="saved constant_data"):
        fitted.Fit(m, ir, idata).value("county_idx")


def test_categorical_codes_keep_their_fit_time_coordinates_after_model_reordering():
    with pm.Model(coords={"group": ["A", "B"], "row": range(4)}) as m:
        pm.Data("group_data", [10, 20], dims="group")
        idx = pm.Data("idx", [0, 0, 1, 1], dims="row")
        tau = pm.HalfNormal("tau")
        a = pm.Normal("a", 0, tau, dims="group")
        pm.Normal("y", a[idx], 1, observed=[4, 8, 8, 10], dims="row")
    idata = _as_posterior(m, draws=5)
    post = idata["posterior"]["a"].values.mean(axis=(0, 1))
    m.set_dim("group", 2, coord_values=["B", "A"])
    ir = to_ir(m)
    assert list(ir.node("a").coords["group"]) == ["B", "A"]
    d = _d("shrinkage", m, ir, idata, "block:hierarchy:a")
    assert d["labels"] == ["A", "B"]
    np.testing.assert_allclose(d["raw"], [6, 9])
    np.testing.assert_allclose(d["post"], post)
    fit = fitted.Fit(m, ir, idata)
    pred = fitted.predictor(fit, "y")
    np.testing.assert_allclose(pred.evaluate(fit, pred.means(fit)), post[[0, 0, 1, 1]])

    conflicting = idata.copy()
    conflicting["posterior"] = idata["posterior"].to_dataset().isel(group=[1, 0])
    with pytest.raises(fitted.Unavailable, match="conflicting fit-time group coordinates"):
        _d("shrinkage", m, ir, conflicting, "block:hierarchy:a")

    idata["posterior"] = idata["posterior"].to_dataset().drop_vars("group")
    with pytest.raises(fitted.Unavailable, match="fit-time group coordinates"):
        _d("shrinkage", m, ir, idata, "block:hierarchy:a")


def test_constant_data_rows_are_aligned_for_group_summaries_and_effect_points():
    m, ir, idata = _setup("radon_noncentered")
    before = _d("shrinkage", m, ir, idata, "block:hierarchy:a")
    # Exercise fitted observation labels rather than the live model coordinates or the IR.
    for name in ("y", "floor", "county_idx"):
        node = ir.node(name)
        if node is not None:
            node.coords = None
    idata["constant_data"] = idata["constant_data"].to_dataset().isel(obs=slice(None, None, -1))
    after = _d("shrinkage", m, ir, idata, "block:hierarchy:a")
    np.testing.assert_allclose(after["raw"], before["raw"])
    effect = _d("effect", m, ir, idata, "block:regression:y")
    np.testing.assert_array_equal(effect["pts_x"], m["floor"].get_value())
    np.testing.assert_array_equal(effect["pts_y"], idata["observed_data"]["y"].values)


@pytest.mark.parametrize(
    "x_dim,y_dim,saved_labels,reason",
    [
        ("row", "row", [0, 1, 9], "incompatible coordinate labels"),
        ("row", "row", [0, 0, 2], "duplicate labels"),
        ("other", "row", None, "incompatible axis names"),
        (None, "row", None, "labelled and positional"),
        ("row", None, None, "labelled and positional"),
    ],
)
def test_predictor_refuses_incompatible_rows(x_dim, y_dim, saved_labels, reason):
    # Even identical integer labels count as positional only on an autogenerated axis.
    with pm.Model(coords={"row": range(3), "other": range(3)}) as m:
        x = pm.Data("x", [1.0, 4.0, 2.0], dims=x_dim)
        b = pm.Normal("b")
        pm.Normal("y", b * x, 1, observed=[3.0, 8.0, 5.0], dims=y_dim)
    idata = _as_posterior(m, draws=5)
    if saved_labels is not None:
        idata["constant_data"] = idata["constant_data"].to_dataset().assign_coords(row=saved_labels)
    with pytest.raises(fitted.Unavailable, match=reason):
        fitted.predictor(fitted.Fit(m, to_ir(m), idata), "y")


def test_predictor_accepts_positional_rows_only_for_a_direct_input_of_the_right_length():
    with pm.Model() as m:
        x = pm.Data("x", [1.0, 4.0, 2.0])
        b = pm.Normal("b")
        pm.Normal("y", b * x, 1, observed=[3.0, 8.0, 5.0])
        pm.Normal("y_reversed", b * x[::-1], 1, observed=[3.0, 8.0, 5.0])
    idata = _as_posterior(m, draws=5)
    ir = to_ir(m)
    fit = fitted.Fit(m, ir, idata)
    pred = fitted.predictor(fit, "y")
    np.testing.assert_allclose(pred.evaluate(fit, {"b": 2}), [2, 8, 4])
    with pytest.raises(fitted.Unavailable, match="not a direct row-wise input"):
        fitted.predictor(fit, "y_reversed")
    idata["constant_data"] = idata["constant_data"].to_dataset().isel(x_dim_0=slice(2))
    with pytest.raises(fitted.Unavailable, match="incompatible positional lengths"):
        fitted.predictor(fitted.Fit(m, ir, idata), "y")


def test_predictor_does_not_probe_anonymous_live_shared_data():
    import pytensor

    with pm.Model() as m:
        b = pm.Normal("b")
        x = pytensor.shared(np.array([1.0, 4.0, 2.0]))
        pm.Normal("y", b * x, 1, observed=[3.0, 8.0, 5.0])
    fit = fitted.Fit(m, to_ir(m), _as_posterior(m, draws=5))
    with pytest.raises(fitted.Unavailable, match="unbound shared value"):
        fitted.predictor(fit, "y")


def test_consumer_discovery_does_not_read_unrelated_likelihood_data():
    with pm.Model(coords={"group": ["a", "b"], "other_row": range(3)}) as m:
        other_x = pm.Data("other_x", [4.0, 2.0, 1.0], dims="other_row")
        other_b = pm.Normal("other_b")
        pm.Normal("other_y", other_b * other_x, 1, observed=[1, 2, 3], dims="other_row")
        a = pm.Normal("a", dims="group")
        pm.Normal("y", a, 1, observed=[5, 8], dims="group")
    idata = _as_posterior(m, draws=5)
    del idata["constant_data"]
    from bayesdag.ir import BlockIR

    b = BlockIR(id="test", kind="hierarchy", title="test", anchor="a")
    d = fitted.shrinkage(fitted.Fit(m, to_ir(m), idata), b)
    np.testing.assert_allclose(d["raw"], [5, 8])


def test_posterior_coordinates_are_aligned_to_model_order():
    m, ir, idata = _setup("eight_schools_noncentered")
    fit = fitted.Fit(m, ir, idata)
    original = fit.draws("theta", None)
    da = idata["posterior"]["theta"]
    axis = da.dims[-1]
    reversed_idata = idata.copy(deep=True)
    reversed_idata["posterior"] = (
        idata["posterior"].to_dataset().isel({axis: slice(None, None, -1)})
    )
    np.testing.assert_allclose(fitted.Fit(m, ir, reversed_idata).draws("theta", None), original)


@pytest.mark.parametrize("literal,coefficient", [(1.0, 1.0), (-1.0, -1.0), (2.0, 2.0), (2.9, 2)])
def test_shrinkage_respects_group_coefficient(literal, coefficient):
    factor = pt.cast(pt.constant(literal), "int64") if literal != coefficient else literal
    with pm.Model(coords={"group": ["a", "b"], "row": range(4)}) as m:
        idx = pm.Data("idx", [0, 0, 1, 1], dims="row")
        a = pm.Normal("a", 0, 1, dims="group")
        pm.Normal("y", factor * a[idx], 2, observed=[4, 8, 8, 10], dims="row")
    ir = to_ir(m)
    idata = _as_posterior(m)
    from bayesdag.ir import BlockIR

    b = BlockIR(id="test", kind="hierarchy", title="test", anchor="a")
    result = fitted.shrinkage(fitted.Fit(m, ir, idata), b)
    np.testing.assert_allclose(result["raw"], np.array([6, 9]) / coefficient)
    np.testing.assert_allclose(result["se"], np.sqrt(2) / abs(coefficient))


def test_shrinkage_refuses_repeated_group_effect():
    with pm.Model(coords={"group": ["a", "b"], "row": range(4)}) as m:
        idx = pm.Data("idx", [0, 0, 1, 1], dims="row")
        a = pm.Normal("a", 0, 1, dims="group")
        pm.Normal("y", a[idx] + a[idx], 2, observed=[4, 8, 8, 10], dims="row")
    ir = to_ir(m)
    from bayesdag.ir import BlockIR

    b = BlockIR(id="test", kind="hierarchy", title="test", anchor="a")
    with pytest.raises(fitted.Unavailable, match="ambiguous"):
        fitted.shrinkage(fitted.Fit(m, ir, _as_posterior(m)), b)


def test_pooling_factor_matches_its_formula():
    m, ir, idata = _setup("radon_noncentered")
    d = _d("pooling", m, ir, idata, "block:hierarchy:a")
    a = idata["posterior"]["a"].values.reshape(-1, 8)
    mu = idata["posterior"]["mu_a"].values.reshape(-1, 1)
    eps = a - mu
    want = 1 - np.var(eps.mean(axis=0)) / np.mean(np.var(eps, axis=1))
    assert d["lambda"] == pytest.approx(want)
    assert len(d["groups"]["pull"]) == 8


def test_population_curves_are_the_implied_family():
    m, ir, idata = _setup("eight_schools_noncentered")
    d = _d("population", m, ir, idata, "block:hierarchy:theta")
    assert d["family"] == "Normal" and d["n_groups"] == 8
    assert len(d["curves"]) == 40 and len(d["predictive"]) == len(d["xs"])
    assert max(d["predictive"]) <= 1.0 + 1e-9


def test_noncentered_population_uses_the_saved_auxiliary_scale():
    with pm.Model(coords={"group": ["a", "b", "c"]}) as m:
        aux_sd = pm.Data("aux_sd", 2.0)
        z = pm.Normal("z", 0, aux_sd, dims="group")
        tau = pm.HalfNormal("tau", 1)
        a = pm.Deterministic("a", tau * z, dims="group")
        pm.Normal("y", a, 1, observed=[1, 2, 3], dims="group")
    ir = to_ir(m)
    idata = _as_posterior(m, draws=50)
    before = _d("population", m, ir, idata, "block:hierarchy:a")
    with m:
        pm.set_data({"aux_sd": 9.0})
    after = _d("population", m, ir, idata, "block:hierarchy:a")
    assert before == after


def test_log_normal_population_for_a_log_scale_hierarchy():
    m, ir, idata = _setup("bridge_measurement_like")
    d = _d("population", m, ir, idata, "block:hierarchy:tube_spread")
    assert d["family"] == "LogNormal"
    assert d["rug"] and min(d["rug"]) > 0  # the effects themselves are positive spreads
    assert all(np.isfinite(d["predictive"]))


def test_effect_line_is_the_average_over_observed_rows():
    """Identity link: the curve at x = g is the row average of a[county] + b*g."""
    m, ir, idata = _setup("radon_noncentered")
    d = _d("effect", m, ir, idata, "block:regression:y")
    idx = m["county_idx"].get_value()
    a = idata["posterior"]["a"].values.reshape(-1, 8)
    b = idata["posterior"]["b"].values.reshape(-1)
    keep = np.unique(np.linspace(0, a.shape[0] - 1, 100).round().astype(int))  # the same thinning
    for i in (0, -1):
        g = d["grid"][i]
        want = np.median((a[keep][:, idx] + b[keep, None] * g).mean(axis=1))
        assert d["mid"][i] == pytest.approx(want)


@pytest.mark.parametrize("scale", ["fixed", "fixed_expression", "posterior", "fixed_independent"])
def test_log_normal_effect_omits_mean_only_when_sigma_depends_on_covariate(scale):
    with pm.Model(coords={"row": range(3)}) as m:
        x = pm.Data("x", [0.5, 1.0, 2.0], dims="row")
        b = pm.Normal("b")
        if scale == "fixed":
            sigma = x
        elif scale == "fixed_expression":
            sigma = x**2
        elif scale == "posterior":
            sigma = pm.Deterministic("sigma", pm.HalfNormal("s") * x.mean())
        else:
            sigma = 1.0
        pm.LogNormal("y", b * x, sigma, observed=[1, 3, 2], dims="row")
    idata = _as_posterior(m, draws=5)
    d = _d("effect", m, to_ir(m), idata, "block:regression:y")
    draws = idata["posterior"]["b"].values.reshape(-1, 1)
    med = np.exp(draws * np.asarray(d["grid"]))
    np.testing.assert_allclose(d["mid"], np.median(med, axis=0))
    if scale == "fixed_independent":
        np.testing.assert_allclose(d["mean"], np.median(med * np.exp(0.5), axis=0))
        assert d["note"] is None
    else:
        assert d["mean"] is None
        assert d["note"] == "mean curve omitted: sigma itself depends on this covariate"


def test_group_lines_and_intercept_slope():
    m, ir, idata = _setup("radon_centered")
    d = _d("group_lines", m, ir, idata, "block:regression:y")
    assert 1 <= len(d["groups"]) <= 8 and d["x"] == "floor"
    s = _d("intercept_slope", m, ir, idata, "block:regression:y")
    assert s["a"] == "a" and s["b"] == "b" and len(s["ma"]) == 8


def test_kappa_matches_its_formula_and_stays_in_the_unit_interval():
    m, ir, idata = _setup("horseshoe")
    d = _d("kappa", m, ir, idata, "block:shrinkage:beta")
    tau = idata["posterior"]["tau"].values.reshape(-1)
    lam = idata["posterior"]["lam"].values.reshape(tau.size, -1)
    sig = idata["posterior"]["sigma"].values.reshape(-1)
    X = m["X"].get_value()
    k = 1 / (1 + (tau[:, None] * lam) ** 2 * (X**2).sum(axis=0) / sig[:, None] ** 2)
    np.testing.assert_allclose(d["mean"], k[: len(k)].mean(axis=0), rtol=0.2)
    assert all(0 <= v <= 1 for v in d["mean"])


@pytest.mark.parametrize("cast", [False, True])
def test_kappa_uses_effective_design_multiplier(cast):
    factor = pt.cast(pt.constant(2.9), "int64") if cast else 2
    with pm.Model(coords={"row": range(4), "coef": ["x", "z"]}) as m:
        X = pm.Data("X", [[1, 1], [1, -1], [-1, 1], [-1, -1]], dims=("row", "coef"))
        tau = pm.HalfCauchy("tau", 1)
        lam = pm.HalfCauchy("lam", 1, dims="coef")
        beta = pm.Normal("beta", 0, tau * lam, dims="coef")
        pm.Normal("y", X @ (factor * beta), 1, observed=[0, 0, 0, 0], dims="row")
    ir = to_ir(m)
    idata = _as_posterior(m)
    for name in ("tau", "lam"):
        idata["posterior"][name].values[:] = 1
    result = _d("kappa", m, ir, idata, "block:shrinkage:beta")
    np.testing.assert_allclose(result["mean"], [1 / 17, 1 / 17])
    idata["constant_data"] = (
        idata["constant_data"].to_dataset().isel(row=[2, 0, 3, 1]).transpose("coef", "row")
    )
    pred = fitted.predictor(fitted.Fit(m, ir, idata), "y")
    np.testing.assert_array_equal(pred.data["X"], m["X"].get_value())


def test_local_scale_marks_the_slab():
    m, ir, idata = _setup("regularized_horseshoe")
    d = _d("local_scale", m, ir, idata, "block:shrinkage:beta")
    c = np.median(np.sqrt(idata["posterior"]["c2"].values))
    assert d["c"] == pytest.approx(c)
    assert len(d["s"]) == 20 and all(s > 0 for s in d["s"])


def test_measured_readings_line_up():
    m, ir, idata = _setup("measurement_error")
    d = _d("measured", m, ir, idata, "block:measured:x_true")
    np.testing.assert_allclose(d["obs"], np.asarray(m.rvs_to_values[m["x_obs"]].eval()))


def test_measured_posterior_is_paired_with_saved_reading_labels():
    m, ir, idata = _setup("measurement_error")
    before = _d("measured", m, ir, idata, "block:measured:x_true")
    ir.node("x_true").coords = None
    ir.node("x_obs").coords = None
    ds = idata["observed_data"].to_dataset().isel(obs=slice(None, None, -1))
    idata["observed_data"] = ds
    after = _d("measured", m, ir, idata, "block:measured:x_true")
    for key in ("obs", "post", "lo", "hi"):
        np.testing.assert_allclose(after[key], before[key][::-1])
    idata["observed_data"] = ds.assign_coords(obs=ds.obs + 1)
    with pytest.raises(fitted.Unavailable, match="incompatible coordinate labels"):
        _d("measured", m, ir, idata, "block:measured:x_true")


def test_non_identity_links_refuse_raw_group_summaries():
    m, ir, idata = _setup("logistic_grouped")
    with pytest.raises(fitted.Unavailable, match="identity link"):
        _d("shrinkage", m, ir, idata, "block:hierarchy:group_effect")
    d = _d("effect", m, ir, idata, "block:regression:y")  # ...but the effect plot works
    assert d["binary"] and all(0 <= v <= 1 for v in d["mid"])


def test_view_offers_renders_and_explains():
    m = STRUCTURE_MODELS["radon_noncentered"]()
    v = bayesdag.view(m, ppc_draws=0)
    assert v.fitted_panels("block:hierarchy:a") == []  # nothing fitted yet
    v.update(_as_posterior(m))
    assert "shrinkage" in v.fitted_panels("block:hierarchy:a")
    svg = v.panel_svg("shrinkage", block_id="block:hierarchy:a")
    assert "shrinkage toward the population" in svg and "raw (no pooling)" in svg
    for kind in ("population", "pooling"):
        assert "<svg" in v.panel_svg(kind, block_id="block:hierarchy:a")
    for kind in ("effect", "group_lines"):
        assert "<svg" in v.panel_svg(kind, block_id="block:regression:y")
    with pytest.raises(ValueError):
        v.panel_svg("shrinkage", node_id="a")


def test_unavailable_panels_say_why_in_the_widget():
    pytest.importorskip("anywidget")
    m = STRUCTURE_MODELS["logistic_grouped"]()
    v = bayesdag.view(m, ppc_draws=0)
    v.update(_as_posterior(m))
    w = v.widget()
    blk = w.spec["blocks"]["block:hierarchy:group_effect"]
    assert {p["kind"] for p in blk["panels"]} >= {"population", "shrinkage"}
    w.panel_request = {"kind": "shrinkage", "block_id": "block:hierarchy:group_effect"}
    assert w.panel_response["status"] == "unavailable"
    assert "identity link" in w.panel_response["reason"]
    w.panel_request = {"kind": "population", "block_id": "block:hierarchy:group_effect"}
    assert w.panel_response["status"] == "ready"
    # a tag-only structure's card lives on its anchor
    reg = w.spec["nodes"]["y"]
    assert "block:regression:y" in reg["blocks_here"]


@pytest.mark.parametrize(
    "name", ["radon_centered", "eight_schools_noncentered", "horseshoe", "vo2max_like"]
)
def test_every_offered_panel_renders_or_explains(name):
    import xml.etree.ElementTree as ET

    m = STRUCTURE_MODELS[name]()
    v = bayesdag.view(m, ppc_draws=0)
    v.update(_as_posterior(m, draws=60))
    for b in v.blocks:
        for kind in v.fitted_panels(b.id):
            status, svg, reason = v._fitted(kind, b)
            assert status in ("ready", "unavailable"), (b.id, kind, reason)
            ET.fromstring(svg)
