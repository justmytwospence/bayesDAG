"""Fitted-structure panels: the classic pictures of hierarchies, regressions and shrinkage priors.

The panels read only posterior draws and the model's data, so these tests feed them prior draws
relabelled as a posterior (no sampler needed) and check the numbers against direct formulas.
"""

import numpy as np
import pymc as pm
import pytest
import xarray as xr
from structures import STRUCTURE_MODELS

import bayesdag
from bayesdag.adapters import fitted
from bayesdag.convert import to_ir


def _as_posterior(model, draws=150):
    with model:
        pri = pm.sample_prior_predictive(draws=draws, random_seed=3)
    return xr.DataTree.from_dict(
        {"posterior": pri["prior"].to_dataset(), "observed_data": pri["observed_data"].to_dataset()}
    )


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
