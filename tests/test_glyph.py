"""Shape-first glyph system: data provider sources + glyph-agnostic registry rendering."""

from bayesdag import glyph
from bayesdag.ir import Box


def test_registry_has_density_and_nonunivariate_kinds():
    ks = glyph.registered_kinds()
    assert {"density", "histogram", "hist_overlay", "bars", "schematic", "heatmap", "curve"} <= ks
    assert {
        "fan",
        "pairplot",
        "mixture",
        "cutpoints",
        "simplex",
        "censored",
        "stem",
        "step",
    } <= ks  # special-construct kinds


def test_zero_reference_marker():
    """A faint x=0 line appears on value-axis glyphs only when 0 is inside the (self-scaled) range —
    restoring sign/centering for ZeroSumNormal, coefficients, correlations, etc."""
    b = Box(0, 0, 100, 40)
    centered = glyph.render("density", {"xs": [-2, -1, 0, 1, 2], "ys": [0.1, 0.6, 1, 0.6, 0.1]}, b)
    positive = glyph.render("density", {"xs": [0.1, 1, 2, 3], "ys": [1, 0.6, 0.3, 0.1]}, b)
    far = glyph.render("density", {"xs": [7, 9, 11], "ys": [0.2, 1, 0.2]}, b)
    assert 'stroke-dasharray="2,2"' in centered  # 0 in range -> marker
    assert 'stroke-dasharray="2,2"' not in positive  # 0 at/below left edge -> no marker
    assert 'stroke-dasharray="2,2"' not in far  # 0 far outside -> no marker


def test_conditional_latent_is_schematic_not_a_random_draw():
    """The prior-vs-latent distinction: a ROOT prior (all params fixed) gets an analytic shape; a
    CONDITIONAL latent (any param is a parent RV) must never be silently sampled into a misleading,
    non-deterministic density (the old bug, where ``.eval()`` drew a random value through the
    parents). A Normal whose loc/scale come from parents has an EXACT standardized shape."""
    import pymc as pm

    from bayesdag.convert import to_ir

    def build(scale=1.0):
        with pm.Model() as m:
            mu = pm.Normal("mu", 0, 5)  # root prior
            sigma = pm.HalfNormal("sigma", scale)  # root prior
            pm.Normal("x", mu, sigma)  # conditional latent: params depend on parents
        return m

    g = {n.id: n.glyph for n in to_ir(build()).nodes}
    assert g["mu"].source == "prior_analytic" and g["sigma"].source == "prior_analytic"
    assert g["x"].source == "conditional_standardized" and g["x"].kind == "schematic"
    assert to_ir(build()).node("x").glyph_data["exact"] is True
    # the schematic is a fixed, parameter-FREE shape: it must be identical regardless of the parent
    # scale (proving it's canonical, never a value sampled through the parents at render time).
    assert to_ir(build(1.0)).node("x").glyph_data == to_ir(build(100.0)).node("x").glyph_data


def test_varying_vector_prior_does_not_plot_element_zero_as_the_node():
    """An iid vector prior broadcasts ONE density across its elements, so a single analytic curve
    is honest. A varying vector prior gives each element its own density — plotting element 0 and
    badging it `prior_analytic` would state something false about the rest. Elements that differ
    only in location/scale still share an exact standardized shape. A Categorical is exempt: its
    vector param IS the pmf."""
    import numpy as np
    import pymc as pm

    from bayesdag.convert import to_ir

    with pm.Model(coords={"k": [0, 1]}) as m:
        pm.Normal("iid", 0.0, 1.0, dims="k")  # broadcast scalars
        pm.Normal("varying", mu=np.array([0.0, 5.0]), sigma=np.array([1.0, 10.0]), dims="k")
        pm.Categorical("choice", p=np.array([0.2, 0.3, 0.5]))

    g = {n.id: n.glyph for n in to_ir(m).nodes}
    assert g["iid"].source == "prior_analytic" and g["iid"].kind == "density"
    assert g["varying"].source == "conditional_standardized" and g["varying"].kind == "schematic"
    assert g["choice"].source == "prior_analytic" and g["choice"].kind == "bars"


def _post(**arrays):
    import xarray as xr

    return xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {
                    k: (("chain", "draw", *[f"{k}_{i}" for i in range(v.ndim - 2)]), v)
                    for k, v in arrays.items()
                }
            )
        }
    )


def test_discrete_posterior_is_bars_and_vectors_are_never_pooled():
    """A discrete variable's posterior is a pmf over integers (a gaussian KDE would put mass on
    values it cannot take). A vector's posterior is never ONE pooled KDE — that curve would be no
    element's posterior: a few elements get a curve each, many a forest of intervals."""
    import numpy as np

    from bayesdag.adapters.glyph_data import posterior_glyph

    rng = np.random.default_rng(0)
    k = posterior_glyph("k", "Poisson", _post(k=rng.poisson(3.0, size=(2, 200))))
    assert k[0].kind == "bars" and k[0].source == "posterior_bars"
    assert all(float(c).is_integer() for c in k[1]["cats"])

    def kind_for(n):
        spec, data = posterior_glyph("t", "Normal", _post(t=rng.normal(size=(2, 200, n))))
        return spec.kind, data

    assert kind_for(1)[0] == "density"
    kind, data = kind_for(8)
    assert kind == "multi" and len(data["curves"]) == 8
    kind, data = kind_for(50)
    assert kind == "forest" and len(data["q50"]) == 50
    assert data["q50"] == sorted(data["q50"])  # sorted by median
    kind, data = kind_for(1000)
    assert kind == "forest" and data["n"] == 1000 and data["shown"] == 400


def test_wide_discrete_draws_are_binned_on_integers_not_smoothed():
    import numpy as np

    from bayesdag.adapters.glyph_data import _discrete_bars

    wide = _discrete_bars(np.arange(0, 1000, 7))
    assert wide["binned"] and len(wide["cats"]) <= 30
    assert all(float(c).is_integer() for c in wide["cats"])
    assert _discrete_bars(np.array([0.5, 1.0])) is None  # non-integers are not rounded silently


def test_vector_panels_say_what_they_show():
    import numpy as np
    import pymc as pm

    from bayesdag.convert import to_ir
    from bayesdag.render_svg import render_node_panel

    rng = np.random.default_rng(0)
    with pm.Model(coords={"g": range(20)}) as m:
        pm.Normal("theta", 0.0, 1.0, dims="g")
    ir = to_ir(m, idata=_post(theta=rng.normal(size=(2, 100, 20))))
    panel = render_node_panel(ir.node("theta"))
    assert "20 elements sorted by median" in panel


def test_marginal_rows_keep_coordinate_order_and_page():
    import numpy as np
    import xarray as xr

    from bayesdag.adapters.glyph_data import marginal_rows

    rng = np.random.default_rng(0)
    vals = rng.normal(size=(2, 100, 30)) + np.arange(30)[::-1]  # medians DEcreasing
    idata = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {"b": (("chain", "draw", "p"), vals)}, coords={"p": [f"P{i}" for i in range(30)]}
            )
        }
    )
    p0 = marginal_rows("b", idata, page=0)
    assert p0["pages"] == 2 and len(p0["rows"]) == 24
    assert [r["label"] for r in p0["rows"][:3]] == ["P0", "P1", "P2"]  # coordinate order, not rank
    p1 = marginal_rows("b", idata, page=1)
    assert len(p1["rows"]) == 6 and p1["lo"] == p0["lo"]  # one shared axis across pages


def test_censored_panel_reports_the_real_mass_behind_the_exaggerated_spike():
    """The spikes are scaled up so a few-percent censored mass is visible next to a
    peak-normalized density. That makes the BAR a marker, not a readable probability — so the
    true value travels with the data and is captioned."""
    import pymc as pm

    from bayesdag.convert import to_ir
    from bayesdag.render_svg import render_node_panel

    with pm.Model() as m:
        pm.Censored("x", pm.Normal.dist(0.0, 1.0), lower=-1.5, upper=1.5)
    n = to_ir(m).node("x")
    ps = [sp["p"] for sp in n.glyph_data["spikes"]]
    assert all(0.0 < p < 0.10 for p in ps)  # ~6.7% per tail for N(0,1) beyond ±1.5
    assert all(sp["h"] > sp["p"] for sp in n.glyph_data["spikes"])  # drawn exaggerated
    assert "censored mass" in render_node_panel(n)


def test_mixture_components_are_weighted_only_when_the_weights_are_known():
    """A 0.9/0.1 mixture drawn with equal-height components misstates where the mass is. Scale
    by the weights when they are numeric; with a Dirichlet prior on the weights there is no
    honest number to scale by, so the overlay stays unweighted (shapes only)."""
    import numpy as np
    import pymc as pm

    from bayesdag.convert import to_ir

    with pm.Model() as known:
        pm.NormalMixture("x", w=[0.9, 0.1], mu=[-2.0, 2.0], sigma=[1.0, 1.0])
    with pm.Model() as prior_w:
        w = pm.Dirichlet("w", a=np.ones(2))
        pm.NormalMixture("x", w=w, mu=[-2.0, 2.0], sigma=[1.0, 1.0])

    kn = to_ir(known).node("x")
    assert kn.glyph_data["weighted"] is True
    peaks = [max(c["ys"]) for c in kn.glyph_data["curves"]]
    assert peaks[0] > peaks[1] * 5  # the 0.9 component dominates, as it should

    pw = to_ir(prior_w).node("x")
    assert pw.glyph_data["weighted"] is False
    assert max(abs(max(c["ys"]) - 1.0) for c in pw.glyph_data["curves"]) < 1e-9  # equal heights


def test_special_glyph_kinds_render():
    b = Box(0, 0, 80, 40)
    assert "<path" in glyph.render(
        "fan", {"mid": [0.5, 0.5, 0.5], "lo": [0.4, 0.3, 0.2], "hi": [0.6, 0.7, 0.8]}, b
    )
    assert "<ellipse" in glyph.render("pairplot", {"cov": [[1.0, 0.6], [0.6, 1.0]]}, b)
    assert glyph.render(
        "mixture", {"curves": [{"xs": [0, 1, 2], "ys": [0, 1, 0]}], "spike": 0.3}, b
    )
    assert "<rect" in glyph.render(
        "cutpoints", {"probs": [0.2, 0.5, 0.3], "cutpoints": [-1.0, 1.0]}, b
    )
    assert "<path" in glyph.render("simplex", {"curves": [{"xs": [0, 0.5, 1], "ys": [0, 1, 0]}]}, b)
    assert "<rect" in glyph.render(
        "censored", {"xs": [0, 1, 2], "ys": [0.2, 1, 0.2], "spikes": [{"x": 0.0, "h": 0.5}]}, b
    )


def test_glyph_sources(eight_schools_ir):
    g = {n.id: n.glyph for n in eight_schools_ir.nodes}
    assert g["mu"].source == "prior_analytic" and g["mu"].kind == "density"
    assert g["tau"].source == "prior_analytic"
    assert g["eta"].source == "prior_analytic"  # iid vector prior resolves to a single shape
    # continuous observed likelihood -> the data histogram, no fitted curve (a family fitted to
    # the marginal of regression data says nothing about the likelihood)
    assert g["y_obs"].source == "observed_hist" and g["y_obs"].kind == "histogram"
    assert g["theta"] is None  # deterministic -> no shape glyph in M0


def test_halfnormal_support_is_nonnegative(eight_schools_ir):
    tau = eight_schools_ir.node("tau")
    assert min(tau.glyph_data["xs"]) >= 0.0  # HalfNormal lives on [0, inf)


def test_density_and_histogram_render(eight_schools_ir):
    box = Box(0, 0, 40, 28)
    d = glyph.render("density", eight_schools_ir.node("mu").glyph_data, box)
    h = glyph.render("histogram", eight_schools_ir.node("y_obs").glyph_data, box)
    assert d.count("<path") >= 2  # filled area + line
    assert "<rect" in h


def test_predictive_check_uses_the_models_own_replicates(eight_schools_ir):
    """With predictive draws in the idata, an observed node shows its data against the model's
    replicates on one shared scale — the check a fitted-family overlay pretended to be."""
    import numpy as np
    import xarray as xr

    from bayesdag.adapters.glyph_data import predictive_glyph

    rng = np.random.default_rng(0)
    y = np.array([28.0, 8, -3, 7, -1, 1, 18, 12])
    idata = xr.DataTree.from_dict(
        {
            "observed_data": xr.Dataset({"y_obs": (("school",), y)}),
            "posterior_predictive": xr.Dataset(
                {"y_obs": (("chain", "draw", "school"), rng.normal(8, 12, size=(2, 50, 8)))}
            ),
        }
    )
    spec, data = predictive_glyph("y_obs", "Normal", idata)
    assert spec.kind == "ppc" and spec.source == "posterior_predictive"
    assert max(max(data["counts"]), max(data["ys"])) <= 1.0 + 1e-9
    assert 1 <= len(data["draws"]) <= 20
    # a shape mismatch between data and replicates is refused, not coerced
    bad = xr.DataTree.from_dict(
        {
            "observed_data": xr.Dataset({"y_obs": (("school",), y[:5])}),
            "posterior_predictive": xr.Dataset(
                {"y_obs": (("chain", "draw", "school"), rng.normal(size=(2, 50, 8)))}
            ),
        }
    )
    assert predictive_glyph("y_obs", "Normal", bad) is None


def test_discrete_predictive_check_has_per_value_whiskers():
    import numpy as np
    import xarray as xr

    from bayesdag.adapters.glyph_data import predictive_glyph

    rng = np.random.default_rng(0)
    idata = xr.DataTree.from_dict(
        {
            "observed_data": xr.Dataset({"k": (("i",), rng.poisson(3, 40))}),
            "prior_predictive": xr.Dataset(
                {"k": (("chain", "draw", "i"), rng.poisson(3, size=(1, 60, 40)))}
            ),
        }
    )
    spec, data = predictive_glyph("k", "Poisson", idata)
    assert spec.source == "prior_predictive"
    assert len(data["pred_lo"]) == len(data["cats"]) == len(data["heights"])
    assert "<circle" in glyph.render("ppc", data, Box(0, 0, 80, 40))


def test_discrete_observed_renders_as_class_bars():
    from conftest import MODEL_BUILDERS

    from bayesdag.convert import to_ir

    for name in ("irt", "mrp"):
        ir = to_ir(MODEL_BUILDERS[name]())
        y = ir.node("y")  # Bernoulli likelihood -> one bar per class, no continuous overlay
        assert y.glyph.source == "observed_hist" and y.glyph.kind == "bars"
        gd = y.glyph_data or {}
        assert "overlay" not in gd
        assert gd.get("cats") == [0, 1] and len(gd["heights"]) == 2


def test_bars_render_is_slot_centered():
    import re

    out = glyph.render("bars", {"cats": [0, 1], "heights": [0.6, 1.0]}, Box(0, 0, 40, 30))
    assert out.count("<rect") == 2  # one bar per class
    # bars sit centered in their slots, not pinned to the box edges
    xs = sorted(float(x) for x in re.findall(r'<rect x="([\d.]+)"', out))
    assert xs[0] > 0.0


def test_legacy_hist_overlay_still_renders():
    """Schema-1.0 IR may carry a fitted overlay; it keeps rendering (captioned as legacy)."""
    legacy = {
        "edges": [0.0, 1.0, 2.0],
        "counts": [0.5, 1.0],
        "overlay": {"xs": [0.0, 1.0, 2.0], "ys": [0.3, 1.0, 0.4]},
    }
    out = glyph.render("hist_overlay", legacy, Box(0, 0, 60, 30))
    assert "<rect" in out  # data bars
    assert "<path" in out  # best-fit family curve


def test_nonunivariate_kind_renders_via_same_registry():
    out = glyph.render("heatmap", {"matrix": [[0.1, 0.9], [0.9, 0.1]]}, Box(0, 0, 40, 28))
    assert out.count("<rect") == 5  # 4 cells + a framing rect


def test_2d_glyphs_get_a_square_area():
    """heatmap/pairplot need a near-square block, not the thin 1-D strip."""
    from bayesdag import geometry

    b = Box(0, 0, 140, 130)
    dens = {"xs": [0, 1], "ys": [0, 1]}
    mat = {"matrix": [[1.0, 0.0], [0.0, 1.0]]}
    cov = {"cov": [[1.0, 0.0], [0.0, 1.0]]}
    strip = geometry.glyph_rect(b, 16.0, "density", dens)
    square = geometry.glyph_rect(b, 16.0, "heatmap", mat)
    assert strip.h == geometry.GLYPH_H
    assert square.h > strip.h and abs(square.w - square.h) < 1.0
    # and the node reserves the taller area
    _, h_strip = geometry.node_size(120, 16, "density", dens)
    _, h_square = geometry.node_size(120, 16, "pairplot", cov)
    assert h_square > h_strip


def test_unknown_or_empty_render_is_blank():
    assert glyph.render("does-not-exist", {"x": 1}, Box(0, 0, 1, 1)) == ""
    assert glyph.render("density", None, Box(0, 0, 1, 1)) == ""


def test_posterior_glyphs_against_a_real_inference_data():
    """Every other posterior test hand-builds a DataTree. This one samples for real, so it is the
    only check that the join key the whole design rests on — node id == the CONSTRAINED idata
    variable name — actually matches what PyMC writes out. `tau` is the interesting case: it is
    stored unconstrained as `tau_log__`, and the glyph must still find `tau`."""
    import numpy as np
    import pymc as pm

    import bayesdag
    from bayesdag.convert import to_ir

    rng = np.random.default_rng(0)
    y = rng.normal(2.0, 1.5, 40)
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfNormal("tau", 5)
        pm.Normal("y", mu, tau, observed=y)
        idata = pm.sample(
            draws=50,
            tune=50,
            chains=1,
            cores=1,
            random_seed=0,
            progressbar=False,
            compute_convergence_checks=False,
        )

    ir = to_ir(m, idata=idata)
    for name in ("mu", "tau"):
        node = ir.node(name)
        assert node.glyph.source == "posterior_kde", f"{name} did not pick up its posterior"
        assert node.glyph_data["ys"], f"{name} has an empty density"
    assert ir.node("tau").idata_unconstrained_key == "tau_log__"
    assert ir.node("y").glyph.source == "observed_hist"  # observed data is never a posterior

    svg = bayesdag.view(m, idata=idata).to_svg()
    assert "#d2691e" in svg  # the posterior colour actually reaches the figure
    assert "posterior" in svg  # ...and the legend says so


def test_to_ir_bounded_on_large_observed_data():
    """The MLE best-fit overlay must thin its input — full-data scipy .fit() took ~12s on 1M
    points. Generous wall bound; the real guard is that thinning stays wired in."""
    import time

    import numpy as np
    import pymc as pm

    from bayesdag.convert import to_ir

    y = np.random.default_rng(0).standard_t(5, size=1_000_000)
    with pm.Model() as m:
        nu = pm.Exponential("nu", 0.1)
        pm.StudentT("y", nu=nu, mu=0, sigma=1, observed=y)
    t0 = time.perf_counter()
    ir = to_ir(m)
    elapsed = time.perf_counter() - t0
    n = ir.node("y")
    assert n.glyph is not None and n.glyph_data  # overlay still computed
    assert elapsed < 5.0, f"to_ir took {elapsed:.1f}s on 1M observed points"


def test_cutpoints_glyph_renders_bars_and_ticks():
    """`cutpoints` is registered but no adapter emits it: OrderedLogistic is indistinguishable
    from Categorical at the op level, so ordinal models route through the generic pmf path. The
    registry is deliberately producer-agnostic though — a hand-built IR may ask for this kind —
    so the renderer itself has to keep working."""
    out = glyph.render(
        "cutpoints",
        {"probs": [0.1, 0.3, 0.4, 0.2], "cutpoints": [-1.0, 0.2, 1.1]},
        Box(0, 0, 80, 44),
    )
    assert out.count("<rect") == 4  # one bar per category
    assert out.count("<line") >= 3  # one tick per cutpoint


def _cond(build):
    import pymc as pm

    from bayesdag.convert import to_ir

    with pm.Model() as m:
        build()
    return to_ir(m).node("x")


def test_exact_shapes_for_location_scale_conditionals():
    """A parent in a location/scale slot leaves the family's shape exact; the curve must be THAT
    family's standardized shape, not one bell for everything."""
    import numpy as np
    import pymc as pm
    import scipy.stats as st

    n = _cond(lambda: pm.HalfStudentT("x", nu=2, sigma=pm.HalfNormal("s", 1)))
    d = n.glyph_data
    assert (
        n.glyph.source == "conditional_standardized"
        and d["exact"]
        and d["family"] == ("HalfStudentT")
    )
    # compare against the analytic half-t(2) on the same (rescaled) grid
    hi = st.t(df=2).ppf(0.995)
    xs = np.asarray(d["xs"]) * hi
    ref = st.t(df=2).pdf(xs)
    assert np.allclose(np.asarray(d["ys"]), ref / ref.max(), atol=1e-6)

    u = _cond(lambda: pm.Uniform("x", lower=pm.Normal("a", 0, 1), upper=5.0))
    assert u.glyph_data["exact"] and np.allclose(u.glyph_data["ys"], 1.0)

    lo = _cond(lambda: pm.LogNormal("x", mu=pm.Normal("m", 0, 1), sigma=0.5))
    assert lo.glyph.source == "conditional_standardized"


def test_parent_governed_shape_gets_no_curve_and_names_the_slot():
    import pymc as pm

    for build, slot in [
        # op-level slot names (Gamma's op calls alpha "shape", StudentT's nu "df")
        (lambda: pm.Gamma("x", alpha=pm.HalfNormal("a", 1), beta=1.0), "shape"),
        (lambda: pm.Beta("x", alpha=pm.HalfNormal("a", 1), beta=2.0), "alpha"),
        (lambda: pm.LogNormal("x", mu=0.0, sigma=pm.HalfNormal("s", 1)), "sigma"),
        (lambda: pm.StudentT("x", nu=pm.Gamma("n", 2, 0.1), mu=0, sigma=1), "df"),
    ]:
        n = _cond(build)
        assert n.glyph is None and n.representable, slot
        assert n.glyph_note == f"no fixed curve: {slot} comes from a parent", n.glyph_note
    p = _cond(lambda: pm.Poisson("x", mu=pm.HalfNormal("m", 1)))
    assert p.glyph is None and "comes from a parent" in p.glyph_note


def test_standardized_shape_is_independent_of_parent_values():
    import pymc as pm

    def gd(scale):
        return _cond(lambda: pm.HalfStudentT("x", nu=3, sigma=pm.HalfNormal("s", scale))).glyph_data

    assert gd(1.0) == gd(50.0)
