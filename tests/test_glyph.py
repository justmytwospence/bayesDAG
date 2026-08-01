"""Shape-first glyph system: data provider sources + glyph-agnostic registry rendering."""

from bayesdag import glyph
from bayesdag.ir import Box


def test_registry_has_density_and_nonunivariate_kinds():
    ks = glyph.registered_kinds()
    assert {"density", "histogram", "hist_overlay", "bars", "schematic", "heatmap", "curve"} <= ks
    assert {"fan", "pairplot", "mixture", "cutpoints", "simplex", "censored", "stem", "step"} <= ks  # special-construct kinds


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
    CONDITIONAL latent (any param is a parent RV) must fall to the family-only schematic rather than
    being silently sampled into a misleading, non-deterministic green density (the old bug, where
    ``.eval()`` drew a random value through the parents)."""
    import pymc as pm

    from bayesdag.convert import to_ir

    def build(scale=1.0):
        with pm.Model() as m:
            mu = pm.Normal("mu", 0, 5)             # root prior
            sigma = pm.HalfNormal("sigma", scale)  # root prior
            pm.Normal("x", mu, sigma)              # conditional latent: params depend on parents
        return m

    g = {n.id: n.glyph for n in to_ir(build()).nodes}
    assert g["mu"].source == "prior_analytic" and g["sigma"].source == "prior_analytic"
    assert g["x"].source == "prior_family_only" and g["x"].kind == "schematic"
    # the schematic is a fixed, parameter-FREE shape: it must be identical regardless of the parent
    # scale (proving it's canonical, never a value sampled through the parents at render time).
    assert to_ir(build(1.0)).node("x").glyph_data == to_ir(build(100.0)).node("x").glyph_data


def test_varying_vector_prior_does_not_plot_element_zero_as_the_node():
    """An iid vector prior broadcasts ONE density across its elements, so a single analytic curve
    is honest. A varying vector prior gives each element its own density — plotting element 0 and
    badging it `prior_analytic` would state something false about the rest, so it degrades to the
    family schematic. A Categorical is exempt: its vector param IS the pmf."""
    import numpy as np
    import pymc as pm

    from bayesdag.convert import to_ir

    with pm.Model(coords={"k": [0, 1]}) as m:
        pm.Normal("iid", 0.0, 1.0, dims="k")                                # broadcast scalars
        pm.Normal("varying", mu=np.array([0.0, 5.0]), sigma=np.array([1.0, 10.0]), dims="k")
        pm.Categorical("choice", p=np.array([0.2, 0.3, 0.5]))

    g = {n.id: n.glyph for n in to_ir(m).nodes}
    assert g["iid"].source == "prior_analytic" and g["iid"].kind == "density"
    assert g["varying"].source == "prior_family_only" and g["varying"].kind == "schematic"
    assert g["choice"].source == "prior_analytic" and g["choice"].kind == "bars"


def test_discrete_posterior_is_bars_and_vector_posterior_says_it_is_pooled():
    """Two honesty fixes on the posterior path: a discrete variable's posterior is a pmf over
    integers (a gaussian KDE would put mass on values it cannot take), and a vector parameter's
    KDE pools every element's draws into one curve — which is not any element's marginal, so the
    panel has to say so."""
    import numpy as np
    import pymc as pm
    import xarray as xr

    from bayesdag.convert import to_ir
    from bayesdag.render_svg import render_node_panel

    rng = np.random.default_rng(0)
    idata = xr.DataTree.from_dict({"posterior": xr.Dataset({
        "k": (("chain", "draw"), rng.poisson(3.0, size=(2, 200))),
        "theta": (("chain", "draw", "g"), rng.normal(size=(2, 200, 4))),
    })})
    with pm.Model(coords={"g": range(4)}) as m:
        pm.Poisson("k", 3.0)
        pm.Normal("theta", 0.0, 1.0, dims="g")

    ir = to_ir(m, idata=idata)
    k, theta = ir.node("k"), ir.node("theta")
    assert k.glyph.kind == "bars" and k.glyph.source == "posterior_bars"
    assert all(float(c).is_integer() for c in k.glyph_data["cats"])
    assert theta.glyph.kind == "density" and theta.glyph.source == "posterior_kde"
    assert theta.glyph_data["pooled"] == 4
    assert "pooled over 4 elements" in render_node_panel(theta)
    assert "pooled" not in render_node_panel(k)


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
    assert "<path" in glyph.render("fan", {"mid": [0.5, 0.5, 0.5], "lo": [0.4, 0.3, 0.2], "hi": [0.6, 0.7, 0.8]}, b)
    assert "<ellipse" in glyph.render("pairplot", {"cov": [[1.0, 0.6], [0.6, 1.0]]}, b)
    assert glyph.render("mixture", {"curves": [{"xs": [0, 1, 2], "ys": [0, 1, 0]}], "spike": 0.3}, b)
    assert "<rect" in glyph.render("cutpoints", {"probs": [0.2, 0.5, 0.3], "cutpoints": [-1.0, 1.0]}, b)
    assert "<path" in glyph.render("simplex", {"curves": [{"xs": [0, 0.5, 1], "ys": [0, 1, 0]}]}, b)
    assert "<rect" in glyph.render("censored", {"xs": [0, 1, 2], "ys": [0.2, 1, 0.2], "spikes": [{"x": 0.0, "h": 0.5}]}, b)


def test_glyph_sources(eight_schools_ir):
    g = {n.id: n.glyph for n in eight_schools_ir.nodes}
    assert g["mu"].source == "prior_analytic" and g["mu"].kind == "density"
    assert g["tau"].source == "prior_analytic"
    assert g["eta"].source == "prior_analytic"  # iid vector prior resolves to a single shape
    # continuous observed likelihood -> data histogram + MLE best-fit family overlay
    assert g["y_obs"].source == "observed_hist" and g["y_obs"].kind == "hist_overlay"
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


def test_observed_overlay_shared_scale(eight_schools_ir):
    gd = eight_schools_ir.node("y_obs").glyph_data
    assert "overlay" in gd and gd["overlay"]["xs"] and gd["overlay"]["ys"]
    # the best-fit curve spans exactly the histogram's x-range...
    assert gd["overlay"]["xs"][0] == gd["edges"][0]
    assert gd["overlay"]["xs"][-1] == gd["edges"][-1]
    # ...and both sit on ONE shared vertical scale: each peaks <=1, the taller one hits ~1
    assert max(gd["counts"]) <= 1.0 + 1e-9
    assert max(gd["overlay"]["ys"]) <= 1.0 + 1e-9
    assert abs(max(max(gd["counts"]), max(gd["overlay"]["ys"])) - 1.0) < 1e-9


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


def test_hist_overlay_renders_bars_and_curve(eight_schools_ir):
    out = glyph.render("hist_overlay", eight_schools_ir.node("y_obs").glyph_data, Box(0, 0, 60, 30))
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
    strip = geometry.glyph_rect(b, "latent", 16.0, "density", dens)
    square = geometry.glyph_rect(b, "latent", 16.0, "heatmap", mat)
    assert strip.h == geometry.GLYPH_H
    assert square.h > strip.h and abs(square.w - square.h) < 1.0
    # and the node reserves the taller area
    _, h_strip = geometry.node_size(120, 16, "latent", "density", dens)
    _, h_square = geometry.node_size(120, 16, "latent", "pairplot", cov)
    assert h_square > h_strip


def test_unknown_or_empty_render_is_blank():
    assert glyph.render("does-not-exist", {"x": 1}, Box(0, 0, 1, 1)) == ""
    assert glyph.render("density", None, Box(0, 0, 1, 1)) == ""


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
