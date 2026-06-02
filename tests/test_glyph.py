"""Shape-first glyph system: data provider sources + glyph-agnostic registry rendering."""

from bayesdag import glyph
from bayesdag.ir import Box


def test_registry_has_density_and_nonunivariate_kinds():
    ks = glyph.registered_kinds()
    assert {"density", "histogram", "schematic", "heatmap"} <= ks


def test_glyph_sources(eight_schools_ir):
    g = {n.id: n.glyph for n in eight_schools_ir.nodes}
    assert g["mu"].source == "prior_analytic" and g["mu"].kind == "density"
    assert g["tau"].source == "prior_analytic"
    assert g["eta"].source == "prior_analytic"  # iid vector prior resolves to a single shape
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


def test_nonunivariate_kind_renders_via_same_registry():
    out = glyph.render("heatmap", {"matrix": [[0.1, 0.9], [0.9, 0.1]]}, Box(0, 0, 40, 28))
    assert out.count("<rect") == 4


def test_unknown_or_empty_render_is_blank():
    assert glyph.render("does-not-exist", {"x": 1}, Box(0, 0, 1, 1)) == ""
    assert glyph.render("density", None, Box(0, 0, 1, 1)) == ""
