"""Context-aware legend: only shows encodings present; embedded and toggleable."""

import xml.etree.ElementTree as ET

from bayesdag import legend
from bayesdag.layout import layout
from bayesdag.render_svg import to_svg


def test_build_is_context_aware(eight_schools_ir):
    swatches = {i.swatch for i in legend.build(eight_schools_ir)}
    assert {"role:latent", "role:observed", "role:deterministic"} <= swatches
    assert {"glyph:prior_analytic", "glyph:observed_hist", "glyph:best_fit"} <= swatches
    assert {"symbol:~", "symbol:=", "plate"} <= swatches
    # no posterior entry without an idata
    assert "glyph:posterior_kde" not in swatches


def test_legend_embedded_by_default(eight_schools_ir):
    svg = to_svg(eight_schools_ir, layout(eight_schools_ir))
    assert "bd-legend" in svg and ">Legend<" in svg
    ET.fromstring(svg)  # still well-formed


def test_legend_can_be_disabled(eight_schools_ir):
    svg = to_svg(eight_schools_ir, layout(eight_schools_ir), legend=False)
    assert "bd-legend" not in svg
