"""TeX->SVG + token-anchor extraction (the M0 spike, as regression tests)."""

import pytest

from bayesdag import mathsvg

_r = mathsvg.get_renderer()
pytestmark = pytest.mark.skipif(
    not _r.available, reason="needs the 'math' extra (mini-racer) + a built mathjax bundle"
)


def test_render_produces_self_contained_svg():
    svg = mathsvg.render(r"\mathcal{N}(\mu, \sigma)")
    assert "<svg" in svg
    assert "<path" in svg  # fontCache='local' => inline glyph paths (self-contained)
    assert "data-mml-node" in svg


def test_token_anchors_are_fractional_and_ordered():
    tex = r"\mathcal{N}(\cssId{tok-mu}{\mu},\ \cssId{tok-sg}{\sigma})"
    svg, anchors = mathsvg.render_with_anchors(tex)
    assert set(anchors) == {"mu", "sg"}
    # mu must sit to the left of sigma in N(mu, sigma)
    assert anchors["mu"][0] < anchors["sg"][0]
    for fx, fy in anchors.values():
        assert 0.0 <= fx <= 1.0
        assert 0.0 <= fy <= 1.0


@pytest.mark.parametrize("tex", [r"\sigma^2", r"\frac{a}{b}", r"\alpha + \beta x", r"\Sigma"])
def test_renders_varied_tex(tex):
    svg = mathsvg.render(tex)
    assert "<svg" in svg and "data-mml-node" in svg


def test_render_is_cached():
    assert mathsvg.render(r"\alpha") == mathsvg.render(r"\alpha")
