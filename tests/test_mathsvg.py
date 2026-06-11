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


def test_broken_bundle_fails_once_not_per_label(tmp_path, monkeypatch):
    """A bundle that throws during eval must be attempted ONCE — every later render re-raises
    the cached error instead of re-evaluating the multi-MB bundle per label."""
    import py_mini_racer

    calls = {"n": 0}
    real = py_mini_racer.MiniRacer

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(py_mini_racer, "MiniRacer", counting)
    bundle = tmp_path / "mathjax.bundle.js"
    bundle.write_text("throw new Error('boom');")
    r = mathsvg.MathRenderer(bundle_path=bundle)
    with pytest.raises(Exception) as e1:
        r.render(r"\alpha")
    with pytest.raises(Exception) as e2:
        r.render(r"\beta")
    assert e2.value is e1.value  # the same cached exception object — no rebuild
    assert calls["n"] == 1


def test_bbox_cache_no_reparse_and_bounded(monkeypatch):
    """Warm re-layouts must hit the (svg, bboxes) cache — token_bboxes ran per label per
    layout before — and the cache must evict past its LRU bound."""
    r = mathsvg.MathRenderer()
    tex = r"\cssId{tok-a}{\alpha} + 1"
    svg1, bb1 = r.render_with_bboxes(tex)
    calls = {"n": 0}
    real = mathsvg.token_bboxes

    def counting(svg):
        calls["n"] += 1
        return real(svg)

    monkeypatch.setattr(mathsvg, "token_bboxes", counting)
    svg2, bb2 = r.render_with_bboxes(tex)
    assert (svg2, bb2) == (svg1, bb1) and "a" in bb2
    assert calls["n"] == 0  # cache hit: no re-parse
    monkeypatch.setattr(r, "_CACHE_MAX", 2)
    for i in range(3):
        r.render_with_bboxes(rf"x_{i}")
    assert len(r._cache) == 2  # LRU bound holds
