"""Shared SVG emitter: well-formedness, embedded labels/glyphs, and saving."""

import re
import xml.etree.ElementTree as ET

import pytest

from bayesdag import mathsvg, render_static
from bayesdag.ir import ModelIR
from bayesdag.layout import layout
from bayesdag.render_svg import to_svg

_math = mathsvg.get_renderer().available


def test_empty_model_gets_the_placeholder_canvas():
    """`Box` is a dataclass and therefore always truthy, so the `layout.canvas or Box(...)`
    placeholder never fired — an empty model emitted a 0x0 SVG."""
    import re

    ir = ModelIR(nodes=[], edges=[], plates=[])
    svg = to_svg(ir, layout(ir))
    w, h = (float(v) for v in re.search(r'width="([\d.]+)" height="([\d.]+)"', svg).groups())
    assert w > 0 and h > 0


def test_save_writes_utf8_regardless_of_locale(tmp_path, eight_schools_ir):
    """The SVG always carries non-ASCII (badge glyphs, elision ellipsis, legend dashes), so the
    write must not depend on the platform's default encoding."""
    svg = to_svg(eight_schools_ir, layout(eight_schools_ir))
    p = render_static.save(svg, tmp_path / "m.svg")
    assert p.read_text(encoding="utf-8") == svg


def test_svg_is_well_formed_xml(eight_schools_ir):
    res = layout(eight_schools_ir)
    svg = to_svg(eight_schools_ir, res)
    root = ET.fromstring(svg)  # composed doc (incl. nested MathJax svgs) must parse
    assert root.tag.endswith("svg")
    assert "viewBox" in root.attrib
    # one rect per node (chrome) at minimum, plus the plate
    assert svg.count("<rect") >= len(eight_schools_ir.nodes)
    assert "marker-end" in svg  # edges have arrowheads
    assert "#c0392b" in svg  # observed node's MLE best-fit family overlay curve


@pytest.mark.skipif(not _math, reason="needs the built mathjax bundle")
def test_svg_embeds_labels_and_glyphs(eight_schools_ir):
    res = layout(eight_schools_ir)
    svg = to_svg(eight_schools_ir, res)
    assert "data-mml-node" in svg  # embedded MathJax labels
    # density glyph paths + observed histogram bars beyond the edge paths
    assert svg.count("<path") > len(eight_schools_ir.edges)
    ET.fromstring(svg)  # still well-formed with nested svgs


def test_save_svg(tmp_path, eight_schools_ir):
    res = layout(eight_schools_ir)
    svg = to_svg(eight_schools_ir, res)
    out = render_static.save(svg, tmp_path / "model.svg")
    assert out.exists() and out.read_text().lstrip().startswith("<svg")


def test_plate_click_target_is_the_border_not_the_interior(eight_schools_ir):
    """A plate encloses most of the canvas. If its interior were a click target it would swallow
    the background click that dismisses the pinned card — which the card itself tells users to
    make ("click empty space to close"). The hit area is a transparent stroke band instead."""
    svg = to_svg(eight_schools_ir, layout(eight_schools_ir))
    assert 'pointer-events="all"' not in svg  # interior must not be hittable
    assert 'stroke-opacity="0" stroke-width="10" pointer-events="stroke"' in svg


@pytest.mark.skipif(not _math, reason="needs the built mathjax bundle")
def test_mathjax_defs_are_deduped(eight_schools_ir):
    """Labels share one content-hashed <defs>: every glyph reference resolves, no duplicate
    path data remains (per-equation MJX defs were >50% of output bytes)."""
    import re

    res = layout(eight_schools_ir)
    svg = to_svg(eight_schools_ir, res)
    ids = re.findall(r'<path id="([^"]+)"', svg)
    refs = set(re.findall(r'(?:xlink:)?href="#((?:bdg|MJX)[^"]+)"', svg))
    assert ids and all(i.startswith("bdg-") for i in ids)  # all label defs hoisted + hashed
    assert len(ids) == len(set(ids))  # one defs entry per unique glyph, document-wide
    assert refs <= set(ids)  # no dangling glyph references
    # the dedupe is the point: fewer defs entries than total glyph uses
    assert len(ids) < svg.count("data-c=")


@pytest.mark.skipif(not _math, reason="needs the built mathjax bundle for real labels")
def test_composed_svg_has_no_duplicate_element_ids(eight_schools_ir):
    """Every label carries the same per-token ids — `tok-loc`, `tok-scale`, `tok-__lhs__` — so a
    document with two Normals had duplicate element ids and was invalid SVG. (The committed hero
    image had `tok-loc` and `tok-scale` four times each.) Namespacing them per node fixes it;
    this is the net that would have caught it."""
    svg = to_svg(eight_schools_ir, layout(eight_schools_ir))
    ids = [el.get("id") for el in ET.fromstring(svg).iter() if el.get("id")]

    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate element ids in the composed SVG: {sorted(dupes)}"
    assert 'id="tok-' not in svg  # raw, un-namespaced token ids must not survive embedding
    assert 'id="bd-y_obs-tok-loc"' in svg  # ...and the namespaced form is what replaced them


def test_dom_slug_keeps_colliding_names_apart():
    """Variable names are arbitrary Python strings, so the sanitizer has to be lossy — which
    means it also has to be injective enough that two different names never share an id."""
    from bayesdag.render_svg import _dom_slug

    assert _dom_slug("y_obs") == "y_obs"  # already safe: left readable, no hash noise
    assert _dom_slug("a-b") == "a-b"
    assert _dom_slug("a b") != _dom_slug("a|b")  # both flatten to `a_b` without the hash
    for raw in ("a b", "a|b", "θ", "x[0]"):
        slug = _dom_slug(raw)
        assert re.fullmatch(r"[A-Za-z0-9_-]+", slug), f"{raw!r} -> {slug!r} is not DOM-safe"
