"""Shared SVG emitter: well-formedness, embedded labels/glyphs, and saving."""

import xml.etree.ElementTree as ET

import pytest

from bayesdag import mathsvg, render_static
from bayesdag.layout import layout
from bayesdag.render_svg import to_svg

_math = mathsvg.get_renderer().available


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


@pytest.mark.skipif(not _math, reason="needs the 'math' extra")
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


@pytest.mark.skipif(not _math, reason="needs the 'math' extra")
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
