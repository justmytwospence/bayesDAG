"""The ONE shared SVG emitter. Both the static renderer and the anywidget consume its
output verbatim, so static == interactive by construction.

Composes (back-to-front): plate boxes, node chrome (role-styled), edges (token-anchored,
on top so port-edges into equations are visible), the embedded MathJax label SVG, and the
distribution-shape glyph. All coordinates are absolute px from the single ``LayoutResult``.
"""

from __future__ import annotations

import re
from xml.sax.saxutils import escape

from . import geometry, glyph
from .ir import Box, LayoutResult, ModelIR, NodeIR

# Role -> (fill, stroke, corner-radius). Rounded rectangles read better with math labels
# than ellipses; observed nodes are shaded (the conditioning cue).
_CHROME = {
    "latent": ("#ffffff", "#333333", 9.0),
    "observed": ("#e9eef5", "#33415c", 9.0),
    "deterministic": ("#ffffff", "#555555", 2.5),
    "data": ("#eeeeee", "#666666", 11.0),
    "potential": ("#f5ecdc", "#8a6d3b", 2.5),
    "factor": ("#f5ecdc", "#8a6d3b", 2.5),
}
_GLYPH_COLORS = {
    "prior_analytic": ("#2a8a55", "#2a8a55"),
    "prior_family_only": ("#999999", "#999999"),
    "posterior_kde": ("#d2691e", "#d2691e"),
    "observed_hist": ("#3a5f95", "#5a7fb5"),
}

_DEFS = (
    '<defs><marker id="bd-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
    'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
    'fill="#555"/></marker></defs>'
)


def _embed_label(label_svg: str, x: float, y: float, w: float, h: float) -> str:
    s = re.sub(r'width="[\d.]+ex"', f'width="{w:.1f}"', label_svg, count=1)
    s = re.sub(r'height="[\d.]+ex"', f'height="{h:.1f}"', s, count=1)
    s = s.replace("<svg ", f'<svg x="{x:.1f}" y="{y:.1f}" ', 1)
    return s


def _node_chrome(n: NodeIR, b: Box) -> str:
    fill, stroke, rx = _CHROME.get(n.role, _CHROME["latent"])
    dash = ' stroke-dasharray="4,3"' if n.role in ("potential", "factor") else ""
    return (
        f'<rect x="{b.x:.1f}" y="{b.y:.1f}" width="{b.w:.1f}" height="{b.h:.1f}" '
        f'rx="{rx}" ry="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"{dash}/>'
    )


def _edge(pts: list[list[float]]) -> str:
    if len(pts) < 2:
        return ""
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return f'<path d="{d}" fill="none" stroke="#555" stroke-width="1.3" marker-end="url(#bd-arrow)"/>'


def _plate(b: Box, label: str) -> str:
    return (
        f'<rect x="{b.x:.1f}" y="{b.y:.1f}" width="{b.w:.1f}" height="{b.h:.1f}" rx="6" ry="6" '
        f'fill="none" stroke="#9aa0a6" stroke-width="1" stroke-dasharray="2,2"/>'
        f'<text x="{b.x + b.w - 4:.1f}" y="{b.y + b.h - 5:.1f}" text-anchor="end" '
        f'font-size="11" fill="#6b7075">{escape(label)}</text>'
    )


def to_svg(ir: ModelIR, layout: LayoutResult, *, overlay_mode: str = "prior") -> str:
    c = layout.canvas or Box(0, 0, 100, 100)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{c.w:.1f}" height="{c.h:.1f}" '
        f'viewBox="0 0 {c.w:.1f} {c.h:.1f}" font-family="system-ui, sans-serif">',
        _DEFS,
    ]
    # plates (behind everything)
    for p in ir.plates:
        b = layout.plate_boxes.get(p.id)
        if b:
            out.append(_plate(b, p.label))
    # node chrome + glyph + label
    for n in ir.nodes:
        b = n.box or layout.node_boxes.get(n.id)
        if b is None:
            continue
        out.append(_node_chrome(n, b))
        if n.label_svg:
            lw, lh = geometry.label_px_size(n.label_svg)
            ox, oy = geometry.label_origin(b, lw, lh)
            out.append(_embed_label(n.label_svg, ox, oy, lw, lh))
        else:
            out.append(
                f'<text x="{b.x + b.w / 2:.1f}" y="{b.y + b.h / 2 + 4:.1f}" text-anchor="middle" '
                f'font-size="12" fill="#222">{escape(n.id)}</text>'
            )
        if n.glyph and n.glyph_data:
            _, lh = geometry.label_px_size(n.label_svg)
            gr = geometry.glyph_rect(b, n.role, lh)
            if gr:
                stroke, fill = _GLYPH_COLORS.get(n.glyph.source, ("#2a8a55", "#2a8a55"))
                out.append(glyph.render(n.glyph.kind, n.glyph_data, gr, stroke=stroke, fill=fill))
    # edges on top so token-anchored arrowheads into equations are visible
    for e in ir.edges:
        pts = layout.edge_paths.get(f"{e.source}|{e.target}")
        if pts:
            out.append(_edge(pts))
    out.append("</svg>")
    return "".join(out)
