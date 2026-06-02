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
from . import legend as _legend
from .ir import Box, LayoutResult, ModelIR, NodeIR

_LEGEND_GAP = 14.0
_BELL = ([0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0], [0.05, 0.2, 0.55, 0.9, 1.0, 0.9, 0.55, 0.2, 0.05])

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
    '<defs><marker id="bd-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5.5" '
    'markerHeight="5.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
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


def _legend_swatch(kind: str, b: Box) -> str:
    if kind.startswith("role:"):
        role = kind.split(":", 1)[1]
        fill, stroke, _ = _CHROME.get(role, _CHROME["latent"])
        dash = ' stroke-dasharray="3,2"' if role in ("potential", "factor") else ""
        return (
            f'<rect x="{b.x:.1f}" y="{b.y:.1f}" width="{b.w:.1f}" height="{b.h:.1f}" rx="3" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1"{dash}/>'
        )
    if kind.startswith("glyph:"):
        src = kind.split(":", 1)[1]
        stroke, fill = _GLYPH_COLORS.get(src, ("#2a8a55", "#2a8a55"))
        if src == "observed_hist":
            data = {"edges": [0, 1, 2, 3], "counts": [0.6, 1.0, 0.45]}
            return glyph.render("histogram", data, b, fill=fill, stroke=stroke)
        kindname = "schematic" if src == "prior_family_only" else "density"
        return glyph.render(kindname, {"xs": _BELL[0], "ys": _BELL[1]}, b, stroke=stroke, fill=fill)
    if kind == "plate":
        return (
            f'<rect x="{b.x:.1f}" y="{b.y:.1f}" width="{b.w:.1f}" height="{b.h:.1f}" rx="3" '
            'fill="none" stroke="#9aa0a6" stroke-dasharray="2,2"/>'
        )
    if kind.startswith("symbol:"):
        s = kind.split(":", 1)[1]
        return (
            f'<text x="{b.x + b.w / 2:.1f}" y="{b.y + b.h - 1:.1f}" text-anchor="middle" '
            f'font-size="13" fill="#333">{escape(s)}</text>'
        )
    if kind == "elision":
        return f'<text x="{b.x:.1f}" y="{b.y + b.h - 1:.1f}" font-size="11" fill="#333">[⋯]</text>'
    return ""


def _render_legend(items, ox: float, oy: float, content_w: float) -> tuple[str, float, float]:
    pad, row_h, sw, title_h = 10.0, 20.0, 26.0, 20.0
    label_w = max((len(it.label) for it in items), default=10) * 6.3
    w = max(content_w, sw + label_w + 2 * pad, 220.0)
    h = pad * 2 + title_h + len(items) * row_h
    out = ['<g class="bd-legend">']
    out.append(
        f'<rect x="{ox:.1f}" y="{oy:.1f}" width="{w:.1f}" height="{h:.1f}" rx="6" '
        'fill="#fcfcfc" stroke="#dddddd"/>'
    )
    out.append(
        f'<text x="{ox + pad:.1f}" y="{oy + pad + 12:.1f}" font-size="12" font-weight="600" '
        'fill="#444">Legend</text>'
    )
    y = oy + pad + title_h
    for it in items:
        out.append(_legend_swatch(it.swatch, Box(ox + pad, y + 2, 18, row_h - 7)))
        out.append(
            f'<text x="{ox + pad + sw:.1f}" y="{y + row_h - 6:.1f}" font-size="11" '
            f'fill="#333">{escape(it.label)}</text>'
        )
        y += row_h
    out.append("</g>")
    return "".join(out), w, h


def to_svg(ir: ModelIR, layout: LayoutResult, *, overlay_mode: str = "prior", legend: bool = True) -> str:
    c = layout.canvas or Box(0, 0, 100, 100)
    body = [_DEFS]
    # plates (behind everything)
    for p in ir.plates:
        b = layout.plate_boxes.get(p.id)
        if b:
            body.append(_plate(b, p.label))
    # node chrome + label + glyph
    for n in ir.nodes:
        b = n.box or layout.node_boxes.get(n.id)
        if b is None:
            continue
        body.append(_node_chrome(n, b))
        if n.label_svg:
            lw, lh = geometry.label_px_size(n.label_svg)
            ox, oy = geometry.label_origin(b, lw, lh)
            body.append(_embed_label(n.label_svg, ox, oy, lw, lh))
        else:
            body.append(
                f'<text x="{b.x + b.w / 2:.1f}" y="{b.y + b.h / 2 + 4:.1f}" text-anchor="middle" '
                f'font-size="12" fill="#222">{escape(n.id)}</text>'
            )
        if n.glyph and n.glyph_data:
            _, lh = geometry.label_px_size(n.label_svg)
            gr = geometry.glyph_rect(b, n.role, lh)
            if gr:
                stroke, fill = _GLYPH_COLORS.get(n.glyph.source, ("#2a8a55", "#2a8a55"))
                body.append(glyph.render(n.glyph.kind, n.glyph_data, gr, stroke=stroke, fill=fill))
    # edges on top so token-anchored arrowheads into equations are visible
    for e in ir.edges:
        pts = layout.edge_paths.get(f"{e.source}|{e.target}")
        if pts:
            body.append(_edge(pts))

    legend_svg, total_w, total_h = "", c.w, c.h
    if legend:
        items = _legend.build(ir)
        if items:
            legend_svg, lw, lh = _render_legend(items, 0.0, c.h + _LEGEND_GAP, c.w)
            total_w, total_h = max(c.w, lw), c.h + _LEGEND_GAP + lh

    header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.1f}" height="{total_h:.1f}" '
        f'viewBox="0 0 {total_w:.1f} {total_h:.1f}" font-family="system-ui, sans-serif">'
    )
    return header + "".join(body) + legend_svg + "</svg>"
