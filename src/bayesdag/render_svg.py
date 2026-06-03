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
    if n.role == "deterministic":
        # no visible box around the equation; a transparent rect keeps it hover/click-able
        # and gives edges a region to land in.
        return (
            f'<rect x="{b.x:.1f}" y="{b.y:.1f}" width="{b.w:.1f}" height="{b.h:.1f}" '
            'fill="transparent" stroke="none"/>'
        )
    fill, stroke, rx = _CHROME.get(n.role, _CHROME["latent"])
    dash = ' stroke-dasharray="4,3"' if n.role in ("potential", "factor") else ""
    return (
        f'<rect x="{b.x:.1f}" y="{b.y:.1f}" width="{b.w:.1f}" height="{b.h:.1f}" '
        f'rx="{rx}" ry="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"{dash}/>'
    )


def _edge(pts: list[list[float]], src: str, tgt: str) -> str:
    if len(pts) < 2:
        return ""
    if len(pts) >= 4 and (len(pts) - 1) % 3 == 0:  # cubic chain: p0 + (c1,c2,p) triples
        d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
        for i in range(1, len(pts), 3):
            c1, c2, p = pts[i], pts[i + 1], pts[i + 2]
            d += f" C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p[0]:.1f},{p[1]:.1f}"
    else:
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return (
        f'<path class="bd-edge" data-src="{escape(src)}" data-tgt="{escape(tgt)}" d="{d}" '
        'fill="none" stroke="#555" stroke-width="1.3" marker-end="url(#bd-arrow)"/>'
    )


def _plate(b: Box, label: str) -> str:
    return (
        f'<rect x="{b.x:.1f}" y="{b.y:.1f}" width="{b.w:.1f}" height="{b.h:.1f}" rx="6" ry="6" '
        f'fill="none" stroke="#9aa0a6" stroke-width="1" stroke-dasharray="2,2" pointer-events="all"/>'
        f'<text x="{b.x + b.w - 4:.1f}" y="{b.y + b.h - 5:.1f}" text-anchor="end" '
        f'font-size="11" fill="#6b7075">{escape(label)}</text>'
    )


def _panel_curve(xs: list[float], ys: list[float], box: Box, color: str = "#3a6ea5") -> str:
    x0, x1 = xs[0], xs[-1]
    span = (x1 - x0) or 1.0
    pts = [
        (box.x + (x - x0) / span * box.w, box.y + box.h - max(0.0, min(1.0, y)) * box.h)
        for x, y in zip(xs, ys)
    ]
    d = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1" opacity="0.3"/>'


def render_plate_panel(expansion: dict) -> str:
    """Standalone SVG for a plate's prior-predictive expansion: per member variable, the N
    per-instance densities overlaid on a shared axis (observed members get data ticks)."""
    members = expansion.get("members", [])
    if not members:
        return ""
    pw, rh, pad, title = 280.0, 60.0, 12.0, 18.0
    w = pw + 2 * pad
    h = pad * 2 + 22 + len(members) * (rh + title + 8)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" '
        f'viewBox="0 0 {w:.0f} {h:.0f}" font-family="system-ui, sans-serif">',
        f'<rect width="{w:.0f}" height="{h:.0f}" rx="8" fill="#ffffff" stroke="#c7c7cc"/>',
        f'<text x="{pad}" y="{pad + 12}" font-size="12" font-weight="600" fill="#333">'
        f'prior predictive — {escape(expansion.get("label", ""))}</text>',
    ]
    y = pad + 22
    for mem in members:
        cap = f" (showing {len(mem['curves'])})" if mem.get("capped") else ""
        obs_note = "  ·  orange ticks = observed data" if mem.get("observed") else ""
        out.append(
            f'<text x="{pad}" y="{y + 11:.1f}" font-size="11" fill="#555">'
            f'{escape(mem["id"])} — {mem["n"]} instances{cap}{obs_note}</text>'
        )
        box = Box(pad, y + title, pw, rh)
        out.append(
            f'<rect x="{box.x}" y="{box.y:.1f}" width="{box.w}" height="{box.h}" '
            'fill="#fafafa" stroke="#eeeeee"/>'
        )
        xs = mem["xs"]
        for c in mem["curves"]:
            out.append(_panel_curve(xs, c, box))
        if mem.get("observed"):
            x0, x1 = xs[0], xs[-1]
            span = (x1 - x0) or 1.0
            for v in mem["observed"]:
                if x0 <= v <= x1:
                    px = box.x + (v - x0) / span * box.w
                    out.append(
                        f'<line x1="{px:.1f}" y1="{box.y + box.h - 8:.1f}" x2="{px:.1f}" '
                        f'y2="{box.y + box.h:.1f}" stroke="#d2691e" stroke-width="1.5"/>'
                    )
        y += title + rh + 8
    out.append("</svg>")
    return "".join(out)


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


def _render_legend(items, ox: float, oy: float) -> tuple[str, float, float]:
    """A compact single-column legend panel placed to the SIDE (widens, not lengthens, the
    figure) so it doesn't eat vertical space."""
    pad, row_h, sw, title_h = 9.0, 19.0, 24.0, 20.0
    label_w = max((len(it.label) for it in items), default=10) * 5.7
    w = 2 * pad + sw + label_w
    h = pad * 2 + title_h + len(items) * row_h
    out = ['<g class="bd-legend">']
    out.append(
        f'<rect x="{ox:.1f}" y="{oy:.1f}" width="{w:.1f}" height="{h:.1f}" rx="6" '
        'fill="#fcfcfc" stroke="#dddddd"/>'
    )
    out.append(
        f'<text x="{ox + pad:.1f}" y="{oy + pad + 11:.1f}" font-size="11.5" font-weight="600" '
        'fill="#444">Legend</text>'
    )
    y0 = oy + pad + title_h
    for i, it in enumerate(items):
        cy = y0 + i * row_h
        out.append(_legend_swatch(it.swatch, Box(ox + pad, cy + 2, 16, row_h - 7)))
        out.append(
            f'<text x="{ox + pad + sw:.1f}" y="{cy + row_h - 6:.1f}" font-size="10.5" '
            f'fill="#333">{escape(it.label)}</text>'
        )
    out.append("</g>")
    return "".join(out), w, h


def to_svg(ir: ModelIR, layout: LayoutResult, *, overlay_mode: str = "prior", legend: bool = True) -> str:
    c = layout.canvas or Box(0, 0, 100, 100)
    body = [_DEFS]
    # plates (behind everything), tagged for click-to-expand
    for p in ir.plates:
        b = layout.plate_boxes.get(p.id)
        if b:
            body.append(f'<g class="bd-plate" data-plate="{escape(p.id)}">' + _plate(b, p.label) + "</g>")
    # Two passes so a later node's (opaque) chrome box can never paint over an earlier node's
    # label/glyph: ALL chrome first, then ALL labels+glyphs on top. Each pass tags its group with
    # data-node, and the widget keys hover/selection off data-node (js/index.js), so splitting a
    # node across two groups is transparent to interactivity. Parity holds: one emitter, both
    # renderers consume this verbatim.
    drawn = [(n, b) for n in ir.nodes if (b := (n.box or layout.node_boxes.get(n.id))) is not None]
    for n, b in drawn:  # chrome (boxes) behind everything else node-ish
        body.append(f'<g class="bd-node" data-node="{escape(n.id)}">' + _node_chrome(n, b) + "</g>")
    for n, b in drawn:  # labels + glyphs above all chrome
        if n.label_svg:
            lw, lh = geometry.label_px_size(n.label_svg)
            ox, oy = geometry.label_origin(b, lw, lh)
            parts = [_embed_label(n.label_svg, ox, oy, lw, lh)]
        else:
            parts = [
                f'<text x="{b.x + b.w / 2:.1f}" y="{b.y + b.h / 2 + 4:.1f}" text-anchor="middle" '
                f'font-size="12" fill="#222">{escape(n.id)}</text>'
            ]
        if n.glyph and n.glyph_data:
            _, lh = geometry.label_px_size(n.label_svg)
            gr = geometry.glyph_rect(b, n.role, lh)
            if gr:
                stroke, fill = _GLYPH_COLORS.get(n.glyph.source, ("#2a8a55", "#2a8a55"))
                parts.append(glyph.render(n.glyph.kind, n.glyph_data, gr, stroke=stroke, fill=fill))
        body.append(f'<g class="bd-node" data-node="{escape(n.id)}">' + "".join(parts) + "</g>")
    # edges on top so token-anchored arrowheads into equations are visible
    for e in ir.edges:
        pts = layout.edge_paths.get(f"{e.source}|{e.target}")
        if pts:
            body.append(_edge(pts, e.source, e.target))

    legend_svg, total_w, total_h = "", c.w, c.h
    if legend:
        items = _legend.build(ir)
        if items:
            legend_svg, lw, lh = _render_legend(items, c.w + _LEGEND_GAP, 0.0)
            total_w, total_h = c.w + _LEGEND_GAP + lw, max(c.h, lh)

    header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.1f}" height="{total_h:.1f}" '
        f'viewBox="0 0 {total_w:.1f} {total_h:.1f}" font-family="system-ui, sans-serif">'
    )
    return header + "".join(body) + legend_svg + "</svg>"
