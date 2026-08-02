"""The ONE shared SVG emitter. Both the static renderer and the anywidget consume its
output verbatim, so static == interactive by construction.

Composes (back-to-front): plate boxes, node chrome (role-styled), edges (token-anchored,
on top so port-edges into equations are visible), the embedded MathJax label SVG, and the
distribution-shape glyph. All coordinates are absolute px from the single ``LayoutResult``.
"""

from __future__ import annotations

import hashlib
import re
from xml.sax.saxutils import escape

from . import geometry, glyph
from . import legend as _legend
from .glyph.kinds import bar_layout as glyph_bar_layout
from .ir import Box, LayoutResult, ModelIR, NodeIR

_LEGEND_GAP = 14.0
_BELL = (
    [0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0],
    [0.05, 0.2, 0.55, 0.9, 1.0, 0.9, 0.55, 0.2, 0.05],
)

# Role -> (fill, stroke, corner-radius). Rounded rectangles read better with math labels
# than ellipses; observed nodes are shaded (the conditioning cue).
_CHROME = {
    "latent": ("#ffffff", "#333333", 9.0),
    "observed": ("#e9eef5", "#33415c", 9.0),
    # deterministic = a crisp, light box (sharper corners + a thinner, softer border than the rounded
    # random-variable nodes) so a *computed* quantity reads as secondary to the things being inferred.
    "deterministic": ("#ffffff", "#9499a2", 3.0),
    "data": ("#eeeeee", "#666666", 11.0),
    "potential": ("#f5ecdc", "#8a6d3b", 2.5),
    "factor": ("#f5ecdc", "#8a6d3b", 2.5),
}
_GLYPH_COLORS = {
    "prior_analytic": ("#2a8a55", "#2a8a55"),
    "prior_family_only": ("#999999", "#999999"),
    "posterior_kde": ("#d2691e", "#d2691e"),
    "posterior_bars": ("#d2691e", "#d2691e"),
    "observed_hist": ("#3a5f95", "#5a7fb5"),
    "deterministic_fn": (
        "#7a5bd0",
        "#7a5bd0",
    ),  # transfer-function curve (a 4th hue: a computed transform)
}
# MLE best-fit family curve drawn over an observed histogram (the conventional "fitted curve" red;
# distinct from data=blue, prior=green, posterior=orange).
_OVERLAY = "#c0392b"


def _arrow(mid: str, color: str) -> str:
    return (
        f'<marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5.5" '
        f'markerHeight="5.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
        f'fill="{color}"/></marker>'
    )


# Default grey arrowhead + the two tinted heads the widget swaps in (via CSS marker-end) for
# the directional causal trace: upstream = blue, downstream = amber. The static renderer never
# applies the trace classes, so it always shows the grey head — parity is unaffected.
_DEFS = (
    "<defs>"
    + _arrow("bd-arrow", "#555")
    + _arrow("bd-arrow-up", "#2563eb")
    + _arrow("bd-arrow-down", "#d97706")
    + "</defs>"
)


_LABEL_DEFS_RE = re.compile(r"<defs>(.*?)</defs>", re.S)
_GLYPH_PATH_RE = re.compile(r'<path id="([^"]+)" d="([^"]*)"\s*(?:/>|>\s*</path>)')


def _hoist_label_defs(s: str, shared: dict[str, str]) -> str:
    """Move an embedded label's MathJax font defs into ONE shared document ``<defs>``,
    rewriting glyph ids to content-hashed ones so identical glyphs collapse across labels
    (MathJax's per-equation ``MJX-<n>-`` id prefix makes every label carry duplicate
    ``<path>`` data — measured >50% of output bytes on multi-node models). A label whose
    defs contain anything but bare glyph paths is left untouched — never guess."""
    m = _LABEL_DEFS_RE.search(s)
    if not m:
        return s
    block = m.group(1)
    if _GLYPH_PATH_RE.sub("", block).strip():
        return s  # unexpected defs content: keep this label's local defs
    out = s[: m.start()] + s[m.end() :]
    for pid, d in _GLYPH_PATH_RE.findall(block):
        nid = "bdg-" + hashlib.sha1(d.encode()).hexdigest()[:10]
        shared[nid] = d
        out = out.replace(f'xlink:href="#{pid}"', f'xlink:href="#{nid}"')
        out = out.replace(f'href="#{pid}"', f'href="#{nid}"')
    return out


def _embed_label(
    label_svg: str, x: float, y: float, w: float, h: float, shared: dict[str, str] | None = None
) -> str:
    s = re.sub(r'width="[\d.]+ex"', f'width="{w:.1f}"', label_svg, count=1)
    s = re.sub(r'height="[\d.]+ex"', f'height="{h:.1f}"', s, count=1)
    s = s.replace("<svg ", f'<svg x="{x:.1f}" y="{y:.1f}" ', 1)
    if shared is not None:
        s = _hoist_label_defs(s, shared)
    return s


def _node_chrome(n: NodeIR, b: Box) -> str:
    fill, stroke, rx = _CHROME.get(n.role, _CHROME["latent"])
    dash = ' stroke-dasharray="4,3"' if n.role in ("potential", "factor") else ""
    sw = 1.1 if n.role == "deterministic" else 1.4  # the deterministic box reads as secondary
    # class "bd-chrome" lets the widget highlight ONLY the box outline (never the equation/glyph).
    return (
        f'<rect class="bd-chrome" x="{b.x:.1f}" y="{b.y:.1f}" width="{b.w:.1f}" height="{b.h:.1f}" '
        f'rx="{rx}" ry="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>'
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
    """A plate's dashed border + corner label, plus an invisible hit band along the border.

    The click target is deliberately the BORDER, not the interior: a plate usually encloses most
    of the canvas, so an interior hit area would swallow every "click empty space to close" the
    pinned card advertises. The visible rect is inert (``pointer-events="none"``); the hit rect is
    a transparent ~10px stroke on the same box, so interior clicks fall through to the background.
    """
    return (
        f'<rect x="{b.x:.1f}" y="{b.y:.1f}" width="{b.w:.1f}" height="{b.h:.1f}" rx="6" ry="6" '
        f'fill="none" stroke="#9aa0a6" stroke-width="1" stroke-dasharray="2,2" '
        f'pointer-events="none"/>'
        f'<rect x="{b.x:.1f}" y="{b.y:.1f}" width="{b.w:.1f}" height="{b.h:.1f}" rx="6" ry="6" '
        f'fill="none" stroke="#000" stroke-opacity="0" stroke-width="10" pointer-events="stroke"/>'
        f'<text x="{b.x + b.w - 4:.1f}" y="{b.y + b.h - 5:.1f}" text-anchor="end" '
        f'font-size="11" fill="#6b7075">{escape(label)}</text>'
    )


def _panel_curve(xs: list[float], ys: list[float], box: Box, color: str = "#3a6ea5") -> str:
    x0, x1 = xs[0], xs[-1]
    span = (x1 - x0) or 1.0
    pts = [
        (box.x + (x - x0) / span * box.w, box.y + box.h - max(0.0, min(1.0, y)) * box.h)
        for x, y in zip(xs, ys, strict=False)
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
        f"prior predictive — {escape(expansion.get('label', ''))}</text>",
    ]
    y = pad + 22
    for mem in members:
        cap = f" (showing {len(mem['curves'])})" if mem.get("capped") else ""
        obs_note = "  ·  orange ticks = observed data" if mem.get("observed") else ""
        out.append(
            f'<text x="{pad}" y="{y + 11:.1f}" font-size="11" fill="#555">'
            f"{escape(mem['id'])} — {mem['n']} instances{cap}{obs_note}</text>"
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


def render_observed_panel(node_id: str, dist: str | None, glyph_data: dict) -> str:
    """Standalone SVG for the widget's pinned card. Continuous likelihoods show the data histogram
    with the MLE best-fit family density overlaid (a HEDGED shape comparison, never a goodness-of-fit
    verdict); discrete likelihoods show per-class proportion bars. Both carry an x-axis + an n/fit
    summary."""
    heights = glyph_data.get("heights")  # discrete pmf bars
    edges, counts = glyph_data.get("edges"), glyph_data.get("counts")  # continuous histogram
    if not heights and not (edges and counts):
        return ""
    overlay = glyph_data.get("overlay")
    fit = glyph_data.get("fit") or {}
    n = fit.get("n") or glyph_data.get("n")
    pad, title_h, ph, axis_h, text_h = 12.0, 20.0, 96.0, 18.0, 14.0
    pw = 300.0
    w = pw + 2 * pad
    plot = Box(pad, pad + title_h, pw, ph)
    sub_y = plot.y + ph + axis_h + 11
    cap_y = sub_y + text_h
    h = cap_y + pad - 4
    n_str = f" (n={n})" if n else ""

    def _fmt(v) -> str:
        return str(int(v)) if float(v).is_integer() else f"{float(v):.3g}"

    if heights:  # discrete: one bar per class
        kind = "bars"
        title = f"{escape(node_id)} — observed data{n_str}"
        sub = "discrete likelihood — observed class proportions"
        caption = ""
        cats = glyph_data.get("cats") or list(range(len(heights)))
        k = len(cats)
        _, centers = glyph_bar_layout(k, plot)  # tick under each bar center (kept in sync)
        idxs = range(k) if k <= 10 else (0, k // 2, k - 1)
        ticks = [((centers[i] - plot.x) / plot.w, cats[i]) for i in idxs]
    elif overlay and (fit.get("family") or dist):  # continuous + best-fit curve
        kind = "hist_overlay"
        fam = fit.get("family") or dist
        title = f"{escape(node_id)} — best-fit {escape(fam)}{n_str}"
        sub = fit.get("params") or ""
        caption = "shape check: data vs best-fit family — not a goodness-of-fit test"
        x0, x1 = float(edges[0]), float(edges[-1])
        ticks = [(0.0, x0), (0.5, 0.5 * (x0 + x1)), (1.0, x1)]
    else:  # continuous, no fit available
        kind = "histogram"
        title = f"{escape(node_id)} — observed data{n_str}"
        sub = "best-fit overlay n/a (unfittable likelihood)"
        caption = ""
        x0, x1 = float(edges[0]), float(edges[-1])
        ticks = [(0.0, x0), (0.5, 0.5 * (x0 + x1)), (1.0, x1)]

    bstroke, bfill = _GLYPH_COLORS["observed_hist"]
    axis_y = plot.y + plot.h
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" '
        f'viewBox="0 0 {w:.0f} {h:.0f}" font-family="system-ui, sans-serif">',
        f'<rect width="{w:.0f}" height="{h:.0f}" rx="8" fill="#ffffff" stroke="#c7c7cc"/>',
        f'<text x="{pad:.0f}" y="{pad + 12:.0f}" font-size="12" font-weight="600" fill="#333">{title}</text>',
        f'<rect x="{plot.x:.1f}" y="{plot.y:.1f}" width="{plot.w:.1f}" height="{plot.h:.1f}" '
        'fill="#fafafa" stroke="#eeeeee"/>',
        glyph.render(kind, glyph_data, plot, stroke=bstroke, fill=bfill, overlay=_OVERLAY),
    ]
    for frac, val in ticks:
        px = plot.x + frac * plot.w
        anchor = "start" if frac < 0.05 else ("end" if frac > 0.95 else "middle")
        out.append(
            f'<line x1="{px:.1f}" y1="{axis_y:.1f}" x2="{px:.1f}" y2="{axis_y + 4:.1f}" '
            'stroke="#999" stroke-width="0.8"/>'
        )
        out.append(
            f'<text x="{px:.1f}" y="{axis_y + 14:.1f}" text-anchor="{anchor}" font-size="9" '
            f'fill="#777">{_fmt(val)}</text>'
        )
    if sub:
        out.append(
            f'<text x="{pad:.0f}" y="{sub_y:.1f}" font-size="10" fill="#555">{escape(sub)}</text>'
        )
    if caption:
        out.append(
            f'<text x="{pad:.0f}" y="{cap_y:.1f}" font-size="9" fill="#9aa0a6">{caption}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def _panel_coord_labels(n) -> list[str]:
    """Row labels for a matrix-shaped panel (heatmap rows / pairplot diagonal): the coords of
    whichever node dim matches the matrix size exactly; [] when nothing matches (no guess)."""
    data = n.glyph_data or {}
    m = data.get("matrix") or data.get("cov")
    if not m:
        return []
    k = len(m)
    for d in n.dims or ():
        vals = (n.coords or {}).get(d)
        if vals is not None and len(vals) == k:
            return [str(v) for v in vals]
    return []


def render_node_panel(n) -> str:
    """Standalone SVG for the widget's pinned card of ANY glyph-bearing node: the node's
    existing ``glyph_data`` drawn large through the same registry as the in-node glyph, plus
    the hedged source caption (single-sourced from the legend wording) and coordinate row
    labels where a node dim matches the matrix size. Widget-only — the static SVG never
    carries panels, so renderer parity is untouched."""
    if not (n.glyph and n.glyph_data):
        return ""
    from .legend import _SOURCE_LABELS

    square = n.glyph.kind in ("pairplot", "heatmap")
    pad, title_h, label_w = 12.0, 20.0, 0.0
    coord_labels = _panel_coord_labels(n) if square else []
    if coord_labels:
        label_w = 6.0 * min(10, max(len(s) for s in coord_labels)) + 6.0
    pw, ph = (220.0, 220.0) if square else (300.0, 140.0)
    w = pad * 2 + label_w + pw
    plot = Box(pad + label_w, pad + title_h, pw, ph)
    cap_y = plot.y + ph + 16.0
    h = cap_y + pad - 4.0
    stroke, fill = _GLYPH_COLORS.get(n.glyph.source, ("#2a8a55", "#2a8a55"))
    title = escape(n.id) + (f" ~ {escape(n.dist)}" if n.dist else "")
    caption = escape(_SOURCE_LABELS.get(n.glyph.source, n.glyph.source))
    pooled = (n.glyph_data or {}).get("pooled")
    if pooled:  # one density built from ALL elements' draws — not any single element's marginal
        caption += f" · pooled over {int(pooled)} elements"
    spikes = (n.glyph_data or {}).get("spikes") or []
    ps = [sp["p"] for sp in spikes if isinstance(sp, dict) and sp.get("p") is not None]
    if ps:  # the bars are deliberately exaggerated to be visible — give the real numbers
        caption += " · censored mass " + ", ".join(f"{p:.1%}" for p in ps)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" '
        f'viewBox="0 0 {w:.0f} {h:.0f}" font-family="system-ui, sans-serif">',
        f'<rect width="{w:.0f}" height="{h:.0f}" rx="8" fill="#ffffff" stroke="#c7c7cc"/>',
        f'<text x="{pad:.0f}" y="{pad + 12:.0f}" font-size="12" font-weight="600" '
        f'fill="#333">{title}</text>',
        f'<rect x="{plot.x:.1f}" y="{plot.y:.1f}" width="{plot.w:.1f}" height="{plot.h:.1f}" '
        'fill="#fafafa" stroke="#eeeeee"/>',
        glyph.render(n.glyph.kind, n.glyph_data, plot, stroke=stroke, fill=fill, overlay=_OVERLAY),
    ]
    if coord_labels:
        ch = plot.h / len(coord_labels)
        for i, s in enumerate(coord_labels):
            s = s if len(s) <= 10 else s[:9] + "…"
            out.append(
                f'<text x="{plot.x - 4:.1f}" y="{plot.y + (i + 0.5) * ch + 3:.1f}" '
                f'text-anchor="end" font-size="8" fill="#777">{escape(s)}</text>'
            )
    out.append(f'<text x="{pad:.0f}" y="{cap_y:.1f}" font-size="9" fill="#9aa0a6">{caption}</text>')
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
        if src == "best_fit":
            return glyph.render(
                "density", {"xs": _BELL[0], "ys": _BELL[1]}, b, stroke=_OVERLAY, fill=_OVERLAY
            )
        if src == "observed_hist":
            data = {"edges": [0, 1, 2, 3], "counts": [0.6, 1.0, 0.45]}
            return glyph.render("histogram", data, b, fill=fill, stroke=stroke)
        if src == "deterministic_fn":
            scurve = {"xs": [0.0, 0.25, 0.5, 0.75, 1.0], "ys": [0.05, 0.2, 0.5, 0.8, 0.95]}
            return glyph.render("curve", scurve, b, stroke=stroke, fill=fill)
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


def to_svg(ir: ModelIR, layout: LayoutResult, *, legend: bool = True) -> str:
    # `Box` is a dataclass and therefore always truthy — test the DIMENSIONS, so an empty model
    # gets the intended placeholder canvas instead of a 0x0 SVG
    c = layout.canvas
    if c is None or c.w <= 0 or c.h <= 0:
        c = Box(0, 0, 100, 100)
    body = [_DEFS]
    # plates (behind everything), tagged for click-to-expand
    for p in ir.plates:
        b = layout.plate_boxes.get(p.id)
        if b:
            body.append(
                f'<g class="bd-plate" data-plate="{escape(p.id)}">' + _plate(b, p.label) + "</g>"
            )
    # Two passes so a later node's (opaque) chrome box can never paint over an earlier node's
    # label/glyph: ALL chrome first, then ALL labels+glyphs on top. Each pass tags its group with
    # data-node, and the widget keys hover/selection off data-node (js/index.js), so splitting a
    # node across two groups is transparent to interactivity. Parity holds: one emitter, both
    # renderers consume this verbatim.
    # the LayoutResult is authoritative; `n.box` is only a convenience mirror of it
    drawn = [(n, b) for n in ir.nodes if (b := (layout.node_boxes.get(n.id) or n.box)) is not None]
    for n, b in drawn:  # chrome (boxes) behind everything else node-ish
        body.append(f'<g class="bd-node" data-node="{escape(n.id)}">' + _node_chrome(n, b) + "</g>")
    font_defs: dict[str, str] = {}  # content-hashed glyph id -> path data (shared across labels)
    for n, b in drawn:  # labels + glyphs above all chrome
        if n.label_svg:
            lw, lh = geometry.label_px_size(n.label_svg)
            ox, oy = geometry.label_origin(b, lw)
            parts = [_embed_label(n.label_svg, ox, oy, lw, lh, shared=font_defs)]
        else:
            parts = [
                f'<text x="{b.x + b.w / 2:.1f}" y="{b.y + b.h / 2 + 4:.1f}" text-anchor="middle" '
                f'font-size="12" fill="#222">{escape(n.id)}</text>'
            ]
        if n.glyph and n.glyph_data:
            _, lh = geometry.label_px_size(n.label_svg)
            gr = geometry.glyph_rect(b, lh, n.glyph.kind, n.glyph_data)
            if gr:
                stroke, fill = _GLYPH_COLORS.get(n.glyph.source, ("#2a8a55", "#2a8a55"))
                parts.append(glyph.render(n.glyph.kind, n.glyph_data, gr, stroke=stroke, fill=fill))
        if getattr(n, "elision_reason", None):  # honesty badge: undrawable construct
            reason = (
                n.elision_reason if len(n.elision_reason) <= 30 else n.elision_reason[:29] + "…"
            )
            parts.append(
                f'<text x="{b.x + 7:.1f}" y="{b.y + b.h - 5:.1f}" font-size="8" fill="#8a6d3b" '
                f'font-style="italic">⚠ {escape(reason)}</text>'
            )
        body.append(f'<g class="bd-node" data-node="{escape(n.id)}">' + "".join(parts) + "</g>")
    if font_defs:  # one shared <defs> for all hoisted MathJax glyphs (sorted: deterministic bytes)
        body.insert(
            1,
            "<defs>"
            + "".join(f'<path id="{i}" d="{d}"></path>' for i, d in sorted(font_defs.items()))
            + "</defs>",
        )
    # edges on top so token-anchored arrowheads into equations are visible
    for e in ir.edges:
        pts = layout.edge_paths.get((e.source, e.target))
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
