"""Built-in glyph kinds (each renders precomputed shape data into an absolute px box).

``density``      data = {"xs": [...], "ys": [...]}   (ys normalized to 0..1)
``histogram``    data = {"edges": [...], "counts": [...]}  (counts normalized to 0..1)
``hist_overlay`` data = {"edges","counts","overlay":{"xs","ys"}}  (histogram + best-fit curve, shared scale)
``bars``         data = {"cats":[...], "heights":[...]}  (discrete pmf: one slot-centered bar per class)
``schematic``    data = {"xs","ys"} drawn faint/dashed (family shape only)
``heatmap``      data = {"matrix": [[...]]}  (proves non-univariate kinds register the same way)
"""

from __future__ import annotations

from typing import Any

from ..ir import Box
from .registry import register


def _poly(xs, ys, box: Box):
    x0, x1 = min(xs), max(xs)
    span = (x1 - x0) or 1.0

    def sx(x):
        return box.x + (x - x0) / span * box.w

    def sy(y):
        return box.y + box.h - max(0.0, min(1.0, y)) * box.h

    return [(sx(x), sy(y)) for x, y in zip(xs, ys)]


def render_density(data: dict[str, Any], box: Box, *, stroke="#2a7", fill="#2a7", fill_opacity=0.18, dashed=False, **_):
    xs, ys = data.get("xs"), data.get("ys")
    if not xs or not ys:
        return ""
    pts = _poly(xs, ys, box)
    line = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    base = box.y + box.h
    area = (
        f"M{pts[0][0]:.1f},{base:.1f} L"
        + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        + f" L{pts[-1][0]:.1f},{base:.1f} Z"
    )
    dash = ' stroke-dasharray="3,2"' if dashed else ""
    return (
        f'<path d="{area}" fill="{fill}" fill-opacity="{fill_opacity}" stroke="none"/>'
        f'<path d="{line}" fill="none" stroke="{stroke}" stroke-width="1.3"{dash}/>'
    )


def render_schematic(data: dict[str, Any], box: Box, **_):
    # Family shape only: faint + dashed, with an "approx" cue.
    return render_density(data, box, stroke="#999", fill="#999", fill_opacity=0.08, dashed=True)


def _bars(edges, counts, box: Box, fill, stroke) -> str:
    x0, x1 = edges[0], edges[-1]
    span = (x1 - x0) or 1.0
    base = box.y + box.h
    out = []
    for i, c in enumerate(counts):
        bx = box.x + (edges[i] - x0) / span * box.w
        bw = max(0.5, (edges[i + 1] - edges[i]) / span * box.w - 1.0)
        bh = max(0.0, min(1.0, c)) * box.h
        out.append(f'<rect x="{bx:.1f}" y="{base - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="0.4"/>')
    return "".join(out)


def render_histogram(data: dict[str, Any], box: Box, *, fill="#5a7fb5", stroke="#3a5f95", **_):
    edges, counts = data.get("edges"), data.get("counts")
    if not edges or not counts:
        return ""
    return _bars(edges, counts, box, fill, stroke)


def render_hist_overlay(data: dict[str, Any], box: Box, *, fill="#5a7fb5", stroke="#3a5f95", overlay="#c0392b", **_):
    """Observed-data histogram (bars) + the MLE best-fit family density (line) on a SHARED scale.
    A poor family choice shows up as a curve that misses the bars; a good one tracks them."""
    edges, counts = data.get("edges"), data.get("counts")
    if not edges or not counts:
        return ""
    out = _bars(edges, counts, box, fill, stroke)
    ov = data.get("overlay") or {}
    xs, ys = ov.get("xs"), ov.get("ys")
    if xs and ys:
        pts = _poly(xs, ys, box)
        line = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        out += f'<path d="{line}" fill="none" stroke="{overlay}" stroke-width="1.4"/>'
    return out


def bar_layout(k: int, box: Box, gap_frac: float = 0.4) -> tuple[float, list[float]]:
    """Geometry for ``k`` discrete-class bars: a CENTERED group (bar width, list of center-x in
    absolute px). Few classes sit together near the middle (so 2-class Bernoulli reads as a tidy
    pair, not edge-pinned or quarter-spread); many classes shrink to fill the width like a
    histogram. Shared by ``render_bars`` and the card panel so its axis ticks stay bar-aligned."""
    k = max(1, k)
    bw = min(box.h * 0.9, box.w / (k + (k - 1) * gap_frac) * 0.9)  # square-ish cap; fill when dense
    gap = bw * gap_frac
    group_w = k * bw + (k - 1) * gap
    x0 = box.x + (box.w - group_w) / 2.0
    return bw, [x0 + i * (bw + gap) + bw / 2.0 for i in range(k)]


def render_bars(data: dict[str, Any], box: Box, *, fill="#5a7fb5", stroke="#3a5f95", **_):
    """Discrete pmf: one bar per integer class as a centered group (see ``bar_layout``). Heights
    are the class proportions (normalized to 0..1)."""
    heights = data.get("heights")
    if not heights:
        return ""
    bw, centers = bar_layout(len(heights), box)
    base = box.y + box.h
    out = []
    for hgt, cx in zip(heights, centers):
        bh = max(0.0, min(1.0, hgt)) * box.h
        out.append(
            f'<rect x="{cx - bw / 2:.1f}" y="{base - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="0.4"/>'
        )
    return "".join(out)


def render_heatmap(data: dict[str, Any], box: Box, **_):
    """Matrix glyph (covariance/correlation) — a non-univariate kind, proving the registry
    is glyph-agnostic. data = {"matrix": [[...], ...]} in 0..1."""
    m = data.get("matrix")
    if not m:
        return ""
    rows = len(m)
    cols = len(m[0]) if rows else 0
    if not cols:
        return ""
    cw, ch = box.w / cols, box.h / rows
    cells = []
    for i, row in enumerate(m):
        for j, v in enumerate(row):
            g = int(max(0.0, min(1.0, v)) * 255)
            cells.append(
                f'<rect x="{box.x + j * cw:.1f}" y="{box.y + i * ch:.1f}" width="{cw:.1f}" height="{ch:.1f}" '
                f'fill="rgb({255 - g},{255 - g},{g})"/>'
            )
    return "".join(cells)


register("density", render_density)
register("schematic", render_schematic)
register("histogram", render_histogram)
register("hist_overlay", render_hist_overlay)
register("bars", render_bars)
register("heatmap", render_heatmap)
