"""Built-in glyph kinds (each renders precomputed shape data into an absolute px box).

``density``      data = {"xs": [...], "ys": [...]}   (ys normalized to 0..1)
``histogram``    data = {"edges": [...], "counts": [...]}  (counts normalized to 0..1)
``hist_overlay`` data = {"edges","counts","overlay":{"xs","ys"}}  (histogram + best-fit curve, shared scale)
``bars``         data = {"cats":[...], "heights":[...]}  (discrete pmf: one slot-centered bar per class)
``schematic``    data = {"xs","ys"} drawn faint/dashed (family shape only)
``heatmap``      data = {"matrix": [[...]]}  (covariance/correlation/adjacency matrices)
``fan``          data = {"mid","lo","hi"}  (random-walk diffusion band)
``pairplot``     data = {"cov": [[...]]}  (low-dim multivariate: marginals + covariance ellipses)
``mixture``      data = {"curves":[{xs,ys}], "base":{cats,heights}, "spike": float}  (mixtures / zero-inflated)
``cutpoints``    data = {"probs":[...], "cutpoints":[...]}  (ordinal)
``simplex``      data = {"curves":[{xs,ys}]}  (Dirichlet marginal Beta curves, filled)
``stem``         data = {"lags":[...], "values":[...]}  (PACF / autocorrelation stems, +/-)
``censored``     data = {"xs","ys","spikes":[{x,h}]}  (base density + bound mass-spikes)
"""

from __future__ import annotations

import math
from typing import Any

from ..ir import Box
from .registry import register


def _path(pts) -> str:
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)


def _poly(xs, ys, box: Box):
    x0, x1 = min(xs), max(xs)
    span = (x1 - x0) or 1.0

    def sx(x):
        return box.x + (x - x0) / span * box.w

    def sy(y):
        return box.y + box.h - max(0.0, min(1.0, y)) * box.h

    return [(sx(x), sy(y)) for x, y in zip(xs, ys)]


def _zero_marker(box: Box, x0: float, x1: float) -> str:
    """A faint dashed vertical reference at x=0, when 0 falls inside the rendered range. Glyphs are
    self-scaled, so this restores the key landmark — sign / centering (e.g. ZeroSumNormal, a
    coefficient straddling 0, an LKJ correlation on [-1, 1])."""
    if x1 <= x0 or not (x0 < 0.0 < x1):
        return ""
    px = box.x + (0.0 - x0) / (x1 - x0) * box.w
    return (
        f'<line x1="{px:.1f}" y1="{box.y:.1f}" x2="{px:.1f}" y2="{box.y + box.h:.1f}" '
        'stroke="#b0b0b0" stroke-width="0.6" stroke-dasharray="2,2"/>'
    )


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
        _zero_marker(box, min(xs), max(xs))
        + f'<path d="{area}" fill="{fill}" fill-opacity="{fill_opacity}" stroke="none"/>'
        + f'<path d="{line}" fill="none" stroke="{stroke}" stroke-width="1.3"{dash}/>'
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
    return _zero_marker(box, edges[0], edges[-1]) + _bars(edges, counts, box, fill, stroke)


def render_hist_overlay(data: dict[str, Any], box: Box, *, fill="#5a7fb5", stroke="#3a5f95", overlay="#c0392b", **_):
    """Observed-data histogram (bars) + the MLE best-fit family density (line) on a SHARED scale.
    A poor family choice shows up as a curve that misses the bars; a good one tracks them."""
    edges, counts = data.get("edges"), data.get("counts")
    if not edges or not counts:
        return ""
    out = _zero_marker(box, edges[0], edges[-1]) + _bars(edges, counts, box, fill, stroke)
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
    cells = [  # subtle frame so the matrix reads as a block
        f'<rect x="{box.x:.1f}" y="{box.y:.1f}" width="{box.w:.1f}" height="{box.h:.1f}" '
        'fill="none" stroke="#cfcfcf" stroke-width="0.5"/>'
    ]
    for i, row in enumerate(m):
        for j, v in enumerate(row):
            t = max(0.0, min(1.0, v))  # sequential white -> blue ramp
            r, g, b = int(255 - 213 * t), int(255 - 162 * t), int(255 - 112 * t)
            cells.append(
                f'<rect x="{box.x + j * cw:.1f}" y="{box.y + i * ch:.1f}" width="{cw + 0.3:.1f}" '
                f'height="{ch + 0.3:.1f}" fill="rgb({r},{g},{b})"/>'
            )
    return "".join(cells)


def render_fan(data: dict[str, Any], box: Box, *, stroke="#3a6ea5", fill="#3a6ea5", **_):
    """Fan chart for a diffusion process (random walk): a median line + a band that widens over
    steps. ``mid``/``lo``/``hi`` are y-values in 0..1 (shared scale, computed by the adapter)."""
    mid, lo, hi = data.get("mid"), data.get("lo"), data.get("hi")
    if not mid or not lo or not hi:
        return ""
    n = len(mid)

    def sx(i):
        return box.x + (i / (n - 1) if n > 1 else 0.0) * box.w

    def sy(y):
        return box.y + box.h - max(0.0, min(1.0, y)) * box.h

    top = [(sx(i), sy(hi[i])) for i in range(n)]
    bot = [(sx(i), sy(lo[i])) for i in range(n - 1, -1, -1)]
    band = _path(top + bot) + " Z"
    line = _path([(sx(i), sy(mid[i])) for i in range(n)])
    return (
        f'<path d="{band}" fill="{fill}" fill-opacity="0.18" stroke="none"/>'
        f'<path d="{line}" fill="none" stroke="{stroke}" stroke-width="1.3"/>'
    )


def render_pairplot(data: dict[str, Any], box: Box, *, stroke="#2a8a55", fill="#2a8a55", **_):
    """Low-dim multivariate corner plot: diagonal marginal bumps + lower-triangle covariance
    ellipses (1-sd, self-scaled per cell so tilt/elongation read the correlation). ``cov`` is the
    numeric covariance matrix."""
    cov = data.get("cov")
    if not cov or len(cov) < 2:
        return ""
    d = len(cov)
    cw, ch = box.w / d, box.h / d
    out = []
    bump = [math.exp(-((-3 + 6 * k / 23) ** 2) / 2) for k in range(24)]
    bm = max(bump)
    for i in range(d):
        for j in range(d):
            x0, y0 = box.x + j * cw, box.y + i * ch
            if i == j:
                pts = [(x0 + (k / 23) * cw, y0 + ch - (bump[k] / bm) * ch * 0.9) for k in range(24)]
                out.append(f'<path d="{_path(pts)}" fill="none" stroke="{stroke}" stroke-width="0.8"/>')
            elif i > j:
                a, b, c = cov[i][i], cov[i][j], cov[j][j]
                tr = a + c
                disc = math.sqrt(max(tr * tr / 4 - (a * c - b * b), 0.0))
                s1, s2 = math.sqrt(max(tr / 2 + disc, 1e-12)), math.sqrt(max(tr / 2 - disc, 1e-12))
                smax = max(s1, s2) or 1.0
                rx, ry = (s1 / smax) * 0.38 * min(cw, ch), (s2 / smax) * 0.38 * min(cw, ch)
                ecx, ecy = x0 + cw / 2, y0 + ch / 2
                deg = -math.degrees(0.5 * math.atan2(2 * b, a - c))
                out.append(
                    f'<ellipse cx="{ecx:.1f}" cy="{ecy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" '
                    f'transform="rotate({deg:.1f} {ecx:.1f} {ecy:.1f})" fill="{fill}" '
                    f'fill-opacity="0.15" stroke="{stroke}" stroke-width="0.8"/>'
                )
    return "".join(out)


def render_mixture(data: dict[str, Any], box: Box, *, stroke="#2a8a55", fill="#2a8a55", **_):
    """Composite: overlaid component densities (continuous mixture) and/or base pmf bars + a
    zero-inflation mass spike at the left (zero-inflated/hurdle counts)."""
    out = []
    base = data.get("base")
    if base:
        out.append(render_bars(base, box, fill=fill, stroke=stroke))
    curves = data.get("curves", [])
    if curves and curves[0].get("xs"):  # shared x-range across overlaid components
        xs0 = curves[0]["xs"]
        out.append(_zero_marker(box, min(xs0), max(xs0)))
    for comp in curves:
        xs, ys = comp.get("xs"), comp.get("ys")
        if xs and ys:
            out.append(f'<path d="{_path(_poly(xs, ys, box))}" fill="none" stroke="{stroke}" stroke-width="1.0" opacity="0.8"/>')
    spike = data.get("spike")
    if spike:
        h = max(0.0, min(1.0, spike)) * box.h
        out.append(f'<rect x="{box.x:.1f}" y="{box.y + box.h - h:.1f}" width="5" height="{h:.1f}" fill="{stroke}"/>')
    return "".join(out)


def render_cutpoints(data: dict[str, Any], box: Box, *, fill="#5a7fb5", stroke="#3a5f95", **_):
    """Ordinal: per-category probability bars + the latent-scale cutpoint ticks on a baseline."""
    out = []
    probs = data.get("probs")
    if probs:
        m = max(probs) or 1.0
        out.append(render_bars({"cats": list(range(len(probs))), "heights": [p / m for p in probs]}, box, fill=fill, stroke=stroke))
    cuts = data.get("cutpoints")
    if cuts:
        lo, hi = min(cuts), max(cuts)
        span = (hi - lo) or 1.0
        y = box.y + 2.0
        out.append(f'<line x1="{box.x:.1f}" y1="{y:.1f}" x2="{box.x + box.w:.1f}" y2="{y:.1f}" stroke="#999" stroke-width="0.6"/>')
        for cpt in cuts:
            cx = box.x + (cpt - lo) / span * box.w
            out.append(f'<line x1="{cx:.1f}" y1="{y - 2:.1f}" x2="{cx:.1f}" y2="{y + 3:.1f}" stroke="#666" stroke-width="0.8"/>')
    return "".join(out)


def render_simplex(data: dict[str, Any], box: Box, *, stroke="#2a8a55", fill="#2a8a55", **_):
    """Dirichlet & friends: overlaid marginal Beta curves (one per component) on [0, 1], each lightly
    filled so even a flat (uniform) marginal reads as a density band rather than a bare line."""
    curves = data.get("curves")
    if not curves:
        return ""
    base = box.y + box.h
    out = []
    for crv in curves:
        xs, ys = crv.get("xs"), crv.get("ys")
        if not (xs and ys):
            continue
        pts = _poly(xs, ys, box)
        area = (
            f"M{pts[0][0]:.1f},{base:.1f} L"
            + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            + f" L{pts[-1][0]:.1f},{base:.1f} Z"
        )
        out.append(f'<path d="{area}" fill="{fill}" fill-opacity="0.12" stroke="none"/>')
        out.append(f'<path d="{_path(pts)}" fill="none" stroke="{stroke}" stroke-width="0.9" opacity="0.85"/>')
    return "".join(out)


def render_stem(data: dict[str, Any], box: Box, *, stroke="#3a5f95", fill="#3a5f95", **_):
    """Stem plot (e.g. an AR partial-autocorrelation function): one stem per lag from a zero
    baseline, handling +/- values. The AR(p) fingerprint is a sharp cutoff after lag p."""
    vals = data.get("values")
    if not vals:
        return ""
    n = len(vals)
    mx = max(1e-9, max(abs(v) for v in vals))
    base = box.y + box.h / 2.0
    out = [
        f'<line x1="{box.x:.1f}" y1="{base:.1f}" x2="{box.x + box.w:.1f}" y2="{base:.1f}" '
        'stroke="#bbb" stroke-width="0.5"/>'
    ]
    for i, v in enumerate(vals):
        cx = box.x + (i + 0.5) / n * box.w
        y = base - (v / mx) * (box.h / 2.0 - 2.0)
        out.append(f'<line x1="{cx:.1f}" y1="{base:.1f}" x2="{cx:.1f}" y2="{y:.1f}" stroke="{stroke}" stroke-width="1.4"/>')
        out.append(f'<circle cx="{cx:.1f}" cy="{y:.1f}" r="1.5" fill="{stroke}"/>')
    return "".join(out)


def render_censored(data: dict[str, Any], box: Box, *, stroke="#2a8a55", fill="#2a8a55", **_):
    """Censored: the base density plus probability-mass spikes piled at the censoring bounds.
    ``spikes`` = list of {x: fraction across box, h: 0..1}."""
    out = [render_density(data, box, stroke=stroke, fill=fill)]
    for sp in data.get("spikes", []):
        x = sp.get("x")
        if x is None:
            continue
        px = box.x + max(0.0, min(1.0, x)) * box.w
        bh = max(0.0, min(1.0, sp.get("h", 1.0))) * box.h
        out.append(f'<rect x="{px - 2:.1f}" y="{box.y + box.h - bh:.1f}" width="4" height="{bh:.1f}" fill="{stroke}"/>')
    return "".join(out)


register("density", render_density)
register("schematic", render_schematic)
register("histogram", render_histogram)
register("hist_overlay", render_hist_overlay)
register("bars", render_bars)
register("heatmap", render_heatmap)
register("fan", render_fan)
register("pairplot", render_pairplot)
register("mixture", render_mixture)
register("cutpoints", render_cutpoints)
register("simplex", render_simplex)
register("stem", render_stem)
register("censored", render_censored)
