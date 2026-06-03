"""Backend-agnostic layout helpers shared by every ``LayoutBackend``.

The label measurement, token-anchor projection, and the smooth fallback edge are identical
regardless of which engine (ELK, dot, …) decides node *positions* — keeping them here is
what lets the backends differ only in placement while the rest stays byte-identical (the
parity principle). Token-level port anchors are computed by us from the MathJax bboxes, so
they are engine-independent.
"""

from __future__ import annotations

from .. import geometry, mathsvg
from ..ir import Box, ModelIR


def render_labels(ir: ModelIR) -> dict[str, dict]:
    """Render each node's label to SVG (set ``node.label_svg``) and collect px size +
    fractional token bboxes. Falls back to a size estimate when math isn't available."""
    renderer = mathsvg.get_renderer()
    use = renderer.available
    info: dict[str, dict] = {}
    for n in ir.nodes:
        svg = None
        bboxes: dict[str, tuple[float, float, float, float]] = {}
        if use and n.label_tex:
            try:
                svg = renderer.render(n.label_tex, display=True)
                bboxes = mathsvg.token_bboxes(svg)
            except Exception:
                svg, bboxes = None, {}
        n.label_svg = svg
        lw, lh = geometry.label_px_size(svg)
        if svg is None and n.label_tex:
            lw = max(lw, 7.0 * len(n.id))  # rough estimate without math
        info[n.id] = {"w": lw, "h": lh, "bboxes": bboxes}
    return info


def node_token_anchors(box: Box, label_w: float, label_h: float, bboxes: dict) -> dict[str, Box]:
    """Project the label's fractional token bboxes into absolute boxes within ``box`` — the
    anchor a port-edge terminates on (token top-center, with a standoff applied at draw)."""
    ox, oy = geometry.label_origin(box, label_w, label_h)
    anchors: dict[str, Box] = {}
    for tok, (fx, fy, fw, fh) in bboxes.items():
        anchors[tok] = Box(ox + fx * label_w, oy + fy * label_h, fw * label_w, fh * label_h)
    return anchors


def smooth_polyline(pts: list) -> list[list[float]]:
    """Catmull-Rom -> cubic-Bezier chain (``[p0, c1, c2, p1, c1, c2, p2, …]``) so a routed
    polyline (ELK's bend-points) renders as one smooth path with no faceting. The chain format
    is exactly what ``render_svg._edge`` consumes. Collinear/duplicate points are dropped so a
    straight run stays straight (and the final arrowhead orients along the real last segment)."""
    P: list[list[float]] = []
    for x, y in pts:
        p = [float(x), float(y)]
        if not P or abs(p[0] - P[-1][0]) > 0.01 or abs(p[1] - P[-1][1]) > 0.01:
            P.append(p)
    if len(P) < 2:
        return P
    n = len(P)
    out = [P[0]]
    for i in range(n - 1):
        p_prev = P[i - 1] if i > 0 else P[i]
        p0, p1 = P[i], P[i + 1]
        p_next = P[i + 2] if i + 2 < n else P[i + 1]
        out.append([p0[0] + (p1[0] - p_prev[0]) / 6.0, p0[1] + (p1[1] - p_prev[1]) / 6.0])
        out.append([p1[0] - (p_next[0] - p0[0]) / 6.0, p1[1] - (p_next[1] - p0[1]) / 6.0])
        out.append([p1[0], p1[1]])
    return out


def _target_point(tb: Box, anchor: Box | None) -> tuple[float, float]:
    if anchor is not None:
        return anchor.x + anchor.w / 2.0, anchor.y - geometry.STANDOFF
    return tb.x + tb.w / 2.0, tb.y


def simple_edge_path(sb: Box, tb: Box, anchor: Box | None) -> list[list[float]]:
    """A gentle cubic from the source bottom-center to the target token. The edge leaves the
    source heading down, and **approaches the token along the source->token line** so the
    arrowhead orients with the line (no vertical kink at the tip). Ends a standoff above the
    glyph so the arrowhead doesn't cover it. Returned as a 4-point cubic ``[p0, c1, c2, p1]``."""
    ex, ey = sb.x + sb.w / 2.0, sb.y + sb.h
    nx, ny = _target_point(tb, anchor)
    dx, dy = nx - ex, ny - ey
    dist = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / dist, dy / dist
    k = max(14.0, min(0.32 * dist, 44.0))
    c1 = [ex + 0.18 * dx, ey + max(12.0, min(0.4 * abs(dy), 30.0))]  # set off downward, slight lean
    c2 = [nx - ux * k, ny - uy * k]  # pull the end handle back ALONG the approach -> aligned tip
    return [[ex, ey], c1, c2, [nx, ny]]


def _cubic4_points(pts: list, n: int = 26) -> list[tuple[float, float]]:
    """Sample a 4-point cubic ``[p0, c1, c2, p3]`` into points along the curve."""
    p0, c1, c2, p3 = pts
    out = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        out.append(
            (
                mt**3 * p0[0] + 3 * mt**2 * t * c1[0] + 3 * mt * t * t * c2[0] + t**3 * p3[0],
                mt**3 * p0[1] + 3 * mt**2 * t * c1[1] + 3 * mt * t * t * c2[1] + t**3 * p3[1],
            )
        )
    return out


def routed_edge_path(sb: Box, tb: Box, anchor: Box | None, obstacles: list) -> list[list[float]]:
    """The direct edge (``simple_edge_path``) when its rendered CURVE is clear; otherwise route
    DOWN a clear vertical channel to one side of the blocking node(s), past their full height,
    then into the target token from below — so the edge never crosses a node. Plate boxes are
    NOT obstacles (edges are meant to cross plate boundaries) — pass node boxes only.

    We test the actual curve (not the straight chord), so a gentle bend that already clears a
    corner doesn't provoke a needless detour, while a curve that bulges into a node does."""
    ex, ey = sb.x + sb.w / 2.0, sb.y + sb.h
    nx, ny = _target_point(tb, anchor)
    if abs(ny - ey) < 1.0:
        return simple_edge_path(sb, tb, anchor)

    direct = simple_edge_path(sb, tb, anchor)
    pts = _cubic4_points(direct)[2:-2]  # ignore the endpoints (they touch source/token)
    clippers = [
        o
        for o in obstacles
        if any(o.x <= x <= o.x + o.w and o.y <= y <= o.y + o.h for x, y in pts)
    ]
    if not clippers:
        return direct

    margin = 16.0
    cx = sum(o.x + o.w / 2.0 for o in clippers) / len(clippers)
    # channel down the side the source is already on (no need to cross over the obstacle)
    if ex >= cx:
        channel = max(o.x + o.w for o in clippers) + margin
    else:
        channel = min(o.x for o in clippers) - margin
    top = max(min(o.y for o in clippers) - margin, ey + 8.0)
    bot = min(max(o.y + o.h for o in clippers) + margin, ny - 8.0)
    if bot <= top:  # obstacles fill the whole gap -> fall back to a direct edge
        return simple_edge_path(sb, tb, anchor)
    # end at the token (no vertical stub) so the arrowhead aligns with the (channel,bot)->token approach
    return smooth_polyline([[ex, ey], [channel, top], [channel, bot], [nx, ny]])
