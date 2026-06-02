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


def simple_edge_path(sb: Box, tb: Box, anchor: Box | None) -> list[list[float]]:
    """A gentle cubic from the source bottom-center to the target (the specific token for a
    port-edge, else the top-center). The curve heads roughly STRAIGHT at the source and lands
    ~vertically on the target token (so the arrowhead sits just above the glyph, standoff).

    The tangent length is short and capped so a parent that's far to the SIDE gets a direct
    diagonal — not a long horizontal segment hugging the plate boundary on its way over.
    Returned as a 4-point cubic ``[p0, c1, c2, p1]`` (what ``render_svg._edge`` consumes)."""
    ex, ey = sb.x + sb.w / 2.0, sb.y + sb.h
    if anchor is not None:
        nx, ny = anchor.x + anchor.w / 2.0, anchor.y - geometry.STANDOFF
    else:
        nx, ny = tb.x + tb.w / 2.0, tb.y
    span = abs(ny - ey)
    k = max(12.0, min(0.35 * span, 26.0))  # short, capped -> direct (no boundary-hugging swoop)
    # lean the departure toward the target so the edge sets off in its general direction,
    # while the landing stays vertical for a clean arrowhead on the token.
    c1x = ex + 0.25 * (nx - ex)
    return [[ex, ey], [c1x, ey + k], [nx, ny - k], [nx, ny]]


def _segment_hits_box(p0: tuple, p1: tuple, box: Box, margin: float) -> bool:
    """True if the straight segment ``p0->p1`` passes through ``box`` expanded by ``margin``."""
    x0, y0 = p0
    x1, y1 = p1
    bx0, by0, bx1, by1 = box.x - margin, box.y - margin, box.x + box.w + margin, box.y + box.h + margin
    n = 24
    for i in range(n + 1):
        t = i / n
        x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        if bx0 <= x <= bx1 and by0 <= y <= by1:
            return True
    return False


def routed_edge_path(sb: Box, tb: Box, anchor: Box | None, obstacles: list) -> list[list[float]]:
    """The direct edge (``simple_edge_path``) when its straight path is clear; otherwise route
    DOWN a clear vertical channel to one side of the blocking node(s), past their full height,
    then into the target token from below — so the edge never crosses a node. Plate boxes are
    NOT obstacles (edges are meant to cross plate boundaries) — pass node boxes only."""
    ex, ey = sb.x + sb.w / 2.0, sb.y + sb.h
    if anchor is not None:
        nx, ny = anchor.x + anchor.w / 2.0, anchor.y - geometry.STANDOFF
    else:
        nx, ny = tb.x + tb.w / 2.0, tb.y
    if abs(ny - ey) < 1.0:
        return simple_edge_path(sb, tb, anchor)

    clippers = [o for o in obstacles if _segment_hits_box((ex, ey), (nx, ny), o, 2.0)]
    if not clippers:
        return simple_edge_path(sb, tb, anchor)

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
    stub = max(10.0, min(0.3 * abs(ny - bot), 20.0))
    return smooth_polyline([[ex, ey], [channel, top], [channel, bot], [nx, ny - stub], [nx, ny]])
