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
    """A gentle cubic from the source bottom-center to the target token, approaching ALONG the
    source->token line. This is the crossing-optimal *body*; the arrowhead is turned to point
    straight DOWN afterwards by ``reflow.prefer_vertical_tips`` wherever a vertical tip doesn't add
    a crossing (it edits only the final handle, leaving the body — and so every crossing /
    through-node count — untouched). Ends a standoff above the glyph so the arrowhead doesn't cover
    it. Returned as a 4-point cubic ``[p0, c1, c2, p1]``."""
    ex, ey = sb.x + sb.w / 2.0, sb.y + sb.h
    nx, ny = _target_point(tb, anchor)
    dx, dy = nx - ex, ny - ey
    dist = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / dist, dy / dist
    k = max(14.0, min(0.32 * dist, 44.0))
    c1 = [ex + 0.18 * dx, ey + max(12.0, min(0.4 * abs(dy), 30.0))]  # set off downward, slight lean
    c2 = [nx - ux * k, ny - uy * k]  # pull the end handle back ALONG the approach
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


def _chain_points(pts, n: int = 12) -> list[tuple[float, float]]:
    """Sample a cubic-Bezier CHAIN (``[p0, c1, c2, p1, c1, c2, p2, …]``) into points."""
    out = []
    if len(pts) >= 4 and (len(pts) - 1) % 3 == 0:
        for i in range(1, len(pts), 3):
            p0 = pts[0] if i == 1 else pts[i - 1]
            c1, c2, p3 = pts[i], pts[i + 1], pts[i + 2]
            for k in range(n + 1):
                t = k / n
                mt = 1 - t
                out.append(
                    (
                        mt**3 * p0[0] + 3 * mt**2 * t * c1[0] + 3 * mt * t * t * c2[0] + t**3 * p3[0],
                        mt**3 * p0[1] + 3 * mt**2 * t * c1[1] + 3 * mt * t * t * c2[1] + t**3 * p3[1],
                    )
                )
    return out


def _obstacle_hits(path: list, obstacles: list) -> int:
    """How many node-box interiors a path passes through (endpoints' touch excluded by the
    1px inset)."""
    s = _chain_points(path)[1:-1]
    return sum(
        1 for o in obstacles
        if any(o.x + 1 <= x <= o.x + o.w - 1 and o.y + 1 <= y <= o.y + o.h - 1 for x, y in s)
    )


def edge_candidates(
    sb: Box, tb: Box, anchor: Box | None, obstacles: list, plates: list | None = None
) -> list[list[list[float]]]:
    """Candidate routes for one edge, best-first: the direct curve, then (if it hits a node) a
    vertical channel PAST the blocking node(s) down each side into the target token. The channel
    side is chosen by, in priority order: fewest node hits, fewest FOREIGN-plate intrusions, then
    the side the target token sits on — so an arrow lands on its token without bowing across a
    plate line (e.g. MRP's ``sigma -> a = z*sigma`` routes right onto the right-most token instead
    of swinging left out of the plate). The global optimizer (``reflow.optimize_routes``) picks
    among these. Node boxes are obstacles; ``plates`` (the caller passes plate boxes already
    excluding this edge's own src/tgt plates) only influence the side choice, never block."""
    ex, ey = sb.x + sb.w / 2.0, sb.y + sb.h
    nx, ny = _target_point(tb, anchor)
    direct = simple_edge_path(sb, tb, anchor)
    if abs(ny - ey) < 1.0 or _obstacle_hits(direct, obstacles) == 0:
        return [direct]
    pts = _cubic4_points(direct)[2:-2]
    clippers = [
        o for o in obstacles
        if any(o.x <= x <= o.x + o.w and o.y <= y <= o.y + o.h for x, y in pts)
    ]
    if not clippers:
        return [direct]
    margin = 16.0
    top = max(min(o.y for o in clippers) - margin, ey + 8.0)
    bot = min(max(o.y + o.h for o in clippers) + margin, ny - 8.0)
    if bot <= top:
        return [direct]
    left = min(o.x for o in clippers) - margin
    right = max(o.x + o.w for o in clippers) + margin
    plates = plates or []

    def _foreign(path: list) -> int:
        s = _chain_points(path)[1:-1]
        return sum(
            1 for pb in plates
            if any(pb.x <= x <= pb.x + pb.w and pb.y <= y <= pb.y + pb.h for x, y in s)
        )

    # build both side channels, then order by (node hits, foreign-plate intrusions, token side)
    built = []
    for cx in (left, right):
        p = smooth_polyline([[ex, ey], [cx, top], [cx, bot], [nx, ny]])
        built.append((p, _obstacle_hits(p, obstacles), _foreign(p), abs(cx - nx)))
    built.sort(key=lambda t: (t[1], t[2], t[3]))
    return [t[0] for t in built] + [direct]


def routed_edge_path(
    sb: Box, tb: Box, anchor: Box | None, obstacles: list, plates: list | None = None
) -> list[list[float]]:
    """Default (locally-best) route for one edge — the first ``edge_candidates`` entry. The
    global optimizer may later swap in a different candidate to reduce total crossings."""
    return edge_candidates(sb, tb, anchor, obstacles, plates)[0]
