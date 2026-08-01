"""Layout-quality metrics over a ``LayoutResult``.

Edge routing is done natively by ELK now (orthogonal edges; see ``elk_backend``), so this module
no longer routes or repairs anything — it only *measures* quality so the test suite can assert it:
edge–edge crossings, edges passing through a non-endpoint node, node-box overlaps, and edges
bowing through a plate that owns neither endpoint. All deterministic; geometry-only.
"""

from __future__ import annotations

from ..ir import LayoutResult, ModelIR


def _cross(a, b, c, d) -> bool:
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])

    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def _samples(pts, n: int = 16):
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
                        mt**3 * p0[0]
                        + 3 * mt**2 * t * c1[0]
                        + 3 * mt * t * t * c2[0]
                        + t**3 * p3[0],
                        mt**3 * p0[1]
                        + 3 * mt**2 * t * c1[1]
                        + 3 * mt * t * t * c2[1]
                        + t**3 * p3[1],
                    )
                )
    return out


def count_crossings(res: LayoutResult) -> int:
    """Number of *proper* edge–edge crossings (ignoring intersections that occur at/near a
    shared endpoint — two edges into the same node legitimately meet there)."""
    paths = {k: _samples(v) for k, v in res.edge_paths.items()}
    keys = [k for k, v in paths.items() if v]
    total = 0
    for i in range(len(keys)):
        A = paths[keys[i]]
        for j in range(i + 1, len(keys)):
            B = paths[keys[j]]
            ends = (A[0], A[-1], B[0], B[-1])
            hit = False
            for x in range(len(A) - 1):
                for y in range(len(B) - 1):
                    if _cross(A[x], A[x + 1], B[y], B[y + 1]):
                        px, py = A[x]
                        if min((px - ex) ** 2 + (py - ey) ** 2 for ex, ey in ends) ** 0.5 >= 10:
                            hit = True
                            break
                if hit:
                    break
            total += hit
    return total


def count_through_nodes(ir: ModelIR, res: LayoutResult) -> int:
    """How many edges pass through a non-endpoint node's interior."""
    total = 0
    for e in ir.edges:
        pts = res.edge_paths.get(f"{e.source}|{e.target}")
        if not pts:
            continue
        s = _samples(pts)[2:-2]
        for nid, b in res.node_boxes.items():
            if nid in (e.source, e.target):
                continue
            if any(b.x + 1 <= x <= b.x + b.w - 1 and b.y + 1 <= y <= b.y + b.h - 1 for x, y in s):
                total += 1
                break
    return total


def count_overlaps(res: LayoutResult) -> int:
    """Pairs of node boxes whose interiors overlap by more than 1px on BOTH axes (a touching edge
    or a 1px graze doesn't count). ELK never overlaps nodes, so this should always be 0."""
    boxes = list(res.node_boxes.values())
    n = 0
    for i in range(len(boxes)):
        bi = boxes[i]
        for j in range(i + 1, len(boxes)):
            bj = boxes[j]
            ox = min(bi.x + bi.w, bj.x + bj.w) - max(bi.x, bj.x)
            oy = min(bi.y + bi.h, bj.y + bj.h) - max(bi.y, bj.y)
            if ox > 1.0 and oy > 1.0:
                n += 1
    return n


def foreign_plate_total(ir: ModelIR, res: LayoutResult) -> int:
    """Total times an edge's interior passes through a 'foreign' plate — one that contains NEITHER
    endpoint. Entering the target's own plate is expected and free; bowing across an unrelated
    group's box is the ugliness we watch for."""
    member_plate: dict[str, str] = {}
    for p in ir.plates:
        for m in p.members:
            member_plate[m] = p.id
    total = 0
    for e in ir.edges:
        pts = res.edge_paths.get(f"{e.source}|{e.target}")
        if not pts:
            continue
        s = _samples(pts)[2:-2]
        own = (member_plate.get(e.source), member_plate.get(e.target))
        for pid, b in res.plate_boxes.items():
            if pid in own:
                continue
            if any(b.x + 1 <= x <= b.x + b.w - 1 and b.y + 1 <= y <= b.y + b.h - 1 for x, y in s):
                total += 1
    return total
