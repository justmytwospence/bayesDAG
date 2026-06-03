"""Post-ELK parent-order reflow.

ELK places a node's parents well *within a layer*, but it cannot order a parent against a
cluster in a *different* layer (verified: no `position`/`semiInteractive`/`forceNodeModelOrder`
/`layerChoiceConstraint` recipe does it). The symptom: a **free scalar** coefficient (e.g. `b1`
in `mu = f(a, county_idx) + b1*x1 + …`, or `b_male` in MRP) lands out of its token order
relative to the plated / in-plate parents, so its edge crosses theirs.

Fix: nudge each **free scalar** parent (not plated, not in the child's own plate — those ELK
already orders) into the x-gap that matches its token position between its neighbours. Plated
parents and their plate boxes are never moved (that's the hard, fragile part we avoid). The
caller applies this only if it *reduces* crossings (see `elk_backend.layout`), so it can never
make a model worse — honest, bounded, deterministic.
"""

from __future__ import annotations

from ..ir import Box, ModelIR, LayoutResult

_GAP = 16.0


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
                        mt**3 * p0[0] + 3 * mt**2 * t * c1[0] + 3 * mt * t * t * c2[0] + t**3 * p3[0],
                        mt**3 * p0[1] + 3 * mt**2 * t * c1[1] + 3 * mt * t * t * c2[1] + t**3 * p3[1],
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
    """Pairs of node boxes whose interiors overlap by more than 1px on BOTH axes (a touching
    edge or a 1px graze doesn't count). Used to guard the free-scalar snap: a reflow that fixes
    crossings but stacks boxes is worse than the crossings, so the caller reverts it."""
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
    group's box is the ugliness we discourage (a tie-breaker weighted far below crossings)."""
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


def optimize_routes(ir: ModelIR, res: LayoutResult) -> None:
    """Global edge-route repair: pick each edge's route (direct / channel-left / channel-right) to
    minimize a strictly **lexicographic** penalty — through-nodes ≫ crossings ≫ foreign-plate
    bows. We never leave an edge slicing through a node even if a forced layout must instead show
    one edge–edge crossing (normal in DAGs), and among equal-crossing routes we prefer the one
    that doesn't carve across an unrelated plate. The weights guarantee the lower tiers can never
    trade up a crossing (max foreign sum ≪ 1000). A greedy local search (per-edge choices
    interact). Skipped entirely when the default routes are already clean. Deterministic; bounded."""
    from . import common

    def penalty() -> int:
        return (
            count_through_nodes(ir, res) * 1_000_000
            + count_crossings(res) * 1_000
            + foreign_plate_total(ir, res)
        )

    if penalty() == 0:  # default routes already clean -> nothing to do (and stay fast)
        return

    member_plate: dict[str, str] = {}
    for p in ir.plates:
        for m in p.members:
            member_plate[m] = p.id

    cands: dict[str, list] = {}
    for e in ir.edges:
        sb = res.node_boxes.get(e.source)
        tb = res.node_boxes.get(e.target)
        if sb is None or tb is None:
            continue
        anchor = (
            res.node_token_anchors.get(e.target, {}).get(e.target_token_id)
            if e.target_token_id
            else None
        )
        obst = [b for nid, b in res.node_boxes.items() if nid not in (e.source, e.target)]
        plates = [
            b for pid, b in res.plate_boxes.items()
            if pid not in (member_plate.get(e.source), member_plate.get(e.target))
        ]
        cands[f"{e.source}|{e.target}"] = common.edge_candidates(sb, tb, anchor, obst, plates)

    pen = penalty()
    for _ in range(4):
        improved = False
        for key, cs in cands.items():
            if len(cs) < 2:
                continue
            best, best_pen = res.edge_paths[key], pen
            for c in cs:
                res.edge_paths[key] = c
                p = penalty()
                if p < best_pen:
                    best, best_pen = c, p
            res.edge_paths[key] = best
            if best_pen < pen:
                pen, improved = best_pen, True
        if pen == 0 or not improved:
            break


def prefer_vertical_tips(ir: ModelIR, res: LayoutResult) -> None:
    """Turn each arrowhead to point straight DOWN into its token wherever it's free to. We move
    ONLY the final control handle of a route directly above its endpoint (making the tip tangent
    vertical), leaving the body — and therefore every crossing / through-node / foreign-plate count
    — exactly as ``optimize_routes`` left it. The change is kept per edge only if it doesn't raise
    the penalty, so a tip that would have to slice a neighbour's edge or a node stays diagonal.
    This is the visual 'point downward' touch, applied without ever trading away layout quality.
    Deterministic; one pass."""

    def penalty() -> int:
        return (
            count_through_nodes(ir, res) * 1_000_000
            + count_crossings(res) * 1_000
            + foreign_plate_total(ir, res)
        )

    base = penalty()
    for e in ir.edges:
        key = f"{e.source}|{e.target}"
        pts = res.edge_paths.get(key)
        if not pts or len(pts) < 4 or (len(pts) - 1) % 3 != 0:  # need a cubic chain
            continue
        c2, p1 = pts[-2], pts[-1]
        if abs(p1[0] - c2[0]) <= 0.4 * abs(p1[1] - c2[1]):
            continue  # tip is already ~vertical
        start_y = pts[-4][1]  # the last segment's start anchor
        tip = max(10.0, min(0.5 * abs(p1[1] - start_y), 20.0))
        if p1[1] - tip <= start_y:  # not enough vertical room to turn without bulging backward
            continue
        saved = pts[-2]
        pts[-2] = [p1[0], p1[1] - tip]  # handle straight above the endpoint -> arrowhead points DOWN
        # reject if the verticalized tip now grazes a non-endpoint node (fine sampling — only THIS
        # edge changed, so only it can gain a through-node) or if it raises the global penalty
        s = _samples(res.edge_paths[key], 48)[2:-2]
        clips = any(
            nid not in (e.source, e.target)
            and any(b.x + 1 <= x <= b.x + b.w - 1 and b.y + 1 <= y <= b.y + b.h - 1 for x, y in s)
            for nid, b in res.node_boxes.items()
        )
        now = penalty()
        if clips or now > base:
            pts[-2] = saved  # would add a crossing/through-node/foreign bow -> keep the diagonal tip
        else:
            base = now


def _shift_node(res: LayoutResult, nid: str, dx: float) -> None:
    b = res.node_boxes[nid]
    res.node_boxes[nid] = Box(b.x + dx, b.y, b.w, b.h)
    anc = res.node_token_anchors.get(nid)
    if anc:
        res.node_token_anchors[nid] = {k: Box(v.x + dx, v.y, v.w, v.h) for k, v in anc.items()}


def snap_free_scalars(ir: ModelIR, res: LayoutResult) -> bool:
    """Nudge free-scalar parents into token order between their neighbours. Returns True if any
    node moved (so the caller knows to re-route edges). Mutates ``res`` node boxes/anchors."""
    member_plate: dict[str, str] = {}
    for p in ir.plates:
        for m in p.members:
            member_plate[m] = p.id

    parents_of: dict[str, list] = {}
    for e in ir.edges:
        if e.target_token_id:
            parents_of.setdefault(e.target, []).append(e)

    moved = False
    for child, edges in parents_of.items():
        toks = res.node_token_anchors.get(child, {})
        rows = []
        for e in edges:
            a = toks.get(e.target_token_id)
            nb = res.node_boxes.get(e.source)
            if a is None or nb is None:
                continue
            # "free" = movable: not plated AND not in the child's own plate
            free = e.source not in member_plate
            rows.append([a.x + a.w / 2.0, e.source, nb, free])
        if len(rows) < 2:
            continue
        rows.sort(key=lambda r: r[0])  # token order

        # Place each FREE scalar in the x-gap matching its token position, packed strictly
        # left-to-right against a monotonic barrier: the running right edge of every prior
        # parent, fixed or free. The barrier NEVER moves left — a narrow fixed node (e.g. a data
        # index) tucked under a wide just-placed scalar must not rewind it, or the next scalar
        # would clear only against the narrow node and land on top of the wide one. So a scalar
        # may sit right of its ideal token-x when crowded; overlap-free wins, and token order is
        # preserved (the edges still don't cross).
        prev_right = None
        for tcx, src, nb, free in rows:
            if not free:  # fixed parent: advance the barrier, never rewind it
                prev_right = nb.x + nb.w if prev_right is None else max(prev_right, nb.x + nb.w)
                continue
            target = tcx
            if prev_right is not None:
                target = max(target, prev_right + _GAP + nb.w / 2.0)  # strict clearance, no left-pull
            cur = nb.x + nb.w / 2.0
            if abs(target - cur) > 0.5:
                _shift_node(res, src, target - cur)
                moved = True
                nb = res.node_boxes[src]
            prev_right = nb.x + nb.w if prev_right is None else max(prev_right, nb.x + nb.w)
    return moved
