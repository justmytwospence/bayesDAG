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


def optimize_routes(ir: ModelIR, res: LayoutResult) -> None:
    """Global edge-route repair: pick each edge's route (direct / channel-left / channel-right)
    to minimize a penalty that weights **through-nodes far above crossings** — so we never leave
    an edge slicing through a node, even if a forced layout must instead show one edge–edge
    crossing (normal in DAGs). A greedy local search (per-edge choices interact, so we optimize
    the set). Skipped entirely when the default routes are already clean — keeps it fast and
    regression-free. Deterministic; bounded passes."""
    from . import common

    def penalty() -> int:
        return count_through_nodes(ir, res) * 1000 + count_crossings(res)

    if penalty() == 0:  # default routes already clean -> nothing to do (and stay fast)
        return

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
        cands[f"{e.source}|{e.target}"] = common.edge_candidates(sb, tb, anchor, obst)

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

        # Place each FREE scalar in the x-gap that matches its token position: to the right of
        # the previous parent and to the left of the next FIXED (plated/in-plate) parent, biased
        # to its own token x. The fixed parents anchor the token gaps the scalar's edge must
        # thread; free scalars are packed left-to-right so they don't overlap each other.
        prev_right = None
        for k, (tcx, src, nb, free) in enumerate(rows):
            if not free:
                prev_right = nb.x + nb.w
                continue
            hi = None
            for tcx2, src2, nb2, free2 in rows[k + 1:]:
                if not free2:  # next fixed parent caps this scalar on the right
                    hi = nb2.x - _GAP - nb.w / 2.0
                    break
            target = tcx
            if prev_right is not None:
                target = max(target, prev_right + _GAP + nb.w / 2.0)
            if hi is not None and hi > target - 1e6:  # cap right only if it stays right of prev
                target = min(target, hi) if hi >= (prev_right or hi) else target
            cur = nb.x + nb.w / 2.0
            if abs(target - cur) > 0.5:
                _shift_node(res, src, target - cur)
                moved = True
                nb = res.node_boxes[src]
            prev_right = nb.x + nb.w
    return moved
