"""Layout quality across the six escalating example models: no edge passes through a node,
the crossing-free models are crossing-free, and the two dense models keep at most the single
forced (plate-contiguity) crossing. Uses an independent fine-grained geometric detector."""

import pytest

from bayesdag.convert import to_ir
from bayesdag.layout import elk_backend, layout
from conftest import CROSSING_FREE_MODELS, MODEL_BUILDERS, RESIDUAL_CROSSING_MODELS

pytestmark = pytest.mark.skipif(not elk_backend.available(), reason="needs mini-racer + elkjs")


def _samples(pts, n=40):
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


def _seg_x(a, b, c, d):
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])

    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def count_crossings(res) -> int:
    paths = {k: _samples(v) for k, v in res.edge_paths.items()}
    keys = [k for k, v in paths.items() if v]
    total = 0
    for i in range(len(keys)):
        A = paths[keys[i]]
        for j in range(i + 1, len(keys)):
            B = paths[keys[j]]
            ends = (A[0], A[-1], B[0], B[-1])
            crossed = False
            for x in range(len(A) - 1):
                for y in range(len(B) - 1):
                    if _seg_x(A[x], A[x + 1], B[y], B[y + 1]):
                        px, py = A[x]
                        # ignore intersections at/near a shared endpoint (edges legitimately meet there)
                        if min((px - ex) ** 2 + (py - ey) ** 2 for ex, ey in ends) ** 0.5 >= 10:
                            crossed = True
                            break
                if crossed:
                    break
            total += crossed
    return total


def through_node_edges(ir, res):
    bad = []
    for e in ir.edges:
        pts = res.edge_paths.get(f"{e.source}|{e.target}")
        s = _samples(pts)[3:-3] if pts else []
        for nid, b in res.node_boxes.items():
            if nid in (e.source, e.target):
                continue
            if any(b.x + 1 <= x <= b.x + b.w - 1 and b.y + 1 <= y <= b.y + b.h - 1 for x, y in s):
                bad.append((f"{e.source}->{e.target}", nid))
                break
    return bad


@pytest.mark.parametrize("name", list(MODEL_BUILDERS))
def test_no_edge_passes_through_a_node(name):
    """The headline guarantee: across ALL six models, no edge cuts through a non-endpoint node."""
    ir = to_ir(MODEL_BUILDERS[name]())
    res = layout(ir)
    assert through_node_edges(ir, res) == []


@pytest.mark.parametrize("name", CROSSING_FREE_MODELS)
def test_crossing_free_models(name):
    ir = to_ir(MODEL_BUILDERS[name]())
    assert count_crossings(layout(ir)) == 0


@pytest.mark.parametrize("name", RESIDUAL_CROSSING_MODELS)
def test_dense_models_at_most_one_crossing(name):
    """IRT and MRP have one *forced* edge-edge crossing (a plate's members bracket another
    plate's token / plated random-effect drift). We hold the line at <= 1 — never a pile-up."""
    ir = to_ir(MODEL_BUILDERS[name]())
    assert count_crossings(layout(ir)) <= 1


def test_reflow_is_deterministic():
    """Same model (fixed seeds) -> byte-identical geometry across runs, reflow included."""
    a = layout(to_ir(MODEL_BUILDERS["hier_reg"]()))
    b = layout(to_ir(MODEL_BUILDERS["hier_reg"]()))
    for k, box in a.node_boxes.items():
        assert (box.x, box.y, box.w, box.h) == (
            b.node_boxes[k].x, b.node_boxes[k].y, b.node_boxes[k].w, b.node_boxes[k].h,
        )
