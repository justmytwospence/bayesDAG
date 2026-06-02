"""ELK layout backend (the default): the M0.3 spike gate + the plate-layout fixes.

ELK runs in-process in mini-racer V8; these are skipped when that isn't available (the
`dot` fallback is exercised by `test_layout`)."""

import pytest

from bayesdag.ir import EdgeIR, ModelIR, NodeIR, PlateIR
from bayesdag.layout import elk_backend

pytestmark = pytest.mark.skipif(not elk_backend.available(), reason="needs mini-racer + elkjs")


def test_spike_gate_compound_graph():
    """The de-risking checkpoint: ELK lays out a 3-node compound graph in V8 and returns
    geometry (a node outside a plate feeding a node inside it)."""
    ir = ModelIR(
        nodes=[
            NodeIR(id="s", role="latent", observed=False, label_tex="s"),
            NodeIR(id="m", role="latent", observed=False, label_tex="m"),
            NodeIR(id="y", role="observed", observed=True, label_tex="y"),
        ],
        edges=[EdgeIR("s", "y"), EdgeIR("m", "y")],
        plates=[PlateIR(id="plate_obs", label="obs", members=["m", "y"])],
    )
    res = elk_backend.layout(ir)
    assert res.canvas.w > 0 and res.canvas.h > 0
    assert {"s", "m", "y"} <= set(res.node_boxes)
    assert "plate_obs" in res.plate_boxes


def test_plate_encloses_members_excludes_externals(radon_ir):
    res = elk_backend.layout(radon_ir)
    obs = res.plate_boxes["plate_obs"]
    for m in ("mu", "y", "county_idx", "floor"):
        b = res.node_boxes[m]
        assert obs.x - 1 <= b.x and b.x + b.w <= obs.x + obs.w + 1
        assert obs.y - 1 <= b.y and b.y + b.h <= obs.y + obs.h + 1
    # the external scalar parents must NOT fall inside the obs plate box
    for ext in ("sigma", "b"):
        b = res.node_boxes[ext]
        cx, cy = b.x + b.w / 2.0, b.y + b.h / 2.0
        inside = obs.x <= cx <= obs.x + obs.w and obs.y <= cy <= obs.y + obs.h
        assert not inside, f"{ext} should be outside the obs plate"


def test_external_scale_param_placed_beside_its_child(radon_ir):
    """The headline fix: `sigma` (parent of `y`) is placed toward `y`'s side, not shoved to
    the opposite edge of the canvas (which is what forced the crossing under dot)."""
    res = elk_backend.layout(radon_ir)
    sigma_cx = res.node_boxes["sigma"].x + res.node_boxes["sigma"].w / 2.0
    mu_cx = res.node_boxes["mu"].x + res.node_boxes["mu"].w / 2.0
    # sigma sits to the right of the mu spine (same side it descends into y from)
    assert sigma_cx > mu_cx


def test_ports_order_parents_to_match_token_positions(radon_ir):
    """Boundary ports: each parent is placed in the x-order of its target token, so edges
    into `mu = f(a, county_idx) + b floor` don't cross / pile up."""
    res = elk_backend.layout(radon_ir)
    anchors = res.node_token_anchors["mu"]
    assert anchors["county_idx"].x < anchors["floor"].x  # token order in the equation
    cx = {k: res.node_boxes[k].x + res.node_boxes[k].w / 2.0 for k in ("county_idx", "floor")}
    assert cx["county_idx"] < cx["floor"]  # ...and the source nodes follow that order


def test_layout_is_deterministic(eight_schools_ir):
    """Fixed randomSeed -> identical geometry across runs (golden-image stability)."""
    a = elk_backend.layout(eight_schools_ir)
    b = elk_backend.layout(eight_schools_ir)
    assert a.canvas.w == b.canvas.w and a.canvas.h == b.canvas.h
    for k, box in a.node_boxes.items():
        assert (box.x, box.y, box.w, box.h) == (
            b.node_boxes[k].x,
            b.node_boxes[k].y,
            b.node_boxes[k].w,
            b.node_boxes[k].h,
        )


def test_layout_works_inside_asyncio_loop(radon_ir):
    """Regression: marimo runs cells inside an asyncio loop, where mini-racer forbids a
    blocking promise.get() on the loop thread. ELK must still run (on its worker thread) and
    the dispatcher must NOT silently fall back to dot."""
    import asyncio

    from bayesdag.layout import layout as dispatch

    direct = elk_backend.layout(radon_ir)

    async def _run():
        return dispatch(radon_ir)

    in_loop = asyncio.run(_run())
    # same engine as the direct ELK call (matching canvas) -> not the dot fallback
    assert (round(in_loop.canvas.w), round(in_loop.canvas.h)) == (
        round(direct.canvas.w),
        round(direct.canvas.h),
    )


def test_default_layout_is_elk_no_silent_fallback(radon_ir):
    """The dispatcher uses ELK by default (same result as calling it directly) — it does NOT
    silently downgrade to dot."""
    from bayesdag.layout import layout as dispatch

    default = dispatch(radon_ir)
    forced = elk_backend.layout(radon_ir)
    assert (round(default.canvas.w), round(default.canvas.h)) == (
        round(forced.canvas.w),
        round(forced.canvas.h),
    )


def test_dot_is_explicit_opt_in_only(radon_ir, monkeypatch):
    """`BAYESDAG_LAYOUT=dot` is the only way to reach the Graphviz backend; it stays working as
    the deliberate rollback target."""
    from bayesdag.layout import layout as dispatch

    monkeypatch.setenv("BAYESDAG_LAYOUT", "dot")
    res = dispatch(radon_ir)
    assert res.canvas.w > 0 and "plate_obs" in res.plate_boxes
    assert {"mu", "y", "sigma", "b"} <= set(res.node_boxes)
