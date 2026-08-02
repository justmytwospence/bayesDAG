"""Layout backend: boxes, plates, and the param-edge -> token-anchor post-pass."""

import pytest

from bayesdag import mathsvg
from bayesdag.geometry import STANDOFF
from bayesdag.layout import layout

_math = mathsvg.get_renderer().available


def test_layout_produces_boxes_and_plate(eight_schools_ir):
    res = layout(eight_schools_ir)
    assert res.canvas.w > 0 and res.canvas.h > 0
    assert set(res.node_boxes) == {"mu", "tau", "eta", "theta", "y_obs"}
    for n in eight_schools_ir.nodes:
        assert n.box is not None
    assert "plate_school" in res.plate_boxes
    # plate encloses its members
    pb = res.plate_boxes["plate_school"]
    for m in ("eta", "theta", "y_obs"):
        b = res.node_boxes[m]
        assert pb.x - 1 <= b.x and b.x + b.w <= pb.x + pb.w + 1


@pytest.mark.parametrize(
    ("rankdir", "axis", "sign"),
    [("TB", 1, +1), ("BT", 1, -1), ("LR", 0, +1), ("RL", 0, -1)],
)
def test_every_advertised_rankdir_actually_flows_that_way(rankdir, axis, sign):
    """`_DIRECTION` advertises four directions; only TB was ever exercised. A parameter that
    silently produces a wrong figure is worse than one that isn't offered, so pin all four:
    the child must sit on the correct side of its parent along the correct axis."""
    import numpy as np
    import pymc as pm

    from bayesdag.convert import to_ir

    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("y", mu, 1.0, observed=np.zeros(5))
    res = layout(to_ir(m), rankdir=rankdir)

    a, b = res.node_boxes["mu"], res.node_boxes["y"]
    centers = ((a.x + a.w / 2, b.x + b.w / 2), (a.y + a.h / 2, b.y + b.h / 2))
    parent_c, child_c = centers[axis]
    assert sign * (child_c - parent_c) > 0, (
        f"{rankdir}: the child landed on the wrong side of its parent "
        f"(parent={parent_c:.0f}, child={child_c:.0f})"
    )
    assert res.canvas.w > 0 and res.canvas.h > 0


@pytest.mark.parametrize("rankdir", ["TB", "BT", "LR", "RL"])
def test_token_edges_currently_approach_from_above_in_every_direction(rankdir):
    """A known, deliberate limitation rather than a hidden one: token ports are placed
    NORTH/SOUTH and `_attach_target` lands every edge on the target's TOP border, whatever the
    flow direction. Nothing is drawn wrongly — no edge cuts through a node in any direction
    (test_reflow covers that) — but a BT or LR figure approaches its tokens the TB way.

    If someone makes attachment direction-aware, this test should fail and be updated on purpose.
    """
    import numpy as np
    import pymc as pm

    from bayesdag.convert import to_ir
    from bayesdag.geometry import STANDOFF

    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("y", mu, 1.0, observed=np.zeros(5))
    ir = to_ir(m)
    res = layout(ir, rankdir=rankdir)
    if not res.node_token_anchors.get("y"):
        pytest.skip("no token anchors without the math bundle")
    target = res.node_boxes["y"]
    end_y = next(iter(res.edge_paths.values()))[-1][1]
    assert abs(end_y - (target.y - STANDOFF)) < 0.6


def test_math_unavailable_warning_points_at_a_real_install_path(monkeypatch, eight_schools_ir):
    """The degradation message used to advertise `pip install 'bayesdag[math]'` — an extra that
    has never existed (mini-racer is a core dependency). Pointing users at a no-op install is
    exactly the kind of dishonest instruction the project's own contract forbids."""
    from bayesdag.layout import common

    class _Unavailable:
        available = False

    monkeypatch.setattr(common.mathsvg, "get_renderer", lambda: _Unavailable())
    monkeypatch.setattr(common, "_warned_math_unavailable", False)
    with pytest.warns(RuntimeWarning) as rec:
        common.render_labels(eight_schools_ir)
    msg = str(rec[0].message)
    assert "[math]" not in msg
    assert "npm run build" in msg


def test_relayout_leaves_no_stale_geometry_on_the_ir(eight_schools_ir):
    """The LayoutResult is the source of truth; the copies on NodeIR are a convenience mirror.
    Laying the same IR out twice (two views, or a different rankdir) must not leave a node
    carrying the previous run's coordinates — stale numbers are indistinguishable from fresh
    ones, and the renderer would happily draw them."""
    from bayesdag.layout import common
    from bayesdag.render_svg import to_svg

    ir = eight_schools_ir
    layout(ir, rankdir="TB")
    stale = {n.id: n.box for n in ir.nodes}
    second = layout(ir, rankdir="LR")

    assert {n.id: n.box for n in ir.nodes} == second.node_boxes  # mirrors the NEW layout
    assert stale != second.node_boxes  # sanity: the two layouts really do differ

    # a node the new layout doesn't place carries no geometry, so it cannot be drawn from a
    # leftover box: the reset clears every node up front and only placed ones are re-populated
    common.reset_geometry(ir)
    assert all(n.box is None and not n.port_anchors for n in ir.nodes)

    dropped = ir.nodes.pop()  # y_obs: a leaf
    ir.edges = [e for e in ir.edges if dropped.id not in (e.source, e.target)]
    for p in ir.plates:
        p.members = [m for m in p.members if m != dropped.id]
    third = layout(ir, rankdir="TB")
    assert dropped.id not in third.node_boxes
    assert f'data-node="{dropped.id}"' not in to_svg(ir, third)


@pytest.mark.skipif(not _math, reason="needs the built mathjax bundle for token anchors")
def test_token_anchors_are_real_bboxes(eight_schools_ir):
    res = layout(eight_schools_ir)
    b = res.node_token_anchors["theta"]["mu"]
    assert b.w > 0 and b.h > 0  # real bbox, not a zero-size point


@pytest.mark.skipif(not _math, reason="needs the built mathjax bundle for token anchors")
def test_param_edges_land_on_token_without_overlap(eight_schools_ir):
    res = layout(eight_schools_ir)
    # every node is bordered (incl. the deterministic equation box): arrows land a STANDOFF above the
    # box's TOP border, in the token's column — the column says WHICH parameter, the box stays
    # uncrossed. (edge, node, token)
    for edge, node, tok in [(("theta", "y_obs"), "y_obs", "loc"), (("mu", "theta"), "theta", "mu")]:
        a = res.node_token_anchors[node][tok]
        cx = a.x + a.w / 2.0
        end = res.edge_paths[edge][-1]
        assert abs(end[0] - cx) < 1.5  # centered on the token column
        assert end[1] < a.y  # arrowhead above the token (never covers it)
        surface = res.node_boxes[node].y  # the box top border
        assert end[1] < surface + 0.5  # outside/above the box
        assert abs((surface - end[1]) - STANDOFF) < 1.5


@pytest.mark.skipif(not _math, reason="needs the built mathjax bundle")
def test_labels_rendered_onto_nodes(eight_schools_ir):
    layout(eight_schools_ir)
    for n in eight_schools_ir.nodes:
        assert n.label_svg and "<svg" in n.label_svg


def test_missing_dot_binary_gives_actionable_error(monkeypatch):
    """BAYESDAG_LAYOUT=dot without graphviz installed must say how to fix it, not dump a raw
    FileNotFoundError traceback."""
    from bayesdag.layout import graphviz_backend as gb

    def boom(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "dot")

    monkeypatch.setattr(gb.subprocess, "run", boom)
    with pytest.raises(RuntimeError, match="BAYESDAG_LAYOUT=dot"):
        gb._run_dot("digraph {}")


def test_math_unavailable_warns_once(monkeypatch, eight_schools_ir):
    """Silent plain-text degradation hid real breakage — the first render without math must warn."""
    import warnings

    from bayesdag.layout import common

    class _Unavailable:
        available = False

    monkeypatch.setattr(common.mathsvg, "get_renderer", lambda: _Unavailable())
    monkeypatch.setattr(common, "_warned_math_unavailable", False)
    with pytest.warns(RuntimeWarning, match="math rendering is unavailable"):
        common.render_labels(eight_schools_ir)
    with warnings.catch_warnings():  # second render: flag set, no second warning
        warnings.simplefilter("error")
        common.render_labels(eight_schools_ir)


def test_nested_plates_nest():
    """`PlateIR.parent` and the recursive branch in elk_backend._build_graph are unreachable from
    from_pymc (it never sets a parent), but they ARE reachable through a hand-built IR or
    ModelIR.from_dict — the published schema advertises the field. Untested reachable code is
    how a cycle guard rots, so drive it directly: the inner plate's box must sit inside the
    outer one's."""
    from bayesdag.ir import EdgeIR, ModelIR, NodeIR, PlateIR

    ir = ModelIR(
        nodes=[
            NodeIR(id="a", role="latent", dist="Normal", label_tex="a"),
            NodeIR(id="b", role="latent", dist="Normal", label_tex="b"),
        ],
        edges=[EdgeIR(source="a", target="b")],
        plates=[
            PlateIR(id="outer", label="outer (3)", members=["a", "b"]),
            PlateIR(id="inner", label="inner (2)", members=["b"], parent="outer"),
        ],
    )
    res = layout(ir)
    outer, inner = res.plate_boxes["outer"], res.plate_boxes["inner"]
    assert outer.x - 1 <= inner.x and inner.x + inner.w <= outer.x + outer.w + 1
    assert outer.y - 1 <= inner.y and inner.y + inner.h <= outer.y + outer.h + 1
    assert res.node_boxes["b"].x >= inner.x - 1  # the member really is in the inner plate


def test_a_pipe_in_a_variable_name_cannot_collide_two_edges():
    """edge_paths was keyed "src|tgt". `pm.Normal("a|b", ...)` is a legal name, so two distinct
    edges could hash to one key and silently share a route. Tuple keys make that unrepresentable."""
    from bayesdag.ir import EdgeIR, ModelIR, NodeIR

    ir = ModelIR(
        nodes=[
            NodeIR(id="a|b", role="latent", dist="Normal", label_tex="x"),
            NodeIR(id="c", role="latent", dist="Normal", label_tex="y"),
            NodeIR(id="a", role="latent", dist="Normal", label_tex="z"),
            NodeIR(id="b|c", role="latent", dist="Normal", label_tex="w"),
        ],
        # both of these collapse to the string key "a|b|c"
        edges=[EdgeIR(source="a|b", target="c"), EdgeIR(source="a", target="b|c")],
    )
    res = layout(ir)
    assert ("a|b", "c") in res.edge_paths
    assert ("a", "b|c") in res.edge_paths
    assert res.edge_paths[("a|b", "c")] != res.edge_paths[("a", "b|c")]
