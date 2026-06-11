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


@pytest.mark.skipif(not _math, reason="needs the 'math' extra for token anchors")
def test_token_anchors_are_real_bboxes(eight_schools_ir):
    res = layout(eight_schools_ir)
    b = res.node_token_anchors["theta"]["mu"]
    assert b.w > 0 and b.h > 0  # real bbox, not a zero-size point


@pytest.mark.skipif(not _math, reason="needs the 'math' extra for token anchors")
def test_param_edges_land_on_token_without_overlap(eight_schools_ir):
    res = layout(eight_schools_ir)
    # arrows land a STANDOFF above the target's VISIBLE surface, in the token's column: the box
    # border for a bordered node (theta|y_obs -> observed), the token glyph for a borderless
    # deterministic equation (mu|theta -> deterministic). (edge, node, token, target_bordered)
    for edge, node, tok, bordered in [("theta|y_obs", "y_obs", "loc", True),
                                      ("mu|theta", "theta", "mu", False)]:
        a = res.node_token_anchors[node][tok]
        cx = a.x + a.w / 2.0
        end = res.edge_paths[edge][-1]
        assert abs(end[0] - cx) < 1.5             # centered on the token column
        assert end[1] < a.y                       # arrowhead above the token glyph (never covers it)
        surface = res.node_boxes[node].y if bordered else a.y
        assert end[1] < surface + 0.5             # outside/above the visible surface
        assert abs((surface - end[1]) - STANDOFF) < 1.5


@pytest.mark.skipif(not _math, reason="needs the 'math' extra")
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
