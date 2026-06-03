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
    for edge, node, tok in [("theta|y_obs", "y_obs", "loc"), ("mu|theta", "theta", "mu")]:
        b = res.node_token_anchors[node][tok]
        cx = b.x + b.w / 2.0
        pts = res.edge_paths[edge]
        end = pts[-1]
        assert abs(end[0] - cx) < 1.5            # centered on the token
        assert end[1] < b.y                       # arrowhead ABOVE the token (no overlap)
        assert abs((b.y - end[1]) - STANDOFF) < 1.5


@pytest.mark.skipif(not _math, reason="needs the 'math' extra")
def test_labels_rendered_onto_nodes(eight_schools_ir):
    layout(eight_schools_ir)
    for n in eight_schools_ir.nodes:
        assert n.label_svg and "<svg" in n.label_svg
