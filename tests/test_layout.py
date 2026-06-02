"""Layout backend: boxes, plates, and the param-edge -> token-anchor post-pass."""

import pytest

from bayesdag import mathsvg
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
def test_param_edges_land_on_token_anchors(eight_schools_ir):
    res = layout(eight_schools_ir)
    loc = res.node_token_anchors["y_obs"]["loc"]
    end = res.edge_paths["theta|y_obs"][-1]
    assert abs(end[0] - loc.x) < 0.5 and abs(end[1] - loc.y) < 0.5
    # deterministic port-edge: mu -> theta targets the mu token inside the equation
    mtok = res.node_token_anchors["theta"]["mu"]
    mend = res.edge_paths["mu|theta"][-1]
    assert abs(mend[0] - mtok.x) < 0.5 and abs(mend[1] - mtok.y) < 0.5


@pytest.mark.skipif(not _math, reason="needs the 'math' extra")
def test_labels_rendered_onto_nodes(eight_schools_ir):
    layout(eight_schools_ir)
    for n in eight_schools_ir.nodes:
        assert n.label_svg and "<svg" in n.label_svg
