"""``subgraph(ir, var_names)`` / ``view(var_names=...)`` — pm.model_to_graphviz-style filtering."""

import pytest

import bayesdag
from bayesdag.convert import subgraph
from bayesdag.ir import ModelIR


def test_subgraph_keeps_selection_plus_direct_parents(eight_schools_ir):
    sub = subgraph(eight_schools_ir, ["theta"])
    assert {n.id for n in sub.nodes} == {"mu", "tau", "eta", "theta"}  # parents kept, y_obs dropped
    assert {(e.source, e.target) for e in sub.edges} == {
        ("mu", "theta"),
        ("tau", "theta"),
        ("eta", "theta"),
    }
    # the school plate survives with only its surviving members
    plate = next(p for p in sub.plates if p.id == "plate_school")
    assert set(plate.members) == {"eta", "theta"}
    # pure function: the input IR is not mutated
    assert {n.id for n in eight_schools_ir.nodes} == {"mu", "tau", "eta", "theta", "y_obs"}
    assert isinstance(sub, ModelIR)


def test_subgraph_drops_empty_plates_and_rejects_unknown(eight_schools_ir):
    sub = subgraph(eight_schools_ir, ["mu"])
    assert {n.id for n in sub.nodes} == {"mu"}
    assert sub.plates == []  # school plate has no surviving member
    with pytest.raises(ValueError, match="nope"):
        subgraph(eight_schools_ir, ["nope"])


def test_view_var_names_filters_and_renders(eight_schools_model):
    v = bayesdag.view(eight_schools_model, var_names=["theta"])
    assert {n.id for n in v.ir.nodes} == {"mu", "tau", "eta", "theta"}
    svg = v.to_svg()
    assert "<svg" in svg and 'data-node="theta"' in svg and "y_obs" not in svg


def test_subgraph_round_trips_through_dict(eight_schools_ir):
    sub = subgraph(eight_schools_ir, ["theta"])
    again = ModelIR.from_dict(sub.to_dict())
    assert {n.id for n in again.nodes} == {n.id for n in sub.nodes}
    assert len(again.edges) == len(sub.edges)
