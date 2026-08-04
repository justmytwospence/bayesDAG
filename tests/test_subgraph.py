"""``subgraph(ir, var_names)`` / ``view(var_names=...)`` — pm.model_to_graphviz-style filtering."""

import pytest

import bayesdag
from bayesdag.convert import subgraph, to_ir
from bayesdag.ir import ModelIR


def test_to_ir_warns_when_idata_cannot_reach_the_adapter(eight_schools_ir):
    """`to_ir` builds glyphs through the ADAPTER, from the model's random variables. Handed an
    IR that already exists, it has nothing to attach a posterior to — and returning silently
    would give a caller who asked for posteriors a prior-only diagram instead.

    This is about `to_ir` specifically. `view()` CAN apply a posterior to a prebuilt IR, because
    it re-derives that layer by name rather than through the adapter — see
    test_update.py::test_a_prebuilt_ir_gains_posteriors_at_construction."""
    with pytest.warns(UserWarning, match="idata is ignored"):
        to_ir(eight_schools_ir, idata=object())
    with pytest.warns(UserWarning, match="idata is ignored"):
        to_ir(eight_schools_ir.to_dict(), idata=object())


def test_view_does_not_warn_for_a_prebuilt_ir_with_idata(eight_schools_ir):
    """The view routes the prebuilt case around the adapter, so the warning must not leak out
    of it — a warning a caller cannot act on is noise."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        bayesdag.view(eight_schools_ir)


def test_no_warning_on_the_ordinary_paths(eight_schools_ir, eight_schools_model):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        to_ir(eight_schools_ir)  # idempotent, no idata
        to_ir(eight_schools_model)  # the real adapter path


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
