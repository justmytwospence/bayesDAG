"""Interoperability: JSON Schema validation, round-trip, and graph-format adapters."""

import json

import pytest

from bayesdag import ir as ir_mod
from bayesdag import schema
from bayesdag.adapters import to_elk, to_networkx
from bayesdag.adapters.graph import markov_blanket
from bayesdag.convert import to_ir
from bayesdag.ir import ModelIR


def test_schema_validates(eight_schools_ir):
    schema.validate(eight_schools_ir.to_dict())


def test_schema_rejects_what_it_should():
    """A validator that accepts everything guards nothing. These two cases pin the parts of the
    generator that are easy to silently lose: `required` emission, and Literal -> enum."""
    import copy

    jsonschema = pytest.importorskip("jsonschema")

    valid = {
        "schema_version": ir_mod.SCHEMA_VERSION,
        "nodes": [{"id": "x", "role": "latent"}],
        "edges": [],
        "plates": [],
    }
    schema.validate(valid)  # sanity: the baseline really is valid

    missing_role = copy.deepcopy(valid)
    del missing_role["nodes"][0]["role"]
    with pytest.raises(jsonschema.ValidationError):
        schema.validate(missing_role)

    bad_role = copy.deepcopy(valid)
    bad_role["nodes"][0]["role"] = "banana"
    with pytest.raises(jsonschema.ValidationError):
        schema.validate(bad_role)


def test_published_schema_in_sync():
    """The committed schema for the current version must match the generated schema."""
    published = json.loads((schema._SCHEMA_FILE).read_text())
    assert published == schema.build_schema()


def test_round_trip(eight_schools_ir):
    assert ModelIR.from_dict(eight_schools_ir.to_dict()) == eight_schools_ir


def test_to_ir_idempotent(eight_schools_ir):
    assert to_ir(eight_schools_ir) is eight_schools_ir


def test_to_ir_dict_escape_hatch(eight_schools_ir):
    assert to_ir(eight_schools_ir.to_dict()) == eight_schools_ir


def test_to_networkx(eight_schools_ir):
    g = to_networkx(eight_schools_ir)
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 4
    assert markov_blanket(eight_schools_ir, "tau") == {"eta", "mu", "theta"}


def test_to_elk_serializable_and_nested(eight_schools_ir):
    elk = to_elk(eight_schools_ir)
    json.dumps(elk)  # must be JSON-serializable
    plate_children = [c for c in elk["children"] if c.get("children")]
    assert plate_children, "expected a nested plate compound node"
    assert {c["id"] for c in plate_children[0]["children"]} == {
        "node:eta",
        "node:theta",
        "node:y_obs",
    }
    assert len(elk["edges"]) == 4


def _elk_ids(elk):
    ids, stack = [], list(elk["children"])
    while stack:
        c = stack.pop()
        ids.append(c["id"])
        ids += [p["id"] for p in c.get("ports", [])]
        stack += c.get("children", [])
    return ids


def test_to_elk_references_only_existing_ports(eight_schools_ir):
    """Every edge endpoint must resolve — including a deterministic's leaf-token ports, which the
    old exporter referenced without ever declaring."""
    elk = to_elk(eight_schools_ir)
    ids = _elk_ids(elk)
    assert len(ids) == len(set(ids))
    for e in elk["edges"]:
        for end in e["sources"] + e["targets"]:
            assert end in ids, end
    theta = next(c for c in _elk_ids(elk) if c.startswith("port:") and "theta" in c)
    assert theta  # theta = mu + tau*eta exposes its leaf tokens as ports


def test_to_elk_nests_plates_and_survives_hostile_names():
    import dataclasses

    from bayesdag.ir import EdgeIR, NodeIR, PlateIR

    ir = ModelIR(
        nodes=[NodeIR(id="plate_a", role="latent"), NodeIR(id="a.b", role="latent")],
        edges=[EdgeIR("plate_a", "a.b")],
        plates=[
            PlateIR(id="plate_a", label="a", members=["plate_a"]),
            PlateIR(id="inner", label="b", members=["a.b"], parent="plate_a"),
        ],
    )
    elk = to_elk(ir)
    ids = _elk_ids(elk)
    assert len(ids) == len(set(ids))  # node "plate_a" and plate "plate_a" do not collide
    outer = next(c for c in elk["children"] if c["id"] == "plate:plate_a")
    assert any(c["id"] == "plate:inner" for c in outer["children"])
    # a cyclic parent chain degrades to the root rather than recursing forever
    cyc = dataclasses.replace(
        ir,
        plates=[
            PlateIR(id="p", label="p", members=["plate_a"], parent="q"),
            PlateIR(id="q", label="q", members=["a.b"], parent="p"),
        ],
    )
    json.dumps(to_elk(cyc))


def test_all_published_schemas_exist():
    """Older artifacts must stay checkable against the contract they were written under."""
    assert schema.PUBLISHED == ("1.0", "1.1", "1.2")
    for v in schema.PUBLISHED:
        assert (schema._SCHEMA_DIR / f"graph-v{v}.json").is_file(), v


def test_reads_a_1_0_artifact(eight_schools_ir):
    """1.0 dicts load: the singular edge token fills the list, and a 1.0 conditional-latent
    schematic (one generic bell for every family) is dropped with a note, not shown as exact."""
    d = eight_schools_ir.to_dict()
    d["meta"]["schema_version"] = "1.0"
    for n in d["nodes"]:
        del n["ndim"], n["observed_sources"]
    for e in d["edges"]:
        e.pop("target_token_ids", None)
    d["nodes"].append(
        {
            "id": "legacy",
            "role": "latent",
            "glyph": {"kind": "schematic", "source": "prior_family_only"},
            "glyph_data": {"xs": [0, 1], "ys": [0, 1]},
        }
    )
    out = ModelIR.from_dict(d)
    assert out.meta.schema_version == ir_mod.SCHEMA_VERSION
    tok = [e for e in out.edges if e.target_token_id]
    assert tok and all(e.target_token_ids == [e.target_token_id] for e in tok)
    legacy = out.node("legacy")
    assert legacy.glyph is None and "legacy" in legacy.glyph_note
    assert all(n.ndim is None and n.observed_sources == [] for n in out.nodes)


def test_reads_1_1_without_reinterpreting_legacy_edges_or_applying_1_0_fixes():
    old = {
        "meta": {"schema_version": "1.1"},
        "nodes": [
            {
                "id": "y",
                "role": "observed",
                "glyph": {"kind": "schematic", "source": "prior_family_only"},
            },
            {"id": "data", "role": "data"},
        ],
        "edges": [{"source": "y", "target": "data", "target_token_id": "old"}],
    }
    upgraded = ModelIR.from_dict(old)
    assert upgraded.meta.schema_version == "1.2"
    assert all(n.ndim is None and not n.observed_sources for n in upgraded.nodes)
    assert upgraded.node("y").glyph.kind == "schematic"
    (edge,) = upgraded.edges
    assert (edge.source, edge.target, edge.target_token_id, edge.target_token_ids) == (
        "y",
        "data",
        "old",
        [],
    )


def test_1_2_round_trip_preserves_rank_and_observation_binding():
    data = {
        "meta": {"schema_version": "1.2"},
        "nodes": [
            {"id": "values", "role": "data", "ndim": 1},
            {"id": "y", "role": "observed", "ndim": 1, "observed_sources": ["values"]},
        ],
    }
    model_ir = to_ir(data)
    assert model_ir.node("y").ndim == 1
    assert model_ir.node("y").observed_sources == ["values"]
    assert model_ir.edges == []
    schema.validate(model_ir.to_dict())
    assert ModelIR.from_json(model_ir.to_json()) == model_ir

    data["nodes"][1]["observed_sources"] = ["missing"]
    with pytest.raises(ValueError, match=r"node y: unknown observation sources .*missing"):
        to_ir(data)


def test_rejects_an_unsupported_major_version(eight_schools_ir):
    d = eight_schools_ir.to_dict()
    d["meta"]["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="not supported"):
        ModelIR.from_dict(d)
