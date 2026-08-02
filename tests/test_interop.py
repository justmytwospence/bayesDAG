"""Interoperability: JSON Schema validation, round-trip, and graph-format adapters."""

import json

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

    import pytest

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
    """The committed schema/graph-v1.0.json must match the generated schema."""
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
    assert {c["id"] for c in plate_children[0]["children"]} == {"eta", "theta", "y_obs"}
    assert len(elk["edges"]) == 4
