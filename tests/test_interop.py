"""Interoperability: JSON Schema validation, round-trip, and graph-format adapters."""

import json

from bayesdag import schema
from bayesdag.adapters import to_elk, to_networkx
from bayesdag.adapters.graph import markov_blanket
from bayesdag.convert import to_ir
from bayesdag.ir import ModelIR


def test_schema_validates(eight_schools_ir):
    schema.validate(eight_schools_ir.to_dict())


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
