"""Published JSON Schema for the topology layer of the IR (draft 2020-12).

Generated from the ``bayesdag.ir`` dataclasses so the contract stays in sync with the code.
``validate(data)`` checks an IR dict; ``write_schema_file`` publishes it to
``schema/graph-v<version>.json`` (the language-neutral contract other tools can consume).
"""

from __future__ import annotations

import dataclasses
import json
import types
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from . import ir

SCHEMA_ID = f"https://bayesdag.dev/schema/graph-v{ir.SCHEMA_VERSION}.json"
SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
_SCHEMA_DIR = Path(__file__).parent / "schema"
_SCHEMA_FILE = _SCHEMA_DIR / f"graph-v{ir.SCHEMA_VERSION}.json"
# Every published version stays in the wheel: an artifact written by an older bayesdag must stay
# checkable against the contract it was written under.
PUBLISHED = ("1.0", ir.SCHEMA_VERSION)


def _type_schema(hint: Any, defs: dict[str, Any]) -> dict[str, Any]:
    if hint in (Any, None) or hint is type(None):
        return {} if hint is Any else {"type": "null"}
    origin = get_origin(hint)
    if origin is Literal:
        return {"enum": list(get_args(hint))}
    if origin is Union or origin is types.UnionType:
        return {"anyOf": [_type_schema(a, defs) for a in get_args(hint)]}
    if origin in (list, tuple):
        args = get_args(hint)
        item = _type_schema(args[0], defs) if args and args[0] is not Ellipsis else {}
        return {"type": "array", "items": item}
    if origin is dict:
        args = get_args(hint)
        val = _type_schema(args[1], defs) if len(args) == 2 else {}
        return {"type": "object", "additionalProperties": val}
    if dataclasses.is_dataclass(hint):
        return {"$ref": _register(hint, defs)}
    if hint is bool:
        return {"type": "boolean"}
    if hint is int:
        return {"type": "integer"}
    if hint is float:
        return {"type": "number"}
    if hint is str:
        return {"type": "string"}
    return {}


def _register(cls: type, defs: dict[str, Any]) -> str:
    name = cls.__name__
    ref = f"#/$defs/{name}"
    if name in defs:
        return ref
    defs[name] = {}  # placeholder to break recursion (e.g. TokenIR.children)
    hints = get_type_hints(cls)
    props: dict[str, Any] = {}
    required: list[str] = []
    for f in dataclasses.fields(cls):
        props[f.name] = _type_schema(hints.get(f.name, Any), defs)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:  # type: ignore[misc]
            required.append(f.name)
    schema: dict[str, Any] = {"type": "object", "properties": props, "additionalProperties": True}
    if required:
        schema["required"] = required
    defs[name] = schema
    return ref


def build_schema() -> dict[str, Any]:
    defs: dict[str, Any] = {}
    root_ref = _register(ir.ModelIR, defs)
    return {
        "$schema": SCHEMA_DIALECT,
        "$id": SCHEMA_ID,
        "title": f"bayesdag Model-Graph IR v{ir.SCHEMA_VERSION}",
        "$ref": root_ref,
        "$defs": defs,
    }


def validate(data: dict[str, Any]) -> None:
    """Raise ``jsonschema.ValidationError`` if ``data`` is not a valid IR dict."""
    import jsonschema

    jsonschema.validate(instance=data, schema=build_schema())


def _tokens(n: ir.NodeIR) -> set[str] | None:
    """The tokens an edge may land on for this node, or None when the node carries no token
    information at all (a hand-built minimal IR) — then there is nothing to check against."""
    toks = {p.token_id for p in n.params}
    stack = [n.label_tree] if n.label_tree is not None else []
    while stack:
        t = stack.pop()
        toks.add(t.token_id)
        stack += t.children
    return toks or None


def check(model_ir: ir.ModelIR) -> None:
    """Semantic validation the JSON Schema cannot express: references resolve.

    Unique node ids and ``(source, target)`` pairs; known edge endpoints and target tokens;
    known plate members and parents with no parent cycle; known block and aux-view references.
    Raises ``ValueError`` listing every problem found."""
    problems: list[str] = []
    ids = [n.id for n in model_ir.nodes]
    known = set(ids)
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        problems.append(f"duplicate node ids {dup}")
    pairs = [(e.source, e.target) for e in model_ir.edges]
    dpairs = sorted({p for p in pairs if pairs.count(p) > 1})
    if dpairs:
        problems.append(f"duplicate edges {dpairs}")
    by_id = {n.id: n for n in model_ir.nodes}
    for e in model_ir.edges:
        for end in (e.source, e.target):
            if end not in known:
                problems.append(f"edge {e.source}->{e.target}: unknown node {end!r}")
        tgt = by_id.get(e.target)
        toks = _tokens(tgt) if tgt is not None else None
        for t in [e.target_token_id, *e.target_token_ids]:
            if t and toks is not None and t not in toks:
                problems.append(f"edge {e.source}->{e.target}: unknown token {t!r}")
    plate_ids = [p.id for p in model_ir.plates]
    if len(plate_ids) != len(set(plate_ids)):
        problems.append("duplicate plate ids")
    parent_of = {p.id: p.parent for p in model_ir.plates}
    for p in model_ir.plates:
        missing = [m for m in p.members if m not in known]
        if missing:
            problems.append(f"plate {p.id}: unknown members {missing}")
        if p.parent is not None and p.parent not in parent_of:
            problems.append(f"plate {p.id}: unknown parent {p.parent!r}")
        seen, q = {p.id}, p.parent
        while q is not None:
            if q in seen:
                problems.append(f"plate {p.id}: parent cycle")
                break
            seen.add(q)
            q = parent_of.get(q)
    for b in model_ir.blocks:
        refs = list(b.members) + ([b.anchor] if b.anchor else [])
        refs += [x for xs in b.roles.values() for x in xs]
        missing = sorted({r for r in refs if r not in known})
        if missing:
            problems.append(f"block {b.id}: unknown nodes {missing}")
    for a in model_ir.aux_views:
        missing = [v for v in [*a.vars, *(a.edge or [])] if v not in known]
        if missing:
            problems.append(f"aux view {a.kind}: unknown nodes {missing}")
    if problems:
        raise ValueError("bayesdag: invalid IR — " + "; ".join(problems))


def write_schema_file(path: Path | None = None) -> Path:
    path = path or _SCHEMA_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_schema(), indent=2) + "\n")
    return path


if __name__ == "__main__":  # `python -m bayesdag.schema` regenerates the published file
    print("wrote", write_schema_file())
