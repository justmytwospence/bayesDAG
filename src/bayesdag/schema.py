"""Published JSON Schema for the topology layer of the IR (draft 2020-12).

Generated from the ``bayesdag.ir`` dataclasses so the contract stays in sync with the code.
``validate(data)`` checks an IR dict; ``write_schema_file`` publishes it to
``schema/graph-v1.0.json`` (the language-neutral contract other tools can consume).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from . import ir

SCHEMA_ID = "https://bayesdag.dev/schema/graph-v1.0.json"
SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
_SCHEMA_FILE = Path(__file__).parent / "schema" / "graph-v1.0.json"


def _type_schema(hint: Any, defs: dict[str, Any]) -> dict[str, Any]:
    if hint in (Any, None) or hint is type(None):
        return {} if hint is Any else {"type": "null"}
    origin = get_origin(hint)
    if origin is Literal:
        return {"enum": list(get_args(hint))}
    if origin is Union:
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


def write_schema_file(path: Path | None = None) -> Path:
    path = path or _SCHEMA_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_schema(), indent=2) + "\n")
    return path


if __name__ == "__main__":  # `python -m bayesdag.schema` regenerates the published file
    print("wrote", write_schema_file())
