"""``to_ir(obj)`` — idempotent, PPL-agnostic dispatch to a ``ModelIR``.

Mirrors ArviZ's ``convert_to_datatree`` pattern: detect the source by **duck-typed
module/class-name match**, never ``isinstance`` against a PPL type, so the core never
imports pymc/numpyro/stan. New PPL adapters slot in here without touching the IR.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .ir import ModelIR


def to_ir(obj: Any, idata: Any = None) -> ModelIR:
    if isinstance(obj, ModelIR):
        return obj  # idempotent
    if isinstance(obj, dict):
        return ModelIR.from_dict(obj)  # low-level escape hatch

    cls = type(obj)
    module = getattr(cls, "__module__", "") or ""
    name = cls.__name__

    if name == "Model" and module.startswith("pymc"):
        from .adapters.pymc import from_pymc

        return from_pymc(obj, idata=idata)

    # future: numpyro (`MCMC`/handlers), stan (cmdstanpy) -> from_numpyro / from_stan
    raise TypeError(
        f"bayesdag.to_ir: don't know how to convert {module}.{name}. "
        "Pass a pymc.Model, a ModelIR, or a ModelIR dict."
    )


def subgraph(ir: ModelIR, var_names: Sequence[str]) -> ModelIR:
    """A new ``ModelIR`` restricted to ``var_names`` plus their direct parents (the same
    context rule as ``pm.model_to_graphviz(var_names=...)``). Pure omission — edges are
    induced on the kept set, plate member lists pruned, empty plates dropped; nothing is
    fabricated and the input IR is not mutated."""
    ids = {n.id for n in ir.nodes}
    unknown = [v for v in var_names if v not in ids]
    if unknown:
        raise ValueError(
            f"bayesdag.subgraph: unknown variable name(s) {unknown}; available: {sorted(ids)}"
        )
    keep = set(var_names)
    for e in ir.edges:  # one hop of context: the direct parents of every selected node
        if e.target in var_names:
            keep.add(e.source)
    plates = []
    for p in ir.plates:
        members = [m for m in p.members if m in keep]
        if members:
            plates.append(dataclasses.replace(p, members=members))
    plate_ids = {p.id for p in plates}
    return dataclasses.replace(
        ir,
        nodes=[n for n in ir.nodes if n.id in keep],
        edges=[e for e in ir.edges if e.source in keep and e.target in keep],
        plates=[p if (p.parent is None or p.parent in plate_ids) else dataclasses.replace(p, parent=None) for p in plates],
    )
