"""``to_ir(obj)`` — idempotent, PPL-agnostic dispatch to a ``ModelIR``.

Mirrors ArviZ's ``convert_to_datatree`` pattern: detect the source by **duck-typed
module/class-name match**, never ``isinstance`` against a PPL type, so the core never
imports pymc/numpyro/stan. New PPL adapters slot in here without touching the IR.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Sequence
from typing import Any

from .ir import ModelIR


def to_ir(obj: Any, idata: Any = None) -> ModelIR:
    if isinstance(obj, ModelIR | dict):
        # Posterior glyphs are computed by the ADAPTER, from the model's random variables —
        # there is nothing left to attach them to once the IR is built. Returning silently
        # would render a prior-only diagram while the caller believes they asked for posteriors.
        if idata is not None:
            warnings.warn(
                "bayesdag.to_ir: idata is ignored for a prebuilt ModelIR/dict — posterior "
                "glyphs are computed by the adapter, so pass the pymc.Model together with "
                "idata (bayesdag.view(model, idata=idata)) to see them.",
                UserWarning,
                stacklevel=2,
            )
        from .schema import check

        out = obj if isinstance(obj, ModelIR) else ModelIR.from_dict(obj)
        check(out)  # a prebuilt IR's references must resolve before anything lays it out
        return out  # idempotent for a ModelIR

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
    """A new ``ModelIR`` restricted to ``var_names`` plus their direct parents (one hop of
    context; PyMC's own ``model_to_graphviz(var_names=...)`` keeps all ancestors). Pure omission — edges are
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

    def kept(refs) -> bool:
        return all(r in keep for r in refs)

    return dataclasses.replace(
        ir,
        nodes=[n for n in ir.nodes if n.id in keep],
        edges=[e for e in ir.edges if e.source in keep and e.target in keep],
        plates=[
            p
            if (p.parent is None or p.parent in plate_ids)
            else dataclasses.replace(p, parent=None)
            for p in plates
        ],
        # a panel or a block about a variable that is no longer shown would dangle: drop it
        # rather than silently widening the selection to complete it
        aux_views=[a for a in ir.aux_views if kept([*a.vars, *(a.edge or [])])],
        blocks=[
            b
            for b in ir.blocks
            if kept([*b.members, *([b.anchor] if b.anchor else [])])
            and kept([x for xs in b.roles.values() for x in xs])
        ],
    )


def drop_index_data(ir: ModelIR) -> ModelIR:
    """A new ``ModelIR`` without the index-only ``pm.Data`` nodes (``usage == "index"``: integer
    lookup tables such as ``county_idx``) — but only those whose name every consumer's equation
    already shows (``a[county_idx]``), so nothing the reader needs leaves the page. Pure
    omission: their edges and plate memberships go with them; the input IR is not mutated."""
    from .labels import symbol_for

    by_id = {n.id: n for n in ir.nodes}
    consumers: dict[str, list[str]] = {}
    for e in ir.edges:
        consumers.setdefault(e.source, []).append(e.target)
    drop = set()
    for n in ir.nodes:
        if n.usage != "index":
            continue
        sym = symbol_for(n.id)
        kids = consumers.get(n.id, [])
        if kids and all(sym in (by_id[k].label_tex or "") for k in kids if k in by_id):
            drop.add(n.id)
    if not drop:
        return ir
    plates = [
        dataclasses.replace(p, members=[m for m in p.members if m not in drop]) for p in ir.plates
    ]
    kept = [p for p in plates if p.members]
    kept_ids = {p.id for p in kept}
    # a plate emptied by the drop disappears; a child plate whose parent went is re-rooted
    kept = [
        p if p.parent is None or p.parent in kept_ids else dataclasses.replace(p, parent=None)
        for p in kept
    ]
    return dataclasses.replace(
        ir,
        nodes=[n for n in ir.nodes if n.id not in drop],
        edges=[e for e in ir.edges if e.source not in drop and e.target not in drop],
        plates=kept,
        blocks=[
            dataclasses.replace(b, members=[m for m in b.members if m not in drop])
            for b in ir.blocks
        ],
    )
