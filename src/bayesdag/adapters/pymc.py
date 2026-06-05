"""``from_pymc(model, idata=None) -> ModelIR``.

All PyMC-isms live here (the rest of bayesdag never imports pymc). We reuse PyMC's own
``ModelGraph`` for the authoritative topology (it already handles observed-edge reversal
and plate grouping) and add the one thing it lacks: **slot-aware** parameter extraction,
so an edge can point at the specific parameter token in the child's equation.

Built against pymc 6.x; relies only on stable surfaces (`model_graph.ModelGraph`,
`model.{free_RVs,observed_RVs,deterministics,data_vars,potentials,named_vars,
named_vars_to_dims,rvs_to_transforms,coords}`, and `op.dist_params` / `op._print_name`).
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

from .. import labels
from ..ir import EdgeIR, Meta, ModelIR, NodeIR, OverlayRef, ParamIR, PlateIR
from .glyph_data import glyph_for
from .pytensor_latex import render_value

_SKIP_PARAMS = {"self", "size", "rng", "dtype", "name", "kwargs", "args"}


def _param_names(op: Any, n: int) -> list[str]:
    """Ordered op-level parameter names (loc, scale, ...) aligned to ``dist_params``.

    Names come from the op's ``__call__`` signature — NOT PyMC's ``dist()`` kwargs, which
    are reparameterized (e.g. ``HalfNormal(sigma=...)`` -> positional ``[0.0, sigma]``)."""
    try:
        sig = inspect.signature(type(op).__call__)
        names = [
            p.name
            for p in sig.parameters.values()
            if p.name not in _SKIP_PARAMS
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)
        ]
    except (TypeError, ValueError):
        names = []
    names = names[:n]
    names += [f"arg{i}" for i in range(len(names), n)]
    return names


def _direct_named_parents(value: Any, named: dict[int, str], exclude: Optional[str] = None) -> list[str]:
    """Named model vars that DIRECTLY feed ``value`` (stop descending at named boundaries;
    raw ``ancestors`` would over-collect transitive parents)."""
    if id(value) in named:
        nm = named[id(value)]
        return [] if nm == exclude else [nm]
    out: list[str] = []
    seen: set[int] = set()
    stack = [value]
    while stack:
        v = stack.pop()
        if id(v) in seen:
            continue
        seen.add(id(v))
        owner = getattr(v, "owner", None)
        if owner is None:
            continue
        for inp in owner.inputs:
            if id(inp) in named:
                nm = named[id(inp)]
                if nm != exclude and nm not in out:
                    out.append(nm)
            else:
                stack.append(inp)
    return out


def _rv_dist_and_params(var: Any, named: dict[int, str]) -> tuple[Optional[str], list[ParamIR]]:
    node = var.owner
    op = node.op
    pn = getattr(op, "_print_name", None)
    dist = pn[0] if pn else type(op).__name__.removesuffix("RV")
    try:
        dparams = list(op.dist_params(node))
    except Exception:
        dparams = list(node.inputs[2:])  # (rng, size, *params)
    names = _param_names(op, len(dparams))
    params: list[ParamIR] = []
    for i, (nm, val) in enumerate(zip(names, dparams)):
        parents = _direct_named_parents(val, named, exclude=var.name)
        value_tex, _ = render_value(val, named, wrap_leaves=False)
        if value_tex.strip() == r"\ldots":  # an elided param that is itself an RV -> show its family
            sym = _rv_family_symbol(val)
            if sym:
                value_tex = sym
        params.append(ParamIR(index=i, name=nm, token_id=nm, parents=parents, value_tex=value_tex))
    return dist, params


def _rv_family_symbol(val: Any) -> Optional[str]:
    """The LaTeX symbol of a param that is itself a random variable (e.g. a mixture's component),
    so it renders as its family (``\\mathcal{N}``) instead of the bare elision ``\\ldots``."""
    op = getattr(getattr(val, "owner", None), "op", None)
    if op is None:
        return None
    pn = getattr(op, "_print_name", None)
    name = pn[0] if pn else type(op).__name__.removesuffix("RV")
    sym = labels.dist_symbol(name)
    return sym if "operatorname" not in sym else None


def _overlays(name: str, role: str, dims: list, idata: Any) -> list[OverlayRef]:
    if idata is None:
        return []
    try:
        groups = set(idata.groups())
    except Exception:
        return []
    out: list[OverlayRef] = []
    if "posterior" in groups and name in getattr(idata, "posterior", {}):
        out.append(OverlayRef("posterior", name, list(dims)))
    if "prior" in groups and name in getattr(idata, "prior", {}):
        out.append(OverlayRef("prior", name, list(dims)))
    if role == "observed" and "observed_data" in groups and name in getattr(idata, "observed_data", {}):
        out.append(OverlayRef("observed_data", name, list(dims)))
    return out


def from_pymc(model: Any, idata: Any = None) -> ModelIR:
    from pymc.model_graph import ModelGraph

    g = ModelGraph(model)
    named: dict[int, str] = {id(v): nm for nm, v in model.named_vars.items()}
    role_of: dict[int, str] = {}
    for v in model.observed_RVs:
        role_of[id(v)] = "observed"
    for v in model.free_RVs:
        role_of.setdefault(id(v), "latent")
    for v in model.deterministics:
        role_of[id(v)] = "deterministic"
    for v in getattr(model, "data_vars", []):
        role_of[id(v)] = "data"
    for v in model.potentials:
        role_of[id(v)] = "potential"

    coords = {k: list(v) for k, v in (model.coords or {}).items() if v is not None}
    n2d = dict(model.named_vars_to_dims)
    transforms = model.rvs_to_transforms

    compute = g.make_compute_graph()
    var_names = list(compute)

    rv_param_map: dict[str, dict[str, str]] = {}  # child -> {parent_name: token_id}
    det_tokens: dict[str, set[str]] = {}  # deterministic child -> leaf names wrapped in its expr
    nodes: list[NodeIR] = []
    for name in var_names:
        var = model[name]
        role = role_of.get(id(var), "latent")
        observed = role == "observed"
        dist: Optional[str] = None
        params: list[ParamIR] = []
        label_tex, label_tree = labels.assemble_bare(name)
        if role in ("latent", "observed") and getattr(var, "owner", None) is not None:
            dist, params = _rv_dist_and_params(var, named)
            pm: dict[str, str] = {}
            for pr in params:
                for parent in pr.parents:
                    pm.setdefault(parent, pr.token_id)
            rv_param_map[name] = pm
            label_tex, label_tree = labels.assemble_stochastic(
                name, dist, [(pr.token_id, pr.value_tex or "") for pr in params]
            )
        elif role == "deterministic" and getattr(var, "owner", None) is not None:
            expr_tex, used = render_value(var, named, wrap_leaves=True, _root=True)
            # nested unknown ops (f(f(...))) = auto-generated matrix/plumbing (e.g. LKJ corr/stds,
            # OrderedLogistic class probs). They render as an illegible f-mess -> elide honestly.
            if r"f\!\left(f\!\left" in expr_tex:
                expr_tex, used = r"[\,\cdots\,]", set()
            det_tokens[name] = used
            label_tex, label_tree = labels.assemble_deterministic(name, expr_tex, sorted(used))
        dims = list(n2d.get(name, ()))
        node_coords = {d: coords[d] for d in dims if d in coords} or None
        tr = transforms.get(var) if hasattr(transforms, "get") else None
        tname = getattr(tr, "name", None) if tr is not None else None
        unconstrained = f"{name}_{tname}__" if tname else None
        glyph_spec, glyph_data, elision = glyph_for(var, role, dist, model, idata, named=named)
        # A deterministic whose equation was elided to illegible matrix plumbing ([⋯]) shouldn't carry a
        # transfer glyph either (the function isn't a meaningful modeling transform — e.g. LKJ corr/stds).
        # Match ONLY the plumbing marker \cdots; \ldots is a legitimate long-vector/budget abbreviation.
        if role == "deterministic" and glyph_spec is not None and r"\cdots" in label_tex:
            glyph_spec, glyph_data = None, None
        nodes.append(
            NodeIR(
                id=name,
                role=role,
                observed=observed,
                dist=dist,
                params=params,
                dims=dims,
                coords=node_coords,
                label_tex=label_tex,
                label_tree=label_tree,
                transform=tname,
                idata_unconstrained_key=unconstrained,
                glyph=glyph_spec,
                glyph_data=glyph_data,
                overlays=_overlays(name, role, dims, idata),
                representable=elision is None,
                elision_reason=elision,
            )
        )

    edges: list[EdgeIR] = []
    for child, parents in compute.items():
        pmap = rv_param_map.get(child, {})
        used = det_tokens.get(child, set())
        for parent in sorted(parents):
            token = pmap.get(parent) or (parent if parent in used else None)
            edges.append(EdgeIR(source=parent, target=child, target_token_id=token))

    plates: list[PlateIR] = []
    try:
        raw_plates = list(g.get_plates())  # pymc evals each var's shape here
    except Exception:
        raw_plates = None
    if raw_plates is not None:
        for plate in raw_plates:
            di = plate.dim_info
            if not di.names:  # the empty-dims group is just ungrouped scalars, not a plate
                continue
            label = " x ".join(
                f"{nm} ({ln})" if nm else f"{ln}" for nm, ln in zip(di.names, di.lengths)
            )
            pid = "plate_" + "_".join(str(nm) for nm in di.names)
            plates.append(PlateIR(id=pid, label=label, members=[ni.var.name for ni in plate.variables]))
    else:
        # A non-samplable RV (Flat/HalfFlat/ICAR, …) broke pymc's shape eval. Derive plates
        # eval-free from the named dims + coords we already hold, grouping by dim signature.
        groups: dict[tuple, list[str]] = {}
        for nm in var_names:
            dims = tuple(d for d in n2d.get(nm, ()) if d)
            if dims and all(d in coords for d in dims):
                groups.setdefault(dims, []).append(nm)
        for dims, members in groups.items():
            label = " x ".join(f"{d} ({len(coords[d])})" for d in dims)
            pid = "plate_" + "_".join(str(d) for d in dims)
            plates.append(PlateIR(id=pid, label=label, members=members))

    meta = Meta.stamp(source_ppl="pymc", model_name=getattr(model, "name", None) or None)
    return ModelIR(nodes=nodes, edges=edges, plates=plates, meta=meta)
