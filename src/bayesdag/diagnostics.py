"""Sampling diagnostics, joined to the graph by node id.

The join is free: a node id IS the constrained ``idata`` variable name, so ``az.rhat(idata)[id]``
lands on the right node with no bookkeeping. That is what lets convergence live *on the diagram*
instead of in a separate table the reader has to align by eye.

**Everything here is a hedged "inspect this", never a verdict.** R-hat above a threshold does not
mean the model is wrong, and a clean R-hat does not mean it is right; a funnel-prone shape is a
reason to look at the geometry, not a diagnosis. The wording lives in ``FLAG_LABELS`` so the
badge, the card and the legend cannot drift into stating something stronger than we know.

Two of the hedges are structural rather than stylistic:

* Vector nodes report their **worst element**. A single number standing for 8 schools would be
  the same lie the pooled-KDE glyph refuses to tell, so the caption says which it is.
* The funnel flag is *structural* — read off the graph, not the samples — so on its own it would
  fire on every non-centered hierarchy that sampled perfectly. It is only surfaced when the run
  actually produced divergences, i.e. when there is something to inspect.

``arviz`` is imported lazily: nothing outside this module needs it.
"""

from __future__ import annotations

from typing import Any

from .ir import ModelIR

# Conventional thresholds. Deliberately the well-known ones rather than anything tuned: the badge
# is an invitation to look, and a reader who knows the convention can calibrate it themselves.
RHAT_THRESHOLD = 1.01
ESS_PER_CHAIN = 100

FLAG_LABELS = {
    "rhat": "chains may not have mixed — inspect the trace",
    "ess": "few effective samples — estimates are noisy",
    "funnel": "funnel-prone geometry — inspect the joint",
}

# scale-family slots: a latent feeding one of these on a PLATED child is the classic funnel neck
_SCALE_SLOTS = frozenset({"scale", "sigma", "tau", "sd"})


def _values(obj, name: str):
    try:
        return float(obj[name].max())
    except Exception:
        return None


def _min_value(obj, name: str):
    try:
        return float(obj[name].min())
    except Exception:
        return None


def per_node(idata: Any, var_names: list[str]) -> dict[str, dict]:
    """``{node_id: {rhat, ess_bulk, ess_tail, n_chains, vector, flags}}`` for the variables that
    are actually in the posterior. Vector nodes report the WORST element, and say so via
    ``vector`` so the caption can qualify the number."""
    try:
        import arviz as az
    except Exception:
        return {}

    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return {}
    present = [v for v in var_names if v in posterior]
    if not present:
        return {}

    try:
        n_chains = int(posterior.sizes.get("chain", 1))
    except Exception:
        n_chains = 1
    # R-hat is meaningless with one chain — it compares between-chain to within-chain variance.
    want_rhat = n_chains > 1

    try:
        rhat = az.rhat(idata, var_names=present) if want_rhat else None
        ess_bulk = az.ess(idata, var_names=present, method="bulk")
        ess_tail = az.ess(idata, var_names=present, method="tail")
    except Exception:
        return {}

    out: dict[str, dict] = {}
    for name in present:
        try:
            vector = int(posterior[name].size // max(posterior[name].sizes.get("draw", 1), 1)) > (
                n_chains
            )
        except Exception:
            vector = False
        entry = {
            "rhat": _values(rhat, name) if rhat is not None else None,
            "ess_bulk": _min_value(ess_bulk, name),
            "ess_tail": _min_value(ess_tail, name),
            "n_chains": n_chains,
            "vector": vector,
            "flags": [],
        }
        if entry["rhat"] is not None and entry["rhat"] > RHAT_THRESHOLD:
            entry["flags"].append("rhat")
        floor = ESS_PER_CHAIN * n_chains
        if entry["ess_bulk"] is not None and entry["ess_bulk"] < floor:
            entry["flags"].append("ess")
        out[name] = entry
    return out


def model_level(idata: Any) -> dict:
    """Whole-run facts: divergence count and the number of draws it is out of. Empty when the
    idata carries no ``sample_stats`` — we never invent a zero."""
    stats = getattr(idata, "sample_stats", None)
    if stats is None:
        return {}
    try:
        if "diverging" not in stats:
            return {}
        diverging = stats["diverging"]
        return {
            "divergences": int(diverging.sum()),
            "draws": int(diverging.size),
        }
    except Exception:
        return {}


def funnel_candidates(ir: ModelIR) -> list[tuple[str, str]]:
    """``[(scale_id, child_id)]`` — latents used as the SCALE of a plated latent child.

    Purely structural: read off ``ParamIR`` slots, no samples involved, so it is testable without
    pymc and costs nothing. This is the shape that produces Neal's funnel — the centered
    hierarchy whose group-level spread is itself being estimated.
    """
    plated = {m for p in ir.plates for m in p.members}
    out: list[tuple[str, str]] = []
    for node in ir.nodes:
        if node.role != "latent" or node.id not in plated:
            continue
        for param in node.params:
            if param.name not in _SCALE_SLOTS:
                continue
            for parent in param.parents:
                parent_node = next((n for n in ir.nodes if n.id == parent), None)
                if parent_node is not None and parent_node.role == "latent":
                    pair = (parent, node.id)
                    if pair not in out:
                        out.append(pair)
    return out


_MAX_POINTS = 1500  # deterministic thinning cap for the non-divergent cloud


def _flat(idata, name: str):
    """A variable's draws as a flat 1-D array (elements pooled), or None."""
    import numpy as np

    try:
        posterior = getattr(idata, "posterior", None)
        if posterior is None or name not in posterior:
            return None
        return np.asarray(posterior[name].values).reshape(-1)
    except Exception:
        return None


def _divergence_mask(idata, n: int):
    import numpy as np

    stats = getattr(idata, "sample_stats", None)
    try:
        if stats is None or "diverging" not in stats:
            return None
        mask = np.asarray(stats["diverging"].values).reshape(-1)
        if mask.size == n:
            return mask
        # a vector child pools k elements per draw, so each draw's flag repeats k times
        if n % mask.size == 0:
            return np.repeat(mask, n // mask.size)
    except Exception:
        return None
    return None


def funnel_joint(idata: Any, scale_id: str, child_id: str, unconstrained_key: str | None) -> dict:
    """Points for the funnel joint: the child against its scale on the LOG axis, divergences apart.

    The neck is only visible on the unconstrained scale, so this prefers the sampler's own
    ``unconstrained_posterior`` when the idata carries it and otherwise takes the log itself —
    saying which, because a computed axis is a slightly different object from the one the sampler
    actually explored.

    Every divergent draw is kept; the rest are thinned deterministically (a stride, never an RNG,
    so the same idata always yields the same picture). Returns ``{}`` when there is nothing to
    draw, so callers can treat "no data" and "no funnel" the same way.
    """
    import numpy as np

    child = _flat(idata, child_id)
    if child is None or child.size == 0:
        return {}

    axis_space, scale = "unconstrained", None
    if unconstrained_key:
        raw = None
        for group in ("unconstrained_posterior", "posterior"):
            ds = getattr(idata, group, None)
            if ds is not None and unconstrained_key in ds:
                raw = np.asarray(ds[unconstrained_key].values).reshape(-1)
                break
        if raw is not None:
            scale, computed = raw, False
    if scale is None:
        constrained = _flat(idata, scale_id)
        if constrained is None or constrained.size == 0:
            return {}
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.log(constrained)
        computed = True

    # a vector child pools k elements per draw; repeat the scalar scale to match
    if child.size != scale.size:
        if child.size % scale.size:
            return {}
        scale = np.repeat(scale, child.size // scale.size)

    ok = np.isfinite(child) & np.isfinite(scale)
    child, scale = child[ok], scale[ok]
    if child.size == 0:
        return {}

    mask = _divergence_mask(idata, ok.size)
    mask = mask[ok] if mask is not None and mask.size == ok.size else np.zeros(child.size, bool)

    div_x, div_y = scale[mask], child[mask]
    keep_x, keep_y = scale[~mask], child[~mask]
    if keep_x.size > _MAX_POINTS:  # deterministic stride, never an RNG
        step = keep_x.size // _MAX_POINTS + 1
        keep_x, keep_y = keep_x[::step], keep_y[::step]

    return {
        "x": [float(v) for v in keep_x],
        "y": [float(v) for v in keep_y],
        "div_x": [float(v) for v in div_x],
        "div_y": [float(v) for v in div_y],
        "x_label": f"log({scale_id})" + (" (computed)" if computed else ""),
        "y_label": child_id,
        "axis_space": axis_space if not computed else "computed log",
        "n_divergent": int(mask.sum()),
        "n_total": int(mask.size),
    }


def joint_views(ir: ModelIR, idata: Any) -> list:
    """``AuxViewIR`` joints for every funnel candidate that a divergent run gives us data for.

    Only built when there are divergences: a funnel-shaped posterior that sampled cleanly is not
    something to send the reader looking at.
    """
    from .ir import AuxViewIR

    if idata is None or not model_level(idata).get("divergences"):
        return []
    out = []
    for scale_id, child_id in funnel_candidates(ir):
        node = next((n for n in ir.nodes if n.id == scale_id), None)
        data = funnel_joint(
            idata, scale_id, child_id, getattr(node, "idata_unconstrained_key", None)
        )
        if data:
            out.append(
                AuxViewIR(
                    kind="joint",
                    vars=[child_id, scale_id],
                    edge=[scale_id, child_id],
                    axis_space="unconstrained",
                    data_ref=data,
                )
            )
    return out


def annotate(ir: ModelIR, idata: Any) -> dict:
    """Attach ``NodeIR.diag`` for every node and return the model-level summary.

    Called with ``idata=None`` this CLEARS every annotation, so a view toggled back to its prior
    does not keep showing diagnostics from a run it is no longer displaying.
    """
    if idata is None:
        for node in ir.nodes:
            node.diag = None
        ir.aux_views = []
        return {}

    stats = per_node(idata, [n.id for n in ir.nodes])
    summary = model_level(idata)
    ir.aux_views = joint_views(ir, idata)

    # Structure alone would flag every non-centered hierarchy that sampled perfectly well, so the
    # funnel hint only appears when the run actually produced divergences to explain.
    if summary.get("divergences"):
        for scale_id, _child in funnel_candidates(ir):
            stats.setdefault(scale_id, {"flags": []})["flags"].append("funnel")

    for node in ir.nodes:
        entry = stats.get(node.id)
        node.diag = entry if entry and entry.get("flags") is not None else None
    return summary


def describe(diag: dict | None) -> list[str]:
    """Human-readable rows for a node's pinned card. Hedged wording, single-sourced."""
    if not diag:
        return []
    rows: list[str] = []
    qualifier = " (worst element)" if diag.get("vector") else ""
    if diag.get("rhat") is not None:
        rows.append(f"R-hat {diag['rhat']:.3f}{qualifier}")
    elif diag.get("n_chains") == 1:
        rows.append("R-hat needs >1 chain")
    if diag.get("ess_bulk") is not None:
        rows.append(f"ESS bulk {diag['ess_bulk']:.0f}{qualifier}")
    if diag.get("ess_tail") is not None:
        rows.append(f"ESS tail {diag['ess_tail']:.0f}{qualifier}")
    rows += [FLAG_LABELS[f] for f in diag.get("flags", []) if f in FLAG_LABELS]
    return rows
