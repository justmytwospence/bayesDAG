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


def annotate(ir: ModelIR, idata: Any) -> dict:
    """Attach ``NodeIR.diag`` for every node and return the model-level summary.

    Called with ``idata=None`` this CLEARS every annotation, so a view toggled back to its prior
    does not keep showing diagnostics from a run it is no longer displaying.
    """
    if idata is None:
        for node in ir.nodes:
            node.diag = None
        return {}

    stats = per_node(idata, [n.id for n in ir.nodes])
    summary = model_level(idata)

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
