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
  fire on every centered hierarchy, including the many that sample perfectly. It is only surfaced when the run
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


def _extreme(obj, name: str, *, worst_is_max: bool) -> tuple[float | None, str | None]:
    """The worst element's value and a readable label for WHICH element it is (``school=S3``),
    so a vector node's single number can be traced back to its coordinate."""
    try:
        da = obj[name]
    except Exception:
        return None, None
    try:
        if da.ndim == 0:
            return float(da), None
        import numpy as np

        vals = np.asarray(da.values, float)
        flat_i = int(np.nanargmax(vals) if worst_is_max else np.nanargmin(vals))
        idx = np.unravel_index(flat_i, vals.shape)
        parts = []
        for dim, i in zip(da.dims, idx, strict=True):
            coord = da.coords[dim].values[i] if dim in da.coords else i
            parts.append(f"{dim}={coord}")
        return float(vals[idx]), ", ".join(parts)
    except Exception:
        return None, None


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
        r, r_at = _extreme(rhat, name, worst_is_max=True) if rhat is not None else (None, None)
        eb, eb_at = _extreme(ess_bulk, name, worst_is_max=False)
        et, _ = _extreme(ess_tail, name, worst_is_max=False)
        entry = {
            "rhat": r,
            "ess_bulk": eb,
            "ess_tail": et,
            "n_chains": n_chains,
            "vector": vector,
            "flags": [],
        }
        if vector:  # which element the worst number belongs to
            entry["rhat_at"] = r_at
            entry["ess_at"] = eb_at
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
_SAMPLE_DIMS = ("chain", "draw")


def _draws(ds, name: str):
    """``ds[name]`` transposed to ``(chain, draw, ...)`` — never flattened in storage order, which
    is what silently paired the wrong draws when a group was stored ``(draw, chain)``."""
    if ds is None or name not in ds:
        return None
    da = ds[name]
    if not all(d in da.dims for d in _SAMPLE_DIMS):
        return None
    return da.transpose(*_SAMPLE_DIMS, ...)


def _align(*arrays):
    """Align DataArrays on chain/draw by their coordinates. Reorders matching labels; returns
    ``None`` when the draw sets differ (an inner join would silently discard draws, an override
    would relabel them)."""
    import xarray as xr

    try:
        out = xr.align(*arrays, join="inner")
    except Exception:
        return None
    for a, b in zip(arrays, out, strict=True):
        if any(a.sizes[d] != b.sizes[d] for d in _SAMPLE_DIMS):
            return None
    return out


def funnel_joint(
    idata: Any,
    scale_id: str,
    child_id: str,
    unconstrained_key: str | None,
    transform: str | None = None,
) -> dict:
    """Points for the funnel joint: the child against its scale, divergences apart.

    The neck is only visible on the log scale, so this prefers the sampler's own log-transformed
    ``unconstrained_posterior`` draws and otherwise takes the log itself — saying which. A log axis
    is used only for a scale that is actually positive; anything else keeps its own axis.

    Draws are joined by their chain/draw COORDINATES, never by position in a flattened array; a
    vector child overlays its k elements against the same per-draw scale (and says so). Counts
    are of transitions, not element-points. Every divergent draw is kept; the rest are thinned
    deterministically. Returns ``{}`` when there is nothing trustworthy to draw.
    """
    import numpy as np

    post = getattr(idata, "posterior", None)
    child = _draws(post, child_id)
    if child is None:
        return {}

    computed = True
    scale = None
    if unconstrained_key and transform == "log":
        for group in ("unconstrained_posterior", "posterior"):
            scale = _draws(getattr(idata, group, None), unconstrained_key)
            if scale is not None:
                computed = False
                break
    if scale is None:
        scale = _draws(post, scale_id)
        if scale is None:
            return {}
    if scale.ndim != 2:  # a vector scale has no single per-draw value to plot against
        return {}

    stats = getattr(idata, "sample_stats", None)
    div = _draws(stats, "diverging")
    aligned = _align(child, scale, div) if div is not None else _align(child, scale)
    if aligned is None:
        return {}
    child, scale = aligned[0], aligned[1]
    div = aligned[2] if div is not None else None

    sv = np.asarray(scale.values, float)  # (chain, draw)
    log_axis = not computed
    if computed:
        with np.errstate(divide="ignore", invalid="ignore"):
            if np.all(sv[np.isfinite(sv)] > 0):
                sv, log_axis = np.log(sv), True
    cv = np.asarray(child.values, float).reshape(sv.shape[0], sv.shape[1], -1)  # (c, d, k)
    k = cv.shape[2]
    mask_t = (
        np.asarray(div.values, bool) if div is not None else np.zeros(sv.shape, bool)
    )  # per transition

    xs = np.repeat(sv[:, :, None], k, axis=2).reshape(-1)
    ys = cv.reshape(-1)
    mask = np.repeat(mask_t[:, :, None], k, axis=2).reshape(-1)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys, mask = xs[ok], ys[ok], mask[ok]
    if xs.size == 0:
        return {}

    div_x, div_y = xs[mask], ys[mask]
    keep_x, keep_y = xs[~mask], ys[~mask]
    thinned = keep_x.size > _MAX_POINTS
    if thinned:  # deterministic stride, never an RNG
        step = keep_x.size // _MAX_POINTS + 1
        keep_x, keep_y = keep_x[::step], keep_y[::step]

    if log_axis:
        x_label = f"log({scale_id})" + (" (computed)" if computed else "")
        axis_space = "computed log" if computed else "unconstrained"
    else:
        x_label, axis_space = scale_id, "constrained"
    return {
        "x": [float(v) for v in keep_x],
        "y": [float(v) for v in keep_y],
        "div_x": [float(v) for v in div_x],
        "div_y": [float(v) for v in div_y],
        "x_label": x_label,
        "y_label": child_id + (f" ({k} elements overlaid)" if k > 1 else ""),
        "axis_space": axis_space,
        # transitions, not element-points; None = the run did not record divergences
        "n_divergent": int(mask_t.sum()) if div is not None else None,
        "n_total": int(mask_t.size),
        "n_elements": int(k),
        "thinned": bool(thinned),
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
            idata,
            scale_id,
            child_id,
            getattr(node, "idata_unconstrained_key", None),
            getattr(node, "transform", None),
        )
        if data:
            out.append(
                AuxViewIR(
                    kind="joint",
                    vars=[child_id, scale_id],
                    edge=[scale_id, child_id],
                    axis_space=data["axis_space"],
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

    # Structure alone would flag every centered hierarchy, even one that sampled perfectly, so the
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
    vector = diag.get("vector")

    def worst(at):
        if not vector:
            return ""
        return f" (worst element: {at})" if at else " (worst element)"

    if diag.get("rhat") is not None:
        rows.append(f"R-hat {diag['rhat']:.3f}{worst(diag.get('rhat_at'))}")
    elif diag.get("n_chains") == 1:
        rows.append("R-hat needs >1 chain")
    if diag.get("ess_bulk") is not None:
        rows.append(f"ESS bulk {diag['ess_bulk']:.0f}{worst(diag.get('ess_at'))}")
    if diag.get("ess_tail") is not None:
        rows.append(f"ESS tail {diag['ess_tail']:.0f}{worst(None)}")
    rows += [FLAG_LABELS[f] for f in diag.get("flags", []) if f in FLAG_LABELS]
    return rows
