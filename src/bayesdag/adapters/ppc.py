"""Plate prior-predictive expansion data (PyMC-specific).

For each plate, forward-simulate from the priors (``pm.sample_prior_predictive`` — no MCMC),
then for each plate-member variable compute the per-instance marginal density across the
plate dimension on a shared axis (the parent hyperparameters integrated out by the sim).
Observed members also carry the observed data points → a genuine prior predictive check.

Interactive-only and computed lazily (when a widget is built), so static rendering pays
nothing. The PPL-agnostic renderer turns this data into the expansion panel.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .glyph_data import _DISCRETE, _observed_values, _thin

_GRID = 48
_MAX_CURVES = 40


def _kde(col: np.ndarray, xs: np.ndarray):
    col = col[np.isfinite(col)]
    if col.size < 2 or np.allclose(col, col[0]):
        return None
    col = _thin(col)
    try:
        from scipy.stats import gaussian_kde

        return gaussian_kde(col)(xs)
    except Exception:
        return None


def potential_note(model: Any) -> str | None:
    """Why forward simulation would misrepresent this model, or None. ``pm.Potential`` terms
    reweight the joint density but are invisible to ancestral sampling, so prior-predictive draws
    would come from a DIFFERENT model than the one drawn."""
    n = len(getattr(model, "potentials", []) or [])
    if not n:
        return None
    return f"forward simulation ignores the model's {n} Potential{'s' if n != 1 else ''}"


def _discrete_curves(flat: np.ndarray, n_inst: int):
    """Per-instance relative frequencies on a shared INTEGER support (a KDE would put mass on
    values a count cannot take)."""
    finite = flat[np.isfinite(flat)]
    if finite.size == 0 or not np.all(finite == np.round(finite)):
        return None
    lo, hi = np.percentile(finite, [0.5, 99.5])
    lo, hi = int(np.floor(lo)), int(np.ceil(hi))
    hi = min(hi, lo + 60)
    cats = np.arange(lo, hi + 1)
    curves = []
    for i in range(min(n_inst, _MAX_CURVES)):
        col = flat[:, i]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        curves.append(np.array([np.mean(col == c) for c in cats], float))
    return (cats.astype(float), curves) if curves else None


def prior_predictive_expansions(model: Any, ir: Any, draws: int = 200) -> dict[str, dict]:
    """``{plate_id: {"status": "ready", "label", "members"}}`` — or, per plate, a
    ``{"status": "unavailable" | "error", "reason"}`` the widget shows instead of an empty box."""
    plates = [p for p in ir.plates if p.members]
    if not plates:
        return {}
    note = potential_note(model)
    if note:
        return {p.id: {"status": "unavailable", "reason": note} for p in plates}
    try:
        import pymc as pm

        with model:
            pp = pm.sample_prior_predictive(draws=draws, random_seed=0)
    except Exception as exc:
        reason = f"prior-predictive simulation failed ({type(exc).__name__})"
        return {p.id: {"status": "error", "reason": reason} for p in plates}

    def get_draws(name: str):
        for grp in ("prior", "prior_predictive"):
            ds = getattr(pp, grp, None)
            if ds is not None and name in ds:
                return np.asarray(ds[name].values)  # (chain, draw, *dims)
        return None

    node_by_id = {n.id: n for n in ir.nodes}
    out: dict[str, dict] = {}
    for plate in plates:
        members = []
        for vid in plate.members:
            node = node_by_id.get(vid)
            arr = get_draws(vid)
            if node is None or arr is None or arr.ndim < 3:
                continue  # need at least (chain, draw, instance-dim)
            flat = arr.reshape(arr.shape[0] * arr.shape[1], -1)  # (samples, N_instances)
            n_inst = flat.shape[1]
            finite = flat[np.isfinite(flat)]
            if finite.size < 2:
                continue
            discrete = bool(node.dist in _DISCRETE) if node.dist else False
            if discrete:
                got = _discrete_curves(flat, n_inst)
                if got is None:
                    continue
                xs, curves = got
            else:
                lo, hi = np.percentile(finite, [0.5, 99.5])
                if not (hi > lo):
                    continue
                xs = np.linspace(lo, hi, _GRID)
                curves = []
                for i in range(min(n_inst, _MAX_CURVES)):
                    ys = _kde(flat[:, i], xs)
                    if ys is not None:
                        curves.append(ys)
            if not curves:
                continue
            peak = max(float(c.max()) for c in curves) or 1.0
            member = {
                "id": vid,
                "role": node.role,
                "n": n_inst,
                "capped": n_inst > _MAX_CURVES,
                "discrete": discrete,
                "xs": [float(x) for x in xs],
                "curves": [[float(y / peak) for y in c] for c in curves],
            }
            if node.role == "observed":
                obs = _observed_values(model[vid], model)
                if obs is not None:
                    o = np.asarray(obs, float).ravel()
                    member["observed"] = [float(v) for v in o[np.isfinite(o)]]
            members.append(member)
        if members:
            out[plate.id] = {"status": "ready", "label": plate.label, "members": members}
        else:
            out[plate.id] = {"status": "unavailable", "reason": "no simulated member to show"}
    return out
