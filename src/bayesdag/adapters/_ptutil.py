"""The ONE place that decides what is numerically knowable about a PyTensor graph.

Every ``.eval()`` in the adapters sits behind ``depends_on_rv``: evaluating a tensor with a
random-variable ancestor silently DRAWS a sample instead of reading a value, which would mislabel
a conditional latent as an analytic root prior and give a different picture on every render. So
the gate fails CLOSED (any doubt reads as "random").

``slot_values`` is the per-slot view the glyph rules build on: one entry per ``dist_params`` slot,
``None`` wherever a parent governs it.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class GlyphNote(str):
    """A caption for a node that is fully representable but carries no curve (or only a partial
    one): "shape depends on alpha (a parent)", "elements differ". Returned in the third slot of a
    provider's ``(spec, data, reason)`` tuple, where a plain ``str`` would be an elision badge —
    the type is what tells the adapter "caption, not warning"."""


def op_is_rv(op: Any, *, unknown: bool = True) -> bool:
    """True if ``op`` is a random-variable op.

    ``unknown`` is what we report when NEITHER base class can be imported, i.e. when the op
    cannot be classified at all. Callers pick the direction that degrades honestly: the
    eval-safety gate passes ``True`` ("assume random", costing a schematic), while component
    *selection* passes ``False`` ("not a component", costing a badge). Never let an
    unclassifiable op read as a known constant — that is what licenses a random `.eval()`."""
    checked = False
    for mod_name, cls_name in (
        ("pytensor.tensor.random.op", "RandomVariable"),
        ("pymc.distributions.distribution", "SymbolicRandomVariable"),
    ):
        try:
            from importlib import import_module

            cls = getattr(import_module(mod_name), cls_name)
        except Exception:
            continue
        checked = True
        if isinstance(op, cls):
            return True
    return unknown if not checked else False


def depends_on_rv(t: Any) -> bool:
    """True if a tensor's value is governed by a random variable (a prior) rather than fixed
    constants — i.e. `.eval()` would return an arbitrary random draw, not the real value.

    **Fails closed.** Any internal failure (a moved pytensor traversal API — it has moved once
    already, hence the import shim below) reports True."""
    try:
        try:
            from pytensor.graph.traversal import ancestors  # pytensor >= 2.x new location
        except Exception:
            from pytensor.graph.basic import ancestors  # older pytensor
    except Exception:
        return True  # can't walk the graph -> assume prior-governed
    try:
        for a in ancestors([t]):
            op = getattr(getattr(a, "owner", None), "op", None)
            if op is not None and op_is_rv(op):
                return True
    except Exception:
        return True
    return False


def ev(t: Any) -> np.ndarray | None:
    """Evaluate a (usually constant) tensor to a float array, or None if it can't be evaluated OR
    is governed by a parent RV."""
    if depends_on_rv(t):
        return None
    try:
        return np.asarray(t.eval(), dtype=float)
    except Exception:
        return None


def dist_name(rv: Any) -> str | None:
    op = getattr(getattr(rv, "owner", None), "op", None)
    if op is None:
        return None
    pn = getattr(op, "_print_name", None)
    return pn[0] if pn else type(op).__name__.removesuffix("RV")


def dist_params(var: Any) -> list:
    node = getattr(var, "owner", None)
    if node is None:
        return []
    try:
        return list(node.op.dist_params(node))
    except Exception:
        return list(node.inputs[2:])  # (rng, size, *params)


def slot_values(var: Any) -> list[np.ndarray | None]:
    """One entry per ``dist_params`` slot: the numeric value, or ``None`` where a parent governs
    the slot (or it cannot be evaluated). Never evaluates a parent-governed slot."""
    out: list[np.ndarray | None] = []
    for dp in dist_params(var):
        if depends_on_rv(dp):
            out.append(None)
            continue
        try:
            out.append(np.asarray(dp.eval()))
        except Exception:
            out.append(None)
    return out


def numeric_params(var: Any) -> list | None:
    """Every slot numerically, or None if any slot is parent-governed (a conditional latent)."""
    vals = slot_values(var)
    return None if any(v is None for v in vals) else vals


def lead_numeric(var: Any, k: int) -> list | None:
    """The first k slots numerically — for constructs that need only a LEADING subset (LKJ's
    ``n, eta``; a walk's drift) whose trailing slots may legitimately be priors."""
    vals = []
    for dp in dist_params(var)[:k]:
        if depends_on_rv(dp):
            return None
        try:
            vals.append(np.asarray(dp.eval()))
        except Exception:
            return None
    return vals


def uniform(values) -> bool:
    """True when every array holds one repeated value, so one scalar per slot describes the whole
    node (an iid vector prior). A genuinely varying vector gives each element its own shape."""
    try:
        for x in values or []:
            a = np.asarray(x).reshape(-1)
            if a.size > 1 and not np.all(a == a.flat[0]):
                return False
    except Exception:
        return False
    return True
