"""PyTensor expression -> LaTeX (the PPL-specific half of the label engine).

Renders a slot value (a distribution parameter) or a ``pm.Deterministic`` right-hand side
to LaTeX, stopping at named model variables (rendered as their symbol). For deterministics
we wrap each leaf in ``\\cssId{tok-<name>}{...}`` so a parent's edge can terminate on the
exact variable inside the equation. Has a node budget + elision so giant graphs degrade
to ``f(...)`` / ``\\ldots`` rather than exploding.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..labels import symbol_for

_UNWRAP_OPS = {"ViewOp"}  # pm.Deterministic wraps its expression in a ViewOp


def _scalar_op_name(op: Any) -> Optional[str]:
    try:
        from pytensor.tensor.elemwise import Elemwise

        if isinstance(op, Elemwise):
            return type(op.scalar_op).__name__
    except Exception:
        pass
    return None


def _is_wrapper(op: Any) -> bool:
    """A value-preserving structural wrapper that a label or glyph can see through: ``ViewOp``
    (``pm.Deterministic``), ``DimShuffle`` (transpose/broadcast), or ``Identity``. NOT ``Cast`` — a
    ``float->int`` cast is a step function (value-changing), so glyph callers must handle it explicitly."""
    if type(op).__name__ in ("ViewOp", "DimShuffle"):
        return True
    return _scalar_op_name(op) == "Identity"


def _fmt_scalar(v: Any) -> str:
    if isinstance(v, float):
        if v != v:
            return r"\mathrm{nan}"
        if v == float("inf"):
            return r"\infty"
        if v == float("-inf"):
            return r"-\infty"
        if v.is_integer():
            return str(int(v))
    return f"{v:.4g}" if isinstance(v, (int, float)) else str(v)  # ~4 sig figs (no -0.666667)


def _const_tex(var: Any) -> Optional[str]:
    data = getattr(var, "data", None)
    if data is None:
        return None
    arr = np.asarray(data)
    if arr.ndim == 0:
        return _fmt_scalar(arr.item())
    flat = arr.ravel()
    if flat.size == 0:
        return r"[\,]"
    # A constant vector (all entries equal) reads as the single value it repeats.
    if np.all(flat == flat[0]):
        return _fmt_scalar(flat[0].item())
    # Otherwise show the actual values — eliding to "[⋯]" wastes the same space while
    # hiding that it's a vector. Show a few entries, then a trailing ellipsis if long.
    if flat.size <= 4:
        return "[" + ",\\,".join(_fmt_scalar(x.item()) for x in flat) + "]"
    head = ",\\,".join(_fmt_scalar(x.item()) for x in flat[:3])
    return "[" + head + r",\,\ldots]"


def render_value(
    var: Any,
    named: dict[int, str],
    *,
    wrap_leaves: bool,
    budget: Optional[list[int]] = None,
    _root: bool = False,
) -> tuple[str, set[str]]:
    """Return ``(latex, used_leaf_names)``. ``named`` maps ``id(var) -> name``."""
    if budget is None:
        budget = [40]
    from pytensor.tensor.elemwise import DimShuffle

    # A named variable (other than the root we were asked to expand) is a leaf.
    if not _root and id(var) in named:
        nm = named[id(var)]
        sym = symbol_for(nm)
        if wrap_leaves:
            return rf"\cssId{{tok-{nm}}}{{{sym}}}", {nm}
        return sym, set()

    owner = getattr(var, "owner", None)
    if owner is None:
        c = _const_tex(var)
        return (c if c is not None else r"\,\cdot\,"), set()

    budget[0] -= 1
    if budget[0] <= 0:
        return r"\ldots", set()

    op = owner.op
    if type(op).__name__ in _UNWRAP_OPS:
        return render_value(owner.inputs[0], named, wrap_leaves=wrap_leaves, budget=budget)
    if isinstance(op, DimShuffle):
        return render_value(owner.inputs[0], named, wrap_leaves=wrap_leaves, budget=budget)

    def R(v):
        return render_value(v, named, wrap_leaves=wrap_leaves, budget=budget)

    used: set[str] = set()
    sname = _scalar_op_name(op)
    if sname:
        parts = []
        for inp in owner.inputs:
            t, u = R(inp)
            parts.append(t)
            used |= u
        if sname == "Add":
            return " + ".join(parts), used
        if sname == "Sub":
            return " - ".join(parts), used
        if sname == "Mul":
            return "\\,".join(parts), used
        if sname in ("TrueDiv", "IntDiv") and len(parts) >= 2:
            return rf"\frac{{{parts[0]}}}{{{parts[1]}}}", used
        if sname == "Pow" and len(parts) >= 2:
            return rf"{parts[0]}^{{{parts[1]}}}", used
        if sname == "Neg":
            return rf"-{parts[0]}", used
        if sname == "Exp":
            return rf"\exp\!\left({parts[0]}\right)", used
        if sname == "Log":
            return rf"\log\!\left({parts[0]}\right)", used
        if sname == "Sqrt":
            return rf"\sqrt{{{parts[0]}}}", used
        if sname == "Reciprocal":  # PyMC exposes Exp/Gamma scale as reciprocal(rate) -> show 1/rate
            return rf"\frac{{1}}{{{parts[0]}}}", used
        if sname == "Second" and len(parts) >= 2:  # second(a, b) broadcasts b to a's shape -> b
            return parts[1], used
        if sname in ("Cast", "Identity") and parts:  # type-cast wrappers are invisible to the math
            return parts[0], used
        return rf"\operatorname{{{sname.lower()}}}\!\left({', '.join(parts)}\right)", used

    if type(op).__name__ == "Softmax":  # non-Elemwise but a recognizable transfer -> name it
        t, u = R(owner.inputs[0])
        return rf"\operatorname{{softmax}}\!\left({t}\right)", u

    # Unknown op: render named-bearing inputs as f(...), else elide.
    sub = []
    for inp in owner.inputs:
        t, u = R(inp)
        if u:
            sub.append(t)
            used |= u
    if sub:
        return rf"f\!\left({', '.join(sub)}\right)", used
    return r"\ldots", set()
