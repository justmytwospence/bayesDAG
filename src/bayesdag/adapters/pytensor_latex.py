"""PyTensor expression -> LaTeX (the PPL-specific half of the label engine).

Renders a slot value (a distribution parameter) or a ``pm.Deterministic`` right-hand side
to LaTeX, stopping at named model variables (rendered as their symbol). For deterministics
we wrap each leaf in ``\\cssId{tok-<name>}{...}`` so a parent's edge can terminate on the
exact variable inside the equation. Has a node budget + elision so giant graphs degrade
to ``f(...)`` / ``\\ldots`` rather than exploding.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..labels import symbol_for

_UNWRAP_OPS = {"ViewOp"}  # pm.Deterministic wraps its expression in a ViewOp


def _scalar_op_name(op: Any) -> str | None:
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


def _const_tex(var: Any) -> str | None:
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


# Operator precedence of a rendered fragment, so a parent op knows when to parenthesize it.
# Higher binds tighter. NEG sits between ADD and MUL: `a\,(-b)` needs parens, `-a + b` does not.
CMP, ADD, NEG, MUL, FRAC, POW, ATOM = 0, 1, 1.5, 2, 2.5, 3, 4


def _paren(t: str) -> str:
    return rf"\left({t}\right)"


def _wrap_if(t: str, prec: float, need: float) -> str:
    return _paren(t) if prec < need else t


def _cast_kind(owner: Any) -> str:
    """'invisible' for a value-preserving (widening / same-kind) cast; 'int' for a float->integer
    cast (truncates toward zero — a step function); 'bool' for a cast to bool (an indicator)."""
    try:
        out_k = np.dtype(owner.outputs[0].dtype).kind
        in_k = np.dtype(owner.inputs[0].dtype).kind
    except Exception:
        return "invisible"
    if out_k == "b" and in_k != "b":
        return "bool"
    if out_k in "iu" and in_k == "f":
        return "int"
    return "invisible"


# CAReduce scalar op -> how the reduction reads. Keyed on the scalar op class (not the Op class)
# so `Sum`, `Prod`, `Max`, `All`, `Any` and any bare `CAReduce` all dispatch the same way.
_REDUCTIONS = {
    "Add": r"\textstyle\sum",
    "Mul": r"\textstyle\prod",
    "Maximum": r"\max",
    "ScalarMaximum": r"\max",
    "Minimum": r"\min",
    "ScalarMinimum": r"\min",
    "AND": r"\textstyle\bigwedge",
    "OR": r"\textstyle\bigvee",
}


def render_value(
    var: Any,
    named: dict[int, str],
    *,
    wrap_leaves: bool,
    budget: list[int] | None = None,
    _root: bool = False,
) -> tuple[str, set[str]]:
    """Return ``(latex, used_leaf_names)``. ``named`` maps ``id(var) -> name``.

    ``used`` always holds EVERY named leaf the expression depends on (the edge set), while only
    the FIRST occurrence of each leaf is token-wrapped: a repeated ``\\cssId{tok-mu}`` would give
    the composed SVG duplicate element ids."""
    state = {"budget": budget if budget is not None else [40], "seen": set()}
    tex, used, _ = _render(var, named, wrap_leaves, state, _root)
    return tex, used


def _render(var, named, wrap, st, root=False) -> tuple[str, set[str], float]:
    from pytensor.tensor.elemwise import DimShuffle

    # A named variable (other than the root we were asked to expand) is a leaf.
    if not root and id(var) in named:
        nm = named[id(var)]
        sym = symbol_for(nm)
        if wrap and nm not in st["seen"]:
            st["seen"].add(nm)
            return rf"\cssId{{tok-{nm}}}{{{sym}}}", {nm}, ATOM
        return sym, {nm}, ATOM

    owner = getattr(var, "owner", None)
    if owner is None:
        c = _const_tex(var)
        if c is None:
            return r"\,\cdot\,", set(), ATOM
        return c, set(), (NEG if c.startswith("-") else ATOM)

    st["budget"][0] -= 1
    if st["budget"][0] <= 0:
        return r"\ldots", set(), ATOM

    op = owner.op

    def R(v):
        return _render(v, named, wrap, st)

    if type(op).__name__ in _UNWRAP_OPS or isinstance(op, DimShuffle):
        return R(owner.inputs[0])

    used: set[str] = set()
    sname = _scalar_op_name(op)
    if sname:
        if sname == "Cast":
            kind = _cast_kind(owner)
            t, u, p = R(owner.inputs[0])
            if kind == "int":
                return rf"\operatorname{{int}}\!\left({t}\right)", u, ATOM
            if kind == "bool":
                return rf"\mathbb{{1}}\!\left[{t}\right]", u, ATOM
            return t, u, p
        parts: list[tuple[str, float]] = []
        for inp in owner.inputs:
            t, u, p = R(inp)
            parts.append((t, p))
            used |= u
        texts = [t for t, _ in parts]
        if sname == "Add":
            out = [
                _paren(t) if (p < ADD or (i > 0 and p == NEG)) else t
                for i, (t, p) in enumerate(parts)
            ]
            return " + ".join(out), used, ADD
        if sname == "Sub" and len(parts) == 2:
            (lt, lp), (rt, rp) = parts
            return f"{_wrap_if(lt, lp, ADD)} - {_wrap_if(rt, rp, MUL)}", used, ADD
        if sname == "Mul":
            return "\\,".join(_wrap_if(t, p, MUL) for t, p in parts), used, MUL
        if sname in ("TrueDiv", "IntDiv") and len(parts) >= 2:
            frac = rf"\frac{{{texts[0]}}}{{{texts[1]}}}"
            if sname == "IntDiv":  # floor division: the floor is part of the value
                return rf"\left\lfloor {frac} \right\rfloor", used, ATOM
            return frac, used, FRAC
        if sname == "Pow" and len(parts) >= 2:
            (bt, bp), (et, _) = parts[0], parts[1]
            return rf"{_wrap_if(bt, bp, ATOM)}^{{{et}}}", used, POW
        if sname == "Neg":
            t, p = parts[0]
            return rf"-{_wrap_if(t, p, MUL)}", used, NEG
        if sname == "Exp":
            return rf"\exp\!\left({texts[0]}\right)", used, ATOM
        if sname == "Log":
            return rf"\log\!\left({texts[0]}\right)", used, ATOM
        if sname == "Sqrt":
            return rf"\sqrt{{{texts[0]}}}", used, ATOM
        if sname == "Reciprocal":  # PyMC exposes Exp/Gamma scale as reciprocal(rate) -> show 1/rate
            if texts[0] == "1":  # fold ONLY the trivial 1/1; 1/0.1 stays a fraction (it
                return "1", used, ATOM  # preserves the user's rate parameterization — honesty)
            return rf"\frac{{1}}{{{texts[0]}}}", used, FRAC
        if sname == "Second" and len(parts) >= 2:  # second(a, b) broadcasts b to a's shape -> b
            return parts[1][0], used, parts[1][1]
        if sname == "Identity" and parts:
            return parts[0][0], used, parts[0][1]
        _CMP = {
            "LE": r"\le",
            "GE": r"\ge",
            "LT": "<",
            "GT": ">",
            "EQ": "=",
            "AND": r"\land",
            "OR": r"\lor",
        }
        if sname in _CMP and len(parts) >= 2:  # a comparison/boolean reads infix
            return rf"{texts[0]} {_CMP[sname]} {texts[1]}", used, CMP
        if sname == "Switch" and len(parts) >= 3:  # where(cond, a, b) -> a conditional
            return rf"\left({texts[0]} \,?\, {texts[1]} : {texts[2]}\right)", used, ATOM
        return (
            rf"\operatorname{{{sname.lower()}}}\!\left({', '.join(texts)}\right)",
            used,
            ATOM,
        )

    if type(op).__name__ == "Softmax":  # non-Elemwise but a recognizable transfer -> name it
        t, u, _ = R(owner.inputs[0])
        return rf"\operatorname{{softmax}}\!\left({t}\right)", u, ATOM

    red = _reduction_symbol(op)
    if red is not None:  # a reduction names its actual operation, never a blanket sum
        t, u, p = R(owner.inputs[0])
        return f"{red} {_wrap_if(t, p, ATOM)}", u, ADD

    # Unknown op: render named-bearing inputs as f(...), else elide.
    sub = []
    for inp in owner.inputs:
        t, u, _ = R(inp)
        if u:
            sub.append(t)
            used |= u
    if sub:
        return rf"f\!\left({', '.join(sub)}\right)", used, ATOM
    return r"\ldots", set(), ATOM


def _reduction_symbol(op: Any) -> str | None:
    try:
        from pytensor.tensor.elemwise import CAReduce
    except Exception:  # pragma: no cover
        return None
    if not isinstance(op, CAReduce):
        return None
    name = type(getattr(op, "scalar_op", None)).__name__
    return _REDUCTIONS.get(name, r"\operatorname{reduce}")
