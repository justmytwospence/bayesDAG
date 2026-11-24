"""Recognized model structures ("blocks"), read off the PyTensor graph (PyMC-specific).

Every rule here is a STRUCTURAL whitelist over op and slot structure — never variable names,
never rendered TeX, never sample values — and every rule states its proof obligations. A block
is claimed only when the structure is a mathematical consequence of the ops (false negatives are
fine; a structure the model does not have must never be claimed). Nothing is ``.eval()``-ed
unless ``depends_on_rv`` has cleared it. Each detector is guarded: a failure means "no block",
never a broken ``to_ir``.

Rules (``BlockIR.rule``):

* ``noncentered/1`` — a named Deterministic ``D = L + S*z`` (optionally ``exp(.)`` /
  ``invlogit(.)`` of it, or ``A @ z``) with ``z`` a root standard Normal (or Student-t) that
  nothing else consumes. Since ``z`` is symmetric, ``D | L, S ~ Family(L, |S|)`` exactly.
* ``shrinkage/1`` — a coefficient whose scale factors into one global and one local positive
  scale (optionally with a co-data ``exp(.)`` modifier and the Piironen–Vehtari slab
  ``c^2 lambda^2 / (c^2 + tau^2 lambda^2)``).
* ``regression/1`` — an observed likelihood whose location, through at most one recognized link,
  is affine in its leaves with at least one data-times-parameter term.
* ``measured/1`` — a latent true value observed with noise AND used elsewhere.
* ``centered/1`` (IR level, see ``bayesdag.blocks``) — a plated latent with a coarser latent
  parent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..ir import BlockIR, GlyphSpec, ModelIR
from ._ptutil import depends_on_rv, dist_name, slot_values, uniform
from .pytensor_latex import _scalar_op_name, render_value

_BUDGET = 2048  # op visits per candidate: exceeding it means "no match"
_POSITIVE = frozenset(
    {
        "HalfNormal",
        "HalfCauchy",
        "HalfStudentT",
        "Exponential",
        "Gamma",
        "InverseGamma",
        "LogNormal",
        "Weibull",
        "HalfFlat",
        "ChiSquared",
    }
)
_DISCRETE_FAMILIES = frozenset(
    {
        "Bernoulli",
        "Binomial",
        "BetaBinomial",
        "Poisson",
        "NegativeBinomial",
        "Categorical",
        "DiscreteUniform",
        "DiscreteWeibull",
        "Geometric",
        "HyperGeometric",
    }
)
REFS_NONCENTERED = [
    "Papaspiliopoulos, Roberts & Sköld (2007)",
    "Betancourt & Girolami (2015)",
]
REFS_SHRINKAGE = ["Carvalho, Polson & Scott (2010)", "Piironen & Vehtari (2017)"]
REFS_REGRESSION = ["Gelman & Hill (2007)"]


class _Budget(Exception):
    pass


@dataclass
class _Ctx:
    model: Any
    named: dict[int, str]
    ir: ModelIR
    consumers: dict[str, set[str]]
    visits: list[int] = field(default_factory=lambda: [0])

    def tick(self):
        self.visits[0] += 1
        if self.visits[0] > _BUDGET:
            raise _Budget

    def node(self, name):
        return self.ir.node(name)

    def role(self, name) -> str | None:
        n = self.node(name)
        return n.role if n is not None else None

    def dims(self, name) -> tuple:
        n = self.node(name)
        return tuple(n.dims) if n is not None else ()

    def var(self, name):
        return self.model[name]

    def slot_values(self, var):
        return slot_values(var)

    def fixed_value(self, expr):
        if not depends_on_rv(expr):
            try:
                return np.asarray(expr.eval())
            except Exception:
                pass
        return None


def _opname(v) -> str:
    o = getattr(v, "owner", None)
    return type(o.op).__name__ if o is not None else ""


def _is_dot(v) -> bool:
    o = getattr(v, "owner", None)
    if o is None:
        return False
    n = type(o.op).__name__
    if n in ("Dot", "Dot22", "BatchedDot"):
        return True
    return n == "Blockwise" and type(getattr(o.op, "core_op", None)).__name__ in ("Dot", "Dot22")


def _unwrap(v, ctx: _Ctx, root_id: int | None = None, *, preserve_axes: bool = False):
    """Strip value-preserving wrappers (ViewOp / DimShuffle / Identity / widening Cast), stopping
    at any named variable other than the root."""
    for _ in range(256):
        if id(v) in ctx.named and id(v) != root_id:
            return v
        o = getattr(v, "owner", None)
        if o is None:
            return v
        n = type(o.op).__name__
        sn = _scalar_op_name(o.op)
        if n in ("ViewOp", "DimShuffle", "SpecifyShape") or sn == "Identity":
            if preserve_axes and n == "DimShuffle" and o.inputs[0].ndim:
                order = tuple(o.op.new_order)
                axes = tuple(i for i in order if i != "x")
                if not order or order[0] != 0 or axes != tuple(range(len(axes))):
                    return v
            v = o.inputs[0]
            continue
        if sn == "Cast":
            try:
                if np.dtype(o.outputs[0].dtype).kind == "f":
                    v = o.inputs[0]
                    continue
            except Exception:
                pass
        return v
    return v


def _named_leaves(t, ctx: _Ctx, *, skip_indices: bool = False) -> list[str]:
    """Named variables directly feeding ``t`` (descent stops at named boundaries). With
    ``skip_indices`` the index inputs of a gather are not followed (an index is not a value)."""
    if id(t) in ctx.named:
        return [ctx.named[id(t)]]
    out, seen, stack = [], set(), [t]
    while stack:
        ctx.tick()
        v = stack.pop()
        if id(v) in seen:
            continue
        seen.add(id(v))
        o = getattr(v, "owner", None)
        if o is None:
            continue
        ins = list(o.inputs)
        if skip_indices and type(o.op).__name__ in (
            "AdvancedSubtensor",
            "AdvancedSubtensor1",
            "Subtensor",
        ):
            ins = ins[:1]
        for i in ins:
            if id(i) in ctx.named:
                nm = ctx.named[id(i)]
                if nm not in out:
                    out.append(nm)
            else:
                stack.append(i)
    return out


def _depends_on(v, z) -> bool:
    try:
        from pytensor.graph.traversal import ancestors
    except Exception:  # pragma: no cover
        return True
    return any(a is z for a in ancestors([v]))


# --------------------------------------------------------------------------- affine decomposition
@dataclass
class Affine:
    """``value = sum(sign_i * loc_i) + sign * prod(scale factors) * (matrix @) z``."""

    loc: list[tuple[int, Any]] = field(default_factory=list)  # (sign, z-free tensor)
    scale: list[tuple[float, Any]] = field(default_factory=list)  # (exponent, z-free tensor)
    sign: int = 1
    matrix: Any = None


def affine_decompose(expr, z, ctx: _Ctx, root_id: int | None = None) -> Affine | None:
    """Decompose ``expr`` as an affine function of the tensor ``z`` over the whitelist: wrappers,
    Add/Sub (one z-dependent term), Neg, Mul (one z-dependent factor, applied before any
    location term), TrueDiv (z in the numerator only), and a left matrix product ``A @ z``.
    Gathers of z are rejected (they would duplicate elements). None = not provably affine."""

    def dec(v) -> Affine | None:
        ctx.tick()
        v = _unwrap(v, ctx, root_id)
        if v is z:
            return Affine()
        if id(v) in ctx.named and id(v) != root_id:
            return None  # a named intermediate between z and D: z would have another consumer
        o = getattr(v, "owner", None)
        if o is None:
            return None
        sn = _scalar_op_name(o.op)
        ins = list(o.inputs)
        dep = [i for i in ins if _depends_on(i, z)]
        if len(dep) != 1:
            return None
        zi = dep[0]
        free = [i for i in ins if i is not zi]
        if sn == "Add":
            r = dec(zi)
            if r is None:
                return None
            r.loc += [(1, f) for f in free]
            return r
        if sn == "Sub" and len(ins) == 2:
            r = dec(zi)
            if r is None:
                return None
            if zi is ins[0]:
                r.loc.append((-1, ins[1]))
            else:  # a - (L + S z) = (a - L) - S z
                r.loc = [(1, ins[0])] + [(-s, t) for s, t in r.loc]
                r.sign = -r.sign
            return r
        if sn == "Neg":
            r = dec(zi)
            if r is None:
                return None
            r.loc = [(-s, t) for s, t in r.loc]
            r.sign = -r.sign
            return r
        if sn == "Mul":
            r = dec(zi)
            if r is None or r.loc:  # (L + S z) * c would scale L too: not handled
                return None
            r.scale += [(1.0, f) for f in free]
            return r
        if sn == "TrueDiv" and zi is ins[0]:
            r = dec(zi)
            if r is None or r.loc:
                return None
            r.scale.append((-1.0, ins[1]))
            return r
        if _is_dot(v) and zi is ins[1]:
            inner = _unwrap(ins[1], ctx, root_id)
            if inner is not z:
                return None
            return Affine(matrix=ins[0])
        return None

    try:
        return dec(expr)
    except _Budget:
        return None


# --------------------------------------------------------------------------- helpers
def std_aux(name: str, ctx: _Ctx) -> dict | None:
    """``{family, scale, nu}`` when ``name`` is a root, zero-location, fixed-scale symmetric
    variable (Normal or Student-t): the ``z`` of a non-centered parameterization."""
    if ctx.role(name) != "latent":
        return None
    var = ctx.var(name)
    fam = dist_name(var)
    if fam not in ("Normal", "StudentT"):
        return None
    vals = ctx.slot_values(var)
    if any(v is None for v in vals) or not uniform(vals):
        return None
    q = [float(np.asarray(v).reshape(-1)[0]) for v in vals]
    if fam == "Normal" and len(q) == 2 and q[0] == 0.0 and q[1] > 0:
        return {"family": "Normal", "scale": q[1], "nu": None}
    if fam == "StudentT" and len(q) == 3 and q[1] == 0.0 and q[2] > 0:
        return {"family": "StudentT", "scale": q[2], "nu": q[0]}
    return None


def _tex(t, ctx: _Ctx, wrap: bool = False, seen: set | None = None) -> str:
    try:
        return render_value(t, ctx.named, wrap_leaves=wrap, seen=seen)[0]
    except Exception:
        return r"\ldots"


def _is_positive_latent(name: str, ctx: _Ctx) -> bool:
    n = ctx.node(name)
    return n is not None and n.role == "latent" and (n.dist in _POSITIVE)


def _coarser(a: tuple, b: tuple) -> bool:
    """``a`` indexes a strict subset of ``b``'s dims (a scalar is coarser than any plated var)."""
    return set(a) < set(b) or (not a and bool(b))


# --------------------------------------------------------------------------- non-centered (H2)
@dataclass
class _NC:
    anchor: str
    aux: str
    aux_info: dict
    family: str
    aff: Affine
    loc_names: list[str]
    scale_names: list[str]


def _noncentered(name: str, ctx: _Ctx) -> _NC | None:
    var = ctx.var(name)
    root = var
    core = _unwrap(var.owner.inputs[0], ctx) if _opname(var) == "ViewOp" else var
    family = "Normal"
    sn = _scalar_op_name(getattr(getattr(core, "owner", None), "op", None))
    if sn == "Exp":
        family, core = "LogNormal", core.owner.inputs[0]
    elif sn == "Sigmoid":
        family, core = "LogitNormal", core.owner.inputs[0]
    leaves = _named_leaves(core, ctx)
    # a standard-Normal leaf that only D consumes, with D's dims; mu ~ N(0, 5) can look like one,
    # so every candidate is tried and exactly one must decompose with a learned scale
    matches = []
    for zname in leaves:
        if ctx.consumers.get(zname, set()) != {name}:
            continue  # z must feed D and nothing else
        if ctx.dims(name) and ctx.dims(zname) != ctx.dims(name):
            continue
        aux_info = std_aux(zname, ctx)
        if aux_info is None:
            continue
        aff = affine_decompose(core, ctx.var(zname), ctx, root_id=id(root))
        if aff is None or (not any(depends_on_rv(t) for _, t in aff.scale) and aff.matrix is None):
            continue
        matches.append((zname, aff, aux_info))
    if len(matches) != 1:
        return None
    zname, aff, aux_info = matches[0]
    if aff.matrix is not None:
        if family != "Normal" or aux_info["family"] != "Normal":
            return None
        family = "MultivariateNormal"
    elif aux_info["family"] == "StudentT":
        if family != "Normal":
            return None
        family = "StudentT"
    scale_terms = [t for _, t in aff.scale] + ([aff.matrix] if aff.matrix is not None else [])
    scale_names: list[str] = []
    for t in scale_terms:
        for nm in _named_leaves(t, ctx, skip_indices=True):
            if ctx.role(nm) == "data":
                return None  # a data-scaled "effect" is a regression term, not a hierarchy
            n = ctx.node(nm)
            if n is not None and n.role == "latent" and n.dist in _DISCRETE_FAMILIES:
                return None  # an inclusion indicator times a slab is a mixture, not a spread
            if nm not in scale_names:
                scale_names.append(nm)
    if not any(depends_on_rv(t) for t in scale_terms):
        return None  # a fixed scale: `5*z` is a reparameterized prior, not a learned spread
    loc_names: list[str] = []
    for _, t in aff.loc:
        for nm in _named_leaves(t, ctx):
            if nm not in loc_names:
                loc_names.append(nm)
    return _NC(name, zname, aux_info, family, aff, loc_names, scale_names)


def _implied_tex(nc: _NC, ctx: _Ctx, wrap: bool) -> tuple[str, str, str]:
    """(loc TeX, scale TeX, the full implied line) for a non-centered match."""
    from ..labels import dist_symbol, symbol_for

    seen: set = set()  # one token per leaf across the whole implied line
    loc_parts = []
    for i, (s, t) in enumerate(nc.aff.loc):
        tx = _tex(t, ctx, wrap, seen)
        if s < 0:
            loc_parts.append(rf"- \left({tx}\right)" if "+" in tx or "-" in tx else f"- {tx}")
        else:
            loc_parts.append(tx if i == 0 else f"+ {tx}")
    loc = " ".join(loc_parts).lstrip("+ ") or "0"
    num, den = [], []
    s0 = nc.aux_info["scale"]
    if s0 != 1.0:
        num.append(f"{s0:.4g}")
    for e, t in nc.aff.scale:
        tx = _tex(t, ctx, wrap, seen)
        if "+" in tx or " - " in tx:
            tx = rf"\left({tx}\right)"
        (num if e > 0 else den).append(tx)
    scale = r"\,".join(num) or "1"
    if den:
        scale = rf"\frac{{{scale}}}{{{r'\,'.join(den)}}}"
    if nc.aff.sign < 0 or any(
        (getattr(t, "owner", None) is None and _const_negative(t)) for _, t in nc.aff.scale
    ):
        scale = rf"\left|{scale}\right|"
    lhs = symbol_for(nc.anchor)
    if nc.family == "MultivariateNormal":
        mat = _tex(nc.aff.matrix, ctx, wrap, seen)
        cov = rf"{mat}\,{mat}^{{\top}}"
        if s0 != 1.0:
            cov = rf"{s0**2:.4g}\,{cov}"
        line = rf"{lhs} \sim \mathcal{{N}}\!\left({loc},\ {cov}\right)"
        return loc, cov, line
    if nc.family == "StudentT":
        nu = nc.aux_info["nu"]
        line = rf"{lhs} \sim \mathrm{{StudentT}}_{{{nu:.4g}}}\!\left({loc},\ {scale}\right)"
        return loc, scale, line
    line = rf"{lhs} \sim {dist_symbol(nc.family)}\!\left({loc},\ {scale}\right)"
    return loc, scale, line


def _const_negative(t) -> bool:
    d = getattr(t, "data", None)
    try:
        return d is not None and bool(np.any(np.asarray(d) < 0))
    except Exception:
        return False


def _implied_glyph(nc: _NC):
    """The exact standardized shape of the implied family, when its shape is fixed: Normal and
    Student-t (location/scale only); LogNormal/LogitNormal only if the scale is a number."""
    from .glyph_data import _conditional_shape_from_values

    if nc.family == "Normal":
        return _conditional_shape_from_values("Normal", [None, None])
    if nc.family == "StudentT":
        return _conditional_shape_from_values(
            "StudentT", [np.asarray(nc.aux_info["nu"]), None, None]
        )
    return None


# --------------------------------------------------------------------------- shrinkage (S1)
def _flatten_factors(t, ctx: _Ctx, e: float, out: list, depth: int = 0) -> bool:
    """Multiplicative factors of ``t`` as ``(exponent, key, tensor)``; key = a named variable,
    ``("exp", id)`` for an exp(.) factor, ``("add", id)`` for a sum, or ``("const", value)``."""
    ctx.tick()
    if depth > 64:
        return False
    t = _unwrap(t, ctx)
    if id(t) in ctx.named:
        out.append((e, ctx.named[id(t)], t))
        return True
    o = getattr(t, "owner", None)
    if o is None:
        d = getattr(t, "data", None)
        if d is None:
            return False
        a = np.asarray(d, float)
        if a.size and np.all(a == a.reshape(-1)[0]):
            out.append((e, ("const", float(a.reshape(-1)[0])), t))
            return True
        return False
    sn = _scalar_op_name(o.op)
    ins = o.inputs
    if sn == "Mul":
        return all(_flatten_factors(i, ctx, e, out, depth + 1) for i in ins)
    if sn == "TrueDiv":
        return _flatten_factors(ins[0], ctx, e, out, depth + 1) and _flatten_factors(
            ins[1], ctx, -e, out, depth + 1
        )
    if sn == "Sqrt":
        return _flatten_factors(ins[0], ctx, e * 0.5, out, depth + 1)
    if sn == "Sqr":
        return _flatten_factors(ins[0], ctx, e * 2.0, out, depth + 1)
    if sn == "Pow":
        k = _const_scalar(ins[1])
        if k is None:
            return False
        return _flatten_factors(ins[0], ctx, e * k, out, depth + 1)
    if sn == "Exp":
        out.append((e, ("exp", id(t)), t))
        return True
    if sn == "Add":
        out.append((e, ("add", id(t)), t))
        return True
    return False


def _const_scalar(t) -> float | None:
    t2 = t
    for _ in range(8):
        o = getattr(t2, "owner", None)
        if o is None:
            break
        if type(o.op).__name__ in ("DimShuffle",) or _scalar_op_name(o.op) in ("Cast", "Identity"):
            t2 = o.inputs[0]
            continue
        break
    d = getattr(t2, "data", None)
    if d is None:
        return None
    a = np.asarray(d, float)
    if a.size and np.all(a == a.reshape(-1)[0]):
        return float(a.reshape(-1)[0])
    return None


def _combine(factors) -> tuple[dict, dict]:
    """(exponents by key, a representative tensor by key)."""
    exps: dict = {}
    tens: dict = {}
    for e, k, t in factors:
        exps[k] = exps.get(k, 0.0) + e
        tens[k] = t
    return exps, tens


@dataclass
class _Shrink:
    coef: str
    aux: str | None
    glob: str
    local: str
    slab: str | None
    slab_power: float | None
    modifiers: list[str]
    center: list[str]
    centered: bool


def _shrinkage(name: str, ctx: _Ctx, nc: _NC | None) -> _Shrink | None:
    """Proof obligations: the coefficient's conditional scale is EXACTLY
    ``g * l [* m] [* sqrt(c^k / (c^k + g^2 l^2 [m^2]))]`` up to positive constants, with ``g`` a
    positive latent coarser than the coefficient, ``l`` a positive latent with its dims, and every
    modifier ``m = exp(.)`` of a random expression. Anything else in the scale: no match."""
    node = ctx.node(name)
    if node is None or not node.dims:
        return None
    if nc is not None:
        if nc.family != "Normal":
            return None
        scale_terms = list(nc.aff.scale)
        center_t = [t for _, t in nc.aff.loc]
        centered = False
        aux = nc.aux
    else:
        if node.role != "latent" or node.dist != "Normal":
            return None
        var = ctx.var(name)
        from ._ptutil import dist_params

        dp = dist_params(var)
        if len(dp) != 2:
            return None
        scale_terms = [(1.0, dp[1])]
        center_t = [dp[0]] if depends_on_rv(dp[0]) or _const_scalar(dp[0]) not in (0.0,) else []
        centered, aux = True, None
    factors: list = []
    for e, t in scale_terms:
        if not _flatten_factors(t, ctx, e, factors):
            return None
    exps, tens = _combine(factors)
    named = {k: v for k, v in exps.items() if isinstance(k, str)}
    exp_mods = {k: v for k, v in exps.items() if isinstance(k, tuple) and k[0] == "exp"}
    adds = {k: v for k, v in exps.items() if isinstance(k, tuple) and k[0] == "add"}
    bdims = tuple(node.dims)
    globs = [
        k
        for k, v in named.items()
        if abs(v - 1.0) < 1e-9 and _is_positive_latent(k, ctx) and _coarser(ctx.dims(k), bdims)
    ]
    locs = [
        k
        for k, v in named.items()
        if abs(v - 1.0) < 1e-9 and _is_positive_latent(k, ctx) and ctx.dims(k) == bdims
    ]
    if len(globs) != 1 or len(locs) != 1:
        return None
    g, lam = globs[0], locs[0]
    if any(abs(v - 1.0) > 1e-9 for v in exp_mods.values()):
        return None
    modifiers: list[str] = []
    for k in exp_mods:
        t = tens[k]
        if not depends_on_rv(t):
            return None
        modifiers += [n for n in _named_leaves(t.owner.inputs[0], ctx) if ctx.role(n) != "data"]
    rest = {k: v for k, v in named.items() if k not in (g, lam)}
    slab, slab_power = None, None
    if adds or rest:
        # the regularized slab: one sum raised to -1/2 and one positive slab variable c^(k/2)
        if len(adds) != 1 or len(rest) != 1:
            return None
        (akey, aexp), (c, cexp) = next(iter(adds.items())), next(iter(rest.items()))
        if abs(aexp + 0.5) > 1e-9 or not _is_positive_latent(c, ctx):
            return None
        if not _coarser(ctx.dims(c), bdims) and ctx.dims(c) != ():
            return None
        k = 2 * cexp
        if not _slab_denominator(tens[akey], ctx, c, k, g, lam, set(exp_mods)):
            return None
        slab, slab_power = c, k
    center: list[str] = []
    for t in center_t:
        center += [n for n in _named_leaves(t, ctx) if n not in center]
    return _Shrink(name, aux, g, lam, slab, slab_power, modifiers, center, centered)


def _slab_denominator(add, ctx, c, k, g, lam, mods) -> bool:
    """``add`` is exactly ``c^k + g^2 * lam^2 [* m^2 ...]`` (either order)."""
    o = add.owner
    if o is None or len(o.inputs) != 2:
        return False
    terms = []
    for t in o.inputs:
        fs: list = []
        if not _flatten_factors(t, ctx, 1.0, fs):
            return False
        exps, _ = _combine(fs)
        exps = {kk: v for kk, v in exps.items() if not (isinstance(kk, tuple) and kk[0] == "const")}
        terms.append(exps)
    want_c = {c: k}
    want_g = {g: 2.0, lam: 2.0}
    for t in terms:
        mod_keys = {kk for kk in t if isinstance(kk, tuple) and kk[0] == "exp"}
        base = {kk: v for kk, v in t.items() if kk not in mod_keys}
        if base == want_c and not mod_keys:
            continue
        if (
            base == want_g
            and len(mod_keys) == len(mods)
            and all(abs(t[kk] - 2.0) < 1e-9 for kk in mod_keys)
        ):
            continue
        return False
    return True


def _shrink_implied(sh: _Shrink) -> str:
    """The conventional reading: ``beta ~ N(center, tau * lambda_tilde)`` and, with a slab,
    ``lambda_tilde^2 = c lambda^2 / (c + tau^2 lambda^2)`` — symbols token-wrapped once so
    aggregated edges land on them."""
    from ..labels import symbol_for

    seen: set[str] = set()

    def sym(nm: str) -> str:
        s = symbol_for(nm)
        if nm in seen:
            return s
        seen.add(nm)
        return rf"\cssId{{tok-{nm}}}{{{s}}}"

    center = ",\\,".join(sym(c) for c in sh.center) if sh.center else "0"
    g = sym(sh.glob)

    def lam_eff() -> str:  # the (possibly co-data modulated) local scale; tokens on first use
        mods = "".join(rf"\,e^{{\mathbf{{C}}\,{sym(m)}}}" for m in sh.modifiers)
        lam = sym(sh.local)
        return rf"\left({lam}{mods}\right)" if mods else lam

    head = rf"{symbol_for(sh.coef)} \sim \mathcal{{N}}\!\left({center},\ {g}\,"
    if sh.slab is None:
        return head + rf"{lam_eff()}\right)"

    def ck() -> str:
        c = sym(sh.slab)
        return c if sh.slab_power == 1.0 else rf"{c}^{{{sh.slab_power:g}}}"

    lt = r"\tilde{\lambda}"
    num = rf"{ck()}\,{lam_eff()}^{{2}}"
    den = rf"{ck()} + {sym(sh.glob)}^{{2}}\,{lam_eff()}^{{2}}"
    return head + rf"{lt}\right),\quad {lt}^{{2}} = \frac{{{num}}}{{{den}}}"


def _local_family(lam: str, ctx: _Ctx) -> tuple[str, float | None]:
    n = ctx.node(lam)
    fam = n.dist if n is not None else None
    nu = None
    if fam == "HalfStudentT":
        vals = ctx.slot_values(ctx.var(lam))
        if vals and vals[0] is not None and uniform([vals[0]]):
            nu = float(np.asarray(vals[0]).reshape(-1)[0])
    return fam, nu


def _shrink_title(sh: _Shrink, ctx: _Ctx) -> tuple[str, str]:
    fam, nu = _local_family(sh.local, ctx)
    reg = sh.slab is not None
    if fam == "HalfCauchy" or (fam == "HalfStudentT" and nu == 1.0):
        base = "horseshoe"
    elif fam == "HalfStudentT" and nu is not None:
        base = f"horseshoe · half-t({nu:g}) locals"
    else:
        base = "global–local shrinkage"
    return ("regularized " + base if reg else base), ("regularized" if reg else "plain")


_HS_GRID: dict = {}


def _horseshoe_marginal(nu: float) -> dict:
    """The exact standardized marginal ``p(beta | tau = 1)`` with ``lambda ~ half-t_nu(0, 1)``
    (nu = 1: half-Cauchy — the horseshoe), integrated numerically once per nu. The density has a
    pole at 0 (for nu <= 1), so the grid skips 0 and the curve is marked ``clipped``."""
    if nu in _HS_GRID:
        return _HS_GRID[nu]
    import scipy.integrate as si
    import scipy.stats as st

    xs = np.linspace(-4.0, 4.0, 64)  # an even count: 0 is not on the grid
    halft = st.t(df=nu)

    def dens(b):
        f = lambda lam: st.norm.pdf(b, scale=lam) * 2.0 * halft.pdf(lam)  # noqa: E731
        return si.quad(f, 0.0, np.inf, limit=200)[0]

    ys = np.array([dens(b) for b in xs])
    m = float(ys.max()) or 1.0
    out = {
        "xs": [float((x + 4.0) / 8.0) for x in xs],
        "ys": [float(y / m) for y in ys],
        "family": "horseshoe" if nu == 1.0 else f"half-t({nu:g}) global-local",
        "exact": True,
        "clipped": True,
    }
    _HS_GRID[nu] = out
    return out


# --------------------------------------------------------------------------- regression (R1)
_REG_FAMILIES = {
    # family -> (location slot index, {transfer: title})
    "Normal": (0, {"identity": "linear regression"}),
    "StudentT": (1, {"identity": "robust (Student-t) regression"}),
    "LogNormal": (0, {"identity": "log-normal regression"}),
    "Bernoulli": (0, {"Sigmoid": "logistic regression", "probit": "probit regression"}),
    "Binomial": (1, {"Sigmoid": "binomial logistic regression", "probit": "probit regression"}),
    "Poisson": (0, {"Exp": "Poisson regression (log link)"}),
    "Categorical": (0, {"Softmax": "multinomial-logit regression"}),
}


def _link(t, ctx: _Ctx):
    """(transfer name, inner expression) — identity when no recognized link wraps ``t``."""
    t = _unwrap(t, ctx)
    o = getattr(t, "owner", None)
    if o is None:
        return "identity", t
    sn = _scalar_op_name(o.op)
    if sn in ("Sigmoid", "Exp"):
        return sn, o.inputs[0]
    if type(o.op).__name__ == "Softmax":
        return "Softmax", o.inputs[0]
    # 0.5 * (1 + erf(x / sqrt(2))) — the probit link as PyMC's invprobit builds it
    if sn == "Mul":
        consts = [i for i in o.inputs if _const_scalar(i) is not None]
        rest = [i for i in o.inputs if _const_scalar(i) is None]
        if len(rest) == 1 and len(consts) == 1 and abs(_const_scalar(consts[0]) - 0.5) < 1e-12:
            a = _unwrap(rest[0], ctx)
            ao = getattr(a, "owner", None)
            if ao is not None and _scalar_op_name(ao.op) == "Add" and len(ao.inputs) == 2:
                one = [i for i in ao.inputs if _const_scalar(i) == 1.0]
                other = [i for i in ao.inputs if _const_scalar(i) != 1.0]
                if one and other:
                    e = _unwrap(other[0], ctx)
                    eo = getattr(e, "owner", None)
                    if eo is not None and _scalar_op_name(eo.op) in ("Erf",):
                        return "probit", eo.inputs[0]
    return "identity", t


@dataclass
class Term:
    """One additive term of a linear predictor.

    ``kind``: ``"leaf"`` (a parameter), ``"const"`` (data/constants only), ``"product"`` (data x
    parameter, incl. a matrix product), ``"gather"`` (a parameter indexed by data). ``params`` are
    the named parameters it is linear in, ``index`` the named data used to gather them, ``data``
    the named data multiplying them, and ``design`` the (RV-free) multiplier tensor itself."""

    kind: str
    params: list = field(default_factory=list)
    index: list = field(default_factory=list)
    data: list = field(default_factory=list)
    design: Any = None
    multiplier: float | None = 1.0
    # Inputs proved to supply one likelihood row per first-axis entry, not just named leaves.
    row_data: list[str] = field(default_factory=list)
    row_params: list[str] = field(default_factory=list)
    row_design: bool = False


def _direct_data(t, ctx: _Ctx) -> list[str]:
    v = _unwrap(t, ctx, preserve_axes=True)
    n = ctx.named.get(id(v))
    return [n] if n is not None and ctx.role(n) == "data" and v.ndim else []


def _affine_terms(t, ctx: _Ctx, blocked: set[str], hops: int = 0) -> list | None:
    """Additive :class:`Term` s of a linear predictor. Named deterministics are expanded (unless
    they anchor another block) up to 6 hops. None = not affine."""
    expr = _unwrap(t, ctx)
    terms = _terms(expr, ctx, blocked, hops)
    if terms and _unwrap(t, ctx, preserve_axes=True) is not expr:
        for term in terms:
            term.row_data = []
            term.row_params = []
            term.row_design = False
    return terms


def _terms(t, ctx: _Ctx, blocked: set[str], hops: int) -> list | None:
    ctx.tick()
    if id(t) in ctx.named:
        nm = ctx.named[id(t)]
        role = ctx.role(nm)
        if role == "data":
            return [Term("const", data=[nm], row_data=[nm] if t.ndim == 1 else [])]
        if role == "deterministic" and nm not in blocked and hops < 6:
            v = ctx.var(nm)
            inner = v.owner.inputs[0] if _opname(v) == "ViewOp" else v
            expanded = _affine_terms(inner, ctx, blocked, hops + 1)
            if expanded is not None:
                return expanded
            # a nonlinear named quantity (a spike-and-slab coefficient, gamma * slab) is still
            # one VALUE the predictor is linear in: keep it as a leaf
        return [Term("leaf", params=[nm], row_params=[nm] if t.ndim == 1 else [])]
    if not depends_on_rv(t):
        return [Term("const", data=[n for n in _named_leaves(t, ctx) if ctx.role(n) == "data"])]
    o = getattr(t, "owner", None)
    if o is None:
        return [Term("const")]
    sn = _scalar_op_name(o.op)
    ins = o.inputs
    if sn in ("Add", "Sub"):
        out = []
        for j, i in enumerate(ins):
            r = _affine_terms(i, ctx, blocked, hops)
            if r is None:
                return None
            if sn == "Sub" and j > 0:
                for term in r:
                    if term.multiplier is not None:
                        term.multiplier *= -1
            out += r
        return out
    if sn == "Neg":
        result = _affine_terms(ins[0], ctx, blocked, hops)
        if result is not None:
            for term in result:
                if term.multiplier is not None:
                    term.multiplier *= -1
        return result
    if sn == "Mul" or _is_dot(t):
        rv = [i for i in ins if depends_on_rv(i)]
        if len(rv) != 1:
            return None  # parameter x parameter: not linear
        sub = _affine_terms(rv[0], ctx, blocked, hops)
        if sub is None:
            return None
        other = [i for i in ins if i is not rv[0]]
        data_like = any(_varying_data(i, ctx) for i in other)
        params = sorted({n for x in sub for n in x.params})
        index = sorted({n for x in sub for n in x.index})
        fixed_data = {n for i in other for n in _named_leaves(i, ctx) if ctx.role(n) == "data"}
        data = sorted(fixed_data | {n for x in sub for n in x.data})
        proof = sub[0].multiplier if len(sub) == 1 and len(params) == 1 else None
        scalar = _literal_multiplier(other[0]) if len(other) == 1 else None
        if scalar is not None:
            term = sub[0] if len(sub) == 1 else None
            if term is not None:
                term.multiplier = proof * scalar if proof is not None else None
                return [term]
        design = other[0] if len(other) == 1 else None
        row_design = False
        row_data, row_params = [], []
        if t.ndim == 1 and sn == "Mul":
            row_design = design is not None and design.ndim == 1
            direct = set(_direct_data(design, ctx)) if row_design else set()
            indirect = (fixed_data - direct) | {
                n for term in sub for n in term.index + term.data if n not in term.row_data
            }
            row_data = sorted((direct | {n for term in sub for n in term.row_data}) - indirect)
            row_params = [n for term in sub for n in term.row_params]
        elif design is ins[0] and design.ndim == 2 and _is_dot(t):
            # PyTensor's matrix @ vector builds a Matmul with a column-vector RHS, then
            # squeezes the singleton axis. In both forms the left operand supplies axis 0.
            rhs = rv[0]
            row_design = (t.ndim == rhs.ndim == 1) or (
                t.ndim == rhs.ndim == 2 and rhs.type.shape[1] == 1
            )
            if row_design:
                row_data = _direct_data(design, ctx)
        if not data_like:
            kind = "gather" if index else "leaf"
            return [
                Term(
                    kind,
                    params=params,
                    index=index,
                    data=data,
                    multiplier=None,
                    row_data=row_data,
                    row_params=row_params,
                )
            ]
        if len(sub) != 1 or sub[0].kind not in ("leaf", "gather"):
            proof = None
        return [
            Term(
                "product",
                params=params,
                index=index,
                data=data,
                design=design,
                multiplier=proof,
                row_data=row_data,
                row_params=row_params,
                row_design=row_design,
            )
        ]
    if type(o.op).__name__ in ("AdvancedSubtensor", "AdvancedSubtensor1", "Subtensor"):
        if any(depends_on_rv(i) for i in ins[1:]):
            return None
        sub = _affine_terms(ins[0], ctx, blocked, hops)
        if sub is None:
            return None
        idx = sorted(
            {n for i in ins[1:] for n in _named_leaves(i, ctx) if ctx.role(n) == "data"}
            | {n for x in sub for n in x.index}
        )
        proof = sub[0].multiplier if len(sub) == 1 and sub[0].kind == "leaf" else None
        row_data = []
        if len(ins) == 2 and len(idx) == 1 and _unwrap(ins[1], ctx) is ctx.var(idx[0]):
            if t.ndim == 1 and ins[1].ndim == 1 and np.dtype(ins[1].dtype).kind in "iu":
                row_data = _direct_data(ins[1], ctx)
        else:
            proof = None
        return [
            Term(
                "gather",
                params=sorted({n for x in sub for n in x.params}),
                index=idx,
                data=sorted({n for x in sub for n in x.data}),
                multiplier=proof,
                row_data=row_data,
            )
        ]
    if type(o.op).__name__ == "Join":
        out = []
        for i in ins[1:]:
            r = _affine_terms(i, ctx, blocked, hops)
            if r is None:
                return None
            for term in r:
                term.row_data = []
                term.row_params = []
                term.row_design = False
            out += r
        return out
    return None


def _literal_multiplier(t) -> float | None:
    """A finite literal scalar, through broadcast wrappers and explicit scalar casts."""
    casts = []
    for _ in range(8):
        owner = getattr(t, "owner", None)
        if owner is None:
            break
        scalar = _scalar_op_name(owner.op)
        if scalar == "Cast":
            if np.dtype(t.dtype).kind not in "biuf":
                return None
            casts.append(owner.op.scalar_op)
        elif type(owner.op).__name__ != "DimShuffle" and scalar != "Identity":
            return None
        t = owner.inputs[0]
    from pytensor.tensor import TensorConstant

    if not isinstance(t, TensorConstant) or t.ndim != 0 or np.dtype(t.dtype).kind not in "biuf":
        return None
    value = t.data
    try:
        with np.errstate(invalid="raise", over="raise"):
            for cast in reversed(casts):
                value = cast.impl(value)
            multiplier = float(value)
    except (FloatingPointError, OverflowError, TypeError, ValueError):
        return None
    return multiplier if np.isfinite(multiplier) else None


def _varying_data(t, ctx: _Ctx) -> bool:
    """Data or a non-constant (varying) array — a covariate, not a scalar coefficient."""
    names = _named_leaves(t, ctx)
    if any(ctx.role(n) == "data" for n in names):
        return True
    a = ctx.fixed_value(t)
    return a is not None and a.size > 1 and not np.all(a == a.reshape(-1)[0])


# --------------------------------------------------------------------------- driver
def detect(model: Any, ir: ModelIR, named: dict[int, str]) -> list[BlockIR]:
    """All recognized blocks, in priority order, with their tags/glyphs applied to ``ir``."""
    consumers: dict[str, set[str]] = {}
    for e in ir.edges:
        consumers.setdefault(e.source, set()).add(e.target)
    ctx = _Ctx(model, named, ir, consumers)
    blocks: list[BlockIR] = []
    ncs: dict[str, _NC] = {}
    for n in ir.nodes:
        if n.role != "deterministic":
            continue
        ctx.visits[0] = 0
        try:
            nc = _noncentered(n.id, ctx)
        except Exception:
            nc = None
        if nc is not None:
            ncs[n.id] = nc

    # shrinkage first (it claims its coefficient over a plain non-centered reading)
    shrink: dict[str, _Shrink] = {}
    for n in ir.nodes:
        if n.role not in ("deterministic", "latent"):
            continue
        ctx.visits[0] = 0
        try:
            sh = (
                _shrinkage(n.id, ctx, ncs.get(n.id))
                if (n.id in ncs or n.role == "latent")
                else None
            )
        except Exception:
            sh = None
        if sh is not None:
            shrink[n.id] = sh

    for name, sh in shrink.items():
        title, variant = _shrink_title(sh, ctx)
        node = ctx.node(name)
        members = [name] + ([sh.aux] if sh.aux else [])
        if (
            sh.local not in members
            and _private(sh.local, set(members) | {name}, ctx)
            and (ctx.dims(sh.local) == tuple(node.dims))
        ):
            members.append(sh.local)
        notes = []
        if sh.slab is not None:
            notes.append(
                "the slab bounds each coefficient's conditional scale (not its value): "
                "scale = tau * lambda_tilde, lambda_tilde^2 = c^2 lambda^2 / (c^2 + tau^2 lambda^2)"
            )
        fam, nu = _local_family(sh.local, ctx)
        if fam == "HalfStudentT" and nu not in (None, 1.0):
            notes.append(
                f"local scales are half-t({nu:g}), not half-Cauchy: a lighter-tailed "
                "generalization of the horseshoe"
            )
        if sh.modifiers:
            notes.append("local scales are modulated by co-data: lambda * exp(C w)")
        if sh.center:
            notes.append("coefficients shrink toward a center, not toward zero")
        if not sh.centered:
            notes.append("non-centered: beta = center + scale * z, z ~ Normal(0, 1)")
        implied = _shrink_implied(sh)
        roles = {"coefficients": [name], "global": [sh.glob], "local": [sh.local]}
        if sh.aux:
            roles["aux"] = [sh.aux]
        if sh.slab:
            roles["slab"] = [sh.slab]
        if sh.modifiers:
            roles["modifier"] = sh.modifiers
        if sh.center:
            roles["center"] = sh.center
        blocks.append(
            BlockIR(
                id=f"block:shrinkage:{name}",
                kind="shrinkage",
                title=title,
                variant=variant,
                anchor=name,
                members=members,
                roles=roles,
                implied=implied,
                rule="shrinkage/1",
                notes=notes,
                references=list(REFS_SHRINKAGE),
                data={"local_family": fam, "nu_local": nu, "slab_power": sh.slab_power},
            )
        )
        _apply_shrinkage_glyph(node, sh, ctx)

    for name, nc in ncs.items():
        node = ctx.node(name)
        coarse = [
            nm
            for nm in nc.loc_names + nc.scale_names
            if ctx.role(nm) in ("latent", "deterministic")
            and _coarser(ctx.dims(nm), tuple(node.dims))
        ]
        kind = "hierarchy" if node.dims and coarse else "reparam"
        fam = "" if nc.family == "Normal" else f" ({_FAMILY_WORD.get(nc.family, nc.family)})"
        title = ("partial pooling · non-centered" if kind == "hierarchy" else "non-centered") + fam
        _loc_tex, _scale_tex, implied = _implied_tex(nc, ctx, wrap=True)
        members = [nc.aux, name]
        roles = {"aux": [nc.aux], "effect": [name]}
        if nc.loc_names:
            roles["loc"] = list(nc.loc_names)
        if nc.scale_names:
            roles["scale"] = list(nc.scale_names)
        blocks.append(
            BlockIR(
                id=f"block:{kind}:{name}",
                kind=kind,
                title=title,
                variant="noncentered",
                anchor=name,
                members=members,
                roles=roles,
                implied=implied,
                rule="noncentered/1",
                notes=[
                    f"{name} = location + scale * {nc.aux} with {nc.aux} standard "
                    f"{nc.aux_info['family']}: exactly {name} ~ {nc.family}(location, |scale|)"
                ],
                references=list(REFS_NONCENTERED),
                data={
                    "family": nc.family,
                    "loc_tex": _implied_tex(nc, ctx, wrap=False)[0],
                    "scale_tex": _implied_tex(nc, ctx, wrap=False)[1],
                },
            )
        )
        if name not in shrink:
            _apply_nc_glyph(node, nc)

    blocked = {b.anchor for b in blocks if b.anchor}
    effects = {b.anchor for b in blocks if b.kind in ("hierarchy", "shrinkage")}
    effects |= {b.anchor for b in centered_blocks(ir)}
    for n in ir.nodes:
        if n.role != "observed" or n.dist not in _REG_FAMILIES:
            continue
        ctx.visits[0] = 0
        try:
            b = _regression(n, ctx, blocked, effects)
        except Exception:
            b = None
        if b is not None:
            blocks.append(b)

    for n in ir.nodes:
        try:
            b = _measured(n, ctx)
        except Exception:
            b = None
        if b is not None:
            blocks.append(b)
    return blocks


_FAMILY_WORD = {
    "LogNormal": "log-normal",
    "LogitNormal": "logit-normal",
    "MultivariateNormal": "multivariate normal",
    "StudentT": "Student-t",
}


def _private(name: str, members: set[str], ctx: _Ctx) -> bool:
    return bool(ctx.consumers.get(name)) and ctx.consumers.get(name, set()) <= members


def _apply_nc_glyph(node, nc: _NC) -> None:
    g = _implied_glyph(nc)
    if g is not None:
        node.glyph = GlyphSpec(kind="schematic", source="implied_family")
        node.glyph_data = g
        node.glyph_note = None
    else:
        node.glyph, node.glyph_data = None, None
        if nc.family in ("LogNormal", "LogitNormal"):
            node.glyph_note = f"implied {nc.family}: its shape depends on the scale's parents"
        else:
            node.glyph_note = f"implied {nc.family}"


def _apply_shrinkage_glyph(node, sh: _Shrink, ctx: _Ctx) -> None:
    if sh.slab is not None:  # conditional on the scales, the coefficient is Normal
        from .glyph_data import _conditional_shape_from_values

        g = _conditional_shape_from_values("Normal", [None, None])
        if g is not None:
            node.glyph = GlyphSpec(kind="schematic", source="implied_family")
            node.glyph_data = g
            node.glyph_note = "given its scales, each coefficient is Normal"
        return
    fam, nu = _local_family(sh.local, ctx)
    lam_vals = ctx.slot_values(ctx.var(sh.local))
    numeric = all(v is not None and uniform([v]) for v in lam_vals)
    if fam == "HalfCauchy" and numeric:
        nu = 1.0
    if nu is None or not numeric or sh.modifiers:
        node.glyph, node.glyph_data = None, None
        node.glyph_note = "global-local shrinkage: shape depends on parents"
        return
    node.glyph = GlyphSpec(kind="schematic", source="implied_family")
    node.glyph_data = dict(_horseshoe_marginal(nu))
    node.glyph_note = "exact marginal given the global scale (pole at 0 clipped)"


def _regression(n, ctx: _Ctx, blocked: set[str], effects: set[str]) -> BlockIR | None:
    slot, titles = _REG_FAMILIES[n.dist]
    from ._ptutil import dist_params

    dp = dist_params(ctx.var(n.id))
    if len(dp) <= slot:
        return None
    transfer, eta = _link(dp[slot], ctx)
    title = titles.get(transfer)
    if title is None:
        return None
    terms = _affine_terms(eta, ctx, blocked)
    if not terms or not any(t.kind == "product" for t in terms):
        return None
    groups = sorted(
        {nm for t in terms if t.kind in ("gather", "leaf") for nm in t.params if nm in effects}
    )
    if groups:
        title += " · with group effects"
    coefs = sorted({nm for t in terms if t.kind == "product" for nm in t.params})
    notes = []
    if n.dist == "LogNormal":
        notes.append(
            "mu is the log-scale location: exp(mu) is the conditional median, and the "
            "conditional mean is exp(mu + sigma^2 / 2)"
        )
    roles = {"likelihood": [n.id], "coefficients": coefs}
    if groups:
        roles["group_effects"] = groups
    return BlockIR(
        id=f"block:regression:{n.id}",
        kind="regression",
        title=title,
        variant=transfer,
        anchor=n.id,
        members=[],
        roles=roles,
        rule="regression/1",
        notes=notes,
        references=list(REFS_REGRESSION),
        data={
            "link": transfer,
            # which data indexes / multiplies which parameter: what the fitted panels need
            "terms": [
                {"kind": t.kind, "params": t.params, "index": t.index, "data": t.data}
                for t in terms
                if t.kind != "const"
            ],
        },
    )


def _measured(n, ctx: _Ctx) -> BlockIR | None:
    """``x_obs ~ Family(x_true, ...)`` observed, with ``x_true`` a latent also used elsewhere: the
    errors-in-variables pattern."""
    if n.role != "observed" or not n.params:
        return None
    loc = n.params[0]
    if len(loc.parents) != 1:
        return None
    x = loc.parents[0]
    xn = ctx.node(x)
    if xn is None or xn.role != "latent" or tuple(xn.dims) != tuple(n.dims) or not xn.dims:
        return None
    from ._ptutil import dist_params

    dp = dist_params(ctx.var(n.id))
    if not dp or _unwrap(dp[0], ctx) is not ctx.var(x):
        return None  # the reading's location must be the true value itself
    if not (ctx.consumers.get(x, set()) - {n.id}):
        return None
    return BlockIR(
        id=f"block:measured:{x}",
        kind="measured",
        title="measured latent (errors-in-variables)",
        anchor=x,
        members=[],
        roles={"latent": [x], "reading": [n.id]},
        rule="measured/1",
        notes=[f"{n.id} is a noisy reading of {x}; {x} is what the rest of the model uses"],
    )


def centered_blocks(ir: ModelIR) -> list[BlockIR]:
    from ..blocks import centered

    return centered(ir)
