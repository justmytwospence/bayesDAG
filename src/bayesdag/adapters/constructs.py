"""Special-construct glyphs (the PyMC families a single 1-D curve can't honestly capture).

Detection keys on the RV class (``type(op).__name__``) — the op's print-name aliases/collapses, so
public-class distinctions are lost there. Sub-distributions and structural params are recovered from
``var.owner.inputs`` (verified per-construct layouts for PyMC 6.x). Every branch is guarded; anything
that can't be drawn faithfully returns an honest ``elision_reason`` badge instead of a wrong picture
or a crash.

``special_glyph(var)`` -> ``(GlyphSpec | None, data | None, elision_reason | None)``.
A non-``None`` GlyphSpec OR a non-``None`` elision_reason means "handled here"; an all-``None`` result
means "not special — let the generic univariate path handle it".
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..ir import GlyphSpec
from . import glyph_data as gd

_GRID = 64


def _ev(t) -> Optional[np.ndarray]:
    """Evaluate a (usually constant) tensor input to a numpy array, or None if it can't sample OR
    is governed by a parent RV (whose ``.eval()`` would be a random draw, not the real value)."""
    if _depends_on_rv(t):
        return None
    try:
        return np.asarray(t.eval(), dtype=float)
    except Exception:
        return None


def _op_is_rv(op, *, unknown: bool = True) -> bool:
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


def _depends_on_rv(t) -> bool:
    """True if a tensor's value is governed by a random variable (a prior) rather than fixed
    constants — i.e. `.eval()` would return an arbitrary random draw, not the real value.

    **Fails closed.** This is the single gate in front of every `.eval()` in the adapters, so
    any internal failure (a moved pytensor traversal API — it has moved once already, hence
    the import shim below) reports True: the node falls to a deterministic schematic instead
    of silently plotting a random draw badged as an analytic prior."""
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
            if op is not None and _op_is_rv(op):
                return True
    except Exception:
        return True
    return False


def _dist_name(rv) -> Optional[str]:
    op = getattr(getattr(rv, "owner", None), "op", None)
    if op is None:
        return None
    pn = getattr(op, "_print_name", None)
    return pn[0] if pn else type(op).__name__.removesuffix("RV")


def _lead_numeric(var, k: int) -> Optional[list]:
    """The first k distribution params as numeric arrays — for constructs that need only a LEADING
    subset (e.g. LKJ's n & eta, a random walk's drift) whose TRAILING params may legitimately be
    priors that must never be sampled (an LKJCholeskyCov's sd_dist, an innovation's scale). Returns
    None if any of the first k is itself a prior or non-evaluable."""
    node = getattr(var, "owner", None)
    if node is None:
        return None
    try:
        dparams = list(node.op.dist_params(node))
    except Exception:
        dparams = list(node.inputs[2:])
    out = []
    for dp in dparams[:k]:
        if _depends_on_rv(dp):
            return None
        try:
            out.append(np.asarray(dp.eval()))
        except Exception:
            return None
    return out


def _base_frozen(rv):
    """A scipy frozen for a (univariate) sub-RV, reusing the verified Phase-2 translations."""
    name = _dist_name(rv)
    params = gd._numeric_params(rv)
    return gd._scipy_frozen(name, params) if (name and params is not None) else None


def _first_matrix(params) -> Optional[np.ndarray]:
    for p in params or []:
        a = np.asarray(p, dtype=float)
        if a.ndim >= 2 and a.shape[-1] > 1 and a.shape[-2] > 1:
            return a[..., :, :] if a.ndim == 2 else a.reshape(a.shape[-2], a.shape[-1])
    return None


def _heatmap(mat: np.ndarray, max_n: int = 12) -> dict:
    m = np.asarray(mat, float)
    m = m[:max_n, :max_n]
    lo, hi = float(np.min(m)), float(np.max(m))
    rng = (hi - lo) or 1.0
    return {"matrix": [[float((v - lo) / rng) for v in row] for row in m]}


def _badge(reason: str):
    return GlyphSpec(kind="schematic", source="prior_family_only"), None, reason


# ---- per-construct handlers (each returns (spec, data) or None) ------------------------------


def _multivariate(var, params, op_cls):
    cov = _first_matrix(params)
    if cov is None:
        return None
    d = cov.shape[0]
    if d <= 4:
        return GlyphSpec(kind="pairplot", source="prior_analytic"), {"cov": [[float(x) for x in r] for r in cov]}
    return GlyphSpec(kind="heatmap", source="prior_analytic"), _heatmap(cov)


def _matrix_only(var, params):
    mat = _first_matrix(params)
    if mat is None:
        return None
    return GlyphSpec(kind="heatmap", source="prior_analytic"), _heatmap(mat)


def _dirichlet(var, params):
    if not params:
        return None
    a = np.asarray(params[0], float).ravel()
    if a.size < 2 or not np.all(np.isfinite(a)):
        return None
    import scipy.stats as st

    a0 = float(a.sum())
    xs = np.linspace(1e-3, 1 - 1e-3, _GRID)
    curves = []
    for ai in a[:6]:
        ys = st.beta(ai, max(a0 - ai, 1e-6)).pdf(xs)
        m = float(np.max(ys)) or 1.0
        curves.append({"xs": [float(x) for x in xs], "ys": [float(y / m) for y in ys]})
    return GlyphSpec(kind="simplex", source="prior_analytic"), {"curves": curves}


def _interpolated(var, params):
    if len(params or []) < 2:
        return None
    xs = np.asarray(params[0], float).ravel()
    ys = np.asarray(params[1], float).ravel()
    if xs.size < 2 or xs.size != ys.size or ys.max() <= 0:
        return None
    m = float(ys.max()) or 1.0
    return GlyphSpec(kind="density", source="prior_analytic"), {
        "xs": [float(x) for x in xs],
        "ys": [float(y / m) for y in ys],
    }


def _censored(var):
    ins = var.owner.inputs
    fr = _base_frozen(ins[0])
    if fr is None:
        return None
    lo, hi = _ev(ins[1]), _ev(ins[2])
    dens = gd._density_from_frozen(fr)
    if dens is None:
        return None
    x0, x1 = dens["xs"][0], dens["xs"][-1]
    span = (x1 - x0) or 1.0
    spikes = []
    if lo is not None and np.isfinite(lo):
        spikes.append({"x": max(0.0, min(1.0, (float(lo) - x0) / span)), "h": float(min(1.0, fr.cdf(float(lo)) * 3))})
    if hi is not None and np.isfinite(hi):
        spikes.append({"x": max(0.0, min(1.0, (float(hi) - x0) / span)), "h": float(min(1.0, fr.sf(float(hi)) * 3))})
    return GlyphSpec(kind="censored", source="prior_analytic"), {**dens, "spikes": spikes}


def _truncated_normal(var, params):
    if len(params or []) < 4:
        return None
    import scipy.stats as st

    loc, scale, lo, hi = (float(np.asarray(params[i]).reshape(-1)[0]) for i in range(4))
    base = st.norm(loc, scale)
    lo = lo if np.isfinite(lo) else base.ppf(0.001)
    hi = hi if np.isfinite(hi) else base.ppf(0.999)
    if hi <= lo:
        return None
    xs = np.linspace(lo, hi, _GRID)
    z = base.cdf(hi) - base.cdf(lo) or 1.0
    ys = base.pdf(xs) / z
    m = float(np.max(ys)) or 1.0
    return GlyphSpec(kind="density", source="prior_analytic"), {
        "xs": [float(x) for x in xs],
        "ys": [float(y / m) for y in ys],
    }


def _random_walk(var):
    ins = var.owner.inputs
    innov = next((i for i in ins[1:] if getattr(i, "owner", None) is not None and _dist_name(i)), None)
    if innov is None:
        return None
    # The fan is `drift*t ± 2*sd*sqrt(t)`, so we must know which slots ARE (mu, sigma). Only a
    # Normal innovation is read positionally: other families reorder/reparametrize the pair (a
    # StudentT is [nu, mu, sigma], so slot 0 is the df — reading it as drift tilts the fan by
    # nu*t). Anything else badges rather than draws a wrong slope.
    if _dist_name(innov) != "Normal":
        return None
    p = gd._numeric_params(innov)
    if p:
        q = [float(np.asarray(x).reshape(-1)[0]) for x in p]
        drift = q[0] if len(q) > 1 else 0.0
        sd = q[-1]
    else:
        # the innovation SCALE is a prior: the fan's absolute width is unknown. But a DRIFTLESS walk's
        # normalized fan (the characteristic sqrt-t spread) is scale-invariant — the sd cancels — so
        # draw it canonically. With drift, the drift/scale balance matters, so honest-badge instead.
        lead = _lead_numeric(innov, 1)  # the innovation mean (drift); trailing scale may be a prior
        if lead is None:
            return None
        drift = float(np.asarray(lead[0]).reshape(-1)[0])
        if drift != 0.0:
            return None
        sd = 1.0
    T = 20
    t = np.arange(T + 1)
    mean = drift * t
    band = 2.0 * sd * np.sqrt(t)
    lo, hi = mean - band, mean + band
    g0, g1 = float(lo.min()), float(hi.max())
    rng = (g1 - g0) or 1.0

    def norm(a):
        return [float((v - g0) / rng) for v in a]

    return GlyphSpec(kind="fan", source="prior_analytic"), {"mid": norm(mean), "lo": norm(lo), "hi": norm(hi)}


def _lkj(var):
    """LKJ prior over correlation matrices -> the marginal density of a single pairwise correlation
    on [-1, 1] (a symmetric Beta whose concentration encodes eta — the AR-style 'how much
    correlation is plausible' signature). Params are [n (dim), eta]; an LKJCholeskyCov also carries a
    trailing sd_dist that may be a prior (and is irrelevant to the correlation marginal), so read ONLY
    the leading n, eta and never sample sd_dist."""
    params = _lead_numeric(var, 2)
    if not params or len(params) < 2:
        return None
    import scipy.stats as st

    n = int(np.asarray(params[0]).ravel()[0])
    eta = float(np.asarray(params[1]).ravel()[0])
    if n < 2:
        return None
    alpha = eta + (n - 2) / 2.0  # marginal of an off-diagonal r: (r+1)/2 ~ Beta(alpha, alpha)
    rs = np.linspace(-0.999, 0.999, _GRID)
    ys = st.beta(alpha, alpha).pdf((rs + 1) / 2.0) / 2.0
    if not np.all(np.isfinite(ys)) or ys.max() <= 0:
        return None
    m = float(ys.max()) or 1.0
    return GlyphSpec(kind="density", source="prior_analytic"), {
        "xs": [float(x) for x in rs],
        "ys": [float(y / m) for y in ys],
    }


def _ar_acf(rho: np.ndarray, k: int) -> list[float]:
    """Theoretical ACF rho(0..k) of a stationary AR(p) via Yule-Walker (first p) then recursion."""
    p = len(rho)
    mat = np.zeros((p, p))
    c = np.zeros(p)
    for kk in range(1, p + 1):
        mat[kk - 1, kk - 1] += 1.0
        for j in range(1, p + 1):
            m = abs(kk - j)
            if m == 0:
                c[kk - 1] += rho[j - 1]
            else:
                mat[kk - 1, m - 1] -= rho[j - 1]
    acf = [1.0] + list(np.linalg.solve(mat, c))
    for kk in range(p + 1, k + 1):
        acf.append(float(sum(rho[j - 1] * acf[kk - j] for j in range(1, p + 1))))
    return acf


def _levinson_pacf(acf: list[float], nlags: int) -> list[float]:
    """Partial autocorrelations from the ACF (Durbin-Levinson). For a pure AR(p) the PACF cuts off
    after lag p — the distinctive AR fingerprint."""
    phi = np.zeros((nlags + 1, nlags + 1))
    v = acf[0]
    pacf = []
    for k in range(1, nlags + 1):
        num = acf[k] - sum(phi[k - 1, j] * acf[k - j] for j in range(1, k))
        pkk = num / v if abs(v) > 1e-12 else 0.0
        phi[k, k] = pkk
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - pkk * phi[k - 1, k - j]
        v *= 1 - pkk * pkk
        pacf.append(float(pkk))
    return pacf


def _ar_order(rho_in) -> Optional[int]:
    """The AR order = number of coefficients. It's STRUCTURAL (the coefficient vector's length), so
    it's knowable from the static shape even when the coefficient VALUES are a prior we must not
    sample — i.e. without ``.eval()``-ing the (random) tensor."""
    st = getattr(getattr(rho_in, "type", None), "shape", None)
    if st is not None:
        if len(st) == 0:
            return 1  # scalar coefficient -> AR(1)
        if st[-1] is not None:
            return int(st[-1])
    try:
        sh = np.asarray(rho_in.shape.eval())  # shape graph is RNG-independent, unlike the value
        return int(sh[-1]) if sh.size else 1
    except Exception:
        return None


def _ar(var):
    rho_in = var.owner.inputs[0]  # AR coefficients
    if _depends_on_rv(rho_in):
        # the coefficients are a prior: their .eval() is a meaningless random draw. Don't fake a
        # PACF — show the AR ORDER honestly (p schematic lags, then the cutoff), in schematic style.
        p = _ar_order(rho_in)
        if not p:
            return None
        nlags = min(10, p + 4)
        vals = [0.7 if i < p else 0.0 for i in range(nlags)]
        return GlyphSpec(kind="stem", source="prior_family_only"), {
            "lags": list(range(1, nlags + 1)),
            "values": vals,
        }
    rho = _ev(rho_in)
    if rho is None:
        return None
    rho = np.atleast_1d(rho).ravel().astype(float)
    p = rho.size
    if p == 0:
        return None
    nlags = min(10, p + 4)
    if np.sum(rho**2) >= 1.0:  # known but non-stationary
        return _badge("autoregressive — non-stationary")
    try:  # known, stationary coefficients -> the true theoretical PACF
        pacf = _levinson_pacf(_ar_acf(rho, nlags), nlags)
    except Exception:
        return None
    return GlyphSpec(kind="stem", source="prior_analytic"), {
        "lags": list(range(1, len(pacf) + 1)),
        "values": pacf,
    }


def _is_rv(i) -> bool:
    """Is this *input* an actual sub-RV (vs a weight vector / plumbing)? Unlike the eval gate,
    an unclassifiable op must NOT be treated as a component — a leaked `MakeVector` would draw
    a wrong glyph, whereas dropping it yields an honest badge."""
    op = getattr(getattr(i, "owner", None), "op", None)
    return _op_is_rv(op, unknown=False) if op is not None else False


def _mixture(var):
    comps = [i for i in var.owner.inputs if _is_rv(i)]  # actual component RVs (not weight vectors)
    names = [_dist_name(c) for c in comps]
    if "DiracDelta" in names:  # zero-inflated / hurdle: a spike at 0 + the base count pmf
        base = next((c for c in comps if _dist_name(c) != "DiracDelta"), None)
        bn, bp = (_dist_name(base), gd._numeric_params(base)) if base is not None else (None, None)
        bars = gd._pmf(bn, bp) if (bn and bp is not None) else None
        if bars is not None:
            return GlyphSpec(kind="mixture", source="prior_analytic"), {"base": bars, "spike": 0.85}
        return None
    # continuous mixture: overlay component densities on ONE shared x-range (a single batched RV
    # carries all components, so expand its vector params element-wise)
    frozens = []
    for c in comps:
        name, params = _dist_name(c), gd._numeric_params(c)
        if not name or params is None:
            continue
        arrs = [np.atleast_1d(np.asarray(p, float).ravel()) for p in params]
        k = max((a.size for a in arrs), default=1)
        for idx in range(min(k, 6)):
            fr = gd._scipy_frozen(name, [a[idx % a.size] for a in arrs])
            if fr is not None:
                frozens.append(fr)
    if len(frozens) < 2:
        return None
    lo = min(float(fr.ppf(0.01)) for fr in frozens)
    hi = max(float(fr.ppf(0.99)) for fr in frozens)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return None
    xs = np.linspace(lo, hi, _GRID)
    pdfs = [np.asarray(fr.pdf(xs), float) for fr in frozens]
    gmax = max(float(np.max(p)) for p in pdfs) or 1.0
    curves = [{"xs": [float(x) for x in xs], "ys": [float(y / gmax) for y in p]} for p in pdfs]
    return GlyphSpec(kind="mixture", source="prior_analytic"), {"curves": curves}


# Canonical step heights for the BART glyph — a fixed, plausible sum-of-trees draw (parameter-free,
# so it is identical on every render). It depicts BART's STRUCTURE (piecewise-constant function),
# never the node's fitted values — the same honest-schematic stance as the random-walk fan.
_BART_STEPS = [0.40, 0.62, 0.55, 0.80, 0.48, 0.68, 0.42, 0.58]


def special_glyph(var):
    """Return (GlyphSpec, data, elision_reason) for a special construct, or (None, None, None)."""
    op = getattr(getattr(var, "owner", None), "op", None)
    if op is None:
        return None, None, None
    cls = type(op).__name__
    params = gd._numeric_params(var)
    try:
        if cls in ("MvNormalRV", "MvStudentTRV"):
            r = _multivariate(var, params, cls)
            return (*r, None) if r else _badge("multivariate — covariance not numeric")
        if cls in ("WishartRV", "MatrixNormalRV", "KroneckerNormalRV"):
            r = _matrix_only(var, params)
            return (*r, None) if r else _badge("matrix-valued — params not numeric")
        if cls in ("LKJCorrRV", "_LKJCholeskyCovRV"):
            r = _lkj(var)
            return (*r, None) if r else _badge("correlation-matrix prior")
        if cls == "DirichletRV":
            r = _dirichlet(var, params)
            return (*r, None) if r else _badge("simplex — concentration not numeric")
        if cls in ("DirichletMultinomialRV", "MultinomialRV", "StickBreakingWeightsRV"):
            return _badge("multivariate count/simplex")
        if cls == "InterpolatedRV":
            r = _interpolated(var, params)
            return (*r, None) if r else _badge("interpolated density")
        if cls == "CensoredRV":
            r = _censored(var)
            return (*r, None) if r else _badge("censored")
        if cls == "TruncatedNormalRV":
            r = _truncated_normal(var, params)
            return (*r, None) if r else _badge("truncated")
        if cls == "TruncatedRV":
            return _badge("truncated")
        if cls == "RandomWalkRV":
            r = _random_walk(var)
            return (*r, None) if r else _badge("random walk")
        if cls == "AutoRegressiveRV":
            r = _ar(var)  # may already be a full badge 3-tuple (non-stationary case)
            return (r if len(r) == 3 else (*r, None)) if r else _badge("autoregressive")
        if cls == "GARCH11RV":
            return _badge("GARCH(1,1) — conditional volatility")
        if cls in ("EulerMaruyamaRV",):
            return _badge("SDE — opaque drift/diffusion")
        if cls in ("MixtureRV", "_HurdleRV"):
            r = _mixture(var)
            return (*r, None) if r else _badge("mixture")
        if cls in ("CARRV", "ICARRV"):
            r = _matrix_only(var, params)
            return (*r, None) if r else _badge("spatial (CAR/ICAR)")
        if cls in ("FlatRV", "HalfFlatRV"):
            return _badge("improper prior")
        if cls.startswith("BART"):  # pymc-bart op is BART_<name>; sum of trees => step-function draws
            return GlyphSpec(kind="step", source="prior_family_only"), {"ys": list(_BART_STEPS)}, None
        if cls in ("CustomDistRV", "SymbolicRandomVariable") or "CustomDist" in cls or "Simulator" in cls:
            return _badge("custom density — elided")
    except Exception:
        return _badge("elided")
    return None, None, None
