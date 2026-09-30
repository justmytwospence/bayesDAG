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

import numpy as np

from ..ir import GlyphSpec
from . import glyph_data as gd

_GRID = 64


def _ev(t) -> np.ndarray | None:
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


def _dist_name(rv) -> str | None:
    op = getattr(getattr(rv, "owner", None), "op", None)
    if op is None:
        return None
    pn = getattr(op, "_print_name", None)
    return pn[0] if pn else type(op).__name__.removesuffix("RV")


def _lead_numeric(var, k: int) -> list | None:
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


class _Differ(Exception):
    """The batch members of a parameter differ, so no single glyph describes the node."""


def _one_member(a: np.ndarray, event_ndim: int) -> np.ndarray:
    """The single member of a batched parameter whose last ``event_ndim`` axes are the event.
    Raises ``_Differ`` when batch members are not all identical — one picture would then state
    something false about every member but the first."""
    a = np.asarray(a, dtype=float)
    if a.ndim <= event_ndim:
        return a
    flat = a.reshape((-1, *a.shape[a.ndim - event_ndim :]))
    if not np.all(flat == flat[0]):
        raise _Differ
    return flat[0]


def _first_matrix(params) -> np.ndarray | None:
    for p in params or []:
        a = np.asarray(p, dtype=float)
        if a.ndim >= 2 and a.shape[-1] > 1 and a.shape[-2] > 1:
            return _one_member(a, 2)
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


# The covariance/scale slot of each multivariate family, in ``dist_params`` order. Reading "the
# first matrix-shaped param" instead would take a batched MEAN (shape (n, d)) for a covariance.
_COV_SLOT = {"MvNormalRV": 1, "MvStudentTRV": 2}


def _multivariate(params, cls: str = "MvNormalRV"):
    slot = _COV_SLOT.get(cls)
    if slot is None or params is None or len(params) <= slot:
        return None
    cov = _one_member(np.asarray(params[slot], float), 2)
    if cov.ndim != 2 or cov.shape[0] < 2:
        return None
    d = cov.shape[0]
    if d <= 4:
        return GlyphSpec(kind="pairplot", source="prior_analytic"), {
            "cov": [[float(x) for x in r] for r in cov]
        }
    return GlyphSpec(kind="heatmap", source="prior_analytic"), _heatmap(cov)


def _matrix_only(params):
    mat = _first_matrix(params)
    if mat is None:
        return None
    return GlyphSpec(kind="heatmap", source="prior_analytic"), _heatmap(mat)


def _dirichlet(params):
    if not params:
        return None
    a = _one_member(np.asarray(params[0], float), 1).ravel()
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


def _interpolated(params):
    if len(params or []) < 2:
        return None
    xs = _one_member(np.asarray(params[0], float), 1).ravel()
    ys = _one_member(np.asarray(params[1], float), 1).ravel()
    if xs.size < 2 or xs.size != ys.size or ys.max() <= 0:
        return None
    m = float(ys.max()) or 1.0
    return GlyphSpec(kind="density", source="prior_analytic"), {
        "xs": [float(x) for x in xs],
        "ys": [float(y / m) for y in ys],
    }


# Censored-mass spikes are scaled up by this factor so a few-percent mass is legible beside a
# peak-normalized density. The bar is therefore a MARKER of censoring, not a readable
# probability — the exact value ships as `spikes[i]["p"]` and is captioned in the panel.
_SPIKE_GAIN = 3.0


def _censored(var):
    """Base density on the INTERIOR of the censoring interval only (no density exists outside
    it — that mass sits on the bounds), plus a spike at each finite bound carrying its true mass."""
    ins = var.owner.inputs
    fr = _base_frozen(ins[0])
    if fr is None:
        return None
    lo, hi = _ev(ins[1]), _ev(ins[2])
    lo = float(np.asarray(lo).reshape(-1)[0]) if lo is not None else -np.inf
    hi = float(np.asarray(hi).reshape(-1)[0]) if hi is not None else np.inf
    if not hi > lo:
        return None  # coincident bounds: all mass on a point, no density to draw
    try:
        a = max(lo, float(fr.ppf(0.005)))
        b = min(hi, float(fr.ppf(0.995)))
    except Exception:
        return None
    if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
        return None
    xs = np.linspace(a, b, _GRID)
    ys = np.asarray(fr.pdf(xs), float)
    if not np.all(np.isfinite(ys)) or ys.max() <= 0:
        return None
    m = float(ys.max())
    span = b - a
    spikes = []
    # A bound outside the drawn range carries < 0.5% mass (that is why it is outside): omitted.
    if np.isfinite(lo) and a <= lo <= b:
        spikes.append({"x": float((lo - a) / span), "p": float(fr.cdf(lo))})
    if np.isfinite(hi) and a <= hi <= b:
        spikes.append({"x": float((hi - a) / span), "p": float(fr.sf(hi))})
    # The spike heights are drawn EXAGGERATED (see `_SPIKE_GAIN`): the censored mass is often a
    # few percent, which is invisible next to a peak-normalized density, yet its presence is the
    # whole point of the glyph. The true probability travels as `p` so the reader can be told.
    for sp in spikes:
        sp["h"] = float(min(1.0, sp["p"] * _SPIKE_GAIN))
    return GlyphSpec(kind="censored", source="prior_analytic"), {
        "xs": [float(x) for x in xs],
        "ys": [float(y / m) for y in ys],
        "spikes": spikes,
        "spike_gain": _SPIKE_GAIN,
    }


def _truncated_normal(params):
    if len(params or []) < 4:
        return None
    if not gd._uniform_params(params):
        raise _Differ
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


def _rw_steps(var) -> int | None:
    sh = getattr(getattr(var, "type", None), "shape", None)
    if sh and sh[-1] is not None:
        return int(sh[-1])
    try:
        return int(np.asarray(var.shape.eval())[-1])  # shape graph is RNG-independent
    except Exception:
        return None


def _normal_moments(rv) -> tuple[float, float] | None:
    """(mean, variance) of a numeric, elementwise-uniform Normal or a DiracDelta point mass."""
    name = _dist_name(rv)
    p = gd._numeric_params(rv)
    if p is None or not gd._uniform_params(p):
        return None
    q = [float(np.asarray(x).reshape(-1)[0]) for x in p]
    if name == "Normal" and len(q) >= 2:
        return q[0], q[1] ** 2
    if name == "DiracDelta" and q:
        return q[0], 0.0
    return None


_Z90 = 1.6448536269514722  # central 90% band of a Normal


def _random_walk(var):
    """The exact marginal band of a Gaussian random walk: ``X_t = X_0 + sum of t innovations``, so
    ``E[X_t] = m0 + t*m`` and ``Var[X_t] = v0 + t*v`` — drawn as the central 90% band over the
    walk's real length. Only drawn when that band is actually knowable."""
    ins = var.owner.inputs
    init, innov = ins[0], ins[1]
    fam = _dist_name(innov)
    if fam != "Normal":  # other innovation families: the band is not a Normal sum
        return None, None, gd.GlyphNote(f"random walk with {fam} innovations: band not drawn")
    unknown = (None, None, gd.GlyphNote("random walk: the band's shape depends on parents"))
    n = _rw_steps(var)
    if not n or n < 2:
        return None
    t = np.arange(n, dtype=float)  # t = 0 is the initial value
    step = _normal_moments(innov)
    start = _normal_moments(init)
    source = "prior_analytic"
    if step is not None and start is not None:
        (m, v), (m0, v0) = step, start
    else:
        # The innovation scale is a prior. The walk's shape survives only when everything else
        # scales with it: zero drift and a point-mass start make the band exactly sd*sqrt(t) up
        # to the unknown sd — a standardized fan. A spread-out start (v0 > 0) does not scale with
        # the innovation sd, so its shape is genuinely unknown.
        lead = _lead_numeric(innov, 1)
        if lead is None or float(np.asarray(lead[0]).reshape(-1)[0]) != 0.0:
            return unknown
        if start is None or start[1] != 0.0:
            return unknown
        m, v, m0, v0 = 0.0, 1.0, start[0], 0.0
        source = "conditional_standardized"
    mean = m0 + t * m
    sd = np.sqrt(v0 + t * v)
    lo, hi = mean - _Z90 * sd, mean + _Z90 * sd
    if n > 64:  # keep the payload bounded; the band is smooth
        idx = np.unique(np.linspace(0, n - 1, 64).round().astype(int))
        mean, lo, hi = mean[idx], lo[idx], hi[idx]
    g0, g1 = float(lo.min()), float(hi.max())
    rng = (g1 - g0) or 1.0

    def norm(a):
        return [float((x - g0) / rng) for x in a]

    data = {"mid": norm(mean), "lo": norm(lo), "hi": norm(hi), "band": 0.9, "steps": int(n)}
    return GlyphSpec(kind="fan", source=source), data, None


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
    acf = [1.0, *np.linalg.solve(mat, c)]
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


def _ar_order(rho_in) -> int | None:
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


def _is_stationary(phi: np.ndarray) -> bool:
    """Every root of the AR lag polynomial outside the unit circle <=> every eigenvalue of the
    companion matrix strictly inside it (with a margin: a unit root is not stationary)."""
    p = phi.size
    comp = np.zeros((p, p))
    comp[0, :] = phi
    if p > 1:
        comp[1:, :-1] = np.eye(p - 1)
    return bool(np.max(np.abs(np.linalg.eigvals(comp))) < 1.0 - 1e-8)


def _ar(var):
    op = var.owner.op
    rho_in = var.owner.inputs[0]  # AR coefficients (with the intercept first when constant=True)
    const = bool(getattr(op, "constant_term", False))
    order = getattr(op, "ar_order", None)
    if _depends_on_rv(rho_in):
        # the coefficients are a prior: their .eval() is a meaningless random draw, and no PACF
        # heights are knowable. The order IS structural — say that, don't draw made-up stems.
        p = order if order else ((_ar_order(rho_in) or 1) - (1 if const else 0))
        return (
            None,
            None,
            gd.GlyphNote(f"AR({p}): partial autocorrelation cuts off after lag {p}"),
        )
    rho = _ev(rho_in)
    if rho is None:
        return None
    try:
        rho = _one_member(np.atleast_1d(rho), 1).ravel().astype(float)
    except _Differ:
        return None, None, gd.GlyphNote("elements differ")
    phi = rho[1:] if const else rho
    if order:
        phi = phi[: int(order)]
    p = phi.size
    if p == 0:
        return None
    if not _is_stationary(phi):
        return _badge("autoregressive — non-stationary")
    nlags = min(10, p + 4)
    try:  # known, stationary coefficients -> the true theoretical PACF of the stationary process
        pacf = _levinson_pacf(_ar_acf(phi, nlags), nlags)
    except Exception:
        return None
    return (
        GlyphSpec(kind="stem", source="prior_analytic"),
        {"lags": list(range(1, len(pacf) + 1)), "values": pacf, "stationary": True},
        None,
    )


def _is_rv(i) -> bool:
    """Is this *input* an actual sub-RV (vs a weight vector / plumbing)? Unlike the eval gate,
    an unclassifiable op must NOT be treated as a component — a leaked `MakeVector` would draw
    a wrong glyph, whereas dropping it yields an honest badge."""
    op = getattr(getattr(i, "owner", None), "op", None)
    return _op_is_rv(op, unknown=False) if op is not None else False


def _mixture_parts(var):
    """(weights | None, components). Weights come from the op's verified first ``dist_params``
    slot (``w``); they are None when a parent governs them (e.g. a Dirichlet) or when batched
    weight rows differ."""
    node = var.owner
    try:
        dp = list(node.op.dist_params(node))
    except Exception:
        return None, []
    if not dp:
        return None, []
    comps = [c for c in dp[1:] if _is_rv(c)]
    w = _ev(dp[0])
    if w is not None:
        try:
            w = _one_member(np.asarray(w, float), 1).ravel()
        except _Differ:
            w = None
    if w is not None and (np.any(w < 0) or abs(float(w.sum()) - 1.0) > 1e-6):
        w = None
    return w, comps


def _mixture(var):
    w, comps = _mixture_parts(var)
    names = [_dist_name(c) for c in comps]
    if "DiracDelta" in names:
        return _point_mass_mixture(w, comps, names)
    # continuous mixture: overlay component densities on ONE shared x-range (a single batched RV
    # carries all components, so expand its vector params element-wise)
    frozens = []
    for c in comps:
        name, params = _dist_name(c), gd._numeric_params(c)
        if not name or params is None:
            return None  # an unknown component: no honest overlay
        arrs = [np.atleast_1d(np.asarray(p, float).ravel()) for p in params]
        k = max((a.size for a in arrs), default=1)
        for idx in range(k):
            fr = gd._scipy_frozen(name, [a[idx % a.size] for a in arrs])
            if fr is None:
                return None
            frozens.append(fr)
    if len(frozens) < 2 or len(frozens) > 8:
        return None
    lo = min(float(fr.ppf(0.01)) for fr in frozens)
    hi = max(float(fr.ppf(0.99)) for fr in frozens)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return None
    xs = np.linspace(lo, hi, _GRID)
    pdfs = [np.asarray(fr.pdf(xs), float) for fr in frozens]
    # Scale each component by its weight when the weights are KNOWN: a 0.9/0.1 mixture whose
    # components are drawn at equal height misrepresents which one carries the mass. Unknown
    # (prior) weights keep the unweighted overlay — shapes only, no claim about proportions.
    weighted = w is not None and w.size == len(pdfs)
    if weighted:
        pdfs = [p * wi for p, wi in zip(pdfs, w, strict=True)]
    gmax = max(float(np.max(p)) for p in pdfs) or 1.0
    curves = [{"xs": [float(x) for x in xs], "ys": [float(y / gmax) for y in p]} for p in pdfs]
    spec = GlyphSpec(kind="mixture", source="prior_analytic")
    data = {"curves": curves, "weighted": weighted}
    if weighted:
        return spec, data, None
    return spec, data, gd.GlyphNote("components only: the mixture weights depend on parents")


def _point_mass_mixture(w, comps, names):
    """Zero-inflated (point-mass) mixture: the EXACT mixture pmf ``sum_i w_i pmf_i(k)``, so the
    bar at the point mass is its true total probability (the point's own weight plus the base
    family's mass there). A DiracDelta keeps its real location."""
    pmfs: list[tuple[str, object]] = []  # ("point", loc) | ("base", frozen)
    for c, name in zip(comps, names, strict=True):
        params = gd._numeric_params(c)
        if params is None or not gd._uniform_params(params):
            return None
        if name == "DiracDelta":
            pmfs.append(("point", round(float(np.asarray(params[0]).reshape(-1)[0]))))
        else:
            fr = gd._discrete_frozen(name, params)
            if fr is None:
                return None
            pmfs.append(("base", fr))
    los, his = [], []
    for kind, v in pmfs:
        if kind == "point":
            los.append(v)
            his.append(v)
        else:
            los.append(int(v.ppf(0.001)))
            his.append(int(v.ppf(0.999)))
    lo, hi = min(los), max(his)
    hi = min(hi, lo + 39)
    cats = np.arange(lo, hi + 1)
    if w is None or w.size != len(pmfs):
        base = next((v for k, v in pmfs if k == "base"), None)
        if base is None:
            return None
        hts = np.asarray(base.pmf(cats), float)
        m = float(hts.max()) or 1.0
        return (
            GlyphSpec(kind="mixture", source="prior_analytic"),
            {"base": {"cats": [int(c) for c in cats], "heights": [float(h / m) for h in hts]}},
            gd.GlyphNote("base component only: the point-mass weight depends on parents"),
        )
    total = np.zeros(cats.size)
    for wi, (kind, v) in zip(w, pmfs, strict=True):
        total += wi * ((cats == v).astype(float) if kind == "point" else np.asarray(v.pmf(cats)))
    m = float(total.max()) or 1.0
    return (
        GlyphSpec(kind="mixture", source="prior_analytic"),
        {
            "base": {"cats": [int(c) for c in cats], "heights": [float(h / m) for h in total]},
            "weighted": True,
        },
        None,
    )


def special_glyph(var):
    """Return (GlyphSpec, data, elision_reason) for a special construct, or (None, None, None)."""
    op = getattr(getattr(var, "owner", None), "op", None)
    if op is None:
        return None, None, None
    cls = type(op).__name__
    try:
        params = gd._numeric_params(var)  # inside the guard: it walks + evals the param graph
        if cls in ("MvNormalRV", "MvStudentTRV"):
            r = _multivariate(params, cls)
            return (*r, None) if r else _badge("multivariate — covariance not numeric")
        if cls in ("WishartRV", "MatrixNormalRV", "KroneckerNormalRV"):
            r = _matrix_only(params)
            return (*r, None) if r else _badge("matrix-valued — params not numeric")
        if cls in ("LKJCorrRV", "_LKJCholeskyCovRV"):
            r = _lkj(var)
            return (*r, None) if r else _badge("correlation-matrix prior")
        if cls == "DirichletRV":
            r = _dirichlet(params)
            return (*r, None) if r else _badge("simplex — concentration not numeric")
        if cls in ("DirichletMultinomialRV", "MultinomialRV", "StickBreakingWeightsRV"):
            return _badge("multivariate count/simplex")
        if cls == "InterpolatedRV":
            r = _interpolated(params)
            return (*r, None) if r else _badge("interpolated density")
        if cls == "CensoredRV":
            r = _censored(var)
            return (*r, None) if r else _badge("censored")
        if cls == "TruncatedNormalRV":
            r = _truncated_normal(params)
            return (*r, None) if r else _badge("truncated")
        if cls == "TruncatedRV":
            return _badge("truncated")
        if cls == "RandomWalkRV":
            r = _random_walk(var)
            return r if r else _badge("random walk")
        if cls == "AutoRegressiveRV":
            r = _ar(var)  # already a full 3-tuple (glyph, badge or note)
            return r if r else _badge("autoregressive")
        if cls == "GARCH11RV":
            return _badge("GARCH(1,1) — conditional volatility")
        if cls in ("EulerMaruyamaRV",):
            return _badge("SDE — opaque drift/diffusion")
        if cls in ("MixtureRV", "_HurdleRV"):
            r = _mixture(var)  # a full 3-tuple (a partial picture carries its note)
            return r if r else _badge("mixture")
        if cls in ("CARRV", "ICARRV"):
            r = _matrix_only(params)
            return (*r, None) if r else _badge("spatial (CAR/ICAR)")
        if cls in ("FlatRV", "HalfFlatRV"):
            return _badge("improper prior")
        if cls.startswith("BART"):  # pymc-bart op is BART_<name>
            # No closed-form prior; any drawn step function would be made-up heights. State the
            # structure in words instead.
            return None, None, gd.GlyphNote("sum of trees: a piecewise-constant function of X")
        if (
            cls in ("CustomDistRV", "SymbolicRandomVariable")
            or "CustomDist" in cls
            or "Simulator" in cls
        ):
            return _badge("custom density — elided")
    except _Differ:
        return None, None, gd.GlyphNote("elements differ")
    except Exception:
        return _badge("elided")
    return None, None, None
