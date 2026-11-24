"""Distribution-shape DATA provider (the PyMC-specific half of the glyph system).

Produces a ``(GlyphSpec, data_dict)`` per node: the *actual parameterized* prior density
when the parameters are numerically resolvable, the family/schematic shape when they
depend on parents, an observed-data histogram for observed nodes, and (when an idata is
given) the posterior KDE. The PPL-agnostic ``glyph`` package renders the data to SVG.

The shape is the primary mark; a ``source`` tag records HOW it was obtained
(prior_analytic / prior_family_only / posterior_kde / observed_hist).
"""

from __future__ import annotations

import numpy as np

from ..ir import GlyphSpec
from ._ptutil import GlyphNote, uniform

_GRID = 64
_MAX_STAT_POINTS = 50_000


def _thin(v: np.ndarray) -> np.ndarray:
    """Deterministic strided subsample for O(n)-sensitive stats (KDE / MLE fit) — an MLE on
    50k points is visually identical to 1M but ~100x faster. Never RNG (glyphs must be
    byte-identical across renders); histograms stay full-data."""
    if v.size <= _MAX_STAT_POINTS:
        return v
    return v[:: int(np.ceil(v.size / _MAX_STAT_POINTS))]


def _scalars(p) -> list:
    # One representative scalar per param. Only valid when the params are elementwise-uniform
    # (an iid vector prior broadcasts a scalar) — callers gate on `_uniform_params` first.
    return [float(np.asarray(x).reshape(-1)[0]) for x in (p or [])]


def _uniform_params(p) -> bool:
    """True when every param array holds a single value (or a constant broadcast of one), so one
    scalar per slot describes the WHOLE node — see ``_ptutil.uniform``."""
    return uniform(p)


# Standardized fill per ``dist_params`` slot, in op order. A number is the canonical value of a
# LOCATION or SCALE slot (0 / 1): a parent there only moves or stretches the curve, so the shape is
# still exact. ``S`` marks a SHAPE slot (a df, an alpha, a log-scale sigma, …): if a parent governs
# it, no single curve is honest. Families not listed have no proven location-scale structure, and
# every discrete family's parameters change its shape.
S = None
_STD_FILL: dict[str, list] = {
    "Normal": [0.0, 1.0],
    "HalfNormal": [0.0, 1.0],
    "Cauchy": [0.0, 1.0],
    "HalfCauchy": [1.0],
    "Laplace": [0.0, 1.0],
    "Logistic": [0.0, 1.0],
    "Gumbel": [0.0, 1.0],
    "Moyal": [0.0, 1.0],
    "ZeroSumNormal": [1.0, S],  # [sigma, n] — n is structural and always numeric
    "Uniform": [0.0, 1.0],
    "Exponential": [1.0],  # op exposes the scale
    "AsymmetricLaplace": [1.0, S, 0.0],  # [b (scale), kappa, mu]
    "StudentT": [S, 0.0, 1.0],
    "HalfStudentT": [S, 1.0],
    "SkewNormal": [0.0, 1.0, S],
    "ExGaussian": [0.0, S, S],  # the shape is nu/sigma: both must be known
    "VonMises": [0.0, S],
    "Gamma": [S, 1.0],
    "InverseGamma": [S, 1.0],
    "Weibull": [S, 1.0],
    "Pareto": [S, 1.0],
    "Rice": [S, 1.0],  # [b = nu/sigma, sigma]
    "Wald": [S, S, 0.0],  # the shape is mu/lam; alpha shifts
    "LogNormal": [0.0, S],  # exp(mu) is a scale; sigma is the shape
}
_S = S
del S


def _conditional_shape(var, dist: str | None, vals: list | None = None):
    """The glyph for a latent with no single set of numbers: some slot is governed by a parent,
    or (``vals`` given) its numeric slots vary across elements.

    Exact when every such slot is a pure location/scale slot of its family and every shape slot
    is a known, elementwise-uniform number: then the standardized curve IS the node's shape up to
    where it sits and how wide it is. Otherwise no curve — and a caption naming the slots whose
    parents decide the shape. Never a generic stand-in bell."""
    from . import _ptutil

    if vals is None:
        vals = _ptutil.slot_values(var)
    names = _slot_names(var, dist, len(vals))
    fill = _STD_FILL.get(dist or "")
    if fill is None or len(fill) != len(vals):
        governed = [n for n, v in zip(names, vals, strict=False) if v is None and n]
        return None, None, GlyphNote(_shape_note(governed))
    shape_unknown = [
        names[i] for i, (f, v) in enumerate(zip(fill, vals, strict=True)) if f is _S and v is None
    ]
    if shape_unknown:
        return None, None, GlyphNote(_shape_note(shape_unknown))
    shape_vals = [v for f, v in zip(fill, vals, strict=True) if f is _S]
    if not uniform(shape_vals):
        return None, None, GlyphNote("elements differ in shape")
    data = _conditional_shape_from_values(dist, vals)
    if data is None:
        return None, None, GlyphNote(_shape_note([]))
    return GlyphSpec(kind="schematic", source="conditional_standardized"), data, None


def _conditional_shape_from_values(dist: str, vals: list) -> dict | None:
    """The standardized curve of ``dist`` given per-slot values (None = set by a parent). Shape
    slots must be known; location/scale slots take their canonical 0/1. None when unknowable."""
    fill = _STD_FILL.get(dist or "")
    if fill is None or len(fill) != len(vals):
        return None
    if any(f is _S and v is None for f, v in zip(fill, vals, strict=True)):
        return None
    params = [
        (np.asarray(v).reshape(-1)[0] if f is _S else f) for f, v in zip(fill, vals, strict=True)
    ]
    frozen = _scipy_frozen(dist, params)
    data = _density_from_frozen(frozen) if frozen is not None else _custom_density(dist, params)
    if data is None:
        return None
    xs = np.asarray(data["xs"], float)
    span = float(xs[-1] - xs[0]) or 1.0
    # the axis carries no values (location/scale are the parents'), so it is rescaled to [0, 1]
    # and the zero marker stays off
    data["xs"] = [float((x - xs[0]) / span) for x in xs]
    data.update({"family": dist, "exact": True})
    return data


def _slot_names(var, dist, n) -> list[str]:
    try:
        from .pymc import _resolved_param_names

        return [x or "" for x in _resolved_param_names(dist, var.owner.op, n)]
    except Exception:
        return [""] * n


def _shape_note(slots: list[str]) -> str:
    if not slots:
        return "no fixed curve: its shape depends on parents"
    if len(slots) == 1:
        return f"no fixed curve: {slots[0]} comes from a parent"
    return f"no fixed curve: {', '.join(slots)} come from parents"


def _scipy_frozen(dist: str, p: list):
    """A scipy frozen continuous distribution for a prior whose params are numeric, in the OP's
    parameter order (verified against PyMC 6.x via logp). Returns None for discrete / custom /
    unmapped families. NB: the op already reparametrizes several families (Exponential/Gamma expose
    SCALE, HalfCauchy a single scale) — these translations match the model's density, not the
    public ``dist()`` kwargs."""
    import scipy.stats as st

    q = _scalars(p)
    builders = {
        # location-scale & friends
        "Normal": lambda: st.norm(q[0], q[1]),
        "HalfNormal": lambda: st.halfnorm(q[0], q[1]),
        "Cauchy": lambda: st.cauchy(q[0], q[1]),
        "HalfCauchy": lambda: st.halfcauchy(scale=q[0]),
        "Laplace": lambda: st.laplace(q[0], q[1]),
        "AsymmetricLaplace": lambda: st.laplace_asymmetric(q[1], loc=q[2], scale=1.0 / q[0]),
        "Logistic": lambda: st.logistic(q[0], q[1]),
        "Gumbel": lambda: st.gumbel_r(q[0], q[1]),
        "Moyal": lambda: st.moyal(q[0], q[1]),
        "StudentT": lambda: st.t(df=q[0], loc=q[1], scale=q[2]),
        # ZeroSumNormal: each element is marginally Normal(0, sigma*sqrt((n-1)/n)) — centered at 0
        "ZeroSumNormal": lambda: st.norm(
            0.0, q[0] * np.sqrt(max(q[1] - 1, 1) / q[1]) if len(q) >= 2 else q[0]
        ),
        "SkewNormal": lambda: st.skewnorm(a=q[2], loc=q[0], scale=q[1]),
        "ExGaussian": lambda: st.exponnorm(K=q[2] / q[1], loc=q[0], scale=q[1]),
        "VonMises": lambda: st.vonmises(kappa=q[1], loc=q[0]),
        # positive support
        "Exponential": lambda: st.expon(scale=q[0]),  # op exposes scale, not rate
        "Gamma": lambda: st.gamma(q[0], scale=q[1]),  # op exposes [alpha, scale]
        "InverseGamma": lambda: st.invgamma(q[0], scale=q[1]),
        "LogNormal": lambda: st.lognorm(s=q[1], scale=np.exp(q[0])),
        "Weibull": lambda: st.weibull_min(c=q[0], scale=q[1]),
        "Pareto": lambda: st.pareto(b=q[0], scale=q[1]),
        "Rice": lambda: st.rice(q[0], scale=q[1]),  # op exposes [b=nu/sigma, sigma]
        "Wald": lambda: st.invgauss(mu=q[0] / q[1], loc=q[2], scale=q[1]),
        # bounded
        "Beta": lambda: st.beta(q[0], q[1]),
        "Uniform": lambda: st.uniform(q[0], q[1] - q[0]),
        "Triangular": lambda: st.triang(
            c=(q[1] - q[0]) / (q[2] - q[0]), loc=q[0], scale=q[2] - q[0]
        ),
    }
    fn = builders.get(dist)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None


def _discrete_frozen(dist: str, p: list):
    """A scipy frozen discrete distribution (verified op-order param translations)."""
    import scipy.stats as st

    q = _scalars(p)
    builders = {
        "Poisson": lambda: st.poisson(q[0]),
        "Bernoulli": lambda: st.bernoulli(q[0]),
        "Binomial": lambda: st.binom(int(q[0]), q[1]),
        "Geometric": lambda: st.geom(q[0]),
        "NegativeBinomial": lambda: st.nbinom(q[0], q[1]),  # op exposes [n, p]
        "DiscreteUniform": lambda: st.randint(int(q[0]), int(q[1]) + 1),
        "BetaBinomial": lambda: st.betabinom(int(q[0]), q[1], q[2]),  # op [n, a, b]
        "HyperGeometric": lambda: st.hypergeom(
            int(q[0] + q[1]), int(q[0]), int(q[2])
        ),  # [good, bad, draws]
    }
    fn = builders.get(dist)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None


def _pmf(dist: str, p: list, max_cats: int = 40) -> dict | None:
    """Analytic pmf for a discrete prior with numeric params -> ``{cats, heights}`` (bar glyph)."""
    try:
        if dist == "Categorical":  # the prob vector IS the pmf
            probs = np.asarray(p[0], float)
            if probs.ndim > 1:  # a batch of categoricals: one picture only if all are identical
                rows = probs.reshape(-1, probs.shape[-1])
                if not np.all(rows == rows[0]):
                    return None
                probs = rows[0]
            probs = probs.ravel()
            cats, heights = list(range(probs.size)), probs
        else:
            fr = _discrete_frozen(dist, p)
            if fr is None:
                return None
            lo, hi = int(fr.ppf(0.001)), int(fr.ppf(0.999))
            bins = _integer_bins(lo, hi, max_cats)
            cats = [a for a, _ in bins]
            heights = np.array(
                [
                    fr.pmf(a)
                    if a == b
                    else (
                        fr.cdf(b) - fr.cdf(a - 1) if fr.cdf(b) <= 0.5 else fr.sf(a - 1) - fr.sf(b)
                    )
                    for a, b in bins
                ]
            )
            tails = {"left": float(fr.cdf(lo - 1)), "right": float(fr.sf(hi))}
        if heights.size == 0 or not np.all(np.isfinite(heights)) or heights.max() <= 0:
            return None
        m = float(heights.max()) or 1.0
        result = {"cats": [int(c) for c in cats], "heights": [float(h / m) for h in heights]}
        if dist != "Categorical":
            result.update(bins=bins, masses=heights.tolist(), tails=tails)
            if any(a != b for a, b in bins):
                result["binned"] = True
        return result
    except Exception:
        return None


def _custom_density(dist: str, p: list) -> dict | None:
    """Analytic density for continuous families with no direct scipy frozen (closed-form pdf)."""
    import scipy.stats as st

    q = _scalars(p)
    try:
        if dist == "Kumaraswamy":
            a, b = q[0], q[1]
            xs = np.linspace(1e-3, 1 - 1e-3, _GRID)
            ys = a * b * xs ** (a - 1) * (1 - xs**a) ** (b - 1)
        elif dist == "LogitNormal":
            mu, sg = q[0], q[1]
            xs = np.linspace(1e-3, 1 - 1e-3, _GRID)
            lg = np.log(xs / (1 - xs))
            ys = np.exp(-((lg - mu) ** 2) / (2 * sg**2)) / (sg * np.sqrt(2 * np.pi) * xs * (1 - xs))
        elif dist == "HalfStudentT":  # folded Student-t: 2 * t.pdf on [0, inf)
            nu, sg = q[0], q[1]
            base = st.t(df=nu, scale=sg)
            xs = np.linspace(0.0, base.ppf(0.995), _GRID)
            ys = 2.0 * base.pdf(xs)
        else:
            return None
        ys = np.asarray(ys, float)
        if not np.all(np.isfinite(ys)) or ys.max() <= 0:
            return None
        m = float(np.max(ys)) or 1.0
        return {"xs": [float(x) for x in xs], "ys": [float(y / m) for y in ys]}
    except Exception:
        return None


def _density_from_frozen(frozen) -> dict | None:
    try:
        lo, hi = frozen.ppf(0.005), frozen.ppf(0.995)
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return None
        xs = np.linspace(lo, hi, _GRID)
        ys = frozen.pdf(xs)
        m = float(np.max(ys)) or 1.0
        return {"xs": [float(x) for x in xs], "ys": [float(y / m) for y in ys]}
    except Exception:
        return None


def _density_from_samples(values) -> dict | None:
    v = np.asarray(values, float).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2 or np.allclose(v, v[0]):
        return None
    v = _thin(v)
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(v)
        xs = np.linspace(v.min(), v.max(), _GRID)
        ys = kde(xs)
        m = float(np.max(ys)) or 1.0
        return {"xs": [float(x) for x in xs], "ys": [float(y / m) for y in ys]}
    except Exception:
        return None


def _binned(v, max_bins: int):
    """numpy "auto" = max(Sturges, Freedman-Diaconis): a robust visual default that avoids
    FD's too-few-bins behavior on small/lightly-tailed samples."""
    edges = np.histogram_bin_edges(v, bins="auto")
    if len(edges) > max_bins + 1:
        edges = np.histogram_bin_edges(v, bins=max_bins)
    return edges


def _histogram(values, max_bins: int = 30) -> dict | None:
    try:
        v = np.asarray(values, float).ravel()
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        counts, edges = np.histogram(v, bins=_binned(v, max_bins))
        m = float(counts.max()) or 1.0
        return {"edges": [float(e) for e in edges], "counts": [float(c / m) for c in counts]}
    except Exception:
        return None


# Families whose values are integers: their data and draws are drawn as per-value frequencies,
# never as a KDE (which would put mass between integers).
_DISCRETE = {
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


def _integer_bins(lo: int, hi: int, max_bins: int) -> list[tuple[int, int]]:
    """Partition an inclusive integer span without allocating for every integer."""
    if hi < lo or max_bins < 1:
        raise ValueError("invalid integer range or bin limit")
    width = hi - lo + 1
    step = (width + max_bins - 1) // max_bins
    return [(start, min(start + step - 1, hi)) for start in range(lo, hi + 1, step)]


def _bin_frequencies(values: np.ndarray, bins: list[tuple[int, int]]) -> np.ndarray:
    ends = np.asarray([b for _, b in bins])
    return np.bincount(np.searchsorted(ends, values), minlength=len(bins)) / values.size


def _discrete_bars(values, max_cats: int = 30) -> dict | None:
    """Per-value frequencies for integer data or draws — one bar per value (empty interior values
    kept, so a pmf's gaps stay visible). A range wider than ``max_cats`` is grouped into at most
    ``max_cats`` integer-aligned bins and marked ``binned``. Non-integer values are not silently
    rounded: they return None."""
    try:
        v = np.asarray(values, float).ravel()
        if v.size == 0 or not np.all(np.isfinite(v)) or not np.all(v == np.round(v)):
            return None
        bins = _integer_bins(int(v.min()), int(v.max()), max_cats)
        frequencies = _bin_frequencies(v, bins)
        m = float(frequencies.max())
        result = {
            "cats": [a for a, _ in bins],
            "bins": bins,
            "frequencies": frequencies.tolist(),
            "heights": (frequencies / m).tolist(),
            "n": int(v.size),
        }
        if any(a != b for a, b in bins):
            result["binned"] = True
        return result
    except Exception:
        return None


def _numeric_params(var) -> list | None:
    """The node's distribution parameters as numpy arrays — but ONLY for a root prior, i.e. one
    whose every parameter is a fixed constant. None for a conditional latent (any parameter
    governed by a parent RV): there ``.eval()`` would silently draw a random sample."""
    from . import _ptutil

    return _ptutil.numeric_params(var)


def _observed_values(var, model):
    try:
        return np.asarray(model.rvs_to_values[var].eval())
    except Exception:
        obs = getattr(getattr(var, "tag", None), "observations", None)
        if obs is not None:
            try:
                return np.asarray(obs.eval())
            except Exception:
                return None
    return None


def _element_count(samples) -> int:
    """How many elements a posterior array carries: arrays are ``(chain, draw, *element dims)``,
    so everything past the first two axes is a distinct element of the variable."""
    try:
        shape = np.asarray(samples).shape
        return int(np.prod(shape[2:])) if len(shape) > 2 else 1
    except Exception:
        return 1


def _group_draws(idata, group: str, name: str):
    """``idata[group][name]`` as a DataArray transposed to ``(chain, draw, ...)``, or None."""
    if idata is None:
        return None
    try:
        ds = getattr(idata, group, None)
        if ds is None or name not in ds:
            return None
        da = ds[name]
        if "chain" in da.dims and "draw" in da.dims:
            return da.transpose("chain", "draw", ...)
        return None
    except Exception:
        return None


def _posterior_samples(name: str, idata):
    da = _group_draws(idata, "posterior", name)
    return None if da is None else np.asarray(da.values)


def element_labels(da) -> list[str]:
    """One readable label per element of a ``(chain, draw, *dims)`` DataArray, in storage
    order: ``S3`` for one dim, ``(P1, b2)`` for several (tuple labels for multi-dim coords)."""
    import itertools

    dims = [d for d in da.dims if d not in ("chain", "draw")]
    if not dims:
        return []
    axes = [
        [str(c) for c in (da.coords[d].values if d in da.coords else range(da.sizes[d]))]
        for d in dims
    ]
    if len(axes) == 1:
        return axes[0]
    return ["(" + ", ".join(t) + ")" for t in itertools.product(*axes)]


MULTI_MAX = 8  # up to this many elements: one overlaid curve each
FOREST_MAX_ROWS = 400  # beyond this, a forest shows evenly spaced ranks (and says so)


def _multi(flat: np.ndarray) -> dict | None:
    """Per-element KDEs on one shared grid, jointly peak-normalized (so heights compare)."""
    finite = flat[np.isfinite(flat)]
    if finite.size < 2:
        return None
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if not hi > lo:
        return None
    xs = np.linspace(lo, hi, _GRID)
    from scipy.stats import gaussian_kde

    curves = []
    for j in range(flat.shape[1]):
        col = flat[:, j]
        col = _thin(col[np.isfinite(col)])
        if col.size < 2 or np.allclose(col, col[0]):
            return None
        curves.append(gaussian_kde(col)(xs))
    m = max(float(c.max()) for c in curves) or 1.0
    return {
        "xs": [float(x) for x in xs],
        "curves": [[float(y / m) for y in c] for c in curves],
        "n": int(flat.shape[1]),
    }


def _forest(flat: np.ndarray) -> dict | None:
    """Per-element 5/25/50/75/95% quantiles, sorted by median (an unlabelled shape summary of the
    whole vector — the labelled, coordinate-ordered version lives in the card)."""
    with np.errstate(all="ignore"):
        q = np.nanpercentile(flat, [5, 25, 50, 75, 95], axis=0)  # (5, k)
    if not np.all(np.isfinite(q)):
        return None
    order = np.argsort(q[2], kind="stable")
    k = q.shape[1]
    shown = order
    if k > FOREST_MAX_ROWS:  # evenly spaced ranks keep the payload bounded
        shown = order[np.unique(np.linspace(0, k - 1, FOREST_MAX_ROWS).round().astype(int))]
    lo, hi = float(q[0].min()), float(q[4].max())
    if not hi > lo:
        return None
    out = {f"q{p:02d}": [float(q[i, j]) for j in shown] for i, p in enumerate((5, 25, 50, 75, 95))}
    out.update({"lo": lo, "hi": hi, "n": int(k), "shown": len(shown)})
    return out


def posterior_glyph(name: str, dist: str | None, idata) -> tuple[GlyphSpec, dict] | None:
    """The posterior glyph for a variable, or None if this idata has no draws for it.

    Deliberately takes only a NAME — no ``pm.Model``, no PyTensor variable — because the node id
    is by construction the constrained idata variable name (invariant 6).

    A vector is never pooled into one curve (that curve would be no element's posterior — the
    shrinkage picture a hierarchy is about would vanish): up to ``MULTI_MAX`` elements get one
    curve each, more get a forest of per-element intervals."""
    da = _group_draws(idata, "posterior", name)
    if da is None:
        return None
    samples = np.asarray(da.values)
    k = _element_count(samples)
    flat = samples.reshape(samples.shape[0] * samples.shape[1], -1).astype(float)
    discrete = bool(dist and dist in _DISCRETE)
    if k == 1:
        if discrete:
            # a discrete posterior is a pmf over integers: a KDE would smear mass between them
            bars = _discrete_bars(samples)
            return (GlyphSpec(kind="bars", source="posterior_bars"), bars) if bars else None
        data = _density_from_samples(samples)
        return (GlyphSpec(kind="density", source="posterior_kde"), data) if data else None
    if k <= MULTI_MAX and not discrete:
        data = _multi(flat)
        if data is not None:
            return GlyphSpec(kind="multi", source="posterior_kde"), data
    data = _forest(flat)
    if data is None:
        return None
    return GlyphSpec(kind="forest", source="posterior_quantiles"), data


def marginal_rows(name: str, idata, page: int = 0, per_page: int = 24) -> dict | None:
    """Labelled per-element posterior intervals in ORIGINAL coordinate order (never ranked), one
    page at a time: ``{rows: [{label, q05, q25, q50, q75, q95}], page, pages, n, lo, hi}``. The
    axis is shared across ALL pages, so rows on different pages compare."""
    da = _group_draws(idata, "posterior", name)
    if da is None:
        return None
    samples = np.asarray(da.values, float)
    flat = samples.reshape(samples.shape[0] * samples.shape[1], -1)
    with np.errstate(all="ignore"):
        q = np.nanpercentile(flat, [5, 25, 50, 75, 95], axis=0)
    if not np.all(np.isfinite(q)):
        return None
    labels = element_labels(da) or [name]
    k = q.shape[1]
    pages = max(1, int(np.ceil(k / per_page)))
    page = max(0, min(int(page), pages - 1))
    idx = range(page * per_page, min(k, (page + 1) * per_page))
    rows = [
        {"label": labels[j] if j < len(labels) else str(j)}
        | {f"q{p:02d}": float(q[i, j]) for i, p in enumerate((5, 25, 50, 75, 95))}
        for j in idx
    ]
    return {
        "rows": rows,
        "page": page,
        "pages": pages,
        "n": int(k),
        "lo": float(q[0].min()),
        "hi": float(q[4].max()),
    }


def predictive_glyph(name: str, dist: str | None, idata) -> tuple[GlyphSpec, dict] | None:
    """A predictive check for an observed node from draws the caller supplied: the observed data
    against ``posterior_predictive`` (preferred) or ``prior_predictive``, on one axis. The data
    are matched element-for-element with the replicates; any mismatch yields None."""
    obs_da = None
    try:
        ds = getattr(idata, "observed_data", None)
        if ds is not None and name in ds:
            obs_da = ds[name]
    except Exception:
        obs_da = None
    if obs_da is None:
        return None
    for group, source in (
        ("posterior_predictive", "posterior_predictive"),
        ("prior_predictive", "prior_predictive"),
    ):
        pred = _group_draws(idata, group, name)
        if pred is None:
            continue
        pv = np.asarray(pred.values, float)
        ov = np.asarray(obs_da.values, float)
        if pv.shape[2:] != ov.shape:
            return None
        flat = pv.reshape(pv.shape[0] * pv.shape[1], -1)  # (S, N)
        o = ov.ravel()
        data = _ppc_discrete(o, flat) if dist and dist in _DISCRETE else _ppc_continuous(o, flat)
        if data is None:
            return None
        data["group"] = group
        return GlyphSpec(kind="ppc", source=source), data
    return None


_PPC_DRAWS = 20


def _ppc_continuous(o: np.ndarray, flat: np.ndarray) -> dict | None:
    from scipy.stats import gaussian_kde

    o = o[np.isfinite(o)]
    allv = flat[np.isfinite(flat)]
    if o.size < 2 or allv.size < 2:
        return None
    lo = float(min(np.percentile(allv, 0.5), o.min()))
    hi = float(max(np.percentile(allv, 99.5), o.max()))
    if not hi > lo:
        return None
    xs = np.linspace(lo, hi, _GRID)
    try:
        dens, edges = np.histogram(o, bins=_binned(o, 30), range=None, density=True)
        pooled = gaussian_kde(_thin(allv))(xs)
        idx = np.unique(
            np.linspace(0, flat.shape[0] - 1, min(_PPC_DRAWS, flat.shape[0])).astype(int)
        )
        draws = []
        for i in idx:
            row = flat[i][np.isfinite(flat[i])]
            if row.size >= 2 and not np.allclose(row, row[0]):
                draws.append(gaussian_kde(row)(xs))
    except Exception:
        return None
    m = max([float(dens.max()), float(pooled.max())] + [float(d.max()) for d in draws]) or 1.0
    return {
        "edges": [float(e) for e in edges],
        "counts": [float(c / m) for c in dens],
        "xs": [float(x) for x in xs],
        "ys": [float(y / m) for y in pooled],
        "draws": [[float(y / m) for y in d] for d in draws],
        "n": int(o.size),
    }


def _ppc_discrete(o: np.ndarray, flat: np.ndarray) -> dict | None:
    o = np.asarray(o, float).ravel()
    flat = np.asarray(flat, float)
    if (
        not o.size
        or flat.ndim != 2
        or not flat.size
        or not np.all(np.isfinite(o))
        or not np.all(np.isfinite(flat))
        or not np.all(o == np.round(o))
        or not np.all(flat == np.round(flat))
    ):
        return None
    bins = _integer_bins(int(min(o.min(), flat.min())), int(max(o.max(), flat.max())), 40)
    obs = _bin_frequencies(o, bins)
    per_draw = np.stack([_bin_frequencies(row, bins) for row in flat])
    med = np.median(per_draw, axis=0)
    q05, q95 = np.percentile(per_draw, [5, 95], axis=0)
    m = float(max(obs.max(), q95.max())) or 1.0
    return {
        "cats": [a for a, _ in bins],
        "bins": bins,
        "binned": any(a != b for a, b in bins),
        "frequencies": obs.tolist(),
        "pred_frequencies": per_draw.tolist(),
        "tails": {"left": 0.0, "right": 0.0},
        "heights": [float(h / m) for h in obs],
        "pred_mid": [float(h / m) for h in med],
        "pred_lo": [float(h / m) for h in q05],
        "pred_hi": [float(h / m) for h in q95],
        "n": int(o.size),
    }


def glyph_for(
    var, role: str, dist: str | None, model, idata=None, named: dict | None = None
) -> tuple[GlyphSpec | None, dict | None, str | None]:
    """Return ``(GlyphSpec, glyph_data, elision_reason)``. ``elision_reason`` is non-None only for
    honestly-undrawable constructs (it sets ``NodeIR.representable = False`` and shows a badge).
    ``named`` is the ``id(var)->name`` leaf map (used by the deterministic transfer-function path)."""
    # Deterministics depict their TRANSFER FUNCTION (a structural, parameter-free shape), NOT a
    # posterior — the function shape is the point; falls to equation-only when not provably depictable.
    if role == "deterministic":
        if getattr(var, "owner", None) is None:
            return None, None, None
        from .deterministic import transform_glyph  # lazy: deterministic <-> glyph_data cycle

        return transform_glyph(var, named)

    # Posterior overlay wins when available (fitted result).
    post = posterior_glyph(getattr(var, "name", ""), dist, idata)
    if post is not None:
        return post[0], post[1], None

    if role == "observed":
        vals = _observed_values(var, model)
        if vals is None:
            return None, None, None
        # discrete likelihood -> per-class proportion bars (Bernoulli/Binomial/Poisson/...)
        if dist and dist in _DISCRETE:
            bars = _discrete_bars(vals)
            if bars is not None:
                return GlyphSpec(kind="bars", source="observed_hist"), bars, None
        # continuous: the data histogram. No fitted curve: a family fitted to the MARGINAL of
        # regression data says nothing about the likelihood (the marginal of N(mu_i, sigma) is a
        # mixture). The model's own predictions belong here instead — see predictive_glyph.
        data = _histogram(vals)
        if data is not None:
            return GlyphSpec(kind="histogram", source="observed_hist"), data, None
        return None, None, None

    if role == "latent" and dist:
        # special constructs (multivariate / matrix / mixture / time-series / bounded / spatial /
        # meta) get a faithful structural glyph or an honest badge before the univariate path.
        from .constructs import special_glyph

        spec, data, elision = special_glyph(var)
        if spec is not None or elision is not None:
            return spec, data, elision

        params = _numeric_params(var)
        if dist == "Categorical" and params is not None:
            probs = np.asarray(params[0])
            if probs.ndim > 1 and not np.all(
                probs.reshape(-1, probs.shape[-1]) == probs.reshape(-1, probs.shape[-1])[0]
            ):
                return None, None, GlyphNote("elements differ")
        # A varying vector prior gives each element its OWN density, so no single curve is the
        # node's prior — those fall through to the schematic. Categorical is exempt: its vector
        # param IS the pmf, not a per-element parameterization.
        if params is not None and (dist == "Categorical" or _uniform_params(params)):
            # continuous analytic pdf
            frozen = _scipy_frozen(dist, params)
            data = _density_from_frozen(frozen) if frozen is not None else None
            if data is not None:
                return GlyphSpec(kind="density", source="prior_analytic"), data, None
            # discrete analytic pmf -> bars
            data = _pmf(dist, params)
            if data is not None:
                return GlyphSpec(kind="bars", source="prior_analytic"), data, None
            # closed-form pdf for families with no scipy frozen (Kumaraswamy/LogitNormal/HalfStudentT)
            data = _custom_density(dist, params)
            if data is not None:
                return GlyphSpec(kind="density", source="prior_analytic"), data, None
        if params is not None:
            # every slot is numeric, yet no single analytic curve applies: a varying vector
            # prior gives each element its own density
            if not _uniform_params(params):
                # elements differ in location/scale only -> the standardized shape is still
                # exact for every element; differing SHAPE -> no single curve
                return _conditional_shape(var, dist, list(params))
            if dist == "DiracDelta" and params:
                v = float(np.asarray(params[0]).reshape(-1)[0])
                return None, None, GlyphNote(f"a point mass at {v:.4g}")
            return None, None, GlyphNote(f"no analytic {dist} shape is drawn")
        # a parent governs some slot: the family's exact standardized shape when location/scale
        # are all that depend on parents, else no curve and a caption — never a sampled value
        return _conditional_shape(var, dist)

    return None, None, None  # data / potential (deterministics dispatch above) -> no shape glyph
