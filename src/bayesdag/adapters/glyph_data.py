"""Distribution-shape DATA provider (the PyMC-specific half of the glyph system).

Produces a ``(GlyphSpec, data_dict)`` per node: the *actual parameterized* prior density
when the parameters are numerically resolvable, the family/schematic shape when they
depend on parents, an observed-data histogram for observed nodes, and (when an idata is
given) the posterior KDE. The PPL-agnostic ``glyph`` package renders the data to SVG.

The shape is the primary mark; a ``source`` tag records HOW it was obtained
(prior_analytic / prior_family_only / posterior_kde / observed_hist).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..ir import GlyphSpec

_GRID = 64


def _scalars(p) -> list:
    # representative scalar per param (iid vector priors broadcast params to arrays)
    return [float(np.asarray(x).reshape(-1)[0]) for x in (p or [])]


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
        "Triangular": lambda: st.triang(c=(q[1] - q[0]) / (q[2] - q[0]), loc=q[0], scale=q[2] - q[0]),
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
        "HyperGeometric": lambda: st.hypergeom(int(q[0] + q[1]), int(q[0]), int(q[2])),  # [good, bad, draws]
    }
    fn = builders.get(dist)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None


def _pmf(dist: str, p: list, max_cats: int = 40) -> Optional[dict]:
    """Analytic pmf for a discrete prior with numeric params -> ``{cats, heights}`` (bar glyph)."""
    try:
        if dist == "Categorical":  # the prob vector IS the pmf
            probs = np.asarray(p[0], float).ravel()
            cats, heights = list(range(probs.size)), probs
        else:
            fr = _discrete_frozen(dist, p)
            if fr is None:
                return None
            lo, hi = int(fr.ppf(0.001)), int(fr.ppf(0.999))
            hi = min(hi, lo + max_cats - 1)
            cats = list(range(lo, hi + 1))
            heights = np.asarray(fr.pmf(np.array(cats)), float)
        if heights.size == 0 or not np.all(np.isfinite(heights)) or heights.max() <= 0:
            return None
        m = float(heights.max()) or 1.0
        return {"cats": [int(c) for c in cats], "heights": [float(h / m) for h in heights]}
    except Exception:
        return None


def _custom_density(dist: str, p: list) -> Optional[dict]:
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


def _density_from_frozen(frozen) -> Optional[dict]:
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


def _density_from_samples(values) -> Optional[dict]:
    v = np.asarray(values, float).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2 or np.allclose(v, v[0]):
        return None
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(v)
        xs = np.linspace(v.min(), v.max(), _GRID)
        ys = kde(xs)
        m = float(np.max(ys)) or 1.0
        return {"xs": [float(x) for x in xs], "ys": [float(y / m) for y in ys]}
    except Exception:
        return None


def _histogram(values, max_bins: int = 30) -> Optional[dict]:
    v = np.asarray(values, float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    # numpy "auto" = max(Sturges, Freedman-Diaconis): a robust visual default that avoids
    # FD's too-few-bins behavior on small/lightly-tailed samples.
    edges = np.histogram_bin_edges(v, bins="auto")
    if len(edges) > max_bins + 1:
        edges = np.histogram_bin_edges(v, bins=max_bins)
    counts, edges = np.histogram(v, bins=edges)
    m = float(counts.max()) or 1.0
    return {"edges": [float(e) for e in edges], "counts": [float(c / m) for c in counts]}


# Discrete likelihoods: a continuous best-fit overlay is meaningless (a best-fit Bernoulli is
# degenerate), so observed nodes with these families fall back to the plain histogram.
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

# PyMC family name -> (scipy class, fixed-param kwargs for `.fit`). Support-constrained families
# are fit with the location pinned (floc=0) — and Beta on the unit interval — so the MLE respects
# the family's domain. Mirrors the continuous coverage of `_scipy_frozen` above.
def _fitters():
    import scipy.stats as st

    return {
        "Normal": (st.norm, {}),
        "StudentT": (st.t, {}),
        "Cauchy": (st.cauchy, {}),
        "Laplace": (st.laplace, {}),
        "Uniform": (st.uniform, {}),
        "Exponential": (st.expon, {"floc": 0.0}),
        "HalfNormal": (st.halfnorm, {"floc": 0.0}),
        "HalfCauchy": (st.halfcauchy, {"floc": 0.0}),
        "LogNormal": (st.lognorm, {"floc": 0.0}),
        "Gamma": (st.gamma, {"floc": 0.0}),
        "InverseGamma": (st.invgamma, {"floc": 0.0}),
        "Weibull": (st.weibull_min, {"floc": 0.0}),
        "Beta": (st.beta, {"floc": 0.0, "fscale": 1.0}),
        "Logistic": (st.logistic, {}),
        "Gumbel": (st.gumbel_r, {}),
        "Moyal": (st.moyal, {}),
        "SkewNormal": (st.skewnorm, {}),
        "Pareto": (st.pareto, {"floc": 0.0}),
        "ChiSquared": (st.chi2, {"floc": 0.0}),
    }


def _fit_frozen(dist: str, v):
    """MLE-fit the continuous family ``dist`` to data ``v``; return a scipy frozen dist or None.

    This is the *best the chosen family can do* on the data — it isolates whether the FAMILY is a
    reasonable shape, independent of the model's priors. Any non-convergence / domain error yields
    ``None`` (caller falls back to the plain histogram)."""
    try:
        fam = _fitters().get(dist)
        if fam is None:
            return None
        cls, kw = fam
        params = cls.fit(v, **kw)  # (shapes..., loc, scale)
        return cls(*params)
    except Exception:
        return None


def _fit_label(frozen, dist: str) -> str:
    """Short, family-agnostic summary of the fit for the card title (empty if no finite moments)."""
    try:
        mean, sd = float(frozen.mean()), float(frozen.std())
        if np.isfinite(mean) and np.isfinite(sd):
            return f"mean={mean:.3g}, sd={sd:.3g}"
    except Exception:
        pass
    return ""


def _observed_overlay(vals, dist: Optional[str], max_bins: int = 30) -> Optional[dict]:
    """Empirical density histogram + MLE best-fit family curve on ONE SHARED vertical scale.

    The histogram is area-normalized (``density=True``) and the best-fit pdf is sampled over the
    same x-span, then BOTH are divided by a single max so they are directly comparable (a naive
    overlay of two independently peak-normalized curves would not line up). Returns None when no
    continuous fit is possible (discrete family / unmapped / degenerate data) so the caller can
    fall back to the plain max-normalized histogram."""
    v = np.asarray(vals, float).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2 or np.allclose(v, v[0]):
        return None
    if not dist or dist in _DISCRETE:
        return None
    frozen = _fit_frozen(dist, v)
    if frozen is None:
        return None
    edges = np.histogram_bin_edges(v, bins="auto")
    if len(edges) > max_bins + 1:
        edges = np.histogram_bin_edges(v, bins=max_bins)
    dens, edges = np.histogram(v, bins=edges, density=True)
    xs = np.linspace(float(edges[0]), float(edges[-1]), _GRID)
    try:
        ys = np.asarray(frozen.pdf(xs), float)
    except Exception:
        return None
    if not np.all(np.isfinite(ys)):
        return None
    m = max(float(dens.max()) if dens.size else 0.0, float(ys.max()) if ys.size else 0.0) or 1.0
    return {
        "edges": [float(e) for e in edges],
        "counts": [float(d / m) for d in dens],
        "overlay": {"xs": [float(x) for x in xs], "ys": [float(y / m) for y in ys]},
        "fit": {"family": dist, "n": int(v.size), "params": _fit_label(frozen, dist)},
    }


def _discrete_bars(values, max_cats: int = 30) -> Optional[dict]:
    """Observed proportions for an integer/categorical likelihood (Bernoulli/Binomial/Poisson/...)
    as ONE bar per class — the honest representation of a pmf. Avoids the continuous auto-histogram,
    which scatters binary data into edge-pinned bins with empty gaps between. Returns None when the
    value range is too wide to be categorical (caller falls back to a continuous histogram)."""
    v = np.asarray(values, float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    vi = np.rint(v).astype(int)
    lo, hi = int(vi.min()), int(vi.max())
    cats = list(range(lo, hi + 1))  # include empty interior classes so a pmf's gaps stay visible
    if len(cats) > max_cats:
        return None
    counts = np.array([float(np.count_nonzero(vi == c)) for c in cats])
    m = float(counts.max()) or 1.0
    return {"cats": cats, "heights": [float(c / m) for c in counts], "n": int(v.size)}


def _numeric_params(var) -> Optional[list]:
    node = var.owner
    op = node.op
    try:
        dparams = list(op.dist_params(node))
    except Exception:
        dparams = list(node.inputs[2:])
    out = []
    for dp in dparams:
        try:
            out.append(np.asarray(dp.eval()))
        except Exception:
            return None
    return out


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


def _posterior_samples(name: str, idata):
    if idata is None:
        return None
    try:
        if name in getattr(idata, "posterior", {}):
            return np.asarray(idata.posterior[name].values)
    except Exception:
        return None
    return None


def glyph_for(var, role: str, dist: Optional[str], model, idata=None) -> tuple[Optional[GlyphSpec], Optional[dict]]:
    # Posterior overlay wins when available (fitted result).
    samples = _posterior_samples(getattr(var, "name", ""), idata)
    if samples is not None:
        data = _density_from_samples(samples)
        if data is not None:
            return GlyphSpec(kind="density", source="posterior_kde"), data

    if role == "observed":
        vals = _observed_values(var, model)
        if vals is None:
            return None, None
        # discrete likelihood -> per-class proportion bars (Bernoulli/Binomial/Poisson/...)
        if dist and dist in _DISCRETE:
            bars = _discrete_bars(vals)
            if bars is not None:
                return GlyphSpec(kind="bars", source="observed_hist"), bars
        # continuous: best-fit family curve overlaid on the data histogram; else plain histogram
        overlay = _observed_overlay(vals, dist)
        if overlay is not None:
            return GlyphSpec(kind="hist_overlay", source="observed_hist"), overlay
        data = _histogram(vals)
        if data is not None:
            return GlyphSpec(kind="histogram", source="observed_hist"), data
        return None, None

    if role == "latent" and dist:
        params = _numeric_params(var)
        if params is not None:
            # continuous analytic pdf
            frozen = _scipy_frozen(dist, params)
            data = _density_from_frozen(frozen) if frozen is not None else None
            if data is not None:
                return GlyphSpec(kind="density", source="prior_analytic"), data
            # discrete analytic pmf -> bars
            data = _pmf(dist, params)
            if data is not None:
                return GlyphSpec(kind="bars", source="prior_analytic"), data
            # closed-form pdf for families with no scipy frozen (Kumaraswamy/LogitNormal/HalfStudentT)
            data = _custom_density(dist, params)
            if data is not None:
                return GlyphSpec(kind="density", source="prior_analytic"), data
        # params depend on parents (hierarchical) or unmapped dist -> family/schematic shape
        return GlyphSpec(kind="schematic", source="prior_family_only"), None

    return None, None  # deterministic / data / potential -> no shape glyph in M0
