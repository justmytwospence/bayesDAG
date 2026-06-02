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


def _scipy_frozen(dist: str, p: list):
    import scipy.stats as st

    def f(x):  # representative scalar (iid vector priors broadcast params to arrays)
        return float(np.asarray(x).reshape(-1)[0])

    try:
        if dist == "Normal":
            return st.norm(loc=f(p[0]), scale=f(p[1]))
        if dist == "HalfNormal":
            return st.halfnorm(loc=f(p[0]), scale=f(p[1]))
        if dist == "Uniform":
            return st.uniform(loc=f(p[0]), scale=f(p[1]) - f(p[0]))
        if dist == "Exponential":
            return st.expon(scale=1.0 / f(p[0]))  # pymc param is rate
        if dist == "Beta":
            return st.beta(f(p[0]), f(p[1]))
        if dist == "Gamma":
            return st.gamma(a=f(p[0]), scale=1.0 / f(p[1]))  # pymc Gamma(alpha, rate)
        if dist == "InverseGamma":
            return st.invgamma(a=f(p[0]), scale=f(p[1]))
        if dist == "StudentT":
            return st.t(df=f(p[0]), loc=f(p[1]), scale=f(p[2]))
        if dist == "Cauchy":
            return st.cauchy(loc=f(p[0]), scale=f(p[1]))
        if dist == "HalfCauchy":
            return st.halfcauchy(loc=f(p[0]), scale=f(p[1]))
        if dist == "Laplace":
            return st.laplace(loc=f(p[0]), scale=f(p[1]))
        if dist == "LogNormal":
            return st.lognorm(s=f(p[1]), scale=np.exp(f(p[0])))
    except Exception:
        return None
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
        data = _histogram(vals) if vals is not None else None
        if data is not None:
            return GlyphSpec(kind="histogram", source="observed_hist"), data
        return None, None

    if role == "latent" and dist:
        params = _numeric_params(var)
        if params is not None:
            frozen = _scipy_frozen(dist, params)
            data = _density_from_frozen(frozen) if frozen is not None else None
            if data is not None:
                return GlyphSpec(kind="density", source="prior_analytic"), data
        # params depend on parents (hierarchical) or unmapped dist -> family/schematic shape
        return GlyphSpec(kind="schematic", source="prior_family_only"), None

    return None, None  # deterministic / data / potential -> no shape glyph in M0
