"""Phase-2 univariate shape coverage: the prior glyph a node ships must match the model's actual
density/pmf (validated against `pm.logp`), across continuous (scipy), custom-pdf, and discrete
families. This locks the PyMC->scipy parameter translations (incl. the Exponential/Gamma/HalfCauchy
fixes) and the closed-form pdfs (Kumaraswamy/LogitNormal/HalfStudentT)."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from bayesdag.adapters.glyph_data import glyph_for
from bayesdag.adapters.pymc import _rv_dist_and_params

CONTINUOUS = {
    "Normal": lambda: pm.Normal("v", 2, 3),
    "Exponential": lambda: pm.Exponential("v", 2),  # op exposes scale, not rate (regression)
    "Gamma": lambda: pm.Gamma("v", 2, 5),  # op exposes [alpha, scale] (regression)
    "HalfCauchy": lambda: pm.HalfCauchy("v", 3),  # single-param op (regression)
    "Weibull": lambda: pm.Weibull("v", 2, 3),
    "Wald": lambda: pm.Wald("v", mu=2, lam=3),
    "Triangular": lambda: pm.Triangular("v", lower=0, c=3, upper=10),
    "VonMises": lambda: pm.VonMises("v", 1, 2),
    "Pareto": lambda: pm.Pareto("v", 3, 2),
    "SkewNormal": lambda: pm.SkewNormal("v", mu=2, sigma=3, alpha=4),
    "Kumaraswamy": lambda: pm.Kumaraswamy("v", 2, 3),  # closed-form custom pdf
    "LogitNormal": lambda: pm.LogitNormal("v", 0, 1),  # closed-form custom pdf
    "HalfStudentT": lambda: pm.HalfStudentT("v", nu=4, sigma=2),  # folded-t custom pdf
}
DISCRETE = {
    "Poisson": lambda: pm.Poisson("v", 3),
    "NegativeBinomial": lambda: pm.NegativeBinomial("v", mu=4, alpha=2),
    "BetaBinomial": lambda: pm.BetaBinomial("v", alpha=2, beta=3, n=10),
    "HyperGeometric": lambda: pm.HyperGeometric("v", N=20, k=7, n=5),
}


def _glyph(build):
    with pm.Model() as m:
        v = build()
    dist, _ = _rv_dist_and_params(v, {})
    spec, data, _elision = glyph_for(v, "latent", dist, m)
    return v, spec, data


def _model_pdf(v, xs):
    x = pt.vector("x")
    return np.exp(pm.logp(v, x).eval({x: np.asarray(xs, float)}))


@pytest.mark.parametrize("name", list(CONTINUOUS))
def test_continuous_prior_glyph_matches_logp(name):
    v, spec, data = _glyph(CONTINUOUS[name])
    assert spec.kind == "density" and spec.source == "prior_analytic"
    xs, ys = np.array(data["xs"]), np.array(data["ys"])
    truth = _model_pdf(v, xs)
    good = np.isfinite(truth) & (truth > 0)
    truth = truth[good] / truth[good].max()
    got = ys[good] / ys[good].max()
    assert np.max(np.abs(got - truth)) < 0.02, name


@pytest.mark.parametrize("name", list(DISCRETE))
def test_discrete_prior_glyph_matches_logp(name):
    v, spec, data = _glyph(DISCRETE[name])
    assert spec.kind == "bars" and spec.source == "prior_analytic"
    cats = np.array(data["cats"])
    truth = _model_pdf(v, cats)
    truth = truth / truth.max()
    got = np.array(data["heights"]) / max(data["heights"])
    assert np.max(np.abs(got - truth)) < 0.02, name
