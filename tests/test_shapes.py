"""Phase-2 univariate shape coverage: the prior glyph a node ships must match the model's actual
density/pmf (validated against `pm.logp`), across continuous (scipy), custom-pdf, and discrete
families. This locks the PyMC->scipy parameter translations (incl. the Exponential/Gamma/HalfCauchy
fixes) and the closed-form pdfs (Kumaraswamy/LogitNormal/HalfStudentT).

The op REPARAMETRIZES several families, so a translation that looks right against the public
``dist()`` kwargs can still be wrong (Exponential/Gamma expose *scale*; HalfCauchy a single
param). `pm.draw` is not a safe oracle either — it samples Cauchy as loc=a/b, scale=1/b — so the
density via ``logp`` is the truth.

`test_every_translation_is_locked` closes the loop: it reads the builder tables directly, so
adding a family to `glyph_data` without a case here FAILS rather than shipping unverified.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from bayesdag.adapters import glyph_data as gd
from bayesdag.adapters.glyph_data import glyph_for
from bayesdag.adapters.pymc import _rv_dist_and_params

CONTINUOUS = {
    "Normal": lambda: pm.Normal("v", 2, 3),
    "HalfNormal": lambda: pm.HalfNormal("v", 3),
    "Exponential": lambda: pm.Exponential("v", 2),  # op exposes scale, not rate (regression)
    "Gamma": lambda: pm.Gamma("v", 2, 5),  # op exposes [alpha, scale] (regression)
    "InverseGamma": lambda: pm.InverseGamma("v", alpha=4, beta=3),
    "Cauchy": lambda: pm.Cauchy("v", 1, 2),
    "HalfCauchy": lambda: pm.HalfCauchy("v", 3),  # single-param op (regression)
    "Laplace": lambda: pm.Laplace("v", 1, 2),
    "AsymmetricLaplace": lambda: pm.AsymmetricLaplace("v", kappa=2.0, mu=1.0, b=3.0),
    "Logistic": lambda: pm.Logistic("v", 1, 2),
    "Gumbel": lambda: pm.Gumbel("v", 1, 2),
    "Moyal": lambda: pm.Moyal("v", 1, 2),
    "StudentT": lambda: pm.StudentT("v", nu=5, mu=1, sigma=2),
    "SkewNormal": lambda: pm.SkewNormal("v", mu=2, sigma=3, alpha=4),
    "ExGaussian": lambda: pm.ExGaussian("v", mu=1.0, sigma=2.0, nu=3.0),
    "VonMises": lambda: pm.VonMises("v", 1, 2),
    "LogNormal": lambda: pm.LogNormal("v", 0.5, 1.2),
    "Weibull": lambda: pm.Weibull("v", 2, 3),
    "Pareto": lambda: pm.Pareto("v", 3, 2),
    "Rice": lambda: pm.Rice("v", nu=2.0, sigma=3.0),
    "Wald": lambda: pm.Wald("v", mu=2, lam=3),
    "Beta": lambda: pm.Beta("v", 2, 5),
    "Uniform": lambda: pm.Uniform("v", -1, 4),
    "Triangular": lambda: pm.Triangular("v", lower=0, c=3, upper=10),
    "ZeroSumNormal": lambda: pm.ZeroSumNormal("v", sigma=2.0, shape=4),
    "Kumaraswamy": lambda: pm.Kumaraswamy("v", 2, 3),  # closed-form custom pdf
    "LogitNormal": lambda: pm.LogitNormal("v", 0, 1),  # closed-form custom pdf
    "HalfStudentT": lambda: pm.HalfStudentT("v", nu=4, sigma=2),  # folded-t custom pdf
}
DISCRETE = {
    "Poisson": lambda: pm.Poisson("v", 3),
    "Bernoulli": lambda: pm.Bernoulli("v", 0.3),
    "Binomial": lambda: pm.Binomial("v", n=10, p=0.4),
    "Geometric": lambda: pm.Geometric("v", 0.3),
    "NegativeBinomial": lambda: pm.NegativeBinomial("v", mu=4, alpha=2),
    "DiscreteUniform": lambda: pm.DiscreteUniform("v", lower=0, upper=8),
    "BetaBinomial": lambda: pm.BetaBinomial("v", alpha=2, beta=3, n=10),
    "HyperGeometric": lambda: pm.HyperGeometric("v", N=20, k=7, n=5),
}

# ZeroSumNormal's glyph is the MARGINAL of one element, which is deliberately not the joint
# logp (the distribution is degenerate: its density lives on the sum-to-zero hyperplane).
_MARGINAL_ONLY = {"ZeroSumNormal"}


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
    if name in _MARGINAL_ONLY:
        assert ys.max() > 0 and np.all(np.isfinite(ys))
        return
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


def test_zero_sum_normal_marginal_is_the_documented_shape():
    """ZeroSumNormal is degenerate (mass on the sum-to-zero hyperplane), so there is no 1-D
    density to match. The glyph shows one element's MARGINAL, N(0, sigma*sqrt((n-1)/n))."""
    import scipy.stats as st

    _v, _spec, data = _glyph(CONTINUOUS["ZeroSumNormal"])
    sigma, n = 2.0, 4
    ref = st.norm(0.0, sigma * np.sqrt((n - 1) / n))
    xs = np.array(data["xs"])
    truth = ref.pdf(xs)
    got = np.array(data["ys"])
    assert np.max(np.abs(got / got.max() - truth / truth.max())) < 0.02


def test_every_translation_is_locked():
    """Read the builder tables and demand a logp-locked case for each entry.

    The translations are the project's highest-risk data: they are hand-derived against a
    reparametrizing op, and a wrong one renders a confident, wrong density. Adding a family to
    `_scipy_frozen`/`_discrete_frozen` without a case above must fail here rather than ship."""
    continuous = _builder_keys(gd._scipy_frozen)
    discrete = _builder_keys(gd._discrete_frozen)
    assert len(continuous) >= 20 and len(discrete) >= 8  # the reader found the real tables

    missing_c = sorted(continuous - set(CONTINUOUS))
    missing_d = sorted(discrete - set(DISCRETE))
    assert not missing_c, f"scipy translations with no logp-matching test: {missing_c}"
    assert not missing_d, f"discrete translations with no logp-matching test: {missing_d}"

    # and the closed-form pdfs, which bypass scipy entirely
    assert {"Kumaraswamy", "LogitNormal", "HalfStudentT"} <= set(CONTINUOUS)


def _builder_keys(fn) -> set:
    """The family names a ``_*_frozen`` builder table knows, read out of the function's own AST
    so this test cannot drift from the implementation it is guarding."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return {
        k.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for k in node.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
