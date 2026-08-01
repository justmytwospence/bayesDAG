"""Full-catalog coverage: every PyMC distribution must convert + lay out + render WITHOUT crashing,
and resolve to a real symbol (named families) or an honest badge. This is the denominator guarantee
behind the distribution-support work — if PyMC adds/changes a family, this surfaces it.

The guarantee is enforced by `test_catalog_covers_every_pymc_family`, which diffs CATALOG against
`pm.distributions.__all__` in both directions. Every family is instantiated for real, including the
ones needing a callable (EulerMaruyama's sde_fn, Simulator's fn, CustomDist's dist) — nothing is
skipped, so there is no silent cap to log."""

import numpy as np
import pymc as pm
import pytest

from bayesdag.convert import to_ir
from bayesdag.labels import dist_symbol
from bayesdag.layout import layout
from bayesdag.render_svg import to_svg

_N = pm.Normal.dist
_W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

# Minimal valid constructor per distribution (name -> builder). Grouped as in the catalog.
CATALOG = {
    # continuous univariate
    "Normal": lambda: pm.Normal("x", 0, 1),
    "HalfNormal": lambda: pm.HalfNormal("x", 1),
    "Beta": lambda: pm.Beta("x", 2, 2),
    "Kumaraswamy": lambda: pm.Kumaraswamy("x", 2, 2),
    "Cauchy": lambda: pm.Cauchy("x", 0, 1),
    "HalfCauchy": lambda: pm.HalfCauchy("x", 1),
    "ChiSquared": lambda: pm.ChiSquared("x", 3),
    "ExGaussian": lambda: pm.ExGaussian("x", mu=0, sigma=1, nu=1),
    "Exponential": lambda: pm.Exponential("x", 1),
    "Gamma": lambda: pm.Gamma("x", 2, 2),
    "InverseGamma": lambda: pm.InverseGamma("x", 3, 2),
    "Gumbel": lambda: pm.Gumbel("x", 0, 1),
    "Laplace": lambda: pm.Laplace("x", 0, 1),
    "AsymmetricLaplace": lambda: pm.AsymmetricLaplace("x", b=1, kappa=1, mu=0),
    "Logistic": lambda: pm.Logistic("x", 0, 1),
    "LogitNormal": lambda: pm.LogitNormal("x", 0, 1),
    "LogNormal": lambda: pm.LogNormal("x", 0, 1),
    "Moyal": lambda: pm.Moyal("x", 0, 1),
    "Pareto": lambda: pm.Pareto("x", 2, 1),
    "Rice": lambda: pm.Rice("x", nu=1, sigma=1),
    "SkewNormal": lambda: pm.SkewNormal("x", mu=0, sigma=1, alpha=2),
    "SkewStudentT": lambda: pm.SkewStudentT("x", a=2, b=2, mu=0, sigma=1),
    "StudentT": lambda: pm.StudentT("x", nu=4, mu=0, sigma=1),
    "HalfStudentT": lambda: pm.HalfStudentT("x", nu=4, sigma=1),
    "Triangular": lambda: pm.Triangular("x", lower=0, c=1, upper=2),
    "Uniform": lambda: pm.Uniform("x", 0, 1),
    "VonMises": lambda: pm.VonMises("x", 0, 1),
    "Wald": lambda: pm.Wald("x", mu=1, lam=1),
    "Weibull": lambda: pm.Weibull("x", 2, 1),
    "PolyaGamma": lambda: pm.PolyaGamma("x", 1, 0),
    # discrete univariate
    "Bernoulli": lambda: pm.Bernoulli("x", 0.5),
    "BetaBinomial": lambda: pm.BetaBinomial("x", alpha=2, beta=2, n=10),
    "Binomial": lambda: pm.Binomial("x", n=10, p=0.5),
    "Categorical": lambda: pm.Categorical("x", [0.2, 0.3, 0.5]),
    "DiracDelta": lambda: pm.DiracDelta("x", 3),
    "DiscreteUniform": lambda: pm.DiscreteUniform("x", 0, 5),
    "DiscreteWeibull": lambda: pm.DiscreteWeibull("x", q=0.9, beta=1),
    "Geometric": lambda: pm.Geometric("x", 0.3),
    "HyperGeometric": lambda: pm.HyperGeometric("x", N=20, k=10, n=5),
    "NegativeBinomial": lambda: pm.NegativeBinomial("x", mu=3, alpha=2),
    "Poisson": lambda: pm.Poisson("x", 3),
    "OrderedLogistic": lambda: pm.OrderedLogistic(
        "x", eta=0.0, cutpoints=np.array([-1.0, 0.0, 1.0])
    ),
    "OrderedProbit": lambda: pm.OrderedProbit("x", eta=0.0, cutpoints=np.array([-1.0, 0.0, 1.0])),
    # multivariate / matrix / simplex
    "MvNormal": lambda: pm.MvNormal("x", mu=[0, 0], cov=np.eye(2)),
    "MvStudentT": lambda: pm.MvStudentT("x", nu=3, mu=[0, 0], scale=np.eye(2)),
    "Wishart": lambda: pm.Wishart("x", nu=3, V=np.eye(2)),
    "MatrixNormal": lambda: pm.MatrixNormal(
        "x", mu=np.zeros((2, 2)), rowcov=np.eye(2), colcov=np.eye(2)
    ),
    "KroneckerNormal": lambda: pm.KroneckerNormal("x", mu=np.zeros(4), covs=[np.eye(2), np.eye(2)]),
    "Dirichlet": lambda: pm.Dirichlet("x", [1.0, 2.0, 3.0]),
    "DirichletMultinomial": lambda: pm.DirichletMultinomial("x", n=10, a=[1.0, 2.0, 3.0]),
    "LKJCorr": lambda: pm.LKJCorr("x", n=3, eta=2),
    "StickBreakingWeights": lambda: pm.StickBreakingWeights("x", alpha=2, K=4),
    # mixtures / inflated
    "Mixture": lambda: pm.Mixture("x", w=[0.5, 0.5], comp_dists=[_N(-2, 1), _N(2, 1)]),
    "NormalMixture": lambda: pm.NormalMixture("x", w=[0.5, 0.5], mu=[-2, 2], sigma=[1, 1]),
    "ZeroInflatedPoisson": lambda: pm.ZeroInflatedPoisson("x", psi=0.7, mu=3),
    "ZeroInflatedBinomial": lambda: pm.ZeroInflatedBinomial("x", psi=0.7, n=10, p=0.5),
    "ZeroInflatedNegativeBinomial": lambda: pm.ZeroInflatedNegativeBinomial(
        "x", psi=0.7, mu=3, alpha=2
    ),
    "HurdlePoisson": lambda: pm.HurdlePoisson("x", psi=0.7, mu=3),
    "HurdleGamma": lambda: pm.HurdleGamma("x", psi=0.7, alpha=2, beta=2),
    "HurdleLogNormal": lambda: pm.HurdleLogNormal("x", psi=0.7, mu=0, sigma=1),
    "HurdleNegativeBinomial": lambda: pm.HurdleNegativeBinomial("x", psi=0.7, mu=3, alpha=2),
    # bounded
    "Censored": lambda: pm.Censored("x", _N(0, 1), lower=-1, upper=1),
    "Truncated": lambda: pm.Truncated("x", pm.Gamma.dist(2, 2), lower=0, upper=5),
    "TruncatedNormal": lambda: pm.TruncatedNormal("x", mu=0, sigma=1, lower=-1, upper=1),
    # time series
    "AR": lambda: pm.AR("x", rho=[0.5], sigma=1, init_dist=_N(0, 1), steps=5),
    "GARCH11": lambda: pm.GARCH11("x", omega=0.1, alpha_1=0.1, beta_1=0.8, initial_vol=1, steps=5),
    "GaussianRandomWalk": lambda: pm.GaussianRandomWalk("x", sigma=1, init_dist=_N(0, 1), steps=5),
    "RandomWalk": lambda: pm.RandomWalk("x", init_dist=_N(0, 1), innovation_dist=_N(0, 1), steps=5),
    # spatial
    "CAR": lambda: pm.CAR("x", mu=np.zeros(3), W=_W, alpha=0.9, tau=1),
    "ICAR": lambda: pm.ICAR("x", W=_W),
    # meta / custom
    "Interpolated": lambda: pm.Interpolated(
        "x", x_points=np.linspace(-3, 3, 30), pdf_points=np.exp(-(np.linspace(-3, 3, 30) ** 2))
    ),
    "Flat": lambda: pm.Flat("x"),
    "HalfFlat": lambda: pm.HalfFlat("x"),
    "CustomDist": lambda: pm.CustomDist(
        "x", 0.0, dist=lambda mu, size: pm.Normal.dist(mu, 1.0, size=size)
    ),
    "DensityDist": lambda: pm.DensityDist(
        "x", 0.0, dist=lambda mu, size: pm.Normal.dist(mu, 1.0, size=size)
    ),
    "Simulator": lambda: pm.Simulator(
        "x", lambda rng, m, size: rng.normal(m, 1.0, size), 0.0, observed=np.zeros(3)
    ),
    "ZeroSumNormal": lambda: pm.ZeroSumNormal("x", sigma=1.0, shape=3),
    "Multinomial": lambda: pm.Multinomial("x", n=5, p=[0.3, 0.3, 0.4]),
    "OrderedMultinomial": lambda: pm.OrderedMultinomial(
        "x", eta=0.0, cutpoints=np.array([-1.0, 1.0]), n=5
    ),
    "LKJCholeskyCov": lambda: pm.LKJCholeskyCov(
        "x", n=3, eta=2.0, sd_dist=pm.Exponential.dist(1.0)
    ),
    "WishartBartlett": lambda: pm.WishartBartlett("x", S=np.eye(3), nu=4),
    "MvGaussianRandomWalk": lambda: pm.MvGaussianRandomWalk(
        "x",
        mu=np.zeros(2),
        cov=np.eye(2),
        init_dist=pm.MvNormal.dist(np.zeros(2), np.eye(2)),
        steps=4,
    ),
    "MvStudentTRandomWalk": lambda: pm.MvStudentTRandomWalk(
        "x",
        nu=4,
        mu=np.zeros(2),
        scale=np.eye(2),
        init_dist=pm.MvNormal.dist(np.zeros(2), np.eye(2)),
        steps=4,
    ),
    "EulerMaruyama": lambda: pm.EulerMaruyama(
        "x", dt=0.1, sde_fn=lambda v, a: (-a * v, 1.0), sde_pars=(1.0,), init_dist=_N(0, 1), steps=4
    ),
}

# Not distribution families: base classes and one alias PyMC re-exports from `__all__`.
_NOT_FAMILIES = {"Continuous", "Discrete", "Distribution", "SymbolicRandomVariable", "Lognormal"}

# Every named family must resolve to a real symbol rather than the \operatorname{} fallback —
# EXCEPT these, whose "symbol" is deliberately an upright word: they have no conventional
# notation, and inventing one would imply a standard meaning they don't have.
_FALLBACK_OK = {"CustomDist", "DensityDist", "Simulator"}


def _node(build):
    with pm.Model() as m:
        build()
    return to_ir(m), m


@pytest.mark.parametrize("name", list(CATALOG))
def test_every_distribution_renders(name):
    """Convert + lay out + render the full SVG; the node must be handled (a glyph or an honest
    badge), never left as an unhandled crash."""
    ir, _m = _node(CATALOG[name])
    n = ir.node("x")
    assert n is not None, name
    # handled = has a shape glyph OR an honest elision badge
    assert (n.glyph is not None) or (n.elision_reason is not None), name
    assert "<svg" in to_svg(ir, layout(ir)), name


@pytest.mark.parametrize("name", list(CATALOG))
def test_every_distribution_has_a_symbol(name):
    ir, _m = _node(CATALOG[name])
    sym = dist_symbol(ir.node("x").dist)
    if name in _FALLBACK_OK:
        # no conventional notation exists; assert we still print the FAMILY, not the op's
        # variable-mangled internal name ("CustomDist_x")
        assert "operatorname" in sym and "_x" not in sym, f"{name}: mangled symbol ({sym})"
        return
    assert "operatorname" not in sym, f"{name}: no symbol ({sym})"


def test_catalog_covers_every_pymc_family():
    """The actual denominator check behind COVERAGE.md's "every distribution in PyMC renders".

    A hand-maintained list plus a size floor could never notice a NEW family — and didn't: 11
    were missing, several of them ticked off as done. Diffing against `pm.distributions.__all__`
    means a PyMC release that adds a family fails here instead of silently going unrendered."""
    published = set(pm.distributions.__all__) - _NOT_FAMILIES
    missing = sorted(published - set(CATALOG))
    assert not missing, f"PyMC families absent from CATALOG: {missing}"

    # the reverse direction: a catalog entry PyMC no longer publishes is stale
    stale = sorted(set(CATALOG) - published)
    assert not stale, f"CATALOG entries not in pm.distributions.__all__: {stale}"


def test_data_and_minibatch_containers_render():
    """`pm.Data` / `pm.Minibatch` are ticked off in COVERAGE.md but Minibatch appeared in no
    test. A minibatched likelihood carries no named data node (the batch tensor is anonymous),
    so what has to hold is that it converts and renders rather than crashing on the
    unnameable observed input."""
    rng = np.random.default_rng(0)
    obs = rng.normal(size=100)

    with pm.Model() as data_model:
        d = pm.Data("d", obs)
        pm.Normal("y", pm.Normal("mu", 0, 1), 1.0, observed=d)
    ir = to_ir(data_model)
    assert ir.node("d") is not None and ir.node("d").role == "data"
    assert "<svg" in to_svg(ir, layout(ir))

    with pm.Model() as mb_model:
        batch = pm.Minibatch(obs, batch_size=10)
        pm.Normal("y", pm.Normal("mu", 0, 1), 1.0, observed=batch, total_size=100)
    ir = to_ir(mb_model)
    assert {n.id for n in ir.nodes} == {"mu", "y"}
    assert ir.node("y").role == "observed"
    assert "<svg" in to_svg(ir, layout(ir))


def test_showcase_models_render():
    """The canonical gallery models (examples/zoo.py) all convert + render."""
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "examples"))
    from zoo import ZOO_MODELS

    assert len(ZOO_MODELS) == 16
    for name, build in ZOO_MODELS.items():
        ir = to_ir(build())
        assert "<svg" in to_svg(ir, layout(ir)), name
