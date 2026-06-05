"""Canonical showcase models that exercise bayesdag's richer distribution glyphs.

Each is a small but *recognizable* Bayesian model (the kind people actually write), chosen so the
set collectively exercises the special-construct glyphs: random-walk fan charts, Weibull/censored
survival, zero-inflated composites, multivariate pairplots + LKJ correlation priors, ordinal
cutpoints, Gaussian-mixture overlays, spatial adjacency heatmaps, and the AR stationary marginal.

Importable from the gallery notebook (``examples/`` is on ``sys.path``) and re-exported to the test
suite via ``conftest``. Builders are pure (seeded) functions returning a ``pm.Model``.
"""

from __future__ import annotations

import numpy as np
import pymc as pm


def build_stochastic_volatility():
    """Finance: heavy-tailed returns with a Gaussian-random-walk log-volatility (StudentT, GRW, Exponential)."""
    rng = np.random.default_rng(0)
    T = 80
    returns = rng.standard_t(6, T) * 0.6
    with pm.Model(coords={"t": np.arange(T)}) as model:
        step = pm.Exponential("step_sd", 10.0)
        nu = pm.Exponential("nu", 0.1)
        h = pm.GaussianRandomWalk("log_vol", sigma=step, init_dist=pm.Normal.dist(0, 1), dims="t")
        pm.StudentT("returns", nu=nu, sigma=pm.math.exp(h / 2), observed=returns, dims="t")
    return model


def build_weibull_survival():
    """Biostatistics: right-censored time-to-event with a Weibull hazard (Weibull, Censored, Gamma)."""
    rng = np.random.default_rng(1)
    n, cens = 60, 8.0
    event = rng.weibull(1.5, n) * 5.0
    observed = np.minimum(event, cens)
    with pm.Model(coords={"subject": np.arange(n)}) as model:
        shape = pm.Gamma("shape", 2.0, 1.0)
        scale = pm.Gamma("scale", 2.0, 0.5)
        pm.Censored(
            "time",
            pm.Weibull.dist(shape, scale),
            lower=-np.inf,
            upper=cens,
            observed=observed,
            dims="subject",
        )
    return model


def build_zero_inflated_counts():
    """Ecology: overdispersed catch counts with excess zeros (ZeroInflatedPoisson, Beta)."""
    rng = np.random.default_rng(2)
    n = 120
    x = rng.normal(0, 1, n)
    y = rng.poisson(np.exp(0.5 + 0.3 * x)) * (rng.random(n) > 0.3)
    with pm.Model(coords={"obs": np.arange(n)}) as model:
        psi = pm.Beta("psi", 2.0, 2.0)
        b0 = pm.Normal("b0", 0, 1)
        # spike-and-slab (sparsity) prior on the slope — a latent Mixture → composite glyph
        b1 = pm.Mixture("b1", w=[0.8, 0.2], comp_dists=[pm.Normal.dist(0, 0.1), pm.Normal.dist(0, 2.0)])
        xx = pm.Data("x", x, dims="obs")
        lam = pm.Deterministic("lam", pm.math.exp(b0 + b1 * xx), dims="obs")
        pm.ZeroInflatedPoisson("y", psi=psi, mu=lam, observed=y, dims="obs")
    return model


def build_correlated_slopes():
    """Multilevel: per-group correlated intercept+slope (MvNormal + LKJCholeskyCov)."""
    rng = np.random.default_rng(3)
    ncafe, nvisit = 12, 8
    n = ncafe * nvisit
    cafe_idx = np.repeat(np.arange(ncafe), nvisit)
    afternoon = np.tile([0.0, 1.0] * (nvisit // 2), ncafe)
    y = rng.normal(3.0, 1.0, n)
    coords = {"cafe": np.arange(ncafe), "effect": ["intercept", "slope"], "obs": np.arange(n)}
    with pm.Model(coords=coords) as model:
        chol, _corr, _sds = pm.LKJCholeskyCov(
            "chol_cov", n=2, eta=2.0, sd_dist=pm.Exponential.dist(1.0), compute_corr=True
        )
        mu = pm.Normal("mu", 0, 5, dims="effect")
        ab = pm.MvNormal("ab", mu=mu, chol=chol, dims=("cafe", "effect"))
        sigma = pm.Exponential("sigma", 1.0)
        ci = pm.Data("cafe_idx", cafe_idx, dims="obs")
        aft = pm.Data("afternoon", afternoon, dims="obs")
        theta = pm.Deterministic("theta", ab[ci, 0] + ab[ci, 1] * aft, dims="obs")
        pm.Normal("y", theta, sigma, observed=y, dims="obs")
    return model


def build_ordinal_ratings():
    """Survey: ordinal responses on a latent scale cut by ordered cutpoints (OrderedLogistic)."""
    rng = np.random.default_rng(4)
    n, K = 150, 5
    x = rng.normal(0, 1, n)
    y = np.clip((x + rng.normal(0, 1, n) + 2).astype(int), 0, K - 1)
    from pymc.distributions.transforms import ordered

    with pm.Model(coords={"obs": np.arange(n), "cut": np.arange(K - 1)}) as model:
        beta = pm.Normal("beta", 0, 1)
        cutpoints = pm.Normal(
            "cutpoints", mu=np.linspace(-2, 2, K - 1), sigma=1, transform=ordered, dims="cut"
        )
        xx = pm.Data("x", x, dims="obs")
        eta = pm.Deterministic("eta", beta * xx, dims="obs")
        pm.OrderedLogistic("y", eta=eta, cutpoints=cutpoints, observed=y, dims="obs")
    return model


def build_gaussian_mixture():
    """Clustering: a two-component Gaussian mixture (NormalMixture, Dirichlet weights)."""
    rng = np.random.default_rng(5)
    n = 200
    y = np.concatenate([rng.normal(-3, 1, n // 2), rng.normal(3, 1, n // 2)])
    with pm.Model(coords={"comp": [0, 1], "obs": np.arange(n)}) as model:
        w = pm.Dirichlet("w", a=np.array([4.0, 2.0]), dims="comp")  # informative weights prior
        mu = pm.Normal("mu", 0, 5, dims="comp")
        sigma = pm.HalfNormal("sigma", 2, dims="comp")
        pm.NormalMixture("y", w=w, mu=mu, sigma=sigma, observed=y, dims="obs")
    return model


def build_disease_mapping():
    """Epidemiology: areal disease counts with an intrinsic spatial effect (ICAR, Poisson)."""
    rng = np.random.default_rng(6)
    R = 12
    W = np.zeros((R, R), dtype=int)
    for i in range(R):  # a ring of regions
        W[i, (i + 1) % R] = 1
        W[i, (i - 1) % R] = 1
    expected = rng.uniform(5.0, 20.0, R)
    y = rng.poisson(expected)
    with pm.Model(coords={"region": np.arange(R)}) as model:
        sd = pm.HalfNormal("spatial_sd", 1.0)
        phi = pm.ICAR("phi", W=W, dims="region")
        b0 = pm.Normal("b0", 0, 1)
        off = pm.Data("expected", expected, dims="region")
        mu = pm.Deterministic("mu", pm.math.exp(b0 + sd * phi) * off, dims="region")
        pm.Poisson("y", mu=mu, observed=y, dims="region")
    return model


def build_ar_forecast():
    """Econometrics: a latent second-order autoregressive trend behind noisy observations
    (state-space) — the latent `level` shows the AR **stationary-marginal** glyph."""
    rng = np.random.default_rng(7)
    T = 120
    trend = np.zeros(T)
    for t in range(2, T):
        trend[t] = 0.6 * trend[t - 1] + 0.2 * trend[t - 2] + rng.normal(0, 0.4)
    y = trend + rng.normal(0, 0.5, T)
    with pm.Model(coords={"t": np.arange(T)}) as model:
        sigma_obs = pm.HalfNormal("sigma_obs", 1.0)
        level = pm.AR(
            "level", rho=[0.6, 0.2], sigma=0.4, init_dist=pm.Normal.dist(0, 1), constant=False, dims="t"
        )
        pm.Normal("y", level, sigma_obs, observed=y, dims="t")
    return model


ZOO_MODELS = {
    "stochastic_volatility": build_stochastic_volatility,
    "weibull_survival": build_weibull_survival,
    "zero_inflated_counts": build_zero_inflated_counts,
    "correlated_slopes": build_correlated_slopes,
    "ordinal_ratings": build_ordinal_ratings,
    "gaussian_mixture": build_gaussian_mixture,
    "disease_mapping": build_disease_mapping,
    "ar_forecast": build_ar_forecast,
}
