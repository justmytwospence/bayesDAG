"""Canonical showcase models that exercise bayesdag's richer distribution glyphs.

Each is a small but *recognizable* Bayesian model (the kind people actually write), chosen so the
set collectively exercises the special-construct glyphs: random-walk fan charts, Weibull/censored
survival, zero-inflated composites, multivariate pairplots + LKJ correlation priors, ordinal
cutpoints, Gaussian-mixture overlays, spatial adjacency heatmaps, the AR stationary marginal, BART
broken out into its sum-of-trees model, and the deterministic **transfer-function** glyphs (a logistic
S-curve, a probit S-curve, a log-link exponential, a softplus positive ramp, a tanh saturating curve,
a quadratic power curve, a softmax simplex, and affine-predictor lines).

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


def build_logistic_regression():
    """Classification: a logistic regression with the linear predictor and inverse-link made explicit —
    `eta` shows the **line** transfer glyph, `p` the **logistic S-curve** (invlogit)."""
    rng = np.random.default_rng(8)
    n = 100
    x = rng.normal(0, 1, n)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(0.4 + 1.5 * x))))
    with pm.Model(coords={"obs": np.arange(n)}) as model:
        a = pm.Normal("a", 0, 1)
        b = pm.Normal("b", 0, 1)
        eta = pm.Deterministic("eta", a + b * x, dims="obs")
        p = pm.Deterministic("p", pm.math.invlogit(eta), dims="obs")
        pm.Bernoulli("y", p=p, observed=y, dims="obs")
    return model


def build_poisson_loglink():
    """Counts: a Poisson GLM with an explicit log link — `eta` is the **line**, `rate` the
    **exponential** transfer glyph (exp)."""
    rng = np.random.default_rng(9)
    n = 100
    x = rng.normal(0, 1, n)
    y = rng.poisson(np.exp(0.5 + 0.7 * x))
    with pm.Model(coords={"obs": np.arange(n)}) as model:
        a = pm.Normal("a", 0, 1)
        b = pm.Normal("b", 0, 1)
        eta = pm.Deterministic("eta", a + b * x, dims="obs")
        rate = pm.Deterministic("rate", pm.math.exp(eta), dims="obs")
        pm.Poisson("y", mu=rate, observed=y, dims="obs")
    return model


def build_softmax_categorical():
    """Multinomial choice: a softmax (multinomial-logit) classifier — `p` shows the **simplex**
    transfer glyph over the K categories."""
    rng = np.random.default_rng(10)
    n, K = 120, 3
    x = rng.normal(0, 1, n)
    beta = np.array([-1.0, 0.0, 1.0])
    logits = np.outer(x, beta)
    p_true = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(K, p=row) for row in p_true])
    with pm.Model(coords={"obs": np.arange(n), "cat": np.arange(K)}) as model:
        a = pm.Normal("a", 0, 1, dims="cat")
        b = pm.Normal("b", 0, 1, dims="cat")
        eta = pm.Deterministic("eta", a + b * x[:, None], dims=("obs", "cat"))
        p = pm.Deterministic("p", pm.math.softmax(eta, axis=-1), dims=("obs", "cat"))
        pm.Categorical("y", p=p, observed=y, dims="obs")
    return model


def build_probit_regression():
    """Classification: a **probit** binary regression — `p = Phi(eta)` written as the standard
    `0.5*(1 + erf(eta/sqrt(2)))`. `p` shows the **probit** S-curve, which the glyph draws from the
    true Gaussian CDF so it is visibly distinct (steeper shoulders) from the logistic S-curve."""
    import pytensor.tensor as pt
    from scipy.stats import norm

    rng = np.random.default_rng(11)
    n = 100
    x = rng.normal(0, 1, n)
    y = rng.binomial(1, norm.cdf(0.3 + 1.2 * x))
    with pm.Model(coords={"obs": np.arange(n)}) as model:
        a = pm.Normal("a", 0, 1)
        b = pm.Normal("b", 0, 1)
        eta = pm.Deterministic("eta", a + b * x, dims="obs")
        p = pm.Deterministic("p", 0.5 * (1.0 + pt.erf(eta / np.sqrt(2.0))), dims="obs")
        pm.Bernoulli("y", p=p, observed=y, dims="obs")
    return model


def build_heteroskedastic_softplus():
    """Non-constant variance: the noise scale itself grows with a predictor, mapped through
    **softplus** to stay positive — `sigma = softplus(a + b*x)`. `sigma` shows the **softplus**
    smooth-positive-ramp transfer glyph (a deterministic that is a scale, not a mean)."""
    rng = np.random.default_rng(12)
    n = 120
    x = rng.normal(0, 1, n)
    sd = np.log1p(np.exp(0.2 + 0.8 * x))
    y = rng.normal(1.0 + 0.5 * x, sd)
    with pm.Model(coords={"obs": np.arange(n)}) as model:
        m0 = pm.Normal("m0", 0, 1)
        m1 = pm.Normal("m1", 0, 1)
        a = pm.Normal("a", 0, 1)
        b = pm.Normal("b", 0, 1)
        mu = pm.Deterministic("mu", m0 + m1 * x, dims="obs")
        eta_s = pm.Deterministic("eta_s", a + b * x, dims="obs")
        sigma = pm.Deterministic("sigma", pm.math.softplus(eta_s), dims="obs")
        pm.Normal("y", mu=mu, sigma=sigma, observed=y, dims="obs")
    return model


def build_saturating_tanh():
    """A bounded/saturating dose-response: the effect tapers to +/-1 through **tanh** —
    `effect = tanh(a + b*x)`. `effect` shows the **tanh** S-curve on [-1, 1] (a different bounded
    shape from the logistic [0, 1])."""
    rng = np.random.default_rng(13)
    n = 100
    x = rng.normal(0, 1.5, n)
    y = rng.normal(np.tanh(0.2 + 1.3 * x), 0.3)
    with pm.Model(coords={"obs": np.arange(n)}) as model:
        a = pm.Normal("a", 0, 1)
        b = pm.Normal("b", 0, 1)
        eta = pm.Deterministic("eta", a + b * x, dims="obs")
        effect = pm.Deterministic("effect", pm.math.tanh(eta), dims="obs")
        pm.Normal("y", mu=effect, sigma=0.3, observed=y, dims="obs")
    return model


def build_quadratic_power():
    """A squared-amplitude mean (signal power ~ amplitude^2): `power = amp**2` with a constant
    exponent. `power` shows the **pow** transfer glyph drawn as the *actual* parabola x^2 (the glyph
    reads the constant exponent from the op graph), not a generic curve."""
    rng = np.random.default_rng(14)
    n = 100
    x = rng.normal(0, 1, n)
    amp_true = 0.5 + 0.9 * x
    y = rng.normal(amp_true**2, 0.4)
    with pm.Model(coords={"obs": np.arange(n)}) as model:
        a = pm.Normal("a", 0, 1)
        b = pm.Normal("b", 0, 1)
        amp = pm.Deterministic("amp", a + b * x, dims="obs")
        power = pm.Deterministic("power", amp**2, dims="obs")
        pm.Normal("y", mu=power, sigma=0.4, observed=y, dims="obs")
    return model


def build_bart_sum_of_trees():
    """**BART broken out into the model it actually is, with every parameter on display.** Rather than
    one opaque `BART(...)` node (bayesdag supports that too, as a step-function glyph), this surfaces
    BART's full parameterization. The tree-structure prior is explicit: a node at depth `d` splits with
    probability `p_split = alpha*(1+depth)^(-beta)` (the `alpha`, `beta` ~2-5, and `depth`
    hyperparameters). Leaf values are shrunk toward 0 by `sigma_mu = 0.5/(k*sqrt(m))` (the `k`, `m`
    hyperparameters). The `tree` plate then holds each tree's parameters — split indicator
    `split ~ Bernoulli(alpha)`, split point `c ~ U`, leaf values `mu_L, mu_R ~ Normal(0, sigma_mu)` —
    and the tree's function is the (gated) split decision `g = (split & x<=c ? mu_L : mu_R)`. The
    regression function is the sum over trees `f = sum(g)`, closed by `y ~ Normal(f, sigma)`. The
    per-tree computation is shown as a depth-1 stump for legibility; the depth prior is what lets real
    BART grow deeper trees."""
    rng = np.random.default_rng(15)
    m, n = 30, 60
    xv = np.linspace(0, 10, n)
    Y = np.sin(xv) + 0.1 * xv + rng.normal(0, 0.3, n)
    with pm.Model(coords={"tree": range(m), "level": range(4), "obs": range(n)}) as model:
        x = pm.Data("x", xv, dims="obs")
        alpha = pm.Data("alpha", 0.95)               # base split probability
        beta = pm.Data("beta", 2.0)                  # depth penalty (typically 2-5)
        k = pm.Data("k", 2.0)                        # leaf-value scale factor
        m_trees = pm.Data("m", float(m))             # number of trees
        depth = pm.Data("depth", np.arange(4), dims="level")
        pm.Deterministic("p_split", alpha * (1.0 + depth) ** (-beta), dims="level")  # P(split | depth); shown, terminal
        sigma_mu = pm.Deterministic("sigma_mu", 0.5 / (k * pm.math.sqrt(m_trees)))             # leaf-value scale
        split = pm.Bernoulli("split", alpha, dims="tree")    # does the tree split (root)?
        c = pm.Uniform("c", 0, 10, dims="tree")              # split point
        mu_L = pm.Normal("mu_L", 0, sigma_mu, dims="tree")   # left leaf value
        mu_R = pm.Normal("mu_R", 0, sigma_mu, dims="tree")   # right leaf value
        g = pm.Deterministic(
            "g",
            pm.math.where(split[:, None].astype("bool") & (x[None, :] <= c[:, None]), mu_L[:, None], mu_R[:, None]),
            dims=("tree", "obs"),
        )
        f = pm.Deterministic("f", g.sum(axis=0), dims="obs")  # the m-tree sum
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y", mu=f, sigma=sigma, observed=Y, dims="obs")
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
    "logistic_regression": build_logistic_regression,
    "poisson_loglink": build_poisson_loglink,
    "softmax_categorical": build_softmax_categorical,
    "probit_regression": build_probit_regression,
    "heteroskedastic_softplus": build_heteroskedastic_softplus,
    "saturating_tanh": build_saturating_tanh,
    "quadratic_power": build_quadratic_power,
    "bart_sum_of_trees": build_bart_sum_of_trees,
}
