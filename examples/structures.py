"""Compound model structures bayesdag recognizes ("blocks") — seeded, synthetic builders.

Each builder returns a ``pm.Model`` built from generated arrays with anonymous coordinates. The
last two are stand-ins for two real analyses (a proteomics measurement model and a VO2max
outcome model): they reproduce the STRUCTURE those models have — 2-D protein x batch arrays,
log-scale hierarchies, bounded readings, Potential likelihoods, a regularized horseshoe with
half-t locals and co-data — without importing, running or copying anything from them.

Negative examples (ridge, spike-and-slab, a slab-less global-local product) live here too: a
recognizer is only as good as the look-alikes it declines.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

# ----------------------------------------------------------------------------- hierarchies


def build_eight_schools_noncentered():
    y = np.array([28.0, 8, -3, 7, -1, 1, 18, 12])
    sigma = np.array([15.0, 10, 16, 11, 9, 11, 10, 18])
    with pm.Model(coords={"school": [f"S{i}" for i in range(8)]}) as m:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfNormal("tau", 5)
        eta = pm.Normal("eta", 0, 1, dims="school")
        theta = pm.Deterministic("theta", mu + tau * eta, dims="school")
        pm.Normal("y_obs", theta, sigma, observed=y, dims="school")
    return m


def build_eight_schools_centered():
    y = np.array([28.0, 8, -3, 7, -1, 1, 18, 12])
    sigma = np.array([15.0, 10, 16, 11, 9, 11, 10, 18])
    with pm.Model(coords={"school": [f"S{i}" for i in range(8)]}) as m:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfNormal("tau", 5)
        theta = pm.Normal("theta", mu, tau, dims="school")
        pm.Normal("y_obs", theta, sigma, observed=y, dims="school")
    return m


def _radon_data(n_county=8, n_obs=60, seed=0):
    rng = np.random.default_rng(seed)
    county = rng.integers(0, n_county, n_obs)
    floor = rng.integers(0, 2, n_obs).astype(float)
    a = rng.normal(1.3, 0.4, n_county)
    b = rng.normal(-0.6, 0.2, n_county)
    y = a[county] + b[county] * floor + rng.normal(0, 0.5, n_obs)
    return county, floor, y


def build_radon_centered():
    """Varying intercepts AND slopes by county, centered."""
    county, floor, y = _radon_data()
    coords = {"county": [f"C{i}" for i in range(8)], "obs": np.arange(len(y))}
    with pm.Model(coords=coords) as m:
        mu_a = pm.Normal("mu_a", 0, 5)
        sigma_a = pm.HalfNormal("sigma_a", 1)
        mu_b = pm.Normal("mu_b", 0, 5)
        sigma_b = pm.HalfNormal("sigma_b", 1)
        a = pm.Normal("a", mu_a, sigma_a, dims="county")
        b = pm.Normal("b", mu_b, sigma_b, dims="county")
        cidx = pm.Data("county_idx", county, dims="obs")
        x = pm.Data("floor", floor, dims="obs")
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y", a[cidx] + b[cidx] * x, sigma, observed=y, dims="obs")
    return m


def build_radon_noncentered():
    county, floor, y = _radon_data()
    coords = {"county": [f"C{i}" for i in range(8)], "obs": np.arange(len(y))}
    with pm.Model(coords=coords) as m:
        mu_a = pm.Normal("mu_a", 0, 5)
        sigma_a = pm.HalfNormal("sigma_a", 1)
        z_a = pm.Normal("z_a", 0, 1, dims="county")
        a = pm.Deterministic("a", mu_a + sigma_a * z_a, dims="county")
        b = pm.Normal("b", 0, 1)
        cidx = pm.Data("county_idx", county, dims="obs")
        x = pm.Data("floor", floor, dims="obs")
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y", a[cidx] + b * x, sigma, observed=y, dims="obs")
    return m


# ----------------------------------------------------------------------------- regressions


def _design(n=80, p=3, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)), rng


def build_logistic_grouped():
    X, rng = _design()
    g = rng.integers(0, 5, len(X))
    y = rng.integers(0, 2, len(X))
    coords = {"obs": np.arange(len(X)), "covariate": ["x1", "x2", "x3"], "group": range(5)}
    with pm.Model(coords=coords) as m:
        Xd = pm.Data("X", X, dims=("obs", "covariate"))
        gi = pm.Data("group_idx", g, dims="obs")
        beta = pm.Normal("beta", 0, 1, dims="covariate")
        s = pm.HalfNormal("group_sd", 1)
        z = pm.Normal("group_z", 0, 1, dims="group")
        u = pm.Deterministic("group_effect", s * z, dims="group")
        pm.Bernoulli("y", logit_p=Xd @ beta + u[gi], observed=y, dims="obs")
    return m


def build_poisson_regression():
    X, rng = _design()
    y = rng.poisson(2.0, len(X))
    coords = {"obs": np.arange(len(X)), "covariate": ["x1", "x2", "x3"]}
    with pm.Model(coords=coords) as m:
        Xd = pm.Data("X", X, dims=("obs", "covariate"))
        a = pm.Normal("alpha", 0, 1)
        beta = pm.Normal("beta", 0, 1, dims="covariate")
        pm.Poisson("y", pm.math.exp(a + Xd @ beta), observed=y, dims="obs")
    return m


def build_lognormal_regression():
    X, rng = _design()
    y = np.exp(rng.normal(size=len(X)))
    coords = {"obs": np.arange(len(X)), "covariate": ["x1", "x2", "x3"]}
    with pm.Model(coords=coords) as m:
        Xd = pm.Data("X", X, dims=("obs", "covariate"))
        a = pm.Normal("alpha", 0, 1)
        beta = pm.Normal("beta", 0, 1, dims="covariate")
        eta = pm.Deterministic("log_mu", a + Xd @ beta, dims="obs")
        pm.LogNormal("y", mu=eta, sigma=pm.HalfNormal("sigma", 1), observed=y, dims="obs")
    return m


# ----------------------------------------------------------------------------- shrinkage


def _hs_data(n=60, p=20, seed=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X[:, :3] @ np.array([1.0, -0.8, 0.5]) + rng.normal(0, 0.5, n)
    return X, y


def build_horseshoe():
    """The classical horseshoe (Carvalho, Polson & Scott 2010): half-Cauchy global and locals,
    no slab, non-centered."""
    X, y = _hs_data()
    coords = {"obs": np.arange(len(y)), "predictor": [f"x{i}" for i in range(X.shape[1])]}
    with pm.Model(coords=coords) as m:
        Xd = pm.Data("X", X, dims=("obs", "predictor"))
        tau = pm.HalfCauchy("tau", 1.0)
        lam = pm.HalfCauchy("lam", 1.0, dims="predictor")
        z = pm.Normal("z", 0, 1, dims="predictor")
        beta = pm.Deterministic("beta", z * tau * lam, dims="predictor")
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y", Xd @ beta, sigma, observed=y, dims="obs")
    return m


def build_regularized_horseshoe(local_nu: float = 1.0, codata: bool = False):
    """The regularized horseshoe (Piironen & Vehtari 2017): global and local scales, an
    inverse-gamma slab, ``beta = center + z * tau * sqrt(c2 lam^2 / (c2 + tau^2 lam^2))``.
    ``local_nu != 1`` gives half-t locals; ``codata`` adds a co-data center and local-scale
    modifier (the reference outcome model's form)."""
    X, y = _hs_data()
    p = X.shape[1]
    rng = np.random.default_rng(3)
    coords = {"obs": np.arange(len(y)), "predictor": [f"x{i}" for i in range(p)]}
    with pm.Model(coords=coords) as m:
        Xd = pm.Data("X", X, dims=("obs", "predictor"))
        sigma = pm.HalfNormal("sigma", 1)
        tau = pm.HalfStudentT("tau", nu=2, sigma=0.1)
        if local_nu == 1.0:
            lam = pm.HalfCauchy("lam", 1.0, dims="predictor")
        else:
            lam = pm.HalfStudentT("lam", nu=local_nu, dims="predictor")
        if codata:
            C = rng.normal(size=(p, 2))
            w = pm.Normal("codata_scale_weight", 0, 0.5, shape=2)
            lam = lam * pt.exp(pt.constant(C) @ w)
        c2 = pm.InverseGamma("c2", alpha=2.0, beta=2.0 * 0.15**2)
        lam_t2 = c2 * lam**2 / (c2 + tau**2 * lam**2)
        z = pm.Normal("z", 0, 1, dims="predictor")
        dev = z * tau * pt.sqrt(lam_t2)
        if codata:
            M = rng.normal(size=(p, 2))
            wm = pm.Normal("codata_mean_weight", 0, 0.5, shape=2)
            dev = pm.Deterministic("prior_center", pt.constant(M) @ wm, dims="predictor") + dev
        beta = pm.Deterministic("beta", dev, dims="predictor")
        pm.Normal("y", Xd @ beta, sigma, observed=y, dims="obs")
    return m


def build_ridge():
    """NOT a horseshoe: one shared scale."""
    X, y = _hs_data()
    coords = {"obs": np.arange(len(y)), "predictor": [f"x{i}" for i in range(X.shape[1])]}
    with pm.Model(coords=coords) as m:
        Xd = pm.Data("X", X, dims=("obs", "predictor"))
        s = pm.HalfNormal("s", 1)
        beta = pm.Normal("beta", 0, s, dims="predictor")
        pm.Normal("y", Xd @ beta, 1.0, observed=y, dims="obs")
    return m


def build_spike_and_slab():
    """NOT a horseshoe: a Bernoulli inclusion times a slab."""
    X, y = _hs_data()
    coords = {"obs": np.arange(len(y)), "predictor": [f"x{i}" for i in range(X.shape[1])]}
    with pm.Model(coords=coords) as m:
        Xd = pm.Data("X", X, dims=("obs", "predictor"))
        gamma = pm.Bernoulli("gamma", 0.2, dims="predictor")
        slab = pm.Normal("slab", 0, 2, dims="predictor")
        beta = pm.Deterministic("beta", gamma * slab, dims="predictor")
        pm.Normal("y", Xd @ beta, 1.0, observed=y, dims="obs")
    return m


def build_global_local_halfnormal():
    """Global-local, but HalfNormal locals and no slab: shrinkage, not a horseshoe."""
    X, y = _hs_data()
    coords = {"obs": np.arange(len(y)), "predictor": [f"x{i}" for i in range(X.shape[1])]}
    with pm.Model(coords=coords) as m:
        Xd = pm.Data("X", X, dims=("obs", "predictor"))
        tau = pm.HalfNormal("tau", 1)
        lam = pm.HalfNormal("lam", 1, dims="predictor")
        beta = pm.Normal("beta", 0, tau * lam, dims="predictor")
        pm.Normal("y", Xd @ beta, 1.0, observed=y, dims="obs")
    return m


# ----------------------------------------------------------------------------- measurement


def build_measurement_error():
    """Errors-in-variables: the covariate is measured with noise; the outcome uses the truth."""
    rng = np.random.default_rng(4)
    n = 40
    x_true = rng.normal(size=n)
    x_obs = x_true + rng.normal(0, 0.3, n)
    y = 1.0 + 2.0 * x_true + rng.normal(0, 0.5, n)
    with pm.Model(coords={"obs": np.arange(n)}) as m:
        mu_x = pm.Normal("mu_x", 0, 1)
        sd_x = pm.HalfNormal("sd_x", 1)
        xt = pm.Normal("x_true", mu_x, sd_x, dims="obs")
        pm.Normal("x_obs", xt, 0.3, observed=x_obs, dims="obs")
        a = pm.Normal("a", 0, 1)
        b = pm.Normal("b", 0, 1)
        pm.Normal("y", a + b * xt, pm.HalfNormal("sigma", 1), observed=y, dims="obs")
    return m


def build_bridge_measurement_like():
    """Stand-in for a proteomics bridge-sample measurement model: protein x batch arrays, a
    truncated protein mean with a parent-dependent bound, log-scale non-centered hierarchies,
    a zero-sum batch center, a constrained run basis with an MvNormal, bounded latent readings
    for flagged cells, and the whole likelihood written as Potentials."""
    rng = np.random.default_rng(5)
    n_p, n_b, n_run = 6, 3, 5
    Q = np.linalg.qr(rng.normal(size=(n_run, n_run - 1)))[0]
    y = rng.normal(size=(n_p, n_b))
    above = rng.random((n_p, n_b)) < 0.15
    coords = {
        "protein": [f"P{i}" for i in range(n_p)],
        "batch": [f"b{i}" for i in range(n_b)],
        "query_batch": [f"b{i}" for i in range(1, n_b)],
        "run": range(n_run),
        "run_free": range(n_run - 1),
        "cell_above": range(int(above.sum())),
    }
    with pm.Model(coords=coords) as m:
        floor = pm.Normal("signal_floor", 0, 1)
        protein_mean = pm.TruncatedNormal("protein_mean", 1.0, 2.0, lower=floor, dims="protein")
        mu_t = pm.Normal("log_tube_spread_mu", -1, 0.5)
        sd_t = pm.HalfNormal("log_tube_spread_sd", 0.5)
        z_t = pm.Normal("tube_spread_z", 0, 1, dims="protein")
        tube_spread = pm.Deterministic("tube_spread", pt.exp(mu_t + sd_t * z_t), dims="protein")
        batch_center = pm.ZeroSumNormal("batch_center", sigma=0.5, dims="batch")
        offset_spread = pm.HalfNormal("offset_spread", 0.3, dims="query_batch")
        offset_q = pm.TruncatedNormal(
            "batch_offset_q",
            batch_center[1:][None, :],
            offset_spread[None, :],
            lower=-2.0,
            dims=("protein", "query_batch"),
        )
        batch_offset = pm.Deterministic(
            "batch_offset",
            pt.concatenate([pt.zeros((n_p, 1)), offset_q], axis=1),
            dims=("protein", "batch"),
        )
        mu_s = pm.Normal("log_assay_sd_mu", -1, 0.5)
        sd_s = pm.HalfNormal("log_assay_sd_sd", 0.5)
        z_s = pm.Normal("assay_sd_z", 0, 1, dims="protein")
        assay_sd = pm.Deterministic("assay_sd", pt.exp(mu_s + sd_s * z_s), dims="protein")
        run_sd = pm.HalfNormal("run_sd", 0.2, dims="batch")
        Qt = pt.as_tensor(Q)
        run_idx = np.arange(n_run) % n_b
        cov_w = Qt.T @ (pt.sqr(run_sd)[run_idx][:, None] * Qt)
        w = pm.MvNormal("run_level_w", mu=pt.zeros(n_run - 1), cov=cov_w, dims="run_free")
        pm.Deterministic("run_level", Qt @ w, dims="run")
        bound = np.full(int(above.sum()), 2.5)
        y_above = pm.Uniform("y_above", lower=bound, upper=bound + 1.0, dims="cell_above")
        y_t = pt.set_subtensor(pt.as_tensor(y)[np.nonzero(above)], y_above)
        mu = protein_mean[:, None] + batch_offset
        var = pt.sqr(tube_spread)[:, None] + pt.sqr(assay_sd)[:, None]
        pm.Potential("bridge_marginal", (-0.5 * pt.log(var) - 0.5 * pt.sqr(y_t - mu) / var).sum())
        pm.Potential(
            "study_censored",
            pm.logcdf(pm.Normal.dist(0.0, 1.0), (mu[:, 0] - 3.0) / pt.sqrt(var[:, 0])).sum(),
        )
    return m


def build_vo2max_like(two_stage: bool = True):
    """Stand-in for a VO2max outcome model: a non-centered random intercept per subject, a
    regularized horseshoe with half-t(2) global and locals, an inverse-gamma slab and co-data,
    a LogNormal likelihood and a heteroskedastic observation sd. ``two_stage`` feeds protein
    summaries as fixed constants (``pt.constant``), as a measurement model's posterior means."""
    rng = np.random.default_rng(6)
    n, n_subj, p, k = 40, 20, 12, 2
    subj = rng.integers(0, n_subj, n)
    Z = rng.normal(size=(n, p))
    Zsd2 = rng.uniform(0.01, 0.05, size=(n, p))
    Xc = rng.normal(size=(n, 2))
    y = np.exp(3.5 + rng.normal(0, 0.1, n))
    coords = {
        "obs": np.arange(n),
        "subject": range(n_subj),
        "covariate": ["age", "sex"],
        "protein": [f"P{i}" for i in range(p)],
    }
    with pm.Model(coords=coords) as m:
        cov = pm.Data("covariates", Xc, dims=("obs", "covariate"))
        yd = pm.Data("vo2max_data", y, dims="obs")
        si = pm.Data("subject_index", subj.astype("int32"), dims="obs")
        intercept = pm.Normal("vo2max_intercept", 3.5, 0.2)
        cov_eff = pm.Normal("covariate_effects", 0, 0.15, dims="covariate")
        between = pm.HalfNormal("between_subject_sd", 0.2)
        raw_s = pm.Normal("subject_offset_raw", 0, 1, dims="subject")
        subject_offset = pm.Deterministic("subject_offset", between * raw_s, dims="subject")
        within = pm.Exponential("within_subject_sd", 10.0)
        lin = intercept + cov @ cov_eff
        g = pm.HalfStudentT("global_shrinkage", nu=2, sigma=0.05)
        lam = pm.HalfStudentT("local_shrinkage", nu=2, dims="protein")
        C = rng.normal(size=(p, k))
        wsc = pm.Normal("codata_scale_weight", 0, 0.5, shape=k)
        lam = lam * pt.exp(pt.constant(C) @ wsc)
        slab = pm.InverseGamma("slab_variance", alpha=2.0, beta=2.0 * 0.15**2)
        shrunk2 = slab * lam**2 / (slab + g**2 * lam**2)
        raw_p = pm.Normal("protein_effect_raw", 0, 1, dims="protein")
        dev = raw_p * g * pt.sqrt(shrunk2)
        M = rng.normal(size=(p, k))
        wm = pm.Normal("codata_mean_weight", 0, 0.5, shape=k)
        dev = pm.Deterministic("protein_prior_centre", pt.constant(M) @ wm, dims="protein") + dev
        effects = pm.Deterministic("protein_effects", dev, dims="protein")
        kappa = 1.0 / (1.0 + n * g**2 * shrunk2)
        pm.Deterministic("n_effective_proteins", (1.0 - kappa).sum())
        if two_stage:
            Zt = pt.constant(Z)
            meas_var = pt.constant(Zsd2) @ pt.sqr(effects)
        else:
            true_z = pm.Normal("protein_level", 0, 1, dims=("obs", "protein"))
            pm.Normal("protein_reading", true_z, 0.2, observed=Z, dims=("obs", "protein"))
            Zt = true_z
            meas_var = pt.zeros(n)
        lin = lin + Zt @ effects
        obs_sd = pm.Deterministic("obs_sd", pt.sqrt(within**2 + meas_var), dims="obs")
        pred = pm.Deterministic("predicted_log_vo2max", lin, dims="obs")
        pm.LogNormal(
            "vo2max_observed",
            mu=pred + subject_offset[si],
            sigma=obs_sd,
            observed=yd,
            dims="obs",
        )
    return m


STRUCTURE_MODELS = {
    "eight_schools_noncentered": build_eight_schools_noncentered,
    "eight_schools_centered": build_eight_schools_centered,
    "radon_centered": build_radon_centered,
    "radon_noncentered": build_radon_noncentered,
    "logistic_grouped": build_logistic_grouped,
    "poisson_regression": build_poisson_regression,
    "lognormal_regression": build_lognormal_regression,
    "horseshoe": build_horseshoe,
    "regularized_horseshoe": build_regularized_horseshoe,
    "regularized_horseshoe_codata": lambda: build_regularized_horseshoe(2.0, codata=True),
    "ridge": build_ridge,
    "spike_and_slab": build_spike_and_slab,
    "global_local_halfnormal": build_global_local_halfnormal,
    "measurement_error": build_measurement_error,
    "bridge_measurement_like": build_bridge_measurement_like,
    "vo2max_like": build_vo2max_like,
    "vo2max_like_joint": lambda: build_vo2max_like(two_stage=False),
}
