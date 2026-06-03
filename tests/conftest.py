"""Shared fixtures + the example-model registry.

Six models of escalating generative-structure complexity, used both as fixtures and (via
``MODEL_BUILDERS``) by the parametrized layout-quality tests:
  1. eight schools  2. radon  3. hierarchical regression  4. IRT (2PL)
  5. MRP (5 crossed grouping factors)  6. joint longitudinal-survival (interlinked sub-hierarchies)
"""

import numpy as np
import pymc as pm
import pytest


def build_eight_schools():
    """M0 slice: hyperpriors, a renderable deterministic, a plate, an observed likelihood."""
    y = np.array([28.0, 8, -3, 7, -1, 1, 18, 12])
    sigma = np.array([15.0, 10, 16, 11, 9, 11, 10, 18])
    with pm.Model(coords={"school": [f"S{i}" for i in range(8)]}) as model:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfNormal("tau", 5)
        eta = pm.Normal("eta", 0, 1, dims="school")
        theta = pm.Deterministic("theta", mu + tau * eta, dims="school")
        pm.Normal("y_obs", theta, sigma, observed=y, dims="school")
    return model


def build_radon():
    """Two plates + an external scalar parent (`sigma`) of a plate-internal node (`y`)."""
    rng = np.random.default_rng(0)
    nc, no = 6, 40
    cidx = rng.integers(0, nc, no)
    floor = rng.integers(0, 2, no).astype(float)
    a_true = rng.normal(1.2, 0.5, nc)
    radon_log = a_true[cidx] - 0.6 * floor + rng.normal(0, 0.4, no)
    with pm.Model(coords={"county": [f"C{i}" for i in range(nc)], "obs": np.arange(no)}) as model:
        mu_a = pm.Normal("mu_a", 0, 5)
        sigma_a = pm.HalfNormal("sigma_a", 5)
        a = pm.Normal("a", mu_a, sigma_a, dims="county")
        b = pm.Normal("b", 0, 5)
        sigma = pm.HalfNormal("sigma", 1)
        cc = pm.Data("county_idx", cidx, dims="obs")
        fl = pm.Data("floor", floor, dims="obs")
        mu = pm.Deterministic("mu", a[cc] + b * fl, dims="obs")
        pm.Normal("y", mu, sigma, observed=radon_log, dims="obs")
    return model


def build_hier_reg():
    """Varying intercept + several fixed coefficients converging on ONE deterministic equation."""
    rng = np.random.default_rng(1)
    nc, no = 6, 60
    cidx = rng.integers(0, nc, no)
    x1, x2, x3 = (rng.normal(size=no) for _ in range(3))
    a_true = rng.normal(0, 1, nc)
    y = a_true[cidx] + 0.5 * x1 - 0.3 * x2 + 0.2 * x3 + rng.normal(0, 0.3, no)
    with pm.Model(coords={"county": np.arange(nc), "obs": np.arange(no)}) as model:
        mu_a = pm.Normal("mu_a", 0, 5)
        sigma_a = pm.HalfNormal("sigma_a", 5)
        a = pm.Normal("a", mu_a, sigma_a, dims="county")
        b1 = pm.Normal("b1", 0, 5)
        b2 = pm.Normal("b2", 0, 5)
        b3 = pm.Normal("b3", 0, 5)
        sigma = pm.HalfNormal("sigma", 1)
        cc = pm.Data("county_idx", cidx, dims="obs")
        X1 = pm.Data("x1", x1, dims="obs")
        X2 = pm.Data("x2", x2, dims="obs")
        X3 = pm.Data("x3", x3, dims="obs")
        mu = pm.Deterministic("mu", a[cc] + b1 * X1 + b2 * X2 + b3 * X3, dims="obs")
        pm.Normal("y", mu, sigma, observed=y, dims="obs")
    return model


def build_irt():
    """2-parameter item-response theory: three crossed plates (student, item, obs)."""
    rng = np.random.default_rng(2)
    ns, ni = 20, 10
    no = ns * ni
    si = np.repeat(np.arange(ns), ni)
    ii = np.tile(np.arange(ni), ns)
    th, a_t, b_t = rng.normal(0, 1, ns), rng.lognormal(0, 0.3, ni), rng.normal(0, 1, ni)
    y = rng.binomial(1, 1 / (1 + np.exp(-(a_t[ii] * (th[si] - b_t[ii])))))
    with pm.Model(coords={"student": np.arange(ns), "item": np.arange(ni), "obs": np.arange(no)}) as model:
        theta = pm.Normal("theta", 0, 1, dims="student")
        mu_a = pm.Normal("mu_a", 0, 1)
        sigma_a = pm.HalfNormal("sigma_a", 1)
        a = pm.LogNormal("a", mu_a, sigma_a, dims="item")
        mu_b = pm.Normal("mu_b", 0, 1)
        sigma_b = pm.HalfNormal("sigma_b", 1)
        b = pm.Normal("b", mu_b, sigma_b, dims="item")
        S = pm.Data("student_idx", si, dims="obs")
        I = pm.Data("item_idx", ii, dims="obs")  # noqa: E741
        eta = pm.Deterministic("eta", a[I] * (theta[S] - b[I]), dims="obs")
        pm.Bernoulli("y", logit_p=eta, observed=y, dims="obs")
    return model


def build_mrp():
    """Multilevel Regression & Poststratification: five crossed grouping factors (non-centered)
    converging on one logit — six plates, ~7 convergent parents."""
    rng = np.random.default_rng(3)
    ns, na, ne, nh, nr, no = 8, 4, 4, 4, 4, 400
    idx = {k: rng.integers(0, n, no) for k, n in (("s", ns), ("a", na), ("e", ne), ("h", nh), ("r", nr))}
    male = rng.integers(0, 2, no).astype(float)
    y = rng.binomial(1, 0.5, no)
    coords = {"state": np.arange(ns), "age": np.arange(na), "edu": np.arange(ne),
              "eth": np.arange(nh), "region": np.arange(nr), "obs": np.arange(no)}
    with pm.Model(coords=coords) as model:
        a = pm.Normal("a", 0, 1)

        def re(name, dim):
            sg = pm.HalfNormal("sigma_" + name, 1)
            z = pm.Normal("z_" + name, 0, 1, dims=dim)
            return pm.Deterministic(name, z * sg, dims=dim)

        a_s, a_a, a_e, a_h, a_r = (
            re("a_state", "state"), re("a_age", "age"), re("a_edu", "edu"),
            re("a_eth", "eth"), re("a_region", "region"),
        )
        bm = pm.Normal("b_male", 0, 1)
        S = pm.Data("state_idx", idx["s"], dims="obs")
        A = pm.Data("age_idx", idx["a"], dims="obs")
        E = pm.Data("edu_idx", idx["e"], dims="obs")
        H = pm.Data("eth_idx", idx["h"], dims="obs")
        R = pm.Data("region_idx", idx["r"], dims="obs")
        M = pm.Data("male", male, dims="obs")
        p = pm.Deterministic("p", a + a_s[S] + a_a[A] + a_e[E] + a_h[H] + a_r[R] + bm * M, dims="obs")
        pm.Bernoulli("y", logit_p=p, observed=y, dims="obs")
    return model


def build_joint():
    """Joint longitudinal-survival: two sub-hierarchies over `subject`, linked by the
    association `alpha*b1` (the random slope feeds BOTH the trajectory and the survival rate)."""
    rng = np.random.default_rng(4)
    nsub, nvis = 40, 4
    nl = nsub * nvis
    si = np.repeat(np.arange(nsub), nvis)
    tm = np.tile(np.linspace(0, 1, nvis), nsub)
    b0t, b1t = rng.normal(1, 0.5, nsub), rng.normal(-0.5, 0.3, nsub)
    yl = b0t[si] + b1t[si] * tm + rng.normal(0, 0.2, nl)
    ev = rng.exponential(1 / np.exp(-1 + 0.8 * b1t))
    with pm.Model(coords={"subject": np.arange(nsub), "visit": np.arange(nl)}) as model:
        mu_b0 = pm.Normal("mu_b0", 0, 2)
        sigma_b0 = pm.HalfNormal("sigma_b0", 1)
        mu_b1 = pm.Normal("mu_b1", 0, 2)
        sigma_b1 = pm.HalfNormal("sigma_b1", 1)
        b0 = pm.Normal("b0", mu_b0, sigma_b0, dims="subject")
        b1 = pm.Normal("b1", mu_b1, sigma_b1, dims="subject")
        sigma_y = pm.HalfNormal("sigma_y", 1)
        S = pm.Data("subj_idx", si, dims="visit")
        T = pm.Data("time", tm, dims="visit")
        traj = pm.Deterministic("traj", b0[S] + b1[S] * T, dims="visit")
        pm.Normal("y_long", traj, sigma_y, observed=yl, dims="visit")
        gamma0 = pm.Normal("gamma0", 0, 2)
        alpha = pm.Normal("alpha", 0, 1)
        log_rate = pm.Deterministic("log_rate", gamma0 + alpha * b1, dims="subject")
        pm.Exponential("event_time", pm.math.exp(-log_rate), observed=ev, dims="subject")
    return model


# Registry for the parametrized layout-quality tests (models that lay out crossing-free vs the
# two dense cases whose residual single crossing is a documented plate-contiguity limit).
MODEL_BUILDERS = {
    "eight_schools": build_eight_schools,
    "radon": build_radon,
    "hier_reg": build_hier_reg,
    "irt": build_irt,
    "mrp": build_mrp,
    "joint": build_joint,
}
CROSSING_FREE_MODELS = ["eight_schools", "radon", "hier_reg", "joint"]
RESIDUAL_CROSSING_MODELS = ["irt", "mrp"]  # 1 forced edge-edge crossing (plate contiguity)


@pytest.fixture
def eight_schools_model():
    return build_eight_schools()


@pytest.fixture
def eight_schools_ir(eight_schools_model):
    from bayesdag.convert import to_ir

    return to_ir(eight_schools_model)


@pytest.fixture
def radon_model():
    return build_radon()


@pytest.fixture
def radon_ir(radon_model):
    from bayesdag.convert import to_ir

    return to_ir(radon_model)
