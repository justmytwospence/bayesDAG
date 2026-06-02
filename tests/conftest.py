"""Shared fixtures. The 8-schools non-centered model is the M0 vertical-slice model:
it exercises hyperpriors, a deterministic with renderable math, a plate, an observed
likelihood, and a transformed (log) scale parameter."""

import numpy as np
import pymc as pm
import pytest


@pytest.fixture
def eight_schools_model():
    y = np.array([28.0, 8, -3, 7, -1, 1, 18, 12])
    sigma = np.array([15.0, 10, 16, 11, 9, 11, 10, 18])
    schools = [f"S{i}" for i in range(8)]
    with pm.Model(coords={"school": schools}) as model:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfNormal("tau", 5)
        eta = pm.Normal("eta", 0, 1, dims="school")
        theta = pm.Deterministic("theta", mu + tau * eta, dims="school")
        pm.Normal("y_obs", theta, sigma, observed=y, dims="school")
    return model


@pytest.fixture
def eight_schools_ir(eight_schools_model):
    from bayesdag.convert import to_ir

    return to_ir(eight_schools_model)


@pytest.fixture
def radon_model():
    """A hierarchical model with two plates and an external scalar parent (`sigma`) of a
    plate-internal node (`y`) — the case that exposes the plate-layout crossing problem."""
    rng = np.random.default_rng(0)
    n_counties, n_obs = 6, 40
    county_idx = rng.integers(0, n_counties, n_obs)
    floor = rng.integers(0, 2, n_obs).astype(float)
    a_true = rng.normal(1.2, 0.5, n_counties)
    radon_log = a_true[county_idx] - 0.6 * floor + rng.normal(0, 0.4, n_obs)
    coords = {"county": [f"C{i}" for i in range(n_counties)], "obs": np.arange(n_obs)}
    with pm.Model(coords=coords) as model:
        mu_a = pm.Normal("mu_a", 0, 5)
        sigma_a = pm.HalfNormal("sigma_a", 5)
        a = pm.Normal("a", mu_a, sigma_a, dims="county")
        b = pm.Normal("b", 0, 5)
        sigma = pm.HalfNormal("sigma", 1)
        cidx = pm.Data("county_idx", county_idx, dims="obs")
        fl = pm.Data("floor", floor, dims="obs")
        mu = pm.Deterministic("mu", a[cidx] + b * fl, dims="obs")
        pm.Normal("y", mu, sigma, observed=radon_log, dims="obs")
    return model


@pytest.fixture
def radon_ir(radon_model):
    from bayesdag.convert import to_ir

    return to_ir(radon_model)
