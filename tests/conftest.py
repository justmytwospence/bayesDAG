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
