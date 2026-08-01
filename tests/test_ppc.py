"""Plate prior-predictive expansions.

`prior_predictive_expansions` had no direct test, and its only caller wraps it in a blanket
`except Exception: plates = {}` (view.py) — so any regression degraded silently to "no plate
panels" on every model, with nothing to notice it. These tests pin the data contract directly.
"""

import numpy as np
import pymc as pm
import pytest

from bayesdag.adapters.ppc import _MAX_CURVES, prior_predictive_expansions
from bayesdag.convert import to_ir


def _expansions(model, **kw):
    return prior_predictive_expansions(model, to_ir(model), **kw)


def test_expansion_shape_and_observed_members(eight_schools_model):
    exp = _expansions(eight_schools_model, draws=50)
    assert "plate_school" in exp
    plate = exp["plate_school"]
    members = {m["id"]: m for m in plate["members"]}

    # every plate member that has per-instance draws gets one curve per instance
    assert {"eta", "theta", "y_obs"} <= set(members)
    for m in members.values():
        assert m["n"] == 8 and len(m["curves"]) == 8
        assert all(len(c) == len(m["xs"]) for c in m["curves"])
        assert max(max(c) for c in m["curves"]) == pytest.approx(1.0)  # peak-normalized together
        assert not m["capped"]

    # the observed member carries the real data alongside the predictive curves — that pairing
    # IS the prior predictive check
    assert members["y_obs"]["role"] == "observed"
    assert len(members["y_obs"]["observed"]) == 8
    assert "observed" not in members["eta"]


def test_curves_are_capped_and_the_cap_is_declared():
    """A wide plate can't draw a curve per instance; the cap has to be visible in the data
    rather than silently truncating (the panel prints it)."""
    n = _MAX_CURVES + 15
    with pm.Model(coords={"g": range(n)}) as m:
        pm.Normal("z", 0.0, 1.0, dims="g")
        pm.Normal("y", m["z"], 1.0, observed=np.zeros(n), dims="g")

    member = next(mm for mm in _expansions(m, draws=40)["plate_g"]["members"] if mm["id"] == "z")
    assert member["n"] == n  # the true instance count is reported...
    assert len(member["curves"]) == _MAX_CURVES  # ...even though fewer curves are drawn
    assert member["capped"] is True


def test_model_without_plates_returns_nothing():
    with pm.Model() as m:
        pm.Normal("mu", 0.0, 1.0)
    assert _expansions(m) == {}


def test_sampling_failure_degrades_to_empty(monkeypatch, eight_schools_model):
    """Prior-predictive simulation is best-effort: a model that can't be forward-simulated must
    yield no panels rather than breaking the widget build."""
    import pymc

    monkeypatch.setattr(
        pymc,
        "sample_prior_predictive",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cannot simulate")),
    )
    assert _expansions(eight_schools_model) == {}


def test_expansions_are_deterministic(eight_schools_model):
    """Seeded simulation: the panel must not change shape between two builds of the same model."""
    a = _expansions(eight_schools_model, draws=40)
    b = _expansions(eight_schools_model, draws=40)
    assert a == b
