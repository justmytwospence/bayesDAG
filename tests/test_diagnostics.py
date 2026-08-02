"""Sampling diagnostics on the diagram: hedged flags, joined by node id.

The contract these tests defend is as much about wording and restraint as about numbers. A
diagnostic badge says "inspect this"; it must never imply a verdict, never invent a value it
does not have, and never move the diagram it is drawn on.
"""

import numpy as np
import pymc as pm
import pytest
import xarray as xr

import bayesdag
from bayesdag import diagnostics


def _centered_eight_schools():
    """theta ~ Normal(mu, tau) directly — the funnel-prone parameterization."""
    y = np.array([28.0, 8, -3, 7, -1, 1, 18, 12])
    sigma = np.array([15.0, 10, 16, 11, 9, 11, 10, 18])
    with pm.Model(coords={"school": [f"S{i}" for i in range(8)]}) as m:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfNormal("tau", 5)
        theta = pm.Normal("theta", mu, tau, dims="school")
        pm.Normal("y_obs", theta, sigma, observed=y, dims="school")
    return m


def _dims_for(name, arr):
    """(chain, draw) plus one named dim per extra axis, so vector variables are well-formed."""
    return ["chain", "draw"] + [f"{name}_dim{i}" for i in range(arr.ndim - 2)]


def _idata(posterior: dict, diverging=None):
    groups = {
        "posterior": xr.Dataset({k: (_dims_for(k, v), v) for k, v in posterior.items()}),
    }
    if diverging is not None:
        groups["sample_stats"] = xr.Dataset({"diverging": (("chain", "draw"), diverging)})
    return xr.DataTree.from_dict(groups)


# --------------------------------------------------------------------- per-node statistics
def test_rhat_flag_trips_on_chains_that_disagree_and_stays_hedged():
    rng = np.random.default_rng(0)
    # two chains sampling different distributions: R-hat should be far above 1
    split = np.concatenate([rng.normal(0, 1, (1, 400)), rng.normal(6, 1, (1, 400))], axis=0)
    agree = rng.normal(0, 1, (2, 400))
    stats = diagnostics.per_node(_idata({"bad": split, "good": agree}), ["bad", "good"])

    assert "rhat" in stats["bad"]["flags"]
    assert stats["bad"]["rhat"] > diagnostics.RHAT_THRESHOLD
    assert "rhat" not in stats["good"]["flags"]

    wording = " ".join(diagnostics.describe(stats["bad"]))
    assert "inspect" in wording
    for verdict in ("failed", "invalid", "wrong", "broken"):
        assert verdict not in wording.lower()


def test_low_ess_is_flagged():
    rng = np.random.default_rng(1)
    # a near-random-walk chain: highly autocorrelated, so very few effective samples
    walk = np.cumsum(rng.normal(0, 1, (2, 300)), axis=1)
    stats = diagnostics.per_node(_idata({"sticky": walk}), ["sticky"])
    assert "ess" in stats["sticky"]["flags"]
    assert "ESS bulk" in " ".join(diagnostics.describe(stats["sticky"]))


def test_a_single_chain_reports_no_rhat_rather_than_a_meaningless_one():
    """R-hat compares between-chain to within-chain variance; with one chain there is no such
    number. Reporting 1.0 (or nan) would read as "converged"."""
    rng = np.random.default_rng(2)
    stats = diagnostics.per_node(_idata({"x": rng.normal(0, 1, (1, 200))}), ["x"])
    assert stats["x"]["rhat"] is None
    assert "rhat" not in stats["x"]["flags"]
    assert "R-hat needs >1 chain" in diagnostics.describe(stats["x"])


def test_a_vector_node_says_its_number_is_the_worst_element():
    """One number standing for eight schools is the same lie the pooled-KDE glyph refuses to
    tell, so the caption has to qualify it."""
    rng = np.random.default_rng(3)
    stats = diagnostics.per_node(_idata({"theta": rng.normal(0, 1, (2, 300, 8))}), ["theta"])
    assert stats["theta"]["vector"] is True
    assert any("worst element" in row for row in diagnostics.describe(stats["theta"]))


def test_nothing_is_reported_without_an_idata_or_without_sample_stats():
    """Never fabricate. No idata means no diagnostics; no sample_stats means no divergence
    count — not a reassuring zero."""
    rng = np.random.default_rng(4)
    assert diagnostics.per_node(None, ["x"]) == {}
    assert diagnostics.model_level(None) == {}
    assert diagnostics.model_level(_idata({"x": rng.normal(size=(2, 50))})) == {}

    with_stats = _idata({"x": rng.normal(size=(2, 50))}, diverging=np.zeros((2, 50), bool))
    assert diagnostics.model_level(with_stats) == {"divergences": 0, "draws": 100}


# --------------------------------------------------------------------- structural funnel flag
def test_funnel_candidates_are_structural(eight_schools_ir):
    """A latent used as the SCALE of a plated latent child is Neal's funnel. The non-centered
    eight schools deliberately avoids it, which is the whole point of that reparameterization."""
    from bayesdag.convert import to_ir

    centered = to_ir(_centered_eight_schools())
    assert diagnostics.funnel_candidates(centered) == [("tau", "theta")]
    assert diagnostics.funnel_candidates(eight_schools_ir) == []  # non-centered: no funnel


def test_the_funnel_hint_waits_for_divergences_to_explain():
    """Structure alone would badge every centered hierarchy that sampled perfectly well. The
    flag is a pointer to something that went wrong, so it needs something to point at."""
    from bayesdag.convert import to_ir

    rng = np.random.default_rng(5)
    posterior = {
        "mu": rng.normal(size=(2, 100)),
        "tau": np.abs(rng.normal(size=(2, 100))) + 0.1,
        "theta": rng.normal(size=(2, 100, 8)),
    }

    clean = to_ir(_centered_eight_schools())
    diagnostics.annotate(clean, _idata(posterior, diverging=np.zeros((2, 100), bool)))
    assert "funnel" not in (clean.node("tau").diag or {}).get("flags", [])

    diverged = to_ir(_centered_eight_schools())
    mask = np.zeros((2, 100), bool)
    mask[0, :7] = True
    summary = diagnostics.annotate(diverged, _idata(posterior, diverging=mask))
    assert summary == {"divergences": 7, "draws": 200}
    assert "funnel" in diverged.node("tau").diag["flags"]
    assert "inspect the joint" in " ".join(diagnostics.describe(diverged.node("tau").diag))


# --------------------------------------------------------------------- rendering + the view
def test_the_badge_reserves_no_space_so_the_diagram_cannot_move():
    """This is what lets update() attach a posterior AND its diagnostics without anything
    shifting: the mark is drawn inside the box the node already has."""
    from bayesdag.convert import to_ir
    from bayesdag.layout import layout
    from bayesdag.render_svg import to_svg

    plain = to_ir(_centered_eight_schools())
    boxes_plain = layout(plain).node_boxes

    badged = to_ir(_centered_eight_schools())
    for n in badged.nodes:
        n.diag = {"flags": ["rhat"], "rhat": 1.4}
    res = layout(badged)

    assert {k: (b.x, b.y, b.w, b.h) for k, b in res.node_boxes.items()} == {
        k: (b.x, b.y, b.w, b.h) for k, b in boxes_plain.items()
    }
    svg = to_svg(badged, res)
    assert 'class="bd-diag"' in svg
    assert "inspect — sampling diagnostic" in svg  # legend explains the mark


def test_a_view_carries_diagnostics_through_construction_and_update():
    pytest.importorskip("anywidget")
    model = _centered_eight_schools()
    with model:
        idata = pm.sample(
            draws=150,
            tune=150,
            chains=2,
            cores=1,
            random_seed=0,
            progressbar=False,
            compute_convergence_checks=False,
        )

    v = bayesdag.view(model, ppc_draws=0)
    assert all(n.diag is None for n in v.ir.nodes)  # no idata, no diagnostics
    assert "diag" not in v.widget().spec  # ...and no model-level strip

    v.update(idata=idata)
    spec = v.widget().spec
    assert isinstance(spec["nodes"]["tau"]["diag"], list)
    n_div = int(idata.sample_stats["diverging"].sum())
    if n_div:
        assert "divergent transition" in spec["diagnostics"]
        assert "not that the model is wrong" in spec["diagnostics"]

    v.update(None)  # back to the prior: the diagnostics must go with it
    assert all(n.diag is None for n in v.ir.nodes)
    assert "diagnostics" not in v.widget().spec
    assert 'class="bd-diag"' not in v.to_svg()
