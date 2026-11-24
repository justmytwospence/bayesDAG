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


# --------------------------------------------------------------------- the funnel joint (aux view)
def _funnel_idata(n_div=6, draws=200):
    """A centered eight-schools-shaped posterior with a genuine neck: theta's spread shrinks
    with tau, and the divergences sit at the small-tau end where the sampler gets stuck."""
    rng = np.random.default_rng(7)
    log_tau = np.linspace(-3.0, 2.0, draws)
    tau = np.exp(log_tau)
    theta = rng.normal(0.0, 1.0, (draws, 8)) * tau[:, None]
    diverging = np.zeros(draws, bool)
    diverging[:n_div] = True  # the smallest taus: the neck
    return xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {
                    "tau": (("chain", "draw"), tau[None, :]),
                    "theta": (("chain", "draw", "school"), theta[None, :, :]),
                }
            ),
            "sample_stats": xr.Dataset({"diverging": (("chain", "draw"), diverging[None, :])}),
        }
    )


def test_funnel_joint_separates_divergent_draws_and_labels_a_computed_axis():
    data = diagnostics.funnel_joint(_funnel_idata(), "tau", "theta", None)
    assert data["n_divergent"] == 6  # divergent TRANSITIONS, not element-points
    assert data["n_total"] == _funnel_idata()["posterior"].sizes["draw"]  # one chain
    assert len(data["div_x"]) == 6 * 8  # ... but every element of a divergent draw is plotted
    assert data["y_label"] == "theta (8 elements overlaid)"
    # no unconstrained_posterior in this idata, so the log axis is ours — and says so
    assert data["x_label"] == "log(tau) (computed)"
    # the divergences really are in the neck: their log(tau) is below the bulk's median
    assert max(data["div_x"]) < float(np.median(data["x"]))


def test_the_joint_prefers_the_samplers_own_unconstrained_draws():
    """The neck is only visible on the unconstrained scale. When the sampler recorded it, use
    that rather than recomputing — and drop the "(computed)" qualifier, because it isn't."""
    base = _funnel_idata()
    tree = base.to_dict()
    tree["unconstrained_posterior"] = xr.Dataset(
        {"tau_log__": (("chain", "draw"), np.log(base["posterior"]["tau"].values))}
    )
    data = diagnostics.funnel_joint(xr.DataTree.from_dict(tree), "tau", "theta", "tau_log__", "log")
    assert data["x_label"] == "log(tau)"
    assert "computed" not in data["x_label"]


def test_the_joint_is_deterministic_and_bounded():
    """Thinning is a stride, never an RNG: the same idata must always give the same picture, and
    every divergent point survives regardless of how many there are."""
    big = _funnel_idata(n_div=40, draws=6000)
    a = diagnostics.funnel_joint(big, "tau", "theta", None)
    b = diagnostics.funnel_joint(big, "tau", "theta", None)
    assert a == b
    assert len(a["x"]) <= diagnostics._MAX_POINTS + 1
    assert len(a["div_x"]) == 40 * 8  # never thinned


def test_no_joint_without_divergences_or_without_a_funnel():
    from bayesdag.convert import to_ir

    centered = to_ir(_centered_eight_schools())
    clean = _funnel_idata(n_div=0)
    assert diagnostics.joint_views(centered, clean) == []  # nothing went wrong: nothing to show
    assert diagnostics.joint_views(centered, None) == []

    non_centered = to_ir(_centered_eight_schools())
    non_centered.plates = []  # theta no longer indexed by a group -> not the funnel shape
    for n in non_centered.nodes:
        n.dims = []
    assert diagnostics.joint_views(non_centered, _funnel_idata()) == []


def test_the_joint_becomes_an_aux_view_and_a_rendered_panel():
    """AuxViewIR was declared in the IR from the start and never constructed. This is its first
    real use, so it also has to round-trip through the published schema."""
    from bayesdag.convert import to_ir
    from bayesdag.ir import ModelIR
    from bayesdag.render_svg import render_joint_panel

    ir = to_ir(_centered_eight_schools())
    diagnostics.annotate(ir, _funnel_idata())

    assert len(ir.aux_views) == 1
    aux = ir.aux_views[0]
    assert (aux.kind, aux.vars, aux.edge) == ("joint", ["theta", "tau"], ["tau", "theta"])

    panel = render_joint_panel(aux)
    assert panel.startswith("<svg") and "divergent draws" in panel
    assert "draws divergent" in panel  # transitions, not element-points
    assert "#c0392b" in panel  # the divergent draws are actually drawn

    assert ModelIR.from_dict(ir.to_dict()).aux_views == ir.aux_views  # schema round trip


def test_the_card_offers_the_joint_from_the_flagged_node():
    pytest.importorskip("anywidget")
    from bayesdag.convert import to_ir

    v = bayesdag.view(to_ir(_centered_eight_schools()), ppc_draws=0)
    diagnostics.annotate(v.ir, _funnel_idata())
    spec = v._build_spec()

    assert list(spec["aux"]) == ["tau"]  # offered from the scale — the neck of the funnel
    assert spec["aux"]["tau"][0]["label"] == "joint: theta vs log(tau)"
    assert "<svg" in spec["aux"]["tau"][0]["panel"]


def test_the_joint_pairs_draws_by_coordinate_not_storage_order():
    """A group stored (draw, chain), or with its chains relabelled in a different order, must
    still pair each theta draw with ITS OWN tau draw."""
    rng = np.random.default_rng(0)
    tau = np.exp(rng.normal(size=(2, 50)))
    theta = tau[:, :, None] * rng.normal(size=(2, 50, 3))
    div = np.zeros((2, 50), bool)
    div[1, :5] = True
    straight = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {
                    "tau": (("chain", "draw"), tau),
                    "theta": (("chain", "draw", "g"), theta),
                },
                coords={"chain": [0, 1]},
            ),
            "sample_stats": xr.Dataset(
                {"diverging": (("chain", "draw"), div)}, coords={"chain": [0, 1]}
            ),
        }
    )
    scrambled = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {
                    "tau": (("draw", "chain"), tau.T),  # stored transposed
                    "theta": (("chain", "draw", "g"), theta),
                },
                coords={"chain": [0, 1]},
            ),
            "sample_stats": xr.Dataset(  # chains stored in the other order
                {"diverging": (("chain", "draw"), div[::-1])}, coords={"chain": [1, 0]}
            ),
        }
    )
    a = diagnostics.funnel_joint(straight, "tau", "theta", None)
    b = diagnostics.funnel_joint(scrambled, "tau", "theta", None)
    assert a["n_divergent"] == b["n_divergent"] == 5
    assert sorted(a["div_x"]) == sorted(b["div_x"])
    assert sorted(a["x"]) == sorted(b["x"])


def test_mismatched_draw_sets_give_no_joint_rather_than_a_wrong_one():
    base = _funnel_idata()
    tree = base.to_dict()
    tree["/sample_stats"] = xr.Dataset(
        {"diverging": (("chain", "draw"), np.zeros((1, 10), bool))}  # too few draws
    )
    assert diagnostics.funnel_joint(xr.DataTree.from_dict(tree), "tau", "theta", None) == {}


def test_a_scale_that_is_not_positive_keeps_its_own_axis():
    base = _funnel_idata()
    tree = base.to_dict()
    tree["/posterior"]["tau"] = tree["/posterior"]["tau"] - 1.0  # crosses zero
    data = diagnostics.funnel_joint(xr.DataTree.from_dict(tree), "tau", "theta", None)
    assert data["x_label"] == "tau" and data["axis_space"] == "constrained"


def test_vector_diagnostics_name_the_worst_coordinate():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 200, 3))
    x[0, :, 2] += 5.0  # element 2 disagrees across chains
    idata = xr.DataTree.from_dict(
        {"posterior": xr.Dataset({"x": (("chain", "draw", "g"), x)}, coords={"g": ["a", "b", "c"]})}
    )
    d = diagnostics.per_node(idata, ["x"])["x"]
    assert d["rhat_at"] == "g=c"
    assert any("worst element: g=c" in r for r in diagnostics.describe(d))


def test_divergences_add_the_parameterization_hints():
    """With divergences, a centered hierarchy's effect gets the hedged "non-centered may help"
    row, and a non-centered hierarchy's scale the converse; without them, neither."""
    import sys

    sys.path.insert(0, "examples")
    from structures import build_eight_schools_centered, build_eight_schools_noncentered

    from bayesdag.convert import to_ir

    div = _funnel_idata()
    c = to_ir(build_eight_schools_centered())
    diagnostics.annotate(c, div)
    assert "reparam_hint" in c.node("theta").diag["flags"]
    assert any(
        "non-centered parameterization may help" in r
        for r in diagnostics.describe(c.node("theta").diag)
    )
    nc = to_ir(build_eight_schools_noncentered())
    diagnostics.annotate(nc, div)
    assert "centered_hint" in nc.node("tau").diag["flags"]
    assert diagnostics.funnel_candidates(nc) == []
    clean = to_ir(build_eight_schools_centered())
    diagnostics.annotate(clean, _funnel_idata(n_div=0))
    assert "reparam_hint" not in (clean.node("theta").diag or {}).get("flags", [])
