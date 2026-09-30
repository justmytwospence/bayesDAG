"""Recognized model structures ("blocks"): detection, annotations, and the overview projection.

The contract is conservative: each rule claims a structure only when it is a consequence of the
op graph. The negative rows matter as much as the positive ones.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
from structures import STRUCTURE_MODELS

import bayesdag
from bayesdag import blocks, schema
from bayesdag.convert import to_ir


def _blocks(build):
    ir = to_ir(build() if callable(build) else build)
    return ir, {(b.rule, b.anchor): b for b in ir.blocks}


def _rules(build):
    return {(b.rule, b.anchor, b.title) for b in to_ir(build()).blocks}


# ------------------------------------------------------------------------------ positives


@pytest.mark.parametrize(
    "name,expected",
    [
        (
            "eight_schools_noncentered",
            {("noncentered/1", "theta", "partial pooling · non-centered")},
        ),
        ("eight_schools_centered", {("centered/1", "theta", "partial pooling · centered")}),
        (
            "radon_centered",
            {
                ("centered/1", "a", "partial pooling · centered"),
                ("centered/1", "b", "partial pooling · centered"),
                ("regression/1", "y", "linear regression · with group effects"),
            },
        ),
        (
            "radon_noncentered",
            {
                ("noncentered/1", "a", "partial pooling · non-centered"),
                ("regression/1", "y", "linear regression · with group effects"),
            },
        ),
        (
            "logistic_grouped",
            {
                ("noncentered/1", "group_effect", "partial pooling · non-centered"),
                ("regression/1", "y", "logistic regression · with group effects"),
            },
        ),
        ("poisson_regression", {("regression/1", "y", "Poisson regression (log link)")}),
        ("lognormal_regression", {("regression/1", "y", "log-normal regression")}),
        (
            "horseshoe",
            {("shrinkage/1", "beta", "horseshoe"), ("regression/1", "y", "linear regression")},
        ),
        (
            "regularized_horseshoe",
            {
                ("shrinkage/1", "beta", "regularized horseshoe"),
                ("regression/1", "y", "linear regression"),
            },
        ),
        (
            "regularized_horseshoe_codata",
            {
                ("shrinkage/1", "beta", "regularized horseshoe · half-t(2) locals"),
                ("regression/1", "y", "linear regression"),
            },
        ),
        (
            "global_local_halfnormal",
            {
                ("shrinkage/1", "beta", "global–local shrinkage"),
                ("regression/1", "y", "linear regression"),
            },
        ),
        (
            "measurement_error",
            {("measured/1", "x_true", "measured latent (errors-in-variables)")},
        ),
    ],
)
def test_recognized_structures(name, expected):
    assert _rules(STRUCTURE_MODELS[name]) == expected


def test_reference_outcome_model_stand_in():
    """The VO2max-shaped model: a half-t(2) regularized horseshoe with co-data (named as such,
    never as the classical horseshoe), a non-centered subject hierarchy, and a log-normal
    regression — with the model-defined effective size and the observation sd left visible."""
    ir, bs = _blocks(STRUCTURE_MODELS["vo2max_like"])
    hs = bs[("shrinkage/1", "protein_effects")]
    assert hs.title == "regularized horseshoe · half-t(2) locals"
    assert hs.roles["global"] == ["global_shrinkage"] and hs.roles["slab"] == ["slab_variance"]
    assert hs.roles["modifier"] == ["codata_scale_weight"]
    assert hs.roles["center"] == ["protein_prior_centre"]
    assert any("half-t(2)" in n for n in hs.notes)
    assert "also: non-centered" in " ".join(hs.notes) or any("non-centered" in n for n in hs.notes)
    sub = bs[("noncentered/1", "subject_offset")]
    assert sub.title == "partial pooling · non-centered"
    reg = bs[("regression/1", "vo2max_observed")]
    assert reg.title == "log-normal regression · with group effects"
    assert any("conditional median" in n for n in reg.notes)
    ov = blocks.overview(ir)
    shown = {n.id for n in ov.nodes}
    assert {"n_effective_proteins", "obs_sd", "global_shrinkage", "slab_variance"} <= shown
    assert "block:shrinkage:protein_effects" in shown and "protein_effect_raw" not in shown


def test_reference_measurement_model_stand_in():
    ir, bs = _blocks(STRUCTURE_MODELS["bridge_measurement_like"])
    for anchor in ("tube_spread", "assay_sd"):
        b = bs[("noncentered/1", anchor)]
        assert b.title == "partial pooling · non-centered (log-normal)"
        assert r"\mathrm{LogNormal}" in b.implied
        # LogNormal's shape depends on its (random) log-scale sd: no curve, a caption instead
        n = ir.node(anchor)
        assert n.glyph is None and "depends on the scale" in n.glyph_note
    assert ("centered/1", "batch_offset_q") in bs
    assert ("centered/1", "protein_mean") not in bs  # a shared truncation bound is not pooling
    inner = next(p for p in ir.plates if p.dims == ["protein", "batch"])
    assert inner.parent == next(p.id for p in ir.plates if p.dims == ["protein"])
    assert {n.id for n in ir.nodes if n.role == "potential"} == {
        "bridge_marginal",
        "study_censored",
    }


# ------------------------------------------------------------------------------ negatives


def _single(build):
    with pm.Model(coords={"g": range(4)}) as m:
        build()
    return {(b.rule, b.anchor) for b in to_ir(m).blocks}


@pytest.mark.parametrize(
    "label,build",
    [
        (
            "a + b*x with a standard-normal a is not non-centering",
            lambda: pm.Deterministic(
                "d", pm.Normal("a", 0, 1, dims="g") + pm.Normal("b", 0, 1) * np.arange(4.0)
            ),
        ),
        ("a fixed scale 5*z", lambda: pm.Deterministic("d", 5.0 * pm.Normal("z", 0, 1, dims="g"))),
        (
            "a non-standard aux",
            lambda: pm.Deterministic(
                "d", pm.HalfNormal("s", 1) * pm.Normal("z", 1.0, 1.0, dims="g")
            ),
        ),
    ],
)
def test_near_misses_are_not_non_centered(label, build):
    assert not any(r == "noncentered/1" for r, _ in _single(build)), label


def test_an_aux_with_two_consumers_is_not_non_centering():
    def build():
        z = pm.Normal("z", 0, 1, dims="g")
        s = pm.HalfNormal("s", 1)
        pm.Deterministic("d1", s * z, dims="g")
        pm.Deterministic("d2", z + 1.0, dims="g")

    assert not any(r == "noncentered/1" for r, _ in _single(build))


def test_negative_shrinkage_lookalikes():
    assert not any(r == "shrinkage/1" for r, *_ in _rules(STRUCTURE_MODELS["ridge"]))
    assert ("centered/1", "beta", "partial pooling · centered") in _rules(STRUCTURE_MODELS["ridge"])
    ss = _rules(STRUCTURE_MODELS["spike_and_slab"])
    assert not any(r in ("shrinkage/1", "noncentered/1") for r, *_ in ss)
    plain = _rules(STRUCTURE_MODELS["horseshoe"])
    assert not any("regularized" in t for *_, t in plain)  # no slab: never "regularized"


def test_same_plate_and_noise_parents_are_not_hierarchies():
    with pm.Model(coords={"g": range(4)}) as m:
        mu = pm.Normal("mu", 0, 1, dims="g")
        pm.Normal("x", mu, 1.0, dims="g")  # a same-plate parent
        s = pm.HalfNormal("s", 1)
        pm.Normal("y", 0, s, observed=np.zeros(4), dims="g")  # observed: likelihood noise
    assert not to_ir(m).blocks


def test_nonlinear_predictors_are_not_regressions():
    with pm.Model(coords={"obs": range(10)}) as m:
        a = pm.Normal("a", 0, 1)
        b = pm.Normal("b", 0, 1)
        x = pm.Data("x", np.arange(10.0), dims="obs")
        pm.Normal("y", pt.exp(a * x) * b, 1.0, observed=np.zeros(10), dims="obs")
    assert not any(b.rule == "regression/1" for b in to_ir(m).blocks)


def test_recognition_ignores_names_and_operand_order():
    def build(names, flip):
        a, s, z, d = names
        with pm.Model(coords={"g": range(4)}) as m:
            mu = pm.Normal(a, 0, 5)
            tau = pm.HalfNormal(s, 5)
            eta = pm.Normal(z, 0, 1, dims="g")
            pm.Deterministic(d, (eta * tau + mu) if flip else (mu + tau * eta), dims="g")
        return m

    for names, flip in ((("mu", "tau", "eta", "theta"), False), (("p", "q", "r", "w"), True)):
        (b,) = to_ir(build(names, flip)).blocks
        assert (
            b.rule == "noncentered/1"
            and b.anchor == names[3]
            and b.members
            == [
                names[2],
                names[3],
            ]
        )


def test_implied_glyph_replaces_the_transfer_curve():
    ir = to_ir(STRUCTURE_MODELS["eight_schools_noncentered"]())
    theta = ir.node("theta")
    assert theta.glyph.source == "implied_family" and theta.glyph_data["exact"]
    assert theta.tag == "partial pooling · non-centered"


def test_horseshoe_glyph_is_the_exact_marginal():
    ir = to_ir(STRUCTURE_MODELS["horseshoe"]())
    beta = ir.node("beta")
    d = beta.glyph_data
    assert beta.glyph.source == "implied_family" and d["exact"] and d["clipped"]
    ys = np.asarray(d["ys"])
    assert ys.argmax() in (31, 32)  # the spike sits at 0 (grid symmetric around it)
    assert ys[0] > 0.005  # heavy tails: still visibly above zero at |beta| = 4


def test_blocks_can_be_switched_off():
    ir = to_ir(STRUCTURE_MODELS["eight_schools_noncentered"](), blocks=False)
    assert not ir.blocks and ir.node("theta").tag is None


# ------------------------------------------------------------------------------ annotations


def test_annotation_validation_and_priority():
    ir = to_ir(STRUCTURE_MODELS["bridge_measurement_like"]())
    spec = bayesdag.BlockSpec(
        kind="measurement",
        title="bridge measurement model",
        members=["bridge_marginal", "study_censored", "y_above"],
        roles={"noise": ["assay_sd"]},
        notes=["likelihood: marginal over tubes, censored study cells"],
        workflow="joint",
    )
    out = blocks.apply(ir, [spec])
    b = out.block("block:bridge-measurement-model")
    assert b.provenance == "annotated" and b.workflow == "joint"
    assert ir.block("block:bridge-measurement-model") is None  # the input is not mutated
    schema.check(out)
    with pytest.raises(ValueError, match="unknown variable"):
        blocks.apply(ir, [bayesdag.BlockSpec("custom", "x", members=["nope"])])
    with pytest.raises(ValueError, match="claimed by both"):
        blocks.apply(
            ir,
            [
                bayesdag.BlockSpec("custom", "a", members=["y_above"]),
                bayesdag.BlockSpec("custom", "b", members=["y_above"]),
            ],
        )
    # an annotation owning a recognized block's member wins
    over = blocks.apply(
        ir, [bayesdag.BlockSpec("custom", "mine", members=["tube_spread", "tube_spread_z"])]
    )
    assert over.block("block:hierarchy:tube_spread") is None


# ------------------------------------------------------------------------------ overview


def test_overview_contracts_a_noncentered_hierarchy():
    ir = to_ir(STRUCTURE_MODELS["eight_schools_noncentered"]())
    ov = blocks.overview(ir)
    assert [n.id for n in ov.nodes] == ["mu", "tau", "block:hierarchy:theta", "y_obs"]
    blk = ov.node("block:hierarchy:theta")
    assert blk.role == "block" and blk.tag == "partial pooling · non-centered"
    edges = {(e.source, e.target): e for e in ov.edges}
    assert edges["mu", "block:hierarchy:theta"].target_token_id == "mu"  # the implied line
    assert edges["block:hierarchy:theta", "y_obs"].via == [["theta", "y_obs"]]
    # every canonical edge is either internal to the block or carried by a display edge
    carried = {tuple(v) for e in ov.edges for v in e.via}
    internal = {("eta", "theta")}
    assert carried | internal == {(e.source, e.target) for e in ir.edges}
    schema.check(ov)
    assert ir.node("theta") is not None  # canonical untouched
    assert blocks.overview(ir, expand={"block:hierarchy:theta"}) is ir


def test_overview_never_creates_a_cycle():
    """z -> theta -> w -> theta2 with z's block {z, theta2}? Not convex: contraction refused."""
    from bayesdag.ir import BlockIR, EdgeIR, ModelIR, NodeIR

    ir = ModelIR(
        nodes=[NodeIR(id=x, role="latent") for x in ("a", "b", "c")],
        edges=[EdgeIR("a", "b"), EdgeIR("b", "c")],
        blocks=[BlockIR(id="k", kind="custom", title="k", members=["a", "c"])],
    )
    assert blocks.overview(ir) is ir


def test_overview_renders_with_unique_ids():
    import xml.etree.ElementTree as ET

    from bayesdag.layout import layout
    from bayesdag.render_svg import to_svg

    ov = blocks.overview(to_ir(STRUCTURE_MODELS["regularized_horseshoe_codata"]()))
    svg = to_svg(ov, layout(ov))
    root = ET.fromstring(svg)
    ids = [el.get("id") for el in root.iter() if el.get("id")]
    assert len(ids) == len(set(ids))
    assert "regularized horseshoe · half-t(2) locals" in svg


# ------------------------------------------------------------------------------ the view


def test_view_defaults_to_the_overview_and_switches_in_place():
    v = bayesdag.view(STRUCTURE_MODELS["eight_schools_noncentered"](), ppc_draws=0)
    assert v.detail == "overview"
    assert "block:hierarchy:theta" in {n.id for n in v.display_ir.nodes}
    assert {n.id for n in v.ir.nodes} >= {"eta", "theta"}  # canonical IR untouched
    full = v.to_svg(detail="full")
    assert 'data-node="eta"' in full and 'data-node="eta"' not in v.to_svg()
    v.set_detail("full")
    assert v.to_svg() == full
    v.set_detail("overview").expand("block:hierarchy:theta")
    assert 'data-node="eta"' in v.to_svg()
    v.collapse("block:hierarchy:theta")
    assert 'data-node="eta"' not in v.to_svg()
    with pytest.raises(ValueError):
        v.expand("block:nope")
    with pytest.raises(ValueError):
        bayesdag.view(STRUCTURE_MODELS["eight_schools_noncentered"](), detail="compact")


def test_view_without_blocks_and_with_annotations():
    m = STRUCTURE_MODELS["eight_schools_noncentered"]()
    off = bayesdag.view(m, blocks=None, ppc_draws=0)
    assert not off.blocks and off.display_ir is off.ir
    spec = bayesdag.BlockSpec("custom", "the school effects", members=["eta", "theta"])
    ann = bayesdag.view(m, blocks=None, annotations=[spec], ppc_draws=0)
    (b,) = ann.blocks
    assert b.provenance == "annotated"
    svg = ann.to_svg()
    assert "the school effects" in svg and "annotated" in svg  # the legend says so


def test_widget_block_selection_and_expansion_round_trip():
    pytest.importorskip("anywidget")
    v = bayesdag.view(STRUCTURE_MODELS["eight_schools_noncentered"](), ppc_draws=0)
    w = v.widget()
    spec = w.spec
    assert spec["detail"] == "overview" and spec["has_overview"]
    blk = spec["blocks"]["block:hierarchy:theta"]
    assert blk["contracted"] and blk["display_id"] == "block:hierarchy:theta"
    assert spec["nodes"]["eta"]["hidden"]  # members still get cards
    assert spec["nodes"]["block:hierarchy:theta"]["role"] == "block"
    w.expanded_blocks = ["block:hierarchy:theta"]  # the JS clicked "expand"
    assert 'data-node="eta"' in w.spec["svg"]
    assert w.spec["nodes"]["eta"]["in_block"] == "block:hierarchy:theta"
    w.detail = "full"
    assert v.detail == "full"
    w.panel_request = {"kind": "block", "block_id": "block:hierarchy:theta"}
    assert w.panel_response["status"] == "ready" and "partial pooling" in w.panel_response["svg"]
    w.panel_request = {"kind": "nope", "node_id": "theta"}
    assert w.panel_response["status"] == "error"
    assert w.selected_node == ""  # a block id never lands in selected_node


def test_update_keeps_the_overview_still():
    import xarray as xr

    m = STRUCTURE_MODELS["eight_schools_noncentered"]()
    v = bayesdag.view(m, ppc_draws=0)
    boxes = {k: (b.x, b.y, b.w, b.h) for k, b in v.layout.node_boxes.items()}
    rng = np.random.default_rng(0)
    post = {
        "mu": (("chain", "draw"), rng.normal(size=(2, 50))),
        "tau": (("chain", "draw"), np.abs(rng.normal(size=(2, 50)))),
        "eta": (("chain", "draw", "school"), rng.normal(size=(2, 50, 8))),
        "theta": (("chain", "draw", "school"), rng.normal(size=(2, 50, 8))),
    }
    idata = xr.DataTree.from_dict({"posterior": xr.Dataset(post)})
    v.update(idata)
    blk = v.display_ir.node("block:hierarchy:theta")
    assert blk.glyph.kind == "multi"  # the block shows its anchor's posterior
    assert {k: (b.x, b.y, b.w, b.h) for k, b in v.layout.node_boxes.items()} == boxes
    assert "posterior per element" in v.panel_svg("marginals", block_id="block:hierarchy:theta")


@pytest.mark.parametrize("name", list(STRUCTURE_MODELS))
def test_every_structure_model_renders_in_both_details(name):
    """Every fixture converts, lays out and renders as the overview AND the full graph, with
    resolvable references and unique element ids — and the overview's display edges account
    for every canonical edge."""
    import xml.etree.ElementTree as ET

    v = bayesdag.view(STRUCTURE_MODELS[name](), ppc_draws=0, hide_index_data=False)
    for detail in ("overview", "full"):
        svg = v.to_svg(detail=detail)
        root = ET.fromstring(svg)
        ids = [el.get("id") for el in root.iter() if el.get("id")]
        assert len(ids) == len(set(ids)), (name, detail)
    schema.check(v.ir)
    schema.check(v.display_ir)
    carried = {tuple(p) for e in v.display_ir.edges for p in (e.via or [[e.source, e.target]])}
    hidden = {m for b in blocks.contracting(v.ir) for m in b.members}
    internal = {(e.source, e.target) for e in v.ir.edges if {e.source, e.target} <= hidden}
    assert carried | internal >= {(e.source, e.target) for e in v.ir.edges}
