"""Graph fidelity: every dependency survives conversion, ids are unambiguous, and references
resolve (schema.check)."""

import numpy as np
import pymc as pm
import pytest

from bayesdag import schema
from bayesdag.convert import subgraph, to_ir
from bayesdag.ir import AuxViewIR, BlockIR, EdgeIR, ModelIR, NodeIR, PlateIR


def test_a_parent_feeding_two_slots_keeps_both():
    """`Normal(m, m/10)`: m feeds loc AND scale. The old adapter kept only the first slot; the
    edge now carries both and anchors at the node boundary instead of one token."""
    with pm.Model() as m:
        mu = pm.HalfNormal("m", 1)
        pm.Normal("x", mu, mu * 0.1)
    e = next(e for e in to_ir(m).edges if (e.source, e.target) == ("m", "x"))
    assert e.target_token_ids == ["loc", "scale"]
    assert e.target_token_id is None


def test_single_slot_edges_keep_their_anchor(eight_schools_ir):
    e = next(e for e in eight_schools_ir.edges if (e.source, e.target) == ("tau", "theta"))
    assert e.target_token_id == "tau" and e.target_token_ids == ["tau"]


def test_unnamed_shapes_of_different_lengths_get_distinct_plates():
    with pm.Model() as m:
        pm.Normal("a", 0, 1, shape=3)
        pm.Normal("b", 0, 1, shape=4)
    ir = to_ir(m)
    ids = sorted(p.id for p in ir.plates)
    assert ids == ["plate_len3", "plate_len4"]
    assert all(p.source == "shape" for p in ir.plates)
    schema.check(ir)


def test_named_plates_record_dims_and_lengths(eight_schools_ir):
    (p,) = eight_schools_ir.plates
    assert p.id == "plate_school" and p.dims == ["school"] and p.lengths == [8]
    assert p.source == "named"


def test_colliding_plate_signatures_get_a_suffix():
    from bayesdag.adapters.pymc import _plate_id

    used: set[str] = set()
    assert _plate_id(("a_b",), (2,), used) == "plate_a_b"
    assert _plate_id(("a", "b"), (2, 3), used) == "plate_a_b~2"


def test_subgraph_drops_dangling_panels_and_blocks(eight_schools_ir):
    import dataclasses

    ir = dataclasses.replace(
        eight_schools_ir,
        aux_views=[AuxViewIR(kind="joint", vars=["theta", "tau"], edge=["tau", "theta"])],
        blocks=[BlockIR(id="b", kind="hierarchy", title="t", anchor="theta", members=["eta"])],
    )
    sub = subgraph(ir, ["y_obs"])  # keeps y_obs + theta; tau and eta leave
    assert sub.aux_views == [] and sub.blocks == []
    schema.check(sub)


def _bad(**kw):
    base = {
        "nodes": [NodeIR(id="a", role="latent"), NodeIR(id="b", role="latent")],
        "edges": [EdgeIR("a", "b")],
        "plates": [],
    }
    base.update(kw)
    return ModelIR(**base)


@pytest.mark.parametrize(
    "ir,msg",
    [
        (_bad(nodes=[NodeIR(id="a", role="latent")] * 2, edges=[]), "duplicate node"),
        (_bad(edges=[EdgeIR("a", "zz")]), "unknown node"),
        (_bad(edges=[EdgeIR("a", "b"), EdgeIR("a", "b")]), "duplicate edges"),
        (_bad(plates=[PlateIR(id="p", label="p", members=["q"])]), "unknown members"),
        (
            _bad(
                plates=[
                    PlateIR(id="p", label="p", parent="q"),
                    PlateIR(id="q", label="q", parent="p"),
                ]
            ),
            "parent cycle",
        ),
        (_bad(blocks=[BlockIR(id="k", kind="x", title="x", members=["nope"])]), "block k"),
        (_bad(aux_views=[AuxViewIR(kind="joint", vars=["nope"])]), "aux view"),
    ],
)
def test_check_rejects_dangling_references(ir, msg):
    with pytest.raises(ValueError, match=msg):
        schema.check(ir)


def test_check_rejects_an_unknown_token(eight_schools_ir):
    d = eight_schools_ir.to_dict()
    d["edges"][0]["target_token_id"] = "nonexistent"
    with pytest.raises(ValueError, match="unknown token"):
        to_ir(d)


def test_real_models_pass_the_check(eight_schools_ir):
    schema.check(eight_schools_ir)
    with pm.Model(coords={"g": range(3)}) as m:
        s = pm.HalfNormal("s", 1)
        z = pm.Normal("z", 0, 1, dims="g")
        pm.Normal("y", z * s, s, observed=np.zeros(3), dims="g")
    schema.check(to_ir(m))


def _radon(**kw):
    rng = np.random.default_rng(0)
    cidx = rng.integers(0, 4, 20)
    with pm.Model(coords={"county": range(4), "obs": range(20)}) as m:
        a = pm.Normal("a", 0, 1, dims="county")
        ci = pm.Data("county_idx", cidx, dims="obs")
        fl = pm.Data("floor", rng.integers(0, 2, 20).astype(float), dims="obs")
        pm.Normal("y", a[ci] + 0.5 * fl, 1.0, observed=rng.normal(size=20), dims="obs")
    return m


def test_index_only_data_is_marked_and_hidden_by_default():
    import bayesdag
    from bayesdag.convert import drop_index_data

    ir = to_ir(_radon())
    assert ir.node("county_idx").usage == "index"
    assert ir.node("floor").usage is None  # used arithmetically
    shown = drop_index_data(ir)
    assert shown.node("county_idx") is None and shown.node("floor") is not None
    assert r"\mathrm{county\_idx}" in shown.node("y").label_tex  # the name is still on the page
    schema.check(shown)
    assert bayesdag.view(_radon(), ppc_draws=0).ir.node("county_idx") is None
    assert bayesdag.view(_radon(), ppc_draws=0, hide_index_data=False).ir.node("county_idx")


def test_integer_data_used_arithmetically_is_kept():
    with pm.Model() as m:
        k = pm.Data("k", np.array([1, 2, 3]))
        pm.Normal("y", k * 2.0, 1.0, observed=np.zeros(3))
    assert to_ir(m).node("k").usage is None


def test_multi_dim_plates_nest_under_their_leading_dim():
    with pm.Model(coords={"protein": range(3), "batch": range(2)}) as m:
        pm.Normal("mean", 0, 1, dims="protein")
        pm.Normal("offset", 0, 1, dims=("protein", "batch"))
    ir = to_ir(m)
    inner = next(p for p in ir.plates if p.dims == ["protein", "batch"])
    outer = next(p for p in ir.plates if p.dims == ["protein"])
    assert inner.parent == outer.id and inner.label == "batch (2)"
    schema.check(ir)
    from bayesdag.layout import layout

    res = layout(ir)
    ob, ib = res.plate_boxes[outer.id], res.plate_boxes[inner.id]
    assert ob.x <= ib.x and ib.x + ib.w <= ob.x + ob.w  # drawn inside


def test_no_prefix_plate_means_no_nesting():
    with pm.Model(coords={"protein": range(3), "batch": range(2)}) as m:
        pm.Normal("offset", 0, 1, dims=("protein", "batch"))
    (p,) = to_ir(m).plates
    assert p.parent is None and p.label == "protein (3) x batch (2)"


def test_potentials_show_their_log_density_expression():
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1, shape=3)
        pm.Potential("pen", -(x**2).sum())
    ir = to_ir(m)
    pen = ir.node("pen")
    assert pen.label_tex.startswith(r"\log\psi_{\mathrm{pen}} = ")
    assert r"\cssId{tok-x}" in pen.label_tex
    e = next(e for e in ir.edges if e.target == "pen")
    assert e.target_token_id == "x"
