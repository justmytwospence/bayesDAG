"""`from_pymc` extraction against the 8-schools model."""


def _node(ir, nid):
    return ir.node(nid)


def test_roles(eight_schools_ir):
    roles = {n.id: n.role for n in eight_schools_ir.nodes}
    assert roles == {
        "mu": "latent",
        "tau": "latent",
        "eta": "latent",
        "theta": "deterministic",
        "y_obs": "observed",
    }


def test_observed_flag(eight_schools_ir):
    assert _node(eight_schools_ir, "y_obs").observed is True
    assert _node(eight_schools_ir, "mu").observed is False


def test_dist_names(eight_schools_ir):
    assert _node(eight_schools_ir, "mu").dist == "Normal"
    assert _node(eight_schools_ir, "tau").dist == "HalfNormal"
    assert _node(eight_schools_ir, "eta").dist == "Normal"
    assert _node(eight_schools_ir, "theta").dist is None  # deterministic


def test_transform_and_unconstrained_key(eight_schools_ir):
    tau = _node(eight_schools_ir, "tau")
    assert tau.transform == "log"
    assert tau.idata_unconstrained_key == "tau_log__"
    assert _node(eight_schools_ir, "mu").transform is None


def test_slot_aware_loc_parent(eight_schools_ir):
    """theta feeds y_obs's *loc* slot (not scale) — the port-edge target."""
    y = _node(eight_schools_ir, "y_obs")
    loc = next(p for p in y.params if p.name == "loc")
    scale = next(p for p in y.params if p.name == "scale")
    assert loc.parents == ["theta"]
    assert scale.parents == []  # sigma is a constant array, no named parent
    edge = next(e for e in eight_schools_ir.edges if e.source == "theta" and e.target == "y_obs")
    assert edge.target_token_id == "loc"


def test_root_priors_have_no_incoming_edges(eight_schools_ir):
    incoming = {e.target for e in eight_schools_ir.edges}
    assert "mu" not in incoming and "tau" not in incoming and "eta" not in incoming


def test_dims_and_plate(eight_schools_ir):
    ir = eight_schools_ir
    assert _node(ir, "eta").dims == ["school"]
    assert _node(ir, "theta").dims == ["school"]
    assert _node(ir, "mu").dims == []
    assert len(ir.plates) == 1
    plate = ir.plates[0]
    assert plate.label == "school (8)"
    assert set(plate.members) == {"eta", "theta", "y_obs"}


def test_provenance(eight_schools_ir):
    meta = eight_schools_ir.meta
    assert meta.source_ppl == "pymc"
    assert meta.schema_version
    assert meta.created_at  # stamped


def test_overlay_refs_name_the_groups_they_can_be_joined_on():
    """`NodeIR.overlays` is the IR's reference-don't-duplicate half: pointers into the user's
    InferenceData rather than copied samples. Nothing consumed it, so nothing checked it was
    right either — and a pointer that names the wrong group is worse than no pointer."""
    import numpy as np
    import pymc as pm
    import xarray as xr

    from bayesdag.convert import to_ir

    rng = np.random.default_rng(0)
    y = rng.normal(size=6)
    idata = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset({"mu": (("chain", "draw"), rng.normal(size=(2, 50)))}),
            "prior": xr.Dataset({"mu": (("chain", "draw"), rng.normal(size=(1, 50)))}),
            "observed_data": xr.Dataset({"y": ("obs", y)}),
        }
    )
    with pm.Model(coords={"obs": range(6)}) as m:
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("y", mu, 1.0, observed=y, dims="obs")

    ir = to_ir(m, idata=idata)
    mu_groups = {o.idata_group for o in ir.node("mu").overlays}
    assert mu_groups == {"posterior", "prior"}  # both present for mu; no observed_data
    assert all(o.var_name == "mu" for o in ir.node("mu").overlays)  # the join key is the node id

    y_overlays = ir.node("y").overlays
    assert [o.idata_group for o in y_overlays] == ["observed_data"]
    assert y_overlays[0].var_dims == ["obs"]
    assert y_overlays[0].sample_dims == ["chain", "draw"]


def test_no_overlays_without_idata(eight_schools_ir):
    assert all(not n.overlays for n in eight_schools_ir.nodes)


def test_overlay_refs_work_on_what_pm_sample_actually_returns():
    """The regression that made `overlays` look merely unused when it was in fact never
    populated: pm.sample returns an xarray DataTree, whose `groups` is a PROPERTY of paths
    ('/posterior'), not a method of names. Calling it raised, and the except turned that into
    "no groups". Both flavours have to work, and the leading slash must not leak into the ref."""
    import numpy as np
    import pymc as pm

    from bayesdag.convert import to_ir

    y = np.random.default_rng(0).normal(size=8)
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("y", mu, 1.0, observed=y)
        idata = pm.sample(
            draws=20,
            tune=20,
            chains=1,
            cores=1,
            random_seed=0,
            progressbar=False,
            compute_convergence_checks=False,
        )

    ir = to_ir(m, idata=idata)
    assert [o.idata_group for o in ir.node("mu").overlays] == ["posterior"]
    assert {o.idata_group for o in ir.node("y").overlays} == {"observed_data"}
    assert all("/" not in o.idata_group for n in ir.nodes for o in n.overlays)
