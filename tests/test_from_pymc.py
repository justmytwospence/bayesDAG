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
