"""Special / awkward PyMC constructs: the adapter must render them honestly and NEVER crash.

Grows alongside the distribution-coverage work; for now it pins the robustness contract that a
non-samplable RV (whose shape can't be eval'd) degrades gracefully instead of breaking `to_ir`.
"""

import numpy as np
import pymc as pm
import pytest

from bayesdag.convert import to_ir
from bayesdag.layout import layout
from bayesdag.render_svg import to_svg

_W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])


@pytest.mark.parametrize(
    "name,build",
    [
        ("Flat", lambda: pm.Flat("x")),
        ("HalfFlat", lambda: pm.HalfFlat("x")),
        ("ICAR", lambda: pm.ICAR("x", W=_W)),
    ],
)
def test_non_samplable_rvs_do_not_crash_to_ir(name, build):
    """Flat/HalfFlat/ICAR raise on shape `.eval()` (can't sample) — pymc's get_plates would blow
    up. The adapter must catch that, still build the IR, and recover plates eval-free from coords."""
    with pm.Model(coords={"r": [0, 1, 2]}) as m:
        build()
        pm.Normal("y", 0, 1, observed=np.array([1.0, 2.0, 3.0]), dims="r")
    ir = to_ir(m)
    assert {n.id for n in ir.nodes} == {"x", "y"}
    assert any(p.id == "plate_r" for p in ir.plates)  # the real plate survives the fallback
    assert "<svg" in to_svg(ir, layout(ir))  # full render succeeds


# (build, expected glyph kind, expected representable) for the special-construct handlers
_SPECIAL = [
    (
        "MvNormal2",
        lambda: pm.MvNormal("x", mu=[0, 0], cov=np.array([[1.0, 0.7], [0.7, 1.0]])),
        "pairplot",
        True,
    ),
    ("MvNormal6", lambda: pm.MvNormal("x", mu=np.zeros(6), cov=np.eye(6)), "heatmap", True),
    ("Wishart", lambda: pm.Wishart("x", nu=4, V=np.eye(3)), "heatmap", True),
    ("Dirichlet", lambda: pm.Dirichlet("x", a=[2.0, 3.0, 5.0]), "simplex", True),
    (
        "Censored",
        lambda: pm.Censored("x", pm.Normal.dist(0, 1), lower=-1, upper=2),
        "censored",
        True,
    ),
    (
        "TruncatedNormal",
        lambda: pm.TruncatedNormal("x", mu=0, sigma=1, lower=-1, upper=2),
        "density",
        True,
    ),
    (
        "GRW",
        lambda: pm.GaussianRandomWalk("x", sigma=1, init_dist=pm.Normal.dist(0, 1), steps=8),
        "fan",
        True,
    ),
    (
        "AR",
        lambda: pm.AR("x", rho=[0.5], sigma=1, init_dist=pm.Normal.dist(0, 1), steps=8),
        "stem",
        True,
    ),
    (
        "LKJCorr",
        lambda: pm.LKJCorr("x", n=3, eta=2),
        "density",
        True,
    ),  # marginal correlation density
    (
        "Interpolated",
        lambda: pm.Interpolated(
            "x",
            x_points=np.linspace(-3, 3, 40),
            pdf_points=np.exp(-(np.linspace(-3, 3, 40) ** 2) / 2),
        ),
        "density",
        True,
    ),
    (
        "NormalMixture",
        lambda: pm.NormalMixture("x", w=[0.5, 0.5], mu=[-2, 2], sigma=[1, 1]),
        "mixture",
        True,
    ),
    ("ZeroInflatedPoisson", lambda: pm.ZeroInflatedPoisson("x", psi=0.7, mu=3), "mixture", True),
    # honest badges (undrawable as a single static shape)
    (
        "GARCH11",
        lambda: pm.GARCH11("x", omega=0.1, alpha_1=0.1, beta_1=0.8, initial_vol=1, steps=8),
        "schematic",
        False,
    ),
    ("Flat", lambda: pm.Flat("x"), "schematic", False),
]


@pytest.mark.parametrize("name,build,kind,representable", _SPECIAL)
def test_special_construct_glyphs(name, build, kind, representable):
    with pm.Model() as m:
        build()
    n = to_ir(m).node("x")
    assert n.glyph is not None and n.glyph.kind == kind, name
    assert n.representable is representable, name
    assert (n.elision_reason is None) is representable, name


def test_constructs_with_a_prior_subparam_keep_their_deterministic_glyph():
    """A construct needing only a NUMERIC SUBSET of params keeps its real, deterministic glyph even
    when a trailing param is a prior: an LKJCholeskyCov's correlation marginal needs only n & eta (not
    its sd_dist prior), and a driftless random walk's normalized fan is scale-invariant (the prior
    innovation scale cancels). A construct whose shape GENUINELY depends on a prior — an MvNormal whose
    covariance is the LKJ draw — honestly badges instead of fabricating a random pairplot."""

    def lkj_chol():
        with pm.Model() as m:
            pm.LKJCholeskyCov(
                "L", n=3, eta=2.0, sd_dist=pm.Exponential.dist(1.0), compute_corr=True
            )
        return m

    def driftless_rw(init=None):
        with pm.Model(coords={"t": range(8)}) as m:
            pm.GaussianRandomWalk(
                "w",
                sigma=pm.HalfNormal("s", 1.0),
                init_dist=init if init is not None else pm.DiracDelta.dist(0.0),
                dims="t",
            )
        return m

    def mvn_lkj_cov():
        with pm.Model() as m:
            chol, _, _ = pm.LKJCholeskyCov(
                "C", n=2, eta=2.0, sd_dist=pm.Exponential.dist(1.0), compute_corr=True
            )
            pm.MvNormal("ab", mu=[0.0, 0.0], chol=chol)
        return m

    lkj = to_ir(lkj_chol()).node("L")
    assert (
        lkj.glyph.kind == "density" and lkj.glyph.source == "prior_analytic"
    )  # not re-broken to a badge
    assert (
        to_ir(lkj_chol()).node("L").glyph_data == lkj.glyph_data
    )  # deterministic: sd_dist never sampled

    rw = to_ir(driftless_rw()).node("w")
    assert rw.glyph.kind == "fan" and rw.glyph.source == "conditional_standardized"
    assert (
        to_ir(driftless_rw()).node("w").glyph_data == rw.glyph_data
    )  # deterministic: scale-invariant

    # a spread-out start does NOT scale with the (unknown) innovation sd: the band's shape is
    # genuinely unknown, so no fan — a caption says why
    spread = to_ir(driftless_rw(pm.Normal.dist(0, 1))).node("w")
    assert (
        spread.glyph is None
        and spread.representable
        and "depends on parents" in (spread.glyph_note or "")
    )

    ab = to_ir(mvn_lkj_cov()).node("ab")
    assert (
        ab.glyph.kind == "schematic" and ab.representable is False
    )  # covariance is a genuine prior


def test_ar_pacf_real_only_when_coefficients_are_known():
    """Fixed AR coefficients -> the true theoretical PACF; prior coefficients -> an honest, DETERMINISTIC
    order schematic (no random per-render draw of the unknown coefficients)."""

    def fixed():
        with pm.Model(coords={"t": range(8)}) as m:
            pm.AR(
                "level",
                rho=[0.6, 0.2],
                sigma=0.4,
                init_dist=pm.Normal.dist(0, 1),
                constant=False,
                dims="t",
            )
        return m

    def prior():
        with pm.Model(coords={"t": range(8)}) as m:
            rho = pm.Normal("rho", 0, 0.5, shape=2)
            pm.AR(
                "level",
                rho=rho,
                sigma=pm.HalfNormal("sigma", 1),
                init_dist=pm.Normal.dist(0, 1),
                constant=False,
                dims="t",
            )
        return m

    f = to_ir(fixed()).node("level")
    assert f.glyph.kind == "stem" and f.glyph.source == "prior_analytic"
    assert [round(v, 2) for v in f.glyph_data["values"][:2]] == [0.75, 0.2]  # real PACF of AR(2)

    # prior coefficients: no stem heights are knowable, so none are drawn; the ORDER is
    # structural and is stated in words
    pn = to_ir(prior()).node("level")
    assert pn.glyph is None and pn.representable
    assert pn.glyph_note == "AR(2): partial autocorrelation cuts off after lag 2"


def test_ar_non_stationary_keeps_its_honesty_badge():
    """Known coefficients with sum(rho^2) >= 1 have no stationary PACF — the node must carry
    the specific elision reason, not a bare schematic with no badge."""
    with pm.Model(coords={"t": range(8)}) as m:
        pm.AR(
            "level",
            rho=[0.9, 0.6],
            sigma=0.4,
            init_dist=pm.Normal.dist(0, 1),
            constant=False,
            dims="t",
        )
    n = to_ir(m).node("level")
    assert n.glyph.kind == "schematic"
    assert n.elision_reason == "autoregressive — non-stationary"


def test_special_constructs_render_without_error():
    """Every special construct must compose into a valid SVG (badge or glyph), never crash."""
    for _name, build, _k, _r in _SPECIAL:
        with pm.Model() as m:
            build()
        ir = to_ir(m)
        assert "<svg" in to_svg(ir, layout(ir))


def test_glyph_and_label_failures_never_crash_to_ir(monkeypatch):
    """The honesty contract's hard floor: a glyph is optional, a label is cosmetic, but `to_ir`
    must survive. `glyph_for` fans out to every shape/construct provider and `render_value` walks
    arbitrary op graphs, so both are guarded at the call site rather than trusting every branch."""
    from bayesdag.adapters import pymc as adapter

    def boom(*_a, **_k):
        raise RuntimeError("provider exploded")

    with pm.Model(coords={"g": [0, 1, 2]}) as m:
        mu = pm.Normal("mu", 0, 1)
        pm.Deterministic("d", mu * 2.0)
        pm.Normal("y", mu, 1.0, observed=np.array([1.0, 2.0, 3.0]), dims="g")

    monkeypatch.setattr(adapter, "glyph_for", boom)
    monkeypatch.setattr(adapter, "render_value", boom)
    ir = to_ir(m)

    assert {n.id for n in ir.nodes} == {"mu", "d", "y"}
    assert all(n.glyph is None for n in ir.nodes)  # degraded to no glyph, not a crash
    assert "<svg" in to_svg(ir, layout(ir))  # and still renders


def test_random_walk_fan_only_trusts_a_normal_innovation():
    """The fan is `drift*t ± 2*sd*sqrt(t)`, read positionally from the innovation's params. That
    layout is only known for a Normal ([mu, sigma]); a StudentT is [nu, mu, sigma], so reading
    slot 0 as the drift would tilt the fan by the degrees of freedom. Non-Normal -> honest badge."""
    with pm.Model(coords={"t": range(8)}) as m:
        pm.RandomWalk(
            "w",
            init_dist=pm.Normal.dist(0, 1),
            innovation_dist=pm.StudentT.dist(nu=4.0, mu=0.0, sigma=1.0),
            dims="t",
        )
    n = to_ir(m).node("w")
    assert n.glyph is None and n.representable
    assert "StudentT innovations" in n.glyph_note

    with pm.Model(coords={"t": range(8)}) as m2:  # the Normal case still draws its fan
        pm.GaussianRandomWalk("w", mu=0.5, sigma=1.0, init_dist=pm.Normal.dist(0, 1), dims="t")
    fan = to_ir(m2).node("w")
    assert fan.glyph.kind == "fan" and fan.glyph.source == "prior_analytic"
    assert fan.glyph_data["mid"][0] < fan.glyph_data["mid"][-1]  # positive drift tilts the fan up


def test_eval_safety_gate_fails_closed(monkeypatch):
    """`_depends_on_rv` is the SINGLE gate in front of every `.eval()` in the adapters, so it must
    fail CLOSED. If the pytensor traversal API moves (it has moved once — hence the import shim),
    an unresolvable graph walk must report "prior-governed", degrading the node to a schematic.
    Failing open would `.eval()` through parent RVs and badge a random draw as an analytic prior."""
    import sys
    import types

    import pytensor.tensor as pt

    from bayesdag.adapters import constructs
    from bayesdag.adapters.glyph_data import glyph_for

    const = pt.as_tensor_variable(3.0)
    assert constructs._depends_on_rv(const) is False  # baseline: a constant reads as constant

    with pm.Model() as m:
        mu = pm.Normal("mu", 0.0, 5.0)  # a genuine ROOT prior -> normally an analytic density
    spec, _data, _elision = glyph_for(mu, "latent", "Normal", m)
    assert spec.source == "prior_analytic"  # baseline

    # break BOTH import locations of `ancestors`
    for mod in ("pytensor.graph.traversal", "pytensor.graph.basic"):
        monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))

    assert constructs._depends_on_rv(const) is True  # unresolvable -> assume prior-governed
    spec, _data, _elision = glyph_for(mu, "latent", "Normal", m)
    assert spec.source == "prior_family_only"  # degraded to the schematic, not a fabricated prior


def test_unclassifiable_op_is_not_taken_for_a_mixture_component():
    """The other direction of the same gate: component SELECTION must fail to "not a component"
    (-> honest badge), because a leaked non-RV input would draw a wrong glyph."""
    from bayesdag.adapters import constructs

    assert constructs._op_is_rv(object(), unknown=False) is False
    assert (
        constructs._op_is_rv(object(), unknown=True) is False
    )  # classifiable: plain object is no RV


def test_bart_states_its_structure_with_clean_label():
    """BART (sum-of-trees) has no closed-form prior density. Any drawn step function would be
    made-up heights, so the structure is stated in a caption instead; the label elides the
    response/tree-prior arrays -> `mu ~ BART(X, m)`."""
    pmb = pytest.importorskip("pymc_bart")
    x = np.linspace(0, 5, 40)
    Y = np.sin(x) + 0.1 * x
    with pm.Model() as m:
        X = pm.Data("X", x[:, None])
        pmb.BART("mu", X=X, Y=Y, m=20)
        pm.Normal("y", mu=m["mu"], sigma=pm.HalfNormal("sigma", 1), observed=Y)
    ir = to_ir(m)
    mu = ir.node("mu")
    assert mu.glyph is None and mu.glyph_note.startswith("sum of trees")
    assert mu.elision_reason is None  # representable: a caption, not an "elided" badge
    assert "arg0" not in mu.label_tex and "BART" in mu.label_tex
    assert r"\cssId{tok-X}" in mu.label_tex and r"\cssId{tok-m}" in mu.label_tex  # X + m shown
    assert ("X", "mu", "X") in [
        (e.source, e.target, e.target_token_id) for e in ir.edges
    ]  # data -> BART edge
    assert "<svg" in to_svg(ir, layout(ir))  # composes without crashing


def _node_of(build):
    with pm.Model() as m:
        build()
    return to_ir(m).node("x")


@pytest.mark.parametrize("psi", [0.0, 0.3, 1.0])
def test_zero_inflation_bar_is_the_true_mass_at_zero(psi):
    """The zero bar used to be a fixed 0.85 whatever the weight. It must be the exact mixture
    pmf: P(0) = (1 - psi) + psi * Poisson(0 | mu)."""
    import scipy.stats as st

    n = _node_of(lambda: pm.ZeroInflatedPoisson("x", psi=psi, mu=3.0))
    base = n.glyph_data["base"]
    cats = np.array(base["cats"])
    truth = (1 - psi) * (cats == 0) + psi * st.poisson(3.0).pmf(cats)
    assert np.allclose(np.array(base["heights"]), truth / truth.max())
    assert "spike" not in n.glyph_data


def test_point_mass_mixture_keeps_its_location():
    n = _node_of(
        lambda: pm.Mixture(
            "x", w=[0.4, 0.6], comp_dists=[pm.DiracDelta.dist(2), pm.Poisson.dist(3.0)]
        )
    )
    b = n.glyph_data["base"]
    assert b["cats"][int(np.argmax(b["heights"]))] == 2  # the spike is AT 2, not at 0


def test_mixture_with_prior_weights_says_it_is_components_only():
    def build():
        w = pm.Dirichlet("w", np.ones(2))
        pm.Mixture("x", w=w, comp_dists=[pm.Normal.dist(0, 1), pm.Normal.dist(3, 1)])

    n = _node_of(build)
    assert n.glyph.kind == "mixture" and n.glyph_data["weighted"] is False
    assert "weights depend on parents" in n.glyph_note


@pytest.mark.parametrize(
    "rho,stationary",
    [
        ([0.9, -0.5], True),
        ([0.8, -0.8], True),
        ([0.8, 0.3], False),  # sum(rho^2) < 1 but a root lies outside the unit circle
        ([0.5, 0.5], False),  # unit root
    ],
)
def test_ar_stationarity_uses_the_lag_polynomial(rho, stationary):
    n = _node_of(lambda: pm.AR("x", rho=rho, sigma=1, init_dist=pm.Normal.dist(0, 1), steps=8))
    if stationary:
        assert n.glyph.kind == "stem" and n.elision_reason is None
    else:
        assert n.elision_reason == "autoregressive — non-stationary"


def test_ar_intercept_is_not_read_as_a_coefficient():
    """With constant=True, rho[0] is the intercept: a large intercept must not make a
    stationary AR(1) look non-stationary."""
    n = _node_of(
        lambda: pm.AR(
            "x", rho=[5.0, 0.5], constant=True, sigma=1, init_dist=pm.Normal.dist(0, 1), steps=8
        )
    )
    assert n.glyph.kind == "stem"
    assert round(n.glyph_data["values"][0], 6) == 0.5


def test_censored_density_lives_inside_the_bounds():
    import scipy.stats as st

    n = _node_of(lambda: pm.Censored("x", pm.Normal.dist(0, 1), lower=-1, upper=2))
    d = n.glyph_data
    assert min(d["xs"]) >= -1 and max(d["xs"]) <= 2
    ps = sorted(sp["p"] for sp in d["spikes"])
    assert np.allclose(ps, sorted([st.norm.cdf(-1), st.norm.sf(2)]))


def test_one_sided_censoring():
    n = _node_of(lambda: pm.Censored("x", pm.Normal.dist(0, 1), lower=0.5, upper=None))
    assert len(n.glyph_data["spikes"]) == 1 and min(n.glyph_data["xs"]) >= 0.5


def test_random_walk_band_includes_the_initial_spread():
    """Var[X_t] = v0 + t*v: a wide initial distribution must widen the band at t = 0."""

    def width0(sd0):
        n = _node_of(
            lambda: pm.GaussianRandomWalk(
                "x", mu=0.0, sigma=1.0, init_dist=pm.Normal.dist(0, sd0), steps=9
            )
        )
        d = n.glyph_data
        return (d["hi"][0] - d["lo"][0]) / (d["hi"][-1] - d["lo"][-1])

    assert width0(0.0001) < 0.01  # a near-point start: the band opens from nothing
    assert width0(3.0) > 0.5  # var0 = 9 against var_9 = 18


def test_batched_dirichlet_with_differing_rows_is_not_one_glyph():
    n = _node_of(lambda: pm.Dirichlet("x", a=np.array([[1.0, 2.0, 3.0], [5.0, 1.0, 1.0]])))
    assert n.glyph is None and n.glyph_note == "elements differ"


def test_batched_mvnormal_reads_the_covariance_slot():
    """A batched MEAN is matrix-shaped too; the covariance must come from its own slot."""
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    n = _node_of(lambda: pm.MvNormal("x", mu=np.zeros((4, 2)), cov=cov, shape=(4, 2)))
    assert n.glyph.kind == "pairplot" and np.allclose(n.glyph_data["cov"], cov)
