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
    ("MvNormal2", lambda: pm.MvNormal("x", mu=[0, 0], cov=np.array([[1.0, 0.7], [0.7, 1.0]])), "pairplot", True),
    ("MvNormal6", lambda: pm.MvNormal("x", mu=np.zeros(6), cov=np.eye(6)), "heatmap", True),
    ("Wishart", lambda: pm.Wishart("x", nu=4, V=np.eye(3)), "heatmap", True),
    ("Dirichlet", lambda: pm.Dirichlet("x", a=[2.0, 3.0, 5.0]), "simplex", True),
    ("Censored", lambda: pm.Censored("x", pm.Normal.dist(0, 1), lower=-1, upper=2), "censored", True),
    ("TruncatedNormal", lambda: pm.TruncatedNormal("x", mu=0, sigma=1, lower=-1, upper=2), "density", True),
    ("GRW", lambda: pm.GaussianRandomWalk("x", sigma=1, init_dist=pm.Normal.dist(0, 1), steps=8), "fan", True),
    ("AR", lambda: pm.AR("x", rho=[0.5], sigma=1, init_dist=pm.Normal.dist(0, 1), steps=8), "stem", True),
    ("LKJCorr", lambda: pm.LKJCorr("x", n=3, eta=2), "density", True),  # marginal correlation density
    ("Interpolated", lambda: pm.Interpolated("x", x_points=np.linspace(-3, 3, 40), pdf_points=np.exp(-np.linspace(-3, 3, 40) ** 2 / 2)), "density", True),
    ("NormalMixture", lambda: pm.NormalMixture("x", w=[0.5, 0.5], mu=[-2, 2], sigma=[1, 1]), "mixture", True),
    ("ZeroInflatedPoisson", lambda: pm.ZeroInflatedPoisson("x", psi=0.7, mu=3), "mixture", True),
    # honest badges (undrawable as a single static shape)
    ("GARCH11", lambda: pm.GARCH11("x", omega=0.1, alpha_1=0.1, beta_1=0.8, initial_vol=1, steps=8), "schematic", False),
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
            pm.LKJCholeskyCov("L", n=3, eta=2.0, sd_dist=pm.Exponential.dist(1.0), compute_corr=True)
        return m

    def driftless_rw():
        with pm.Model(coords={"t": range(8)}) as m:
            pm.GaussianRandomWalk("w", sigma=pm.HalfNormal("s", 1.0), init_dist=pm.Normal.dist(0, 1), dims="t")
        return m

    def mvn_lkj_cov():
        with pm.Model() as m:
            chol, _, _ = pm.LKJCholeskyCov("C", n=2, eta=2.0, sd_dist=pm.Exponential.dist(1.0), compute_corr=True)
            pm.MvNormal("ab", mu=[0.0, 0.0], chol=chol)
        return m

    lkj = to_ir(lkj_chol()).node("L")
    assert lkj.glyph.kind == "density" and lkj.glyph.source == "prior_analytic"  # not re-broken to a badge
    assert to_ir(lkj_chol()).node("L").glyph_data == lkj.glyph_data  # deterministic: sd_dist never sampled

    rw = to_ir(driftless_rw()).node("w")
    assert rw.glyph.kind == "fan" and rw.glyph.source == "prior_analytic"
    assert to_ir(driftless_rw()).node("w").glyph_data == rw.glyph_data  # deterministic: scale-invariant

    ab = to_ir(mvn_lkj_cov()).node("ab")
    assert ab.glyph.kind == "schematic" and ab.representable is False  # covariance is a genuine prior


def test_ar_pacf_real_only_when_coefficients_are_known():
    """Fixed AR coefficients -> the true theoretical PACF; prior coefficients -> an honest, DETERMINISTIC
    order schematic (no random per-render draw of the unknown coefficients)."""

    def fixed():
        with pm.Model(coords={"t": range(8)}) as m:
            pm.AR("level", rho=[0.6, 0.2], sigma=0.4, init_dist=pm.Normal.dist(0, 1), constant=False, dims="t")
        return m

    def prior():
        with pm.Model(coords={"t": range(8)}) as m:
            rho = pm.Normal("rho", 0, 0.5, shape=2)
            pm.AR("level", rho=rho, sigma=pm.HalfNormal("sigma", 1), init_dist=pm.Normal.dist(0, 1), constant=False, dims="t")
        return m

    f = to_ir(fixed()).node("level")
    assert f.glyph.kind == "stem" and f.glyph.source == "prior_analytic"
    assert [round(v, 2) for v in f.glyph_data["values"][:2]] == [0.75, 0.2]  # real PACF of AR(2)

    runs = [to_ir(prior()).node("level").glyph_data["values"] for _ in range(2)]
    assert runs[0] == runs[1]  # deterministic — not a random coefficient draw
    pn = to_ir(prior()).node("level")
    assert pn.glyph.source == "prior_family_only"  # schematic (order only)
    vals = pn.glyph_data["values"]
    assert vals[0] > 0 and vals[1] > 0 and all(v == 0 for v in vals[2:])  # p=2 lags then cutoff


def test_ar_non_stationary_keeps_its_honesty_badge():
    """Known coefficients with sum(rho^2) >= 1 have no stationary PACF — the node must carry
    the specific elision reason, not a bare schematic with no badge."""
    with pm.Model(coords={"t": range(8)}) as m:
        pm.AR("level", rho=[0.9, 0.6], sigma=0.4, init_dist=pm.Normal.dist(0, 1), constant=False, dims="t")
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
    assert constructs._op_is_rv(object(), unknown=True) is False  # classifiable: plain object is no RV


def test_bart_renders_as_step_function_with_clean_label():
    """BART (sum-of-trees) has no closed-form prior density; its draws are piecewise-constant, so we
    depict it with the canonical STEP-function schematic (honest structure, not the misleading bell),
    and the label elides the response/tree-prior arrays -> `mu ~ BART(X, m)`."""
    pmb = pytest.importorskip("pymc_bart")
    x = np.linspace(0, 5, 40)
    Y = np.sin(x) + 0.1 * x
    with pm.Model() as m:
        X = pm.Data("X", x[:, None])
        pmb.BART("mu", X=X, Y=Y, m=20)
        pm.Normal("y", mu=m["mu"], sigma=pm.HalfNormal("sigma", 1), observed=Y)
    ir = to_ir(m)
    mu = ir.node("mu")
    assert mu.glyph is not None and mu.glyph.kind == "step"
    assert mu.elision_reason is None  # honest canonical depiction, not an "elided" badge
    assert "arg0" not in mu.label_tex and "BART" in mu.label_tex
    assert r"\cssId{tok-X}" in mu.label_tex and r"\cssId{tok-m}" in mu.label_tex  # X + m shown
    assert ("X", "mu", "X") in [(e.source, e.target, e.target_token_id) for e in ir.edges]  # data -> BART edge
    assert "<svg" in to_svg(ir, layout(ir))  # composes without crashing
