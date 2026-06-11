"""Label engine: symbols, distribution templates, assembly, and the 8-schools labels."""

import numpy as np
import pymc as pm
import pytest

from bayesdag import labels, mathsvg
from bayesdag.convert import to_ir
from bayesdag.labels import dist_symbol, symbol_for


def _label(model, node_id):
    return next(n.label_tex for n in to_ir(model).nodes if n.id == node_id)


def test_label_polish():
    """Regression for the rendering warts: reciprocal->fraction, rounded numbers, infinite bounds,
    and elision of auto-generated nested-op (matrix plumbing) deterministics."""
    with pm.Model(coords={"k": range(3)}) as m:
        pm.Exponential("e", 0.1)  # op exposes scale = 1/rate -> a fraction, not reciprocal(...)
        pm.Normal("c", mu=np.array([-2.0, -2.0 / 3, 2.0]), sigma=1, dims="k")  # rounded numbers
        pm.Censored("z", pm.Normal.dist(0, 1), lower=-np.inf, upper=2, observed=np.zeros(4))
    assert r"\frac{1}{0.1}" in _label(m, "e") and "reciprocal" not in _label(m, "e")
    assert "-0.6667" in _label(m, "c") and "0.666667" not in _label(m, "c")
    assert r"-\infty" in _label(m, "z")
    # an LKJCholeskyCov's auto-generated corr/stds deterministics are illegible matrix plumbing -> elided
    with pm.Model() as m2:
        pm.LKJCholeskyCov("L", n=3, eta=2, sd_dist=pm.Exponential.dist(1), compute_corr=True)
    corr = _label(m2, "L_corr")
    assert r"\cdots" in corr and r"f\!\left(f\!\left" not in corr


def test_unknown_op_over_named_leaf_shows_f_of_leaf():
    """A named leaf must count as 'used' even un-wrapped, so an unknown op over it (e.g. sum)
    renders f(v) rather than eliding to dots."""
    with pm.Model() as m:
        v = pm.Normal("v", 0, 1, shape=3)
        pm.Normal("yy", v.sum(), 1)
    ir = to_ir(m)
    yloc = next(p for p in ir.node("yy").params if p.name == "loc")
    assert r"f\!\left(" in yloc.value_tex and "v" in yloc.value_tex
    assert yloc.value_tex != r"\ldots"


def test_symbol_for():
    assert symbol_for("mu") == r"\mu"
    assert symbol_for("tau") == r"\tau"
    assert symbol_for("y_obs") == r"y_{\mathrm{obs}}"
    assert symbol_for("beta_1") == r"\beta_{1}"
    assert symbol_for("Sigma") == r"\Sigma"
    assert symbol_for("x") == "x"


def test_dist_symbol():
    assert dist_symbol("Normal") == r"\mathcal{N}"
    assert dist_symbol("HalfNormal") == r"\mathcal{N}^{+}"
    assert dist_symbol("Womble") == r"\operatorname{Womble}"
    # full-catalog coverage: derived op names (incl. the collapsed/aliased ones) resolve
    assert dist_symbol("MultivariateNormal") == r"\mathcal{N}"  # MvNormal's op print-name (was a dead key)
    assert dist_symbol("Mixture") == r"\mathrm{Mix}"  # also NormalMixture/ZeroInflated*
    assert dist_symbol("Hurdle") == r"\mathrm{Hurdle}"
    assert dist_symbol("VonMises") == r"\mathrm{VonMises}"
    assert dist_symbol("PG") == r"\mathrm{PG}"  # PolyaGamma
    assert dist_symbol("RandomWalk") == r"\mathrm{RW}"  # also GaussianRandomWalk
    # a generic Truncated(<Base>) renders the base symbol with a truncation subscript
    assert dist_symbol("TruncatedGamma") == r"\mathrm{Gamma}_{[\,]}"


def test_assemble_stochastic_wraps_tokens():
    tex, tree = labels.assemble_stochastic("mu", "Normal", [("loc", "0"), ("scale", "5")])
    assert r"\cssId{tok-loc}{0}" in tex
    assert r"\cssId{tok-scale}{5}" in tex
    assert tex.startswith(r"\mu \sim \mathcal{N}")
    assert [c.token_id for c in tree.children] == ["loc", "scale"]


def test_eight_schools_labels(eight_schools_ir):
    nd = {n.id: n for n in eight_schools_ir.nodes}
    # deterministic rendered as real math, each leaf anchorable
    # the LHS variable is wrapped so it's anchorable (its outgoing edge originates from it)
    assert nd["theta"].label_tex.startswith(rf"\cssId{{tok-{labels.LHS_TOKEN}}}{{\theta}} = ")
    for tok in (r"\cssId{tok-mu}", r"\cssId{tok-tau}", r"\cssId{tok-eta}"):
        assert tok in nd["theta"].label_tex
    # observed likelihood: loc slot shows the parent symbol
    yloc = next(p for p in nd["y_obs"].params if p.name == "loc")
    assert yloc.value_tex == r"\theta"
    assert r"\mathcal{N}" in nd["mu"].label_tex


def test_deterministic_port_edges(eight_schools_ir):
    e = {(x.source, x.target): x.target_token_id for x in eight_schools_ir.edges}
    assert e[("mu", "theta")] == "mu"
    assert e[("tau", "theta")] == "tau"
    assert e[("theta", "y_obs")] == "loc"


@pytest.mark.skipif(
    not mathsvg.get_renderer().available, reason="needs the 'math' extra + built bundle"
)
def test_all_labels_render_in_mathjax(eight_schools_ir):
    for n in eight_schools_ir.nodes:
        svg, _anchors = mathsvg.render_with_anchors(n.label_tex)
        assert "<svg" in svg, f"label failed to render: {n.id}: {n.label_tex}"


def test_param_name_templates_kill_arg_noise():
    """SymbolicRandomVariables have a generic (inputs, kwargs) signature -> arg0/arg1 noise.
    The verified per-construct templates name the slots and hide structural params (steps)."""
    with pm.Model(coords={"t": range(8)}) as m:
        pm.Exponential("e", 1.0)
        pm.GaussianRandomWalk("rw", mu=0.1, sigma=0.5, init_dist=pm.Normal.dist(0, 1), dims="t")
        pm.AR("ar", rho=[0.6, 0.2], sigma=0.4, init_dist=pm.Normal.dist(0, 1), constant=False, dims="t")
        pm.Censored("cz", pm.Normal.dist(0, 1), lower=-1.0, upper=2.0)
        pm.NormalMixture("mix", w=[0.3, 0.7], mu=[-1.0, 1.0], sigma=[0.5, 0.5])
    ir = to_ir(m)
    nd = {n.id: n for n in ir.nodes}
    for nid in ("rw", "ar", "cz", "mix"):
        assert "arg0" not in nd[nid].label_tex, nid
    assert [p.name for p in nd["ar"].params] == ["rho", "sigma", "init"]  # steps hidden
    assert [p.name for p in nd["rw"].params] == ["init", "innov"]
    assert [p.name for p in nd["cz"].params] == ["dist", "lower", "upper"]
    assert [p.name for p in nd["mix"].params] == ["w", "comp"]
    # ZeroInflated* derives to a 3-param Mixture -> the arity-matched variant applies
    with pm.Model() as m_zi:
        pm.ZeroInflatedPoisson("zi", psi=0.8, mu=3.0)
    zi = next(n for n in to_ir(m_zi).nodes if n.id == "zi")
    assert [p.name for p in zi.params] == ["w", "comp1", "comp2"]
    assert "arg0" not in zi.label_tex
    # the hidden steps param leaves no trailing elision in the label
    assert not nd["ar"].label_tex.rstrip(r"\right)").endswith(r"\ldots")
    # trivial reciprocal folds: Exp(rate=1) shows 1, not 1/1
    assert r"\cssId{tok-scale}{1}" in nd["e"].label_tex
    assert r"\frac{1}{1}" not in nd["e"].label_tex
