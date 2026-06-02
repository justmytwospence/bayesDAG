"""Label engine: symbols, distribution templates, assembly, and the 8-schools labels."""

import pytest

from bayesdag import labels, mathsvg
from bayesdag.labels import dist_symbol, symbol_for


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


def test_assemble_stochastic_wraps_tokens():
    tex, tree = labels.assemble_stochastic("mu", "Normal", [("loc", "0"), ("scale", "5")])
    assert r"\cssId{tok-loc}{0}" in tex
    assert r"\cssId{tok-scale}{5}" in tex
    assert tex.startswith(r"\mu \sim \mathcal{N}")
    assert [c.token_id for c in tree.children] == ["loc", "scale"]


def test_eight_schools_labels(eight_schools_ir):
    nd = {n.id: n for n in eight_schools_ir.nodes}
    # deterministic rendered as real math, each leaf anchorable
    assert nd["theta"].label_tex.startswith(r"\theta =")
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
