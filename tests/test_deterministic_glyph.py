"""Deterministic transfer-function glyphs.

A glyph is drawn ONLY when its shape is a provable consequence of the op graph (zero false positives);
it is parameter-free and deterministic. These tests pin the detection table (draws + honest skips), the
parameter-independence, the unfilled `curve` rendering, the presence-based geometry gate, and the
glyph-deterministic edge exit.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from bayesdag import geometry, glyph
from bayesdag.convert import to_ir
from bayesdag.ir import Box


def _glyph(builder):
    with pm.Model() as m:
        builder()
    return to_ir(m).node("d").glyph


# (label, builder, expected kind) — each builds a deterministic named "d" that IS provably depictable.
_DRAW = [
    ("invlogit", lambda: pm.Deterministic("d", pm.math.invlogit(pm.Normal("x", 0, 1))), "curve"),
    (
        "exp",
        lambda: pm.Deterministic("d", pm.math.exp(pm.Normal("a", 0, 1) + pm.Normal("b", 0, 1))),
        "curve",
    ),
    ("log", lambda: pm.Deterministic("d", pm.math.log(pm.HalfNormal("x", 1))), "curve"),
    ("softplus", lambda: pm.Deterministic("d", pt.softplus(pm.Normal("x", 0, 1))), "curve"),
    ("tanh", lambda: pm.Deterministic("d", pt.tanh(pm.Normal("x", 0, 1))), "curve"),
    ("sqrt", lambda: pm.Deterministic("d", pt.sqrt(pm.HalfNormal("x", 1))), "curve"),
    ("abs", lambda: pm.Deterministic("d", abs(pm.Normal("x", 0, 1))), "curve"),
    ("pow2", lambda: pm.Deterministic("d", pm.Normal("x", 0, 1) ** 2), "curve"),
    (
        "probit (0.5(1+erf))",
        lambda: pm.Deterministic("d", 0.5 * (1 + pt.erf(pm.Normal("x", 0, 1)))),
        "curve",
    ),
    (
        "scaled invlogit",
        lambda: pm.Deterministic("d", 2.0 * pm.math.invlogit(pm.Normal("x", 0, 1))),
        "curve",
    ),
    ("reflected -exp", lambda: pm.Deterministic("d", -pm.math.exp(pm.Normal("x", 0, 1))), "curve"),
    (
        "affine (a + b*data)",
        lambda: pm.Deterministic("d", pm.Normal("a", 0, 1) + pm.Normal("b", 0, 1) * np.arange(5.0)),
        "curve",
    ),
    (
        "softmax",
        lambda: pm.Deterministic("d", pm.math.softmax(pm.Normal("e", 0, 1, shape=3))),
        "bars",
    ),
]

# Builders whose deterministic "d" is NOT provably depictable -> equation-only (glyph is None).
_SKIP = [
    (
        "bilinear tau*eta",
        lambda: pm.Deterministic("d", pm.HalfNormal("t", 1) * pm.Normal("e", 0, 1)),
    ),
    (
        "manual sigmoid 1/(1+e^-x)",
        lambda: pm.Deterministic("d", 1.0 / (1.0 + pm.math.exp(-pm.Normal("x", 0, 1)))),
    ),
    ("reciprocal 1/x", lambda: pm.Deterministic("d", 1.0 / pm.HalfNormal("x", 1))),
    ("sum reduction", lambda: pm.Deterministic("d", pt.sum(pm.Normal("x", 0, 1, shape=4)))),
    ("mean reduction", lambda: pm.Deterministic("d", pt.mean(pm.Normal("x", 0, 1, shape=4)))),
    (
        "exp(x)*parent",
        lambda: pm.Deterministic("d", pm.math.exp(pm.Normal("x", 0, 1)) * pm.HalfNormal("s", 1)),
    ),
    (
        "two transfers exp+log",
        lambda: pm.Deterministic(
            "d", pm.math.exp(pm.Normal("a", 0, 1)) + pm.math.log(pm.HalfNormal("b", 1))
        ),
    ),
    (
        "pure gather a[idx]",
        lambda: pm.Deterministic("d", pm.Normal("a", 0, 1, shape=3)[np.array([0, 1, 2, 0])]),
    ),
    (
        "non-const exponent x**k",
        lambda: pm.Deterministic("d", pm.HalfNormal("x", 1) ** pm.Normal("k", 0, 1)),
    ),
]


@pytest.mark.parametrize("label,builder,kind", _DRAW, ids=[c[0] for c in _DRAW])
def test_provably_depictable_draws(label, builder, kind):
    g = _glyph(builder)
    assert g is not None, label
    assert g.kind == kind and g.source == "deterministic_fn", label


@pytest.mark.parametrize("label,builder", _SKIP, ids=[c[0] for c in _SKIP])
def test_unprovable_skips_to_equation_only(label, builder):
    assert _glyph(builder) is None, label  # zero false positives: never a misleading curve


def test_curve_is_parameter_free_and_deterministic():
    """The shape is canonical — independent of the parents' values (never sampled through them) — and
    identical on every build."""

    def gd(scale):
        with pm.Model() as m:
            pm.Deterministic("d", pm.math.invlogit(pm.Normal("x", 0, scale)))
        return to_ir(m).node("d").glyph_data

    assert gd(1.0) == gd(100.0)  # parent-independent
    assert gd(1.0) == gd(1.0)  # deterministic


def test_distinct_transfers_have_distinct_shapes():
    """Specificity: logistic vs probit, and x**2 vs x**3, are drawn as the actual (different) curves."""

    def gd(builder):
        with pm.Model() as m:
            builder()
        return to_ir(m).node("d").glyph_data

    logit = gd(lambda: pm.Deterministic("d", pm.math.invlogit(pm.Normal("x", 0, 1))))
    probit = gd(lambda: pm.Deterministic("d", 0.5 * (1 + pt.erf(pm.Normal("x", 0, 1)))))
    sq = gd(lambda: pm.Deterministic("d", pm.Normal("x", 0, 1) ** 2))
    cube = gd(lambda: pm.Deterministic("d", pm.Normal("x", 0, 1) ** 3))
    assert logit["ys"] != probit["ys"]
    assert sq["ys"] != cube["ys"]


def test_curve_renders_unfilled_with_baseline():
    out = glyph.render("curve", {"xs": [-3, 0, 3], "ys": [0.0, 0.5, 1.0]}, Box(0, 0, 80, 40))
    assert 'fill="none"' in out  # the function line is unfilled (a curve, not a density area)
    assert "<line" in out  # faint baseline tethers it
    assert 'Z"' not in out  # no closed area path (distinguishes it from render_density)


def test_curve_in_registry():
    assert "curve" in glyph.registered_kinds()


def test_geometry_gate_is_presence_based():
    b = Box(0, 0, 120, 80)
    data = {"xs": [0, 1], "ys": [0, 1]}
    # a deterministic WITH a glyph reserves a strip; equation-only reserves nothing
    assert geometry.glyph_rect(b, "deterministic", 16.0, "curve", data) is not None
    assert geometry.glyph_rect(b, "deterministic", 16.0, None, None) is None
    _, h_glyph = geometry.node_size(100, 16, "deterministic", "curve", data)
    _, h_plain = geometry.node_size(100, 16, "deterministic", None, None)
    assert h_glyph > h_plain
    # latent/observed sizing unchanged (still reserves a strip when it carries glyph data)
    _, h_latent = geometry.node_size(100, 16, "latent", "density", data)
    assert h_latent > h_plain


def test_curve_width_decoupled_from_equation_width():
    """A deterministic transfer curve has a canonical shape; its width must not stretch with a
    wide equation. The strip is capped and centered, unlike a density strip (which spans the box)."""
    data = {"xs": [0, 1], "ys": [0, 1]}
    wide = Box(0, 0, 400, 80)
    cr = geometry.glyph_rect(wide, "deterministic", 16.0, "curve", data)
    assert cr is not None
    assert cr.w == geometry._FN_GLYPH_MAX_W  # capped, NOT 400 - 2*PAD
    assert abs((cr.x + cr.w / 2.0) - (wide.x + wide.w / 2.0)) < 0.01  # centered under the equation
    # a density strip (a distribution) still spans the full box width — only `curve` is capped
    dr = geometry.glyph_rect(wide, "latent", 16.0, "density", data)
    assert dr.w == wide.w - 2 * geometry.PAD
    # a narrow node uses the available width (cap is a ceiling, not a fixed size)
    narrow = Box(0, 0, 56, 80)
    assert (
        geometry.glyph_rect(narrow, "deterministic", 16.0, "curve", data).w == 56 - 2 * geometry.PAD
    )


def test_deterministic_node_draws_a_box():
    """A deterministic equation now renders inside a visible box (not a transparent, border-less
    region), so it reads as a node like the others."""
    from bayesdag.layout import layout
    from bayesdag.render_svg import _node_chrome, to_svg

    with pm.Model() as m:
        x = pm.Normal("x", 0, 1)
        pm.Deterministic("d", x + 1.0)
        pm.Normal("y", mu=m["d"], sigma=1, observed=np.zeros(3))
    ir = to_ir(m)
    det = ir.node("d")
    chrome = _node_chrome(det, Box(0, 0, 80, 40))
    assert 'stroke="none"' not in chrome and "transparent" not in chrome  # a real bordered box
    assert 'stroke="#9499a2"' in chrome  # the deterministic box stroke
    svg = to_svg(ir, layout(ir))
    assert "<svg" in svg


def test_deterministic_incoming_edge_lands_on_box_border():
    """An arrow into a deterministic points at the box's TOP border (above the token), in the token's
    column — it no longer penetrates the equation to the token glyph."""
    from bayesdag.layout import layout

    with pm.Model() as m:
        a = pm.Normal("a", 0, 1)
        bb = pm.Normal("bb", 0, 1)
        pm.Deterministic("s", a + bb)
        pm.Normal("y", mu=m["s"], sigma=1, observed=np.zeros(3))
    ir = to_ir(m)
    res = layout(ir)
    box = res.node_boxes["s"]
    tok = res.node_token_anchors["s"]["a"]  # the `a` token inside `s = a + bb`
    end = res.edge_paths["a|s"][-1]
    assert abs(end[0] - (tok.x + tok.w / 2.0)) < 1.5  # vertically under the `a` token (which one)
    assert abs((box.y - end[1]) - geometry.STANDOFF) < 1.5  # lands above the box top border
    assert end[1] < tok.y  # stays out of the equation (above the token)


def test_glyph_deterministic_edge_exits_from_node_box():
    """A glyph-bearing deterministic's outgoing edge exits at/below the glyph strip (the node box), not
    lifted up into the equation where it would cross the curve."""
    from bayesdag.layout import layout

    with pm.Model() as m:
        x = pm.Normal("x", 0, 1)
        d = pm.Deterministic("d", pm.math.invlogit(x))
        pm.Bernoulli("y", p=d, observed=np.array([1, 0, 1]))
    ir = to_ir(m)
    res = layout(ir)
    n = ir.node("d")
    assert n.glyph is not None
    box = res.node_boxes["d"]
    _, lh = geometry.label_px_size(n.label_svg)
    gr = geometry.glyph_rect(box, n.role, lh, n.glyph.kind, n.glyph_data)
    pts = res.edge_paths.get("d|y")
    assert pts is not None and gr is not None
    assert pts[0][1] >= gr.y + gr.h - 2.0  # starts at/below the glyph, i.e. the box bottom
