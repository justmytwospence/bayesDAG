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
    ("AR", lambda: pm.AR("x", rho=[0.5], sigma=1, init_dist=pm.Normal.dist(0, 1), steps=8), "density", True),
    ("Interpolated", lambda: pm.Interpolated("x", x_points=np.linspace(-3, 3, 40), pdf_points=np.exp(-np.linspace(-3, 3, 40) ** 2 / 2)), "density", True),
    ("NormalMixture", lambda: pm.NormalMixture("x", w=[0.5, 0.5], mu=[-2, 2], sigma=[1, 1]), "mixture", True),
    ("ZeroInflatedPoisson", lambda: pm.ZeroInflatedPoisson("x", psi=0.7, mu=3), "mixture", True),
    # honest badges (undrawable as a single static shape)
    ("LKJCorr", lambda: pm.LKJCorr("x", n=3, eta=2), "schematic", False),
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


def test_special_constructs_render_without_error():
    """Every special construct must compose into a valid SVG (badge or glyph), never crash."""
    for _name, build, _k, _r in _SPECIAL:
        with pm.Model() as m:
            build()
        ir = to_ir(m)
        assert "<svg" in to_svg(ir, layout(ir))
