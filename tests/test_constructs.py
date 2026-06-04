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


@pytest.mark.parametrize(
    "name,build",
    [
        ("Flat", lambda: pm.Flat("x")),
        ("HalfFlat", lambda: pm.HalfFlat("x")),
        ("ICAR", lambda: pm.ICAR("x", W=np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))),
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
