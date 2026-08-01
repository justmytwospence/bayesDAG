"""PyTensor expression -> LaTeX, tested directly.

This module had no test file of its own; it was exercised only transitively through the
assembled labels in test_labels.py, which meant the constant-formatting and elision edge cases
below were never pinned.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from bayesdag.adapters.pytensor_latex import _const_tex, _fmt_scalar, _is_wrapper, render_value


@pytest.mark.parametrize(
    "value,expected",
    [
        (2.0, "2"),                       # a whole float reads as an integer, not "2.0000"
        (0.6666666666, "0.6667"),         # ~4 significant figures
        (-1.5, "-1.5"),
        (float("inf"), r"\infty"),
        (float("-inf"), r"-\infty"),
        (float("nan"), r"\mathrm{nan}"),
    ],
)
def test_scalar_formatting(value, expected):
    assert _fmt_scalar(value) == expected


@pytest.mark.parametrize(
    "array,expected",
    [
        (np.array(3.0), "3"),
        (np.array([]), r"[\,]"),
        (np.array([2.0, 2.0, 2.0]), "2"),                       # constant vector -> the value
        (np.array([1.0, 2.0]), r"[1,\,2]"),
        (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), r"[1,\,2,\,3,\,\ldots]"),  # long -> head + ellipsis
    ],
)
def test_constant_rendering(array, expected):
    assert _const_tex(pt.as_tensor_variable(array)) == expected


def test_const_tex_ignores_non_constants():
    assert _const_tex(pt.vector("v")) is None


def test_wrapper_detection_excludes_cast():
    """`_is_wrapper` decides what a label may see THROUGH. A float->int Cast changes the value
    (it is a step function), so it must never be treated as transparent."""
    x = pt.vector("x")
    assert _is_wrapper(pt.specify_shape(x, (3,)).owner.op) is False
    assert _is_wrapper(x.dimshuffle("x", 0).owner.op) is True
    assert _is_wrapper(pt.cast(x, "int64").owner.op) is False


def test_render_value_stops_at_named_variables():
    """The leaf boundary: a named model variable renders as its symbol rather than being
    descended into, so an equation shows `mu + tau*eta`, not the whole graph beneath it."""
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1)
        tau = pm.HalfNormal("tau", 1)
        eta = pm.Normal("eta", 0, 1)
        expr = pm.Deterministic("theta", mu + tau * eta)

    named = {id(v): n for n, v in m.named_vars.items()}
    tex, leaves = render_value(expr, named, wrap_leaves=True, _root=True)
    assert leaves == {"mu", "tau", "eta"}
    for leaf in leaves:
        assert rf"\cssId{{tok-{leaf}}}" in tex
    assert "theta" not in tex  # the node being defined is not a leaf of its own equation


def test_oversized_graph_degrades_instead_of_exploding():
    """The node budget: a giant expression must elide to f(...)/ellipsis rather than emitting an
    unbounded label."""
    with pm.Model() as m:
        xs = [pm.Normal(f"x{i}", 0, 1) for i in range(60)]
        big = pm.Deterministic("big", sum(xs))

    named = {id(v): n for n, v in m.named_vars.items()}
    tex, _leaves = render_value(big, named, wrap_leaves=True, _root=True)
    assert len(tex) < 4000
    assert r"\ldots" in tex or r"\cdots" in tex or "f" in tex
