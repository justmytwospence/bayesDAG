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
        (2.0, "2"),  # a whole float reads as an integer, not "2.0000"
        (0.6666666666, "0.6667"),  # ~4 significant figures
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
        (np.array([2.0, 2.0, 2.0]), "2"),  # constant vector -> the value
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


def _tex(build):
    """Render a deterministic built by ``build(a, b, c)`` from three named scalar leaves."""
    with pm.Model() as m:
        a, b, c = (pm.Normal(n, 0, 1) for n in "abc")
        d = pm.Deterministic("d", build(a, b, c))
    named = {id(v): n for n, v in m.named_vars.items()}
    return render_value(d, named, wrap_leaves=False, _root=True)[0]


@pytest.mark.parametrize(
    "build,expected",
    [
        (lambda a, b, c: a * (b + c), r"a\,\left(b + c\right)"),
        (lambda a, b, c: a - (b + c), r"a - \left(b + c\right)"),
        (lambda a, b, c: a - (b - c), r"a - \left(b - c\right)"),
        (lambda a, b, c: -(a + b), r"-\left(a + b\right)"),
        (lambda a, b, c: (a + b) ** 2, r"\left(a + b\right)^{2}"),
        (lambda a, b, c: (a * b) ** c, r"\left(a\,b\right)^{c}"),
        (lambda a, b, c: a * b + c, r"a\,b + c"),  # no spurious parens
        (lambda a, b, c: a + b * c, r"a + b\,c"),
        (lambda a, b, c: a * -b, r"a\,\left(-b\right)"),
    ],
)
def test_precedence_is_preserved(build, expected):
    """`a*(b+c)` used to render as `a\\,b + c` — a different number. Every operand whose
    precedence is lower than its context must be parenthesized."""
    assert _tex(build) == expected


def test_value_changing_casts_are_visible():
    assert _tex(lambda a, b, c: pt.cast(a, "int64")) == r"\operatorname{int}\!\left(a\right)"
    assert _tex(lambda a, b, c: pt.cast(a > b, "float64")) == "a > b"  # bool -> float widens
    assert _tex(lambda a, b, c: pt.cast(a, "float32")) == "a"  # float -> float is invisible


def test_reductions_name_their_operation():
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1, shape=3)
        ds = {
            "sum": pm.Deterministic("s", x.sum()),
            "prod": pm.Deterministic("p", x.prod()),
            "max": pm.Deterministic("mx", x.max()),
        }
    named = {id(v): n for n, v in m.named_vars.items()}
    out = {k: render_value(v, named, wrap_leaves=False, _root=True)[0] for k, v in ds.items()}
    assert r"\sum" in out["sum"]
    assert r"\prod" in out["prod"] and r"\sum" not in out["prod"]
    assert r"\max" in out["max"] and r"\sum" not in out["max"]


def test_repeated_leaf_is_token_wrapped_once():
    """Two `\\cssId{tok-a}` in one label would give the composed SVG duplicate element ids; the
    dependency itself must survive in `used`."""
    with pm.Model() as m:
        a = pm.Normal("a", 0, 1)
        d = pm.Deterministic("d", a + a * a)
    named = {id(v): n for n, v in m.named_vars.items()}
    tex, used = render_value(d, named, wrap_leaves=True, _root=True)
    assert tex.count(r"\cssId{tok-a}") == 1
    assert used == {"a"}
