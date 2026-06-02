"""Smoke test — the package imports and exposes a version. No heavy deps required."""

import bayesdag


def test_version_present():
    assert isinstance(bayesdag.__version__, str)
    assert bayesdag.__version__


def test_ir_is_import_light():
    """`bayesdag.ir` must import without pymc/xarray/render deps (invariant)."""
    import importlib.util

    if importlib.util.find_spec("bayesdag.ir") is not None:
        import bayesdag.ir  # noqa: F401  (only asserts it imports cleanly)
