"""Smoke test — the package imports and exposes a version. No heavy deps required."""

import bayesdag


def test_version_present():
    assert isinstance(bayesdag.__version__, str)
    assert bayesdag.__version__


# The import-light invariant is checked by tests/test_import_light.py, which runs in a clean
# subprocess and asserts pymc/pytensor/numpy/scipy/anywidget stay out of sys.modules. The version
# that lived here guarded on `find_spec(...) is not None` and then merely imported, asserting
# nothing — it read like a second guard while checking nothing at all.
