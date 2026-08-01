"""Invariant: `import bayesdag` must not pull in pymc / anywidget / numpy / scipy.

Those are heavy or optional (extras / adapters). Checked in a clean subprocess so other
tests' imports don't pollute sys.modules.
"""

import subprocess
import sys
from pathlib import Path


def test_import_bayesdag_is_light():
    """The whole public surface must resolve without dragging in the heavy stack — including
    the exports that shadow a same-named submodule (`view`, `layout`), which have to be the
    callables, not the modules."""
    code = (
        "import sys, bayesdag\n"
        "heavy = [m for m in ('pymc', 'pytensor', 'anywidget', 'numpy', 'scipy') "
        "if m in sys.modules]\n"
        "assert not heavy, f'import bayesdag pulled in: {heavy}'\n"
        "for name in bayesdag.__all__:\n"
        "    assert hasattr(bayesdag, name), f'{name} in __all__ but not importable'\n"
        "import inspect\n"
        "for name in ('view', 'layout', 'to_ir', 'to_svg', 'subgraph'):\n"
        "    obj = getattr(bayesdag, name)\n"
        "    assert inspect.isfunction(obj), f'bayesdag.{name} is {obj!r}, not the function'\n"
        "assert inspect.isclass(bayesdag.ModelGraphView) and inspect.isclass(bayesdag.ModelIR)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_py_typed_marker_is_present():
    """The API is fully annotated; without this marker none of it reaches a type checker."""
    import bayesdag

    assert (Path(bayesdag.__file__).parent / "py.typed").is_file()
