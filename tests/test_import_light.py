"""Invariant: `import bayesdag` must not pull in pymc / anywidget / numpy / scipy.

Those are heavy or optional (extras / adapters). Checked in a clean subprocess so other
tests' imports don't pollute sys.modules.
"""

import subprocess
import sys


def test_import_bayesdag_is_light():
    code = (
        "import sys, bayesdag\n"
        "heavy = [m for m in ('pymc', 'pytensor', 'anywidget', 'numpy', 'scipy') "
        "if m in sys.modules]\n"
        "assert not heavy, f'import bayesdag pulled in: {heavy}'\n"
        "assert callable(bayesdag.view)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
