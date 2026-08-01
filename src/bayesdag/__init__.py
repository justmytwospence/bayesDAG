"""bayesdag — shape-first, posterior-aware, interactive visualizations of PyMC models.

Public API::

    import bayesdag
    v = bayesdag.view(model, idata=None)   # -> ModelGraphView (auto static/interactive)
    v.to_svg(); v.save("model.svg"); v.widget()

    ir = bayesdag.to_ir(model)             # the neutral IR
    sub = bayesdag.subgraph(ir, ["tau"])   # restrict to variables + their direct parents
    svg = bayesdag.to_svg(ir, bayesdag.layout(ir))

``bayesdag.ir`` is intentionally import-light (stdlib only), and so is this whole chain:
pymc/pytensor/numpy/scipy/anywidget arrive only when an adapter or renderer actually needs
them. ``tests/test_import_light.py`` checks that in a clean subprocess.

Note that ``view`` and ``layout`` are the FUNCTIONS here; they deliberately shadow the
same-named submodules, which stay reachable as ``bayesdag.view``/``bayesdag.layout`` via
``importlib.import_module``.
"""

from __future__ import annotations

import importlib.metadata

try:
    __version__ = importlib.metadata.version("bayesdag")
except importlib.metadata.PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0+dev"

# noqa: E402 — these must follow __version__, and each shadows a submodule attribute of the
# same name, so binding order matters (the import machinery sets the module attr first).
from .convert import subgraph, to_ir  # noqa: E402
from .ir import ModelIR  # noqa: E402
from .layout import layout  # noqa: E402
from .render_svg import to_svg  # noqa: E402
from .view import ModelGraphView, view  # noqa: E402

__all__ = [
    "__version__",
    "ModelGraphView",
    "ModelIR",
    "layout",
    "subgraph",
    "to_ir",
    "to_svg",
    "view",
]
