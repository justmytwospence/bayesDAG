"""bayesdag — shape-first, posterior-aware, interactive visualizations of PyMC models.

Public API::

    import bayesdag
    v = bayesdag.view(model, idata=None)   # -> ModelGraphView (auto static/interactive)
    v.to_svg(); v.save("model.svg"); v.widget()

    ir = bayesdag.to_ir(model)             # the neutral IR
    sub = bayesdag.subgraph(ir, ["tau"])   # restrict to variables + their direct parents

    from bayesdag.layout import layout     # the low-level pass, if you need it directly
    svg = bayesdag.to_svg(ir, layout(ir))

``bayesdag.ir`` is intentionally import-light (stdlib only), and so is this whole chain:
pymc/pytensor/numpy/scipy/anywidget arrive only when an adapter or renderer actually needs
them. ``tests/test_import_light.py`` checks that in a clean subprocess.

``view`` here is the FUNCTION, shadowing the ``bayesdag.view`` submodule of the same name.
``layout`` is deliberately NOT re-exported: it would shadow the ``bayesdag.layout`` *package*,
which breaks ``import bayesdag.layout.elk_backend``.
"""

from __future__ import annotations

import importlib.metadata

try:
    __version__ = importlib.metadata.version("bayesdag")
except importlib.metadata.PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0+dev"


# These sit below __version__ deliberately, and `view` shadows the submodule attribute of the
# same name, so binding order matters (the import machinery sets the module attr first).
from .convert import subgraph, to_ir
from .ir import ModelIR
from .render_svg import to_svg
from .view import ModelGraphView, view

__all__ = [
    "ModelGraphView",
    "ModelIR",
    "__version__",
    "subgraph",
    "to_ir",
    "to_svg",
    "view",
]
