"""bayesdag — shape-first, posterior-aware, interactive visualizations of PyMC models.

Public API (implemented incrementally during M0; see
``.claude/plans/please-review-all-the-streamed-storm.md``)::

    import bayesdag
    view = bayesdag.view(model, idata=None)   # -> ModelGraphView (auto static/interactive)
    view.to_svg(); view.save("model.svg"); view.widget()

The ``bayesdag.ir`` module is intentionally import-light (stdlib only) so the IR
can be produced/validated without pymc, xarray, or any renderer installed.
"""

from __future__ import annotations

import importlib.metadata

try:
    __version__ = importlib.metadata.version("bayesdag")
except importlib.metadata.PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0+dev"

from .view import view  # noqa: E402  (binds the function, shadowing the submodule attr)

__all__ = ["__version__", "to_ir", "view"]


def __getattr__(name: str):  # lazy: keep heavy/optional deps out of `import bayesdag`
    if name == "to_ir":
        from .convert import to_ir

        return to_ir
    raise AttributeError(f"module 'bayesdag' has no attribute {name!r}")
