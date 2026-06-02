"""``ModelGraphView`` — the public entry point. Builds the IR + layout + shared SVG once,
then renders the right surface per environment: an interactive anywidget in Jupyter/marimo
(when available), a static SVG everywhere else. The widget consumes the *same* SVG, so the
two are identical by construction.
"""

from __future__ import annotations

from typing import Any

from . import render_static
from .convert import to_ir
from .layout import layout
from .render_svg import to_svg


def _in_marimo() -> bool:
    try:
        import marimo as mo

        return bool(mo.running_in_notebook())
    except Exception:
        return False


def _interactive_available() -> bool:
    try:
        import anywidget  # noqa: F401
        import ipywidgets  # noqa: F401
    except Exception:
        return False
    return True


class ModelGraphView:
    def __init__(self, model_or_ir: Any, idata: Any = None, *, rankdir: str = "TB") -> None:
        self.ir = to_ir(model_or_ir, idata=idata)
        self.layout = layout(self.ir, rankdir=rankdir)
        self._svg = to_svg(self.ir, self.layout)
        self._widget = None

    # ---- outputs ---------------------------------------------------------------
    def to_svg(self) -> str:
        return self._svg

    def save(self, path):
        """Save to .svg / .png / .pdf (format from the extension)."""
        return render_static.save(self._svg, path)

    def widget(self):
        if self._widget is None:
            from .widget import ModelGraphWidget

            self._widget = ModelGraphWidget(spec={"svg": self._svg})
        return self._widget

    # ---- display protocol ------------------------------------------------------
    def _repr_svg_(self) -> str:
        return self._svg

    def _repr_mimebundle_(self, **kwargs):
        if _interactive_available():
            try:
                return self.widget()._repr_mimebundle_(**kwargs)
            except Exception:
                pass
        return {"image/svg+xml": self._svg}

    def _display_(self):  # marimo hook (takes precedence there)
        try:
            import marimo as mo
        except Exception:
            return None
        if _interactive_available():
            return mo.ui.anywidget(self.widget())
        return mo.Html(self._svg)


def view(model_or_ir: Any, idata: Any = None, *, rankdir: str = "TB") -> ModelGraphView:
    """Visualize a PyMC model (or a ``ModelIR``). Returns a :class:`ModelGraphView` that
    renders interactively in a notebook and statically elsewhere."""
    return ModelGraphView(model_or_ir, idata=idata, rankdir=rankdir)
