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
    def __init__(
        self, model_or_ir: Any, idata: Any = None, *, rankdir: str = "TB", legend: bool = True
    ) -> None:
        self.ir = to_ir(model_or_ir, idata=idata)
        self.layout = layout(self.ir, rankdir=rankdir)
        self._svg = to_svg(self.ir, self.layout, legend=legend)
        self._widget = None

    # ---- outputs ---------------------------------------------------------------
    def to_svg(self) -> str:
        return self._svg

    def save(self, path):
        """Save to .svg / .png / .pdf (format from the extension)."""
        return render_static.save(self._svg, path)

    def _build_spec(self) -> dict:
        """SVG + per-node detail + adjacency (Markov blanket) for the interactive layer."""
        from .adapters.graph import to_networkx

        g = to_networkx(self.ir)
        try:
            import networkx as nx

            moral = nx.moral_graph(g)
        except Exception:
            moral = None
        parents: dict[str, list] = {n.id: [] for n in self.ir.nodes}
        children: dict[str, list] = {n.id: [] for n in self.ir.nodes}
        for e in self.ir.edges:
            children.setdefault(e.source, []).append(e.target)
            parents.setdefault(e.target, []).append(e.source)
        nodes = {}
        for n in self.ir.nodes:
            blanket = sorted(moral.neighbors(n.id)) if (moral is not None and n.id in moral) else []
            nodes[n.id] = {
                "role": n.role,
                "dist": n.dist,
                "observed": n.observed,
                "dims": list(n.dims),
                "params": [{"name": p.name, "value": p.value_tex} for p in n.params],
                "transform": n.transform,
                "parents": parents.get(n.id, []),
                "children": children.get(n.id, []),
                "blanket": blanket,
            }
        return {"svg": self._svg, "nodes": nodes, "selected": ""}

    def widget(self):
        if self._widget is None:
            from .widget import ModelGraphWidget

            self._widget = ModelGraphWidget(spec=self._build_spec())
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


def view(
    model_or_ir: Any, idata: Any = None, *, rankdir: str = "TB", legend: bool = True
) -> ModelGraphView:
    """Visualize a PyMC model (or a ``ModelIR``). Returns a :class:`ModelGraphView` that
    renders interactively in a notebook and statically elsewhere. ``legend=True`` embeds a
    context-aware legend in the figure (set ``legend=False`` for a bare diagram)."""
    return ModelGraphView(model_or_ir, idata=idata, rankdir=rankdir, legend=legend)
