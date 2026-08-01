"""``ModelGraphView`` — the public entry point. Builds the IR + layout + shared SVG once,
then renders the right surface per environment: an interactive anywidget in Jupyter/marimo
(when available), a static SVG everywhere else. The widget consumes the *same* SVG, so the
two are identical by construction.
"""

from __future__ import annotations

import logging
from typing import Any

from . import render_static
from .convert import subgraph, to_ir
from .ir import ModelIR
from .layout import layout
from .render_svg import render_node_panel, render_observed_panel, to_svg


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


def _warm_layout_engine() -> None:
    """Fire-and-forget the ELK V8 context build on its dedicated worker thread, overlapping
    the ~0.2s isolate init with ``to_ir``. Failures are deliberately ignored here — they
    re-raise with the real, actionable error on the first ``layout()`` call."""
    import os

    if os.environ.get("BAYESDAG_LAYOUT", "").strip().lower() == "dot":
        return
    try:
        from .layout.elk_backend import get_engine

        eng = get_engine()
        if eng.available:
            # retrieve the Future's exception so a failed warm-up is a logged debug record
            # rather than an "exception never retrieved" surprise at interpreter shutdown
            eng._worker().submit(eng._context).add_done_callback(_log_warmup_failure)
    except Exception:
        pass


def _log_warmup_failure(fut) -> None:
    try:
        exc = fut.exception()
    except Exception:
        return
    if exc is not None:
        logging.getLogger(__name__).debug(
            "ELK warm-up failed (will re-raise on layout)", exc_info=exc
        )


class ModelGraphView:
    def __init__(
        self,
        model_or_ir: Any,
        idata: Any = None,
        *,
        rankdir: str = "TB",
        legend: bool = True,
        widget_legend: bool = False,
        var_names: Any = None,
    ) -> None:
        _warm_layout_engine()  # build the ELK V8 isolate on its worker WHILE to_ir runs
        # keep the source model (if any) so the interactive plate prior-predictive panels
        # can be computed lazily — static rendering never pays that cost.
        self._model = None if isinstance(model_or_ir, ModelIR) else model_or_ir
        self.ir = to_ir(model_or_ir, idata=idata)
        if var_names is not None:
            self.ir = subgraph(self.ir, list(var_names))
        self.layout = layout(self.ir, rankdir=rankdir)
        # The static figure carries the legend by default (it can't hover); the interactive
        # widget omits it by default since the same info is one hover away.
        self._legend = legend
        self._widget_legend = widget_legend
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
            nx = None
            moral = None
        parents: dict[str, list] = {n.id: [] for n in self.ir.nodes}
        children: dict[str, list] = {n.id: [] for n in self.ir.nodes}
        for e in self.ir.edges:
            children.setdefault(e.source, []).append(e.target)
            parents.setdefault(e.target, []).append(e.source)
        nodes = {}
        for n in self.ir.nodes:
            blanket = sorted(moral.neighbors(n.id)) if (moral is not None and n.id in moral) else []
            # Transitive lineage for the directional causal trace shown on pin (click): every
            # node that flows INTO this one (ancestors) vs. every node it flows into
            # (descendants). Computed here so the JS layer stays graph-algorithm-free.
            in_graph = nx is not None and n.id in g
            ancestors = sorted(nx.ancestors(g, n.id)) if in_graph else []
            descendants = sorted(nx.descendants(g, n.id)) if in_graph else []
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
                "ancestors": ancestors,
                "descendants": descendants,
                # the SAME MathJax SVG embedded in the diagram -> the tooltip/card show real
                # rendered math (parity), not raw LaTeX source.
                "label_svg": n.label_svg,
            }
            # every glyph-bearing node gets a large pinned-card panel built straight from the
            # precomputed glyph_data (JS just injects this SVG): observed nodes keep the
            # histogram + best-fit overlay; everything else renders its glyph large with the
            # hedged source caption + coord labels.
            if n.glyph_data:
                if n.role == "observed":
                    panel = render_observed_panel(n.id, n.dist, n.glyph_data)
                else:
                    panel = render_node_panel(n)
                if panel:
                    nodes[n.id]["panel"] = panel
        plates: dict = {}
        if self._model is not None and self.ir.plates:
            try:
                from .adapters.ppc import prior_predictive_expansions
                from .render_svg import render_plate_panel

                for pid, exp in prior_predictive_expansions(self._model, self.ir).items():
                    panel = render_plate_panel(exp)
                    if panel:
                        plates[pid] = {"panel": panel}
            except Exception:
                plates = {}
        # widget SVG omits the legend by default (hover surfaces the same info); the static
        # `self._svg` keeps it. Re-render is cheap (layout + math are already computed).
        widget_svg = (
            self._svg
            if self._widget_legend == self._legend
            else to_svg(self.ir, self.layout, legend=self._widget_legend)
        )
        return {"svg": widget_svg, "nodes": nodes, "plates": plates, "selected": ""}

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
            try:
                return mo.ui.anywidget(self.widget())
            except Exception:
                pass  # same contract as _repr_mimebundle_: degrade to the static figure
        return mo.Html(self._svg)


def view(
    model_or_ir: Any,
    idata: Any = None,
    *,
    rankdir: str = "TB",
    legend: bool = True,
    widget_legend: bool = False,
    var_names: Any = None,
) -> ModelGraphView:
    """Visualize a PyMC model (or a ``ModelIR``). Returns a :class:`ModelGraphView` that
    renders interactively in a notebook and statically elsewhere.

    ``legend`` (default ``True``) embeds a context-aware legend in the **static** figure.
    ``widget_legend`` (default ``False``) controls it for the **interactive** widget, which
    omits the legend by default since hovering a node already surfaces the same information.
    ``var_names`` (default ``None`` = everything) restricts the diagram to those variables
    plus their direct parents — the same semantics as ``pm.model_to_graphviz(var_names=…)``.
    """
    return ModelGraphView(
        model_or_ir,
        idata=idata,
        rankdir=rankdir,
        legend=legend,
        widget_legend=widget_legend,
        var_names=var_names,
    )
