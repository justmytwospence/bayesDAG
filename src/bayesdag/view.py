"""``ModelGraphView`` — the public entry point. Builds the IR + layout + shared SVG once,
then renders the right surface per environment: an interactive anywidget in Jupyter/marimo
(when available), a static SVG everywhere else. The widget consumes the *same* SVG, so the
two are identical by construction.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from . import diagnostics, geometry, render_static
from .convert import drop_index_data, subgraph, to_ir
from .ir import ModelIR
from .layout import layout
from .render_svg import render_node_panel, render_observed_panel, strip_token_ids, to_svg


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
        ppc_draws: int | None = 200,
        hide_index_data: bool = True,
    ) -> None:
        _warm_layout_engine()  # build the ELK V8 isolate on its worker WHILE to_ir runs
        # A prebuilt IR (or its dict form) carries no pymc model: the adapter already ran, so
        # there is nothing for the glyph providers or the plate prior-predictive to work from.
        prebuilt = isinstance(model_or_ir, ModelIR | dict)
        # keep the source model (if any) so the interactive plate prior-predictive panels
        # can be computed lazily — static rendering never pays that cost.
        self._model = None if prebuilt else model_or_ir
        self._ppc_draws = ppc_draws
        # The draws themselves stay out of the IR (which is JSON-serializable and ships only
        # curves); this is the handle that makes `OverlayRef` dereferenceable — see `idata`.
        self._idata = None
        # The IR is ALWAYS built prior-only and owned by this view: a posterior is attached through
        # `update()` alone, so `update(None)` can always restore what was built (the prior) — and
        # a caller's IR object (or a second view of it) is never mutated.
        base = to_ir(model_or_ir)
        self.ir = copy.deepcopy(base) if prebuilt else base
        if hide_index_data:  # lookup-table Data whose name every consumer's equation shows
            self.ir = drop_index_data(self.ir)
        if var_names is not None:
            self.ir = subgraph(self.ir, list(var_names))
        self._rankdir = rankdir
        # The static figure carries the legend by default (it can't hover); the interactive
        # widget omits it by default since the same info is one hover away.
        self._legend = legend
        self._widget_legend = widget_legend
        self._widget = None
        # The as-built data layer, so update() can put it back (a prior<->posterior toggle). A
        # prebuilt IR may already hold POSTERIOR glyphs with no prior behind them: those are not
        # a base to return to, so detaching shows no curve and says why rather than keeping a
        # posterior on screen while claiming the prior.
        self._base_glyphs = {}
        for n in self.ir.nodes:
            src = n.glyph.source if n.glyph is not None else ""
            if src.startswith("posterior"):
                self._base_glyphs[n.id] = (None, None, "prior unavailable (posterior-only IR)")
            else:
                self._base_glyphs[n.id] = (n.glyph, n.glyph_data, n.glyph_note)
        self._plate_panels: dict | None = None  # built once; independent of idata
        # a prebuilt IR shows what it was given until something is attached or detached
        self._diagnostics = self._apply_data(idata) if idata is not None else {}
        self.layout = layout(self.ir, rankdir=rankdir)
        self._svg = to_svg(self.ir, self.layout, legend=legend)

    # ---- outputs ---------------------------------------------------------------
    def to_svg(self) -> str:
        return self._svg

    def save(self, path):
        """Save to .svg / .png / .pdf (format from the extension)."""
        return render_static.save(self._svg, path)

    # ---- live update -----------------------------------------------------------
    def update(self, idata: Any = None):
        """Re-render this diagram's data layer against ``idata`` — in place.

        Sample in one cell, call this in the next, and every prior curve becomes its posterior
        **without the diagram moving**: the same ``LayoutResult`` is reused whenever no node's
        size changes, which is the common case (a 30px density strip becomes a 30px density
        strip). ``update(None)`` restores the as-built view, so the two are a toggle.

        Geometry is not assumed. A posterior can change a node's size class — an MvNormal's
        pairplot square becoming a pooled KDE strip, or a `Flat` that had no glyph at all
        gaining one — so every node is re-measured and a full relayout runs if any differ.
        Honest either way; the no-move promise is just the fast path.

        This is imperative mutation, deliberately outside marimo's dataflow graph — which is
        exactly what lets the figure change without being rebuilt. Passing ``idata=`` to
        ``view()`` remains the pure-dataflow alternative.
        """
        self._diagnostics = self._apply_data(idata)

        if self._layout_still_fits():
            layout_result = self.layout  # nothing resized: the diagram must not jump
        else:
            logging.getLogger(__name__).debug(
                "update(): a glyph changed size class, so the diagram is being laid out again"
            )
            layout_result = self.layout = layout(self.ir, rankdir=self._rankdir)

        self._svg = to_svg(self.ir, layout_result, legend=self._legend)
        if self._widget is not None:
            self._widget.spec = self._build_spec()
        return self

    def _apply_data(self, idata: Any) -> dict:
        """Set the data layer (posterior glyphs, overlays, diagnostics) from ``idata`` — or clear
        all of it together when ``idata`` is None. Returns the model-level diagnostics."""
        from .adapters.glyph_data import posterior_glyph, predictive_glyph
        from .adapters.pymc import overlays_for

        self._idata = idata
        # a deterministic that anchors a recognized effect (a non-centered hierarchy, shrunken
        # coefficients) is a quantity with a posterior of its own, not just a transfer function
        effects = {
            b.anchor
            for b in self.ir.blocks
            if b.anchor and (b.variant == "noncentered" or b.kind == "shrinkage")
        }
        for n in self.ir.nodes:
            base_spec, base_data, base_note = self._base_glyphs[n.id]
            # deterministics depict a transfer FUNCTION (their posterior goes to the card); an
            # observed node shows its data — against the model's replicates when the caller
            # supplied predictive draws
            post = None
            if idata is not None and n.role == "observed":
                post = predictive_glyph(n.id, n.dist, idata)
            elif idata is not None and (
                n.role == "latent" or (n.role == "deterministic" and n.id in effects)
            ):
                post = posterior_glyph(n.id, n.dist, idata)
            if post is not None:
                n.glyph, n.glyph_data = post
                n.glyph_note = None
            else:
                n.glyph, n.glyph_data, n.glyph_note = base_spec, base_data, base_note
            n.overlays = overlays_for(n.id, n.role, n.dims, idata)
        # diagnostics follow the attached idata, and are cleared when it is removed — a diagram
        # showing its prior must not still be badged with a run it is no longer displaying
        return diagnostics.annotate(self.ir, idata)

    @property
    def idata(self):
        """The ``InferenceData`` currently attached (from the constructor or :meth:`update`), or
        ``None`` when the diagram is showing its priors.

        The IR deliberately ships only *curves* — a peak-normalized grid per node — because it is
        JSON-serializable against a published schema. ``OverlayRef`` is its pointer back to the
        draws, and this handle is what makes that pointer dereferenceable: without it an interval
        or ROPE annotation would have nothing to compute from. Held, never serialized, never in
        the IR. ``update(None)`` always returns to the prior layer, whether the idata came from
        the constructor or from :meth:`update`.
        """
        return self._idata

    def _layout_still_fits(self) -> bool:
        """Whether every node still wants exactly the box the current layout gave it."""
        for n in self.ir.nodes:
            box = self.layout.node_boxes.get(n.id)
            if box is None:
                return False
            lw, lh = geometry.label_px_size(n.label_svg)
            kind = n.glyph.kind if n.glyph else None
            w, h = geometry.node_size(lw, lh, kind, n.glyph_data, **geometry.flags(n))
            if abs(w - box.w) > 0.5 or abs(h - box.h) > 0.5:
                return False
        return True

    def marginals_svg(self, node_id: str, page: int = 0) -> str:
        """Labelled per-element posterior intervals for ``node_id`` (coordinate order, 24 rows
        per page, one axis shared across pages) — the static twin of the widget card's panel.
        Empty when no idata is attached or it has no draws for that node."""
        from .adapters.glyph_data import marginal_rows
        from .render_svg import render_marginals_panel

        if self.ir.node(node_id) is None:
            raise ValueError(f"bayesdag: unknown node {node_id!r}")
        m = marginal_rows(node_id, self._idata, page=page) if self._idata is not None else None
        return render_marginals_panel(node_id, m) if m else ""

    @property
    def _can_expand_plates(self) -> bool:
        return bool(self._model is not None and self.ir.plates and self._ppc_draws)

    def expand_plates(self) -> dict:
        """Compute (once) the plate prior-predictive panels: ``{plate_id: {"panel": svg}}``.

        This forward-simulates the user's model — the one place bayesdag samples anything, and by
        far the most expensive thing it can do. So it is **on demand**: building a widget costs
        nothing, and the cost is paid the first time someone actually opens a plate. That is what
        makes a slider → rebuild → re-render loop usable, where a per-construction simulation
        would put seconds between every drag.

        One simulation yields every plate, so all of them are cached together. The panels describe
        the PRIOR and cannot change when a posterior is attached, which is also why ``update()``
        never re-runs this. ``ppc_draws=0`` disables it entirely.
        """
        if self._plate_panels is not None:
            return self._plate_panels
        panels: dict = {}
        if self._can_expand_plates:
            try:
                from .adapters.ppc import prior_predictive_expansions
                from .render_svg import render_plate_panel

                expansions = prior_predictive_expansions(
                    self._model, self.ir, draws=self._ppc_draws
                )
                for pid, exp in expansions.items():
                    if exp.get("status", "ready") != "ready":
                        panels[pid] = {"status": exp["status"], "reason": exp.get("reason", "")}
                        continue
                    panel = render_plate_panel(exp)
                    panels[pid] = (
                        {"status": "ready", "panel": panel}
                        if panel
                        else {"status": "unavailable", "reason": "nothing to draw"}
                    )
            except Exception:
                panels = {}
                logging.getLogger(__name__).debug(
                    "plate prior-predictive expansion failed; the widget degrades to no plate "
                    "panels",
                    exc_info=True,
                )
        self._plate_panels = panels
        return panels

    def _on_plate_expand(self, change) -> None:
        """A plate was clicked in the browser: compute the panels and push them back."""
        if not change["new"] or self._widget is None:
            return
        if self._plate_panels is None:
            self.expand_plates()
            self._widget.spec = self._build_spec()

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
                "glyph_note": n.glyph_note,
                "parents": parents.get(n.id, []),
                "children": children.get(n.id, []),
                "blanket": blanket,
                "ancestors": ancestors,
                "descendants": descendants,
                # the SAME MathJax SVG embedded in the diagram -> the tooltip/card show real
                # rendered math (parity), not raw LaTeX source. Token ids are stripped: these
                # copies land in the same document as the diagram and would duplicate its ids.
                "label_svg": strip_token_ids(n.label_svg) if n.label_svg else n.label_svg,
                # hedged diagnostic rows for the pinned card ([] when there is no idata)
                "diag": diagnostics.describe(n.diag),
            }
            # every glyph-bearing node gets a large pinned-card panel built straight from the
            # precomputed glyph_data (JS just injects this SVG): observed nodes keep the
            # histogram + best-fit overlay; everything else renders its glyph large with the
            # hedged source caption + coord labels.
            if self._idata is not None and n.role in ("latent", "deterministic"):
                from .adapters.glyph_data import marginal_rows

                m = marginal_rows(n.id, self._idata)
                if m and (m["n"] > 1 or n.role == "deterministic"):
                    from .render_svg import render_marginals_panel

                    nodes[n.id]["marginals"] = {
                        "svg": render_marginals_panel(n.id, m),
                        "page": m["page"],
                        "pages": m["pages"],
                    }
            if n.glyph_data:
                if n.role == "observed":
                    panel = render_observed_panel(n.id, n.dist, n.glyph_data)
                else:
                    panel = render_node_panel(n)
                if panel:
                    nodes[n.id]["panel"] = panel
        # Panels only if they have already been computed — clicking a plate is what triggers the
        # simulation. `expandable` tells the JS which plates to offer the affordance for.
        plates = self._plate_panels or {}
        # widget SVG omits the legend by default (hover surfaces the same info); the static
        # `self._svg` keeps it. Re-render is cheap (layout + math are already computed).
        widget_svg = (
            self._svg
            if self._widget_legend == self._legend
            else to_svg(self.ir, self.layout, legend=self._widget_legend)
        )
        # posterior-geometry panels, keyed by the node whose card offers them (the scale that
        # forms the funnel's neck). Built only for a run that actually diverged.
        aux: dict = {}
        for view_ir in self.ir.aux_views:
            from .render_svg import render_joint_panel

            panel = render_joint_panel(view_ir)
            if panel and view_ir.edge and len(view_ir.vars) == 2:
                child, scale = view_ir.vars
                aux.setdefault(view_ir.edge[0], []).append(
                    {"label": f"joint: {child} vs log({scale})", "panel": panel}
                )

        spec = {
            "svg": widget_svg,
            "nodes": nodes,
            "plates": plates,
            "aux": aux,
            "expandable": [p.id for p in self.ir.plates] if self._can_expand_plates else [],
        }
        if self._model is not None:
            from .adapters.ppc import potential_note

            note = potential_note(self._model)
            if note:
                spec["ppc_note"] = note + "; prior-predictive panels are off"
                spec["expandable"] = []
        if self._diagnostics.get("divergences"):
            d, total = self._diagnostics["divergences"], self._diagnostics["draws"]
            spec["diagnostics"] = (
                f"{d} divergent transition{'s' if d != 1 else ''} out of {total} draws — "
                "inspect the flagged nodes; divergences mean the sampler struggled with the "
                "geometry, not that the model is wrong."
            )
        return spec

    def widget(self):
        if self._widget is None:
            from .widget import ModelGraphWidget

            self._widget = ModelGraphWidget(spec=self._build_spec())
            self._widget.observe(self._on_plate_expand, names="expanded_plate")
        return self._widget

    # ---- linked views ----------------------------------------------------------
    def ui(self):
        """The marimo UI element for this diagram, so a *neighbouring cell* can read which node
        is selected: ``w = view(model).ui()`` in one cell, ``w.value["selected_node"]`` in the
        next. marimo re-runs the reader whenever the selection changes, which makes the diagram
        a navigation surface for the whole ArviZ ecosystem::

            w = bayesdag.view(model, idata=idata).ui()          # cell 1
            sel = w.value.get("selected_node")                  # cell 2
            az.plot_trace(idata, var_names=[sel]) if sel else mo.md("click a node")

        The node id IS the constrained idata variable name, so it drops straight into
        ``var_names``. Displaying the view directly also works, but only a *named* element can
        be read back — hence this method. ``w.value`` carries every synced trait, so a cell
        reading it also re-runs when the diagram itself is updated (see ``update``).
        """
        import marimo as mo

        return mo.ui.anywidget(self.widget())

    def on_select(self, fn):
        """Call ``fn(node_id)`` whenever the selection changes (``""`` when cleared).

        The callback form, for Jupyter and for scripts; in marimo prefer :meth:`ui`, whose
        dataflow does this without a callback. Returns an unsubscribe callable.
        """
        w = self.widget()

        def _handler(change):
            fn(change["new"])

        w.observe(_handler, names="selected_node")
        return lambda: w.unobserve(_handler, names="selected_node")

    @property
    def selected_node(self) -> str:
        """The currently pinned node id (``""`` if none) — the same trait the JS writes."""
        return self.widget().selected_node if self._widget is not None else ""

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
    ppc_draws: int | None = 200,
    hide_index_data: bool = True,
) -> ModelGraphView:
    """Visualize a PyMC model (or a ``ModelIR``). Returns a :class:`ModelGraphView` that
    renders interactively in a notebook and statically elsewhere.

    ``legend`` (default ``True``) embeds a context-aware legend in the **static** figure.
    ``widget_legend`` (default ``False``) controls it for the **interactive** widget, which
    omits the legend by default since hovering a node already surfaces the same information.
    ``var_names`` (default ``None`` = everything) restricts the diagram to those variables
    plus their direct parents — one hop of context (``pm.model_to_graphviz(var_names=…)`` keeps
    all ancestors instead).
    ``hide_index_data`` (default ``True``) leaves out integer ``pm.Data`` used only as a gather
    index (``county_idx``) when every consumer's equation already shows its name
    (``a[county_idx]``); ``False`` keeps those nodes.
    ``ppc_draws`` (default 200) is the draw count for the **interactive** plate
    prior-predictive panels, which forward-simulate the model; ``0`` skips them entirely.
    Static rendering never pays this cost.
    """
    return ModelGraphView(
        model_or_ir,
        idata=idata,
        rankdir=rankdir,
        legend=legend,
        widget_legend=widget_legend,
        var_names=var_names,
        ppc_draws=ppc_draws,
        hide_index_data=hide_index_data,
    )
