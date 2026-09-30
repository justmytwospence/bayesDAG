"""``ModelGraphView`` — the public entry point. Builds the IR + layout + shared SVG once,
then renders the right surface per environment: an interactive anywidget in Jupyter/marimo
(when available), a static SVG everywhere else. The widget consumes the *same* SVG, so the
two are identical by construction.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from . import blocks as _blocks
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
        detail: str = "overview",
        blocks: str | None = "auto",
        annotations: Any = (),
        expand: Any = (),
    ) -> None:
        if detail not in ("overview", "full"):
            raise ValueError(f"bayesdag.view: detail must be 'overview' or 'full', not {detail!r}")
        if blocks not in ("auto", None):
            raise ValueError(f"bayesdag.view: blocks must be 'auto' or None, not {blocks!r}")
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
        base = to_ir(model_or_ir, blocks=blocks == "auto")
        self.ir = copy.deepcopy(base) if prebuilt else base
        if blocks is None:
            self.ir = _blocks.strip(self.ir)
        elif prebuilt and not self.ir.blocks:
            self.ir = _blocks.detect_ir(self.ir)  # structure an IR carries by itself
        # the user's declarations win over anything recognized
        self.ir = _blocks.apply(self.ir, list(annotations or ()))
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
        self._detail = detail
        self._expand: set[str] = set(expand or ())
        self._layouts: dict = {}  # (detail, expanded, rankdir) -> LayoutResult
        self._render()

    # ---- projection + layout ---------------------------------------------------
    def _project(self, detail: str) -> ModelIR:
        return _blocks.overview(self.ir, expand=self._expand) if detail == "overview" else self.ir

    def _layout_for(self, shown: ModelIR, detail: str):
        key = (detail, frozenset(self._expand), self._rankdir)
        cache = self.__dict__.setdefault("_label_cache", {})
        _restore_label_svgs(shown, cache)  # a projection is re-derived; its labels are not
        res = self._layouts.get(key)
        if res is None or not _fits(shown, res):
            res = self._layouts[key] = layout(shown, rankdir=self._rankdir)
        else:
            _reapply_geometry(shown, res)
        _remember_label_svgs(shown, cache)
        return res

    def _render(self) -> None:
        """Project (overview or full), lay out (cached per projection) and emit the SVG."""
        shown = self._project(self._detail)
        self.layout = self._layout_for(shown, self._detail)
        self.display_ir = shown
        self._svg = to_svg(shown, self.layout, legend=self._legend)

    def _push(self) -> None:
        if self._widget is not None:
            self._widget.spec = self._build_spec()
            if self._widget.detail != self._detail:
                self._widget.detail = self._detail
            if sorted(self._widget.expanded_blocks) != sorted(self._expand):
                self._widget.expanded_blocks = sorted(self._expand)

    @property
    def detail(self) -> str:
        """``"overview"`` (recognized/annotated structures contracted) or ``"full"``."""
        return self._detail

    @property
    def blocks(self) -> list:
        """The structures on this diagram (:class:`~bayesdag.ir.BlockIR`), most specific first."""
        return list(self.ir.blocks)

    def set_detail(self, detail: str):
        """Switch between the overview and the full variable graph, in place."""
        if detail not in ("overview", "full"):
            raise ValueError(f"detail must be 'overview' or 'full', not {detail!r}")
        if detail != self._detail:
            self._detail = detail
            self._render()
            self._push()
        return self

    def expand(self, block_ids):
        """Show these blocks' member variables instead of their compound box."""
        ids = {block_ids} if isinstance(block_ids, str) else set(block_ids)
        unknown = ids - {b.id for b in self.ir.blocks}
        if unknown:
            raise ValueError(f"bayesdag: unknown block(s) {sorted(unknown)}")
        self._expand |= ids
        self._render()
        self._push()
        return self

    def collapse(self, block_ids):
        """Undo :meth:`expand` for these blocks."""
        ids = {block_ids} if isinstance(block_ids, str) else set(block_ids)
        self._expand -= ids
        self._render()
        self._push()
        return self

    # ---- outputs ---------------------------------------------------------------
    def to_svg(self, *, detail: str | None = None) -> str:
        """The figure's SVG; ``detail`` renders the other projection without switching."""
        if detail is None or detail == self._detail:
            return self._svg
        if detail not in ("overview", "full"):
            raise ValueError(f"detail must be 'overview' or 'full', not {detail!r}")
        shown = self._project(detail)
        return to_svg(shown, self._layout_for(shown, detail), legend=self._legend)

    def save(self, path, *, detail: str | None = None):
        """Save to .svg / .png / .pdf (format from the extension)."""
        return render_static.save(self.to_svg(detail=detail), path)

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
        # the cached layouts stay valid exactly when no node changed size (the common case: a
        # density strip becomes a density strip); `_layout_for` re-measures and relays otherwise
        self._render()
        self._push()
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
        """Whether every shown node still wants exactly the box the current layout gave it."""
        return _fits(self.display_ir, self.layout)

    def marginals_svg(self, node_id: str, page: int = 0) -> str:
        """Labelled per-element posterior intervals for ``node_id`` (coordinate order, 24 rows
        per page, one axis shared across pages) — the static twin of the widget card's panel.
        Empty when no idata is attached or it has no draws for that node."""
        return self.panel_svg("marginals", node_id=node_id, page=page)

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

    def _node_entry(self, n, g, moral, nx, parents, children) -> dict:
        blanket = sorted(moral.neighbors(n.id)) if (moral is not None and n.id in moral) else []
        # Transitive lineage for the directional trace shown on pin (click): every node that
        # flows INTO this one (ancestors) vs. every node it flows into (descendants). Computed
        # here so the JS layer stays graph-algorithm-free.
        in_graph = nx is not None and n.id in g
        entry = {
            "role": n.role,
            "dist": n.dist,
            "observed": n.observed,
            "dims": list(n.dims),
            "params": [{"name": p.name, "value": p.value_tex} for p in n.params],
            "transform": n.transform,
            "glyph_note": n.glyph_note,
            "tag": n.tag,
            "block": n.block,
            "parents": parents.get(n.id, []),
            "children": children.get(n.id, []),
            "blanket": blanket,
            "ancestors": sorted(nx.ancestors(g, n.id)) if in_graph else [],
            "descendants": sorted(nx.descendants(g, n.id)) if in_graph else [],
            # the SAME MathJax SVG embedded in the diagram -> the tooltip/card show real
            # rendered math (parity), not raw LaTeX source. Token ids are stripped: these
            # copies land in the same document as the diagram and would duplicate its ids.
            "label_svg": strip_token_ids(n.label_svg) if n.label_svg else _label_svg(n),
            # hedged diagnostic rows for the pinned card ([] when there is no idata)
            "diag": diagnostics.describe(n.diag),
        }
        # the variable whose posterior a card shows: itself, or a compound node's anchor
        var = self.ir.block(n.block).anchor if n.role == "block" and n.block else n.id
        canon = self.ir.node(var) if var else None
        if (
            self._idata is not None
            and canon is not None
            and canon.role
            in (
                "latent",
                "deterministic",
            )
        ):
            m = self._marginals(var, 0)
            if m and (m["n"] > 1 or canon.role == "deterministic" or n.role == "block"):
                entry["marginals"] = m
                entry["marginals_of"] = var
        # every glyph-bearing node gets a large pinned-card panel built straight from the
        # precomputed glyph_data (JS just injects this SVG)
        if n.glyph_data:
            if n.role == "observed":
                panel = render_observed_panel(n.id, n.dist, n.glyph_data)
            else:
                panel = render_node_panel(n)
            if panel:
                entry["panel"] = panel
        return entry

    def fitted_panels(self, block_id: str) -> list[str]:
        """The fitted-structure panels offered for a block (they need a model and a posterior)."""
        from .adapters.fitted import applicable

        b = self.ir.block(block_id)
        if b is None:
            raise ValueError(f"bayesdag: unknown block {block_id!r}")
        if self._model is None or self._idata is None:
            return []
        return applicable(b, self.ir)

    def _fitted(self, kind: str, b) -> tuple[str, str, str]:
        """(status, svg, reason) of a fitted panel, cached per attached idata."""
        from .adapters import fitted
        from .render_panels import RENDERERS, render_unavailable

        cache = self.__dict__.setdefault("_fitted_cache", {})
        key = (kind, b.id, id(self._idata))
        if key in cache:
            return cache[key]
        label = fitted.PANELS[kind][0]
        try:
            data = fitted.compute(kind, self._model, self.ir, self._idata, b)
            out = ("ready", RENDERERS[kind](data), "")
        except fitted.Unavailable as exc:
            out = ("unavailable", render_unavailable(label, str(exc)), str(exc))
        except Exception as exc:  # a panel failure is reported, never raised into the notebook
            logging.getLogger(__name__).debug("fitted panel %s failed", kind, exc_info=True)
            reason = f"could not compute this panel ({type(exc).__name__})"
            out = ("error", render_unavailable(label, reason), reason)
        cache[key] = out
        return out

    def _marginals(self, node_id: str, page: int) -> dict | None:
        from .adapters.glyph_data import marginal_rows
        from .render_svg import render_marginals_panel

        m = marginal_rows(node_id, self._idata, page=page) if self._idata is not None else None
        if not m:
            return None
        return {
            "svg": render_marginals_panel(node_id, m),
            "page": m["page"],
            "pages": m["pages"],
            "n": m["n"],
        }

    def _build_spec(self) -> dict:
        """SVG + per-node detail + adjacency (Markov blanket) + structures for the widget."""
        from .adapters.graph import to_networkx

        shown = self.display_ir
        g = to_networkx(shown)
        try:
            import networkx as nx

            moral = nx.moral_graph(g)
        except Exception:
            nx = None
            moral = None
        parents: dict[str, list] = {n.id: [] for n in shown.nodes}
        children: dict[str, list] = {n.id: [] for n in shown.nodes}
        for e in shown.edges:
            children.setdefault(e.source, []).append(e.target)
            parents.setdefault(e.target, []).append(e.source)
        nodes = {n.id: self._node_entry(n, g, moral, nx, parents, children) for n in shown.nodes}
        # the members hidden inside compound nodes still get cards (the block card lists them)
        cg = to_networkx(self.ir)
        cparents: dict[str, list] = {}
        cchildren: dict[str, list] = {}
        for e in self.ir.edges:
            cchildren.setdefault(e.source, []).append(e.target)
            cparents.setdefault(e.target, []).append(e.source)
        for n in self.ir.nodes:
            if n.id not in nodes:
                entry = self._node_entry(n, cg, None, nx, cparents, cchildren)
                entry["hidden"] = True
                nodes[n.id] = entry
        contracted = {n.block: n.id for n in shown.nodes if n.role == "block" and n.block}
        blocks_spec = {}
        from .adapters.fitted import PANELS

        for b in self.ir.blocks:
            blocks_spec[b.id] = {
                "title": b.title,
                "kind": b.kind,
                "variant": b.variant,
                "provenance": b.provenance,
                "workflow": b.workflow,
                "anchor": b.anchor,
                "members": list(b.members),
                "roles": {k: list(v) for k, v in b.roles.items()},
                "notes": list(b.notes),
                "references": list(b.references),
                "collapsible": b.collapsible and bool(b.members),
                "contracted": b.id in contracted,
                "display_id": contracted.get(b.id),
                "expanded": b.id in self._expand,
                # the classic fitted pictures for this structure, computed when asked for
                "panels": [{"kind": k, "label": PANELS[k][0]} for k in self.fitted_panels(b.id)],
            }
            # a tag-only structure's card lives on its anchor node
            if b.anchor in nodes and not b.members:
                nodes[b.anchor].setdefault("blocks_here", []).append(b.id)
            for m in b.members:
                if m in nodes and b.id in self._expand:
                    nodes[m]["in_block"] = b.id
        # Panels only if they have already been computed — clicking a plate is what triggers the
        # simulation. `expandable` tells the JS which plates to offer the affordance for.
        plates = self._plate_panels or {}
        # widget SVG omits the legend by default (hover surfaces the same info); the static
        # `self._svg` keeps it. Re-render is cheap (layout + math are already computed).
        widget_svg = (
            self._svg
            if self._widget_legend == self._legend
            else to_svg(shown, self.layout, legend=self._widget_legend)
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
            "blocks": blocks_spec,
            "detail": self._detail,
            # whether an overview differs from the full graph at all (the toggle is pointless
            # otherwise)
            "has_overview": bool(_blocks.contracting(self.ir)) or bool(self._expand),
            "expandable": [p.id for p in shown.plates] if self._can_expand_plates else [],
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

            self._widget = ModelGraphWidget(
                spec=self._build_spec(),
                detail=self._detail,
                expanded_blocks=sorted(self._expand),
            )
            self._widget.observe(self._on_plate_expand, names="expanded_plate")
            self._widget.observe(self._on_detail, names="detail")
            self._widget.observe(self._on_expanded, names="expanded_blocks")
            self._widget.observe(self._on_panel_request, names="panel_request")
        return self._widget

    def _on_detail(self, change) -> None:
        if change["new"] in ("overview", "full") and change["new"] != self._detail:
            self.set_detail(change["new"])

    def _on_expanded(self, change) -> None:
        want = {b for b in (change["new"] or []) if self.ir.block(b) is not None}
        if want != self._expand:
            self._expand = want
            self._render()
            self._push()

    def _on_panel_request(self, change) -> None:
        """The JS asks for a panel page (it computes nothing itself); answer on panel_response."""
        req = change["new"] or {}
        if not req or self._widget is None:
            return
        try:
            from .adapters.fitted import PANELS

            kind = req.get("kind", "marginals")
            if kind in PANELS and req.get("block_id") and self.ir.block(req["block_id"]):
                status, svg, reason = self._fitted(kind, self.ir.block(req["block_id"]))
                resp = {"request": req, "status": status, "svg": svg, "reason": reason}
                self._widget.panel_response = resp
                return
            svg = self.panel_svg(
                kind,
                node_id=req.get("node_id"),
                block_id=req.get("block_id"),
                page=int(req.get("page", 0)),
            )
            resp = {"request": req, "status": "ready" if svg else "unavailable", "svg": svg}
        except Exception as exc:  # a bad request is answered, never raised into the kernel
            resp = {"request": req, "status": "error", "reason": str(exc), "svg": ""}
        self._widget.panel_response = resp

    def panel_svg(
        self,
        kind: str,
        *,
        node_id: str | None = None,
        block_id: str | None = None,
        page: int = 0,
    ) -> str:
        """The same Python-rendered panels the widget card offers, as standalone SVG.

        ``kind`` is ``"marginals"`` (per-element intervals, paged; needs idata), ``"node"``
        (the node's glyph drawn large), ``"block"`` (a structure's summary), or one of the
        fitted-structure panels (:meth:`fitted_panels` lists them for a block: ``shrinkage``,
        ``population``, ``pooling``, ``group_lines``, ``intercept_slope``, ``effect``, ``kappa``,
        ``local_scale``, ``measured``). Exactly one of ``node_id``/``block_id``; an empty string
        means no data to draw; a fitted panel that cannot be drawn honestly says why."""
        if (node_id is None) == (block_id is None):
            raise ValueError("bayesdag.panel_svg: pass exactly one of node_id / block_id")
        from .adapters.fitted import PANELS

        if kind in PANELS:
            b = self.ir.block(block_id) if block_id else None
            if b is None:
                raise ValueError(f"bayesdag: the {kind!r} panel needs a known block_id")
            return self._fitted(kind, b)[1]
        if block_id is not None:
            b = self.ir.block(block_id)
            if b is None:
                raise ValueError(f"bayesdag: unknown block {block_id!r}")
            if kind == "block":
                from .render_svg import render_block_panel

                return render_block_panel(b)
            if b.anchor is None:
                return ""
            node_id = b.anchor
        n = self.ir.node(node_id)
        if n is None:
            raise ValueError(f"bayesdag: unknown node {node_id!r}")
        if kind == "marginals":
            m = self._marginals(node_id, page)
            return m["svg"] if m else ""
        if kind == "node":
            if n.role == "observed" and n.glyph_data:
                return render_observed_panel(n.id, n.dist, n.glyph_data)
            return render_node_panel(n)
        raise ValueError(f"bayesdag.panel_svg: unknown panel kind {kind!r}")

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


def _label_svg(n) -> str | None:
    """A label rendered for a card (a member hidden inside a compound node was never laid out)."""
    if not n.label_tex:
        return None
    try:
        from . import mathsvg

        r = mathsvg.get_renderer()
        if not r.available:
            return None
        svg, _ = r.render_with_bboxes(n.label_tex, display=True)
        return strip_token_ids(svg)
    except Exception:
        return None


def _fits(shown: ModelIR, res) -> bool:
    """Whether every node of ``shown`` still wants exactly the box ``res`` gave it."""
    if set(res.node_boxes) != {n.id for n in shown.nodes}:
        return False
    for n in shown.nodes:
        box = res.node_boxes[n.id]
        svg = n.label_svg
        if svg is None:
            return False  # an unmeasured label (a fresh projection): lay out to measure it
        lw, lh = geometry.label_px_size(svg)
        kind = n.glyph.kind if n.glyph else None
        w, h = geometry.node_size(lw, lh, kind, n.glyph_data, **geometry.flags(n))
        if abs(w - box.w) > 0.5 or abs(h - box.h) > 0.5:
            return False
    return True


def _reapply_geometry(shown: ModelIR, res) -> None:
    """A reused layout's boxes/anchors, mirrored onto a freshly derived projection's nodes."""
    for n in shown.nodes:
        n.box = res.node_boxes.get(n.id)
        n.port_anchors = res.node_token_anchors.get(n.id, {})


def _remember_label_svgs(shown: ModelIR, cache: dict) -> None:
    for n in shown.nodes:
        if n.label_svg:
            cache[(n.id, n.label_tex)] = n.label_svg


def _restore_label_svgs(shown: ModelIR, cache: dict) -> None:
    """A projection is re-derived on every render; its labels were measured the first time."""
    for n in shown.nodes:
        if n.label_svg is None:
            n.label_svg = cache.get((n.id, n.label_tex))


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
    detail: str = "overview",
    blocks: str | None = "auto",
    annotations: Any = (),
    expand: Any = (),
) -> ModelGraphView:
    """Visualize a PyMC model (or a ``ModelIR``). Returns a :class:`ModelGraphView` that
    renders interactively in a notebook and statically elsewhere.

    ``legend`` (default ``True``) embeds a context-aware legend in the **static** figure.
    ``widget_legend`` (default ``False``) controls it for the **interactive** widget, which
    omits the legend by default since hovering a node already surfaces the same information.
    ``var_names`` (default ``None`` = everything) restricts the diagram to those variables
    plus their direct parents — one hop of context (``pm.model_to_graphviz(var_names=…)`` keeps
    all ancestors instead).
    ``detail`` (default ``"overview"``) contracts each recognized or annotated structure —
    a non-centered hierarchy, a shrinkage prior, a declared measurement model — into one box
    carrying its equations; ``"full"`` shows every variable. ``blocks`` (``"auto"``) recognizes
    structures from the graph; ``None`` turns that off (``annotations`` still apply).
    ``annotations`` is a sequence of :class:`bayesdag.BlockSpec` the user declares; ``expand``
    lists block ids to show expanded from the start.
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
        detail=detail,
        blocks=blocks,
        annotations=annotations,
        expand=expand,
    )
