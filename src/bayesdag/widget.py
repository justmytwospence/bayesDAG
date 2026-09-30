"""anywidget front-end. Ships the SAME SVG the static renderer produced (parity by
construction); the JS only adds hover/selection/plate-expansion. Imported lazily so the rest of
bayesdag never requires anywidget."""

from __future__ import annotations

import pathlib

import anywidget
import traitlets

_STATIC = pathlib.Path(__file__).parent / "static"


class ModelGraphWidget(anywidget.AnyWidget):
    _esm = _STATIC / "widget.js"
    _css = _STATIC / "widget.css"

    # The full Python-emitted SVG + (future) adjacency/overlays/aux panels.
    spec = traitlets.Dict({}).tag(sync=True)
    # Two-way state for linked views (read back in marimo/Jupyter).
    selected_node = traitlets.Unicode("").tag(sync=True)
    # Set by the JS when a plate is clicked. The prior-predictive expansion forward-simulates the
    # user's model, so it is computed on demand rather than for every widget ever built; the view
    # observes this, computes, and pushes the panel back through `spec`.
    expanded_plate = traitlets.Unicode("").tag(sync=True)
    # "overview" (structures contracted into compound nodes) or "full"; the JS toggles it and
    # Python re-projects, lays out and pushes `spec` (the JS never computes geometry)
    detail = traitlets.Unicode("overview").tag(sync=True)
    # block ids shown expanded (their member variables instead of one compound box)
    expanded_blocks = traitlets.List(traitlets.Unicode()).tag(sync=True)
    # a selected structure (a presentation id — never a variable, so never in selected_node)
    selected_block = traitlets.Unicode("").tag(sync=True)
    # a page of a Python-rendered panel the card asks for: {kind, node_id|block_id, page}
    panel_request = traitlets.Dict({}).tag(sync=True)
    panel_response = traitlets.Dict({}).tag(sync=True)
