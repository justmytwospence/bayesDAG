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
