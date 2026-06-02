"""Layout backends. ``layout(ir)`` is the single Python pass that produces the
``LayoutResult`` both renderers consume (the parity guarantee)."""

from __future__ import annotations

from .graphviz_backend import layout

__all__ = ["layout"]
