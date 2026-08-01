"""Shape-first glyph system: a glyph-agnostic registry keyed by ``kind`` (PPL-agnostic,
renders precomputed shape data to SVG) plus the univariate density/histogram kinds.

The density curve is the primary mark; ``interval``/``point`` are optional annotations a
kind may ignore — so non-univariate kinds (heatmap/ternary/rose/joint) register the same
way and are first-class, not exceptions.
"""

from __future__ import annotations

from . import kinds  # noqa: F401  (registers the built-in kinds on import)
from .registry import register, registered_kinds, render

__all__ = ["register", "registered_kinds", "render"]
