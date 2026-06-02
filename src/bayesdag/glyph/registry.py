"""Glyph-kind registry. The core knows nothing about univariate-vs-not: a kind is just a
``render(data, box, **opts) -> svg`` callable keyed by name. New kinds (heatmap, ternary,
rose, kde2d, ...) register identically, so non-univariate glyphs are first-class."""

from __future__ import annotations

from typing import Any, Callable

from ..ir import Box

RenderFn = Callable[..., str]
_REGISTRY: dict[str, RenderFn] = {}


def register(kind: str, fn: RenderFn) -> None:
    _REGISTRY[kind] = fn


def registered_kinds() -> set[str]:
    return set(_REGISTRY)


def render(kind: str, data: dict[str, Any] | None, box: Box, **opts: Any) -> str:
    """Render a glyph of ``kind`` into ``box`` (absolute px). Unknown/empty -> ''."""
    fn = _REGISTRY.get(kind)
    if fn is None or data is None:
        return ""
    return fn(data, box, **opts)
