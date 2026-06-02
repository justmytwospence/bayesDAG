"""Geometry constants + helpers shared by the layout backend AND the SVG emitter.

Both import these so node sizing and label/glyph placement agree exactly — that shared
agreement (plus one ``LayoutResult``) is what makes the static and interactive renderers
identical by construction.
"""

from __future__ import annotations

import re

from .ir import Box

EX_PX = 8.0     # px per MathJax 'ex' unit (we set the embedded SVG to this scale)
PAD = 10.0      # node interior padding
GLYPH_H = 30.0  # reserved height for the distribution-shape glyph strip
GAP = 4.0       # gap between label and glyph
MIN_W = 56.0
MIN_H = 38.0

_W = re.compile(r'\bwidth="([\d.]+)ex"')
_H = re.compile(r'\bheight="([\d.]+)ex"')


def has_glyph(role: str) -> bool:
    return role in ("latent", "observed")


def label_px_size(svg: str | None) -> tuple[float, float]:
    """(width, height) in px for a MathJax SVG (from its ``ex`` dimensions)."""
    if not svg:
        return 40.0, 16.0
    w = _W.search(svg)
    h = _H.search(svg)
    return (
        float(w.group(1)) * EX_PX if w else 40.0,
        float(h.group(1)) * EX_PX if h else 16.0,
    )


def node_size(label_w: float, label_h: float, role: str) -> tuple[float, float]:
    gh = (GLYPH_H + GAP) if has_glyph(role) else 0.0
    w = max(MIN_W, label_w + 2 * PAD)
    h = max(MIN_H, PAD + label_h + gh + PAD)
    return w, h


def label_origin(box: Box, label_w: float, label_h: float) -> tuple[float, float]:
    """Top-left px of the label SVG within a node box (centered horizontally, top-aligned)."""
    return box.x + (box.w - label_w) / 2.0, box.y + PAD


def glyph_rect(box: Box, role: str, label_h: float) -> Box | None:
    if not has_glyph(role):
        return None
    return Box(box.x + PAD, box.y + PAD + label_h + GAP, box.w - 2 * PAD, GLYPH_H)
