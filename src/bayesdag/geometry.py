"""Geometry constants + helpers shared by the layout backend AND the SVG emitter.

Both import these so node sizing and label/glyph placement agree exactly — that shared
agreement (plus one ``LayoutResult``) is what makes the static and interactive renderers
identical by construction.
"""

from __future__ import annotations

import re

from .ir import Box

EX_PX = 8.0  # px per MathJax 'ex' unit (we set the embedded SVG to this scale)
PAD = 10.0  # node interior padding
GLYPH_H = 30.0  # reserved height for a 1-D distribution-shape glyph strip
GAP = 4.0  # gap between label and glyph
MIN_W = 56.0
MIN_H = 38.0
TAG_H = 12.0  # a recognized-structure tag strip above the label ("partial pooling · centered")
CAP_H = 11.0  # a caption row below everything (glyph note / elision badge)
CAPTION_PX_PER_CHAR = 4.6  # 8px italic system-ui, a conservative average advance
TAG_PX_PER_CHAR = 4.8
MAX_TEXT_W = 190.0  # a caption/tag widens its node up to this, then truncates (the card has it all)
STANDOFF = 4.0  # gap above a token edge's VISIBLE target surface (box border for bordered nodes,
# token glyph for borderless) so the arrowhead tip sits ~3px clear and never covers the surface

# 2-D glyphs need a near-square block (scaled by dimension); multi-element 1-D glyphs (several
# overlaid curves, a fan, a stem plot) read better in a slightly taller strip than a single curve.
_TALL_GLYPHS = frozenset({"heatmap", "pairplot"})
_TALLER_STRIP = frozenset({"fan", "stem", "cutpoints", "simplex", "mixture"})
# `multi`, `forest` and `ppc` deliberately use the STANDARD strip: they replace a prior density
# or a data histogram when idata is attached, and keeping the height is what lets update() leave
# the diagram where it is.
# A deterministic transfer function (the `curve` kind) has a CANONICAL shape — its width carries
# no information, so it must not stretch with the equation. Cap + center it under the equation;
# a wide `θ = μ + τ·η` no longer smears the S-curve/line across the whole node.
_FN_GLYPH_MAX_W = 72.0

_W = re.compile(r'\bwidth="([\d.]+)ex"')
_H = re.compile(r'\bheight="([\d.]+)ex"')


def has_glyph_data(glyph_kind: str | None, glyph_data: dict | None) -> bool:
    """Whether a node actually carries a drawable glyph. Sizing/placement gate on THIS (presence),
    not on role — so a deterministic with a transfer-function glyph reserves a strip, while an
    equation-only deterministic (and any glyph-less node) stays compact."""
    return glyph_kind is not None and bool(glyph_data)


def caption_text(n) -> str | None:
    """The one caption row a node carries: an elision badge (a warning) wins over a glyph note (a
    plain caption). Duck-typed on the NodeIR fields so this module stays IR-light."""
    reason = getattr(n, "elision_reason", None)
    if reason:
        return "⚠ " + reason
    return getattr(n, "glyph_note", None) or None


def flags(n) -> dict:
    """The optional text rows a node reserves space for — passed to every sizing/placement helper
    by BOTH layout and render, so they agree exactly."""
    return {"tag": getattr(n, "tag", None) or None, "caption": caption_text(n)}


def fit_text(text: str, width: float, px_per_char: float) -> str:
    """Truncate to the drawable width with an ellipsis (the full text lives in the card)."""
    n = max(4, int((width - 2 * 7.0) / px_per_char))
    return text if len(text) <= n else text[: n - 1] + "…"


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


def glyph_area(glyph_kind: str | None, glyph_data: dict | None = None) -> tuple[float, float]:
    """The (min_width, height) px a glyph region wants. 2-D glyphs (heatmap/pairplot) ask for a
    near-square block that grows with the matrix dimension (so big covariances/adjacencies aren't
    squished); multi-element 1-D glyphs a taller strip; a plain density the standard strip. This is
    what generalizes node sizing to whatever a node needs to show."""
    data = glyph_data or {}
    if glyph_kind in _TALL_GLYPHS:
        mat = data.get("cov") or data.get("matrix") or []
        d = len(mat) if mat else 2
        side = max(88.0, min(d * 26.0, 180.0))
        return side, side
    if glyph_kind in _TALLER_STRIP:
        return 0.0, 44.0
    return 0.0, GLYPH_H


def node_size(
    label_w: float,
    label_h: float,
    glyph_kind: str | None = None,
    glyph_data: dict | None = None,
    *,
    tag: str | None = None,
    caption: str | None = None,
) -> tuple[float, float]:
    if has_glyph_data(glyph_kind, glyph_data):
        gmin_w, gh = glyph_area(glyph_kind, glyph_data)
        block = gh + GAP
    else:
        gmin_w, block = 0.0, 0.0
    text_w = 0.0
    if tag:
        text_w = max(text_w, min(MAX_TEXT_W, len(tag) * TAG_PX_PER_CHAR + 14.0))
    if caption:
        text_w = max(text_w, min(MAX_TEXT_W, len(caption) * CAPTION_PX_PER_CHAR + 14.0))
    w = max(MIN_W, label_w + 2 * PAD, gmin_w + 2 * PAD, text_w)
    h = PAD + (TAG_H if tag else 0.0) + label_h + block + (CAP_H if caption else 0.0) + PAD
    return w, max(MIN_H, h)


def label_origin(box: Box, label_w: float, *, tag: str | None = None) -> tuple[float, float]:
    """Top-left px of the label SVG within a node box (centered horizontally, top-aligned, below
    the tag strip when there is one)."""
    return box.x + (box.w - label_w) / 2.0, box.y + PAD + (TAG_H if tag else 0.0)


def glyph_rect(
    box: Box,
    label_h: float,
    glyph_kind: str | None = None,
    glyph_data: dict | None = None,
    *,
    tag: str | None = None,
) -> Box | None:
    if not has_glyph_data(glyph_kind, glyph_data):
        return None
    top = box.y + PAD + (TAG_H if tag else 0.0) + label_h + GAP
    _, gh = glyph_area(glyph_kind, glyph_data)
    if glyph_kind in _TALL_GLYPHS:  # a centered (near-)square block for 2-D glyphs
        side = min(box.w - 2 * PAD, gh)
        return Box(box.x + (box.w - side) / 2.0, top, side, side)
    if (
        glyph_kind == "curve"
    ):  # canonical transfer shape: bounded width, centered under the equation
        cw = min(box.w - 2 * PAD, _FN_GLYPH_MAX_W)
        return Box(box.x + (box.w - cw) / 2.0, top, cw, gh)
    return Box(box.x + PAD, top, box.w - 2 * PAD, gh)
