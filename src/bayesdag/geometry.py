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
STANDOFF = 4.0  # gap above a token edge's VISIBLE target surface (box border for bordered nodes,
# token glyph for borderless) so the arrowhead tip sits ~3px clear and never covers the surface

# 2-D glyphs need a near-square block (scaled by dimension); multi-element 1-D glyphs (several
# overlaid curves, a fan, a stem plot) read better in a slightly taller strip than a single curve.
_TALL_GLYPHS = frozenset({"heatmap", "pairplot"})
_TALLER_STRIP = frozenset({"fan", "stem", "cutpoints", "simplex", "mixture"})
# A deterministic transfer function (the `curve` kind) has a CANONICAL shape — its width carries
# no information, so it must not stretch with the equation. Cap + center it under the equation;
# a wide `θ = μ + τ·η` no longer smears the S-curve/line across the whole node.
_FN_GLYPH_MAX_W = 72.0

_W = re.compile(r'\bwidth="([\d.]+)ex"')
_H = re.compile(r'\bheight="([\d.]+)ex"')


def has_glyph(role: str) -> bool:
    return role in ("latent", "observed")


def has_glyph_data(glyph_kind: str | None, glyph_data: dict | None) -> bool:
    """Whether a node actually carries a drawable glyph. Sizing/placement gate on THIS (presence),
    not on role — so a deterministic with a transfer-function glyph reserves a strip, while an
    equation-only deterministic (and any glyph-less node) stays compact."""
    return glyph_kind is not None and bool(glyph_data)


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
) -> tuple[float, float]:
    if has_glyph_data(glyph_kind, glyph_data):
        gmin_w, gh = glyph_area(glyph_kind, glyph_data)
        block = gh + GAP
    else:
        gmin_w, block = 0.0, 0.0
    w = max(MIN_W, label_w + 2 * PAD, gmin_w + 2 * PAD)
    h = max(MIN_H, PAD + label_h + block + PAD)
    return w, h


def label_origin(box: Box, label_w: float) -> tuple[float, float]:
    """Top-left px of the label SVG within a node box (centered horizontally, top-aligned)."""
    return box.x + (box.w - label_w) / 2.0, box.y + PAD


def glyph_rect(
    box: Box,
    label_h: float,
    glyph_kind: str | None = None,
    glyph_data: dict | None = None,
) -> Box | None:
    if not has_glyph_data(glyph_kind, glyph_data):
        return None
    top = box.y + PAD + label_h + GAP
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
