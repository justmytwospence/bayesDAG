"""Backend-agnostic layout helpers shared by every ``LayoutBackend``.

The label measurement, token-anchor projection, and the smooth fallback edge are identical
regardless of which engine (ELK, dot, …) decides node *positions* — keeping them here is
what lets the backends differ only in placement while the rest stays byte-identical (the
parity principle). Token-level port anchors are computed by us from the MathJax bboxes, so
they are engine-independent.
"""

from __future__ import annotations

from .. import geometry, mathsvg
from ..ir import Box, ModelIR


def render_labels(ir: ModelIR) -> dict[str, dict]:
    """Render each node's label to SVG (set ``node.label_svg``) and collect px size +
    fractional token bboxes. Falls back to a size estimate when math isn't available."""
    renderer = mathsvg.get_renderer()
    use = renderer.available
    info: dict[str, dict] = {}
    for n in ir.nodes:
        svg = None
        bboxes: dict[str, tuple[float, float, float, float]] = {}
        if use and n.label_tex:
            try:
                svg = renderer.render(n.label_tex, display=True)
                bboxes = mathsvg.token_bboxes(svg)
            except Exception:
                svg, bboxes = None, {}
        n.label_svg = svg
        lw, lh = geometry.label_px_size(svg)
        if svg is None and n.label_tex:
            lw = max(lw, 7.0 * len(n.id))  # rough estimate without math
        info[n.id] = {"w": lw, "h": lh, "bboxes": bboxes}
    return info


def node_token_anchors(box: Box, label_w: float, label_h: float, bboxes: dict) -> dict[str, Box]:
    """Project the label's fractional token bboxes into absolute boxes within ``box`` — the
    anchor a port-edge terminates on (token top-center, with a standoff applied at draw)."""
    ox, oy = geometry.label_origin(box, label_w, label_h)
    anchors: dict[str, Box] = {}
    for tok, (fx, fy, fw, fh) in bboxes.items():
        anchors[tok] = Box(ox + fx * label_w, oy + fy * label_h, fw * label_w, fh * label_h)
    return anchors


def simple_edge_path(sb: Box, tb: Box, anchor: Box | None) -> list[list[float]]:
    """A gentle cubic from the source bottom-center to the target (the specific token for a
    port-edge, else the top-center). The curve heads roughly STRAIGHT at the source and lands
    ~vertically on the target token (so the arrowhead sits just above the glyph, standoff).

    The tangent length is short and capped so a parent that's far to the SIDE gets a direct
    diagonal — not a long horizontal segment hugging the plate boundary on its way over.
    Returned as a 4-point cubic ``[p0, c1, c2, p1]`` (what ``render_svg._edge`` consumes)."""
    ex, ey = sb.x + sb.w / 2.0, sb.y + sb.h
    if anchor is not None:
        nx, ny = anchor.x + anchor.w / 2.0, anchor.y - geometry.STANDOFF
    else:
        nx, ny = tb.x + tb.w / 2.0, tb.y
    span = abs(ny - ey)
    k = max(12.0, min(0.35 * span, 26.0))  # short, capped -> direct (no boundary-hugging swoop)
    # lean the departure toward the target so the edge sets off in its general direction,
    # while the landing stays vertical for a clean arrowhead on the token.
    c1x = ex + 0.25 * (nx - ex)
    return [[ex, ey], [c1x, ey + k], [nx, ny - k], [nx, ny]]
