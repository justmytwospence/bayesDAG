"""Backend-agnostic layout helpers shared by every ``LayoutBackend``.

Label measurement and token-anchor projection are identical regardless of which engine (ELK,
dot, …) decides node *positions* — keeping them here is what lets the backends differ only in
placement while the rest stays byte-identical (the parity principle). Token-level port anchors
are computed by us from the MathJax bboxes, so they are engine-independent. ``orthogonal_path``
turns an engine's right-angle bend points into the rounded cubic chain the SVG emitter draws.
"""

from __future__ import annotations

import warnings

from .. import geometry, mathsvg
from ..ir import Box, ModelIR

_warned_math_unavailable = False  # once per process — the degradation is global, not per-render


def render_labels(ir: ModelIR) -> dict[str, dict]:
    """Render each node's label to SVG (set ``node.label_svg``) and collect px size +
    fractional token bboxes. Falls back to a size estimate when math isn't available."""
    global _warned_math_unavailable
    renderer = mathsvg.get_renderer()
    use = renderer.available
    if not use and not _warned_math_unavailable and any(n.label_tex for n in ir.nodes):
        warnings.warn(
            "bayesdag: math rendering is unavailable — labels degrade to plain text and "
            "edges to center anchors. Install the 'math' extra "
            "(pip install 'bayesdag[math]') or build the bundle (npm install && npm run build).",
            RuntimeWarning,
            stacklevel=2,
        )
        _warned_math_unavailable = True
    info: dict[str, dict] = {}
    failures: list[tuple[str, Exception]] = []
    for n in ir.nodes:
        svg = None
        bboxes: dict[str, tuple[float, float, float, float]] = {}
        if use and n.label_tex:
            try:
                svg, bboxes = renderer.render_with_bboxes(n.label_tex, display=True)
            except Exception as exc:
                svg, bboxes = None, {}
                failures.append((n.id, exc))
        n.label_svg = svg
        lw, lh = geometry.label_px_size(svg)
        if svg is None and n.label_tex:
            lw = max(lw, 7.0 * len(n.id))  # rough estimate without math
        info[n.id] = {"w": lw, "h": lh, "bboxes": bboxes}
    if failures:
        nid, exc = failures[0]
        warnings.warn(
            f"bayesdag: math rendering failed for {len(failures)} label(s) "
            f"(first: {nid}: {exc}); they degrade to plain text.",
            RuntimeWarning,
            stacklevel=2,
        )
    return info


def node_token_anchors(box: Box, label_w: float, label_h: float, bboxes: dict) -> dict[str, Box]:
    """Project the label's fractional token bboxes into absolute boxes within ``box`` — the
    anchor a port-edge terminates on (token top-center, with a standoff applied at draw)."""
    ox, oy = geometry.label_origin(box, label_w, label_h)
    anchors: dict[str, Box] = {}
    for tok, (fx, fy, fw, fh) in bboxes.items():
        anchors[tok] = Box(ox + fx * label_w, oy + fy * label_h, fw * label_w, fh * label_h)
    return anchors


def orthogonal_path(points: list, radius: float = 5.0) -> list[list[float]]:
    """Render an orthogonal polyline (ELK's right-angle bend points) as a cubic-Bezier chain with
    small rounded corners. Straight runs stay straight; each interior corner is filleted with a
    fixed radius (clamped to half each adjacent segment), giving clean right angles that read as
    'down, over, down' rather than swerves. Output is the ``[p0, c1,c2,p1, c1,c2,p2, …]`` chain
    that ``render_svg._edge`` consumes, so the arrowhead still orients along the final segment."""
    pts: list[list[float]] = []
    for p in points:  # drop consecutive duplicates
        q = [float(p[0]), float(p[1])]
        if not pts or abs(q[0] - pts[-1][0]) > 0.01 or abs(q[1] - pts[-1][1]) > 0.01:
            pts.append(q)
    if len(pts) < 2:
        return pts
    if len(pts) == 2:
        a, b = pts
        return [a, a[:], b[:], b]  # straight line as a degenerate cubic

    def _lerp(p, q, t):
        return [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]

    def _dist(p, q):
        return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5

    out = [pts[0]]
    cur = pts[0]
    for i in range(1, len(pts) - 1):
        prev, v, nxt = pts[i - 1], pts[i], pts[i + 1]
        dp, dn = _dist(prev, v), _dist(v, nxt)
        r = min(radius, dp / 2.0, dn / 2.0)
        a = _lerp(v, prev, r / dp) if dp else v  # corner entry: r before the vertex
        b = _lerp(v, nxt, r / dn) if dn else v   # corner exit: r after the vertex
        out += [_lerp(cur, a, 1 / 3.0), _lerp(cur, a, 2 / 3.0), a]  # straight run into the corner
        out += [v[:], v[:], b]                                       # rounded corner (control at v)
        cur = b
    last = pts[-1]
    out += [_lerp(cur, last, 1 / 3.0), _lerp(cur, last, 2 / 3.0), last]  # final straight run
    return out
