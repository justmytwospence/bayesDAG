"""Layout via Graphviz ``dot`` (the default ``LayoutBackend``).

We use ``dot`` purely as a layout *oracle*: build DOT with each node sized to its rendered
label, run ``dot -Tjson0``, parse node positions + cluster boxes + edge splines, and apply
ONE coordinate transform (points, y-up bottom-left -> px, y-down top-left). Then a
param-edge post-pass re-routes each edge whose ``target_token_id`` is set to the exact
token anchor inside the child's equation (computed from the MathJax token fractions);
unresolved edges keep their spline (center-anchored).

`dot` is the only requirement (a small system binary). An ELK-subprocess backend can slot
in behind the same signature later.
"""

from __future__ import annotations

import json
import subprocess
from typing import Optional

from .. import geometry, mathsvg
from ..ir import Box, LayoutResult, ModelIR


def _render_labels(ir: ModelIR) -> dict[str, dict]:
    """Render each node's label to SVG (set ``node.label_svg``) and collect px size +
    fractional token anchors. Falls back to a size estimate if math isn't available."""
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


def _build_dot(ir: ModelIR, info: dict[str, dict], rankdir: str) -> str:
    lines = [
        "digraph G {",
        f"  graph [rankdir={rankdir}, nodesep=0.32, ranksep=0.6, pad=0.1, "
        "newrank=true, compound=true];",
        '  node [shape=box, fixedsize=true, label=""];',
    ]
    member_of: dict[str, str] = {}
    for p in ir.plates:
        for m in p.members:
            member_of[m] = p.id

    def node_line(n) -> str:
        w, h = geometry.node_size(info[n.id]["w"], info[n.id]["h"], n.role)
        return f'    {json.dumps(n.id)} [width={w / 72.0:.4f}, height={h / 72.0:.4f}];'

    by_id = {n.id: n for n in ir.nodes}
    for p in ir.plates:
        lines.append(f'  subgraph "cluster_{p.id}" {{')
        lines.append(f'    label={json.dumps(p.label)}; labelloc=b; labeljust=r; style=rounded;')
        for m in p.members:
            if m in by_id:
                lines.append(node_line(by_id[m]))
        lines.append("  }")
    for n in ir.nodes:
        if n.id not in member_of:
            lines.append(node_line(n))
    for e in ir.edges:
        # Heavily weight edges INSIDE a plate so the plate's spine stays straight/vertical
        # and only the cross-plate (external-parent) edges bend -> fewer crossings.
        same_plate = (
            member_of.get(e.source) is not None
            and member_of.get(e.source) == member_of.get(e.target)
        )
        weight = 8 if same_plate else 1
        lines.append(f"  {json.dumps(e.source)} -> {json.dumps(e.target)} [weight={weight}];")
    lines.append("}")
    return "\n".join(lines)


def _run_dot(dot_text: str) -> dict:
    proc = subprocess.run(
        ["dot", "-Tjson0"], input=dot_text, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"graphviz `dot` failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def _parse_spline(pos: str, height: float) -> list[list[float]]:
    pts: list[tuple[float, float]] = []
    end: Optional[tuple[float, float]] = None
    for tok in pos.split():
        if tok.startswith("e,"):
            _, x, y = tok.split(",")
            end = (float(x), float(y))
        elif tok.startswith("s,"):
            continue
        else:
            x, y = tok.split(",")
            pts.append((float(x), float(y)))
    if end is not None:
        pts.append(end)
    return [[x, height - y] for (x, y) in pts]


def layout(ir: ModelIR, *, rankdir: str = "TB") -> LayoutResult:
    info = _render_labels(ir)
    data = _run_dot(_build_dot(ir, info, rankdir))

    _, _, gw, gh = (float(v) for v in data["bb"].split(","))
    res = LayoutResult(canvas=Box(0.0, 0.0, gw, gh))

    objects = data.get("objects", [])
    idx2name = [o.get("name", "") for o in objects]
    by_id = {n.id: n for n in ir.nodes}

    for o in objects:
        name = o.get("name", "")
        if "pos" in o and name in by_id:  # a node
            px, py = (float(v) for v in o["pos"].split(","))
            w = float(o["width"]) * 72.0
            h = float(o["height"]) * 72.0
            box = Box(px - w / 2.0, (gh - py) - h / 2.0, w, h)
            n = by_id[name]
            n.box = box
            res.node_boxes[name] = box
            # absolute token anchors from the label's fractional anchors
            lw, lh = info[name]["w"], info[name]["h"]
            ox, oy = geometry.label_origin(box, lw, lh)
            anchors: dict[str, Box] = {}
            for tok, (fx, fy, fw, fh) in info[name]["bboxes"].items():
                anchors[tok] = Box(ox + fx * lw, oy + fy * lh, fw * lw, fh * lh)
            n.port_anchors = anchors
            res.node_token_anchors[name] = anchors
        elif name.startswith("cluster_") and "bb" in o:  # a plate
            llx, lly, urx, ury = (float(v) for v in o["bb"].split(","))
            pid = name[len("cluster_"):]
            res.plate_boxes[pid] = Box(llx, gh - ury, urx - llx, ury - lly)

    for e in data.get("edges", []):
        src = idx2name[e["tail"]]
        tgt = idx2name[e["head"]]
        pts = _parse_spline(e.get("pos", ""), gh) if e.get("pos") else []
        # param-edge post-pass: re-route to "enter from the top" and stop just ABOVE the
        # token so the arrowhead points at the glyph without covering it (and the final
        # segment is ~vertical, which removes the interior-diagonal crossings).
        edge_ir = next(
            (x for x in ir.edges if x.source == src and x.target == tgt), None
        )
        if edge_ir is not None and edge_ir.target_token_id and pts:
            anchor = res.node_token_anchors.get(tgt, {}).get(edge_ir.target_token_id)
            tbox = res.node_boxes.get(tgt)
            if anchor is not None and tbox is not None:
                cx = anchor.x + anchor.w / 2.0  # token center-x
                top = anchor.y                  # token visual top
                pts = pts[:-1] + [[cx, tbox.y], [cx, top - geometry.STANDOFF]]
        res.edge_paths[f"{src}|{tgt}"] = pts

    return res
