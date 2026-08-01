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

from .. import geometry
from ..ir import Box, LayoutResult, ModelIR
from . import common
from .common import render_labels as _render_labels


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
        w, h = geometry.node_size(
            info[n.id]["w"],
            info[n.id]["h"],
            n.glyph.kind if n.glyph else None,
            n.glyph_data if n.glyph else None,
        )
        return f"    {json.dumps(n.id)} [width={w / 72.0:.4f}, height={h / 72.0:.4f}];"

    by_id = {n.id: n for n in ir.nodes}
    for p in ir.plates:
        lines.append(f'  subgraph "cluster_{p.id}" {{')
        lines.append(f"    label={json.dumps(p.label)}; labelloc=b; labeljust=r; style=rounded;")
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
        same_plate = member_of.get(e.source) is not None and member_of.get(
            e.source
        ) == member_of.get(e.target)
        weight = 8 if same_plate else 1
        lines.append(f"  {json.dumps(e.source)} -> {json.dumps(e.target)} [weight={weight}];")
    lines.append("}")
    return "\n".join(lines)


def _flip_spline(
    pos: str, gh: float
) -> tuple[list[tuple[float, float]], tuple[float, float] | None]:
    """Parse a graphviz edge ``pos`` into flipped px coords.

    Format: ``e,EX,EY  B0 B1 B2 …`` — the B-points are cubic-Bezier control points (``B0``
    start, then triples), and ``EX,EY`` is the arrowhead tip. Returns ``(control_pts, tip)``.
    """
    tip: tuple[float, float] | None = None
    body: list[tuple[float, float]] = []
    for t in pos.split():
        if t.startswith("e,"):
            _, x, y = t.split(",")
            tip = (float(x), gh - float(y))
        elif t.startswith("s,"):
            continue
        else:
            x, y = t.split(",")
            body.append((float(x), gh - float(y)))
    return body, tip


_DOT_TIMEOUT_S = 60.0


def _run_dot(dot_text: str) -> dict:
    try:
        proc = subprocess.run(
            ["dot", "-Tjson0"],
            input=dot_text,
            capture_output=True,
            text=True,
            timeout=_DOT_TIMEOUT_S,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "BAYESDAG_LAYOUT=dot is set but the Graphviz 'dot' binary is not on PATH — "
            "install graphviz (brew install graphviz / apt install graphviz) or unset "
            "BAYESDAG_LAYOUT"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        # dot can spin indefinitely on a pathological graph; bound it rather than hanging
        # the interpreter (ELK, the default backend, already takes a timeout)
        raise RuntimeError(
            f"graphviz `dot` did not finish within {_DOT_TIMEOUT_S:.0f}s — "
            "unset BAYESDAG_LAYOUT to use the default ELK backend"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(f"graphviz `dot` failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def layout(ir: ModelIR, *, rankdir: str = "TB") -> LayoutResult:
    common.reset_geometry(ir)  # never let a previous layout's coordinates survive into this one
    info = _render_labels(ir)
    data = _run_dot(_build_dot(ir, info, rankdir))

    _, _, gw, gh = (float(v) for v in data["bb"].split(","))
    res = LayoutResult(canvas=Box(0.0, 0.0, gw, gh))

    objects = data.get("objects", [])
    by_id = {n.id: n for n in ir.nodes}

    for o in objects:
        name = o.get("name", "")
        if "pos" in o and name in by_id:  # a node
            px, py = (float(v) for v in o["pos"].split(","))
            w = float(o["width"]) * 72.0
            h = float(o["height"]) * 72.0
            box = Box(px - w / 2.0, (gh - py) - h / 2.0, w, h)
            n = by_id[name]
            anchors = common.node_token_anchors(
                box, info[name]["w"], info[name]["h"], info[name]["bboxes"]
            )
            n.box, n.port_anchors = box, anchors
            res.node_boxes[name] = box
            res.node_token_anchors[name] = anchors
        elif name.startswith("cluster_") and "bb" in o:  # a plate
            llx, lly, urx, ury = (float(v) for v in o["bb"].split(","))
            pid = name[len("cluster_") :]
            res.plate_boxes[pid] = Box(llx, gh - ury, urx - llx, ury - lly)

    # Edges follow dot's OWN spline routing (it minimizes crossings) rendered as one smooth
    # cubic chain — no faceting. `edge_paths` is a flat list of cubic control points
    # (`[p0, c1, c2, p1, c1, c2, p2, …]` -> `M p0 C c1 c2 p1 C …`). For a port-edge we keep
    # dot's routing for the bulk and graft a short vertical tail onto the exact target token.
    gvid2name = {o.get("_gvid"): o.get("name") for o in objects}
    dot_pos: dict[tuple[str, str], str] = {}
    for de in data.get("edges", []):
        s = gvid2name.get(de.get("tail"))
        t = gvid2name.get(de.get("head"))
        if s is not None and t is not None and "pos" in de:
            dot_pos[(s, t)] = de["pos"]

    for e in ir.edges:
        sb = res.node_boxes.get(e.source)
        tb = res.node_boxes.get(e.target)
        if sb is None or tb is None:
            continue
        anchor = (
            res.node_token_anchors.get(e.target, {}).get(e.target_token_id)
            if e.target_token_id
            else None
        )
        body, tip = _flip_spline(dot_pos.get((e.source, e.target), ""), gh)
        if len(body) >= 4:
            n_seg = (len(body) - 1) // 3
            ctrl = body[: 1 + 3 * n_seg]
            if anchor is not None:
                ax, ay = anchor.x + anchor.w / 2.0, anchor.y - geometry.STANDOFF
                lx, ly = ctrl[-1]
                dy = max(12.0, 0.4 * abs(ay - ly))
                ctrl = [*ctrl, (lx, ly + dy), (ax, ay - dy), (ax, ay)]
            elif tip is not None:
                lx, ly = ctrl[-1]
                ctrl = [*ctrl, (lx, ly), tip, tip]
        else:
            # No usable spline -> a plain cubic with vertical tangents at both ends.
            ex, ey = sb.x + sb.w / 2.0, sb.y + sb.h
            if anchor is not None:
                nx, ny = anchor.x + anchor.w / 2.0, anchor.y - geometry.STANDOFF
            else:
                nx, ny = tb.x + tb.w / 2.0, tb.y
            dy = max(16.0, 0.42 * abs(ny - ey))
            ctrl = [(ex, ey), (ex, ey + dy), (nx, ny - dy), (nx, ny)]
        res.edge_paths[f"{e.source}|{e.target}"] = [[x, y] for x, y in ctrl]

    return res
