"""Layout via ELK (``elkjs``) run **in-process** in ``mini-racer`` V8 — the default backend.

Why ELK over Graphviz ``dot``: plates are first-class here but second-class in ``dot``,
which flattens clusters before crossing minimization, so an external scalar parent of a
plate-internal node (``sigma -> y`` with ``y`` in the ``obs`` plate) gets shoved aside and
its edge crosses. ELK's ``layered`` with ``hierarchyHandling=INCLUDE_CHILDREN`` lays out a
node and all descendants in one pass, so cross-hierarchy edges participate in global
crossing minimization (the reason Mermaid moved dagre->ELK). ELK fixes node *placement*;
edges are then drawn by our own smooth cubic (``common.simple_edge_path``) and the
token-level port anchors are computed by us from the MathJax bboxes (engine-independent).

Node-free integration (proven by the M0.3 spike), in its own V8 isolate (MathJax uses a
separate one — two isolates total, ~35MB / ~67MB RSS respectively):
  * the modern ``mini-racer`` (bpcreech) has an event loop + ``JSPromise.get()``;
  * a GWT globals shim (``window``/``global``/``self`` = ``globalThis``);
  * a ``setTimeout(fn, 0) -> 1ms`` coercion (mini-racer's 0ms path throws);
  * a synchronous in-process worker shim wiring ``elk-api.js`` <-> ``elk-worker.min.js``
    (the bundled build's default worker needs a Web ``Worker``, absent in bare V8).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .. import geometry
from ..ir import Box, LayoutResult, ModelIR
from ..labels import LHS_TOKEN
from . import common

_DIRECTION = {"TB": "DOWN", "BT": "UP", "LR": "RIGHT", "RL": "LEFT"}
_PLATE_PADDING = "[top=14.0,left=14.0,bottom=28.0,right=14.0]"  # bottom: room for the plate label
# corner radii of the rendered node chrome (render_svg._CHROME) — edges must exit/enter on the
# straight part of the border, not the cut corner (deterministic boxes use a small radius).
_CORNER_RX = {"latent": 9.0, "observed": 9.0, "deterministic": 3.0, "data": 11.0, "potential": 3.0, "factor": 3.0}

# Synchronous in-process worker: run the GWT engine in its own `self` (inheriting
# Error/Math from globalThis) and expose a Worker-like handle the elk-api talks to.
_WORKER_SHIM = r"""
globalThis.__mkWorker = function(workerSrc){ return function(){
  var W = Object.create(globalThis); var apiOnmsg = null;
  W.postMessage = function(m){ if(apiOnmsg) apiOnmsg({data:m}); };
  W.importScripts = function(){};
  var mod = {exports:{}};
  (new Function('self','postMessage','module','exports','window','global', workerSrc))
    (W, W.postMessage, mod, mod.exports, W, W);
  return { postMessage:function(m){ W.onmessage({data:m}); },
    addEventListener:function(t,fn){ if(t==='message') apiOnmsg=fn; },
    removeEventListener:function(){},
    set onmessage(fn){ apiOnmsg=fn; }, get onmessage(){ return apiOnmsg; },
    terminate:function(){} }; }; };
"""


def _static_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "static"


def _first_error_line(exc: BaseException) -> str:
    """The informative first line of a V8/ELK error (the rest is minified GWT stack)."""
    lines = str(exc).splitlines() or [repr(exc)]
    for ln in lines:
        if "Error" in ln or "Exception" in ln:
            return ln.strip()
    return lines[0].strip()


def _graph_counts(g: dict) -> tuple[int, int]:
    kids = g.get("children") or []
    n, e = len(kids), len(g.get("edges") or [])
    for k in kids:
        kn, ke = _graph_counts(k)
        n, e = n + kn, e + ke
    return n, e


class ElkEngine:
    """Lazy in-process ELK layout engine. Construct once and reuse (V8 init is the cost)."""

    def __init__(self, api_path: Optional[Path] = None, worker_path: Optional[Path] = None) -> None:
        sd = _static_dir()
        self._api_path = api_path or (sd / "elk-api.js")
        self._worker_path = worker_path or (sd / "elk-worker.min.js")
        self._ctx = None
        self._executor = None

    @property
    def available(self) -> bool:
        try:
            import py_mini_racer  # noqa: F401
        except Exception:
            return False
        return self._api_path.exists() and self._worker_path.exists()

    def _worker(self):
        # All mini-racer work runs on ONE dedicated thread. mini-racer binds its event loop to
        # the thread that creates the context; if that's a thread with a live asyncio loop (a
        # marimo cell), `promise.get()` either asserts or deadlocks. A single private thread that
        # never runs an asyncio loop lets mini-racer own its loop and `.get()` block safely.
        if self._executor is None:
            import concurrent.futures

            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="bayesdag-elk"
            )
        return self._executor

    def _context(self):  # must run on the worker thread (see _worker)
        if self._ctx is None:
            from py_mini_racer import MiniRacer

            ctx = MiniRacer()
            ctx.eval("var global=globalThis, self=globalThis, window=globalThis;")
            # mini-racer setTimeout(fn,0) hits a non-promise Atomics.waitAsync path -> coerce 0->1ms
            ctx.eval(
                "var __bd_st=globalThis.setTimeout;"
                "globalThis.setTimeout=function(f,d){return __bd_st(f,(d&&d>0)?d:1);};"
            )
            ctx.eval(_WORKER_SHIM)
            ctx.eval("globalThis.__elkWorkerSrc = " + json.dumps(self._worker_path.read_text()))
            ctx.eval(
                "var module={exports:{}},exports=module.exports,require=function(){return {};};\n"
                + self._api_path.read_text()
                + "\nglobalThis.__ELK = (module.exports.default||module.exports);"
            )
            ctx.eval("globalThis.__elk = new __ELK({ workerFactory: __mkWorker(__elkWorkerSrc) });")
            ctx.eval("globalThis.__elkRun = async (g)=>JSON.stringify(await __elk.layout(g));")
            self._ctx = ctx
        return self._ctx

    def _layout_blocking(self, graph: dict, timeout_ms: int) -> dict:
        ctx = self._context()
        ctx.eval("globalThis.__elkG = " + json.dumps(graph))
        try:
            return json.loads(ctx.eval("__elkRun(__elkG)").get(timeout=timeout_ms))
        except Exception as exc:
            # str(exc) is a minified GWT stack; surface only the useful first line
            n, e = _graph_counts(graph)
            raise RuntimeError(
                f"ELK layout failed: {_first_error_line(exc)} "
                f"(graph: {n} nodes / {e} edges; likely a bayesdag bug — please report)"
            ) from exc
        finally:
            ctx.eval("globalThis.__elkG = undefined")  # don't retain the last graph in V8

    def layout_graph(self, graph: dict, timeout_ms: int = 30000) -> dict:
        # marshal onto the dedicated thread (context is created there on first use)
        return self._worker().submit(self._layout_blocking, graph, timeout_ms).result()


_ENGINE: Optional[ElkEngine] = None


def get_engine() -> ElkEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ElkEngine()
    return _ENGINE


def available() -> bool:
    return get_engine().available


def _port_id(node_id: str, tok: str) -> str:
    return f"{node_id}\x1e{tok}"  # unit-separator keeps it unambiguous vs node/token names


def _build_graph(ir: ModelIR, info: dict, rankdir: str) -> dict:
    by_id = {n.id: n for n in ir.nodes}
    member_plate: dict[str, str] = {}
    for p in ir.plates:
        for m in p.members:
            member_plate[m] = p.id  # PyMC plates are keyed by dim-set -> one plate per var

    # Which tokens are edge targets on each node -> fixed-position ports at the token's x, so
    # ELK orders each node's parents to match the equation's token order (no crossing pile-up).
    targeted: dict[str, set] = {}
    for e in ir.edges:
        if e.target_token_id:
            targeted.setdefault(e.target, set()).add(e.target_token_id)
    edge_sources = {e.source for e in ir.edges}

    port_ids: set[str] = set()

    def node_json(n) -> dict:
        lw, lh = info[n.id]["w"], info[n.id]["h"]
        w, h = geometry.node_size(
            lw, lh, n.role, n.glyph.kind if n.glyph else None, n.glyph_data if n.glyph else None
        )
        d: dict = {"id": n.id, "width": float(w), "height": float(h)}
        ports = []
        bboxes = info[n.id]["bboxes"]
        label_ox = (w - lw) / 2.0  # label is centered in the node (geometry.label_origin)
        for tok in sorted(targeted.get(n.id, ())):
            bb = bboxes.get(tok)
            if bb is None:
                continue
            fx, _fy, fw, _fh = bb
            # token center x — must equal common.node_token_anchors' cx so ELK routes to the exact
            # token column (clamp only to the node, never narrow it off the token)
            px = max(0.0, min(w, label_ox + (fx + fw / 2.0) * lw))
            pid = _port_id(n.id, tok)
            port_ids.add(pid)
            ports.append(
                {"id": pid, "x": px, "y": 0.0, "width": 1.0, "height": 1.0,
                 "layoutOptions": {"elk.port.side": "NORTH"}}
            )
        # a deterministic equation's value flows out of its LHS variable: give it a SOUTH port at
        # that variable's x so ELK routes the outgoing edge from the variable, not the box edge.
        if n.role == "deterministic" and n.id in edge_sources:
            bb = bboxes.get(LHS_TOKEN)
            if bb is not None:
                fx, _fy, fw, _fh = bb
                px = max(0.0, min(w, label_ox + (fx + fw / 2.0) * lw))
                pid = _port_id(n.id, LHS_TOKEN)
                port_ids.add(pid)
                ports.append(
                    {"id": pid, "x": px, "y": float(h), "width": 1.0, "height": 1.0,
                     "layoutOptions": {"elk.port.side": "SOUTH"}}
                )
        if ports:
            d["ports"] = ports
            d["layoutOptions"] = {"elk.portConstraints": "FIXED_POS"}
        return d

    def plate_json(p) -> dict:
        children = [node_json(by_id[m]) for m in p.members if m in by_id and member_plate.get(m) == p.id]
        children += [plate_json(cp) for cp in ir.plates if cp.parent == p.id]
        return {"id": p.id, "layoutOptions": {"elk.padding": _PLATE_PADDING}, "children": children}

    root_children = [plate_json(p) for p in ir.plates if p.parent is None]
    root_children += [node_json(n) for n in ir.nodes if n.id not in member_plate]

    # Feed edges to ELK grouped by target and ordered by their target-token x, so
    # considerModelOrder orders each node's parents to match the equation's token order.
    def _tok_x(e) -> float:
        bb = info.get(e.target, {}).get("bboxes", {}).get(e.target_token_id or "")
        return (bb[0] + bb[2] / 2.0) if bb else 0.5

    ordered = sorted(enumerate(ir.edges), key=lambda ie: (ie[1].target, _tok_x(ie[1])))
    edges = []
    for i, e in ordered:
        spid = _port_id(e.source, LHS_TOKEN)
        src = spid if spid in port_ids else e.source  # deterministic sources exit via their LHS port
        pid = _port_id(e.target, e.target_token_id) if e.target_token_id else None
        tgt = pid if pid in port_ids else e.target
        edges.append({"id": f"e{i}", "sources": [src], "targets": [tgt]})
    return {
        "id": "root",
        "layoutOptions": {
            "elk.algorithm": "layered",
            "elk.direction": _DIRECTION.get(rankdir, "DOWN"),
            "elk.hierarchyHandling": "INCLUDE_CHILDREN",
            "elk.randomSeed": "1",  # determinism for golden tests
            "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
            "elk.layered.crossingMinimization.greedySwitchHierarchical.type": "TWO_SIDED",
            # ELK routes the edges itself, as right angles around the nodes (we draw what it returns)
            "elk.edgeRouting": "ORTHOGONAL",
            # balanced placement centers a single-child node (e.g. y) under its parent
            "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
            "elk.layered.nodePlacement.bk.fixedAlignment": "BALANCED",
            "elk.spacing.nodeNode": "34",
            "elk.layered.spacing.nodeNodeBetweenLayers": "40",
            "elk.spacing.edgeNode": "18",
        },
        "children": root_children,
        "edges": edges,
    }


def _collect_boxes(data: dict) -> dict[str, Box]:
    """Walk the ELK result, accumulating parent offsets -> absolute boxes for every node."""
    boxes: dict[str, Box] = {}

    def walk(node: dict, ox: float, oy: float) -> None:
        for c in node.get("children", []):
            x, y = ox + float(c.get("x", 0.0)), oy + float(c.get("y", 0.0))
            boxes[c["id"]] = Box(x, y, float(c.get("width", 0.0)), float(c.get("height", 0.0)))
            walk(c, x, y)

    walk(data, 0.0, 0.0)
    return boxes


def _container_offsets(data: dict) -> dict[str, tuple[float, float]]:
    """Absolute (x, y) of every node/plate container, for converting an edge's container-relative
    section points to absolute coordinates."""
    off: dict[str, tuple[float, float]] = {}

    def walk(node: dict, ox: float, oy: float) -> None:
        for c in node.get("children", []):
            x, y = ox + float(c.get("x", 0.0)), oy + float(c.get("y", 0.0))
            off[c["id"]] = (x, y)
            walk(c, x, y)

    walk(data, 0.0, 0.0)
    return off


def _seg_clear(x0: float, y0: float, x1: float, y1: float, boxes: list) -> bool:
    """True if the axis-aligned segment (x0,y0)->(x1,y1) misses every box interior (1px inset)."""
    for t in range(25):
        x = x0 + (x1 - x0) * t / 24.0
        y = y0 + (y1 - y0) * t / 24.0
        for b in boxes:
            if b.x + 1 <= x <= b.x + b.w - 1 and b.y + 1 <= y <= b.y + b.h - 1:
                return False
    return True


_RUN_TOL = 1.0  # treat points within this x of each other as one vertical run (absorbs ELK's 0.5px
# NORTH-port half-offset so a snapped terminal run is exactly one column)


def _attach_source(pts: list, e, res: LayoutResult, roles: dict, anchor) -> None:
    """Exit the source aligned under the target token (a clean vertical when the token is over the
    source, otherwise the box edge nearest it). The exit stays on the source's STRAIGHT bottom
    border (>= its corner radius from the ends) so a rounded box doesn't leave the edge floating off
    its cut corner; the whole leading vertical run is shifted, and only if the moved drop + its
    connector stay clear of every other node (never creates a through-node)."""
    sb = res.node_boxes.get(e.source)
    if sb is None or abs(pts[1][0] - pts[0][0]) >= _RUN_TOL:
        return
    # A deterministic's value exits its BOX BOTTOM at the LHS-variable's fixed SOUTH port (ELK placed
    # it there). With a visible box that bottom edge is the real boundary — keep the exit under the
    # LHS variable (so `theta = …` flows out from `theta`); no target-alignment, no lift.
    if roles.get(e.source) == "deterministic":
        return
    tb = res.node_boxes.get(e.target)
    tok = (anchor.x + anchor.w / 2.0) if anchor is not None else (
        tb.x + tb.w / 2.0 if tb is not None else pts[0][0]
    )
    rx = _CORNER_RX.get(roles.get(e.source), 9.0)
    lo, hi = sb.x + rx, sb.x + sb.w - rx
    if lo >= hi:
        lo = hi = sb.x + sb.w / 2.0
    sx0 = pts[0][0]
    run = 1
    while run < len(pts) - 1 and abs(pts[run][0] - sx0) < _RUN_TOL:
        run += 1
    obstacles = [b for nid, b in res.node_boxes.items() if nid not in (e.source, e.target)]
    y0, y1, nxt_x = pts[0][1], pts[run - 1][1], pts[run][0]
    for cand in (min(max(tok, lo), hi), min(max(sx0, lo), hi)):
        if _seg_clear(cand, y0, cand, y1, obstacles) and _seg_clear(cand, y1, nxt_x, y1, obstacles):
            for k in range(run):
                pts[k][0] = cand
            break


def _attach_target(pts: list, e, res: LayoutResult, roles: dict, anchor) -> None:
    """Land the arrow a standoff above the target box's TOP border, in the token's column: every node
    (now including deterministic equation boxes) is bordered, so the arrowhead clears the border and
    points at the box edge directly above the parameter it feeds — the column carries 'which token',
    the box stays uncrossed. Snap the WHOLE trailing vertical run to the token column (mirror of the
    source side) so the final approach is exactly vertical, then collapse any same-column point that
    would sit below the landing so the last segment always descends cleanly."""
    tb = res.node_boxes.get(e.target)
    if anchor is None or tb is None:
        return
    tx = anchor.x + anchor.w / 2.0
    ex0 = pts[-1][0]
    j = len(pts) - 1
    while j > 0 and abs(pts[j - 1][0] - ex0) < _RUN_TOL:  # back over ELK's trailing vertical run
        j -= 1
    for k in range(j, len(pts)):
        pts[k][0] = tx
    pts[-1][1] = tb.y - geometry.STANDOFF
    # drop same-column points at/below the landing so the final segment never reverses
    while len(pts) > 2 and abs(pts[-2][0] - tx) < 0.5 and pts[-2][1] >= pts[-1][1] - 0.5:
        del pts[-2]


def _collect_edges(ir: ModelIR, data: dict, res: LayoutResult) -> None:
    """Consume ELK's native orthogonal edge routes. Each ELK edge (id ``e{i}`` -> ``ir.edges[i]``)
    carries a ``container`` whose absolute position offsets the section's relative points; the
    polyline is ``[startPoint] + bendPoints + [endPoint]``. Both terminals are then attached
    symmetrically (``_attach_source``/``_attach_target``) and emitted as a rounded cubic chain."""
    offsets = _container_offsets(data)
    by_eid = {e.get("id"): e for e in data.get("edges", [])}
    roles = {n.id: n.role for n in ir.nodes}
    for i, e in enumerate(ir.edges):
        elk_e = by_eid.get(f"e{i}")
        if elk_e is None:
            continue
        ox, oy = offsets.get(elk_e.get("container", "root"), (0.0, 0.0))
        pts: list[list[float]] = []
        for s in elk_e.get("sections", []):
            sp = s.get("startPoint")
            if sp is not None:
                pts.append([ox + sp["x"], oy + sp["y"]])
            for bp in s.get("bendPoints", []):
                pts.append([ox + bp["x"], oy + bp["y"]])
            ep = s.get("endPoint")
            if ep is not None:
                pts.append([ox + ep["x"], oy + ep["y"]])
        if len(pts) < 2:
            continue
        anchor = (
            res.node_token_anchors.get(e.target, {}).get(e.target_token_id)
            if e.target_token_id
            else None
        )
        _attach_source(pts, e, res, roles, anchor)
        _attach_target(pts, e, res, roles, anchor)
        res.edge_paths[f"{e.source}|{e.target}"] = common.orthogonal_path(pts, radius=5.0)


def layout(ir: ModelIR, *, rankdir: str = "TB") -> LayoutResult:
    info = common.render_labels(ir)
    data = get_engine().layout_graph(_build_graph(ir, info, rankdir))

    boxes = _collect_boxes(data)
    canvas = Box(0.0, 0.0, float(data.get("width", 0.0)), float(data.get("height", 0.0)))
    res = LayoutResult(canvas=canvas)

    for n in ir.nodes:
        b = boxes.get(n.id)
        if b is None:
            continue
        n.box = b
        res.node_boxes[n.id] = b
        anchors = common.node_token_anchors(b, info[n.id]["w"], info[n.id]["h"], info[n.id]["bboxes"])
        n.port_anchors = anchors
        res.node_token_anchors[n.id] = anchors

    for p in ir.plates:
        b = boxes.get(p.id)
        if b is not None:
            res.plate_boxes[p.id] = b

    # ELK routed the edges orthogonally; consume those routes (no custom routing / reflow).
    _collect_edges(ir, data, res)

    # sync node geometry back onto the IR nodes + recompute the canvas to cover everything
    for n in ir.nodes:
        if n.id in res.node_boxes:
            n.box = res.node_boxes[n.id]
            n.port_anchors = res.node_token_anchors.get(n.id, n.port_anchors)
    allboxes = list(res.node_boxes.values()) + list(res.plate_boxes.values())
    if allboxes:
        w = max(b.x + b.w for b in allboxes)
        h = max(b.y + b.h for b in allboxes)
        res.canvas = Box(0.0, 0.0, max(res.canvas.w, w), max(res.canvas.h, h))
    return res
