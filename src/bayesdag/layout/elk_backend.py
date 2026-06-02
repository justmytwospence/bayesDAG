"""Layout via ELK (``elkjs``) run **in-process** in ``mini-racer`` V8 — the default backend.

Why ELK over Graphviz ``dot``: plates are first-class here but second-class in ``dot``,
which flattens clusters before crossing minimization, so an external scalar parent of a
plate-internal node (``sigma -> y`` with ``y`` in the ``obs`` plate) gets shoved aside and
its edge crosses. ELK's ``layered`` with ``hierarchyHandling=INCLUDE_CHILDREN`` lays out a
node and all descendants in one pass, so cross-hierarchy edges participate in global
crossing minimization (the reason Mermaid moved dagre->ELK). ELK fixes node *placement*;
edges are then drawn by our own smooth cubic (``common.simple_edge_path``) and the
token-level port anchors are computed by us from the MathJax bboxes (engine-independent).

Node-free integration (proven by the M0.3 spike), all in the same V8 we use for MathJax:
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
from . import common

_DIRECTION = {"TB": "DOWN", "BT": "UP", "LR": "RIGHT", "RL": "LEFT"}
_PLATE_PADDING = "[top=14.0,left=14.0,bottom=28.0,right=14.0]"  # bottom: room for the plate label

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
        return json.loads(ctx.eval("__elkRun(__elkG)").get(timeout=timeout_ms))

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

    port_ids: set[str] = set()

    def node_json(n) -> dict:
        lw, lh = info[n.id]["w"], info[n.id]["h"]
        w, h = geometry.node_size(lw, lh, n.role)
        d: dict = {"id": n.id, "width": float(w), "height": float(h)}
        ports = []
        bboxes = info[n.id]["bboxes"]
        label_ox = (w - lw) / 2.0  # label is centered in the node (geometry.label_origin)
        for tok in sorted(targeted.get(n.id, ())):
            bb = bboxes.get(tok)
            if bb is None:
                continue
            fx, _fy, fw, _fh = bb
            px = max(1.0, min(w - 1.0, label_ox + (fx + fw / 2.0) * lw))  # token center x
            pid = _port_id(n.id, tok)
            port_ids.add(pid)
            ports.append(
                {"id": pid, "x": px, "y": 0.0, "width": 1.0, "height": 1.0,
                 "layoutOptions": {"elk.port.side": "NORTH"}}
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
    edges = []
    for i, e in enumerate(ir.edges):
        pid = _port_id(e.target, e.target_token_id) if e.target_token_id else None
        tgt = pid if pid in port_ids else e.target
        edges.append({"id": f"e{i}", "sources": [e.source], "targets": [tgt]})
    return {
        "id": "root",
        "layoutOptions": {
            "elk.algorithm": "layered",
            "elk.direction": _DIRECTION.get(rankdir, "DOWN"),
            "elk.hierarchyHandling": "INCLUDE_CHILDREN",
            "elk.randomSeed": "1",  # determinism for golden tests
            "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
            "elk.layered.crossingMinimization.greedySwitchHierarchical.type": "TWO_SIDED",
            "elk.spacing.nodeNode": "28",
            "elk.layered.spacing.nodeNodeBetweenLayers": "36",
            "elk.spacing.edgeNode": "16",
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

    # ELK fixes placement; we draw our own smooth cubic to the exact token. With external
    # parents now placed on the correct side (sigma above y, not opposite it), the simple
    # source->token cubic stays clean — and reads better than ELK's orthogonal routing.
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
        res.edge_paths[f"{e.source}|{e.target}"] = common.simple_edge_path(sb, tb, anchor)

    return res
