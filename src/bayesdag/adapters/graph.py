"""Renderer-neutral graph-format projections of a ``ModelIR``.

* ``to_networkx`` — a plain ``DiGraph`` of variable-name nodes + dependency edges, for
  graph algorithms (``ancestors``/``descendants``/``nx.moral_graph`` -> Markov blanket).
* ``to_elk`` — ELK JSON (the one mainstream layout format with first-class **ports** +
  **nested compounds**): plates become nested ``children``, params become ``ports``, edges
  reference ``node:port``. Used as an optional alternate layout backend / wire format.

Both are projections (lossy on purpose); the typed ``ModelIR`` stays the source of truth.
"""

from __future__ import annotations

from typing import Any

from ..ir import ModelIR


def to_networkx(ir: ModelIR):
    import networkx as nx

    g = nx.DiGraph()
    for n in ir.nodes:
        g.add_node(n.id, role=n.role, observed=n.observed, dist=n.dist)
    for e in ir.edges:
        g.add_edge(e.source, e.target, port=e.target_token_id)
    return g


def markov_blanket(ir: ModelIR, node_id: str) -> set[str]:
    """Parents + children + co-parents of ``node_id`` (via the moral graph)."""
    import networkx as nx

    g = to_networkx(ir)
    if node_id not in g:
        return set()
    moral = nx.moral_graph(g)
    return set(moral.neighbors(node_id))


def to_elk(ir: ModelIR, default_size: tuple[float, float] = (120.0, 60.0)) -> dict[str, Any]:
    """ELK JSON for INTEROP — handing the graph to an external ELK consumer.

    This is deliberately not the graph bayesdag lays out with: ``layout.elk_backend._build_graph``
    builds that one, with measured node sizes, fixed-position token ports and its own port-id
    scheme. Keep the two separate rather than merging them — this one is a stable exchange format,
    that one tracks whatever the layout engine currently needs.

    ELK ids are one global namespace, so every id is prefixed by its kind (``node:``, ``plate:``,
    ``port:``, ``edge:``) — a variable may legally be called ``plate_school`` or ``a.b``. Ports are
    exactly the tokens edges land on (including a deterministic's leaf tokens), and plates nest by
    ``PlateIR.parent``. The original names travel in ``labels``."""
    import json

    w, h = default_size

    def nid(x: str) -> str:
        return "node:" + x

    def gid(x: str) -> str:
        return "plate:" + x

    def pid(node: str, token: str) -> str:
        return "port:" + json.dumps([node, token], ensure_ascii=False)

    ports: dict[str, list[str]] = {}
    for e in ir.edges:
        if e.target_token_id:
            toks = ports.setdefault(e.target, [])
            if e.target_token_id not in toks:
                toks.append(e.target_token_id)

    def elk_node(n) -> dict[str, Any]:
        node: dict[str, Any] = {"id": nid(n.id), "width": w, "height": h}
        if ports.get(n.id):
            node["ports"] = [{"id": pid(n.id, t), "labels": [{"text": t}]} for t in ports[n.id]]
        node["labels"] = [{"text": n.id}, {"text": n.label_tex or n.id}]
        return node

    member_of: dict[str, str] = {}
    for p in ir.plates:
        for m in p.members:
            member_of[m] = p.id

    plate_ids = {p.id for p in ir.plates}
    plate_json: dict[str, dict] = {
        p.id: {"id": gid(p.id), "labels": [{"text": p.label}], "children": []} for p in ir.plates
    }
    top: list[dict] = []
    for n in ir.nodes:
        en = elk_node(n)
        owner = member_of.get(n.id)
        (plate_json[owner]["children"] if owner in plate_json else top).append(en)
    for p in ir.plates:  # nest by parent (a missing/cyclic parent falls back to the root)
        parent = p.parent if p.parent in plate_ids and p.parent != p.id else None
        seen, q = {p.id}, parent
        while q is not None and q not in seen:  # guard malformed parent cycles
            seen.add(q)
            q = next((x.parent for x in ir.plates if x.id == q), None)
        if q is not None:
            parent = None
        (plate_json[parent]["children"] if parent else top).append(plate_json[p.id])

    edges = []
    for i, e in enumerate(ir.edges):
        tgt = pid(e.target, e.target_token_id) if e.target_token_id else nid(e.target)
        edges.append({"id": f"edge:{i}", "sources": [nid(e.source)], "targets": [tgt]})

    return {"id": "root", "children": top, "edges": edges}
