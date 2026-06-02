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
    w, h = default_size

    def elk_node(n) -> dict[str, Any]:
        node: dict[str, Any] = {"id": n.id, "width": w, "height": h}
        ports = [{"id": f"{n.id}.{p.token_id}"} for p in n.params]
        if ports:
            node["ports"] = ports
        node["labels"] = [{"text": n.label_tex or n.id}]
        return node

    member_of: dict[str, str] = {}
    for p in ir.plates:
        for m in p.members:
            member_of[m] = p.id

    plate_children: dict[str, list[dict]] = {p.id: [] for p in ir.plates}
    top: list[dict] = []
    for n in ir.nodes:
        en = elk_node(n)
        pid = member_of.get(n.id)
        if pid is not None:
            plate_children[pid].append(en)
        else:
            top.append(en)
    for p in ir.plates:
        top.append({"id": p.id, "labels": [{"text": p.label}], "children": plate_children[p.id]})

    edges = []
    for i, e in enumerate(ir.edges):
        tgt = f"{e.target}.{e.target_token_id}" if e.target_token_id else e.target
        edges.append({"id": f"e{i}", "sources": [e.source], "targets": [tgt]})

    return {"id": "root", "children": top, "edges": edges}
