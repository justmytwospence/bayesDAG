"""Model structures ("blocks") over the canonical graph — PPL-neutral.

A block is presentation metadata: a recognized structure (``provenance="verified"``, found by
an adapter or by :func:`centered` here) or one the user declared (:class:`BlockSpec`,
``provenance="annotated"``). Blocks never add variables or dependencies. The canonical IR stays
authoritative; :func:`overview` derives a DISPLAY graph in which contracting blocks become one
compound node each — a pure function of the IR, never serialized as a model.

Functions here take and return :class:`~bayesdag.ir.ModelIR` without mutating their input
(except :func:`resolve`, which the adapter calls on the IR it is building).
"""

from __future__ import annotations

import copy
import dataclasses
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field

from .ir import BlockIR, EdgeIR, ModelIR, NodeIR, PlateIR

# Which recognized structure wins a node both would tag, most specific first.
_PRIORITY = {
    "annotated": 0,
    "shrinkage/1": 1,
    "noncentered/1": 2,
    # "a noisy reading of this latent" says more about x_true than "partial pooling" does
    "measured/1": 3,
    "centered/1": 4,
    "regression/1": 5,
}
REFS_CENTERED = ["Gelman & Hill (2007)"]

# families a "centered hierarchy" can be read off (univariate, one value per element)
_MULTIVARIATE = frozenset(
    {
        "MultivariateNormal",
        "MvNormal",
        "MvStudentT",
        "MultivariateStudentT",
        "Dirichlet",
        "Multinomial",
        "DirichletMultinomial",
        "Wishart",
        "LKJCorr",
        "_lkjcholeskycov",
        "MatrixNormal",
        "KroneckerNormal",
        "CAR",
        "ICAR",
        "RandomWalk",
        "AR",
        "GARCH11",
        "EulerMaruyama",
        "Mixture",
        "StickBreakingWeights",
    }
)
_LOC_SLOTS = frozenset({"loc", "mu", "mean", "alpha_loc"})
_SCALE_SLOTS = frozenset({"scale", "sigma", "tau", "sd", "beta", "lam"})


@dataclass(frozen=True)
class BlockSpec:
    """A structure the user declares — for what bayesdag cannot prove from the graph: a
    measurement model written as Potentials, a two-stage workflow feeding fixed posterior
    summaries, a custom construction. Rendered with an "annotated" marker, never as verified.

    ``members`` are canonical node ids the block owns (contracted into one box in the
    overview when ``collapse``); ``roles`` maps role names to node ids (external shared nodes
    allowed); ``workflow`` is ``"joint"``, ``"fixed_summary"`` or None."""

    kind: str
    title: str
    members: Sequence[str]
    roles: Mapping[str, Sequence[str]] = field(default_factory=dict)
    notes: Sequence[str] = ()
    workflow: str | None = None
    id: str | None = None
    collapse: bool = True


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-").lower() or "block"


def _coarser(a, b) -> bool:
    return set(a) < set(b) or (not a and bool(b))


def centered(ir: ModelIR) -> list[BlockIR]:
    """Centered hierarchies, read off the IR alone (so any PPL's IR gets them): a plated,
    univariate latent with a latent or deterministic parent over strictly coarser dims — a
    group-level quantity drawn around population-level ones.

    Proof obligations: the parent really is shared across the child's elements (its dims are a
    strict subset) and it sets the child's LOCATION or SCALE. A same-plate parent, a data parent,
    a shared truncation bound, or an observed child is not a hierarchy."""
    by_id = {n.id: n for n in ir.nodes}
    out = []
    for n in ir.nodes:
        if n.role != "latent" or not n.dims or not n.params or None in n.dims:
            continue
        if not n.dist or n.dist in _MULTIVARIATE:
            continue
        loc, scale = [], []
        for p in n.params:
            for par in p.parents:
                pn = by_id.get(par)
                if pn is None or pn.role not in ("latent", "deterministic"):
                    continue
                if None in pn.dims or not _coarser(pn.dims, n.dims):
                    continue
                bucket = loc if p.name in _LOC_SLOTS else scale if p.name in _SCALE_SLOTS else None
                if bucket is not None and par not in bucket:
                    bucket.append(par)
        if not (loc or scale):
            continue
        roles = {"effect": [n.id]}
        if loc:
            roles["loc"] = loc
        if scale:
            roles["scale"] = scale
        out.append(
            BlockIR(
                id=f"block:hierarchy:{n.id}",
                kind="hierarchy",
                title="partial pooling · centered",
                variant="centered",
                anchor=n.id,
                members=[],
                roles=roles,
                rule="centered/1",
                notes=[
                    f"{n.id} is drawn around population-level parents shared by all its "
                    "elements (a centered parameterization)"
                ],
                references=list(REFS_CENTERED),
            )
        )
    return out


def _priority(b: BlockIR) -> tuple:
    key = "annotated" if b.provenance == "annotated" else (b.rule or "")
    return (_PRIORITY.get(key, 9), b.anchor or "", b.id)


def resolve(ir: ModelIR, blocks: list[BlockIR]) -> None:
    """Install ``blocks`` on ``ir`` (in place): most specific first; a block whose anchor is
    already claimed merges into the claiming block's notes instead of competing; a verified
    block whose members an annotation owns is dropped (the user's declaration wins). Sets one
    tag per anchor."""
    kept: list[BlockIR] = []
    claimed: dict[str, BlockIR] = {}
    owned: set[str] = set()
    for b in sorted(blocks, key=_priority):
        if b.provenance != "annotated" and set(b.members) & owned:
            continue
        if b.anchor and b.anchor in claimed:
            host = claimed[b.anchor]
            extra = f"also: {b.title}"
            if extra not in host.notes:
                host.notes.append(extra)
            continue
        if b.anchor:
            claimed[b.anchor] = b
        owned |= set(b.members)
        kept.append(b)
    ir.blocks = kept
    for n in ir.nodes:
        n.tag = None
    by_id = {n.id: n for n in ir.nodes}
    for b in kept:
        if b.anchor and b.anchor in by_id:
            by_id[b.anchor].tag = b.title


def strip(ir: ModelIR) -> ModelIR:
    """A copy with no blocks and no tags (``blocks=None``)."""
    out = copy.deepcopy(ir)
    out.blocks = []
    for n in out.nodes:
        n.tag = None
    return out


def detect_ir(ir: ModelIR) -> ModelIR:
    """A copy with the IR-level structures added (centered hierarchies) for an IR that did not
    come through an adapter which detects them — e.g. one built by another PPL."""
    out = copy.deepcopy(ir)
    have = {b.id for b in out.blocks}
    resolve(out, list(out.blocks) + [b for b in centered(out) if b.id not in have])
    return out


def apply(ir: ModelIR, specs: Sequence[BlockSpec]) -> ModelIR:
    """A copy of ``ir`` with the user's :class:`BlockSpec` declarations installed.

    Raises ``ValueError`` for an unknown node id (listing the available ones) or for two
    declarations that own the same member."""
    if not specs:
        return ir
    ids = {n.id for n in ir.nodes}
    seen: dict[str, str] = {}
    new: list[BlockIR] = []
    for spec in specs:
        members = list(spec.members)
        refs = members + [x for xs in spec.roles.values() for x in xs]
        unknown = sorted({r for r in refs if r not in ids})
        if unknown:
            raise ValueError(
                f"bayesdag.BlockSpec {spec.title!r}: unknown variable(s) {unknown}; "
                f"available: {sorted(ids)}"
            )
        if not members:
            raise ValueError(f"bayesdag.BlockSpec {spec.title!r}: needs at least one member")
        for m in members:
            if m in seen:
                raise ValueError(
                    f"bayesdag.BlockSpec: {m!r} is claimed by both {seen[m]!r} and {spec.title!r}"
                )
            seen[m] = spec.title
        new.append(
            BlockIR(
                id=f"block:{spec.id or _slug(spec.title)}",
                kind=spec.kind,
                title=spec.title,
                provenance="annotated",
                members=members,
                roles={k: list(v) for k, v in spec.roles.items()},
                workflow=spec.workflow,
                notes=[str(x) for x in spec.notes],
                collapsible=bool(spec.collapse),
            )
        )
    out = copy.deepcopy(ir)
    resolve(out, new + list(out.blocks))
    return out


# --------------------------------------------------------------------------- overview projection
def _reach(adj: dict[str, set[str]], start: set[str]) -> set[str]:
    seen, stack = set(start), list(start)
    while stack:
        for w in adj.get(stack.pop(), ()):
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return seen


def _convex(members: set[str], fwd, bwd) -> bool:
    """No outside node lies on a path between two members (so contracting keeps a DAG)."""
    down = _reach(fwd, members) - members
    up = _reach(bwd, members) - members
    return not (down & up)


def _acyclic(nodes: set[str], edges: set[tuple[str, str]]) -> bool:
    indeg = dict.fromkeys(nodes, 0)
    fwd: dict[str, list[str]] = {n: [] for n in nodes}
    for a, b in edges:
        fwd[a].append(b)
        indeg[b] += 1
    queue = [n for n, d in indeg.items() if d == 0]
    count = 0
    while queue:
        n = queue.pop()
        count += 1
        for w in fwd[n]:
            indeg[w] -= 1
            if indeg[w] == 0:
                queue.append(w)
    return count == len(nodes)


MAX_BLOCK_LINES = 6


def contracting(ir: ModelIR, expand: Collection[str] = ()) -> list[BlockIR]:
    """The blocks the overview contracts: collapsible, owning members, not expanded, and
    path-convex (contracting them cannot create a cycle)."""
    fwd: dict[str, set[str]] = {}
    bwd: dict[str, set[str]] = {}
    for e in ir.edges:
        fwd.setdefault(e.source, set()).add(e.target)
        bwd.setdefault(e.target, set()).add(e.source)
    out = []
    for b in ir.blocks:
        if not b.members or not b.collapsible or b.id in expand:
            continue
        if not _convex(set(b.members), fwd, bwd):
            continue
        out.append(b)
    return out


def _topo(members: list[str], ir: ModelIR) -> list[str]:
    ms = set(members)
    deps = {m: {e.source for e in ir.edges if e.target == m and e.source in ms} for m in members}
    order, done = [], set()
    while len(order) < len(members):
        ready = [m for m in members if m not in done and deps[m] <= done]
        if not ready:  # defensive: a cycle cannot happen in a DAG
            ready = [m for m in members if m not in done]
        m = ready[0]
        order.append(m)
        done.add(m)
    return order


_TOK = re.compile(r"\\cssId\{tok-([^}]*)\}")


def _block_label(b: BlockIR, members: list[str], by_id: dict[str, NodeIR]) -> str:
    lines = []
    if b.implied:
        lines.append(b.implied)
    for m in members:
        tex = by_id[m].label_tex or m
        # every member line gets its own token namespace (`theta::mu`), so repeated leaves in
        # different lines never collide and an aggregated edge can land on the right line
        lines.append(_TOK.sub(lambda mt, m=m: r"\cssId{tok-" + m + "::" + mt.group(1) + "}", tex))
    if len(lines) > MAX_BLOCK_LINES:
        extra = len(lines) - (MAX_BLOCK_LINES - 1)
        lines = [*lines[: MAX_BLOCK_LINES - 1], rf"+{extra}\ \text{{more}}"]
    return r"\begin{array}{l}" + r" \\ ".join(lines) + r"\end{array}"


def _implied_tokens(b: BlockIR) -> set[str]:
    return set(_TOK.findall(b.implied or ""))


def _worst_diag(members: list[str], by_id: dict[str, NodeIR]) -> dict | None:
    best = None
    for m in members:
        d = by_id[m].diag
        if d and d.get("flags") and (best is None or len(d["flags"]) > len(best[1]["flags"])):
            best = (m, d)
    if best is None:
        return None
    return {**best[1], "member": best[0]}


def overview(ir: ModelIR, *, expand: Collection[str] = ()) -> ModelIR:
    """The display projection: each contracting block becomes ONE compound node carrying the
    block's implied line and its members' equations; every edge crossing the block boundary is
    kept (``EdgeIR.via`` lists the canonical edges it stands for); internal edges go. The input
    is not mutated. With nothing to contract, returns ``ir`` itself."""
    blocks = contracting(ir, expand)
    if not blocks:
        return ir
    by_id = {n.id: n for n in ir.nodes}
    ids = set(by_id)
    # contract one block at a time; a block whose contraction would make the quotient cyclic
    # stays expanded (and says so)
    rep: dict[str, str] = {}
    chosen: list[tuple[BlockIR, str]] = []
    for b in blocks:
        bid = b.id if b.id not in ids else b.id + "#2"
        trial = dict(rep)
        trial.update(dict.fromkeys(b.members, bid))
        qn = {trial.get(n, n) for n in ids}
        qe = {(trial.get(e.source, e.source), trial.get(e.target, e.target)) for e in ir.edges}
        qe = {(a, c) for a, c in qe if a != c}
        if _acyclic(qn, qe):
            rep, chosen = trial, [*chosen, (b, bid)]
    if not chosen:
        return ir
    out = copy.deepcopy(ir)
    by_id = {n.id: n for n in out.nodes}
    member_to_block = {m: bid for b, bid in chosen for m in b.members}
    new_nodes: list[NodeIR] = []
    placed: set[str] = set()
    token_map: dict[tuple[str, str], str] = {}
    for n in out.nodes:
        bid = member_to_block.get(n.id)
        if bid is None:
            new_nodes.append(n)
            continue
        if bid in placed:
            continue
        placed.add(bid)
        b, _ = next((bb, x) for bb, x in chosen if x == bid)
        members = _topo(list(b.members), out)
        anchor = by_id.get(b.anchor) if b.anchor else None
        src = anchor or by_id[members[-1]]
        dims = list(anchor.dims) if anchor else _common_dims(members, by_id)
        node = NodeIR(
            id=bid,
            role="block",
            dims=dims,
            coords=(anchor.coords if anchor else None),
            label_tex=_block_label(b, members, by_id),
            glyph=copy.deepcopy(src.glyph) if anchor else None,
            glyph_data=copy.deepcopy(src.glyph_data) if anchor else None,
            glyph_note=None,
            tag=b.title,
            block=b.id,
            diag=_worst_diag(members, by_id),
        )
        new_nodes.append(node)
        implied = _implied_tokens(b)
        for m in members:
            for e in out.edges:
                if e.target == m and e.target_token_id:
                    if b.implied and e.source in implied:
                        token_map[(e.source, m)] = e.source
                    else:
                        token_map[(e.source, m)] = f"{m}::{e.target_token_id}"
    # edges: re-point across the boundary, drop internal ones, merge parallels
    merged: dict[tuple[str, str], EdgeIR] = {}
    for e in out.edges:
        a = member_to_block.get(e.source, e.source)
        c = member_to_block.get(e.target, e.target)
        if a == c:
            continue
        if e.target in member_to_block:
            tok = token_map.get((e.source, e.target))
            toks = [tok] if tok else []
        else:
            tok, toks = e.target_token_id, list(e.target_token_ids)
        key = (a, c)
        if key in merged:
            m = merged[key]
            m.via.append([e.source, e.target])
            for t in toks:
                if t not in m.target_token_ids:
                    m.target_token_ids.append(t)
            m.target_token_id = m.target_token_ids[0] if len(m.target_token_ids) == 1 else None
        else:
            merged[key] = EdgeIR(
                source=a,
                target=c,
                target_token_id=tok,
                target_token_ids=toks,
                via=[[e.source, e.target]],
            )
    # plates: members leave; the block joins its anchor's plate (or the one all members share)
    plates: list[PlateIR] = []
    for p in out.plates:
        mem = [m for m in p.members if m not in member_to_block]
        for b, bid in chosen:
            home = _home_plate(b, out)
            if home == p.id and bid not in mem:
                mem.append(bid)
        if mem:
            plates.append(dataclasses.replace(p, members=mem))
    keep = {p.id for p in plates}
    plates = [
        p if p.parent is None or p.parent in keep else dataclasses.replace(p, parent=None)
        for p in plates
    ]
    shown = {n.id for n in new_nodes}
    contracted_ids = {b.id for b, _ in chosen}

    def remap(xs):
        out_ = []
        for x in xs:
            y = member_to_block.get(x, x)
            if y not in out_:
                out_.append(y)
        return out_

    # blocks that stay expanded keep their references, re-pointed at the compound nodes
    blocks_out = [
        b
        if b.id in contracted_ids
        else dataclasses.replace(
            b,
            members=remap(b.members),
            anchor=member_to_block.get(b.anchor, b.anchor) if b.anchor else None,
            roles={k: remap(v) for k, v in b.roles.items()},
        )
        for b in out.blocks
    ]
    return dataclasses.replace(
        out,
        nodes=new_nodes,
        edges=list(merged.values()),
        plates=plates,
        blocks=blocks_out,
        aux_views=[a for a in out.aux_views if all(v in shown for v in a.vars)],
    )


def _common_dims(members: list[str], by_id: dict[str, NodeIR]) -> list:
    dims = [tuple(by_id[m].dims) for m in members]
    return list(dims[0]) if dims and all(d == dims[0] for d in dims) else []


def _home_plate(b: BlockIR, ir: ModelIR) -> str | None:
    plate_of = {m: p.id for p in ir.plates for m in p.members}
    if b.anchor:
        return plate_of.get(b.anchor)
    homes = {plate_of.get(m) for m in b.members}
    return homes.pop() if len(homes) == 1 else None
