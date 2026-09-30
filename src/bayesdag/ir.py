"""bayesdag intermediate representation — the single source of truth.

IMPORT-LIGHT INVARIANT (see AGENTS.md): this module imports **stdlib only**. No pymc,
xarray, numpy, graphviz, or renderer imports. That keeps the IR (and its JSON Schema
validation / format adapters) usable without a PPL or a rendering stack installed, and
makes the IR a neutral interchange ("ArviZ for model structure") rather than a silo.

Two-substrate design:
  * topology + presentation (this module)  -> JSON  (+ a published JSON Schema)
  * data-bearing overlays                  -> referenced in the user's ArviZ InferenceData
    via `OverlayRef` (never duplicated here).

The dataclasses below are JSON-serializable. ``to_dict``/``from_dict`` round-trip exactly
(a tested invariant), and ``meta`` stamps provenance + schema version into every artifact.
"""

from __future__ import annotations

import dataclasses
import types
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

SCHEMA_VERSION = "1.1"
SCHEMA_MAJOR = 1

# Structural identity of a node (fixed). NOTE: ``observed`` below is a *separate*,
# mutable flag (current conditioning state) — the workflow toggle re-shades the observed
# set to move between prior / prior-predictive / posterior views without changing roles.
Role = Literal["latent", "observed", "deterministic", "data", "potential", "factor", "block"]
# ``block`` exists ONLY in a presentation projection (``bayesdag.blocks.overview``): a recognized or
# annotated structure contracted into one box. It is never a model variable and never an idata key.

# How a glyph's shape was obtained (orthogonal to the glyph ``kind``).
GlyphSource = Literal[
    "prior_analytic",  # family density with the node's real numeric params plugged in
    "prior_family_only",  # params depend on parents -> prior-predictive or canonical shape
    "posterior_kde",  # KDE of idata.posterior[name]
    "posterior_bars",  # posterior of a DISCRETE variable -> per-class proportions, not a KDE
    "observed_hist",  # observed data -> histogram (auto-binned), not a KDE
    "deterministic_fn",  # canonical, parameter-free shape of a Deterministic's transfer function
    # exact shape of the family with location/scale left to parents (never the marginal prior)
    "conditional_standardized",
    "implied_family",  # exact family implied by a recognized reparameterization (a block)
    "posterior_predictive",  # observed data + draws from idata.posterior_predictive
    "prior_predictive",  # observed data + draws from idata.prior_predictive
    "posterior_quantiles",  # per-element posterior quantiles (a forest of intervals)
]


# --------------------------------------------------------------------------- geometry
@dataclass
class Box:
    """Axis-aligned box in SVG px (top-left origin)."""

    x: float
    y: float
    w: float
    h: float


# --------------------------------------------------------------------------- labels
@dataclass
class TokenIR:
    """A node in the LaTeX token-tree. ``token_id`` is the stable anchor a port-edge
    terminates at (matched to the ``data-mml-node`` / ``\\cssId`` in the rendered SVG)."""

    token_id: str
    tex: str
    children: list[TokenIR] = field(default_factory=list)


@dataclass
class ParamIR:
    """A distribution parameter slot. ``name`` is the *op-level* name (e.g. ``loc``,
    ``scale``), recovered from the op signature — not PyMC's ``dist()`` kwargs."""

    index: int
    name: str
    token_id: str
    parents: list[str] = field(default_factory=list)  # source node ids feeding this slot
    value_tex: str | None = None  # LaTeX for the slot content (constant / parent symbol / expr)


# --------------------------------------------------------------------------- glyphs
@dataclass
class GlyphSpec:
    """How to draw a node's distribution. ``kind`` is a key in the glyph registry — see
    ``bayesdag.glyph.registered_kinds()`` for the kinds that actually render; the density/shape
    is always the primary mark. ``interval``/``point`` are OPTIONAL annotations a kind may
    ignore (the registry core is glyph-agnostic, so non-univariate kinds are first-class).
    They are the declared M2 annotation contract and nothing sets them yet."""

    kind: str = "density"
    source: GlyphSource = "prior_analytic"
    interval: list[float] | None = None  # credible-interval probabilities, e.g. [0.5, 0.94]
    point: str | None = None  # "median" | "mean" | "mode" | None


@dataclass
class OverlayRef:
    """A pointer into the user's ArviZ InferenceData (we reference, never duplicate)."""

    idata_group: str  # "posterior" | "prior" | "observed_data" | "posterior_predictive" | ...
    var_name: str  # MUST equal the idata variable name
    var_dims: list[str] = field(default_factory=list)
    sample_dims: list[str] = field(default_factory=lambda: ["chain", "draw"])


# --------------------------------------------------------------------------- graph
@dataclass
class NodeIR:
    id: str  # = the constrained idata variable name (universal join key)
    role: Role
    observed: bool = False  # current conditioning state (mutable; drives shading)
    dist: str | None = None  # distribution name ("Normal"); None for deterministic/factor/data
    params: list[ParamIR] = field(default_factory=list)
    dims: list[str | None] = field(default_factory=list)
    coords: dict[str, list[Any]] | None = None
    label_tex: str = ""
    label_tree: TokenIR | None = None
    transform: str | None = None  # e.g. "log", "logodds", "simplex"
    idata_unconstrained_key: str | None = None  # e.g. "tau_log__" in unconstrained_posterior
    glyph: GlyphSpec | None = None
    glyph_data: dict[str, Any] | None = (
        None  # precomputed shape (xs/ys or edges/counts), shipped in-band
    )
    overlays: list[OverlayRef] = field(default_factory=list)
    representable: bool = True
    elision_reason: str | None = None
    # sampling diagnostics joined by id (see bayesdag.diagnostics); None when there is no idata.
    # Hedged flags only — this never carries a verdict about the model.
    diag: dict[str, Any] | None = None
    # why a node that is otherwise representable carries no curve ("shape depends on alpha (a
    # parent)", "elements differ"). A caption, not a warning badge: nothing is wrong with the model.
    glyph_note: str | None = None
    # the recognized-structure tag drawn on this node ("partial pooling · centered"), if any
    tag: str | None = None
    # the BlockIR id this node stands for (display projections only; role == "block")
    block: str | None = None
    # "index" for an integer pm.Data used only to gather (it can be folded into its consumers)
    usage: str | None = None
    # filled by render/layout stages:
    label_svg: str | None = None
    box: Box | None = None
    port_anchors: dict[str, Box] = field(default_factory=dict)  # token_id -> bbox in node-local px


@dataclass
class EdgeIR:
    source: str
    target: str
    # the ONE token the arrow lands on; None => boundary anchor (no slot, or several slots)
    target_token_id: str | None = None
    # every slot/token this parent feeds, in slot order (a parent may feed loc AND scale)
    target_token_ids: list[str] = field(default_factory=list)
    # display projections only: the canonical (source, target) edges this display edge stands for
    via: list[list[str]] = field(default_factory=list)


@dataclass
class PlateIR:
    id: str
    label: str  # e.g. "school (8)"
    members: list[str] = field(default_factory=list)
    parent: str | None = None  # enclosing plate id (nested plates)
    box: Box | None = None
    dims: list[str | None] = field(default_factory=list)  # dim names (None = unnamed axis)
    lengths: list[int] = field(default_factory=list)
    # "named" when grouped by declared dims, "shape" when inferred from an unnamed shape. Either
    # way a plate means "indexed by" — never a claim of exchangeability or statistical nesting.
    source: str | None = None


@dataclass
class BlockIR:
    """A compound model structure: recognized from the graph (``provenance="verified"``) or
    declared by the user (``"annotated"``). Presentation metadata over the canonical graph — it
    never adds variables or dependencies."""

    id: str
    kind: str  # hierarchy | reparam | shrinkage | regression | measured | measurement | custom
    title: str  # the tag text
    provenance: Literal["verified", "annotated"] = "verified"
    variant: str | None = None  # centered | noncentered | regularized | ...
    anchor: str | None = None  # node that carries the tag when not contracted
    members: list[str] = field(default_factory=list)  # owned nodes ([] = tag-only)
    roles: dict[str, list[str]] = field(default_factory=dict)
    implied: str | None = None  # TeX of the implied family line
    rule: str | None = None  # versioned rule id, e.g. "noncentered/1"
    workflow: str | None = None  # "joint" | "fixed_summary"
    notes: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    collapsible: bool = True
    data: dict[str, Any] | None = None


@dataclass
class AuxViewIR:
    """A linked auxiliary panel (posterior geometry, parcoord, energy, ...). Stats are
    precomputed in Python; the JS layer only re-styles on selection (M2+)."""

    kind: str  # "joint" | "parcoord" | "energy" | "marginal"
    vars: list[str] = field(default_factory=list)
    edge: list[str] | None = None  # [source, target] when the panel is edge-driven
    axis_space: str = "constrained"  # "constrained" | "unconstrained"
    data_ref: dict[str, Any] | None = None  # precomputed bins/density/divergence masks


@dataclass
class Meta:
    schema_version: str = SCHEMA_VERSION
    source_ppl: str | None = None  # "pymc" | "numpyro" | "stan" | ...
    creation_library: str = "bayesdag"
    creation_library_version: str | None = None
    creation_library_language: str = "python"
    created_at: str | None = None  # ISO-8601, stamped at build time
    model_name: str | None = None

    @classmethod
    def stamp(cls, source_ppl: str | None = None, model_name: str | None = None) -> Meta:
        import datetime
        import importlib.metadata

        try:
            ver = importlib.metadata.version("bayesdag")
        except importlib.metadata.PackageNotFoundError:
            ver = None
        return cls(
            source_ppl=source_ppl,
            creation_library_version=ver,
            created_at=datetime.datetime.now(datetime.UTC).isoformat(),
            model_name=model_name,
        )


@dataclass
class LayoutResult:
    """The geometry both renderers consume verbatim (parity guarantee). All coords in
    SVG px (top-left origin) after the single coordinate transform."""

    canvas: Box | None = None
    node_boxes: dict[str, Box] = field(default_factory=dict)
    node_token_anchors: dict[str, dict[str, Box]] = field(default_factory=dict)
    # (source, target) -> [[x,y],...]. A tuple key, not "src|tgt": `pm.Normal("a|b", ...)` is a
    # legal variable name, and a string join would let two different edges collide silently.
    edge_paths: dict[tuple[str, str], list[list[float]]] = field(default_factory=dict)
    plate_boxes: dict[str, Box] = field(default_factory=dict)


@dataclass
class ModelIR:
    nodes: list[NodeIR] = field(default_factory=list)
    edges: list[EdgeIR] = field(default_factory=list)
    plates: list[PlateIR] = field(default_factory=list)
    aux_views: list[AuxViewIR] = field(default_factory=list)
    blocks: list[BlockIR] = field(default_factory=list)
    meta: Meta = field(default_factory=Meta)

    # ---- convenience accessors -------------------------------------------------
    def node(self, node_id: str) -> NodeIR | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    # ---- (de)serialization -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def block(self, block_id: str) -> BlockIR | None:
        return next((b for b in self.blocks if b.id == block_id), None)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelIR:
        """Read any 1.x IR dict. A 1.0 artifact is upgraded: its singular edge token fills the
        new token list, and a ``prior_family_only`` schematic (which 1.0 drew as one generic bell
        whatever the family) is dropped with a note rather than shown as if it were exact."""
        version = str((data.get("meta") or {}).get("schema_version") or SCHEMA_VERSION)
        try:
            major = int(version.split(".")[0])
        except ValueError:
            major = -1
        if major != SCHEMA_MAJOR:
            raise ValueError(
                f"bayesdag IR schema {version!r} is not supported (this bayesdag reads "
                f"{SCHEMA_MAJOR}.x)"
            )
        out = _from_dict(cls, data)
        if version != SCHEMA_VERSION:
            for e in out.edges:
                if e.target_token_id and not e.target_token_ids:
                    e.target_token_ids = [e.target_token_id]
            for n in out.nodes:
                if (
                    n.glyph is not None
                    and n.glyph.source == "prior_family_only"
                    and n.glyph.kind == "schematic"
                    and not (n.glyph_data or {}).get("exact")
                    and n.elision_reason is None
                ):
                    n.glyph, n.glyph_data = None, None
                    n.glyph_note = "legacy schematic (unverified shape) omitted"
            out.meta.schema_version = SCHEMA_VERSION
        return out

    def to_json(self, **kw: Any) -> str:
        import json

        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_json(cls, text: str) -> ModelIR:
        import json

        return cls.from_dict(json.loads(text))


# --------------------------------------------------------------------------- structuring
def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Reconstruct a (possibly nested) dataclass from a plain dict, honoring type hints."""
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = _structure(data[f.name], hints.get(f.name, Any))
    return cls(**kwargs)


def _structure(value: Any, hint: Any) -> Any:
    if value is None:
        return None
    origin = get_origin(hint)
    if origin is Union or origin is types.UnionType:  # Optional[X] / Union[...]
        args = [a for a in get_args(hint) if a is not type(None)]
        return _structure(value, args[0]) if args else value
    if origin in (list,):
        (elem,) = get_args(hint) or (Any,)
        return [_structure(v, elem) for v in value]
    if origin in (dict,):
        targs = get_args(hint)
        vt = targs[1] if len(targs) == 2 else Any
        return {k: _structure(v, vt) for k, v in value.items()}
    if origin in (tuple,):
        targs = get_args(hint)
        if len(targs) == 2 and targs[1] is Ellipsis:
            return tuple(_structure(v, targs[0]) for v in value)
        return tuple(_structure(v, a) for v, a in zip(value, targs, strict=False))
    if is_dataclass(hint) and isinstance(value, dict):
        return _from_dict(hint, value)
    return value
