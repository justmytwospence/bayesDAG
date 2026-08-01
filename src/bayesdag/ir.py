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
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Literal, Optional, Union, get_args, get_origin, get_type_hints

SCHEMA_VERSION = "1.0"

# Structural identity of a node (fixed). NOTE: ``observed`` below is a *separate*,
# mutable flag (current conditioning state) — the workflow toggle re-shades the observed
# set to move between prior / prior-predictive / posterior views without changing roles.
Role = Literal["latent", "observed", "deterministic", "data", "potential", "factor"]

# How a glyph's shape was obtained (orthogonal to the glyph ``kind``).
GlyphSource = Literal[
    "prior_analytic",  # family density with the node's real numeric params plugged in
    "prior_family_only",  # params depend on parents -> prior-predictive or canonical shape
    "posterior_kde",  # KDE of idata.posterior[name]
    "posterior_bars",  # posterior of a DISCRETE variable -> per-class proportions, not a KDE
    "observed_hist",  # observed data -> histogram (auto-binned), not a KDE
    "deterministic_fn",  # canonical, parameter-free shape of a Deterministic's transfer function
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
    children: list["TokenIR"] = field(default_factory=list)


@dataclass
class ParamIR:
    """A distribution parameter slot. ``name`` is the *op-level* name (e.g. ``loc``,
    ``scale``), recovered from the op signature — not PyMC's ``dist()`` kwargs."""

    index: int
    name: str
    token_id: str
    parents: list[str] = field(default_factory=list)  # source node ids feeding this slot
    value_tex: Optional[str] = None  # LaTeX for the slot content (constant / parent symbol / expr)


# --------------------------------------------------------------------------- glyphs
@dataclass
class GlyphSpec:
    """How to draw a node's distribution. ``kind`` is the registry key; the density/shape
    is always the primary mark. ``interval``/``point`` are OPTIONAL annotations a kind may
    ignore (the registry core is glyph-agnostic, so non-univariate kinds are first-class)."""

    kind: str = (
        "density"  # density|cdf|ccdf|histogram|gradient|dotplot|band|heatmap|ternary|rose|...
    )
    source: GlyphSource = "prior_analytic"
    interval: Optional[list[float]] = None  # credible-interval probabilities, e.g. [0.5, 0.94]
    point: Optional[str] = None  # "median" | "mean" | "mode" | None
    layout: Optional[str] = None  # "ridgeline" for vector-valued params, else None
    transform: Optional[dict[str, Any]] = None  # e.g. {"animate": "hops", "frames": 20}


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
    dist: Optional[str] = None  # distribution name ("Normal"); None for deterministic/factor/data
    params: list[ParamIR] = field(default_factory=list)
    dims: list[Optional[str]] = field(default_factory=list)
    coords: Optional[dict[str, list[Any]]] = None
    label_tex: str = ""
    label_tree: Optional[TokenIR] = None
    transform: Optional[str] = None  # e.g. "log", "logodds", "simplex"
    idata_unconstrained_key: Optional[str] = None  # e.g. "tau_log__" in unconstrained_posterior
    glyph: Optional[GlyphSpec] = None
    glyph_data: Optional[dict[str, Any]] = (
        None  # precomputed shape (xs/ys or edges/counts), shipped in-band
    )
    overlays: list[OverlayRef] = field(default_factory=list)
    representable: bool = True
    elision_reason: Optional[str] = None
    docstring: Optional[str] = None
    # filled by render/layout stages:
    label_svg: Optional[str] = None
    box: Optional[Box] = None
    port_anchors: dict[str, Box] = field(default_factory=dict)  # token_id -> bbox in node-local px


@dataclass
class EdgeIR:
    source: str
    target: str
    target_token_id: Optional[str] = None  # which param token; None => center-anchor fallback


@dataclass
class PlateIR:
    id: str
    label: str  # e.g. "school (8)"
    members: list[str] = field(default_factory=list)
    parent: Optional[str] = None  # enclosing plate id (nested plates)
    box: Optional[Box] = None


@dataclass
class AuxViewIR:
    """A linked auxiliary panel (posterior geometry, parcoord, energy, ...). Stats are
    precomputed in Python; the JS layer only re-styles on selection (M2+)."""

    kind: str  # "joint" | "parcoord" | "energy" | "marginal"
    vars: list[str] = field(default_factory=list)
    edge: Optional[list[str]] = None  # [source, target] when the panel is edge-driven
    axis_space: str = "constrained"  # "constrained" | "unconstrained"
    data_ref: Optional[dict[str, Any]] = None  # precomputed bins/density/divergence masks


@dataclass
class Meta:
    schema_version: str = SCHEMA_VERSION
    source_ppl: Optional[str] = None  # "pymc" | "numpyro" | "stan" | ...
    creation_library: str = "bayesdag"
    creation_library_version: Optional[str] = None
    creation_library_language: str = "python"
    created_at: Optional[str] = None  # ISO-8601, stamped at build time
    model_name: Optional[str] = None

    @classmethod
    def stamp(cls, source_ppl: Optional[str] = None, model_name: Optional[str] = None) -> "Meta":
        import datetime
        import importlib.metadata

        try:
            ver = importlib.metadata.version("bayesdag")
        except importlib.metadata.PackageNotFoundError:
            ver = None
        return cls(
            source_ppl=source_ppl,
            creation_library_version=ver,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            model_name=model_name,
        )


@dataclass
class LayoutResult:
    """The geometry both renderers consume verbatim (parity guarantee). All coords in
    SVG px (top-left origin) after the single coordinate transform."""

    canvas: Optional[Box] = None
    node_boxes: dict[str, Box] = field(default_factory=dict)
    node_token_anchors: dict[str, dict[str, Box]] = field(default_factory=dict)
    edge_paths: dict[str, list[list[float]]] = field(
        default_factory=dict
    )  # "src|tgt" -> [[x,y],...]
    plate_boxes: dict[str, Box] = field(default_factory=dict)


@dataclass
class ModelIR:
    nodes: list[NodeIR] = field(default_factory=list)
    edges: list[EdgeIR] = field(default_factory=list)
    plates: list[PlateIR] = field(default_factory=list)
    aux_views: list[AuxViewIR] = field(default_factory=list)
    meta: Meta = field(default_factory=Meta)

    # ---- convenience accessors -------------------------------------------------
    def node(self, node_id: str) -> Optional[NodeIR]:
        return next((n for n in self.nodes if n.id == node_id), None)

    # ---- (de)serialization -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelIR":
        return _from_dict(cls, data)

    def to_json(self, **kw: Any) -> str:
        import json

        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_json(cls, text: str) -> "ModelIR":
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
    if origin is Union:  # Optional[X] / Union[...]
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
        return tuple(_structure(v, a) for v, a in zip(value, targs))
    if is_dataclass(hint) and isinstance(value, dict):
        return _from_dict(hint, value)
    return value
