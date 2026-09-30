"""Context-aware legend: decide which entries to show from the IR (only encodings that
actually appear). The SVG swatches are drawn in ``render_svg`` (which owns the styles)."""

from __future__ import annotations

from dataclasses import dataclass

from .ir import ModelIR

_ROLE_LABELS = {
    "latent": "latent variable",
    "observed": "observed variable",
    "deterministic": "deterministic (computed)",
    "data": "data",
    "potential": "potential (factor)",
    "factor": "factor",
    "block": "block — a collapsed structure (several variables)",
}
_SOURCE_LABELS = {
    "prior_analytic": "prior density (parameters known)",
    "prior_family_only": "prior shape (depends on parents)",
    "posterior_kde": "posterior (from idata)",
    "posterior_bars": "posterior (from idata, per class)",
    "observed_hist": "observed data (histogram)",
    "deterministic_fn": "transfer function (canonical shape)",
    "conditional_standardized": "exact shape — location/scale set by parents or per element",
    "implied_family": "implied distribution (recognized structure)",
    "posterior_predictive": "posterior predictive check (from idata)",
    "prior_predictive": "prior predictive check (from idata)",
    "posterior_quantiles": "posterior intervals per element (from idata)",
}


@dataclass
class LegendItem:
    swatch: str  # "role:<r>" | "glyph:<source>" | "symbol:~" | "symbol:=" | "plate" | "elision"
    label: str


def build(ir: ModelIR) -> list[LegendItem]:
    items: list[LegendItem] = []

    for role in ("latent", "observed", "deterministic", "data", "potential", "factor", "block"):
        if any(n.role == role for n in ir.nodes):
            items.append(LegendItem(f"role:{role}", _ROLE_LABELS[role]))

    for src in (
        "prior_analytic",
        "posterior_kde",
        "posterior_bars",
        "observed_hist",
        "prior_family_only",
        "conditional_standardized",
        "implied_family",
        "posterior_predictive",
        "prior_predictive",
        "posterior_quantiles",
        "deterministic_fn",
    ):
        if any(n.glyph and n.glyph.source == src for n in ir.nodes):
            items.append(LegendItem(f"glyph:{src}", _SOURCE_LABELS[src]))
    if any(n.glyph and n.glyph.kind == "hist_overlay" for n in ir.nodes):
        items.append(LegendItem("glyph:best_fit", "best-fit family (legacy IR)"))

    if any(n.role in ("latent", "observed") for n in ir.nodes):
        items.append(LegendItem("symbol:~", "“~”  distributed as"))
    if any(n.role == "deterministic" for n in ir.nodes):
        items.append(LegendItem("symbol:=", "“=”  computed from"))
    if ir.plates:
        items.append(LegendItem("plate", "plate — indexed by its dimension"))
    if any(n.tag for n in ir.nodes):
        items.append(LegendItem("tag", "tag — a structure read off the graph"))
    if any(b.provenance == "annotated" for b in ir.blocks):
        items.append(LegendItem("annotated", "annotated — declared by the user, not verified"))
    if any(
        ("\\cdots" in (n.label_tex or "")) or ("\\ldots" in (n.label_tex or "")) or n.elision_reason
        for n in ir.nodes
    ):
        items.append(LegendItem("elision", "[⋯]  more values / elided"))
    if any((n.diag or {}).get("flags") for n in ir.nodes):
        items.append(LegendItem("diag", "inspect — sampling diagnostic"))

    return items
