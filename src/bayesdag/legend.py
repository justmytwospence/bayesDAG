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
}
_SOURCE_LABELS = {
    "prior_analytic": "prior density (parameters known)",
    "prior_family_only": "prior shape (depends on parents)",
    "posterior_kde": "posterior (from idata)",
    "posterior_bars": "posterior (from idata, per class)",
    "observed_hist": "observed data (histogram)",
    "deterministic_fn": "transfer function (canonical shape)",
}


@dataclass
class LegendItem:
    swatch: str  # "role:<r>" | "glyph:<source>" | "symbol:~" | "symbol:=" | "plate" | "elision"
    label: str


def build(ir: ModelIR) -> list[LegendItem]:
    items: list[LegendItem] = []

    for role in ("latent", "observed", "deterministic", "data", "potential", "factor"):
        if any(n.role == role for n in ir.nodes):
            items.append(LegendItem(f"role:{role}", _ROLE_LABELS[role]))

    for src in (
        "prior_analytic",
        "posterior_kde",
        "posterior_bars",
        "observed_hist",
        "prior_family_only",
        "deterministic_fn",
    ):
        if any(n.glyph and n.glyph.source == src for n in ir.nodes):
            items.append(LegendItem(f"glyph:{src}", _SOURCE_LABELS[src]))
    if any(n.glyph and n.glyph.kind == "hist_overlay" for n in ir.nodes):
        items.append(LegendItem("glyph:best_fit", "best-fit family (shape check)"))

    if any(n.role in ("latent", "observed") for n in ir.nodes):
        items.append(LegendItem("symbol:~", "“~”  distributed as"))
    if any(n.role == "deterministic" for n in ir.nodes):
        items.append(LegendItem("symbol:=", "“=”  computed from"))
    if ir.plates:
        items.append(LegendItem("plate", "plate — repeats over its dimension"))
    if any(
        ("\\cdots" in (n.label_tex or "")) or ("\\ldots" in (n.label_tex or "")) or n.elision_reason
        for n in ir.nodes
    ):
        items.append(LegendItem("elision", "[⋯]  more values / elided"))

    return items
