"""Static output: SVG (primary) + PNG/PDF (via cairosvg) + TikZ (planned).

The SVG is produced by the shared emitter (:mod:`bayesdag.render_svg`); this module only
handles persistence and raster/vector conversion, so static and interactive share byte-
identical SVG content.
"""

from __future__ import annotations

from pathlib import Path


def save(svg: str, path) -> Path:
    """Write ``svg`` to ``path``; format inferred from the extension (.svg/.png/.pdf)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".svg":
        path.write_text(svg)
        return path
    if ext in (".png", ".pdf"):
        try:
            import cairosvg
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "PNG/PDF export needs the 'export' extra and a system cairo library: "
                "pip install 'bayesdag[export]' (and e.g. `brew install cairo`)."
            ) from exc
        data = svg.encode("utf-8")
        if ext == ".png":
            cairosvg.svg2png(bytestring=data, write_to=str(path))
        else:
            cairosvg.svg2pdf(bytestring=data, write_to=str(path))
        return path
    if ext == ".tikz":  # pragma: no cover
        raise NotImplementedError("TikZ export is planned for M2 (publication path).")
    raise ValueError(f"unsupported output format: {ext!r}")
