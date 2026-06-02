"""TeX -> SVG rendering + per-token anchor extraction.

This is the load-bearing module the M0 spike validated:
  * ``mathjax-full`` runs **in-process** in ``py_mini_racer`` (bare V8, no DOM, no runtime
    Node) once the bundle is built with ``PACKAGE_VERSION`` defined (kills MathJax's
    ``eval("require")`` node-detection branch). See ``js/mathjax_entry.js``.
  * Each parameter token is wrapped ``\\cssId{tok-<id>}{...}`` by the label engine; MathJax
    tags the corresponding ``<g data-mml-node id="tok-<id>">``. We recover the token's
    position by composing the ancestor ``transform`` chain and expressing it as a fraction
    of the SVG's viewBox -> node-local fractional anchors that the layout post-pass turns
    into absolute port-edge endpoints.

Rendering once here and embedding the identical SVG in both renderers is what makes the
math byte-identical across the static and interactive outputs (parity principle #2).

Backend ladder: in-process ``py_mini_racer`` (preferred; ships the bundle, no Node) ->
[TODO: bundled Node subprocess] -> [TODO: matplotlib.mathtext, which loses token anchors].
Only the first is implemented in M0; the others raise a clear, actionable error.
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

_BUNDLE_NAME = "mathjax.bundle.js"
_PROCESS_SHIM = "globalThis.process = globalThis.process || {env:{}};"


def _bundle_path() -> Path:
    return Path(__file__).parent / "static" / _BUNDLE_NAME


# --------------------------------------------------------------------------- affine 2x3
# An affine transform is [a, b, c, d, e, f] mapping (x,y) -> (a*x + c*y + e, b*x + d*y + f).
_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _mat_mul(A: tuple, B: tuple) -> tuple:
    a1, b1, c1, d1, e1, f1 = A
    a2, b2, c2, d2, e2, f2 = B
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _parse_transform(s: Optional[str]) -> tuple:
    """Parse an SVG ``transform`` attribute into a composed affine (translate/scale/matrix)."""
    M = _IDENTITY
    if not s:
        return M
    for kind, args in re.findall(r"(\w+)\(([^)]*)\)", s):
        v = [float(x) for x in re.split(r"[ ,]+", args.strip()) if x]
        if kind == "translate":
            T = (1.0, 0.0, 0.0, 1.0, v[0], v[1] if len(v) > 1 else 0.0)
        elif kind == "scale":
            T = (v[0], 0.0, 0.0, (v[1] if len(v) > 1 else v[0]), 0.0, 0.0)
        elif kind == "matrix" and len(v) == 6:
            T = tuple(v)
        else:
            T = _IDENTITY
        M = _mat_mul(M, T)
    return M


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def token_anchors(svg: str) -> dict[str, tuple[float, float]]:
    """Map ``token_id -> (fx, fy)`` node-local fractional anchors (0..1 within the SVG box).

    Token ids are taken from ``<g id="tok-...">`` groups (the ``tok-`` prefix is stripped to
    match ``ParamIR.token_id``). The layout post-pass scales these by the node's box to get
    absolute port-edge endpoints; unresolved params simply get no anchor (center fallback).
    """
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return {}
    vb = root.get("viewBox")
    if not vb:
        return {}
    min_x, min_y, vb_w, vb_h = (float(x) for x in vb.split())
    if vb_w == 0 or vb_h == 0:
        return {}

    anchors: dict[str, tuple[float, float]] = {}

    def walk(el: ET.Element, M: tuple) -> None:
        M2 = _mat_mul(M, _parse_transform(el.get("transform")))
        nid = el.get("id")
        if nid and nid.startswith("tok-"):
            e, f = M2[4], M2[5]  # image of the token-local origin (0,0)
            fx = (e - min_x) / vb_w
            fy = (f - min_y) / vb_h
            anchors[nid[len("tok-"):]] = (fx, fy)
        for child in el:
            walk(child, M2)

    walk(root, _IDENTITY)
    return anchors


class MathRenderer:
    """Lazy in-process MathJax renderer. Construct once and reuse (V8 init is the cost)."""

    def __init__(self, bundle_path: Optional[Path] = None) -> None:
        self._bundle_path = bundle_path or _bundle_path()
        self._ctx = None
        self._cache: dict[str, str] = {}

    @property
    def available(self) -> bool:
        try:
            import py_mini_racer  # noqa: F401
        except Exception:
            return False
        return self._bundle_path.exists()

    def _context(self):
        if self._ctx is None:
            try:
                from py_mini_racer import MiniRacer
            except Exception as exc:  # pragma: no cover - exercised only without the extra
                raise RuntimeError(
                    "bayesdag math rendering needs the 'math' extra: "
                    "pip install 'bayesdag[math]' (provides mini-racer). "
                    "A matplotlib.mathtext fallback (without token anchors) is not yet implemented."
                ) from exc
            if not self._bundle_path.exists():
                raise RuntimeError(
                    f"MathJax bundle missing at {self._bundle_path}. Build it with "
                    "`npm install && npm run build` (produces static/mathjax.bundle.js)."
                )
            ctx = MiniRacer()
            ctx.eval(_PROCESS_SHIM)
            ctx.eval(self._bundle_path.read_text())
            self._ctx = ctx
        return self._ctx

    def render(self, tex: str, display: bool = True) -> str:
        """Return the SVG markup for ``tex`` (cached by content)."""
        key = hashlib.sha256(f"{int(display)}:{tex}".encode()).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        svg = self._context().call("tex2svg", tex, bool(display))
        self._cache[key] = svg
        return svg

    def render_with_anchors(self, tex: str, display: bool = True) -> tuple[str, dict[str, tuple[float, float]]]:
        svg = self.render(tex, display)
        return svg, token_anchors(svg)


_RENDERER: Optional[MathRenderer] = None


def get_renderer() -> MathRenderer:
    """Process-wide singleton (so the V8 context + cache are reused)."""
    global _RENDERER
    if _RENDERER is None:
        _RENDERER = MathRenderer()
    return _RENDERER


def render(tex: str, display: bool = True) -> str:
    return get_renderer().render(tex, display)


def render_with_anchors(tex: str, display: bool = True) -> tuple[str, dict[str, tuple[float, float]]]:
    return get_renderer().render_with_anchors(tex, display)
