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

import re
import threading
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

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


def _parse_transform(s: str | None) -> tuple:
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


_XLINK = "{http://www.w3.org/1999/xlink}href"
_NUM = re.compile(r"-?\d+\.?\d*(?:[eE]-?\d+)?")
# very rough per-glyph em box used when a <use>'s referenced path can't be measured
_FALLBACK_EM_W = 0.5
_FALLBACK_EM_H = 0.7
_EM_UNITS = 1000.0


def _defs_path_extents(svg: str) -> dict[str, tuple[float, float, float, float]]:
    """Crude bounding box (minx,miny,maxx,maxy) for each ``<path id=.. d=..>`` in <defs>.

    Treats the ``d`` numbers as alternating x,y — approximate (includes Bezier control
    points / descenders) but sufficient to aim an edge at a token's top-center."""
    out: dict[str, tuple[float, float, float, float]] = {}
    for m in re.finditer(r'<path\b[^>]*\bid="([^"]+)"[^>]*\bd="([^"]+)"', svg):
        nums = [float(x) for x in _NUM.findall(m.group(2))]
        xs, ys = nums[0::2], nums[1::2]
        if xs and ys:
            out[m.group(1)] = (min(xs), min(ys), max(xs), max(ys))
    return out


def token_bboxes(svg: str) -> dict[str, tuple[float, float, float, float]]:
    """Map ``token_id -> (fx, fy, fw, fh)``: the token's fractional bounding box within the
    SVG (fx,fy = top-left; 0..1). Used to aim a port-edge at the token's top-center with a
    standoff so the arrowhead doesn't cover the glyph."""
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
    extents = _defs_path_extents(svg)
    boxes: dict[str, list[float]] = {}
    origins: dict[str, tuple[float, float]] = {}

    def grow(tok: str, x: float, y: float) -> None:
        b = boxes.get(tok)
        if b is None:
            boxes[tok] = [x, y, x, y]
        else:
            b[0], b[1], b[2], b[3] = min(b[0], x), min(b[1], y), max(b[2], x), max(b[3], y)

    def walk(el: ET.Element, M: tuple, tok: str | None) -> None:
        M2 = _mat_mul(M, _parse_transform(el.get("transform")))
        nid = el.get("id")
        if nid and nid.startswith("tok-"):
            tok = nid[len("tok-") :]
            origins[tok] = (M2[4], M2[5])
        if tok is not None and _local(el.tag) == "use":
            href = (el.get(_XLINK) or el.get("href") or "").lstrip("#")
            ux, uy = float(el.get("x", 0) or 0), float(el.get("y", 0) or 0)
            Mu = _mat_mul(M2, (1.0, 0.0, 0.0, 1.0, ux, uy))
            ex = extents.get(href)
            if ex:
                x0, y0, x1, y1 = ex
                for cx, cy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                    grow(tok, Mu[0] * cx + Mu[2] * cy + Mu[4], Mu[1] * cx + Mu[3] * cy + Mu[5])
            else:
                grow(tok, Mu[4], Mu[5])
        for child in el:
            walk(child, M2, tok)

    walk(root, _IDENTITY, None)

    result: dict[str, tuple[float, float, float, float]] = {}
    seen = set(boxes) | set(origins)
    for tok in seen:
        if tok in boxes and boxes[tok][2] > boxes[tok][0] and boxes[tok][3] > boxes[tok][1]:
            x0, y0, x1, y1 = boxes[tok]
        else:  # degenerate -> small em box centered on the token origin
            ox, oy = origins.get(tok, (min_x, min_y))
            w, h = _FALLBACK_EM_W * _EM_UNITS, _FALLBACK_EM_H * _EM_UNITS
            x0, y0, x1, y1 = ox, oy - h, ox + w, oy
        result[tok] = (
            (x0 - min_x) / vb_w,
            (y0 - min_y) / vb_h,
            (x1 - x0) / vb_w,
            (y1 - y0) / vb_h,
        )
    return result


class MathRenderer:
    """Lazy in-process MathJax renderer. Construct once and reuse (V8 init is the cost).

    Like the ELK engine, ALL V8 work runs on one dedicated thread: mini-racer binds its event
    loop to the thread that creates the context, so an isolate built on a thread that owns a
    live asyncio loop (a marimo cell) can assert or deadlock. Labels are rendered eagerly from
    the caller's thread — which IS the main thread — so the pinning has to live here, not only
    in the layout backend that happens to call us."""

    _CACHE_MAX = 4096  # LRU bound — long-lived kernels render many models; memory stays flat

    def __init__(self, bundle_path: Path | None = None) -> None:
        self._bundle_path = bundle_path or _bundle_path()
        self._ctx = None
        self._ctx_error: Exception | None = None
        self._executor = None
        self._lock = threading.Lock()
        # (display, tex) -> (svg, token bboxes): bboxes are cached WITH the SVG so a warm
        # re-layout never re-parses the same label (token_bboxes was ~40% of warm layout time)
        self._cache: OrderedDict[tuple[bool, str], tuple[str, dict]] = OrderedDict()

    def _worker(self):
        """The single thread every V8 call is marshalled onto (see the class docstring)."""
        with self._lock:
            if self._executor is None:
                import concurrent.futures

                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="bayesdag-mathjax"
                )
            return self._executor

    @property
    def available(self) -> bool:
        try:
            import py_mini_racer  # noqa: F401
        except Exception:
            return False
        return self._bundle_path.exists()

    def _context(self):  # must run on the worker thread (see _worker)
        if self._ctx_error is not None:
            # a failed V8/bundle build is permanent for this renderer — re-raise the cached
            # error instead of paying the multi-MB bundle eval again for every label
            raise self._ctx_error
        if self._ctx is None:
            try:
                from py_mini_racer import MiniRacer
            except Exception as exc:  # pragma: no cover - exercised only without the extra
                raise RuntimeError(
                    "bayesdag math rendering needs mini-racer, which is a core dependency — "
                    "reinstall bayesdag (pip install --force-reinstall bayesdag). "
                    "A matplotlib.mathtext fallback (without token anchors) is not implemented."
                ) from exc
            if not self._bundle_path.exists():
                raise RuntimeError(
                    f"MathJax bundle missing at {self._bundle_path}. Build it with "
                    "`npm install && npm run build` (produces static/mathjax.bundle.js)."
                )
            try:
                ctx = MiniRacer()
                ctx.eval(_PROCESS_SHIM)
                ctx.eval(self._bundle_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self._ctx_error = exc
                raise
            self._ctx = ctx
        return self._ctx

    def render(self, tex: str, display: bool = True) -> str:
        """Return the SVG markup for ``tex`` (cached by content)."""
        return self.render_with_bboxes(tex, display)[0]

    def _tex2svg(self, tex: str, display: bool) -> str:  # runs on the worker thread
        return self._context().call("tex2svg", tex, display)

    def render_with_bboxes(self, tex: str, display: bool = True) -> tuple[str, dict]:
        """Return ``(svg, fractional token bboxes)`` — both cached together, LRU-bounded."""
        key = (bool(display), tex)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
        # marshal onto the dedicated thread (the context is created there on first use)
        svg = self._worker().submit(self._tex2svg, tex, bool(display)).result()
        bboxes = token_bboxes(svg)
        with self._lock:
            self._cache[key] = (svg, bboxes)
            if len(self._cache) > self._CACHE_MAX:
                self._cache.popitem(last=False)
        return svg, bboxes


_RENDERER: MathRenderer | None = None
_RENDERER_LOCK = threading.Lock()


def get_renderer() -> MathRenderer:
    """Process-wide singleton (so the V8 context + cache are reused)."""
    global _RENDERER
    with _RENDERER_LOCK:
        if _RENDERER is None:
            _RENDERER = MathRenderer()
        return _RENDERER


def render(tex: str, display: bool = True) -> str:
    return get_renderer().render(tex, display)
