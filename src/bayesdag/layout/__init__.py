"""Layout backends. ``layout(ir)`` is the single Python pass that produces the
``LayoutResult`` both renderers consume (the parity guarantee).

Backend selection: **ELK** (hierarchy-aware, lays plates out correctly) when ``mini-racer``
is available, else **Graphviz dot** (the fallback). Override with ``BAYESDAG_LAYOUT=elk|dot``.
Both implement the same ``layout(ir, *, rankdir) -> LayoutResult`` signature.
"""

from __future__ import annotations

import os

from ..ir import LayoutResult, ModelIR


def _select(name: str):
    if name == "dot":
        from .graphviz_backend import layout as _dot

        return _dot
    if name == "elk":
        from .elk_backend import layout as _elk

        return _elk
    return None


def layout(ir: ModelIR, *, rankdir: str = "TB") -> LayoutResult:
    forced = os.environ.get("BAYESDAG_LAYOUT", "").strip().lower()
    if forced in ("elk", "dot"):
        return _select(forced)(ir, rankdir=rankdir)

    # default: ELK when usable, dot otherwise
    try:
        from .elk_backend import available as _elk_available

        if _elk_available():
            from .elk_backend import layout as _elk

            return _elk(ir, rankdir=rankdir)
    except Exception:
        pass
    from .graphviz_backend import layout as _dot

    return _dot(ir, rankdir=rankdir)


__all__ = ["layout"]
