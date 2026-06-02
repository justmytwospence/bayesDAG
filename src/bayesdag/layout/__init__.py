"""Layout backends. ``layout(ir)`` is the single Python pass that produces the
``LayoutResult`` both renderers consume (the parity guarantee).

**ELK is the layout engine** (hierarchy-aware; lays plates out correctly). There is NO
automatic runtime fallback: a silent downgrade to a different engine would ship a worse
layout while reporting success (it's how a real bug stayed hidden once). If ELK can't run we
**raise** — fix it, or roll back via git. ``dot`` remains only as a *deliberate, explicit*
opt-in / rollback target via ``BAYESDAG_LAYOUT=dot``. Both backends implement the same
``layout(ir, *, rankdir) -> LayoutResult`` signature.
"""

from __future__ import annotations

import os

from ..ir import LayoutResult, ModelIR


def layout(ir: ModelIR, *, rankdir: str = "TB") -> LayoutResult:
    if os.environ.get("BAYESDAG_LAYOUT", "").strip().lower() == "dot":
        from .graphviz_backend import layout as _dot

        return _dot(ir, rankdir=rankdir)

    from .elk_backend import available as _elk_available

    if not _elk_available():
        raise RuntimeError(
            "bayesdag: the ELK layout backend is unavailable (needs `mini-racer` and the "
            "bundled elkjs in src/bayesdag/static/). Install mini-racer and run `npm run build`, "
            "or set BAYESDAG_LAYOUT=dot to use the Graphviz fallback explicitly."
        )
    from .elk_backend import layout as _elk

    return _elk(ir, rankdir=rankdir)  # errors propagate — no silent downgrade


__all__ = ["layout"]
