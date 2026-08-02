"""One process-wide lock around mini-racer isolate CONSTRUCTION.

bayesdag runs two independent V8 isolates — MathJax (``mathsvg``) and ELK (``layout.elk_backend``)
— each pinned to its own dedicated thread. Each module serializes its *own* construction, but
nothing serialized the two against each other, and building both at the same moment aborts the
whole interpreter inside V8's allocator:

    [FATAL:address_pool_manager.cc(67)] Check failed: !pool->IsInitialized().

That is reachable on the ordinary path, not just under contrived threading: ``view()`` warms the
ELK context on its worker thread precisely so it overlaps the work that goes on to render labels
through MathJax. Whether the two cold starts collide is a timing accident, which is the worst
kind of crash to ship — it reproduces on someone else's machine and not on yours.

Only the ``MiniRacer()`` call needs the lock (it is the V8 platform/allocator init). Evaluating
the bundles afterwards is per-isolate and stays parallel, so the warm-up overlap keeps its value;
the cost is that two cold isolates start one after the other rather than simultaneously.
"""

from __future__ import annotations

import threading

_CONSTRUCTION_LOCK = threading.Lock()


def new_isolate():
    """Construct a mini-racer isolate under the shared lock. Import errors propagate to the
    caller, which knows how to phrase them for its own missing-dependency story."""
    from py_mini_racer import MiniRacer

    with _CONSTRUCTION_LOCK:
        return MiniRacer()
