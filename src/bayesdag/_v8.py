"""One loop-free executor for renderer work submitted by sync and notebook callers.

Linux uses separate Python children for MathJax and ELK; elsewhere the packaged bundles
share one lazy, in-process isolate. Custom bundles always keep private globals.
"""

from __future__ import annotations

import atexit
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from importlib.util import find_spec
from pathlib import Path

from ._v8_ipc import Child

_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bayesdag-v8")
_LIVE_ISOLATES = weakref.WeakSet()
_DEFAULT_CHILDREN = {component: Child(component) for component in ("mathjax", "elk")}


def isolated() -> bool:
    return sys.platform.startswith("linux")


def child(component: str, *paths: Path) -> Child:
    if paths:
        return Child(component, tuple(str(path.absolute()) for path in paths))
    return _DEFAULT_CHILDREN[component]


def available() -> bool:
    if isolated():
        # Inspect the installation without loading the native runtime in the parent.
        return find_spec("py_mini_racer") is not None
    try:
        import py_mini_racer  # noqa: F401
    except Exception:
        return False
    return True


def worker():
    return _WORKER


class Runtime:
    """Bundle state accessed exclusively by the owner thread. No V8 work at import."""

    def __init__(self):
        self._ctx = None
        self._ready = set()
        self._errors = {}

    def context(self, component, install):
        from py_mini_racer import MiniRacer

        if component in self._errors:
            raise self._errors[component]
        if self._ctx is None:
            self._ctx = MiniRacer()
            if hasattr(self._ctx, "close"):
                _LIVE_ISOLATES.add(self._ctx)
        if component not in self._ready:
            try:
                install(self._ctx)
            except Exception as exc:
                # Never run a partially installed bundle's initialization twice.
                self._errors[component] = exc
                raise
            self._ready.add(component)
        return self._ctx


shared = Runtime()


@atexit.register
def _close_isolates() -> None:
    # Executor workers have joined, but mini-racer's daemon event loops still run.
    # Close now, before late finalization can hang joining an already stopped loop.
    with ExitStack() as cleanup:
        for isolate in _LIVE_ISOLATES:
            cleanup.callback(isolate.close)
