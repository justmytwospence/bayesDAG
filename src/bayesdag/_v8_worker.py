"""Private Python entry point for one renderer. No executor or recursive IPC here."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

from ._v8_ipc import encode, receive, send


def _start(component: str, paths: list[str]):
    if component == "mathjax" and len(paths) in (0, 1):
        from .mathsvg import MathRenderer

        renderer = MathRenderer(*(Path(p) for p in paths), _local=True)
        renderer._context()
        return renderer._tex2svg, (str, bool)
    if component == "elk" and len(paths) in (0, 2):
        from .layout.elk_backend import ElkEngine

        engine = ElkEngine(*(Path(p) for p in paths), _local=True)
        engine._local_context()
        return engine._layout_blocking, (dict, int)
    raise ValueError("invalid renderer component or bundle paths")


def _error(exc: Exception) -> bytes:
    # Send diagnostics, not pickled exceptions or potentially huge JS stack traces.
    detail = str(exc)[:4096].partition("\n")[0].strip()
    return encode(["error", detail or type(exc).__name__])


def _serve(conn: socket.socket, component: str, paths: list[str]) -> None:
    try:
        # The child main thread owns the context for its entire life, with no asyncio loop.
        run, types = _start(component, paths)
    except Exception as exc:
        send(conn, _error(exc))
        return
    send(conn, encode(["ok", None]))
    while True:
        args = receive(conn)
        try:
            if not isinstance(args, list) or [type(arg) for arg in args] != list(types):
                raise ValueError("invalid renderer arguments")
            reply = encode(["ok", run(*args)])
        except Exception as exc:
            reply = _error(exc)
        send(conn, reply)


def main() -> None:
    fd, component, *paths = sys.argv[1:]
    with socket.socket(fileno=int(fd)) as conn:
        conn.set_inheritable(False)
        try:
            _serve(conn, component, paths)
        except (EOFError, BrokenPipeError, ConnectionResetError):
            pass  # parent closed the connection (including on kernel exit)


if __name__ == "__main__":
    main()
