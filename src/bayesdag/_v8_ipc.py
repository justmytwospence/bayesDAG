"""Bounded, serial JSON over a private socketpair (trusted parent and renderer child)."""

from __future__ import annotations

import json
import socket
import struct
import subprocess
import sys
import time
import weakref

_HEADER = struct.Struct("!I")
_MAX_FRAME = 64 * 1024 * 1024
_STARTUP_MS = 30_000
_RENDER_MS = 30_000
_NAMES = {"mathjax": "MathJax", "elk": "ELK"}


class WorkerError(RuntimeError):
    """A renderer child could not complete a request."""


def _remaining(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError
    return left


def encode(value) -> bytes:
    data = json.dumps(value, separators=(",", ":")).encode("ascii")
    if len(data) > _MAX_FRAME:
        raise ValueError("renderer message exceeds the 64 MiB frame limit")
    return data


def send(conn: socket.socket, data: bytes, deadline: float | None = None) -> None:
    for part in (_HEADER.pack(len(data)), data):
        if deadline is not None:
            conn.settimeout(_remaining(deadline))
        conn.sendall(part)


def _read(conn: socket.socket, size: int, deadline: float | None) -> bytearray:
    data = bytearray(size)
    pending = memoryview(data)
    while pending:
        if deadline is not None:
            conn.settimeout(_remaining(deadline))
        count = conn.recv_into(pending)
        if not count:
            raise EOFError("closed its connection")
        pending = pending[count:]
    return data


def receive(conn: socket.socket, deadline: float | None = None):
    (size,) = _HEADER.unpack(_read(conn, _HEADER.size, deadline))
    if not 0 < size <= _MAX_FRAME:
        raise ValueError("invalid renderer frame size")
    return json.loads(_read(conn, size, deadline))


def _stop(process: subprocess.Popen, conn: socket.socket, *, kill: bool = False) -> None:
    conn.close()  # the child's request loop exits on EOF, then closes its isolate
    if not kill:
        try:
            process.wait(timeout=0.5)
            return
        except subprocess.TimeoutExpired:
            pass
    process.kill()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass  # even a process stuck in the OS must not block the kernel's shutdown


class _Link:
    def __init__(self, component: str, paths: tuple[str, ...]) -> None:
        conn, peer = socket.socketpair()
        try:
            with peer:
                fd = peer.fileno()
                process = subprocess.Popen(
                    [sys.executable, "-m", "bayesdag._v8_worker", str(fd), component, *paths],
                    pass_fds=(fd,),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    start_new_session=True,
                )
        except BaseException:
            conn.close()
            raise
        self.process = process
        self.conn = conn
        self._cleanup = weakref.finalize(self, _stop, process, conn)

    def close(self, *, kill: bool = False) -> None:
        if self._cleanup.detach() is not None:
            _stop(self.process, self.conn, kill=kill)


def _response(conn: socket.socket, deadline: float, result_type: type) -> tuple[bool, object]:
    reply = receive(conn, deadline)
    _remaining(deadline)
    if isinstance(reply, list) and len(reply) == 2:
        status, value = reply
        if status == "ok" and isinstance(value, result_type):
            return True, value
        if status == "error" and isinstance(value, str):
            return False, value
    raise ValueError("invalid renderer response")


class Child:
    """A lazy child for one component/bundle set; owned by the shared parent executor."""

    def __init__(self, component: str, paths: tuple[str, ...] = ()) -> None:
        self._component = component
        self._paths = paths
        self._name = _NAMES[component]
        self._link: _Link | None = None
        self._startup_error: WorkerError | None = None

    @property
    def pid(self) -> int | None:
        return None if self._link is None else self._link.process.pid

    def close(self) -> None:
        """Shut down the child, if started. Call on the parent executor."""
        if self._link is not None:
            link, self._link = self._link, None
            link.close()

    def warm(self) -> None:
        self._exchange(None, _STARTUP_MS, type(None))

    def render(self, tex: str, display: bool) -> str:
        return self._exchange([tex, display], _RENDER_MS, str)

    def layout(self, graph: dict, timeout_ms: int) -> dict:
        return self._exchange([graph, timeout_ms], timeout_ms, dict)

    def _exchange(self, args: list | None, timeout_ms: int, result_type: type):
        if self._startup_error is not None:
            raise self._startup_error
        deadline = time.monotonic() + timeout_ms / 1000
        data = None if args is None else encode(args)
        link = self._link
        try:
            if link is None:
                _remaining(deadline)
                link = self._link = _Link(self._component, self._paths)
                ok, value = _response(link.conn, deadline, type(None))
                if not ok:
                    self._startup_error = WorkerError(
                        f"{self._name} worker failed to start: {value}"
                    )
                    self.close()
                    raise self._startup_error
            if data is None:
                if link.process.poll() is not None:
                    raise EOFError("closed its connection")
                return None
            send(link.conn, data, deadline)
            ok, value = _response(link.conn, deadline, result_type)
        except (OSError, EOFError, ValueError, RecursionError) as exc:
            self._link = None
            if isinstance(exc, TimeoutError):
                reason = f"timed out after {timeout_ms} ms"
            else:
                status = None if link is None else link.process.poll()
                reason = str(exc)
                if status is not None:
                    reason = f"exited with status {status} ({reason})"
            if link is not None:
                link.close(kill=True)
            # Never replay a request after losing the connection. A later call can start fresh.
            raise WorkerError(f"{self._name} worker {reason}") from exc
        if not ok:
            raise WorkerError(f"{self._name} worker: {value}")
        return value
