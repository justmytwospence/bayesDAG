"""TeX->SVG + token-anchor extraction (the M0 spike, as regression tests)."""

import subprocess
import sys
import textwrap

import pytest

from bayesdag import mathsvg

_r = mathsvg.get_renderer()
pytestmark = pytest.mark.skipif(
    not _r.available, reason="needs a built mathjax bundle (npm run build)"
)


def test_render_produces_self_contained_svg():
    svg = mathsvg.render(r"\mathcal{N}(\mu, \sigma)")
    assert "<svg" in svg
    assert "<path" in svg  # fontCache='local' => inline glyph paths (self-contained)
    assert "data-mml-node" in svg


def test_token_bboxes_are_fractional_and_ordered():
    """Token bboxes are what the layout anchors port-edges to: fractional within the label SVG,
    positioned in reading order, and with real extent (not a zero-size point)."""
    tex = r"\mathcal{N}(\cssId{tok-mu}{\mu},\ \cssId{tok-sg}{\sigma})"
    _svg, bboxes = mathsvg.get_renderer().render_with_bboxes(tex)
    assert set(bboxes) == {"mu", "sg"}
    # mu must sit to the left of sigma in N(mu, sigma)
    assert bboxes["mu"][0] < bboxes["sg"][0]
    for fx, fy, fw, fh in bboxes.values():
        assert 0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0
        assert fw > 0.0 and fh > 0.0


@pytest.mark.parametrize("tex", [r"\sigma^2", r"\frac{a}{b}", r"\alpha + \beta x", r"\Sigma"])
def test_renders_varied_tex(tex):
    svg = mathsvg.render(tex)
    assert "<svg" in svg and "data-mml-node" in svg


def test_render_is_cached():
    assert mathsvg.render(r"\alpha") == mathsvg.render(r"\alpha")


def test_broken_bundle_fails_once_not_per_label(tmp_path):
    """A failed installation stays failed for this renderer, even if the bundle is fixed."""
    bundle = tmp_path / "mathjax.bundle.js"
    bundle.write_text("throw new Error('boom');", encoding="utf-8")
    r = mathsvg.MathRenderer(bundle_path=bundle)
    with pytest.raises(Exception, match="boom") as first:
        r.render(r"\alpha")

    # Use new TeX so the second call cannot hit the label cache. Re-reading and evaluating
    # the bundle on that call would now succeed, rather than reporting the original error.
    bundle.write_text("function tex2svg() { return '<svg/>'; }", encoding="utf-8")
    with pytest.raises(type(first.value), match="boom") as second:
        r.render(r"\beta")
    assert str(second.value) == str(first.value)
    if not sys.platform.startswith("linux"):
        assert second.value is first.value


def test_renders_in_a_notebook_kernel_without_deadlocking():
    """Warm MathJax in an async cell, then call it from a synchronous cell.

    A V8 context created on the kernel's event loop hangs when off-loop callers drive it after
    that loop stops. Both the Linux child and the dedicated owner thread on other platforms
    must keep V8 independent of the kernel loop.
    """
    import asyncio
    import threading

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:

        async def build():
            r = mathsvg.MathRenderer()
            r.render(r"\gamma")  # first render builds the V8 context — the binding moment
            return r

        r = loop.run_until_complete(build())  # an async cell warms the process-wide renderer

        out: dict = {}
        done = threading.Event()

        def sync_cell():  # ordinary synchronous notebook-cell code
            try:
                out["svg"] = r.render(r"\cssId{tok-a}{\alpha} + \beta")
            except BaseException as exc:
                out["err"] = f"{type(exc).__name__}: {exc}"
            done.set()

        threading.Thread(target=sync_cell, daemon=True).start()
        assert done.wait(timeout=60), "MathJax render deadlocked under a notebook kernel loop"
        assert "err" not in out, out.get("err")
        assert "<svg" in out["svg"] and "data-mml-node" in out["svg"]
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def _run_renderer_host(code, *args):
    """A fresh interpreter with bounded startup, calls and normal shutdown."""
    try:
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", textwrap.dedent(code), *args],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = (exc.stderr or b"").decode(errors="replace")
        pytest.fail(f"Renderer host hung; stdout={exc.stdout!r}\n{stderr}", pytrace=False)
    assert result.returncode == 0, (
        f"Renderer host failed (returncode {result.returncode}); "
        f"stdout={result.stdout!r}\nstderr={result.stderr}"
    )
    assert result.stdout.strip() == "OK"


@pytest.mark.parametrize("first", ["math", "elk", "concurrent"])
def test_default_engines_start_call_and_shut_down(first):
    """Exercise real geometry and simultaneous callers with cold and warm V8 runtimes.

    A completion marker alone would miss the shutdown deadlock, so let the interpreter exit
    normally with both engines still live. Persistent Linux children isolate V8 from the
    host; other platforms keep a single V8 context on a dedicated owner thread.
    """
    _run_renderer_host(
        r"""
        import concurrent.futures
        import faulthandler
        import math
        import sys
        import threading
        import py_mini_racer

        faulthandler.dump_traceback_later(45)
        linux = sys.platform.startswith('linux')
        real_mini_racer = py_mini_racer.MiniRacer
        constructor_threads = []
        def record_context(*args, **kwargs):
            constructor_threads.append(threading.get_ident())
            assert not linux, 'Linux renderer constructed MiniRacer in the host'
            return real_mini_racer(*args, **kwargs)
        py_mini_racer.MiniRacer = record_context

        from bayesdag import _v8, mathsvg
        from bayesdag.layout.elk_backend import ElkEngine, get_engine

        renderers = [mathsvg.get_renderer(), mathsvg.MathRenderer()]
        engines = [get_engine(), ElkEngine()]

        def render(i, suffix):
            token = f'alpha-{suffix}'
            svg, boxes = renderers[i].render_with_bboxes(
                r'(\cssId{tok-' + token + r'}{\alpha} + \cssId{tok-beta}{\beta})')
            assert '<svg' in svg and '<path' in svg
            assert set(boxes) == {token, 'beta'}
            assert boxes[token][0] < boxes['beta'][0]
            for x, y, w, h in boxes.values():
                assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1

        def layout(i, suffix):
            source, target = f'a{suffix}', f'b{suffix}'
            graph = {
                'id': 'root', 'layoutOptions': {'elk.algorithm': 'layered'},
                'children': [{'id': source, 'width': 30, 'height': 20},
                             {'id': target, 'width': 30, 'height': 20}],
                'edges': [{'id': 'ab', 'sources': [source], 'targets': [target]}],
            }
            result = engines[i].layout_graph(graph)
            assert {child['id'] for child in result['children']} == {source, target}
            assert all(math.isfinite(c[axis]) for c in result['children'] for axis in ('x', 'y'))
            section, = result['edges'][0]['sections']
            assert section['startPoint'] != section['endPoint']
            assert all(math.isfinite(section[p][axis])
                       for p in ('startPoint', 'endPoint') for axis in ('x', 'y'))

        def concurrent_calls(*calls):
            start = threading.Barrier(len(calls))
            def call(fn, i, suffix):
                start.wait()
                fn(i, suffix)
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(calls)) as callers:
                futures = [callers.submit(call, *args) for args in calls]
                for future in futures:
                    future.result()

        if linux:
            children = {'math': _v8.child('mathjax'), 'elk': _v8.child('elk')}
            assert all(child.pid is None for child in children.values())
        first = sys.argv[1]
        if first == 'concurrent':
            concurrent_calls((render, 0, 0), (layout, 0, 0))
        else:
            calls = {'math': render, 'elk': layout}
            second = 'elk' if first == 'math' else 'math'
            calls[first](0, 0)
            if linux:
                assert children[first].pid is not None and children[second].pid is None
            calls[second](0, 0)
        if linux:
            pids = [child.pid for child in children.values()]
            assert None not in pids and len(set(pids)) == 2
        concurrent_calls((render, 0, 1), (render, 1, 2), (layout, 0, 1), (layout, 1, 2))

        if linux:
            assert not constructor_threads, 'expected zero MiniRacer constructors in the host'
            assert [child.pid for child in children.values()] == pids, 'helpers were not reused'
        else:
            workers = [obj._worker() for obj in [*renderers, *engines]]
            assert all(worker is workers[0] for worker in workers)
            assert constructor_threads == [workers[0].submit(threading.get_ident).result()]
        print('OK', flush=True)
    """,
        first,
    )


if sys.platform.startswith("linux"):
    # No other platform has a helper to stop; do not collect a skip under REQUIRE_FULL.
    def test_exited_mathjax_helper_is_contained():
        """Lose one helper between calls without generating a native fault."""
        _run_renderer_host(
            r"""
            import os
            import signal
            import time

            from bayesdag import _v8, mathsvg
            from bayesdag._v8_ipc import WorkerError
            from bayesdag.layout.elk_backend import get_engine

            assert '<svg' in mathsvg.render(r'\alpha')
            engine = get_engine()
            graph = {'id': 'root', 'children': [{'id': 'a', 'width': 20, 'height': 20}]}
            assert engine.layout_graph(graph)['children'][0]['id'] == 'a'
            elk_pid = _v8.child('elk').pid

            # Stop the idle helper without a native fault. Wait without reaping it so the
            # renderer still owns its process status when the next caller hits the dead link.
            child = _v8.child('mathjax')
            pid = child.pid
            os.kill(pid, signal.SIGTERM)
            status = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
            assert status.si_code == os.CLD_KILLED and status.si_status == signal.SIGTERM

            start = time.monotonic()
            try:
                mathsvg.render(r'\beta')  # uncached, so it must contact the helper
            except WorkerError as exc:
                message = str(exc)
                assert 'MathJax worker' in message and 'exited' in message, message
                assert len(message) < 500 and '\n' not in message, message
            else:
                raise AssertionError('lost helper did not surface an error to the caller')
            assert time.monotonic() - start < 10, 'the exited helper was not detected promptly'

            assert engine.layout_graph(graph)['children'][0]['id'] == 'a'
            assert _v8.child('elk').pid == elk_pid, 'unrelated helper was restarted'
            assert '<svg' in mathsvg.render(r'\beta')  # a later ordinary call can recover
            assert child.pid is not None and child.pid != pid
            print('OK', flush=True)
            """
        )


def test_bbox_cache_no_reparse_and_bounded(monkeypatch):
    """Warm re-layouts must hit the (svg, bboxes) cache — token_bboxes ran per label per
    layout before — and the cache must evict past its LRU bound."""
    r = mathsvg.MathRenderer()
    tex = r"\cssId{tok-a}{\alpha} + 1"
    svg1, bb1 = r.render_with_bboxes(tex)
    calls = {"n": 0}
    real = mathsvg.token_bboxes

    def counting(svg):
        calls["n"] += 1
        return real(svg)

    monkeypatch.setattr(mathsvg, "token_bboxes", counting)
    svg2, bb2 = r.render_with_bboxes(tex)
    assert (svg2, bb2) == (svg1, bb1) and "a" in bb2
    assert calls["n"] == 0  # cache hit: no re-parse
    monkeypatch.setattr(r, "_CACHE_MAX", 2)
    for i in range(3):
        r.render_with_bboxes(rf"x_{i}")
    assert len(r._cache) == 2  # LRU bound holds
