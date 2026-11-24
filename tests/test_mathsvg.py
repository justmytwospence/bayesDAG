"""TeX->SVG + token-anchor extraction (the M0 spike, as regression tests)."""

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


def test_broken_bundle_fails_once_not_per_label(tmp_path, monkeypatch):
    """A bundle that throws during eval must be attempted ONCE — every later render re-raises
    the cached error instead of re-evaluating the multi-MB bundle per label."""
    import py_mini_racer

    calls = {"n": 0}
    real = py_mini_racer.MiniRacer

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(py_mini_racer, "MiniRacer", counting)
    bundle = tmp_path / "mathjax.bundle.js"
    bundle.write_text("throw new Error('boom');")
    r = mathsvg.MathRenderer(bundle_path=bundle)
    with pytest.raises(Exception) as e1:
        r.render(r"\alpha")
    with pytest.raises(Exception) as e2:
        r.render(r"\beta")
    assert e2.value is e1.value  # the same cached exception object — no rebuild
    assert calls["n"] == 1


def test_renders_in_a_notebook_kernel_without_deadlocking():
    """The marimo hazard, and why the MathJax isolate is thread-pinned like ELK's.

    mini-racer binds its event loop to whatever loop is current when the context is BUILT, and
    a later synchronous call from off-loop does `run_coroutine_threadsafe(...).result()`. So an
    isolate built while a notebook kernel's loop is current, then driven from ordinary
    synchronous cell code, blocks forever on a loop nobody is running — verified: the
    un-pinned construction deadlocks in exactly this shape. Building on a dedicated, loop-free
    thread makes mini-racer own a private loop instead, so the call always completes.

    Labels are rendered eagerly from the caller's thread, so pinning ELK alone did not cover this.
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


@pytest.mark.parametrize("first", ["math", "elk"])
def test_default_engines_share_worker_and_context(first):
    """Both initialization orders, multiple instances and concurrent callers share V8.

    Run in a fresh interpreter so neither the module singleton nor another test can warm the
    context. Successful process exit also exercises normal interpreter shutdown.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(r"""
        import concurrent.futures
        import sys
        import threading
        import py_mini_racer
        from bayesdag import mathsvg
        from bayesdag.layout.elk_backend import ElkEngine, get_engine

        real_mini_racer = py_mini_racer.MiniRacer
        contexts = []
        def record_context(*args, **kwargs):
            ctx = real_mini_racer(*args, **kwargs)
            contexts.append((ctx, threading.get_ident()))
            return ctx
        py_mini_racer.MiniRacer = record_context

        renderers = [mathsvg.get_renderer(), mathsvg.MathRenderer()]
        engines = [get_engine(), ElkEngine()]
        graph = {
            'id': 'root', 'layoutOptions': {'elk.algorithm': 'layered'},
            'children': [{'id': 'a', 'width': 30, 'height': 20},
                         {'id': 'b', 'width': 30, 'height': 20}],
            'edges': [{'id': 'ab', 'sources': ['a'], 'targets': ['b']}],
        }
        def render(i, suffix=0):
            svg, boxes = renderers[i].render_with_bboxes(
                r'\cssId{tok-alpha}{\alpha} + \beta' + f' + {suffix}')
            assert '<svg' in svg and '<path' in svg and 'alpha' in boxes
            assert boxes['alpha'][2] > 0 and boxes['alpha'][3] > 0
        def layout(i):
            result = engines[i].layout_graph(graph)
            assert {child['id'] for child in result['children']} == {'a', 'b'}
            assert all('x' in child and 'y' in child for child in result['children'])
            assert result['edges'][0]['sections']

        operations = {'math': lambda: render(0), 'elk': lambda: layout(0)}
        first = sys.argv[1]
        operations[first]()
        operations['elk' if first == 'math' else 'math']()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as callers:
            futures = [callers.submit(render, i, i + 1) for i in range(2)]
            futures += [callers.submit(layout, i) for i in range(2)]
            for future in futures:
                future.result()
        assert len(contexts) == 1, f'expected one V8 context, got {len(contexts)}'
        workers = [obj._worker() for obj in [*renderers, *engines]]
        assert all(worker is workers[0] for worker in workers)
        assert workers[0].submit(threading.get_ident).result() == contexts[0][1]
        print('OK', flush=True)
    """)
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, first], capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"V8 startup, calls or shutdown hung: {exc}", pytrace=False)
    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr}"
    assert result.stdout.strip() == "OK"


def test_concurrent_v8_cold_start_and_shutdown():
    """Concurrent cold starts can abort inside V8's allocator, and on Python 3.13
    leaving both live isolates to finalization can hang after the work has finished.

    `view()` warms ELK while MathJax renders labels, so exercise both in a fresh process.
    Require ordinary interpreter shutdown with the isolates still held by their singletons:
    a completion marker alone would miss the shutdown deadlock.
    """
    import subprocess
    import sys

    code = (
        "import faulthandler, threading\n"
        "faulthandler.dump_traceback_later(45)\n"
        "from bayesdag import mathsvg\n"
        "from bayesdag.layout import elk_backend\n"
        "err = []\n"
        "start = threading.Barrier(2)\n"
        "def m():\n"
        "    start.wait()\n"
        "    try: mathsvg.get_renderer().render(r'x \\sim \\mathcal{N}(0,1)')\n"
        "    except Exception as e: err.append(('math', repr(e)))\n"
        "def e():\n"
        "    start.wait()\n"
        "    try:\n"
        "        eng = elk_backend.get_engine()\n"
        "        graph = {'id': 'root', 'layoutOptions': {'elk.algorithm': 'layered'},\n"
        "                 'children': [{'id': 'a', 'width': 20, 'height': 20},\n"
        "                              {'id': 'b', 'width': 20, 'height': 20}],\n"
        "                 'edges': [{'id': 'ab', 'sources': ['a'], 'targets': ['b']}]}\n"
        "        result = eng.layout_graph(graph)\n"
        "        assert result['edges'][0]['sections']\n"
        "    except Exception as exc: err.append(('elk', repr(exc)))\n"
        "ts = [threading.Thread(target=m), threading.Thread(target=e)]\n"
        "for t in ts: t.start()\n"
        "for t in ts: t.join()\n"
        "assert not err, err\n"
        "print('OK', flush=True)\n"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = (exc.stderr or b"").decode(errors="replace")
        pytest.fail(f"V8 startup or shutdown hung; stdout={exc.stdout!r}\n{stderr}", pytrace=False)
    assert r.returncode == 0, (
        f"concurrent V8 startup or shutdown failed (returncode {r.returncode}); "
        f"stdout={r.stdout!r}\nstderr={r.stderr}"
    )
    assert r.stdout.strip() == "OK"


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
