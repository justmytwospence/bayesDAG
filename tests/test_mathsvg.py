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


def test_building_both_v8_isolates_at_once_does_not_abort_the_process():
    """bayesdag runs two isolates — MathJax and ELK — each pinned to its own thread. Building
    both at the same moment used to abort the whole interpreter inside V8's allocator
    ("Check failed: !pool->IsInitialized()"), taking the user's kernel with it.

    This is reachable on the ordinary path: `view()` warms the ELK context on its worker thread
    precisely so it overlaps the work that goes on to render labels through MathJax, so whether
    the cold starts collide is a timing accident. Run in a subprocess — the failure mode is a
    fatal signal, not an exception, so an in-process check would kill the test session itself.
    """
    import subprocess
    import sys

    code = (
        "import threading\n"
        "from bayesdag import mathsvg\n"
        "from bayesdag.layout import elk_backend\n"
        "err = []\n"
        "def m():\n"
        "    try: mathsvg.get_renderer().render(r'x \\sim \\mathcal{N}(0,1)')\n"
        "    except Exception as e: err.append(('math', repr(e)))\n"
        "def e():\n"
        "    try:\n"
        "        eng = elk_backend.get_engine()\n"
        "        eng._worker().submit(eng._context).result()\n"
        "    except Exception as exc: err.append(('elk', repr(exc)))\n"
        "ts = [threading.Thread(target=m), threading.Thread(target=e)]\n"
        "for t in ts: t.start()\n"
        "for t in ts: t.join()\n"
        "assert not err, err\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (
        f"concurrent V8 isolate construction failed (returncode {r.returncode}); "
        f"stdout={r.stdout!r} stderr={r.stderr[-2000:]!r}"
    )
    assert "OK" in r.stdout


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
