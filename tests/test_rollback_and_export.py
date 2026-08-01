"""The escape hatches: the BAYESDAG_LAYOUT=dot rollback target and the raster/vector exports.

Both are documented, user-reachable paths with no test of their own. A rollback path that is
never exercised is a rollback path that won't work on the day it's needed.
"""

import shutil
import subprocess

import pytest

from bayesdag.layout import layout
from bayesdag.render_static import save
from bayesdag.render_svg import to_svg

_HAS_DOT = shutil.which("dot") is not None


@pytest.mark.skipif(not _HAS_DOT, reason="needs the graphviz `dot` binary")
def test_dot_rollback_backend_produces_a_usable_layout(monkeypatch, eight_schools_ir):
    """AGENTS.md keeps `dot` as the deliberate opt-in rollback. It must still lay out every node,
    nest the plate, and render."""
    monkeypatch.setenv("BAYESDAG_LAYOUT", "dot")
    res = layout(eight_schools_ir)

    assert set(res.node_boxes) == {"mu", "tau", "eta", "theta", "y_obs"}
    assert res.canvas.w > 0 and res.canvas.h > 0
    assert "plate_school" in res.plate_boxes
    assert "<svg" in to_svg(eight_schools_ir, res)


@pytest.mark.skipif(not _HAS_DOT, reason="needs the graphviz `dot` binary")
def test_dot_backend_is_only_reachable_by_explicit_opt_in(monkeypatch, eight_schools_ir):
    """No silent downgrade: without the env var the ELK backend is used even though `dot` is
    installed and available."""
    import bayesdag.layout.graphviz_backend as gvb

    monkeypatch.delenv("BAYESDAG_LAYOUT", raising=False)
    calls = []
    monkeypatch.setattr(gvb, "layout", lambda *a, **k: calls.append(1))
    layout(eight_schools_ir)
    assert calls == []


def test_dot_timeout_is_reported_not_hung(monkeypatch, eight_schools_ir):
    """A pathological graph must not hang the interpreter forever."""
    import bayesdag.layout.graphviz_backend as gvb

    def timeout(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="dot", timeout=gvb._DOT_TIMEOUT_S)

    monkeypatch.setattr(gvb.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="did not finish"):
        gvb._run_dot("digraph {}")


def test_unsupported_export_format_is_rejected(tmp_path, eight_schools_ir):
    svg = to_svg(eight_schools_ir, layout(eight_schools_ir))
    with pytest.raises(ValueError, match="unsupported output format"):
        save(svg, tmp_path / "m.jpeg")


def test_png_export_round_trip(tmp_path, eight_schools_ir):
    try:  # cairosvg imports fine but raises OSError when the system cairo lib is absent
        import cairosvg  # noqa: F401
    except Exception as exc:
        pytest.skip(f"needs the 'export' extra + a system cairo library ({type(exc).__name__})")
    svg = to_svg(eight_schools_ir, layout(eight_schools_ir))
    out = save(svg, tmp_path / "m.png")
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # a real PNG, not an SVG with a .png name


def test_package_versions_agree():
    """pyproject and package.json both hardcode the version; nothing kept them in sync, and the
    wheel bundles the JS built from the latter."""
    import json
    import pathlib
    import tomllib

    root = pathlib.Path(__file__).resolve().parent.parent
    py = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    js = json.loads((root / "package.json").read_text())["version"]
    assert py == js, f"pyproject {py} != package.json {js}"
