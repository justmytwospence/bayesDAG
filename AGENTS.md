# AGENTS.md — working in the `bayesdag` repo

Context for AI agents (and humans) contributing to `bayesdag`. Read this, the live
work-plan in [`COVERAGE.md`](COVERAGE.md), the full design in
[`.claude/plans/please-review-all-the-streamed-storm.md`](.claude/plans/please-review-all-the-streamed-storm.md),
and the source/decision log in [`docs/RESEARCH.md`](docs/RESEARCH.md) before making changes.

## Dev / test / build commands
```bash
uv sync                              # create .venv (Python 3.12), install deps + dev group, editable install
npm install                          # JS deps (esbuild, d3-*)  — run before the first `uv sync` build, or set HATCH_JUPYTER_BUILDER_SKIP_NPM=1
npm run dev                          # esbuild --watch -> src/bayesdag/static/widget.js
uv run env ANYWIDGET_HMR=1 jupyter lab     # live-reload widget dev (Jupyter)
uv run env ANYWIDGET_HMR=1 marimo edit examples/eight_schools.py   # marimo dev
uv run pytest                        # test suite
uv build                             # wheel (runs esbuild via hatch-jupyter-builder; bundle is included)
```
Toolchain present in this environment: `uv`, `node`/`npx`, Graphviz `dot`, `git`. System Python is 3.14 (too new for PyMC) — the venv is pinned to **3.12** via `.python-version`.

## Module map (`src/bayesdag/`)
- `ir.py` — neutral, **import-light** dataclasses (the single source of truth). No pymc/xarray/render imports.
- `convert.py` — `to_ir(obj)` idempotent, **duck-typed** dispatch (never `isinstance` against a PPL).
- `adapters/pymc.py` — `from_pymc(model, idata=None)`; all PyMC-isms isolated here.
- `adapters.py` / `export.py` — `to_elk`, `to_networkx`; DOT / GraphML / PROV-JSON-LD exporters.
- `labels.py` — per-op LaTeX template registry + deterministic graph→LaTeX visitor + token-tree.
- `mathsvg.py` — TeX→SVG via `mini-racer`/MathJax (cached) + per-token bbox extraction. **Highest-risk module.**
- `layout/graphviz_backend.py` — `dot -Tjson` → `LayoutResult` + coordinate transform + param-edge post-pass (behind a `LayoutBackend` interface).
- `glyph/` — `registry.py` (glyph-agnostic), `spec.py` (`GlyphSpec`), `distribution.py` (scipy/idata adapter), `kinds/`.
- `render_svg.py` — the ONE shared SVG emitter (nodes/edges/plates/glyphs/panels). Both renderers consume it.
- `render_static.py` — standalone SVG + cairosvg PNG/PDF + TikZ.
- `widget.py` — `anywidget.AnyWidget` subclass + synced traitlets.
- `view.py` — `ModelGraphView`: env detection + `_repr_mimebundle_`/`_repr_svg_`/`_display_` fallback.
- `diagnostics.py` — per-node/edge diagnostics + funnel scores + `AuxViewIR` (M2+).
- `js/index.js` — thin d3 controller (zoom/hover/highlight only; **never** layout or stats).

## Load-bearing invariants (do not break)
1. **Layout once, two dumb emitters.** Python computes one `LayoutResult` + one set of MathJax-SVG fragments; static and widget both consume them verbatim. Parity is checked by a golden-SVG diff test.
2. **One math artifact.** Each TeX → SVG once; both renderers embed the same bytes; token bboxes from that SVG anchor port-edges.
3. **`bayesdag.ir` stays import-light** (stdlib only). pymc/xarray/render are extras.
4. **Honesty/representability contract.** Undrawable constructs → factor glyph / "density-only / elided" badge; every auto-flag (incl. funnels) is a hedged "inspect this," never a verdict.
5. **Shape-first glyphs.** The density curve is the primary mark; `interval`/`point` are optional annotations. The glyph registry core is glyph-agnostic — non-univariate kinds (heatmap/ternary/rose/joint) are first-class.
6. **node id = the constrained `idata` variable name** (the universal join key between graph, plates, and posterior).
7. The JS layer never computes geometry or statistics.

## Recipes
- **Add a distribution/construct handler:** extend the label registry in `labels.py` (LaTeX template + op param names from `inspect.signature(type(op).__call__)`), add a coverage row in `COVERAGE.md`, and a golden test. For `SymbolicRandomVariable`s recurse via the inner graph.
- **Register a new glyph kind:** `glyph/registry.register(kind, validate_fn, render_fn)` in `glyph/kinds/`; add it to the auto-selection table; it consumes a distribution-object (density/cdf/quantile/sample) so it's source-agnostic. Do NOT assume `interval`/`point` exist.
- **Add an exporter:** new function in `export.py` taking a `ModelIR`; lossy is fine for reach (document what's dropped).

## Conventions
Conventional commits (`feat:`/`fix:`/`refactor:`/`test:`/`docs:`/`chore:`); atomic commits; never force-push main; no Co-Authored-By lines. Run `uv run ruff` and `uv run pytest` before committing.
