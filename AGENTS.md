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
uv run env ANYWIDGET_HMR=1 marimo edit examples/bayesdag_gallery.py   # marimo dev
uv run pytest                        # test suite
uv build                             # wheel (runs esbuild via hatch-jupyter-builder; bundle is included)
```
Toolchain present in this environment: `uv`, `node`/`npx`, Graphviz `dot`, `git`. System Python is 3.14 (too new for PyMC) — the venv is pinned to **3.12** via `.python-version`.

## Module map (`src/bayesdag/`)
- `ir.py` — neutral, **import-light** dataclasses (the single source of truth). No pymc/xarray/render imports.
- `convert.py` — `to_ir(obj)` idempotent, **duck-typed** dispatch (never `isinstance` against a PPL).
- `adapters/pymc.py` — `from_pymc(model, idata=None)`; all PyMC-isms isolated here.
- `adapters/glyph_data.py` — distribution-shape DATA provider: per-node `(GlyphSpec, glyph_data, elision_reason)`. Verified PyMC→scipy param translations (continuous pdf, discrete pmf, closed-form for Kumaraswamy/LogitNormal/HalfStudentT), observed histogram + best-fit overlay, posterior KDE.
- `adapters/constructs.py` — special-construct glyphs. Detect by **RV class** (`type(op).__name__`); recover sub-RVs/params from `var.owner.inputs`. Faithful glyph (pairplot/heatmap/simplex/fan/censored/mixture/AR-marginal) or an honest `elision_reason` badge. Every branch guarded — never crash, never lie.
- `adapters.py` / `export.py` — `to_elk`, `to_networkx`; DOT / GraphML / PROV-JSON-LD exporters.
- `labels.py` — `DIST_SYMBOLS` (full catalog, keyed on derived op names) + deterministic graph→LaTeX visitor + token-tree.
- `mathsvg.py` — TeX→SVG via `mini-racer`/MathJax (cached) + per-token bbox extraction. **Highest-risk module.**
- `layout/` — **ELK is the engine.** `elk_backend.py` runs `elkjs` in-process in `mini-racer` V8 (`hierarchyHandling=INCLUDE_CHILDREN` + fixed-position token **ports** so plates and equation-edges lay out correctly). It runs all V8 work on ONE dedicated thread (mini-racer binds its loop to the creating thread and forbids blocking `.get()` under a live asyncio loop, e.g. a marimo cell). **No automatic fallback**: if ELK can't run, `layout()` *raises* (a silent downgrade once shipped a worse layout while reporting success). `graphviz_backend.py` (`dot -Tjson`) is reachable ONLY via the explicit `BAYESDAG_LAYOUT=dot` opt-in / git-rollback target. `common.py` holds the engine-agnostic bits (label measurement, token-anchor projection, the smooth edge). Token-level port anchors are computed by us from MathJax bboxes (engine-independent).
- `glyph/` — `registry.py` (glyph-agnostic kind→render-fn) + `kinds.py` (density/histogram/bars/hist_overlay/heatmap/fan/pairplot/mixture/cutpoints/simplex/censored). `GlyphSpec` lives in `ir.py`; the data provider is `adapters/glyph_data.py`.
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
- **Add a distribution/construct handler:** (1) add its symbol to `DIST_SYMBOLS` in `labels.py` keyed on the **derived** name (`op._print_name[0]`, NOT the public class — many collapse, e.g. `MvNormal`→`"MultivariateNormal"`, `ZeroInflated*`→`"Mixture"`); (2) for a univariate shape, add the verified PyMC→scipy param translation to `glyph_data._scipy_frozen`/`_discrete_frozen` (op params are in `op.dist_params` order and may already be reparametrized — e.g. Exponential exposes *scale*; **lock it with a logp-matching test** in `test_shapes.py`); (3) for a special construct, add a branch in `constructs.special_glyph` keyed on `type(op).__name__`, pulling sub-RVs/params from `var.owner.inputs` — return a glyph or an honest `elision_reason` badge; (4) add it to `tests/test_coverage.py::CATALOG`. Everything guards with try/except and degrades to a badge — never crash `to_ir`.
- **Register a new glyph kind:** add `render_<kind>(data, box, *, stroke, fill, **_)` to `glyph/kinds.py` and `register("<kind>", …)`; reuse `_poly`/`_bars`/`bar_layout`. The data dict is whatever the provider ships (registry doesn't validate). Do NOT assume `interval`/`point` exist.
- **Add an exporter:** new function in `export.py` taking a `ModelIR`; lossy is fine for reach (document what's dropped).

## Conventions
Conventional commits (`feat:`/`fix:`/`refactor:`/`test:`/`docs:`/`chore:`); atomic commits; never force-push main; no Co-Authored-By lines. Run `uv run ruff` and `uv run pytest` before committing.
