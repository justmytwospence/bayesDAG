# AGENTS.md — working in the `bayesdag` repo

Context for AI agents (and humans) contributing to `bayesdag`. Read this, the live
work-plan in [`COVERAGE.md`](COVERAGE.md), and the source/decision log in
[`docs/RESEARCH.md`](docs/RESEARCH.md) before making changes.

## Dev / test / build commands
```bash
uv sync                              # create .venv (Python 3.12+), install deps + dev group + extras, editable install
npm ci                               # JS deps (esbuild, elkjs, mathjax-full) — before the first `uv sync` build, or set HATCH_JUPYTER_BUILDER_SKIP_NPM=1
npm run dev                          # esbuild --watch -> src/bayesdag/static/widget.js
uv run env ANYWIDGET_HMR=1 jupyter lab     # live-reload widget dev (Jupyter)
uv run env ANYWIDGET_HMR=1 marimo edit examples/bayesdag_gallery.py   # marimo dev
uv run pytest                        # test suite (BAYESDAG_REQUIRE_FULL=1 makes skips failures, as CI does)
uv run ruff check && uv run ruff format --check   # lint + format gate (both run in CI)
uv build                             # sdist + wheel. ALWAYS re-runs the npm build (no skip-if-exists —
                                     # it once shipped a stale bundle); `ensured-targets` fails the build
                                     # if any of the five static artifacts is missing.
```
Toolchain present in this environment: `uv`, `node`/`npx`, Graphviz `dot`, `git`. System Python is 3.14 (too new for PyMC) — the venv is pinned to **3.12** via `.python-version`. `requires-python` is **>=3.12** because PyMC 6 itself requires it. Graphviz is optional (the `BAYESDAG_LAYOUT=dot` rollback only).

## Module map (`src/bayesdag/`)
- `ir.py` — neutral, **import-light** dataclasses (the single source of truth). No pymc/xarray/render imports.
- `convert.py` — `to_ir(obj)` idempotent, **duck-typed** dispatch (never `isinstance` against a PPL).
- `adapters/pymc.py` — `from_pymc(model, idata=None)`; all PyMC-isms isolated here.
- `adapters/glyph_data.py` — distribution-shape DATA provider: per-node `(GlyphSpec, glyph_data, elision_reason)`. Verified PyMC→scipy param translations (continuous pdf, discrete pmf, closed-form for Kumaraswamy/LogitNormal/HalfStudentT), observed histogram + best-fit overlay, posterior KDE.
- `adapters/constructs.py` — special-construct glyphs. Detect by **RV class** (`type(op).__name__`); recover sub-RVs/params from `var.owner.inputs`. Faithful glyph (pairplot/heatmap/simplex/fan/censored/mixture/AR-marginal) or an honest `elision_reason` badge. Every branch guarded — never crash, never lie.
- `adapters/deterministic.py` — `pm.Deterministic` **transfer-function** glyphs (`curve`/`bars`). Find the PRINCIPAL op (unwrap value-preserving wrappers + strip scalar-constant affine framing, sign-tracked), then map a recognized transfer (Sigmoid/Erf/Tanh/Exp/Log/Softplus/Sqrt/Abs/Pow-const-exp) → its canonical curve, Softmax → bars, provably-affine → line. **ZERO false positives by construction**: only draw when the shape is a mathematical consequence of the op graph (elementwise ⇒ pointwise T; `is_affine` is a degree-preserving whitelist that rejects parent×parent); everything else skips. Curves are evaluated from the true function (parameter-free, deterministic). Leaf set = the `named` model vars (mirrors `pytensor_latex`).
- `adapters/graph.py` — `to_elk` (an INTEROP export, deliberately separate from the layout's own graph builder), `to_networkx`, `markov_blanket`. (DOT / GraphML / PROV-JSON-LD exporters are M3 roadmap — not yet written.)
- `adapters/ppc.py` — per-plate prior-predictive expansions (forward simulation → per-instance marginals + the observed points). Interactive-only, computed lazily.
- `geometry.py` — node sizing, label origin, and the glyph strip. Reserves space by glyph **presence** (`has_glyph_data`), never by role.
- `legend.py` — the context-aware legend; `_SOURCE_LABELS` is the single source of the glyph-source wording (the node panels reuse it).
- `schema.py` — builds the JSON Schema from the IR dataclasses and validates against it; `python -m bayesdag.schema` regenerates the published `schema/graph-v1.0.json` (a test asserts the two agree).
- `labels.py` — `DIST_SYMBOLS` (full catalog, keyed on derived op names) + deterministic graph→LaTeX visitor + token-tree.
- `mathsvg.py` — TeX→SVG via `mini-racer`/MathJax (LRU-cached with the token bboxes) + per-token bbox extraction. Its V8 isolate is pinned to a dedicated thread for the same reason ELK's is (see `layout/`). **Highest-risk module.**
- `layout/` — **ELK is the engine.** `elk_backend.py` runs `elkjs` in-process in `mini-racer` V8 (`hierarchyHandling=INCLUDE_CHILDREN` + fixed-position token **ports** so plates and equation-edges lay out correctly). It runs all V8 work on ONE dedicated thread (mini-racer binds its loop to the creating thread and forbids blocking `.get()` under a live asyncio loop, e.g. a marimo cell). **No automatic fallback**: if ELK can't run, `layout()` *raises* (a silent downgrade once shipped a worse layout while reporting success). `graphviz_backend.py` (`dot -Tjson0`) is reachable ONLY via the explicit `BAYESDAG_LAYOUT=dot` opt-in / git-rollback target. `common.py` holds the engine-agnostic bits (label measurement, token-anchor projection, the smooth edge). Token-level port anchors are computed by us from MathJax bboxes (engine-independent).
- `glyph/` — `registry.py` (glyph-agnostic kind→render-fn) + `kinds.py` (density/schematic/histogram/bars/hist_overlay/heatmap/fan/pairplot/mixture/cutpoints/simplex/censored/`curve`/`stem`/`step`). `curve` is an unfilled polyline (a deterministic's transfer function — a function, not a density). `GlyphSpec` lives in `ir.py`; the data provider is `adapters/glyph_data.py`.
- `render_svg.py` — the ONE shared SVG emitter (nodes/edges/plates/glyphs/panels). Both renderers consume it.
- `render_static.py` — standalone SVG + cairosvg PNG/PDF (TikZ is declared but raises `NotImplementedError` — M2).
- `widget.py` — `anywidget.AnyWidget` subclass + synced traitlets.
- `view.py` — `ModelGraphView`: env detection + `_repr_mimebundle_`/`_repr_svg_`/`_display_` fallback.
- `diagnostics.py` — **not yet written** (M2): per-node/edge diagnostics + funnel scores. Its `AuxViewIR` placeholder already lives in `ir.py`.
- `js/index.js` — thin controller: hover-highlight, tooltips, click-to-pin cards, plate expansion. Imports **nothing** (no d3, no pan/zoom — the diagram renders at natural size) and **never** computes layout or stats. `render()` returns anywidget's cleanup function.

## Load-bearing invariants (do not break)
1. **Layout once, two dumb emitters.** Python computes one `LayoutResult` + one set of MathJax-SVG fragments; static and widget both consume them verbatim. `tests/test_view.py` asserts the widget ships the same `to_svg` bytes (the two differ only by the legend, which is opt-out per renderer). That is only a consistency check, so the regression net is `tests/test_golden.py`: a committed reference render of eight-schools (`tests/golden/`), compared as an exact non-numeric skeleton plus per-number tolerance of one emitted decimal. Regenerate it deliberately with `pytest --golden-update` and review the diff.
   The `LayoutResult` is the geometry source of truth; the `box`/`port_anchors` copies on `NodeIR` are a convenience mirror, cleared at the start of every layout.
2. **One math artifact.** Each TeX → SVG once; both renderers embed the same bytes; token bboxes from that SVG anchor port-edges.
3. **`bayesdag.ir` stays import-light** (stdlib only). pymc/xarray/render are extras.
4. **Honesty/representability contract.** Undrawable constructs → an `elision_reason` badge naming the reason (`representable = False`); every auto-flag (incl. funnels) is a hedged "inspect this," never a verdict. A glyph must never state something the model does not say: no value sampled through a parent, no single curve standing in for elements that differ, no exaggerated mark presented as a readable quantity (the censored spikes ship their true mass in the data and caption it).
5. **Shape-first glyphs.** The density curve is the primary mark; `interval`/`point` annotations are M2 (declared on `GlyphSpec`, nothing sets them). The glyph registry core is glyph-agnostic, so non-univariate kinds are first-class by construction — `heatmap` and `pairplot` are built; ternary/rose/joint would slot in the same way but do not exist yet. `glyph.registered_kinds()` is the authority on what actually renders.
6. **node id = the constrained `idata` variable name** (the universal join key between graph, plates, and posterior).
7. The JS layer never computes geometry or statistics.

## Recipes
- **Add a distribution/construct handler:** (1) add its symbol to `DIST_SYMBOLS` in `labels.py` keyed on the **derived** name (`op._print_name[0]`, NOT the public class — many collapse, e.g. `MvNormal`→`"MultivariateNormal"`, `ZeroInflated*`→`"Mixture"`); (2) for a univariate shape, add the verified PyMC→scipy param translation to `glyph_data._scipy_frozen`/`_discrete_frozen` (op params are in `op.dist_params` order and may already be reparametrized — e.g. Exponential exposes *scale*; **lock it with a logp-matching test** in `test_shapes.py`); (3) for a special construct, add a branch in `constructs.special_glyph` keyed on `type(op).__name__`, pulling sub-RVs/params from `var.owner.inputs` — return a glyph or an honest `elision_reason` badge; (4) add it to `tests/test_coverage.py::CATALOG` — this is ENFORCED: `test_catalog_covers_every_pymc_family` diffs CATALOG against `pm.distributions.__all__` both ways, and `test_every_translation_is_locked` reads the `_scipy_frozen`/`_discrete_frozen` tables out of their own AST, so a translation with no logp-matching case fails the suite. Everything guards with try/except and degrades to a badge — never crash `to_ir`. **Never `.eval()` a param without first checking `_depends_on_rv`** — `.eval()` silently *draws a random sample* through any parent RV, which would mislabel a conditional latent as an analytic root prior and produce a non-deterministic shape. `_numeric_params` returns None for any prior-governed param (→ family-only schematic); if a construct needs only a numeric *subset* (e.g. LKJ's leading `n, eta`, not its `sd_dist` prior), read it with `_lead_numeric(var, k)` so a legitimate prior sub-param doesn't sink a knowable, deterministic shape.
- **Register a new glyph kind:** add `render_<kind>(data, box, *, stroke, fill, **_)` to `glyph/kinds.py` and `register("<kind>", …)`; reuse `_poly`/`_bars`/`bar_layout`. The data dict is whatever the provider ships (registry doesn't validate). Do NOT assume `interval`/`point` exist. Geometry reserves a glyph strip by **presence** (`geometry.has_glyph_data(kind, data)`), not by role — so any node carrying glyph data (incl. a deterministic) gets space, and a kind-but-no-data badge does not.
- **Add a deterministic transfer glyph:** extend `_TRANSFER_BUILDERS` in `adapters/deterministic.py` with the principal scalar-op name → a canonical-curve builder (evaluate the TRUE function on a fixed grid, normalize to [0,1]). **Honesty rule: only emit a glyph whose shape is provable from the op structure** — a recognized elementwise transfer (pointwise by definition) or a `_is_affine` line; never reconstruct a function from a guess (e.g. manual `1/(1+e^-x)` stays equation-only). Add a row to `tests/test_deterministic_glyph.py` (draw or skip).
- **Add an exporter:** create `export.py` (doesn't exist yet) with a function taking a `ModelIR`; lossy is fine for reach (document what's dropped).
- **Bundle a new JS dependency:** add it to `package.json`, wire a `build:*` step, add its output to BOTH `ensured-targets` and the CI wheel-contents check — and add its license to `THIRD_PARTY_LICENSES`. The wheel redistributes these binaries (elkjs is EPL-2.0, MathJax Apache-2.0), and both licenses require the notice to travel with them.
- **Change the IR:** re-run `uv run python -m bayesdag.schema` to regenerate `schema/graph-v1.0.json`; `tests/test_interop.py` fails if the published file drifts from `build_schema()`. Note the schema/IR reader accepts both `Optional[X]` and `X | None` — it must, since either spelling can appear.

## Conventions
Conventional commits (`feat:`/`fix:`/`refactor:`/`test:`/`docs:`/`chore:`); atomic commits; never force-push main; no Co-Authored-By lines. Run `uv run ruff check`, `uv run ruff format --check` and `uv run pytest` before committing.

A skip is not a pass: ~17% of the suite sits behind optional-dependency gates, so `-rs` lists every
skip and CI runs with `BAYESDAG_REQUIRE_FULL=1`, which turns skips into failures. If you add a gated
test, make sure the dependency is in the dev group so CI actually exercises it.
