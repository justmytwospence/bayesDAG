# COVERAGE.md — living work-plan

The single source of "done / next." Tick boxes as constructs/features land; each should
arrive with a golden test.

## Milestones
- [x] **M0** — vertical slice (8-schools, both renderers, narrow coverage) ✅
- [ ] **M1** — coverage to the hard list
- [ ] **M2** — posterior-geometry explorer + polish (first slice landed: live posterior attachment, hedged diagnostic badges, funnel joint, linked views)
- [ ] **M3** — interop exporters, reparam suggestions, scale, cross-PPL, upstream

## M0 build checklist
- [x] Project scaffold (pyproject/package.json/docs/git)
- [x] uv env (Python 3.12) + deps installed; `pymc` 6.0.1 / `arviz` 1.1.0 import
- [x] **Math + token-anchor spike** — mathjax-full runs **in-process** in py_mini_racer (no runtime Node; `PACKAGE_VERSION` define kills the `eval("require")` branch); `\cssId` token anchors extracted via transform-chain composition. `src/bayesdag/mathsvg.py`.
- [x] PPL-agnostic `ir` + `from_pymc` (slot-aware edges, roles, dims/coords, log-transform key) + published JSON Schema + `to_elk`/`to_networkx` + duck-typed `to_ir` dispatch
- [x] Label engine (per-dist LaTeX templates + symbol naming + PyTensor deterministic visitor with port-tokens + token-tree); renders cleanly in MathJax
- [x] Layout backend — **ELK** in-process (`dot -Tjson0` remains only as the `BAYESDAG_LAYOUT=dot` rollback) → `LayoutResult` + coordinate transform + param-edge -> token-anchor post-pass
- [x] Glyph registry (glyph-agnostic) + distribution-data provider: analytic prior densities, observed histograms (FD bins), posterior KDE; `density`/`histogram`/`schematic`/`heatmap` kinds
- [x] Shared SVG emitter (plates, role-styled chrome, embedded MathJax labels, token-anchored edges, glyphs) + static renderer (SVG; PNG/PDF via cairosvg)
- [x] anywidget widget (ships identical SVG) + thin dependency-free controller (hover/pin/expand; no pan/zoom) + `view.py` env-fallback (`_repr_mimebundle_`/`_repr_svg_`/`_display_`, all three tested); `bayesdag.view(model)`
- [x] 8-schools example + marimo notebook (runs end-to-end: prior -> interactive -> fit -> posterior -> export)
- [x] M0 tests: token-bbox, param/label, widget==static SVG bytes, port-edge, glyphs, interop, env-fallback, import-light invariant (the suite is now ~420 tests overall)

## PyMC construct coverage

**Full-catalog status:** every distribution PyMC publishes (84 families in 6.0.1) renders correctly
and honestly — a real symbol + a shape glyph where one exists, a faithful structural glyph for the rich
families, or a hedged `elision_reason` badge where a static picture would mislead. Guaranteed by the
parametrized `tests/test_coverage.py` (all families: convert + layout + render + non-fallback symbol),
whose `test_catalog_covers_every_pymc_family` diffs CATALOG against `pm.distributions.__all__` in both
directions — so a family PyMC adds fails the suite instead of silently going unrendered.
Detection keys on the **RV class** (`type(op).__name__`); sub-RVs/params come from `var.owner.inputs`
(see `adapters/constructs.py`). Param→scipy translations are locked by logp-matching tests.

Catalog coverage (every family renders and is handled) is not the same claim as mathematical
verification (every drawn number is right) or model comprehension (a reader can see what the
model is). The first is `test_coverage.py`; the second is the logp-locked translations plus the
oracle tests in `test_constructs.py` / `test_glyph.py`; the third is the structure layer below.

### Easy (canonical `name ~ Dist(params)` + shape glyph)
- [x] Scalar continuous — **all** univariate families (scipy-backed pdf; closed-form for Kumaraswamy/LogitNormal/HalfStudentT)
- [x] Scalar discrete — **all** univariate families (analytic pmf bars; class bars for observed)
- [x] Dirichlet (simplex marginal-Beta glyph) · Multinomial/DirichletMultinomial/StickBreakingWeights (badge)
- [x] `pm.Data` / `pm.Minibatch`
- [x] `pm.Potential` → log-density factor: `log ψ = <expression>` with token edges; turns forward simulation off (it would ignore the factor)
- [x] `Flat` / `HalfFlat` → improper-prior badge (and they no longer crash `to_ir`)

### Medium
- [x] `pm.Deterministic` (math-mode rendering with node budget + elision; leaf port-tokens) — precedence-aware (`a(b+c)`), visible value-changing casts, named reductions (∑/∏/max/min/mean), matrix products (`@`, `pt.dot`, pytensor 3's `Blockwise(Dot)`), gathers with index tokens, constant slices, concatenate, sort, `set_subtensor`, constant matrices as `C_{r×c}`
- [x] `pm.Deterministic` **transfer-function glyph** (`curve`): canonical, parameter-free shape of the function — logistic/probit/tanh S-curves, exp/log/softplus/sqrt/abs, `x**k` (constant exponent), affine→line (softmax: a caption, not invented bar heights). Drawn ONLY when provable from the op graph (zero false positives); equation-only otherwise. Glyph-bearing deterministics exit their outgoing edge from the node box; equation-only ones from the LHS token.
- [ ] Transforms (log/logodds/simplex/…) as badges via `rvs_to_transforms`
- [x] MvNormal / MvStudentT → `pairplot` (low-dim, covariance ellipses) → `heatmap`; matrix dists (Wishart/MatrixNormal/Kronecker) → `heatmap`
- [ ] Nested / prefixed submodels (group by prefix)
- [ ] `pm.do` / `pm.observe`

### Hard (special-cased)
- [x] Truncated — `TruncatedNormal` clipped+renormalized density; generic `Truncated(X)` → caption
- [x] Censored — base density on the interior of the bounds + probability-mass spikes carrying their true mass
- [x] Mixtures / NormalMixture / ZeroInflated* (weighted overlaid components / the exact mixture pmf, true mass at the point) · Hurdle* → badge
- [x] Timeseries: RandomWalk/GaussianRandomWalk → exact 90% `fan` (`m0 + t·m`, `v0 + t·v`) · AR → PACF only for companion-matrix-stationary coefficients · GARCH11/EulerMaruyama → badge
- [x] LKJCholeskyCov / LKJCorr → marginal-correlation density (or badge) · Wishart → `heatmap`
- [ ] Missing-data imputation (`_observed` + `_unobserved` + join; `PartialObservedRV`)

### Honest degradation (render what's recoverable + "elided" badge)
- [x] CustomDist / DensityDist → badge · Simulator (SMC) → badge (all three in CATALOG; their symbol is deliberately an upright word, since no conventional notation exists)
- [x] Interpolated → density from its `x_points`/`pdf_points` · DiracDelta → caption with its point
- [x] Conditional latents: the exact standardized family shape when parents set only location/scale; otherwise no curve and a caption naming the slot (no generic bell)
- [x] Batched parameters that differ across members never share one picture (caption "elements differ")
- [x] Spatial CAR / ICAR → adjacency `heatmap` of `W`
- [ ] Oversized Deterministic / ODE (DifferentialEquation)
- [ ] GPs (Latent/Marginal/HSGP/TP/Kron) — not in `pm.distributions`; out of scope here

### Known limitations / follow-ups
- Rich glyphs (`pairplot`/`heatmap`/`fan`/`mixture`) get a dimension-scaled in-node block AND
  a large pinned-card panel on click (with coord row labels where a dim matches the matrix
  size) — raising the pairplot dimensionality threshold on hover is now unblocked.
- Multivariate/symbolic labels can be cosmetically ugly (`second(...)`, `cast(...)`) — per-construct
  param-name templates would clean this up.
- Deterministic transfer glyphs are intentionally conservative (zero false positives): a hand-written
  sigmoid `1/(1+e^-x)` (looks like `TrueDiv`), cloglog, probit via non-`erf` helpers, `mean`, `cumsum`
  (non-monotone on signed summands), and multi-transfer composites are NOT detected — they render
  equation-only. The affine line is schematic (canonical), not the node's actual values.
- [ ] Experimental `pymc.dims` xtensor RVs (`XRV`) — best-effort or declared experimental

## Glyph kinds (registry)
- [x] `density` · `histogram` · `schematic` (exact standardized shape) · `curve` (deterministic transfer function) · `heatmap` · `bars` (discrete pmf / observed classes / discrete posterior) · `hist_overlay` (legacy 1.0 IR only) · `stem` (AR PACF) · `step`
- [x] `multi` (one posterior curve per element, ≤8) · `forest` (per-element 50/90% intervals, sorted) · `ppc` (data vs the model's replicates)
- [x] special-construct kinds: `fan` (random-walk band) · `pairplot` (marginals + covariance ellipses) · `mixture` (overlaid components / zero-spike) · `cutpoints` (ordinal) · `simplex` (Dirichlet marginal Beta) · `censored` (base + bound spikes)
- [x] interval display: the `forest` glyph and the paged per-element interval panel (`marginals_svg`)
- [ ] `cdf`/`ccdf`/`gradient`/`dotplot`/`quantile_dotplot`/`band`; `interval`/`point` annotations on a density
- [ ] Cross-cutting: `transform.animate="hops"` · `layout="ridgeline"`

## Exporters / interop
- [x] `to_elk` (ports for every token an edge lands on, nested plates, kind-prefixed ids) · `to_networkx` (+ `markov_blanket`)
- [ ] DOT · GraphML · PROV-JSON-LD · (optional) Cytoscape / Hugin
- [x] Published JSON Schema (draft 2020-12) + validation — 1.1 (blocks, multi-slot edges, plate dims, glyph notes); 1.0 still published and readable
- [x] `schema.check`: references resolve (ids, endpoints, tokens, plate ancestry, blocks, panels)

## Posterior geometry / diagnostics (M2+)
- [x] live posterior attachment (`view.update(idata=…)`) — same layout reused unless a glyph changes size class
- [x] diagnostic badges: R-hat / ESS (worst element on vectors, hedged wording, no R-hat from a single chain) + a model-level divergence note
- [x] funnel auto-flag (structural, and only once a run has divergences to explain)
- [x] graph-selected divergence joint — the first real `AuxViewIR`; prefers `unconstrained_posterior`, labels a computed `log τ` axis as computed; draws joined by chain/draw coordinates; counts transitions
- [x] predictive checks from supplied `posterior_predictive` / `prior_predictive` draws (`ppc` glyph); wide count ranges use shared integer bins instead of dropping tail values
- [x] linked views: `selected_node` as a read-back API (`view.ui()` / `view.on_select`)
- [ ] MCSE badges · divergence attribution · energy/BFMI · parallel-coordinates
- [ ] layout persistence — capture a hand-arranged layout from the widget and replay it byte-identically from the static renderer (the lavaangui `plot`/`export_plot` split; the one-layout-pass design is most of the way there)
- [x] reparameterization suggestions, structural → diagnostic: hedged hints on centered (and non-centered) hierarchies, only when a run diverges
- [ ] … → VIP (variationally inferred parameterization)

## Recognized structures (blocks)
- [x] partial pooling · non-centered (Normal, Student-t aux; exp → LogNormal, invlogit → LogitNormal, `A @ z` → MvNormal) — contracts in the overview
- [x] partial pooling · centered (IR-level, any PPL) — tag only when rank and named dimensions establish sharing
- [x] global–local shrinkage: horseshoe, regularized horseshoe (slab matched structurally), half-t locals named, co-data center/modifier roles — contracts; exact horseshoe marginal glyph
- [x] regressions: linear, robust, log-normal, logistic, binomial, probit, Poisson — tag, `· with group effects`
- [x] measured latent (errors-in-variables) — tag
- [x] user annotations (`BlockSpec`), drawn as annotated
- [x] overview projection (default), expand/collapse, widget block cards, `selected_block`
- [x] fitted panels (`adapters/fitted.py` + `render_panels.py`): shrinkage plot, population distribution, pooling factor, fitted effect, lines per group, intercepts vs slopes, horseshoe κ, local scale vs coefficient, readings vs truth; fitted data comparisons use saved snapshots and coordinates; update invalidates cached panels
- [ ] fitted panels for non-identity links (shrinkage plot on the link scale with a clear caveat) · correlated-effects ellipses from an LKJ/MvNormal fit · grouped posterior predictive checks
- [ ] nested blocks · block hulls in the full graph · missing-data split/join recognition · matrix/constrained-basis reparameterizations (annotate them for now)
- [ ] Nested / prefixed submodels as blocks
