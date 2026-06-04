# COVERAGE.md — living work-plan

The single source of "done / next." Tick boxes as constructs/features land; each should
arrive with a golden test. Tiers and treatments come from the design plan
([`.claude/plans/please-review-all-the-streamed-storm.md`](.claude/plans/please-review-all-the-streamed-storm.md)).

## Milestones
- [x] **M0** — vertical slice (8-schools, both renderers, narrow coverage) ✅
- [ ] **M1** — coverage to the hard list
- [ ] **M2** — posterior-geometry explorer + polish
- [ ] **M3** — interop exporters, reparam suggestions, scale, cross-PPL, upstream

## M0 build checklist
- [x] Project scaffold (pyproject/package.json/docs/git)
- [x] uv env (Python 3.12) + deps installed; `pymc` 6.0.1 / `arviz` 1.1.0 import
- [x] **Math + token-anchor spike** — mathjax-full runs **in-process** in py_mini_racer (no runtime Node; `PACKAGE_VERSION` define kills the `eval("require")` branch); `\cssId` token anchors extracted via transform-chain composition. `src/bayesdag/mathsvg.py`.
- [x] PPL-agnostic `ir` + `from_pymc` (slot-aware edges, roles, dims/coords, log-transform key) + published JSON Schema + `to_elk`/`to_networkx` + duck-typed `to_ir` dispatch
- [x] Label engine (per-dist LaTeX templates + symbol naming + PyTensor deterministic visitor with port-tokens + token-tree); renders cleanly in MathJax
- [x] Layout backend (`dot -Tjson0` → `LayoutResult` + coordinate transform + param-edge -> token-anchor post-pass)
- [x] Glyph registry (glyph-agnostic) + distribution-data provider: analytic prior densities, observed histograms (FD bins), posterior KDE; `density`/`histogram`/`schematic`/`heatmap` kinds
- [x] Shared SVG emitter (plates, role-styled chrome, embedded MathJax labels, token-anchored edges, glyphs) + static renderer (SVG; PNG/PDF via cairosvg)
- [x] anywidget widget (ships identical SVG) + thin d3 controller (pan/zoom) + `view.py` env-fallback (`_repr_mimebundle_`/`_repr_svg_`/`_display_`); `bayesdag.view(model)`
- [x] 8-schools example + marimo notebook (runs end-to-end: prior -> interactive -> fit -> posterior -> export)
- [x] M0 tests (48): token-anchor, param/label, parity (widget==static SVG), port-edge, glyphs, interop, env-fallback, import-light invariant

## PyMC construct coverage

**Full-catalog status:** every distribution in PyMC 6.0.1 (~82 families) now renders correctly and
honestly — a real symbol + a shape glyph where one exists, a faithful structural glyph for the rich
families, or a hedged `elision_reason` badge where a static picture would mislead. Guaranteed by the
parametrized `tests/test_coverage.py` (all families: convert + layout + render + non-fallback symbol).
Detection keys on the **RV class** (`type(op).__name__`); sub-RVs/params come from `var.owner.inputs`
(see `adapters/constructs.py`). Param→scipy translations are locked by logp-matching tests.

### Easy (canonical `name ~ Dist(params)` + shape glyph)
- [x] Scalar continuous — **all** univariate families (scipy-backed pdf; closed-form for Kumaraswamy/LogitNormal/HalfStudentT)
- [x] Scalar discrete — **all** univariate families (analytic pmf bars; class bars for observed)
- [x] Dirichlet (simplex marginal-Beta glyph) · Multinomial/DirichletMultinomial/StickBreakingWeights (badge)
- [x] `pm.Data` / `pm.Minibatch`
- [ ] `pm.Potential` → factor glyph
- [x] `Flat` / `HalfFlat` → improper-prior badge (and they no longer crash `to_ir`)

### Medium
- [x] `pm.Deterministic` (math-mode rendering with node budget + elision; leaf port-tokens) — basic ops (add/mul/sub/div/pow/exp/log/sqrt); more ops as needed
- [ ] Transforms (log/logodds/simplex/…) as badges via `rvs_to_transforms`
- [x] MvNormal / MvStudentT → `pairplot` (low-dim, covariance ellipses) → `heatmap`; matrix dists (Wishart/MatrixNormal/Kronecker) → `heatmap`
- [ ] Nested / prefixed submodels (group by prefix)
- [ ] `pm.do` / `pm.observe`

### Hard (special-cased)
- [x] Truncated — `TruncatedNormal` clipped+renormalized density; generic `Truncated(X)` → badge
- [x] Censored — base density + probability-mass spikes at the bounds
- [x] Mixtures / NormalMixture / ZeroInflated* (composite: overlaid components / base pmf + zero-spike) · Hurdle* → badge
- [x] Timeseries: RandomWalk/GaussianRandomWalk → `fan` chart · AR → stationary marginal · GARCH11/EulerMaruyama → badge
- [x] LKJCholeskyCov / LKJCorr → badge · Wishart → `heatmap`
- [ ] Missing-data imputation (`_observed` + `_unobserved` + join; `PartialObservedRV`)

### Honest degradation (render what's recoverable + "elided" badge)
- [x] CustomDist / DensityDist → badge · Simulator (SMC) → badge
- [x] Interpolated → density from its `x_points`/`pdf_points` · DiracDelta → point
- [x] Spatial CAR / ICAR → adjacency `heatmap` of `W`
- [ ] Oversized Deterministic / ODE (DifferentialEquation)
- [ ] GPs (Latent/Marginal/HSGP/TP/Kron) — not in `pm.distributions`; out of scope here

### Known limitations / follow-ups
- In-node `pairplot` is cramped in the 30px glyph strip — a squarer glyph area (or a card-panel
  upgrade) would let the dimensionality threshold rise on hover (as designed).
- Multivariate/symbolic labels can be cosmetically ugly (`second(...)`, `cast(...)`) — per-construct
  param-name templates would clean this up.
- [ ] Experimental `pymc.dims` xtensor RVs (`XRV`) — best-effort or declared experimental

## Glyph kinds (registry)
- [x] `density` · `histogram` · `schematic` · `heatmap` · `bars` (discrete pmf) · `hist_overlay` (observed data + best-fit family)
- [x] special-construct kinds: `fan` (random-walk band) · `pairplot` (marginals + covariance ellipses) · `mixture` (overlaid components / zero-spike) · `cutpoints` (ordinal) · `simplex` (Dirichlet marginal Beta) · `censored` (base + bound spikes)
- [ ] `cdf`/`ccdf`/`gradient`/`dotplot`/`quantile_dotplot`/`band`; `interval`/`point` annotations
- [ ] Cross-cutting: `transform.animate="hops"` · `layout="ridgeline"`

## Exporters / interop
- [x] `to_elk` · `to_networkx` (+ `markov_blanket`)
- [ ] DOT · GraphML · PROV-JSON-LD · (optional) Cytoscape / Hugin
- [x] Published JSON Schema (draft 2020-12) + validation

## Posterior geometry / diagnostics (M2+)
- [ ] graph-selected divergence joints · auto unconstrained axis (`log τ`) · diagnostic badges (R-hat/ESS/MCSE/divergence-involvement)
- [ ] funnel auto-flag (tiered, hedged) · divergence attribution · energy/BFMI · parallel-coordinates
- [ ] reparameterization suggestions (structural → diagnostic → VIP)
