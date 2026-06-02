# COVERAGE.md — living work-plan

The single source of "done / next." Tick boxes as constructs/features land; each should
arrive with a golden test. Tiers and treatments come from the design plan
([`.claude/plans/please-review-all-the-streamed-storm.md`](.claude/plans/please-review-all-the-streamed-storm.md)).

## Milestones
- [ ] **M0** — vertical slice (8-schools, both renderers, narrow coverage)
- [ ] **M1** — coverage to the hard list
- [ ] **M2** — posterior-geometry explorer + polish
- [ ] **M3** — interop exporters, reparam suggestions, scale, cross-PPL, upstream

## M0 build checklist
- [ ] Project scaffold (pyproject/package.json/docs/git) — *in progress*
- [ ] uv env (Python 3.12) + deps installed; `pymc` imports
- [ ] **Math + token-anchor spike** (mathjax-full in mini-racer? fallback ladder; token bboxes) — *highest risk, first*
- [ ] PPL-agnostic `ir` + `from_pymc` + JSON Schema + provenance + `to_elk`/`to_networkx`
- [ ] Label engine (LaTeX templates + deterministic visitor + token-tree)
- [ ] Layout backend (`dot -Tjson` → `LayoutResult` + param-edge post-pass)
- [ ] Glyph registry + distribution-object adapter (univariate `density` kinds)
- [ ] Shared SVG emitter + static renderer (SVG/PNG/PDF)
- [ ] anywidget widget + thin d3 controller + `view.py` fallback
- [ ] 8-schools example + marimo notebook
- [ ] M0 tests (token-anchor, param/label, parity, port-edge, glyphs, interop, env)

## PyMC construct coverage

### Easy (canonical `name ~ Dist(params)` + shape glyph)
- [ ] Scalar continuous (Normal, HalfNormal, Beta, Gamma, Exponential, StudentT, Uniform, …)
- [ ] Scalar discrete (Bernoulli, Binomial, Poisson, NegativeBinomial, Categorical, …)
- [ ] Dirichlet, Multinomial
- [ ] `pm.Data` / `pm.Minibatch`
- [ ] `pm.Potential` → factor glyph
- [ ] `Flat` / `HalfFlat` → name + prior-linter badge

### Medium
- [ ] `pm.Deterministic` (math-mode rendering with node budget + elision)
- [ ] Transforms (log/logodds/simplex/…) as badges via `rvs_to_transforms`
- [ ] MvNormal / matrix params → `heatmap` glyph
- [ ] Nested / prefixed submodels (group by prefix)
- [ ] `pm.do` / `pm.observe`

### Hard (special-cased)
- [ ] Truncated (`op.base_rv_op`)
- [ ] Censored
- [ ] Mixtures / NormalMixture / ZeroInflated* / Hurdle* (nested components; RV weights)
- [ ] Timeseries: AR, RandomWalk/GaussianRandomWalk, GARCH11, EulerMaruyama (init_dist + scan; sde_fn opaque)
- [ ] LKJCholeskyCov (→ ≤3 nodes + nested `sd_dist`) / LKJCorr / Wishart
- [ ] Missing-data imputation (`_observed` + `_unobserved` + join; `PartialObservedRV`)

### Honest degradation (render what's recoverable + "elided" badge)
- [ ] CustomDist / DensityDist
- [ ] Simulator (SMC)
- [ ] Oversized Deterministic / ODE (DifferentialEquation)
- [ ] GPs (Latent/Marginal/HSGP/TP/Kron) — `_rotated_`/`_hsgp_coeffs_` heuristics

### Scope decision
- [ ] Experimental `pymc.dims` xtensor RVs (`XRV`) — best-effort or declared experimental

## Glyph kinds (registry)
- [ ] `density` (default univariate, the primary mark) · `cdf` · `ccdf` · `histogram` · `gradient` · `dotplot` · `quantile_dotplot` · `band`
- [ ] `interval` / `point` annotations
- [ ] Non-univariate: `kde2d`/`scatter2d`/`contour2d`/`hexbin` · `heatmap`/`corr_ellipses` · `ternary` · `rose`/`polar` · `stem`/`bar`
- [ ] Cross-cutting: `transform.animate="hops"` · `layout="ridgeline"`

## Exporters / interop
- [ ] `to_elk` · `to_networkx`
- [ ] DOT · GraphML · PROV-JSON-LD · (optional) Cytoscape / Hugin
- [ ] Published JSON Schema (draft 2020-12) + validation

## Posterior geometry / diagnostics (M2+)
- [ ] graph-selected divergence joints · auto unconstrained axis (`log τ`) · diagnostic badges (R-hat/ESS/MCSE/divergence-involvement)
- [ ] funnel auto-flag (tiered, hedged) · divergence attribution · energy/BFMI · parallel-coordinates
- [ ] reparameterization suggestions (structural → diagnostic → VIP)
