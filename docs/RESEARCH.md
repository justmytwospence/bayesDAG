# docs/RESEARCH.md — session provenance & annotated bibliography

This package was designed from an extended research session. This file captures the
**sources** and the **decisions they drove** so the context can be reconstructed later
without re-doing the research. The decision *log* is the "Foundational decisions" table in
[the plan](../.claude/plans/please-review-all-the-streamed-storm.md); this file is the
*bibliography*. Grouped by topic; each line = key sources + the takeaway that shaped the design.

- **PyMC internals & coverage** — `pymc/model_graph.py`, `pymc/printing.py`, `pymc/distributions/distribution.py`, `pymc/model/core.py`, `model/fgraph.py`; [`model_to_graphviz` docs](https://www.pymc.io/projects/docs/en/stable/api/model/generated/pymc.model_to_graphviz.html). *Takeaway:* clean `ModelGraph`→render pipeline; `printing.py` already does nested-RV labels (reuse); `SymbolicRandomVariable` introspection (`dist_params`/`signature`/`inner_outputs`/`_print_name`) is the coverage master key; observed-edge reversal + plate-from-dims already exist.
- **PGM-viz literature** — Buntine 1994 (plates, JAIR); Kruschke *DBDA* (shape glyphs, `~`/`=`); Dietz 2010 + [tikz-bayesnet](https://github.com/jluttine/tikz-bayesnet) (directed factor graphs); [daft-pgm](https://docs.daft-pgm.org/); Pyro/NumPyro [`render_model`](https://num.pyro.ai/en/stable/tutorials/model_rendering.html); [causact](https://www.causact.com/); [IPME, Frontiers 2020](https://www.frontiersin.org/articles/10.3389/fcomp.2020.567344/full). *Takeaway:* multi-representation; distributions-on-nodes is the bar; IPME = closest interactive-DAG prior art (marginals only, no geometry).
- **Prior art / PyMC community** — issues [#6716](https://github.com/pymc-devs/pymc/issues/6716) (minimal mode), [#3458](https://github.com/pymc-devs/pymc/issues/3458) (perf hang), [#3413](https://github.com/pymc-devs/pymc/issues/3413) (colon bug); PR [#4159](https://github.com/pymc-devs/pymc/pull/4159); [pykrusch](https://pypi.org/project/pykrusch/); Discourse Kruschke/LaTeX threads. *Takeaway:* real demand; `pymc-extras` is the incubator; params-in-labels + richer `node_formatter` context are upstreamable wins.
- **Stan world & the honesty doctrine** — Carpenter blog [design history (2017-02-28)](https://statmodeling.stat.columbia.edu/2017/02/28/stan-language-design-history/), [abstractions (2017-09-07)](https://statmodeling.stat.columbia.edu/2017/09/07/fundamental-abstractions-underlying-bugs-stan-probabilistic-programming-languages/), [Turing complete (2014-06-12)](https://statmodeling.stat.columbia.edu/2014/06/12/stan-turing-completeo-probabilistic-programming-language/); [Stan User's Guide BUGS-vs-Stan](https://mc-stan.org/docs/stan-users-guide/); [JSS 2017](https://www.jstatsoft.org/v076/i01); Stan Discourse ["auto-generate model diagrams"](https://discourse.mc-stan.org/t/tool-to-auto-generate-model-diagrams/7361) + [stanc3 #177](https://github.com/stan-dev/stanc3/issues/177). *Takeaway:* Stan is imperative, not a DAG → diagrams are fundamentally limited for Stan (the cautionary case); Betancourt's "more danger than benefit" → the representability/honesty contract; also why a `from_stan` adapter is partial-by-nature.
- **Rendering / layout engine** — [Graphviz JSON output](https://graphviz.org/docs/outputs/json/); [elkjs](https://www.npmjs.com/package/elkjs) (+ issues [#142](https://github.com/kieler/elkjs/issues/142)/[#401](https://github.com/eclipse/elk/issues/401); EPL, Node-subprocess only); [mini-racer](https://pypi.org/project/mini-racer/); [MathJax tex2svg](https://docs.mathjax.org/en/v4.0/web/convert.html); [XYFlow](https://github.com/xyflow/xyflow) + [~1k-node ceiling #3003](https://github.com/xyflow/xyflow/discussions/3003); [Cytoscape png drops html labels #2219](https://github.com/cytoscape/cytoscape.js/issues/2219); [Sigma renderers](https://www.sigmajs.org/docs/advanced/renderers/); [d3-dag no-clusters #7](https://github.com/erikbrinkman/d3-dag/issues/7); [d3-zoom](https://github.com/d3/d3-zoom). *Takeaway:* `dot` layout oracle + raw-SVG + thin D3 + render-once MathJax = parity by construction; every framework reintroduces a second box model.
- **IR / interoperability** — [ArviZ InferenceData schema](https://python.arviz.org/en/stable/schema/schema.html); `arviz_base.convert_to_datatree` (duck-typed dispatch); [networkx node-link](https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.node_link_data.html); [JSON Canvas 1.0](https://jsoncanvas.org/spec/1.0/); [ELK JSON format](https://eclipse.dev/elk/documentation/tooldevelopers/graphdatastructure/jsonformat.html); [GraphML primer](http://graphml.graphdrawing.org/primer/graphml-primer.html); [PROV-JSON-LD (W3C 2024)](https://www.w3.org/submissions/2024/SUBM-prov-jsonld-20240825/); pgmpy readers; [Hugin .net](https://download.hugin.com/webdocs/manuals/8.9/htmlhelp/pages/Tutorials/CaseAndData/NetLanguage.html). *Takeaway:* own IR + ArviZ-style PPL-agnostic core + two-substrate (JSON topology / xarray overlays); ELK+networkx adapters; DOT/GraphML/PROV-JSON-LD exporters; discrete-CPT PGM formats don't fit.
- **Uncertainty-viz / glyph design** — [Padilla, Kay & Hullman 2022, *Uncertainty Visualization*](http://space.ucmerced.edu/Downloads/publications/Uncertainty_Visualization_Padilla_Kay_Hullman_2022.pdf); [quantile dotplots, Fernandes et al. CHI 2018](https://idl.uw.edu/papers/uncertainty-bus); HOPs (Hullman et al.); [arviz-plots visuals](https://python.arviz.org/projects/plots/en/latest/) (light naming alignment only); bayesplot (naming cross-check); [Petek et al. 2025, arXiv:2508.00937](https://arxiv.org/html/2508.00937v1) (distribution-as-functional view incl. bivariate/simplex/function-valued). *Takeaway:* shape-first (density is the primary mark); an open glyph-kind registry with `interval`/`point` as optional annotations; non-univariate kinds + HOPs/ridgeline are first-class; **design the glyph vocabulary on its own terms, not pinned to any plotting grammar.**
- **Packaging** — [uv build backend](https://docs.astral.sh/uv/concepts/build-backend/) + [package guide](https://docs.astral.sh/uv/guides/package/) + [deps/extras](https://docs.astral.sh/uv/concepts/projects/dependencies/); [anywidget bundling](https://anywidget.dev/en/bundling/) + [getting started](https://anywidget.dev/en/getting-started/); [hatch-jupyter-builder config](https://hatch-jupyter-builder.readthedocs.io/en/latest/source/get_started/config.html); [create-anywidget](https://github.com/manzt/anywidget/blob/main/packages/create-anywidget/create.js); [marimo anywidget](https://docs.marimo.io/api/inputs/anywidget/). *Takeaway:* `uv` + `hatchling` + `hatch-jupyter-builder` + `esbuild`; Node build-time-only; `mo.ui.anywidget` for marimo.
- **Posterior geometry / diagnostics** — Betancourt ["Diagnosing Biased Inference with Divergences"](https://betanalpha.github.io/assets/case_studies/divergences_and_bias.html) ([PyMC port](https://www.pymc.io/projects/examples/en/latest/diagnostics_and_criticism/Diagnosing_biased_Inference_with_Divergences.html)); Betancourt & Girolami 2015; Neal's funnel; ArviZ [`plot_pair`](https://python.arviz.org/en/stable/api/generated/arviz.plot_pair.html)/`plot_parallel`/`plot_energy`/`bfmi`/`ess`; bayesplot `mcmc_pairs`/`mcmc_parcoord`; [Gorinova et al. 2020, arXiv:1906.03028](https://arxiv.org/abs/1906.03028) + [`pymc_extras…vip_reparametrize`](https://www.pymc.io/projects/extras/en/stable/generated/pymc_extras.model.transforms.autoreparam.vip_reparametrize.html); ArviZ `unconstrained_posterior` group; [Mosaic, TVCG 2024](https://idl.cs.washington.edu/files/2024-Mosaic-TVCG.pdf). *Takeaway:* funnels are joint + live in unconstrained space; structure-aware pair-selection + auto-`log(τ)` axis is the wedge; VIP gives reparameterization suggestions; Mosaic only if SPLOM brushing must scale.

## Decision log — full distribution coverage (2026-06)

Goal: every PyMC distribution renders correctly + honestly. Co-designed per-family representations
(the agreed table) and findings verified by introspecting the installed **PyMC 6.0.1** (not docs/training).

- **Verified PyMC-6.0.1 facts that shaped the implementation** (introspection, not assumption):
  - Detect by **RV class** (`type(op).__name__`), not the op print-name: the print-name aliases/collapses —
    `MvNormal→"MultivariateNormal"` (so the old `DIST_SYMBOLS["MvNormal"]` was a dead key), `LKJCholeskyCov→"_lkjcholeskycov"`,
    `ChiSquared→"Gamma"`, `PolyaGamma→"PG"`; and `NormalMixture`/`ZeroInflated*`→`MixtureRV`, `GaussianRandomWalk`→`RandomWalkRV`,
    `OrderedLogistic/Probit`→`CategoricalRV` (these distinctions are lost at the op level).
  - **Sub-RVs/params are reachable via `var.owner.inputs`** (per-construct layout): Censored `[base, lo, hi]`;
    generic Truncated `op.base_rv_op`; RandomWalk `[init_dist, innovation_dist, …]`; AR `[rho, sigma, init_dist, …]`;
    Mixture inputs carry the component RVs (a `DiracDelta` component ⇒ zero-inflation). Filter to actual RVs
    (`isinstance` RandomVariable/SymbolicRandomVariable) — `MakeVector` weight ops otherwise leak in.
  - **The op reparametrizes** several families, so analytic shapes must match `logp`, not the public kwargs:
    Exponential/Gamma expose *scale* (the old code double-inverted → latent bugs), HalfCauchy a single scale param.
    Every PyMC→scipy translation is locked by a **logp-matching test** (`test_shapes.py`) — `pm.draw` is *not*
    a safe oracle (it samples Cauchy as `loc=α/β, scale=1/β`; the density via `logp` is the truth).
  - **Robustness:** `Flat`/`HalfFlat`/`ICAR` raise on shape `.eval()` and crashed `to_ir` via pymc's `get_plates`;
    guarded with an eval-free plate fallback.
  - **`.eval()` silently samples through parent RVs** (the second face of the `pm.draw` hazard): `_numeric_params`
    used to `.eval()` every distribution param, which *succeeds* for a conditional latent by drawing a random value
    through its parents. So `mu ~ N(0,5)` (root prior) and `x ~ N(mu, sigma)` (conditional latent) both came back
    `prior_analytic` and rendered as identical green densities — and `x`'s "shape" jumped randomly every build
    (centre −14→+6 across renders), breaking the prior-vs-latent distinction, the honesty contract, and determinism
    (static ≠ static, let alone static == widget). Fix: gate every param `.eval()` on `_depends_on_rv` — a **root
    prior** (all params fixed constants) renders the analytic shape; a **conditional latent** (any param governed by a
    parent RV) falls to the family-only grey schematic, deterministically. Constructs that need only a numeric
    *subset* read it via `_lead_numeric` so a legitimate prior sub-param doesn't nuke a knowable shape: LKJCholeskyCov's
    correlation marginal needs only `n, eta` (not its `sd_dist` prior); a **driftless** random walk's normalized fan is
    scale-invariant (the prior innovation `sigma` cancels). Only a shape that *genuinely* depends on a prior — an
    MvNormal whose covariance is the LKJ draw — honestly badges rather than fabricating a random pairplot.
  - **Ordinal caveat:** `OrderedLogistic` is a `CategoricalRV` with a computed `p` — not reliably distinguishable
    from a plain Categorical at the op level, so the cutpoints glyph is best-effort.
- **Representations** (`adapters/constructs.py`): multivariate→pairplot(ellipses)/heatmap, Dirichlet→simplex
  (marginal Beta), Censored→base+mass-spikes, TruncatedNormal→clipped density, RandomWalk→fan chart,
  AR→stationary marginal, Interpolated→density-from-points, CAR/ICAR→adjacency heatmap, mixtures/zero-inflated→composite;
  GARCH/SDE/LKJ/CustomDist/Simulator/Flat→honest `elision_reason` badge (wires the long-dormant `representable` field).
- **Showcase** (`examples/zoo.py`): canonical models (stochastic volatility, Weibull survival, zero-inflated counts,
  correlated slopes, ordinal ratings, Gaussian mixture, disease mapping, AR, + logistic regression, Poisson log-link,
  softmax classifier) chosen so the set exercises the rich glyphs with *recognizable* models, not contrived zoos; the
  long tail is guaranteed by the all-catalog `test_coverage.py`.

## Decision log — deterministic transfer-function glyphs (2026-06)

A `pm.Deterministic("y", f(parents))` now gets a small **canonical glyph of the function `f` itself** — a
logistic S-curve for `invlogit`, an exponential for `exp`, a line for an affine predictor, bars for `softmax`.
The design constraint (user's) was **zero false positives, argued from first principles, not from test examples**:

- **A glyph is emitted only when its shape is a *mathematical consequence* of the op graph.** (1) A transfer
  curve `T` is drawn only when, after stripping value-preserving wrappers (ViewOp/DimShuffle/Identity; a
  `float→int` Cast is NOT value-preserving) and **scalar-constant** affine framing (sign-tracked; this also reaches
  `Erf` inside the standard probit `0.5(1+erf(·))`), the core op IS an `Elemwise{T}` — elementwise application
  *means* `yᵢ=T(zᵢ)` pointwise, so the shape is true by definition, and the framing is shape-preserving by the
  normalization identity `normalize(c·T+d) ≡ normalize(±T)`. (2) A **line** is drawn only when `is_affine` holds —
  a whitelist closed under the degree-preserving ops with a `Mul`/`Dot` "≤1 parent-dependent factor" guard, so by
  structural induction the expression is degree ≤1 in the parent RVs (rejects `tau*eta`, `a*(θ−b)` — bilinear).
  (3) Everything else (`TrueDiv` ⇒ manual-sigmoid/reciprocal/mean ambiguity, reductions, gather, non-constant
  `Pow`, vector framing, fused `Composite`, unknown ops) falls through to **skip**. The rule is a conservative
  whitelist, so the only failure mode is a false *negative*; a false positive is structurally impossible.
- **First-principles analysis caught what an example sweep didn't:** `cumsum` was dropped (a staircase is only
  monotone for non-negative summands — a real FP on signed values); curves must be evaluated from the true
  function (not hand-stylized) so the drawn shape *is* `T`. Verified pytensor op signatures: `invlogit→Elemwise/Sigmoid`,
  `softplus→Softplus`, `erf→Erf`, `x**k→Pow`, `softmax→Softmax`, and `1/x`/manual-sigmoid/`mean` all `→TrueDiv`.
- **Leaf set = the `named` model vars** (RVs + deterministics + data), mirroring `pytensor_latex.render_value` — so a
  linear predictor referencing *other* deterministics (mrp `p`) resolves to a line, not over-descended.
- **Determinism/parity:** curves are parameter-free (fixed grid + closed form), never `.eval()`-ing a parent-bearing
  tensor — identical on every render (the same contract as the `_canonical_bell` conditional-latent schematic).
- **Edge anchoring:** a glyph-bearing deterministic exits its outgoing edge from the node box (below the glyph), like
  a distribution-glyph node; an equation-only deterministic keeps the LHS-token exit. Geometry reserves the strip by
  glyph **presence**, not role.
