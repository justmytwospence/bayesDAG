# docs/RESEARCH.md — session provenance & annotated bibliography

This package was designed from an extended research session. This file captures the
**sources** and the **decisions they drove** so the context can be reconstructed later
without re-doing the research — the annotated *bibliography* of that work. Grouped by topic;
each line = key sources + the takeaway that shaped the design.

- **PyMC internals & coverage** — `pymc/model_graph.py`, `pymc/printing.py`, `pymc/distributions/distribution.py`, `pymc/model/core.py`, `model/fgraph.py`; [`model_to_graphviz` docs](https://www.pymc.io/projects/docs/en/stable/api/model/generated/pymc.model_to_graphviz.html). *Takeaway:* clean `ModelGraph`→render pipeline; `printing.py` already does nested-RV labels (reuse); `SymbolicRandomVariable` introspection (`dist_params`/`signature`/`inner_outputs`/`_print_name`) is the coverage master key; observed-edge reversal + plate-from-dims already exist.
- **PGM-viz literature** — Buntine 1994 (plates, JAIR); Kruschke *DBDA* (shape glyphs, `~`/`=`); Dietz 2010 + [tikz-bayesnet](https://github.com/jluttine/tikz-bayesnet) (directed factor graphs); [daft-pgm](https://docs.daft-pgm.org/); Pyro/NumPyro [`render_model`](https://num.pyro.ai/en/stable/tutorials/model_rendering.html); [causact](https://www.causact.com/); [IPME, Frontiers 2020](https://www.frontiersin.org/articles/10.3389/fcomp.2020.567344/full). *Takeaway:* multi-representation; distributions-on-nodes is the bar; IPME = closest interactive-DAG prior art (marginals only, no geometry).
- **Prior art / PyMC community** — issues [#6716](https://github.com/pymc-devs/pymc/issues/6716) (minimal mode), [#3458](https://github.com/pymc-devs/pymc/issues/3458) (perf hang), [#3413](https://github.com/pymc-devs/pymc/issues/3413) (colon bug); PR [#4159](https://github.com/pymc-devs/pymc/pull/4159); [pykrusch](https://pypi.org/project/pykrusch/); Discourse Kruschke/LaTeX threads. *Takeaway:* real demand; `pymc-extras` is the incubator; params-in-labels + richer `node_formatter` context are upstreamable wins.
- **Stan world & the honesty doctrine** — Carpenter blog [design history (2017-02-28)](https://statmodeling.stat.columbia.edu/2017/02/28/stan-language-design-history/), [abstractions (2017-09-07)](https://statmodeling.stat.columbia.edu/2017/09/07/fundamental-abstractions-underlying-bugs-stan-probabilistic-programming-languages/), [Turing complete (2014-06-12)](https://statmodeling.stat.columbia.edu/2014/06/12/stan-turing-completeo-probabilistic-programming-language/); [Stan User's Guide BUGS-vs-Stan](https://mc-stan.org/docs/stan-users-guide/); [JSS 2017](https://www.jstatsoft.org/v076/i01); Stan Discourse ["auto-generate model diagrams"](https://discourse.mc-stan.org/t/tool-to-auto-generate-model-diagrams/7361) + [stanc3 #177](https://github.com/stan-dev/stanc3/issues/177). *Takeaway:* Stan is imperative, not a DAG → diagrams are fundamentally limited for Stan (the cautionary case); Betancourt's "more danger than benefit" → the representability/honesty contract; also why a `from_stan` adapter is partial-by-nature.
- **Rendering / layout engine** — [Graphviz JSON output](https://graphviz.org/docs/outputs/json/); [elkjs](https://www.npmjs.com/package/elkjs) (+ issues [#142](https://github.com/kieler/elkjs/issues/142)/[#401](https://github.com/eclipse/elk/issues/401); EPL, Node-subprocess only); [mini-racer](https://pypi.org/project/mini-racer/); [MathJax tex2svg](https://docs.mathjax.org/en/v4.0/web/convert.html); [XYFlow](https://github.com/xyflow/xyflow) + [~1k-node ceiling #3003](https://github.com/xyflow/xyflow/discussions/3003); [Cytoscape png drops html labels #2219](https://github.com/cytoscape/cytoscape.js/issues/2219); [Sigma renderers](https://www.sigmajs.org/docs/advanced/renderers/); [d3-dag no-clusters #7](https://github.com/erikbrinkman/d3-dag/issues/7); [d3-zoom](https://github.com/d3/d3-zoom). *Takeaway:* `dot` layout oracle + raw-SVG + thin D3 + render-once MathJax = parity by construction; every framework reintroduces a second box model.
- **IR / interoperability** — [ArviZ InferenceData schema](https://python.arviz.org/en/stable/schema/schema.html); `arviz_base.convert_to_datatree` (duck-typed dispatch); [networkx node-link](https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.node_link_data.html); [JSON Canvas 1.0](https://jsoncanvas.org/spec/1.0/); [ELK JSON format](https://eclipse.dev/elk/documentation/tooldevelopers/graphdatastructure/jsonformat.html); [GraphML primer](http://graphml.graphdrawing.org/primer/graphml-primer.html); [PROV-JSON-LD (W3C 2024)](https://www.w3.org/submissions/2024/SUBM-prov-jsonld-20240825/); pgmpy readers; [Hugin .net](https://download.hugin.com/webdocs/manuals/8.9/htmlhelp/pages/Tutorials/CaseAndData/NetLanguage.html). *Takeaway:* own IR + ArviZ-style PPL-agnostic core + two-substrate (JSON topology / xarray overlays); ELK+networkx adapters; DOT/GraphML/PROV-JSON-LD exporters; discrete-CPT PGM formats don't fit.
- **Uncertainty-viz / glyph design** — [Padilla, Kay & Hullman 2022, *Uncertainty Visualization*](http://space.ucmerced.edu/Downloads/publications/Uncertainty_Visualization_Padilla_Kay_Hullman_2022.pdf); [quantile dotplots, Fernandes et al. CHI 2018](https://idl.uw.edu/papers/uncertainty-bus); HOPs (Hullman et al.); [arviz-plots visuals](https://python.arviz.org/projects/plots/en/latest/) (light naming alignment only); bayesplot (naming cross-check); [Petek et al. 2025, arXiv:2508.00937](https://arxiv.org/html/2508.00937v1) (distribution-as-functional view incl. bivariate/simplex/function-valued). *Takeaway:* shape-first (density is the primary mark); an open glyph-kind registry with `interval`/`point` as optional annotations; non-univariate kinds + HOPs/ridgeline are first-class; **design the glyph vocabulary on its own terms, not pinned to any plotting grammar.**
- **SEM plotting ecosystem (the comparison class)** — [lavaan](https://lavaan.ugent.be/) model syntax + parameter table; [semPlot](https://github.com/SachaEpskamp/semPlot) (`semPlotModel@Pars`) + [semptools](https://sfcheung.github.io/semptools/); [lavaanPlot](https://github.com/alishinski/lavaanPlot); [tidySEM plotting](https://cjvanlissa.github.io/tidySEM/articles/Plotting_graphs.html); [lavaangui](https://www.tandfonline.com/doi/full/10.1080/10705511.2024.2420678); [blavaan](https://blavaan.org/) (Merkle & Rosseel, JSS 2018); [Hunter 2019, *Beyond path diagrams*](https://pmc.ncbi.nlm.nih.gov/articles/PMC6760248/); [pathmc](https://pathmc.pymc-labs.com/) (PyMC Labs). *Takeaways:* **semPlot's `@Pars` keeps `est`/`std` and no `se`/CI — uncertainty is destroyed at the IR layer, so nothing downstream can ever draw it, and `semPaths(blavaan_fit)` silently renders posterior means as MLEs** (the cautionary case that motivated retaining the idata handle on the view); lavaan auto-names every parameter `lhs op rhs` and that one string joins summary/plots/exports/constraints — the canonical-id lesson; lavaanPlot silently omits variances/residuals and semPlot silently drops the mean structure, so *naming* an omission is a differentiator, not table stakes; blavaan prints the `Prior` column beside `Estimate`/`Post.SD`/`pi.lower`/`pi.upper`/`Rhat` (a textual prior→posterior overlay), turns fit indices and modification indices into posteriors, and argues explicitly against cutoffs; lavaangui splits interactive layout from `export_plot()` replay (the reproducibility/customization tradeoff, solved); tidySEM stores layout as an editable data frame; Hunter 2019 shows three datasets with identical r = 0.65 and identical fit indices whose relationships are linear, curvilinear and discontinuous — invisible in a path diagram, which is the shape-first argument made by SEM methodologists about their own field. pathmc compiles a lavaan-style DSL into a real `pm.Model` (`.pymc_model` is public), but its `graph()` is structure-only and identical before and after fitting, and its diagnostics are a one-line `az.summary` passthrough — verified that `bayesdag.view(m.pymc_model, idata=…)` renders it with no adapter code. Its colored ✓/✗/⚠ verdicts sit alongside carefully hedged docstrings: **hedging has to live on the rendered surface or it does not exist.**
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
  tensor — identical on every render. (The `_canonical_bell` conditional-latent schematic that this line once cited was
  removed in 2026-09: one bell for every family was itself a shape the model did not say.)
- **Edge anchoring:** a glyph-bearing deterministic exits its outgoing edge from the node box (below the glyph), like
  a distribution-glyph node; an equation-only deterministic keeps the LHS-token exit. Geometry reserves the strip by
  glyph **presence**, not role.

## Decision log — research review (2026-08)

A comparative review against the SEM plotting ecosystem (lavaan/semPlot/blavaan/lavaangui/tidySEM)
and PyMC Labs' pathmc. Both bodies of prior art were read against this codebase, and the findings
that changed code are recorded here with the ones that were deliberately refused.

- **Retain the `InferenceData` handle on the view** (semPlot's lesson, applied before it bit us).
  The IR ships only curves — a peak-normalized grid per node — because it is JSON-serializable
  against a published schema, and `OverlayRef` is the pointer back to the draws. But the view
  discarded the idata, so that pointer had nothing to dereference. semPlot made the same choice one
  layer down and cannot undo it: `@Pars` has no `se`, so no renderer in that ecosystem can show
  uncertainty and blavaan posteriors get drawn as MLEs. Keeping the handle (never serialized, never
  in the IR) is what makes M2's interval/ROPE annotations possible at all. It also let
  `view(prebuilt_ir, idata=…)` stop warning and start working, since `update()` is keyed purely on
  node ids.
- **Canonical per-node token ids** (lavaan's lesson, plus a real defect). lavaan's auto-derived
  `lhs op rhs` name is the join key for its whole ecosystem; bayesdag had the node half (node id ==
  constrained idata variable name) but every label reused the same `tok-loc`/`tok-scale`/`tok-__lhs__`
  ids, so the composed SVG had duplicate element ids — nine of them in the committed hero image.
  Namespaced to `bd-<slug>-tok-<token>` at embed time, never in the TeX (that would destroy the math
  cache's cross-node hits). Edge identity stays the `(source, target)` tuple: any joined string
  re-imports the `pm.Normal("a|b")` collision the tuple-key refactor removed.
- **REFUSED**, each against the honesty contract:
  - *Significance stars* (`*`/`**`/`***`) — a continuous quantity flattened to three levels at
    arbitrary thresholds. The most entrenched convention in SEM diagrams; blavaan argues against it too.
  - *p-filtered edges* (lavaanPlot's `sig=.05` deletes non-significant paths) — draws a model that is
    not the model.
  - *Standardized coefficients* — SEM standardizes by observed SDs; PyMC parameters have no canonical
    standardization, so there is no honest `Std.all` to compute. Expect the request; decline it.
  - *Point estimates on edges* — inverts the thesis (node carries the distribution, edge carries
    structure) and is exactly the compression Hunter 2019 dismantles.
  - *Causal vocabulary* — a `pm.Model` is not a causal DAG. Its arrows are conditional-dependence
    arrows in a factorization and carry no interventional meaning. pathmc can make causal claims only
    because its user hand-wrote structural assumptions in a DSL; bayesdag recovers the graph from a
    model object and has no access to those. Decorating edges causally would state something the model
    does not say.
  - *CFI/RMSEA/TLI analogues* — need a saturated model, a baseline model and a model-implied covariance
    matrix. A general PyMC model has none. The transferable idea is "fit summaries are distributions
    with no cutoffs", not the indices.
- **Deferred to roadmap:** layout persistence in lavaangui's shape — arrange the diagram in the widget,
  replay it byte-identically from the static renderer. Most-wanted feature in that ecosystem (semptools
  and tidySEM both exist largely because semPlot lacks it), and bayesdag's one-layout-pass/two-emitters
  design is most of the way there already.

## Decision log — audit: honest glyphs and recognized structures (2026-09)

An audit from the perspective of Bayesian workflow (Gelman et al. 2020; Gabry et al. 2019,
*Visualization in Bayesian workflow*) found places where the diagram stated things the model does
not say, and no vocabulary for the compound structures a Bayesian reads a model by. Three product
decisions were confirmed by the owner: **overview by default**, a **user annotation API**, and
**exact shapes only**.

- **Exact shapes only.** A conditional latent used to get one generic bell whatever its family — a
  Gamma, a HalfStudentT and a Uniform all looked the same. Now: when parents set only location and
  scale, the family's standardized curve IS the shape (exact); when a parent sets a shape parameter
  (a Gamma's alpha, a LogNormal's sigma), no curve is drawn and a caption names the slot. A
  representative member (Gamma with alpha = 2) was considered and rejected: a picture of a shape
  the model does not have. The same rule removed three other invented marks: softmax's equal bars,
  BART's fixed step heights, and the AR "order" stems of height 0.7.
- **No fitted-family overlay on observed data.** The MLE curve fitted the family to the data's
  MARGINAL, but the likelihood is conditional: the marginal of `N(mu_i, sigma)` regression data is a
  mixture, so "a Normal fits poorly" said nothing about the model. The honest comparison is the
  predictive check, and it is now drawn from the caller's own `posterior_predictive` /
  `prior_predictive` draws (Gabry et al. 2019). Legacy IR keeps rendering, captioned as legacy.
- **Vectors are never pooled.** One KDE over all of `theta[school]`'s draws is no school's
  posterior and hides exactly the shrinkage a hierarchy is about. ≤ 8 elements: one curve each;
  more: a forest of per-element 50/90% intervals sorted by median (an unlabelled shape summary);
  the card lists labelled intervals in coordinate order — never ranked by apparent significance
  (bayesplot's `mcmc_intervals` is the reference presentation).
- **Recognized structures ("blocks").** Rules are structural whitelists on the PyTensor graph with
  stated proof obligations; nothing is recognized by name, TeX or samples, and a false negative is
  always preferred to a false claim.
  - *Non-centered:* `D = L + S*z` with `z` a symmetric standard variable nothing else consumes is
    EXACTLY `D ~ Family(L, |S|)` (Papaspiliopoulos, Roberts & Sköld 2007; the equivalence VIP
    exploits, Gorinova et al. 2020). `exp`/`invlogit` wrappers give LogNormal/LogitNormal — whose
    shape depends on the (random) scale, so the exact-shape rule draws no curve for them.
  - *Centered:* a plated latent with a location/scale parent over strictly coarser dims (Gelman &
    Hill 2007). A shared truncation bound is not pooling.
  - *Shrinkage:* the scale must factor into one global and one local positive scale, optionally
    times the Piironen & Vehtari (2017) slab `c^2 lambda^2 / (c^2 + tau^2 lambda^2)`, matched on the
    op tree modulo commutativity and sqr/pow. "Horseshoe" is reserved for half-Cauchy locals
    (Carvalho, Polson & Scott 2010); half-t(nu) locals are named as the generalization they are.
    Ridge (one shared scale) and spike-and-slab (a Bernoulli indicator) are rejected by construction.
    A plain horseshoe's glyph is its exact marginal given the global scale, integrated numerically,
    with the pole at 0 clipped and captioned. The slab bounds the conditional SCALE, not the
    coefficient, and the card says so. No inclusion probabilities or "effective number of
    proteins" are invented; a model-defined effective-size deterministic stays visible as itself.
  - *Regression:* an observed likelihood whose location is, through one recognized link, affine
    with a data × parameter term. The LogNormal note — `exp(mu)` is the conditional median, the mean
    is `exp(mu + sigma^2 / 2)` — is there because that confusion is common.
  - *Measured latent:* a latent read by a noisy observation and used elsewhere (Stan User's Guide,
    measurement error). Anything richer — a likelihood written as Potentials, a two-stage workflow —
    is declared with `BlockSpec` and drawn as annotated: the user's statement, never verified.
- **Overview by default.** Blocks that own variables (a non-centered `z`, raw coefficients) contract
  into one box carrying the implied distribution and the member equations; population-level
  hyperparameters stay visible as boundary inputs. Contraction must be path-convex and keep the
  quotient acyclic, and every crossing edge keeps the canonical edges it stands for (`EdgeIR.via`).
  The projection is presentation only: exporters, `to_json` and `selected_node` see the canonical IR.
- **Parameterization hints are structural AND diagnostic.** A centered hierarchy's card suggests
  (hedged) a non-centered form only when the run diverged, and a non-centered one's the converse —
  the right parameterization depends on how informative the data are about each group (Betancourt
  & Girolami 2015; the Stan reparameterization chapter). Stan's pedantic mode is the precedent for a
  structural lint that stays advisory.
- **REFUSED / deferred:** a non-centered "collapse" transform that rewrites the model (the overview
  shows the implied form without pretending the model was written that way); block hulls in the full
  graph (ELK cannot express plate-crossing groups honestly); significance ranking of forest rows;
  Playwright/shadow-DOM (the JS queries only inside its own element; node:test + jsdom cover the
  controller); inferring statistical nesting from dims (plates mean "indexed by" — a
  `(protein, batch)` array nests under `protein` as plate notation, nothing more).

## Decision log — fitted-structure panels (2026-09)

Each recognized structure now offers the pictures its literature uses, once a posterior is
attached: the Efron–Morris shrinkage plot (Efron & Morris 1977; Gelman & Hill 2007, ch. 12), the
population distribution with a new group's posterior predictive, Gelman & Pardoe's (2006) pooling
factor, the radon "lines per group" figure (complete / no / partial pooling), varying intercepts
vs slopes, the fitted effect on the outcome scale, the horseshoe's shrinkage factors κ and local
scale vs coefficient (Piironen & Vehtari 2017), and readings vs inferred truth.

- **Raw estimates are exact bookkeeping, not a second model.** A group's raw estimate is its data
  mean minus the rest of the predictor evaluated at posterior means; because the regression rule
  proved the predictor affine in its leaves, that equals the rest's posterior mean exactly.
- **Only like-for-like comparisons.** Raw group summaries are drawn only on the predictor's own
  scale: an identity link, or log for a LogNormal. Under a logit or log link the panel refuses
  with its reason rather than comparing group proportions to a linear-predictor effect.
- **Average predictive comparisons, not an invented mean observation.** Effect curves and
  per-group lines vary ONE covariate over every observed row and average (Gelman & Pardoe 2007).
  This also works when the predictor carries unnamed per-row constants (a fixed matrix of
  first-stage summaries), where "hold everything else at its mean" has nothing to hold.
- **κ is labelled for what it is:** exact for an orthogonal design, the standard approximation
  otherwise, with the effective number of coefficients Σ(1 − κ_j) as a summary; heteroskedastic
  likelihoods use each row's own σ_i.
- **No selection by significance, no inclusion probabilities,** and the intercept–slope panel's
  correlation of posterior means is captioned as descriptive (the model may estimate none).
