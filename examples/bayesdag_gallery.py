import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # bayesdag

    Shape-first, posterior-aware, interactive visualization of PyMC models. Run from the
    project root: `uv run marimo edit examples/bayesdag_gallery.py`.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pymc as pm

    import bayesdag

    return bayesdag, np, pm


@app.cell
def _(mo, pm):
    def graphviz_fit(model):
        """PyMC's `model_to_graphviz` for a model, rendered as an SVG that shrinks to fit the cell
        width. Degrades to a note when PyMC can't render it — e.g. it can't sample an `ICAR`/`Flat`
        to infer shapes — which bayesdag handles regardless."""
        try:
            svg = pm.model_to_graphviz(model).pipe(format="svg").decode()
        except Exception as e:
            return mo.md(
                f"*(PyMC's `model_to_graphviz` can't render this model — `{type(e).__name__}`: "
                f"{e}. bayesdag renders it above.)*"
            )
        return mo.Html(svg.replace("<svg ", '<svg style="max-width:100%;height:auto;" ', 1))

    return (graphviz_fit,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Eight schools

    Non-centered: hyperpriors `mu`, `tau`; a deterministic `theta = mu + tau*eta`; an observed
    Normal likelihood over a `school` plate.
    """)
    return


@app.cell
def _(np, pm):
    es_y = np.array([28.0, 8, -3, 7, -1, 1, 18, 12])
    es_sigma = np.array([15.0, 10, 16, 11, 9, 11, 10, 18])
    with pm.Model(coords={"school": [f"S{i}" for i in range(8)]}) as es_model:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfNormal("tau", 5)
        eta = pm.Normal("eta", 0, 1, dims="school")
        theta = pm.Deterministic("theta", mu + tau * eta, dims="school")
        pm.Normal("y_obs", theta, es_sigma, observed=es_y, dims="school")
    return (es_model,)


@app.cell
def _(mo):
    mo.md(r"""
    ### The diagram — hover and click

    Each node shows its distribution's **shape** (note `tau`'s half-normal starting at 0); the
    deterministic is real math; the observed node is a **histogram**; arrows from `mu`/`tau`/`eta`
    land on those exact tokens inside `theta = mu + tau*eta` (port-level edges). The interactive
    view drops the legend (hover shows the same info); the static `save(...)` figure keeps it.

    - **Hover** a node: everything outside its Markov blanket fades; a tooltip shows the distribution.
    - **Click** a node for a detail card (distribution, parameters, dims, a copyable `pm.*` line).
    - **Click the `school` plate's border or label** to expand its **prior predictive check** — the 8 per-school
      curves overlaid (θ spreads via the shared μ/τ; η's coincide → exchangeability; y_obs shows
      the prior-predictive vs. the observed data as orange ticks).
    """)
    return


@app.cell
def _(bayesdag, es_model, mo):
    es_view = bayesdag.view(es_model)
    mo.ui.anywidget(es_view.widget())
    return (es_view,)


@app.cell
def _(es_model, graphviz_fit, mo):
    mo.vstack(
        [
            mo.md("**Baseline — PyMC's built-in `model_to_graphviz` for the same model:**"),
            graphviz_fit(es_model),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Fit, then posterior overlays (glyphs turn orange)
    """)
    return


@app.cell
def _(es_model, pm):
    with es_model:
        es_idata = pm.sample(draws=300, tune=300, chains=2, random_seed=0, progressbar=False)
    return (es_idata,)


@app.cell
def _(bayesdag, es_idata, es_model, mo):
    mo.ui.anywidget(bayesdag.view(es_model, idata=es_idata).widget())
    return


@app.cell
def _(es_view, mo):
    # This cell IS the regeneration path for the README's hero image. Anchored to the notebook
    # directory so it always lands in examples/, whatever the working directory happens to be.
    es_view.save(mo.notebook_dir() / "eight_schools.svg")  # publication SVG (PNG/PDF via [export])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Radon

    A varying-intercept regression: per-county intercepts `a[county]` from hyperpriors `mu_a`,
    `sigma_a`; a shared floor effect `b`; observed `y`. Two plates (`county`, `obs`) + data nodes
    make this busy enough that hovering to isolate a node — and expanding a plate — really helps.
    """)
    return


@app.cell
def _(np, pm):
    rng = np.random.default_rng(0)
    n_counties, n_obs = 6, 40
    county_idx = rng.integers(0, n_counties, n_obs)
    floor = rng.integers(0, 2, n_obs).astype(float)
    a_true = rng.normal(1.2, 0.5, n_counties)
    radon_log = a_true[county_idx] - 0.6 * floor + rng.normal(0, 0.4, n_obs)
    with pm.Model(
        coords={"county": [f"C{i}" for i in range(n_counties)], "obs": np.arange(n_obs)}
    ) as radon_model:
        mu_a = pm.Normal("mu_a", 0, 5)
        sigma_a = pm.HalfNormal("sigma_a", 5)
        a = pm.Normal("a", mu_a, sigma_a, dims="county")
        b = pm.Normal("b", 0, 5)
        sigma = pm.HalfNormal("sigma", 1)
        cidx = pm.Data("county_idx", county_idx, dims="obs")
        fl = pm.Data("floor", floor, dims="obs")
        radon_mu = pm.Deterministic("mu", a[cidx] + b * fl, dims="obs")
        pm.Normal("y", radon_mu, sigma, observed=radon_log, dims="obs")
    return (radon_model,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Interactive

    Hover `a` to see it depends on `mu_a`/`sigma_a` and feeds `mu`. Click the `county` plate for
    the prior-predictive spread of the 6 intercepts; click the `obs` plate for prior-predictive
    `y` vs. the observed log-radon.
    """)
    return


@app.cell
def _(bayesdag, mo, radon_model):
    radon_view = bayesdag.view(radon_model)
    mo.ui.anywidget(radon_view.widget())
    return


@app.cell
def _(graphviz_fit, mo, radon_model):
    mo.vstack(
        [
            mo.md("**Baseline — PyMC `model_to_graphviz`:**"),
            graphviz_fit(radon_model),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Fit, then posterior overlays
    """)
    return


@app.cell
def _(pm, radon_model):
    with radon_model:
        radon_idata = pm.sample(draws=300, tune=300, chains=2, random_seed=0, progressbar=False)
    return (radon_idata,)


@app.cell
def _(bayesdag, mo, radon_idata, radon_model):
    mo.ui.anywidget(bayesdag.view(radon_model, idata=radon_idata).widget())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Hierarchical regression

    A varying intercept `a[county]` plus several fixed coefficients `b1,b2,b3`, all converging on
    one deterministic `mu = a[county_idx] + b1*x1 + b2*x2 + b3*x3`. Watch how the free coefficients
    are placed in the equation's token order so their arrows don't cross.
    """)
    return


@app.cell
def _(np, pm):
    hr_rng = np.random.default_rng(1)
    hr_nc, hr_no = 6, 60
    hr_cidx = hr_rng.integers(0, hr_nc, hr_no)
    hr_x1, hr_x2, hr_x3 = (hr_rng.normal(size=hr_no) for _ in range(3))
    hr_at = hr_rng.normal(0, 1, hr_nc)
    hr_y = hr_at[hr_cidx] + 0.5 * hr_x1 - 0.3 * hr_x2 + 0.2 * hr_x3 + hr_rng.normal(0, 0.3, hr_no)
    with pm.Model(coords={"county": np.arange(hr_nc), "obs": np.arange(hr_no)}) as hr_model:
        hr_mu_a = pm.Normal("mu_a", 0, 5)
        hr_sa = pm.HalfNormal("sigma_a", 5)
        hr_a = pm.Normal("a", hr_mu_a, hr_sa, dims="county")
        hr_b1 = pm.Normal("b1", 0, 5)
        hr_b2 = pm.Normal("b2", 0, 5)
        hr_b3 = pm.Normal("b3", 0, 5)
        hr_s = pm.HalfNormal("sigma", 1)
        hr_cc = pm.Data("county_idx", hr_cidx, dims="obs")
        hr_X1 = pm.Data("x1", hr_x1, dims="obs")
        hr_X2 = pm.Data("x2", hr_x2, dims="obs")
        hr_X3 = pm.Data("x3", hr_x3, dims="obs")
        pm.Deterministic(
            "mu", hr_a[hr_cc] + hr_b1 * hr_X1 + hr_b2 * hr_X2 + hr_b3 * hr_X3, dims="obs"
        )
        pm.Normal("y", hr_model.named_vars["mu"], hr_s, observed=hr_y, dims="obs")
    return (hr_model,)


@app.cell
def _(bayesdag, hr_model, mo):
    mo.ui.anywidget(bayesdag.view(hr_model).widget())
    return


@app.cell
def _(graphviz_fit, hr_model, mo):
    mo.vstack(
        [
            mo.md("**Baseline — PyMC `model_to_graphviz`:**"),
            graphviz_fit(hr_model),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · IRT (2-parameter item response theory)

    Three crossed plates — `student`, `item`, `obs` — with a deterministic logit
    `eta = a[item]*(theta[student] - b[item])` combining three plated parents via crossed indexing.
    A canonical psychometrics model; an order more structurally complex than radon.
    """)
    return


@app.cell
def _(np, pm):
    irt_rng = np.random.default_rng(2)
    irt_ns, irt_ni = 20, 10
    irt_no = irt_ns * irt_ni
    irt_si = np.repeat(np.arange(irt_ns), irt_ni)
    irt_ii = np.tile(np.arange(irt_ni), irt_ns)
    irt_th = irt_rng.normal(0, 1, irt_ns)
    irt_at = irt_rng.lognormal(0, 0.3, irt_ni)
    irt_bt = irt_rng.normal(0, 1, irt_ni)
    irt_obs = irt_rng.binomial(
        1, 1 / (1 + np.exp(-(irt_at[irt_ii] * (irt_th[irt_si] - irt_bt[irt_ii]))))
    )
    with pm.Model(
        coords={"student": np.arange(irt_ns), "item": np.arange(irt_ni), "obs": np.arange(irt_no)}
    ) as irt_model:
        irt_theta = pm.Normal("theta", 0, 1, dims="student")
        irt_mua = pm.Normal("mu_a", 0, 1)
        irt_saa = pm.HalfNormal("sigma_a", 1)
        irt_a = pm.LogNormal("a", irt_mua, irt_saa, dims="item")
        irt_mub = pm.Normal("mu_b", 0, 1)
        irt_sbb = pm.HalfNormal("sigma_b", 1)
        irt_b = pm.Normal("b", irt_mub, irt_sbb, dims="item")
        irt_S = pm.Data("student_idx", irt_si, dims="obs")
        irt_I = pm.Data("item_idx", irt_ii, dims="obs")
        pm.Deterministic("eta", irt_a[irt_I] * (irt_theta[irt_S] - irt_b[irt_I]), dims="obs")
        pm.Bernoulli("y", logit_p=irt_model.named_vars["eta"], observed=irt_obs, dims="obs")
    return (irt_model,)


@app.cell
def _(bayesdag, irt_model, mo):
    mo.ui.anywidget(bayesdag.view(irt_model).widget())
    return


@app.cell
def _(graphviz_fit, irt_model, mo):
    mo.vstack(
        [
            mo.md("**Baseline — PyMC `model_to_graphviz`:**"),
            graphviz_fit(irt_model),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · MRP (multilevel regression & poststratification)

    Five crossed grouping factors (`state`, `age`, `education`, `ethnicity`, `region`), each a
    non-centered random effect, all converging on one logit — six plates and ~7 convergent parents.
    Gelman's canonical polling model; the maximal single-equation layout stress test.
    """)
    return


@app.cell
def _(np, pm):
    mrp_rng = np.random.default_rng(3)
    mrp_ns, mrp_na, mrp_ne, mrp_nh, mrp_nr, mrp_no = 8, 4, 4, 4, 4, 400
    mrp_idx = {
        k: mrp_rng.integers(0, n, mrp_no)
        for k, n in (("s", mrp_ns), ("a", mrp_na), ("e", mrp_ne), ("h", mrp_nh), ("r", mrp_nr))
    }
    mrp_male = mrp_rng.integers(0, 2, mrp_no).astype(float)
    mrp_y = mrp_rng.binomial(1, 0.5, mrp_no)
    mrp_coords = {
        "state": np.arange(mrp_ns),
        "age": np.arange(mrp_na),
        "edu": np.arange(mrp_ne),
        "eth": np.arange(mrp_nh),
        "region": np.arange(mrp_nr),
        "obs": np.arange(mrp_no),
    }
    with pm.Model(coords=mrp_coords) as mrp_model:
        mrp_a = pm.Normal("a", 0, 1)

        def mrp_re(name, dim):
            sg = pm.HalfNormal("sigma_" + name, 1)
            z = pm.Normal("z_" + name, 0, 1, dims=dim)
            return pm.Deterministic(name, z * sg, dims=dim)

        mrp_as = mrp_re("a_state", "state")
        mrp_aa = mrp_re("a_age", "age")
        mrp_ae = mrp_re("a_edu", "edu")
        mrp_ah = mrp_re("a_eth", "eth")
        mrp_ar = mrp_re("a_region", "region")
        mrp_bm = pm.Normal("b_male", 0, 1)
        mrp_S = pm.Data("state_idx", mrp_idx["s"], dims="obs")
        mrp_A = pm.Data("age_idx", mrp_idx["a"], dims="obs")
        mrp_E = pm.Data("edu_idx", mrp_idx["e"], dims="obs")
        mrp_H = pm.Data("eth_idx", mrp_idx["h"], dims="obs")
        mrp_R = pm.Data("region_idx", mrp_idx["r"], dims="obs")
        mrp_M = pm.Data("male", mrp_male, dims="obs")
        pm.Deterministic(
            "p",
            mrp_a
            + mrp_as[mrp_S]
            + mrp_aa[mrp_A]
            + mrp_ae[mrp_E]
            + mrp_ah[mrp_H]
            + mrp_ar[mrp_R]
            + mrp_bm * mrp_M,
            dims="obs",
        )
        pm.Bernoulli("y", logit_p=mrp_model.named_vars["p"], observed=mrp_y, dims="obs")
    return (mrp_model,)


@app.cell
def _(bayesdag, mo, mrp_model):
    mo.ui.anywidget(bayesdag.view(mrp_model).widget())
    return


@app.cell
def _(graphviz_fit, mo, mrp_model):
    mo.vstack(
        [
            mo.md("**Baseline — PyMC `model_to_graphviz`:**"),
            graphviz_fit(mrp_model),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Joint longitudinal–survival model

    Two interlinked sub-hierarchies over `subject`: a longitudinal mixed model (random intercept
    `b0`, slope `b1`, trajectory `traj`) **and** a survival model whose log-rate `gamma0 + alpha*b1`
    depends on the longitudinal slope. The shared parents `b0`/`b1` feed *two* likelihood subtrees —
    the joint-model "association". Canonical in biostatistics.
    """)
    return


@app.cell
def _(np, pm):
    jm_rng = np.random.default_rng(4)
    jm_nsub, jm_nvis = 40, 4
    jm_nl = jm_nsub * jm_nvis
    jm_si = np.repeat(np.arange(jm_nsub), jm_nvis)
    jm_tm = np.tile(np.linspace(0, 1, jm_nvis), jm_nsub)
    jm_b0t = jm_rng.normal(1, 0.5, jm_nsub)
    jm_b1t = jm_rng.normal(-0.5, 0.3, jm_nsub)
    jm_yl = jm_b0t[jm_si] + jm_b1t[jm_si] * jm_tm + jm_rng.normal(0, 0.2, jm_nl)
    jm_ev = jm_rng.exponential(1 / np.exp(-1 + 0.8 * jm_b1t))
    with pm.Model(coords={"subject": np.arange(jm_nsub), "visit": np.arange(jm_nl)}) as jm_model:
        jm_mb0 = pm.Normal("mu_b0", 0, 2)
        jm_s0 = pm.HalfNormal("sigma_b0", 1)
        jm_mb1 = pm.Normal("mu_b1", 0, 2)
        jm_s1 = pm.HalfNormal("sigma_b1", 1)
        jm_b0 = pm.Normal("b0", jm_mb0, jm_s0, dims="subject")
        jm_b1 = pm.Normal("b1", jm_mb1, jm_s1, dims="subject")
        jm_sy = pm.HalfNormal("sigma_y", 1)
        jm_S = pm.Data("subj_idx", jm_si, dims="visit")
        jm_T = pm.Data("time", jm_tm, dims="visit")
        pm.Deterministic("traj", jm_b0[jm_S] + jm_b1[jm_S] * jm_T, dims="visit")
        pm.Normal("y_long", jm_model.named_vars["traj"], jm_sy, observed=jm_yl, dims="visit")
        jm_g0 = pm.Normal("gamma0", 0, 2)
        jm_al = pm.Normal("alpha", 0, 1)
        pm.Deterministic("log_rate", jm_g0 + jm_al * jm_b1, dims="subject")
        pm.Exponential(
            "event_time",
            pm.math.exp(-jm_model.named_vars["log_rate"]),
            observed=jm_ev,
            dims="subject",
        )
    return (jm_model,)


@app.cell
def _(bayesdag, jm_model, mo):
    mo.ui.anywidget(bayesdag.view(jm_model).widget())
    return


@app.cell
def _(graphviz_fit, jm_model, mo):
    mo.vstack(
        [
            mo.md("**Baseline — PyMC `model_to_graphviz`:**"),
            graphviz_fit(jm_model),
        ]
    )
    return


@app.cell
def _():
    import zoo

    return (zoo,)


@app.cell
def _(bayesdag, graphviz_fit, mo):
    def show(model, title, blurb):
        return mo.vstack(
            [
                mo.md(f"## {title}\n\n{blurb}"),
                mo.ui.anywidget(bayesdag.view(model).widget()),
                mo.md("**PyMC `model_to_graphviz` for the same model:**"),
                graphviz_fit(model),
            ]
        )

    return (show,)


@app.cell
def _(show, zoo):
    show(
        zoo.build_stochastic_volatility(),
        "7 · Stochastic volatility (finance)",
        "Heavy-tailed `StudentT` returns over a latent Gaussian-random-walk log-volatility — the "
        "**fan chart** on `log_vol` is the random walk's widening uncertainty band.",
    )
    return


@app.cell
def _(show, zoo):
    show(
        zoo.build_weibull_survival(),
        "8 · Weibull survival with right-censoring (biostatistics)",
        "A `Censored` Weibull time-to-event likelihood (observations censored at t=8); the observed "
        "times render as a data histogram with the censored events piled at the bound.",
    )
    return


@app.cell
def _(show, zoo):
    show(
        zoo.build_zero_inflated_counts(),
        "9 · Zero-inflated counts (ecology)",
        "Excess-zero catch counts via `ZeroInflatedPoisson`; the regression `slope` carries a spike-and-slab "
        "`Mixture` prior — its **composite glyph** shows the narrow spike over the wide slab.",
    )
    return


@app.cell
def _(show, zoo):
    show(
        zoo.build_correlated_slopes(),
        "10 · Correlated varying slopes (multilevel)",
        "Per-café intercept+slope drawn `MvNormal` with an **LKJ** Cholesky correlation prior — `cafe_effect` "
        "shows the **pairplot** (marginals + covariance ellipse) and the LKJ node its **marginal "
        "correlation density** on [-1, 1].",
    )
    return


@app.cell
def _(show, zoo):
    show(
        zoo.build_ordinal_ratings(),
        "11 · Ordinal ratings (survey)",
        "An `OrderedLogistic` response over a latent scale cut by ordered `cutpoints`; the ordinal "
        "data renders as per-class bars.",
    )
    return


@app.cell
def _(show, zoo):
    show(
        zoo.build_gaussian_mixture(),
        "12 · Gaussian mixture (clustering)",
        "A two-component `NormalMixture` with `Dirichlet` weights — the **simplex** glyph on `weights` shows "
        "the components' marginal Beta densities.",
    )
    return


@app.cell
def _(show, zoo):
    show(
        zoo.build_disease_mapping(),
        "13 · Disease mapping (epidemiology)",
        "Areal disease counts with an intrinsic spatial effect (`ICAR`) — its neighbourhood matrix "
        "renders as an **adjacency heatmap**.",
    )
    return


@app.cell
def _(show, zoo):
    show(
        zoo.build_ar_forecast(),
        "14 · Autoregressive trend (econometrics)",
        "A latent second-order `AR` trend behind noisy observations (state-space) — `level` shows the "
        "**partial-autocorrelation (PACF)** stem plot, whose spikes cut off after the AR order.",
    )
    return


@app.cell
def _(show, zoo):
    show(
        zoo.build_logistic_regression(),
        "15 · Logistic regression (classification)",
        "The linear predictor and inverse-link are explicit deterministics: `linear_pred` shows the affine "
        "**line** transfer glyph, and `prob = sigmoid(linear_pred)` the **logistic S-curve** — drawn only "
        "because each function is provable from the op graph (a hand-written `1/(1+exp(-x))` would stay "
        "equation-only).",
    )
    return


@app.cell
def _(show, zoo):
    show(
        zoo.build_poisson_loglink(),
        "16 · Poisson regression with a log link (counts)",
        "An explicit log link: `linear_pred` is the **line**, `rate = exp(linear_pred)` the **exponential** "
        "transfer-function curve (distinct from the logistic S above).",
    )
    return


@app.cell
def _(show, zoo):
    show(
        zoo.build_softmax_categorical(),
        "17 · Softmax classifier (multinomial choice)",
        "A multinomial-logit model: `probs = softmax(category_logits)` renders as the **k-category bars** of "
        "its simplex output (one probability vector over the categories).",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Transfer-function glyphs — newly showcased

    The deterministic transfer detector draws a canonical curve only when the shape is *provable*
    from the op graph (zero false positives). These four sections exercise transfers the gallery
    did not previously show — each curve is generated from the **true** function on a fixed grid,
    so it is identical on every render and its width is fixed (it no longer stretches with the equation).
    """)
    return


@app.cell(hide_code=True)
def _(show, zoo):
    show(
        zoo.build_probit_regression(),
        "18 - Probit regression (erf)",
        "`prob = Phi(linear_pred)` written as `0.5*(1 + erf(linear_pred/sqrt(2)))`. `prob` shows the **probit** "
        "S-curve, drawn from the true Gaussian CDF - visibly steeper-shouldered than the logistic S in section 15.",
    )
    return


@app.cell(hide_code=True)
def _(show, zoo):
    show(
        zoo.build_heteroskedastic_softplus(),
        "19 - Heteroskedastic scale (softplus)",
        "Non-constant noise: `sigma = softplus(scale_pred)` keeps the scale positive. `sigma` shows the "
        "**softplus** smooth-positive-ramp - a deterministic that feeds a *scale*, not a mean (`mu`, `scale_pred` are lines).",
    )
    return


@app.cell(hide_code=True)
def _(show, zoo):
    show(
        zoo.build_saturating_tanh(),
        "20 - Saturating response (tanh)",
        "A bounded dose-response: `effect = tanh(linear_pred)` tapers to +/-1. The **tanh** S-curve on [-1, 1] "
        "is a different bounded shape from the logistic [0, 1].",
    )
    return


@app.cell(hide_code=True)
def _(show, zoo):
    show(
        zoo.build_quadratic_power(),
        "21 - Squared amplitude (pow)",
        "Signal power ~ amplitude^2: `power = amplitude**2`. The **pow** glyph reads the constant exponent "
        "from the op graph and draws the *actual* parabola x^2, not a generic curve.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Nonparametric regression — fully-Bayesian BART, every parameter named

    BART (Bayesian Additive Regression Trees) is a **sum of regression trees**. Rather than one opaque
    `BART(...)` node, this is the whole model with **plain-English names** and its hyperparameters
    *learned* (given priors) rather than fixed. The tree-structure controls are now distributions:
    `split_prob ~ Beta(2, 5)` (how readily a node splits), `depth_penalty ~ Gamma` (how fast splitting
    decays with depth), `leaf_shrinkage ~ Gamma` (regularization of leaf values). From them come the
    depth prior `split_prob_by_depth = split_prob*(1+depth)^(-depth_penalty)` and the leaf spread
    `leaf_scale`. The `tree` plate carries each tree's parameters — `splits`, `split_point`,
    `leaf_left`, `leaf_right` — combined by the decision `tree_output`, summed to `prediction`. (The
    per-tree computation is a depth-1 stump for legibility; the depth prior is what grows real BART
    deeper. bayesdag *also* renders an opaque `pmb.BART` node as a step-function glyph, in the tests.)
    """)
    return


@app.cell(hide_code=True)
def _(show, zoo):
    show(
        zoo.build_bart_sum_of_trees(),
        "22 - Fully-Bayesian BART, every parameter named",
        "BART with its hyperparameters learned and labelled in plain English: `split_prob ~ Beta(2, 5)` "
        "(base split probability), `depth_penalty ~ Gamma` and `leaf_shrinkage ~ Gamma`. These give the "
        "depth prior `split_prob_by_depth` and the `leaf_scale`. The `tree` plate holds `splits`, "
        "`split_point`, and leaf values `leaf_left, leaf_right`; each tree is the decision "
        "`tree_output = (splits & x<=split_point ? leaf_left : leaf_right)`, and the regression mean is "
        "`prediction = sum(tree_output)`, closed by `y ~ Normal(prediction, noise)`.",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Notes

    Each model fits quickly (`pm.sample` with small `draws`); pass `idata=` to `view(...)` to turn
    the prior glyphs into posterior KDEs (as in §1–2). Outside a notebook the same
    `bayesdag.view(...)` degrades automatically to a static SVG via `_repr_svg_`; pass
    `legend=False` for a bare figure.
    """)
    return


if __name__ == "__main__":
    app.run()
