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
    # bayesdag — a tour

    Shape-first, posterior-aware, interactive visualization of PyMC models. Run from the
    project root: `uv run marimo edit examples/tour.py`.

    Two models, in sections: **eight schools** (the clean intro) and **radon** (a bigger
    hierarchical model where the interactive view earns its keep).
    """)
    return


@app.cell
def _():
    import numpy as np
    import pymc as pm

    import bayesdag

    return bayesdag, np, pm


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
    land on those exact tokens inside `theta = mu + tau*eta` (port-level edges). The legend (to the
    side) is context-aware — it only lists encodings actually present.

    - **Hover** a node: everything outside its Markov blanket fades; a tooltip shows the distribution.
    - **Click** a node for a detail card (distribution, parameters, dims, a copyable `pm.*` line).
    - **Click the `school` plate** to expand its **prior predictive check** — the 8 per-school
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
def _(es_view):
    es_view.save("examples/eight_schools.svg")  # publication SVG (PNG/PDF via [export])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Radon (where interactivity earns its keep)

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
    with pm.Model(coords={"county": [f"C{i}" for i in range(n_counties)], "obs": np.arange(n_obs)}) as radon_model:
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
    ## Notes

    Outside a notebook (a plain script, nbconvert), the same `bayesdag.view(...)` degrades
    automatically to a static SVG via `_repr_svg_`. Pass `legend=False` for a bare figure.
    """)
    return


if __name__ == "__main__":
    app.run()
