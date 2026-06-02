import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
    # bayesdag — radon (a hierarchical model worth exploring)

    A varying-intercept regression: per-county intercepts `a[county]` drawn from
    hyperpriors `mu_a`, `sigma_a`, a shared floor effect `b`, and an observed `y`.
    With two plates (`county`, `obs`) and data nodes, the **interactive** view earns
    its keep — hover to trace dependencies, click a node for its card, and click a
    **plate** to expand its prior-predictive check.
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import pymc as pm

    import bayesdag

    return bayesdag, np, pm


@app.cell
def _(np, pm):
    rng = np.random.default_rng(0)
    n_counties, n_obs = 6, 40
    counties = [f"C{i}" for i in range(n_counties)]
    county_idx = rng.integers(0, n_counties, n_obs)
    floor = rng.integers(0, 2, n_obs).astype(float)
    a_true = rng.normal(1.2, 0.5, n_counties)
    log_radon = a_true[county_idx] - 0.6 * floor + rng.normal(0, 0.4, n_obs)

    with pm.Model(coords={"county": counties, "obs": np.arange(n_obs)}) as model:
        mu_a = pm.Normal("mu_a", 0, 5)
        sigma_a = pm.HalfNormal("sigma_a", 5)
        a = pm.Normal("a", mu_a, sigma_a, dims="county")
        b = pm.Normal("b", 0, 5)
        sigma = pm.HalfNormal("sigma", 1)
        cidx = pm.Data("county_idx", county_idx, dims="obs")
        fl = pm.Data("floor", floor, dims="obs")
        mu = pm.Deterministic("mu", a[cidx] + b * fl, dims="obs")
        pm.Normal("y", mu, sigma, observed=log_radon, dims="obs")
    return log_radon, model


@app.cell
def _(mo):
    mo.md(
        """
    ## Interactive view — try it

    - **Hover** `a` (the county intercepts): everything outside its Markov blanket fades,
      so you see it depends on `mu_a`, `sigma_a` and feeds `mu`. A tooltip shows its distribution.
    - **Click** any node for a detail card (distribution, parameters, dims, a copyable `pm.*` line).
    - **Click the `county` plate** to expand a prior-predictive check of the 6 intercepts;
      **click the `obs` plate** to see the prior-predictive `y` vs. the observed log-radon.
    """
    )
    return


@app.cell
def _(bayesdag, mo, model):
    radon_view = bayesdag.view(model)
    mo.ui.anywidget(radon_view.widget())
    return (radon_view,)


@app.cell
def _(mo):
    mo.md("## Static figure (publication SVG) — with the legend")
    return


@app.cell
def _(mo, radon_view):
    mo.Html(radon_view.to_svg())
    return


@app.cell
def _(mo):
    mo.md("## Fit, then posterior overlays")
    return


@app.cell
def _(model, pm):
    with model:
        idata = pm.sample(draws=300, tune=300, chains=2, random_seed=0, progressbar=False)
    return (idata,)


@app.cell
def _(bayesdag, idata, mo, model):
    mo.Html(bayesdag.view(model, idata=idata).to_svg())
    return


if __name__ == "__main__":
    app.run()
