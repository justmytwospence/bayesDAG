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
        # bayesdag — eight schools (M0 vertical slice)

        Shape-first, posterior-aware visualization of a PyMC model. Run from the project
        root: `uv run marimo edit examples/eight_schools.py`.
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
def _(mo):
    mo.md(
        """
        ## The model — non-centered eight schools

        Hyperpriors `mu`, `tau`; a non-centered deterministic `theta = mu + tau*eta`; an
        observed Normal likelihood over a `school` plate.
        """
    )
    return


@app.cell
def _(np, pm):
    y = np.array([28.0, 8, -3, 7, -1, 1, 18, 12])
    sigma = np.array([15.0, 10, 16, 11, 9, 11, 10, 18])
    with pm.Model(coords={"school": [f"S{i}" for i in range(8)]}) as model:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfNormal("tau", 5)
        eta = pm.Normal("eta", 0, 1, dims="school")
        theta = pm.Deterministic("theta", mu + tau * eta, dims="school")
        pm.Normal("y_obs", theta, sigma, observed=y, dims="school")
    return (model,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Static render

        Each node shows its distribution's **shape** (note `tau`'s half-normal starting at
        0); the deterministic is rendered as real math; the observed node is a **histogram**
        of the data; and the arrows from `mu`/`tau`/`eta` land on those exact tokens inside
        `theta = mu + tau*eta` (port-level edges).
        """
    )
    return


@app.cell
def _(bayesdag, mo, model):
    prior_view = bayesdag.view(model)
    mo.Html(prior_view.to_svg())
    return (prior_view,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Interactive render

        The **identical** SVG, in an anywidget with pan/zoom. (Static and interactive are
        the same bytes — parity by construction.)
        """
    )
    return


@app.cell
def _(mo, prior_view):
    mo.ui.anywidget(prior_view.widget())
    return


@app.cell
def _(mo):
    mo.md("## Fit, then posterior overlays")
    return


@app.cell
def _(model, pm):
    with model:
        idata = pm.sample(
            draws=300, tune=300, chains=2, random_seed=0, progressbar=False
        )
    return (idata,)


@app.cell
def _(mo):
    mo.md("Node glyphs are now **posterior** KDEs (orange) from the fitted `idata`:")
    return


@app.cell
def _(bayesdag, idata, mo, model):
    mo.Html(bayesdag.view(model, idata=idata).to_svg())
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Export & fallback

        `view.save("eight_schools.svg")` writes publication SVG (PNG/PDF via the `[export]`
        extra). Outside a notebook (a plain script, nbconvert), the same `bayesdag.view(...)`
        degrades automatically to static SVG via `_repr_svg_`.
        """
    )
    return


@app.cell
def _(prior_view):
    prior_view.save("examples/eight_schools.svg")
    return


if __name__ == "__main__":
    app.run()
