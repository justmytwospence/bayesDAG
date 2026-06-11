# bayesdag

**Shape-first, posterior-aware, interactive visualizations of PyMC models.**

`bayesdag` renders a PyMC model as a rich generative-model diagram where **every node shows the full shape of its distribution** — so you can read the data-generating story off the curves (Kruschke-style) — with real **LaTeX math inside nodes**, **edges that point at the specific parameter** they feed, prior/posterior overlays, and a model-aware **posterior-geometry** explorer (funnels, divergences). It has a **static SVG renderer** (publication-quality) and an **interactive [anywidget](https://anywidget.dev)** renderer (Jupyter + [marimo](https://marimo.io)) that are *identical by construction*, and it falls back to static automatically when interactivity isn't available.

> **Status: pre-alpha (M0 in progress).** See [`COVERAGE.md`](COVERAGE.md) for the live work-plan. Background and sources: [`docs/RESEARCH.md`](docs/RESEARCH.md).

## Install

```bash
pip install "bayesdag[interactive]"     # or: uv add "bayesdag[interactive]"
```

Requires Python ≥ 3.10 and the Graphviz `dot` binary (`brew install graphviz` / `apt install graphviz`). Extras: `[interactive]` (widget), `[math]` (in-process MathJax for crisp math), `[export]` (PNG/PDF), `[interop]` (PROV export), `[reparam]` (VIP reparameterization suggestions).

## Quickstart

```python
import numpy as np, pymc as pm, bayesdag

J = 8
y = np.array([28., 8, -3, 7, -1, 1, 18, 12])
sigma = np.array([15., 10, 16, 11, 9, 11, 10, 18])

with pm.Model(coords={"school": [f"S{i}" for i in range(J)]}) as model:
    mu = pm.Normal("mu", 0, 5)
    tau = pm.HalfNormal("tau", 5)
    eta = pm.Normal("eta", 0, 1, dims="school")
    theta = pm.Deterministic("theta", mu + tau * eta, dims="school")
    y_obs = pm.Normal("y_obs", theta, sigma, observed=y, dims="school")

bayesdag.view(model)                 # interactive in a notebook, static SVG elsewhere
bayesdag.view(model).save("eight_schools.svg")

idata = pm.sample(model=model)
bayesdag.view(model, idata=idata)    # nodes gain posterior overlays
```

## What you see (glyph legend)

Every node carries a **density curve** (the primary mark); an optional credible interval / point summary are annotations, never a substitute. A small badge says where the shape came from:

- **actual prior** — the family density with the node's real parameters plugged in.
- **shape only / from parents** — a hierarchical prior whose parameters are themselves random (prior-predictive or canonical family shape).
- **posterior** — KDE from a fitted `InferenceData`.
- **observed data** — a histogram (auto-binned), not a KDE.

Node shapes follow PyMC's vocabulary (latent = ellipse, observed = filled, deterministic = box, data = rounded box, `Potential` = factor glyph). Constructs that can't be drawn honestly are badged "density-only / elided" rather than shown as a misleading arrow.

## Roadmap

- **M0** — vertical slice: 8-schools rendered in both renderers, LaTeX nodes, port-edges, prior/posterior/observed glyphs, fallback.
- **M1** — ~100% PyMC coverage (mixtures, truncated, timeseries, LKJ, imputation, …).
- **M2** — posterior-geometry explorer (funnels/divergences), workflow toggle, prior linter, distribution cards.
- **M3** — interop exporters (GraphML/PROV), reparameterization suggestions, scale, cross-PPL (`from_numpyro`), upstreaming to PyMC.

## License

MIT — see [`LICENSE`](LICENSE).
