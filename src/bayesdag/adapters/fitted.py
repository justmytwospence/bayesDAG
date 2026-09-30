"""Fitted-model panels for recognized structures (PyMC side): the classic pictures of what a
hierarchy, a regression or a shrinkage prior DID once the model is fit.

Each panel function takes a :class:`Fit` (model + IR + idata) and a block, and returns plain
data for ``render_panels`` — or raises :class:`Unavailable` with the reason, shown instead of a
picture. The honesty rules:

* Posterior quantities are exact functions of the draws. Where a panel evaluates the linear
  predictor at posterior MEANS (to get "the rest of the predictor"), that is exact only because
  the recognized predictor is affine in its leaves — which the regression rule proved.
* Data summaries (a group's raw mean, least-squares lines) are labelled as data summaries, and
  only drawn where they are a like-for-like comparison: an identity (or log, for a LogNormal)
  link, one grouping index, one covariate. Anything else is refused with a reason.
* Nothing samples: every number comes from the caller's idata or the model's data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..ir import BlockIR, ModelIR
from . import blocks as B
from ._ptutil import depends_on_rv, dist_params

MAX_DRAWS = 400  # draws used for draw-wise curves (evenly spaced, deterministic)


class Unavailable(Exception):
    """This panel cannot be drawn honestly for this model/fit; the message says why."""


@dataclass
class Fit:
    model: Any
    ir: ModelIR
    idata: Any

    def __post_init__(self):
        if self.model is None:
            raise Unavailable("fitted panels need the model (a prebuilt IR has no data to show)")
        if self.idata is None or getattr(self.idata, "posterior", None) is None:
            raise Unavailable("attach a posterior (view.update(idata)) to see this panel")
        named = {id(v): n for n, v in self.model.named_vars.items()}
        consumers: dict[str, set[str]] = {}
        for e in self.ir.edges:
            consumers.setdefault(e.source, set()).add(e.target)
        self.ctx = _FitCtx(self.model, named, self.ir, consumers)

    # ---- draws and data
    def has(self, name: str) -> bool:
        post = self.idata.posterior
        return name in post

    def draws(self, name: str, n: int | None = MAX_DRAWS) -> np.ndarray:
        """``(S, *element shape)`` posterior draws of ``name`` (chain/draw flattened, in
        coordinate order), thinned to at most ``n`` evenly spaced draws."""
        from .glyph_data import _group_draws

        da = _group_draws(self.idata, "posterior", name)
        if da is None:
            raise Unavailable(f"the posterior has no draws for {name!r}")
        v = np.asarray(da.values, float)
        v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
        if n is not None and v.shape[0] > n:
            v = v[np.unique(np.linspace(0, v.shape[0] - 1, n).round().astype(int))]
        return v

    def mean(self, name: str) -> np.ndarray:
        return self.draws(name, None).mean(axis=0)

    def value(self, name: str) -> np.ndarray:
        """A pm.Data (or other RV-free named) variable's current value."""
        v = self.model[name]
        if depends_on_rv(v):
            raise Unavailable(f"{name!r} is not data")
        return np.asarray(v.eval())

    def observed(self, name: str) -> np.ndarray:
        from .glyph_data import _observed_values

        v = _observed_values(self.model[name], self.model)
        if v is None:
            raise Unavailable(f"no observed values for {name!r}")
        return np.asarray(v, float)

    def labels(self, name: str, k: int) -> list[str]:
        n = self.ir.node(name)
        if n is not None and n.dims and n.coords and n.dims[0] in n.coords:
            vals = n.coords[n.dims[0]]
            if len(vals) == k:
                return [str(x) for x in vals]
        return [str(i) for i in range(k)]


class _FitCtx(B._Ctx):
    """The detection context, able to see variables the DISPLAYED IR left out (index data folded
    into equations is still data the panels must read)."""

    def role(self, name):
        r = super().role(name)
        if r is not None:
            return r
        v = self.model.named_vars.get(name)
        if v is None:
            return None
        if v in getattr(self.model, "data_vars", []):
            return "data"
        if v in self.model.deterministics:
            return "deterministic"
        return "latent" if v in self.model.free_RVs else None

    def dims(self, name) -> tuple:
        n = self.node(name)
        if n is not None:
            return tuple(n.dims)
        return tuple(self.model.named_vars_to_dims.get(name, ()))


def _compile(expr, model, names: list[str]):
    """A function of the named variables ``names`` (in order) computing ``expr``, with everything
    else fixed. Refuses a graph that would still draw random numbers."""
    import pytensor
    import pytensor.tensor as pt
    from pytensor.graph.replace import graph_replace

    vs = [model[n] for n in names]
    phs = [
        pt.tensor(dtype=v.dtype, shape=(None,) * v.ndim, name=n)
        for v, n in zip(vs, names, strict=True)
    ]
    out = graph_replace(expr, dict(zip(vs, phs, strict=True)), strict=False)
    rest = [
        a
        for a in _ancestors(out)
        if getattr(a, "owner", None) is not None and _is_rv_op(a.owner.op)
    ]
    if rest:
        raise Unavailable("the predictor depends on a quantity with no draws to plug in")
    return pytensor.function(phs, out, on_unused_input="ignore", allow_input_downcast=True)


def _ancestors(v):
    from pytensor.graph.traversal import ancestors

    return list(ancestors([v]))


def _is_rv_op(op) -> bool:
    from ._ptutil import op_is_rv

    return op_is_rv(op, unknown=False)


# ---------------------------------------------------------------------------------- predictors
_LOC_SLOT = {
    "Normal": 0,
    "StudentT": 1,
    "LogNormal": 0,
    "Bernoulli": 0,
    "Binomial": 1,
    "Poisson": 0,
}
_SCALE_SLOT = {"Normal": 1, "StudentT": 2, "LogNormal": 1}


@dataclass
class Predictor:
    y: str
    family: str
    link: str
    eta: Any
    terms: list
    leaves: list[str]
    data: list[str]
    n_obs: int
    fn: Any = None

    def evaluate(self, fit: Fit, values: dict) -> np.ndarray:
        """The linear predictor with ``values`` (name -> array) plugged in; data default to
        their current values."""
        if self.fn is None:
            self.fn = _compile(self.eta, fit.model, self.leaves + self.data)
        args = [values[n] if n in values else fit.value(n) for n in self.data]
        return np.asarray(self.fn(*[values[n] for n in self.leaves], *args), float)


def predictor(fit: Fit, y: str) -> Predictor:
    """The recognized affine predictor of the observed ``y``: its terms, the parameters it is
    linear in, and the data it reads."""
    node = fit.ir.node(y)
    if node is None or node.role != "observed" or node.dist not in _LOC_SLOT:
        raise Unavailable(f"{y!r} has no recognized location")
    dp = dist_params(fit.model[y])
    ctx = fit.ctx
    ctx.visits[0] = 0
    link, eta = B._link(dp[_LOC_SLOT[node.dist]], ctx)
    blocked = {b.anchor for b in fit.ir.blocks if b.anchor and b.kind != "regression"}
    terms = B._affine_terms(eta, ctx, blocked)
    if not terms:
        raise Unavailable(f"{y!r}'s location is not affine in its parameters")
    leaves = sorted({p for t in terms for p in t.params})
    data = sorted({d for t in terms for d in (t.index + t.data)})
    y_val = fit.observed(y)
    return Predictor(y, node.dist, link, eta, terms, leaves, data, int(y_val.reshape(-1).size))


def _consumers_of(fit: Fit, effect: str) -> list[str]:
    """Observed likelihoods whose recognized predictor uses ``effect`` directly."""
    out = []
    for n in fit.ir.nodes:
        if n.role != "observed" or n.dist not in _LOC_SLOT:
            continue
        try:
            p = predictor(fit, n.id)
        except Exception:
            continue
        if any(effect in t.params for t in p.terms):
            out.append(n.id)
    return out


def _means(fit: Fit, names) -> dict:
    return {n: fit.mean(n) for n in names}


def _data_at(fit: Fit, pred: Predictor, n_rows: int, override: dict) -> dict:
    """Per-observation data replaced by ``n_rows`` rows (overrides, else the column means);
    data that is not per-observation keeps its value."""
    out = {}
    for d in pred.data:
        if d in override:
            out[d] = override[d]
            continue
        v = fit.value(d)
        if v.ndim >= 1 and v.shape[0] == pred.n_obs:
            if np.issubdtype(v.dtype, np.integer):
                out[d] = np.zeros((n_rows, *v.shape[1:]), v.dtype)  # replaced by callers
            else:
                out[d] = np.broadcast_to(v.mean(axis=0), (n_rows, *v.shape[1:])).copy()
        else:
            out[d] = v
    return out


def _y_scale(pred: Predictor, y: np.ndarray) -> tuple[np.ndarray, str]:
    """The observations on the predictor's scale (identity link; log for a LogNormal)."""
    if pred.link != "identity":
        raise Unavailable(
            f"this comparison needs an identity link ({pred.y} uses {pred.link}); raw group "
            "summaries are not on the predictor's scale"
        )
    if pred.family == "LogNormal":
        if np.any(y <= 0):
            raise Unavailable("log-normal data must be positive")
        return np.log(y), f"log {pred.y}"
    if pred.family not in ("Normal", "StudentT"):
        raise Unavailable(f"a {pred.family} likelihood has no identity-scale data summary")
    return y, pred.y


def _obs_sd(fit: Fit, pred: Predictor) -> np.ndarray | None:
    """Per-observation residual sd at its posterior mean (or its fixed value), if knowable."""
    slot = _SCALE_SLOT.get(pred.family)
    if slot is None:
        return None
    t = dist_params(fit.model[pred.y])[slot]
    if not depends_on_rv(t):
        return np.broadcast_to(np.asarray(t.eval(), float), (pred.n_obs,)).copy()
    u = B._unwrap(t, fit.ctx)
    if id(u) in fit.ctx.named and fit.has(fit.ctx.named[id(u)]):
        m = fit.mean(fit.ctx.named[id(u)])
        return np.broadcast_to(m, (pred.n_obs,)).copy()
    return None


# ---------------------------------------------------------------------------------- hierarchies
@dataclass
class _Hier:
    effect: str
    family: str  # Normal | LogNormal
    loc: tuple  # ("param", name) | ("const", value)
    scale: tuple  # ("param", name, factor) | ("const", value)


def _hier(fit: Fit, b: BlockIR) -> _Hier:
    a = b.anchor
    node = fit.ir.node(a)
    ctx = fit.ctx
    ctx.visits[0] = 0
    if b.variant == "noncentered":
        nc = B._noncentered(a, ctx)
        if nc is None or nc.family not in ("Normal", "LogNormal"):
            raise Unavailable(
                "the population is not a Normal/LogNormal with a single location/scale"
            )
        if not nc.aff.loc:
            loc = ("const", 0.0)
        elif len(nc.aff.loc) == 1 and nc.aff.loc[0][0] == 1:
            loc = _single(nc.aff.loc[0][1], ctx, "location")
        else:
            raise Unavailable("the population location is not a single parameter")
        if len(nc.aff.scale) != 1 or nc.aff.scale[0][0] != 1.0 or nc.aff.matrix is not None:
            raise Unavailable("the population scale is not a single parameter")
        sc = _single(nc.aff.scale[0][1], ctx, "scale")
        s0 = nc.aux_info["scale"]
        scale = (sc[0], sc[1], s0) if sc[0] == "param" else ("const", sc[1] * s0)
        return _Hier(a, nc.family, loc, scale)
    if node is None or node.dist not in ("Normal", "LogNormal"):
        raise Unavailable("the population is not a Normal/LogNormal")
    dp = dist_params(fit.model[a])
    return _Hier(
        a, node.dist, _single(dp[0], ctx, "location"), _add_factor(_single(dp[1], ctx, "scale"))
    )


def _add_factor(s):
    return (s[0], s[1], 1.0) if s[0] == "param" else s


def _single(t, ctx, what: str) -> tuple:
    if not depends_on_rv(t):
        a = np.asarray(t.eval(), float)
        if a.size and np.all(a == a.reshape(-1)[0]):
            return ("const", float(a.reshape(-1)[0]))
        raise Unavailable(f"the population {what} varies by element")
    u = B._unwrap(t, ctx)
    if id(u) in ctx.named:
        n = ctx.named[id(u)]
        if tuple(ctx.dims(n)) == ():
            return ("param", n)
    raise Unavailable(f"the population {what} is not a single scalar parameter")


def _loc_draws(fit: Fit, h: _Hier) -> np.ndarray | float:
    return fit.draws(h.loc[1]) if h.loc[0] == "param" else h.loc[1]


def _scale_draws(fit: Fit, h: _Hier) -> np.ndarray | float:
    return fit.draws(h.scale[1]) * h.scale[2] if h.scale[0] == "param" else h.scale[1]


def population(fit: Fit, b: BlockIR) -> dict:
    """Draws of the population distribution the group effects are drawn from, with the groups'
    posterior means as a rug: where a NEW group would fall, and how spread the groups are."""
    import scipy.stats as st

    h = _hier(fit, b)
    eff = fit.draws(h.effect)
    means = eff.reshape(eff.shape[0], -1).mean(axis=0)
    mu = np.broadcast_to(_loc_draws(fit, h), (eff.shape[0],))
    tau = np.broadcast_to(_scale_draws(fit, h), (eff.shape[0],))
    if h.family == "LogNormal":
        frozen = [st.lognorm(s=t, scale=np.exp(m)) for m, t in zip(mu, tau, strict=True)]
    else:
        frozen = [st.norm(m, t) for m, t in zip(mu, tau, strict=True)]
    qs = np.array([[f.ppf(0.02), f.ppf(0.98)] for f in frozen[:: max(1, len(frozen) // 60)]])
    lo = float(min(np.median(qs[:, 0]), means.min()))
    hi = float(max(np.median(qs[:, 1]), means.max()))
    pad = 0.05 * (hi - lo or 1.0)
    xs = np.linspace(lo - pad, hi + pad, 120)
    dens = np.array([f.pdf(xs) for f in frozen])
    pred = dens.mean(axis=0)
    idx = np.unique(np.linspace(0, len(frozen) - 1, min(40, len(frozen))).round().astype(int))
    curves = dens[idx]
    top = float(max(pred.max(), np.percentile(curves.max(axis=1), 90))) or 1.0
    return {
        "effect": h.effect,
        "family": h.family,
        "xs": xs.tolist(),
        "curves": (np.minimum(curves / top, 1.2)).tolist(),
        "predictive": (pred / top).tolist(),
        "rug": means.tolist(),
        "n_groups": int(means.size),
    }


def pooling(fit: Fit, b: BlockIR) -> dict:
    """Gelman & Pardoe's (2006) pooling factor for the batch, from the draws alone:
    ``lambda = 1 - V_j(E[eps_j]) / E[V_j(eps_j)]``, ``eps_j = theta_j - mu`` (on the log scale
    for a LogNormal population). 0 = no pooling, 1 = complete pooling. With raw group estimates
    available (see :func:`shrinkage`), also each group's fraction of the way pulled in."""
    h = _hier(fit, b)
    eff = fit.draws(h.effect, None)
    eff = eff.reshape(eff.shape[0], -1)
    if h.family == "LogNormal":
        if np.any(eff <= 0):
            raise Unavailable("a log-normal effect must be positive")
        eff = np.log(eff)
    mu = h.loc[1] if h.loc[0] == "const" else fit.draws(h.loc[1], None)
    eps = eff - np.asarray(mu).reshape(-1, 1)
    num = np.var(eps.mean(axis=0))
    den = np.mean(np.var(eps, axis=1))
    lam = float(1.0 - num / den) if den > 0 else float("nan")
    out = {"effect": h.effect, "lambda": lam, "groups": None}
    try:
        s = shrinkage(fit, b)
        pull = []
        for raw, post in zip(s["raw"], s["post"], strict=True):
            d = raw - s["pop"] if s["pop"] is not None else None
            pull.append(float((raw - post) / d) if d not in (None, 0.0) else None)
        out["groups"] = {"labels": s["labels"], "pull": pull, "n": s["n"]}
    except Unavailable as exc:
        out["groups_reason"] = str(exc)
    return out


def shrinkage(fit: Fit, b: BlockIR) -> dict:
    """The Efron–Morris / Gelman–Hill shrinkage plot: each group's raw estimate (the data)
    against its posterior mean, pulled toward the population mean.

    The raw estimate is the group mean of ``y`` minus the REST of the predictor at its posterior
    mean — exact bookkeeping because the predictor is affine — so it is the no-pooling estimate
    of this effect with everything else held at the fit."""
    a = b.anchor
    eff = fit.draws(a, None)
    if eff.ndim != 2:
        raise Unavailable("the group effect must be one value per group")
    cons = _consumers_of(fit, a)
    if not cons:
        raise Unavailable(f"no observed likelihood uses {a!r} directly")
    pred = predictor(fit, cons[0])
    y_raw = fit.observed(pred.y).reshape(-1)
    y, ylab = _y_scale(pred, y_raw)
    term = next(t for t in pred.terms if a in t.params)
    if term.kind == "gather" and len(term.index) == 1 and term.params == [a]:
        codes = fit.value(term.index[0]).astype(int).reshape(-1)
    elif term.kind == "leaf" and term.params == [a] and eff.shape[1] == pred.n_obs:
        codes = np.arange(pred.n_obs)
    else:
        raise Unavailable(f"{a!r} does not enter {pred.y}'s predictor as a plain group intercept")
    means = _means(fit, pred.leaves)
    eta_hat = pred.evaluate(fit, means).reshape(-1)
    rest = eta_hat - means[a][codes]
    resid = y - rest
    G = eff.shape[1]
    sd = _obs_sd(fit, pred)
    raw, n, se = [], [], []
    for j in range(G):
        m = codes == j
        n.append(int(m.sum()))
        raw.append(float(resid[m].mean()) if m.any() else float("nan"))
        if sd is not None and m.any():
            se.append(float(np.sqrt((sd[m] ** 2).sum()) / m.sum()))
        elif m.sum() >= 2:
            se.append(float(resid[m].std(ddof=1) / np.sqrt(m.sum())))
        else:
            se.append(None)
    try:
        h = _hier(fit, b)
        pop = float(np.mean(_loc_draws(fit, h)))
        if h.family == "LogNormal":
            pop = None  # a log-scale population is not on this effect's scale
    except Unavailable:
        pop = None
    post = eff.mean(axis=0)
    lo, hi = np.percentile(eff, [5, 95], axis=0)
    ok = [j for j in range(G) if n[j] > 0]
    x_kind = "n" if len({n[j] for j in ok}) > 1 else ("se" if all(se[j] for j in ok) else "rank")
    return {
        "effect": a,
        "y": ylab,
        "labels": [fit.labels(a, G)[j] for j in ok],
        "raw": [raw[j] for j in ok],
        "n": [n[j] for j in ok],
        "se": [se[j] for j in ok],
        "post": [float(post[j]) for j in ok],
        "post_lo": [float(lo[j]) for j in ok],
        "post_hi": [float(hi[j]) for j in ok],
        "pop": pop,
        "x_kind": x_kind,
    }


# ---------------------------------------------------------------------------------- regressions
def _group_index(pred: Predictor, effect: str | None = None) -> tuple[str, list]:
    idx = sorted({i for t in pred.terms for i in t.index if (effect is None or effect in t.params)})
    if len(idx) != 1:
        raise Unavailable("needs exactly one grouping index")
    return idx[0], [t for t in pred.terms if idx[0] in t.index]


def _covariate(fit: Fit, pred: Predictor) -> tuple[str, int | None, str]:
    """(data name, column or None, label) of the covariate to put on the x axis: the first
    per-observation data vector in a product term, else column 0 of a design matrix."""
    mats = []
    for t in pred.terms:
        if t.kind != "product":
            continue
        for d in t.data:
            v = fit.value(d)
            if v.ndim == 1 and v.shape[0] == pred.n_obs and np.unique(v).size > 1:
                return d, None, d
            if v.ndim == 2 and v.shape[0] == pred.n_obs:
                mats.append(d)
    if mats:
        d = mats[0]
        node = fit.ir.node(d)
        lab = d
        if node is not None and len(node.dims) == 2 and node.coords and node.dims[1] in node.coords:
            lab = f"{d}[{node.coords[node.dims[1]][0]}]"
        return d, 0, lab
    raise Unavailable("no data covariate multiplies a parameter in this predictor")


def _set_covariate(fit, name, col, value) -> dict:
    """The observed data with ONE covariate set to ``value`` in every row. Everything else —
    other covariates, group indices, unnamed per-row constants — stays as observed, so a curve
    built from it is averaged over the data actually seen (an average predictive comparison)
    rather than evaluated at an invented "mean" observation."""
    v = np.array(fit.value(name), float)
    if col is None:
        v[:] = value
    else:
        v[:, col] = value
    return {name: v}


def group_lines(fit: Fit, b: BlockIR) -> dict:
    """Gelman & Hill's radon figure: for a few groups, the data, the complete-pooling line (one
    least-squares line for everyone), the no-pooling line (least squares within the group), and
    the model's partial-pooling line at the posterior mean."""
    pred = _regression_predictor(fit, b)
    y_raw = fit.observed(pred.y).reshape(-1)
    y, ylab = _y_scale(pred, y_raw)
    iname, _ = _group_index(pred)
    xname, col, xlab = _covariate(fit, pred)
    xv = fit.value(xname)
    x = xv if col is None else xv[:, col]
    codes = fit.value(iname).astype(int).reshape(-1)
    G = int(codes.max()) + 1
    means = _means(fit, pred.leaves)
    lo, hi = float(x.min()), float(x.max())
    if not hi > lo:
        raise Unavailable("the covariate does not vary")
    grid = np.array([lo, hi])
    # the model's line for each group, at both ends: every row's predictor with the covariate set
    # to the end value, averaged over that group's rows (its other covariates as observed)
    ends = [pred.evaluate(fit, {**means, **_set_covariate(fit, xname, col, g)}) for g in grid]
    ends = [e.reshape(-1) for e in ends]
    cp = np.polyfit(x, y, 1)
    counts = np.bincount(codes, minlength=G)
    order = np.argsort(counts, kind="stable")
    pick = order[np.unique(np.linspace(0, G - 1, min(8, G)).round().astype(int))]
    effect = next((p for t in pred.terms if iname in t.index for p in t.params), None)
    labels = fit.labels(effect, G) if effect else [str(j) for j in range(G)]
    panels = []
    for j in sorted(pick.tolist(), key=lambda j: -counts[j]):
        m = codes == j
        if not m.any():
            continue
        partial = [float(e[m].mean()) for e in ends]
        nopool = None
        if m.sum() >= 2 and np.unique(x[m]).size > 1:
            c = np.polyfit(x[m], y[m], 1)
            nopool = [float(np.polyval(c, g)) for g in grid]
        panels.append(
            {
                "label": labels[j],
                "n": int(m.sum()),
                "x": x[m].tolist(),
                "y": y[m].tolist(),
                "partial": partial,
                "nopool": nopool,
            }
        )
    return {
        "x": xlab,
        "y": ylab,
        "grid": grid.tolist(),
        "complete": [float(np.polyval(cp, g)) for g in grid],
        "groups": panels,
        "ylim": [float(y.min()), float(y.max())],
        "n_groups": G,
    }


def _regression_predictor(fit: Fit, b: BlockIR) -> Predictor:
    if b.kind == "regression":
        return predictor(fit, b.anchor)
    cons = _consumers_of(fit, b.anchor)
    if not cons:
        raise Unavailable(f"no observed likelihood uses {b.anchor!r}")
    return predictor(fit, cons[0])


def intercept_slope(fit: Fit, b: BlockIR) -> dict:
    """Varying intercepts against varying slopes, one point per group with 90% intervals — the
    structure a correlated-effects model would add a covariance for."""
    pred = _regression_predictor(fit, b)
    _, terms = _group_index(pred)
    inter = [t for t in terms if t.kind == "gather" and len(t.params) == 1]
    slope = [t for t in terms if t.kind == "product" and len(t.params) == 1]
    if not inter or not slope:
        raise Unavailable("needs both a varying intercept and a varying slope on one index")
    a, s = inter[0].params[0], slope[0].params[0]
    da, ds = fit.draws(a, None), fit.draws(s, None)
    if da.shape != ds.shape or da.ndim != 2:
        raise Unavailable("intercepts and slopes must be one value per group each")
    ma, ms = da.mean(axis=0), ds.mean(axis=0)
    la, ha = np.percentile(da, [5, 95], axis=0)
    ls, hs = np.percentile(ds, [5, 95], axis=0)
    r = float(np.corrcoef(ma, ms)[0, 1]) if ma.size > 2 else None
    return {
        "a": a,
        "b": s,
        "labels": fit.labels(a, ma.size),
        "ma": ma.tolist(),
        "mb": ms.tolist(),
        "a_lo": la.tolist(),
        "a_hi": ha.tolist(),
        "b_lo": ls.tolist(),
        "b_hi": hs.tolist(),
        "r_means": r,
    }


def _inverse_link(link: str, eta: np.ndarray) -> np.ndarray:
    if link == "identity":
        return eta
    if link == "Exp":
        return np.exp(eta)
    if link == "Sigmoid":
        return 1.0 / (1.0 + np.exp(-eta))
    if link == "probit":
        from scipy.special import erf

        return 0.5 * (1.0 + erf(eta / np.sqrt(2.0)))
    raise Unavailable(f"no inverse link for {link}")


def effect(fit: Fit, b: BlockIR) -> dict:
    """The fitted mean against one covariate, on the OUTCOME's scale (the link undone), with a
    90% credible band and the data behind it. The curve is an average predictive comparison:
    every observed row with this covariate set to the grid value, averaged — so the other
    covariates and the groups are as observed, never an invented "average observation". For a
    LogNormal outcome the median exp(eta) and the mean exp(eta + sigma^2/2) are different curves,
    and both are drawn."""
    pred = _regression_predictor(fit, b)
    if pred.link == "Softmax":
        raise Unavailable("a multinomial-logit effect has one curve per category")
    xname, col, xlab = _covariate(fit, pred)
    xv = fit.value(xname)
    x = xv if col is None else xv[:, col]
    grid = np.linspace(float(x.min()), float(x.max()), 30)
    draws = {n: fit.draws(n, 100) for n in pred.leaves}
    S = min(v.shape[0] for v in draws.values())
    # rows x draws x grid: each draw's predictor at every observed row with the covariate set to
    # the grid value; the curve is the row average (other covariates and groups as observed)
    rows = [_set_covariate(fit, xname, col, g) for g in grid]
    eta = np.array(
        [
            [
                pred.evaluate(fit, {**{n: v[s] for n, v in draws.items()}, **r}).reshape(-1)
                for r in rows
            ]
            for s in range(S)
        ]
    )  # (S, G, n)
    y_raw = fit.observed(pred.y).reshape(-1)
    note = None
    mean_curve = None
    if pred.family == "LogNormal":
        med = np.exp(eta)
        slot = dist_params(fit.model[pred.y])[_SCALE_SLOT["LogNormal"]]
        u = B._unwrap(slot, fit.ctx)
        sig = None
        if not depends_on_rv(slot):
            a = np.asarray(slot.eval(), float)
            sig = float(a.reshape(-1)[0]) if a.size and np.all(a == a.reshape(-1)[0]) else None
        elif id(u) in fit.ctx.named and fit.ctx.dims(fit.ctx.named[id(u)]) == ():
            sig = fit.draws(fit.ctx.named[id(u)], 100)[:S].reshape(-1, 1)
        else:
            sig_node = fit.ctx.named.get(id(u)) if id(u) in fit.ctx.named else None
            if sig_node and fit.has(sig_node):  # a per-observation sd: exact row by row …
                if any(a is fit.model[xname] for a in _ancestors(fit.model[sig_node])):
                    note = "mean curve omitted: sigma itself depends on this covariate"
                else:  # … unless it moves with the covariate we are varying
                    sig = fit.draws(sig_node, 100)[:S].reshape(S, 1, -1)
        if sig is not None:
            sig = np.asarray(sig, float)
            if sig.ndim == 2:
                sig = sig.reshape(S, 1, 1)
            mean_curve = np.median(np.exp(eta + np.square(sig) / 2.0).mean(axis=2), axis=0)
            mean_curve = mean_curve.tolist()
        elif note is None:
            note = "mean curve omitted: sigma has no draws"
        mu = med.mean(axis=2)
    else:
        mu = _inverse_link(pred.link, eta).mean(axis=2)
    y_plot = y_raw
    if pred.family == "Binomial":
        ntr = dist_params(fit.model[pred.y])[0]
        if depends_on_rv(ntr):
            raise Unavailable("binomial trials are not fixed")
        y_plot = y_raw / np.broadcast_to(np.asarray(ntr.eval(), float), y_raw.shape)
    keep = np.unique(np.linspace(0, x.size - 1, min(400, x.size)).round().astype(int))
    q05, q50, q95 = np.percentile(mu, [5, 50, 95], axis=0)
    return {
        "x": xlab,
        "y": pred.y,
        "family": pred.family,
        "link": pred.link,
        "grid": grid.tolist(),
        "lo": q05.tolist(),
        "mid": q50.tolist(),
        "hi": q95.tolist(),
        "mean": mean_curve,
        "pts_x": x[keep].tolist(),
        "pts_y": y_plot[keep].tolist(),
        "binary": pred.family == "Bernoulli",
        "note": note,
    }


# ---------------------------------------------------------------------------------- shrinkage
def _scale_fn(fit: Fit, b: BlockIR):
    """(function of the scale's named leaves -> per-coefficient conditional scale, leaves)."""
    ctx = fit.ctx
    ctx.visits[0] = 0
    a = b.anchor
    node = fit.ir.node(a)
    if node.role == "deterministic":
        nc = B._noncentered(a, ctx)
        if nc is None:
            raise Unavailable("the coefficient's scale could not be isolated")
        expr = None
        for e, t in nc.aff.scale:
            f = t**e if e != 1.0 else t
            expr = f if expr is None else expr * f
        expr = expr * nc.aux_info["scale"]
        center = None
        for s, t in nc.aff.loc:
            center = (s * t) if center is None else center + s * t
    else:
        dp = dist_params(fit.model[a])
        expr, center = dp[1], (dp[0] if depends_on_rv(dp[0]) else None)
    leaves = sorted({n for n in B._named_leaves(expr, ctx) if ctx.role(n) != "data"})
    f = _compile(expr, fit.model, leaves)
    cf, cleaves = None, []
    if center is not None:
        cleaves = sorted({n for n in B._named_leaves(center, ctx) if ctx.role(n) != "data"})
        cf = _compile(center, fit.model, cleaves)
    return f, leaves, cf, cleaves


def _scales(fit, b, S=MAX_DRAWS):
    f, leaves, cf, cleaves = _scale_fn(fit, b)
    draws = {n: fit.draws(n, S) for n in set(leaves) | set(cleaves)}
    n = min(v.shape[0] for v in draws.values())
    beta = fit.draws(b.anchor, S)[:n]
    P = beta.shape[1]
    sc = np.array(
        [np.broadcast_to(f(*[draws[k][i] for k in leaves]), (P,)) for i in range(n)], float
    )
    cen = None
    if cf is not None:
        cen = np.array(
            [np.broadcast_to(cf(*[draws[k][i] for k in cleaves]), (P,)) for i in range(n)], float
        )
    return np.abs(sc), beta, cen


def kappa(fit: Fit, b: BlockIR) -> dict:
    """The horseshoe's shrinkage factors ``kappa_j = 1 / (1 + s_j^2 sum_i x_ij^2 / sigma_i^2)``
    (Piironen & Vehtari 2017), draw by draw: near 1 = shrunk away, near 0 = left alone. Exact
    for an orthogonal design; for correlated predictors it is the standard approximation, and
    the caption says so."""
    a = b.anchor
    cons = _consumers_of(fit, a)
    if not cons:
        raise Unavailable(f"no observed likelihood uses {a!r}")
    pred = predictor(fit, cons[0])
    if pred.link != "identity" or pred.family not in ("Normal", "StudentT", "LogNormal"):
        raise Unavailable("kappa is defined for a Gaussian (or log-Gaussian) likelihood")
    term = next((t for t in pred.terms if a in t.params and t.kind == "product"), None)
    if term is None or term.design is None or depends_on_rv(term.design):
        raise Unavailable(f"{a!r} is not multiplied by a fixed design")
    X = np.asarray(term.design.eval(), float)
    if X.ndim == 1:
        X = X[:, None]
    sc, _, _ = _scales(fit, b)
    if X.shape[1] != sc.shape[1]:
        raise Unavailable("the design does not match the coefficients")
    slot = dist_params(fit.model[pred.y])[_SCALE_SLOT[pred.family]]
    u = B._unwrap(slot, fit.ctx)
    S = sc.shape[0]
    if not depends_on_rv(slot):
        inv = np.broadcast_to(1.0 / np.square(np.asarray(slot.eval(), float)), (X.shape[0],))
        w = np.broadcast_to((np.square(X) * inv[:, None]).sum(axis=0), (S, X.shape[1]))
    elif id(u) in fit.ctx.named and fit.has(fit.ctx.named[id(u)]):
        sig = fit.draws(fit.ctx.named[id(u)], MAX_DRAWS)[:S].reshape(S, -1)
        sig = np.broadcast_to(sig, (S, X.shape[0]))
        w = (1.0 / np.square(sig)) @ np.square(X)
    else:
        raise Unavailable("the residual scale has no draws")
    k = 1.0 / (1.0 + np.square(sc) * w)
    hist, edges = np.histogram(k.reshape(-1), bins=20, range=(0.0, 1.0))
    return {
        "coef": a,
        "hist": (hist / max(hist.max(), 1)).tolist(),
        "edges": edges.tolist(),
        "mean": k.mean(axis=0).tolist(),
        "labels": fit.labels(a, k.shape[1]),
        "m_eff": float((1.0 - k).sum(axis=1).mean()),
        "orthogonal": bool(_near_orthogonal(X)),
    }


def _near_orthogonal(X: np.ndarray) -> bool:
    if X.shape[1] < 2:
        return True
    c = np.corrcoef(X, rowvar=False)
    off = np.abs(c[~np.eye(c.shape[0], dtype=bool)])
    return bool(np.nanmax(off) < 0.3)


def local_scale(fit: Fit, b: BlockIR) -> dict:
    """Each coefficient's conditional prior scale ``s_j = tau * lambda_tilde_j`` against its
    posterior (deviation from the center), with the slab's ceiling on ``s`` marked: coefficients
    that escaped shrinkage sit at large ``s``, and the slab caps their SCALE, not their size."""
    sc, beta, cen = _scales(fit, b)
    dev = beta - cen if cen is not None else beta
    c = None
    slab = (b.roles.get("slab") or [None])[0]
    if slab:
        v = fit.draws(slab, None)
        power = (b.data or {}).get("slab_power") or 1.0
        c = float(np.median(np.sqrt(v) if power == 1.0 else v))
    lo, mid, hi = np.percentile(dev, [5, 50, 95], axis=0)
    return {
        "coef": b.anchor,
        "labels": fit.labels(b.anchor, beta.shape[1]),
        "s": np.median(sc, axis=0).tolist(),
        "mid": mid.tolist(),
        "lo": lo.tolist(),
        "hi": hi.tolist(),
        "c": c,
        "centered": cen is not None,
    }


# ---------------------------------------------------------------------------------- measured
def measured(fit: Fit, b: BlockIR) -> dict:
    """Each noisy reading against the inferred true value (90% interval), on the 1:1 line: how
    far the measurement model moved each observation."""
    latent = b.roles["latent"][0]
    reading = b.roles["reading"][0]
    obs = fit.observed(reading).reshape(-1)
    d = fit.draws(latent, None)
    d = d.reshape(d.shape[0], -1)
    if d.shape[1] != obs.size:
        raise Unavailable("readings and latent values do not line up")
    lo, hi = np.percentile(d, [5, 95], axis=0)
    return {
        "latent": latent,
        "reading": reading,
        "obs": obs.tolist(),
        "post": d.mean(axis=0).tolist(),
        "lo": lo.tolist(),
        "hi": hi.tolist(),
    }


# ---------------------------------------------------------------------------------- registry
PANELS: dict[str, tuple[str, Any]] = {
    "shrinkage": ("shrinkage toward the population", shrinkage),
    "population": ("population distribution", population),
    "pooling": ("how much pooling", pooling),
    "group_lines": ("lines per group", group_lines),
    "intercept_slope": ("intercepts vs slopes", intercept_slope),
    "effect": ("fitted effect", effect),
    "kappa": ("shrinkage factors (kappa)", kappa),
    "local_scale": ("local scale vs coefficient", local_scale),
    "measured": ("readings vs true values", measured),
}


def applicable(b: BlockIR, ir: ModelIR) -> list[str]:
    """The fitted panels worth offering for ``b`` — a cheap structural check; computing one may
    still end in :class:`Unavailable` with a reason."""
    if b.kind in ("hierarchy",) and b.anchor:
        out = ["population", "pooling", "shrinkage"]
        if _has_varying_slope(ir, b.anchor):
            out.append("intercept_slope")
        return out
    if b.kind == "regression":
        terms = (b.data or {}).get("terms") or []
        out = ["effect"]
        if any(t.get("index") for t in terms):
            out.append("group_lines")
        if _varying_pair(terms):
            out.append("intercept_slope")
        return out
    if b.kind == "shrinkage":
        return ["kappa", "local_scale"]
    if b.kind == "measured":
        return ["measured"]
    return []


def _varying_pair(terms) -> bool:
    gi = {i for t in terms if t.get("kind") == "gather" for i in t.get("index", [])}
    pi = {i for t in terms if t.get("kind") == "product" for i in t.get("index", [])}
    return bool(gi & pi)


def _has_varying_slope(ir: ModelIR, anchor: str) -> bool:
    for b in ir.blocks:
        if b.kind != "regression":
            continue
        terms = (b.data or {}).get("terms") or []
        if _varying_pair(terms) and any(anchor in t.get("params", []) for t in terms):
            return True
    return False


def compute(kind: str, model: Any, ir: ModelIR, idata: Any, b: BlockIR) -> dict:
    if kind not in PANELS:
        raise ValueError(f"unknown fitted panel {kind!r}")
    fit = Fit(model, ir, idata)
    return PANELS[kind][1](fit, b)
