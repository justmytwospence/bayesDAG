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
* Nothing samples: values come from the caller's idata or immutable graph constants.
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


@dataclass(frozen=True)
class _Axis:
    """One axis in a saved array. The model provides its name, not its labels."""

    name: str | None
    size: int
    labels: Any  # pandas.Index or None (positional)

    def indexer(self, source: _Axis, what: str):
        if self.name is not None and source.name is not None and self.name != source.name:
            raise Unavailable(f"{what} has incompatible axis names: {source.name!r}, {self.name!r}")
        if (self.labels is None) != (source.labels is None):
            raise Unavailable(f"{what} cannot pair labelled and positional axes")
        if self.labels is None:
            if self.size != source.size:
                raise Unavailable(f"{what} has incompatible positional lengths")
            return slice(None)
        if not self.labels.is_unique or not source.labels.is_unique:
            raise Unavailable(f"{what} has duplicate labels")
        if self.size != source.size:
            raise Unavailable(f"{what} has incompatible coordinate labels")
        if source.labels.equals(self.labels):
            return slice(None)
        try:
            positions = source.labels.get_indexer(self.labels)
        except (TypeError, ValueError):
            raise Unavailable(f"{what} has incompatible coordinate labels") from None
        if np.any(positions < 0):
            raise Unavailable(f"{what} has incompatible coordinate labels")
        return positions

    def text_labels(self) -> list[str]:
        values = self.labels if self.labels is not None else range(self.size)
        return [str(x) for x in values]


def _axis(da, axis: str, name: str | None, var: str, i: int) -> _Axis:
    import pandas as pd

    size = da.sizes[axis]
    labels = pd.Index(da.coords[axis].values) if axis in da.coords else None
    if (
        labels is not None
        and name is None
        and axis == f"{var}_dim_{i}"
        and labels.equals(pd.RangeIndex(size))
    ):
        labels = None  # PyMC/ArviZ's generated default coordinate is not a row identifier.
    if labels is not None and not labels.is_unique:
        raise Unavailable(f"{var!r} axis {axis!r} has duplicate labels")
    return _Axis(name, size, labels)


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
        self.ctx = _FitCtx(self.model, named, self.ir, consumers, fit=self)

    # ---- draws and data
    def has(self, name: str) -> bool:
        post = self.idata.posterior
        return name in post

    def _dims(self, name: str, ndim: int) -> tuple:
        return tuple(self.model.named_vars_to_dims.get(name, ())) or (None,) * ndim

    def _saved_dataarray(self, name: str, group: str):
        from .glyph_data import _group_draws

        samples = group == "posterior"
        ds = getattr(self.idata, group, None)
        da = (
            _group_draws(self.idata, group, name)
            if samples
            else (ds[name] if ds is not None and name in ds else None)
        )
        if da is None:
            if samples:
                raise Unavailable(f"the posterior has no draws for {name!r}")
            raise Unavailable(f"the fit has no saved {group} for {name!r}")
        prefix = ("chain", "draw") if samples else ()
        axes = da.dims[len(prefix) :]
        dims = self._dims(name, len(axes))
        declared = {d for d in dims if d is not None}
        if len(dims) != len(axes) or not declared.issubset(axes):
            raise Unavailable(f"{name!r} has incompatible axis names")
        # Restore graph axis order, including unnamed axes when PyMC's generated names are saved.
        order = [d if d is not None else f"{name}_dim_{i}" for i, d in enumerate(dims)]
        spare = iter(d for d in axes if d not in order)
        order = [d if d in axes else next(spare) for d in order]
        if tuple(order) != axes:
            da = da.transpose(*prefix, *order)
        return da, dims

    def group_axis(self, name: str) -> _Axis:
        """Integer gathers index the stored fit axis, not the current model's coordinates."""
        da, dims = self._saved_dataarray(name, "posterior")
        if not dims:
            raise Unavailable(f"{name!r} has no saved group axis")
        axis = _axis(da, da.dims[2], dims[0], name, 0)
        dim = axis.name
        if dim is not None:
            if axis.labels is None:
                raise Unavailable(f"{name!r} has no saved fit-time group coordinates")
            # If another fit snapshot records this dimension, its order must agree. Unlike
            # ordinary labelled joins, reordering either sequence here could reinterpret codes.
            for group in ("constant_data", "observed_data"):
                ds = getattr(self.idata, group, None)
                if ds is not None and dim in ds.dims and dim in ds.coords:
                    saved = _axis(ds, dim, dim, name, 0)
                    if not axis.labels.equals(saved.labels):
                        raise Unavailable(f"{name!r} has conflicting fit-time group coordinates")
        return axis

    def _dataarray(self, name: str, group: str, *, rows: _Axis | None = None):
        da, dims = self._saved_dataarray(name, group)
        order = da.dims[2:] if group == "posterior" else da.dims
        node = self.ir.node(name)
        coords = node.coords if node is not None and node.coords else {}
        for i, (axis, dim) in enumerate(zip(order, dims, strict=True)):
            source = _axis(da, axis, dim, name, i)
            target = rows if i == 0 else None
            if target is None and dim in coords:
                import pandas as pd

                labels = pd.Index(coords[dim])
                target = _Axis(dim, len(labels), labels)
            if target is not None:
                da = da.isel({axis: target.indexer(source, f"{name!r} axis {axis!r}")})
        if rows is not None and not order:
            raise Unavailable(f"{name!r} has no observation axis")
        return da

    def draws(
        self, name: str, n: int | None = MAX_DRAWS, *, rows: _Axis | None = None
    ) -> np.ndarray:
        """``(S, *element shape)`` posterior draws of ``name`` (chain/draw flattened, in
        coordinate order), thinned to at most ``n`` evenly spaced draws."""
        v = np.asarray(self._dataarray(name, "posterior", rows=rows).values, float)
        v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
        if n is not None and v.shape[0] > n:
            v = v[np.unique(np.linspace(0, v.shape[0] - 1, n).round().astype(int))]
        return v

    def mean(self, name: str) -> np.ndarray:
        return self.draws(name, None).mean(axis=0)

    def value(self, name: str, *, rows: _Axis | None = None) -> np.ndarray:
        """The fit-time snapshot of a named data variable."""
        v = self.model[name]
        if depends_on_rv(v):
            raise Unavailable(f"{name!r} is not data")
        return np.asarray(self._dataarray(name, "constant_data", rows=rows).values)

    def eval_fixed(self, expr, *, data: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Evaluate an RV-free expression using only saved data, never live shared values."""
        import pytensor
        import pytensor.tensor as pt
        from pytensor.compile.sharedvalue import SharedVariable
        from pytensor.graph.replace import graph_replace

        ancestors = _ancestors(expr)
        replacements = {}
        for name, var in self.model.named_vars.items():
            if var in getattr(self.model, "data_vars", ()) and any(a is var for a in ancestors):
                value = data[name] if data is not None and name in data else self.value(name)
                replacements[var] = pt.as_tensor_variable(value)
        try:
            out = graph_replace(expr, replacements, strict=False)
            if depends_on_rv(out) or any(isinstance(a, SharedVariable) for a in _ancestors(out)):
                raise Unavailable(
                    "fixed expression depends on an unbound shared value or random variable"
                )
            return np.asarray(pytensor.function([], out)())
        except Unavailable:
            raise
        except Exception as exc:
            raise Unavailable("the fixed expression cannot be evaluated from saved data") from exc

    def observed(self, name: str) -> np.ndarray:
        """The fit-time observations, in captured IR order when available."""
        return np.asarray(self._dataarray(name, "observed_data").values, float)

    def observation_rows(self, name: str):
        da = self._dataarray(name, "observed_data")
        if not da.ndim:
            raise Unavailable(f"{name!r} has no observation axis")
        dim = self._dims(name, da.ndim)[0]
        return da, _axis(da, da.dims[0], dim, name, 0)

    def labels(self, name: str, k: int) -> list[str]:
        n = self.ir.node(name)
        if n is not None and n.dims and n.coords and n.dims[0] in n.coords:
            vals = n.coords[n.dims[0]]
            if len(vals) == k:
                return [str(x) for x in vals]
        return [str(i) for i in range(k)]


@dataclass(kw_only=True)
class _FitCtx(B._Ctx):
    """The detection context, able to see variables the DISPLAYED IR left out (index data folded
    into equations is still data the panels must read)."""

    fit: Fit

    def fixed_value(self, expr):
        return None if depends_on_rv(expr) else self.fit.eval_fixed(expr)

    def slot_values(self, var):
        return [self.fixed_value(slot) for slot in dist_params(var)]

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


def _compile(expr, fit: Fit, names: list[str]):
    """A function of the named variables ``names`` (in order) computing ``expr``, with everything
    else fixed. Refuses a graph that would still draw random numbers."""
    import pytensor
    import pytensor.tensor as pt
    from pytensor.graph.replace import graph_replace

    model = fit.model
    vs = [model[n] for n in names]
    phs = [
        pt.tensor(dtype=v.dtype, shape=(None,) * v.ndim, name=n)
        for v, n in zip(vs, names, strict=True)
    ]
    replacements = dict(zip(vs, phs, strict=True))
    ancestors = _ancestors(expr)
    for name, var in model.named_vars.items():
        if (
            var in getattr(model, "data_vars", ())
            and var not in replacements
            and any(a is var for a in ancestors)
        ):
            replacements[var] = pt.as_tensor_variable(fit.value(name))
    out = graph_replace(expr, replacements, strict=False)
    from pytensor.compile.sharedvalue import SharedVariable

    if any(isinstance(a, SharedVariable) for a in _ancestors(out)):
        raise Unavailable("the predictor depends on an unbound shared value")
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
    data: dict[str, np.ndarray]
    row_params: set[str]
    group_axes: dict[str, _Axis]
    rows: _Axis
    observed: np.ndarray
    fn: Any = None

    @property
    def n_obs(self) -> int:
        return self.rows.size

    def draws(self, fit: Fit, name: str, n: int | None) -> np.ndarray:
        axis = self.rows if name in self.row_params else self.group_axes.get(name)
        return fit.draws(name, n, rows=axis)

    def group_codes(self, index: str, *effects: str) -> np.ndarray:
        codes = self.data[index]
        if (
            codes.shape != (self.n_obs,)
            or codes.dtype.kind not in "iu"
            or np.any(codes < 0)
            or any(
                p not in self.group_axes or np.any(codes >= self.group_axes[p].size)
                for p in effects
            )
        ):
            raise Unavailable(f"{index!r} does not index the saved group axes")
        return codes.astype(int)

    def means(self, fit: Fit) -> dict[str, np.ndarray]:
        return {n: self.draws(fit, n, None).mean(axis=0) for n in self.leaves}

    def evaluate(self, fit: Fit, values: dict) -> np.ndarray:
        """Evaluate with supplied parameters/overrides and the row-aligned saved data."""
        if self.fn is None:
            self.fn = _compile(self.eta, fit, self.leaves + list(self.data))
        args = [values.get(n, v) for n, v in self.data.items()]
        out = np.asarray(self.fn(*[values[n] for n in self.leaves], *args), float)
        if out.ndim > 1 or out.size not in (1, self.n_obs):
            raise Unavailable(f"{self.y!r}'s predictor does not yield one value per observation")
        return np.broadcast_to(out, (self.n_obs,))


def _predictor_graph(fit: Fit, y: str):
    node = fit.ir.node(y)
    if node is None or node.role != "observed" or node.dist not in _LOC_SLOT:
        raise Unavailable(f"{y!r} has no recognized location")
    dp = dist_params(fit.model[y])
    ctx = fit.ctx
    ctx.visits[0] = 0
    location = dp[_LOC_SLOT[node.dist]]
    link, eta = B._link(location, ctx)
    if link == "identity":
        eta = location  # Keep the graph's shape wrappers when evaluating it.
    elif B._unwrap(location, ctx, preserve_axes=True) is not B._unwrap(location, ctx):
        raise Unavailable(f"{y!r}'s link does not preserve the observation axis")
    blocked = {b.anchor for b in fit.ir.blocks if b.anchor and b.kind != "regression"}
    terms = B._affine_terms(eta, ctx, blocked)
    if not terms:
        raise Unavailable(f"{y!r}'s location is not affine in its parameters")
    return node.dist, link, eta, terms


def _bind_predictor(fit: Fit, y: str, family: str, link: str, eta, terms) -> Predictor:
    da, rows = fit.observation_rows(y)
    if da.ndim != 1:
        raise Unavailable(f"{y!r} needs one outcome value per observation row")
    leaves = sorted({p for t in terms for p in t.params})
    row_params = {p for t in terms for p in t.row_params}
    if any((row_params & set(t.params)) - set(t.row_params) for t in terms):
        raise Unavailable(f"{y!r} uses a parameter both row-wise and outside the observation axis")
    for p in sorted(row_params):
        fit._dataarray(p, "posterior", rows=rows)
    grouped = {p for t in terms if t.index for p in t.params if p not in row_params}
    group_axes = {p: fit.group_axis(p) for p in sorted(grouped) if fit.model[p].ndim}
    row_data = {d for t in terms for d in t.row_data}
    unproved = {d for t in terms for d in (t.index + t.data) if d not in t.row_data}
    data = {}
    for d in sorted(row_data | unproved):
        if d in unproved and fit.model[d].ndim:
            raise Unavailable(f"{d!r} is not a direct row-wise input to {y!r}")
        data[d] = fit.value(d, rows=rows if d in row_data else None)
    return Predictor(
        y,
        family,
        link,
        eta,
        terms,
        leaves,
        data,
        row_params,
        group_axes,
        rows,
        np.asarray(da.values, float),
    )


def predictor(fit: Fit, y: str) -> Predictor:
    """Recognize the affine predictor and bind its direct inputs to this likelihood's rows."""
    return _bind_predictor(fit, y, *_predictor_graph(fit, y))


def _consumer_predictor(fit: Fit, effect: str) -> Predictor:
    """First usable likelihood that directly uses ``effect``; unrelated fits cannot poison it."""
    error = None
    for n in fit.ir.nodes:
        if n.role != "observed" or n.dist not in _LOC_SLOT:
            continue
        try:
            graph = _predictor_graph(fit, n.id)
        except (Unavailable, B._Budget):
            continue
        if any(effect in t.params for t in graph[-1]):
            try:
                return _bind_predictor(fit, n.id, *graph)
            except Unavailable as exc:
                if error is None:
                    error = exc
    if error is not None:
        raise error
    raise Unavailable(f"no observed likelihood uses {effect!r} directly")


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


def _fixed_likelihood_value(fit: Fit, pred: Predictor, expr) -> np.ndarray:
    """A fixed likelihood slot in observation order (e.g. errors or binomial trials)."""
    direct = B._direct_data(expr, fit.ctx)
    if expr.ndim == 1 and direct:
        name = direct[0]
        return fit.eval_fixed(expr, data={name: fit.value(name, rows=pred.rows)})
    value = fit.eval_fixed(expr)
    if value.size > 1 and any(
        v in getattr(fit.model, "data_vars", ()) and v.ndim for v in _ancestors(expr)
    ):
        raise Unavailable("the fixed likelihood parameter is not a direct row-wise input")
    if value.size > 1 and value.shape != (pred.n_obs,):
        raise Unavailable("the fixed likelihood parameter does not match the observation rows")
    return value


def _obs_sd(fit: Fit, pred: Predictor) -> np.ndarray | None:
    """Per-observation residual sd at its posterior mean (or its fixed value), if knowable."""
    slot = _SCALE_SLOT.get(pred.family)
    if slot is None:
        return None
    t = dist_params(fit.model[pred.y])[slot]
    if not depends_on_rv(t):
        return np.broadcast_to(_fixed_likelihood_value(fit, pred, t), (pred.n_obs,)).copy()
    u = B._unwrap(t, fit.ctx, preserve_axes=True)
    if u.ndim <= 1 and id(u) in fit.ctx.named and fit.has(fit.ctx.named[id(u)]):
        m = fit.draws(fit.ctx.named[id(u)], None, rows=pred.rows if u.ndim else None).mean(axis=0)
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
            loc = _single(nc.aff.loc[0][1], fit, "location")
        else:
            raise Unavailable("the population location is not a single parameter")
        if len(nc.aff.scale) != 1 or nc.aff.scale[0][0] != 1.0 or nc.aff.matrix is not None:
            raise Unavailable("the population scale is not a single parameter")
        sc = _single(nc.aff.scale[0][1], fit, "scale")
        s0 = nc.aux_info["scale"]
        scale = (sc[0], sc[1], s0) if sc[0] == "param" else ("const", sc[1] * s0)
        return _Hier(a, nc.family, loc, scale)
    if node is None or node.dist not in ("Normal", "LogNormal"):
        raise Unavailable("the population is not a Normal/LogNormal")
    dp = dist_params(fit.model[a])
    return _Hier(
        a, node.dist, _single(dp[0], fit, "location"), _add_factor(_single(dp[1], fit, "scale"))
    )


def _add_factor(s):
    return (s[0], s[1], 1.0) if s[0] == "param" else s


def _single(t, fit: Fit, what: str) -> tuple:
    ctx = fit.ctx
    if not depends_on_rv(t):
        a = np.asarray(fit.eval_fixed(t), float)
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
    pred = _consumer_predictor(fit, a)
    eff = pred.draws(fit, a, None)
    if eff.ndim != 2:
        raise Unavailable("the group effect must be one value per group")
    y, ylab = _y_scale(pred, pred.observed)
    occurrences = [t for t in pred.terms if a in t.params]
    if len(occurrences) != 1 or occurrences[0].multiplier in (None, 0):
        raise Unavailable(f"{a!r} has an ambiguous or unproved predictor coefficient")
    term = occurrences[0]
    c = term.multiplier
    if term.kind == "gather" and len(term.index) == 1 and term.params == [a]:
        codes = pred.group_codes(term.index[0], a)
    elif term.kind == "leaf" and term.params == [a] and eff.shape[1] == pred.n_obs:
        codes = np.arange(pred.n_obs)
    else:
        raise Unavailable(f"{a!r} does not enter {pred.y}'s predictor as a plain group intercept")
    means = pred.means(fit)
    rest = pred.evaluate(fit, {**means, a: np.zeros_like(means[a])})
    resid = y - rest
    G = eff.shape[1]
    sd = _obs_sd(fit, pred)
    raw, n, se = [], [], []
    for j in range(G):
        m = codes == j
        n.append(int(m.sum()))
        raw.append(float(resid[m].mean() / c) if m.any() else float("nan"))
        if sd is not None and m.any():
            se.append(float(np.sqrt((sd[m] ** 2).sum()) / m.sum() / abs(c)))
        elif m.sum() >= 2:
            se.append(float(resid[m].std(ddof=1) / np.sqrt(m.sum()) / abs(c)))
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
    if a in pred.group_axes:
        labels = pred.group_axes[a].text_labels()
    elif a in pred.row_params:
        labels = pred.rows.text_labels()
    else:
        labels = fit.labels(a, G)
    return {
        "effect": a,
        "y": ylab,
        "labels": [labels[j] for j in ok],
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
            if d not in t.row_data:
                continue
            v = pred.data[d]
            if v.ndim == 1 and np.unique(v).size > 1:
                return d, None, d
            if v.ndim == 2:
                mats.append(d)
    if mats:
        d = mats[0]
        node = fit.ir.node(d)
        lab = d
        if node is not None and len(node.dims) == 2 and node.coords and node.dims[1] in node.coords:
            lab = f"{d}[{node.coords[node.dims[1]][0]}]"
        return d, 0, lab
    raise Unavailable("no data covariate multiplies a parameter in this predictor")


def _set_covariate(pred, name, col, value) -> dict:
    """The observed data with ONE covariate set to ``value`` in every row. Everything else —
    other covariates, group indices, unnamed per-row constants — stays as observed, so a curve
    built from it is averaged over the data actually seen (an average predictive comparison)
    rather than evaluated at an invented "mean" observation."""
    v = np.array(pred.data[name], float)
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
    y, ylab = _y_scale(pred, pred.observed)
    iname, group_terms = _group_index(pred)
    group_params = [p for t in group_terms for p in t.params]
    if (
        not group_params
        or len(group_params) != len(set(group_params))
        or any(t.multiplier != 1 for t in group_terms)
    ):
        raise Unavailable("group effects need unambiguous unit coefficients")
    xname, col, xlab = _covariate(fit, pred)
    xv = pred.data[xname]
    x = xv if col is None else xv[:, col]
    codes = pred.group_codes(iname, *group_params)
    G = int(codes.max()) + 1
    means = pred.means(fit)
    lo, hi = float(x.min()), float(x.max())
    if not hi > lo:
        raise Unavailable("the covariate does not vary")
    grid = np.array([lo, hi])
    # the model's line for each group, at both ends: every row's predictor with the covariate set
    # to the end value, averaged over that group's rows (its other covariates as observed)
    ends = [pred.evaluate(fit, {**means, **_set_covariate(pred, xname, col, g)}) for g in grid]
    ends = [e.reshape(-1) for e in ends]
    cp = np.polyfit(x, y, 1)
    counts = np.bincount(codes, minlength=G)
    order = np.argsort(counts, kind="stable")
    pick = order[np.unique(np.linspace(0, G - 1, min(8, G)).round().astype(int))]
    labels = pred.group_axes[group_params[0]].text_labels()
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
    return _consumer_predictor(fit, b.anchor)


def intercept_slope(fit: Fit, b: BlockIR) -> dict:
    """Varying intercepts against varying slopes, one point per group with 90% intervals — the
    structure a correlated-effects model would add a covariance for."""
    pred = _regression_predictor(fit, b)
    _, terms = _group_index(pred)
    inter = [t for t in terms if t.kind == "gather" and len(t.params) == 1]
    slope = [t for t in terms if t.kind == "product" and len(t.params) == 1]
    if not inter or not slope:
        raise Unavailable("needs both a varying intercept and a varying slope on one index")
    if len(inter) != 1 or len(slope) != 1 or inter[0].multiplier != 1 or slope[0].multiplier != 1:
        raise Unavailable("intercepts and slopes need unambiguous unit coefficients")
    a, s = inter[0].params[0], slope[0].params[0]
    da, ds = pred.draws(fit, a, None), pred.draws(fit, s, None)
    if a not in pred.group_axes or s not in pred.group_axes or da.shape != ds.shape or da.ndim != 2:
        raise Unavailable("intercepts and slopes must be one value per group each")
    ma, ms = da.mean(axis=0), ds.mean(axis=0)
    la, ha = np.percentile(da, [5, 95], axis=0)
    ls, hs = np.percentile(ds, [5, 95], axis=0)
    r = float(np.corrcoef(ma, ms)[0, 1]) if ma.size > 2 else None
    return {
        "a": a,
        "b": s,
        "labels": pred.group_axes[a].text_labels(),
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
    and both are drawn when sigma is known and does not depend on the varied covariate."""
    pred = _regression_predictor(fit, b)
    if pred.link == "Softmax":
        raise Unavailable("a multinomial-logit effect has one curve per category")
    xname, col, xlab = _covariate(fit, pred)
    xv = pred.data[xname]
    x = xv if col is None else xv[:, col]
    grid = np.linspace(float(x.min()), float(x.max()), 30)
    draws = {n: pred.draws(fit, n, 100) for n in pred.leaves}
    S = min(v.shape[0] for v in draws.values())
    # rows x draws x grid: each draw's predictor at every observed row with the covariate set to
    # the grid value; the curve is the row average (other covariates and groups as observed)
    rows = [_set_covariate(pred, xname, col, g) for g in grid]
    eta = np.array(
        [
            [
                pred.evaluate(fit, {**{n: v[s] for n, v in draws.items()}, **r}).reshape(-1)
                for r in rows
            ]
            for s in range(S)
        ]
    )  # (S, G, n)
    y_raw = pred.observed
    note = None
    mean_curve = None
    if pred.family == "LogNormal":
        med = np.exp(eta)
        slot = dist_params(fit.model[pred.y])[_SCALE_SLOT["LogNormal"]]
        u = B._unwrap(slot, fit.ctx, preserve_axes=True)
        sig = None
        if B._depends_on(slot, fit.model[xname]):
            note = "mean curve omitted: sigma itself depends on this covariate"
        elif not depends_on_rv(slot):
            a = _fixed_likelihood_value(fit, pred, slot)
            sig = float(a.reshape(-1)[0]) if a.size == 1 else a.reshape(1, 1, -1)
        elif id(u) in fit.ctx.named and u.ndim == 0:
            sig = fit.draws(fit.ctx.named[id(u)], 100)[:S].reshape(-1, 1)
        else:
            sig_node = fit.ctx.named.get(id(u))
            if u.ndim == 1 and sig_node and fit.has(sig_node):
                sig = fit.draws(sig_node, 100, rows=pred.rows)[:S].reshape(S, 1, -1)
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
        y_plot = y_raw / np.broadcast_to(_fixed_likelihood_value(fit, pred, ntr), y_raw.shape)
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
    f = _compile(expr, fit, leaves)
    cf, cleaves = None, []
    if center is not None:
        cleaves = sorted({n for n in B._named_leaves(center, ctx) if ctx.role(n) != "data"})
        cf = _compile(center, fit, cleaves)
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
    pred = _consumer_predictor(fit, a)
    if pred.link != "identity" or pred.family not in ("Normal", "StudentT", "LogNormal"):
        raise Unavailable("kappa is defined for a Gaussian (or log-Gaussian) likelihood")
    occurrences = [t for t in pred.terms if a in t.params]
    term = occurrences[0] if len(occurrences) == 1 else None
    if (
        term is None
        or term.kind != "product"
        or term.multiplier in (None, 0)
        or not term.row_design
        or term.design is None
        or depends_on_rv(term.design)
    ):
        raise Unavailable(f"{a!r} is not multiplied by a fixed design")
    X = term.multiplier * np.asarray(fit.eval_fixed(term.design, data=pred.data), float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or X.shape[0] != pred.n_obs:
        raise Unavailable("the design does not match the observation rows")
    sc, _, _ = _scales(fit, b)
    if X.shape[1] != sc.shape[1]:
        raise Unavailable("the design does not match the coefficients")
    slot = dist_params(fit.model[pred.y])[_SCALE_SLOT[pred.family]]
    u = B._unwrap(slot, fit.ctx, preserve_axes=True)
    S = sc.shape[0]
    if not depends_on_rv(slot):
        inv = np.broadcast_to(
            1.0 / np.square(_fixed_likelihood_value(fit, pred, slot)), (X.shape[0],)
        )
        w = np.broadcast_to((np.square(X) * inv[:, None]).sum(axis=0), (S, X.shape[1]))
    elif u.ndim <= 1 and id(u) in fit.ctx.named and fit.has(fit.ctx.named[id(u)]):
        sig = fit.draws(fit.ctx.named[id(u)], MAX_DRAWS, rows=pred.rows if u.ndim else None)
        sig = sig[:S].reshape(S, -1)
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
    da, rows = fit.observation_rows(reading)
    slots = dist_params(fit.model[reading])
    if not slots or B._unwrap(slots[0], fit.ctx, preserve_axes=True) is not fit.model[latent]:
        raise Unavailable("the readings must measure the latent values directly")
    draws = fit._dataarray(latent, "posterior", rows=rows)
    if draws.shape[2:] != da.shape:
        raise Unavailable("readings and latent values do not line up")
    dims = fit._dims(reading, da.ndim)
    ldims = fit._dims(latent, da.ndim)
    for i, (axis, laxis) in enumerate(zip(da.dims[1:], draws.dims[3:], strict=True), 1):
        obs_axis = _axis(da, axis, dims[i], reading, i)
        latent_axis = _axis(draws, laxis, ldims[i], latent, i)
        positions = obs_axis.indexer(latent_axis, f"{latent!r} axis {laxis!r}")
        if not isinstance(positions, slice):
            raise Unavailable("readings and latent values have different trailing-axis order")
    obs = np.asarray(da.values, float).reshape(-1)
    d = np.asarray(draws.values, float).reshape(draws.shape[0] * draws.shape[1], -1)
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
