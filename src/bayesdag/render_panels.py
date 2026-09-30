"""SVG for the fitted-structure panels (PPL-neutral: plain data in, standalone SVG out).

Each renderer takes the dict ``adapters.fitted`` computed and draws it — no statistics here.
Every panel says what it shows in its title and what each mark is in a caption, because these
are the pictures a reader will screenshot without the surrounding card.
"""

from __future__ import annotations

import math
from xml.sax.saxutils import escape

ORANGE = "#d2691e"  # posterior (same as the node glyphs)
BLUE = "#3a5f95"  # data
GREY = "#8a8f98"
LIGHT = "#c9ccd1"


def _fmt(v: float) -> str:
    if v == 0 or not math.isfinite(v):
        return "0" if v == 0 else "—"
    return f"{v:.3g}"


class _Plot:
    """A plot area with linear (or log-x) scales and the frame/ticks/labels around it."""

    def __init__(self, x, y, w, h, xlim, ylim, logx=False):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.logx = logx
        x0, x1 = xlim
        if logx:
            x0, x1 = math.log(max(x0, 1e-12)), math.log(max(x1, 1e-12))
        if not x1 > x0:
            x0, x1 = x0 - 0.5, x1 + 0.5
        y0, y1 = ylim
        if not y1 > y0:
            y0, y1 = y0 - 0.5, y1 + 0.5
        padx, pady = 0.04 * (x1 - x0), 0.06 * (y1 - y0)
        self.x0, self.x1, self.y0, self.y1 = x0 - padx, x1 + padx, y0 - pady, y1 + pady

    def sx(self, v):
        if self.logx:
            v = math.log(max(v, 1e-12))
        return self.x + (v - self.x0) / (self.x1 - self.x0) * self.w

    def sy(self, v):
        return self.y + self.h - (v - self.y0) / (self.y1 - self.y0) * self.h

    def frame(self, xlabel="", ylabel="", yticks=True) -> str:
        out = [
            f'<rect x="{self.x:.1f}" y="{self.y:.1f}" width="{self.w:.1f}" height="{self.h:.1f}" '
            'fill="#fafafa" stroke="#e5e7eb"/>'
        ]
        xs = [self.x0 + (self.x1 - self.x0) * f for f in (0.04 / 1.08, 0.5, 1.04 / 1.08)]
        for v in xs:
            lab = math.exp(v) if self.logx else v
            px = self.x + (v - self.x0) / (self.x1 - self.x0) * self.w
            out.append(
                f'<text x="{px:.1f}" y="{self.y + self.h + 11:.1f}" text-anchor="middle" '
                f'font-size="8" fill="#777">{_fmt(lab)}</text>'
            )
        if yticks:
            for v in (self.y0 + (self.y1 - self.y0) * f for f in (0.06 / 1.12, 0.5, 1.06 / 1.12)):
                out.append(
                    f'<text x="{self.x - 4:.1f}" y="{self.sy(v) + 3:.1f}" text-anchor="end" '
                    f'font-size="8" fill="#777">{_fmt(v)}</text>'
                )
        if xlabel:
            out.append(
                f'<text x="{self.x + self.w / 2:.1f}" y="{self.y + self.h + 23:.1f}" '
                f'text-anchor="middle" font-size="9" fill="#555">{escape(xlabel)}</text>'
            )
        if ylabel:
            cx, cy = self.x - 30, self.y + self.h / 2
            out.append(
                f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" font-size="9" '
                f'fill="#555" transform="rotate(-90 {cx:.1f} {cy:.1f})">{escape(ylabel)}</text>'
            )
        return "".join(out)

    def hline(self, v, color=GREY, dash="3,2", width=0.8) -> str:
        if not (self.y0 <= v <= self.y1):
            return ""
        y = self.sy(v)
        return (
            f'<line x1="{self.x:.1f}" y1="{y:.1f}" x2="{self.x + self.w:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>'
        )

    def vline(self, v, color=GREY, dash="3,2", width=0.8) -> str:
        x = self.sx(v)
        if not (self.x - 0.5 <= x <= self.x + self.w + 0.5):
            return ""
        return (
            f'<line x1="{x:.1f}" y1="{self.y:.1f}" x2="{x:.1f}" y2="{self.y + self.h:.1f}" '
            f'stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>'
        )

    def path(self, xs, ys, color, width=1.2, dash=None, opacity=1.0) -> str:
        pts = " L".join(f"{self.sx(a):.1f},{self.sy(b):.1f}" for a, b in zip(xs, ys, strict=False))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<path d="M{pts}" fill="none" stroke="{color}" stroke-width="{width}"{d} '
            f'opacity="{opacity}"/>'
        )

    def band(self, xs, lo, hi, color, opacity=0.18) -> str:
        top = " L".join(f"{self.sx(a):.1f},{self.sy(b):.1f}" for a, b in zip(xs, hi, strict=False))
        bot = " L".join(
            f"{self.sx(a):.1f},{self.sy(b):.1f}"
            for a, b in zip(reversed(xs), reversed(lo), strict=False)
        )
        return f'<path d="M{top} L{bot} Z" fill="{color}" fill-opacity="{opacity}" stroke="none"/>'


def _doc(w, h, title, body, caption_lines, plot=None) -> str:
    if (
        plot is not None
    ):  # fit the height to what sits under the plot: ticks, label, legend, caption
        h = plot.y + plot.h + 56 + 12 * len(caption_lines)
    cap = "".join(
        f'<text x="12" y="{h - 10 - 12 * (len(caption_lines) - 1 - i):.1f}" font-size="9" '
        f'fill="#8a8f98">{escape(c)}</text>'
        for i, c in enumerate(caption_lines)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" '
        f'viewBox="0 0 {w:.0f} {h:.0f}" font-family="system-ui, sans-serif">'
        f'<rect width="{w:.0f}" height="{h:.0f}" rx="8" fill="#ffffff" stroke="#c7c7cc"/>'
        f'<text x="12" y="22" font-size="12" font-weight="600" fill="#333">{escape(title)}</text>'
        + "".join(body)
        + cap
        + "</svg>"
    )


def _legend(items, x, y) -> str:
    """``items``: (kind, color, label) with kind in line/dash/dot/point/hollow/band."""
    out = []
    for kind, color, label in items:
        if kind in ("line", "dash", "dot"):
            d = {"line": "", "dash": ' stroke-dasharray="4,2"', "dot": ' stroke-dasharray="1.5,2"'}
            out.append(
                f'<line x1="{x}" y1="{y - 3}" x2="{x + 14}" y2="{y - 3}" stroke="{color}" '
                f'stroke-width="1.4"{d[kind]}/>'
            )
        elif kind == "band":
            out.append(
                f'<rect x="{x}" y="{y - 7}" width="14" height="8" fill="{color}" '
                'fill-opacity="0.2"/>'
            )
        else:
            fill = "#fff" if kind == "hollow" else color
            out.append(
                f'<circle cx="{x + 7}" cy="{y - 3}" r="3" fill="{fill}" stroke="{color}" '
                'stroke-width="1.2"/>'
            )
        out.append(f'<text x="{x + 18}" y="{y}" font-size="9" fill="#555">{escape(label)}</text>')
        x += 24 + 5.2 * len(label)
    return "".join(out)


# ---------------------------------------------------------------------------------- hierarchy
def render_shrinkage(d: dict) -> str:
    """Raw group estimate (hollow, blue) -> posterior mean (filled, orange) with its 90%
    interval, against group size (or measurement sd); dashed line = population mean."""
    kind = d["x_kind"]
    if kind == "n":
        xs, xlab, logx = d["n"], "group size (observations, log scale)", True
    elif kind == "se":
        xs, xlab, logx = d["se"], "standard error of the raw estimate", False
    else:
        order = sorted(range(len(d["raw"])), key=lambda i: d["raw"][i])
        xs = [order.index(i) + 1 for i in range(len(d["raw"]))]
        xlab, logx = "group (ranked by raw estimate)", False
    if kind == "n":  # spread tied sizes so the arrows don't overlap
        seen: dict = {}
        jit = []
        for v in xs:
            k = seen.get(v, 0)
            seen[v] = k + 1
            jit.append(v * (1 + 0.04 * k))
        xs = jit
    vals = d["raw"] + d["post_lo"] + d["post_hi"] + ([d["pop"]] if d["pop"] is not None else [])
    w, h = 440.0, 300.0
    p = _Plot(52, 34, w - 70, h - 110, (min(xs), max(xs)), (min(vals), max(vals)), logx=logx)
    body = [p.frame(xlab, d["y"])]
    if d["pop"] is not None:
        body.append(p.hline(d["pop"], GREY))
    for x, r, m, lo, hi in zip(xs, d["raw"], d["post"], d["post_lo"], d["post_hi"], strict=True):
        px = p.sx(x)
        body.append(
            f'<line x1="{px:.1f}" y1="{p.sy(lo):.1f}" x2="{px:.1f}" y2="{p.sy(hi):.1f}" '
            f'stroke="{ORANGE}" stroke-width="1" opacity="0.45"/>'
            f'<line x1="{px:.1f}" y1="{p.sy(r):.1f}" x2="{px:.1f}" y2="{p.sy(m):.1f}" '
            f'stroke="{GREY}" stroke-width="1"/>'
            f'<circle cx="{px:.1f}" cy="{p.sy(r):.1f}" r="3" fill="#fff" stroke="{BLUE}" '
            'stroke-width="1.2"/>'
            f'<circle cx="{px:.1f}" cy="{p.sy(m):.1f}" r="3" fill="{ORANGE}"/>'
        )
    body.append(
        _legend(
            [
                ("hollow", BLUE, "raw (no pooling)"),
                ("point", ORANGE, "posterior mean, 90%"),
                ("dash", GREY, "population mean"),
            ],
            14,
            p.y + p.h + 40,
        )
    )
    return _doc(
        w,
        h,
        f"{d['effect']}: shrinkage toward the population",
        body,
        [
            "raw = group mean of the data minus the rest of the predictor (at the fit);",
            "small or noisy groups are pulled furthest — that pull is the partial pooling",
        ],
        plot=p,
    )


def render_population(d: dict) -> str:
    """Draws of the population density (faint), their average — where a NEW group would fall —
    (bold), and each group's posterior mean as a tick."""
    xs = d["xs"]
    w, h = 440.0, 250.0
    p = _Plot(20, 34, w - 40, h - 100, (xs[0], xs[-1]), (0.0, 1.05))
    body = [p.frame(d["effect"], yticks=False)]
    for c in d["curves"]:
        body.append(p.path(xs, [min(v, 1.05) for v in c], ORANGE, width=0.6, opacity=0.25))
    body.append(p.path(xs, d["predictive"], ORANGE, width=1.8))
    for v in d["rug"]:
        px = p.sx(v)
        body.append(
            f'<line x1="{px:.1f}" y1="{p.y + p.h:.1f}" x2="{px:.1f}" y2="{p.y + p.h - 8:.1f}" '
            f'stroke="{BLUE}" stroke-width="1.2"/>'
        )
    body.append(
        _legend(
            [
                ("line", ORANGE, "a new group (posterior predictive)"),
                ("dot", ORANGE, "population draws"),
                ("line", BLUE, "groups' posterior means"),
            ],
            14,
            p.y + p.h + 40,
        )
    )
    fam = "log-normal" if d["family"] == "LogNormal" else "normal"
    return _doc(
        w,
        h,
        f"{d['effect']}: the population it is drawn from",
        body,
        [f"{fam} population, one faint curve per posterior draw; {d['n_groups']} groups ticked"],
        plot=p,
    )


def render_pooling(d: dict) -> str:
    lam = d["lambda"]
    w = 440.0
    rows = (d.get("groups") or {}).get("labels") or []
    h = 120.0 + (14.0 * len(rows) + 20 if rows else 0.0)
    body = []
    gx, gy, gw = 20.0, 46.0, w - 40
    body.append(
        f'<rect x="{gx}" y="{gy}" width="{gw}" height="10" rx="3" fill="#eef0f3"/>'
        + (
            f'<rect x="{gx}" y="{gy}" width="{gw * max(0.0, min(1.0, lam)):.1f}" height="10" '
            f'rx="3" fill="{ORANGE}" fill-opacity="0.75"/>'
            if math.isfinite(lam)
            else ""
        )
        + f'<text x="{gx}" y="{gy + 24}" font-size="9" fill="#777">0 · no pooling</text>'
        f'<text x="{gx + gw}" y="{gy + 24}" text-anchor="end" font-size="9" fill="#777">'
        "complete pooling · 1</text>"
        f'<text x="{gx + gw / 2}" y="{gy - 6}" text-anchor="middle" font-size="11" fill="#333">'
        f"pooling factor λ = {_fmt(lam)}</text>"
    )
    if rows:
        y0 = gy + 44
        body.append(
            f'<text x="{gx}" y="{y0}" font-size="9" fill="#555">each group: fraction of the '
            "way from its raw estimate to the population mean</text>"
        )
        bx, bw = gx + 70, gw - 110
        for i, (lab, f, n) in enumerate(
            zip(rows, d["groups"]["pull"], d["groups"]["n"], strict=True)
        ):
            y = y0 + 8 + 14 * i
            body.append(
                f'<text x="{bx - 6}" y="{y + 8}" text-anchor="end" font-size="8" fill="#555">'
                f"{escape(lab[:12])}</text>"
            )
            if f is not None and math.isfinite(f):
                fw = bw * max(0.0, min(1.0, f))
                body.append(
                    f'<rect x="{bx}" y="{y}" width="{fw:.1f}" height="9" fill="{ORANGE}" '
                    'fill-opacity="0.6"/>'
                    f'<text x="{bx + bw + 4}" y="{y + 8}" font-size="8" fill="#777">'
                    f"{f:.0%} (n={n})</text>"
                )
    return _doc(
        w,
        h,
        f"{d['effect']}: how much pooling",
        body,
        ["λ = 1 − Var_j(E[θ_j − μ]) / E[Var_j(θ_j − μ)] (Gelman & Pardoe 2006)"],
    )


# ---------------------------------------------------------------------------------- regression
def render_group_lines(d: dict) -> str:
    """Gelman–Hill small multiples: per group, data, complete pooling (grey dashed), no pooling
    (blue dotted) and the model's partial pooling (orange)."""
    groups = d["groups"]
    cols = 4
    rows = math.ceil(len(groups) / cols)
    cw, ch = 118.0, 104.0
    w = 24 + cols * cw
    h = 44 + rows * ch + 46
    ylo, yhi = d["ylim"]
    vals = [ylo, yhi] + d["complete"] + [v for g in groups for v in g["partial"]]
    lo, hi = min(vals), max(vals)
    import hashlib

    # clip each mini plot (a within-group least-squares line can shoot off); ids are
    # content-derived so two panels in one page never collide
    tag = hashlib.sha1(repr((d["x"], d["y"], d["grid"])).encode()).hexdigest()[:8]
    body = []
    for i, g in enumerate(groups):
        r, c = divmod(i, cols)
        p = _Plot(
            20 + c * cw, 40 + r * ch, cw - 14, ch - 34, (d["grid"][0], d["grid"][1]), (lo, hi)
        )
        cid = f"bdp-{tag}-{i}"
        body.append(
            f'<clipPath id="{cid}"><rect x="{p.x:.1f}" y="{p.y:.1f}" width="{p.w:.1f}" '
            f'height="{p.h:.1f}"/></clipPath>'
        )
        body.append(
            f'<rect x="{p.x:.1f}" y="{p.y:.1f}" width="{p.w:.1f}" height="{p.h:.1f}" '
            'fill="#fafafa" stroke="#e5e7eb"/>'
            f'<text x="{p.x:.1f}" y="{p.y - 3:.1f}" font-size="8" fill="#555">'
            f"{escape(g['label'][:14])} (n={g['n']})</text>"
        )
        for x, y in zip(g["x"], g["y"], strict=True):
            body.append(
                f'<circle cx="{p.sx(x):.1f}" cy="{p.sy(y):.1f}" r="1.6" fill="{BLUE}" '
                'fill-opacity="0.55"/>'
            )
        lines = [p.path(d["grid"], d["complete"], GREY, width=1, dash="4,2")]
        if g["nopool"]:
            lines.append(p.path(d["grid"], g["nopool"], BLUE, width=1, dash="1.5,2"))
        lines.append(p.path(d["grid"], g["partial"], ORANGE, width=1.5))
        body.append(f'<g clip-path="url(#{cid})">' + "".join(lines) + "</g>")
    body.append(
        _legend(
            [
                ("dash", GREY, "complete pooling (least squares)"),
                ("dot", BLUE, "no pooling (least squares in group)"),
                ("line", ORANGE, "the model"),
            ],
            14,
            h - 32,
        )
    )
    return _doc(
        w,
        h,
        f"{d['y']} against {d['x']}, by group ({len(groups)} of {d['n_groups']})",
        body,
        ["the model's line averages each group's rows with only the covariate changed"],
    )


def render_intercept_slope(d: dict) -> str:
    ma, mb = d["ma"], d["mb"]
    w, h = 380.0, 320.0
    p = _Plot(
        56, 34, w - 76, h - 110, (min(d["a_lo"]), max(d["a_hi"])), (min(d["b_lo"]), max(d["b_hi"]))
    )
    body = [p.frame(d["a"] + " (intercept)", d["b"] + " (slope)")]
    for a, b, alo, ahi, blo, bhi in zip(
        ma, mb, d["a_lo"], d["a_hi"], d["b_lo"], d["b_hi"], strict=True
    ):
        body.append(
            f'<line x1="{p.sx(alo):.1f}" y1="{p.sy(b):.1f}" x2="{p.sx(ahi):.1f}" y2="{p.sy(b):.1f}" '
            f'stroke="{ORANGE}" stroke-width="0.8" opacity="0.4"/>'
            f'<line x1="{p.sx(a):.1f}" y1="{p.sy(blo):.1f}" x2="{p.sx(a):.1f}" y2="{p.sy(bhi):.1f}" '
            f'stroke="{ORANGE}" stroke-width="0.8" opacity="0.4"/>'
            f'<circle cx="{p.sx(a):.1f}" cy="{p.sy(b):.1f}" r="3" fill="{ORANGE}"/>'
        )
    cap = ["one point per group: posterior means with 90% intervals"]
    if d.get("r_means") is not None:
        cap.append(
            f"correlation of the posterior means: {d['r_means']:.2f} (descriptive — the model "
            "does not estimate one)"
        )
    return _doc(w, h, "varying intercepts vs varying slopes", body, cap, plot=p)


def render_effect(d: dict) -> str:
    g = d["grid"]
    ys = d["pts_y"] + d["lo"] + d["hi"] + (d["mean"] or [])
    w, h = 440.0, 300.0
    p = _Plot(
        56, 34, w - 76, h - 118, (min(d["pts_x"] + g), max(d["pts_x"] + g)), (min(ys), max(ys))
    )
    body = [p.frame(d["x"], d["y"])]
    import random

    rnd = random.Random(0)  # deterministic jitter for 0/1 outcomes
    for x, y in zip(d["pts_x"], d["pts_y"], strict=True):
        jy = y + (rnd.uniform(-0.03, 0.03) if d["binary"] else 0.0)
        body.append(
            f'<circle cx="{p.sx(x):.1f}" cy="{p.sy(jy):.1f}" r="1.6" fill="{BLUE}" '
            'fill-opacity="0.35"/>'
        )
    body.append(p.band(g, d["lo"], d["hi"], ORANGE))
    body.append(p.path(g, d["mid"], ORANGE, width=1.8))
    items = [("band", ORANGE, "90% credible band")]
    if d["family"] == "LogNormal":
        items.insert(0, ("line", ORANGE, "median exp(η)"))
        if d["mean"]:
            body.append(p.path(g, d["mean"], ORANGE, width=1.4, dash="4,2"))
            items.insert(1, ("dash", ORANGE, "mean exp(η + σ²/2)"))
    else:
        items.insert(0, ("line", ORANGE, "fitted mean"))
    items.append(("point", BLUE, "data"))
    body.append(_legend(items, 14, p.y + p.h + 40))
    cap = ["averaged over the other covariates and groups as observed (only this one varies)"]
    if d.get("note"):
        cap.append(d["note"])
    link = {"Sigmoid": "logit", "Exp": "log", "probit": "probit"}.get(d["link"])
    title = f"{d['y']} against {d['x']}" + (f" (the {link} link undone)" if link else "")
    return _doc(w, h, title, body, cap, plot=p)


# ---------------------------------------------------------------------------------- shrinkage
def render_kappa(d: dict) -> str:
    """Pooled histogram of the shrinkage factors kappa over all coefficients and draws, with each
    coefficient's mean kappa as a tick: a horseshoe's U — most near 1, a few near 0."""
    w, h = 440.0, 260.0
    p = _Plot(20, 34, w - 40, h - 110, (0.0, 1.0), (0.0, 1.05))
    body = [p.frame("κ  (0 = left alone · 1 = shrunk to the center)", yticks=False)]
    e = d["edges"]
    for i, v in enumerate(d["hist"]):
        x0, x1 = p.sx(e[i]), p.sx(e[i + 1])
        body.append(
            f'<rect x="{x0:.1f}" y="{p.sy(v):.1f}" width="{max(0.5, x1 - x0 - 1):.1f}" '
            f'height="{p.sy(0) - p.sy(v):.1f}" fill="{ORANGE}" fill-opacity="0.5"/>'
        )
    for v in d["mean"]:
        px = p.sx(v)
        body.append(
            f'<line x1="{px:.1f}" y1="{p.y + p.h:.1f}" x2="{px:.1f}" y2="{p.y + p.h - 9:.1f}" '
            f'stroke="{BLUE}" stroke-width="1.2"/>'
        )
    body.append(
        _legend(
            [("band", ORANGE, "all draws"), ("line", BLUE, "each coefficient's mean κ")],
            14,
            p.y + p.h + 40,
        )
    )
    cap = [
        f"κ_j = 1/(1 + s_j² Σ_i x_ij²/σ_i²) (Piironen & Vehtari 2017); Σ_j (1 − κ_j) ≈ "
        f"{d['m_eff']:.1f} effective coefficients",
    ]
    if not d["orthogonal"]:
        cap.append("predictors are correlated: κ is the orthogonal-design approximation")
    return _doc(w, h, f"{d['coef']}: shrinkage factors", body, cap, plot=p)


def render_local_scale(d: dict) -> str:
    s = d["s"]
    vals = d["lo"] + d["hi"]
    xs = s + ([d["c"]] if d["c"] else [])
    w, h = 440.0, 300.0
    p = _Plot(56, 34, w - 76, h - 110, (min(xs), max(xs)), (min(vals), max(vals)), logx=True)
    ylab = f"{d['coef']} − center" if d["centered"] else d["coef"]
    body = [p.frame("conditional prior scale s = τ · regularized λ (log)", ylab)]
    body.append(p.hline(0.0))
    if d["c"]:
        body.append(p.vline(d["c"], "#b45309", dash="4,2", width=1))
    for x, m, lo, hi in zip(s, d["mid"], d["lo"], d["hi"], strict=True):
        px = p.sx(x)
        body.append(
            f'<line x1="{px:.1f}" y1="{p.sy(lo):.1f}" x2="{px:.1f}" y2="{p.sy(hi):.1f}" '
            f'stroke="{ORANGE}" stroke-width="1" opacity="0.55"/>'
            f'<circle cx="{px:.1f}" cy="{p.sy(m):.1f}" r="2.6" fill="{ORANGE}"/>'
        )
    items = [("point", ORANGE, "posterior median, 90%")]
    if d["c"]:
        items.append(("dash", "#b45309", "slab: the ceiling on s"))
    body.append(_legend(items, 14, p.y + p.h + 40))
    return _doc(
        w,
        h,
        f"{d['coef']}: local scale vs coefficient",
        body,
        [
            "coefficients that escape shrinkage sit at large s; the slab caps their scale,",
            "not their size",
        ],
        plot=p,
    )


# ---------------------------------------------------------------------------------- measurement
def render_measured(d: dict) -> str:
    vals = d["obs"] + d["lo"] + d["hi"]
    lo, hi = min(vals), max(vals)
    w, h = 380.0, 330.0
    p = _Plot(56, 34, w - 76, h - 110, (lo, hi), (lo, hi))
    body = [p.frame(f"{d['reading']} (reading)", f"{d['latent']} (inferred)")]
    body.append(p.path([lo, hi], [lo, hi], GREY, width=0.8, dash="3,2"))
    for o, m, a, b in zip(d["obs"], d["post"], d["lo"], d["hi"], strict=True):
        px = p.sx(o)
        body.append(
            f'<line x1="{px:.1f}" y1="{p.sy(a):.1f}" x2="{px:.1f}" y2="{p.sy(b):.1f}" '
            f'stroke="{ORANGE}" stroke-width="0.9" opacity="0.5"/>'
            f'<circle cx="{px:.1f}" cy="{p.sy(m):.1f}" r="2.4" fill="{ORANGE}"/>'
        )
    return _doc(
        w,
        h,
        "readings vs inferred true values",
        body,
        ["posterior mean and 90% interval per reading; dashed = 1:1 (no correction)"],
        plot=p,
    )


def render_unavailable(title: str, reason: str) -> str:
    return _doc(
        380.0,
        76.0,
        title,
        [f'<text x="12" y="44" font-size="10" fill="#6b7280">{escape(reason)}</text>'],
        [],
    )


RENDERERS = {
    "shrinkage": render_shrinkage,
    "population": render_population,
    "pooling": render_pooling,
    "group_lines": render_group_lines,
    "intercept_slope": render_intercept_slope,
    "effect": render_effect,
    "kappa": render_kappa,
    "local_scale": render_local_scale,
    "measured": render_measured,
}
