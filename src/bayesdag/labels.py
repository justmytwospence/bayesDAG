"""PPL-agnostic LaTeX label assembly: variable-name -> symbol, per-distribution
templates, and assembly of ``name \\sim Dist(args)`` / ``name = expr`` with each
parameter token wrapped in ``\\cssId{tok-<id>}{...}`` so it is anchorable for port-edges.

This module does **not** import pymc/pytensor. The PyTensor-specific rendering of a slot
value or a deterministic expression to LaTeX lives in the pymc adapter, which feeds the
resulting strings here for assembly.
"""

from __future__ import annotations

from typing import Optional

from .ir import TokenIR

_GREEK_LOWER = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
    "varphi",
    "vartheta",
    "varepsilon",
}
_GREEK_UPPER = {
    "Gamma",
    "Delta",
    "Theta",
    "Lambda",
    "Xi",
    "Pi",
    "Sigma",
    "Upsilon",
    "Phi",
    "Psi",
    "Omega",
}

# Distribution name (the adapter's derived `dist`, i.e. op._print_name[0] or the RV class minus
# "RV") -> LaTeX symbol. Keys are the VERIFIED derived names for PyMC 6.x; several public dists
# collapse to a shared op (NormalMixture/ZeroInflated*->"Mixture", GaussianRandomWalk->"RandomWalk",
# OrderedLogistic/Probit->"Categorical", ChiSquared->"Gamma", MvNormal->"MultivariateNormal").
DIST_SYMBOLS = {
    # --- continuous univariate ---
    "Normal": r"\mathcal{N}",
    "HalfNormal": r"\mathcal{N}^{+}",
    "ZeroSumNormal": r"\mathcal{N}_{0}",  # Normal constrained to sum to zero over its last dim
    "Uniform": r"\mathrm{U}",
    "Beta": r"\mathrm{Beta}",
    "Kumaraswamy": r"\mathrm{Kumaraswamy}",
    "Gamma": r"\mathrm{Gamma}",
    "InverseGamma": r"\mathrm{InvGamma}",
    "ChiSquared": r"\chi^{2}",
    "Exponential": r"\mathrm{Exp}",
    "Laplace": r"\mathrm{Laplace}",
    "AsymmetricLaplace": r"\mathrm{AL}",
    "StudentT": r"\mathrm{StudentT}",
    "HalfStudentT": r"\mathrm{StudentT}^{+}",
    "SkewStudentT": r"\mathrm{SkewT}",
    "Cauchy": r"\mathrm{Cauchy}",
    "HalfCauchy": r"\mathrm{HalfCauchy}",
    "LogNormal": r"\mathrm{LogNormal}",
    "LogitNormal": r"\mathrm{LogitN}",
    "Logistic": r"\mathrm{Logistic}",
    "Weibull": r"\mathrm{Weibull}",
    "Gumbel": r"\mathrm{Gumbel}",
    "Moyal": r"\mathrm{Moyal}",
    "Pareto": r"\mathrm{Pareto}",
    "Rice": r"\mathrm{Rice}",
    "SkewNormal": r"\mathrm{SkewN}",
    "Triangular": r"\mathrm{Tri}",
    "VonMises": r"\mathrm{VonMises}",
    "Wald": r"\mathrm{Wald}",  # a.k.a. InverseGaussian
    "ExGaussian": r"\mathrm{ExGauss}",
    "PG": r"\mathrm{PG}",  # PolyaGamma (op _print_name = "PG")
    # --- discrete univariate ---
    "Poisson": r"\mathrm{Pois}",
    "Bernoulli": r"\mathrm{Bern}",
    "Binomial": r"\mathrm{Bin}",
    "BetaBinomial": r"\mathrm{BetaBin}",
    "NegativeBinomial": r"\mathrm{NegBin}",
    "Geometric": r"\mathrm{Geom}",
    "HyperGeometric": r"\mathrm{HyperGeom}",
    "DiscreteUniform": r"\mathrm{U}_{d}",
    "DiscreteWeibull": r"\mathrm{Weibull}_{d}",
    "Categorical": r"\mathrm{Cat}",
    "DiracDelta": r"\delta",
    # --- multivariate / matrix / simplex ---
    "MultivariateNormal": r"\mathcal{N}",  # MvNormal's op print-name
    "MvNormal": r"\mathcal{N}",  # defensive alias
    "MvStudentT": r"\mathcal{T}",
    "Dirichlet": r"\mathrm{Dir}",
    "Multinomial": r"\mathrm{Mult}",
    "DirichletMultinomial": r"\mathrm{DirMult}",
    "StickBreakingWeights": r"\mathrm{SBW}",
    "Wishart": r"\mathcal{W}",
    "MatrixNormal": r"\mathcal{MN}",
    "KroneckerNormal": r"\mathcal{N}_{\otimes}",
    "LKJCorr": r"\mathrm{LKJ}",
    "LKJCorrRV": r"\mathrm{LKJ}",  # op exposes the RV class name here
    "_lkjcholeskycov": r"\mathrm{LKJ}_{\mathrm{chol}}",
    # --- mixtures / inflated ---
    "Mixture": r"\mathrm{Mix}",  # also NormalMixture / ZeroInflated*
    "Hurdle": r"\mathrm{Hurdle}",
    # --- bounded ---
    "TruncatedNormal": r"\mathcal{N}_{[\,]}",
    "Censored": r"\mathrm{Censored}",
    # --- time series ---
    "RandomWalk": r"\mathrm{RW}",  # also GaussianRandomWalk
    "AR": r"\mathrm{AR}",
    "GARCH11": r"\mathrm{GARCH}",
    "EulerMaruyama": r"\mathrm{SDE}",  # Euler-Maruyama discretization of an SDE
    # --- spatial ---
    "CAR": r"\mathrm{CAR}",
    "ICAR": r"\mathrm{ICAR}",
    # --- meta / custom ---
    "Interpolated": r"\mathrm{Interp}",
    "Flat": r"\mathrm{Flat}",
    "HalfFlat": r"\mathrm{Flat}^{+}",
    "CustomDist": r"\operatorname{Custom}",
    "DensityDist": r"\operatorname{Custom}",
    "Simulator": r"\operatorname{Simulator}",
}


def _atom(token: str) -> str:
    if token in _GREEK_LOWER or token in _GREEK_UPPER:
        return "\\" + token
    if len(token) == 1:
        return token
    if token.isdigit():
        return token
    return r"\mathrm{" + token + "}"


def symbol_for(name: str) -> str:
    """LaTeX symbol for a variable name (greek detection + subscripts).

    ``mu`` -> ``\\mu``, ``y_obs`` -> ``y_{\\mathrm{obs}}``, ``beta_1`` -> ``\\beta_{1}``.
    """
    base, _, sub = name.partition("_")
    out = _atom(base)
    if sub:
        out += "_{" + ",".join(_atom(s) for s in sub.split("_") if s) + "}"
    return out


# Ordered op-level param names for SymbolicRandomVariables, whose ``__call__`` signature is
# the generic ``(self, inputs, kwargs)`` and yields useless ``arg0/arg1/…`` names. Keyed on
# the DERIVED op name (same keying rule as DIST_SYMBOLS); a key may carry several variants
# for ops whose arity differs by construction (Mixture). ``None`` = structural plumbing
# (steps/size) hidden from the label. A variant applies ONLY when its length matches the
# node's actual ``dist_params`` count (exact match or fall back — never guess). Each entry's
# order is verified against ``op.dist_params`` introspection; verify before adding.
DIST_PARAM_TEMPLATES: dict[str, list[list[Optional[str]]]] = {
    "RandomWalk": [["init", "innov", None]],  # [init_dist, innovation_dist, steps]
    "AR": [["rho", "sigma", "init", None]],  # [rho, sigma, init_dist, steps]
    "Censored": [["dist", "lower", "upper"]],
    "Mixture": [
        ["w", "comp"],  # NormalMixture (single packed component)
        ["w", "comp1", "comp2"],  # ZeroInflated*/two-component (e.g. [w, zero-spike, count])
    ],
    "_lkjcholeskycov": [["n", "eta", "sd_dist"]],
    "LKJCorrRV": [["n", "eta"]],
    # pymc-bart dist_params = [X, Y, m, alpha, beta]: show the predictor X and tree count m; hide the
    # response array and the tree-prior hyperparameters (noise in a diagram) -> `mu ~ BART(X, m)`.
    "BART": [["X", None, "m", None, None]],
}


def dist_symbol(dist_name: Optional[str]) -> str:
    if not dist_name:
        return r"\operatorname{?}"
    if dist_name in DIST_SYMBOLS:
        return DIST_SYMBOLS[dist_name]
    # A generic Truncated(<Base>) collapses to a dynamic op name like "TruncatedGamma": render the
    # base distribution's symbol with a truncation subscript.
    if dist_name.startswith("Truncated"):
        base = dist_name[len("Truncated") :]
        base_sym = DIST_SYMBOLS.get(base, rf"\operatorname{{{base}}}")
        return base_sym + r"_{[\,]}"
    # CustomDist/DensityDist name their op after the VARIABLE ("CustomDist_theta"), so the plain
    # key never matches — fall back to the family symbol rather than printing the mangled name.
    if dist_name.startswith("CustomDist"):
        return DIST_SYMBOLS["CustomDist"]
    return rf"\operatorname{{{dist_name}}}"


LHS_TOKEN = "__lhs__"  # reserved token id for a deterministic equation's left-hand-side variable,
# so its outgoing edge can originate from the variable it defines (never a port-edge target)


def _cssid(token_id: str, content: str) -> str:
    return rf"\cssId{{tok-{token_id}}}{{{content}}}"


def assemble_stochastic(
    node_name: str, dist_name: Optional[str], args: list[tuple[str, str]]
) -> tuple[str, TokenIR]:
    """``args`` = list of (token_id, value_tex). Returns (label_tex, token_tree)."""
    sym = symbol_for(node_name)
    dsym = dist_symbol(dist_name)
    wrapped = [_cssid(tid, vt) for tid, vt in args]
    rhs = dsym + (r"\!\left(" + ",\\ ".join(wrapped) + r"\right)" if wrapped else "")
    label = f"{sym} \\sim {rhs}"
    tree = TokenIR(token_id="root", tex=label, children=[TokenIR(tid, vt) for tid, vt in args])
    return label, tree


def assemble_deterministic(
    node_name: str, expr_tex: str, leaf_tokens: list[str]
) -> tuple[str, TokenIR]:
    sym = symbol_for(node_name)
    # wrap the LHS variable so it's anchorable — its outgoing edge originates from the variable
    label = f"{_cssid(LHS_TOKEN, sym)} = {expr_tex}"
    tree = TokenIR(
        token_id="root",
        tex=label,
        children=[TokenIR(LHS_TOKEN, sym), *[TokenIR(t, t) for t in leaf_tokens]],
    )
    return label, tree


def assemble_bare(node_name: str, kind_tex: Optional[str] = None) -> tuple[str, TokenIR]:
    """For data/potential/elided nodes: just the symbol (optionally annotated)."""
    sym = symbol_for(node_name)
    label = f"{sym} {kind_tex}" if kind_tex else sym
    return label, TokenIR(token_id="root", tex=label)
