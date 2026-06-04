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
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota",
    "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "upsilon",
    "phi", "chi", "psi", "omega", "varphi", "vartheta", "varepsilon",
}
_GREEK_UPPER = {"Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega"}

# Distribution name (from op._print_name[0]) -> LaTeX symbol.
DIST_SYMBOLS = {
    "Normal": r"\mathcal{N}",
    "HalfNormal": r"\mathcal{N}^{+}",
    "TruncatedNormal": r"\mathcal{N}_{[\,]}",
    "MvNormal": r"\mathcal{N}",
    "Uniform": r"\mathrm{U}",
    "Beta": r"\mathrm{Beta}",
    "Gamma": r"\mathrm{Gamma}",
    "InverseGamma": r"\mathrm{InvGamma}",
    "Exponential": r"\mathrm{Exp}",
    "Laplace": r"\mathrm{Laplace}",
    "StudentT": r"\mathrm{StudentT}",
    "HalfStudentT": r"\mathrm{StudentT}^{+}",
    "Cauchy": r"\mathrm{Cauchy}",
    "HalfCauchy": r"\mathrm{HalfCauchy}",
    "LogNormal": r"\mathrm{LogNormal}",
    "Poisson": r"\mathrm{Pois}",
    "Bernoulli": r"\mathrm{Bern}",
    "Binomial": r"\mathrm{Bin}",
    "NegativeBinomial": r"\mathrm{NegBin}",
    "Categorical": r"\mathrm{Cat}",
    "Dirichlet": r"\mathrm{Dir}",
    "Multinomial": r"\mathrm{Mult}",
    "Weibull": r"\mathrm{Weibull}",
    "Gumbel": r"\mathrm{Gumbel}",
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


def dist_symbol(dist_name: Optional[str]) -> str:
    if not dist_name:
        return r"\operatorname{?}"
    return DIST_SYMBOLS.get(dist_name, rf"\operatorname{{{dist_name}}}")


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


def assemble_deterministic(node_name: str, expr_tex: str, leaf_tokens: list[str]) -> tuple[str, TokenIR]:
    sym = symbol_for(node_name)
    # wrap the LHS variable so it's anchorable — its outgoing edge originates from the variable
    label = f"{_cssid(LHS_TOKEN, sym)} = {expr_tex}"
    tree = TokenIR(
        token_id="root", tex=label,
        children=[TokenIR(LHS_TOKEN, sym), *[TokenIR(t, t) for t in leaf_tokens]],
    )
    return label, tree


def assemble_bare(node_name: str, kind_tex: Optional[str] = None) -> tuple[str, TokenIR]:
    """For data/potential/elided nodes: just the symbol (optionally annotated)."""
    sym = symbol_for(node_name)
    label = f"{sym} {kind_tex}" if kind_tex else sym
    return label, TokenIR(token_id="root", tex=label)
