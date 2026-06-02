"""Adapters: PPL front-ends (``from_pymc``) and graph-format projections (``to_elk``,
``to_networkx``). Kept out of the import-light ``bayesdag.ir`` core."""

from __future__ import annotations

from .graph import to_elk, to_networkx

__all__ = ["from_pymc", "to_elk", "to_networkx"]


def __getattr__(name: str):  # lazy so importing `adapters` doesn't require pymc
    if name == "from_pymc":
        from .pymc import from_pymc

        return from_pymc
    raise AttributeError(name)
