"""``to_ir(obj)`` — idempotent, PPL-agnostic dispatch to a ``ModelIR``.

Mirrors ArviZ's ``convert_to_datatree`` pattern: detect the source by **duck-typed
module/class-name match**, never ``isinstance`` against a PPL type, so the core never
imports pymc/numpyro/stan. New PPL adapters slot in here without touching the IR.
"""

from __future__ import annotations

from typing import Any

from .ir import ModelIR


def to_ir(obj: Any, idata: Any = None) -> ModelIR:
    if isinstance(obj, ModelIR):
        return obj  # idempotent
    if isinstance(obj, dict):
        return ModelIR.from_dict(obj)  # low-level escape hatch

    cls = type(obj)
    module = getattr(cls, "__module__", "") or ""
    name = cls.__name__

    if name == "Model" and module.startswith("pymc"):
        from .adapters.pymc import from_pymc

        return from_pymc(obj, idata=idata)

    # future: numpyro (`MCMC`/handlers), stan (cmdstanpy) -> from_numpyro / from_stan
    raise TypeError(
        f"bayesdag.to_ir: don't know how to convert {module}.{name}. "
        "Pass a pymc.Model, a ModelIR, or a ModelIR dict."
    )
