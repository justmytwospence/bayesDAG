"""Where the time goes in a render, so "fast enough to feel live" is a number, not a claim.

The interactive loop bayesdag is aiming at is: move a slider -> rebuild the `pm.Model` in the
cell -> `view(model)` -> look. That is only pleasant if a warm re-render stays in the low
hundreds of milliseconds, and the only way to keep it there is to know which stage costs what.

Run it::

    uv run python examples/benchmark.py            # the six conftest models
    uv run python examples/benchmark.py eight_schools radon

Cold numbers include one-time process costs — building the two V8 isolates (~0.2s each) and
filling the MathJax cache. Warm numbers are what a slider drag actually pays. Note that changing
a prior's *value* changes the LaTeX for that label, so it misses the math cache and can resize
the node: the honest slider loop re-runs layout, which is why the warm layout number matters.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from conftest import MODEL_BUILDERS

REPEATS = 5


def _time(fn, repeats=REPEATS):
    """(median, min) seconds over `repeats` runs, after one warm-up."""
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times)


def _row(name, cold, stages, total_warm):
    parts = "  ".join(f"{k} {v * 1000:6.1f}" if v is not None else f"{k}    n/a" for k, v in stages)
    print(f"{name:<14} cold {cold * 1000:7.1f}   {parts}   warm total {total_warm * 1000:7.1f}")


def main(names: list[str]) -> None:
    import bayesdag
    from bayesdag.convert import to_ir
    from bayesdag.layout import layout
    from bayesdag.render_svg import to_svg

    print(f"bayesdag benchmark — median of {REPEATS} runs, milliseconds\n")
    print("cold = first render in this process (includes V8 isolate build + empty math cache)")
    print("warm = steady state, which is what an interactive rebuild loop pays\n")

    for name in names:
        build = MODEL_BUILDERS[name]

        t0 = time.perf_counter()
        bayesdag.view(build(), ppc_draws=0).to_svg()
        cold = time.perf_counter() - t0

        ir_med, _ = _time(lambda b=build: to_ir(b()))
        ir = to_ir(build())
        layout_med, _ = _time(lambda ir=ir: layout(ir))
        res = layout(ir)
        svg_med, _ = _time(lambda ir=ir, res=res: to_svg(ir, res))
        view_med, _ = _time(lambda b=build: bayesdag.view(b(), ppc_draws=0).to_svg())

        spec_med = None
        try:
            v = bayesdag.view(build(), ppc_draws=0)
            v.widget()
            spec_med, _ = _time(v._build_spec)
        except Exception:
            pass  # anywidget not installed: the spec stage simply isn't measured

        _row(
            name,
            cold,
            [("to_ir", ir_med), ("layout", layout_med), ("to_svg", svg_med), ("spec", spec_med)],
            view_med,
        )

    print(
        "\nThe plate prior-predictive expansion is NOT in these numbers: it forward-simulates the\n"
        "model and is computed only when a plate is opened (view.expand_plates). Timing it here\n"
        "would measure PyMC's sampler, not bayesdag's render path."
    )


if __name__ == "__main__":
    requested = sys.argv[1:] or list(MODEL_BUILDERS)
    unknown = [n for n in requested if n not in MODEL_BUILDERS]
    if unknown:
        sys.exit(f"unknown model(s) {unknown}; available: {list(MODEL_BUILDERS)}")
    main(requested)
