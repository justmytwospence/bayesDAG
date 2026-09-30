"""The golden fixture: a committed reference SVG the renderer must keep reproducing.

Every other rendering test asserts a *property* (well-formed XML, defs deduped, no edge
through a node). The parity test compares the widget's SVG to the static one — but both come
from the same emitter, so a regression that changes both passes it. Nothing pinned the actual
bytes, which is what AGENTS.md means by "a real golden fixture is still owed".

The comparison is deliberately not byte-equality. Coordinates come out of ELK and MathJax
through floating-point math, and the last emitted decimal can differ across platforms and BLAS
builds. So: the non-numeric SKELETON must match exactly (every tag, attribute name, class,
content-hashed glyph id, caption and legend string), and each number is compared with a
tolerance of one emitted decimal. That catches any structural or visible-geometry regression
while ignoring noise no human could see.

Regenerate deliberately with `pytest --golden-update` and read the diff before committing it.
"""

import re

import pytest
from conftest import build_eight_schools

from bayesdag import mathsvg
from bayesdag.convert import to_ir
from bayesdag.layout import layout
from bayesdag.render_svg import to_svg

TOLERANCE = 0.1  # one emitted decimal: `.1f` coordinates, `.3g` glyph values

# Numbers OUTSIDE attribute values that must match exactly. Content-hashed glyph ids
# (`bdg-1a2b3c…`) and their references carry digits but are identities, not measurements: a
# changed hash means changed path data, which is a real regression.
_EXACT_ATTRS = re.compile(
    r'\b(?:id|href|xlink:href|data-c|class|data-node|data-src|data-tgt|data-plate)="[^"]*"'
)
_NUMBER = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def _tokenize(svg: str) -> tuple[str, list[float]]:
    """Split an SVG into (skeleton, numbers): the skeleton keeps identity attributes verbatim and
    replaces every other number with a placeholder, so it compares exactly; the extracted numbers
    compare with tolerance.

    Identity spans are copied through directly rather than masked-and-restored — any placeholder
    scheme would itself have to carry an index, and the number pass would eat it."""
    numbers: list[float] = []

    def _take(m: re.Match) -> str:
        numbers.append(float(m.group(0)))
        return "\x01"

    parts: list[str] = []
    pos = 0
    for m in _EXACT_ATTRS.finditer(svg):
        parts.append(_NUMBER.sub(_take, svg[pos : m.start()]))
        parts.append(m.group(0))  # identity: compared exactly, never numerically
        pos = m.end()
    parts.append(_NUMBER.sub(_take, svg[pos:]))
    return "".join(parts), numbers


def _full() -> str:
    ir = to_ir(build_eight_schools())
    return to_svg(ir, layout(ir), legend=True)


def _overview(build, annotations=()):
    from bayesdag import blocks

    ir = blocks.apply(to_ir(build()), list(annotations))
    ov = blocks.overview(ir)
    return to_svg(ov, layout(ov), legend=True)


def _eight_schools_overview() -> str:
    from structures import build_eight_schools_noncentered

    return _overview(build_eight_schools_noncentered)


def _horseshoe_overview() -> str:
    from structures import build_regularized_horseshoe

    return _overview(lambda: build_regularized_horseshoe(2.0, codata=True))


def _bridge_overview() -> str:
    """The measurement stand-in with its likelihood declared as an annotated block."""
    from structures import build_bridge_measurement_like

    from bayesdag.blocks import BlockSpec

    spec = BlockSpec(
        kind="measurement",
        title="bridge measurement model",
        members=["bridge_marginal", "study_censored", "y_above"],
        notes=["the likelihood: marginal over tubes plus censored study cells"],
        workflow="joint",
    )
    return _overview(build_bridge_measurement_like, [spec])


GOLDENS = {
    "eight_schools_TB.svg": _full,  # the canonical (full) graph
    "eight_schools_overview_TB.svg": _eight_schools_overview,
    "horseshoe_overview_TB.svg": _horseshoe_overview,
    "bridge_measurement_overview_TB.svg": _bridge_overview,
}


@pytest.mark.skipif(
    not mathsvg.get_renderer().available, reason="needs the built mathjax bundle for real labels"
)
@pytest.mark.parametrize("name", list(GOLDENS))
def test_golden_svg(name, golden_path, golden_update):
    """The committed reference renders. Under `--golden-update` this rewrites the fixtures."""
    actual = GOLDENS[name]()
    path = golden_path(name)

    if golden_update:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
        pytest.skip(f"--golden-update: rewrote {path.name}; review the diff before committing")

    assert path.exists(), f"missing golden fixture {path}; regenerate with `pytest --golden-update`"
    expected = path.read_text(encoding="utf-8")

    exp_skeleton, exp_numbers = _tokenize(expected)
    act_skeleton, act_numbers = _tokenize(actual)

    assert act_skeleton == exp_skeleton, (
        "the rendered SVG's structure changed (tags, classes, ids, captions or glyph hashes). "
        "If this is intended, regenerate with `pytest --golden-update` and review the diff."
    )
    assert len(act_numbers) == len(exp_numbers)
    drift = [
        (i, e, a)
        for i, (e, a) in enumerate(zip(exp_numbers, act_numbers, strict=True))
        if abs(e - a) > TOLERANCE
    ]
    assert not drift, (
        f"{len(drift)} coordinate(s) moved by more than {TOLERANCE}px "
        f"(first: expected {drift[0][1]}, got {drift[0][2]}). Geometry changed — regenerate with "
        "`pytest --golden-update` if intended."
    )


def test_golden_tokenizer_catches_what_it_claims_to():
    """The comparison is only worth as much as its sensitivity: a moved box must fail, a
    last-decimal wobble must not, and a changed glyph hash must fail (it means changed paths)."""
    base = '<rect class="bd-chrome" x="10.0" y="20.0"/><path id="bdg-abc123" d="M1.0,2.0"/>'

    wobble = base.replace('x="10.0"', 'x="10.05"')
    s1, n1 = _tokenize(base)
    s2, n2 = _tokenize(wobble)
    assert s1 == s2 and max(abs(a - b) for a, b in zip(n1, n2, strict=True)) <= TOLERANCE

    moved = base.replace('x="10.0"', 'x="48.0"')
    _, n3 = _tokenize(moved)
    assert max(abs(a - b) for a, b in zip(n1, n3, strict=True)) > TOLERANCE

    rehashed = base.replace("bdg-abc123", "bdg-def456")
    assert _tokenize(rehashed)[0] != s1  # identity attrs live in the skeleton, so this fails loudly

    restructured = base.replace('class="bd-chrome"', 'class="bd-frame"')
    assert _tokenize(restructured)[0] != s1
