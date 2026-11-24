"""Wide count supports preserve probability and observed values."""

import numpy as np
import pytest
from scipy.stats import poisson

from bayesdag import glyph
from bayesdag.adapters.glyph_data import _discrete_bars, _pmf, _ppc_discrete
from bayesdag.adapters.ppc import _discrete_curves
from bayesdag.ir import Box


def test_analytic_count_bins_cover_mode_and_account_for_tails():
    data = _pmf("Poisson", [1000])
    assert len(data["bins"]) <= 40
    assert any(a <= 1000 <= b for a, b in data["bins"])
    for (a, b), mass in zip(data["bins"], data["masses"], strict=True):
        assert mass == pytest.approx(poisson.cdf(b, 1000) - poisson.cdf(a - 1, 1000))
    assert sum(data["masses"]) + sum(data["tails"].values()) == pytest.approx(1)
    small = _pmf("Bernoulli", [0.3])
    assert small["bins"] == [(0, 0), (1, 1)]
    assert small["masses"] == pytest.approx([0.7, 0.3])


def test_empirical_bins_include_full_range_and_per_draw_frequencies():
    observed = np.array([0, 10000])
    draws = np.array([[0, 1, 2], [1, 2, 3]])
    data = _ppc_discrete(observed, draws)
    assert len(data["bins"]) <= 40
    assert data["bins"][0][0] == 0 and data["bins"][-1][1] == 10000
    assert sum(data["frequencies"]) == pytest.approx(1)
    assert np.sum(data["pred_frequencies"], axis=1) == pytest.approx([1, 1])
    assert _ppc_discrete(observed, np.array([[np.nan]])) is None
    assert _ppc_discrete(observed, np.empty((0, 3))) is None
    assert _discrete_bars([1, np.nan]) is None
    xs, curves, bins = _discrete_curves(draws.T, 2, observed)
    assert len(xs) <= 40 and bins[-1][1] == 10000
    assert [sum(c) for c in curves] == pytest.approx([1, 1])


def test_wide_bins_are_labelled_in_node_and_plate_panels():
    from bayesdag.ir import GlyphSpec, NodeIR
    from bayesdag.render_svg import render_node_panel, render_plate_panel

    data = _pmf("Poisson", [1000])
    node = NodeIR("count", "latent", glyph=GlyphSpec(kind="bars", source="prior"), glyph_data=data)
    svg = render_node_panel(node)
    assert "probability per bin" in svg and "omitted tails" in svg
    a, b = data["bins"][0]
    assert f"{a}–{b}" in svg
    plate = render_plate_panel(
        {
            "label": "p",
            "members": [
                {
                    "id": "y",
                    "n": 2,
                    "xs": [0.5, 2.5],
                    "bins": [[0, 1], [2, 3]],
                    "curves": [[1.0, 0.5]],
                    "discrete": True,
                    "observed": [0, 3],
                }
            ],
        }
    )
    assert plate.count('stroke="#d2691e"') == 2


def test_bin_titles_are_escaped_and_legacy_bars_render():
    data = {"heights": [1], "binned": True, "bins": [["<bad>", "&"]]}
    svg = glyph.render("bars", data, Box(0, 0, 100, 40))
    assert "&lt;bad&gt;–&amp;" in svg
    assert "<title>" not in glyph.render("bars", {"heights": [1]}, Box(0, 0, 100, 40))
