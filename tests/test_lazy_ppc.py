"""Plate prior-predictive expansion is computed on demand, not for every widget ever built.

Forward-simulating the user's model is the single most expensive thing bayesdag does. Paying for
it at construction put seconds between every re-render, which is what made a slider-driven
rebuild loop unusable — and most of those simulations were for panels nobody ever opened.
"""

import pytest

import bayesdag


@pytest.fixture
def counting_ppc(monkeypatch):
    import bayesdag.adapters.ppc as ppc_mod

    calls = {"n": 0}
    real = ppc_mod.prior_predictive_expansions

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(ppc_mod, "prior_predictive_expansions", counting)
    return calls


def test_building_a_widget_simulates_nothing(counting_ppc, eight_schools_model):
    pytest.importorskip("anywidget")
    spec = bayesdag.view(eight_schools_model).widget().spec
    assert counting_ppc["n"] == 0
    assert spec["plates"] == {}
    # ...but the JS is told which plates CAN be opened, so the affordance is still offered
    assert spec["expandable"] == ["plate_school"]


def test_clicking_a_plate_computes_and_pushes_the_panel(counting_ppc, eight_schools_model):
    """The browser sets `expanded_plate`; Python observes it, simulates once, and pushes the
    panel back through `spec`. That round trip is the whole mechanism."""
    pytest.importorskip("anywidget")
    v = bayesdag.view(eight_schools_model)
    w = v.widget()
    assert counting_ppc["n"] == 0

    w.expanded_plate = "plate_school"  # what the JS does on click

    assert counting_ppc["n"] == 1
    assert "prior predictive" in w.spec["plates"]["plate_school"]["panel"]


def test_a_second_click_is_free(counting_ppc, eight_schools_model):
    """One simulation yields every plate, so they are all cached together."""
    pytest.importorskip("anywidget")
    v = bayesdag.view(eight_schools_model)
    w = v.widget()

    w.expanded_plate = "plate_school"
    w.expanded_plate = ""
    w.expanded_plate = "plate_school"

    assert counting_ppc["n"] == 1


def test_expand_plates_is_callable_without_a_widget(counting_ppc, eight_schools_model):
    """The panels are ordinary data; a script or a test must be able to ask for them directly."""
    v = bayesdag.view(eight_schools_model)
    panels = v.expand_plates()
    assert "plate_school" in panels
    assert v.expand_plates() is panels  # cached
    assert counting_ppc["n"] == 1
    assert v._widget is None


def test_clearing_the_selection_does_not_simulate(counting_ppc, eight_schools_model):
    pytest.importorskip("anywidget")
    w = bayesdag.view(eight_schools_model).widget()
    w.expanded_plate = ""
    assert counting_ppc["n"] == 0
