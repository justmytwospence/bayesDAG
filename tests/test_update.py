"""`ModelGraphView.update(idata=...)` — attaching a posterior to a diagram already on screen.

The promise is that priors become posteriors IN PLACE: same layout, same node boxes, nothing
jumping. That only holds while no node changes size, so the honest part of the feature is the
check that decides between reusing the layout and laying out again.
"""

import numpy as np
import pymc as pm
import pytest

import bayesdag


def _fit(model, **kw):
    with model:
        return pm.sample(
            draws=kw.get("draws", 60),
            tune=60,
            chains=2,
            cores=1,
            random_seed=0,
            progressbar=False,
            compute_convergence_checks=False,
        )


@pytest.fixture(scope="module")
def fitted_eight_schools():
    y = np.array([28.0, 8, -3, 7, -1, 1, 18, 12])
    sigma = np.array([15.0, 10, 16, 11, 9, 11, 10, 18])
    with pm.Model(coords={"school": [f"S{i}" for i in range(8)]}) as m:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfNormal("tau", 5)
        eta = pm.Normal("eta", 0, 1, dims="school")
        theta = pm.Deterministic("theta", mu + tau * eta, dims="school")
        pm.Normal("y_obs", theta, sigma, observed=y, dims="school")
    return m, _fit(m)


def _sources(view):
    return {n.id: (n.glyph.source if n.glyph else None) for n in view.ir.nodes}


def _boxes(view):
    return {k: (b.x, b.y, b.w, b.h) for k, b in view.layout.node_boxes.items()}


def test_the_diagram_does_not_move_when_the_posterior_arrives(fitted_eight_schools):
    """The headline: same LayoutResult object, same boxes, prior curves replaced by posteriors."""
    model, idata = fitted_eight_schools
    v = bayesdag.view(model, ppc_draws=0)
    before_layout, before_boxes = v.layout, _boxes(v)

    v.update(idata=idata)

    assert v.layout is before_layout  # literally reused, not recomputed to the same numbers
    assert _boxes(v) == before_boxes
    assert _sources(v) == {
        "mu": "posterior_kde",
        "tau": "posterior_kde",
        "eta": "posterior_kde",
        "theta": None,  # a deterministic depicts its transfer function, never a posterior
        "y_obs": "observed_hist",  # observed nodes keep their data
    }
    assert "#d2691e" in v.to_svg()  # the posterior colour reaches the figure


def test_update_none_restores_the_as_built_view(fitted_eight_schools):
    """Prior <-> posterior is a toggle, so the way back has to be exact, not approximate."""
    model, idata = fitted_eight_schools
    v = bayesdag.view(model, ppc_draws=0)
    prior_svg, prior_sources = v.to_svg(), _sources(v)

    v.update(idata=idata)
    assert v.to_svg() != prior_svg

    v.update(None)
    assert v.to_svg() == prior_svg
    assert _sources(v) == prior_sources


def test_update_returns_self_for_chaining(fitted_eight_schools):
    model, idata = fitted_eight_schools
    v = bayesdag.view(model, ppc_draws=0)
    assert v.update(idata=idata) is v


def test_overlay_refs_follow_the_attached_idata(fitted_eight_schools):
    model, idata = fitted_eight_schools
    v = bayesdag.view(model, ppc_draws=0)
    assert all(not n.overlays for n in v.ir.nodes)

    v.update(idata=idata)
    assert [o.idata_group for o in v.ir.node("mu").overlays] == ["posterior"]

    v.update(None)
    assert all(not n.overlays for n in v.ir.nodes)


def test_a_glyph_that_changes_size_class_forces_a_relayout():
    """An MvNormal's prior is a pairplot — a ~90px square. Its posterior is a pooled KDE strip of
    30px. Reusing the old boxes there would leave the node sized for a glyph it no longer draws,
    so the size check has to catch it and lay out again."""
    from bayesdag import geometry

    cov = np.array([[1.0, 0.6], [0.6, 1.0]])
    with pm.Model(coords={"axis": ["a", "b"]}) as m:
        pm.MvNormal("z", mu=np.zeros(2), cov=cov, dims="axis")
    idata = _fit(m)

    v = bayesdag.view(m, ppc_draws=0)
    assert v.ir.node("z").glyph.kind == "pairplot"
    before_layout, before_boxes = v.layout, _boxes(v)

    v.update(idata=idata)

    assert v.ir.node("z").glyph.kind == "density"  # posterior wins over the construct glyph
    assert v.layout is not before_layout, "the size class changed; the layout must be redone"
    assert _boxes(v) != before_boxes
    # and the new box is the one the new glyph actually wants
    n = v.ir.node("z")
    lw, lh = geometry.label_px_size(n.label_svg)
    want_w, want_h = geometry.node_size(lw, lh, n.glyph.kind, n.glyph_data)
    box = v.layout.node_boxes["z"]
    assert abs(box.w - want_w) < 0.5 and abs(box.h - want_h) < 0.5


def test_the_widget_is_pushed_and_stays_in_parity(fitted_eight_schools):
    """Parity is the load-bearing invariant: whatever the widget shows must be the same bytes the
    static renderer produces. An update path that re-emits one and not the other would break it
    silently, since nothing else compares them after construction."""
    pytest.importorskip("anywidget")
    from bayesdag.render_svg import to_svg

    model, idata = fitted_eight_schools
    v = bayesdag.view(model, ppc_draws=0)
    w = v.widget()
    before = w.spec["svg"]

    v.update(idata=idata)

    assert w.spec["svg"] != before  # the push actually happened
    assert w.spec["svg"] == to_svg(v.ir, v.layout, legend=False)  # ...and parity survived it
    assert w.spec["nodes"]["mu"]["panel"]  # the card panel was rebuilt too


def test_update_never_resamples_the_prior_predictive(monkeypatch, fitted_eight_schools):
    """The plate panels describe the PRIOR, so they cannot change when a posterior arrives.
    Recomputing them would forward-simulate the user's model on every push — seconds of work, per
    update, for bytes that are identical."""
    pytest.importorskip("anywidget")
    import bayesdag.adapters.ppc as ppc_mod

    model, idata = fitted_eight_schools
    calls = {"n": 0}
    real = ppc_mod.prior_predictive_expansions

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(ppc_mod, "prior_predictive_expansions", counting)

    v = bayesdag.view(model)
    w = v.widget()
    assert calls["n"] == 1
    panels_before = w.spec["plates"]

    v.update(idata=idata)
    v.update(None)
    v.update(idata=idata)

    assert calls["n"] == 1, "the prior predictive was recomputed on a spec push"
    assert w.spec["plates"] == panels_before


def test_update_works_without_a_widget(fitted_eight_schools):
    """A static/script user must be able to attach a posterior without anywidget in the picture."""
    model, idata = fitted_eight_schools
    v = bayesdag.view(model, ppc_draws=0)
    v.update(idata=idata)
    assert v._widget is None
    assert "#d2691e" in v.to_svg()
