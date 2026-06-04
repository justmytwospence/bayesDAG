"""ModelGraphView: outputs, save, and the widget/static parity guarantee."""

import pytest

import bayesdag
from bayesdag.view import ModelGraphView


def test_view_produces_svg(eight_schools_model):
    v = bayesdag.view(eight_schools_model)
    s = v.to_svg()
    assert s.lstrip().startswith("<svg")
    assert v._repr_svg_() == s


def test_view_accepts_ir(eight_schools_ir):
    v = ModelGraphView(eight_schools_ir)
    assert "<svg" in v.to_svg()


def test_save(tmp_path, eight_schools_model):
    v = bayesdag.view(eight_schools_model)
    assert v.save(tmp_path / "m.svg").exists()


def test_widget_ships_identical_graph_svg(eight_schools_model):
    pytest.importorskip("anywidget")
    from bayesdag.render_svg import to_svg

    # The widget omits the legend by default (hover covers it), but the GRAPH itself is
    # byte-identical to the static renderer (same LayoutResult + emitter) — parity holds.
    v = bayesdag.view(eight_schools_model)
    w = v.widget()
    assert "bd-legend" not in w.spec["svg"]                       # no legend in the widget
    assert "bd-legend" in v.to_svg()                              # ...but yes in the static SVG
    assert w.spec["svg"] == to_svg(v.ir, v.layout, legend=False)  # identical graph bytes


def test_widget_legend_opt_in(eight_schools_model):
    pytest.importorskip("anywidget")
    v = bayesdag.view(eight_schools_model, widget_legend=True)
    assert "bd-legend" in v.widget().spec["svg"]


def test_repr_mimebundle(eight_schools_model):
    mb = bayesdag.view(eight_schools_model)._repr_mimebundle_()
    # either a data dict (static fallback) or a (data, metadata) tuple (anywidget)
    data = mb[0] if isinstance(mb, tuple) else mb
    assert isinstance(data, dict) and data


def test_widget_spec_has_nodes_adjacency_and_tags(eight_schools_model):
    pytest.importorskip("anywidget")
    spec = bayesdag.view(eight_schools_model).widget().spec
    assert "svg" in spec and "nodes" in spec
    assert set(spec["nodes"]) == {"mu", "tau", "eta", "theta", "y_obs"}
    assert "theta" in spec["nodes"]["tau"]["blanket"]  # Markov blanket adjacency
    # transitive lineage for the directional pin trace (mu -> theta -> y_obs)
    assert spec["nodes"]["mu"]["descendants"] == ["theta", "y_obs"]
    assert spec["nodes"]["mu"]["ancestors"] == []
    assert set(spec["nodes"]["y_obs"]["ancestors"]) == {"mu", "tau", "eta", "theta"}
    assert spec["nodes"]["y_obs"]["descendants"] == []
    assert spec["nodes"]["y_obs"]["params"]            # per-node detail (loc/scale)
    assert "<svg" in spec["nodes"]["y_obs"]["panel"]   # observed: histogram + best-fit overlay panel
    assert 'class="bd-node"' in spec["svg"] and 'class="bd-edge"' in spec["svg"]
    assert 'class="bd-plate"' in spec["svg"]


def test_plate_prior_predictive_panel(eight_schools_model):
    pytest.importorskip("anywidget")
    plates = bayesdag.view(eight_schools_model).widget().spec.get("plates", {})
    assert "plate_school" in plates
    panel = plates["plate_school"]["panel"]
    assert "prior predictive" in panel
    assert panel.count("<path") >= 3  # overlaid per-instance density curves
    assert "y_obs" in panel           # observed member row (with data ticks)
