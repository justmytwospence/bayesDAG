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


def test_widget_ships_identical_svg(eight_schools_model):
    pytest.importorskip("anywidget")
    v = bayesdag.view(eight_schools_model)
    w = v.widget()
    # the widget renders the EXACT bytes the static renderer produced
    assert w.spec["svg"] == v.to_svg()


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
    assert spec["nodes"]["y_obs"]["params"]            # per-node detail (loc/scale)
    assert 'class="bd-node"' in spec["svg"] and 'class="bd-edge"' in spec["svg"]
