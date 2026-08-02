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
    assert "bd-legend" not in w.spec["svg"]  # no legend in the widget
    assert "bd-legend" in v.to_svg()  # ...but yes in the static SVG
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


def test_falls_back_to_static_without_the_interactive_extra(monkeypatch, eight_schools_model):
    """The advertised "falls back to static automatically when interactivity isn't available".
    Both extras are installed in dev, so this arm is only ever reached by forcing it."""
    import importlib

    # NB: `bayesdag.view` the attribute is the FUNCTION (it shadows the submodule of the same
    # name), so the module has to be fetched through the import machinery.
    view_mod = importlib.import_module("bayesdag.view")

    v = bayesdag.view(eight_schools_model)
    monkeypatch.setattr(view_mod, "_interactive_available", lambda: False)
    mb = v._repr_mimebundle_()
    assert mb == {"image/svg+xml": v.to_svg()}


def test_display_degrades_to_static_when_the_widget_fails(monkeypatch, eight_schools_model):
    """marimo's `_display_` must degrade like `_repr_mimebundle_` does. `widget()` builds the
    whole spec (moral graph, prior-predictive panels), so letting it raise would blow up the
    cell instead of showing the figure that was already rendered."""
    mo = pytest.importorskip("marimo")

    v = bayesdag.view(eight_schools_model)
    monkeypatch.setattr(type(v), "widget", lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    out = v._display_()
    assert isinstance(out, mo.Html) and "<svg" in out.text


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
    assert spec["nodes"]["y_obs"]["params"]  # per-node detail (loc/scale)
    assert "<svg" in spec["nodes"]["y_obs"]["panel"]  # observed: histogram + best-fit overlay panel
    assert 'class="bd-node"' in spec["svg"] and 'class="bd-edge"' in spec["svg"]
    assert 'class="bd-plate"' in spec["svg"]


def test_plate_prior_predictive_panel(eight_schools_model):
    pytest.importorskip("anywidget")
    v = bayesdag.view(eight_schools_model)
    plates = v.expand_plates()  # on demand: see test_lazy_ppc.py
    assert "plate_school" in plates
    panel = plates["plate_school"]["panel"]
    assert "prior predictive" in panel
    assert panel.count("<path") >= 3  # overlaid per-instance density curves
    assert "y_obs" in panel  # observed member row (with data ticks)


def test_spec_carries_every_key_the_js_reads(eight_schools_model):
    """The Python/JS contract, pinned from the Python side.

    js/index.js has no tests of its own — deliberately: the "JS computes nothing" invariant keeps
    all the logic here, and a DOM harness for 200 lines of class-toggling would buy little. What
    it cannot catch is a rename on THIS side quietly stranding the front end, since the JS reads
    every field with `?.`-ish defaults and simply renders nothing. So enumerate the contract.

    Keep this list in sync with js/index.js when either side changes.
    """
    pytest.importorskip("anywidget")
    spec = bayesdag.view(eight_schools_model, widget_legend=False).widget().spec

    assert set(spec) >= {"svg", "nodes", "plates"}

    # per-node detail: tooltip, pinned card, constructor line, and the highlight/trace sets
    required = {
        "role",
        "dist",
        "observed",
        "dims",
        "params",
        "transform",
        "parents",
        "children",
        "blanket",
        "ancestors",
        "descendants",
        "label_svg",
        "diag",
    }
    for nid, node in spec["nodes"].items():
        missing = required - set(node)
        assert not missing, f"{nid} is missing {sorted(missing)} that js/index.js reads"
        for p in node["params"]:
            assert set(p) >= {"name", "value"}  # paramsText()/constructorText()

    # plate panels are looked up as plates[pid].panel
    for pid, plate in spec["plates"].items():
        assert "panel" in plate, f"plate {pid} has no panel key"

    # the SVG hooks the JS queries for: selectors and the data-* attributes it keys off
    svg = spec["svg"]
    for hook in (
        'class="bd-node"',
        'class="bd-edge"',
        'class="bd-plate"',
        'class="bd-chrome"',
        "data-node=",
        "data-src=",
        "data-tgt=",
        "data-plate=",
    ):
        assert hook in svg, f"js/index.js queries {hook} but the emitter no longer produces it"

    # the arrowhead markers the pinned-trace CSS swaps in must exist in <defs>
    for marker in ("bd-arrow", "bd-arrow-up", "bd-arrow-down"):
        assert f'id="{marker}"' in svg


def test_ppc_draws_zero_skips_the_forward_simulation(monkeypatch, eight_schools_model):
    """Building a widget forward-simulates the user's model for the plate panels — the one place
    bayesdag samples anything. `ppc_draws=0` must opt out of it entirely, not just discard it."""
    pytest.importorskip("anywidget")
    import bayesdag.adapters.ppc as ppc_mod

    monkeypatch.setattr(
        ppc_mod,
        "prior_predictive_expansions",
        lambda *a, **k: pytest.fail("prior predictive ran despite ppc_draws=0"),
    )
    v = bayesdag.view(eight_schools_model, ppc_draws=0)
    assert v.widget().spec["plates"] == {}
    assert v.widget().spec["expandable"] == []  # no affordance offered either
    assert v.expand_plates() == {}  # even when asked directly


def test_ppc_draws_is_threaded_through(monkeypatch, eight_schools_model):
    pytest.importorskip("anywidget")
    import bayesdag.adapters.ppc as ppc_mod

    seen = {}
    real = ppc_mod.prior_predictive_expansions

    def spy(model, ir, draws=200):
        seen["draws"] = draws
        return real(model, ir, draws=draws)

    monkeypatch.setattr(ppc_mod, "prior_predictive_expansions", spy)
    bayesdag.view(eight_schools_model, ppc_draws=25).expand_plates()
    assert seen["draws"] == 25


def test_failing_prior_predictive_still_builds_a_widget(monkeypatch, caplog, eight_schools_model):
    """A model whose forward simulation raises must degrade to "no plate panels" — visibly in the
    log, never as a silently swallowed exception."""
    pytest.importorskip("anywidget")
    import logging

    import bayesdag.adapters.ppc as ppc_mod

    def boom(*a, **k):
        raise RuntimeError("no forward sampling here")

    monkeypatch.setattr(ppc_mod, "prior_predictive_expansions", boom)
    v = bayesdag.view(eight_schools_model)
    with caplog.at_level(logging.DEBUG, logger="bayesdag.view"):
        assert v.expand_plates() == {}
    assert "<svg" in v.widget().spec["svg"]  # the diagram itself is unaffected
    assert any("prior-predictive" in r.message for r in caplog.records)


def test_rich_glyph_nodes_get_card_panels():
    """Every glyph-bearing node ships a large pinned-card panel — not just observed nodes.
    Panels are widget-only: the static SVG bytes are untouched (parity)."""
    import xml.etree.ElementTree as ET

    import numpy as np
    import pymc as pm

    pytest.importorskip("anywidget")
    cov = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.3], [0.2, 0.3, 1.0]])
    with pm.Model(coords={"axis": ["a", "b", "c"]}) as m:
        pm.MvNormal("z", mu=np.zeros(3), cov=cov, dims="axis")
        pm.Normal("mu", 0, 1)
    v = bayesdag.view(m)
    spec = v.widget().spec
    panel = spec["nodes"]["z"].get("panel")
    assert panel and "<svg" in panel
    ET.fromstring(panel)  # well-formed standalone SVG
    for coord in ("a", "b", "c"):  # coord labels on the matrix rows
        assert f">{coord}</text>" in panel
    assert "panel" in spec["nodes"]["mu"]  # plain latent density gets one too
    assert "bd-card" not in v.to_svg()  # static SVG carries no panels


def test_on_select_fires_and_unsubscribes(eight_schools_model):
    """`selected_node` was synced in both directions but read by nobody — "linked views" was
    aspirational. The callback form is the Jupyter/script half of making it real."""
    pytest.importorskip("anywidget")
    v = bayesdag.view(eight_schools_model, ppc_draws=0)
    seen = []
    unsubscribe = v.on_select(seen.append)

    v.widget().selected_node = "tau"  # what the JS does on click
    assert seen == ["tau"]
    assert v.selected_node == "tau"

    v.widget().selected_node = ""  # background click clears the pin
    assert seen == ["tau", ""]

    unsubscribe()
    v.widget().selected_node = "mu"
    assert seen == ["tau", ""], "unsubscribe left the handler attached"


def test_selected_node_is_empty_before_the_widget_exists(eight_schools_model):
    v = bayesdag.view(eight_schools_model, ppc_draws=0)
    assert v.selected_node == ""  # must not force the widget (and its spec) into existence
    assert v._widget is None


def test_ui_returns_a_readable_marimo_element(eight_schools_model):
    """marimo's anywidget wrapper exposes every synced trait through `.value`, so a neighbouring
    cell can read the selection with no JS and no callback. It also caches on widget identity,
    so calling ui() twice must not fork the state into two elements."""
    pytest.importorskip("anywidget")
    pytest.importorskip("marimo")

    v = bayesdag.view(eight_schools_model, ppc_draws=0)
    ui = v.ui()
    assert "selected_node" in ui.value
    assert ui.value["selected_node"] == ""

    v.widget().selected_node = "theta"
    assert v.ui().value["selected_node"] == "theta"  # same element, updated value
