// bayesdag anywidget front-end (thin controller).
//
// INVARIANT (see AGENTS.md): this module NEVER computes geometry or statistics.
// Python ships a fully laid-out SVG plus a `spec` (ModelIR + LayoutResult + glyph
// fragments + adjacency + precomputed aux-view data). The JS only injects that SVG
// and adds pan/zoom, hover, selection, highlight, and collapse via class/transform
// toggles. This is what guarantees static == interactive by construction.
//
// M0 stub: inject the shipped SVG and wire pan/zoom. The full controller
// (Markov-blanket highlight, plate collapse, linked aux panels) lands in task #9.

export default {
  render({ model, el }) {
    el.classList.add("bayesdag");
    const draw = () => {
      const spec = model.get("spec") || {};
      el.innerHTML =
        spec.svg ||
        '<div class="bayesdag-placeholder">bayesdag widget &mdash; no spec yet</div>';
    };
    draw();
    model.on("change:spec", draw);
  },
};
