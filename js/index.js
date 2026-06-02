// bayesdag anywidget front-end (thin controller).
//
// INVARIANT (AGENTS.md): never computes geometry or statistics. Python ships a fully
// laid-out SVG in `spec.svg`; the JS injects it verbatim (parity with the static
// renderer) and adds only pan/zoom + hover/selection via transforms/classes.

import { select } from "d3-selection";
import { zoom } from "d3-zoom";

const NS = "http://www.w3.org/2000/svg";

export default {
  render({ model, el }) {
    el.classList.add("bayesdag");

    function draw() {
      const spec = model.get("spec") || {};
      el.innerHTML =
        spec.svg || '<div class="bayesdag-placeholder">bayesdag &mdash; no spec yet</div>';
      const svg = el.querySelector("svg");
      if (!svg) return;

      // Wrap all drawn content in a <g> so we can pan/zoom without touching geometry.
      const g = document.createElementNS(NS, "g");
      const defs = svg.querySelector("defs");
      for (const child of Array.from(svg.childNodes)) {
        if (child === defs) continue; // keep <defs> (markers) at the svg root
        g.appendChild(child);
      }
      svg.appendChild(g);

      const z = zoom()
        .scaleExtent([0.2, 8])
        .on("zoom", (ev) => g.setAttribute("transform", ev.transform.toString()));
      select(svg).call(z).style("cursor", "grab");
    }

    draw();
    model.on("change:spec", draw);
  },
};
