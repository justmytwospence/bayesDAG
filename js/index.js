// bayesdag anywidget front-end (thin controller).
//
// INVARIANT (AGENTS.md): never computes geometry or statistics. Python ships a fully
// laid-out SVG in `spec.svg` plus `spec.nodes` (per-node detail + adjacency). The JS
// injects the SVG verbatim (parity with the static renderer) and adds only pan/zoom,
// hover-highlight (Markov blanket), a tooltip, and a click-to-pin detail card via
// CSS classes / DOM overlays.

import { select } from "d3-selection";
import { zoom } from "d3-zoom";

const NS = "http://www.w3.org/2000/svg";

function esc(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

function paramsText(node) {
  if (!node.params || !node.params.length) return "";
  return node.params.map((p) => `${p.name}=${p.value ?? "?"}`).join(", ");
}

function constructorText(id, node) {
  if (!node.dist) return "";
  const args = (node.params || []).map((p) => (p.value ?? "").replace(/[\\{}]/g, "")).filter(Boolean);
  const dims = node.dims && node.dims.length ? `, dims=${JSON.stringify(node.dims)}` : "";
  return `pm.${node.dist}("${id}", ${args.join(", ")}${dims})`;
}

export default {
  render({ model, el }) {
    el.classList.add("bayesdag");
    el.style.position = "relative";

    const tooltip = document.createElement("div");
    tooltip.className = "bd-tooltip";
    tooltip.style.display = "none";
    el.appendChild(tooltip);

    const card = document.createElement("div");
    card.className = "bd-card";
    card.style.display = "none";
    el.appendChild(card);

    function draw() {
      const spec = model.get("spec") || {};
      const nodes = spec.nodes || {};
      // wipe previous render but keep tooltip/card
      for (const c of Array.from(el.childNodes)) {
        if (c !== tooltip && c !== card) el.removeChild(c);
      }
      const holder = document.createElement("div");
      holder.innerHTML = spec.svg || '<div class="bd-placeholder">bayesdag &mdash; no spec</div>';
      el.insertBefore(holder, tooltip);
      const svg = holder.querySelector("svg");
      if (!svg) return;

      // wrap drawable content (not <defs>) in a <g> for pan/zoom
      const g = document.createElementNS(NS, "g");
      const defs = svg.querySelector("defs");
      for (const child of Array.from(svg.childNodes)) {
        if (child !== defs) g.appendChild(child);
      }
      svg.appendChild(g);
      const z = zoom().scaleExtent([0.2, 8]).on("zoom", (ev) =>
        g.setAttribute("transform", ev.transform.toString())
      );
      select(svg).call(z).style("cursor", "grab");

      const nodeEls = Array.from(svg.querySelectorAll(".bd-node"));
      const edgeEls = Array.from(svg.querySelectorAll(".bd-edge"));
      let pinned = null;

      function highlight(id) {
        const keep = new Set([id, ...((nodes[id] && nodes[id].blanket) || [])]);
        nodeEls.forEach((n) => n.classList.toggle("bd-dim", !keep.has(n.dataset.node)));
        edgeEls.forEach((e) =>
          e.classList.toggle("bd-dim", e.dataset.src !== id && e.dataset.tgt !== id)
        );
      }
      function clear() {
        nodeEls.forEach((n) => n.classList.remove("bd-dim"));
        edgeEls.forEach((e) => e.classList.remove("bd-dim"));
      }
      function showTooltip(id, ev) {
        const n = nodes[id];
        if (!n) return;
        const rel = n.dist ? `~ ${n.dist}(${esc(paramsText(n))})` : "deterministic";
        const dims = n.dims && n.dims.length ? `<div class="bd-dim-line">dims: ${esc(n.dims.join(" × "))}</div>` : "";
        tooltip.innerHTML = `<b>${esc(id)}</b> · ${esc(n.role)}<div>${rel}</div>${dims}`;
        const r = el.getBoundingClientRect();
        tooltip.style.left = ev.clientX - r.left + 12 + "px";
        tooltip.style.top = ev.clientY - r.top + 12 + "px";
        tooltip.style.display = "block";
      }
      function showCard(id) {
        const n = nodes[id];
        if (!n) return;
        const rows = [`<div class="bd-card-title">${esc(id)} <span>${esc(n.role)}</span></div>`];
        if (n.dist) rows.push(`<div>distribution: <b>${esc(n.dist)}</b></div>`);
        if (n.params && n.params.length)
          rows.push(`<div>parameters: ${esc(paramsText(n))}</div>`);
        if (n.dims && n.dims.length) rows.push(`<div>dims: ${esc(n.dims.join(" × "))}</div>`);
        if (n.transform) rows.push(`<div>transform: ${esc(n.transform)}</div>`);
        const ctor = constructorText(id, n);
        if (ctor) rows.push(`<pre class="bd-ctor">${esc(ctor)}</pre>`);
        rows.push('<div class="bd-card-hint">click empty space to close</div>');
        card.innerHTML = rows.join("");
        card.style.display = "block";
      }

      nodeEls.forEach((nodeEl) => {
        const id = nodeEl.dataset.node;
        nodeEl.style.cursor = "pointer";
        nodeEl.addEventListener("mouseenter", (ev) => {
          if (pinned) return;
          highlight(id);
          showTooltip(id, ev);
        });
        nodeEl.addEventListener("mousemove", (ev) => {
          if (!pinned) showTooltip(id, ev);
        });
        nodeEl.addEventListener("mouseleave", () => {
          if (pinned) return;
          clear();
          tooltip.style.display = "none";
        });
        nodeEl.addEventListener("click", (ev) => {
          ev.stopPropagation();
          pinned = id;
          model.set("selected_node", id);
          model.save_changes();
          highlight(id);
          tooltip.style.display = "none";
          showCard(id);
        });
      });

      svg.addEventListener("click", () => {
        pinned = null;
        model.set("selected_node", "");
        model.save_changes();
        clear();
        card.style.display = "none";
      });
    }

    draw();
    model.on("change:spec", draw);
  },
};
