// bayesdag anywidget front-end (thin controller).
//
// INVARIANT (AGENTS.md): never computes geometry or statistics. Python ships a fully
// laid-out SVG in `spec.svg` plus `spec.nodes` (per-node detail + adjacency) and
// `spec.plates` (prior-predictive panels). The JS injects the SVG verbatim (parity with
// the static renderer) and adds, via CSS classes / DOM overlays:
//   - hover-highlight of a node's Markov blanket + a tooltip,
//   - click-to-pin a node detail card,
//   - click-to-expand a plate's prior-predictive check.
// (No pan/zoom: the diagram renders at its natural size.)

function esc(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

function paramsText(node) {
  if (!node.params || !node.params.length) return "";
  return node.params.map((p) => `${p.name}=${p.value ?? "?"}`).join(", ");
}

function constructorText(id, node) {
  if (!node.dist) return "";
  // a param whose value carries LaTeX markup (or is long) can't be shown as plain Python — elide it
  const args = (node.params || [])
    .map((p) => {
      const v = (p.value ?? "").trim();
      if (!v) return "";
      return /[\\{}^]/.test(v) || v.length > 18 ? "…" : v;
    })
    .filter(Boolean);
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

    const panel = document.createElement("div");
    panel.className = "bd-panel";
    panel.style.display = "none";
    el.appendChild(panel);

    function draw() {
      const spec = model.get("spec") || {};
      const nodes = spec.nodes || {};
      for (const c of Array.from(el.childNodes)) {
        if (c !== tooltip && c !== card && c !== panel) el.removeChild(c);
      }
      const holder = document.createElement("div");
      holder.innerHTML = spec.svg || '<div class="bd-placeholder">bayesdag &mdash; no spec</div>';
      // model-level sampling note (divergences), above the figure — Python writes the wording
      if (spec.diagnostics) {
        const strip = document.createElement("div");
        strip.className = "bd-diag-strip";
        strip.textContent = spec.diagnostics;
        holder.insertBefore(strip, holder.firstChild);
      }
      el.insertBefore(holder, tooltip);
      const svg = holder.querySelector("svg");
      if (!svg) return;

      const nodeEls = Array.from(svg.querySelectorAll(".bd-node"));
      const edgeEls = Array.from(svg.querySelectorAll(".bd-edge"));
      let pinned = null;

      const NODE_HL = ["bd-dim", "bd-focus", "bd-up", "bd-down"];
      const EDGE_HL = ["bd-dim", "bd-edge-up", "bd-edge-down"];

      // hover: fade everything outside the focused node's Markov blanket (local, crisp).
      function highlight(id) {
        const keep = new Set([id, ...((nodes[id] && nodes[id].blanket) || [])]);
        nodeEls.forEach((n) => {
          n.classList.remove("bd-focus", "bd-up", "bd-down");
          n.classList.toggle("bd-dim", !keep.has(n.dataset.node));
        });
        edgeEls.forEach((e) => {
          e.classList.remove("bd-edge-up", "bd-edge-down");
          e.classList.toggle("bd-dim", e.dataset.src !== id && e.dataset.tgt !== id);
        });
      }
      // pin (click): directional causal trace — upstream (ancestors) vs downstream
      // (descendants), so the full lineage reads as a flow instead of one flat dim/undim split.
      function trace(id) {
        const n = nodes[id] || {};
        const up = new Set(n.ancestors || []);
        const down = new Set(n.descendants || []);
        nodeEls.forEach((el) => {
          const k = el.dataset.node;
          el.classList.remove(...NODE_HL);
          if (k === id) el.classList.add("bd-focus");
          else if (up.has(k)) el.classList.add("bd-up");
          else if (down.has(k)) el.classList.add("bd-down");
          else el.classList.add("bd-dim");
        });
        edgeEls.forEach((e) => {
          const s = e.dataset.src;
          const t = e.dataset.tgt;
          e.classList.remove(...EDGE_HL);
          if (up.has(s) && (up.has(t) || t === id)) e.classList.add("bd-edge-up");
          else if (down.has(t) && (down.has(s) || s === id)) e.classList.add("bd-edge-down");
          else e.classList.add("bd-dim");
        });
      }
      function clearHl() {
        nodeEls.forEach((n) => n.classList.remove(...NODE_HL));
        edgeEls.forEach((e) => e.classList.remove(...EDGE_HL));
      }
      function showTooltip(id, ev) {
        const n = nodes[id];
        if (!n) return;
        // render the SAME MathJax SVG as the diagram; fall back to a plain relation
        const math = n.label_svg
          ? `<div class="bd-math">${n.label_svg}</div>`
          : `<div>${n.dist ? `~ ${esc(n.dist)}(${esc(paramsText(n))})` : "deterministic"}</div>`;
        const dims = n.dims && n.dims.length ? `<div class="bd-dim-line">dims: ${esc(n.dims.join(" × "))}</div>` : "";
        tooltip.innerHTML = `<b>${esc(id)}</b> · ${esc(n.role)}${math}${dims}`;
        const r = el.getBoundingClientRect();
        tooltip.style.left = ev.clientX - r.left + 12 + "px";
        tooltip.style.top = ev.clientY - r.top + 12 + "px";
        tooltip.style.display = "block";
      }
      function showCard(id) {
        const n = nodes[id];
        if (!n) return;
        const rows = [`<div class="bd-card-title">${esc(id)} <span>${esc(n.role)}</span></div>`];
        if (n.label_svg) rows.push(`<div class="bd-math">${n.label_svg}</div>`);
        if (n.dist) rows.push(`<div>distribution: <b>${esc(n.dist)}</b></div>`);
        if (n.dims && n.dims.length) rows.push(`<div>dims: ${esc(n.dims.join(" × "))}</div>`);
        if (n.transform) rows.push(`<div>transform: ${esc(n.transform)}</div>`);
        if (n.diag && n.diag.length) {
          rows.push(
            `<div class="bd-diag-rows">${n.diag.map((d) => `<div>${esc(d)}</div>`).join("")}</div>`
          );
        }
        const ctor = constructorText(id, n);
        if (ctor) rows.push(`<pre class="bd-ctor">${esc(ctor)}</pre>`);
        // observed nodes: the Python-rendered histogram + best-fit-family overlay (data vs theory)
        if (n.panel) rows.push(`<div class="bd-overlay">${n.panel}</div>`);
        rows.push('<div class="bd-card-hint">click empty space to close</div>');
        card.innerHTML = rows.join("");
        card.style.display = "block";
      }

      nodeEls.forEach((nodeEl) => {
        const id = nodeEl.dataset.node;
        nodeEl.style.cursor = "pointer";
        nodeEl.addEventListener("mouseenter", (ev) => {
          if (!pinned) {
            highlight(id);
            showTooltip(id, ev);
          }
        });
        nodeEl.addEventListener("mousemove", (ev) => {
          if (!pinned) showTooltip(id, ev);
        });
        nodeEl.addEventListener("mouseleave", () => {
          if (!pinned) {
            clearHl();
            tooltip.style.display = "none";
          }
        });
        nodeEl.addEventListener("click", (ev) => {
          ev.stopPropagation();
          pinned = id;
          model.set("selected_node", id);
          model.save_changes();
          trace(id);
          tooltip.style.display = "none";
          panel.style.display = "none";
          showCard(id);
        });
      });

      Array.from(svg.querySelectorAll(".bd-plate")).forEach((plEl) => {
        const pid = plEl.dataset.plate;
        // only advertise the affordance when there is actually a panel behind it
        if ((spec.plates || {})[pid]) plEl.style.cursor = "zoom-in";
        plEl.addEventListener("click", (ev) => {
          // bail BEFORE stopPropagation, so a panel-less plate never swallows the
          // background click that dismisses the pinned card
          const p = (spec.plates || {})[pid];
          if (!p || !p.panel) return;
          ev.stopPropagation();
          card.style.display = "none";
          panel.innerHTML =
            `<div class="bd-panel-head">${esc(pid)}<span>click empty space to close</span></div>` + p.panel;
          panel.style.display = "block";
        });
      });

      svg.addEventListener("click", () => {
        pinned = null;
        model.set("selected_node", "");
        model.save_changes();
        clearHl();
        card.style.display = "none";
        panel.style.display = "none";
      });

      // A spec push (view.update(...)) redraws from scratch, and `pinned` lives in this closure.
      // Without this the user's selection silently drops on every update — which is exactly when
      // they are watching one node. Restore it from the synced trait; classes/innerHTML only.
      const wasPinned = model.get("selected_node");
      if (wasPinned && nodes[wasPinned]) {
        pinned = wasPinned;
        trace(wasPinned);
        showCard(wasPinned);
      }
    }

    draw();
    model.on("change:spec", draw);
    // anywidget calls this when the view goes away (cell re-run, notebook close). Without it the
    // model keeps a reference to `draw` — and to this `el` — for every view ever rendered, so a
    // re-executed cell redraws through all its predecessors' dead DOM.
    return () => {
      model.off("change:spec", draw);
    };
  },
};
