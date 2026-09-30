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

// The op-level parameter line: `Normal(loc = μ, scale = τ)`. Deliberately NOT a `pm.Normal(...)`
// call: op-level slots are reparametrized and reordered relative to PyMC's public constructor
// (HalfNormal's op takes `(loc, scale)`), so printing them as code would show something that is
// not valid PyMC.
function familyText(node) {
  if (!node.dist) return "";
  const args = (node.params || []).map((p) => {
    const v = (p.value ?? "").trim();
    return `${p.name} = ${!v || /[\\{}^]/.test(v) || v.length > 18 ? "…" : v}`;
  });
  return `${node.dist}(${args.join(", ")})`;
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
      const toolbar = document.createElement("div");
      toolbar.className = "bd-toolbar";
      const fitBtn = document.createElement("button");
      fitBtn.type = "button";
      fitBtn.textContent = "fit ↔ 1:1";
      fitBtn.title = "toggle between natural size (scroll) and fit-to-width";
      fitBtn.setAttribute("aria-pressed", String(el.classList.contains("bd-fit")));
      fitBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        el.classList.toggle("bd-fit");
        fitBtn.setAttribute("aria-pressed", String(el.classList.contains("bd-fit")));
      });
      toolbar.appendChild(fitBtn);
      if (spec.has_overview) {
        // Python re-projects and lays out; the JS only flips the trait
        const detailBtn = document.createElement("button");
        detailBtn.type = "button";
        const full = spec.detail === "full";
        detailBtn.textContent = full ? "show overview" : "show all variables";
        detailBtn.title = "overview: recognized structures as single boxes · full: every variable";
        detailBtn.setAttribute("aria-pressed", String(full));
        detailBtn.addEventListener("click", (ev) => {
          ev.stopPropagation();
          model.set("detail", full ? "overview" : "full");
          model.save_changes();
        });
        toolbar.appendChild(detailBtn);
      }
      const holder = document.createElement("div");
      holder.className = "bd-holder";
      holder.innerHTML = spec.svg || '<div class="bd-placeholder">bayesdag &mdash; no spec</div>';
      // model-level sampling note (divergences), above the figure — Python writes the wording
      for (const note of [spec.diagnostics, spec.ppc_note]) {
        if (!note) continue;
        const strip = document.createElement("div");
        strip.className = "bd-diag-strip";
        strip.textContent = note;
        holder.insertBefore(strip, holder.firstChild);
      }
      el.insertBefore(toolbar, tooltip);
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
      // pin (click): directional lineage trace — upstream (ancestors) vs downstream
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
        // the container scrolls (natural size), and the tooltip lives in its scrolled content
        tooltip.style.left = ev.clientX - r.left + el.scrollLeft + 12 + "px";
        tooltip.style.top = ev.clientY - r.top + el.scrollTop + 12 + "px";
        tooltip.style.display = "block";
      }
      function blockRows(bid) {
        const b = (spec.blocks || {})[bid];
        if (!b) return [];
        const rows = [];
        const prov = b.provenance === "annotated" ? "annotated by the user" : "read off the graph";
        rows.push(`<div class="bd-block-sub">${esc(prov)}${b.workflow ? " · " + esc(b.workflow.replace("_", " ")) : ""}</div>`);
        for (const [role, names] of Object.entries(b.roles || {})) {
          rows.push(`<div class="bd-role"><span>${esc(role.replace("_", " "))}</span> ${esc(names.join(", "))}</div>`);
        }
        (b.notes || []).forEach((t) => rows.push(`<div class="bd-note">${esc(t)}</div>`));
        if (b.references && b.references.length) {
          rows.push(`<div class="bd-note">see ${esc(b.references.join("; "))}</div>`);
        }
        if (b.members && b.members.length) {
          const inner = b.members
            .map((m) => {
              const mn = nodes[m] || {};
              return `<div class="bd-member"><b>${esc(m)}</b>${mn.label_svg ? `<div class="bd-math">${mn.label_svg}</div>` : ""}</div>`;
            })
            .join("");
          rows.push(`<details class="bd-members"><summary>inside (${b.members.length})</summary>${inner}</details>`);
        }
        if (b.contracted) rows.push(`<button class="bd-expand-btn" data-block="${esc(bid)}">expand</button>`);
        else if (b.expanded) rows.push(`<button class="bd-collapse-btn" data-block="${esc(bid)}">collapse</button>`);
        return rows;
      }
      function setExpanded(bid, on) {
        const cur = new Set(model.get("expanded_blocks") || []);
        if (on) cur.add(bid);
        else cur.delete(bid);
        model.set("expanded_blocks", Array.from(cur).sort());
        model.save_changes();
      }
      function wireCard(id) {
        card.querySelectorAll(".bd-expand-btn").forEach((btn) =>
          btn.addEventListener("click", (ev) => {
            ev.stopPropagation();
            setExpanded(btn.dataset.block, true);
          })
        );
        card.querySelectorAll(".bd-collapse-btn").forEach((btn) =>
          btn.addEventListener("click", (ev) => {
            ev.stopPropagation();
            setExpanded(btn.dataset.block, false);
          })
        );
        card.querySelectorAll(".bd-page-btn").forEach((btn) =>
          btn.addEventListener("click", (ev) => {
            ev.stopPropagation();
            // ask Python for the page; the answer arrives on panel_response
            model.set("panel_request", {
              kind: "marginals",
              node_id: btn.dataset.node,
              page: Number(btn.dataset.page),
              nonce: Date.now(),
            });
            model.save_changes();
          })
        );
      }
      function showCard(id) {
        const n = nodes[id];
        if (!n) return;
        const isBlock = n.role === "block";
        const b = isBlock ? (spec.blocks || {})[n.block] || {} : null;
        const title = isBlock ? b.title || id : id;
        const rows = [`<div class="bd-card-title">${esc(title)} <span>${esc(isBlock ? "structure" : n.role)}</span></div>`];
        if (n.tag && !isBlock) rows.push(`<div class="bd-tag-line">${esc(n.tag)}</div>`);
        if (n.label_svg) rows.push(`<div class="bd-math">${n.label_svg}</div>`);
        if (n.dist) rows.push(`<div>distribution: <b>${esc(n.dist)}</b></div>`);
        if (n.dims && n.dims.length) rows.push(`<div>dims: ${esc(n.dims.join(" × "))}</div>`);
        if (n.transform) rows.push(`<div>transform: ${esc(n.transform)}</div>`);
        if (n.diag && n.diag.length) {
          rows.push(
            `<div class="bd-diag-rows">${n.diag.map((d) => `<div>${esc(d)}</div>`).join("")}</div>`
          );
        }
        const fam = familyText(n);
        if (fam) rows.push(`<div class="bd-ctor">${esc(fam)}</div>`);
        if (n.glyph_note) rows.push(`<div class="bd-note">${esc(n.glyph_note)}</div>`);
        // observed nodes: the Python-rendered histogram + best-fit-family overlay (data vs theory)
        if (n.panel) rows.push(`<div class="bd-overlay">${n.panel}</div>`);
        // labelled per-element posterior intervals (coordinate order), rendered by Python
        if (n.marginals && n.marginals.svg) {
          const m = n.marginals;
          let pager = "";
          if (m.pages > 1) {
            const of = esc(n.marginals_of || id);
            pager =
              `<div class="bd-pager">` +
              (m.page > 0 ? `<button class="bd-page-btn" data-node="${of}" data-page="${m.page - 1}">‹ prev</button>` : "") +
              `<span>page ${m.page + 1} / ${m.pages}</span>` +
              (m.page + 1 < m.pages ? `<button class="bd-page-btn" data-node="${of}" data-page="${m.page + 1}">next ›</button>` : "") +
              `</div>`;
          }
          rows.push(`<div class="bd-overlay bd-marginals">${m.svg}${pager}</div>`);
        }
        if (isBlock) rows.push(...blockRows(n.block));
        else if (n.in_block) rows.push(...blockRows(n.in_block));
        // posterior-geometry panels offered by this node (the funnel joint)
        const auxes = (spec.aux || {})[id] || [];
        auxes.forEach((a, i) => {
          rows.push(`<button class="bd-aux-btn" data-aux="${i}">${esc(a.label)}</button>`);
        });
        rows.push('<div class="bd-card-hint">click empty space to close</div>');
        card.innerHTML = rows.join("");
        card.querySelectorAll(".bd-aux-btn").forEach((btn) => {
          btn.addEventListener("click", (ev) => {
            ev.stopPropagation();
            const a = auxes[Number(btn.dataset.aux)];
            if (!a) return;
            panel.innerHTML =
              `<div class="bd-panel-head">${esc(a.label)}<span>click empty space to close</span></div>` +
              a.panel;
            panel.style.display = "block";
          });
        });
        wireCard(id);
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
          pin(id);
        });
        // keyboard: the body group is the node's one tab stop (the chrome group is paint only)
        if (nodeEl.classList.contains("bd-body")) {
          nodeEl.setAttribute("tabindex", "0");
          nodeEl.setAttribute("role", "button");
          nodeEl.setAttribute("aria-label", `${id} (${(nodes[id] || {}).role || "node"})`);
          const mates = nodeEls.filter((x) => x.dataset.node === id);
          nodeEl.addEventListener("focus", () => mates.forEach((x) => x.classList.add("bd-kbd")));
          nodeEl.addEventListener("blur", () => mates.forEach((x) => x.classList.remove("bd-kbd")));
          nodeEl.addEventListener("keydown", (ev) => {
            if (ev.key === "Enter" || ev.key === " ") {
              ev.preventDefault();
              pin(id);
            }
          });
        }
      });

      function pin(id) {
        pinned = id;
        // a compound node is a STRUCTURE, not a variable: it goes to selected_block, and
        // selected_node (which neighbouring cells feed to ArviZ) is cleared
        const isBlock = (nodes[id] || {}).role === "block";
        model.set("selected_node", isBlock ? "" : id);
        model.set("selected_block", isBlock ? (nodes[id] || {}).block || "" : "");
        model.save_changes();
        trace(id);
        tooltip.style.display = "none";
        panel.style.display = "none";
        showCard(id);
      }

      function openPlate(pid) {
        const p = (spec.plates || {})[pid];
        card.style.display = "none";
        panel.innerHTML =
          `<div class="bd-panel-head">${esc(pid)}<span>click empty space to close</span></div>` +
          (p && p.panel
            ? p.panel
            : p && p.reason
              ? `<div class="bd-placeholder">${esc(p.reason)}</div>`
              : '<div class="bd-placeholder">simulating the prior predictive&hellip;</div>');
        panel.style.display = "block";
      }

      Array.from(svg.querySelectorAll(".bd-plate")).forEach((plEl) => {
        const pid = plEl.dataset.plate;
        const expandable = (spec.expandable || []).includes(pid) || (spec.plates || {})[pid];
        // only advertise the affordance when there is something behind it
        if (expandable) plEl.style.cursor = "zoom-in";
        plEl.addEventListener("click", (ev) => {
          // bail BEFORE stopPropagation, so a non-expandable plate never swallows the
          // background click that dismisses the pinned card
          if (!expandable) return;
          ev.stopPropagation();
          openPlate(pid);
          // ask Python for the panel; it forward-simulates on first request and pushes `spec`
          if (!(spec.plates || {})[pid]) {
            model.set("expanded_plate", pid);
            model.save_changes();
          }
        });
      });

      function unpin() {
        pinned = null;
        model.set("selected_node", "");
        model.set("selected_block", "");
        model.save_changes();
        clearHl();
        card.style.display = "none";
        panel.style.display = "none";
        if (model.get("expanded_plate")) {
          model.set("expanded_plate", "");
          model.save_changes();
        }
      }
      svg.addEventListener("click", unpin);
      el.onkeydown = (ev) => {
        if (ev.key === "Escape") unpin();
      };

      // A spec push (view.update(...)) redraws from scratch, and `pinned` lives in this closure.
      // Without this the user's selection silently drops on every update — which is exactly when
      // they are watching one node. Restore it from the synced trait; classes/innerHTML only.
      let wasPinned = model.get("selected_node");
      const selBlock = model.get("selected_block");
      if (!wasPinned && selBlock) {
        // the block may be contracted (its compound node) or expanded (its anchor)
        const b = (spec.blocks || {})[selBlock] || {};
        wasPinned = b.display_id || b.anchor || "";
      }
      if (wasPinned && nodes[wasPinned] && !nodes[wasPinned].hidden) {
        pinned = wasPinned;
        trace(wasPinned);
        showCard(wasPinned);
      }
      // a push carrying the plate panel the user is waiting on replaces the placeholder in place
      const openPid = model.get("expanded_plate");
      if (openPid && (spec.plates || {})[openPid] && panel.style.display === "block") {
        openPlate(openPid);
      }
    }

    function onPanel() {
      // a page of a Python-rendered panel arrived: swap it into the open card
      const resp = model.get("panel_response") || {};
      const slot = card.querySelector(".bd-marginals");
      if (!slot || !resp.request || resp.request.kind !== "marginals") return;
      const node = resp.request.node_id;
      const page = Number(resp.request.page || 0);
      const spec = model.get("spec") || {};
      const holder = Object.values(spec.nodes || {}).find((x) => x.marginals_of === node) || {};
      const pages = (holder.marginals || {}).pages || 1;
      const pager =
        `<div class="bd-pager">` +
        (page > 0 ? `<button class="bd-page-btn" data-node="${esc(node)}" data-page="${page - 1}">‹ prev</button>` : "") +
        `<span>page ${page + 1} / ${pages}</span>` +
        (page + 1 < pages ? `<button class="bd-page-btn" data-node="${esc(node)}" data-page="${page + 1}">next ›</button>` : "") +
        `</div>`;
      slot.innerHTML = resp.status === "ready" ? resp.svg + pager : `<div class="bd-placeholder">${esc(resp.reason || "unavailable")}</div>`;
      slot.querySelectorAll(".bd-page-btn").forEach((btn) =>
        btn.addEventListener("click", (ev) => {
          ev.stopPropagation();
          model.set("panel_request", { kind: "marginals", node_id: btn.dataset.node, page: Number(btn.dataset.page), nonce: Date.now() });
          model.save_changes();
        })
      );
    }

    draw();
    model.on("change:spec", draw);
    model.on("change:panel_response", onPanel);
    // anywidget calls this when the view goes away (cell re-run, notebook close). Without it the
    // model keeps a reference to `draw` — and to this `el` — for every view ever rendered, so a
    // re-executed cell redraws through all its predecessors' dead DOM.
    return () => {
      model.off("change:spec", draw);
      model.off("change:panel_response", onPanel);
    };
  },
};
