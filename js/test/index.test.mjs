// Controller tests for js/index.js (node:test + jsdom; run with `npm test`).
//
// The widget computes no geometry and no statistics, so what is left to test is the contract:
// which traits a gesture writes, which Python-rendered fragment it shows, and that it cleans up.
import { test } from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";

function setup(specOverrides = {}, traits = {}) {
  const dom = new JSDOM("<!doctype html><html><body><div id=root></div></body></html>");
  globalThis.document = dom.window.document;
  globalThis.window = dom.window;
  const svg =
    '<svg xmlns="http://www.w3.org/2000/svg">' +
    '<g class="bd-node" data-node="mu"><rect class="bd-chrome"/></g>' +
    '<g class="bd-node bd-body" data-node="mu"><text>mu</text></g>' +
    '<g class="bd-node" data-node="block:hierarchy:theta"><rect class="bd-chrome"/></g>' +
    '<g class="bd-node bd-body" data-node="block:hierarchy:theta"><text>theta</text></g>' +
    '<path class="bd-edge" data-src="mu" data-tgt="block:hierarchy:theta"/>' +
    "</svg>";
  const spec = {
    svg,
    detail: "overview",
    has_overview: true,
    nodes: {
      mu: { role: "latent", dist: "Normal", params: [{ name: "loc", value: "0" }], dims: [], blanket: ["block:hierarchy:theta"], ancestors: [], descendants: ["block:hierarchy:theta"], diag: [] },
      "block:hierarchy:theta": {
        role: "block",
        block: "block:hierarchy:theta",
        params: [],
        dims: ["school"],
        blanket: ["mu"],
        ancestors: ["mu"],
        descendants: [],
        diag: [],
        marginals: { svg: "<svg id=m0></svg>", page: 0, pages: 2, n: 30 },
        marginals_of: "theta",
      },
      theta: { role: "deterministic", hidden: true, params: [], dims: ["school"], diag: [] },
    },
    blocks: {
      "block:hierarchy:theta": {
        title: "partial pooling · non-centered",
        provenance: "verified",
        workflow: null,
        anchor: "theta",
        members: ["eta", "theta"],
        roles: { loc: ["mu"] },
        notes: ["a note"],
        references: [],
        contracted: true,
        display_id: "block:hierarchy:theta",
        expanded: false,
        panels: [{ kind: "shrinkage", label: "shrinkage toward the population" }],
      },
    },
    plates: {},
    aux: {},
    expandable: [],
    ...specOverrides,
  };
  const state = { spec, selected_node: "", selected_block: "", expanded_plate: "", detail: "overview", expanded_blocks: [], panel_request: {}, panel_response: {}, ...traits };
  const handlers = {};
  const model = {
    get: (k) => state[k],
    set: (k, v) => {
      state[k] = v;
    },
    save_changes: () => {},
    on: (ev, fn) => {
      (handlers[ev] ||= []).push(fn);
    },
    off: (ev, fn) => {
      handlers[ev] = (handlers[ev] || []).filter((f) => f !== fn);
    },
    fire: (ev) => (handlers[ev] || []).forEach((f) => f()),
    handlers,
  };
  const el = dom.window.document.getElementById("root");
  return { dom, el, model, state };
}

async function mount(...args) {
  const ctx = setup(...args);
  const mod = await import("../index.js");
  ctx.cleanup = mod.default.render({ model: ctx.model, el: ctx.el });
  return ctx;
}

function body(el, id) {
  return el.querySelector(`.bd-body[data-node="${id}"]`);
}

test("the fit toggle only flips a class", async () => {
  const { el } = await mount();
  const btn = el.querySelector(".bd-toolbar button");
  assert.equal(btn.textContent, "fit ↔ 1:1");
  btn.dispatchEvent(new el.ownerDocument.defaultView.MouseEvent("click", { bubbles: true }));
  assert.ok(el.classList.contains("bd-fit"));
});

test("keyboard: one tab stop per node, Enter pins, Escape clears", async () => {
  const { el, state } = await mount();
  const b = body(el, "mu");
  assert.equal(b.getAttribute("tabindex"), "0");
  assert.equal(el.querySelectorAll('[data-node="mu"][tabindex]').length, 1);
  const KE = el.ownerDocument.defaultView.KeyboardEvent;
  b.dispatchEvent(new KE("keydown", { key: "Enter", bubbles: true }));
  assert.equal(state.selected_node, "mu");
  assert.equal(el.querySelector(".bd-card").style.display, "block");
  assert.match(el.querySelector(".bd-card").innerHTML, /Normal\(loc = 0\)/); // no pm. code
  el.dispatchEvent(new KE("keydown", { key: "Escape", bubbles: true }));
  assert.equal(state.selected_node, "");
});

test("a block pins as a structure, never as a variable", async () => {
  const { el, state } = await mount();
  const ME = el.ownerDocument.defaultView.MouseEvent;
  body(el, "block:hierarchy:theta").dispatchEvent(new ME("click", { bubbles: true }));
  assert.equal(state.selected_node, "");
  assert.equal(state.selected_block, "block:hierarchy:theta");
  const card = el.querySelector(".bd-card").innerHTML;
  assert.match(card, /partial pooling · non-centered/);
  assert.match(card, /read off the graph/);
  el.querySelector(".bd-expand-btn").dispatchEvent(new ME("click", { bubbles: true }));
  assert.deepEqual(state.expanded_blocks, ["block:hierarchy:theta"]);
});

test("the detail toggle writes the trait and nothing else", async () => {
  const { el, state } = await mount();
  const ME = el.ownerDocument.defaultView.MouseEvent;
  const btn = Array.from(el.querySelectorAll(".bd-toolbar button")).find((x) => /all variables/.test(x.textContent));
  btn.dispatchEvent(new ME("click", { bubbles: true }));
  assert.equal(state.detail, "full");
});

test("paging asks Python and swaps in its answer", async () => {
  const { el, state, model } = await mount();
  const ME = el.ownerDocument.defaultView.MouseEvent;
  body(el, "block:hierarchy:theta").dispatchEvent(new ME("click", { bubbles: true }));
  el.querySelector(".bd-page-btn").dispatchEvent(new ME("click", { bubbles: true }));
  assert.equal(state.panel_request.kind, "marginals");
  assert.equal(state.panel_request.node_id, "theta");
  assert.equal(state.panel_request.page, 1);
  state.panel_response = { request: state.panel_request, status: "ready", svg: "<svg id=m1></svg>" };
  model.fire("change:panel_response");
  assert.ok(el.querySelector(".bd-marginals #m1"));
});

test("cleanup unsubscribes every listener", async () => {
  const { model, cleanup } = await mount();
  assert.ok(model.handlers["change:spec"].length && model.handlers["change:panel_response"].length);
  cleanup();
  assert.equal(model.handlers["change:spec"].length, 0);
  assert.equal(model.handlers["change:panel_response"].length, 0);
});

test("a fitted-structure panel is requested from Python and shown with its reason", async () => {
  const { el, state, model } = await mount();
  const ME = el.ownerDocument.defaultView.MouseEvent;
  body(el, "block:hierarchy:theta").dispatchEvent(new ME("click", { bubbles: true }));
  el.querySelector(".bd-fit-panel-btn").dispatchEvent(new ME("click", { bubbles: true }));
  assert.equal(state.panel_request.kind, "shrinkage");
  assert.equal(state.panel_request.block_id, "block:hierarchy:theta");
  assert.match(el.querySelector(".bd-panel").innerHTML, /computing/);
  state.panel_response = { request: state.panel_request, status: "unavailable", svg: "", reason: "needs an identity link" };
  model.fire("change:panel_response");
  assert.match(el.querySelector(".bd-panel").innerHTML, /needs an identity link/);
});
