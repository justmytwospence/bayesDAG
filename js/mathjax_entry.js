// Self-contained MathJax TeX->SVG bundle for running inside a bare JS engine
// (py_mini_racer's V8 — no DOM, no Node). Uses the liteAdaptor (DOM-free), the same
// approach mathjax-node / the tex2svg CLI use server-side.
//
// After this bundle is eval'd, `globalThis.tex2svg(tex, display)` returns the SVG markup
// for the equation, with `data-mml-node` groups intact (the anchors for port-edges).

import { mathjax } from "mathjax-full/js/mathjax.js";
import { TeX } from "mathjax-full/js/input/tex.js";
import { SVG } from "mathjax-full/js/output/svg.js";
import { liteAdaptor } from "mathjax-full/js/adaptors/liteAdaptor.js";
import { RegisterHTMLHandler } from "mathjax-full/js/handlers/html.js";
import { AllPackages } from "mathjax-full/js/input/tex/AllPackages.js";

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);

const texInput = new TeX({ packages: AllPackages });
// fontCache 'local' => glyphs are inline <path> data, so each SVG is self-contained
// (critical for embedding the identical SVG in both the static file and the widget).
const svgOutput = new SVG({ fontCache: "local" });
const doc = mathjax.document("", { InputJax: texInput, OutputJax: svgOutput });

globalThis.tex2svg = function (tex, display) {
  const node = doc.convert(String(tex), { display: !!display });
  // node is an <mjx-container>; return its inner <svg> markup.
  const svg = adaptor.tags(node, "svg")[0];
  return svg ? adaptor.outerHTML(svg) : adaptor.outerHTML(node);
};

// Smoke marker so we can confirm the bundle initialized.
globalThis.__bayesdag_mathjax_ready = true;
