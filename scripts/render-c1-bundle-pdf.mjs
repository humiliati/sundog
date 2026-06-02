#!/usr/bin/env node
// scripts/render-c1-bundle-pdf.mjs
//
// Render internal/outreach/PDE_C1_REVIEW_BUNDLE.md to a print-quality PDF for
// emailing to a reviewer. Path: markdown-it -> styled print HTML -> headless
// Chrome --print-to-pdf. Chrome's font stack handles the Unicode-heavy content
// (Phi_K, epsilon_K, delta_H, R^2, mu, ->, <=, etc.) with no glyph boxes.
//
// Output: internal/outreach/PDE_C1_REVIEW_BUNDLE.pdf
//
// This is a render-only convenience; it does not modify the bundle. Re-run after
// `node scripts/build-c1-review-bundle.mjs` if the bundle changes.

import { readFileSync, writeFileSync, existsSync, mkdtempSync, rmSync, statSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const MarkdownIt = require("markdown-it");

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SRC = path.resolve(REPO, "internal/outreach/PDE_C1_REVIEW_BUNDLE.md");
const OUT = path.resolve(REPO, "internal/outreach/PDE_C1_REVIEW_BUNDLE.pdf");

const CHROME_CANDIDATES = [
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
];

function findChrome() {
  for (const c of CHROME_CANDIDATES) if (existsSync(c)) return c;
  throw new Error("No Chrome/Edge found for headless PDF rendering");
}

const CSS = `
  @page { size: Letter; margin: 0.9in 0.85in; }
  html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  body {
    font-family: "Georgia", "Cambria", "Times New Roman", serif;
    font-size: 10.5pt; line-height: 1.5; color: #1a1a1a; max-width: 100%;
  }
  h1 { font-size: 19pt; line-height: 1.15; margin: 0 0 0.3em; color: #1A3A52; }
  h2 { font-size: 14pt; margin: 1.1em 0 0.35em; color: #1A3A52; border-bottom: 1px solid #ccd; padding-bottom: 0.15em; }
  h3 { font-size: 11.5pt; margin: 0.9em 0 0.3em; color: #244; }
  h4 { font-size: 10.5pt; margin: 0.8em 0 0.25em; color: #355; }
  p { margin: 0.45em 0; }
  code, kbd {
    font-family: "Consolas", "DejaVu Sans Mono", "Courier New", monospace;
    font-size: 9pt; background: #f3f4f6; padding: 0.05em 0.3em; border-radius: 3px;
  }
  pre {
    background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 5px;
    padding: 0.7em 0.9em; overflow-x: auto; font-size: 8.6pt; line-height: 1.4;
    white-space: pre-wrap; word-break: break-word;
  }
  pre code { background: none; padding: 0; font-size: inherit; }
  blockquote {
    margin: 0.6em 0; padding: 0.3em 0 0.3em 1em; border-left: 3px solid #B8831E;
    background: rgba(255,244,214,0.4); color: #40403a;
  }
  table { border-collapse: collapse; margin: 0.7em 0; font-size: 9pt; width: 100%; }
  th, td { border: 1px solid #ccd; padding: 0.35em 0.55em; text-align: left; vertical-align: top; }
  th { background: #eef2f5; font-weight: 700; }
  hr { border: none; border-top: 2px solid #1A3A52; margin: 1.4em 0 0.8em; }
  a { color: #1A3A52; text-decoration: none; }
  strong { color: #111; }
  /* keep section dividers from orphaning their heading */
  h2, h3 { break-after: avoid; }
  table, pre, blockquote { break-inside: avoid; }

  /* closing page. NOTE: break-before:page directly on this flex/centered
     section makes Chrome's headless print path DROP the flowed prose after the
     graphic (verified 2026-06-02). Use a zero-height spacer with break-before
     instead, and keep the section itself together. */
  .c1-pagebreak { break-before: page; height: 0; }
  .c1-closing { text-align: center; padding-top: 0.4in; break-inside: avoid; }
  .c1-closing svg { width: 320px; height: auto; display: block; margin: 0 auto 0.5em; }
  .closing-cap {
    font-family: "Consolas","DejaVu Sans Mono",monospace; font-size: 8pt;
    letter-spacing: 0.08em; text-transform: uppercase; color: #6A7680; margin: 0 0 1.4em;
  }
  .c1-closing h3 { font-size: 15pt; color: #1A3A52; border: none; margin: 0.2em 0 0.6em; text-align: center; }
  .closing-body { text-align: left; max-width: 5.4in; margin: 0 auto; }
  .closing-body blockquote {
    border-left: 3px solid #B8831E; background: rgba(255,244,214,0.45);
    font-size: 10pt; line-height: 1.6;
  }
  /* the signature line is the last <p><em>…</em></p>; center + italicize it */
  .closing-body > p:last-child { text-align: center; margin-top: 1.5em; color: #40505C; }
`;

// On-brand halo graphic for the closing page. The geometry encodes the result:
// the bright central sun is the full hidden state; the highlighted right-hand
// parhelion is Phi_K — a partial-but-readable projection on the 22-degree ring.
const HALO_SVG = `
<svg viewBox="0 0 400 230" role="img" aria-label="A sundog halo: central sun with the 22-degree ring and two parhelia; the right parhelion is highlighted as the readable low-band projection.">
  <defs>
    <radialGradient id="sun" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#fff4b2"/><stop offset="45%" stop-color="#f4c430"/><stop offset="100%" stop-color="#b97812"/>
    </radialGradient>
    <radialGradient id="dog" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#fff8d8"/><stop offset="55%" stop-color="#f6cf52"/><stop offset="100%" stop-color="#caa233" stop-opacity="0.15"/>
    </radialGradient>
  </defs>
  <!-- 22-degree halo ring -->
  <circle cx="200" cy="115" r="92" fill="none" stroke="#1A3A52" stroke-width="1.4" stroke-opacity="0.5" stroke-dasharray="2 4"/>
  <!-- parhelic circle (horizontal) -->
  <line x1="40" y1="115" x2="360" y2="115" stroke="#1A3A52" stroke-width="0.8" stroke-opacity="0.28"/>
  <!-- central sun = full hidden state -->
  <circle cx="200" cy="115" r="17" fill="url(#sun)"/>
  <!-- left parhelion (faint) -->
  <circle cx="108" cy="115" r="8" fill="url(#dog)" opacity="0.55"/>
  <!-- right parhelion = Phi_K, highlighted -->
  <circle cx="292" cy="115" r="11" fill="url(#dog)"/>
  <circle cx="292" cy="115" r="11" fill="none" stroke="#B8831E" stroke-width="1.2"/>
  <line x1="292" y1="104" x2="292" y2="70" stroke="#B8831E" stroke-width="0.8" stroke-opacity="0.6"/>
  <text x="292" y="62" text-anchor="middle" font-family="Consolas,monospace" font-size="11" fill="#684811" font-weight="700">&#934;_K</text>
  <text x="200" y="150" text-anchor="middle" font-family="Consolas,monospace" font-size="9" fill="#40505C">full state</text>
</svg>`;

const CLOSING_HTML = `${HALO_SVG}
  <p class="closing-cap">read the shadow &middot; test the boundary</p>`;

function main() {
  if (!existsSync(SRC)) {
    throw new Error(`Bundle not found: ${SRC}. Run scripts/build-c1-review-bundle.mjs first.`);
  }
  const mdText = readFileSync(SRC, "utf8");
  const md = new MarkdownIt({ html: false, linkify: true, typographer: false });
  let bodyHtml = md.render(mdText);

  // Upgrade the closing page: swap the escaped sentinel paragraph for the
  // on-brand halo graphic, and wrap the whole closing section (graphic + the
  // thank-you prose that follows it, to end of document) in one styled section
  // so the page-break and centered layout apply. markdown-it (html:false)
  // renders the sentinel as an escaped <p>, so we target that exact string.
  // The closing is the LAST content in the bundle, so opening the wrapper at the
  // sentinel and closing it at end-of-body yields balanced markup.
  const SENTINEL = "<p>&lt;!-- PDF_CLOSING_GRAPHIC --&gt;</p>";
  if (!bodyHtml.includes(SENTINEL)) {
    throw new Error("closing sentinel not found in rendered HTML; rebuild the bundle (build-c1-review-bundle.mjs) before rendering");
  }
  bodyHtml = bodyHtml.replace(SENTINEL, `<div class="c1-pagebreak"></div><section class="c1-closing">${CLOSING_HTML}<div class="closing-body">`);
  bodyHtml = bodyHtml + "</div></section>";

  const html = `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>PDE C1 Finite-Galerkin Separation — Review Bundle</title>
<style>${CSS}</style></head>
<body>${bodyHtml}</body></html>`;

  const tmp = mkdtempSync(path.join(tmpdir(), "c1pdf-"));
  const htmlPath = path.join(tmp, "bundle.html");
  writeFileSync(htmlPath, html, "utf8");

  const chrome = findChrome();
  // headless print-to-pdf. file:// URL so relative nothing is needed.
  const fileUrl = "file:///" + htmlPath.replaceAll("\\", "/");
  try {
    execFileSync(chrome, [
      "--headless",
      "--disable-gpu",
      "--no-pdf-header-footer",
      `--print-to-pdf=${OUT}`,
      fileUrl,
    ], { stdio: ["ignore", "pipe", "pipe"], timeout: 120000 });
  } catch (err) {
    // Older Chrome rejects --no-pdf-header-footer; retry without it.
    execFileSync(chrome, [
      "--headless",
      "--disable-gpu",
      `--print-to-pdf=${OUT}`,
      fileUrl,
    ], { stdio: ["ignore", "pipe", "pipe"], timeout: 120000 });
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }

  if (!existsSync(OUT)) throw new Error("Chrome did not produce the PDF");
  const kb = Math.round(statSync(OUT).size / 1024);
  console.log(`PDF written: ${path.relative(REPO, OUT).replaceAll("\\", "/")} (${kb} KB)`);
  console.log(`renderer: ${path.basename(chrome)}`);
}

main();
