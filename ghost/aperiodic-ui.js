// Ghost Phase 3 aperiodic reader UI.
//
// Renders a rhombic Penrose tiling from the pure core and overlays supertile
// ancestry + a window/recognizability probe. Reader surface only: no public
// route, no theorem-shaped metric, no reused assets.

import * as G from "./aperiodic-core.js";

const state = {
  depth: 5,
  level: 2,
  cx: 0,
  cy: 0,
  r: 0.6,
};

let model = G.makePenrose(state.depth);

const $ = (id) => document.getElementById(id);

function hashHue(key) {
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) >>> 0;
  return h % 360;
}

const STATUS_STROKE = { contained: "#2f6f52", crossing: "#9a4a3f", outside: "#c4ccc7" };

function fmt(n) {
  return Number.isFinite(n) ? n.toFixed(3) : "n/a";
}

function renderControls() {
  state.level = G.clampInt(state.level, 1, model.depth);
  const lr = $("level-range");
  lr.max = String(model.depth);
  lr.value = String(state.level);
  $("depth-range").value = String(state.depth);
  $("cx-range").value = String(state.cx);
  $("cy-range").value = String(state.cy);
  $("r-range").value = String(state.r);
  $("depth-readout").textContent = String(state.depth);
  $("level-readout").textContent = String(state.level);
  $("cx-readout").textContent = state.cx.toFixed(2);
  $("cy-readout").textContent = state.cy.toFixed(2);
  $("r-readout").textContent = state.r.toFixed(2);
}

function renderTiling() {
  const center = { x: state.cx, y: state.cy };
  const status = G.classifySupertiles(model, center, state.r, state.level);
  const parts = [];
  for (const t of model.triangles) {
    const key = G.supertileKey(t, state.level);
    const st = status.get(key) || "outside";
    const hue = hashHue(key);
    const opacity = st === "outside" ? 0.16 : 0.92;
    const stroke = STATUS_STROKE[st];
    const sw = st === "crossing" ? 0.006 : 0.0025;
    // SVG y-down: negate y so the figure reads math-oriented.
    const pts = `${t.A.x},${-t.A.y} ${t.B.x},${-t.B.y} ${t.C.x},${-t.C.y}`;
    parts.push(
      `<polygon points="${pts}" fill="hsl(${hue} 55% 62%)" fill-opacity="${opacity}" stroke="${stroke}" stroke-width="${sw}" stroke-linejoin="round"/>`,
    );
  }
  // window circle (y-flipped center)
  parts.push(
    `<circle cx="${state.cx}" cy="${-state.cy}" r="${state.r}" fill="none" stroke="#122633" stroke-width="0.007" stroke-dasharray="0.02 0.015"/>`,
  );
  $("tiling").innerHTML = parts.join("");
}

function renderPanels() {
  const center = { x: state.cx, y: state.cy };
  const a = G.analyzeWindow(model, center, state.r, state.level);
  $("regime").textContent = "bounded recognition";
  $("metric-counts").textContent = `${a.tileCounts.thin} / ${a.tileCounts.thick}`;
  $("metric-ratio").textContent = `${fmt(a.thickThinRatio)} (phi = ${fmt(G.GOLDEN)})`;
  $("metric-supertiles").textContent = `${a.supertilesTouching} touching @ L${a.ancestryLevel}`;
  $("metric-inwindow").textContent = String(a.tilesInWindow);
  $("metric-touching").textContent = String(a.supertilesTouching);
  $("metric-contained").textContent = String(a.supertilesContained);
  $("metric-crossing").textContent = String(a.supertilesCrossing);

  const v = $("verdict");
  if (a.supertilesCrossing === 0 && a.supertilesTouching > 0) {
    v.textContent = "all touched supertiles are inside: ancestry recoverable here";
    v.className = "verdict closed";
  } else {
    v.textContent = "supertiles cross the edge: ancestry is ghost at the boundary";
    v.className = "verdict open";
  }
  $("recognition-note").textContent = a.note;
  $("cliff-note").textContent = a.cliff.note;
  $("export-out").textContent = JSON.stringify(
    G.exportReaderAnalysis(model, center, state.r, state.level),
    null,
    2,
  );
}

function render() {
  renderControls();
  renderTiling();
  renderPanels();
}

function rebuild() {
  model = G.makePenrose(state.depth);
  state.level = G.clampInt(state.level, 1, model.depth);
  render();
}

$("depth-range").addEventListener("input", (e) => {
  state.depth = Number(e.target.value);
  rebuild();
});
$("level-range").addEventListener("input", (e) => {
  state.level = Number(e.target.value);
  render();
});
$("cx-range").addEventListener("input", (e) => {
  state.cx = Number(e.target.value);
  render();
});
$("cy-range").addEventListener("input", (e) => {
  state.cy = Number(e.target.value);
  render();
});
$("r-range").addEventListener("input", (e) => {
  state.r = Number(e.target.value);
  render();
});

render();
