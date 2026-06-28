// Public Ghost reader (ghost.html). Renders a rhombic Penrose tiling and a
// movable circle that shows which supertiles are recoverable from inside it.
// Imports the same verified generator as the internal reader (single source of
// truth); no new math, no reused figures.

import { makePenrose, supertileKey, classifySupertiles, analyzeWindow, GOLDEN, clampInt } from "../ghost/aperiodic-core.js";

const state = { depth: 5, level: 2, cx: 0, cy: 0, r: 0.55 };
let model = makePenrose(state.depth);

const $ = (id) => document.getElementById(id);
const svg = $("ghost-tiling");

function hashHue(key) {
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) >>> 0;
  return h % 360;
}

const STROKE = { contained: "#2E7D5A", crossing: "#B53A2C", outside: "#c4ccc7" };

function renderTiling() {
  const center = { x: state.cx, y: state.cy };
  const status = classifySupertiles(model, center, state.r, state.level);
  const parts = [];
  for (const t of model.triangles) {
    const key = supertileKey(t, state.level);
    const st = status.get(key) || "outside";
    const hue = hashHue(key);
    const opacity = st === "outside" ? 0.16 : 0.9;
    const sw = st === "crossing" ? 0.006 : 0.0025;
    const pts = `${t.A.x},${-t.A.y} ${t.B.x},${-t.B.y} ${t.C.x},${-t.C.y}`;
    parts.push(`<polygon points="${pts}" fill="hsl(${hue} 55% 62%)" fill-opacity="${opacity}" stroke="${STROKE[st]}" stroke-width="${sw}" stroke-linejoin="round"/>`);
  }
  parts.push(`<circle cx="${state.cx}" cy="${-state.cy}" r="${state.r}" fill="none" stroke="#1A3A52" stroke-width="0.008" stroke-dasharray="0.022 0.016"/>`);
  svg.innerHTML = parts.join("");
}

function renderControls() {
  state.level = clampInt(state.level, 1, model.depth);
  const lvl = $("ghost-level");
  lvl.max = String(model.depth);
  lvl.value = String(state.level);
  $("ghost-depth").value = String(state.depth);
  $("ghost-radius").value = String(state.r);
  $("ghost-depth-val").textContent = String(state.depth);
  $("ghost-level-val").textContent = String(state.level);
  $("ghost-radius-val").textContent = state.r.toFixed(2);
}

function renderReadout() {
  const a = analyzeWindow(model, { x: state.cx, y: state.cy }, state.r, state.level);
  const ratio = Number.isFinite(a.thickThinRatio) ? a.thickThinRatio.toFixed(3) : "n/a";
  $("ghost-readout").innerHTML =
    `tiles <b>${a.tileCounts.total}</b> (thick:thin <b>${ratio}</b>, &phi; = ${GOLDEN.toFixed(3)}) &middot; ` +
    `supertiles inside the circle <b>${a.supertilesContained}</b> recoverable, <b>${a.supertilesCrossing}</b> still crossing the edge`;
}

function render() {
  renderControls();
  renderTiling();
  renderReadout();
}

function rebuild() {
  model = makePenrose(state.depth);
  render();
}

// pointer -> tiling coordinates (viewBox -1.15..1.15, y drawn negated)
function moveCircleFromEvent(e) {
  const rect = svg.getBoundingClientRect();
  const fx = (e.clientX - rect.left) / rect.width;
  const fy = (e.clientY - rect.top) / rect.height;
  const drawnX = -1.15 + fx * 2.3;
  const drawnY = -1.15 + fy * 2.3;
  state.cx = Math.max(-0.95, Math.min(0.95, drawnX));
  state.cy = Math.max(-0.95, Math.min(0.95, -drawnY));
  render();
}

let dragging = false;
svg.addEventListener("pointerdown", (e) => {
  dragging = true;
  svg.setPointerCapture(e.pointerId);
  moveCircleFromEvent(e);
});
svg.addEventListener("pointermove", (e) => {
  if (dragging) moveCircleFromEvent(e);
});
svg.addEventListener("pointerup", () => {
  dragging = false;
});

$("ghost-depth").addEventListener("input", (e) => {
  state.depth = Number(e.target.value);
  rebuild();
});
$("ghost-level").addEventListener("input", (e) => {
  state.level = Number(e.target.value);
  render();
});
$("ghost-radius").addEventListener("input", (e) => {
  state.r = Number(e.target.value);
  render();
});

render();
