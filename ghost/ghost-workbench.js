// Ghost Phase 2 internal workbench UI.
//
// Renders a periodic control and a Fibonacci substitution stripe from the pure
// core. This is a reader surface only: no Hat/Spectre assets, no public route,
// no theorem-shaped metric.

import * as G from "./ghost-core.js";

const state = {
  systemId: G.DEFAULT_SYSTEM_ID,
  center: 64,
  radius: 12,
  ancestryLevel: 3,
};

const $ = (id) => document.getElementById(id);

function currentSystem() {
  return G.makeSystem(state.systemId);
}

function countsText(counts) {
  return Object.keys(counts)
    .sort()
    .map((key) => `${key}:${counts[key]}`)
    .join("  ");
}

function candidatesText(candidates) {
  return candidates.length ? candidates.join(", ") : "none under cap";
}

function clampStateToSystem(system) {
  state.radius = G.clampInt(state.radius, 2, Math.max(2, Math.floor(system.length / 3)));
  state.center = G.clampInt(state.center, state.radius, system.length - state.radius - 1);
  state.ancestryLevel = G.clampInt(state.ancestryLevel, 0, system.maxAncestryLevel || 0);
}

function setupControls() {
  const systemSelect = $("system-select");
  systemSelect.innerHTML = "";
  for (const id of G.SYSTEM_IDS) {
    const system = G.makeSystem(id);
    const option = document.createElement("option");
    option.value = id;
    option.textContent = system.label;
    systemSelect.appendChild(option);
  }
  systemSelect.value = state.systemId;
  systemSelect.addEventListener("change", () => {
    state.systemId = systemSelect.value;
    const system = currentSystem();
    state.center = Math.floor(system.length / 2);
    state.radius = Math.min(12, Math.floor(system.length / 4));
    state.ancestryLevel = Math.min(3, system.maxAncestryLevel || 0);
    render();
  });

  $("center-range").addEventListener("input", (event) => {
    state.center = Number(event.target.value);
    render();
  });
  $("radius-range").addEventListener("input", (event) => {
    state.radius = Number(event.target.value);
    render();
  });
  $("ancestry-level").addEventListener("change", (event) => {
    state.ancestryLevel = Number(event.target.value);
    render();
  });
  $("btn-center").addEventListener("click", () => {
    const system = currentSystem();
    state.center = Math.floor(system.length / 2);
    render();
  });
  $("btn-grow").addEventListener("click", () => {
    state.radius += 3;
    render();
  });
  $("btn-shrink").addEventListener("click", () => {
    state.radius -= 3;
    render();
  });
}

function renderControls(system) {
  clampStateToSystem(system);
  const center = $("center-range");
  center.min = String(state.radius);
  center.max = String(system.length - state.radius - 1);
  center.value = String(state.center);

  const radius = $("radius-range");
  radius.min = "2";
  radius.max = String(Math.max(2, Math.min(40, Math.floor(system.length / 3))));
  radius.value = String(state.radius);

  const levelSelect = $("ancestry-level");
  levelSelect.innerHTML = "";
  for (let level = 0; level <= (system.maxAncestryLevel || 0); level++) {
    const option = document.createElement("option");
    option.value = String(level);
    option.textContent = `level ${level}`;
    levelSelect.appendChild(option);
  }
  levelSelect.value = String(state.ancestryLevel);
  levelSelect.disabled = system.type !== "substitution";

  $("center-readout").textContent = String(state.center);
  $("radius-readout").textContent = String(state.radius);
  $("length-readout").textContent = String(system.length);
}

function boundariesForVisible(system, visibleStart, visibleEnd) {
  if (system.type !== "substitution") return new Set();
  const intervals = G.intervalsAtLevel(system, state.ancestryLevel);
  const markers = new Set();
  for (const interval of intervals) {
    if (interval.start >= visibleStart && interval.start < visibleEnd) markers.add(interval.start);
    if (interval.end > visibleStart && interval.end < visibleEnd) markers.add(interval.end);
  }
  return markers;
}

function renderStripe(system, analysis) {
  const stripe = $("stripe");
  const visible = G.visibleBounds(system, state.center, state.radius);
  const markers = boundariesForVisible(system, visible.start, visible.end);
  stripe.style.setProperty("--visible-count", String(visible.length));
  stripe.innerHTML = "";

  for (let i = visible.start; i < visible.end; i++) {
    const symbol = system.symbols[i];
    const cell = document.createElement("div");
    const inWindow = i >= analysis.window.start && i < analysis.window.end;
    const atLeftEdge = i === analysis.window.start;
    const atRightEdge = i === analysis.window.end - 1;
    cell.className = [
      "stripe-cell",
      `sym-${symbol.toLowerCase()}`,
      inWindow ? "selected" : "context",
      atLeftEdge ? "left-edge" : "",
      atRightEdge ? "right-edge" : "",
      markers.has(i) ? "ancestry-boundary" : "",
    ]
      .filter(Boolean)
      .join(" ");
    cell.textContent = symbol;
    cell.title = `index ${i}, symbol ${symbol}`;
    stripe.appendChild(cell);
  }

  $("visible-range").textContent = `${visible.start}..${visible.end - 1}`;
  $("window-range").textContent = `${analysis.window.start}..${analysis.window.end - 1}`;
}

function renderMetrics(system, analysis) {
  $("regime").textContent = analysis.regime;
  $("verdict").textContent = analysis.verdict;
  $("verdict").className = "verdict " + (analysis.periodicClosed ? "closed" : "open");
  $("metric-window").textContent = String(analysis.window.length);
  $("metric-counts").textContent = countsText(analysis.counts);
  $("metric-periods").textContent = candidatesText(analysis.periodCandidates);

  if (system.type === "periodic") {
    $("metric-ancestry").textContent = "repeat cell: ABCD";
    $("ancestry-note").textContent =
      "Periodic control: the ghost collapses to a finite motif and repeat rule.";
  } else {
    const a = analysis.ancestry;
    $("metric-ancestry").textContent = `level ${a.level}, blocks ${a.intersectingBlocks}, boundaries ${a.interiorBoundaries}`;
    $("ancestry-note").textContent =
      a.containingBlockLength
        ? `Window sits inside one level-${a.level} supertile of length ${a.containingBlockLength}.`
        : `Window crosses level-${a.level} supertile boundaries; ancestry is visible but not a repeat cell.`;
  }

  $("cliff-note").textContent = analysis.cliff.note;
}

function renderExport(analysis) {
  const system = currentSystem();
  const data = G.exportReaderAnalysis(system, state.center, state.radius, state.ancestryLevel);
  $("export-out").textContent = JSON.stringify(data, null, 2);
}

function render() {
  const system = currentSystem();
  renderControls(system);
  const analysis = G.analyzeWindow(system, state.center, state.radius, state.ancestryLevel);
  renderStripe(system, analysis);
  renderMetrics(system, analysis);
  renderExport(analysis);
}

setupControls();
render();

