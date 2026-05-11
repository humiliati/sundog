import {
  ACTION,
  TILE,
  MINES_PRESETS,
  applyMinesAction,
  getPublicMemory,
  initializeBoardState,
  makeRng,
} from "./mines-core.mjs";
import { createSensorRuntime, normalizeSensorConfig } from "./mines-sensor.mjs";
import {
  MINES_CONTROLLER_MODES,
  chooseMinesAction,
  frontierIndices,
} from "./mines-controllers.mjs";
import { assessMinesBoundary } from "./mines-boundary.mjs";

const canvas = document.getElementById("mines-canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("mines-status");
const phaseEl = document.getElementById("phase-display");
const metricsEl = document.getElementById("mines-metrics");
const modeSelect = document.getElementById("mines-mode");
const compareModeSelect = document.getElementById("mines-compare-mode");
const presetSelect = document.getElementById("mines-preset");
const sensorSelect = document.getElementById("mines-sensor");
const seedInput = document.getElementById("mines-seed");
const speedInput = document.getElementById("mines-speed");
const compareToggle = document.getElementById("mines-compare");
const auditToggle = document.getElementById("mines-audit");
const runButton = document.getElementById("mines-run");
const stepButton = document.getElementById("mines-step");
const resetButton = document.getElementById("mines-reset");
const nextSeedButton = document.getElementById("mines-next-seed");
const copyReplayButton = document.getElementById("mines-copy-replay");
const loadBestButton = document.getElementById("mines-load-best");
const loadWorstButton = document.getElementById("mines-load-worst");
const bodyLoadBestButton = document.getElementById("mines-body-load-best");
const bodyLoadWorstButton = document.getElementById("mines-body-load-worst");
const boundaryPanel = document.getElementById("mines-boundary-panel");
const boundaryStatus = document.getElementById("mines-boundary-status");
const boundarySummary = document.getElementById("mines-boundary-summary");
const boundaryList = document.getElementById("mines-boundary-list");

const BEST_CELL_PARAMS = Object.freeze({
  preset: "easy_sparse",
  seed: "47",
  mode: "sundog_minimal",
  sensor: "doc_default",
  compare: "naive_pressure",
  mine_count: "13",
  scan_budget: "0",
  sigma_noise: "2",
  dropout: "0.2",
});

const WORST_CELL_PARAMS = Object.freeze({
  preset: "easy_sparse",
  seed: "39",
  mode: "sundog_lean",
  sensor: "doc_default",
  compare: "naive_pressure",
  mine_count: "18",
  scan_budget: "0",
  sigma_noise: "1",
  dropout: "0.35",
});

const SENSOR_CELLS = Object.freeze({
  doc_default: Object.freeze({
    label: "Doc default",
    config: Object.freeze({ sigma: 1.0, sigmaNoise: 0.1, dropoutRate: 0.1, delaySteps: 0 }),
  }),
  blur_noise_cliff: Object.freeze({
    label: "Blur/noise cliff",
    config: Object.freeze({ sigma: 8.0, sigmaNoise: 10.0, dropoutRate: 0.8, delaySteps: 0 }),
  }),
});

const MODE_ORDER = Object.freeze([
  "sundog_lean",
  "naive_pressure",
  "threshold_flagger",
  "sundog_minimal",
  "sundog_controller",
  "sundog_no_gradient",
  "sundog_no_scan",
  "sundog_no_action_history",
  "sundog_no_confidence_gate",
  "random_reveal",
  "naive_pressure_shuffled",
  "naive_pressure_delayed",
  "oracle_safe",
]);

const COLORS = Object.freeze({
  paper: "#f2f7f6",
  ink: "#0c1b26",
  panel: "rgba(7, 17, 27, 0.68)",
  grid: "rgba(244, 196, 48, 0.24)",
  gold: "#f4c430",
  safe: "#d8ebe8",
  flag: "#f4c430",
  mine: "#d84f3f",
  scan: "#7fd1ff",
  dropout: "#34414d",
});

let app = {
  lanes: [],
  running: true,
  lastStepAt: 0,
  speedMs: 520,
  compare: true,
  audit: false,
  boardOverride: {},
  sensorOverride: {},
};

function hashText(text) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function centerAction(board) {
  return {
    type: ACTION.REVEAL,
    x: Math.floor(board.config.width / 2),
    y: Math.floor(board.config.height / 2),
  };
}

function modeLabel(mode) {
  return MINES_CONTROLLER_MODES[mode]?.label ?? mode;
}

function seedValue() {
  const parsed = Number.parseInt(seedInput.value, 10);
  return Number.isInteger(parsed) ? parsed : 1;
}

function intParam(params, key) {
  const raw = params.get(key);
  if (raw === null) return null;
  const value = Number.parseInt(raw, 10);
  return Number.isInteger(value) ? value : null;
}

function floatParam(params, key) {
  const raw = params.get(key);
  if (raw === null) return null;
  const value = Number.parseFloat(raw);
  return Number.isFinite(value) ? value : null;
}

function readReplayOverrides(params) {
  const boardOverride = {};
  const mineCount = intParam(params, "mine_count");
  const width = intParam(params, "width");
  const height = intParam(params, "height");
  const scanBudget = intParam(params, "scan_budget");
  const clusterStrength = floatParam(params, "cluster_strength");
  if (mineCount !== null) boardOverride.mineCount = mineCount;
  if (width !== null) boardOverride.width = width;
  if (height !== null) boardOverride.height = height;
  if (scanBudget !== null) boardOverride.scanBudget = scanBudget;
  if (clusterStrength !== null) boardOverride.generator = { clusterStrength };

  const sensorOverride = {};
  const sigma = floatParam(params, "sigma");
  const sigmaNoise = floatParam(params, "sigma_noise");
  const dropoutRate = floatParam(params, "dropout");
  const delaySteps = intParam(params, "delay");
  if (sigma !== null) sensorOverride.sigma = sigma;
  if (sigmaNoise !== null) sensorOverride.sigmaNoise = sigmaNoise;
  if (dropoutRate !== null) sensorOverride.dropoutRate = dropoutRate;
  if (delaySteps !== null) sensorOverride.delaySteps = delaySteps;

  return { boardOverride, sensorOverride };
}

function sensorConfigFor(mode, seed) {
  const cell = SENSOR_CELLS[sensorSelect.value] ?? SENSOR_CELLS.doc_default;
  const definition = MINES_CONTROLLER_MODES[mode];
  return normalizeSensorConfig({
    ...cell.config,
    ...(definition.sensorOverride ?? {}),
    ...app.sensorOverride,
    sensorSeed: seed + 7919 + hashText(mode),
  });
}

function createLane({ role, mode, seedOffset }) {
  const seed = seedValue() + seedOffset;
  const definition = MINES_CONTROLLER_MODES[mode];
  const board = initializeBoardState({
    preset: presetSelect.value,
    seed,
    turnCap: 160,
    ...(definition.boardOverride ?? {}),
    ...app.boardOverride,
  });
  applyMinesAction(board, centerAction(board));
  const sensorRuntime = createSensorRuntime(sensorConfigFor(mode, seed));
  const lane = {
    role,
    mode,
    board,
    sensorRuntime,
    rng: makeRng(seed ^ hashText(mode) ^ hashText(sensorSelect.value)),
    sensor: null,
    action: null,
    illegalActions: 0,
  };
  lane.sensor = lane.sensorRuntime.step(lane.board);
  return lane;
}

function resetWorkbench() {
  app.speedMs = Number.parseInt(speedInput.value, 10) || 520;
  app.compare = compareToggle.checked;
  app.audit = auditToggle.checked;
  const lanes = [
    createLane({ role: "Primary", mode: modeSelect.value, seedOffset: 0 }),
  ];
  if (app.compare) {
    lanes.push(createLane({ role: "Compare", mode: compareModeSelect.value, seedOffset: 0 }));
  }
  app.lanes = lanes;
  app.lastStepAt = 0;
  updateChrome();
  draw();
}

function fallbackAction(lane) {
  return chooseMinesAction({
    mode: "random_reveal",
    memory: getPublicMemory(lane.board),
    sensor: lane.sensor,
    boardState: lane.board,
    rng: lane.rng,
  });
}

function commitAction(lane, action) {
  const result = applyMinesAction(lane.board, action);
  if (result.applied && action.type === ACTION.SCAN) {
    const scan = lane.sensorRuntime.scan(lane.board, action.x, action.y);
    const lastEntry = lane.board.actionLedger[lane.board.actionLedger.length - 1];
    if (lastEntry?.type === ACTION.SCAN && lastEntry.index === scan.index) {
      lastEntry.scanReading = scan.reading;
    }
  }
  if (!result.applied) {
    lane.illegalActions += 1;
    applyMinesAction(lane.board, fallbackAction(lane));
  }
  lane.action = action;
}

function stepLane(lane) {
  if (lane.board.terminal !== null) return;
  lane.sensor = lane.sensorRuntime.step(lane.board);
  const action = chooseMinesAction({
    mode: lane.mode,
    memory: getPublicMemory(lane.board),
    sensor: lane.sensor,
    boardState: lane.board,
    rng: lane.rng,
    options: { threshold: 1.2 },
  });
  commitAction(lane, action);
}

function stepWorkbench() {
  app.lanes.forEach(stepLane);
  updateChrome();
  draw();
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function pressureColor(value, confidence) {
  if (!Number.isFinite(value) || confidence <= 0) return COLORS.dropout;
  const t = Math.max(0, Math.min(1, value / 2.4));
  const r = Math.round(26 + 205 * t);
  const g = Math.round(89 + 124 * (1 - Math.abs(t - 0.45)));
  const b = Math.round(112 - 82 * t);
  return `rgb(${r}, ${g}, ${b})`;
}

function terminalLabel(terminal) {
  if (terminal === "mine_triggered") return "mine";
  if (terminal === "full_clear") return "clear";
  if (terminal === "turn_cap") return "cap";
  if (terminal === "scan_budget_exhausted") return "scan cap";
  return "running";
}

function actionLabel(action) {
  if (!action) return "opening";
  if (action.type === ACTION.ABSTAIN) return "abstain";
  return `${action.type} ${action.x},${action.y}`;
}

function laneStats(lane) {
  const memory = getPublicMemory(lane.board);
  const frontier = frontierIndices(memory);
  const confidenceValues = frontier
    .map((idx) => lane.sensor?.confidence?.[idx])
    .filter((value) => Number.isFinite(value));
  const meanConfidence = confidenceValues.length > 0
    ? confidenceValues.reduce((acc, value) => acc + value, 0) / confidenceValues.length
    : null;
  return {
    frontier: frontier.length,
    meanConfidence,
    flags: Array.from(lane.board.flags).filter(Boolean).length,
    scansUsed: lane.board.config.scanBudget - lane.board.scansRemaining,
  };
}

function updateChrome() {
  const primary = app.lanes[0];
  if (!primary) return;
  const status = primary.board.terminal === null
    ? `${modeLabel(primary.mode)} taking ${actionLabel(primary.action)}`
    : `${modeLabel(primary.mode)} ended: ${terminalLabel(primary.board.terminal)}`;
  statusEl.textContent = status;
  phaseEl.textContent = app.compare ? "MATCHED" : "LIVE";
  metricsEl.innerHTML = app.lanes.map((lane) => {
    const stats = laneStats(lane);
    const confidence = stats.meanConfidence === null ? "n/a" : stats.meanConfidence.toFixed(2);
    return [
      `<span>${lane.role}: ${lane.board.revealedSafeCount} safe</span>`,
      `<span>${stats.flags} flags</span>`,
      `<span>${lane.board.falseFlagCount} false</span>`,
      `<span>${stats.scansUsed} scans</span>`,
      `<span>${stats.frontier} frontier</span>`,
      `<span>${confidence} conf</span>`,
    ].join("");
  }).join("");
  updateBoundaryPanel(primary);
}

function updateBoundaryPanel(primary) {
  if (!boundaryPanel || !boundaryStatus || !boundarySummary || !boundaryList) return;
  const stats = laneStats(primary);
  const assessment = assessMinesBoundary({
    boardConfig: primary.board.config,
    sensorConfig: sensorConfigFor(primary.mode, seedValue()),
    mode: primary.mode,
    live: {
      frontierSize: stats.frontier,
      meanFrontierConfidence: stats.meanConfidence,
      falseFlagCount: primary.board.falseFlagCount,
      terminal: primary.board.terminal,
    },
  });
  boundaryPanel.dataset.status = assessment.status;
  boundaryStatus.textContent = assessment.label;
  boundarySummary.textContent = assessment.summary;
  if (assessment.mechanisms.length === 0) {
    boundaryList.innerHTML = "<li>No active density, blur, delay, dropout, scan-budget, or live-confidence warning.</li>";
    return;
  }
  boundaryList.innerHTML = assessment.mechanisms.slice(0, 5).map((mechanism) => {
    const value = mechanism.value === null || mechanism.value === undefined ? "" : ` (${mechanism.value})`;
    return `<li>${mechanism.code}${value}</li>`;
  }).join("");
}

function drawPanelFrame(box, lane) {
  ctx.fillStyle = COLORS.panel;
  ctx.fillRect(box.x, box.y, box.w, box.h);
  ctx.strokeStyle = "rgba(244, 196, 48, 0.32)";
  ctx.lineWidth = 1;
  ctx.strokeRect(box.x + 0.5, box.y + 0.5, box.w - 1, box.h - 1);
  ctx.fillStyle = COLORS.gold;
  ctx.font = "700 13px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(`${lane.role}: ${modeLabel(lane.mode)}`, box.x + 14, box.y + 12);
  ctx.fillStyle = "rgba(245, 245, 245, 0.76)";
  ctx.font = "12px ui-monospace, SFMono-Regular, Consolas, monospace";
  ctx.fillText(`turn ${lane.board.turn} | ${terminalLabel(lane.board.terminal)} | ${actionLabel(lane.action)}`, box.x + 14, box.y + 32);
}

function drawFlag(x, y, size) {
  ctx.fillStyle = COLORS.flag;
  ctx.beginPath();
  ctx.moveTo(x + size * 0.34, y + size * 0.22);
  ctx.lineTo(x + size * 0.72, y + size * 0.34);
  ctx.lineTo(x + size * 0.34, y + size * 0.48);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "rgba(10, 19, 27, 0.9)";
  ctx.lineWidth = Math.max(1, size * 0.045);
  ctx.beginPath();
  ctx.moveTo(x + size * 0.34, y + size * 0.22);
  ctx.lineTo(x + size * 0.34, y + size * 0.78);
  ctx.stroke();
}

function drawBoard(box, lane) {
  const { width, height } = lane.board.config;
  const boardMaxW = box.w - 28;
  const boardMaxH = box.h - 80;
  const tile = Math.floor(Math.min(boardMaxW / width, boardMaxH / height));
  const boardW = tile * width;
  const boardH = tile * height;
  const x0 = box.x + (box.w - boardW) / 2;
  const y0 = box.y + 58 + (boardMaxH - boardH) / 2;
  const observed = lane.sensor?.observed ?? [];
  const confidence = lane.sensor?.confidence ?? [];

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      const px = x0 + x * tile;
      const py = y0 + y * tile;
      const tileState = lane.board.tiles[idx];
      const conf = Number.isFinite(confidence[idx]) ? confidence[idx] : 0;
      ctx.fillStyle = pressureColor(observed[idx], conf);
      ctx.fillRect(px, py, tile - 1, tile - 1);

      if (tileState === TILE.CONCEALED) {
        ctx.fillStyle = "rgba(4, 13, 22, 0.58)";
        ctx.fillRect(px, py, tile - 1, tile - 1);
      } else if (tileState === TILE.REVEALED_SAFE) {
        ctx.fillStyle = "rgba(229, 244, 240, 0.72)";
        ctx.fillRect(px + 1, py + 1, tile - 3, tile - 3);
      } else if (tileState === TILE.REVEALED_MINE) {
        ctx.fillStyle = COLORS.mine;
        ctx.fillRect(px + 1, py + 1, tile - 3, tile - 3);
      }

      if (lane.board.scanned[idx]) {
        ctx.strokeStyle = COLORS.scan;
        ctx.lineWidth = Math.max(1, tile * 0.06);
        ctx.beginPath();
        ctx.arc(px + tile / 2, py + tile / 2, tile * 0.28, 0, Math.PI * 2);
        ctx.stroke();
      }

      if (lane.board.flags[idx]) drawFlag(px, py, tile);

      if (app.audit && lane.board.privileged.occupancy[idx] === 1) {
        ctx.fillStyle = "rgba(216, 79, 63, 0.82)";
        ctx.beginPath();
        ctx.arc(px + tile * 0.5, py + tile * 0.5, Math.max(2, tile * 0.11), 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      ctx.strokeRect(px + 0.5, py + 0.5, tile - 1, tile - 1);
    }
  }
}

function draw() {
  if (!canvas.width || !canvas.height) return;
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  const gradient = ctx.createLinearGradient(0, 0, rect.width, rect.height);
  gradient.addColorStop(0, "#0a1a28");
  gradient.addColorStop(0.55, "#132a32");
  gradient.addColorStop(1, "#301f23");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, rect.width, rect.height);

  const margin = 18;
  const gap = app.lanes.length > 1 ? 12 : 0;
  const panelW = (rect.width - margin * 2 - gap) / app.lanes.length;
  const panelH = rect.height - margin * 2;
  app.lanes.forEach((lane, index) => {
    const box = {
      x: margin + index * (panelW + gap),
      y: margin,
      w: panelW,
      h: panelH,
    };
    drawPanelFrame(box, lane);
    drawBoard(box, lane);
  });
}

function populateControls() {
  for (const preset of Object.keys(MINES_PRESETS)) {
    const option = document.createElement("option");
    option.value = preset;
    option.textContent = MINES_PRESETS[preset].label;
    presetSelect.append(option);
  }
  presetSelect.value = "easy_sparse";

  for (const [key, cell] of Object.entries(SENSOR_CELLS)) {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = cell.label;
    sensorSelect.append(option);
  }
  // Initial control values are overwritten by replay params below. Keep these
  // as mundane fallbacks for stale or stripped URLs.
  sensorSelect.value = "doc_default";

  for (const mode of MODE_ORDER) {
    if (!MINES_CONTROLLER_MODES[mode]) continue;
    for (const select of [modeSelect, compareModeSelect]) {
      const option = document.createElement("option");
      option.value = mode;
      option.textContent = modeLabel(mode);
      select.append(option);
    }
  }
  modeSelect.value = "sundog_lean";
  compareModeSelect.value = "naive_pressure";
}

// Replay URL contract — kept in lockstep with scripts/mines-phase4-baselines.mjs.
//   Required params: preset, seed, mode, sensor
//   Optional params: compare (a second mode for matched-seed side-by-side)
function buildReplayURL() {
  const params = new URLSearchParams({
    preset: presetSelect.value,
    seed: String(seedValue()),
    mode: modeSelect.value,
    sensor: sensorSelect.value,
  });
  if (compareToggle.checked && compareModeSelect.value) {
    params.set("compare", compareModeSelect.value);
  }
  appendReplayOverrides(params);
  const base = `${window.location.origin}${window.location.pathname}`;
  return `${base}?${params.toString()}`;
}

function appendReplayOverrides(params) {
  if (Number.isInteger(app.boardOverride.mineCount)) params.set("mine_count", String(app.boardOverride.mineCount));
  if (Number.isInteger(app.boardOverride.width)) params.set("width", String(app.boardOverride.width));
  if (Number.isInteger(app.boardOverride.height)) params.set("height", String(app.boardOverride.height));
  if (Number.isInteger(app.boardOverride.scanBudget)) params.set("scan_budget", String(app.boardOverride.scanBudget));
  if (Number.isFinite(app.boardOverride.generator?.clusterStrength)) {
    params.set("cluster_strength", String(app.boardOverride.generator.clusterStrength));
  }
  if (Number.isFinite(app.sensorOverride.sigma)) params.set("sigma", String(app.sensorOverride.sigma));
  if (Number.isFinite(app.sensorOverride.sigmaNoise)) params.set("sigma_noise", String(app.sensorOverride.sigmaNoise));
  if (Number.isFinite(app.sensorOverride.dropoutRate)) params.set("dropout", String(app.sensorOverride.dropoutRate));
  if (Number.isInteger(app.sensorOverride.delaySteps)) params.set("delay", String(app.sensorOverride.delaySteps));
}

async function copyReplayURL() {
  const url = buildReplayURL();
  try {
    await navigator.clipboard.writeText(url);
    copyReplayButton.textContent = "Copied";
    setTimeout(() => { copyReplayButton.textContent = "Copy replay URL"; }, 1400);
  } catch {
    // Clipboard API may be unavailable (file:// origin or denied perms).
    // Fall back to selecting the URL in a prompt so the user can copy manually.
    window.prompt("Copy this replay URL:", url);
  }
}

// Hydrate controls from any replay params present on the initial URL. Runs
// once before the first resetWorkbench(). Silently ignores params that don't
// correspond to known options so a stale or malformed URL degrades gracefully.
function hydrateFromURL() {
  const params = new URLSearchParams(window.location.search);
  if (params.size === 0) return false;
  const preset = params.get("preset");
  const seed = params.get("seed");
  const mode = params.get("mode");
  const sensor = params.get("sensor");
  const compare = params.get("compare");
  let applied = false;
  if (preset && MINES_PRESETS[preset]) {
    presetSelect.value = preset;
    applied = true;
  }
  const seedInt = Number.parseInt(seed ?? "", 10);
  if (Number.isInteger(seedInt) && seedInt >= 0) {
    seedInput.value = String(seedInt);
    applied = true;
  }
  if (mode && MINES_CONTROLLER_MODES[mode]) {
    modeSelect.value = mode;
    applied = true;
  }
  if (sensor && SENSOR_CELLS[sensor]) {
    sensorSelect.value = sensor;
    applied = true;
  }
  if (compare && MINES_CONTROLLER_MODES[compare]) {
    compareModeSelect.value = compare;
    compareToggle.checked = true;
    applied = true;
  } else if (params.has("compare") && !compare) {
    compareToggle.checked = false;
  }
  const overrides = readReplayOverrides(params);
  app.boardOverride = overrides.boardOverride;
  app.sensorOverride = overrides.sensorOverride;
  if (Object.keys(app.boardOverride).length > 0 || Object.keys(app.sensorOverride).length > 0) {
    applied = true;
  }
  return applied;
}

function applyReplayParams(replayParams, { replace = false } = {}) {
  const params = new URLSearchParams(replayParams);
  const target = `${window.location.pathname}?${params.toString()}`;
  if (replace) window.history.replaceState(null, "", target);
  else window.location.assign(target);
}

function bindControls() {
  runButton.addEventListener("click", () => {
    app.running = !app.running;
    runButton.textContent = app.running ? "Pause" : "Run";
  });
  stepButton.addEventListener("click", () => {
    app.running = false;
    runButton.textContent = "Run";
    stepWorkbench();
  });
  resetButton.addEventListener("click", resetWorkbench);
  nextSeedButton.addEventListener("click", () => {
    seedInput.stepUp();
    resetWorkbench();
  });
  copyReplayButton.addEventListener("click", () => { copyReplayURL(); });
  loadBestButton?.addEventListener("click", () => {
    applyReplayParams(BEST_CELL_PARAMS);
  });
  loadWorstButton?.addEventListener("click", () => {
    applyReplayParams(WORST_CELL_PARAMS);
  });
  bodyLoadBestButton?.addEventListener("click", () => {
    applyReplayParams(BEST_CELL_PARAMS);
  });
  bodyLoadWorstButton?.addEventListener("click", () => {
    applyReplayParams(WORST_CELL_PARAMS);
  });
  for (const input of [modeSelect, compareModeSelect, presetSelect, sensorSelect, seedInput, compareToggle, auditToggle]) {
    input.addEventListener("change", resetWorkbench);
  }
  speedInput.addEventListener("input", () => {
    app.speedMs = Number.parseInt(speedInput.value, 10) || 520;
  });
  window.addEventListener("resize", resizeCanvas);
}

function frame(now) {
  if (app.running && now - app.lastStepAt >= app.speedMs) {
    stepWorkbench();
    app.lastStepAt = now;
  } else {
    draw();
  }
  requestAnimationFrame(frame);
}

populateControls();
// Replay URL params override the page's defaults if present. Runs after
// populateControls so the option lists exist, before bindControls/reset so
// the hydrated values are what the first render sees.
if (!hydrateFromURL()) {
  applyReplayParams(BEST_CELL_PARAMS, { replace: true });
  hydrateFromURL();
}
bindControls();
resetWorkbench();
resizeCanvas();
requestAnimationFrame(frame);
