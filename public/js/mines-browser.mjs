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

function buildReplayURL() {
  const url = new URL(window.location.href);
  url.search = "";
  url.searchParams.set("preset", presetSelect.value);
  url.searchParams.set("seed", String(seedValue()));
  url.searchParams.set("mode", modeSelect.value);
  url.searchParams.set("sensor", sensorSelect.value);
  if (compareToggle.checked && compareModeSelect.value) {
    url.searchParams.set("compare", compareModeSelect.value);
  }
  // Re-emit any board/sensor overrides currently in app state so the URL
  // is a complete round-trip of the rendered configuration. The UI itself
  // does not expose controls for these knobs (they enter only via URL).
  const boardKeyMap = { mineCount: "mine_count", scanBudget: "scan_budget" };
  for (const [key, value] of Object.entries(app.boardOverride)) {
    if (key === "generator") {
      if (value?.clusterStrength != null) {
        url.searchParams.set("cluster_strength", String(value.clusterStrength));
      }
    } else {
      const urlKey = boardKeyMap[key] ?? key;
      url.searchParams.set(urlKey, String(value));
    }
  }
  const sensorKeyMap = {
    sigmaNoise: "sigma_noise",
    dropoutRate: "dropout",
    delaySteps: "delay",
  };
  for (const [key, value] of Object.entries(app.sensorOverride)) {
    const urlKey = sensorKeyMap[key] ?? key;
    url.searchParams.set(urlKey, String(value));
  }
  return url.toString();
}

async function copyReplayURL() {
  const url = buildReplayURL();
  try {
    await navigator.clipboard.writeText(url);
    copyReplayButton.textContent = "Copied";
    setTimeout(() => { copyReplayButton.textContent = "Copy replay URL"; }, 1400);
  } catch {
    window.prompt("Copy this replay URL:", url);
  }
}

function paramsFromCell(cellParams) {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(cellParams)) {
    params.set(key, String(value));
  }
  return params;
}

function hydrateControlsFromParams(params) {
  let applied = false;
  const preset = params.get("preset");
  if (preset && MINES_PRESETS[preset]) {
    presetSelect.value = preset;
    applied = true;
  }
  const seedRaw = params.get("seed");
  const seedInt = Number.parseInt(seedRaw ?? "", 10);
  if (Number.isInteger(seedInt) && seedInt >= 0) {
    seedInput.value = String(seedInt);
    applied = true;
  }
  const mode = params.get("mode");
  if (mode && MINES_CONTROLLER_MODES[mode]) {
    modeSelect.value = mode;
    applied = true;
  }
  const sensor = params.get("sensor");
  if (sensor && SENSOR_CELLS[sensor]) {
    sensorSelect.value = sensor;
    applied = true;
  }
  const compare = params.get("compare");
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

function hydrateFromURL() {
  const params = new URLSearchParams(window.location.search);
  if (Array.from(params).length === 0) return false;
  return hydrateControlsFromParams(params);
}

function applyDefaultCellParams(cellParams) {
  const params = paramsFromCell(cellParams);
  hydrateControlsFromParams(params);
  const url = new URL(window.location.href);
  url.search = params.toString();
  window.history.replaceState(null, "", url.toString());
}

function loadCellParams(cellParams) {
  const url = new URL(window.location.href);
  url.search = paramsFromCell(cellParams).toString();
  // Assigning to location.href triggers a navigation; the new page load
  // calls hydrateFromURL during its boot sequence.
  window.location.assign(url.toString());
}

function loadBestCell() { loadCellParams(BEST_CELL_PARAMS); }
function loadWorstCell() { loadCellParams(WORST_CELL_PARAMS); }

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
  // Phase 11 best/worst-cell shortcuts in the right rail and body copy.
  loadBestButton?.addEventListener("click", loadBestCell);
  loadWorstButton?.addEventListener("click", loadWorstCell);
  bodyLoadBestButton?.addEventListener("click", loadBestCell);
  bodyLoadWorstButton?.addEventListener("click", loadWorstCell);
  // Preset and sensor changes invalidate URL-hydrated overrides because the
  // overrides were specific to the cell that was hydrated. Other inputs
  // (mode/compare/seed/audit) preserve overrides so the user can sweep
  // controllers across the same cell.
  const presetOrSensorInputs = new Set([presetSelect, sensorSelect]);
  for (const input of [modeSelect, compareModeSelect, presetSelect, sensorSelect, seedInput, compareToggle, auditToggle]) {
    input.addEventListener("change", () => {
      if (presetOrSensorInputs.has(input)) {
        app.boardOverride = {};
        app.sensorOverride = {};
      }
      resetWorkbench();
    });
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

// Boot. populateControls fills the dropdowns; hydrateFromURL runs after so
// any ?preset/seed/mode/sensor/compare query params replace the defaults
// before the first reset renders.
populateControls();
if (!hydrateFromURL() && window.location.search === "") {
  applyDefaultCellParams(WORST_CELL_PARAMS);
}
bindControls();
resetWorkbench();
resizeCanvas();
requestAnimationFrame(frame);
