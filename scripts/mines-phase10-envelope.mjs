import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  ACTION,
  applyMinesAction,
  getPublicMemory,
  initializeBoardState,
  makeRng,
  normalizeMinesConfig,
} from "../public/js/mines-core.mjs";
import { createSensorRuntime, normalizeSensorConfig } from "../public/js/mines-sensor.mjs";
import {
  IMPLEMENTED_MINES_MODES,
  MINES_CONTROLLER_MODES,
  chooseMinesAction,
  frontierIndices,
} from "../public/js/mines-controllers.mjs";
import { assessStaticBoundary } from "../public/js/mines-boundary.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const DEFAULT_MODES = Object.freeze([
  "random_reveal",
  "naive_pressure",
  "threshold_flagger",
  "sundog_lean",
  "sundog_minimal",
  "sundog_controller",
  "oracle_safe",
]);

const CANDIDATE_MODES = Object.freeze(["sundog_lean", "sundog_minimal"]);
const LINEAGE_MODES = Object.freeze(["sundog_controller"]);
const BASELINE_MODES = Object.freeze(["random_reveal", "naive_pressure", "threshold_flagger", "oracle_safe"]);
const REPLAY_SENSOR_CELL = "doc_default";
const REPLAY_PRESET = "easy_sparse";

const BASE_BOARD = Object.freeze({
  preset: REPLAY_PRESET,
  width: 9,
  height: 9,
  mineCount: 10,
  scanBudget: 3,
  generator: Object.freeze({ clusterStrength: 0 }),
});

const BASE_SENSOR = Object.freeze({
  sigma: 1.0,
  sigmaNoise: 0.1,
  dropoutRate: 0.1,
  delaySteps: 0,
});

const PRE_REGISTERED_GRID = Object.freeze({
  mineDensities: Object.freeze([0.10, 0.16, 0.22]),
  pressureNoise: Object.freeze([0.1, 0.5, 1.0, 2.0]),
  dropoutRates: Object.freeze([0.05, 0.20, 0.35]),
  sensorDelay: Object.freeze([0, 1, 2]),
  clusteringStrength: Object.freeze([0, 0.35, 0.65]),
  kernelBlur: Object.freeze([1.0, 2.0, 3.0]),
  scanBudget: Object.freeze([0, 1, 3, 6]),
});

function parseList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function parseArgs(argv) {
  const args = {
    phase: "phase10-envelope",
    out: "results/mines/phase10-envelope",
    seedStart: 0,
    seeds: 64,
    turnCap: 160,
    pressureThreshold: 1.2,
    modes: [...DEFAULT_MODES],
    bootstrapIterations: 1000,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    if (!flag.startsWith("--")) continue;
    const value = argv[i + 1];
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`${flag} requires a value`);
    }
    i += 1;

    if (flag === "--phase") args.phase = value;
    else if (flag === "--out") args.out = value;
    else if (flag === "--seed-start") args.seedStart = Number.parseInt(value, 10);
    else if (flag === "--seeds") args.seeds = Number.parseInt(value, 10);
    else if (flag === "--turn-cap") args.turnCap = Number.parseInt(value, 10);
    else if (flag === "--pressure-threshold") args.pressureThreshold = Number.parseFloat(value);
    else if (flag === "--modes") args.modes = parseList(value);
    else if (flag === "--bootstrap-iterations") args.bootstrapIterations = Number.parseInt(value, 10);
    else throw new Error(`Unknown flag: ${flag}`);
  }

  validateArgs(args);
  return args;
}

function validateArgs(args) {
  if (!Number.isInteger(args.seedStart) || args.seedStart < 0) {
    throw new Error("--seed-start must be a non-negative integer");
  }
  if (!Number.isInteger(args.seeds) || args.seeds < 1) {
    throw new Error("--seeds must be a positive integer");
  }
  if (!Number.isInteger(args.turnCap) || args.turnCap < 1) {
    throw new Error("--turn-cap must be a positive integer");
  }
  if (!Number.isInteger(args.bootstrapIterations) || args.bootstrapIterations < 100) {
    throw new Error("--bootstrap-iterations must be an integer >= 100");
  }
  if (!Number.isFinite(args.pressureThreshold)) {
    throw new Error("--pressure-threshold must be numeric");
  }
  for (const mode of args.modes) {
    if (!MINES_CONTROLLER_MODES[mode]) throw new Error(`Unknown mines mode: ${mode}`);
    if (!IMPLEMENTED_MINES_MODES.includes(mode)) {
      throw new Error(`Mode ${mode} is declared but not implemented yet`);
    }
  }
  for (const required of ["random_reveal", "naive_pressure", ...CANDIDATE_MODES]) {
    if (!args.modes.includes(required)) throw new Error(`Phase 10 requires mode ${required}`);
  }
}

function roundMetric(value, digits = 6) {
  if (value === null || value === undefined || Number.isNaN(value)) return null;
  if (!Number.isFinite(value)) return value;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function mean(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) return null;
  return finite.reduce((acc, value) => acc + value, 0) / finite.length;
}

function sum(values) {
  return values.reduce((acc, value) => acc + (Number.isFinite(value) ? value : 0), 0);
}

function rate(rows, predicate) {
  if (rows.length === 0) return null;
  return rows.filter(predicate).length / rows.length;
}

function quantile(sortedValues, q) {
  if (sortedValues.length === 0) return null;
  const pos = (sortedValues.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sortedValues[lo];
  const weight = pos - lo;
  return sortedValues[lo] * (1 - weight) + sortedValues[hi] * weight;
}

function stringHash(text) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function hashHex(text) {
  return stringHash(text).toString(16).padStart(8, "0");
}

function safeId(value) {
  return String(value)
    .replaceAll(".", "p")
    .replaceAll("-", "m")
    .replace(/[^A-Za-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .toLowerCase();
}

function bootstrapMean(values, iterations, salt) {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return { low: null, high: null };
  const rng = makeRng(stringHash(salt));
  const samples = [];
  for (let i = 0; i < iterations; i += 1) {
    let total = 0;
    for (let j = 0; j < finite.length; j += 1) {
      total += finite[Math.floor(rng() * finite.length)];
    }
    samples.push(total / finite.length);
  }
  samples.sort((a, b) => a - b);
  return {
    low: quantile(samples, 0.025),
    high: quantile(samples, 0.975),
  };
}

function centerAction(board) {
  return {
    type: ACTION.REVEAL,
    x: Math.floor(board.config.width / 2),
    y: Math.floor(board.config.height / 2),
  };
}

function terminalClass(terminal) {
  if (terminal === "full_clear") return "full_clear";
  if (terminal === "mine_triggered") return "mine_triggered";
  if (terminal === "turn_cap") return "turn_cap";
  if (terminal === "scan_budget_exhausted") return "scan_budget_exhausted";
  return terminal ?? "none";
}

function normalizedActionTrace(ledger) {
  return ledger.map((entry) => ({
    turn: entry.turn,
    type: entry.type,
    x: Number.isInteger(entry.x) ? entry.x : null,
    y: Number.isInteger(entry.y) ? entry.y : null,
    outcome: entry.outcome ?? "",
    scansRemainingAfter: Number.isInteger(entry.scansRemainingAfter) ? entry.scansRemainingAfter : null,
    scanReading: Number.isFinite(entry.scanReading) ? roundMetric(entry.scanReading, 12) : null,
  }));
}

function countLedger(board, type) {
  return board.actionLedger.filter((entry) => entry.type === type).length;
}

function mineCountForDensity(width, height, density) {
  return Math.max(1, Math.round(width * height * density));
}

function configSignature(board, sensor) {
  const normalizedBoard = normalizeMinesConfig(board);
  const normalizedSensor = normalizeSensorConfig(sensor);
  return JSON.stringify({
    width: normalizedBoard.width,
    height: normalizedBoard.height,
    mineCount: normalizedBoard.mineCount,
    scanBudget: normalizedBoard.scanBudget,
    clusterStrength: normalizedBoard.generator?.clusterStrength ?? 0,
    sigma: normalizedSensor.sigma,
    sigmaNoise: normalizedSensor.sigmaNoise,
    dropoutRate: normalizedSensor.dropoutRate,
    delaySteps: normalizedSensor.delaySteps,
  });
}

function cellIdOf(gridKind, axis, axisValue, board, sensor) {
  const normalizedBoard = normalizeMinesConfig(board);
  const normalizedSensor = normalizeSensorConfig(sensor);
  return [
    safeId(gridKind),
    safeId(axis),
    safeId(axisValue),
    `w${normalizedBoard.width}`,
    `h${normalizedBoard.height}`,
    `m${normalizedBoard.mineCount}`,
    `c${safeId(normalizedBoard.generator?.clusterStrength ?? 0)}`,
    `s${safeId(normalizedSensor.sigma)}`,
    `n${safeId(normalizedSensor.sigmaNoise)}`,
    `d${safeId(normalizedSensor.dropoutRate)}`,
    `lag${safeId(normalizedSensor.delaySteps)}`,
    `scan${safeId(normalizedBoard.scanBudget)}`,
  ].join("__");
}

function makeCell(gridKind, axis, axisValue, { board = {}, sensor = {}, label = "" } = {}) {
  const mergedBoard = {
    ...BASE_BOARD,
    ...board,
    generator: {
      ...(BASE_BOARD.generator ?? {}),
      ...(board.generator ?? {}),
    },
  };
  const mergedSensor = {
    ...BASE_SENSOR,
    ...sensor,
  };
  return Object.freeze({
    cellId: cellIdOf(gridKind, axis, axisValue, mergedBoard, mergedSensor),
    gridKind,
    axis,
    axisValue,
    label: label || `${axis}=${axisValue}`,
    board: Object.freeze(mergedBoard),
    sensor: normalizeSensorConfig(mergedSensor),
    signature: configSignature(mergedBoard, mergedSensor),
  });
}

function makeEnvelopeCells() {
  const cells = [];
  const seen = new Set();
  const add = (cell) => {
    if (seen.has(cell.signature)) return;
    seen.add(cell.signature);
    cells.push(cell);
  };

  for (const density of PRE_REGISTERED_GRID.mineDensities) {
    for (const sigmaNoise of PRE_REGISTERED_GRID.pressureNoise) {
      for (const dropoutRate of PRE_REGISTERED_GRID.dropoutRates) {
        add(makeCell("primary_factorial", "density_noise_dropout", `density_${density}__noise_${sigmaNoise}__dropout_${dropoutRate}`, {
          board: { mineCount: mineCountForDensity(BASE_BOARD.width, BASE_BOARD.height, density) },
          sensor: { sigmaNoise, dropoutRate },
          label: `density ${density}, noise ${sigmaNoise}, dropout ${dropoutRate}`,
        }));
      }
    }
  }

  for (const delaySteps of PRE_REGISTERED_GRID.sensorDelay) {
    add(makeCell("spoke", "sensor_delay", delaySteps, {
      sensor: { delaySteps },
    }));
  }
  for (const clusterStrength of PRE_REGISTERED_GRID.clusteringStrength) {
    add(makeCell("spoke", "clustering_strength", clusterStrength, {
      board: { generator: { clusterStrength } },
    }));
  }
  for (const sigma of PRE_REGISTERED_GRID.kernelBlur) {
    add(makeCell("spoke", "kernel_blur", sigma, {
      sensor: { sigma },
    }));
  }
  for (const scanBudget of PRE_REGISTERED_GRID.scanBudget) {
    add(makeCell("spoke", "scan_budget", scanBudget, {
      board: { scanBudget },
    }));
  }

  return cells;
}

function sensorConfigFor(cell, mode, seed) {
  const definition = MINES_CONTROLLER_MODES[mode];
  return normalizeSensorConfig({
    ...cell.sensor,
    ...(definition.sensorOverride ?? {}),
    sensorSeed: seed + 7919 + stringHash(mode),
  });
}

function meanFrontierConfidence(memory, sensor) {
  const frontier = frontierIndices(memory);
  return {
    frontierSize: frontier.length,
    meanFrontierConfidence: mean(frontier.map((idx) => sensor?.confidence?.[idx])),
  };
}

function trialIdOf({ phase, cell, mode, seed }) {
  return `${phase}:${cell.cellId}:${mode}:seed${seed}`;
}

function runTrial(args, { cell, mode, seed }) {
  const modeDefinition = MINES_CONTROLLER_MODES[mode];
  const board = initializeBoardState({
    ...cell.board,
    seed,
    turnCap: args.turnCap,
    ...(modeDefinition.boardOverride ?? {}),
  });
  applyMinesAction(board, centerAction(board));
  const openingSafeCount = board.revealedSafeCount;
  const sensorConfig = sensorConfigFor(cell, mode, seed);
  const sensorRuntime = createSensorRuntime(sensorConfig);
  const rng = makeRng(seed ^ stringHash(mode) ^ stringHash(REPLAY_SENSOR_CELL));
  let illegalActionCount = 0;
  let sensor = null;

  while (board.terminal === null) {
    sensor = sensorRuntime.step(board);
    const memory = getPublicMemory(board);
    const action = chooseMinesAction({
      mode,
      memory,
      sensor,
      boardState: board,
      rng,
      options: { threshold: args.pressureThreshold },
    });
    let result = applyMinesAction(board, action);
    if (result.applied && action.type === ACTION.SCAN) {
      const scan = sensorRuntime.scan(board, action.x, action.y);
      const lastEntry = board.actionLedger[board.actionLedger.length - 1];
      if (lastEntry?.type === ACTION.SCAN && lastEntry.index === scan.index) {
        lastEntry.scanReading = scan.reading;
      }
    }
    if (!result.applied) {
      illegalActionCount += 1;
      const fallback = chooseMinesAction({
        mode: "random_reveal",
        memory,
        sensor,
        boardState: board,
        rng,
      });
      result = applyMinesAction(board, fallback);
      if (result.applied && fallback.type === ACTION.SCAN) {
        const scan = sensorRuntime.scan(board, fallback.x, fallback.y);
        const lastEntry = board.actionLedger[board.actionLedger.length - 1];
        if (lastEntry?.type === ACTION.SCAN && lastEntry.index === scan.index) {
          lastEntry.scanReading = scan.reading;
        }
      }
    }
  }

  const memory = getPublicMemory(board);
  const frontierStats = meanFrontierConfidence(memory, sensor);
  const staticBoundary = assessStaticBoundary({
    boardConfig: board.config,
    sensorConfig,
    mode,
  });
  const revealCount = countLedger(board, ACTION.REVEAL);
  const flagCount = countLedger(board, ACTION.FLAG);
  const scanCount = countLedger(board, ACTION.SCAN);
  const rawSafeTiles = board.revealedSafeCount;
  const budgetAdjustedSafeTiles = rawSafeTiles - scanCount;
  const actionTraceHash = hashHex(JSON.stringify(normalizedActionTrace(board.actionLedger)));
  const replayUrl = makeReplayURL({
    seed,
    mode,
    compareMode: "naive_pressure",
    boardConfig: board.config,
    sensorConfig,
  });

  return {
    phase: args.phase,
    cell_id: cell.cellId,
    trial_id: trialIdOf({ phase: args.phase, cell, mode, seed }),
    grid_kind: cell.gridKind,
    axis: cell.axis,
    axis_value: cell.axisValue,
    label: cell.label,
    preset: REPLAY_PRESET,
    sensor_cell: REPLAY_SENSOR_CELL,
    mode,
    seed,
    action_trace_hash: actionTraceHash,
    mode_status: modeDefinition.status,
    uses_privileged: modeDefinition.usesPrivileged,
    uses_scan: modeDefinition.usesScan,
    width: board.config.width,
    height: board.config.height,
    mine_count: board.config.mineCount,
    mine_density: roundMetric(board.config.mineCount / (board.config.width * board.config.height)),
    cluster_strength: roundMetric(board.config.generator?.clusterStrength ?? 0),
    scan_budget: board.config.scanBudget,
    configured_scan_budget: normalizeMinesConfig(cell.board).scanBudget,
    turn_cap: args.turnCap,
    terminal: terminalClass(board.terminal),
    survived: board.terminal !== "mine_triggered",
    full_clear: board.terminal === "full_clear",
    mine_triggered: board.terminal === "mine_triggered",
    turns: board.turn,
    controller_turns: Math.max(0, board.turn - 1),
    reveal_count: revealCount,
    flag_count: flagCount,
    scan_count: scanCount,
    illegal_action_count: illegalActionCount,
    opening_safe_count: openingSafeCount,
    raw_safe_tiles: rawSafeTiles,
    safe_tiles_after_opening: Math.max(0, rawSafeTiles - openingSafeCount),
    budget_adjusted_safe_tiles: budgetAdjustedSafeTiles,
    false_flag_count: board.falseFlagCount,
    frontier_size: frontierStats.frontierSize,
    mean_frontier_confidence: roundMetric(frontierStats.meanFrontierConfidence),
    pressure_threshold: args.pressureThreshold,
    sigma: sensorConfig.sigma,
    sigma_noise: sensorConfig.sigmaNoise,
    dropout_rate: sensorConfig.dropoutRate,
    delay_steps: sensorConfig.delaySteps,
    kernel_family: sensorConfig.kernel,
    static_boundary_status: staticBoundary.status,
    static_boundary_mechanisms: staticBoundary.mechanismCodes.join("|"),
    replay_url: replayUrl,
    harness_replay_command: `npm run mines:phase7:replay -- "${replayUrl}"`,
  };
}

function makeReplayURL({ seed, mode, compareMode, boardConfig, sensorConfig }) {
  const params = new URLSearchParams({
    preset: REPLAY_PRESET,
    seed: String(seed),
    mode,
    sensor: REPLAY_SENSOR_CELL,
  });
  if (compareMode) params.set("compare", compareMode);
  const baseBoard = normalizeMinesConfig({ preset: REPLAY_PRESET });
  const clusterStrength = boardConfig.generator?.clusterStrength ?? 0;
  if (boardConfig.mineCount !== baseBoard.mineCount) params.set("mine_count", String(boardConfig.mineCount));
  if (boardConfig.width !== baseBoard.width) params.set("width", String(boardConfig.width));
  if (boardConfig.height !== baseBoard.height) params.set("height", String(boardConfig.height));
  if (boardConfig.scanBudget !== baseBoard.scanBudget) params.set("scan_budget", String(boardConfig.scanBudget));
  if (clusterStrength !== (baseBoard.generator?.clusterStrength ?? 0)) {
    params.set("cluster_strength", String(roundMetric(clusterStrength, 6)));
  }
  if (sensorConfig.sigma !== BASE_SENSOR.sigma) params.set("sigma", String(sensorConfig.sigma));
  if (sensorConfig.sigmaNoise !== BASE_SENSOR.sigmaNoise) params.set("sigma_noise", String(sensorConfig.sigmaNoise));
  if (sensorConfig.dropoutRate !== BASE_SENSOR.dropoutRate) params.set("dropout", String(sensorConfig.dropoutRate));
  if (sensorConfig.delaySteps !== BASE_SENSOR.delaySteps) params.set("delay", String(sensorConfig.delaySteps));
  return `https://sundog.cc/mines?${params.toString()}`;
}

function groupBy(rows, keyFn) {
  const groups = new Map();
  for (const row of rows) {
    const key = keyFn(row);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  return groups;
}

function mostCommon(values) {
  const counts = new Map();
  for (const value of values.filter(Boolean)) counts.set(value, (counts.get(value) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0]?.[0] ?? "";
}

function pairRows(rows, leftMode, rightMode) {
  const bySeed = groupBy(rows, (row) => row.seed);
  const pairs = [];
  for (const seedRows of bySeed.values()) {
    const left = seedRows.find((row) => row.mode === leftMode);
    const right = seedRows.find((row) => row.mode === rightMode);
    if (left && right) pairs.push({ left, right });
  }
  return pairs.sort((a, b) => a.left.seed - b.left.seed);
}

function deltaRowsFor(pairs, baselineMode) {
  return pairs.map(({ left, right }) => ({
    phase: left.phase,
    cell_id: left.cell_id,
    grid_kind: left.grid_kind,
    axis: left.axis,
    axis_value: left.axis_value,
    seed: left.seed,
    mode: left.mode,
    baseline_mode: baselineMode,
    target_terminal: left.terminal,
    baseline_terminal: right.terminal,
    raw_safe_tiles_delta: left.raw_safe_tiles - right.raw_safe_tiles,
    safe_tiles_after_opening_delta: left.safe_tiles_after_opening - right.safe_tiles_after_opening,
    budget_adjusted_safe_tiles_delta: left.budget_adjusted_safe_tiles - right.budget_adjusted_safe_tiles,
    survival_delta: (left.survived ? 1 : 0) - (right.survived ? 1 : 0),
    mine_trigger_delta: (left.mine_triggered ? 1 : 0) - (right.mine_triggered ? 1 : 0),
    false_flag_delta: left.false_flag_count - right.false_flag_count,
    scan_delta: left.scan_count - right.scan_count,
    replay_url: left.replay_url,
    harness_replay_command: left.harness_replay_command,
  }));
}

function makeMatchedComparisons(trialRows) {
  const rows = [];
  const byCell = groupBy(trialRows, (row) => row.cell_id);
  for (const cellRows of byCell.values()) {
    for (const mode of [...CANDIDATE_MODES, ...LINEAGE_MODES]) {
      for (const baseline of BASELINE_MODES) {
        if (mode === baseline) continue;
        rows.push(...deltaRowsFor(pairRows(cellRows, mode, baseline), baseline));
      }
    }
  }
  return rows.sort((a, b) => (
    a.cell_id.localeCompare(b.cell_id)
    || a.mode.localeCompare(b.mode)
    || a.baseline_mode.localeCompare(b.baseline_mode)
    || a.seed - b.seed
  ));
}

function summarizeMode(rows) {
  return {
    n: rows.length,
    survival_rate: roundMetric(rate(rows, (row) => row.survived)),
    full_clear_rate: roundMetric(rate(rows, (row) => row.full_clear)),
    mine_trigger_rate: roundMetric(rate(rows, (row) => row.mine_triggered)),
    mean_raw_safe_tiles: roundMetric(mean(rows.map((row) => row.raw_safe_tiles))),
    mean_safe_tiles_after_opening: roundMetric(mean(rows.map((row) => row.safe_tiles_after_opening))),
    mean_budget_adjusted_safe_tiles: roundMetric(mean(rows.map((row) => row.budget_adjusted_safe_tiles))),
    mean_false_flag_count: roundMetric(mean(rows.map((row) => row.false_flag_count))),
    mean_scan_count: roundMetric(mean(rows.map((row) => row.scan_count))),
    mean_turns: roundMetric(mean(rows.map((row) => row.turns))),
  };
}

function empiricalMechanisms({ staticCodes, naiveDelta, falseFlagDelta, mineTriggerDelta, randomDelta }) {
  const codes = new Set(staticCodes.filter(Boolean));
  if (falseFlagDelta > 1) codes.add("overflagged");
  if (mineTriggerDelta > 0.10) codes.add("controller_overcommitted");
  if (randomDelta < 0 || naiveDelta <= -1) codes.add("controller_overcommitted");
  if (codes.size === 0) codes.add("none");
  return [...codes].join("|");
}

function representativePair(pairs, targetMean, prefer = "closest") {
  if (pairs.length === 0) return null;
  const scored = pairs.map((pair) => ({
    pair,
    delta: pair.left.budget_adjusted_safe_tiles - pair.right.budget_adjusted_safe_tiles,
  }));
  if (prefer === "best") return scored.sort((a, b) => b.delta - a.delta || a.pair.left.seed - b.pair.left.seed)[0].pair;
  if (prefer === "worst") return scored.sort((a, b) => a.delta - b.delta || a.pair.left.seed - b.pair.left.seed)[0].pair;
  return scored.sort((a, b) => (
    Math.abs(a.delta - targetMean) - Math.abs(b.delta - targetMean)
    || a.pair.left.seed - b.pair.left.seed
  ))[0].pair;
}

function makeEnvelopeRows(trialRows, args) {
  const byCell = groupBy(trialRows, (row) => row.cell_id);
  const envelope = [];
  for (const [cellId, cellRows] of byCell.entries()) {
    const first = cellRows[0];
    const byMode = groupBy(cellRows, (row) => row.mode);
    for (const mode of [...CANDIDATE_MODES, ...LINEAGE_MODES]) {
      const targetRows = byMode.get(mode) ?? [];
      if (targetRows.length === 0) continue;
      const naivePairs = pairRows(cellRows, mode, "naive_pressure");
      const randomPairs = pairRows(cellRows, mode, "random_reveal");
      const naiveDeltas = naivePairs.map(({ left, right }) => left.budget_adjusted_safe_tiles - right.budget_adjusted_safe_tiles);
      const randomDeltas = randomPairs.map(({ left, right }) => left.budget_adjusted_safe_tiles - right.budget_adjusted_safe_tiles);
      const falseFlagDeltas = naivePairs.map(({ left, right }) => left.false_flag_count - right.false_flag_count);
      const mineTriggerDeltas = naivePairs.map(({ left, right }) => (left.mine_triggered ? 1 : 0) - (right.mine_triggered ? 1 : 0));
      const ci = bootstrapMean(naiveDeltas, args.bootstrapIterations, `${cellId}:${mode}:budget-vs-naive`);
      const targetSummary = summarizeMode(targetRows);
      const naiveSummary = summarizeMode(byMode.get("naive_pressure") ?? []);
      const randomSummary = summarizeMode(byMode.get("random_reveal") ?? []);
      const thresholdSummary = summarizeMode(byMode.get("threshold_flagger") ?? []);
      const oracleSummary = summarizeMode(byMode.get("oracle_safe") ?? []);
      const staticBoundary = assessStaticBoundary({
        boardConfig: {
          preset: REPLAY_PRESET,
          width: targetRows[0].width,
          height: targetRows[0].height,
          mineCount: targetRows[0].mine_count,
          scanBudget: targetRows[0].scan_budget,
          generator: { clusterStrength: targetRows[0].cluster_strength },
        },
        sensorConfig: {
          sigma: targetRows[0].sigma,
          sigmaNoise: targetRows[0].sigma_noise,
          dropoutRate: targetRows[0].dropout_rate,
          delaySteps: targetRows[0].delay_steps,
        },
        mode,
      });
      const naiveDelta = mean(naiveDeltas);
      const randomDelta = mean(randomDeltas);
      const falseFlagDelta = mean(falseFlagDeltas);
      const mineTriggerDelta = mean(mineTriggerDeltas);
      const gateBudgetNaive = naiveDelta >= 1.0;
      const gateBudgetRandom = randomDelta >= 0;
      const gateStatic = staticBoundary.status !== "do_not_use";
      const gateFalseFlags = falseFlagDelta <= 1.0;
      const gateMineTriggers = mineTriggerDelta <= 0.10;
      const gateBootstrap = Number.isFinite(ci.low) && ci.low > 0;
      const gradedCandidate = CANDIDATE_MODES.includes(mode);
      const candidate = gradedCandidate
        && gateBudgetNaive
        && gateBudgetRandom
        && gateStatic
        && gateFalseFlags
        && gateMineTriggers
        && gateBootstrap;
      const failureRegime = gradedCandidate && naiveDelta <= -1.0;
      const representative = representativePair(
        naivePairs,
        naiveDelta,
        candidate ? "best" : failureRegime ? "worst" : "closest",
      );
      const mechanismCodes = empiricalMechanisms({
        staticCodes: staticBoundary.mechanismCodes,
        naiveDelta,
        falseFlagDelta,
        mineTriggerDelta,
        randomDelta,
      });
      const cellClass = candidate
        ? "candidate_positive"
        : failureRegime
          ? "failure_regime"
          : staticBoundary.status === "do_not_use"
            ? "static_boundary"
            : staticBoundary.status === "watch_boundary"
              ? "watch_boundary"
              : "neutral";

      envelope.push({
        phase: args.phase,
        cell_id: cellId,
        grid_kind: first.grid_kind,
        axis: first.axis,
        axis_value: first.axis_value,
        label: first.label,
        mode,
        graded_candidate: gradedCandidate,
        cell_class: cellClass,
        candidate,
        failure_regime: failureRegime,
        width: first.width,
        height: first.height,
        mine_count: first.mine_count,
        mine_density: first.mine_density,
        cluster_strength: first.cluster_strength,
        scan_budget: first.scan_budget,
        configured_scan_budget: first.configured_scan_budget,
        sigma: first.sigma,
        sigma_noise: first.sigma_noise,
        dropout_rate: first.dropout_rate,
        delay_steps: first.delay_steps,
        seed_count: naivePairs.length,
        target_survival_rate: targetSummary.survival_rate,
        naive_survival_rate: naiveSummary.survival_rate,
        random_survival_rate: randomSummary.survival_rate,
        threshold_survival_rate: thresholdSummary.survival_rate,
        oracle_survival_rate: oracleSummary.survival_rate,
        target_mine_trigger_rate: targetSummary.mine_trigger_rate,
        naive_mine_trigger_rate: naiveSummary.mine_trigger_rate,
        target_budget_adjusted_safe_tiles_mean: targetSummary.mean_budget_adjusted_safe_tiles,
        naive_budget_adjusted_safe_tiles_mean: naiveSummary.mean_budget_adjusted_safe_tiles,
        random_budget_adjusted_safe_tiles_mean: randomSummary.mean_budget_adjusted_safe_tiles,
        threshold_budget_adjusted_safe_tiles_mean: thresholdSummary.mean_budget_adjusted_safe_tiles,
        oracle_budget_adjusted_safe_tiles_mean: oracleSummary.mean_budget_adjusted_safe_tiles,
        target_false_flag_count_mean: targetSummary.mean_false_flag_count,
        naive_false_flag_count_mean: naiveSummary.mean_false_flag_count,
        budget_delta_vs_naive_mean: roundMetric(naiveDelta),
        budget_delta_vs_random_mean: roundMetric(randomDelta),
        false_flag_delta_vs_naive_mean: roundMetric(falseFlagDelta),
        mine_trigger_rate_delta_vs_naive: roundMetric(mineTriggerDelta),
        budget_delta_vs_naive_ci_low: roundMetric(ci.low),
        budget_delta_vs_naive_ci_high: roundMetric(ci.high),
        gate_budget_vs_naive: gateBudgetNaive,
        gate_budget_vs_random: gateBudgetRandom,
        gate_static_boundary: gateStatic,
        gate_false_flags: gateFalseFlags,
        gate_mine_triggers: gateMineTriggers,
        gate_bootstrap: gateBootstrap,
        static_boundary_status: staticBoundary.status,
        static_boundary_mechanisms: staticBoundary.mechanismCodes.join("|"),
        mechanism_codes: mechanismCodes,
        representative_seed: representative?.left.seed ?? "",
        replay_url: representative?.left.replay_url ?? targetRows[0]?.replay_url ?? "",
        harness_replay_command: representative?.left.harness_replay_command ?? targetRows[0]?.harness_replay_command ?? "",
      });
    }
  }
  return envelope.sort((a, b) => (
    a.mode.localeCompare(b.mode)
    || b.candidate - a.candidate
    || a.cell_class.localeCompare(b.cell_class)
    || a.cell_id.localeCompare(b.cell_id)
  ));
}

function makeCellManifestRows(cells) {
  return cells.map((cell) => {
    const board = normalizeMinesConfig(cell.board);
    const sensor = normalizeSensorConfig(cell.sensor);
    const boundary = assessStaticBoundary({
      boardConfig: board,
      sensorConfig: sensor,
      mode: "sundog_lean",
    });
    return {
      cell_id: cell.cellId,
      grid_kind: cell.gridKind,
      axis: cell.axis,
      axis_value: cell.axisValue,
      label: cell.label,
      preset: REPLAY_PRESET,
      sensor_cell: REPLAY_SENSOR_CELL,
      width: board.width,
      height: board.height,
      mine_count: board.mineCount,
      mine_density: roundMetric(board.mineCount / (board.width * board.height)),
      cluster_strength: roundMetric(board.generator?.clusterStrength ?? 0),
      scan_budget: board.scanBudget,
      sigma: sensor.sigma,
      sigma_noise: sensor.sigmaNoise,
      dropout_rate: sensor.dropoutRate,
      delay_steps: sensor.delaySteps,
      kernel_family: sensor.kernel,
      static_boundary_status: boundary.status,
      static_boundary_mechanisms: boundary.mechanismCodes.join("|"),
    };
  }).sort((a, b) => a.cell_id.localeCompare(b.cell_id));
}

function makeBestWorstRows(envelopeRows) {
  const candidates = envelopeRows.filter((row) => row.candidate);
  const failures = envelopeRows.filter((row) => row.failure_regime);
  const best = [...candidates].sort((a, b) => (
    b.budget_delta_vs_naive_mean - a.budget_delta_vs_naive_mean
    || a.mode.localeCompare(b.mode)
    || a.cell_id.localeCompare(b.cell_id)
  ))[0] ?? null;
  const worst = [...failures].sort((a, b) => (
    a.budget_delta_vs_naive_mean - b.budget_delta_vs_naive_mean
    || a.mode.localeCompare(b.mode)
    || a.cell_id.localeCompare(b.cell_id)
  ))[0] ?? null;
  return [
    worst ? { selection: "worst_cell", rule: "lowest candidate-mode budget-adjusted delta versus naive_pressure among failure_regime cells", ...worst } : null,
    best ? { selection: "best_cell", rule: "highest candidate-mode budget-adjusted delta versus naive_pressure among candidate cells", ...best } : null,
  ].filter(Boolean);
}

function makeCellClassRows(envelopeRows) {
  return envelopeRows
    .filter((row) => row.graded_candidate)
    .map((row) => ({
      cell_id: row.cell_id,
      grid_kind: row.grid_kind,
      axis: row.axis,
      axis_value: row.axis_value,
      mode: row.mode,
      cell_class: row.cell_class,
      candidate: row.candidate,
      failure_regime: row.failure_regime,
      budget_delta_vs_naive_mean: row.budget_delta_vs_naive_mean,
      budget_delta_vs_naive_ci_low: row.budget_delta_vs_naive_ci_low,
      false_flag_delta_vs_naive_mean: row.false_flag_delta_vs_naive_mean,
      mine_trigger_rate_delta_vs_naive: row.mine_trigger_rate_delta_vs_naive,
      static_boundary_status: row.static_boundary_status,
      mechanism_codes: row.mechanism_codes,
      replay_url: row.replay_url,
    }));
}

function candidateDominance(envelopeRows) {
  const byCell = groupBy(envelopeRows.filter((row) => CANDIDATE_MODES.includes(row.mode)), (row) => row.cell_id);
  let comparable = 0;
  let leanDominates = 0;
  const losses = [];
  for (const rows of byCell.values()) {
    const lean = rows.find((row) => row.mode === "sundog_lean");
    const minimal = rows.find((row) => row.mode === "sundog_minimal");
    if (!lean || !minimal) continue;
    comparable += 1;
    const dominates = lean.budget_delta_vs_naive_mean >= minimal.budget_delta_vs_naive_mean
      && (lean.candidate || !minimal.candidate);
    if (dominates) leanDominates += 1;
    else losses.push({
      cell_id: lean.cell_id,
      lean_delta: lean.budget_delta_vs_naive_mean,
      minimal_delta: minimal.budget_delta_vs_naive_mean,
      lean_candidate: lean.candidate,
      minimal_candidate: minimal.candidate,
    });
  }
  return {
    comparable_cells: comparable,
    lean_dominates_cells: leanDominates,
    lean_dominates_all: comparable > 0 && leanDominates === comparable,
    minimal_retirement_recommended: comparable > 0 && leanDominates === comparable,
    exceptions: losses.slice(0, 12),
  };
}

function makeVerdict(envelopeRows, trialRows, bestWorstRows, args) {
  const candidateRows = envelopeRows.filter((row) => row.candidate);
  const failureRows = envelopeRows.filter((row) => row.failure_regime);
  const contradictionRows = candidateRows.filter((row) => (
    row.static_boundary_status === "do_not_use"
    || row.mechanism_codes.split("|").includes("controller_overcommitted")
    || row.mechanism_codes.split("|").includes("overflagged")
  ));
  const promotion = candidateRows.length > 0
    && failureRows.length > 0
    && contradictionRows.length === 0;
  const byMode = groupBy(candidateRows, (row) => row.mode);
  const verdict = promotion ? "CONFIRM" : "NO_ENVELOPE";
  const dominance = candidateDominance(envelopeRows);
  const reasons = [];
  if (candidateRows.length === 0) reasons.push("No candidate cell passed all pre-registered gates.");
  if (failureRows.length === 0) reasons.push("No paired failure-regime cell was available for publication.");
  if (contradictionRows.length > 0) reasons.push(`${contradictionRows.length} candidate rows carried contradictory empirical/static labels.`);
  if (promotion) {
    const bestCell = bestWorstRows.find((row) => row.selection === "best_cell") ?? null;
    const worstCell = bestWorstRows.find((row) => row.selection === "worst_cell") ?? null;
    reasons.push(`${candidateRows.length} candidate rows passed all pre-registered gates, with ${failureRows.length} paired failure-regime rows available for negative-region publication.`);
    if (bestCell) {
      const sameCellCandidates = candidateRows
        .filter((row) => row.cell_id === bestCell.cell_id)
        .sort((a, b) => b.budget_delta_vs_naive_mean - a.budget_delta_vs_naive_mean)
        .map((row) => `${row.mode} delta ${row.budget_delta_vs_naive_mean} CI [${row.budget_delta_vs_naive_ci_low}, ${row.budget_delta_vs_naive_ci_high}]`)
        .join("; ");
      reasons.push(`Confirmed pocket: density ${bestCell.mine_density}, pressure noise ${bestCell.sigma_noise}, dropout ${bestCell.dropout_rate}; ${sameCellCandidates}.`);
      reasons.push(`Caveat: in the best cell, target and naive mine-trigger rates are ${bestCell.target_mine_trigger_rate} and ${bestCell.naive_mine_trigger_rate}; the confirmed outcome is safe-tile progress before mine trigger, not field clearance.`);
      reasons.push(`Comparator caveat: threshold_flagger survival is ${bestCell.threshold_survival_rate} in the best cell, so the promoted comparator claim remains against naive_pressure rather than survival dominance over threshold_flagger.`);
      if (bestCell.static_boundary_status === "watch_boundary") {
        reasons.push(`Boundary caveat: the confirmed cell is watch_boundary with static mechanisms ${bestCell.static_boundary_mechanisms}; the pocket is provisional and sits near the field-informativeness caution edge.`);
      }
    }
    if (worstCell) {
      reasons.push(`Mapped failure: ${worstCell.mode} at density ${worstCell.mine_density}, pressure noise ${worstCell.sigma_noise}, dropout ${worstCell.dropout_rate} lost ${worstCell.budget_delta_vs_naive_mean} budget-adjusted safe tiles versus naive_pressure, CI [${worstCell.budget_delta_vs_naive_ci_low}, ${worstCell.budget_delta_vs_naive_ci_high}], mechanisms ${worstCell.mechanism_codes}.`);
    }
    reasons.push(`Variant caveat: lean_dominates_all=${dominance.lean_dominates_all} and minimal_retirement_recommended=${dominance.minimal_retirement_recommended}; keep both sundog_lean and sundog_minimal visible until a later controller redesign retires one.`);
  }
  return {
    schema: "sundog.mines.phase10-verdict.v1",
    generatedAt: new Date().toISOString(),
    phase: args.phase,
    verdict,
    disposition: promotion
      ? "Promote Pressure Mines to Operating-Envelope Study tier after Phase 11 public polish."
      : "Keep Pressure Mines at Planned Workbench tier and publish the mapped boundary rather than an envelope claim.",
    gates: {
      budget_delta_vs_naive_min: 1.0,
      budget_delta_vs_random_min: 0,
      false_flag_delta_vs_naive_max: 1.0,
      mine_trigger_rate_delta_vs_naive_max: 0.10,
      bootstrap_ci_low_must_exceed: 0,
      bootstrap_iterations: args.bootstrapIterations,
    },
    cell_counts: {
      envelope_rows: envelopeRows.length,
      candidate_rows: candidateRows.length,
      failure_rows: failureRows.length,
      trial_rows: trialRows.length,
    },
    candidates_by_mode: Object.fromEntries(CANDIDATE_MODES.map((mode) => [mode, (byMode.get(mode) ?? []).length])),
    contradiction_count: contradictionRows.length,
    contradiction_rows: contradictionRows.slice(0, 12),
    dominance,
    bestCell: bestWorstRows.find((row) => row.selection === "best_cell") ?? null,
    worstCell: bestWorstRows.find((row) => row.selection === "worst_cell") ?? null,
    reasons,
  };
}

function verdictMarkdown(verdict, args) {
  return [
    "# Sundog Pressure Mines Phase 10 Verdict",
    "",
    `Generated: ${verdict.generatedAt}`,
    `Phase: ${args.phase}`,
    `Verdict: ${verdict.verdict}`,
    `Disposition: ${verdict.disposition}`,
    "",
    "## Candidate Gates",
    "",
    `Candidate rows: ${verdict.cell_counts.candidate_rows}`,
    `Failure-regime rows: ${verdict.cell_counts.failure_rows}`,
    `Contradiction rows: ${verdict.contradiction_count}`,
    `Candidates by mode: ${JSON.stringify(verdict.candidates_by_mode)}`,
    "",
    "## Best / Worst",
    "",
    verdict.worstCell
      ? `Worst: ${verdict.worstCell.mode} ${verdict.worstCell.axis}=${verdict.worstCell.axis_value}, delta ${verdict.worstCell.budget_delta_vs_naive_mean}, ${verdict.worstCell.replay_url}`
      : "Worst: none",
    verdict.bestCell
      ? `Best: ${verdict.bestCell.mode} ${verdict.bestCell.axis}=${verdict.bestCell.axis_value}, delta ${verdict.bestCell.budget_delta_vs_naive_mean}, ${verdict.bestCell.replay_url}`
      : "Best: none",
    "",
    "## Dominance",
    "",
    `Lean dominates all comparable cells: ${verdict.dominance.lean_dominates_all}`,
    `Minimal retirement recommended: ${verdict.dominance.minimal_retirement_recommended}`,
    "",
    "## Reasons",
    "",
    ...(verdict.reasons.length > 0 ? verdict.reasons.map((reason) => `- ${reason}`) : ["- All pre-registered promotion gates held."]),
    "",
  ].join("\n");
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const normalized = typeof value === "number" ? roundMetric(value) : value;
  const text = String(normalized);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function rowsToCsv(rows, columns = null) {
  if (rows.length === 0 && !columns) return "";
  const headers = columns ?? Object.keys(rows[0]);
  const lines = [headers.join(",")];
  for (const row of rows) lines.push(headers.map((header) => csvEscape(row[header])).join(","));
  return `${lines.join("\n")}\n`;
}

function runSelfChecks(args, cells, trialRows) {
  const expected = cells.length * args.modes.length * args.seeds;
  if (trialRows.length !== expected) {
    throw new Error(`Expected ${expected} trial rows, got ${trialRows.length}`);
  }
  for (const row of trialRows) {
    if (row.budget_adjusted_safe_tiles > row.raw_safe_tiles) {
      throw new Error(`Budget-adjusted safe tiles exceeded raw safe tiles for ${row.trial_id}`);
    }
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const cells = makeEnvelopeCells();
  const trialRows = [];

  for (const cell of cells) {
    for (let i = 0; i < args.seeds; i += 1) {
      const seed = args.seedStart + i;
      for (const mode of args.modes) {
        trialRows.push(runTrial(args, { cell, mode, seed }));
      }
    }
  }

  runSelfChecks(args, cells, trialRows);
  const cellRows = makeCellManifestRows(cells);
  const comparisonRows = makeMatchedComparisons(trialRows);
  const envelopeRows = makeEnvelopeRows(trialRows, args);
  const cellClassRows = makeCellClassRows(envelopeRows);
  const bestWorstRows = makeBestWorstRows(envelopeRows);
  const verdict = makeVerdict(envelopeRows, trialRows, bestWorstRows, args);
  const manifest = {
    schema: "sundog.mines.phase10-envelope.v1",
    generatedAt: new Date().toISOString(),
    phase: args.phase,
    seedStart: args.seedStart,
    seeds: args.seeds,
    turnCap: args.turnCap,
    pressureThreshold: args.pressureThreshold,
    bootstrapIterations: args.bootstrapIterations,
    modes: args.modes,
    candidateModes: CANDIDATE_MODES,
    lineageModes: LINEAGE_MODES,
    grid: PRE_REGISTERED_GRID,
    cellCount: cells.length,
    trialCount: trialRows.length,
    envelopeRows: envelopeRows.length,
    verdict: verdict.verdict,
    note: "Ignored local Phase 10 operating-envelope output. Verdict surfaces are pre-registered in docs/sundog_v_minesweeper.md.",
  };

  const outDir = path.resolve(repoRoot, args.out);
  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  await writeFile(path.join(outDir, "summary.json"), `${JSON.stringify({ ...manifest, verdict }, null, 2)}\n`);
  await writeFile(path.join(outDir, "verdict.json"), `${JSON.stringify(verdict, null, 2)}\n`);
  await writeFile(path.join(outDir, "verdict.md"), verdictMarkdown(verdict, args));
  await writeFile(path.join(outDir, "cell-manifest.csv"), rowsToCsv(cellRows));
  await writeFile(path.join(outDir, "trial-outcomes.csv"), rowsToCsv(trialRows));
  await writeFile(path.join(outDir, "matched-comparison.csv"), rowsToCsv(comparisonRows));
  await writeFile(path.join(outDir, "envelope.csv"), rowsToCsv(envelopeRows));
  await writeFile(path.join(outDir, "cell-class-map.csv"), rowsToCsv(cellClassRows));
  await writeFile(path.join(outDir, "best-worst-cells.csv"), rowsToCsv(bestWorstRows));

  console.log(`Mines ${args.phase}: ${trialRows.length} trials across ${cells.length} envelope cells`);
  console.log(`Wrote ${path.relative(repoRoot, outDir)}`);
  console.log(`Verdict: ${verdict.verdict}; candidates ${verdict.cell_counts.candidate_rows}; failures ${verdict.cell_counts.failure_rows}`);
  if (verdict.bestCell) {
    console.log(`Best: ${verdict.bestCell.mode} ${verdict.bestCell.axis}=${verdict.bestCell.axis_value}, delta ${verdict.bestCell.budget_delta_vs_naive_mean}`);
  }
  if (verdict.worstCell) {
    console.log(`Worst: ${verdict.worstCell.mode} ${verdict.worstCell.axis}=${verdict.worstCell.axis_value}, delta ${verdict.worstCell.budget_delta_vs_naive_mean}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
