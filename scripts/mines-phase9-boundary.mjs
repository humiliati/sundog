import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  ACTION,
  applyMinesAction,
  getPublicMemory,
  initializeBoardState,
  makeRng,
  MINES_PRESETS,
  normalizeMinesConfig,
} from "../public/js/mines-core.mjs";
import { createSensorRuntime, normalizeSensorConfig } from "../public/js/mines-sensor.mjs";
import {
  IMPLEMENTED_MINES_MODES,
  MINES_CONTROLLER_MODES,
  chooseMinesAction,
  frontierIndices,
} from "../public/js/mines-controllers.mjs";
import { assessMinesBoundary } from "../public/js/mines-boundary.mjs";

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

const DEFAULT_AXES = Object.freeze([
  "preset",
  "mine_density",
  "clustering_strength",
  "pressure_noise",
  "sensor_delay",
  "scan_dropout",
  "kernel_blur",
  "board_size",
  "scan_budget",
]);

const TARGET_MODES = Object.freeze(["sundog_lean", "sundog_minimal", "sundog_controller"]);
const BASELINE_MODES = Object.freeze(["random_reveal", "naive_pressure", "threshold_flagger", "oracle_safe"]);

const BASE_BOARD = Object.freeze({
  preset: "easy_sparse",
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

function parseList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function parseArgs(argv) {
  const args = {
    phase: "phase9-boundary",
    out: "results/mines/phase9-boundary",
    seedStart: 0,
    seeds: 8,
    turnCap: 160,
    pressureThreshold: 1.2,
    modes: [...DEFAULT_MODES],
    axes: [...DEFAULT_AXES],
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
    else if (flag === "--axes") args.axes = parseList(value);
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
  if (!Number.isFinite(args.pressureThreshold)) {
    throw new Error("--pressure-threshold must be numeric");
  }
  for (const mode of args.modes) {
    if (!MINES_CONTROLLER_MODES[mode]) throw new Error(`Unknown mines mode: ${mode}`);
    if (!IMPLEMENTED_MINES_MODES.includes(mode)) {
      throw new Error(`Mode ${mode} is declared but not implemented yet`);
    }
  }
  for (const axis of args.axes) {
    if (!DEFAULT_AXES.includes(axis)) throw new Error(`Unknown Phase 9 axis: ${axis}`);
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

function primaryMechanism(assessment) {
  return assessment.mechanisms.find((mechanism) => mechanism.severity === "unsafe")?.code
    ?? assessment.mechanisms[0]?.code
    ?? "none";
}

function boundaryCellId(axis, axisValue, board, sensor) {
  return [
    axis,
    safeId(axisValue),
    `w${board.width}`,
    `h${board.height}`,
    `m${board.mineCount}`,
    `c${safeId(board.generator?.clusterStrength ?? 0)}`,
    `s${safeId(sensor.sigma)}`,
    `n${safeId(sensor.sigmaNoise)}`,
    `d${safeId(sensor.dropoutRate)}`,
    `lag${safeId(sensor.delaySteps)}`,
    `scan${safeId(board.scanBudget)}`,
  ].join("__");
}

function addBoundaryCell(cells, axis, axisValue, { preset = BASE_BOARD.preset, board = {}, sensor = {}, label = "" } = {}) {
  const mergedBoard = {
    ...BASE_BOARD,
    preset,
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
  const normalizedBoard = normalizeMinesConfig(mergedBoard);
  const normalizedSensor = normalizeSensorConfig(mergedSensor);
  cells.push(Object.freeze({
    cellId: boundaryCellId(axis, axisValue, normalizedBoard, normalizedSensor),
    axis,
    axisValue,
    preset,
    label: label || `${axis}=${axisValue}`,
    board: Object.freeze(mergedBoard),
    sensor: normalizedSensor,
  }));
}

function addPresetCell(cells, preset) {
  const definition = MINES_PRESETS[preset];
  addBoundaryCell(cells, "preset", preset, {
    preset,
    board: {
      ...definition.config,
      generator: definition.generator ?? { clusterStrength: 0 },
    },
    label: definition.label,
  });
}

function makeBoundaryCells(args) {
  const cells = [];
  const wants = (axis) => args.axes.includes(axis);

  if (wants("preset")) {
    addPresetCell(cells, "easy_sparse");
    addPresetCell(cells, "clustered");
    addPresetCell(cells, "dense");
  }

  if (wants("mine_density")) {
    for (const value of [0.1, 0.16, 0.22, 0.28]) {
      addBoundaryCell(cells, "mine_density", value, {
        board: { mineCount: Math.round(9 * 9 * value) },
      });
    }
  }

  if (wants("clustering_strength")) {
    for (const value of [0, 0.35, 0.65, 0.9]) {
      addBoundaryCell(cells, "clustering_strength", value, {
        preset: "clustered",
        board: {
          width: 12,
          height: 12,
          mineCount: 24,
          scanBudget: 4,
          generator: { clusterStrength: value },
        },
      });
    }
  }

  if (wants("pressure_noise")) {
    for (const value of [0.1, 1, 5, 10]) {
      addBoundaryCell(cells, "pressure_noise", value, {
        sensor: { sigmaNoise: value },
      });
    }
  }

  if (wants("sensor_delay")) {
    for (const value of [0, 1, 2, 4]) {
      addBoundaryCell(cells, "sensor_delay", value, {
        sensor: { delaySteps: value },
      });
    }
  }

  if (wants("scan_dropout")) {
    for (const value of [0.1, 0.35, 0.65, 0.85]) {
      addBoundaryCell(cells, "scan_dropout", value, {
        sensor: { dropoutRate: value },
      });
    }
  }

  if (wants("kernel_blur")) {
    for (const value of [1, 2.5, 5, 8]) {
      addBoundaryCell(cells, "kernel_blur", value, {
        sensor: { sigma: value },
      });
    }
  }

  if (wants("board_size")) {
    for (const cell of [
      { axisValue: "9x9", width: 9, height: 9, mineCount: 10 },
      { axisValue: "12x12", width: 12, height: 12, mineCount: 18 },
      { axisValue: "16x16", width: 16, height: 16, mineCount: 32 },
    ]) {
      addBoundaryCell(cells, "board_size", cell.axisValue, {
        board: {
          width: cell.width,
          height: cell.height,
          mineCount: cell.mineCount,
          scanBudget: cell.width >= 16 ? 6 : 3,
        },
      });
    }
  }

  if (wants("scan_budget")) {
    for (const value of [0, 1, 3, 6]) {
      addBoundaryCell(cells, "scan_budget", value, {
        board: { scanBudget: value },
      });
    }
  }

  return cells;
}

function sensorConfigFor(cell, mode, seed) {
  const definition = MINES_CONTROLLER_MODES[mode];
  return normalizeSensorConfig({
    ...cell.sensor,
    ...(definition.sensorOverride ?? {}),
    sensorSeed: seed + 7919 + stringHash(mode) + stringHash(cell.cellId),
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

function runBoundaryTrial(args, { cell, mode, seed }) {
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
  const rng = makeRng(seed ^ stringHash(mode) ^ stringHash(cell.cellId));
  let illegalActionCount = 0;
  let sensor = sensorRuntime.step(board);

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
  const boundary = assessMinesBoundary({
    boardConfig: board.config,
    sensorConfig,
    mode,
    live: {
      ...frontierStats,
      falseFlagCount: board.falseFlagCount,
      terminal: null,
    },
  });
  const revealCount = countLedger(board, ACTION.REVEAL);
  const flagCount = countLedger(board, ACTION.FLAG);
  const scanCount = countLedger(board, ACTION.SCAN);
  const rawSafeTiles = board.revealedSafeCount;
  const budgetAdjustedSafeTiles = rawSafeTiles - scanCount;
  const actionTraceHash = hashHex(JSON.stringify(normalizedActionTrace(board.actionLedger)));

  return {
    phase: args.phase,
    cellId: cell.cellId,
    trialId: trialIdOf({ phase: args.phase, cell, mode, seed }),
    axis: cell.axis,
    axisValue: cell.axisValue,
    preset: cell.preset,
    mode,
    seed,
    actionTraceHash,
    modeStatus: modeDefinition.status,
    usesPrivileged: modeDefinition.usesPrivileged,
    usesScan: modeDefinition.usesScan,
    width: board.config.width,
    height: board.config.height,
    mineCount: board.config.mineCount,
    mineDensity: roundMetric(board.config.mineCount / (board.config.width * board.config.height)),
    clusterStrength: roundMetric(board.config.generator?.clusterStrength ?? 0),
    scanBudget: board.config.scanBudget,
    turnCap: args.turnCap,
    terminal: terminalClass(board.terminal),
    survived: board.terminal !== "mine_triggered",
    fullClear: board.terminal === "full_clear",
    turns: board.turn,
    controllerTurns: Math.max(0, board.turn - 1),
    revealCount,
    flagCount,
    scanCount,
    illegalActionCount,
    openingSafeCount,
    rawSafeTiles,
    safeTilesAfterOpening: Math.max(0, rawSafeTiles - openingSafeCount),
    budgetAdjustedSafeTiles,
    falseFlagCount: board.falseFlagCount,
    frontierSize: frontierStats.frontierSize,
    meanFrontierConfidence: roundMetric(frontierStats.meanFrontierConfidence),
    pressureThreshold: args.pressureThreshold,
    sigma: sensorConfig.sigma,
    sigmaNoise: sensorConfig.sigmaNoise,
    dropoutRate: sensorConfig.dropoutRate,
    delaySteps: sensorConfig.delaySteps,
    kernelFamily: sensorConfig.kernel,
    boundaryStatus: boundary.status,
    boundaryPrimaryMechanism: primaryMechanism(boundary),
    boundaryMechanisms: boundary.mechanisms.map((mechanism) => mechanism.code).join("|"),
  };
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

function summarizeTrials(rows) {
  const groups = groupBy(rows, (row) => [
    row.phase,
    row.cellId,
    row.axis,
    row.axisValue,
    row.preset,
    row.mode,
  ].join("|"));

  return Array.from(groups.entries()).map(([key, group]) => {
    const [phase, cellId, axis, axisValue, preset, mode] = key.split("|");
    const first = group[0];
    return {
      phase,
      cellId,
      axis,
      axisValue,
      preset,
      mode,
      n: group.length,
      usesPrivileged: first.usesPrivileged,
      usesScan: first.usesScan,
      width: first.width,
      height: first.height,
      mineCount: first.mineCount,
      mineDensity: first.mineDensity,
      clusterStrength: first.clusterStrength,
      scanBudget: first.scanBudget,
      sigma: first.sigma,
      sigmaNoise: first.sigmaNoise,
      dropoutRate: first.dropoutRate,
      delaySteps: first.delaySteps,
      kernelFamily: first.kernelFamily,
      survivalRate: roundMetric(rate(group, (row) => row.survived)),
      fullClearRate: roundMetric(rate(group, (row) => row.fullClear)),
      mineTriggerRate: roundMetric(rate(group, (row) => row.terminal === "mine_triggered")),
      meanRawSafeTiles: roundMetric(mean(group.map((row) => row.rawSafeTiles))),
      meanSafeTilesAfterOpening: roundMetric(mean(group.map((row) => row.safeTilesAfterOpening))),
      meanBudgetAdjustedSafeTiles: roundMetric(mean(group.map((row) => row.budgetAdjustedSafeTiles))),
      meanFalseFlagCount: roundMetric(mean(group.map((row) => row.falseFlagCount))),
      meanScanCount: roundMetric(mean(group.map((row) => row.scanCount))),
      meanTurns: roundMetric(mean(group.map((row) => row.turns))),
      meanFrontierConfidence: roundMetric(mean(group.map((row) => row.meanFrontierConfidence))),
      illegalActionCount: sum(group.map((row) => row.illegalActionCount)),
      doNotUseRate: roundMetric(rate(group, (row) => row.boundaryStatus === "do_not_use")),
      watchBoundaryRate: roundMetric(rate(group, (row) => row.boundaryStatus === "watch_boundary")),
      primaryFailureMechanism: mostCommon(group
        .map((row) => row.boundaryPrimaryMechanism)
        .filter((value) => value !== "none")),
    };
  }).sort((a, b) => (
    a.axis.localeCompare(b.axis)
    || String(a.axisValue).localeCompare(String(b.axisValue), undefined, { numeric: true })
    || a.mode.localeCompare(b.mode)
  ));
}

function makeComparisonRows(trialRows) {
  const groups = groupBy(trialRows, (row) => `${row.cellId}\t${row.seed}`);
  const comparisons = [];
  for (const group of groups.values()) {
    const byMode = new Map(group.map((row) => [row.mode, row]));
    for (const mode of TARGET_MODES) {
      const target = byMode.get(mode);
      if (!target) continue;
      for (const baselineMode of BASELINE_MODES) {
        if (baselineMode === mode) continue;
        const baseline = byMode.get(baselineMode);
        if (!baseline) continue;
        comparisons.push({
          phase: target.phase,
          cellId: target.cellId,
          axis: target.axis,
          axisValue: target.axisValue,
          preset: target.preset,
          seed: target.seed,
          mode,
          baselineMode,
          width: target.width,
          height: target.height,
          mineCount: target.mineCount,
          mineDensity: target.mineDensity,
          clusterStrength: target.clusterStrength,
          scanBudget: target.scanBudget,
          sigma: target.sigma,
          sigmaNoise: target.sigmaNoise,
          dropoutRate: target.dropoutRate,
          delaySteps: target.delaySteps,
          targetTerminal: target.terminal,
          baselineTerminal: baseline.terminal,
          rawSafeTilesDelta: target.rawSafeTiles - baseline.rawSafeTiles,
          safeTilesAfterOpeningDelta: target.safeTilesAfterOpening - baseline.safeTilesAfterOpening,
          budgetAdjustedSafeTilesDelta: target.budgetAdjustedSafeTiles - baseline.budgetAdjustedSafeTiles,
          survivalDelta: (target.survived ? 1 : 0) - (baseline.survived ? 1 : 0),
          mineTriggerDelta: (target.terminal === "mine_triggered" ? 1 : 0) - (baseline.terminal === "mine_triggered" ? 1 : 0),
          falseFlagDelta: target.falseFlagCount - baseline.falseFlagCount,
          scanDelta: target.scanCount - baseline.scanCount,
          targetBoundaryStatus: target.boundaryStatus,
          targetFailureMechanism: target.boundaryPrimaryMechanism,
        });
      }
    }
  }
  return comparisons.sort((a, b) => (
    a.axis.localeCompare(b.axis)
    || String(a.axisValue).localeCompare(String(b.axisValue), undefined, { numeric: true })
    || a.seed - b.seed
    || a.mode.localeCompare(b.mode)
    || a.baselineMode.localeCompare(b.baselineMode)
  ));
}

function summarizeComparisons(rows) {
  const groups = groupBy(rows, (row) => [
    row.phase,
    row.cellId,
    row.axis,
    row.axisValue,
    row.preset,
    row.mode,
    row.baselineMode,
  ].join("|"));

  return Array.from(groups.entries()).map(([key, group]) => {
    const [phase, cellId, axis, axisValue, preset, mode, baselineMode] = key.split("|");
    const first = group[0];
    return {
      phase,
      cellId,
      axis,
      axisValue,
      preset,
      mode,
      baselineMode,
      n: group.length,
      width: first.width,
      height: first.height,
      mineCount: first.mineCount,
      mineDensity: first.mineDensity,
      clusterStrength: first.clusterStrength,
      scanBudget: first.scanBudget,
      sigma: first.sigma,
      sigmaNoise: first.sigmaNoise,
      dropoutRate: first.dropoutRate,
      delaySteps: first.delaySteps,
      meanRawSafeTilesDelta: roundMetric(mean(group.map((row) => row.rawSafeTilesDelta))),
      meanSafeTilesAfterOpeningDelta: roundMetric(mean(group.map((row) => row.safeTilesAfterOpeningDelta))),
      meanBudgetAdjustedSafeTilesDelta: roundMetric(mean(group.map((row) => row.budgetAdjustedSafeTilesDelta))),
      meanSurvivalDelta: roundMetric(mean(group.map((row) => row.survivalDelta))),
      meanMineTriggerDelta: roundMetric(mean(group.map((row) => row.mineTriggerDelta))),
      meanFalseFlagDelta: roundMetric(mean(group.map((row) => row.falseFlagDelta))),
      meanScanDelta: roundMetric(mean(group.map((row) => row.scanDelta))),
      doNotUseRate: roundMetric(rate(group, (row) => row.targetBoundaryStatus === "do_not_use")),
      primaryFailureMechanism: mostCommon(group
        .map((row) => row.targetFailureMechanism)
        .filter((value) => value !== "none")),
    };
  }).sort((a, b) => (
    a.axis.localeCompare(b.axis)
    || String(a.axisValue).localeCompare(String(b.axisValue), undefined, { numeric: true })
    || a.mode.localeCompare(b.mode)
    || a.baselineMode.localeCompare(b.baselineMode)
  ));
}

function comparisonIndex(comparisonSummaryRows) {
  return new Map(comparisonSummaryRows.map((row) => [
    `${row.cellId}\t${row.mode}\t${row.baselineMode}`,
    row,
  ]));
}

function unsafeReason(summaryRow, naiveComparison) {
  if (summaryRow.primaryFailureMechanism) return summaryRow.primaryFailureMechanism;
  if (naiveComparison?.meanFalseFlagDelta > 1) return "overflagged";
  if (naiveComparison?.meanSurvivalDelta < 0 || naiveComparison?.meanMineTriggerDelta > 0) return "controller_overcommitted";
  if (naiveComparison?.meanBudgetAdjustedSafeTilesDelta < -1) return "performance_boundary";
  return "performance_boundary";
}

function makeUnsafeCells(summaryRows, comparisonSummaryRows) {
  const comparisons = comparisonIndex(comparisonSummaryRows);
  return summaryRows
    .filter((row) => TARGET_MODES.includes(row.mode))
    .map((row) => {
      const naive = comparisons.get(`${row.cellId}\t${row.mode}\tnaive_pressure`);
      return {
        ...row,
        naiveBudgetAdjustedSafeTilesDelta: naive?.meanBudgetAdjustedSafeTilesDelta ?? null,
        naiveSurvivalDelta: naive?.meanSurvivalDelta ?? null,
        naiveMineTriggerDelta: naive?.meanMineTriggerDelta ?? null,
        naiveFalseFlagDelta: naive?.meanFalseFlagDelta ?? null,
        unsafeReason: unsafeReason(row, naive),
      };
    })
    .filter((row) => (
      row.doNotUseRate > 0
      || row.naiveBudgetAdjustedSafeTilesDelta < -1
      || row.naiveSurvivalDelta < 0
      || row.naiveFalseFlagDelta > 1
    ))
    .sort((a, b) => (
      b.doNotUseRate - a.doNotUseRate
      || (a.naiveBudgetAdjustedSafeTilesDelta ?? 0) - (b.naiveBudgetAdjustedSafeTilesDelta ?? 0)
      || (b.naiveFalseFlagDelta ?? 0) - (a.naiveFalseFlagDelta ?? 0)
      || a.axis.localeCompare(b.axis)
      || String(a.axisValue).localeCompare(String(b.axisValue), undefined, { numeric: true })
      || a.mode.localeCompare(b.mode)
    ));
}

function makeCellManifestRows(cells) {
  return cells.map((cell) => {
    const board = normalizeMinesConfig(cell.board);
    const sensor = normalizeSensorConfig(cell.sensor);
    const boundary = assessMinesBoundary({
      boardConfig: board,
      sensorConfig: sensor,
      mode: "sundog_lean",
    });
    return {
      cellId: cell.cellId,
      axis: cell.axis,
      axisValue: cell.axisValue,
      preset: cell.preset,
      label: cell.label,
      width: board.width,
      height: board.height,
      mineCount: board.mineCount,
      mineDensity: roundMetric(board.mineCount / (board.width * board.height)),
      clusterStrength: roundMetric(board.generator?.clusterStrength ?? 0),
      scanBudget: board.scanBudget,
      sigma: sensor.sigma,
      sigmaNoise: sensor.sigmaNoise,
      dropoutRate: sensor.dropoutRate,
      delaySteps: sensor.delaySteps,
      kernelFamily: sensor.kernel,
      staticBoundaryStatus: boundary.status,
      staticBoundaryMechanisms: boundary.mechanisms.map((mechanism) => mechanism.code).join("|"),
    };
  }).sort((a, b) => (
    a.axis.localeCompare(b.axis)
    || String(a.axisValue).localeCompare(String(b.axisValue), undefined, { numeric: true })
  ));
}

function makePanelReport(unsafeRows, summaryRows, args) {
  const topUnsafe = unsafeRows.slice(0, 12).map((row) => ({
    axis: row.axis,
    axisValue: row.axisValue,
    preset: row.preset,
    mode: row.mode,
    unsafeReason: row.unsafeReason,
    doNotUseRate: row.doNotUseRate,
    survivalRate: row.survivalRate,
    mineTriggerRate: row.mineTriggerRate,
    naiveBudgetAdjustedSafeTilesDelta: row.naiveBudgetAdjustedSafeTilesDelta,
    naiveSurvivalDelta: row.naiveSurvivalDelta,
    naiveFalseFlagDelta: row.naiveFalseFlagDelta,
  }));
  const sundogGroups = summaryRows.filter((row) => TARGET_MODES.includes(row.mode));
  return {
    schema: "sundog.mines.phase9-boundary-panel.v1",
    generatedAt: new Date().toISOString(),
    phase: args.phase,
    headline: unsafeRows.length
      ? "Phase 9 found explicit Mines cells where Sundog should be treated as boundary material, not a success claim."
      : "Phase 9 did not find an unsafe Sundog cell in this first sweep; expand the grid before promotion.",
    trialGroups: sundogGroups.length,
    unsafeGroupCount: unsafeRows.length,
    topUnsafe,
    note: "This is a first-pass boundary diagnostic. It is not the Phase 10 operating-envelope verdict.",
  };
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const normalized = typeof value === "number" ? roundMetric(value) : value;
  const text = String(normalized);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(rows, columns) {
  const lines = [columns.join(",")];
  for (const row of rows) lines.push(columns.map((column) => csvEscape(row[column])).join(","));
  return `${lines.join("\n")}\n`;
}

function markdownReport({ args, cellRows, summaryRows, comparisonSummaryRows, unsafeRows }) {
  const leanVsNaive = comparisonSummaryRows
    .filter((row) => row.mode === "sundog_lean" && row.baselineMode === "naive_pressure")
    .slice(0, 18);
  const topUnsafe = unsafeRows.slice(0, 12);
  return [
    "# Sundog Pressure Mines Phase 9 Boundary Sweep",
    "",
    `Phase: \`${args.phase}\``,
    `Seeds per cell: \`${args.seeds}\``,
    `Boundary cells: \`${cellRows.length}\``,
    `Turn cap: \`${args.turnCap}\``,
    "",
    "## Outputs",
    "",
    "- `manifest.json` and `summary.json`: run shape and headline counts.",
    "- `boundary-panel.json`: compact workbench-facing no-use summary.",
    "- `cell-manifest.csv`: the explicit axis grid.",
    "- `trial-outcomes.csv`: one row per matched seed / mode / cell.",
    "- `boundary-summary.csv`: grouped mode metrics by boundary cell.",
    "- `matched-comparison.csv` and `matched-comparison-summary.csv`: Sundog variants versus random, naive, threshold, and oracle.",
    "- `unsafe-cells.csv`: first-pass cells to show as failure-boundary material.",
    "",
    "## Top No-Use Cells",
    "",
    "| axis | value | mode | reason | no-use | safe delta vs naive | survival delta | false-flag delta |",
    "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ...(topUnsafe.length > 0
      ? topUnsafe.map((row) => `| ${row.axis} | ${row.axisValue} | ${row.mode} | ${row.unsafeReason} | ${row.doNotUseRate} | ${row.naiveBudgetAdjustedSafeTilesDelta} | ${row.naiveSurvivalDelta} | ${row.naiveFalseFlagDelta} |`)
      : ["| n/a | n/a | n/a | none | 0 | 0 | 0 | 0 |"]),
    "",
    "## Lean Versus Naive Pressure",
    "",
    "| axis | value | budget delta | survival delta | mine delta | false flags | primary mechanism |",
    "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ...leanVsNaive.map((row) => `| ${row.axis} | ${row.axisValue} | ${row.meanBudgetAdjustedSafeTilesDelta} | ${row.meanSurvivalDelta} | ${row.meanMineTriggerDelta} | ${row.meanFalseFlagDelta} | ${row.primaryFailureMechanism || "none"} |`),
    "",
    "## Scope",
    "",
    "Phase 9 is a boundary diagnostic, not a promoted operating-envelope result. It exists to make the negative region visible before Phase 10 locks any claim.",
    "",
    `Summary groups: \`${summaryRows.length}\`. Unsafe Sundog groups: \`${unsafeRows.length}\`.`,
    "",
  ].join("\n");
}

function runSelfChecks(args, cells, trialRows) {
  const expected = cells.length * args.modes.length * args.seeds;
  if (trialRows.length !== expected) {
    throw new Error(`Expected ${expected} trial rows, got ${trialRows.length}`);
  }
  for (const row of trialRows) {
    if (row.budgetAdjustedSafeTiles > row.rawSafeTiles) {
      throw new Error(`Budget-adjusted safe tiles exceeded raw safe tiles for ${row.trialId}`);
    }
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const cells = makeBoundaryCells(args);
  const trialRows = [];

  for (const cell of cells) {
    for (let i = 0; i < args.seeds; i += 1) {
      const seed = args.seedStart + i;
      for (const mode of args.modes) {
        trialRows.push(runBoundaryTrial(args, { cell, mode, seed }));
      }
    }
  }

  runSelfChecks(args, cells, trialRows);
  const cellRows = makeCellManifestRows(cells);
  const summaryRows = summarizeTrials(trialRows);
  const comparisonRows = makeComparisonRows(trialRows);
  const comparisonSummaryRows = summarizeComparisons(comparisonRows);
  const unsafeRows = makeUnsafeCells(summaryRows, comparisonSummaryRows);
  const panelReport = makePanelReport(unsafeRows, summaryRows, args);
  const manifest = {
    schema: "sundog.mines.phase9-boundary.v1",
    generatedAt: new Date().toISOString(),
    phase: args.phase,
    axes: args.axes,
    modes: args.modes,
    seedStart: args.seedStart,
    seeds: args.seeds,
    turnCap: args.turnCap,
    pressureThreshold: args.pressureThreshold,
    cells: cells.length,
    trialCount: trialRows.length,
    summaryGroupCount: summaryRows.length,
    unsafeGroupCount: unsafeRows.length,
    note: "Ignored local Phase 9 boundary output. This is a sensor-degradation diagnostic, not the Phase 10 verdict.",
  };

  const outDir = path.resolve(repoRoot, args.out);
  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  await writeFile(path.join(outDir, "summary.json"), `${JSON.stringify({
    ...manifest,
    boundaryPanel: panelReport,
    topUnsafe: unsafeRows.slice(0, 12),
  }, null, 2)}\n`);
  await writeFile(path.join(outDir, "boundary-panel.json"), `${JSON.stringify(panelReport, null, 2)}\n`);
  await writeFile(path.join(outDir, "cell-manifest.csv"), toCsv(cellRows, [
    "cellId",
    "axis",
    "axisValue",
    "preset",
    "label",
    "width",
    "height",
    "mineCount",
    "mineDensity",
    "clusterStrength",
    "scanBudget",
    "sigma",
    "sigmaNoise",
    "dropoutRate",
    "delaySteps",
    "kernelFamily",
    "staticBoundaryStatus",
    "staticBoundaryMechanisms",
  ]));
  await writeFile(path.join(outDir, "trial-outcomes.csv"), toCsv(trialRows, [
    "phase",
    "cellId",
    "trialId",
    "axis",
    "axisValue",
    "preset",
    "mode",
    "seed",
    "actionTraceHash",
    "modeStatus",
    "usesPrivileged",
    "usesScan",
    "width",
    "height",
    "mineCount",
    "mineDensity",
    "clusterStrength",
    "scanBudget",
    "turnCap",
    "terminal",
    "survived",
    "fullClear",
    "turns",
    "controllerTurns",
    "revealCount",
    "flagCount",
    "scanCount",
    "illegalActionCount",
    "openingSafeCount",
    "rawSafeTiles",
    "safeTilesAfterOpening",
    "budgetAdjustedSafeTiles",
    "falseFlagCount",
    "frontierSize",
    "meanFrontierConfidence",
    "pressureThreshold",
    "sigma",
    "sigmaNoise",
    "dropoutRate",
    "delaySteps",
    "kernelFamily",
    "boundaryStatus",
    "boundaryPrimaryMechanism",
    "boundaryMechanisms",
  ]));
  await writeFile(path.join(outDir, "boundary-summary.csv"), toCsv(summaryRows, [
    "phase",
    "cellId",
    "axis",
    "axisValue",
    "preset",
    "mode",
    "n",
    "usesPrivileged",
    "usesScan",
    "width",
    "height",
    "mineCount",
    "mineDensity",
    "clusterStrength",
    "scanBudget",
    "sigma",
    "sigmaNoise",
    "dropoutRate",
    "delaySteps",
    "kernelFamily",
    "survivalRate",
    "fullClearRate",
    "mineTriggerRate",
    "meanRawSafeTiles",
    "meanSafeTilesAfterOpening",
    "meanBudgetAdjustedSafeTiles",
    "meanFalseFlagCount",
    "meanScanCount",
    "meanTurns",
    "meanFrontierConfidence",
    "illegalActionCount",
    "doNotUseRate",
    "watchBoundaryRate",
    "primaryFailureMechanism",
  ]));
  await writeFile(path.join(outDir, "matched-comparison.csv"), toCsv(comparisonRows, [
    "phase",
    "cellId",
    "axis",
    "axisValue",
    "preset",
    "seed",
    "mode",
    "baselineMode",
    "width",
    "height",
    "mineCount",
    "mineDensity",
    "clusterStrength",
    "scanBudget",
    "sigma",
    "sigmaNoise",
    "dropoutRate",
    "delaySteps",
    "targetTerminal",
    "baselineTerminal",
    "rawSafeTilesDelta",
    "safeTilesAfterOpeningDelta",
    "budgetAdjustedSafeTilesDelta",
    "survivalDelta",
    "mineTriggerDelta",
    "falseFlagDelta",
    "scanDelta",
    "targetBoundaryStatus",
    "targetFailureMechanism",
  ]));
  await writeFile(path.join(outDir, "matched-comparison-summary.csv"), toCsv(comparisonSummaryRows, [
    "phase",
    "cellId",
    "axis",
    "axisValue",
    "preset",
    "mode",
    "baselineMode",
    "n",
    "width",
    "height",
    "mineCount",
    "mineDensity",
    "clusterStrength",
    "scanBudget",
    "sigma",
    "sigmaNoise",
    "dropoutRate",
    "delaySteps",
    "meanRawSafeTilesDelta",
    "meanSafeTilesAfterOpeningDelta",
    "meanBudgetAdjustedSafeTilesDelta",
    "meanSurvivalDelta",
    "meanMineTriggerDelta",
    "meanFalseFlagDelta",
    "meanScanDelta",
    "doNotUseRate",
    "primaryFailureMechanism",
  ]));
  await writeFile(path.join(outDir, "unsafe-cells.csv"), toCsv(unsafeRows, [
    "phase",
    "cellId",
    "axis",
    "axisValue",
    "preset",
    "mode",
    "n",
    "usesPrivileged",
    "usesScan",
    "width",
    "height",
    "mineCount",
    "mineDensity",
    "clusterStrength",
    "scanBudget",
    "sigma",
    "sigmaNoise",
    "dropoutRate",
    "delaySteps",
    "survivalRate",
    "mineTriggerRate",
    "meanBudgetAdjustedSafeTiles",
    "meanFalseFlagCount",
    "doNotUseRate",
    "watchBoundaryRate",
    "primaryFailureMechanism",
    "naiveBudgetAdjustedSafeTilesDelta",
    "naiveSurvivalDelta",
    "naiveMineTriggerDelta",
    "naiveFalseFlagDelta",
    "unsafeReason",
  ]));
  await writeFile(path.join(outDir, "phase9-boundary.md"), markdownReport({
    args,
    cellRows,
    summaryRows,
    comparisonSummaryRows,
    unsafeRows,
  }));

  console.log(`Mines ${args.phase}: ${trialRows.length} trials across ${cells.length} boundary cells`);
  console.log(`Wrote ${path.relative(repoRoot, outDir)}`);
  console.log(`Unsafe Sundog groups: ${unsafeRows.length}/${summaryRows.filter((row) => TARGET_MODES.includes(row.mode)).length}`);
  for (const row of unsafeRows.slice(0, 6)) {
    console.log(`${row.axis}=${row.axisValue} ${row.mode}: ${row.unsafeReason}, delta ${row.naiveBudgetAdjustedSafeTilesDelta}, no-use ${row.doNotUseRate}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
