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
} from "../public/js/mines-core.mjs";
import {
  IMPLEMENTED_MINES_MODES,
  MINES_CONTROLLER_MODES,
  chooseMinesAction,
} from "../public/js/mines-controllers.mjs";
import { createSensorRuntime, normalizeSensorConfig } from "../public/js/mines-sensor.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const DEFAULT_PRESETS = Object.freeze(["easy_sparse", "clustered", "dense"]);
const DEFAULT_MODES = Object.freeze([
  "random_reveal",
  "naive_pressure",
  "threshold_flagger",
  "naive_pressure_shuffled",
  "naive_pressure_delayed",
  "oracle_safe",
]);

const SENSOR_CELLS = Object.freeze([
  Object.freeze({
    name: "doc_default",
    publishable: true,
    sensor: Object.freeze({ sigma: 1.0, sigmaNoise: 0.1, dropoutRate: 0.1, delaySteps: 0 }),
  }),
  Object.freeze({
    name: "blur_noise_cliff",
    publishable: true,
    sensor: Object.freeze({ sigma: 8.0, sigmaNoise: 10.0, dropoutRate: 0.8, delaySteps: 0 }),
  }),
]);

function parseList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function parseArgs(argv) {
  const args = {
    phase: "phase4-baselines",
    out: "results/mines/phase4-baselines",
    seedStart: 0,
    seeds: 16,
    presets: [...DEFAULT_PRESETS],
    modes: [...DEFAULT_MODES],
    cells: SENSOR_CELLS.map((cell) => cell.name),
    turnCap: 160,
    pressureThreshold: 1.2,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    i += 1;

    if (flag === "--phase") args.phase = value;
    else if (flag === "--out") args.out = value;
    else if (flag === "--seed-start") args.seedStart = Number.parseInt(value, 10);
    else if (flag === "--seeds") args.seeds = Number.parseInt(value, 10);
    else if (flag === "--presets") args.presets = parseList(value);
    else if (flag === "--modes") args.modes = parseList(value);
    else if (flag === "--cells") args.cells = parseList(value);
    else if (flag === "--turn-cap") args.turnCap = Number.parseInt(value, 10);
    else if (flag === "--pressure-threshold") args.pressureThreshold = Number.parseFloat(value);
    else throw new Error(`Unknown flag: ${flag}`);
  }

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
  for (const preset of args.presets) {
    if (!MINES_PRESETS[preset]) throw new Error(`Unknown mines preset: ${preset}`);
  }
  for (const mode of args.modes) {
    if (!MINES_CONTROLLER_MODES[mode]) throw new Error(`Unknown mines mode: ${mode}`);
    if (!IMPLEMENTED_MINES_MODES.includes(mode)) {
      throw new Error(`Mode ${mode} is declared but not implemented yet`);
    }
  }
  for (const cell of args.cells) {
    if (!SENSOR_CELLS.some((candidate) => candidate.name === cell)) {
      throw new Error(`Unknown sensor cell: ${cell}`);
    }
  }
  return args;
}

function stringHash(text) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
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

function ratio(numerator, denominator) {
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator <= 0) return null;
  return numerator / denominator;
}

function centerAction(boardState) {
  return {
    type: ACTION.REVEAL,
    x: Math.floor(boardState.config.width / 2),
    y: Math.floor(boardState.config.height / 2),
  };
}

function mergedSensorConfig(cell, mode, seed) {
  const definition = MINES_CONTROLLER_MODES[mode];
  return normalizeSensorConfig({
    ...cell.sensor,
    ...(definition.sensorOverride ?? {}),
    sensorSeed: seed + 7919 + stringHash(mode),
  });
}

function terminalClass(terminal) {
  if (terminal === "full_clear") return "full_clear";
  if (terminal === "mine_triggered") return "mine_triggered";
  if (terminal === "turn_cap") return "turn_cap";
  if (terminal === "scan_budget_exhausted") return "scan_budget_exhausted";
  return terminal ?? "none";
}

function scoreTrial({ preset, mode, cell, seed, args }) {
  const board = initializeBoardState({
    preset,
    seed,
    turnCap: args.turnCap,
  });
  applyMinesAction(board, centerAction(board));
  const openingSafeCount = board.revealedSafeCount;
  const sensorConfig = mergedSensorConfig(cell, mode, seed);
  const sensorRuntime = createSensorRuntime(sensorConfig);
  const rng = makeRng(seed ^ stringHash(mode) ^ stringHash(cell.name));

  let illegalActionCount = 0;
  while (board.terminal === null) {
    const sensor = sensorRuntime.step(board);
    const memory = getPublicMemory(board);
    const action = chooseMinesAction({
      mode,
      memory,
      sensor,
      boardState: board,
      rng,
      options: { threshold: args.pressureThreshold },
    });
    const result = applyMinesAction(board, action);
    if (!result.applied) {
      illegalActionCount += 1;
      const fallback = chooseMinesAction({
        mode: "random_reveal",
        memory,
        sensor,
        boardState: board,
        rng,
      });
      applyMinesAction(board, fallback);
    }
  }

  const revealCount = board.actionLedger.filter((entry) => entry.type === ACTION.REVEAL).length;
  const flagCount = board.actionLedger.filter((entry) => entry.type === ACTION.FLAG).length;
  const scanCount = board.actionLedger.filter((entry) => entry.type === ACTION.SCAN).length;
  const budgetAdjustedSafeTiles = board.revealedSafeCount - scanCount;
  const safeTilesAfterOpening = Math.max(0, board.revealedSafeCount - openingSafeCount);
  return {
    phase: args.phase,
    preset,
    sensorCell: cell.name,
    mode,
    seed,
    publishableSensorCell: cell.publishable,
    modeStatus: MINES_CONTROLLER_MODES[mode].status,
    usesPrivileged: MINES_CONTROLLER_MODES[mode].usesPrivileged,
    width: board.config.width,
    height: board.config.height,
    mineCount: board.config.mineCount,
    mineDensity: roundMetric(board.config.mineCount / (board.config.width * board.config.height)),
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
    rawSafeTiles: board.revealedSafeCount,
    safeTilesAfterOpening,
    budgetAdjustedSafeTiles,
    falseFlagCount: board.falseFlagCount,
    pressureThreshold: args.pressureThreshold,
    sigma: sensorConfig.sigma,
    sigmaNoise: sensorConfig.sigmaNoise,
    dropoutRate: sensorConfig.dropoutRate,
    delaySteps: sensorConfig.delaySteps,
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

function summarizeTrials(rows) {
  return Array.from(groupBy(rows, (row) => `${row.preset}\t${row.sensorCell}\t${row.mode}`).entries())
    .map(([key, group]) => {
      const [preset, sensorCell, mode] = key.split("\t");
      const first = group[0];
      return {
        phase: first.phase,
        preset,
        sensorCell,
        mode,
        n: group.length,
        usesPrivileged: first.usesPrivileged,
        survivalRate: roundMetric(mean(group.map((row) => row.survived ? 1 : 0))),
        fullClearRate: roundMetric(mean(group.map((row) => row.fullClear ? 1 : 0))),
        meanRawSafeTiles: roundMetric(mean(group.map((row) => row.rawSafeTiles))),
        meanSafeTilesAfterOpening: roundMetric(mean(group.map((row) => row.safeTilesAfterOpening))),
        meanBudgetAdjustedSafeTiles: roundMetric(mean(group.map((row) => row.budgetAdjustedSafeTiles))),
        meanFalseFlagCount: roundMetric(mean(group.map((row) => row.falseFlagCount))),
        meanScanCount: roundMetric(mean(group.map((row) => row.scanCount))),
        meanTurns: roundMetric(mean(group.map((row) => row.turns))),
        illegalActionCount: sum(group.map((row) => row.illegalActionCount)),
      };
    })
    .sort((a, b) => (
      a.preset.localeCompare(b.preset)
      || a.sensorCell.localeCompare(b.sensorCell)
      || a.mode.localeCompare(b.mode)
    ));
}

function makeComparisonRows(trialRows, baselineMode = "naive_pressure") {
  const byKey = new Map();
  for (const row of trialRows) {
    byKey.set(`${row.preset}\t${row.sensorCell}\t${row.seed}\t${row.mode}`, row);
  }
  return trialRows
    .filter((row) => row.mode !== baselineMode)
    .map((row) => {
      const baseline = byKey.get(`${row.preset}\t${row.sensorCell}\t${row.seed}\t${baselineMode}`);
      if (!baseline) return null;
      return {
        phase: row.phase,
        preset: row.preset,
        sensorCell: row.sensorCell,
        seed: row.seed,
        mode: row.mode,
        baselineMode,
        rawSafeTilesDelta: row.rawSafeTiles - baseline.rawSafeTiles,
        budgetAdjustedSafeTilesDelta: row.budgetAdjustedSafeTiles - baseline.budgetAdjustedSafeTiles,
        survivalDelta: (row.survived ? 1 : 0) - (baseline.survived ? 1 : 0),
        falseFlagDelta: row.falseFlagCount - baseline.falseFlagCount,
      };
    })
    .filter(Boolean);
}

function summarizeComparisons(comparisonRows) {
  return Array.from(groupBy(comparisonRows, (row) => `${row.preset}\t${row.sensorCell}\t${row.mode}\t${row.baselineMode}`).entries())
    .map(([key, group]) => {
      const [preset, sensorCell, mode, baselineMode] = key.split("\t");
      return {
        phase: group[0].phase,
        preset,
        sensorCell,
        mode,
        baselineMode,
        n: group.length,
        meanRawSafeTilesDelta: roundMetric(mean(group.map((row) => row.rawSafeTilesDelta))),
        meanBudgetAdjustedSafeTilesDelta: roundMetric(mean(group.map((row) => row.budgetAdjustedSafeTilesDelta))),
        meanSurvivalDelta: roundMetric(mean(group.map((row) => row.survivalDelta))),
        meanFalseFlagDelta: roundMetric(mean(group.map((row) => row.falseFlagDelta))),
      };
    })
    .sort((a, b) => (
      a.preset.localeCompare(b.preset)
      || a.sensorCell.localeCompare(b.sensorCell)
      || a.mode.localeCompare(b.mode)
    ));
}

function modeBudgetRows() {
  return Object.entries(MINES_CONTROLLER_MODES).map(([mode, definition]) => ({
    mode,
    label: definition.label,
    status: definition.status,
    informationBudget: definition.informationBudget,
    usesPressure: definition.usesPressure,
    usesGradient: definition.usesGradient,
    usesConfidence: definition.usesConfidence,
    usesActionHistory: definition.usesActionHistory,
    usesScan: definition.usesScan,
    usesPrivileged: definition.usesPrivileged,
    ablation: definition.ablation ?? "",
  }));
}

function rowsToCsv(rows) {
  if (rows.length === 0) return "";
  const headers = Object.keys(rows[0]);
  const escape = (value) => {
    if (value === null || value === undefined) return "";
    const text = String(value);
    return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  return [
    headers.join(","),
    ...rows.map((row) => headers.map((header) => escape(row[header])).join(",")),
  ].join("\n") + "\n";
}

function markdownReport({ args, budgetRows, summaryRows, comparisonSummaryRows }) {
  const implemented = budgetRows.filter((row) => row.status === "implemented");
  const pending = budgetRows.filter((row) => row.status !== "implemented");
  const docDefaultRows = summaryRows.filter((row) => row.sensorCell === "doc_default");
  return [
    "# Sundog Pressure Mines Phase 4 Baseline Set",
    "",
    `Phase: \`${args.phase}\``,
    `Seeds per cell: \`${args.seeds}\``,
    `Turn cap: \`${args.turnCap}\``,
    `Budget-adjusted safe tiles: \`rawSafeTiles - scanCount\``,
    "",
    "## Implemented Lanes",
    "",
    "| mode | budget | privileged? | scans? |",
    "| --- | --- | --- | --- |",
    ...implemented.map((row) => `| ${row.mode} | ${row.informationBudget} | ${row.usesPrivileged} | ${row.usesScan} |`),
    "",
    "## Pending Phase 5 Lanes",
    "",
    "| mode | planned budget |",
    "| --- | --- |",
    ...pending.map((row) => `| ${row.mode} | ${row.informationBudget} |`),
    "",
    "## Doc-Default Summary",
    "",
    "| preset | mode | survival | raw safe | budget safe | false flags |",
    "| --- | --- | ---: | ---: | ---: | ---: |",
    ...docDefaultRows.map((row) => `| ${row.preset} | ${row.mode} | ${row.survivalRate} | ${row.meanRawSafeTiles} | ${row.meanBudgetAdjustedSafeTiles} | ${row.meanFalseFlagCount} |`),
    "",
    "## Matched-Seed Deltas Versus Naive Pressure",
    "",
    "| preset | sensor cell | mode | raw delta | budget delta | survival delta |",
    "| --- | --- | --- | ---: | ---: | ---: |",
    ...comparisonSummaryRows.map((row) => `| ${row.preset} | ${row.sensorCell} | ${row.mode} | ${row.meanRawSafeTilesDelta} | ${row.meanBudgetAdjustedSafeTilesDelta} | ${row.meanSurvivalDelta} |`),
    "",
    "## Scope",
    "",
    "This is a Phase 4 fairness scaffold. It does not claim a Sundog win; the Sundog controller and its ablations remain pending until Phase 5.",
    "",
  ].join("\n");
}

function runSelfChecks({ args, budgetRows, trialRows }) {
  const missingBudget = args.modes.filter((mode) => !budgetRows.some((row) => row.mode === mode));
  if (missingBudget.length > 0) {
    throw new Error(`Missing mode budgets for: ${missingBudget.join(", ")}`);
  }
  for (const row of trialRows) {
    if (row.budgetAdjustedSafeTiles > row.rawSafeTiles) {
      throw new Error(`Budget-adjusted safe tiles exceeded raw safe tiles for ${row.mode}`);
    }
  }
  const expected = args.presets.length * args.cells.length * args.modes.length * args.seeds;
  if (trialRows.length !== expected) {
    throw new Error(`Expected ${expected} trial rows, got ${trialRows.length}`);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const selectedCells = SENSOR_CELLS.filter((cell) => args.cells.includes(cell.name));
  const trialRows = [];

  for (const preset of args.presets) {
    for (const cell of selectedCells) {
      for (const mode of args.modes) {
        for (let i = 0; i < args.seeds; i += 1) {
          trialRows.push(scoreTrial({
            preset,
            mode,
            cell,
            seed: args.seedStart + i,
            args,
          }));
        }
      }
    }
  }

  const budgetRows = modeBudgetRows();
  runSelfChecks({ args, budgetRows, trialRows });
  const summaryRows = summarizeTrials(trialRows);
  const comparisonRows = makeComparisonRows(trialRows);
  const comparisonSummaryRows = summarizeComparisons(comparisonRows);

  const outDir = path.resolve(repoRoot, args.out);
  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "mode-budgets.csv"), rowsToCsv(budgetRows));
  await writeFile(path.join(outDir, "trial-rows.csv"), rowsToCsv(trialRows));
  await writeFile(path.join(outDir, "summary-rows.csv"), rowsToCsv(summaryRows));
  await writeFile(path.join(outDir, "matched-comparisons.csv"), rowsToCsv(comparisonRows));
  await writeFile(path.join(outDir, "matched-comparison-summary.csv"), rowsToCsv(comparisonSummaryRows));
  await writeFile(path.join(outDir, "summary.json"), JSON.stringify({
    schema: "sundog.mines.phase4-baselines.v1",
    args,
    sensorCells: selectedCells,
    modeBudgets: budgetRows,
    summaryRows,
    comparisonSummaryRows,
  }, null, 2));
  await writeFile(path.join(outDir, "phase4-baselines.md"), markdownReport({
    args,
    budgetRows,
    summaryRows,
    comparisonSummaryRows,
  }));

  console.log(`Mines Phase 4 baselines wrote ${path.relative(repoRoot, outDir)}`);
  console.log(`Trial rows: ${trialRows.length}; comparison rows: ${comparisonRows.length}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
