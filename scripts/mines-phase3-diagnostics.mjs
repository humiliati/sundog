import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  ACTION,
  applyMinesAction,
  initializeBoardState,
  MINES_PRESETS,
} from "../public/js/mines-core.mjs";
import {
  computeAuroc,
  computeCorrelation,
  computeLocalRisk,
  createSensorRuntime,
  normalizeSensorConfig,
} from "../public/js/mines-sensor.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const DEFAULT_PRESETS = Object.freeze([
  "easy_sparse",
  "clustered",
  "ambiguous_overlap",
  "noisy_sensor",
  "delayed_sensor",
  "dense",
]);

const SENSOR_CELLS = Object.freeze([
  Object.freeze({
    name: "doc_default",
    publishable: true,
    note: "Pinned Phase 2 operating point: sigma=1.0, noise=0.10, dropout=0.10, delay=0.",
    sensor: Object.freeze({ sigma: 1.0, sigmaNoise: 0.1, dropoutRate: 0.1, delaySteps: 0 }),
  }),
  Object.freeze({
    name: "wide_blur",
    publishable: true,
    note: "Wider support should smooth local distinctions without violating the floor.",
    sensor: Object.freeze({ sigma: 1.5, sigmaNoise: 0.1, dropoutRate: 0.1, delaySteps: 0 }),
  }),
  Object.freeze({
    name: "high_noise",
    publishable: true,
    note: "Noise degradation cell.",
    sensor: Object.freeze({ sigma: 1.0, sigmaNoise: 0.25, dropoutRate: 0.1, delaySteps: 0 }),
  }),
  Object.freeze({
    name: "dropout_heavy",
    publishable: true,
    note: "Dropout degradation cell.",
    sensor: Object.freeze({ sigma: 1.0, sigmaNoise: 0.1, dropoutRate: 0.35, delaySteps: 0 }),
  }),
  Object.freeze({
    name: "delayed_2",
    publishable: true,
    note: "Two-turn delayed observation cell. Static board pressure makes this mostly a controller-facing degradation.",
    sensor: Object.freeze({ sigma: 1.0, sigmaNoise: 0.1, dropoutRate: 0.1, delaySteps: 2 }),
  }),
  Object.freeze({
    name: "blur_noise_cliff",
    publishable: true,
    note: "Above-floor ambiguity cliff: broad blur plus severe noise/dropout.",
    sensor: Object.freeze({ sigma: 8.0, sigmaNoise: 10.0, dropoutRate: 0.8, delaySteps: 0 }),
  }),
  Object.freeze({
    name: "degenerate_sanity",
    publishable: false,
    note: "Sub-floor exactish sanity cell. Kept out of public evidence by construction.",
    sensor: Object.freeze({
      sigma: 0.35,
      sigmaNoise: 0,
      dropoutRate: 0,
      delaySteps: 0,
      allowSubFloor: true,
    }),
  }),
]);

function parseList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function parseArgs(argv) {
  const args = {
    phase: "phase3-diagnostic",
    out: "results/mines/phase3-diagnostic",
    seedStart: 0,
    seeds: 24,
    presets: [...DEFAULT_PRESETS],
    cells: SENSOR_CELLS.map((cell) => cell.name),
    namedEnvelopePreset: "easy_sparse",
    namedEnvelopeCell: "doc_default",
    aurocThreshold: 0.65,
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
    else if (flag === "--cells") args.cells = parseList(value);
    else if (flag === "--named-envelope-preset") args.namedEnvelopePreset = value;
    else if (flag === "--named-envelope-cell") args.namedEnvelopeCell = value;
    else if (flag === "--auroc-threshold") args.aurocThreshold = Number.parseFloat(value);
    else throw new Error(`Unknown flag: ${flag}`);
  }

  if (!Number.isInteger(args.seedStart) || args.seedStart < 0) {
    throw new Error("--seed-start must be a non-negative integer");
  }
  if (!Number.isInteger(args.seeds) || args.seeds < 1) {
    throw new Error("--seeds must be a positive integer");
  }
  if (!Number.isFinite(args.aurocThreshold) || args.aurocThreshold <= 0 || args.aurocThreshold >= 1) {
    throw new Error("--auroc-threshold must be between 0 and 1");
  }
  for (const preset of args.presets) {
    if (!MINES_PRESETS[preset]) throw new Error(`Unknown mines preset: ${preset}`);
  }
  for (const cell of args.cells) {
    if (!SENSOR_CELLS.some((candidate) => candidate.name === cell)) {
      throw new Error(`Unknown sensor cell: ${cell}`);
    }
  }
  return args;
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

function gradientMagnitude(sensor) {
  const n = sensor.gradientX.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    const gx = sensor.gradientX[i];
    const gy = sensor.gradientY[i];
    out[i] = Number.isFinite(gx) && Number.isFinite(gy) ? Math.hypot(gx, gy) : NaN;
  }
  return out;
}

function finiteCount(values) {
  let count = 0;
  for (const value of values) {
    if (Number.isFinite(value)) count += 1;
  }
  return count;
}

function indicesWhere(length, predicate) {
  const out = [];
  for (let i = 0; i < length; i += 1) {
    if (predicate(i)) out.push(i);
  }
  return out;
}

function pick(values, indices) {
  return indices.map((index) => values[index]);
}

function labelsFrom(values, indices, predicate) {
  return indices.map((index) => predicate(values[index], index) ? 1 : 0);
}

function evaluateThresholds(scores, labels) {
  const pairs = [];
  for (let i = 0; i < scores.length; i += 1) {
    if (Number.isFinite(scores[i])) pairs.push({ score: scores[i], label: labels[i] ? 1 : 0 });
  }
  if (pairs.length === 0) return emptyThresholdMetrics();
  const positives = pairs.filter((pair) => pair.label === 1).length;
  const negatives = pairs.length - positives;
  if (positives === 0 || negatives === 0) return emptyThresholdMetrics(pairs.length, positives);

  const thresholds = [...new Set(pairs.map((pair) => pair.score))].sort((a, b) => a - b);
  let best = null;
  for (const threshold of thresholds) {
    let tp = 0;
    let fp = 0;
    let tn = 0;
    let fn = 0;
    for (const pair of pairs) {
      const warned = pair.score >= threshold;
      if (warned && pair.label) tp += 1;
      else if (warned) fp += 1;
      else if (pair.label) fn += 1;
      else tn += 1;
    }
    const precision = ratio(tp, tp + fp);
    const recall = ratio(tp, tp + fn);
    const f1 = precision !== null && recall !== null && precision + recall > 0
      ? 2 * precision * recall / (precision + recall)
      : null;
    const candidate = {
      threshold,
      precision,
      recall,
      f1,
      falseAlarmRate: ratio(fp, fp + tn),
      truePositive: tp,
      falsePositive: fp,
      trueNegative: tn,
      falseNegative: fn,
    };
    if (best === null || (candidate.f1 ?? -1) > (best.f1 ?? -1)) best = candidate;
  }
  return {
    sampleCount: pairs.length,
    positiveCount: positives,
    bestThreshold: roundMetric(best.threshold),
    bestPrecision: roundMetric(best.precision),
    bestRecall: roundMetric(best.recall),
    bestF1: roundMetric(best.f1),
    falseAlarmRateAtBestF1: roundMetric(best.falseAlarmRate),
  };
}

function emptyThresholdMetrics(sampleCount = 0, positiveCount = 0) {
  return {
    sampleCount,
    positiveCount,
    bestThreshold: null,
    bestPrecision: null,
    bestRecall: null,
    bestF1: null,
    falseAlarmRateAtBestF1: null,
  };
}

function adjacentIndices(width, height, index) {
  const x = index % width;
  const y = Math.floor(index / width);
  const out = [];
  for (let dy = -1; dy <= 1; dy += 1) {
    for (let dx = -1; dx <= 1; dx += 1) {
      if (dx === 0 && dy === 0) continue;
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      out.push(ny * width + nx);
    }
  }
  return out;
}

function makeOpeningFrontier(boardState) {
  const { width, height } = boardState.config;
  const center = {
    x: Math.floor(width / 2),
    y: Math.floor(height / 2),
  };
  applyMinesAction(boardState, { type: ACTION.REVEAL, ...center });
  const frontier = [];
  for (let idx = 0; idx < width * height; idx += 1) {
    if (boardState.tiles[idx] !== "concealed") continue;
    if (adjacentIndices(width, height, idx).some((neighbor) => boardState.tiles[neighbor] === "revealed_safe")) {
      frontier.push(idx);
    }
  }
  return frontier;
}

function precisionAtLowest(scores, labels, k) {
  const pairs = [];
  for (let i = 0; i < scores.length; i += 1) {
    if (Number.isFinite(scores[i])) pairs.push({ score: scores[i], label: labels[i] ? 1 : 0 });
  }
  if (pairs.length === 0) return null;
  pairs.sort((a, b) => a.score - b.score);
  const selected = pairs.slice(0, Math.min(k, pairs.length));
  return selected.filter((pair) => pair.label === 1).length / selected.length;
}

function scoreTrial({ preset, seed, cell }) {
  const board = initializeBoardState({ preset, seed });
  const sensorConfig = normalizeSensorConfig({
    ...cell.sensor,
    sensorSeed: seed + 1009,
  });
  const runtime = createSensorRuntime(sensorConfig);
  let sensor = null;
  for (let i = 0; i <= sensorConfig.delaySteps; i += 1) {
    sensor = runtime.step(board);
  }

  const { width, height } = board.config;
  const n = width * height;
  const occupancy = board.privileged.occupancy;
  const adjacency = board.privileged.adjacency;
  const safeIndices = indicesWhere(n, (idx) => occupancy[idx] === 0);
  const localRisk = computeLocalRisk(board);
  const gradMag = gradientMagnitude(sensor);

  const occupancyLabels = Array.from(occupancy);
  const safeAdjacencyLabels = labelsFrom(adjacency, safeIndices, (value) => value > 0);
  const occupancyAuroc = computeAuroc(Array.from(sensor.observed), occupancyLabels);
  const adjacencyAuroc = computeAuroc(pick(sensor.observed, safeIndices), safeAdjacencyLabels);
  const gradientOccupancyAuroc = computeAuroc(Array.from(gradMag), occupancyLabels);
  const gradientAdjacencyAuroc = computeAuroc(pick(gradMag, safeIndices), safeAdjacencyLabels);
  const pressureRiskCorrelation = computeCorrelation(sensor.observed, localRisk);
  const gradientRiskCorrelation = computeCorrelation(gradMag, localRisk);
  const pressureTrueCorrelation = computeCorrelation(sensor.observed, sensor.truePressure);
  const pr = evaluateThresholds(Array.from(sensor.observed), occupancyLabels);

  const frontierBoard = initializeBoardState({ preset, seed });
  const frontier = makeOpeningFrontier(frontierBoard);
  const frontierRuntime = createSensorRuntime({
    ...cell.sensor,
    sensorSeed: seed + 2003,
  });
  let frontierSensor = null;
  const frontierConfig = frontierRuntime.config;
  for (let i = 0; i <= frontierConfig.delaySteps; i += 1) {
    frontierSensor = frontierRuntime.step(frontierBoard);
  }
  const frontierGradMag = gradientMagnitude(frontierSensor);
  const frontierSafeLabels = labelsFrom(frontierBoard.privileged.occupancy, frontier, (value) => value === 0);
  const frontierHazardLabels = frontierSafeLabels.map((label) => label ? 0 : 1);
  const frontierPressureScores = pick(frontierSensor.observed, frontier);
  const frontierGradientScores = pick(frontierGradMag, frontier);

  return {
    phase: "phase3-diagnostic",
    preset,
    seed,
    sensorCell: cell.name,
    publishable: cell.publishable,
    width,
    height,
    mineCount: board.config.mineCount,
    mineDensity: roundMetric(board.config.mineCount / n),
    sigma: sensorConfig.sigma,
    sigmaNoise: sensorConfig.sigmaNoise,
    dropoutRate: sensorConfig.dropoutRate,
    delaySteps: sensorConfig.delaySteps,
    observedCoverage: roundMetric(finiteCount(sensor.observed) / n),
    meanConfidence: roundMetric(mean(Array.from(sensor.confidence))),
    mineOccupancyAuroc: roundMetric(occupancyAuroc),
    safeTileAdjacencyAuroc: roundMetric(adjacencyAuroc),
    gradientMineOccupancyAuroc: roundMetric(gradientOccupancyAuroc),
    gradientSafeTileAdjacencyAuroc: roundMetric(gradientAdjacencyAuroc),
    pressureLocalRiskCorrelation: roundMetric(pressureRiskCorrelation),
    gradientLocalRiskCorrelation: roundMetric(gradientRiskCorrelation),
    pressureTruePressureCorrelation: roundMetric(pressureTrueCorrelation),
    occupancyBestThreshold: pr.bestThreshold,
    occupancyBestPrecision: pr.bestPrecision,
    occupancyBestRecall: pr.bestRecall,
    occupancyBestF1: pr.bestF1,
    occupancyFalseAlarmRateAtBestF1: pr.falseAlarmRateAtBestF1,
    frontierCount: frontier.length,
    frontierObservedCoverage: roundMetric(frontierPressureScores.filter(Number.isFinite).length / Math.max(frontier.length, 1)),
    frontierBaseSafeRate: roundMetric(ratio(sum(frontierSafeLabels), frontierSafeLabels.length)),
    frontierHazardAuroc: roundMetric(computeAuroc(frontierPressureScores, frontierHazardLabels)),
    frontierGradientHazardAuroc: roundMetric(computeAuroc(frontierGradientScores, frontierHazardLabels)),
    frontierSafePrecisionAt1: roundMetric(precisionAtLowest(frontierPressureScores, frontierSafeLabels, 1)),
    frontierSafePrecisionAt3: roundMetric(precisionAtLowest(frontierPressureScores, frontierSafeLabels, 3)),
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

function summarizeRows(trialRows) {
  return Array.from(groupBy(trialRows, (row) => `${row.preset}\t${row.sensorCell}`).entries())
    .map(([key, rows]) => {
      const [preset, sensorCell] = key.split("\t");
      const first = rows[0];
      return {
        phase: "phase3-diagnostic",
        preset,
        sensorCell,
        publishable: first.publishable,
        n: rows.length,
        mineDensity: roundMetric(mean(rows.map((row) => row.mineDensity))),
        sigma: first.sigma,
        sigmaNoise: first.sigmaNoise,
        dropoutRate: first.dropoutRate,
        delaySteps: first.delaySteps,
        meanObservedCoverage: roundMetric(mean(rows.map((row) => row.observedCoverage))),
        meanMineOccupancyAuroc: roundMetric(mean(rows.map((row) => row.mineOccupancyAuroc))),
        meanSafeTileAdjacencyAuroc: roundMetric(mean(rows.map((row) => row.safeTileAdjacencyAuroc))),
        meanGradientMineOccupancyAuroc: roundMetric(mean(rows.map((row) => row.gradientMineOccupancyAuroc))),
        meanGradientSafeTileAdjacencyAuroc: roundMetric(mean(rows.map((row) => row.gradientSafeTileAdjacencyAuroc))),
        meanPressureLocalRiskCorrelation: roundMetric(mean(rows.map((row) => row.pressureLocalRiskCorrelation))),
        meanGradientLocalRiskCorrelation: roundMetric(mean(rows.map((row) => row.gradientLocalRiskCorrelation))),
        meanPressureTruePressureCorrelation: roundMetric(mean(rows.map((row) => row.pressureTruePressureCorrelation))),
        meanOccupancyBestPrecision: roundMetric(mean(rows.map((row) => row.occupancyBestPrecision))),
        meanOccupancyBestRecall: roundMetric(mean(rows.map((row) => row.occupancyBestRecall))),
        meanOccupancyBestF1: roundMetric(mean(rows.map((row) => row.occupancyBestF1))),
        meanFrontierCount: roundMetric(mean(rows.map((row) => row.frontierCount))),
        meanFrontierObservedCoverage: roundMetric(mean(rows.map((row) => row.frontierObservedCoverage))),
        meanFrontierBaseSafeRate: roundMetric(mean(rows.map((row) => row.frontierBaseSafeRate))),
        meanFrontierHazardAuroc: roundMetric(mean(rows.map((row) => row.frontierHazardAuroc))),
        meanFrontierGradientHazardAuroc: roundMetric(mean(rows.map((row) => row.frontierGradientHazardAuroc))),
        meanFrontierSafePrecisionAt1: roundMetric(mean(rows.map((row) => row.frontierSafePrecisionAt1))),
        meanFrontierSafePrecisionAt3: roundMetric(mean(rows.map((row) => row.frontierSafePrecisionAt3))),
      };
    })
    .sort((a, b) => (
      a.preset.localeCompare(b.preset)
      || a.sensorCell.localeCompare(b.sensorCell)
    ));
}

function cellDisposition(row, threshold) {
  if (!row.publishable) return "sanity_only";
  if (row.meanMineOccupancyAuroc === null) return "untestable";
  if (row.meanMineOccupancyAuroc >= threshold) return "diagnostic_positive";
  if (row.meanMineOccupancyAuroc >= 0.55) return "weak_signal";
  if (row.meanMineOccupancyAuroc >= 0.45) return "chance_boundary";
  return "inverted_or_misleading";
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

function markdownReport({ args, summaryRows, namedEnvelope, bestRows, chanceRows }) {
  return [
    "# Sundog Pressure Mines Phase 3 Diagnostic Benchmark",
    "",
    `Phase: \`${args.phase}\``,
    `Seeds per cell: \`${args.seeds}\``,
    `Pre-committed informativeness threshold: mine-occupancy AUROC >= \`${args.aurocThreshold}\``,
    `Named envelope cell: \`${args.namedEnvelopePreset}\` / \`${args.namedEnvelopeCell}\``,
    "",
    "## Named Envelope Verdict",
    "",
    namedEnvelope
      ? `- ${namedEnvelope.disposition}: mean mine-occupancy AUROC ${namedEnvelope.meanMineOccupancyAuroc}, safe-tile adjacency AUROC ${namedEnvelope.meanSafeTileAdjacencyAuroc}, local-risk correlation ${namedEnvelope.meanPressureLocalRiskCorrelation}.`
      : "- Missing named envelope row.",
    "",
    "## Top Publishable Cells",
    "",
    "| preset | sensor cell | AUROC | adjacency AUROC | local-risk r | frontier safe@1 | disposition |",
    "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ...bestRows.map((row) => `| ${row.preset} | ${row.sensorCell} | ${row.meanMineOccupancyAuroc} | ${row.meanSafeTileAdjacencyAuroc} | ${row.meanPressureLocalRiskCorrelation} | ${row.meanFrontierSafePrecisionAt1} | ${row.disposition} |`),
    "",
    "## Chance / Collapse Cells",
    "",
    chanceRows.length > 0
      ? "| preset | sensor cell | AUROC | coverage | disposition |"
      : "No publishable cells landed in the chance-boundary band on this smoke slate.",
    chanceRows.length > 0 ? "| --- | --- | ---: | ---: | --- |" : "",
    ...chanceRows.map((row) => `| ${row.preset} | ${row.sensorCell} | ${row.meanMineOccupancyAuroc} | ${row.meanObservedCoverage} | ${row.disposition} |`),
    "",
    "## Scope",
    "",
    "This benchmark tests field informativeness only. It does not run a Sundog controller and does not promote Pressure Mines beyond Planned Workbench.",
    "",
  ].filter((line) => line !== "").join("\n") + "\n";
}

function arraysEqualWithNaN(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    const av = a[i];
    const bv = b[i];
    if (Number.isNaN(av) && Number.isNaN(bv)) continue;
    if (av !== bv) return false;
  }
  return true;
}

function runSelfChecks() {
  let floorThrew = false;
  try {
    normalizeSensorConfig({ sigmaNoise: 0 });
  } catch {
    floorThrew = true;
  }
  if (!floorThrew) {
    throw new Error("Expected sub-floor sensor config to throw without allowSubFloor");
  }
  normalizeSensorConfig({ sigmaNoise: 0, allowSubFloor: true });

  const board = initializeBoardState({ preset: "easy_sparse", seed: 77 });
  const runtime = createSensorRuntime({ delaySteps: 1, sensorSeed: 93 });
  const first = runtime.step(board);
  const second = runtime.step(board);
  if (!arraysEqualWithNaN(first.observed, second.observed)) {
    throw new Error("delaySteps=1 should return the previous observed snapshot on the second call");
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  runSelfChecks();

  const selectedCells = SENSOR_CELLS.filter((cell) => args.cells.includes(cell.name));
  const trialRows = [];
  for (const preset of args.presets) {
    for (const cell of selectedCells) {
      for (let i = 0; i < args.seeds; i += 1) {
        trialRows.push(scoreTrial({
          preset,
          seed: args.seedStart + i,
          cell,
        }));
      }
    }
  }

  const summaryRows = summarizeRows(trialRows).map((row) => ({
    ...row,
    disposition: cellDisposition(row, args.aurocThreshold),
  }));
  const namedEnvelope = summaryRows.find((row) => (
    row.preset === args.namedEnvelopePreset && row.sensorCell === args.namedEnvelopeCell
  )) ?? null;
  const bestRows = summaryRows
    .filter((row) => row.publishable)
    .sort((a, b) => (b.meanMineOccupancyAuroc ?? -Infinity) - (a.meanMineOccupancyAuroc ?? -Infinity))
    .slice(0, 8);
  const chanceRows = summaryRows.filter((row) => (
    row.publishable && row.disposition === "chance_boundary"
  ));

  const outDir = path.resolve(repoRoot, args.out);
  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "trial-rows.csv"), rowsToCsv(trialRows));
  await writeFile(path.join(outDir, "summary-rows.csv"), rowsToCsv(summaryRows));
  await writeFile(path.join(outDir, "summary.json"), JSON.stringify({
    schema: "sundog.mines.phase3-diagnostic.v1",
    args,
    sensorCells: selectedCells,
    namedEnvelope,
    bestRows,
    chanceRows,
  }, null, 2));
  await writeFile(path.join(outDir, "phase3-diagnostic.md"), markdownReport({
    args,
    summaryRows,
    namedEnvelope,
    bestRows,
    chanceRows,
  }));

  console.log(`Mines Phase 3 diagnostics wrote ${path.relative(repoRoot, outDir)}`);
  if (namedEnvelope) {
    console.log(`Named envelope ${namedEnvelope.disposition}: AUROC ${namedEnvelope.meanMineOccupancyAuroc}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
