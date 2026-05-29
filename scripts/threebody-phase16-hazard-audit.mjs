import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const CHANNELS = [
  { name: "energy", direction: 1 },
  { name: "kineticEnergy", direction: 1 },
  { name: "potentialEnergy", direction: -1 },
  { name: "virial", direction: 1 },
  { name: "inertia", direction: 1 },
  { name: "tidalMagnitude", direction: 1 },
  { name: "localAccelerationMagnitude", direction: 1 },
  { name: "radius", direction: 1 },
  { name: "minPrimaryDistance", direction: -1 },
  { name: "speed", direction: 1 },
];

const BOOTSTRAP_SEED = 160016;
const BOOTSTRAP_VALID_REPLICATES = 2000;
const BOOTSTRAP_MAX_ATTEMPTS = 20000;
const FAVORABLE_VELOCITY_MIN = 1.05;
const PASS_BAR = 0.70;
const CV_FOLDS = [[0, 1], [2, 3], [4, 5], [6, 7]];
const LOGISTIC_C = 1.0;
const LOGISTIC_MAX_ITER = 200;
const LOGISTIC_GRAD_TOL = 1e-8;

function parseArgs(argv) {
  const args = {
    inDir: "results/threebody/phase16-hazard-channel-audit-lock",
    out: null,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    i += 1;
    if (flag === "--in") args.inDir = value;
    else if (flag === "--out") args.out = value;
    else throw new Error(`Unknown flag: ${flag}`);
  }
  return args;
}

function csvValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "";
  if (typeof value === "boolean") return value ? "true" : "false";
  const text = String(value);
  if (/[",\n]/.test(text)) return `"${text.replaceAll('"', '""')}"`;
  return text;
}

function rowsToCsv(rows) {
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const lines = [columns.join(",")];
  for (const row of rows) lines.push(columns.map((column) => csvValue(row[column])).join(","));
  return `${lines.join("\n")}\n`;
}

function round(value, digits = 6) {
  if (!Number.isFinite(value)) return null;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function quantile(values, q) {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (finite.length === 0) return null;
  const pos = (finite.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return finite[lo];
  return finite[lo] + (finite[hi] - finite[lo]) * (pos - lo);
}

function makeRng(seed) {
  let t = seed >>> 0;
  return function rng() {
    t += 0x6d2b79f5;
    let x = t;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function computeAuroc(samples) {
  const clean = samples.filter((sample) => Number.isFinite(sample.score) && typeof sample.label === "boolean");
  const nPos = clean.filter((sample) => sample.label).length;
  const nNeg = clean.length - nPos;
  if (nPos === 0 || nNeg === 0) return null;

  const sorted = clean.slice().sort((a, b) => a.score - b.score);
  let rank = 1;
  let positiveRankSum = 0;
  for (let i = 0; i < sorted.length;) {
    let j = i + 1;
    while (j < sorted.length && sorted[j].score === sorted[i].score) j += 1;
    const averageRank = (rank + rank + (j - i) - 1) / 2;
    for (let k = i; k < j; k += 1) {
      if (sorted[k].label) positiveRankSum += averageRank;
    }
    rank += j - i;
    i = j;
  }

  return (positiveRankSum - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

async function readTrials(inDir) {
  const jsonlPath = path.join(repoRoot, inDir, "trials-minimal.jsonl");
  const text = await readFile(jsonlPath, "utf8");
  return text
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function trajectoryKey(trial) {
  return [
    trial.regime,
    trial.massRatio,
    trial.timestep,
    trial.radiusScale,
    trial.velocityScale,
    trial.seed,
  ].join("\t");
}

function cellKey(trial) {
  return [
    trial.regime,
    trial.massRatio,
    trial.timestep,
    trial.radiusScale,
    trial.velocityScale,
  ].join("\t");
}

function trialSamples(trial, channel, directional = true) {
  const direction = directional ? channel.direction : 1;
  return (trial.hazardSamples ?? [])
    .map((sample) => ({
      label: sample.label,
      score: direction * Number(sample.channels?.[channel.name]),
      trajectoryKey: trajectoryKey(trial),
      cellKey: cellKey(trial),
      seed: trial.seed,
    }))
    .filter((sample) => Number.isFinite(sample.score) && typeof sample.label === "boolean");
}

function allSamples(trials, channel, predicate = () => true, directional = true) {
  return trials
    .filter((trial) => trial.controllerMode === "off" && predicate(trial))
    .flatMap((trial) => trialSamples(trial, channel, directional));
}

function groupSamplesByTrajectory(samples) {
  const groups = new Map();
  for (const sample of samples) {
    if (!groups.has(sample.trajectoryKey)) groups.set(sample.trajectoryKey, []);
    groups.get(sample.trajectoryKey).push(sample);
  }
  return [...groups.values()];
}

function bootstrapCiFromGroups(groups, scoreMapper = (sample) => sample.score) {
  if (groups.length === 0) {
    return { lo: null, hi: null, validReplicates: 0, attempts: 0, status: "undecidable" };
  }
  const rng = makeRng(BOOTSTRAP_SEED);
  const values = [];
  let attempts = 0;
  while (values.length < BOOTSTRAP_VALID_REPLICATES && attempts < BOOTSTRAP_MAX_ATTEMPTS) {
    attempts += 1;
    const resampled = [];
    for (let i = 0; i < groups.length; i += 1) {
      const group = groups[Math.floor(rng() * groups.length)];
      for (const sample of group) {
        resampled.push({ label: sample.label, score: scoreMapper(sample) });
      }
    }
    const auc = computeAuroc(resampled);
    if (auc !== null) values.push(auc);
  }
  if (values.length < BOOTSTRAP_VALID_REPLICATES) {
    return {
      lo: null,
      hi: null,
      validReplicates: values.length,
      attempts,
      status: "undecidable",
    };
  }
  return {
    lo: quantile(values, 0.025),
    hi: quantile(values, 0.975),
    validReplicates: values.length,
    attempts,
    status: "defined",
  };
}

function perCellMeanAuroc(samples) {
  const byCell = new Map();
  for (const sample of samples) {
    if (!byCell.has(sample.cellKey)) byCell.set(sample.cellKey, []);
    byCell.get(sample.cellKey).push(sample);
  }
  const aucs = [...byCell.values()]
    .map((cellSamples) => computeAuroc(cellSamples))
    .filter(Number.isFinite);
  if (aucs.length === 0) return { value: null, definedCells: 0, totalCells: byCell.size };
  return {
    value: aucs.reduce((sum, value) => sum + value, 0) / aucs.length,
    definedCells: aucs.length,
    totalCells: byCell.size,
  };
}

function standardizeTrainTest(trainRows, testRows) {
  const n = CHANNELS.length;
  const means = Array(n).fill(0);
  const stds = Array(n).fill(1);
  for (let j = 0; j < n; j += 1) {
    const values = trainRows.map((row) => row.x[j]);
    means[j] = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + (value - means[j]) ** 2, 0) / values.length;
    stds[j] = Math.sqrt(variance) || 1;
  }
  const apply = (row) => ({
    ...row,
    x: row.x.map((value, j) => (value - means[j]) / stds[j]),
  });
  return { train: trainRows.map(apply), test: testRows.map(apply) };
}

function sigmoid(z) {
  if (z >= 0) {
    const expNeg = Math.exp(-z);
    return 1 / (1 + expNeg);
  }
  const expPos = Math.exp(z);
  return expPos / (1 + expPos);
}

function solveLinearSystem(matrix, vector) {
  const n = vector.length;
  const a = matrix.map((row, i) => [...row, vector[i]]);
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    if (Math.abs(a[pivot][col]) < 1e-12) return null;
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const pivotValue = a[col][col];
    for (let c = col; c <= n; c += 1) a[col][c] /= pivotValue;
    for (let row = 0; row < n; row += 1) {
      if (row === col) continue;
      const factor = a[row][col];
      for (let c = col; c <= n; c += 1) a[row][c] -= factor * a[col][c];
    }
  }
  return a.map((row) => row[n]);
}

function fitLogistic(rows) {
  const featureCount = CHANNELS.length;
  const paramCount = featureCount + 1;
  const lambda = 1 / LOGISTIC_C;
  const weights = Array(paramCount).fill(0);
  for (let iter = 0; iter < LOGISTIC_MAX_ITER; iter += 1) {
    const grad = Array(paramCount).fill(0);
    const hessian = Array.from({ length: paramCount }, () => Array(paramCount).fill(0));
    for (const row of rows) {
      const x = [1, ...row.x];
      const p = sigmoid(x.reduce((sum, value, j) => sum + value * weights[j], 0));
      const y = row.label ? 1 : 0;
      for (let j = 0; j < paramCount; j += 1) {
        grad[j] += (p - y) * x[j] / rows.length;
        for (let k = 0; k < paramCount; k += 1) {
          hessian[j][k] += p * (1 - p) * x[j] * x[k] / rows.length;
        }
      }
    }
    for (let j = 1; j < paramCount; j += 1) {
      grad[j] += lambda * weights[j];
      hessian[j][j] += lambda;
    }
    const gradNorm = Math.sqrt(grad.reduce((sum, value) => sum + value * value, 0));
    if (gradNorm < LOGISTIC_GRAD_TOL) return { weights, converged: true, iterations: iter };
    const step = solveLinearSystem(hessian, grad);
    if (!step) return { weights, converged: false, iterations: iter, reason: "singular_hessian" };
    for (let j = 0; j < paramCount; j += 1) weights[j] -= step[j];
  }
  return { weights, converged: false, iterations: LOGISTIC_MAX_ITER, reason: "max_iter" };
}

function predict(row, weights) {
  return sigmoid(weights[0] + row.x.reduce((sum, value, j) => sum + value * weights[j + 1], 0));
}

function logisticRows(trials, predicate = () => true) {
  return trials
    .filter((trial) => trial.controllerMode === "off" && predicate(trial))
    .flatMap((trial) => (trial.hazardSamples ?? []).map((sample) => ({
      label: sample.label,
      seed: trial.seed,
      trajectoryKey: trajectoryKey(trial),
      x: CHANNELS.map((channel) => Number(sample.channels?.[channel.name])),
    })))
    .filter((row) => typeof row.label === "boolean" && row.x.every(Number.isFinite));
}

function heldOutSingleChannelAuroc(rows, channelIndex) {
  const heldOut = [];
  for (const foldSeeds of CV_FOLDS) {
    const foldSeedSet = new Set(foldSeeds);
    for (const row of rows) {
      if (foldSeedSet.has(row.seed)) {
        heldOut.push({
          label: row.label,
          score: CHANNELS[channelIndex].direction * row.x[channelIndex],
        });
      }
    }
  }
  return computeAuroc(heldOut);
}

function logisticCrossValidatedRows(rows) {
  if (rows.length === 0) return { predictions: [], converged: false, reason: "no_rows" };
  const predictions = [];
  const foldReports = [];
  for (const foldSeeds of CV_FOLDS) {
    const foldSeedSet = new Set(foldSeeds);
    const trainRows = rows.filter((row) => !foldSeedSet.has(row.seed));
    const testRows = rows.filter((row) => foldSeedSet.has(row.seed));
    if (trainRows.length === 0 || testRows.length === 0) {
      return { predictions, converged: false, reason: "empty_fold" };
    }
    if (computeAuroc(trainRows.map((row) => ({ label: row.label, score: row.label ? 1 : 0 }))) === null) {
      return { predictions, converged: false, reason: "one_class_train_fold" };
    }
    const standardized = standardizeTrainTest(trainRows, testRows);
    const fit = fitLogistic(standardized.train);
    foldReports.push({ foldSeeds: foldSeeds.join("|"), ...fit });
    if (!fit.converged) return { predictions, foldReports, converged: false, reason: fit.reason ?? "not_converged" };
    for (const row of standardized.test) {
      predictions.push({
        label: row.label,
        score: predict(row, fit.weights),
        trajectoryKey: row.trajectoryKey,
      });
    }
  }
  return { predictions, foldReports, converged: true };
}

function inSampleLogisticAuroc(rows) {
  if (rows.length === 0) return { auc: null, converged: false };
  const standardized = standardizeTrainTest(rows, rows);
  const fit = fitLogistic(standardized.train);
  if (!fit.converged) return { auc: null, converged: false, reason: fit.reason };
  const predictions = standardized.test.map((row) => ({ label: row.label, score: predict(row, fit.weights) }));
  return { auc: computeAuroc(predictions), converged: true, iterations: fit.iterations };
}

function channelRow(trials, channel) {
  const favorable = allSamples(trials, channel, (trial) => trial.velocityScale >= FAVORABLE_VELOCITY_MIN);
  const fullGrid = allSamples(trials, channel);
  const reverseFavorable = favorable.map((sample) => ({ ...sample, score: -sample.score }));
  const favorableAuroc = computeAuroc(favorable);
  const reverseAuroc = computeAuroc(reverseFavorable);
  const discriminability = favorableAuroc === null ? null : Math.max(favorableAuroc, 1 - favorableAuroc);
  const ci = bootstrapCiFromGroups(groupSamplesByTrajectory(favorable));
  const perCell = perCellMeanAuroc(favorable);
  const logisticBaseRows = logisticRows(trials, (trial) => trial.velocityScale >= FAVORABLE_VELOCITY_MIN);
  const channelIndex = CHANNELS.findIndex((candidate) => candidate.name === channel.name);
  const heldOut = heldOutSingleChannelAuroc(logisticBaseRows, channelIndex);
  return {
    rowType: "single_channel",
    channel: channel.name,
    direction: channel.direction > 0 ? "+" : "-",
    favorableSampleCount: favorable.length,
    favorablePositiveCount: favorable.filter((sample) => sample.label).length,
    directionalAuroc: round(favorableAuroc),
    ciLo: round(ci.lo),
    ciHi: round(ci.hi),
    ciStatus: ci.status,
    bootstrapValidReplicates: ci.validReplicates,
    bootstrapAttempts: ci.attempts,
    lowerCiPass: ci.lo !== null && ci.lo >= PASS_BAR,
    discriminability: round(discriminability),
    reverseDirectionAuroc: round(reverseAuroc),
    signMisregistered: discriminability !== null && discriminability >= PASS_BAR && favorableAuroc < 0.5,
    perCellMeanAuroc: round(perCell.value),
    perCellDefinedCount: perCell.definedCells,
    perCellTotalCount: perCell.totalCells,
    fullGridAuroc: round(computeAuroc(fullGrid)),
    heldOutFoldAuroc: round(heldOut),
  };
}

function comboRow(trials) {
  const rows = logisticRows(trials, (trial) => trial.velocityScale >= FAVORABLE_VELOCITY_MIN);
  const cv = logisticCrossValidatedRows(rows);
  const inSample = inSampleLogisticAuroc(rows);
  const auc = cv.converged ? computeAuroc(cv.predictions) : null;
  const ci = cv.converged ? bootstrapCiFromGroups(groupSamplesByTrajectory(cv.predictions)) : {
    lo: null,
    hi: null,
    status: "undecidable",
    validReplicates: 0,
    attempts: 0,
  };
  return {
    rowType: "fitted_combo",
    channel: "l2_logistic_combo",
    direction: "fit",
    favorableSampleCount: rows.length,
    favorablePositiveCount: rows.filter((row) => row.label).length,
    directionalAuroc: round(auc),
    ciLo: round(ci.lo),
    ciHi: round(ci.hi),
    ciStatus: cv.converged ? ci.status : "undecidable",
    bootstrapValidReplicates: ci.validReplicates,
    bootstrapAttempts: ci.attempts,
    lowerCiPass: ci.lo !== null && ci.lo >= PASS_BAR,
    heldOutFoldAuroc: round(auc),
    inSampleAuroc: round(inSample.auc),
    overfitGap: round(inSample.auc !== null && auc !== null ? inSample.auc - auc : null),
    comboConverged: cv.converged && inSample.converged,
    comboReason: cv.converged ? "" : cv.reason,
    folds: CV_FOLDS.map((fold) => fold.join("|")).join(";"),
    C: LOGISTIC_C,
    maxIter: LOGISTIC_MAX_ITER,
    gradTol: LOGISTIC_GRAD_TOL,
  };
}

function classifyBranch(rows) {
  const singles = rows.filter((row) => row.rowType === "single_channel");
  const combo = rows.find((row) => row.rowType === "fitted_combo");
  const singlePasses = singles.filter((row) => row.lowerCiPass);
  if (singlePasses.length > 0) return "A_hazard_warnable";
  if (rows.some((row) => row.ciStatus === "undecidable" || (
    Number.isFinite(row.ciLo) && row.ciLo < PASS_BAR && Number.isFinite(row.ciHi) && row.ciHi >= PASS_BAR
  ))) {
    return "C_mixed_provisional";
  }
  if (combo?.lowerCiPass) return "C_mixed_provisional";
  return "B_warnability_capped";
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const inDir = path.normalize(args.inDir);
  const outPath = args.out
    ? path.resolve(repoRoot, args.out)
    : path.join(repoRoot, inDir, "hazard-channel-audit.csv");
  const trials = await readTrials(inDir);
  const rows = CHANNELS.map((channel) => channelRow(trials, channel));
  rows.push(comboRow(trials));
  const branch = classifyBranch(rows);
  const manifest = {
    schema: "sundog.threebody.phase16-hazard-channel-audit.v1",
    generatedAt: new Date().toISOString(),
    input: inDir,
    output: path.relative(repoRoot, outPath),
    branch,
    passBar: PASS_BAR,
    bootstrapSeed: BOOTSTRAP_SEED,
    bootstrapValidReplicates: BOOTSTRAP_VALID_REPLICATES,
    bootstrapMaxAttempts: BOOTSTRAP_MAX_ATTEMPTS,
    folds: CV_FOLDS,
    logistic: {
      C: LOGISTIC_C,
      maxIter: LOGISTIC_MAX_ITER,
      gradTol: LOGISTIC_GRAD_TOL,
      intercept: "included_unpenalized",
      classWeights: "none",
    },
  };
  await mkdir(path.dirname(outPath), { recursive: true });
  await writeFile(outPath, rowsToCsv(rows), "utf8");
  await writeFile(path.join(path.dirname(outPath), "hazard-channel-audit-manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  console.log(`[threebody] wrote hazard-channel-audit.csv to ${path.relative(repoRoot, outPath)}`);
  console.log(`[threebody] branch ${branch}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
