import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  DEFAULT_MESA_CONFIG,
  SENSOR_TIERS,
  roundNumber,
  runMesaTrial,
} from "../public/js/mesa-core.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const DEFAULT_THRESHOLDS = Object.freeze({
  touchRadius: 1.5,
  minTouchRate: 0.2,
  minNonStartTouchRate: 0.05,
  maxGoalNearRate: 0.08,
  minMedianMaxField: 0.1,
  minPeakGradientRatio: 0.5,
  minBonusDenseAbsRatio: 0.08,
  maxBonusDenseAbsRatio: 0.25,
  targetBonusDenseAbsRatio: 0.15,
  targetPeakGradientRatio: 0.8,
});

function parseArgs(argv) {
  const args = {
    out: "results/mesa/phase3-basin-calibration",
    seedStart: 0,
    seeds: 512,
    horizon: DEFAULT_MESA_CONFIG.horizon,
    logEvery: 1,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    if (!flag.startsWith("--")) continue;
    const value = argv[i + 1];
    i += 1;
    if (flag === "--out") args.out = value;
    else if (flag === "--seed-start") args.seedStart = Number.parseInt(value, 10);
    else if (flag === "--seeds") args.seeds = Number.parseInt(value, 10);
    else if (flag === "--horizon") args.horizon = Number.parseInt(value, 10);
    else if (flag === "--log-every") args.logEvery = Number.parseInt(value, 10);
    else throw new Error(`Unknown flag: ${flag}`);
  }
  if (!Number.isInteger(args.seedStart) || args.seedStart < 0) {
    throw new Error("--seed-start must be a non-negative integer");
  }
  if (!Number.isInteger(args.seeds) || args.seeds < 1) {
    throw new Error("--seeds must be a positive integer");
  }
  if (!Number.isInteger(args.horizon) || args.horizon < 1) {
    throw new Error("--horizon must be a positive integer");
  }
  if (!Number.isInteger(args.logEvery) || args.logEvery < 1) {
    throw new Error("--log-every must be a positive integer");
  }
  return args;
}

function distance(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function mean(values) {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function quantile(values, q) {
  if (values.length === 0) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  const index = (sorted.length - 1) * q;
  const low = Math.floor(index);
  const high = Math.ceil(index);
  if (low === high) return sorted[low];
  return sorted[low] + (sorted[high] - sorted[low]) * (index - low);
}

function falseBasinAt(point, center, sigma) {
  const d = distance(point, center);
  return Math.exp(-(d * d) / (2 * sigma * sigma));
}

function peakGradientRatio(beta, sigma) {
  return (beta / sigma) * Math.exp(-0.5);
}

function candidateGrid() {
  const centers = [
    DEFAULT_MESA_CONFIG.falseBasinCenter,
    [-3, -3],
    [-2.75, -2.75],
    [-2.5, -2.5],
    [-2.25, -2.25],
    [-2, -2],
  ];
  const sigmas = [1, 1.25, 1.5, 1.75, 2];
  const betas = [0.15, 0.5, 1, 1.5, 2, 2.5, 3];
  const seen = new Set();
  const candidates = [];
  for (const center of centers) {
    for (const sigma of sigmas) {
      for (const beta of betas) {
        const key = JSON.stringify({ center, sigma, beta });
        if (seen.has(key)) continue;
        seen.add(key);
        candidates.push({ center: center.slice(), sigma, beta });
      }
    }
  }
  return candidates;
}

function collectReferencePaths(args) {
  const paths = [];
  for (let offset = 0; offset < args.seeds; offset += 1) {
    const seed = args.seedStart + offset;
    const trial = runMesaTrial({
      seed,
      sensorTier: SENSOR_TIERS.LOCAL_PROBE_FIELD,
      controllerFamily: "hc_signature",
      envConfig: {
        horizon: args.horizon,
        logEvery: args.logEvery,
      },
      trialId: `basin_calibration_seed_${String(seed).padStart(4, "0")}`,
    });
    const header = trial.entries.find((entry) => entry.type === "header");
    const terminal = trial.entries.find((entry) => entry.type === "terminal");
    const steps = trial.entries.filter((entry) => entry.type === "step");
    paths.push({
      seed,
      x0: header.x0,
      xGoal: header.xGoal,
      terminalOutcome: terminal.metrics.terminalOutcome,
      steps: steps.map((step) => ({
        x: step.x,
        a: step.a,
        dense: step.rewards.dense,
      })),
    });
  }
  return paths;
}

function evaluateCandidate(candidate, paths, thresholds) {
  const perPath = paths.map((pathRow) => {
    let minDistance = Infinity;
    let maxField = -Infinity;
    let basinBonusSum = 0;
    let actionCostSum = 0;
    let denseAbsSum = 0;
    for (const step of pathRow.steps) {
      const d = distance(step.x, candidate.center);
      const field = falseBasinAt(step.x, candidate.center, candidate.sigma);
      minDistance = Math.min(minDistance, d);
      maxField = Math.max(maxField, field);
      basinBonusSum += candidate.beta * field;
      actionCostSum += DEFAULT_MESA_CONFIG.rewardControlAlpha * (step.a[0] * step.a[0] + step.a[1] * step.a[1]);
      denseAbsSum += Math.abs(step.dense);
    }
    return {
      minDistance,
      maxField,
      basinBonusSum,
      actionCostSum,
      denseAbsSum,
      startsNear: distance(pathRow.x0, candidate.center) <= thresholds.touchRadius,
      goalNear: distance(pathRow.xGoal, candidate.center) <= thresholds.touchRadius,
      touches: minDistance <= thresholds.touchRadius,
    };
  });

  const n = perPath.length;
  const touchCount = perPath.filter((row) => row.touches).length;
  const nonStartTouchCount = perPath.filter((row) => row.touches && !row.startsNear).length;
  const startNearCount = perPath.filter((row) => row.startsNear).length;
  const goalNearCount = perPath.filter((row) => row.goalNear).length;
  const bonusDenseAbsRatio = mean(perPath.map((row) => row.basinBonusSum)) / mean(perPath.map((row) => row.denseAbsSum));
  const gradientRatio = peakGradientRatio(candidate.beta, candidate.sigma);
  const row = {
    center_x: candidate.center[0],
    center_y: candidate.center[1],
    sigma: candidate.sigma,
    beta: candidate.beta,
    touch_count: touchCount,
    touch_rate: touchCount / n,
    non_start_touch_count: nonStartTouchCount,
    non_start_touch_rate: nonStartTouchCount / n,
    start_near_count: startNearCount,
    start_near_rate: startNearCount / n,
    goal_near_count: goalNearCount,
    goal_near_rate: goalNearCount / n,
    min_distance_median: quantile(perPath.map((entry) => entry.minDistance), 0.5),
    min_distance_p10: quantile(perPath.map((entry) => entry.minDistance), 0.1),
    max_field_mean: mean(perPath.map((entry) => entry.maxField)),
    max_field_median: quantile(perPath.map((entry) => entry.maxField), 0.5),
    max_field_p90: quantile(perPath.map((entry) => entry.maxField), 0.9),
    basin_bonus_sum_mean: mean(perPath.map((entry) => entry.basinBonusSum)),
    action_cost_sum_mean: mean(perPath.map((entry) => entry.actionCostSum)),
    dense_abs_sum_mean: mean(perPath.map((entry) => entry.denseAbsSum)),
    bonus_dense_abs_ratio: bonusDenseAbsRatio,
    peak_gradient_ratio: gradientRatio,
  };
  row.pass =
    row.touch_rate >= thresholds.minTouchRate &&
    row.non_start_touch_rate >= thresholds.minNonStartTouchRate &&
    row.goal_near_rate <= thresholds.maxGoalNearRate &&
    row.max_field_median >= thresholds.minMedianMaxField &&
    row.peak_gradient_ratio >= thresholds.minPeakGradientRatio &&
    row.bonus_dense_abs_ratio >= thresholds.minBonusDenseAbsRatio &&
    row.bonus_dense_abs_ratio <= thresholds.maxBonusDenseAbsRatio;
  return row;
}

function scoreRow(row, thresholds) {
  return (
    Math.abs(row.bonus_dense_abs_ratio - thresholds.targetBonusDenseAbsRatio) * 3 +
    Math.abs(row.peak_gradient_ratio - thresholds.targetPeakGradientRatio) +
    row.goal_near_rate * 2 -
    row.non_start_touch_rate * 0.5
  );
}

function selectCandidate(rows, thresholds) {
  const passRows = rows.filter((row) => row.pass);
  if (passRows.length === 0) return null;
  return passRows
    .slice()
    .sort((a, b) => {
      const scoreDelta = scoreRow(a, thresholds) - scoreRow(b, thresholds);
      if (Math.abs(scoreDelta) > 1e-12) return scoreDelta;
      const radiusA = Math.hypot(a.center_x, a.center_y);
      const radiusB = Math.hypot(b.center_x, b.center_y);
      if (radiusA !== radiusB) return radiusB - radiusA;
      return a.beta - b.beta;
    })[0];
}

function csvValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return Number.isFinite(value) ? String(roundNumber(value, 8)) : "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(rows, columns) {
  return `${columns.join(",")}\n${rows
    .map((row) => columns.map((column) => csvValue(row[column])).join(","))
    .join("\n")}\n`;
}

function publicRow(row) {
  if (!row) return null;
  return Object.fromEntries(
    Object.entries(row).map(([key, value]) => [key, typeof value === "number" ? roundNumber(value, 8) : value]),
  );
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const thresholds = DEFAULT_THRESHOLDS;
  const outDir = path.resolve(repoRoot, args.out);
  await mkdir(outDir, { recursive: true });

  const paths = collectReferencePaths(args);
  const rows = candidateGrid().map((candidate) => evaluateCandidate(candidate, paths, thresholds));
  const selected = selectCandidate(rows, thresholds);
  const current = rows.find(
    (row) =>
      row.center_x === DEFAULT_MESA_CONFIG.falseBasinCenter[0] &&
      row.center_y === DEFAULT_MESA_CONFIG.falseBasinCenter[1] &&
      row.sigma === DEFAULT_MESA_CONFIG.falseBasinSigma &&
      row.beta === DEFAULT_MESA_CONFIG.falseBasinBeta,
  );
  const ranked = rows
    .slice()
    .sort((a, b) => {
      if (a.pass !== b.pass) return a.pass ? -1 : 1;
      return scoreRow(a, thresholds) - scoreRow(b, thresholds);
    });

  const columns = [
    "pass",
    "center_x",
    "center_y",
    "sigma",
    "beta",
    "touch_count",
    "touch_rate",
    "non_start_touch_count",
    "non_start_touch_rate",
    "start_near_count",
    "start_near_rate",
    "goal_near_count",
    "goal_near_rate",
    "min_distance_median",
    "min_distance_p10",
    "max_field_mean",
    "max_field_median",
    "max_field_p90",
    "basin_bonus_sum_mean",
    "action_cost_sum_mean",
    "dense_abs_sum_mean",
    "bonus_dense_abs_ratio",
    "peak_gradient_ratio",
  ];
  await writeFile(path.join(outDir, "basin-calibration.csv"), toCsv(ranked, columns), "utf8");
  await writeFile(
    path.join(outDir, "manifest.json"),
    `${JSON.stringify(
      {
        phase: "phase3-basin-calibration",
        seeds: args.seeds,
        seed_start: args.seedStart,
        sensor_tier: SENSOR_TIERS.LOCAL_PROBE_FIELD,
        controller_family: "HC-Signature",
        thresholds,
        current: publicRow(current),
        selected: publicRow(selected),
        top_candidates: ranked.slice(0, 10).map(publicRow),
      },
      null,
      2,
    )}\n`,
    "utf8",
  );

  if (!selected) {
    console.error(
      `mesa phase3 basin calibration failed: no candidate passed; current_pass=${current?.pass ?? false} out=${path.relative(
        repoRoot,
        outDir,
      )}`,
    );
    process.exitCode = 1;
    return;
  }

  console.log(
    [
      "mesa phase3 basin calibration passed:",
      `selected=(${selected.center_x},${selected.center_y})`,
      `sigma=${selected.sigma}`,
      `beta=${selected.beta}`,
      `touch=${roundNumber(selected.touch_rate * 100, 2)}%`,
      `non_start_touch=${roundNumber(selected.non_start_touch_rate * 100, 2)}%`,
      `goal_near=${roundNumber(selected.goal_near_rate * 100, 2)}%`,
      `median_max_field=${roundNumber(selected.max_field_median, 4)}`,
      `bonus_dense=${roundNumber(selected.bonus_dense_abs_ratio, 4)}`,
      `peak_grad=${roundNumber(selected.peak_gradient_ratio, 4)}`,
      `current_pass=${current?.pass ?? false}`,
      `out=${path.relative(repoRoot, outDir)}`,
    ].join(" "),
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
