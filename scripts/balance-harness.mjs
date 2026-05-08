import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  BALANCE_PRESETS,
  clamp,
  computeBalanceControl,
  createBalanceRuntime,
  initializeBalanceState,
  integrateBalanceStep,
  makeRng,
  normalizeBalanceConfig,
  roundNumber,
  sampleShadowSensor,
  serializeBalanceSample,
} from "../public/js/balance-core.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const MODE_DEFINITIONS = Object.freeze({
  passive: "No control force.",
  naive_cart: "Cart-centering baseline with proprioception only.",
  naive_shadow: "Shadow residual baseline without dynamics or observability gating.",
  sundog_shadow: "Prototype Sundog shadow controller: residual and residual velocity gated by shadow confidence.",
  oracle: "Privileged true-angle controller; diagnostic ceiling, not an allowed Sundog input.",
});

function parseList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function parseNumberList(value) {
  return parseList(value).map((item) => Number.parseFloat(item));
}

function parseArgs(argv) {
  const args = {
    phase: "phase7-smoke",
    out: "results/balance/phase7-smoke",
    seedStart: 0,
    seeds: 12,
    presets: ["easy", "recoverable", "near_fall", "noisy_shadow", "delayed_shadow"],
    modes: ["passive", "naive_cart", "naive_shadow", "sundog_shadow", "oracle"],
    lightElevations: [28],
    duration: 10,
    dt: 1 / 120,
    forceLimit: 12,
    logEvery: 30,
    jitterScale: 1,
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
    else if (flag === "--light-elevations") args.lightElevations = parseNumberList(value);
    else if (flag === "--duration") args.duration = Number.parseFloat(value);
    else if (flag === "--dt") args.dt = Number.parseFloat(value);
    else if (flag === "--force-limit") args.forceLimit = Number.parseFloat(value);
    else if (flag === "--log-every") args.logEvery = Number.parseInt(value, 10);
    else if (flag === "--jitter-scale") args.jitterScale = Number.parseFloat(value);
    else throw new Error(`Unknown flag: ${flag}`);
  }

  if (!Number.isInteger(args.seedStart) || args.seedStart < 0) {
    throw new Error("--seed-start must be a non-negative integer");
  }
  if (!Number.isInteger(args.seeds) || args.seeds < 1) {
    throw new Error("--seeds must be a positive integer");
  }
  if (!Number.isFinite(args.duration) || args.duration <= 0) {
    throw new Error("--duration must be positive");
  }
  if (!Number.isFinite(args.dt) || args.dt <= 0) {
    throw new Error("--dt must be positive");
  }
  if (!Number.isFinite(args.forceLimit) || args.forceLimit <= 0) {
    throw new Error("--force-limit must be positive");
  }
  if (!Number.isInteger(args.logEvery) || args.logEvery < 1) {
    throw new Error("--log-every must be a positive integer");
  }
  if (!Number.isFinite(args.jitterScale) || args.jitterScale < 0) {
    throw new Error("--jitter-scale must be non-negative");
  }
  if (args.presets.some((preset) => !BALANCE_PRESETS[preset])) {
    throw new Error(`Unknown preset in --presets: ${args.presets.join(",")}`);
  }
  if (args.modes.some((mode) => !MODE_DEFINITIONS[mode])) {
    throw new Error(`Unknown mode in --modes: ${args.modes.join(",")}`);
  }
  if (args.lightElevations.some((value) => !Number.isFinite(value) || value <= 1 || value >= 89)) {
    throw new Error("--light-elevations values must be between 1 and 89 degrees");
  }

  return args;
}

function stringHash(text) {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function seededInitialState(presetName, seed, jitterScale) {
  const base = BALANCE_PRESETS[presetName].state;
  const rng = makeRng(((seed + 1) * 2654435761) ^ stringHash(presetName));
  const jitter = (span) => (rng() - 0.5) * span * jitterScale;

  return {
    x: clamp(base.x + jitter(0.08), -1.9, 1.9),
    xDot: base.xDot + jitter(0.1),
    theta: clamp(base.theta + jitter(0.04), -0.55, 0.55),
    thetaDot: base.thetaDot + jitter(0.08),
  };
}

function trialId({ preset, mode, seed, lightElevationDeg }) {
  return [
    preset,
    mode,
    `elev_${String(lightElevationDeg).replace(".", "p")}`,
    `seed_${String(seed).padStart(3, "0")}`,
  ].join("_");
}

function terminalOutcome(state, cfg) {
  if (state.fallen) return "fallen";
  if (state.railHit) return "rail_hit";
  if (state.t >= cfg.duration) return "timeout";
  return "max_steps";
}

function runTrial(args, { preset, mode, seed, lightElevationDeg }) {
  const initialState = seededInitialState(preset, seed, args.jitterScale);
  const cfg = normalizeBalanceConfig({
    preset,
    controllerMode: mode,
    seed,
    initialState,
    duration: args.duration,
    dt: args.dt,
    forceLimit: args.forceLimit,
    lightElevationDeg,
  });
  let state = initializeBalanceState(cfg);
  const runtime = createBalanceRuntime(cfg);
  const controllerState = {};
  const samples = [];
  let sensor = sampleShadowSensor(state, runtime, cfg);
  let control = { force: 0, rawForce: 0, saturated: false, phase: "PASSIVE", reason: "initial" };
  let thetaSquareSum = 0;
  let confidenceSum = 0;
  let absForceSum = 0;
  let maxAbsTheta = Math.abs(state.theta);
  let maxAbsX = Math.abs(state.x);
  let saturationCount = 0;
  let confidenceLossCount = 0;
  let steps = 0;
  const maxSteps = Math.ceil(args.duration / args.dt);
  const id = trialId({ preset, mode, seed, lightElevationDeg });

  while (steps < maxSteps && !state.fallen && !state.railHit && state.t < cfg.duration) {
    sensor = sampleShadowSensor(state, runtime, cfg);
    control = computeBalanceControl(state, sensor, controllerState, cfg);

    thetaSquareSum += state.theta * state.theta;
    confidenceSum += sensor.confidence;
    absForceSum += Math.abs(control.force);
    maxAbsTheta = Math.max(maxAbsTheta, Math.abs(state.theta));
    maxAbsX = Math.max(maxAbsX, Math.abs(state.x));
    if (control.saturated) saturationCount += 1;
    if (!sensor.valid || sensor.confidence < 0.35) confidenceLossCount += 1;

    if (steps % args.logEvery === 0) {
      samples.push({
        phase: args.phase,
        trialId: id,
        preset,
        seed,
        lightElevationDeg,
        ...serializeBalanceSample(state, sensor, control, cfg),
      });
    }

    state = integrateBalanceStep(state, control.force, cfg);
    steps += 1;
  }

  const outcome = terminalOutcome(state, cfg);
  const simulatedTime = Math.min(state.t, cfg.duration);
  const denom = Math.max(1, steps);
  const result = {
    phase: args.phase,
    trialId: id,
    preset,
    mode,
    seed,
    lightElevationDeg,
    outcome,
    success: outcome === "timeout",
    simulatedTime: roundNumber(simulatedTime),
    normalizedSurvival: roundNumber(simulatedTime / cfg.duration),
    rmsTheta: roundNumber(Math.sqrt(thetaSquareSum / denom)),
    maxAbsTheta: roundNumber(maxAbsTheta),
    maxAbsX: roundNumber(maxAbsX),
    meanShadowConfidence: roundNumber(confidenceSum / denom),
    meanAbsForce: roundNumber(absForceSum / denom),
    saturationCount,
    confidenceLossCount,
    steps,
    initialX: roundNumber(initialState.x),
    initialXDot: roundNumber(initialState.xDot),
    initialTheta: roundNumber(initialState.theta),
    initialThetaDot: roundNumber(initialState.thetaDot),
  };

  samples.push({
    phase: args.phase,
    trialId: id,
    preset,
    seed,
    lightElevationDeg,
    ...serializeBalanceSample(state, sensor, control, cfg),
    terminalOutcome: outcome,
  });

  return { result, samples };
}

function mean(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) return null;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function rate(rows, predicate) {
  if (rows.length === 0) return null;
  return rows.filter(predicate).length / rows.length;
}

function groupBy(rows, keyFn) {
  const groups = new Map();
  for (const row of rows) {
    const key = keyFn(row);
    const existing = groups.get(key) ?? [];
    existing.push(row);
    groups.set(key, existing);
  }
  return groups;
}

function summarize(results) {
  const groups = groupBy(results, (row) => [row.phase, row.preset, row.mode, row.lightElevationDeg].join("|"));
  return Array.from(groups.entries()).map(([key, rows]) => {
    const [phase, preset, mode, lightElevationDeg] = key.split("|");
    return {
      phase,
      preset,
      mode,
      lightElevationDeg: Number.parseFloat(lightElevationDeg),
      n: rows.length,
      successRate: roundNumber(rate(rows, (row) => row.success)),
      fallRate: roundNumber(rate(rows, (row) => row.outcome === "fallen")),
      railHitRate: roundNumber(rate(rows, (row) => row.outcome === "rail_hit")),
      meanSurvival: roundNumber(mean(rows.map((row) => row.simulatedTime))),
      meanNormalizedSurvival: roundNumber(mean(rows.map((row) => row.normalizedSurvival))),
      meanRmsTheta: roundNumber(mean(rows.map((row) => row.rmsTheta))),
      meanMaxAbsTheta: roundNumber(mean(rows.map((row) => row.maxAbsTheta))),
      meanShadowConfidence: roundNumber(mean(rows.map((row) => row.meanShadowConfidence))),
      meanSaturationCount: roundNumber(mean(rows.map((row) => row.saturationCount))),
      meanConfidenceLossCount: roundNumber(mean(rows.map((row) => row.confidenceLossCount))),
    };
  }).sort((a, b) => (
    a.preset.localeCompare(b.preset)
    || a.lightElevationDeg - b.lightElevationDeg
    || a.mode.localeCompare(b.mode)
  ));
}

function compareSurvival(subject, baseline) {
  if (!baseline) return "";
  const delta = subject.simulatedTime - baseline.simulatedTime;
  if (delta > 0.05) return "better";
  if (delta < -0.05) return "worse";
  return "tie";
}

function makeComparisons(results) {
  const groups = groupBy(results, (row) => [row.preset, row.seed, row.lightElevationDeg].join("|"));
  const rows = [];
  for (const trialRows of groups.values()) {
    const byMode = new Map(trialRows.map((row) => [row.mode, row]));
    const passive = byMode.get("passive");
    const naiveShadow = byMode.get("naive_shadow");
    for (const row of trialRows) {
      rows.push({
        phase: row.phase,
        preset: row.preset,
        seed: row.seed,
        lightElevationDeg: row.lightElevationDeg,
        mode: row.mode,
        outcome: row.outcome,
        simulatedTime: row.simulatedTime,
        vsPassiveSurvivalDelta: passive ? roundNumber(row.simulatedTime - passive.simulatedTime) : "",
        vsNaiveShadowSurvivalDelta: naiveShadow ? roundNumber(row.simulatedTime - naiveShadow.simulatedTime) : "",
        vsPassive: compareSurvival(row, passive),
        vsNaiveShadow: compareSurvival(row, naiveShadow),
      });
    }
  }
  return rows.sort((a, b) => (
    a.preset.localeCompare(b.preset)
    || a.seed - b.seed
    || a.lightElevationDeg - b.lightElevationDeg
    || a.mode.localeCompare(b.mode)
  ));
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const normalized = typeof value === "number" ? roundNumber(value) : value;
  const text = String(normalized);
  if (/[",\r\n]/.test(text)) return `"${text.replaceAll("\"", "\"\"")}"`;
  return text;
}

function toCsv(rows, columns) {
  const lines = [columns.join(",")];
  for (const row of rows) {
    lines.push(columns.map((column) => csvEscape(row[column])).join(","));
  }
  return `${lines.join("\n")}\n`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(repoRoot, args.out);
  const results = [];
  const samples = [];

  for (const preset of args.presets) {
    for (const lightElevationDeg of args.lightElevations) {
      for (let i = 0; i < args.seeds; i += 1) {
        const seed = args.seedStart + i;
        for (const mode of args.modes) {
          const trial = runTrial(args, { preset, mode, seed, lightElevationDeg });
          results.push(trial.result);
          samples.push(...trial.samples);
        }
      }
    }
  }

  const summaryRows = summarize(results);
  const comparisonRows = makeComparisons(results);
  const manifest = {
    schema: "sundog.balance.phase7-smoke.v1",
    generatedAt: new Date().toISOString(),
    phase: args.phase,
    modes: args.modes,
    presets: args.presets,
    lightElevations: args.lightElevations,
    seedStart: args.seedStart,
    seeds: args.seeds,
    duration: args.duration,
    dt: args.dt,
    forceLimit: args.forceLimit,
    jitterScale: args.jitterScale,
    trialCount: results.length,
    sampleCount: samples.length,
    modeDefinitions: MODE_DEFINITIONS,
    note: "Ignored local smoke output. This is a Phase 7 reproducibility scaffold, not public evidence.",
  };

  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  await writeFile(path.join(outDir, "samples.jsonl"), `${samples.map((row) => JSON.stringify(row)).join("\n")}\n`);
  await writeFile(path.join(outDir, "trial-outcomes.csv"), toCsv(results, [
    "phase",
    "trialId",
    "preset",
    "mode",
    "seed",
    "lightElevationDeg",
    "outcome",
    "success",
    "simulatedTime",
    "normalizedSurvival",
    "rmsTheta",
    "maxAbsTheta",
    "maxAbsX",
    "meanShadowConfidence",
    "meanAbsForce",
    "saturationCount",
    "confidenceLossCount",
    "steps",
    "initialX",
    "initialXDot",
    "initialTheta",
    "initialThetaDot",
  ]));
  await writeFile(path.join(outDir, "summary.csv"), toCsv(summaryRows, [
    "phase",
    "preset",
    "mode",
    "lightElevationDeg",
    "n",
    "successRate",
    "fallRate",
    "railHitRate",
    "meanSurvival",
    "meanNormalizedSurvival",
    "meanRmsTheta",
    "meanMaxAbsTheta",
    "meanShadowConfidence",
    "meanSaturationCount",
    "meanConfidenceLossCount",
  ]));
  await writeFile(path.join(outDir, "comparison.csv"), toCsv(comparisonRows, [
    "phase",
    "preset",
    "seed",
    "lightElevationDeg",
    "mode",
    "outcome",
    "simulatedTime",
    "vsPassiveSurvivalDelta",
    "vsNaiveShadowSurvivalDelta",
    "vsPassive",
    "vsNaiveShadow",
  ]));

  const sundogRows = summaryRows.filter((row) => row.mode === "sundog_shadow");
  console.log(`Balance ${args.phase}: ${results.length} trials, ${samples.length} sampled rows`);
  console.log(`Wrote ${path.relative(repoRoot, outDir)}`);
  for (const row of sundogRows) {
    console.log(`${row.preset} sundog_shadow: success ${row.successRate}, mean survival ${row.meanSurvival}s`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
