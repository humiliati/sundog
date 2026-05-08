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
    phase: "phase8-recovery",
    out: "results/balance/phase8-recovery",
    seedStart: 0,
    seeds: 12,
    presets: ["easy", "recoverable", "near_fall", "noisy_shadow", "delayed_shadow"],
    modes: ["passive", "naive_cart", "naive_shadow", "sundog_shadow", "oracle"],
    lightElevations: [28],
    duration: 10,
    dt: 1 / 120,
    forceLimit: 12,
    sensorNoiseStd: null,
    sensorDelaySteps: null,
    jitterScale: 1,
    logEvery: 6,
    curveBin: 0.1,
    disturbanceAt: 0.25,
    disturbanceDuration: 0.15,
    disturbanceForce: 4.5,
    disturbanceMode: "adversarial",
    nearFallTheta: 0.45,
    recoveryTheta: 0.09,
    recoveryThetaDot: 0.35,
    recoveryHold: 0.25,
    confidenceLossThreshold: 0.35,
    defaultResidualWarningThreshold: 0.16,
    defaultVelocityWarningThreshold: 0.75,
    residualWarningThresholds: [0.04, 0.08, 0.12, 0.16, 0.24, 0.32, 0.44],
    velocityWarningThresholds: [0.25, 0.5, 0.75, 1, 1.5, 2.5, 4],
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
    else if (flag === "--sensor-noise-std" || flag === "--noise") args.sensorNoiseStd = Number.parseFloat(value);
    else if (flag === "--sensor-delay-steps" || flag === "--delay") args.sensorDelaySteps = Number.parseInt(value, 10);
    else if (flag === "--jitter-scale") args.jitterScale = Number.parseFloat(value);
    else if (flag === "--log-every") args.logEvery = Number.parseInt(value, 10);
    else if (flag === "--curve-bin") args.curveBin = Number.parseFloat(value);
    else if (flag === "--disturbance-at") args.disturbanceAt = Number.parseFloat(value);
    else if (flag === "--disturbance-duration") args.disturbanceDuration = Number.parseFloat(value);
    else if (flag === "--disturbance-force") args.disturbanceForce = Number.parseFloat(value);
    else if (flag === "--disturbance-mode") args.disturbanceMode = value;
    else if (flag === "--near-fall-theta") args.nearFallTheta = Number.parseFloat(value);
    else if (flag === "--recovery-theta") args.recoveryTheta = Number.parseFloat(value);
    else if (flag === "--recovery-theta-dot") args.recoveryThetaDot = Number.parseFloat(value);
    else if (flag === "--recovery-hold") args.recoveryHold = Number.parseFloat(value);
    else if (flag === "--confidence-loss-threshold") args.confidenceLossThreshold = Number.parseFloat(value);
    else if (flag === "--default-residual-warning-threshold") args.defaultResidualWarningThreshold = Number.parseFloat(value);
    else if (flag === "--default-velocity-warning-threshold") args.defaultVelocityWarningThreshold = Number.parseFloat(value);
    else if (flag === "--residual-warning-thresholds") args.residualWarningThresholds = parseNumberList(value);
    else if (flag === "--velocity-warning-thresholds") args.velocityWarningThresholds = parseNumberList(value);
    else throw new Error(`Unknown flag: ${flag}`);
  }

  validateArgs(args);
  args.residualWarningThresholds = [...new Set(args.residualWarningThresholds)].sort((a, b) => a - b);
  args.velocityWarningThresholds = [...new Set(args.velocityWarningThresholds)].sort((a, b) => a - b);
  return args;
}

function validateArgs(args) {
  const positiveNumbers = [
    ["--seeds", args.seeds],
    ["--duration", args.duration],
    ["--dt", args.dt],
    ["--force-limit", args.forceLimit],
    ["--log-every", args.logEvery],
    ["--curve-bin", args.curveBin],
    ["--disturbance-duration", args.disturbanceDuration],
    ["--disturbance-force", args.disturbanceForce],
    ["--near-fall-theta", args.nearFallTheta],
    ["--recovery-theta", args.recoveryTheta],
    ["--recovery-theta-dot", args.recoveryThetaDot],
    ["--recovery-hold", args.recoveryHold],
    ["--confidence-loss-threshold", args.confidenceLossThreshold],
    ["--default-residual-warning-threshold", args.defaultResidualWarningThreshold],
    ["--default-velocity-warning-threshold", args.defaultVelocityWarningThreshold],
  ];

  if (!Number.isInteger(args.seedStart) || args.seedStart < 0) {
    throw new Error("--seed-start must be a non-negative integer");
  }
  if (!Number.isInteger(args.seeds) || args.seeds < 1) {
    throw new Error("--seeds must be a positive integer");
  }
  if (!Number.isInteger(args.logEvery) || args.logEvery < 1) {
    throw new Error("--log-every must be a positive integer");
  }
  for (const [flag, value] of positiveNumbers) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`${flag} must be positive`);
  }
  if (!Number.isFinite(args.disturbanceAt) || args.disturbanceAt < 0) {
    throw new Error("--disturbance-at must be non-negative");
  }
  if (!["adversarial", "fixed"].includes(args.disturbanceMode)) {
    throw new Error("--disturbance-mode must be adversarial or fixed");
  }
  if (args.sensorNoiseStd !== null && (!Number.isFinite(args.sensorNoiseStd) || args.sensorNoiseStd < 0)) {
    throw new Error("--sensor-noise-std must be non-negative");
  }
  if (args.sensorDelaySteps !== null && (!Number.isInteger(args.sensorDelaySteps) || args.sensorDelaySteps < 0)) {
    throw new Error("--sensor-delay-steps must be a non-negative integer");
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
  if (args.residualWarningThresholds.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new Error("--residual-warning-thresholds values must be positive");
  }
  if (args.velocityWarningThresholds.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new Error("--velocity-warning-thresholds values must be positive");
  }
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
  if (state.t + cfg.dt * 0.5 >= cfg.duration) return "timeout";
  return "max_steps";
}

function disturbanceForceAt(state, args) {
  const active = state.t >= args.disturbanceAt && state.t < args.disturbanceAt + args.disturbanceDuration;
  if (!active) return 0;
  if (args.disturbanceMode === "fixed") return args.disturbanceForce;
  return -Math.sign(state.theta || 1) * args.disturbanceForce;
}

function roundedParam(value, digits = 9) {
  return String(Number.isFinite(value) ? Number.parseFloat(value.toFixed(digits)) : value);
}

function makeBrowserInitialUrl({ preset, mode, seed, lightElevationDeg, cfg, initialState }) {
  const params = new URLSearchParams();
  params.set("mode", mode);
  params.set("preset", preset);
  params.set("seed", String(seed));
  params.set("light", roundedParam(lightElevationDeg, 3));
  params.set("force", roundedParam(cfg.forceLimit, 3));
  params.set("noise", roundedParam(cfg.sensorNoiseStd, 6));
  params.set("delay", String(Math.round(cfg.sensorDelaySteps)));
  params.set("duration", roundedParam(cfg.duration, 3));
  params.set("x", roundedParam(initialState.x));
  params.set("xDot", roundedParam(initialState.xDot));
  params.set("theta", roundedParam(initialState.theta));
  params.set("thetaDot", roundedParam(initialState.thetaDot));
  return `balance.html?${params}`;
}

function maybeRound(value) {
  return Number.isFinite(value) ? roundNumber(value) : "";
}

function firstWarningTime(series, key, threshold) {
  const row = series.find((sample) => sample[key] >= threshold);
  return row ? row.t : null;
}

function pushEvent(events, base, label, state, sensor, extra = {}) {
  events.push({
    ...base,
    label,
    t: roundNumber(state.t),
    theta: roundNumber(state.theta),
    x: roundNumber(state.x),
    shadowResidual: roundNumber(sensor?.residual ?? 0),
    shadowResidualVelocity: roundNumber(sensor?.residualVelocity ?? 0),
    shadowConfidence: roundNumber(sensor?.confidence ?? 0),
    ...extra,
  });
}

function runMetricTrial(args, { preset, mode, seed, lightElevationDeg }) {
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
    ...(args.sensorNoiseStd !== null ? { sensorNoiseStd: args.sensorNoiseStd } : {}),
    ...(args.sensorDelaySteps !== null ? { sensorDelaySteps: args.sensorDelaySteps } : {}),
  });
  const id = trialId({ preset, mode, seed, lightElevationDeg });
  const base = { phase: args.phase, trialId: id, preset, mode, seed, lightElevationDeg };
  const disturbanceEnd = args.disturbanceAt + args.disturbanceDuration;
  const holdSteps = Math.max(1, Math.ceil(args.recoveryHold / cfg.dt));
  const browserInitialUrl = makeBrowserInitialUrl({ preset, mode, seed, lightElevationDeg, cfg, initialState });
  let state = initializeBalanceState(cfg);
  const runtime = createBalanceRuntime(cfg);
  const controllerState = {};
  const samples = [];
  const curveSamples = [];
  const events = [];
  const warningSeries = [];
  let previousDisturbanceActive = false;
  let nearFallTime = null;
  let recoveryTime = null;
  let recoveryCandidateTime = null;
  let recoveryHoldCount = 0;
  let thetaSquareSum = 0;
  let postThetaSquareSum = 0;
  let postStepCount = 0;
  let confidenceSum = 0;
  let absForceIntegral = 0;
  let maxAbsTheta = Math.abs(state.theta);
  let maxAbsThetaAfterDisturbance = 0;
  let maxAbsThetaAfterDisturbanceTime = null;
  let maxAbsX = Math.abs(state.x);
  let saturationCount = 0;
  let confidenceLossCount = 0;
  let steps = 0;
  let sensor = sampleShadowSensor(state, runtime, cfg);
  let control = { force: 0, rawForce: 0, saturated: false, phase: "PASSIVE", reason: "initial" };
  const maxSteps = Math.ceil(args.duration / args.dt);

  while (steps < maxSteps && !state.fallen && !state.railHit && state.t < cfg.duration) {
    sensor = sampleShadowSensor(state, runtime, cfg);
    control = computeBalanceControl(state, sensor, controllerState, cfg);
    const disturbanceForce = disturbanceForceAt(state, args);
    const disturbanceActive = disturbanceForce !== 0;
    const absTheta = Math.abs(state.theta);
    const postDisturbance = state.t >= disturbanceEnd;

    if (disturbanceActive && !previousDisturbanceActive) {
      pushEvent(events, base, "disturbance_start", state, sensor, { disturbanceForce: roundNumber(disturbanceForce) });
    }
    if (!disturbanceActive && previousDisturbanceActive) {
      pushEvent(events, base, "disturbance_end", state, sensor);
    }
    previousDisturbanceActive = disturbanceActive;

    thetaSquareSum += state.theta * state.theta;
    confidenceSum += sensor.confidence;
    absForceIntegral += Math.abs(control.force) * cfg.dt;
    maxAbsTheta = Math.max(maxAbsTheta, absTheta);
    maxAbsX = Math.max(maxAbsX, Math.abs(state.x));
    if (control.saturated) saturationCount += 1;
    if (!sensor.valid || sensor.confidence < args.confidenceLossThreshold) confidenceLossCount += 1;

    if (postDisturbance) {
      postThetaSquareSum += state.theta * state.theta;
      postStepCount += 1;
      if (absTheta > maxAbsThetaAfterDisturbance) {
        maxAbsThetaAfterDisturbance = absTheta;
        maxAbsThetaAfterDisturbanceTime = state.t;
      }
      if (recoveryTime === null) {
        const recoveredNow = absTheta <= args.recoveryTheta && Math.abs(state.thetaDot) <= args.recoveryThetaDot;
        if (recoveredNow) {
          recoveryHoldCount += 1;
          if (recoveryCandidateTime === null) recoveryCandidateTime = state.t;
          if (recoveryHoldCount >= holdSteps) {
            recoveryTime = recoveryCandidateTime - disturbanceEnd;
            pushEvent(events, base, "recovery", state, sensor, { recoveryTime: roundNumber(recoveryTime) });
          }
        } else {
          recoveryHoldCount = 0;
          recoveryCandidateTime = null;
        }
      }
    }

    if (nearFallTime === null && absTheta >= args.nearFallTheta) {
      nearFallTime = state.t;
      pushEvent(events, base, "near_fall", state, sensor);
    }

    warningSeries.push({
      t: state.t,
      residualNorm: Math.abs(sensor.residual) / Math.max(cfg.poleLength, 1e-9),
      velocityNorm: Math.abs(sensor.residualVelocity) / Math.max(cfg.poleLength, 1e-9),
    });

    if (steps % args.logEvery === 0) {
      const sample = {
        ...base,
        relativeTime: roundNumber(state.t - disturbanceEnd),
        disturbanceForce: roundNumber(disturbanceForce),
        residualNorm: roundNumber(Math.abs(sensor.residual) / Math.max(cfg.poleLength, 1e-9)),
        residualVelocityNorm: roundNumber(Math.abs(sensor.residualVelocity) / Math.max(cfg.poleLength, 1e-9)),
        rawForce: roundNumber(control.rawForce),
        saturated: control.saturated,
        controlPhase: control.phase,
        controlReason: control.reason,
        sensorValid: sensor.valid,
        ...serializeBalanceSample(state, sensor, control, cfg),
      };
      samples.push(sample);
      curveSamples.push(sample);
    }

    state = integrateBalanceStep(state, control.force + disturbanceForce, cfg);
    steps += 1;
  }

  if (previousDisturbanceActive) pushEvent(events, base, "disturbance_end", state, sensor);
  const outcome = terminalOutcome(state, cfg);
  if (state.fallen) pushEvent(events, base, "fallen", state, sensor);
  if (state.railHit) pushEvent(events, base, "rail_hit", state, sensor);

  const simulatedTime = Math.min(state.t, cfg.duration);
  const denom = Math.max(1, steps);
  const residualWarningTime = firstWarningTime(warningSeries, "residualNorm", args.defaultResidualWarningThreshold);
  const velocityWarningTime = firstWarningTime(warningSeries, "velocityNorm", args.defaultVelocityWarningThreshold);
  const fallEvent = outcome === "fallen";
  const result = {
    ...base,
    outcome,
    success: outcome === "timeout",
    fallEvent,
    railHit: outcome === "rail_hit",
    recovered: recoveryTime !== null,
    simulatedTime: roundNumber(simulatedTime),
    normalizedSurvival: roundNumber(simulatedTime / cfg.duration),
    rmsTheta: roundNumber(Math.sqrt(thetaSquareSum / denom)),
    postDisturbanceRmsTheta: postStepCount > 0 ? roundNumber(Math.sqrt(postThetaSquareSum / postStepCount)) : "",
    maxAbsTheta: roundNumber(maxAbsTheta),
    maxAbsThetaAfterDisturbance: roundNumber(maxAbsThetaAfterDisturbance),
    maxAbsThetaAfterDisturbanceTime: maybeRound(maxAbsThetaAfterDisturbanceTime),
    maxAbsX: roundNumber(maxAbsX),
    recoveryTime: maybeRound(recoveryTime),
    nearFallTime: maybeRound(nearFallTime),
    meanShadowConfidence: roundNumber(confidenceSum / denom),
    forceBudget: roundNumber(absForceIntegral),
    meanAbsForce: roundNumber(absForceIntegral / Math.max(simulatedTime, cfg.dt)),
    saturationCount,
    confidenceLossCount,
    residualWarningTime: maybeRound(residualWarningTime),
    residualWarningLeadTime: fallEvent && residualWarningTime !== null ? roundNumber(simulatedTime - residualWarningTime) : "",
    velocityWarningTime: maybeRound(velocityWarningTime),
    velocityWarningLeadTime: fallEvent && velocityWarningTime !== null ? roundNumber(simulatedTime - velocityWarningTime) : "",
    steps,
    disturbanceAt: args.disturbanceAt,
    disturbanceDuration: args.disturbanceDuration,
    disturbanceForce: args.disturbanceForce,
    disturbanceMode: args.disturbanceMode,
    initialX: roundNumber(initialState.x),
    initialXDot: roundNumber(initialState.xDot),
    initialTheta: roundNumber(initialState.theta),
    initialThetaDot: roundNumber(initialState.thetaDot),
    browserInitialUrl,
    warningSeries,
  };

  return { result, samples, curveSamples, events };
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

function ratio(numerator, denominator) {
  return denominator > 0 ? numerator / denominator : null;
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

function summarizeMetrics(results) {
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
      recoveryRate: roundNumber(rate(rows, (row) => row.recovered)),
      fallRate: roundNumber(rate(rows, (row) => row.outcome === "fallen")),
      railHitRate: roundNumber(rate(rows, (row) => row.outcome === "rail_hit")),
      nearFallRate: roundNumber(rate(rows, (row) => Number.isFinite(row.nearFallTime))),
      meanSurvival: roundNumber(mean(rows.map((row) => row.simulatedTime))),
      meanRmsTheta: roundNumber(mean(rows.map((row) => row.rmsTheta))),
      meanPostDisturbanceRmsTheta: roundNumber(mean(rows.map((row) => row.postDisturbanceRmsTheta))),
      meanMaxAbsThetaAfterDisturbance: roundNumber(mean(rows.map((row) => row.maxAbsThetaAfterDisturbance))),
      meanRecoveryTime: roundNumber(mean(rows.map((row) => row.recoveryTime))),
      meanForceBudget: roundNumber(mean(rows.map((row) => row.forceBudget))),
      meanSaturationCount: roundNumber(mean(rows.map((row) => row.saturationCount))),
      meanConfidenceLossCount: roundNumber(mean(rows.map((row) => row.confidenceLossCount))),
    };
  }).sort((a, b) => (
    a.preset.localeCompare(b.preset)
    || a.lightElevationDeg - b.lightElevationDeg
    || a.mode.localeCompare(b.mode)
  ));
}

function evaluateWarnings(results, args) {
  const groups = groupBy(results, (row) => [row.phase, row.preset, row.mode, row.lightElevationDeg].join("|"));
  const rows = [];

  for (const [key, trials] of groups.entries()) {
    const [phase, preset, mode, lightElevationDeg] = key.split("|");
    for (const [source, thresholds, field] of [
      ["shadow_residual", args.residualWarningThresholds, "residualNorm"],
      ["shadow_residual_velocity", args.velocityWarningThresholds, "velocityNorm"],
    ]) {
      for (const threshold of thresholds) {
        let truePositive = 0;
        let falsePositive = 0;
        let trueNegative = 0;
        let falseNegative = 0;
        const leadTimes = [];

        for (const trial of trials) {
          const firstTime = firstWarningTime(trial.warningSeries, field, threshold);
          if (trial.fallEvent) {
            if (firstTime !== null && firstTime <= trial.simulatedTime) {
              truePositive += 1;
              leadTimes.push(trial.simulatedTime - firstTime);
            } else {
              falseNegative += 1;
            }
          } else if (firstTime !== null) {
            falsePositive += 1;
          } else {
            trueNegative += 1;
          }
        }

        const precision = ratio(truePositive, truePositive + falsePositive);
        const recall = ratio(truePositive, truePositive + falseNegative);
        const f1 = precision !== null && recall !== null && precision + recall > 0
          ? 2 * precision * recall / (precision + recall)
          : null;
        rows.push({
          phase,
          preset,
          mode,
          lightElevationDeg: Number.parseFloat(lightElevationDeg),
          source,
          threshold,
          n: trials.length,
          truePositive,
          falsePositive,
          trueNegative,
          falseNegative,
          precision: roundNumber(precision),
          recall: roundNumber(recall),
          f1: roundNumber(f1),
          falseAlarmRate: roundNumber(ratio(falsePositive, falsePositive + trueNegative)),
          meanLeadTime: roundNumber(mean(leadTimes)),
          minLeadTime: roundNumber(leadTimes.length ? Math.min(...leadTimes) : null),
        });
      }
    }
  }

  return rows;
}

function makeComparisons(results) {
  const groups = groupBy(results, (row) => [row.preset, row.seed, row.lightElevationDeg].join("|"));
  const rows = [];

  for (const trialRows of groups.values()) {
    const byMode = new Map(trialRows.map((row) => [row.mode, row]));
    const passive = byMode.get("passive");
    const naiveShadow = byMode.get("naive_shadow");
    for (const row of trialRows) {
      for (const [baselineName, baseline] of [["passive", passive], ["naive_shadow", naiveShadow]]) {
        if (!baseline || row.mode === baselineName) continue;
        rows.push({
          phase: row.phase,
          preset: row.preset,
          seed: row.seed,
          lightElevationDeg: row.lightElevationDeg,
          mode: row.mode,
          baselineMode: baselineName,
          outcome: row.outcome,
          baselineOutcome: baseline.outcome,
          survivalDelta: roundNumber(row.simulatedTime - baseline.simulatedTime),
          rmsThetaDelta: roundNumber(row.rmsTheta - baseline.rmsTheta),
          postDisturbanceRmsThetaDelta: roundNumber(row.postDisturbanceRmsTheta - baseline.postDisturbanceRmsTheta),
          maxAbsThetaAfterDisturbanceDelta: roundNumber(row.maxAbsThetaAfterDisturbance - baseline.maxAbsThetaAfterDisturbance),
          recoveryTimeDelta: Number.isFinite(row.recoveryTime) && Number.isFinite(baseline.recoveryTime)
            ? roundNumber(row.recoveryTime - baseline.recoveryTime)
            : "",
          recoveryAdvantage: row.recovered && !baseline.recovered ? "recovered_vs_not" : "",
          forceBudgetDelta: roundNumber(row.forceBudget - baseline.forceBudget),
          saturationCountDelta: roundNumber(row.saturationCount - baseline.saturationCount),
          confidenceLossCountDelta: roundNumber(row.confidenceLossCount - baseline.confidenceLossCount),
        });
      }
    }
  }

  return rows.sort((a, b) => (
    a.preset.localeCompare(b.preset)
    || a.seed - b.seed
    || a.lightElevationDeg - b.lightElevationDeg
    || a.mode.localeCompare(b.mode)
    || a.baselineMode.localeCompare(b.baselineMode)
  ));
}

function makeRecoveryCurves(curveSamples, args) {
  const groups = groupBy(curveSamples, (sample) => {
    const timeBin = Math.round(sample.relativeTime / args.curveBin) * args.curveBin;
    return [sample.phase, sample.preset, sample.mode, sample.lightElevationDeg, roundNumber(timeBin)].join("|");
  });

  return Array.from(groups.entries()).map(([key, rows]) => {
    const [phase, preset, mode, lightElevationDeg, relativeTime] = key.split("|");
    return {
      phase,
      preset,
      mode,
      lightElevationDeg: Number.parseFloat(lightElevationDeg),
      relativeTime: Number.parseFloat(relativeTime),
      n: rows.length,
      meanAbsTheta: roundNumber(mean(rows.map((row) => Math.abs(row.theta)))),
      meanTheta: roundNumber(mean(rows.map((row) => row.theta))),
      meanAbsShadowResidual: roundNumber(mean(rows.map((row) => Math.abs(row.shadowResidual)))),
      meanShadowConfidence: roundNumber(mean(rows.map((row) => row.shadowConfidence))),
      meanAbsForce: roundNumber(mean(rows.map((row) => Math.abs(row.force)))),
    };
  }).sort((a, b) => (
    a.preset.localeCompare(b.preset)
    || a.lightElevationDeg - b.lightElevationDeg
    || a.mode.localeCompare(b.mode)
    || a.relativeTime - b.relativeTime
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

function formatSeconds(value) {
  return Number.isFinite(value) ? `${value}s` : "n/a";
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(repoRoot, args.out);
  const results = [];
  const samples = [];
  const curveSamples = [];
  const events = [];

  for (const preset of args.presets) {
    for (const lightElevationDeg of args.lightElevations) {
      for (let i = 0; i < args.seeds; i += 1) {
        const seed = args.seedStart + i;
        for (const mode of args.modes) {
          const trial = runMetricTrial(args, { preset, mode, seed, lightElevationDeg });
          results.push(trial.result);
          samples.push(...trial.samples);
          curveSamples.push(...trial.curveSamples);
          events.push(...trial.events);
        }
      }
    }
  }

  const publicResults = results.map(({ warningSeries, ...row }) => row);
  const summaryRows = summarizeMetrics(publicResults);
  const warningRows = evaluateWarnings(results, args);
  const comparisonRows = makeComparisons(publicResults);
  const recoveryCurveRows = makeRecoveryCurves(curveSamples, args);
  const manifest = {
    schema: "sundog.balance.phase8-recovery.v1",
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
    disturbance: {
      at: args.disturbanceAt,
      duration: args.disturbanceDuration,
      force: args.disturbanceForce,
      mode: args.disturbanceMode,
    },
    recovery: {
      theta: args.recoveryTheta,
      thetaDot: args.recoveryThetaDot,
      hold: args.recoveryHold,
    },
    warningThresholds: {
      defaultResidual: args.defaultResidualWarningThreshold,
      defaultVelocity: args.defaultVelocityWarningThreshold,
      residual: args.residualWarningThresholds,
      velocity: args.velocityWarningThresholds,
    },
    trialCount: publicResults.length,
    sampleCount: samples.length,
    eventCount: events.length,
    note: "Ignored local Phase 8 metric output. This is recovery/event instrumentation, not an operating-envelope verdict.",
  };

  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  await writeFile(path.join(outDir, "samples.jsonl"), `${samples.map((row) => JSON.stringify(row)).join("\n")}\n`);
  await writeFile(path.join(outDir, "trial-metrics.csv"), toCsv(publicResults, [
    "phase",
    "trialId",
    "preset",
    "mode",
    "seed",
    "lightElevationDeg",
    "outcome",
    "success",
    "fallEvent",
    "railHit",
    "recovered",
    "simulatedTime",
    "normalizedSurvival",
    "rmsTheta",
    "postDisturbanceRmsTheta",
    "maxAbsTheta",
    "maxAbsThetaAfterDisturbance",
    "maxAbsThetaAfterDisturbanceTime",
    "maxAbsX",
    "recoveryTime",
    "nearFallTime",
    "meanShadowConfidence",
    "forceBudget",
    "meanAbsForce",
    "saturationCount",
    "confidenceLossCount",
    "residualWarningTime",
    "residualWarningLeadTime",
    "velocityWarningTime",
    "velocityWarningLeadTime",
    "steps",
    "disturbanceAt",
    "disturbanceDuration",
    "disturbanceForce",
    "disturbanceMode",
    "initialX",
    "initialXDot",
    "initialTheta",
    "initialThetaDot",
    "browserInitialUrl",
  ]));
  await writeFile(path.join(outDir, "event-log.csv"), toCsv(events, [
    "phase",
    "trialId",
    "preset",
    "mode",
    "seed",
    "lightElevationDeg",
    "label",
    "t",
    "theta",
    "x",
    "shadowResidual",
    "shadowResidualVelocity",
    "shadowConfidence",
    "disturbanceForce",
    "recoveryTime",
  ]));
  await writeFile(path.join(outDir, "metric-summary.csv"), toCsv(summaryRows, [
    "phase",
    "preset",
    "mode",
    "lightElevationDeg",
    "n",
    "successRate",
    "recoveryRate",
    "fallRate",
    "railHitRate",
    "nearFallRate",
    "meanSurvival",
    "meanRmsTheta",
    "meanPostDisturbanceRmsTheta",
    "meanMaxAbsThetaAfterDisturbance",
    "meanRecoveryTime",
    "meanForceBudget",
    "meanSaturationCount",
    "meanConfidenceLossCount",
  ]));
  await writeFile(path.join(outDir, "warning-thresholds.csv"), toCsv(warningRows, [
    "phase",
    "preset",
    "mode",
    "lightElevationDeg",
    "source",
    "threshold",
    "n",
    "truePositive",
    "falsePositive",
    "trueNegative",
    "falseNegative",
    "precision",
    "recall",
    "f1",
    "falseAlarmRate",
    "meanLeadTime",
    "minLeadTime",
  ]));
  await writeFile(path.join(outDir, "matched-comparison.csv"), toCsv(comparisonRows, [
    "phase",
    "preset",
    "seed",
    "lightElevationDeg",
    "mode",
    "baselineMode",
    "outcome",
    "baselineOutcome",
    "survivalDelta",
    "rmsThetaDelta",
    "postDisturbanceRmsThetaDelta",
    "maxAbsThetaAfterDisturbanceDelta",
    "recoveryTimeDelta",
    "recoveryAdvantage",
    "forceBudgetDelta",
    "saturationCountDelta",
    "confidenceLossCountDelta",
  ]));
  await writeFile(path.join(outDir, "recovery-curves.csv"), toCsv(recoveryCurveRows, [
    "phase",
    "preset",
    "mode",
    "lightElevationDeg",
    "relativeTime",
    "n",
    "meanAbsTheta",
    "meanTheta",
    "meanAbsShadowResidual",
    "meanShadowConfidence",
    "meanAbsForce",
  ]));

  const sundogSummary = summaryRows.filter((row) => row.mode === "sundog_shadow");
  console.log(`Balance ${args.phase}: ${publicResults.length} trials, ${events.length} events, ${samples.length} sampled rows`);
  console.log(`Wrote ${path.relative(repoRoot, outDir)}`);
  for (const row of sundogSummary) {
    console.log(`${row.preset} sundog_shadow: recovery ${row.recoveryRate}, fall ${row.fallRate}, mean recovery ${formatSeconds(row.meanRecoveryTime)}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
