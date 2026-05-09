import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  BALANCE_PRESETS,
  assessBalanceBoundary,
  clamp,
  computeBalanceControl,
  createBalanceRuntime,
  initializeBalanceState,
  integrateBalanceStep,
  makeRng,
  normalizeBalanceConfig,
  roundNumber,
  sampleShadowSensor,
} from "../public/js/balance-core.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const MODE_DEFINITIONS = Object.freeze({
  passive: "No control force.",
  naive_shadow: "Shadow residual baseline without dynamics or observability gating.",
  sundog_shadow: "Prototype Sundog shadow controller with confidence gating.",
  oracle: "Privileged true-angle controller; diagnostic ceiling only.",
});

function parseList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function parseNumberList(value) {
  return parseList(value).map((item) => Number.parseFloat(item));
}

function parseIntegerList(value) {
  return parseList(value).map((item) => Number.parseInt(item, 10));
}

function parseArgs(argv) {
  const args = {
    phase: "phase9-boundary",
    out: "results/balance/phase9-boundary",
    seedStart: 0,
    seeds: 8,
    presets: ["recoverable", "near_fall"],
    modes: ["passive", "naive_shadow", "sundog_shadow", "oracle"],
    duration: 8,
    dt: 1 / 120,
    baseLightElevation: 28,
    baseForceLimit: 12,
    baseRailLimit: 2.4,
    baseSensorNoiseStd: 0,
    baseSensorDelaySteps: 0,
    baseSensorDropoutRate: 0,
    lightElevations: [8, 12, 28, 55, 72, 84],
    sensorDelaySteps: [0, 6, 12, 24, 30],
    sensorNoiseStd: [0, 0.015, 0.03, 0.055, 0.08],
    sensorDropoutRates: [0, 0.05, 0.1, 0.2, 0.35],
    forceLimits: [4, 6, 8, 12, 16],
    railLimits: [1.0, 1.4, 1.8, 2.4],
    disturbanceForces: [2.5, 4.5, 6.5, 8.5],
    disturbanceAt: 0.25,
    disturbanceDuration: 0.15,
    disturbanceMode: "adversarial",
    jitterScale: 1,
    confidenceLossThreshold: 0.35,
    recoveryTheta: 0.09,
    recoveryThetaDot: 0.35,
    recoveryHold: 0.25,
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
    else if (flag === "--duration") args.duration = Number.parseFloat(value);
    else if (flag === "--dt") args.dt = Number.parseFloat(value);
    else if (flag === "--base-light-elevation") args.baseLightElevation = Number.parseFloat(value);
    else if (flag === "--base-force-limit") args.baseForceLimit = Number.parseFloat(value);
    else if (flag === "--base-rail-limit") args.baseRailLimit = Number.parseFloat(value);
    else if (flag === "--base-sensor-noise-std" || flag === "--base-noise") args.baseSensorNoiseStd = Number.parseFloat(value);
    else if (flag === "--base-sensor-delay-steps" || flag === "--base-delay") args.baseSensorDelaySteps = Number.parseInt(value, 10);
    else if (flag === "--base-sensor-dropout-rate" || flag === "--base-dropout") args.baseSensorDropoutRate = Number.parseFloat(value);
    else if (flag === "--light-elevations") args.lightElevations = parseNumberList(value);
    else if (flag === "--sensor-delay-steps" || flag === "--delay-steps") args.sensorDelaySteps = parseIntegerList(value);
    else if (flag === "--sensor-noise-std" || flag === "--noise") args.sensorNoiseStd = parseNumberList(value);
    else if (flag === "--sensor-dropout-rates" || flag === "--dropout-rates") args.sensorDropoutRates = parseNumberList(value);
    else if (flag === "--force-limits") args.forceLimits = parseNumberList(value);
    else if (flag === "--rail-limits") args.railLimits = parseNumberList(value);
    else if (flag === "--disturbance-forces") args.disturbanceForces = parseNumberList(value);
    else if (flag === "--disturbance-at") args.disturbanceAt = Number.parseFloat(value);
    else if (flag === "--disturbance-duration") args.disturbanceDuration = Number.parseFloat(value);
    else if (flag === "--disturbance-mode") args.disturbanceMode = value;
    else if (flag === "--jitter-scale") args.jitterScale = Number.parseFloat(value);
    else if (flag === "--confidence-loss-threshold") args.confidenceLossThreshold = Number.parseFloat(value);
    else if (flag === "--recovery-theta") args.recoveryTheta = Number.parseFloat(value);
    else if (flag === "--recovery-theta-dot") args.recoveryThetaDot = Number.parseFloat(value);
    else if (flag === "--recovery-hold") args.recoveryHold = Number.parseFloat(value);
    else throw new Error(`Unknown flag: ${flag}`);
  }

  validateArgs(args);
  args.lightElevations = uniqueSorted(args.lightElevations);
  args.sensorDelaySteps = uniqueSorted(args.sensorDelaySteps);
  args.sensorNoiseStd = uniqueSorted(args.sensorNoiseStd);
  args.sensorDropoutRates = uniqueSorted(args.sensorDropoutRates);
  args.forceLimits = uniqueSorted(args.forceLimits);
  args.railLimits = uniqueSorted(args.railLimits);
  args.disturbanceForces = uniqueSorted(args.disturbanceForces);
  return args;
}

function uniqueSorted(values) {
  return [...new Set(values)].sort((a, b) => a - b);
}

function validateArgs(args) {
  if (!Number.isInteger(args.seedStart) || args.seedStart < 0) throw new Error("--seed-start must be a non-negative integer");
  if (!Number.isInteger(args.seeds) || args.seeds < 1) throw new Error("--seeds must be a positive integer");
  for (const [flag, value] of [
    ["--duration", args.duration],
    ["--dt", args.dt],
    ["--base-light-elevation", args.baseLightElevation],
    ["--base-force-limit", args.baseForceLimit],
    ["--base-rail-limit", args.baseRailLimit],
    ["--disturbance-at", args.disturbanceAt],
    ["--disturbance-duration", args.disturbanceDuration],
    ["--jitter-scale", args.jitterScale],
    ["--confidence-loss-threshold", args.confidenceLossThreshold],
    ["--recovery-theta", args.recoveryTheta],
    ["--recovery-theta-dot", args.recoveryThetaDot],
    ["--recovery-hold", args.recoveryHold],
  ]) {
    if (!Number.isFinite(value) || value < 0) throw new Error(`${flag} must be finite and non-negative`);
  }
  if (args.duration <= 0 || args.dt <= 0 || args.baseForceLimit <= 0 || args.baseRailLimit <= 0) {
    throw new Error("--duration, --dt, --base-force-limit, and --base-rail-limit must be positive");
  }
  if (!["adversarial", "fixed"].includes(args.disturbanceMode)) {
    throw new Error("--disturbance-mode must be adversarial or fixed");
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
  for (const [flag, values] of [
    ["--sensor-delay-steps", args.sensorDelaySteps],
    ["--sensor-noise-std", args.sensorNoiseStd],
    ["--sensor-dropout-rates", args.sensorDropoutRates],
    ["--force-limits", args.forceLimits],
    ["--rail-limits", args.railLimits],
    ["--disturbance-forces", args.disturbanceForces],
  ]) {
    if (!Array.isArray(values) || values.length === 0 || values.some((value) => !Number.isFinite(value) || value < 0)) {
      throw new Error(`${flag} must contain finite non-negative values`);
    }
  }
  if (args.forceLimits.some((value) => value <= 0) || args.railLimits.some((value) => value <= 0)) {
    throw new Error("--force-limits and --rail-limits values must be positive");
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

function makeCases(args) {
  const base = {
    lightElevationDeg: args.baseLightElevation,
    sensorDelaySteps: args.baseSensorDelaySteps,
    sensorNoiseStd: args.baseSensorNoiseStd,
    sensorDropoutRate: args.baseSensorDropoutRate,
    forceLimit: args.baseForceLimit,
    railLimit: args.baseRailLimit,
    disturbanceForce: 4.5,
  };
  const cases = [];
  const addAxis = (axis, value, overrides) => {
    cases.push({
      axis,
      axisValue: value,
      ...base,
      ...overrides,
    });
  };

  for (const value of args.lightElevations) addAxis("light_elevation", value, { lightElevationDeg: value });
  for (const value of args.sensorDelaySteps) addAxis("sensor_delay", value, { sensorDelaySteps: value });
  for (const value of args.sensorNoiseStd) addAxis("sensor_noise", value, { sensorNoiseStd: value });
  for (const value of args.sensorDropoutRates) addAxis("sensor_dropout", value, { sensorDropoutRate: value });
  for (const value of args.forceLimits) addAxis("force_limit", value, { forceLimit: value });
  for (const value of args.railLimits) addAxis("rail_limit", value, { railLimit: value });
  for (const value of args.disturbanceForces) addAxis("disturbance_force", value, { disturbanceForce: value });
  return cases;
}

function safeId(value) {
  return String(value).replaceAll(".", "p").replaceAll("-", "m");
}

function caseId(boundaryCase) {
  return [
    boundaryCase.axis,
    safeId(boundaryCase.axisValue),
    `light_${safeId(boundaryCase.lightElevationDeg)}`,
    `delay_${safeId(boundaryCase.sensorDelaySteps)}`,
    `noise_${safeId(boundaryCase.sensorNoiseStd)}`,
    `drop_${safeId(boundaryCase.sensorDropoutRate)}`,
    `force_${safeId(boundaryCase.forceLimit)}`,
    `rail_${safeId(boundaryCase.railLimit)}`,
    `push_${safeId(boundaryCase.disturbanceForce)}`,
  ].join("__");
}

function trialId({ boundaryCase, preset, mode, seed }) {
  return [
    caseId(boundaryCase),
    preset,
    mode,
    `seed_${String(seed).padStart(3, "0")}`,
  ].join("__");
}

function terminalOutcome(state, cfg) {
  if (state.fallen) return "fallen";
  if (state.railHit) return "rail_hit";
  if (state.t + cfg.dt * 0.5 >= cfg.duration) return "timeout";
  return "max_steps";
}

function disturbanceForceAt(state, args, boundaryCase) {
  const active = state.t >= args.disturbanceAt && state.t < args.disturbanceAt + args.disturbanceDuration;
  if (!active) return 0;
  if (args.disturbanceMode === "fixed") return boundaryCase.disturbanceForce;
  return -Math.sign(state.theta || 1) * boundaryCase.disturbanceForce;
}

function roundedParam(value, digits = 9) {
  return String(Number.isFinite(value) ? Number.parseFloat(value.toFixed(digits)) : value);
}

function makeBrowserInitialUrl({ preset, mode, seed, cfg, boundaryCase, initialState }) {
  const params = new URLSearchParams();
  params.set("mode", mode);
  params.set("preset", preset);
  params.set("seed", String(seed));
  params.set("light", roundedParam(cfg.lightElevationDeg, 3));
  params.set("force", roundedParam(cfg.forceLimit, 3));
  params.set("noise", roundedParam(cfg.sensorNoiseStd, 6));
  params.set("delay", String(Math.round(cfg.sensorDelaySteps)));
  params.set("dropout", roundedParam(cfg.sensorDropoutRate, 6));
  params.set("rail", roundedParam(cfg.railLimit, 3));
  params.set("duration", roundedParam(cfg.duration, 3));
  params.set("disturbanceForce", roundedParam(boundaryCase.disturbanceForce, 3));
  params.set("x", roundedParam(initialState.x));
  params.set("xDot", roundedParam(initialState.xDot));
  params.set("theta", roundedParam(initialState.theta));
  params.set("thetaDot", roundedParam(initialState.thetaDot));
  return `balance.html?${params}`;
}

function maybeRound(value) {
  return Number.isFinite(value) ? roundNumber(value) : "";
}

function runBoundaryTrial(args, { boundaryCase, preset, mode, seed }) {
  const initialState = seededInitialState(preset, seed, args.jitterScale);
  const cfg = normalizeBalanceConfig({
    preset,
    controllerMode: mode,
    seed,
    initialState,
    duration: args.duration,
    dt: args.dt,
    forceLimit: boundaryCase.forceLimit,
    railLimit: boundaryCase.railLimit,
    lightElevationDeg: boundaryCase.lightElevationDeg,
    sensorNoiseStd: boundaryCase.sensorNoiseStd,
    sensorDelaySteps: boundaryCase.sensorDelaySteps,
    sensorDropoutRate: boundaryCase.sensorDropoutRate,
  });
  const id = trialId({ boundaryCase, preset, mode, seed });
  const disturbanceEnd = args.disturbanceAt + args.disturbanceDuration;
  const holdSteps = Math.max(1, Math.ceil(args.recoveryHold / cfg.dt));
  let state = initializeBalanceState(cfg);
  const runtime = createBalanceRuntime(cfg);
  const controllerState = {};
  let sensor = sampleShadowSensor(state, runtime, cfg);
  let control = { force: 0, rawForce: 0, saturated: false, phase: "PASSIVE", reason: "initial" };
  const initialAssessment = assessBalanceBoundary(cfg, sensor, control, state);
  let recoveryTime = null;
  let recoveryCandidateTime = null;
  let recoveryHoldCount = 0;
  let thetaSquareSum = 0;
  let postThetaSquareSum = 0;
  let postStepCount = 0;
  let confidenceSum = 0;
  let shadowLengthSum = 0;
  let absForceIntegral = 0;
  let maxAbsTheta = Math.abs(state.theta);
  let maxAbsThetaAfterDisturbance = 0;
  let maxAbsX = Math.abs(state.x);
  let saturationCount = 0;
  let confidenceLossCount = 0;
  let dropoutCount = 0;
  let steps = 0;
  const maxSteps = Math.ceil(args.duration / args.dt);

  while (steps < maxSteps && !state.fallen && !state.railHit && state.t < cfg.duration) {
    sensor = sampleShadowSensor(state, runtime, cfg);
    control = computeBalanceControl(state, sensor, controllerState, cfg);
    const disturbanceForce = disturbanceForceAt(state, args, boundaryCase);
    const absTheta = Math.abs(state.theta);
    const postDisturbance = state.t >= disturbanceEnd;

    thetaSquareSum += state.theta * state.theta;
    confidenceSum += sensor.confidence;
    shadowLengthSum += sensor.length;
    absForceIntegral += Math.abs(control.force) * cfg.dt;
    maxAbsTheta = Math.max(maxAbsTheta, absTheta);
    maxAbsX = Math.max(maxAbsX, Math.abs(state.x));
    if (control.saturated) saturationCount += 1;
    if (!sensor.valid) dropoutCount += 1;
    if (!sensor.valid || sensor.confidence < args.confidenceLossThreshold) confidenceLossCount += 1;

    if (postDisturbance) {
      postThetaSquareSum += state.theta * state.theta;
      postStepCount += 1;
      maxAbsThetaAfterDisturbance = Math.max(maxAbsThetaAfterDisturbance, absTheta);
      if (recoveryTime === null) {
        const recoveredNow = absTheta <= args.recoveryTheta && Math.abs(state.thetaDot) <= args.recoveryThetaDot;
        if (recoveredNow) {
          recoveryHoldCount += 1;
          if (recoveryCandidateTime === null) recoveryCandidateTime = state.t;
          if (recoveryHoldCount >= holdSteps) recoveryTime = recoveryCandidateTime - disturbanceEnd;
        } else {
          recoveryHoldCount = 0;
          recoveryCandidateTime = null;
        }
      }
    }

    state = integrateBalanceStep(state, control.force + disturbanceForce, cfg);
    steps += 1;
  }

  const outcome = terminalOutcome(state, cfg);
  const finalAssessment = assessBalanceBoundary(cfg, sensor, control, state);
  const simulatedTime = Math.min(state.t, cfg.duration);
  const denom = Math.max(1, steps);
  const boundaryMechanism = classifyTrialMechanism({
    outcome,
    meanShadowConfidence: confidenceSum / denom,
    saturationRate: saturationCount / denom,
    confidenceLossRate: confidenceLossCount / denom,
    dropoutRate: dropoutCount / denom,
    recovered: recoveryTime !== null,
    sensorDelayMs: cfg.sensorDelaySteps * cfg.dt * 1000,
    sensorDropoutRate: cfg.sensorDropoutRate,
  }, initialAssessment, finalAssessment);

  return {
    phase: args.phase,
    caseId: caseId(boundaryCase),
    trialId: id,
    axis: boundaryCase.axis,
    axisValue: boundaryCase.axisValue,
    preset,
    mode,
    seed,
    outcome,
    success: outcome === "timeout",
    recovered: recoveryTime !== null,
    lightElevationDeg: cfg.lightElevationDeg,
    sensorDelaySteps: cfg.sensorDelaySteps,
    sensorDelayMs: roundNumber(cfg.sensorDelaySteps * cfg.dt * 1000),
    sensorNoiseStd: cfg.sensorNoiseStd,
    sensorDropoutRate: cfg.sensorDropoutRate,
    forceLimit: cfg.forceLimit,
    railLimit: cfg.railLimit,
    disturbanceForce: boundaryCase.disturbanceForce,
    simulatedTime: roundNumber(simulatedTime),
    normalizedSurvival: roundNumber(simulatedTime / cfg.duration),
    rmsTheta: roundNumber(Math.sqrt(thetaSquareSum / denom)),
    postDisturbanceRmsTheta: postStepCount > 0 ? roundNumber(Math.sqrt(postThetaSquareSum / postStepCount)) : "",
    maxAbsTheta: roundNumber(maxAbsTheta),
    maxAbsThetaAfterDisturbance: roundNumber(maxAbsThetaAfterDisturbance),
    maxAbsX: roundNumber(maxAbsX),
    meanShadowConfidence: roundNumber(confidenceSum / denom),
    meanShadowLength: roundNumber(shadowLengthSum / denom),
    forceBudget: roundNumber(absForceIntegral),
    saturationCount,
    saturationRate: roundNumber(saturationCount / denom),
    confidenceLossCount,
    confidenceLossRate: roundNumber(confidenceLossCount / denom),
    dropoutCount,
    dropoutRate: roundNumber(dropoutCount / denom),
    recoveryTime: maybeRound(recoveryTime),
    initialBoundaryStatus: initialAssessment.status,
    finalBoundaryStatus: finalAssessment.status,
    boundaryMechanism,
    boundaryMechanisms: initialAssessment.mechanisms.map((mechanism) => mechanism.code).join("|"),
    browserInitialUrl: makeBrowserInitialUrl({ preset, mode, seed, cfg, boundaryCase, initialState }),
  };
}

function classifyTrialMechanism(trial, initialAssessment, finalAssessment) {
  const staticUnsafe = [...initialAssessment.mechanisms, ...finalAssessment.mechanisms]
    .find((mechanism) => mechanism.severity === "unsafe");
  if (staticUnsafe) return staticUnsafe.code;
  if (trial.meanShadowConfidence < 0.25 || trial.confidenceLossRate > 0.6) return "shadow_unobservable";
  if (trial.sensorDelayMs >= 200 && !trial.recovered) return "delay_destabilized";
  if (trial.saturationRate > 0.2) return "force_saturated";
  if (trial.outcome === "rail_hit") return "rail_limited";
  if (trial.sensorDropoutRate >= 0.2 || trial.dropoutRate > 0.2) return "dropped_frames";
  if (trial.outcome === "fallen") return "controller_overcorrected";
  return "none";
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

function mostCommon(values) {
  const counts = new Map();
  for (const value of values.filter(Boolean)) counts.set(value, (counts.get(value) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0]?.[0] ?? "";
}

function summarize(rows) {
  const groups = groupBy(rows, (row) => [
    row.phase,
    row.axis,
    row.axisValue,
    row.preset,
    row.mode,
    row.lightElevationDeg,
    row.sensorDelaySteps,
    row.sensorNoiseStd,
    row.sensorDropoutRate,
    row.forceLimit,
    row.railLimit,
    row.disturbanceForce,
  ].join("|"));

  return Array.from(groups.entries()).map(([key, groupRows]) => {
    const [
      phase,
      axis,
      axisValue,
      preset,
      mode,
      lightElevationDeg,
      sensorDelaySteps,
      sensorNoiseStd,
      sensorDropoutRate,
      forceLimit,
      railLimit,
      disturbanceForce,
    ] = key.split("|");
    return {
      phase,
      axis,
      axisValue: Number.parseFloat(axisValue),
      preset,
      mode,
      n: groupRows.length,
      lightElevationDeg: Number.parseFloat(lightElevationDeg),
      sensorDelaySteps: Number.parseInt(sensorDelaySteps, 10),
      sensorDelayMs: roundNumber(mean(groupRows.map((row) => row.sensorDelayMs))),
      sensorNoiseStd: Number.parseFloat(sensorNoiseStd),
      sensorDropoutRate: Number.parseFloat(sensorDropoutRate),
      forceLimit: Number.parseFloat(forceLimit),
      railLimit: Number.parseFloat(railLimit),
      disturbanceForce: Number.parseFloat(disturbanceForce),
      successRate: roundNumber(rate(groupRows, (row) => row.success)),
      recoveryRate: roundNumber(rate(groupRows, (row) => row.recovered)),
      fallRate: roundNumber(rate(groupRows, (row) => row.outcome === "fallen")),
      railHitRate: roundNumber(rate(groupRows, (row) => row.outcome === "rail_hit")),
      doNotUseRate: roundNumber(rate(groupRows, (row) => row.initialBoundaryStatus === "do_not_use" || row.finalBoundaryStatus === "do_not_use")),
      meanSurvival: roundNumber(mean(groupRows.map((row) => row.simulatedTime))),
      meanRmsTheta: roundNumber(mean(groupRows.map((row) => row.rmsTheta))),
      meanPostDisturbanceRmsTheta: roundNumber(mean(groupRows.map((row) => row.postDisturbanceRmsTheta))),
      meanRecoveryTime: roundNumber(mean(groupRows.map((row) => row.recoveryTime))),
      meanShadowConfidence: roundNumber(mean(groupRows.map((row) => row.meanShadowConfidence))),
      meanShadowLength: roundNumber(mean(groupRows.map((row) => row.meanShadowLength))),
      meanForceBudget: roundNumber(mean(groupRows.map((row) => row.forceBudget))),
      meanSaturationRate: roundNumber(mean(groupRows.map((row) => row.saturationRate))),
      meanConfidenceLossRate: roundNumber(mean(groupRows.map((row) => row.confidenceLossRate))),
      meanDropoutRate: roundNumber(mean(groupRows.map((row) => row.dropoutRate))),
      primaryFailureMechanism: mostCommon(groupRows.map((row) => row.boundaryMechanism).filter((value) => value !== "none")),
      representativeReplay: groupRows.find((row) => row.mode === mode)?.browserInitialUrl ?? groupRows[0]?.browserInitialUrl ?? "",
    };
  }).sort((a, b) => (
    a.axis.localeCompare(b.axis)
    || a.axisValue - b.axisValue
    || a.preset.localeCompare(b.preset)
    || a.mode.localeCompare(b.mode)
  ));
}

function makeComparisons(rows) {
  const groups = groupBy(rows, (row) => [row.caseId, row.preset, row.seed].join("|"));
  const comparisons = [];
  for (const groupRows of groups.values()) {
    const byMode = new Map(groupRows.map((row) => [row.mode, row]));
    const sundog = byMode.get("sundog_shadow");
    if (!sundog) continue;
    for (const baselineName of ["passive", "naive_shadow", "oracle"]) {
      const baseline = byMode.get(baselineName);
      if (!baseline) continue;
      comparisons.push({
        phase: sundog.phase,
        caseId: sundog.caseId,
        axis: sundog.axis,
        axisValue: sundog.axisValue,
        preset: sundog.preset,
        seed: sundog.seed,
        baselineMode: baselineName,
        lightElevationDeg: sundog.lightElevationDeg,
        sensorDelaySteps: sundog.sensorDelaySteps,
        sensorDelayMs: sundog.sensorDelayMs,
        sensorNoiseStd: sundog.sensorNoiseStd,
        sensorDropoutRate: sundog.sensorDropoutRate,
        forceLimit: sundog.forceLimit,
        railLimit: sundog.railLimit,
        disturbanceForce: sundog.disturbanceForce,
        sundogOutcome: sundog.outcome,
        baselineOutcome: baseline.outcome,
        survivalDelta: roundNumber(sundog.simulatedTime - baseline.simulatedTime),
        recoveryDelta: Number.isFinite(sundog.recoveryTime) && Number.isFinite(baseline.recoveryTime)
          ? roundNumber(sundog.recoveryTime - baseline.recoveryTime)
          : "",
        rmsThetaDelta: roundNumber(sundog.rmsTheta - baseline.rmsTheta),
        confidenceDelta: roundNumber(sundog.meanShadowConfidence - baseline.meanShadowConfidence),
        sundogBoundaryStatus: sundog.initialBoundaryStatus,
        sundogFailureMechanism: sundog.boundaryMechanism,
        browserInitialUrl: sundog.browserInitialUrl,
      });
    }
  }
  return comparisons.sort((a, b) => (
    a.axis.localeCompare(b.axis)
    || a.axisValue - b.axisValue
    || a.preset.localeCompare(b.preset)
    || a.seed - b.seed
    || a.baselineMode.localeCompare(b.baselineMode)
  ));
}

function makeUnsafeCells(summaryRows) {
  return summaryRows
    .filter((row) => row.mode === "sundog_shadow")
    .filter((row) => (
      row.doNotUseRate > 0
      || row.recoveryRate <= 0.25
      || row.fallRate >= 0.5
      || row.meanShadowConfidence < 0.35
    ))
    .map((row) => ({
      ...row,
      unsafeReason: row.primaryFailureMechanism || (
        row.meanShadowConfidence < 0.35
          ? "shadow_unobservable"
          : row.recoveryRate <= 0.25
            ? "recovery_failed"
            : "performance_boundary"
      ),
    }))
    .sort((a, b) => (
      b.doNotUseRate - a.doNotUseRate
      || b.fallRate - a.fallRate
      || a.recoveryRate - b.recoveryRate
      || a.axis.localeCompare(b.axis)
      || a.axisValue - b.axisValue
    ));
}

function makePanelReport(unsafeRows, summaryRows, args) {
  const topUnsafe = unsafeRows.slice(0, 10).map((row) => ({
    axis: row.axis,
    axisValue: row.axisValue,
    preset: row.preset,
    unsafeReason: row.unsafeReason,
    recoveryRate: row.recoveryRate,
    fallRate: row.fallRate,
    meanShadowConfidence: row.meanShadowConfidence,
    representativeReplay: row.representativeReplay,
  }));
  const sundogRows = summaryRows.filter((row) => row.mode === "sundog_shadow");
  return {
    schema: "sundog.balance.phase9-boundary-panel.v1",
    generatedAt: new Date().toISOString(),
    phase: args.phase,
    headline: unsafeRows.length
      ? "Phase 9 found explicit cells where the shadow controller should not be used."
      : "Phase 9 did not find an unsafe cell in this first sweep; expand the grid before promotion.",
    trialGroups: sundogRows.length,
    unsafeGroupCount: unsafeRows.length,
    topUnsafe,
    note: "This is a first-pass boundary diagnostic. It is not the Phase 10 operating-envelope verdict.",
  };
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
  for (const row of rows) lines.push(columns.map((column) => csvEscape(row[column])).join(","));
  return `${lines.join("\n")}\n`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(repoRoot, args.out);
  const boundaryCases = makeCases(args);
  const results = [];

  for (const boundaryCase of boundaryCases) {
    for (const preset of args.presets) {
      for (let i = 0; i < args.seeds; i += 1) {
        const seed = args.seedStart + i;
        for (const mode of args.modes) {
          results.push(runBoundaryTrial(args, { boundaryCase, preset, mode, seed }));
        }
      }
    }
  }

  const summaryRows = summarize(results);
  const comparisonRows = makeComparisons(results);
  const unsafeRows = makeUnsafeCells(summaryRows);
  const panelReport = makePanelReport(unsafeRows, summaryRows, args);
  const manifest = {
    schema: "sundog.balance.phase9-boundary.v1",
    generatedAt: new Date().toISOString(),
    phase: args.phase,
    modes: args.modes,
    presets: args.presets,
    seedStart: args.seedStart,
    seeds: args.seeds,
    duration: args.duration,
    dt: args.dt,
    axes: {
      lightElevations: args.lightElevations,
      sensorDelaySteps: args.sensorDelaySteps,
      sensorNoiseStd: args.sensorNoiseStd,
      sensorDropoutRates: args.sensorDropoutRates,
      forceLimits: args.forceLimits,
      railLimits: args.railLimits,
      disturbanceForces: args.disturbanceForces,
    },
    baseCell: {
      lightElevationDeg: args.baseLightElevation,
      sensorDelaySteps: args.baseSensorDelaySteps,
      sensorNoiseStd: args.baseSensorNoiseStd,
      sensorDropoutRate: args.baseSensorDropoutRate,
      forceLimit: args.baseForceLimit,
      railLimit: args.baseRailLimit,
    },
    disturbance: {
      at: args.disturbanceAt,
      duration: args.disturbanceDuration,
      mode: args.disturbanceMode,
    },
    trialCount: results.length,
    summaryGroupCount: summaryRows.length,
    unsafeGroupCount: unsafeRows.length,
    note: "Ignored local Phase 9 boundary output. This is a sensor-degradation diagnostic, not the Phase 10 verdict.",
  };

  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  await writeFile(path.join(outDir, "boundary-panel.json"), `${JSON.stringify(panelReport, null, 2)}\n`);
  await writeFile(path.join(outDir, "trial-outcomes.csv"), toCsv(results, [
    "phase",
    "caseId",
    "trialId",
    "axis",
    "axisValue",
    "preset",
    "mode",
    "seed",
    "outcome",
    "success",
    "recovered",
    "lightElevationDeg",
    "sensorDelaySteps",
    "sensorDelayMs",
    "sensorNoiseStd",
    "sensorDropoutRate",
    "forceLimit",
    "railLimit",
    "disturbanceForce",
    "simulatedTime",
    "normalizedSurvival",
    "rmsTheta",
    "postDisturbanceRmsTheta",
    "maxAbsTheta",
    "maxAbsThetaAfterDisturbance",
    "maxAbsX",
    "meanShadowConfidence",
    "meanShadowLength",
    "forceBudget",
    "saturationCount",
    "saturationRate",
    "confidenceLossCount",
    "confidenceLossRate",
    "dropoutCount",
    "dropoutRate",
    "recoveryTime",
    "initialBoundaryStatus",
    "finalBoundaryStatus",
    "boundaryMechanism",
    "boundaryMechanisms",
    "browserInitialUrl",
  ]));
  await writeFile(path.join(outDir, "boundary-summary.csv"), toCsv(summaryRows, [
    "phase",
    "axis",
    "axisValue",
    "preset",
    "mode",
    "n",
    "lightElevationDeg",
    "sensorDelaySteps",
    "sensorDelayMs",
    "sensorNoiseStd",
    "sensorDropoutRate",
    "forceLimit",
    "railLimit",
    "disturbanceForce",
    "successRate",
    "recoveryRate",
    "fallRate",
    "railHitRate",
    "doNotUseRate",
    "meanSurvival",
    "meanRmsTheta",
    "meanPostDisturbanceRmsTheta",
    "meanRecoveryTime",
    "meanShadowConfidence",
    "meanShadowLength",
    "meanForceBudget",
    "meanSaturationRate",
    "meanConfidenceLossRate",
    "meanDropoutRate",
    "primaryFailureMechanism",
    "representativeReplay",
  ]));
  await writeFile(path.join(outDir, "matched-comparison.csv"), toCsv(comparisonRows, [
    "phase",
    "caseId",
    "axis",
    "axisValue",
    "preset",
    "seed",
    "baselineMode",
    "lightElevationDeg",
    "sensorDelaySteps",
    "sensorDelayMs",
    "sensorNoiseStd",
    "sensorDropoutRate",
    "forceLimit",
    "railLimit",
    "disturbanceForce",
    "sundogOutcome",
    "baselineOutcome",
    "survivalDelta",
    "recoveryDelta",
    "rmsThetaDelta",
    "confidenceDelta",
    "sundogBoundaryStatus",
    "sundogFailureMechanism",
    "browserInitialUrl",
  ]));
  await writeFile(path.join(outDir, "unsafe-cells.csv"), toCsv(unsafeRows, [
    "phase",
    "axis",
    "axisValue",
    "preset",
    "mode",
    "n",
    "lightElevationDeg",
    "sensorDelaySteps",
    "sensorDelayMs",
    "sensorNoiseStd",
    "sensorDropoutRate",
    "forceLimit",
    "railLimit",
    "disturbanceForce",
    "successRate",
    "recoveryRate",
    "fallRate",
    "railHitRate",
    "doNotUseRate",
    "meanShadowConfidence",
    "meanShadowLength",
    "meanSaturationRate",
    "meanConfidenceLossRate",
    "meanDropoutRate",
    "primaryFailureMechanism",
    "unsafeReason",
    "representativeReplay",
  ]));

  console.log(`Balance ${args.phase}: ${results.length} trials across ${boundaryCases.length} axis cells`);
  console.log(`Wrote ${path.relative(repoRoot, outDir)}`);
  console.log(`Unsafe Sundog groups: ${unsafeRows.length}/${summaryRows.filter((row) => row.mode === "sundog_shadow").length}`);
  for (const row of unsafeRows.slice(0, 6)) {
    console.log(`${row.axis}=${row.axisValue} ${row.preset}: ${row.unsafeReason}, recovery ${row.recoveryRate}, fall ${row.fallRate}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
