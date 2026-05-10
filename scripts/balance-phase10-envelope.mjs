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

const REFUTE_BANNER = "Sundog Balance attempted shadow-derived stabilisation of an inverted pendulum and did not beat naive shadow-centering on the tested slate; the workbench remains as a Planned Workbench.";
const REFUTE_HOOK = "The hidden body was controlled through its shadow only where the lighting geometry allowed it; on this slate, that was not enough to beat naive shadow-centering. The failure boundary is the finding.";

function parseList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function parseNumberList(value) {
  return parseList(value).map((item) => Number.parseFloat(item));
}

function parseIntegerList(value) {
  return parseList(value).map((item) => Number.parseInt(item, 10));
}

function uniqueSorted(values) {
  return [...new Set(values)].sort((a, b) => a - b);
}

function parseArgs(argv) {
  const args = {
    phase: "phase10-envelope",
    out: "results/balance/phase10-envelope",
    seedStart: 0,
    seeds: 100,
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
    bootstrapIterations: 1000,
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
    else if (flag === "--bootstrap-iterations") args.bootstrapIterations = Number.parseInt(value, 10);
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

function validateArgs(args) {
  if (!Number.isInteger(args.seedStart) || args.seedStart < 0) throw new Error("--seed-start must be a non-negative integer");
  if (!Number.isInteger(args.seeds) || args.seeds < 1) throw new Error("--seeds must be a positive integer");
  if (!Number.isInteger(args.bootstrapIterations) || args.bootstrapIterations < 100) {
    throw new Error("--bootstrap-iterations must be an integer >= 100");
  }
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
  for (const [flag, values] of [
    ["--light-elevations", args.lightElevations],
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
  if (args.lightElevations.some((value) => value <= 1 || value >= 89)) {
    throw new Error("--light-elevations values must be between 1 and 89 degrees");
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
  const addAxis = (axis, axisValue, overrides) => {
    cases.push({ axis, axisValue, ...base, ...overrides });
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

function caseId(cell) {
  return [
    cell.axis,
    safeId(cell.axisValue),
    `light_${safeId(cell.lightElevationDeg)}`,
    `delay_${safeId(cell.sensorDelaySteps)}`,
    `noise_${safeId(cell.sensorNoiseStd)}`,
    `drop_${safeId(cell.sensorDropoutRate)}`,
    `force_${safeId(cell.forceLimit)}`,
    `rail_${safeId(cell.railLimit)}`,
    `push_${safeId(cell.disturbanceForce)}`,
  ].join("__");
}

function cellId(cell, preset) {
  return `${caseId(cell)}__preset_${preset}`;
}

function trialId({ cell, preset, mode, seed }) {
  return `${cellId(cell, preset)}__${mode}__seed_${String(seed).padStart(3, "0")}`;
}

function terminalOutcome(state, cfg) {
  if (state.fallen) return "fallen";
  if (state.railHit) return "rail_hit";
  if (state.t + cfg.dt * 0.5 >= cfg.duration) return "timeout";
  return "max_steps";
}

function disturbanceForceAt(state, args, cell) {
  const active = state.t >= args.disturbanceAt && state.t < args.disturbanceAt + args.disturbanceDuration;
  if (!active) return 0;
  if (args.disturbanceMode === "fixed") return cell.disturbanceForce;
  return -Math.sign(state.theta || 1) * cell.disturbanceForce;
}

function roundedParam(value, digits = 9) {
  return String(Number.isFinite(value) ? Number.parseFloat(value.toFixed(digits)) : value);
}

function makeBrowserInitialUrl({ preset, mode, seed, cfg, cell, initialState }) {
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
  params.set("disturbanceForce", roundedParam(cell.disturbanceForce, 3));
  params.set("x", roundedParam(initialState.x));
  params.set("xDot", roundedParam(initialState.xDot));
  params.set("theta", roundedParam(initialState.theta));
  params.set("thetaDot", roundedParam(initialState.thetaDot));
  return `balance.html?${params}`;
}

function maybeRound(value) {
  return Number.isFinite(value) ? roundNumber(value) : "";
}

function runEnvelopeTrial(args, { cell, preset, mode, seed }) {
  const initialState = seededInitialState(preset, seed, args.jitterScale);
  const cfg = normalizeBalanceConfig({
    preset,
    controllerMode: mode,
    seed,
    initialState,
    duration: args.duration,
    dt: args.dt,
    forceLimit: cell.forceLimit,
    railLimit: cell.railLimit,
    lightElevationDeg: cell.lightElevationDeg,
    sensorNoiseStd: cell.sensorNoiseStd,
    sensorDelaySteps: cell.sensorDelaySteps,
    sensorDropoutRate: cell.sensorDropoutRate,
    disturbanceForce: cell.disturbanceForce,
  });
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
    const disturbanceForce = disturbanceForceAt(state, args, cell);
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

  return {
    phase: args.phase,
    cellId: cellId(cell, preset),
    caseId: caseId(cell),
    trialId: trialId({ cell, preset, mode, seed }),
    axis: cell.axis,
    axisValue: cell.axisValue,
    preset,
    mode,
    seed,
    outcome,
    success: outcome === "timeout",
    recovered: recoveryTime !== null,
    light_elev_deg: cfg.lightElevationDeg,
    delay_steps: cfg.sensorDelaySteps,
    delay_ms: roundNumber(cfg.sensorDelaySteps * cfg.dt * 1000),
    noise_sigma: cfg.sensorNoiseStd,
    dropout_rate: cfg.sensorDropoutRate,
    force_limit: cfg.forceLimit,
    rail_limit: cfg.railLimit,
    disturbance_mag: cell.disturbanceForce,
    simulated_time: roundNumber(simulatedTime),
    normalized_survival: roundNumber(simulatedTime / cfg.duration),
    rms_theta: roundNumber(Math.sqrt(thetaSquareSum / denom)),
    post_disturbance_rms_theta: postStepCount > 0 ? roundNumber(Math.sqrt(postThetaSquareSum / postStepCount)) : "",
    max_abs_theta: roundNumber(maxAbsTheta),
    max_abs_theta_after_disturbance: roundNumber(maxAbsThetaAfterDisturbance),
    max_abs_x: roundNumber(maxAbsX),
    mean_shadow_confidence: roundNumber(confidenceSum / denom),
    mean_shadow_length: roundNumber(shadowLengthSum / denom),
    force_budget: roundNumber(absForceIntegral),
    saturation_count: saturationCount,
    saturation_rate: roundNumber(saturationCount / denom),
    confidence_loss_count: confidenceLossCount,
    confidence_loss_rate: roundNumber(confidenceLossCount / denom),
    dropout_count: dropoutCount,
    observed_dropout_rate: roundNumber(dropoutCount / denom),
    recovery_time_after_impulse: maybeRound(recoveryTime),
    initial_boundary_status: initialAssessment.status,
    final_boundary_status: finalAssessment.status,
    initial_boundary_mechanisms: initialAssessment.mechanisms.map((mechanism) => mechanism.code).join("|"),
    final_boundary_mechanisms: finalAssessment.mechanisms.map((mechanism) => mechanism.code).join("|"),
    replay_url: makeBrowserInitialUrl({ preset, mode, seed, cfg, cell, initialState }),
  };
}

function mean(values) {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return null;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function quantile(values, q) {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (finite.length === 0) return null;
  const index = (finite.length - 1) * q;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return finite[lower];
  const weight = index - lower;
  return finite[lower] * (1 - weight) + finite[upper] * weight;
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

function classifyCell(row) {
  const cfg = normalizeBalanceConfig({
    duration: row.duration,
    dt: row.dt,
    lightElevationDeg: row.lightElevationDeg,
    sensorDelaySteps: row.sensorDelaySteps,
    sensorNoiseStd: row.sensorNoiseStd,
    sensorDropoutRate: row.sensorDropoutRate,
    forceLimit: row.forceLimit,
    railLimit: row.railLimit,
    disturbanceForce: row.disturbanceForce,
  });
  const assessment = assessBalanceBoundary(cfg);
  if (assessment.status === "do_not_use") return "failure_regime";
  if (assessment.status === "watch_boundary") return "borderline";
  return "diagnostic_positive";
}

function staticMechanisms(row) {
  const cfg = normalizeBalanceConfig({
    duration: row.duration,
    dt: row.dt,
    lightElevationDeg: row.lightElevationDeg,
    sensorDelaySteps: row.sensorDelaySteps,
    sensorNoiseStd: row.sensorNoiseStd,
    sensorDropoutRate: row.sensorDropoutRate,
    forceLimit: row.forceLimit,
    railLimit: row.railLimit,
    disturbanceForce: row.disturbanceForce,
  });
  return assessBalanceBoundary(cfg).mechanisms.map((mechanism) => mechanism.code).join("|");
}

function bootstrapMean(values, iterations, seedText) {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return { low: null, high: null };
  const rng = makeRng(stringHash(seedText));
  const samples = [];
  for (let i = 0; i < iterations; i += 1) {
    let sum = 0;
    for (let j = 0; j < finite.length; j += 1) {
      sum += finite[Math.floor(rng() * finite.length)];
    }
    samples.push(sum / finite.length);
  }
  return {
    low: quantile(samples, 0.025),
    high: quantile(samples, 0.975),
  };
}

function bootstrapYesFraction(flags, iterations, seedText) {
  if (flags.length === 0) return { low: null, high: null };
  const rng = makeRng(stringHash(seedText));
  const samples = [];
  for (let i = 0; i < iterations; i += 1) {
    let yes = 0;
    for (let j = 0; j < flags.length; j += 1) {
      yes += flags[Math.floor(rng() * flags.length)] ? 1 : 0;
    }
    samples.push(yes / flags.length);
  }
  return {
    low: quantile(samples, 0.025),
    high: quantile(samples, 0.975),
  };
}

function makeEnvelopeRows(trialRows, args) {
  const groups = groupBy(trialRows, (row) => row.cellId);
  return Array.from(groups.entries()).map(([id, rows]) => {
    const first = rows[0];
    const byMode = groupBy(rows, (row) => row.mode);
    const passive = byMode.get("passive") ?? [];
    const naive = byMode.get("naive_shadow") ?? [];
    const sundog = byMode.get("sundog_shadow") ?? [];
    const oracle = byMode.get("oracle") ?? [];
    const naiveBySeed = new Map(naive.map((row) => [row.seed, row]));
    const oracleBySeed = new Map(oracle.map((row) => [row.seed, row]));
    const pairedNaiveMargins = sundog
      .map((row) => {
        const baseline = naiveBySeed.get(row.seed);
        return baseline ? row.simulated_time - baseline.simulated_time : null;
      })
      .filter(Number.isFinite);
    const pairedOracleMargins = oracle
      .map((row) => {
        const controller = sundog.find((candidate) => candidate.seed === row.seed);
        return controller ? row.simulated_time - controller.simulated_time : null;
      })
      .filter(Number.isFinite);
    const survivalSundog = mean(sundog.map((row) => row.simulated_time));
    const survivalNaive = mean(naive.map((row) => row.simulated_time));
    const survivalPassive = mean(passive.map((row) => row.simulated_time));
    const survivalOracle = mean(oracle.map((row) => row.simulated_time));
    const rmsSundog = mean(sundog.map((row) => row.rms_theta));
    const rmsNaive = mean(naive.map((row) => row.rms_theta));
    const rmsOracle = mean(oracle.map((row) => row.rms_theta));
    const pairedBootstrap = bootstrapMean(pairedNaiveMargins, args.bootstrapIterations, `${id}:paired-margin`);
    const ratio = survivalNaive && survivalNaive > 0 ? survivalSundog / survivalNaive : null;
    const cellClass = classifyCell({
      duration: args.duration,
      dt: args.dt,
      lightElevationDeg: first.light_elev_deg,
      sensorDelaySteps: first.delay_steps,
      sensorNoiseStd: first.noise_sigma,
      sensorDropoutRate: first.dropout_rate,
      forceLimit: first.force_limit,
      railLimit: first.rail_limit,
      disturbanceForce: first.disturbance_mag,
    });
    const marginMean = mean(pairedNaiveMargins);
    const yesCell = cellClass === "diagnostic_positive"
      && Number.isFinite(ratio)
      && ratio >= 1.5
      && Number.isFinite(marginMean)
      && marginMean > 0;
    const oracleMarginMean = mean(pairedOracleMargins);
    const sundogReplay = sundog[0]?.replay_url ?? "";

    return {
      cell_id: id,
      case_id: first.caseId,
      axis: first.axis,
      axis_value: first.axisValue,
      preset: first.preset,
      light_elev_deg: first.light_elev_deg,
      delay_ms: first.delay_ms,
      delay_steps: first.delay_steps,
      noise_sigma: first.noise_sigma,
      dropout_rate: first.dropout_rate,
      rail_limit: first.rail_limit,
      force_limit: first.force_limit,
      disturbance_mag: first.disturbance_mag,
      cell_class: cellClass,
      static_boundary_mechanisms: staticMechanisms({
        duration: args.duration,
        dt: args.dt,
        lightElevationDeg: first.light_elev_deg,
        sensorDelaySteps: first.delay_steps,
        sensorNoiseStd: first.noise_sigma,
        sensorDropoutRate: first.dropout_rate,
        forceLimit: first.force_limit,
        railLimit: first.rail_limit,
        disturbanceForce: first.disturbance_mag,
      }),
      survival_passive_mean: roundNumber(survivalPassive ?? NaN),
      survival_sundog_mean: roundNumber(survivalSundog ?? NaN),
      survival_naive_mean: roundNumber(survivalNaive ?? NaN),
      survival_oracle_mean: roundNumber(survivalOracle ?? NaN),
      rms_theta_sundog_mean: roundNumber(rmsSundog ?? NaN),
      rms_theta_naive_mean: roundNumber(rmsNaive ?? NaN),
      rms_theta_oracle_mean: roundNumber(rmsOracle ?? NaN),
      recovery_time_after_impulse: roundNumber(mean(sundog.map((row) => row.recovery_time_after_impulse)) ?? NaN),
      sundog_naive_paired_margin_mean: roundNumber(marginMean ?? NaN),
      sundog_naive_survival_ratio: roundNumber(ratio ?? NaN),
      sundog_beats_naive_1p5x: yesCell,
      oracle_sundog_paired_margin_mean: roundNumber(oracleMarginMean ?? NaN),
      seed_count: pairedNaiveMargins.length,
      paired_margin_bootstrap_low: roundNumber(pairedBootstrap.low ?? NaN),
      paired_margin_bootstrap_high: roundNumber(pairedBootstrap.high ?? NaN),
      sundog_saturation_rate_mean: roundNumber(mean(sundog.map((row) => row.saturation_rate)) ?? NaN),
      sundog_force_budget_mean: roundNumber(mean(sundog.map((row) => row.force_budget)) ?? NaN),
      replay_url: sundogReplay,
      naive_replay_url: naive[0]?.replay_url ?? "",
      oracle_replay_url: oracleBySeed.get(sundog[0]?.seed)?.replay_url ?? oracle[0]?.replay_url ?? "",
    };
  }).sort((a, b) => (
    a.cell_class.localeCompare(b.cell_class)
    || a.axis.localeCompare(b.axis)
    || a.axis_value - b.axis_value
    || a.preset.localeCompare(b.preset)
  ));
}

function makeMatchedComparisons(trialRows) {
  const groups = groupBy(trialRows, (row) => `${row.cellId}\t${row.seed}`);
  const rows = [];
  for (const groupRows of groups.values()) {
    const byMode = new Map(groupRows.map((row) => [row.mode, row]));
    const sundog = byMode.get("sundog_shadow");
    if (!sundog) continue;
    for (const baselineName of ["passive", "naive_shadow", "oracle"]) {
      const baseline = byMode.get(baselineName);
      if (!baseline) continue;
      rows.push({
        phase: sundog.phase,
        cell_id: sundog.cellId,
        case_id: sundog.caseId,
        axis: sundog.axis,
        axis_value: sundog.axisValue,
        preset: sundog.preset,
        seed: sundog.seed,
        baseline_mode: baselineName,
        light_elev_deg: sundog.light_elev_deg,
        delay_ms: sundog.delay_ms,
        noise_sigma: sundog.noise_sigma,
        dropout_rate: sundog.dropout_rate,
        force_limit: sundog.force_limit,
        rail_limit: sundog.rail_limit,
        disturbance_mag: sundog.disturbance_mag,
        sundog_outcome: sundog.outcome,
        baseline_outcome: baseline.outcome,
        survival_delta: roundNumber(sundog.simulated_time - baseline.simulated_time),
        recovery_delta: Number.isFinite(sundog.recovery_time_after_impulse) && Number.isFinite(baseline.recovery_time_after_impulse)
          ? roundNumber(sundog.recovery_time_after_impulse - baseline.recovery_time_after_impulse)
          : "",
        rms_theta_delta: roundNumber(sundog.rms_theta - baseline.rms_theta),
        confidence_delta: roundNumber(sundog.mean_shadow_confidence - baseline.mean_shadow_confidence),
        replay_url: sundog.replay_url,
      });
    }
  }
  return rows.sort((a, b) => (
    a.axis.localeCompare(b.axis)
    || a.axis_value - b.axis_value
    || a.preset.localeCompare(b.preset)
    || a.seed - b.seed
    || a.baseline_mode.localeCompare(b.baseline_mode)
  ));
}

function makeBestWorstRows(envelopeRows) {
  const diagnostic = envelopeRows.filter((row) => row.cell_class === "diagnostic_positive");
  const failure = envelopeRows.filter((row) => row.cell_class === "failure_regime");
  const best = [...diagnostic].sort((a, b) => (
    (b.sundog_naive_paired_margin_mean ?? -Infinity) - (a.sundog_naive_paired_margin_mean ?? -Infinity)
  ))[0] ?? null;
  const worst = [...failure].sort((a, b) => (
    (a.sundog_naive_paired_margin_mean ?? Infinity) - (b.sundog_naive_paired_margin_mean ?? Infinity)
  ))[0] ?? null;
  return [
    best ? { selection: "best_cell", rule: "highest Sundog-minus-naive survival margin among diagnostic_positive cells", ...best } : null,
    worst ? { selection: "worst_cell", rule: "largest Sundog loss to naive among failure_regime cells", ...worst } : null,
  ].filter(Boolean);
}

function makeCellClassMap(envelopeRows) {
  return envelopeRows.map((row) => ({
    cell_id: row.cell_id,
    axis: row.axis,
    axis_value: row.axis_value,
    preset: row.preset,
    light_elev_deg: row.light_elev_deg,
    delay_ms: row.delay_ms,
    noise_sigma: row.noise_sigma,
    dropout_rate: row.dropout_rate,
    rail_limit: row.rail_limit,
    force_limit: row.force_limit,
    disturbance_mag: row.disturbance_mag,
    cell_class: row.cell_class,
    static_boundary_mechanisms: row.static_boundary_mechanisms,
    sundog_naive_paired_margin_mean: row.sundog_naive_paired_margin_mean,
    sundog_beats_naive_1p5x: row.sundog_beats_naive_1p5x,
  }));
}

function checkP1(envelopeRows, args) {
  const diagnostic = envelopeRows.filter((row) => row.cell_class === "diagnostic_positive");
  const yesFlags = diagnostic.map((row) => row.sundog_beats_naive_1p5x === true);
  const yesCount = yesFlags.filter(Boolean).length;
  const fraction = diagnostic.length > 0 ? yesCount / diagnostic.length : 0;
  const ci = bootstrapYesFraction(yesFlags, args.bootstrapIterations, "phase10-p1-yes-fraction");
  return {
    holds: Number.isFinite(ci.low) && ci.low > 0.30,
    diagnostic_positive_cell_count: diagnostic.length,
    yes_cell_count: yesCount,
    yes_cell_fraction: roundNumber(fraction),
    yes_fraction_bootstrap_low: roundNumber(ci.low ?? NaN),
    yes_fraction_bootstrap_high: roundNumber(ci.high ?? NaN),
  };
}

function pairedControllerRows(trialRows, cellId, leftMode, rightMode) {
  const rows = trialRows.filter((row) => row.cellId === cellId);
  const bySeed = groupBy(rows, (row) => row.seed);
  const pairs = [];
  for (const seedRows of bySeed.values()) {
    const left = seedRows.find((row) => row.mode === leftMode);
    const right = seedRows.find((row) => row.mode === rightMode);
    if (left && right) pairs.push({ left, right });
  }
  return pairs;
}

function controllerSucceeded(row) {
  // Phase 10.5 treats failed trials with transient recovery markers as
  // degradation telemetry, not hard failure-boundary success.
  return row?.outcome === "timeout";
}

function checkP2(envelopeRows, trialRows) {
  const failureRows = envelopeRows.filter((row) => (
    row.light_elev_deg >= 80
    || row.delay_ms >= 200
  ));
  const cellReports = failureRows.map((row) => {
    const pairs = pairedControllerRows(trialRows, row.cell_id, "sundog_shadow", "naive_shadow");
    let hardViolationCount = 0;
    let allFailPairCount = 0;
    let sundogWins = 0;
    let naiveWins = 0;
    let ties = 0;
    const margins = [];

    for (const { left: sundog, right: naive } of pairs) {
      const sundogSuccess = controllerSucceeded(sundog);
      const naiveSuccess = controllerSucceeded(naive);
      const margin = sundog.simulated_time - naive.simulated_time;
      margins.push(margin);
      if (sundogSuccess && !naiveSuccess) hardViolationCount += 1;
      if (!sundogSuccess && !naiveSuccess) allFailPairCount += 1;
      if (margin > 1e-6) sundogWins += 1;
      else if (margin < -1e-6) naiveWins += 1;
      else ties += 1;
    }

    const marginMean = mean(margins);
    const allFailMargin = pairs.length > 0 && allFailPairCount === pairs.length && marginMean > 0;
    return {
      cell_id: row.cell_id,
      axis: row.axis,
      axis_value: row.axis_value,
      preset: row.preset,
      light_elev_deg: row.light_elev_deg,
      delay_ms: row.delay_ms,
      seed_pairs: pairs.length,
      hard_violation_count: hardViolationCount,
      all_fail_pair_count: allFailPairCount,
      all_fail_survival_margin: allFailMargin,
      sundog_seed_wins: sundogWins,
      naive_seed_wins: naiveWins,
      tied_seed_pairs: ties,
      sundog_naive_paired_margin_mean: roundNumber(marginMean ?? NaN),
      replay_url: row.replay_url,
    };
  });
  const hardViolations = cellReports.filter((row) => row.hard_violation_count > 0);
  const allFailMargins = cellReports.filter((row) => row.all_fail_survival_margin);
  return {
    holds: hardViolations.length === 0,
    failure_regime_cell_count: failureRows.length,
    hard_violation_count: hardViolations.reduce((sum, row) => sum + row.hard_violation_count, 0),
    hard_violation_cell_count: hardViolations.length,
    p2b_all_fail_margin_cell_count: allFailMargins.length,
    p2b_all_fail_margins_reported_only: allFailMargins,
    cells: cellReports,
  };
}

function recoveryScore(row, args) {
  return Number.isFinite(row.recovery_time_after_impulse) ? row.recovery_time_after_impulse : args.duration;
}

function checkP3(envelopeRows, args) {
  const delayRows = envelopeRows
    .filter((row) => row.axis === "sensor_delay")
    .sort((a, b) => a.delay_ms - b.delay_ms);
  const grouped = groupBy(delayRows, (row) => row.delay_ms);
  const series = Array.from(grouped.entries()).map(([delayMs, rows]) => ({
    delay_ms: Number.parseFloat(delayMs),
    recovery_time_after_impulse: roundNumber(mean(rows.map((row) => recoveryScore(row, args))) ?? NaN),
  })).sort((a, b) => a.delay_ms - b.delay_ms);
  let holds = series.length >= 2;
  for (let i = 1; i < series.length; i += 1) {
    if (series[i].recovery_time_after_impulse + 1e-9 < series[i - 1].recovery_time_after_impulse) {
      holds = false;
      break;
    }
  }
  return { holds, series };
}

function checkP4(envelopeRows, trialRows, args) {
  const cellReports = envelopeRows.map((row) => {
    const pairs = pairedControllerRows(trialRows, row.cell_id, "oracle", "sundog_shadow");
    const capped = pairs.length > 0 && pairs.every(({ left: oracle, right: sundog }) => (
      oracle.simulated_time >= args.duration - 1e-6
      && sundog.simulated_time >= args.duration - 1e-6
    ));
    const oracleSurvivalExceeds = row.survival_oracle_mean > row.survival_sundog_mean + 1e-6;
    const oracleLowerRms = row.rms_theta_oracle_mean + 1e-6 < row.rms_theta_sundog_mean;
    return {
      cell_id: row.cell_id,
      axis: row.axis,
      axis_value: row.axis_value,
      preset: row.preset,
      cell_class: row.cell_class,
      seed_pairs: pairs.length,
      capped,
      survival_oracle_mean: row.survival_oracle_mean,
      survival_sundog_mean: row.survival_sundog_mean,
      oracle_survival_exceeds: oracleSurvivalExceeds,
      rms_theta_oracle_mean: row.rms_theta_oracle_mean,
      rms_theta_sundog_mean: row.rms_theta_sundog_mean,
      oracle_lower_rms: oracleLowerRms,
      replay_url: row.replay_url,
      oracle_replay_url: row.oracle_replay_url,
    };
  });
  const uncapped = cellReports.filter((row) => !row.capped);
  const capped = cellReports.filter((row) => row.capped);
  const uncappedOracleExceeds = uncapped.filter((row) => row.oracle_survival_exceeds);
  const cappedOracleLowerRms = capped.filter((row) => row.oracle_lower_rms);
  const p4aFraction = uncapped.length > 0 ? uncappedOracleExceeds.length / uncapped.length : 1;
  const p4bFraction = capped.length > 0 ? cappedOracleLowerRms.length / capped.length : 1;
  const p4aHolds = uncapped.length === 0 || p4aFraction >= 0.8;
  const p4bHolds = capped.length === 0 || p4bFraction >= 0.8;
  return {
    holds: p4aHolds && p4bHolds,
    p4a_holds: p4aHolds,
    p4a_uncapped_cell_count: uncapped.length,
    p4a_oracle_survival_exceeds_cell_count: uncappedOracleExceeds.length,
    p4a_oracle_survival_exceeds_fraction: roundNumber(p4aFraction),
    p4b_holds: p4bHolds,
    p4b_capped_cell_count: capped.length,
    p4b_oracle_lower_rms_cell_count: cappedOracleLowerRms.length,
    p4b_oracle_lower_rms_fraction: roundNumber(p4bFraction),
    cells: cellReports,
  };
}

function makeVerdict(envelopeRows, trialRows, bestWorstRows, args) {
  const p1 = checkP1(envelopeRows, args);
  const p2 = checkP2(envelopeRows, trialRows);
  const p3 = checkP3(envelopeRows, args);
  const p4 = checkP4(envelopeRows, trialRows, args);
  const verdict = !p1.holds || !p2.holds
    ? "REFUTE"
    : p3.holds && p4.holds
      ? "CONFIRM"
      : "AMBIGUOUS";
  const reasons = [];
  if (!p1.holds) reasons.push("P1 failed the diagnostic-positive yes-cell bootstrap threshold.");
  if (!p2.holds) reasons.push(`P2a hard failure-boundary violations found: ${p2.hard_violation_count}.`);
  if (p2.p2b_all_fail_margin_cell_count > 0) reasons.push(`P2b all-fail survival margins reported only: ${p2.p2b_all_fail_margin_cell_count} cells.`);
  if (!p3.holds) reasons.push("P3 failed monotonic recovery in delay; sensor-model audit required.");
  if (!p4.holds) reasons.push(`P4 failed repaired dual-ceiling rule: P4a=${p4.p4a_holds}, P4b=${p4.p4b_holds}.`);
  return {
    schema: "sundog.balance.phase10-verdict.v1",
    generatedAt: new Date().toISOString(),
    verdict,
    disposition: verdict === "CONFIRM"
      ? "Promote to Operating-Envelope Study tier."
      : verdict === "REFUTE"
        ? "Keep at Planned Workbench tier and publish the negative-finding banner."
        : "Hold tier unchanged until audit and rerun.",
    p1,
    p2,
    p3,
    p4,
    reasons,
    refuteBanner: REFUTE_BANNER,
    refuteHook: REFUTE_HOOK,
    bestCell: bestWorstRows.find((row) => row.selection === "best_cell") ?? null,
    worstCell: bestWorstRows.find((row) => row.selection === "worst_cell") ?? null,
  };
}

function verdictMarkdown(verdict, args) {
  const lines = [
    "# Sundog Balance Phase 10 Verdict",
    "",
    `Generated: ${verdict.generatedAt}`,
    `Phase: ${args.phase}`,
    `Verdict: ${verdict.verdict}`,
    `Disposition: ${verdict.disposition}`,
    "",
    "## P1 - Central Effect",
    "",
    `Holds: ${verdict.p1.holds}`,
    `Diagnostic-positive cells: ${verdict.p1.diagnostic_positive_cell_count}`,
    `Yes cells: ${verdict.p1.yes_cell_count}`,
    `Yes fraction: ${verdict.p1.yes_cell_fraction}`,
    `Bootstrap CI: ${verdict.p1.yes_fraction_bootstrap_low} to ${verdict.p1.yes_fraction_bootstrap_high}`,
    "",
    "## P2 - Failure Boundary Realness",
    "",
    `Holds: ${verdict.p2.holds}`,
    `P2 cells checked: ${verdict.p2.failure_regime_cell_count}`,
    `P2a hard violation count: ${verdict.p2.hard_violation_count}`,
    `P2a hard violation cells: ${verdict.p2.hard_violation_cell_count}`,
    `P2b all-fail survival-margin cells (reported only): ${verdict.p2.p2b_all_fail_margin_cell_count}`,
    "",
    ...(verdict.p2.p2b_all_fail_margins_reported_only.length ? [
      "| cell_id | preset | axis | value | margin | seed wins | replay |",
      "| --- | --- | --- | ---: | ---: | --- | --- |",
      ...verdict.p2.p2b_all_fail_margins_reported_only.map((row) => `| ${row.cell_id} | ${row.preset} | ${row.axis} | ${row.axis_value} | ${row.sundog_naive_paired_margin_mean} | Sundog ${row.sundog_seed_wins}, naive ${row.naive_seed_wins}, ties ${row.tied_seed_pairs} | ${row.replay_url} |`),
      "",
    ] : []),
    "## P3 - Recovery Monotonicity",
    "",
    `Holds: ${verdict.p3.holds}`,
    "",
    "| delay_ms | recovery_time_after_impulse |",
    "| ---: | ---: |",
    ...verdict.p3.series.map((row) => `| ${row.delay_ms} | ${row.recovery_time_after_impulse} |`),
    "",
    "## P4 - Privileged Oracle Ceiling",
    "",
    `Holds: ${verdict.p4.holds}`,
    `P4a survival ceiling on uncapped cells: ${verdict.p4.p4a_holds}`,
    `P4a oracle survival exceeds cells: ${verdict.p4.p4a_oracle_survival_exceeds_cell_count}/${verdict.p4.p4a_uncapped_cell_count}`,
    `P4a oracle survival exceeds fraction: ${verdict.p4.p4a_oracle_survival_exceeds_fraction}`,
    `P4b quality ceiling on capped cells: ${verdict.p4.p4b_holds}`,
    `P4b oracle lower-RMS cells: ${verdict.p4.p4b_oracle_lower_rms_cell_count}/${verdict.p4.p4b_capped_cell_count}`,
    `P4b oracle lower-RMS fraction: ${verdict.p4.p4b_oracle_lower_rms_fraction}`,
    "",
    "## Reasons",
    "",
    ...(verdict.reasons.length ? verdict.reasons.map((reason) => `- ${reason}`) : ["- All pre-registered predictions held."]),
    "",
    "## Broadcast Surfaces",
    "",
    `REFUTE banner: "${verdict.refuteBanner}"`,
    "",
    `REFUTE hook: "${verdict.refuteHook}"`,
    "",
    "## Replay Picks",
    "",
    verdict.bestCell ? `Best cell replay: ${verdict.bestCell.replay_url}` : "Best cell replay: none",
    verdict.worstCell ? `Worst cell replay: ${verdict.worstCell.replay_url}` : "Worst cell replay: none",
    "",
  ];
  return `${lines.join("\n")}\n`;
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number" && !Number.isFinite(value)) return "";
  const normalized = typeof value === "number" ? roundNumber(value) : value;
  const text = String(normalized);
  if (/[",\r\n]/.test(text)) return `"${text.replaceAll("\"", "\"\"")}"`;
  return text;
}

function rowsToCsv(rows, columns = null) {
  const explicitColumns = columns ?? [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const lines = [explicitColumns.join(",")];
  for (const row of rows) lines.push(explicitColumns.map((column) => csvEscape(row[column])).join(","));
  return `${lines.join("\n")}\n`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(repoRoot, args.out);
  const cells = makeCases(args);
  const trialRows = [];

  for (const cell of cells) {
    for (const preset of args.presets) {
      for (let i = 0; i < args.seeds; i += 1) {
        const seed = args.seedStart + i;
        for (const mode of args.modes) {
          trialRows.push(runEnvelopeTrial(args, { cell, preset, mode, seed }));
        }
      }
    }
  }

  const matchedRows = makeMatchedComparisons(trialRows);
  const envelopeRows = makeEnvelopeRows(trialRows, args);
  const cellClassRows = makeCellClassMap(envelopeRows);
  const bestWorstRows = makeBestWorstRows(envelopeRows);
  const verdict = makeVerdict(envelopeRows, trialRows, bestWorstRows, args);
  const manifest = {
    schema: "sundog.balance.phase10-envelope.v1",
    generatedAt: verdict.generatedAt,
    phase: args.phase,
    args,
    modes: args.modes,
    presets: args.presets,
    seedStart: args.seedStart,
    seeds: args.seeds,
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
      disturbanceForce: 4.5,
    },
    disturbance: {
      at: args.disturbanceAt,
      duration: args.disturbanceDuration,
      mode: args.disturbanceMode,
    },
    trialCount: trialRows.length,
    cellCount: envelopeRows.length,
    verdict: verdict.verdict,
    note: "Ignored local Phase 10 operating-envelope output. Verdict surfaces are pre-registered in docs/SUNDOG_V_BALANCE.md.",
  };

  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  await writeFile(path.join(outDir, "trial-outcomes.csv"), rowsToCsv(trialRows));
  await writeFile(path.join(outDir, "matched-comparison.csv"), rowsToCsv(matchedRows));
  await writeFile(path.join(outDir, "envelope.csv"), rowsToCsv(envelopeRows));
  await writeFile(path.join(outDir, "cell-class-map.csv"), rowsToCsv(cellClassRows));
  await writeFile(path.join(outDir, "best-worst-cells.csv"), rowsToCsv(bestWorstRows));
  await writeFile(path.join(outDir, "verdict.json"), `${JSON.stringify(verdict, null, 2)}\n`);
  await writeFile(path.join(outDir, "verdict.md"), verdictMarkdown(verdict, args));

  console.log(`Balance ${args.phase}: ${trialRows.length} trials across ${envelopeRows.length} operating-envelope cells`);
  console.log(`Wrote ${path.relative(repoRoot, outDir)}`);
  console.log(`Verdict: ${verdict.verdict}`);
  console.log(`P1 yes cells ${verdict.p1.yes_cell_count}/${verdict.p1.diagnostic_positive_cell_count}, lower CI ${verdict.p1.yes_fraction_bootstrap_low}`);
  console.log(`P2a hard violations ${verdict.p2.hard_violation_count}; P2b reported-only cells ${verdict.p2.p2b_all_fail_margin_cell_count}; P3 ${verdict.p3.holds}`);
  console.log(`P4a ${verdict.p4.p4a_oracle_survival_exceeds_cell_count}/${verdict.p4.p4a_uncapped_cell_count}; P4b ${verdict.p4.p4b_oracle_lower_rms_cell_count}/${verdict.p4.p4b_capped_cell_count}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
