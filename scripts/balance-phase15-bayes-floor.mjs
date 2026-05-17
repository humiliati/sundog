import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import {
  BALANCE_CONTROLLER_MODES,
  BALANCE_PRESETS,
  clamp,
  computeBalanceControl,
  computeShadowGeometry,
  createBalanceRuntime,
  initializeBalanceState,
  integrateBalanceStep,
  makeRng,
  normalizeBalanceConfig,
  roundNumber,
  sampleShadowSensor,
  serializeBalanceObservation,
} from "../public/js/balance-core.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const FORBIDDEN_OBSERVATION_KEYS = Object.freeze([
  "theta",
  "thetaDot",
  "poleEnergy",
  "oracleAction",
  "terminalOutcome",
  "raw",
]);
const PHASE10_DEFAULTS = Object.freeze({
  presets: Object.freeze(["recoverable", "near_fall"]),
  lightElevations: Object.freeze([8, 12, 28, 55, 72, 84]),
  sensorDelaySteps: Object.freeze([0, 6, 12, 24, 30]),
  sensorNoiseStd: Object.freeze([0, 0.015, 0.03, 0.055, 0.08]),
  sensorDropoutRates: Object.freeze([0, 0.05, 0.1, 0.2, 0.35]),
  forceLimits: Object.freeze([4, 6, 8, 12, 16]),
  railLimits: Object.freeze([1.0, 1.4, 1.8, 2.4]),
  disturbanceForces: Object.freeze([2.5, 4.5, 6.5, 8.5]),
  baseLightElevation: 28,
  baseSensorDelaySteps: 0,
  baseSensorNoiseStd: 0,
  baseSensorDropoutRate: 0,
  baseForceLimit: 12,
  baseRailLimit: 2.4,
  baseDisturbanceForce: 4.5,
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
    phase: "phase15-bayesian-floor-smoke",
    out: "results/balance/phase15-bayesian-floor-smoke",
    seedStart: 0,
    seeds: 8,
    presets: ["recoverable", "near_fall"],
    modes: ["naive_shadow", "sundog_shadow", "bayes_floor_shadow_particle", "oracle"],
    lightElevations: [28, 84],
    sensorDelaySteps: [0],
    sensorNoiseStd: [0],
    sensorDropoutRates: [0],
    cellSlate: "smoke",
    phase10Out: "results/balance/phase10-envelope",
    phase10Cells: null,
    cellClasses: null,
    limitCells: null,
    duration: 8,
    dt: 1 / 120,
    forceLimit: 12,
    railLimit: 2.4,
    disturbanceAt: 0.25,
    disturbanceDuration: 0.15,
    disturbanceMode: "adversarial",
    logEvery: 30,
    jitterScale: 1,
    particleCount: 61,
    horizonSeconds: 0.05,
    estimateFullPhase10Trials: 0,
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
    else if (flag === "--sensor-delay-steps" || flag === "--delay-steps") args.sensorDelaySteps = parseIntegerList(value);
    else if (flag === "--sensor-noise-std" || flag === "--noise") args.sensorNoiseStd = parseNumberList(value);
    else if (flag === "--sensor-dropout-rates" || flag === "--dropout-rates") args.sensorDropoutRates = parseNumberList(value);
    else if (flag === "--cell-slate") args.cellSlate = value;
    else if (flag === "--phase10-out") args.phase10Out = value;
    else if (flag === "--phase10-cells") args.phase10Cells = value;
    else if (flag === "--cell-classes") args.cellClasses = parseList(value);
    else if (flag === "--limit-cells") args.limitCells = value === "all" ? null : Number.parseInt(value, 10);
    else if (flag === "--duration") args.duration = Number.parseFloat(value);
    else if (flag === "--dt") args.dt = Number.parseFloat(value);
    else if (flag === "--force-limit") args.forceLimit = Number.parseFloat(value);
    else if (flag === "--rail-limit" || flag === "--rail") args.railLimit = Number.parseFloat(value);
    else if (flag === "--disturbance-at") args.disturbanceAt = Number.parseFloat(value);
    else if (flag === "--disturbance-duration") args.disturbanceDuration = Number.parseFloat(value);
    else if (flag === "--disturbance-mode") args.disturbanceMode = value;
    else if (flag === "--log-every") args.logEvery = Number.parseInt(value, 10);
    else if (flag === "--jitter-scale") args.jitterScale = Number.parseFloat(value);
    else if (flag === "--particle-count") args.particleCount = Number.parseInt(value, 10);
    else if (flag === "--horizon-seconds") args.horizonSeconds = Number.parseFloat(value);
    else if (flag === "--estimate-full-phase10-trials") args.estimateFullPhase10Trials = Number.parseInt(value, 10);
    else throw new Error(`Unknown flag: ${flag}`);
  }

  validateArgs(args);
  return args;
}

function validateArgs(args) {
  if (!Number.isInteger(args.seedStart) || args.seedStart < 0) throw new Error("--seed-start must be non-negative");
  if (!Number.isInteger(args.seeds) || args.seeds < 1) throw new Error("--seeds must be positive");
  if (!Number.isFinite(args.duration) || args.duration <= 0) throw new Error("--duration must be positive");
  if (!Number.isFinite(args.dt) || args.dt <= 0) throw new Error("--dt must be positive");
  if (!Number.isFinite(args.forceLimit) || args.forceLimit <= 0) throw new Error("--force-limit must be positive");
  if (!Number.isFinite(args.railLimit) || args.railLimit <= 0) throw new Error("--rail-limit must be positive");
  if (!["smoke", "phase10-default", "phase10-output"].includes(args.cellSlate)) {
    throw new Error("--cell-slate must be one of smoke, phase10-default, or phase10-output");
  }
  if (args.limitCells !== null && (!Number.isInteger(args.limitCells) || args.limitCells < 1)) {
    throw new Error("--limit-cells must be a positive integer or all");
  }
  if (!Number.isFinite(args.disturbanceAt) || args.disturbanceAt < 0) throw new Error("--disturbance-at must be non-negative");
  if (!Number.isFinite(args.disturbanceDuration) || args.disturbanceDuration < 0) throw new Error("--disturbance-duration must be non-negative");
  if (!["adversarial", "fixed"].includes(args.disturbanceMode)) {
    throw new Error("--disturbance-mode must be adversarial or fixed");
  }
  if (!Number.isInteger(args.logEvery) || args.logEvery < 1) throw new Error("--log-every must be positive");
  if (!Number.isInteger(args.particleCount) || args.particleCount < 31) throw new Error("--particle-count must be an integer >= 31");
  if (!Number.isFinite(args.horizonSeconds) || args.horizonSeconds <= 0) throw new Error("--horizon-seconds must be positive");
  if (args.presets.some((preset) => !BALANCE_PRESETS[preset])) {
    throw new Error(`Unknown preset in --presets: ${args.presets.join(",")}`);
  }
  const unknownMode = args.modes.find((mode) => !BALANCE_CONTROLLER_MODES[mode]);
  if (unknownMode) throw new Error(`Unknown mode in --modes: ${unknownMode}`);
  const nonRunnableMode = args.modes.find((mode) => BALANCE_CONTROLLER_MODES[mode].status !== "implemented");
  if (nonRunnableMode) {
    throw new Error(`Balance mode ${nonRunnableMode} is ${BALANCE_CONTROLLER_MODES[nonRunnableMode].status}, not runnable yet`);
  }
  if (args.lightElevations.some((value) => !Number.isFinite(value) || value <= 1 || value >= 89)) {
    throw new Error("--light-elevations values must be between 1 and 89 degrees");
  }
  for (const [flag, values] of [
    ["--sensor-delay-steps", args.sensorDelaySteps],
    ["--sensor-noise-std", args.sensorNoiseStd],
    ["--sensor-dropout-rates", args.sensorDropoutRates],
  ]) {
    if (!Array.isArray(values) || values.length === 0 || values.some((value) => !Number.isFinite(value) || value < 0)) {
      throw new Error(`${flag} must contain finite non-negative values`);
    }
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

function safeId(value) {
  return String(value).replaceAll(".", "p").replaceAll("-", "m");
}

function caseId(cell) {
  return [
    cell.axis ?? "smoke",
    safeId(cell.axisValue ?? "grid"),
    `light_${safeId(cell.lightElevationDeg)}`,
    `delay_${safeId(cell.sensorDelaySteps)}`,
    `noise_${safeId(cell.sensorNoiseStd)}`,
    `drop_${safeId(cell.sensorDropoutRate)}`,
    `force_${safeId(cell.forceLimit)}`,
    `rail_${safeId(cell.railLimit)}`,
    `push_${safeId(cell.disturbanceForce)}`,
  ].join("__");
}

function makeSmokeCells(args) {
  const cells = [];
  for (const lightElevationDeg of args.lightElevations) {
    for (const sensorDelaySteps of args.sensorDelaySteps) {
      for (const sensorNoiseStd of args.sensorNoiseStd) {
        for (const sensorDropoutRate of args.sensorDropoutRates) {
          const cell = {
            axis: "smoke",
            axisValue: lightElevationDeg,
            lightElevationDeg,
            sensorDelaySteps,
            sensorNoiseStd,
            sensorDropoutRate,
            forceLimit: args.forceLimit,
            railLimit: args.railLimit,
            disturbanceForce: 0,
            cellClass: "smoke",
            source: "smoke",
          };
          cells.push({
            ...cell,
            cellId: caseId(cell),
          });
        }
      }
    }
  }
  return cells;
}

function makePhase10DefaultCells(args) {
  const base = {
    lightElevationDeg: PHASE10_DEFAULTS.baseLightElevation,
    sensorDelaySteps: PHASE10_DEFAULTS.baseSensorDelaySteps,
    sensorNoiseStd: PHASE10_DEFAULTS.baseSensorNoiseStd,
    sensorDropoutRate: PHASE10_DEFAULTS.baseSensorDropoutRate,
    forceLimit: PHASE10_DEFAULTS.baseForceLimit,
    railLimit: PHASE10_DEFAULTS.baseRailLimit,
    disturbanceForce: PHASE10_DEFAULTS.baseDisturbanceForce,
  };
  const cases = [];
  const addAxis = (axis, axisValue, overrides) => {
    const cell = { axis, axisValue, ...base, ...overrides, source: "phase10-default" };
    cases.push({ ...cell, caseId: caseId(cell) });
  };

  for (const value of PHASE10_DEFAULTS.lightElevations) addAxis("light_elevation", value, { lightElevationDeg: value });
  for (const value of PHASE10_DEFAULTS.sensorDelaySteps) addAxis("sensor_delay", value, { sensorDelaySteps: value });
  for (const value of PHASE10_DEFAULTS.sensorNoiseStd) addAxis("sensor_noise", value, { sensorNoiseStd: value });
  for (const value of PHASE10_DEFAULTS.sensorDropoutRates) addAxis("sensor_dropout", value, { sensorDropoutRate: value });
  for (const value of PHASE10_DEFAULTS.forceLimits) addAxis("force_limit", value, { forceLimit: value });
  for (const value of PHASE10_DEFAULTS.railLimits) addAxis("rail_limit", value, { railLimit: value });
  for (const value of PHASE10_DEFAULTS.disturbanceForces) addAxis("disturbance_force", value, { disturbanceForce: value });

  const presets = args.presets.length > 0 ? args.presets : PHASE10_DEFAULTS.presets;
  return cases.flatMap((cell) => presets.map((preset) => ({
    ...cell,
    preset,
    cellId: `${cell.caseId}__preset_${preset}`,
  })));
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (inQuotes) {
      if (ch === "\"" && next === "\"") {
        field += "\"";
        i += 1;
      } else if (ch === "\"") {
        inQuotes = false;
      } else {
        field += ch;
      }
    } else if (ch === "\"") {
      inQuotes = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field);
      field = "";
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
    } else if (ch !== "\r") {
      field += ch;
    }
  }
  row.push(field);
  if (row.some((value) => value !== "")) rows.push(row);
  if (rows.length === 0) return [];
  const headers = rows[0];
  return rows.slice(1).map((values) => Object.fromEntries(
    headers.map((header, index) => [header, values[index] ?? ""]),
  ));
}

function numberFrom(row, names, fallback = null) {
  for (const name of names) {
    if (!Object.hasOwn(row, name) || row[name] === "") continue;
    const value = Number.parseFloat(row[name]);
    if (Number.isFinite(value)) return value;
  }
  return fallback;
}

async function readPhase10Rows(args) {
  const explicit = args.phase10Cells ? [args.phase10Cells] : [];
  const phase10Out = path.resolve(repoRoot, args.phase10Out);
  const candidates = [
    ...explicit,
    path.join(phase10Out, "envelope.csv"),
    path.join(phase10Out, "cell-class-map.csv"),
  ];
  const errors = [];
  for (const candidate of candidates) {
    const resolved = path.resolve(repoRoot, candidate);
    try {
      const text = await readFile(resolved, "utf8");
      return {
        sourcePath: resolved,
        rows: parseCsv(text),
      };
    } catch (error) {
      errors.push(`${resolved}: ${error.code ?? error.message}`);
    }
  }
  throw new Error(`Could not read Phase 10 cell slate. Tried:\n${errors.join("\n")}`);
}

async function loadPhase10OutputCells(args) {
  const { sourcePath, rows } = await readPhase10Rows(args);
  const cells = rows.map((row) => {
    const sensorDelaySteps = numberFrom(row, ["delay_steps"], null)
      ?? Math.round((numberFrom(row, ["delay_ms"], 0) / 1000) / args.dt);
    return {
      source: "phase10-output",
      sourcePath: path.relative(repoRoot, sourcePath),
      cellId: row.cell_id,
      caseId: row.case_id,
      axis: row.axis,
      axisValue: numberFrom(row, ["axis_value"], row.axis_value),
      preset: row.preset,
      lightElevationDeg: numberFrom(row, ["light_elev_deg"], PHASE10_DEFAULTS.baseLightElevation),
      sensorDelaySteps,
      sensorNoiseStd: numberFrom(row, ["noise_sigma"], PHASE10_DEFAULTS.baseSensorNoiseStd),
      sensorDropoutRate: numberFrom(row, ["dropout_rate"], PHASE10_DEFAULTS.baseSensorDropoutRate),
      forceLimit: numberFrom(row, ["force_limit"], PHASE10_DEFAULTS.baseForceLimit),
      railLimit: numberFrom(row, ["rail_limit"], PHASE10_DEFAULTS.baseRailLimit),
      disturbanceForce: numberFrom(row, ["disturbance_mag"], PHASE10_DEFAULTS.baseDisturbanceForce),
      cellClass: row.cell_class || "",
      staticBoundaryMechanisms: row.static_boundary_mechanisms || "",
    };
  }).filter((cell) => cell.cellId && cell.preset && BALANCE_PRESETS[cell.preset]);

  return filterAndLimitCells(cells, args);
}

function filterAndLimitCells(cells, args) {
  let selected = cells;
  if (args.cellClasses) {
    const wanted = new Set(args.cellClasses);
    selected = selected.filter((cell) => wanted.has(cell.cellClass));
  }
  if (args.presets.length > 0) {
    const wantedPresets = new Set(args.presets);
    selected = selected.filter((cell) => !cell.preset || wantedPresets.has(cell.preset));
  }
  selected = selected.sort((a, b) => (
    String(a.cellClass ?? "").localeCompare(String(b.cellClass ?? ""))
    || String(a.axis ?? "").localeCompare(String(b.axis ?? ""))
    || Number(a.axisValue ?? 0) - Number(b.axisValue ?? 0)
    || String(a.preset ?? "").localeCompare(String(b.preset ?? ""))
    || String(a.cellId).localeCompare(String(b.cellId))
  ));
  return args.limitCells === null ? selected : selected.slice(0, args.limitCells);
}

async function loadCells(args) {
  if (args.cellSlate === "phase10-output") return loadPhase10OutputCells(args);
  if (args.cellSlate === "phase10-default") return filterAndLimitCells(makePhase10DefaultCells(args), args);
  return filterAndLimitCells(makeSmokeCells(args), args);
}

function trialId({ preset, cell, mode, seed }) {
  return `${preset}__${cell.cellId}__${mode}__seed_${String(seed).padStart(3, "0")}`;
}

function terminalOutcome(state, cfg) {
  if (state.fallen) return "fallen";
  if (state.railHit) return "rail_hit";
  if (state.t + cfg.dt * 0.5 >= cfg.duration) return "timeout";
  return "max_steps";
}

function disturbanceForceAt(state, args, cell) {
  const magnitude = cell.disturbanceForce ?? 0;
  if (!magnitude) return 0;
  const active = state.t >= args.disturbanceAt && state.t < args.disturbanceAt + args.disturbanceDuration;
  if (!active) return 0;
  if (args.disturbanceMode === "fixed") return magnitude;
  return -Math.sign(state.theta || 1) * magnitude;
}

function roundObservation(row) {
  const out = {};
  for (const [key, value] of Object.entries(row)) {
    out[key] = typeof value === "number" ? roundNumber(value, 9) : value;
  }
  return out;
}

function runTrial(args, { preset, cell, mode, seed }) {
  const initialState = seededInitialState(preset, seed, args.jitterScale);
  const cfg = normalizeBalanceConfig({
    preset,
    controllerMode: mode,
    seed,
    initialState,
    duration: args.duration,
    dt: args.dt,
    forceLimit: cell.forceLimit ?? args.forceLimit,
    railLimit: cell.railLimit ?? args.railLimit,
    lightElevationDeg: cell.lightElevationDeg,
    sensorDelaySteps: cell.sensorDelaySteps,
    sensorNoiseStd: cell.sensorNoiseStd,
    sensorDropoutRate: cell.sensorDropoutRate,
    bayesParticleCount: args.particleCount,
    bayesHorizonSeconds: args.horizonSeconds,
  });
  let state = initializeBalanceState(cfg);
  const runtime = createBalanceRuntime(cfg);
  const controllerState = {};
  let sensor = sampleShadowSensor(state, runtime, cfg);
  let control = { force: 0, rawForce: 0, saturated: false, phase: "PASSIVE", reason: "initial" };
  const id = trialId({ preset, cell, mode, seed });
  const observationRows = [];
  const beliefRows = [];
  const actionRows = [];
  let thetaSquareSum = 0;
  let confidenceSum = 0;
  let absForceIntegral = 0;
  let maxAbsTheta = Math.abs(state.theta);
  let maxAbsX = Math.abs(state.x);
  let saturationCount = 0;
  let confidenceLossCount = 0;
  let steps = 0;
  const maxSteps = Math.ceil(cfg.duration / cfg.dt);

  while (steps < maxSteps && !state.fallen && !state.railHit && state.t < cfg.duration) {
    sensor = sampleShadowSensor(state, runtime, cfg);
    const observation = serializeBalanceObservation(state, sensor, cfg);
    control = computeBalanceControl(state, sensor, controllerState, cfg);

    thetaSquareSum += state.theta * state.theta;
    confidenceSum += sensor.valid ? sensor.confidence : 0;
    absForceIntegral += Math.abs(control.force) * cfg.dt;
    maxAbsTheta = Math.max(maxAbsTheta, Math.abs(state.theta));
    maxAbsX = Math.max(maxAbsX, Math.abs(state.x));
    if (control.saturated) saturationCount += 1;
    if (!sensor.valid || sensor.confidence < 0.35) confidenceLossCount += 1;

    if (steps % args.logEvery === 0) {
      observationRows.push(roundObservation({
        phase: args.phase,
        trialId: id,
        preset,
        mode,
        seed,
        cellId: cell.cellId,
        axis: cell.axis,
        axisValue: cell.axisValue,
        cellClass: cell.cellClass,
        step: steps,
        ...observation,
      }));
      actionRows.push({
        phase: args.phase,
        trialId: id,
        preset,
        mode,
        seed,
        cellId: cell.cellId,
        axis: cell.axis,
        axisValue: cell.axisValue,
        cellClass: cell.cellClass,
        step: steps,
        t: roundNumber(state.t, 9),
        force: roundNumber(control.force, 9),
        disturbanceForce: roundNumber(disturbanceForceAt(state, args, cell), 9),
        phaseLabel: control.phase,
        reason: control.reason,
        saturated: control.saturated,
      });
      if (mode === "bayes_floor_shadow_particle" && control.belief) {
        beliefRows.push({
          phase: args.phase,
          trialId: id,
          preset,
          mode,
          seed,
          cellId: cell.cellId,
          axis: cell.axis,
          axisValue: cell.axisValue,
          cellClass: cell.cellClass,
          step: steps,
          t: roundNumber(state.t, 9),
          thetaMean: roundNumber(control.belief.thetaMean, 9),
          thetaDotMean: roundNumber(control.belief.thetaDotMean, 9),
          thetaStd: roundNumber(control.belief.thetaStd, 9),
          thetaDotStd: roundNumber(control.belief.thetaDotStd, 9),
          effectiveSampleSize: roundNumber(control.belief.effectiveSampleSize, 6),
          particleCount: control.belief.particleCount,
          candidateCount: control.belief.candidateCount,
          selectedScore: roundNumber(control.belief.selectedScore, 9),
          posteriorReady: control.belief.posteriorReady,
          inverseValid: control.belief.inverseValid,
          inverseTheta: Number.isFinite(control.belief.inverseTheta) ? roundNumber(control.belief.inverseTheta, 9) : "",
          inverseThetaDot: Number.isFinite(control.belief.inverseThetaDot) ? roundNumber(control.belief.inverseThetaDot, 9) : "",
          inverseWeight: roundNumber(control.belief.inverseWeight, 9),
          sundogScore: roundNumber(control.belief.sundogScore, 9),
          proposalScore: roundNumber(control.belief.proposalScore, 9),
          scoreAdvantage: roundNumber(control.belief.scoreAdvantage, 9),
          baseAdvantageThreshold: roundNumber(control.belief.baseAdvantageThreshold, 9),
          advantageThreshold: roundNumber(control.belief.advantageThreshold, 9),
          observationStress: roundNumber(control.belief.observationStress, 9),
          degradationReady: control.belief.degradationReady,
          selectedCandidate: control.belief.selectedCandidate,
          force: roundNumber(control.force, 9),
        });
      }
    }

    const disturbanceForce = disturbanceForceAt(state, args, cell);
    state = integrateBalanceStep(state, control.force + disturbanceForce, cfg);
    steps += 1;
  }

  const outcome = terminalOutcome(state, cfg);
  const simulatedTime = Math.min(state.t, cfg.duration);
  const denom = Math.max(1, steps);
  return {
    result: {
      phase: args.phase,
      trialId: id,
      preset,
      mode,
      seed,
      cellId: cell.cellId,
      axis: cell.axis,
      axisValue: cell.axisValue,
      cellClass: cell.cellClass,
      staticBoundaryMechanisms: cell.staticBoundaryMechanisms ?? "",
      lightElevationDeg: cfg.lightElevationDeg,
      sensorDelaySteps: cfg.sensorDelaySteps,
      sensorNoiseStd: cfg.sensorNoiseStd,
      sensorDropoutRate: cfg.sensorDropoutRate,
      disturbanceForce: cell.disturbanceForce ?? 0,
      outcome,
      success: outcome === "timeout",
      duration: cfg.duration,
      dt: cfg.dt,
      simulatedTime: roundNumber(simulatedTime, 9),
      normalizedSurvival: roundNumber(simulatedTime / cfg.duration, 9),
      rmsTheta: roundNumber(Math.sqrt(thetaSquareSum / denom), 9),
      maxAbsTheta: roundNumber(maxAbsTheta, 9),
      maxAbsX: roundNumber(maxAbsX, 9),
      meanShadowConfidence: roundNumber(confidenceSum / denom, 9),
      forceBudget: roundNumber(absForceIntegral, 9),
      saturationCount,
      confidenceLossCount,
      steps,
      initialX: roundNumber(initialState.x, 9),
      initialXDot: roundNumber(initialState.xDot, 9),
      initialTheta: roundNumber(initialState.theta, 9),
      initialThetaDot: roundNumber(initialState.thetaDot, 9),
    },
    observationRows,
    beliefRows,
    actionRows,
  };
}

function groupBy(rows, keyFn) {
  const groups = new Map();
  for (const row of rows) {
    const key = keyFn(row);
    const group = groups.get(key) ?? [];
    group.push(row);
    groups.set(key, group);
  }
  return groups;
}

function mean(values) {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return null;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function axisFromCellId(cellId) {
  return String(cellId ?? "").split("__")[0] || "";
}

function observationAdmission(row) {
  const axis = row.axis || axisFromCellId(row.cellId);
  const cellClass = row.cellClass || "";
  const mechanisms = String(row.staticBoundaryMechanisms ?? "");
  const observationAxis = ["sensor_delay", "sensor_noise", "sensor_dropout"].includes(axis);
  const failureRegime = cellClass === "failure_regime"
    || /delay_destabilized|sensor_noise_floor|dropped_frames/.test(mechanisms);

  if (observationAxis && failureRegime) {
    return {
      claimGateRequired: false,
      bayesSanityGateRequired: false,
      admissionLane: "reported_only",
      admissionReason: `${axis}_failure_regime`,
    };
  }

  if (observationAxis) {
    return {
      claimGateRequired: true,
      bayesSanityGateRequired: false,
      admissionLane: "observation_parity_gate",
      admissionReason: `${axis}_diagnostic_or_borderline`,
    };
  }

  return {
    claimGateRequired: true,
    bayesSanityGateRequired: true,
    admissionLane: "hard_gate",
    admissionReason: "standard_cell",
  };
}

function makeRegretRows(results) {
  const byKey = groupBy(results, (row) => `${row.preset}\t${row.cellId}\t${row.seed}`);
  const rows = [];
  for (const group of byKey.values()) {
    const byMode = new Map(group.map((row) => [row.mode, row]));
    const bayes = byMode.get("bayes_floor_shadow_particle");
    const sundog = byMode.get("sundog_shadow");
    if (!bayes || !sundog) continue;
    const naive = byMode.get("naive_shadow");
    rows.push({
      phase: bayes.phase,
      preset: bayes.preset,
      cellId: bayes.cellId,
      axis: bayes.axis,
      axisValue: bayes.axisValue,
      cellClass: bayes.cellClass,
      staticBoundaryMechanisms: bayes.staticBoundaryMechanisms,
      sensorDelaySteps: bayes.sensorDelaySteps,
      sensorNoiseStd: bayes.sensorNoiseStd,
      sensorDropoutRate: bayes.sensorDropoutRate,
      seed: bayes.seed,
      bayesOutcome: bayes.outcome,
      sundogOutcome: sundog.outcome,
      bayesNormalizedSurvival: bayes.normalizedSurvival,
      sundogNormalizedSurvival: sundog.normalizedSurvival,
      naiveNormalizedSurvival: naive?.normalizedSurvival ?? "",
      regretVsSundog: roundNumber(bayes.normalizedSurvival - sundog.normalizedSurvival, 9),
      bayesMinusNaive: Number.isFinite(naive?.normalizedSurvival)
        ? roundNumber(bayes.normalizedSurvival - naive.normalizedSurvival, 9)
        : "",
      bayesForceBudget: bayes.forceBudget,
      sundogForceBudget: sundog.forceBudget,
      bayesRmsTheta: bayes.rmsTheta,
      sundogRmsTheta: sundog.rmsTheta,
    });
  }
  return rows.sort((a, b) => (
    a.preset.localeCompare(b.preset)
    || a.cellId.localeCompare(b.cellId)
    || a.seed - b.seed
  ));
}

function makeRegretSummary(regretRows) {
  return Array.from(groupBy(regretRows, (row) => `${row.preset}\t${row.cellId}`).entries())
    .map(([key, rows]) => {
      const [preset, cellId] = key.split("\t");
      const regretValues = rows.map((row) => row.regretVsSundog);
      const bayesMinusNaive = rows.map((row) => row.bayesMinusNaive).filter(Number.isFinite);
      const negativeRegretCount = rows.filter((row) => row.regretVsSundog < -1e-9).length;
      const bayesWorseThanNaiveCount = rows.filter((row) => Number.isFinite(row.bayesMinusNaive) && row.bayesMinusNaive < -1e-9).length;
      const meanRegretVsSundog = roundNumber(mean(regretValues) ?? NaN, 9);
      const bayesSanityPass = bayesWorseThanNaiveCount === 0;
      const admission = observationAdmission(rows[0]);
      const sundogParityPass = meanRegretVsSundog >= -1e-9;
      const claimGatePass = !admission.claimGateRequired
        || (sundogParityPass && (!admission.bayesSanityGateRequired || bayesSanityPass));
      return {
        phase: rows[0].phase,
        preset,
        cellId,
        axis: rows[0].axis,
        axisValue: rows[0].axisValue,
        cellClass: rows[0].cellClass,
        staticBoundaryMechanisms: rows[0].staticBoundaryMechanisms,
        admissionLane: admission.admissionLane,
        admissionReason: admission.admissionReason,
        claimGateRequired: admission.claimGateRequired,
        bayesSanityGateRequired: admission.bayesSanityGateRequired,
        n: rows.length,
        meanRegretVsSundog,
        negativeRegretRate: roundNumber(negativeRegretCount / Math.max(1, rows.length), 9),
        sundogParityPass,
        meanBayesMinusNaive: roundNumber(mean(bayesMinusNaive) ?? NaN, 9),
        bayesWorseThanNaiveRate: roundNumber(bayesWorseThanNaiveCount / Math.max(1, rows.length), 9),
        bayesSanityPass,
        claimGatePass,
      };
    })
    .sort((a, b) => a.preset.localeCompare(b.preset) || a.cellId.localeCompare(b.cellId));
}

function runObservationParityAudit() {
  const cfg = normalizeBalanceConfig({
    preset: "recoverable",
    controllerMode: "sundog_shadow",
    seed: 4101,
    lightElevationDeg: 28,
  });
  const state = initializeBalanceState(cfg);
  const runtime = createBalanceRuntime(cfg);
  const sensor = sampleShadowSensor(state, runtime, cfg);
  const observation = serializeBalanceObservation(state, sensor, cfg);
  const forbiddenPresent = FORBIDDEN_OBSERVATION_KEYS.filter((key) => Object.hasOwn(observation, key));
  const parityChecks = {
    x: observation.x === state.x,
    xDot: observation.xDot === state.xDot,
    shadowResidual: observation.shadowResidual === sensor.residual,
    residualVelocity: observation.residualVelocity === sensor.residualVelocity,
    shadowConfidence: observation.shadowConfidence === sensor.confidence,
    shadowLength: observation.shadowLength === sensor.length,
  };
  return {
    name: "observation_parity",
    pass: forbiddenPresent.length === 0 && Object.values(parityChecks).every(Boolean),
    forbiddenPresent,
    parityChecks,
    admittedKeys: Object.keys(observation),
  };
}

function runNoStateLeakAudit(args) {
  const cfg = normalizeBalanceConfig({
    preset: "recoverable",
    controllerMode: "bayes_floor_shadow_particle",
    seed: 9109,
    lightElevationDeg: 28,
    bayesParticleCount: args.particleCount,
    bayesHorizonSeconds: args.horizonSeconds,
  });
  const baseState = {
    x: 0.125,
    xDot: -0.04,
    theta: 0.18,
    thetaDot: -0.07,
    t: 0,
    fallen: false,
    railHit: false,
  };
  const runtime = createBalanceRuntime(cfg);
  const baseSensor = sampleShadowSensor(baseState, runtime, cfg);
  const variants = [
    { theta: -0.42, thetaDot: -1.1 },
    { theta: 0.01, thetaDot: 0.3 },
    { theta: 0.48, thetaDot: 1.2 },
  ].map((hidden, index) => {
    const state = { ...baseState, ...hidden };
    const sensor = {
      ...baseSensor,
      raw: computeShadowGeometry(state, cfg),
    };
    const controllerState = {};
    const control = computeBalanceControl(state, sensor, controllerState, cfg);
    return {
      index,
      theta: hidden.theta,
      thetaDot: hidden.thetaDot,
      force: control.force,
      phase: control.phase,
    };
  });
  const forces = variants.map((row) => row.force);
  const minForce = Math.min(...forces);
  const maxForce = Math.max(...forces);
  return {
    name: "no_state_leak",
    pass: maxForce - minForce <= 1e-9,
    forceSpan: roundNumber(maxForce - minForce, 12),
    variants,
  };
}

function runUnknownModeAudit() {
  const cfg = normalizeBalanceConfig({ controllerMode: "bogus_balance_mode" });
  const state = initializeBalanceState(cfg);
  const runtime = createBalanceRuntime(cfg);
  const sensor = sampleShadowSensor(state, runtime, cfg);
  try {
    computeBalanceControl(state, sensor, {}, cfg);
    return { name: "unknown_mode_rejection", pass: false, error: null };
  } catch (error) {
    return {
      name: "unknown_mode_rejection",
      pass: /Unknown balance controller mode/.test(error.message),
      error: error.message,
    };
  }
}

function runAudits(args) {
  const audits = [
    runObservationParityAudit(),
    runNoStateLeakAudit(args),
    runUnknownModeAudit(),
  ];
  return {
    pass: audits.every((audit) => audit.pass),
    audits,
  };
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const normalized = typeof value === "number" ? roundNumber(value, 9) : value;
  const text = String(normalized);
  if (/[",\r\n]/.test(text)) return `"${text.replaceAll("\"", "\"\"")}"`;
  return text;
}

function rowsToCsv(rows, columns = null) {
  const cols = columns ?? [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const lines = [cols.join(",")];
  for (const row of rows) lines.push(cols.map((column) => csvEscape(row[column])).join(","));
  return `${lines.join("\n")}\n`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(repoRoot, args.out);
  const cells = await loadCells(args);
  const results = [];
  const observationRows = [];
  const beliefRows = [];
  const actionRows = [];
  const started = performance.now();

  for (const cell of cells) {
    const presets = cell.preset ? [cell.preset] : args.presets;
    for (const preset of presets) {
      for (let i = 0; i < args.seeds; i += 1) {
        const seed = args.seedStart + i;
        for (const mode of args.modes) {
          const trial = runTrial(args, { preset, cell, mode, seed });
          results.push(trial.result);
          observationRows.push(...trial.observationRows);
          beliefRows.push(...trial.beliefRows);
          actionRows.push(...trial.actionRows);
        }
      }
    }
  }

  const elapsedSeconds = (performance.now() - started) / 1000;
  const regretRows = makeRegretRows(results);
  const regretSummaryRows = makeRegretSummary(regretRows);
  const claimGateRows = regretSummaryRows.filter((row) => row.claimGateRequired);
  const reportOnlyRows = regretSummaryRows.filter((row) => !row.claimGateRequired);
  const claimGateFailures = claimGateRows.filter((row) => !row.claimGatePass);
  const audits = runAudits(args);
  const trialRate = results.length / Math.max(elapsedSeconds, 1e-9);
  const manifest = {
    schema: "sundog.balance.phase15-bayesian-floor.v1",
    generatedAt: new Date().toISOString(),
    phase: args.phase,
    out: path.relative(repoRoot, outDir),
    modes: args.modes,
    modeDefinitions: Object.fromEntries(args.modes.map((mode) => [mode, BALANCE_CONTROLLER_MODES[mode]])),
    presets: args.presets,
    cellSlate: args.cellSlate,
    phase10Out: args.phase10Out,
    phase10Cells: args.phase10Cells,
    cellClasses: args.cellClasses,
    limitCells: args.limitCells,
    cells,
    seedStart: args.seedStart,
    seeds: args.seeds,
    duration: args.duration,
    dt: args.dt,
    disturbanceAt: args.disturbanceAt,
    disturbanceDuration: args.disturbanceDuration,
    disturbanceMode: args.disturbanceMode,
    particleCount: args.particleCount,
    horizonSeconds: args.horizonSeconds,
    trialCount: results.length,
    elapsedSeconds: roundNumber(elapsedSeconds, 6),
    trialsPerSecond: roundNumber(trialRate, 6),
    estimatedFullPhase10Trials: args.estimateFullPhase10Trials,
    estimatedFullPhase10Seconds: args.estimateFullPhase10Trials > 0
      ? roundNumber(args.estimateFullPhase10Trials / Math.max(trialRate, 1e-9), 3)
      : null,
    claimGate: {
      policy: "standard cells gate on Bayes-vs-naive sanity and Sundog parity; observation-degradation margin cells gate on Sundog parity while reporting Bayes-vs-naive as a boundary diagnostic; Phase 10 failure-regime observation-degradation cells are reported-only until a separate admission spec promotes them.",
      pass: claimGateFailures.length === 0,
      hardGateCells: claimGateRows.length,
      hardGatePassCells: claimGateRows.length - claimGateFailures.length,
      hardGateFailureCells: claimGateFailures.length,
      reportedOnlyCells: reportOnlyRows.length,
      failures: claimGateFailures.map((row) => ({
        preset: row.preset,
        cellId: row.cellId,
        admissionReason: row.admissionReason,
        meanRegretVsSundog: row.meanRegretVsSundog,
        bayesSanityPass: row.bayesSanityPass,
        bayesSanityGateRequired: row.bayesSanityGateRequired,
        sundogParityPass: row.sundogParityPass,
      })),
    },
    audits,
    note: "Ignored local Phase 15 smoke receipts. Bayesian mode is a same-shadow particle-belief baseline, not a privileged oracle.",
  };
  const profile = {
    profileId: "balance-bayesian-floor-v1",
    implementedMode: "bayes_floor_shadow_particle",
    admittedObservationKeys: [
      "x",
      "xDot",
      "t",
      "shadowTipX",
      "shadowCentroidX",
      "shadowLength",
      "shadowResidual",
      "shadowConfidence",
      "shadowValid",
      "residualVelocity",
      "lengthVelocity",
      "lightElevationDeg",
      "sensorNoiseStd",
      "sensorDelaySteps",
      "sensorDropoutRate",
      "dt",
      "preset",
      "forceLimit",
      "railLimit",
    ],
    forbiddenKeys: FORBIDDEN_OBSERVATION_KEYS,
    claimGateAdmission: {
      hardGate: "standard diagnostic-positive and borderline cells gate on Bayes-vs-naive sanity plus Sundog parity",
      observationParityGate: "sensor_delay, sensor_noise, and sensor_dropout diagnostic/borderline cells gate on Sundog parity; Bayes-vs-naive remains a reported boundary diagnostic",
      reportedOnly: "Phase 10 failure-regime sensor_delay, sensor_noise, and sensor_dropout cells",
    },
    objective: "E_mu[normalized_survival]",
    regret: "bayes_floor_shadow_particle.normalized_survival - sundog_shadow.normalized_survival",
  };

  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  await writeFile(path.join(outDir, "profile.json"), `${JSON.stringify(profile, null, 2)}\n`);
  await writeFile(path.join(outDir, "signature-observations.jsonl"), `${observationRows.map((row) => JSON.stringify(row)).join("\n")}\n`);
  await writeFile(path.join(outDir, "observation-parity.jsonl"), `${audits.audits.map((row) => JSON.stringify(row)).join("\n")}\n`);
  await writeFile(path.join(outDir, "belief-diagnostics.csv"), rowsToCsv(beliefRows));
  await writeFile(path.join(outDir, "bayes-actions.csv"), rowsToCsv(actionRows));
  await writeFile(path.join(outDir, "trial-outcomes.csv"), rowsToCsv(results));
  await writeFile(path.join(outDir, "bayes-regret.csv"), rowsToCsv(regretRows));
  await writeFile(path.join(outDir, "bayes-regret-summary.csv"), rowsToCsv(regretSummaryRows));
  await writeFile(path.join(outDir, "observability-fibers.json"), `${JSON.stringify({
    schema: "sundog.balance.phase15-observability-fibers.v1",
    cells: regretSummaryRows.map((row) => ({
      preset: row.preset,
      cellId: row.cellId,
      meanRegretVsSundog: row.meanRegretVsSundog,
      negativeRegretRate: row.negativeRegretRate,
      bayesSanityPass: row.bayesSanityPass,
      admissionLane: row.admissionLane,
      claimGateRequired: row.claimGateRequired,
      bayesSanityGateRequired: row.bayesSanityGateRequired,
      claimGatePass: row.claimGatePass,
    })),
  }, null, 2)}\n`);

  console.log(`Balance ${args.phase}: ${results.length} trials in ${roundNumber(elapsedSeconds, 3)}s (${roundNumber(trialRate, 2)} trials/s)`);
  console.log(`Audits: ${audits.pass ? "pass" : "FAIL"}`);
  console.log(`Claim gate: ${manifest.claimGate.pass ? "pass" : "FAIL"} (${manifest.claimGate.hardGatePassCells}/${manifest.claimGate.hardGateCells} hard-gate cells; ${manifest.claimGate.reportedOnlyCells} reported-only cells)`);
  console.log(`Wrote ${path.relative(repoRoot, outDir)}`);
  for (const row of regretSummaryRows) {
    console.log(`${row.preset} ${row.cellId}: mean regret vs sundog ${row.meanRegretVsSundog}, Bayes sanity ${row.bayesSanityPass}, claim gate ${row.claimGateRequired ? row.claimGatePass : "reported-only"}`);
  }
  if (!audits.pass) process.exitCode = 1;
}

if (path.resolve(process.argv[1] ?? "") === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
