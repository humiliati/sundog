import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  normalizeConfig,
  runTrial,
  seededInitialParticle,
} from "../public/js/threebody-core.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function parseNumberList(value) {
  return parseList(value).map((item) => Number.parseFloat(item));
}

function parseArgs(argv) {
  const args = {
    phase: "phase9",
    out: "results/threebody/phase9-operating-envelope",
    seedStart: 0,
    seeds: 3,
    regimes: ["stable", "near_escape", "chaotic"],
    modes: ["off", "track_sensor_accel_guarded"],
    duration: 8,
    dt: 0.01,
    logEvery: 40,
    massRatio: 1,
    targetTidal: 2,
    tidalSpikeThreshold: 50,
    localAccelerationWarningThreshold: 10,
    eventWarningHorizon: 1,
    radiusScales: [0.95, 1, 1.05],
    velocityScales: [0.9, 1, 1.1],
    thrustLimits: [0.4, 0.6],
    sensorNoiseSweep: [0, 0.01, 0.03],
    trackGuardMinRadiusSweep: [1.15],
    trackGuardMaxLocalAccelerationSweep: [2.5, 3.5],
    trackGuardMaxTidalMagnitudeSweep: [35],
    candidateMaxWorsenedRate: 0.1,
    candidateMinSurvivalDelta: 0.001,
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
    else if (flag === "--regimes") args.regimes = parseList(value);
    else if (flag === "--modes") args.modes = parseList(value);
    else if (flag === "--duration") args.duration = Number.parseFloat(value);
    else if (flag === "--dt") args.dt = Number.parseFloat(value);
    else if (flag === "--log-every") args.logEvery = Number.parseInt(value, 10);
    else if (flag === "--mass-ratio") args.massRatio = Number.parseFloat(value);
    else if (flag === "--target-tidal") args.targetTidal = Number.parseFloat(value);
    else if (flag === "--tidal-spike-threshold") args.tidalSpikeThreshold = Number.parseFloat(value);
    else if (flag === "--local-acceleration-warning-threshold") args.localAccelerationWarningThreshold = Number.parseFloat(value);
    else if (flag === "--event-warning-horizon") args.eventWarningHorizon = Number.parseFloat(value);
    else if (flag === "--radius-scales") args.radiusScales = parseNumberList(value);
    else if (flag === "--velocity-scales") args.velocityScales = parseNumberList(value);
    else if (flag === "--thrust-limits") args.thrustLimits = parseNumberList(value);
    else if (flag === "--sensor-noise-sweep") args.sensorNoiseSweep = parseNumberList(value);
    else if (flag === "--track-guard-min-radius-sweep") args.trackGuardMinRadiusSweep = parseNumberList(value);
    else if (flag === "--track-guard-max-local-acceleration-sweep") args.trackGuardMaxLocalAccelerationSweep = parseNumberList(value);
    else if (flag === "--track-guard-max-tidal-magnitude-sweep") args.trackGuardMaxTidalMagnitudeSweep = parseNumberList(value);
    else if (flag === "--candidate-max-worsened-rate") args.candidateMaxWorsenedRate = Number.parseFloat(value);
    else if (flag === "--candidate-min-survival-delta") args.candidateMinSurvivalDelta = Number.parseFloat(value);
    else throw new Error(`Unknown flag: ${flag}`);
  }

  if (!Number.isInteger(args.seeds) || args.seeds < 1) throw new Error("--seeds must be a positive integer");
  if (!Number.isFinite(args.duration) || args.duration <= 0) throw new Error("--duration must be positive");
  if (!Number.isFinite(args.dt) || args.dt <= 0) throw new Error("--dt must be positive");
  if (!Number.isInteger(args.logEvery) || args.logEvery < 1) throw new Error("--log-every must be a positive integer");
  for (const [name, values] of [
    ["--radius-scales", args.radiusScales],
    ["--velocity-scales", args.velocityScales],
    ["--thrust-limits", args.thrustLimits],
    ["--sensor-noise-sweep", args.sensorNoiseSweep],
    ["--track-guard-min-radius-sweep", args.trackGuardMinRadiusSweep],
    ["--track-guard-max-local-acceleration-sweep", args.trackGuardMaxLocalAccelerationSweep],
    ["--track-guard-max-tidal-magnitude-sweep", args.trackGuardMaxTidalMagnitudeSweep],
  ]) {
    if (!Array.isArray(values) || values.length === 0 || values.some((value) => !Number.isFinite(value) || value < 0)) {
      throw new Error(`${name} must contain non-negative finite numbers`);
    }
  }
  if (!args.modes.includes("off")) {
    throw new Error("--modes must include off for matched passive baselines");
  }

  args.radiusScales = [...new Set(args.radiusScales)].sort((a, b) => a - b);
  args.velocityScales = [...new Set(args.velocityScales)].sort((a, b) => a - b);
  args.thrustLimits = [...new Set(args.thrustLimits)].sort((a, b) => a - b);
  args.sensorNoiseSweep = [...new Set(args.sensorNoiseSweep)].sort((a, b) => a - b);
  args.trackGuardMinRadiusSweep = [...new Set(args.trackGuardMinRadiusSweep)].sort((a, b) => a - b);
  args.trackGuardMaxLocalAccelerationSweep = [...new Set(args.trackGuardMaxLocalAccelerationSweep)].sort((a, b) => a - b);
  args.trackGuardMaxTidalMagnitudeSweep = [...new Set(args.trackGuardMaxTidalMagnitudeSweep)].sort((a, b) => a - b);
  return args;
}

function csvValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function rowsToCsv(rows, explicitColumns = null) {
  const columns = explicitColumns ?? [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const lines = [columns.join(",")];
  for (const row of rows) lines.push(columns.map((column) => csvValue(row[column])).join(","));
  return `${lines.join("\n")}\n`;
}

function mean(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) return null;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function ratio(numerator, denominator) {
  if (denominator <= 0) return null;
  return numerator / denominator;
}

function roundMetric(value, digits = 6) {
  if (!Number.isFinite(value)) return null;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function compareOutcome(outcome, baselineOutcome) {
  const rank = {
    invalid: 0,
    close_approach: 1,
    escape: 2,
    bounded: 3,
  };
  return (rank[outcome] ?? 0) - (rank[baselineOutcome] ?? 0);
}

function scaleInitialParticle(particle, radiusScale, velocityScale) {
  return {
    x: particle.x * radiusScale,
    y: particle.y * radiusScale,
    vx: particle.vx * velocityScale,
    vy: particle.vy * velocityScale,
  };
}

function envelopeCases(args) {
  const cases = [];
  for (const radiusScale of args.radiusScales) {
    for (const velocityScale of args.velocityScales) {
      for (const thrustLimit of args.thrustLimits) {
        for (const sensorNoiseStd of args.sensorNoiseSweep) {
          for (const trackGuardMinRadius of args.trackGuardMinRadiusSweep) {
            for (const trackGuardMaxLocalAcceleration of args.trackGuardMaxLocalAccelerationSweep) {
              for (const trackGuardMaxTidalMagnitude of args.trackGuardMaxTidalMagnitudeSweep) {
                cases.push({
                  radiusScale,
                  velocityScale,
                  thrustLimit,
                  sensorNoiseStd,
                  sensorDelaySteps: 0,
                  microManeuverContaminationStd: 0.08,
                  trackGuardMinRadius,
                  trackGuardMaxLocalAcceleration,
                  trackGuardMaxTidalMagnitude,
                });
              }
            }
          }
        }
      }
    }
  }
  return cases;
}

function caseId(envelopeCase) {
  return [
    `r_${envelopeCase.radiusScale}`,
    `v_${envelopeCase.velocityScale}`,
    `thrust_${envelopeCase.thrustLimit}`,
    `noise_${envelopeCase.sensorNoiseStd}`,
    `gminr_${envelopeCase.trackGuardMinRadius}`,
    `gaccel_${envelopeCase.trackGuardMaxLocalAcceleration}`,
    `gtidal_${envelopeCase.trackGuardMaxTidalMagnitude}`,
  ].join("__").replaceAll(".", "p");
}

function trialId(envelopeCase, regime, mode, seed) {
  return `${caseId(envelopeCase)}__${regime}__${mode}__seed_${String(seed).padStart(3, "0")}`;
}

function makeTrialConfig(args, envelopeCase, seed, regime, mode) {
  const baseParticle = seededInitialParticle(seed, regime);
  return normalizeConfig({
    seed,
    regime,
    controllerMode: mode,
    duration: args.duration,
    dt: args.dt,
    logEvery: args.logEvery,
    massRatio: args.massRatio,
    thrustLimit: envelopeCase.thrustLimit,
    targetTidal: args.targetTidal,
    tidalSpikeThreshold: args.tidalSpikeThreshold,
    localAccelerationWarningThreshold: args.localAccelerationWarningThreshold,
    eventWarningHorizon: args.eventWarningHorizon,
    sensorAuditVariants: [],
    sensorNoiseStd: envelopeCase.sensorNoiseStd,
    sensorDelaySteps: envelopeCase.sensorDelaySteps,
    microManeuverContaminationStd: envelopeCase.microManeuverContaminationStd,
    trackGuardMinRadius: envelopeCase.trackGuardMinRadius,
    trackGuardMaxLocalAcceleration: envelopeCase.trackGuardMaxLocalAcceleration,
    trackGuardMaxTidalMagnitude: envelopeCase.trackGuardMaxTidalMagnitude,
    initialParticle: scaleInitialParticle(baseParticle, envelopeCase.radiusScale, envelopeCase.velocityScale),
  });
}

function makePairedRows(trials) {
  const byKey = new Map();
  for (const trial of trials) {
    byKey.set(`${trial.caseId}\t${trial.regime}\t${trial.seed}\t${trial.controllerMode}`, trial);
  }

  return trials.filter((trial) => trial.controllerMode !== "off").map((trial) => {
    const passive = byKey.get(`${trial.caseId}\t${trial.regime}\t${trial.seed}\toff`);
    return {
      caseId: trial.caseId,
      regime: trial.regime,
      seed: trial.seed,
      controllerMode: trial.controllerMode,
      radiusScale: trial.radiusScale,
      velocityScale: trial.velocityScale,
      thrustLimit: trial.thrustLimit,
      sensorNoiseStd: trial.sensorNoiseStd,
      trackGuardMinRadius: trial.trackGuardMinRadius,
      trackGuardMaxLocalAcceleration: trial.trackGuardMaxLocalAcceleration,
      trackGuardMaxTidalMagnitude: trial.trackGuardMaxTidalMagnitude,
      terminalOutcome: trial.summary.terminalOutcome,
      passiveOutcome: passive?.summary.terminalOutcome,
      outcomeDeltaVsPassive: passive ? compareOutcome(trial.summary.terminalOutcome, passive.summary.terminalOutcome) : null,
      simulatedTime: trial.summary.simulatedTime,
      passiveSimulatedTime: passive?.summary.simulatedTime,
      simulatedTimeDeltaVsPassive: passive ? trial.summary.simulatedTime - passive.summary.simulatedTime : null,
      totalDeltaV: trial.summary.totalDeltaV,
      minPrimaryDistance: trial.summary.minPrimaryDistance,
      passiveMinPrimaryDistance: passive?.summary.minPrimaryDistance,
      tidalMagnitudeAuroc: trial.summary.tidalMagnitudeAuroc,
      localAccelerationMagnitudeAuroc: trial.summary.localAccelerationMagnitudeAuroc,
      log: trial.log,
    };
  }).sort((a, b) => (
    a.regime.localeCompare(b.regime)
    || a.radiusScale - b.radiusScale
    || a.velocityScale - b.velocityScale
    || a.thrustLimit - b.thrustLimit
    || a.sensorNoiseStd - b.sensorNoiseStd
    || a.trackGuardMaxLocalAcceleration - b.trackGuardMaxLocalAcceleration
    || a.seed - b.seed
    || a.controllerMode.localeCompare(b.controllerMode)
  ));
}

function summarizeRows(rows, args, includeRegime = true) {
  const groups = new Map();
  for (const row of rows) {
    const key = [
      ...(includeRegime ? [row.regime] : []),
      row.controllerMode,
      row.radiusScale,
      row.velocityScale,
      row.thrustLimit,
      row.sensorNoiseStd,
      row.trackGuardMinRadius,
      row.trackGuardMaxLocalAcceleration,
      row.trackGuardMaxTidalMagnitude,
    ].join("\t");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }

  return Array.from(groups.entries()).map(([key, group]) => {
    const values = key.split("\t");
    const offset = includeRegime ? 1 : 0;
    const regime = includeRegime ? values[0] : "all";
    const controllerMode = values[offset];
    const bounded = group.filter((row) => row.terminalOutcome === "bounded").length;
    const passiveBounded = group.filter((row) => row.passiveOutcome === "bounded").length;
    const worsened = group.filter((row) => row.outcomeDeltaVsPassive < 0).length;
    const improved = group.filter((row) => row.outcomeDeltaVsPassive > 0).length;
    const survivalRate = ratio(bounded, group.length);
    const passiveSurvivalRate = ratio(passiveBounded, group.length);
    const survivalDeltaVsPassive = survivalRate !== null && passiveSurvivalRate !== null
      ? survivalRate - passiveSurvivalRate
      : null;
    const worsenedRate = ratio(worsened, group.length);
    const candidateEnvelope = survivalDeltaVsPassive !== null
      && survivalDeltaVsPassive >= args.candidateMinSurvivalDelta
      && worsenedRate !== null
      && worsenedRate <= args.candidateMaxWorsenedRate;
    const regionClass = candidateEnvelope
      ? "promising"
      : worsenedRate !== null && worsenedRate > 0.25 ? "risky"
        : survivalDeltaVsPassive !== null && survivalDeltaVsPassive < 0 ? "negative"
          : survivalDeltaVsPassive === 0 && worsenedRate === 0 ? "neutral"
            : "mixed";

    return {
      regime,
      controllerMode,
      radiusScale: Number.parseFloat(values[offset + 1]),
      velocityScale: Number.parseFloat(values[offset + 2]),
      thrustLimit: Number.parseFloat(values[offset + 3]),
      sensorNoiseStd: Number.parseFloat(values[offset + 4]),
      trackGuardMinRadius: Number.parseFloat(values[offset + 5]),
      trackGuardMaxLocalAcceleration: Number.parseFloat(values[offset + 6]),
      trackGuardMaxTidalMagnitude: Number.parseFloat(values[offset + 7]),
      n: group.length,
      bounded,
      passiveBounded,
      survivalRate: roundMetric(survivalRate),
      passiveSurvivalRate: roundMetric(passiveSurvivalRate),
      survivalDeltaVsPassive: roundMetric(survivalDeltaVsPassive),
      improvedOutcomeVsPassive: improved,
      worsenedOutcomeVsPassive: worsened,
      worsenedRate: roundMetric(worsenedRate),
      meanOutcomeDeltaVsPassive: roundMetric(mean(group.map((row) => row.outcomeDeltaVsPassive))),
      meanTimeDeltaVsPassive: roundMetric(mean(group.map((row) => row.simulatedTimeDeltaVsPassive))),
      meanDeltaV: roundMetric(mean(group.map((row) => row.totalDeltaV))),
      meanMinPrimaryDistance: roundMetric(mean(group.map((row) => row.minPrimaryDistance))),
      meanTidalMagnitudeAuroc: roundMetric(mean(group.map((row) => row.tidalMagnitudeAuroc))),
      meanLocalAccelerationMagnitudeAuroc: roundMetric(mean(group.map((row) => row.localAccelerationMagnitudeAuroc))),
      candidateEnvelope,
      regionClass,
    };
  }).sort((a, b) => (
    Number(b.candidateEnvelope) - Number(a.candidateEnvelope)
    || (b.survivalDeltaVsPassive ?? -Infinity) - (a.survivalDeltaVsPassive ?? -Infinity)
    || (a.worsenedRate ?? Infinity) - (b.worsenedRate ?? Infinity)
    || a.regime.localeCompare(b.regime)
    || a.radiusScale - b.radiusScale
    || a.velocityScale - b.velocityScale
    || a.thrustLimit - b.thrustLimit
    || a.sensorNoiseStd - b.sensorNoiseStd
  ));
}

function makeBestByCellRows(envelopeRows) {
  const groups = new Map();
  for (const row of envelopeRows) {
    const key = `${row.regime}\t${row.radiusScale}\t${row.velocityScale}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }

  return Array.from(groups.entries()).map(([key, rows]) => {
    const [regime, radiusScaleText, velocityScaleText] = key.split("\t");
    const best = [...rows].sort((a, b) => (
      Number(b.candidateEnvelope) - Number(a.candidateEnvelope)
      || (b.survivalDeltaVsPassive ?? -Infinity) - (a.survivalDeltaVsPassive ?? -Infinity)
      || (a.worsenedRate ?? Infinity) - (b.worsenedRate ?? Infinity)
      || (b.survivalRate ?? -Infinity) - (a.survivalRate ?? -Infinity)
    ))[0];
    return {
      regime,
      radiusScale: Number.parseFloat(radiusScaleText),
      velocityScale: Number.parseFloat(velocityScaleText),
      bestControllerMode: best.controllerMode,
      bestRegionClass: best.regionClass,
      bestCandidateEnvelope: best.candidateEnvelope,
      bestThrustLimit: best.thrustLimit,
      bestSensorNoiseStd: best.sensorNoiseStd,
      bestTrackGuardMaxLocalAcceleration: best.trackGuardMaxLocalAcceleration,
      bestSurvivalRate: best.survivalRate,
      bestPassiveSurvivalRate: best.passiveSurvivalRate,
      bestSurvivalDeltaVsPassive: best.survivalDeltaVsPassive,
      bestWorsenedRate: best.worsenedRate,
      bestMeanDeltaV: best.meanDeltaV,
    };
  }).sort((a, b) => (
    a.regime.localeCompare(b.regime)
    || a.radiusScale - b.radiusScale
    || a.velocityScale - b.velocityScale
  ));
}

function velocityColumn(value) {
  return `v_${String(value).replace(".", "p")}`;
}

function makeCellMatrixRows(bestByCellRows, valueKey) {
  const velocities = [...new Set(bestByCellRows.map((row) => row.velocityScale))].sort((a, b) => a - b);
  const groups = new Map();
  for (const row of bestByCellRows) {
    const key = `${row.regime}\t${row.radiusScale}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }

  return Array.from(groups.entries()).map(([key, rows]) => {
    const [regime, radiusScaleText] = key.split("\t");
    const byVelocity = new Map(rows.map((row) => [row.velocityScale, row]));
    return {
      regime,
      radiusScale: Number.parseFloat(radiusScaleText),
      ...Object.fromEntries(velocities.map((velocity) => [
        velocityColumn(velocity),
        byVelocity.get(velocity)?.[valueKey] ?? null,
      ])),
    };
  }).sort((a, b) => (
    a.regime.localeCompare(b.regime)
    || a.radiusScale - b.radiusScale
  ));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(repoRoot, args.out);
  const trialDir = path.join(outDir, "trials");
  await mkdir(trialDir, { recursive: true });

  const cases = envelopeCases(args);
  const trials = [];
  const manifest = {
    schema: `sundog.threebody.${args.phase}.v1`,
    startedAt: new Date().toISOString(),
    purpose: "Phase 9 operating-envelope map for guarded sensor-tier control across initial-condition, thrust, noise, and guard settings.",
    args,
    cases,
    trials: [],
  };

  for (const envelopeCase of cases) {
    for (const regime of args.regimes) {
      for (const mode of args.modes) {
        for (let offset = 0; offset < args.seeds; offset += 1) {
          const seed = args.seedStart + offset;
          const config = makeTrialConfig(args, envelopeCase, seed, regime, mode);
          const id = trialId(envelopeCase, regime, mode, seed);
          const trial = runTrial(config);
          const relativePath = `trials/${id}.jsonl`;
          const logPath = path.join(outDir, relativePath);
          const lines = trial.records.map((record) => JSON.stringify(record)).join("\n");
          await writeFile(logPath, `${lines}\n`, "utf8");

          const row = {
            caseId: caseId(envelopeCase),
            ...envelopeCase,
            seed,
            regime,
            controllerMode: mode,
            log: relativePath,
            summary: trial.summary,
          };
          trials.push(row);
          manifest.trials.push({
            id,
            caseId: row.caseId,
            ...envelopeCase,
            seed,
            regime,
            controllerMode: mode,
            initialParticle: config.initialParticle,
            log: relativePath,
            summary: trial.summary,
          });
        }
      }
    }
  }

  manifest.completedAt = new Date().toISOString();
  const pairedRows = makePairedRows(trials);
  const envelopeRows = summarizeRows(pairedRows, args, true);
  const aggregateRows = summarizeRows(pairedRows, args, false);
  const bestByCellRows = makeBestByCellRows(envelopeRows);
  const cellClassMapRows = makeCellMatrixRows(bestByCellRows, "bestRegionClass");
  const cellDeltaMapRows = makeCellMatrixRows(bestByCellRows, "bestSurvivalDeltaVsPassive");
  const candidateRows = envelopeRows.filter((row) => row.candidateEnvelope);

  await writeFile(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  await writeFile(path.join(outDir, "paired.csv"), rowsToCsv(pairedRows), "utf8");
  await writeFile(path.join(outDir, "envelope-map.csv"), rowsToCsv(envelopeRows), "utf8");
  await writeFile(path.join(outDir, "aggregate-envelope.csv"), rowsToCsv(aggregateRows), "utf8");
  await writeFile(path.join(outDir, "best-by-cell.csv"), rowsToCsv(bestByCellRows), "utf8");
  await writeFile(path.join(outDir, "cell-class-map.csv"), rowsToCsv(cellClassMapRows), "utf8");
  await writeFile(path.join(outDir, "cell-delta-map.csv"), rowsToCsv(cellDeltaMapRows), "utf8");
  await writeFile(
    path.join(outDir, "candidate-envelope.csv"),
    rowsToCsv(candidateRows, envelopeRows.length > 0 ? Object.keys(envelopeRows[0]) : []),
    "utf8",
  );

  const outcomeCounts = manifest.trials.reduce((counts, trial) => {
    counts[trial.summary.terminalOutcome] = (counts[trial.summary.terminalOutcome] ?? 0) + 1;
    return counts;
  }, {});

  console.log(`[threebody] wrote ${manifest.trials.length} phase 9 trials to ${path.relative(repoRoot, outDir)}`);
  console.log(`[threebody] wrote paired.csv, envelope-map.csv, aggregate-envelope.csv, best-by-cell.csv, cell-class-map.csv, cell-delta-map.csv, and candidate-envelope.csv`);
  console.log(`[threebody] candidate envelope rows ${candidateRows.length}/${envelopeRows.length}`);
  console.log(`[threebody] outcomes ${JSON.stringify(outcomeCounts)}`);
}

main().catch((error) => {
  console.error(`[threebody] ${error.message}`);
  process.exitCode = 1;
});
