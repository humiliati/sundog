import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  normalizeConfig,
  runTrial,
  seededInitialParticle,
} from "../public/js/threebody-core.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const SENSOR_TIER_DEFINITIONS = Object.freeze({
  accelerometer_array_noisy: {
    label: "accelerometer_array_noisy",
    modes: ["seek_sensor_accel", "track_sensor_accel", "track_sensor_accel_guarded"],
    variants: ["accelerometer_array_noisy"],
    sweep: "noise",
  },
  delayed_local_probe: {
    label: "delayed_local_probe",
    modes: ["seek_sensor_delayed", "track_sensor_delayed"],
    variants: ["delayed_local_probe"],
    sweep: "delay",
  },
  micro_maneuver_noisy: {
    label: "micro_maneuver_noisy",
    modes: ["seek_sensor_micro", "track_sensor_micro"],
    variants: ["micro_maneuver_noisy"],
    sweep: "noise_delay_contamination",
  },
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
    phase: "phase8-calibration",
    out: "results/threebody/phase8-calibration-sweep",
    seedStart: 0,
    seeds: 3,
    regimes: ["stable", "near_escape", "near_collision", "chaotic"],
    sensorTiers: ["accelerometer_array_noisy", "delayed_local_probe", "micro_maneuver_noisy"],
    duration: 8,
    dt: 0.01,
    logEvery: 20,
    massRatio: 1,
    thrustLimit: 0.5,
    targetTidal: 2,
    tidalSpikeThreshold: 50,
    localAccelerationWarningThreshold: 10,
    eventWarningHorizon: 1,
    sensorAuditEvery: 40,
    sensorNoiseSweep: [0, 0.01, 0.03, 0.06, 0.12],
    sensorDelaySweep: [0, 2, 5, 10],
    microManeuverContaminationSweep: [0.08],
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
    else if (flag === "--sensor-tiers") args.sensorTiers = parseList(value);
    else if (flag === "--duration") args.duration = Number.parseFloat(value);
    else if (flag === "--dt") args.dt = Number.parseFloat(value);
    else if (flag === "--log-every") args.logEvery = Number.parseInt(value, 10);
    else if (flag === "--mass-ratio") args.massRatio = Number.parseFloat(value);
    else if (flag === "--thrust-limit") args.thrustLimit = Number.parseFloat(value);
    else if (flag === "--target-tidal") args.targetTidal = Number.parseFloat(value);
    else if (flag === "--tidal-spike-threshold") args.tidalSpikeThreshold = Number.parseFloat(value);
    else if (flag === "--local-acceleration-warning-threshold") args.localAccelerationWarningThreshold = Number.parseFloat(value);
    else if (flag === "--event-warning-horizon") args.eventWarningHorizon = Number.parseFloat(value);
    else if (flag === "--sensor-audit-every") args.sensorAuditEvery = Number.parseInt(value, 10);
    else if (flag === "--sensor-noise-sweep") args.sensorNoiseSweep = parseNumberList(value);
    else if (flag === "--sensor-delay-sweep") args.sensorDelaySweep = parseIntegerList(value);
    else if (flag === "--micro-maneuver-contamination-sweep") args.microManeuverContaminationSweep = parseNumberList(value);
    else throw new Error(`Unknown flag: ${flag}`);
  }

  if (!Number.isInteger(args.seeds) || args.seeds < 1) throw new Error("--seeds must be a positive integer");
  if (!Number.isFinite(args.duration) || args.duration <= 0) throw new Error("--duration must be positive");
  if (!Number.isFinite(args.dt) || args.dt <= 0) throw new Error("--dt must be positive");
  if (!Number.isInteger(args.sensorAuditEvery) || args.sensorAuditEvery < 1) {
    throw new Error("--sensor-audit-every must be a positive integer");
  }
  if (args.sensorTiers.some((tier) => !SENSOR_TIER_DEFINITIONS[tier])) {
    throw new Error(`--sensor-tiers contains an unknown tier; valid tiers: ${Object.keys(SENSOR_TIER_DEFINITIONS).join(",")}`);
  }
  if (args.sensorNoiseSweep.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error("--sensor-noise-sweep values must be non-negative numbers");
  }
  if (args.sensorDelaySweep.some((value) => !Number.isInteger(value) || value < 0)) {
    throw new Error("--sensor-delay-sweep values must be non-negative integers");
  }
  if (args.microManeuverContaminationSweep.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error("--micro-maneuver-contamination-sweep values must be non-negative numbers");
  }

  args.sensorNoiseSweep = [...new Set(args.sensorNoiseSweep)].sort((a, b) => a - b);
  args.sensorDelaySweep = [...new Set(args.sensorDelaySweep)].sort((a, b) => a - b);
  args.microManeuverContaminationSweep = [...new Set(args.microManeuverContaminationSweep)].sort((a, b) => a - b);
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

function calibrationCases(args) {
  const cases = [];
  for (const sensorTier of args.sensorTiers) {
    const definition = SENSOR_TIER_DEFINITIONS[sensorTier];
    if (definition.sweep === "noise") {
      for (const sensorNoiseStd of args.sensorNoiseSweep) {
        cases.push({
          sensorTier,
          sensorNoiseStd,
          sensorDelaySteps: 0,
          microManeuverContaminationStd: args.microManeuverContaminationSweep[0] ?? 0,
        });
      }
    } else if (definition.sweep === "delay") {
      for (const sensorDelaySteps of args.sensorDelaySweep) {
        cases.push({
          sensorTier,
          sensorNoiseStd: 0,
          sensorDelaySteps,
          microManeuverContaminationStd: args.microManeuverContaminationSweep[0] ?? 0,
        });
      }
    } else {
      for (const sensorNoiseStd of args.sensorNoiseSweep) {
        for (const sensorDelaySteps of args.sensorDelaySweep) {
          for (const microManeuverContaminationStd of args.microManeuverContaminationSweep) {
            cases.push({
              sensorTier,
              sensorNoiseStd,
              sensorDelaySteps,
              microManeuverContaminationStd,
            });
          }
        }
      }
    }
  }
  return cases;
}

function caseId(calibrationCase) {
  return [
    calibrationCase.sensorTier,
    `noise_${calibrationCase.sensorNoiseStd}`,
    `delay_${calibrationCase.sensorDelaySteps}`,
    `micro_${calibrationCase.microManeuverContaminationStd}`,
  ].join("__").replaceAll(".", "p");
}

function trialId(calibrationCase, regime, mode, seed) {
  return `${caseId(calibrationCase)}__${regime}__${mode}__seed_${String(seed).padStart(3, "0")}`;
}

function modesForCase(calibrationCase) {
  return ["off", ...SENSOR_TIER_DEFINITIONS[calibrationCase.sensorTier].modes];
}

function variantsForCase(calibrationCase) {
  return SENSOR_TIER_DEFINITIONS[calibrationCase.sensorTier].variants;
}

function makeTrialConfig(args, calibrationCase, seed, regime, mode) {
  return normalizeConfig({
    seed,
    regime,
    controllerMode: mode,
    duration: args.duration,
    dt: args.dt,
    logEvery: args.logEvery,
    massRatio: args.massRatio,
    thrustLimit: args.thrustLimit,
    targetTidal: args.targetTidal,
    tidalSpikeThreshold: args.tidalSpikeThreshold,
    localAccelerationWarningThreshold: args.localAccelerationWarningThreshold,
    eventWarningHorizon: args.eventWarningHorizon,
    sensorAuditVariants: variantsForCase(calibrationCase),
    sensorAuditEvery: args.sensorAuditEvery,
    sensorNoiseStd: calibrationCase.sensorNoiseStd,
    sensorDelaySteps: calibrationCase.sensorDelaySteps,
    microManeuverContaminationStd: calibrationCase.microManeuverContaminationStd,
    initialParticle: seededInitialParticle(seed, regime),
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
      sensorTier: trial.sensorTier,
      sensorNoiseStd: trial.sensorNoiseStd,
      sensorDelaySteps: trial.sensorDelaySteps,
      microManeuverContaminationStd: trial.microManeuverContaminationStd,
      regime: trial.regime,
      seed: trial.seed,
      controllerMode: trial.controllerMode,
      terminalOutcome: trial.summary.terminalOutcome,
      passiveOutcome: passive?.summary.terminalOutcome,
      outcomeDeltaVsPassive: passive ? compareOutcome(trial.summary.terminalOutcome, passive.summary.terminalOutcome) : null,
      simulatedTime: trial.summary.simulatedTime,
      passiveSimulatedTime: passive?.summary.simulatedTime,
      simulatedTimeDeltaVsPassive: passive ? trial.summary.simulatedTime - passive.summary.simulatedTime : null,
      totalDeltaV: trial.summary.totalDeltaV,
      minPrimaryDistance: trial.summary.minPrimaryDistance,
      passiveMinPrimaryDistance: passive?.summary.minPrimaryDistance,
      tidalWarningF1: trial.summary.tidalWarningF1,
      tidalMagnitudeAuroc: trial.summary.tidalMagnitudeAuroc,
      localAccelerationWarningF1: trial.summary.localAccelerationWarningF1,
      localAccelerationMagnitudeAuroc: trial.summary.localAccelerationMagnitudeAuroc,
      log: trial.log,
    };
  }).sort((a, b) => (
    a.sensorTier.localeCompare(b.sensorTier)
    || a.sensorNoiseStd - b.sensorNoiseStd
    || a.sensorDelaySteps - b.sensorDelaySteps
    || a.microManeuverContaminationStd - b.microManeuverContaminationStd
    || a.regime.localeCompare(b.regime)
    || a.seed - b.seed
    || a.controllerMode.localeCompare(b.controllerMode)
  ));
}

function makeSummaryRows(pairedRows) {
  const groups = new Map();
  for (const row of pairedRows) {
    const key = [
      row.sensorTier,
      row.sensorNoiseStd,
      row.sensorDelaySteps,
      row.microManeuverContaminationStd,
      row.regime,
      row.controllerMode,
    ].join("\t");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }

  return Array.from(groups.entries()).map(([key, rows]) => {
    const [
      sensorTier,
      sensorNoiseStdText,
      sensorDelayStepsText,
      microManeuverContaminationStdText,
      regime,
      controllerMode,
    ] = key.split("\t");
    const bounded = rows.filter((row) => row.terminalOutcome === "bounded").length;
    const closeApproach = rows.filter((row) => row.terminalOutcome === "close_approach").length;
    const escape = rows.filter((row) => row.terminalOutcome === "escape").length;
    const improved = rows.filter((row) => row.outcomeDeltaVsPassive > 0).length;
    const worsened = rows.filter((row) => row.outcomeDeltaVsPassive < 0).length;
    const tied = rows.filter((row) => row.outcomeDeltaVsPassive === 0).length;
    return {
      sensorTier,
      sensorNoiseStd: Number.parseFloat(sensorNoiseStdText),
      sensorDelaySteps: Number.parseInt(sensorDelayStepsText, 10),
      microManeuverContaminationStd: Number.parseFloat(microManeuverContaminationStdText),
      regime,
      controllerMode,
      n: rows.length,
      bounded,
      closeApproach,
      escape,
      survivalRate: roundMetric(ratio(bounded, rows.length)),
      improvedOutcomeVsPassive: improved,
      worsenedOutcomeVsPassive: worsened,
      tiedOutcomeVsPassive: tied,
      meanOutcomeDeltaVsPassive: mean(rows.map((row) => row.outcomeDeltaVsPassive)),
      meanTimeDeltaVsPassive: mean(rows.map((row) => row.simulatedTimeDeltaVsPassive)),
      meanDeltaV: mean(rows.map((row) => row.totalDeltaV)),
      meanMinPrimaryDistance: mean(rows.map((row) => row.minPrimaryDistance)),
      meanTidalWarningF1: mean(rows.map((row) => row.tidalWarningF1)),
      meanTidalMagnitudeAuroc: mean(rows.map((row) => row.tidalMagnitudeAuroc)),
      meanLocalAccelerationWarningF1: mean(rows.map((row) => row.localAccelerationWarningF1)),
      meanLocalAccelerationMagnitudeAuroc: mean(rows.map((row) => row.localAccelerationMagnitudeAuroc)),
    };
  }).sort((a, b) => (
    a.sensorTier.localeCompare(b.sensorTier)
    || a.sensorNoiseStd - b.sensorNoiseStd
    || a.sensorDelaySteps - b.sensorDelaySteps
    || a.microManeuverContaminationStd - b.microManeuverContaminationStd
    || a.regime.localeCompare(b.regime)
    || a.controllerMode.localeCompare(b.controllerMode)
  ));
}

function makeSensorErrorRows(trials) {
  const groups = new Map();
  for (const trial of trials) {
    for (const sample of trial.sensorAudit) {
      const key = [
        trial.sensorTier,
        trial.sensorNoiseStd,
        trial.sensorDelaySteps,
        trial.microManeuverContaminationStd,
        trial.regime,
        trial.controllerMode,
        sample.sensorVariant,
      ].join("\t");
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(sample);
    }
  }

  return Array.from(groups.entries()).map(([key, samples]) => {
    const [
      sensorTier,
      sensorNoiseStdText,
      sensorDelayStepsText,
      microManeuverContaminationStdText,
      regime,
      controllerMode,
      sensorVariant,
    ] = key.split("\t");
    return {
      sensorTier,
      sensorNoiseStd: Number.parseFloat(sensorNoiseStdText),
      sensorDelaySteps: Number.parseInt(sensorDelayStepsText, 10),
      microManeuverContaminationStd: Number.parseFloat(microManeuverContaminationStdText),
      regime,
      controllerMode,
      sensorVariant,
      n: samples.length,
      delayWarmupCount: samples.filter((sample) => sample.delayWarmup === true).length,
      meanIdealMagnitude: mean(samples.map((sample) => sample.idealMagnitude)),
      meanEstimatedMagnitude: mean(samples.map((sample) => sample.estimatedMagnitude)),
      meanMagnitudeAbsError: mean(samples.map((sample) => sample.magnitudeAbsError)),
      meanMagnitudeRelError: mean(samples.map((sample) => sample.magnitudeRelError)),
      rmsMagnitudeAbsError: roundMetric(Math.sqrt(mean(samples.map((sample) => sample.magnitudeAbsError ** 2)) ?? 0)),
      maxMagnitudeAbsError: Math.max(...samples.map((sample) => sample.magnitudeAbsError).filter(Number.isFinite)),
      meanComponentRmse: mean(samples.map((sample) => sample.componentRmse)),
    };
  }).sort((a, b) => (
    a.sensorTier.localeCompare(b.sensorTier)
    || a.sensorNoiseStd - b.sensorNoiseStd
    || a.sensorDelaySteps - b.sensorDelaySteps
    || a.microManeuverContaminationStd - b.microManeuverContaminationStd
    || a.regime.localeCompare(b.regime)
    || a.controllerMode.localeCompare(b.controllerMode)
  ));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(repoRoot, args.out);
  const trialDir = path.join(outDir, "trials");
  await mkdir(trialDir, { recursive: true });

  const startedAt = new Date().toISOString();
  const cases = calibrationCases(args);
  const trials = [];
  const manifest = {
    schema: `sundog.threebody.${args.phase}.v1`,
    startedAt,
    purpose: "Phase 8 sensor-tier calibration sweep across sensor noise, delay, and micro-maneuver contamination.",
    args,
    sensorTierDefinitions: SENSOR_TIER_DEFINITIONS,
    cases,
    trials: [],
  };

  for (const calibrationCase of cases) {
    for (const regime of args.regimes) {
      for (const mode of modesForCase(calibrationCase)) {
        for (let offset = 0; offset < args.seeds; offset += 1) {
          const seed = args.seedStart + offset;
          const config = makeTrialConfig(args, calibrationCase, seed, regime, mode);
          const id = trialId(calibrationCase, regime, mode, seed);
          const trial = runTrial(config);
          const relativePath = `trials/${id}.jsonl`;
          const logPath = path.join(outDir, relativePath);
          const lines = trial.records.map((record) => JSON.stringify(record)).join("\n");
          await writeFile(logPath, `${lines}\n`, "utf8");

          const row = {
            caseId: caseId(calibrationCase),
            ...calibrationCase,
            seed,
            regime,
            controllerMode: mode,
            log: relativePath,
            summary: trial.summary,
            sensorAudit: trial.sensorAudit,
          };
          trials.push(row);
          manifest.trials.push({
            id,
            caseId: row.caseId,
            ...calibrationCase,
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
  const summaryRows = makeSummaryRows(pairedRows);
  const sensorErrorRows = makeSensorErrorRows(trials);
  await writeFile(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  await writeFile(path.join(outDir, "paired.csv"), rowsToCsv(pairedRows), "utf8");
  await writeFile(path.join(outDir, "summary.csv"), rowsToCsv(summaryRows), "utf8");
  await writeFile(path.join(outDir, "sensor-error-summary.csv"), rowsToCsv(sensorErrorRows), "utf8");

  const outcomeCounts = manifest.trials.reduce((counts, trial) => {
    counts[trial.summary.terminalOutcome] = (counts[trial.summary.terminalOutcome] ?? 0) + 1;
    return counts;
  }, {});

  console.log(`[threebody] wrote ${manifest.trials.length} calibration trials to ${path.relative(repoRoot, outDir)}`);
  console.log(`[threebody] wrote paired.csv, summary.csv, and sensor-error-summary.csv`);
  console.log(`[threebody] outcomes ${JSON.stringify(outcomeCounts)}`);
}

main().catch((error) => {
  console.error(`[threebody] ${error.message}`);
  process.exitCode = 1;
});
