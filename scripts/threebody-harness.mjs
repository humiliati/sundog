import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  evaluateLocalAccelerationWarningThreshold,
  evaluateTidalWarningThreshold,
  makeEventDiagnosticSamples,
  normalizeConfig,
  runTrial,
  seededInitialParticle,
} from "../public/js/threebody-core.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const MODE_DEFINITIONS = Object.freeze({
  off: "Passive baseline: no thrust.",
  naive: "Naive local baseline: local acceleration opposition plus velocity damping; no tidal-gradient structure.",
  scan: "SCAN ablation: periodic thrust perturbation only.",
  seek: "Sundog SEEK prototype: descends local tidal-magnitude gradient.",
  track: "Sundog TRACK prototype: follows local tidal-gradient direction to maintain target tidal magnitude.",
  seek_noisy: "Tidal-signal ablation: SEEK with noisy tidal magnitude and gradient observations.",
  track_noisy: "Tidal-signal ablation: TRACK with noisy tidal magnitude and gradient observations.",
  seek_shuffled: "Tidal-signal ablation: SEEK with deterministic shuffled tidal-gradient direction.",
  track_shuffled: "Tidal-signal ablation: TRACK with deterministic shuffled tidal-gradient direction.",
  seek_sensor_accel: "Phase 8 sensor-tier controller: SEEK using noisy accelerometer-array tidal estimates.",
  track_sensor_accel: "Phase 8 sensor-tier controller: TRACK using noisy accelerometer-array tidal estimates.",
  seek_sensor_delayed: "Phase 8 sensor-tier controller: SEEK using delayed local-probe tidal estimates.",
  track_sensor_delayed: "Phase 8 sensor-tier controller: TRACK using delayed local-probe tidal estimates.",
  seek_sensor_micro: "Phase 8 sensor-tier controller: SEEK using noisy delayed micro-maneuver tidal estimates.",
  track_sensor_micro: "Phase 8 sensor-tier controller: TRACK using noisy delayed micro-maneuver tidal estimates.",
  oracle: "Privileged full-state lookahead guard: scores candidate thrust vectors using simulator state; heuristic, not optimal.",
});

const PRIMARY_METRICS = Object.freeze([
  "terminalOutcome",
  "simulatedTime",
  "totalDeltaV",
  "minPrimaryDistance",
  "saturationCount",
  "targetBandLossCount",
  "tidalWarningLeadTime",
  "tidalWarningPrecision",
  "tidalWarningRecall",
  "tidalWarningF1",
  "tidalWarningFalseAlarmRate",
  "tidalMagnitudeAuroc",
  "localAccelerationWarningLeadTime",
  "localAccelerationWarningPrecision",
  "localAccelerationWarningRecall",
  "localAccelerationWarningF1",
  "localAccelerationWarningFalseAlarmRate",
  "localAccelerationMagnitudeAuroc",
]);

const EVENT_DIAGNOSTIC_METRICS = Object.freeze([
  "firstHazardTime",
  "firstTidalWarningTime",
  "tidalWarningLeadTime",
  "tidalWarningThreshold",
  "eventWarningHorizon",
  "eventSampleCount",
  "positiveEventSampleCount",
  "tidalWarningCount",
  "tidalWarningTruePositive",
  "tidalWarningFalsePositive",
  "tidalWarningTrueNegative",
  "tidalWarningFalseNegative",
  "tidalWarningPrecision",
  "tidalWarningRecall",
  "tidalWarningF1",
  "tidalWarningFalseAlarmRate",
  "tidalMagnitudeAuroc",
  "firstLocalAccelerationWarningTime",
  "localAccelerationWarningLeadTime",
  "localAccelerationWarningThreshold",
  "localAccelerationWarningCount",
  "localAccelerationWarningTruePositive",
  "localAccelerationWarningFalsePositive",
  "localAccelerationWarningTrueNegative",
  "localAccelerationWarningFalseNegative",
  "localAccelerationWarningPrecision",
  "localAccelerationWarningRecall",
  "localAccelerationWarningF1",
  "localAccelerationWarningFalseAlarmRate",
  "localAccelerationMagnitudeAuroc",
]);

function parseList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function parseNumberList(value) {
  return parseList(value).map((item) => Number.parseFloat(item));
}

function parseArgs(argv) {
  const args = {
    phase: "phase5",
    out: "results/threebody/phase5-smoke",
    seedStart: 0,
    seeds: 3,
    regimes: ["stable", "near_escape", "near_collision", "chaotic"],
    modes: ["off"],
    duration: 8,
    dt: 0.01,
    logEvery: 20,
    massRatio: 1,
    thrustLimit: 0.5,
    targetTidal: 2,
    tidalSpikeThreshold: 50,
    localAccelerationWarningThreshold: 10,
    eventWarningHorizon: 1,
    thresholdSweep: [5, 10, 20, 35, 50, 75, 100, 150, 250],
    localAccelerationThresholdSweep: [1, 2, 3, 5, 8, 13, 21, 34, 55],
    calibrationBins: 8,
    sensorVariants: [],
    sensorAuditEvery: 20,
    sensorNoiseStd: 0.03,
    sensorDelaySteps: 5,
    microManeuverContaminationStd: 0.08,
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
    else if (flag === "--thrust-limit") args.thrustLimit = Number.parseFloat(value);
    else if (flag === "--target-tidal") args.targetTidal = Number.parseFloat(value);
    else if (flag === "--tidal-spike-threshold") args.tidalSpikeThreshold = Number.parseFloat(value);
    else if (flag === "--local-acceleration-warning-threshold") args.localAccelerationWarningThreshold = Number.parseFloat(value);
    else if (flag === "--event-warning-horizon") args.eventWarningHorizon = Number.parseFloat(value);
    else if (flag === "--threshold-sweep") args.thresholdSweep = parseNumberList(value);
    else if (flag === "--local-acceleration-threshold-sweep") args.localAccelerationThresholdSweep = parseNumberList(value);
    else if (flag === "--calibration-bins") args.calibrationBins = Number.parseInt(value, 10);
    else if (flag === "--sensor-variants") args.sensorVariants = parseList(value);
    else if (flag === "--sensor-audit-every") args.sensorAuditEvery = Number.parseInt(value, 10);
    else if (flag === "--sensor-noise-std") args.sensorNoiseStd = Number.parseFloat(value);
    else if (flag === "--sensor-delay-steps") args.sensorDelaySteps = Number.parseInt(value, 10);
    else if (flag === "--micro-maneuver-contamination-std") args.microManeuverContaminationStd = Number.parseFloat(value);
    else throw new Error(`Unknown flag: ${flag}`);
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
  if (!Number.isFinite(args.eventWarningHorizon) || args.eventWarningHorizon <= 0) {
    throw new Error("--event-warning-horizon must be positive");
  }
  if (!Number.isFinite(args.tidalSpikeThreshold) || args.tidalSpikeThreshold <= 0) {
    throw new Error("--tidal-spike-threshold must be positive");
  }
  if (!Number.isFinite(args.localAccelerationWarningThreshold) || args.localAccelerationWarningThreshold <= 0) {
    throw new Error("--local-acceleration-warning-threshold must be positive");
  }
  if (!Array.isArray(args.thresholdSweep) || args.thresholdSweep.length === 0) {
    throw new Error("--threshold-sweep must contain at least one number");
  }
  if (args.thresholdSweep.some((threshold) => !Number.isFinite(threshold) || threshold <= 0)) {
    throw new Error("--threshold-sweep values must be positive numbers");
  }
  if (!Array.isArray(args.localAccelerationThresholdSweep) || args.localAccelerationThresholdSweep.length === 0) {
    throw new Error("--local-acceleration-threshold-sweep must contain at least one number");
  }
  if (args.localAccelerationThresholdSweep.some((threshold) => !Number.isFinite(threshold) || threshold <= 0)) {
    throw new Error("--local-acceleration-threshold-sweep values must be positive numbers");
  }
  if (!Number.isInteger(args.calibrationBins) || args.calibrationBins < 2) {
    throw new Error("--calibration-bins must be an integer >= 2");
  }
  if (!Number.isInteger(args.sensorAuditEvery) || args.sensorAuditEvery < 1) {
    throw new Error("--sensor-audit-every must be a positive integer");
  }
  if (!Number.isFinite(args.sensorNoiseStd) || args.sensorNoiseStd < 0) {
    throw new Error("--sensor-noise-std must be non-negative");
  }
  if (!Number.isInteger(args.sensorDelaySteps) || args.sensorDelaySteps < 0) {
    throw new Error("--sensor-delay-steps must be a non-negative integer");
  }
  if (!Number.isFinite(args.microManeuverContaminationStd) || args.microManeuverContaminationStd < 0) {
    throw new Error("--micro-maneuver-contamination-std must be non-negative");
  }
  args.thresholdSweep = [...new Set(args.thresholdSweep)].sort((a, b) => a - b);
  args.localAccelerationThresholdSweep = [...new Set(args.localAccelerationThresholdSweep)].sort((a, b) => a - b);
  return args;
}

function trialId({ regime, mode, seed }) {
  return `${regime}_${mode}_seed_${String(seed).padStart(3, "0")}`;
}

function mean(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) return null;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function sum(values) {
  return values.reduce((total, value) => total + (Number.isFinite(value) ? value : 0), 0);
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

function csvValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function makeSummaryRows(trials) {
  const groups = new Map();
  for (const trial of trials) {
    const key = `${trial.regime}\t${trial.controllerMode}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(trial);
  }

  return Array.from(groups.entries()).map(([key, group]) => {
    const [regime, controllerMode] = key.split("\t");
    const countOutcome = (outcome) => group.filter((trial) => trial.summary.terminalOutcome === outcome).length;
    return {
      regime,
      controllerMode,
      n: group.length,
      bounded: countOutcome("bounded"),
      closeApproach: countOutcome("close_approach"),
      escape: countOutcome("escape"),
      invalid: countOutcome("invalid"),
      meanSimulatedTime: mean(group.map((trial) => trial.summary.simulatedTime)),
      meanTotalDeltaV: mean(group.map((trial) => trial.summary.totalDeltaV)),
      meanSaturationCount: mean(group.map((trial) => trial.summary.saturationCount)),
      meanTargetBandLossCount: mean(group.map((trial) => trial.summary.targetBandLossCount)),
      meanMinPrimaryDistance: mean(group.map((trial) => trial.summary.minPrimaryDistance)),
      meanMaxRadius: mean(group.map((trial) => trial.summary.maxRadius)),
      meanTidalWarningLeadTime: mean(group.map((trial) => trial.summary.tidalWarningLeadTime)),
      meanTidalWarningPrecision: mean(group.map((trial) => trial.summary.tidalWarningPrecision)),
      meanTidalWarningRecall: mean(group.map((trial) => trial.summary.tidalWarningRecall)),
      meanTidalWarningF1: mean(group.map((trial) => trial.summary.tidalWarningF1)),
      meanTidalWarningFalseAlarmRate: mean(group.map((trial) => trial.summary.tidalWarningFalseAlarmRate)),
      meanTidalMagnitudeAuroc: mean(group.map((trial) => trial.summary.tidalMagnitudeAuroc)),
      meanLocalAccelerationWarningLeadTime: mean(group.map((trial) => trial.summary.localAccelerationWarningLeadTime)),
      meanLocalAccelerationWarningPrecision: mean(group.map((trial) => trial.summary.localAccelerationWarningPrecision)),
      meanLocalAccelerationWarningRecall: mean(group.map((trial) => trial.summary.localAccelerationWarningRecall)),
      meanLocalAccelerationWarningF1: mean(group.map((trial) => trial.summary.localAccelerationWarningF1)),
      meanLocalAccelerationWarningFalseAlarmRate: mean(group.map((trial) => trial.summary.localAccelerationWarningFalseAlarmRate)),
      meanLocalAccelerationMagnitudeAuroc: mean(group.map((trial) => trial.summary.localAccelerationMagnitudeAuroc)),
    };
  }).sort((a, b) => (
    a.regime.localeCompare(b.regime) || a.controllerMode.localeCompare(b.controllerMode)
  ));
}

function rowsToCsv(rows, explicitColumns = null) {
  const columns = explicitColumns ?? [
    ...new Set(rows.flatMap((row) => Object.keys(row))),
  ];
  const lines = [columns.join(",")];
  for (const row of rows) {
    lines.push(columns.map((column) => csvValue(row[column])).join(","));
  }
  return `${lines.join("\n")}\n`;
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

function makePairedRows(trials) {
  const byKey = new Map();
  for (const trial of trials) {
    byKey.set(`${trial.regime}\t${trial.seed}\t${trial.controllerMode}`, trial);
  }

  return trials.map((trial) => {
    const passive = byKey.get(`${trial.regime}\t${trial.seed}\toff`);
    const oracle = byKey.get(`${trial.regime}\t${trial.seed}\toracle`);
    const passiveSummary = passive?.summary;
    const oracleSummary = oracle?.summary;
    return {
      regime: trial.regime,
      seed: trial.seed,
      controllerMode: trial.controllerMode,
      terminalOutcome: trial.summary.terminalOutcome,
      passiveOutcome: passiveSummary?.terminalOutcome,
      oracleOutcome: oracleSummary?.terminalOutcome,
      outcomeDeltaVsPassive: passiveSummary ? compareOutcome(trial.summary.terminalOutcome, passiveSummary.terminalOutcome) : null,
      outcomeDeltaVsOracle: oracleSummary ? compareOutcome(trial.summary.terminalOutcome, oracleSummary.terminalOutcome) : null,
      simulatedTime: trial.summary.simulatedTime,
      passiveSimulatedTime: passiveSummary?.simulatedTime,
      oracleSimulatedTime: oracleSummary?.simulatedTime,
      simulatedTimeDeltaVsPassive: passiveSummary ? trial.summary.simulatedTime - passiveSummary.simulatedTime : null,
      simulatedTimeDeltaVsOracle: oracleSummary ? trial.summary.simulatedTime - oracleSummary.simulatedTime : null,
      totalDeltaV: trial.summary.totalDeltaV,
      passiveTotalDeltaV: passiveSummary?.totalDeltaV,
      oracleTotalDeltaV: oracleSummary?.totalDeltaV,
      minPrimaryDistance: trial.summary.minPrimaryDistance,
      passiveMinPrimaryDistance: passiveSummary?.minPrimaryDistance,
      oracleMinPrimaryDistance: oracleSummary?.minPrimaryDistance,
      tidalWarningLeadTime: trial.summary.tidalWarningLeadTime,
      tidalWarningPrecision: trial.summary.tidalWarningPrecision,
      tidalWarningRecall: trial.summary.tidalWarningRecall,
      tidalWarningF1: trial.summary.tidalWarningF1,
      tidalWarningFalseAlarmRate: trial.summary.tidalWarningFalseAlarmRate,
      tidalMagnitudeAuroc: trial.summary.tidalMagnitudeAuroc,
      localAccelerationWarningLeadTime: trial.summary.localAccelerationWarningLeadTime,
      localAccelerationWarningPrecision: trial.summary.localAccelerationWarningPrecision,
      localAccelerationWarningRecall: trial.summary.localAccelerationWarningRecall,
      localAccelerationWarningF1: trial.summary.localAccelerationWarningF1,
      localAccelerationWarningFalseAlarmRate: trial.summary.localAccelerationWarningFalseAlarmRate,
      localAccelerationMagnitudeAuroc: trial.summary.localAccelerationMagnitudeAuroc,
      log: trial.log,
    };
  }).sort((a, b) => (
    a.regime.localeCompare(b.regime)
    || a.seed - b.seed
    || a.controllerMode.localeCompare(b.controllerMode)
  ));
}

function makeComparisonRows(pairedRows) {
  const groups = new Map();
  for (const row of pairedRows) {
    const key = `${row.regime}\t${row.controllerMode}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }

  return Array.from(groups.entries()).map(([key, group]) => {
    const [regime, controllerMode] = key.split("\t");
    const nonPassive = group.filter((row) => row.controllerMode !== "off");
    const rows = nonPassive.length > 0 ? nonPassive : group;
    return {
      regime,
      controllerMode,
      n: rows.length,
      improvedOutcomeVsPassive: rows.filter((row) => row.outcomeDeltaVsPassive > 0).length,
      worsenedOutcomeVsPassive: rows.filter((row) => row.outcomeDeltaVsPassive < 0).length,
      tiedOutcomeVsPassive: rows.filter((row) => row.outcomeDeltaVsPassive === 0).length,
      meanTimeDeltaVsPassive: mean(rows.map((row) => row.simulatedTimeDeltaVsPassive)),
      meanTimeDeltaVsOracle: mean(rows.map((row) => row.simulatedTimeDeltaVsOracle)),
      meanDeltaV: mean(rows.map((row) => row.totalDeltaV)),
      meanMinPrimaryDistance: mean(rows.map((row) => row.minPrimaryDistance)),
      meanTidalWarningLeadTime: mean(rows.map((row) => row.tidalWarningLeadTime)),
      meanTidalWarningF1: mean(rows.map((row) => row.tidalWarningF1)),
      meanTidalWarningFalseAlarmRate: mean(rows.map((row) => row.tidalWarningFalseAlarmRate)),
      meanTidalMagnitudeAuroc: mean(rows.map((row) => row.tidalMagnitudeAuroc)),
      meanLocalAccelerationWarningLeadTime: mean(rows.map((row) => row.localAccelerationWarningLeadTime)),
      meanLocalAccelerationWarningF1: mean(rows.map((row) => row.localAccelerationWarningF1)),
      meanLocalAccelerationWarningFalseAlarmRate: mean(rows.map((row) => row.localAccelerationWarningFalseAlarmRate)),
      meanLocalAccelerationMagnitudeAuroc: mean(rows.map((row) => row.localAccelerationMagnitudeAuroc)),
    };
  }).sort((a, b) => (
    a.regime.localeCompare(b.regime) || a.controllerMode.localeCompare(b.controllerMode)
  ));
}

function makeEventMetricRows(trials) {
  return trials.map((trial) => ({
    regime: trial.regime,
    seed: trial.seed,
    controllerMode: trial.controllerMode,
    terminalOutcome: trial.summary.terminalOutcome,
    ...Object.fromEntries(EVENT_DIAGNOSTIC_METRICS.map((metric) => [metric, trial.summary[metric]])),
    log: trial.log,
  })).sort((a, b) => (
    a.regime.localeCompare(b.regime)
    || a.seed - b.seed
    || a.controllerMode.localeCompare(b.controllerMode)
  ));
}

function warningScoreSpecs(args) {
  return [
    {
      scoreType: "tidal_magnitude",
      scoreColumn: "tidalMagnitude",
      thresholds: args.thresholdSweep,
      evaluate: evaluateTidalWarningThreshold,
      mapMetrics: (metrics) => ({
        firstWarningTime: metrics.firstTidalWarningTime,
        warningLeadTime: metrics.tidalWarningLeadTime,
        warningCount: metrics.tidalWarningCount,
        warningTruePositive: metrics.tidalWarningTruePositive,
        warningFalsePositive: metrics.tidalWarningFalsePositive,
        warningTrueNegative: metrics.tidalWarningTrueNegative,
        warningFalseNegative: metrics.tidalWarningFalseNegative,
        warningPrecision: metrics.tidalWarningPrecision,
        warningRecall: metrics.tidalWarningRecall,
        warningF1: metrics.tidalWarningF1,
        warningFalseAlarmRate: metrics.tidalWarningFalseAlarmRate,
      }),
    },
    {
      scoreType: "local_acceleration",
      scoreColumn: "localAccelerationMagnitude",
      thresholds: args.localAccelerationThresholdSweep,
      evaluate: evaluateLocalAccelerationWarningThreshold,
      mapMetrics: (metrics) => ({
        firstWarningTime: metrics.firstLocalAccelerationWarningTime,
        warningLeadTime: metrics.localAccelerationWarningLeadTime,
        warningCount: metrics.localAccelerationWarningCount,
        warningTruePositive: metrics.localAccelerationWarningTruePositive,
        warningFalsePositive: metrics.localAccelerationWarningFalsePositive,
        warningTrueNegative: metrics.localAccelerationWarningTrueNegative,
        warningFalseNegative: metrics.localAccelerationWarningFalseNegative,
        warningPrecision: metrics.localAccelerationWarningPrecision,
        warningRecall: metrics.localAccelerationWarningRecall,
        warningF1: metrics.localAccelerationWarningF1,
        warningFalseAlarmRate: metrics.localAccelerationWarningFalseAlarmRate,
      }),
    },
  ];
}

function makeThresholdSweepRows(analysisTrials, scoreSpecs) {
  return analysisTrials.flatMap((trial) => scoreSpecs.flatMap((scoreSpec) => scoreSpec.thresholds.map((threshold) => {
    const metrics = scoreSpec.evaluate(trial.eventHistory, threshold, trial.config);
    const mapped = scoreSpec.mapMetrics(metrics);
    return {
      regime: trial.regime,
      seed: trial.seed,
      controllerMode: trial.controllerMode,
      terminalOutcome: trial.summary.terminalOutcome,
      scoreType: scoreSpec.scoreType,
      threshold,
      firstHazardTime: metrics.firstHazardTime,
      firstWarningTime: mapped.firstWarningTime,
      warningLeadTime: mapped.warningLeadTime,
      eventSampleCount: metrics.eventSampleCount,
      positiveEventSampleCount: metrics.positiveEventSampleCount,
      warningCount: mapped.warningCount,
      warningTruePositive: mapped.warningTruePositive,
      warningFalsePositive: mapped.warningFalsePositive,
      warningTrueNegative: mapped.warningTrueNegative,
      warningFalseNegative: mapped.warningFalseNegative,
      warningPrecision: mapped.warningPrecision,
      warningRecall: mapped.warningRecall,
      warningF1: mapped.warningF1,
      warningFalseAlarmRate: mapped.warningFalseAlarmRate,
      log: trial.log,
    };
  }))).sort((a, b) => (
    a.regime.localeCompare(b.regime)
    || a.seed - b.seed
    || a.controllerMode.localeCompare(b.controllerMode)
    || a.scoreType.localeCompare(b.scoreType)
    || a.threshold - b.threshold
  ));
}

function makeThresholdSummaryRows(thresholdSweepRows) {
  const groups = new Map();
  for (const row of thresholdSweepRows) {
    const key = `${row.regime}\t${row.controllerMode}\t${row.scoreType}\t${row.threshold}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }

  return Array.from(groups.entries()).map(([key, rows]) => {
    const [regime, controllerMode, scoreType, thresholdText] = key.split("\t");
    const truePositive = sum(rows.map((row) => row.warningTruePositive));
    const falsePositive = sum(rows.map((row) => row.warningFalsePositive));
    const trueNegative = sum(rows.map((row) => row.warningTrueNegative));
    const falseNegative = sum(rows.map((row) => row.warningFalseNegative));
    const precision = ratio(truePositive, truePositive + falsePositive);
    const recall = ratio(truePositive, truePositive + falseNegative);
    const f1 = precision !== null && recall !== null && precision + recall > 0
      ? 2 * precision * recall / (precision + recall)
      : null;
    return {
      regime,
      controllerMode,
      scoreType,
      threshold: Number.parseFloat(thresholdText),
      n: rows.length,
      eventSampleCount: sum(rows.map((row) => row.eventSampleCount)),
      positiveEventSampleCount: sum(rows.map((row) => row.positiveEventSampleCount)),
      warningCount: sum(rows.map((row) => row.warningCount)),
      warningTruePositive: truePositive,
      warningFalsePositive: falsePositive,
      warningTrueNegative: trueNegative,
      warningFalseNegative: falseNegative,
      meanWarningLeadTime: mean(rows.map((row) => row.warningLeadTime)),
      warningPrecision: roundMetric(precision),
      warningRecall: roundMetric(recall),
      warningF1: roundMetric(f1),
      warningFalseAlarmRate: roundMetric(ratio(falsePositive, falsePositive + trueNegative)),
    };
  }).sort((a, b) => (
    a.regime.localeCompare(b.regime)
    || a.controllerMode.localeCompare(b.controllerMode)
    || a.scoreType.localeCompare(b.scoreType)
    || a.threshold - b.threshold
  ));
}

function makeCalibrationRows(analysisTrials, scoreSpecs, binCount) {
  const groups = new Map();
  for (const trial of analysisTrials) {
    for (const scoreSpec of scoreSpecs) {
      const key = `${trial.regime}\t${trial.controllerMode}\t${scoreSpec.scoreType}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(...makeEventDiagnosticSamples(trial.eventHistory, trial.config, scoreSpec.scoreColumn));
    }
  }

  return Array.from(groups.entries()).flatMap(([key, samples]) => {
    const [regime, controllerMode, scoreType] = key.split("\t");
    const sorted = samples
      .filter((sample) => Number.isFinite(sample.score))
      .sort((a, b) => a.score - b.score);
    if (sorted.length === 0) return [];

    const rows = [];
    const effectiveBinCount = Math.min(binCount, sorted.length);
    for (let bin = 0; bin < effectiveBinCount; bin += 1) {
      const start = Math.floor(bin * sorted.length / effectiveBinCount);
      const end = Math.floor((bin + 1) * sorted.length / effectiveBinCount);
      const binSamples = sorted.slice(start, end);
      const positives = binSamples.filter((sample) => sample.label).length;
      rows.push({
        regime,
        controllerMode,
        scoreType,
        bin,
        n: binSamples.length,
        scoreMin: roundMetric(binSamples[0].score),
        scoreMax: roundMetric(binSamples.at(-1).score),
        meanScore: roundMetric(mean(binSamples.map((sample) => sample.score))),
        positiveEventSamples: positives,
        observedEventRate: roundMetric(ratio(positives, binSamples.length)),
      });
    }
    return rows;
  }).sort((a, b) => (
    a.regime.localeCompare(b.regime)
    || a.controllerMode.localeCompare(b.controllerMode)
    || a.scoreType.localeCompare(b.scoreType)
    || a.bin - b.bin
  ));
}

function makeSensorModelSampleRows(analysisTrials) {
  return analysisTrials.flatMap((trial) => trial.sensorAudit.map((sample) => ({
    regime: trial.regime,
    seed: trial.seed,
    controllerMode: trial.controllerMode,
    terminalOutcome: trial.summary.terminalOutcome,
    ...sample,
    log: trial.log,
  }))).sort((a, b) => (
    a.regime.localeCompare(b.regime)
    || a.seed - b.seed
    || a.controllerMode.localeCompare(b.controllerMode)
    || a.sensorVariant.localeCompare(b.sensorVariant)
    || a.time - b.time
  ));
}

function makeSensorModelSummaryRows(sensorModelSampleRows) {
  const groups = new Map();
  for (const row of sensorModelSampleRows) {
    const key = `${row.regime}\t${row.controllerMode}\t${row.sensorVariant}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }

  return Array.from(groups.entries()).map(([key, rows]) => {
    const [regime, controllerMode, sensorVariant] = key.split("\t");
    return {
      regime,
      controllerMode,
      sensorVariant,
      sensorTier: rows[0]?.sensorTier,
      n: rows.length,
      delayWarmupCount: rows.filter((row) => row.delayWarmup === true).length,
      meanIdealMagnitude: mean(rows.map((row) => row.idealMagnitude)),
      meanEstimatedMagnitude: mean(rows.map((row) => row.estimatedMagnitude)),
      meanMagnitudeAbsError: mean(rows.map((row) => row.magnitudeAbsError)),
      meanMagnitudeRelError: mean(rows.map((row) => row.magnitudeRelError)),
      rmsMagnitudeAbsError: Math.sqrt(mean(rows.map((row) => row.magnitudeAbsError ** 2)) ?? 0),
      maxMagnitudeAbsError: Math.max(...rows.map((row) => row.magnitudeAbsError).filter(Number.isFinite)),
      meanComponentRmse: mean(rows.map((row) => row.componentRmse)),
      noiseStd: rows[0]?.noiseStd,
      delaySteps: rows[0]?.delaySteps,
      probeDelta: rows[0]?.probeDelta,
    };
  }).sort((a, b) => (
    a.regime.localeCompare(b.regime)
    || a.controllerMode.localeCompare(b.controllerMode)
    || a.sensorVariant.localeCompare(b.sensorVariant)
  ));
}

function makeTrialConfig(args, seed, regime, mode) {
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
    sensorAuditVariants: args.sensorVariants,
    sensorAuditEvery: args.sensorAuditEvery,
    sensorNoiseStd: args.sensorNoiseStd,
    sensorDelaySteps: args.sensorDelaySteps,
    microManeuverContaminationStd: args.microManeuverContaminationStd,
    initialParticle: seededInitialParticle(seed, regime),
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(repoRoot, args.out);
  const trialDir = path.join(outDir, "trials");
  await mkdir(trialDir, { recursive: true });

  const startedAt = new Date().toISOString();
  const manifest = {
    schema: `sundog.threebody.${args.phase}.v1`,
    startedAt,
    purpose: `${args.phase.toUpperCase()} deterministic harness run; not a benchmark claim unless separately analyzed.`,
    harness: {
      script: "scripts/threebody-harness.mjs",
      core: "public/js/threebody-core.mjs",
    },
    modeDefinitions: MODE_DEFINITIONS,
    primaryMetrics: PRIMARY_METRICS,
    eventDiagnosticMetrics: EVENT_DIAGNOSTIC_METRICS,
    warningScoreDefinitions: {
      tidal_magnitude: "Frobenius norm of the local tidal tensor estimate.",
      local_acceleration: "Magnitude of local gravitational acceleration at the test particle.",
    },
    sensorVariantDefinitions: {
      simulated_local_probe: "Reference tier: exact virtual local probe samples from the simulator field model.",
      accelerometer_array_noisy: "Sensor tier: short-baseline acceleration samples with deterministic Gaussian noise.",
      delayed_local_probe: "Timing tier: exact virtual local probe estimate delayed by configured steps.",
      micro_maneuver_noisy: "Proxy tier: larger-baseline probe with delay and extra maneuver-contamination noise.",
    },
    args,
    trials: [],
  };

  const analysisTrials = [];

  for (const regime of args.regimes) {
    for (const mode of args.modes) {
      for (let offset = 0; offset < args.seeds; offset += 1) {
        const seed = args.seedStart + offset;
        const config = makeTrialConfig(args, seed, regime, mode);
        const id = trialId({ regime, mode, seed });
        const trial = runTrial(config);
        const relativePath = `trials/${id}.jsonl`;
        const logPath = path.join(outDir, relativePath);
        const lines = trial.records.map((record) => JSON.stringify(record)).join("\n");
        await writeFile(logPath, `${lines}\n`, "utf8");
        analysisTrials.push({
          seed,
          regime,
          controllerMode: mode,
          config,
          log: relativePath,
          summary: trial.summary,
          eventHistory: trial.eventHistory,
          sensorAudit: trial.sensorAudit,
        });
        manifest.trials.push({
          id,
          seed,
          regime,
          controllerMode: mode,
          initialParticle: config.initialParticle,
          config: {
            massRatio: config.massRatio,
            masses: config.masses,
            separation: config.separation,
            dt: config.dt,
            duration: config.duration,
            logEvery: config.logEvery,
            thrustLimit: config.thrustLimit,
            targetTidal: config.targetTidal,
            tidalSpikeThreshold: config.tidalSpikeThreshold,
            localAccelerationWarningThreshold: config.localAccelerationWarningThreshold,
            eventWarningHorizon: config.eventWarningHorizon,
            sensorAuditVariants: config.sensorAuditVariants,
            sensorAuditEvery: config.sensorAuditEvery,
            sensorNoiseStd: config.sensorNoiseStd,
            sensorDelaySteps: config.sensorDelaySteps,
            microManeuverContaminationStd: config.microManeuverContaminationStd,
          },
          log: relativePath,
          summary: trial.summary,
        });
      }
    }
  }

  manifest.completedAt = new Date().toISOString();
  const summaryRows = makeSummaryRows(manifest.trials);
  const pairedRows = makePairedRows(manifest.trials);
  const comparisonRows = makeComparisonRows(pairedRows);
  const eventMetricRows = makeEventMetricRows(manifest.trials);
  const scoreSpecs = warningScoreSpecs(args);
  const thresholdSweepRows = makeThresholdSweepRows(analysisTrials, scoreSpecs);
  const thresholdSummaryRows = makeThresholdSummaryRows(thresholdSweepRows);
  const calibrationRows = makeCalibrationRows(analysisTrials, scoreSpecs, args.calibrationBins);
  const sensorModelSampleRows = makeSensorModelSampleRows(analysisTrials);
  const sensorModelSummaryRows = makeSensorModelSummaryRows(sensorModelSampleRows);
  await writeFile(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  await writeFile(path.join(outDir, "summary.csv"), rowsToCsv(summaryRows), "utf8");
  await writeFile(path.join(outDir, "paired.csv"), rowsToCsv(pairedRows), "utf8");
  await writeFile(path.join(outDir, "comparison.csv"), rowsToCsv(comparisonRows), "utf8");
  await writeFile(path.join(outDir, "event-metrics.csv"), rowsToCsv(eventMetricRows), "utf8");
  await writeFile(path.join(outDir, "threshold-sweep.csv"), rowsToCsv(thresholdSweepRows), "utf8");
  await writeFile(path.join(outDir, "threshold-summary.csv"), rowsToCsv(thresholdSummaryRows), "utf8");
  await writeFile(path.join(outDir, "calibration.csv"), rowsToCsv(calibrationRows), "utf8");
  if (sensorModelSampleRows.length > 0) {
    await writeFile(path.join(outDir, "sensor-model-samples.csv"), rowsToCsv(sensorModelSampleRows), "utf8");
    await writeFile(path.join(outDir, "sensor-model-summary.csv"), rowsToCsv(sensorModelSummaryRows), "utf8");
  }

  const outcomeCounts = manifest.trials.reduce((counts, trial) => {
    counts[trial.summary.terminalOutcome] = (counts[trial.summary.terminalOutcome] ?? 0) + 1;
    return counts;
  }, {});

  console.log(`[threebody] wrote ${manifest.trials.length} trials to ${path.relative(repoRoot, outDir)}`);
  console.log(`[threebody] wrote summary.csv with ${summaryRows.length} condition rows`);
  console.log(`[threebody] wrote paired.csv, comparison.csv, event-metrics.csv, threshold-sweep.csv, threshold-summary.csv, and calibration.csv`);
  if (sensorModelSampleRows.length > 0) {
    console.log(`[threebody] wrote sensor-model-samples.csv and sensor-model-summary.csv`);
  }
  console.log(`[threebody] outcomes ${JSON.stringify(outcomeCounts)}`);
}

main().catch((error) => {
  console.error(`[threebody] ${error.message}`);
  process.exitCode = 1;
});
