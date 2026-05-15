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
    timesteps: [0.01],
    logEvery: 40,
    massRatios: [1],
    targetTidal: 2,
    tidalSpikeThreshold: 50,
    localAccelerationWarningThreshold: 10,
    eventWarningHorizon: 1,
    sensorAuditEvery: 80,
    radiusScales: [0.95, 1, 1.05],
    velocityScales: [0.9, 1, 1.1],
    thrustLimits: [0.4, 0.6],
    sensorNoiseSweep: [0, 0.01, 0.03],
    trackGuardMode: "constant",
    trackGuardQuantiles: [0.75],
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
    else if (flag === "--dt") args.timesteps = [Number.parseFloat(value)];
    else if (flag === "--timesteps") args.timesteps = parseNumberList(value);
    else if (flag === "--log-every") args.logEvery = Number.parseInt(value, 10);
    else if (flag === "--mass-ratio") args.massRatios = [Number.parseFloat(value)];
    else if (flag === "--mass-ratios") args.massRatios = parseNumberList(value);
    else if (flag === "--target-tidal") args.targetTidal = Number.parseFloat(value);
    else if (flag === "--tidal-spike-threshold") args.tidalSpikeThreshold = Number.parseFloat(value);
    else if (flag === "--local-acceleration-warning-threshold") args.localAccelerationWarningThreshold = Number.parseFloat(value);
    else if (flag === "--event-warning-horizon") args.eventWarningHorizon = Number.parseFloat(value);
    else if (flag === "--sensor-audit-every") args.sensorAuditEvery = Number.parseInt(value, 10);
    else if (flag === "--radius-scales") args.radiusScales = parseNumberList(value);
    else if (flag === "--velocity-scales") args.velocityScales = parseNumberList(value);
    else if (flag === "--thrust-limits") args.thrustLimits = parseNumberList(value);
    else if (flag === "--sensor-noise-sweep") args.sensorNoiseSweep = parseNumberList(value);
    else if (flag === "--track-guard-mode") args.trackGuardMode = value;
    else if (flag === "--track-guard-quantile") args.trackGuardQuantiles = [Number.parseFloat(value)];
    else if (flag === "--track-guard-quantiles") args.trackGuardQuantiles = parseNumberList(value);
    else if (flag === "--track-guard-min-radius-sweep") args.trackGuardMinRadiusSweep = parseNumberList(value);
    else if (flag === "--track-guard-max-local-acceleration-sweep") args.trackGuardMaxLocalAccelerationSweep = parseNumberList(value);
    else if (flag === "--track-guard-max-tidal-magnitude-sweep") args.trackGuardMaxTidalMagnitudeSweep = parseNumberList(value);
    else if (flag === "--candidate-max-worsened-rate") args.candidateMaxWorsenedRate = Number.parseFloat(value);
    else if (flag === "--candidate-min-survival-delta") args.candidateMinSurvivalDelta = Number.parseFloat(value);
    else if (flag === "--track-action-coupling") args.trackActionCoupling = !["0", "false", "no"].includes(String(value).toLowerCase());
    else throw new Error(`Unknown flag: ${flag}`);
  }

  if (!Number.isInteger(args.seeds) || args.seeds < 1) throw new Error("--seeds must be a positive integer");
  if (!Number.isFinite(args.duration) || args.duration <= 0) throw new Error("--duration must be positive");
  if (!Number.isInteger(args.logEvery) || args.logEvery < 1) throw new Error("--log-every must be a positive integer");
  if (!Number.isInteger(args.sensorAuditEvery) || args.sensorAuditEvery < 1) {
    throw new Error("--sensor-audit-every must be a positive integer");
  }
  for (const [name, values] of [
    ["--mass-ratios", args.massRatios],
    ["--timesteps", args.timesteps],
    ["--radius-scales", args.radiusScales],
    ["--velocity-scales", args.velocityScales],
    ["--thrust-limits", args.thrustLimits],
    ["--sensor-noise-sweep", args.sensorNoiseSweep],
    ["--track-guard-quantiles", args.trackGuardQuantiles],
    ["--track-guard-min-radius-sweep", args.trackGuardMinRadiusSweep],
    ["--track-guard-max-local-acceleration-sweep", args.trackGuardMaxLocalAccelerationSweep],
    ["--track-guard-max-tidal-magnitude-sweep", args.trackGuardMaxTidalMagnitudeSweep],
  ]) {
    if (!Array.isArray(values) || values.length === 0 || values.some((value) => !Number.isFinite(value) || value < 0)) {
      throw new Error(`${name} must contain non-negative finite numbers`);
    }
  }
  if (args.massRatios.some((value) => value <= 0)) throw new Error("--mass-ratios values must be positive");
  if (args.timesteps.some((value) => value <= 0)) throw new Error("--timesteps values must be positive");
  if (!args.modes.includes("off")) {
    throw new Error("--modes must include off for matched passive baselines");
  }
  if (!["constant", "hazard_quantile"].includes(args.trackGuardMode)) {
    throw new Error("--track-guard-mode must be constant or hazard_quantile");
  }
  if (args.trackGuardQuantiles.some((value) => value < 0 || value > 1)) {
    throw new Error("--track-guard-quantile(s) must be between 0 and 1");
  }

  args.radiusScales = [...new Set(args.radiusScales)].sort((a, b) => a - b);
  args.velocityScales = [...new Set(args.velocityScales)].sort((a, b) => a - b);
  args.thrustLimits = [...new Set(args.thrustLimits)].sort((a, b) => a - b);
  args.massRatios = [...new Set(args.massRatios)].sort((a, b) => a - b);
  args.timesteps = [...new Set(args.timesteps)].sort((a, b) => a - b);
  args.sensorNoiseSweep = [...new Set(args.sensorNoiseSweep)].sort((a, b) => a - b);
  args.trackGuardQuantiles = [...new Set(args.trackGuardQuantiles)].sort((a, b) => a - b);
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

function quantile(values, q) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (sorted.length === 0) return null;
  const index = (sorted.length - 1) * q;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
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

function summarizeSensorAudit(sensorAudit) {
  if (!Array.isArray(sensorAudit) || sensorAudit.length === 0) {
    return {
      sensorSampleCount: 0,
      sensorDelayWarmupCount: 0,
      meanSensorMagnitudeRelError: null,
      meanSensorMagnitudeAbsError: null,
      meanSensorComponentRmse: null,
      maxSensorMagnitudeAbsError: null,
    };
  }

  return {
    sensorSampleCount: sensorAudit.length,
    sensorDelayWarmupCount: sensorAudit.filter((sample) => sample.delayWarmup === true).length,
    meanSensorMagnitudeRelError: mean(sensorAudit.map((sample) => sample.magnitudeRelError)),
    meanSensorMagnitudeAbsError: mean(sensorAudit.map((sample) => sample.magnitudeAbsError)),
    meanSensorComponentRmse: mean(sensorAudit.map((sample) => sample.componentRmse)),
    maxSensorMagnitudeAbsError: Math.max(...sensorAudit.map((sample) => sample.magnitudeAbsError).filter(Number.isFinite)),
  };
}

function outcomeEffect(row) {
  if (!Number.isFinite(row.outcomeDeltaVsPassive)) return "unpaired";
  if (row.outcomeDeltaVsPassive > 0) return "helped";
  if (row.outcomeDeltaVsPassive < 0) return "hurt";
  if (Number.isFinite(row.simulatedTimeDeltaVsPassive)) {
    if (row.simulatedTimeDeltaVsPassive > 1e-9) return "time_helped";
    if (row.simulatedTimeDeltaVsPassive < -1e-9) return "time_hurt";
  }
  return "tied";
}

function failureMechanism(row) {
  const effect = outcomeEffect(row);
  if (effect !== "hurt" && effect !== "time_hurt") return "none";

  const passiveSurvived = row.passiveOutcome === "bounded";
  const controllerFailed = row.terminalOutcome !== "bounded";
  const passiveLastedLonger = Number.isFinite(row.simulatedTimeDeltaVsPassive)
    && row.simulatedTimeDeltaVsPassive < -0.5;
  const deltaVSpike = Number.isFinite(row.totalDeltaV)
    && Number.isFinite(row.passiveTotalDeltaV)
    && row.totalDeltaV > row.passiveTotalDeltaV + 0.5;
  const saturationSpike = Number.isFinite(row.saturationCount)
    && Number.isFinite(row.passiveSaturationCount)
    && row.saturationCount > row.passiveSaturationCount + 10;
  const noisyEstimate = Number.isFinite(row.meanSensorMagnitudeRelError)
    && row.meanSensorMagnitudeRelError > 1;

  if ((passiveSurvived && controllerFailed) || passiveLastedLonger) {
    return "controller_destabilized_or_shortened_passive";
  }
  if (deltaVSpike || saturationSpike) {
    return "control_effort_or_saturation";
  }
  if (noisyEstimate) {
    return "sensor_noise_floor";
  }
  return "unclassified_harm";
}

function scaleInitialParticle(particle, radiusScale, velocityScale) {
  return {
    x: particle.x * radiusScale,
    y: particle.y * radiusScale,
    vx: particle.vx * velocityScale,
    vy: particle.vy * velocityScale,
  };
}

function calibrationKey(envelopeCase, regime) {
  return [
    caseId(envelopeCase),
    regime,
  ].join("\t");
}

function nonHazardPassiveSamples(passiveTrials) {
  return passiveTrials.flatMap((trial) => {
    const firstHazard = trial.eventHistory.find((entry) => entry.events.escape || entry.events.closeApproach);
    const hazardTime = firstHazard?.time ?? null;
    return trial.eventHistory.filter((entry) => hazardTime === null || entry.time < hazardTime);
  });
}

function deriveGuardThresholds(passiveTrials, args, envelopeCase) {
  const samples = nonHazardPassiveSamples(passiveTrials);
  const guardQuantile = envelopeCase.trackGuardQuantile;
  const tidalThreshold = quantile(samples.map((sample) => sample.tidalMagnitude), guardQuantile);
  const localAccelerationThreshold = quantile(
    samples.map((sample) => sample.localAccelerationMagnitude),
    guardQuantile,
  );
  const radiusThreshold = quantile(
    samples.map((sample) => sample.events.testParticleRadius),
    1 - guardQuantile,
  );

  return {
    trackGuardMode: args.trackGuardMode,
    trackGuardQuantile: guardQuantile,
    trackGuardMinRadius: radiusThreshold ?? 0,
    trackGuardMaxLocalAcceleration: localAccelerationThreshold ?? Infinity,
    trackGuardMaxTidalMagnitude: tidalThreshold ?? Infinity,
    guardCalibrationSampleCount: samples.length,
  };
}

function envelopeCases(args) {
  const cases = [];
  for (const massRatio of args.massRatios) {
    for (const timestep of args.timesteps) {
      for (const radiusScale of args.radiusScales) {
        for (const velocityScale of args.velocityScales) {
          for (const thrustLimit of args.thrustLimits) {
            for (const sensorNoiseStd of args.sensorNoiseSweep) {
              for (const trackGuardQuantile of args.trackGuardQuantiles) {
                for (const trackGuardMinRadius of args.trackGuardMinRadiusSweep) {
                  for (const trackGuardMaxLocalAcceleration of args.trackGuardMaxLocalAccelerationSweep) {
                    for (const trackGuardMaxTidalMagnitude of args.trackGuardMaxTidalMagnitudeSweep) {
                      cases.push({
                        massRatio,
                        timestep,
                        radiusScale,
                        velocityScale,
                        thrustLimit,
                        sensorNoiseStd,
                        sensorDelaySteps: 0,
                        microManeuverContaminationStd: 0.08,
                        trackGuardQuantile,
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
      }
    }
  }
  return cases;
}

function caseId(envelopeCase) {
  return [
    `mu_${envelopeCase.massRatio}`,
    `dt_${envelopeCase.timestep}`,
    `r_${envelopeCase.radiusScale}`,
    `v_${envelopeCase.velocityScale}`,
    `thrust_${envelopeCase.thrustLimit}`,
    `noise_${envelopeCase.sensorNoiseStd}`,
    `gq_${envelopeCase.trackGuardQuantile}`,
    `gminr_${envelopeCase.trackGuardMinRadius}`,
    `gaccel_${envelopeCase.trackGuardMaxLocalAcceleration}`,
    `gtidal_${envelopeCase.trackGuardMaxTidalMagnitude}`,
  ].join("__").replaceAll(".", "p");
}

function trialId(envelopeCase, regime, mode, seed) {
  return `${caseId(envelopeCase)}__${regime}__${mode}__seed_${String(seed).padStart(3, "0")}`;
}

function makeTrialConfig(args, envelopeCase, seed, regime, mode, guardThresholds = null) {
  const baseParticle = seededInitialParticle(seed, regime);
  const thresholds = guardThresholds ?? {
    trackGuardMinRadius: envelopeCase.trackGuardMinRadius,
    trackGuardMaxLocalAcceleration: envelopeCase.trackGuardMaxLocalAcceleration,
    trackGuardMaxTidalMagnitude: envelopeCase.trackGuardMaxTidalMagnitude,
  };
  return normalizeConfig({
    seed,
    regime,
    controllerMode: mode,
    duration: args.duration,
    dt: envelopeCase.timestep,
    logEvery: args.logEvery,
    massRatio: envelopeCase.massRatio,
    thrustLimit: envelopeCase.thrustLimit,
    targetTidal: args.targetTidal,
    tidalSpikeThreshold: args.tidalSpikeThreshold,
    localAccelerationWarningThreshold: args.localAccelerationWarningThreshold,
    eventWarningHorizon: args.eventWarningHorizon,
    sensorAuditVariants: ["accelerometer_array_noisy"],
    sensorAuditEvery: args.sensorAuditEvery,
    sensorNoiseStd: envelopeCase.sensorNoiseStd,
    sensorDelaySteps: envelopeCase.sensorDelaySteps,
    microManeuverContaminationStd: envelopeCase.microManeuverContaminationStd,
    trackGuardMinRadius: thresholds.trackGuardMinRadius,
    trackGuardMaxLocalAcceleration: thresholds.trackGuardMaxLocalAcceleration,
    trackGuardMaxTidalMagnitude: thresholds.trackGuardMaxTidalMagnitude,
    trackActionCoupling: args.trackActionCoupling ?? false,
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
    const row = {
      caseId: trial.caseId,
      regime: trial.regime,
      seed: trial.seed,
      controllerMode: trial.controllerMode,
      massRatio: trial.massRatio,
      timestep: trial.timestep,
      radiusScale: trial.radiusScale,
      velocityScale: trial.velocityScale,
      thrustLimit: trial.thrustLimit,
      sensorNoiseStd: trial.sensorNoiseStd,
      trackGuardMode: trial.trackGuardMode,
      trackGuardQuantile: trial.trackGuardQuantile,
      trackGuardMinRadius: trial.trackGuardMinRadius,
      trackGuardMaxLocalAcceleration: trial.trackGuardMaxLocalAcceleration,
      trackGuardMaxTidalMagnitude: trial.trackGuardMaxTidalMagnitude,
      guardCalibrationSampleCount: trial.guardCalibrationSampleCount,
      terminalOutcome: trial.summary.terminalOutcome,
      passiveOutcome: passive?.summary.terminalOutcome,
      outcomeDeltaVsPassive: passive ? compareOutcome(trial.summary.terminalOutcome, passive.summary.terminalOutcome) : null,
      simulatedTime: trial.summary.simulatedTime,
      passiveSimulatedTime: passive?.summary.simulatedTime,
      simulatedTimeDeltaVsPassive: passive ? trial.summary.simulatedTime - passive.summary.simulatedTime : null,
      totalDeltaV: trial.summary.totalDeltaV,
      passiveTotalDeltaV: passive?.summary.totalDeltaV,
      deltaVDeltaVsPassive: passive ? trial.summary.totalDeltaV - passive.summary.totalDeltaV : null,
      minPrimaryDistance: trial.summary.minPrimaryDistance,
      passiveMinPrimaryDistance: passive?.summary.minPrimaryDistance,
      minPrimaryDistanceDeltaVsPassive: passive ? trial.summary.minPrimaryDistance - passive.summary.minPrimaryDistance : null,
      saturationCount: trial.summary.saturationCount,
      passiveSaturationCount: passive?.summary.saturationCount,
      saturationDeltaVsPassive: passive ? trial.summary.saturationCount - passive.summary.saturationCount : null,
      targetBandLossCount: trial.summary.targetBandLossCount,
      passiveTargetBandLossCount: passive?.summary.targetBandLossCount,
      targetBandLossDeltaVsPassive: passive ? trial.summary.targetBandLossCount - passive.summary.targetBandLossCount : null,
      maxRadius: trial.summary.maxRadius,
      passiveMaxRadius: passive?.summary.maxRadius,
      tidalMagnitudeAuroc: trial.summary.tidalMagnitudeAuroc,
      localAccelerationMagnitudeAuroc: trial.summary.localAccelerationMagnitudeAuroc,
      ...(trial.summary.actionCouplingEligibleSteps !== undefined
        ? {
          actionCouplingEligibleSteps: trial.summary.actionCouplingEligibleSteps,
          actionCouplingAgreementRate: trial.summary.actionCouplingAgreementRate,
          actionCouplingSignedEffect: trial.summary.actionCouplingSignedEffect,
          passiveTidalMagnitudeAuroc: passive ? passive.summary.tidalMagnitudeAuroc : null,
          passiveTidalWarningLeadTime: passive ? passive.summary.tidalWarningLeadTime : null,
        }
        : {}),
      ...summarizeSensorAudit(trial.sensorAudit),
      log: trial.log,
    };
    row.outcomeEffect = outcomeEffect(row);
    row.failureMechanism = failureMechanism(row);
    return row;
  }).sort((a, b) => (
    a.regime.localeCompare(b.regime)
    || a.massRatio - b.massRatio
    || a.timestep - b.timestep
    || a.radiusScale - b.radiusScale
    || a.velocityScale - b.velocityScale
    || a.thrustLimit - b.thrustLimit
    || a.sensorNoiseStd - b.sensorNoiseStd
    || (a.trackGuardQuantile ?? -1) - (b.trackGuardQuantile ?? -1)
    || a.trackGuardMaxLocalAcceleration - b.trackGuardMaxLocalAcceleration
    || a.seed - b.seed
    || a.controllerMode.localeCompare(b.controllerMode)
  ));
}

function makeTrialOutcomeRows(pairedRows) {
  return pairedRows.map((row) => ({
    seed: row.seed,
    regime: row.regime,
    mass_ratio: row.massRatio,
    timestep: row.timestep,
    radius_scale: row.radiusScale,
    velocity_scale: row.velocityScale,
    thrust_limit: row.thrustLimit,
    sensor_noise_std: row.sensorNoiseStd,
    guard_min_radius: row.trackGuardMinRadius,
    guard_max_local_acceleration: row.trackGuardMaxLocalAcceleration,
    guard_max_tidal_magnitude: row.trackGuardMaxTidalMagnitude,
    guard_mode: row.trackGuardMode,
    guard_quantile: row.trackGuardQuantile,
    guard_calibration_sample_count: row.guardCalibrationSampleCount,
    controller_mode: row.controllerMode,
    terminal_outcome: row.terminalOutcome,
    passive_terminal_outcome: row.passiveOutcome,
    outcome_delta_vs_passive: row.outcomeDeltaVsPassive,
    outcome_effect: row.outcomeEffect,
    failure_mechanism: row.failureMechanism,
    simulated_time: row.simulatedTime,
    passive_simulated_time: row.passiveSimulatedTime,
    simulated_time_delta_vs_passive: row.simulatedTimeDeltaVsPassive,
    total_delta_v: row.totalDeltaV,
    passive_total_delta_v: row.passiveTotalDeltaV,
    delta_v_delta_vs_passive: row.deltaVDeltaVsPassive,
    min_primary_distance: row.minPrimaryDistance,
    passive_min_primary_distance: row.passiveMinPrimaryDistance,
    min_primary_distance_delta_vs_passive: row.minPrimaryDistanceDeltaVsPassive,
    saturation_count: row.saturationCount,
    passive_saturation_count: row.passiveSaturationCount,
    saturation_delta_vs_passive: row.saturationDeltaVsPassive,
    target_band_loss_count: row.targetBandLossCount,
    passive_target_band_loss_count: row.passiveTargetBandLossCount,
    target_band_loss_delta_vs_passive: row.targetBandLossDeltaVsPassive,
    max_radius: row.maxRadius,
    passive_max_radius: row.passiveMaxRadius,
    tidal_magnitude_auroc: row.tidalMagnitudeAuroc,
    local_acceleration_magnitude_auroc: row.localAccelerationMagnitudeAuroc,
    sensor_sample_count: row.sensorSampleCount,
    sensor_delay_warmup_count: row.sensorDelayWarmupCount,
    mean_sensor_magnitude_rel_error: row.meanSensorMagnitudeRelError,
    mean_sensor_magnitude_abs_error: row.meanSensorMagnitudeAbsError,
    mean_sensor_component_rmse: row.meanSensorComponentRmse,
    max_sensor_magnitude_abs_error: row.maxSensorMagnitudeAbsError,
    ...(row.actionCouplingEligibleSteps !== undefined
      ? {
        action_coupling_eligible_steps: row.actionCouplingEligibleSteps,
        action_coupling_agreement_rate: row.actionCouplingAgreementRate,
        action_coupling_signed_effect: row.actionCouplingSignedEffect,
        passive_tidal_magnitude_auroc: row.passiveTidalMagnitudeAuroc,
        passive_tidal_warning_lead_time: row.passiveTidalWarningLeadTime,
      }
      : {}),
    trial_log: row.log,
    case_id: row.caseId,
  }));
}

function mostCommon(values) {
  const counts = new Map();
  for (const value of values.filter(Boolean)) {
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0]?.[0] ?? null;
}

function summarizeRows(rows, args, includeRegime = true) {
  const groups = new Map();
  for (const row of rows) {
    const key = [
      ...(includeRegime ? [row.regime] : []),
      row.controllerMode,
      row.massRatio,
      row.timestep,
      row.radiusScale,
      row.velocityScale,
      row.thrustLimit,
      row.sensorNoiseStd,
      row.trackGuardMode,
      row.trackGuardQuantile,
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
    const harmfulRows = group.filter((row) => row.failureMechanism && row.failureMechanism !== "none");
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
      massRatio: Number.parseFloat(values[offset + 1]),
      timestep: Number.parseFloat(values[offset + 2]),
      radiusScale: Number.parseFloat(values[offset + 3]),
      velocityScale: Number.parseFloat(values[offset + 4]),
      thrustLimit: Number.parseFloat(values[offset + 5]),
      sensorNoiseStd: Number.parseFloat(values[offset + 6]),
      trackGuardMode: values[offset + 7],
      trackGuardQuantile: Number.parseFloat(values[offset + 8]),
      trackGuardMinRadius: Number.parseFloat(values[offset + 9]),
      trackGuardMaxLocalAcceleration: Number.parseFloat(values[offset + 10]),
      trackGuardMaxTidalMagnitude: Number.parseFloat(values[offset + 11]),
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
      meanPassiveDeltaV: roundMetric(mean(group.map((row) => row.passiveTotalDeltaV))),
      meanDeltaVDeltaVsPassive: roundMetric(mean(group.map((row) => row.deltaVDeltaVsPassive))),
      meanMinPrimaryDistance: roundMetric(mean(group.map((row) => row.minPrimaryDistance))),
      meanPassiveSimulatedTime: roundMetric(mean(group.map((row) => row.passiveSimulatedTime))),
      meanPassiveMinPrimaryDistance: roundMetric(mean(group.map((row) => row.passiveMinPrimaryDistance))),
      meanSaturationCount: roundMetric(mean(group.map((row) => row.saturationCount))),
      meanSaturationDeltaVsPassive: roundMetric(mean(group.map((row) => row.saturationDeltaVsPassive))),
      meanSensorMagnitudeRelError: roundMetric(mean(group.map((row) => row.meanSensorMagnitudeRelError))),
      meanSensorMagnitudeAbsError: roundMetric(mean(group.map((row) => row.meanSensorMagnitudeAbsError))),
      meanGuardCalibrationSampleCount: roundMetric(mean(group.map((row) => row.guardCalibrationSampleCount))),
      meanTidalMagnitudeAuroc: roundMetric(mean(group.map((row) => row.tidalMagnitudeAuroc))),
      meanLocalAccelerationMagnitudeAuroc: roundMetric(mean(group.map((row) => row.localAccelerationMagnitudeAuroc))),
      ...(group.some((row) => row.actionCouplingEligibleSteps !== undefined)
        ? {
          meanActionCouplingAgreementRate: roundMetric(mean(group.map((row) => row.actionCouplingAgreementRate))),
          meanActionCouplingSignedEffect: roundMetric(mean(group.map((row) => row.actionCouplingSignedEffect))),
          meanPassiveTidalMagnitudeAuroc: roundMetric(mean(group.map((row) => row.passiveTidalMagnitudeAuroc))),
          totalActionCouplingEligibleSteps: group.reduce((sum, row) => sum + (row.actionCouplingEligibleSteps ?? 0), 0),
        }
        : {}),
      dominantFailureMechanism: mostCommon(harmfulRows.map((row) => row.failureMechanism)),
      candidateEnvelope,
      regionClass,
    };
  }).sort((a, b) => (
    Number(b.candidateEnvelope) - Number(a.candidateEnvelope)
    || (b.survivalDeltaVsPassive ?? -Infinity) - (a.survivalDeltaVsPassive ?? -Infinity)
    || (a.worsenedRate ?? Infinity) - (b.worsenedRate ?? Infinity)
    || a.regime.localeCompare(b.regime)
    || a.massRatio - b.massRatio
    || a.timestep - b.timestep
    || a.radiusScale - b.radiusScale
    || a.velocityScale - b.velocityScale
    || a.thrustLimit - b.thrustLimit
    || a.sensorNoiseStd - b.sensorNoiseStd
  ));
}

function makeBestByCellRows(envelopeRows) {
  const groups = new Map();
  for (const row of envelopeRows) {
    const key = `${row.regime}\t${row.massRatio}\t${row.timestep}\t${row.radiusScale}\t${row.velocityScale}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }

  return Array.from(groups.entries()).map(([key, rows]) => {
    const [regime, massRatioText, timestepText, radiusScaleText, velocityScaleText] = key.split("\t");
    const best = [...rows].sort((a, b) => (
      Number(b.candidateEnvelope) - Number(a.candidateEnvelope)
      || (b.survivalDeltaVsPassive ?? -Infinity) - (a.survivalDeltaVsPassive ?? -Infinity)
      || (a.worsenedRate ?? Infinity) - (b.worsenedRate ?? Infinity)
      || (b.survivalRate ?? -Infinity) - (a.survivalRate ?? -Infinity)
    ))[0];
    return {
      regime,
      massRatio: Number.parseFloat(massRatioText),
      timestep: Number.parseFloat(timestepText),
      radiusScale: Number.parseFloat(radiusScaleText),
      velocityScale: Number.parseFloat(velocityScaleText),
      bestControllerMode: best.controllerMode,
      bestRegionClass: best.regionClass,
      bestCandidateEnvelope: best.candidateEnvelope,
      bestThrustLimit: best.thrustLimit,
      bestSensorNoiseStd: best.sensorNoiseStd,
      bestTrackGuardMaxLocalAcceleration: best.trackGuardMaxLocalAcceleration,
      bestTrackGuardMode: best.trackGuardMode,
      bestTrackGuardQuantile: best.trackGuardQuantile,
      bestSurvivalRate: best.survivalRate,
      bestPassiveSurvivalRate: best.passiveSurvivalRate,
      bestSurvivalDeltaVsPassive: best.survivalDeltaVsPassive,
      bestWorsenedRate: best.worsenedRate,
      bestMeanDeltaV: best.meanDeltaV,
      bestDominantFailureMechanism: best.dominantFailureMechanism,
    };
  }).sort((a, b) => (
    a.regime.localeCompare(b.regime)
    || a.massRatio - b.massRatio
    || a.timestep - b.timestep
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
    const key = `${row.regime}\t${row.massRatio}\t${row.timestep}\t${row.radiusScale}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }

  return Array.from(groups.entries()).map(([key, rows]) => {
    const [regime, massRatioText, timestepText, radiusScaleText] = key.split("\t");
    const byVelocity = new Map(rows.map((row) => [row.velocityScale, row]));
    return {
      regime,
      massRatio: Number.parseFloat(massRatioText),
      timestep: Number.parseFloat(timestepText),
      radiusScale: Number.parseFloat(radiusScaleText),
      ...Object.fromEntries(velocities.map((velocity) => [
        velocityColumn(velocity),
        byVelocity.get(velocity)?.[valueKey] ?? null,
      ])),
    };
  }).sort((a, b) => (
    a.regime.localeCompare(b.regime)
    || a.massRatio - b.massRatio
    || a.timestep - b.timestep
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
  const passiveCalibration = new Map();
  const manifest = {
    schema: `sundog.threebody.${args.phase}.v1`,
    startedAt: new Date().toISOString(),
    purpose: `${args.phase} operating-envelope map for guarded sensor-tier control across initial-condition, thrust, noise, and guard settings.`,
    args,
    cases,
    trials: [],
  };

  for (const envelopeCase of cases) {
    for (const regime of args.regimes) {
      for (const mode of ["off", ...args.modes.filter((candidate) => candidate !== "off")]) {
        const thresholdKey = calibrationKey(envelopeCase, regime);
        let guardThresholds = null;
        if (mode !== "off" && args.trackGuardMode === "hazard_quantile") {
          guardThresholds = passiveCalibration.get(thresholdKey);
          if (!guardThresholds) {
            throw new Error(`Missing passive guard calibration for ${thresholdKey}`);
          }
        }
        for (let offset = 0; offset < args.seeds; offset += 1) {
          const seed = args.seedStart + offset;
          const config = makeTrialConfig(args, envelopeCase, seed, regime, mode, guardThresholds);
          const id = trialId(envelopeCase, regime, mode, seed);
          const trial = runTrial(config);
          const relativePath = `trials/${id}.jsonl`;
          const logPath = path.join(outDir, relativePath);
          const lines = trial.records.map((record) => JSON.stringify(record)).join("\n");
          await writeFile(logPath, `${lines}\n`, "utf8");

          const row = {
            caseId: caseId(envelopeCase),
            ...envelopeCase,
            trackGuardMode: mode === "off" ? "passive" : args.trackGuardMode,
            trackGuardQuantile: mode === "off" ? null : guardThresholds?.trackGuardQuantile ?? null,
            trackGuardMinRadius: config.trackGuardMinRadius,
            trackGuardMaxLocalAcceleration: config.trackGuardMaxLocalAcceleration,
            trackGuardMaxTidalMagnitude: config.trackGuardMaxTidalMagnitude,
            guardCalibrationSampleCount: mode === "off" ? null : guardThresholds?.guardCalibrationSampleCount ?? null,
            seed,
            regime,
            controllerMode: mode,
            log: relativePath,
            summary: trial.summary,
            sensorAudit: trial.sensorAudit,
            eventHistory: trial.eventHistory,
          };
          trials.push(row);
          manifest.trials.push({
            id,
            caseId: row.caseId,
            ...envelopeCase,
            trackGuardMode: row.trackGuardMode,
            trackGuardQuantile: row.trackGuardQuantile,
            trackGuardMinRadius: row.trackGuardMinRadius,
            trackGuardMaxLocalAcceleration: row.trackGuardMaxLocalAcceleration,
            trackGuardMaxTidalMagnitude: row.trackGuardMaxTidalMagnitude,
            guardCalibrationSampleCount: row.guardCalibrationSampleCount,
            seed,
            regime,
            controllerMode: mode,
            initialParticle: config.initialParticle,
            log: relativePath,
            summary: trial.summary,
            sensorAuditSampleCount: trial.sensorAudit.length,
          });
        }
        if (mode === "off" && args.trackGuardMode === "hazard_quantile") {
          const passiveRows = trials.filter((trial) => (
            trial.controllerMode === "off"
            && trial.caseId === caseId(envelopeCase)
            && trial.regime === regime
          ));
          passiveCalibration.set(thresholdKey, deriveGuardThresholds(passiveRows, args, envelopeCase));
        }
      }
    }
  }

  manifest.completedAt = new Date().toISOString();
  const pairedRows = makePairedRows(trials);
  const trialOutcomeRows = makeTrialOutcomeRows(pairedRows);
  const envelopeRows = summarizeRows(pairedRows, args, true);
  const aggregateRows = summarizeRows(pairedRows, args, false);
  const bestByCellRows = makeBestByCellRows(envelopeRows);
  const cellClassMapRows = makeCellMatrixRows(bestByCellRows, "bestRegionClass");
  const cellDeltaMapRows = makeCellMatrixRows(bestByCellRows, "bestSurvivalDeltaVsPassive");
  const candidateRows = envelopeRows.filter((row) => row.candidateEnvelope);

  let cellWarningQualityMapRows = null;
  if (args.trackActionCoupling) {
    const warningGroups = new Map();
    for (const row of envelopeRows) {
      const key = `${row.regime}\t${row.massRatio}\t${row.timestep}\t${row.radiusScale}\t${row.velocityScale}`;
      if (!warningGroups.has(key)) warningGroups.set(key, []);
      warningGroups.get(key).push(row);
    }
    const warningRows = Array.from(warningGroups.entries()).map(([key, rows]) => {
      const [regime, massRatioText, timestepText, radiusScaleText, velocityScaleText] = key.split("\t");
      return {
        regime,
        massRatio: Number.parseFloat(massRatioText),
        timestep: Number.parseFloat(timestepText),
        radiusScale: Number.parseFloat(radiusScaleText),
        velocityScale: Number.parseFloat(velocityScaleText),
        meanPassiveTidalMagnitudeAuroc: roundMetric(mean(rows.map((row) => row.meanPassiveTidalMagnitudeAuroc))),
      };
    });
    cellWarningQualityMapRows = makeCellMatrixRows(warningRows, "meanPassiveTidalMagnitudeAuroc");
  }

  await writeFile(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  await writeFile(path.join(outDir, "trial-outcomes.csv"), rowsToCsv(trialOutcomeRows), "utf8");
  await writeFile(path.join(outDir, "paired.csv"), rowsToCsv(pairedRows), "utf8");
  await writeFile(path.join(outDir, "envelope-map.csv"), rowsToCsv(envelopeRows), "utf8");
  await writeFile(path.join(outDir, "aggregate-envelope.csv"), rowsToCsv(aggregateRows), "utf8");
  await writeFile(path.join(outDir, "best-by-cell.csv"), rowsToCsv(bestByCellRows), "utf8");
  await writeFile(path.join(outDir, "cell-class-map.csv"), rowsToCsv(cellClassMapRows), "utf8");
  await writeFile(path.join(outDir, "cell-delta-map.csv"), rowsToCsv(cellDeltaMapRows), "utf8");
  if (cellWarningQualityMapRows) {
    await writeFile(path.join(outDir, "cell-warning-quality-map.csv"), rowsToCsv(cellWarningQualityMapRows), "utf8");
  }
  await writeFile(
    path.join(outDir, "candidate-envelope.csv"),
    rowsToCsv(candidateRows, envelopeRows.length > 0 ? Object.keys(envelopeRows[0]) : []),
    "utf8",
  );

  const outcomeCounts = manifest.trials.reduce((counts, trial) => {
    counts[trial.summary.terminalOutcome] = (counts[trial.summary.terminalOutcome] ?? 0) + 1;
    return counts;
  }, {});

  console.log(`[threebody] wrote ${manifest.trials.length} ${args.phase} trials to ${path.relative(repoRoot, outDir)}`);
  console.log(`[threebody] wrote trial-outcomes.csv, paired.csv, envelope-map.csv, aggregate-envelope.csv, best-by-cell.csv, cell-class-map.csv, cell-delta-map.csv, and candidate-envelope.csv`);
  console.log(`[threebody] candidate envelope rows ${candidateRows.length}/${envelopeRows.length}`);
  console.log(`[threebody] outcomes ${JSON.stringify(outcomeCounts)}`);
}

main().catch((error) => {
  console.error(`[threebody] ${error.message}`);
  process.exitCode = 1;
});
