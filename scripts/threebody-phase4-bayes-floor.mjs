import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  classifyEvents,
  computeAcceleration,
  computeSignatures,
  computeTidalTensor,
  initializeState,
  integrateStep,
  makeRng,
  normalizeConfig,
  observeGuardedAccelSignature,
  oracleCandidateThrusts,
  roundNumber,
  runTrial,
  seededInitialParticle,
  summarizeOutcome,
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
    phase: "phase4-bayes-floor-smoke",
    out: "results/proof/phase4/bayes-floor-smoke",
    seedStart: 0,
    seeds: 2,
    regimes: ["near_escape"],
    duration: 16,
    timesteps: [0.01],
    logEvery: 1,
    massRatios: [1],
    targetTidal: 2,
    tidalSpikeThreshold: 50,
    localAccelerationWarningThreshold: 10,
    eventWarningHorizon: 1,
    radiusScales: [1.075],
    velocityScales: [1.1],
    thrustLimits: [0.4],
    sensorNoiseSweep: [0],
    trackGuardMode: "hazard_quantile",
    trackGuardQuantiles: [0.75],
    trackGuardMinRadiusSweep: [1.15],
    trackGuardMaxLocalAccelerationSweep: [2.5],
    trackGuardMaxTidalMagnitudeSweep: [35],
    particleCount: 256,
    planningHorizonSteps: 16,
    resampleThreshold: 0.5,
    shapeFraction: 0.5,
    signatureAdvantageDtMultiplier: 1,
    candidateHoldSteps: 1,
    bootstrapSeed: 40604,
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
    else if (flag === "--particle-count") args.particleCount = Number.parseInt(value, 10);
    else if (flag === "--planning-horizon-steps") args.planningHorizonSteps = Number.parseInt(value, 10);
    else if (flag === "--resample-threshold") args.resampleThreshold = Number.parseFloat(value);
    else if (flag === "--shape-fraction") args.shapeFraction = Number.parseFloat(value);
    else if (flag === "--signature-advantage-dt-multiplier") args.signatureAdvantageDtMultiplier = Number.parseFloat(value);
    else if (flag === "--candidate-hold-steps") args.candidateHoldSteps = Number.parseInt(value, 10);
    else if (flag === "--bootstrap-seed") args.bootstrapSeed = Number.parseInt(value, 10);
    else throw new Error(`Unknown flag: ${flag}`);
  }

  if (!Number.isInteger(args.seeds) || args.seeds < 1) throw new Error("--seeds must be a positive integer");
  if (!Number.isFinite(args.duration) || args.duration <= 0) throw new Error("--duration must be positive");
  if (!Number.isInteger(args.logEvery) || args.logEvery < 1) throw new Error("--log-every must be a positive integer");
  if (!Number.isInteger(args.particleCount) || args.particleCount < 1) throw new Error("--particle-count must be positive");
  if (!Number.isInteger(args.planningHorizonSteps) || args.planningHorizonSteps < 1) {
    throw new Error("--planning-horizon-steps must be positive");
  }
  if (!Number.isFinite(args.resampleThreshold) || args.resampleThreshold <= 0 || args.resampleThreshold > 1) {
    throw new Error("--resample-threshold must be in (0, 1]");
  }
  if (!Number.isFinite(args.shapeFraction) || args.shapeFraction < 0 || args.shapeFraction >= 1) {
    throw new Error("--shape-fraction must be in [0, 1) so shaping stays strictly below one dt (floor-validity)");
  }
  if (!Number.isFinite(args.signatureAdvantageDtMultiplier) || args.signatureAdvantageDtMultiplier < 0) {
    throw new Error("--signature-advantage-dt-multiplier must be non-negative");
  }
  if (!Number.isInteger(args.candidateHoldSteps) || args.candidateHoldSteps < 1) {
    throw new Error("--candidate-hold-steps must be a positive integer (1 = prior single-step behaviour)");
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

function trialId(envelopeCase, regime, seed) {
  return `${caseId(envelopeCase)}__${regime}__bayes_floor_particle_mpc__seed_${String(seed).padStart(3, "0")}`;
}

function makeTrialConfig(args, envelopeCase, seed, regime, guardThresholds = null, controllerMode = "bayes_floor_particle_mpc") {
  const baseParticle = seededInitialParticle(seed, regime);
  const thresholds = guardThresholds ?? {
    trackGuardMinRadius: envelopeCase.trackGuardMinRadius,
    trackGuardMaxLocalAcceleration: envelopeCase.trackGuardMaxLocalAcceleration,
    trackGuardMaxTidalMagnitude: envelopeCase.trackGuardMaxTidalMagnitude,
  };
  return normalizeConfig({
    seed,
    regime,
    controllerMode,
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
    sensorNoiseStd: envelopeCase.sensorNoiseStd,
    sensorDelaySteps: envelopeCase.sensorDelaySteps,
    microManeuverContaminationStd: envelopeCase.microManeuverContaminationStd,
    trackGuardMinRadius: thresholds.trackGuardMinRadius,
    trackGuardMaxLocalAcceleration: thresholds.trackGuardMaxLocalAcceleration,
    trackGuardMaxTidalMagnitude: thresholds.trackGuardMaxTidalMagnitude,
    initialParticle: scaleInitialParticle(baseParticle, envelopeCase.radiusScale, envelopeCase.velocityScale),
  });
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

function guardThresholdsForCase(args, envelopeCase, regime) {
  if (args.trackGuardMode !== "hazard_quantile") return null;
  const passiveTrials = [];
  for (let offset = 0; offset < args.seeds; offset += 1) {
    const seed = args.seedStart + offset;
    const config = makeTrialConfig(args, envelopeCase, seed, regime, null, "off");
    passiveTrials.push(runTrial(config));
  }
  return deriveGuardThresholds(passiveTrials, args, envelopeCase);
}

function particleInitialState(args, envelopeCase, regime, trialSeed, particleIndex, config) {
  const particleSeed = (
    args.bootstrapSeed
    + trialSeed * 1000003
    + particleIndex * 9176
    + 101
  ) >>> 0;
  const priorParticle = seededInitialParticle(particleSeed, regime);
  return initializeState({
    ...config,
    seed: particleSeed,
    initialParticle: scaleInitialParticle(priorParticle, envelopeCase.radiusScale, envelopeCase.velocityScale),
  });
}

function initializeParticles(args, envelopeCase, regime, trialSeed, config) {
  const particles = [];
  const weight = 1 / args.particleCount;
  for (let i = 0; i < args.particleCount; i += 1) {
    particles.push({
      state: particleInitialState(args, envelopeCase, regime, trialSeed, i, config),
      weight,
      sensorState: {},
    });
  }
  return particles;
}

function effectiveSampleSize(particles) {
  const sumSquares = particles.reduce((sum, particle) => sum + particle.weight * particle.weight, 0);
  return sumSquares > 0 ? 1 / sumSquares : 0;
}

function normalizeWeights(particles) {
  const total = particles.reduce((sum, particle) => sum + particle.weight, 0);
  if (!Number.isFinite(total) || total <= 0) {
    const weight = 1 / particles.length;
    for (const particle of particles) particle.weight = weight;
    return;
  }
  for (const particle of particles) particle.weight /= total;
}

function cloneSensorState(sensorState) {
  return JSON.parse(JSON.stringify(sensorState ?? {}));
}

function resampleParticles(particles, rng) {
  const cumulative = [];
  let running = 0;
  for (const particle of particles) {
    running += particle.weight;
    cumulative.push(running);
  }
  const resampled = [];
  const uniformWeight = 1 / particles.length;
  for (let i = 0; i < particles.length; i += 1) {
    const draw = rng() * running;
    const index = cumulative.findIndex((value) => value >= draw);
    const selected = particles[index < 0 ? particles.length - 1 : index];
    resampled.push({
      state: selected.state.slice(),
      weight: uniformWeight,
      sensorState: cloneSensorState(selected.sensorState),
    });
  }
  return resampled;
}

function signatureDistanceSquared(observed, predicted, config) {
  const magScale = Math.max(1, Math.abs(observed.tidalMagnitude), Math.abs(predicted.tidalMagnitude));
  const observedGradMag = Math.sqrt(observed.gradX * observed.gradX + observed.gradY * observed.gradY);
  const predictedGradMag = Math.sqrt(predicted.gradX * predicted.gradX + predicted.gradY * predicted.gradY);
  const gradScale = Math.max(1, observedGradMag, predictedGradMag);
  const noiseScale = Math.max(config.sensorNoiseStd, 0.01);
  const magSigma = Math.max(1e-6, magScale * noiseScale);
  const gradSigma = Math.max(1e-6, gradScale * noiseScale);
  const dMag = (observed.tidalMagnitude - predicted.tidalMagnitude) / magSigma;
  const dX = (observed.gradX - predicted.gradX) / gradSigma;
  const dY = (observed.gradY - predicted.gradY) / gradSigma;
  const guardPenalty = observed.guard === predicted.guard ? 0 : 9;
  return dMag * dMag + dX * dX + dY * dY + guardPenalty;
}

function updateParticleBelief(particles, observed, config, args, rng) {
  const predictionConfig = { ...config, sensorNoiseStd: 0 };
  let maxLogLikelihood = -Infinity;
  const logLikelihoods = [];
  for (const particle of particles) {
    const predicted = observeGuardedAccelSignature(particle.state, predictionConfig, particle.sensorState);
    const distance = signatureDistanceSquared(observed, predicted, config);
    const logLikelihood = -0.5 * Math.min(distance, 120);
    logLikelihoods.push(logLikelihood);
    if (logLikelihood > maxLogLikelihood) maxLogLikelihood = logLikelihood;
  }
  for (let i = 0; i < particles.length; i += 1) {
    particles[i].weight *= Math.exp(logLikelihoods[i] - maxLogLikelihood);
  }
  normalizeWeights(particles);
  const essBeforeResample = effectiveSampleSize(particles);
  if (essBeforeResample < args.resampleThreshold * particles.length) {
    const resampled = resampleParticles(particles, rng);
    particles.splice(0, particles.length, ...resampled);
    return { effectiveSampleSize: essBeforeResample, resampled: true };
  }
  return { effectiveSampleSize: essBeforeResample, resampled: false };
}

function actionKey(thrust) {
  return `${roundNumber(thrust[0], 8)}:${roundNumber(thrust[1], 8)}`;
}

function guardedSignatureThrust(signature, config) {
  if (!signature.guard) return [0, 0];
  const gradMag = Math.sqrt(signature.gradX * signature.gradX + signature.gradY * signature.gradY);
  if (gradMag <= 0.001) return [0, 0];
  const error = signature.tidalMagnitude - config.targetTidal;
  const thrustMagnitude = Math.min(Math.abs(0.5 * error), config.thrustLimit);
  const direction = error > 0 ? -1 : 1;
  return [
    direction * thrustMagnitude * signature.gradX / gradMag,
    direction * thrustMagnitude * signature.gradY / gradMag,
  ];
}

function terminalEventsForState(state, config) {
  const signatures = computeSignatures(state, config);
  const tidal = computeTidalTensor(state, config);
  const events = classifyEvents(state, signatures, tidal, [0, 0], config);
  return events.invalid || events.escape || events.closeApproach;
}

// Energy-trend terminal value (BF-2 repair after the BF-4b inert-floor
// receipt). The earlier bounded radius/close-approach margin was
// non-discriminating over a short horizon: 16 integrator steps barely move the
// trajectory, so the cross-candidate margin spread was ~1e-6 and every
// deviation fell under the signature-baseline guard (floor valid but inert
// off-set). Total energy = kinetic + (negative) potential; lower/more-negative
// energy is more bound, higher is toward escape. Thrust does work and changes
// energy *immediately*, so the energy trend over the rollout discriminates a
// one-step deviation even when within-horizon survival is flat. The value is
// self-scaled by the particle's own binding energy (belief-only: computed from
// the particle rollout, never the true state; no oracle), centered at 0.5, and
// clamped to [0,1] so the shaping term stays strictly below one dt and the
// floor-validity invariant (never prefer an earlier-escape action) is
// preserved exactly as before — only what `margin in [0,1]` measures changed.
function terminalEnergyMargin(startState, endState, config) {
  const startEnergy = computeSignatures(startState, config).energy;
  const endEnergy = computeSignatures(endState, config).energy;
  const scale = Math.max(Math.abs(startEnergy), 1e-9);
  const trend = (startEnergy - endEnergy) / scale;
  return Math.min(1, Math.max(0, 0.5 + 0.5 * trend));
}
// Candidate rollout: the signature-policy baseline is pure signature policy at
// every step; a lattice candidate holds its thrust for the first
// `candidateHoldSteps` steps, then follows the signature policy. Holding for >1
// step is required for the information-accessibility diagnostic: a single-step
// perturbation does not propagate over a short horizon, so K=1 makes every
// deviation invisible (the BF-4b inert-floor root cause). Default
// `candidateHoldSteps = 1` preserves all prior BF-4 / BF-4b behaviour exactly.
function candidateThrustAtStep(candidate, simState, config, sensorState, stepIndex, holdSteps) {
  if (candidate.kind === "signature_policy" || stepIndex >= holdSteps) {
    const predicted = observeGuardedAccelSignature(simState, { ...config, sensorNoiseStd: 0 }, sensorState);
    return guardedSignatureThrust(predicted, config);
  }
  return candidate.thrust;
}

function scoreCandidateAction(particles, candidate, config, args) {
  let expectedSafeTime = 0;
  let expectedTerminalMargin = 0;
  let expectedScore = 0;
  let expectedDeltaV = 0;
  for (const particle of particles) {
    let simState = particle.state.slice();
    const rolloutSensorState = cloneSensorState(particle.sensorState);
    let safeTime = 0;
    let hazardReached = false;
    for (let step = 0; step < args.planningHorizonSteps; step += 1) {
      if (terminalEventsForState(simState, config)) {
        hazardReached = true;
        break;
      }
      const thrust = candidateThrustAtStep(candidate, simState, config, rolloutSensorState, step, args.candidateHoldSteps);
      const thrustMagnitude = Math.sqrt(thrust[0] * thrust[0] + thrust[1] * thrust[1]);
      safeTime += config.dt;
      expectedDeltaV += particle.weight * thrustMagnitude * config.dt;
      simState = integrateStep(simState, config.dt, config, thrust);
    }
    const margin = hazardReached ? 0 : terminalEnergyMargin(particle.state, simState, config);
    const particleScore = safeTime + args.shapeFraction * config.dt * margin;
    expectedSafeTime += particle.weight * safeTime;
    expectedTerminalMargin += particle.weight * margin;
    expectedScore += particle.weight * particleScore;
  }
  return { expectedSafeTime, expectedTerminalMargin, expectedScore, expectedDeltaV };
}

const SCORE_TIE_EPS = 1e-9;

function makeActionCandidates(config, observed) {
  const signatureThrust = guardedSignatureThrust(observed, config);
  const candidates = [{
    index: 0,
    kind: "signature_policy",
    thrust: signatureThrust,
    actionKey: actionKey(signatureThrust),
  }];
  const seen = new Set([candidates[0].actionKey]);
  for (const thrust of oracleCandidateThrusts(config)) {
    const key = actionKey(thrust);
    if (seen.has(key)) continue;
    candidates.push({
      index: candidates.length,
      kind: "lattice_first_step_then_signature_policy",
      thrust,
      actionKey: key,
    });
    seen.add(key);
  }
  return candidates;
}

function chooseBayesAction(particles, config, args, observed) {
  const candidates = makeActionCandidates(config, observed);
  let best = null;
  const scores = [];
  for (const candidate of candidates) {
    const score = scoreCandidateAction(particles, candidate, config, args);
    const row = {
      ...candidate,
      ...score,
    };
    scores.push(row);
    // Pinned tie order: (1) maximize the shaped planning score;
    // (2) within SCORE_TIE_EPS of the best shaped score, minimize expected
    // delta-V; (3) first in the pre-registered lattice order (zero-first,
    // preserved by only replacing on strict improvement).
    if (
      best === null
      || row.expectedScore > best.expectedScore + SCORE_TIE_EPS
      || (
        Math.abs(row.expectedScore - best.expectedScore) <= SCORE_TIE_EPS
        && row.expectedDeltaV < best.expectedDeltaV - 1e-12
      )
    ) {
      best = row;
    }
  }
  const signatureBaseline = scores[0];
  const advantageThreshold = args.signatureAdvantageDtMultiplier * config.dt;
  const selected = best.expectedScore > signatureBaseline.expectedScore + advantageThreshold
    ? best
    : signatureBaseline;
  return {
    selected,
    scores,
    signatureBaseline,
    bestBeforeGuard: best,
    advantageThreshold,
    fallbackGuardApplied: selected === signatureBaseline && best !== signatureBaseline,
  };
}

function eventHistoryEntry(time, state, thrust, config) {
  const signatures = computeSignatures(state, config);
  const tidal = computeTidalTensor(state, config);
  const events = classifyEvents(state, signatures, tidal, thrust, config);
  const [localAx, localAy] = computeAcceleration(2, state.slice(0, 6), config);
  return {
    time,
    events,
    tidalMagnitude: tidal.magnitude,
    localAccelerationMagnitude: Math.sqrt(localAx * localAx + localAy * localAy),
  };
}

function runBayesTrial(args, envelopeCase, regime, seed, guardThresholds) {
  const config = makeTrialConfig(args, envelopeCase, seed, regime, guardThresholds);
  const id = trialId(envelopeCase, regime, seed);
  const rng = makeRng((args.bootstrapSeed + seed * 4099) >>> 0);
  const particles = initializeParticles(args, envelopeCase, regime, seed, config);
  const actualSensorState = {};
  const stateRows = [];
  const beliefRows = [];
  const actionRows = [];
  const eventHistory = [];
  let state = initializeState(config);
  let totalDeltaV = 0;
  const steps = Math.max(1, Math.round(config.duration / config.dt));

  for (let step = 0; step <= steps; step += 1) {
    const time = step * config.dt;
    const observed = observeGuardedAccelSignature(state, config, actualSensorState);
    const belief = updateParticleBelief(particles, observed, config, args, rng);
    const decision = chooseBayesAction(particles, config, args, observed);
    const thrust = decision.selected.thrust;
    const historyEntry = eventHistoryEntry(time, state, thrust, config);
    eventHistory.push(historyEntry);

    stateRows.push({
      trial_id: id,
      case_id: caseId(envelopeCase),
      seed,
      regime,
      step,
      time: roundMetric(time),
      mass_ratio: envelopeCase.massRatio,
      timestep: envelopeCase.timestep,
      radius_scale: envelopeCase.radiusScale,
      velocity_scale: envelopeCase.velocityScale,
      thrust_limit: envelopeCase.thrustLimit,
      sensor_noise_std: envelopeCase.sensorNoiseStd,
      guard: observed.guard,
      tidal_magnitude: roundNumber(observed.tidalMagnitude),
      grad_x: roundNumber(observed.gradX),
      grad_y: roundNumber(observed.gradY),
      gradient_magnitude: roundNumber(Math.sqrt(observed.gradX * observed.gradX + observed.gradY * observed.gradY)),
      sensor_tier: observed.sensorTier,
      probe_delta: observed.probeDelta,
      action_index: decision.selected.index,
      action_family: decision.selected.kind,
      action_key: decision.selected.actionKey,
      thrust_x: roundNumber(thrust[0]),
      thrust_y: roundNumber(thrust[1]),
    });

    beliefRows.push({
      trial_id: id,
      case_id: caseId(envelopeCase),
      seed,
      regime,
      step,
      time: roundMetric(time),
      particle_count: particles.length,
      effective_sample_size: roundMetric(belief.effectiveSampleSize),
      resampled: belief.resampled,
    });

    actionRows.push({
      trial_id: id,
      case_id: caseId(envelopeCase),
      seed,
      regime,
      step,
      time: roundMetric(time),
      action_index: decision.selected.index,
      action_family: decision.selected.kind,
      action_key: decision.selected.actionKey,
      thrust_x: roundNumber(thrust[0]),
      thrust_y: roundNumber(thrust[1]),
      expected_safe_time: roundMetric(decision.selected.expectedSafeTime),
      expected_terminal_margin: roundMetric(decision.selected.expectedTerminalMargin),
      expected_score: roundMetric(decision.selected.expectedScore),
      expected_delta_v: roundMetric(decision.selected.expectedDeltaV),
      signature_baseline_action_key: decision.signatureBaseline.actionKey,
      signature_baseline_score: roundMetric(decision.signatureBaseline.expectedScore),
      best_pre_guard_action_key: decision.bestBeforeGuard.actionKey,
      best_pre_guard_score: roundMetric(decision.bestBeforeGuard.expectedScore),
      signature_advantage_threshold: roundMetric(decision.advantageThreshold),
      fallback_guard_applied: decision.fallbackGuardApplied,
      candidate_scores: JSON.stringify(decision.scores.map((score) => ({
        index: score.index,
        kind: score.kind,
        action_key: score.actionKey,
        expected_safe_time: roundMetric(score.expectedSafeTime),
        expected_terminal_margin: roundMetric(score.expectedTerminalMargin),
        expected_score: roundMetric(score.expectedScore),
        expected_delta_v: roundMetric(score.expectedDeltaV),
      }))),
    });

    if (step === steps || historyEntry.events.invalid || historyEntry.events.escape || historyEntry.events.closeApproach) break;
    const thrustMagnitude = Math.sqrt(thrust[0] * thrust[0] + thrust[1] * thrust[1]);
    totalDeltaV += thrustMagnitude * config.dt;
    state = integrateStep(state, config.dt, config, thrust);
    for (const particle of particles) {
      particle.state = integrateStep(particle.state, config.dt, config, thrust);
    }
  }

  const summary = summarizeOutcome(eventHistory);
  return {
    id,
    config,
    stateRows,
    beliefRows,
    actionRows,
    outcomeRow: {
      trial_id: id,
      case_id: caseId(envelopeCase),
      seed,
      regime,
      mass_ratio: envelopeCase.massRatio,
      timestep: envelopeCase.timestep,
      radius_scale: envelopeCase.radiusScale,
      velocity_scale: envelopeCase.velocityScale,
      thrust_limit: envelopeCase.thrustLimit,
      sensor_noise_std: envelopeCase.sensorNoiseStd,
      guard_mode: args.trackGuardMode,
      guard_quantile: envelopeCase.trackGuardQuantile,
      guard_min_radius: config.trackGuardMinRadius,
      guard_max_local_acceleration: config.trackGuardMaxLocalAcceleration,
      guard_max_tidal_magnitude: config.trackGuardMaxTidalMagnitude,
      guard_calibration_sample_count: guardThresholds?.guardCalibrationSampleCount ?? null,
      controller_mode: "bayes_floor_particle_mpc",
      terminal_outcome: summary.terminalOutcome,
      simulated_time: roundMetric(eventHistory.at(-1)?.time ?? 0),
      total_delta_v: roundNumber(totalDeltaV),
      min_primary_distance: roundNumber(summary.minPrimaryDistance),
      max_radius: roundNumber(summary.maxRadius),
      particle_count: args.particleCount,
      planning_horizon_steps: args.planningHorizonSteps,
      resample_threshold: args.resampleThreshold,
    },
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(repoRoot, args.out);
  await mkdir(outDir, { recursive: true });

  const cases = envelopeCases(args);
  const manifest = {
    schema: `sundog.threebody.${args.phase}.bayes_floor.v1`,
    startedAt: new Date().toISOString(),
    purpose: "Phase 4 same-signature particle-belief MPC Bayesian-floor evaluator.",
    args,
    cases,
    actionSelectionInputs: ["guarded_accelerometer_signature_history", "particle_belief"],
    rawStateUse: ["truth_rollout", "particle_simulation", "readout_only"],
    controllerMode: "bayes_floor_particle_mpc",
    observationModel: {
      observedSignature: "observeGuardedAccelSignature(state, cfg, actualSensorState)",
      particlePrediction: "observeGuardedAccelSignature(particle.state, { ...cfg, sensorNoiseStd: 0 }, particle.sensorState)",
      likelihood: "diagonal Gaussian-like score over |T_hat|, gradX, gradY plus guard mismatch penalty",
    },
    planningObjective: {
      form: "expected within-horizon survival time + shapeFraction * dt * expected terminal energy-trend margin",
      shapeFraction: args.shapeFraction,
      signatureAdvantageDtMultiplier: args.signatureAdvantageDtMultiplier,
      candidateHoldSteps: args.candidateHoldSteps,
      candidateHoldNote: "lattice candidate holds its thrust for candidateHoldSteps steps then follows signature policy; 1 = prior single-step rollout (one-step deviations are invisible over a short horizon); >1 is the information-accessibility diagnostic rollout",
      terminalMargin: "energy-trend margin clamp(0.5 + 0.5 * (E_start - E_end) / max(|E_start|, 1e-9), 0, 1) over the particle rollout (total energy = kinetic + potential; lower = more bound); 0 if a terminal hazard was reached in the rollout. Belief-only: computed from particle rollout states, never the true state.",
      floorValidityInvariant: "terminal margin in [0,1] and shapeFraction < 1 => shaping term < one dt, so a candidate whose true survival is longer by >= one dt is never overtaken on margin; the floor never prefers an earlier-escape action",
      signatureFallbackInvariant: "the guarded-signature policy is an explicit same-information candidate; the evaluator deviates from it only when predicted shaped-score advantage exceeds signatureAdvantageDtMultiplier * dt",
      supersedes: "BF-4 pure steps-survived objective (degenerated to passive) then the bounded radius/close-approach margin (BF-4b: valid but inert off-set, cross-candidate spread ~1e-6); replaced 2026-05-16 with the energy-trend margin so a one-step deviation is discriminable even when within-horizon survival is flat",
      tieOrder: "signature baseline unless best predicted shaped score beats it by the configured dt-scaled advantage threshold; within 1e-9 -> min expected delta-V; then first in candidate order",
    },
    actionCandidates: [
      {
        index: 0,
        kind: "signature_policy",
        description: "guarded-signature controller action computed from the admitted observation",
      },
      ...oracleCandidateThrusts(normalizeConfig({ thrustLimit: args.thrustLimits[0] })).map((thrust, index) => ({
        index: index + 1,
        kind: "lattice_first_step_then_signature_policy",
        actionKey: actionKey(thrust),
        thrust,
      })),
    ],
    trials: [],
  };
  const signatureRows = [];
  const beliefRows = [];
  const actionRows = [];
  const outcomeRows = [];

  for (const envelopeCase of cases) {
    for (const regime of args.regimes) {
      const guardThresholds = guardThresholdsForCase(args, envelopeCase, regime);
      for (let offset = 0; offset < args.seeds; offset += 1) {
        const seed = args.seedStart + offset;
        const trial = runBayesTrial(args, envelopeCase, regime, seed, guardThresholds);
        signatureRows.push(...trial.stateRows);
        beliefRows.push(...trial.beliefRows);
        actionRows.push(...trial.actionRows);
        outcomeRows.push(trial.outcomeRow);
        manifest.trials.push({
          id: trial.id,
          caseId: caseId(envelopeCase),
          seed,
          regime,
          initialParticle: trial.config.initialParticle,
          summary: {
            terminalOutcome: trial.outcomeRow.terminal_outcome,
            simulatedTime: trial.outcomeRow.simulated_time,
            totalDeltaV: trial.outcomeRow.total_delta_v,
          },
        });
      }
    }
  }

  manifest.completedAt = new Date().toISOString();
  await writeFile(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  await writeFile(
    path.join(outDir, "signature-observations.jsonl"),
    `${signatureRows.map((row) => JSON.stringify(row)).join("\n")}\n`,
    "utf8",
  );
  await writeFile(path.join(outDir, "belief-diagnostics.csv"), rowsToCsv(beliefRows), "utf8");
  await writeFile(path.join(outDir, "bayes-actions.csv"), rowsToCsv(actionRows), "utf8");
  await writeFile(path.join(outDir, "bayes-trial-outcomes.csv"), rowsToCsv(outcomeRows), "utf8");

  console.log(`[threebody-bayes-floor] wrote ${outcomeRows.length} trials to ${path.relative(repoRoot, outDir)}`);
  console.log("[threebody-bayes-floor] wrote manifest.json, signature-observations.jsonl, belief-diagnostics.csv, bayes-actions.csv, bayes-trial-outcomes.csv");
}

main().then(() => {
  process.exit(0);
}).catch((error) => {
  console.error(error);
  process.exit(1);
});
