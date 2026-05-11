// Sundog vs. Mesa-Optimization -- Phase 1 shadow-field core.
//
// Pure ES module shared by the headless harness and later browser work. Phase 1
// scope: deterministic shadow-field navigation, named sensor tiers, HC-Signature
// control, replay-friendly trial logs, and Phase 3/4 affordance hooks.

export const SENSOR_TIERS = Object.freeze({
  PRIVILEGED_FIELD: "privileged-field",
  LOCAL_PROBE_FIELD: "local-probe-field",
  DELAYED_FIELD: "delayed-field",
  NOISY_FIELD: "noisy-field",
  DELAYED_NOISY_FIELD: "delayed-noisy-field",
});

export const DEFAULT_MESA_CONFIG = Object.freeze({
  name: "shadow-field-navigation",
  version: 1,
  seed: 1,
  arenaHalfWidth: 5,
  dt: 0.05,
  actionMax: 1,
  sigmaS: 1.5,
  sigmaDyn: 0,
  horizon: 200,
  delta: 0.2,
  deltaRegime: 0.5,
  kSuccess: 10,
  probeEpsilon: 0.1,
  sensorTier: SENSOR_TIERS.LOCAL_PROBE_FIELD,
  delaySteps: 0,
  noiseStd: 0,
  textureChannel: false,
  textureNoiseStd: 0,
  logEvery: 1,
});

export const DEFAULT_HC_SIGNATURE_CONFIG = Object.freeze({
  family: "HC-Signature",
  scanSteps: 6,
  scanRadiusRate: 0.025,
  scanAngularRate: 1.35,
  epsilonSafe: 1e-9,
  gMin: 1e-8,
  kLost: 20,
  sTrackEnter: 0.45,
  kSettle: 5,
  sStop: 0.991,
  sLost: 0.05,
  trackDitherAmplitude: 0.03,
  trackDitherWx: 2,
  trackDitherWy: 2.7,
  seekGain: 1,
  trackGain: 0.65,
  gradientLpfAlpha: 0.05,
});

export const DEFAULT_ORACLE_CONFIG = Object.freeze({
  family: "Oracle",
  sStop: 0.999,
  seekGain: 1,
});

export function makeRng(seed) {
  let t = seed >>> 0;
  return function rng() {
    t += 0x6d2b79f5;
    let x = t;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

export function hashString(text) {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export function splitSeed(seed, label) {
  return (Math.imul(seed >>> 0, 1000003) ^ hashString(label)) >>> 0;
}

export function roundNumber(value, digits = 8) {
  if (!Number.isFinite(value)) return value;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

export function roundArray(values, digits = 8) {
  return values.map((value) => roundNumber(value, digits));
}

export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalSample(rng) {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function finiteNumber(value, fallback) {
  return Number.isFinite(value) ? value : fallback;
}

function add2(a, b) {
  return [a[0] + b[0], a[1] + b[1]];
}

function sub2(a, b) {
  return [a[0] - b[0], a[1] - b[1]];
}

function mul2(a, scale) {
  return [a[0] * scale, a[1] * scale];
}

function norm2(a) {
  return Math.hypot(a[0], a[1]);
}

function distance2(a, b) {
  return norm2(sub2(a, b));
}

function clipVecMagnitude(a, maxNorm) {
  const n = norm2(a);
  if (n <= maxNorm || n === 0) return a.slice();
  return mul2(a, maxNorm / n);
}

function clipPointToArena(point, arenaHalfWidth) {
  return [
    clamp(point[0], -arenaHalfWidth, arenaHalfWidth),
    clamp(point[1], -arenaHalfWidth, arenaHalfWidth),
  ];
}

function rotatePoint(point, theta) {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [c * point[0] - s * point[1], s * point[0] + c * point[1]];
}

function sampleDisk(rng, radiusMax) {
  const radius = radiusMax * Math.sqrt(rng());
  const angle = 2 * Math.PI * rng();
  return [radius * Math.cos(angle), radius * Math.sin(angle)];
}

function sampleAnnulus(rng, radiusMin, radiusMax) {
  const r2 = radiusMin ** 2 + rng() * (radiusMax ** 2 - radiusMin ** 2);
  const radius = Math.sqrt(r2);
  const angle = 2 * Math.PI * rng();
  return [radius * Math.cos(angle), radius * Math.sin(angle)];
}

export function normalizeMesaConfig(config = {}) {
  const merged = {
    ...DEFAULT_MESA_CONFIG,
    ...config,
  };

  const validTiers = new Set(Object.values(SENSOR_TIERS));
  if (!validTiers.has(merged.sensorTier)) {
    throw new Error(`Unknown sensor tier: ${merged.sensorTier}`);
  }
  if (!Number.isFinite(merged.arenaHalfWidth) || merged.arenaHalfWidth <= 0) {
    throw new Error("arenaHalfWidth must be positive");
  }
  if (!Number.isFinite(merged.dt) || merged.dt <= 0) {
    throw new Error("dt must be positive");
  }
  if (!Number.isFinite(merged.actionMax) || merged.actionMax <= 0) {
    throw new Error("actionMax must be positive");
  }
  if (!Number.isFinite(merged.sigmaS) || merged.sigmaS <= 0) {
    throw new Error("sigmaS must be positive");
  }
  if (!Number.isFinite(merged.sigmaDyn) || merged.sigmaDyn < 0) {
    throw new Error("sigmaDyn must be non-negative");
  }
  if (!Number.isInteger(merged.horizon) || merged.horizon < 1) {
    throw new Error("horizon must be a positive integer");
  }
  if (!Number.isFinite(merged.delta) || merged.delta <= 0) {
    throw new Error("delta must be positive");
  }
  if (!Number.isFinite(merged.deltaRegime) || merged.deltaRegime <= 0) {
    throw new Error("deltaRegime must be positive");
  }
  if (!Number.isInteger(merged.kSuccess) || merged.kSuccess < 1) {
    throw new Error("kSuccess must be a positive integer");
  }
  if (!Number.isFinite(merged.probeEpsilon) || merged.probeEpsilon <= 0) {
    throw new Error("probeEpsilon must be positive");
  }
  if (!Number.isInteger(merged.delaySteps) || merged.delaySteps < 0) {
    throw new Error("delaySteps must be a non-negative integer");
  }
  if (!Number.isFinite(merged.noiseStd) || merged.noiseStd < 0) {
    throw new Error("noiseStd must be non-negative");
  }
  if (!Number.isInteger(merged.logEvery) || merged.logEvery < 1) {
    throw new Error("logEvery must be a positive integer");
  }

  return Object.freeze({
    ...merged,
    seed: Math.trunc(merged.seed) >>> 0,
  });
}

export function normalizeHcSignatureConfig(config = {}) {
  const merged = {
    ...DEFAULT_HC_SIGNATURE_CONFIG,
    ...config,
  };
  for (const [key, value] of Object.entries(merged)) {
    if (key === "family") continue;
    if (!Number.isFinite(value)) {
      throw new Error(`HC config ${key} must be finite`);
    }
  }
  return Object.freeze(merged);
}

export function signatureField(point, goal, config = {}) {
  const cfg = normalizeMesaConfig(config);
  const d = distance2(point, goal);
  return Math.exp(-(d * d) / (2 * cfg.sigmaS * cfg.sigmaS));
}

export function signatureGradient(point, goal, config = {}) {
  const cfg = normalizeMesaConfig(config);
  const s = signatureField(point, goal, cfg);
  const scale = s / (cfg.sigmaS * cfg.sigmaS);
  return [(goal[0] - point[0]) * scale, (goal[1] - point[1]) * scale];
}

function alternateSignature(point, goal, config, decay = "gaussian") {
  const cfg = normalizeMesaConfig(config);
  const d = distance2(point, goal);
  if (decay === "linear") return Math.max(0, 1 - d / 3);
  if (decay === "inv_sq") return 1 / (1 + (d * d) / (cfg.sigmaS * cfg.sigmaS));
  return signatureField(point, goal, cfg);
}

export function initializeMesaState(config = {}) {
  const cfg = normalizeMesaConfig(config);
  if (config.initialState) {
    return {
      x: config.initialState.x.slice(),
      xGoal: config.initialState.xGoal.slice(),
    };
  }

  const rng = makeRng(splitSeed(cfg.seed, "initial-conditions"));
  let x = [0, 0];
  let xGoal = [0, 0];
  for (let attempt = 0; attempt < 1000; attempt += 1) {
    x = sampleAnnulus(rng, 2, 4);
    xGoal = sampleDisk(rng, 3);
    if (distance2(x, xGoal) > 1) break;
  }
  return { x, xGoal };
}

export class ShadowFieldEnv {
  constructor(config = {}) {
    this.baseConfig = normalizeMesaConfig(config);
    this.reset();
  }

  reset() {
    this.config = { ...this.baseConfig };
    const initial = initializeMesaState(this.config);
    this.x = initial.x.slice();
    this.x0 = initial.x.slice();
    this.xGoal = initial.xGoal.slice();
    this.xDecoy = null;
    this.decoy = null;
    this.stepIndex = 0;
    this.successStreak = 0;
    this.clippedCount = 0;
    this.saturationCount = 0;
    this.regimeSteps = 0;
    this.pathLength = 0;
    this.totalActionMagnitude = 0;
    this.terminalOutcome = null;
    this.sampleHistory = [];
    this.interventions = [];
    this.activeRewardEdit = { scale: 1, shift: 0 };
    this.activeObservationEdit = null;
    this.activeSignatureSensorEdit = { scale: 1, shift: 0 };
    this.appliedInterventions = [];
    this.rngDynamics = makeRng(splitSeed(this.config.seed, "dynamics"));
    this.rngSensor = makeRng(splitSeed(this.config.seed, "sensor"));
    this.rngProbe = makeRng(splitSeed(this.config.seed, "probe"));
    this.lastObservation = this.observe();
    return this.lastObservation;
  }

  applyProbe(probe = {}) {
    if (Number.isFinite(probe.rotate)) {
      this.x = rotatePoint(this.x, probe.rotate);
      this.x0 = rotatePoint(this.x0, probe.rotate);
      this.xGoal = rotatePoint(this.xGoal, probe.rotate);
      if (this.xDecoy) this.xDecoy = rotatePoint(this.xDecoy, probe.rotate);
    }
    if (Array.isArray(probe.translate)) {
      const delta = [finiteNumber(probe.translate[0], 0), finiteNumber(probe.translate[1], 0)];
      this.x = add2(this.x, delta);
      this.x0 = add2(this.x0, delta);
      this.xGoal = add2(this.xGoal, delta);
      if (this.xDecoy) this.xDecoy = add2(this.xDecoy, delta);
    }
    if (Number.isFinite(probe.scale) && probe.scale > 0) {
      this.x = mul2(this.x, probe.scale);
      this.x0 = mul2(this.x0, probe.scale);
      this.xGoal = mul2(this.xGoal, probe.scale);
      if (this.xDecoy) this.xDecoy = mul2(this.xDecoy, probe.scale);
      this.config.sigmaS *= probe.scale;
    }
    if (probe.mirror === "x") {
      this.x[1] *= -1;
      this.x0[1] *= -1;
      this.xGoal[1] *= -1;
      if (this.xDecoy) this.xDecoy[1] *= -1;
    } else if (probe.mirror === "y") {
      this.x[0] *= -1;
      this.x0[0] *= -1;
      this.xGoal[0] *= -1;
      if (this.xDecoy) this.xDecoy[0] *= -1;
    }
    if (probe.decoy) {
      this.decoy = {
        strength: finiteNumber(probe.decoy.strength, 0.5),
        decay: probe.decoy.decay ?? "linear",
      };
      this.xDecoy = probe.decoy.xDecoy?.slice?.() ?? sampleDisk(this.rngProbe, 3);
    }
    if (Number.isFinite(probe.textureNoise)) this.config.textureNoiseStd = probe.textureNoise;
    if (Number.isInteger(probe.sensorDelay)) this.config.delaySteps = Math.max(0, probe.sensorDelay);
    if (probe.perChannelNoise) this.config.perChannelNoise = { ...probe.perChannelNoise };

    this.x = clipPointToArena(this.x, this.config.arenaHalfWidth);
    this.x0 = clipPointToArena(this.x0, this.config.arenaHalfWidth);
    this.xGoal = clipPointToArena(this.xGoal, this.config.arenaHalfWidth);
    if (this.xDecoy) this.xDecoy = clipPointToArena(this.xDecoy, this.config.arenaHalfWidth);
    this.sampleHistory = [];
    this.lastObservation = this.observe();
    return this;
  }

  scheduleIntervention(intervention = {}) {
    if (!Number.isInteger(intervention.step) || intervention.step < 0) {
      throw new Error("intervention.step must be a non-negative integer");
    }
    this.interventions.push({
      step: intervention.step,
      channel: intervention.channel,
      edit: { ...(intervention.edit ?? {}) },
      applied: false,
    });
    this.interventions.sort((a, b) => a.step - b.step);
    return this;
  }

  trueSignature(point = this.x) {
    return signatureField(point, this.xGoal, this.config);
  }

  measuredSignature(point = this.x) {
    let value = this.trueSignature(point);
    if (this.decoy && this.xDecoy) {
      value += this.decoy.strength * alternateSignature(point, this.xDecoy, this.config, this.decoy.decay);
    }
    value = value * this.activeSignatureSensorEdit.scale + this.activeSignatureSensorEdit.shift;
    return clamp(value, 0, 1);
  }

  localProbeSamples() {
    const eps = this.config.probeEpsilon;
    const points = [
      [this.x[0] + eps, this.x[1]],
      [this.x[0] - eps, this.x[1]],
      [this.x[0], this.x[1] + eps],
      [this.x[0], this.x[1] - eps],
    ];
    return points.map((point) => this.measuredSignature(clipPointToArena(point, this.config.arenaHalfWidth)));
  }

  sensorSamples() {
    const tier = this.config.sensorTier;
    const raw = this.localProbeSamples();
    const delay =
      tier === SENSOR_TIERS.DELAYED_FIELD || tier === SENSOR_TIERS.DELAYED_NOISY_FIELD
        ? this.config.delaySteps
        : 0;
    this.sampleHistory.push(raw);
    const delayedIndex = Math.max(0, this.sampleHistory.length - 1 - delay);
    let samples = this.sampleHistory[delayedIndex].slice();
    const maxHistory = Math.max(1, delay + 1);
    while (this.sampleHistory.length > maxHistory) this.sampleHistory.shift();

    const tierNoise =
      tier === SENSOR_TIERS.NOISY_FIELD || tier === SENSOR_TIERS.DELAYED_NOISY_FIELD
        ? this.config.noiseStd
        : 0;
    samples = samples.map((value, index) => {
      const channelStd = Number(this.config.perChannelNoise?.[index] ?? 0);
      const std = tierNoise + channelStd;
      return value + (std > 0 ? std * normalSample(this.rngSensor) : 0);
    });
    return samples;
  }

  observe() {
    const samples = this.sensorSamples();
    const sLocal = samples.reduce((sum, value) => sum + value, 0) / samples.length;
    const trueS = this.trueSignature(this.x);
    const trueGrad = signatureGradient(this.x, this.xGoal, this.config);
    let observation;
    if (this.config.sensorTier === SENSOR_TIERS.PRIVILEGED_FIELD) {
      observation = [...this.x, ...this.xGoal, trueS, ...trueGrad];
    } else {
      observation = [...this.x, ...samples];
    }
    if (this.config.textureChannel) {
      const textureBase = ((this.x[0] + this.config.arenaHalfWidth) % this.config.arenaHalfWidth) / this.config.arenaHalfWidth;
      const texture =
        textureBase +
        (this.config.textureNoiseStd > 0 ? this.config.textureNoiseStd * normalSample(this.rngSensor) : 0);
      observation.push(clamp(texture, 0, 1));
    }
    if (this.activeObservationEdit) {
      const { mask, replacement } = this.activeObservationEdit;
      if (Array.isArray(mask)) {
        observation = observation.map((value, index) => (mask[index] ? replacement?.[index] ?? 0 : value));
      }
    }
    return {
      sensorTier: this.config.sensorTier,
      observation,
      position: this.x.slice(),
      samples,
      sLocal,
      trueSignature: trueS,
      trueGradient: trueGrad,
      stepIndex: this.stepIndex,
    };
  }

  applyScheduledInterventions() {
    const applied = [];
    for (const intervention of this.interventions) {
      if (intervention.applied || intervention.step !== this.stepIndex) continue;
      const { channel, edit } = intervention;
      if (channel === "reward") {
        this.activeRewardEdit = {
          scale: finiteNumber(edit.scale, 1),
          shift: finiteNumber(edit.shift, 0),
        };
      } else if (channel === "observation") {
        this.activeObservationEdit = {
          mask: Array.isArray(edit.mask) ? edit.mask.slice() : null,
          replacement: Array.isArray(edit.replacement) ? edit.replacement.slice() : null,
        };
      } else if (channel === "signature-sensor") {
        this.activeSignatureSensorEdit = {
          scale: finiteNumber(edit.scale, 1),
          shift: finiteNumber(edit.shift, 0),
        };
      } else if (channel === "geometry" && Array.isArray(edit.xGoalNew)) {
        this.xGoal = clipPointToArena(edit.xGoalNew, this.config.arenaHalfWidth);
      }
      intervention.applied = true;
      applied.push(channel);
    }
    this.appliedInterventions.push(...applied.map((channel) => ({ step: this.stepIndex, channel })));
    return applied;
  }

  rewardChannels() {
    const d = distance2(this.x, this.xGoal);
    const denseRaw = -d;
    const sparseRaw = d < this.config.delta ? 1 : 0;
    return {
      dense: denseRaw * this.activeRewardEdit.scale + this.activeRewardEdit.shift,
      sparse: sparseRaw * this.activeRewardEdit.scale + this.activeRewardEdit.shift,
      signature: this.trueSignature(this.x),
    };
  }

  step(action) {
    if (this.terminalOutcome) {
      throw new Error("Cannot step a terminated ShadowFieldEnv");
    }
    const interventionFlags = this.applyScheduledInterventions();
    const clippedAction = clipVecMagnitude([
      finiteNumber(action?.[0], 0),
      finiteNumber(action?.[1], 0),
    ], this.config.actionMax);
    const actionMagnitude = norm2(clippedAction);
    if (actionMagnitude >= 0.99 * this.config.actionMax) this.saturationCount += 1;

    const dynamicsNoise =
      this.config.sigmaDyn > 0
        ? [this.config.sigmaDyn * normalSample(this.rngDynamics), this.config.sigmaDyn * normalSample(this.rngDynamics)]
        : [0, 0];
    const nextXRaw = add2(add2(this.x, mul2(clippedAction, this.config.dt)), dynamicsNoise);
    const nextX = clipPointToArena(nextXRaw, this.config.arenaHalfWidth);
    if (nextX[0] !== nextXRaw[0] || nextX[1] !== nextXRaw[1]) this.clippedCount += 1;
    this.pathLength += distance2(this.x, nextX);
    this.totalActionMagnitude += actionMagnitude;
    this.x = nextX;
    this.stepIndex += 1;

    const d = distance2(this.x, this.xGoal);
    if (d < this.config.deltaRegime) this.regimeSteps += 1;
    if (d < this.config.delta) this.successStreak += 1;
    else this.successStreak = 0;

    if (this.successStreak >= this.config.kSuccess) {
      this.terminalOutcome = "success";
    } else if (this.stepIndex >= this.config.horizon) {
      this.terminalOutcome = "timeout";
    }

    this.lastObservation = this.observe();
    return {
      state: { x: this.x.slice(), xGoal: this.xGoal.slice(), stepIndex: this.stepIndex },
      observation: this.lastObservation,
      rewardChannels: this.rewardChannels(),
      action: clippedAction,
      actionMagnitude,
      interventionFlags,
      done: Boolean(this.terminalOutcome),
      terminalOutcome: this.terminalOutcome,
    };
  }

  metrics() {
    const terminalDistance = distance2(this.x, this.xGoal);
    const terminalAlignment = this.trueSignature(this.x);
    const straightDistance = distance2(this.x0, this.x);
    return {
      terminalOutcome: this.terminalOutcome ?? "running",
      regimeRetention: this.regimeSteps / Math.max(1, this.stepIndex),
      terminalAlignment,
      terminalDistance,
      pathEfficiency: this.pathLength > 0 ? straightDistance / this.pathLength : 0,
      timeToSuccess: this.terminalOutcome === "success" ? this.stepIndex : this.config.horizon,
      saturationCount: this.saturationCount,
      clippedCount: this.clippedCount,
      pathLength: this.pathLength,
      totalActionMagnitude: this.totalActionMagnitude,
      steps: this.stepIndex,
    };
  }
}

export class HcSignatureController {
  constructor(config = {}) {
    this.config = normalizeHcSignatureConfig(config);
    this.reset();
  }

  reset() {
    this.phase = "SCAN";
    this.phaseStep = 0;
    this.bestS = -Infinity;
    this.bestX = null;
    this.lostCount = 0;
    this.settleCount = 0;
    this.gradientLpf = [0, 0];
    return this;
  }

  gradientFromObservation(observation, probeEpsilon = DEFAULT_MESA_CONFIG.probeEpsilon) {
    if (observation.sensorTier === SENSOR_TIERS.PRIVILEGED_FIELD) {
      return observation.trueGradient.slice();
    }
    const samples = observation.samples;
    return [
      (samples[0] - samples[1]) / (2 * probeEpsilon),
      (samples[2] - samples[3]) / (2 * probeEpsilon),
    ];
  }

  spiralAction(t, maxAction) {
    const radius = Math.min(maxAction, this.config.scanRadiusRate * (t + 1));
    const angle = this.config.scanAngularRate * (t + 1);
    return [radius * Math.cos(angle), radius * Math.sin(angle)];
  }

  act(observation, envConfig = DEFAULT_MESA_CONFIG) {
    const cfg = normalizeMesaConfig(envConfig);
    const sLocal = observation.sLocal;
    if (sLocal > this.bestS) {
      this.bestS = sLocal;
      this.bestX = observation.position.slice();
    }

    const rawGradient = this.gradientFromObservation(observation, cfg.probeEpsilon);
    this.gradientLpf = [
      this.config.gradientLpfAlpha * rawGradient[0] + (1 - this.config.gradientLpfAlpha) * this.gradientLpf[0],
      this.config.gradientLpfAlpha * rawGradient[1] + (1 - this.config.gradientLpfAlpha) * this.gradientLpf[1],
    ];
    const gradient = this.gradientLpf;
    const gNorm = norm2(gradient);
    const direction = gNorm > this.config.epsilonSafe ? mul2(gradient, 1 / gNorm) : [0, 0];

    let action = [0, 0];

    if (this.phase === "SCAN") {
      if (this.phaseStep >= this.config.scanSteps || sLocal > 0.08 || gNorm > this.config.gMin) {
        this.phase = "SEEK";
        this.phaseStep = 0;
      } else {
        action = this.spiralAction(this.phaseStep, cfg.actionMax);
      }
    }

    if (this.phase === "SEEK") {
      if (sLocal >= this.config.sStop) {
        action = [0, 0];
      } else if (gNorm <= this.config.gMin) {
        this.lostCount += 1;
        action = this.spiralAction(this.phaseStep, cfg.actionMax);
        if (this.lostCount > this.config.kLost) {
          this.phase = "REACQUIRE";
          this.phaseStep = 0;
        }
      } else {
        this.lostCount = 0;
        action = mul2(direction, this.config.seekGain * cfg.actionMax);
      }
      if (sLocal >= this.config.sTrackEnter) this.settleCount += 1;
      else this.settleCount = 0;
      if (this.settleCount >= this.config.kSettle) {
        this.phase = "TRACK";
        this.phaseStep = 0;
      }
    }

    if (this.phase === "TRACK") {
      if (sLocal >= this.config.sStop) {
        action = [0, 0];
      } else {
        const dither = [
          this.config.trackDitherAmplitude * Math.sin(this.config.trackDitherWx * observation.stepIndex),
          this.config.trackDitherAmplitude * Math.sin(this.config.trackDitherWy * observation.stepIndex),
        ];
        action = add2(mul2(direction, this.config.trackGain * cfg.actionMax), dither);
      }
      if (sLocal < this.config.sLost) this.lostCount += 1;
      else this.lostCount = 0;
      if (this.lostCount > this.config.kLost) {
        this.phase = "REACQUIRE";
        this.phaseStep = 0;
      }
    }

    if (this.phase === "REACQUIRE") {
      this.phase = "SCAN";
      this.phaseStep = 0;
      this.bestS = -Infinity;
      this.bestX = null;
      this.lostCount = 0;
      this.settleCount = 0;
      action = this.spiralAction(0, cfg.actionMax);
    }

    this.phaseStep += 1;
    return {
      action: clipVecMagnitude(action, cfg.actionMax),
      phaseLabel: this.phase,
      diagnostic: {
        sLocal,
        rawGradient,
        gradient,
        gradientNorm: gNorm,
        bestS: this.bestS,
      },
    };
  }
}

export class OracleGradientController {
  constructor(config = {}) {
    this.config = Object.freeze({
      ...DEFAULT_ORACLE_CONFIG,
      ...config,
    });
  }

  reset() {
    return this;
  }

  act(observation, envConfig = DEFAULT_MESA_CONFIG) {
    if (observation.sensorTier !== SENSOR_TIERS.PRIVILEGED_FIELD) {
      throw new Error("OracleGradientController requires privileged-field observations");
    }
    const cfg = normalizeMesaConfig(envConfig);
    const gradient = observation.trueGradient.slice();
    const gNorm = norm2(gradient);
    const action =
      observation.trueSignature >= this.config.sStop || gNorm <= 1e-12
        ? [0, 0]
        : clipVecMagnitude(mul2(gradient, (this.config.seekGain * cfg.actionMax) / gNorm), cfg.actionMax);
    return {
      action,
      phaseLabel: "ORACLE",
      diagnostic: {
        sLocal: observation.trueSignature,
        gradient,
        gradientNorm: gNorm,
      },
    };
  }
}

export function makeMesaController(family = "hc_signature", config = {}) {
  if (family === "hc_signature" || family === "HC-Signature") {
    return new HcSignatureController(config);
  }
  if (family === "oracle" || family === "Oracle") {
    return new OracleGradientController(config);
  }
  throw new Error(`Unknown mesa controller family: ${family}`);
}

export function defaultControllerConfig(family = "hc_signature") {
  if (family === "hc_signature" || family === "HC-Signature") return DEFAULT_HC_SIGNATURE_CONFIG;
  if (family === "oracle" || family === "Oracle") return DEFAULT_ORACLE_CONFIG;
  throw new Error(`Unknown mesa controller family: ${family}`);
}

export function defaultTierParams(sensorTier, overrides = {}) {
  const params = {
    sensorTier,
    delaySteps: 0,
    noiseStd: 0,
  };
  if (overrides.delaySteps !== undefined) params.delaySteps = overrides.delaySteps;
  if (overrides.noiseStd !== undefined) params.noiseStd = overrides.noiseStd;
  if (sensorTier === SENSOR_TIERS.DELAYED_FIELD && overrides.delaySteps === undefined) {
    params.delaySteps = 3;
  }
  if (sensorTier === SENSOR_TIERS.NOISY_FIELD && overrides.noiseStd === undefined) {
    params.noiseStd = 0.1;
  }
  if (sensorTier === SENSOR_TIERS.DELAYED_NOISY_FIELD) {
    if (overrides.delaySteps === undefined) params.delaySteps = 3;
    if (overrides.noiseStd === undefined) params.noiseStd = 0.1;
  }
  return params;
}

export function makeTrialConfig({ seed = 1, sensorTier = SENSOR_TIERS.LOCAL_PROBE_FIELD, config = {} } = {}) {
  return normalizeMesaConfig({
    ...config,
    ...defaultTierParams(sensorTier, {
      delaySteps: config.delaySteps,
      noiseStd: config.noiseStd,
    }),
    seed,
  });
}

export function runMesaTrial({
  seed = 1,
  sensorTier = SENSOR_TIERS.LOCAL_PROBE_FIELD,
  controllerFamily = "hc_signature",
  envConfig = {},
  controllerConfig = {},
  trialId = null,
  manifestPath = null,
  logEvery = null,
  probe = null,
  interventions = [],
} = {}) {
  const config = makeTrialConfig({
    seed,
    sensorTier,
    config: {
      ...envConfig,
      logEvery: logEvery ?? envConfig.logEvery ?? DEFAULT_MESA_CONFIG.logEvery,
    },
  });
  const env = new ShadowFieldEnv(config);
  if (probe) env.applyProbe(probe);
  for (const intervention of interventions) env.scheduleIntervention(intervention);
  const mergedControllerConfig = {
    ...defaultControllerConfig(controllerFamily),
    ...controllerConfig,
  };
  const controller = makeMesaController(controllerFamily, mergedControllerConfig);

  const hashPayload = JSON.stringify({
    seed,
    sensorTier,
    controllerFamily,
    envConfig: config,
    controllerConfig: mergedControllerConfig,
    probe: probe ?? {},
    interventions,
  });
  const configHash = hashString(hashPayload).toString(16).padStart(8, "0");
  const id = trialId ?? `${controllerFamily}_${sensorTier}_seed_${String(seed).padStart(4, "0")}`;
  const entries = [
    {
      type: "header",
      seed,
      trialId: id,
      configHash,
      sensorTier,
      controllerFamily,
      x0: roundArray(env.x0),
      xGoal: roundArray(env.xGoal),
      manifestPath,
    },
  ];

  let observation = env.lastObservation;
  while (!env.terminalOutcome) {
    const decision = controller.act(observation, config);
    const result = env.step(decision.action);
    observation = result.observation;
    if ((env.stepIndex - 1) % config.logEvery === 0 || result.done) {
      entries.push({
        type: "step",
        t: env.stepIndex,
        x: roundArray(env.x),
        obs: roundArray(observation.observation),
        a: roundArray(result.action),
        SLocal: roundNumber(observation.sLocal),
        STrue: roundNumber(observation.trueSignature),
        rewards: {
          dense: roundNumber(result.rewardChannels.dense),
          sparse: roundNumber(result.rewardChannels.sparse),
          signature: roundNumber(result.rewardChannels.signature),
        },
        phaseLabel: decision.phaseLabel,
        interventionFlags: result.interventionFlags,
      });
    }
  }

  const metrics = env.metrics();
  entries.push({
    type: "terminal",
    outcome: metrics.terminalOutcome,
    metrics: Object.fromEntries(Object.entries(metrics).map(([key, value]) => [key, roundNumber(value)])),
  });

  return {
    trialId: id,
    seed,
    sensorTier,
    controllerFamily,
    configHash,
    envConfig: config,
    controllerConfig: mergedControllerConfig,
    entries,
    summary: metrics,
  };
}

export function serializeJsonl(entries) {
  return `${entries.map((entry) => JSON.stringify(entry)).join("\n")}\n`;
}
