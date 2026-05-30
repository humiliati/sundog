export const DEFAULT_THREEBODY_CONFIG = Object.freeze({
  G: 1,
  masses: [1, 1, 0.01],
  separation: 1.5,
  dt: 0.01,
  duration: 20,
  controllerMode: "off",
  thrustLimit: 0.5,
  targetTidal: 2,
  tidalProbeDelta: 0.05,
  targetBandFraction: 0.25,
  closeApproachRadius: 0.08,
  escapeRadius: 4,
  tidalSpikeThreshold: 50,
  tidalNoiseStd: 0.15,
  tidalShufflePeriod: 25,
  eventWarningHorizon: 1,
  localAccelerationWarningThreshold: 10,
  sensorAuditVariants: [],
  sensorAuditEvery: 20,
  sensorNoiseStd: 0.03,
  sensorDelaySteps: 5,
  microManeuverContaminationStd: 0.08,
  trackGuardMinRadius: 1.15,
  trackGuardMaxLocalAcceleration: 2.5,
  trackGuardMaxTidalMagnitude: 35,
  radiusInwardMagnitude: 0.4,
  precisionReceipts: false,
  counterfactualAudit: false,
  multiStepAudit: false,
  hazardChannelAudit: false,
  hazardCounterfactualAudit: false,
  counterfactualNormalizerFloor: 1e-9,
  logEvery: 10,
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

export function roundNumber(value, digits = 8) {
  if (!Number.isFinite(value)) return value;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

export function roundArray(values, digits = 8) {
  return values.map((value) => roundNumber(value, digits));
}

function normalSample(rng) {
  const u1 = Math.max(rng(), 1e-9);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function modeHash(mode = "off") {
  let hash = 2166136261;
  for (let i = 0; i < mode.length; i += 1) {
    hash ^= mode.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export function normalizeConfig(config = {}) {
  const merged = {
    ...DEFAULT_THREEBODY_CONFIG,
    ...config,
  };
  const massRatio = config.massRatio ?? merged.masses[1] / merged.masses[0];
  const m1 = config.m1 ?? merged.masses[0];
  const m2 = config.m2 ?? m1 * massRatio;
  const m3 = config.m3 ?? merged.masses[2];
  return {
    ...merged,
    masses: [m1, m2, m3],
    massRatio,
  };
}

export function seededInitialParticle(seed, regime = "stable") {
  const rng = makeRng(seed * 1000003 + 17);
  const jitter = (span) => (rng() - 0.5) * span;

  if (regime === "near_escape") {
    return {
      x: 1.45 + jitter(0.3),
      y: 0.15 + jitter(0.3),
      vx: 0.2 + jitter(0.3),
      vy: 1.45 + jitter(0.3),
    };
  }

  if (regime === "near_collision") {
    return {
      x: 0.72 + jitter(0.12),
      y: 0.06 + jitter(0.12),
      vx: -0.1 + jitter(0.3),
      vy: 0.75 + jitter(0.25),
    };
  }

  if (regime === "chaotic") {
    return {
      x: jitter(0.35),
      y: jitter(0.35),
      vx: jitter(1.2),
      vy: 0.7 + jitter(1.1),
    };
  }

  return {
    x: 1 + jitter(0.3),
    y: jitter(0.25),
    vx: jitter(0.3),
    vy: 1 + jitter(0.25),
  };
}

export function initializeState(config = {}) {
  const cfg = normalizeConfig(config);
  const particle = config.initialParticle ?? seededInitialParticle(cfg.seed ?? 0, cfg.regime);
  const [m1, m2] = cfg.masses;
  const totalMass = m1 + m2;
  const x1 = -m2 * cfg.separation / totalMass;
  const x2 = m1 * cfg.separation / totalMass;
  const omega = Math.sqrt(cfg.G * totalMass / cfg.separation ** 3);
  const v1 = omega * Math.abs(x1);
  const v2 = omega * Math.abs(x2);

  return [
    x1, 0,
    x2, 0,
    particle.x, particle.y,
    0, v1,
    0, -v2,
    particle.vx, particle.vy,
  ];
}

export function computeAcceleration(index, positions, config = {}, thrust = [0, 0]) {
  const cfg = normalizeConfig(config);
  let ax = 0;
  let ay = 0;
  const xi = positions[index * 2];
  const yi = positions[index * 2 + 1];

  for (let j = 0; j < 3; j += 1) {
    if (index === j) continue;
    if (index < 2 && j === 2) continue;

    const xj = positions[j * 2];
    const yj = positions[j * 2 + 1];
    const dx = xj - xi;
    const dy = yj - yi;
    const r2 = dx * dx + dy * dy;
    const r = Math.sqrt(r2);
    if (r < 0.01) continue;

    const force = cfg.G * cfg.masses[j] / (r2 * r);
    ax += force * dx;
    ay += force * dy;
  }

  if (index === 2) {
    ax += thrust[0];
    ay += thrust[1];
  }

  return [ax, ay];
}

export function integrateStep(state, dt, config = {}, thrust = [0, 0]) {
  const positions = state.slice(0, 6);
  const velocities = state.slice(6, 12);

  const accelerationsFor = (pos) => {
    const values = [];
    for (let i = 0; i < 3; i += 1) {
      values.push(...computeAcceleration(i, pos, config, thrust));
    }
    return values;
  };

  const k1v = accelerationsFor(positions);
  const k1x = velocities.slice();

  const pos2 = positions.map((p, i) => p + 0.5 * dt * k1x[i]);
  const vel2 = velocities.map((v, i) => v + 0.5 * dt * k1v[i]);
  const k2v = accelerationsFor(pos2);
  const k2x = vel2.slice();

  const pos3 = positions.map((p, i) => p + 0.5 * dt * k2x[i]);
  const vel3 = velocities.map((v, i) => v + 0.5 * dt * k2v[i]);
  const k3v = accelerationsFor(pos3);
  const k3x = vel3.slice();

  const pos4 = positions.map((p, i) => p + dt * k3x[i]);
  const vel4 = velocities.map((v, i) => v + dt * k3v[i]);
  const k4v = accelerationsFor(pos4);
  const k4x = vel4.slice();

  return [
    ...positions.map((p, i) => p + (dt / 6) * (k1x[i] + 2 * k2x[i] + 2 * k3x[i] + k4x[i])),
    ...velocities.map((v, i) => v + (dt / 6) * (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i])),
  ];
}

export function computeSignatures(state, config = {}) {
  const cfg = normalizeConfig(config);
  const positions = state.slice(0, 6);
  const velocities = state.slice(6, 12);
  let kineticEnergy = 0;
  for (let i = 0; i < 3; i += 1) {
    const vx = velocities[i * 2];
    const vy = velocities[i * 2 + 1];
    kineticEnergy += 0.5 * cfg.masses[i] * (vx * vx + vy * vy);
  }

  let potentialEnergy = 0;
  for (let i = 0; i < 3; i += 1) {
    for (let j = i + 1; j < 3; j += 1) {
      const dx = positions[j * 2] - positions[i * 2];
      const dy = positions[j * 2 + 1] - positions[i * 2 + 1];
      const r = Math.sqrt(dx * dx + dy * dy);
      if (r > 0.01) potentialEnergy -= cfg.G * cfg.masses[i] * cfg.masses[j] / r;
    }
  }

  const totalMass = cfg.masses.reduce((a, b) => a + b, 0);
  let cmx = 0;
  let cmy = 0;
  for (let i = 0; i < 3; i += 1) {
    cmx += cfg.masses[i] * positions[i * 2];
    cmy += cfg.masses[i] * positions[i * 2 + 1];
  }
  cmx /= totalMass;
  cmy /= totalMass;

  let inertiaTrace = 0;
  for (let i = 0; i < 3; i += 1) {
    const dx = positions[i * 2] - cmx;
    const dy = positions[i * 2 + 1] - cmy;
    inertiaTrace += cfg.masses[i] * (dx * dx + dy * dy);
  }

  return {
    virial: Math.abs(potentialEnergy) > 0.001 ? 2 * kineticEnergy / Math.abs(potentialEnergy) : 0,
    inertia: inertiaTrace,
    energy: kineticEnergy + potentialEnergy,
    kineticEnergy,
    potentialEnergy,
  };
}

export function computeTidalTensor(state, config = {}) {
  const cfg = normalizeConfig(config);
  const positions = state.slice(0, 6);
  const x3 = positions[4];
  const y3 = positions[5];
  const delta = cfg.tidalProbeDelta;
  const [ax, ay] = computeAcceleration(2, positions, cfg);

  const positionsXp = positions.slice();
  positionsXp[4] = x3 + delta;
  const [axpx, axpy] = computeAcceleration(2, positionsXp, cfg);

  const positionsYp = positions.slice();
  positionsYp[5] = y3 + delta;
  const [aypx, aypy] = computeAcceleration(2, positionsYp, cfg);

  const T_xx = (axpx - ax) / delta;
  const T_yx = (axpy - ay) / delta;
  const T_xy = (aypx - ax) / delta;
  const T_yy = (aypy - ay) / delta;
  const magnitude = Math.sqrt(T_xx * T_xx + T_xy * T_xy + T_yx * T_yx + T_yy * T_yy);

  return { magnitude, T_xx, T_xy, T_yx, T_yy };
}

function tensorFromAccelerationSamples(center, xProbe, yProbe, delta) {
  const T_xx = (xProbe[0] - center[0]) / delta;
  const T_yx = (xProbe[1] - center[1]) / delta;
  const T_xy = (yProbe[0] - center[0]) / delta;
  const T_yy = (yProbe[1] - center[1]) / delta;
  const magnitude = Math.sqrt(T_xx * T_xx + T_xy * T_xy + T_yx * T_yx + T_yy * T_yy);
  return { magnitude, T_xx, T_xy, T_yx, T_yy };
}

function addAccelerationNoise(acceleration, rng, noiseStd) {
  if (noiseStd <= 0) return acceleration;
  const scale = Math.max(1, Math.sqrt(acceleration[0] * acceleration[0] + acceleration[1] * acceleration[1]));
  return [
    acceleration[0] + scale * noiseStd * normalSample(rng),
    acceleration[1] + scale * noiseStd * normalSample(rng),
  ];
}

function delayedEstimate(sensorState, estimate, delaySteps) {
  if (delaySteps <= 0) return { ...estimate, delayWarmup: false };
  if (!sensorState.buffer) sensorState.buffer = [];
  sensorState.buffer.push(estimate);
  if (sensorState.buffer.length <= delaySteps) return { ...estimate, delayWarmup: true };
  return { ...sensorState.buffer.shift(), delayWarmup: false };
}

function sensorVariantSettings(variant, config) {
  if (variant === "accelerometer_array_noisy") {
    return {
      delta: config.tidalProbeDelta,
      noiseStd: config.sensorNoiseStd,
      delaySteps: 0,
      sensorTier: "accelerometer_array",
    };
  }
  if (variant === "delayed_local_probe") {
    return {
      delta: config.tidalProbeDelta,
      noiseStd: 0,
      delaySteps: config.sensorDelaySteps,
      sensorTier: "delayed_simulated_probe",
    };
  }
  if (variant === "micro_maneuver_noisy") {
    return {
      delta: config.tidalProbeDelta * 2,
      noiseStd: config.sensorNoiseStd + config.microManeuverContaminationStd,
      delaySteps: config.sensorDelaySteps,
      sensorTier: "micro_maneuver_proxy",
    };
  }
  return {
    delta: config.tidalProbeDelta,
    noiseStd: 0,
    delaySteps: 0,
    sensorTier: "simulated_local_probe",
  };
}

export function computeSensorTidalTensor(state, config = {}, variant = "simulated_local_probe", sensorState = {}) {
  const cfg = normalizeConfig(config);
  const settings = sensorVariantSettings(variant, cfg);
  const positions = state.slice(0, 6);
  const x3 = positions[4];
  const y3 = positions[5];
  const rng = controllerRng(sensorState, { ...cfg, controllerMode: `sensor_${variant}` });

  const positionsXp = positions.slice();
  positionsXp[4] = x3 + settings.delta;
  const positionsYp = positions.slice();
  positionsYp[5] = y3 + settings.delta;

  const center = addAccelerationNoise(computeAcceleration(2, positions, cfg), rng, settings.noiseStd);
  const xProbe = addAccelerationNoise(computeAcceleration(2, positionsXp, cfg), rng, settings.noiseStd);
  const yProbe = addAccelerationNoise(computeAcceleration(2, positionsYp, cfg), rng, settings.noiseStd);
  const estimate = {
    ...tensorFromAccelerationSamples(center, xProbe, yProbe, settings.delta),
    sensorVariant: variant,
    sensorTier: settings.sensorTier,
    noiseStd: settings.noiseStd,
    delaySteps: settings.delaySteps,
    probeDelta: settings.delta,
  };

  return delayedEstimate(sensorState, estimate, settings.delaySteps);
}

export function computeTidalGradient(state, config = {}) {
  const cfg = normalizeConfig(config);
  const positions = state.slice(0, 6);
  const x3 = positions[4];
  const y3 = positions[5];
  const delta = cfg.tidalProbeDelta;
  const tidal = computeTidalTensor(state, cfg);

  const positionsXp = positions.slice();
  positionsXp[4] = x3 + delta;
  const tidalXp = computeTidalTensor([...positionsXp, ...state.slice(6)], cfg);

  const positionsYp = positions.slice();
  positionsYp[5] = y3 + delta;
  const tidalYp = computeTidalTensor([...positionsYp, ...state.slice(6)], cfg);

  return {
    tidal,
    gradX: (tidalXp.magnitude - tidal.magnitude) / delta,
    gradY: (tidalYp.magnitude - tidal.magnitude) / delta,
  };
}

function controllerSensorVariant(mode) {
  if (mode.endsWith("_sensor_accel")) return "accelerometer_array_noisy";
  if (mode === "track_sensor_accel_guarded") return "accelerometer_array_noisy";
  if (mode === "track_radius_guard") return "accelerometer_array_noisy";
  if (PHASE14_ABLATION_MODES.has(mode)) return "accelerometer_array_noisy";
  if (mode.endsWith("_sensor_delayed")) return "delayed_local_probe";
  if (mode.endsWith("_sensor_micro")) return "micro_maneuver_noisy";
  return null;
}

function shouldRunGuardedTrack(state, tidal, config) {
  const positions = state.slice(0, 6);
  const x3 = positions[4];
  const y3 = positions[5];
  const radius = Math.sqrt(x3 * x3 + y3 * y3);
  const [ax, ay] = computeAcceleration(2, positions, config);
  const localAccelerationMagnitude = Math.sqrt(ax * ax + ay * ay);
  return (
    radius >= config.trackGuardMinRadius
    && localAccelerationMagnitude <= config.trackGuardMaxLocalAcceleration
    && tidal.magnitude <= config.trackGuardMaxTidalMagnitude
  );
}

function computeSensorTidalGradient(state, config, variant, controllerState) {
  const cfg = normalizeConfig(config);
  const positions = state.slice(0, 6);
  const x3 = positions[4];
  const y3 = positions[5];
  const delta = cfg.tidalProbeDelta;
  if (!controllerState.sensorGradientStates) controllerState.sensorGradientStates = {};
  if (!controllerState.sensorGradientStates[variant]) {
    controllerState.sensorGradientStates[variant] = {
      center: {},
      xp: {},
      yp: {},
    };
  }
  const sensorStates = controllerState.sensorGradientStates[variant];
  const tidal = computeSensorTidalTensor(state, cfg, variant, sensorStates.center);

  const positionsXp = positions.slice();
  positionsXp[4] = x3 + delta;
  const tidalXp = computeSensorTidalTensor([...positionsXp, ...state.slice(6)], cfg, variant, sensorStates.xp);

  const positionsYp = positions.slice();
  positionsYp[5] = y3 + delta;
  const tidalYp = computeSensorTidalTensor([...positionsYp, ...state.slice(6)], cfg, variant, sensorStates.yp);

  return {
    tidal,
    gradX: (tidalXp.magnitude - tidal.magnitude) / delta,
    gradY: (tidalYp.magnitude - tidal.magnitude) / delta,
    sensorVariant: variant,
    sensorTier: tidal.sensorTier,
    delayWarmup: tidal.delayWarmup || tidalXp.delayWarmup || tidalYp.delayWarmup,
  };
}

function observeGuardedAccelSignatureDetails(state, config, controllerState = {}) {
  const cfg = normalizeConfig(config);
  const gradient = computeSensorTidalGradient(
    state,
    cfg,
    "accelerometer_array_noisy",
    controllerState,
  );
  const guard = shouldRunGuardedTrack(state, gradient.tidal, cfg);
  const signature = {
    tidalMagnitude: gradient.tidal.magnitude,
    absTidalMagnitude: gradient.tidal.magnitude,
    gradX: gradient.gradX,
    gradY: gradient.gradY,
    guard,
    sensorVariant: "accelerometer_array_noisy",
    sensorTier: gradient.sensorTier,
    sensorNoiseStd: cfg.sensorNoiseStd,
    probeDelta: gradient.tidal.probeDelta ?? cfg.tidalProbeDelta,
    delayWarmup: Boolean(gradient.delayWarmup),
    seed: cfg.seed ?? null,
  };
  return { signature, gradient };
}

export function observeGuardedAccelSignature(state, config = {}, controllerState = {}) {
  return observeGuardedAccelSignatureDetails(state, config, controllerState).signature;
}

function limitVector(x, y, maxMagnitude) {
  const magnitude = Math.sqrt(x * x + y * y);
  if (magnitude <= 0.001) return [0, 0];
  if (magnitude <= maxMagnitude) return [x, y];
  return [maxMagnitude * x / magnitude, maxMagnitude * y / magnitude];
}

function controllerRng(controllerState, config) {
  if (!controllerState.rng) {
    controllerState.rng = makeRng(((config.seed ?? 0) * 1009 + modeHash(config.controllerMode)) >>> 0);
  }
  return controllerState.rng;
}

const PHASE14_ABLATION_MODES = new Set([
  "track_sensor_accel_signal_shuffle",
  "track_sensor_accel_action_shuffle",
  "track_sensor_accel_signal_delay",
  "track_sensor_accel_sign_flip",
]);

export const KNOWN_CONTROLLER_MODES = new Set([
  "off",
  "scan",
  "naive",
  "oracle",
  "forward_oracle_strict",
  "seek",
  "track",
  "seek_noisy",
  "track_noisy",
  "seek_shuffled",
  "track_shuffled",
  "seek_sensor_accel",
  "track_sensor_accel",
  "track_sensor_accel_guarded",
  "seek_sensor_delayed",
  "track_sensor_delayed",
  "seek_sensor_micro",
  "track_sensor_micro",
  "track_radius_guard",
  "track_radius_inward",
  ...PHASE14_ABLATION_MODES,
]);

function phase14PlannedSteps(config) {
  return Math.max(1, Math.round(config.duration / config.dt)) + 1;
}

function phase14Permutation(controllerState, config, kind) {
  const cacheKey = `${kind}Perm`;
  if (!controllerState[cacheKey]) {
    const n = phase14PlannedSteps(config);
    const perm = Array.from({ length: n }, (_, i) => i);
    const rng = controllerRng(controllerState, config);
    for (let i = n - 1; i > 0; i -= 1) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = perm[i];
      perm[i] = perm[j];
      perm[j] = tmp;
    }
    controllerState[cacheKey] = perm;
  }
  return controllerState[cacheKey];
}

function phase14SignalAblation(mode, gradient, controllerState, config) {
  if (mode === "track_sensor_accel_signal_delay") {
    const delaySteps = Math.max(0, Math.round(0.5 / config.dt));
    if (!controllerState.signalDelayBuffer) controllerState.signalDelayBuffer = [];
    controllerState.signalDelayBuffer.push(gradient);
    if (controllerState.signalDelayBuffer.length <= delaySteps) return { suppressStep: true };
    return { gradient: controllerState.signalDelayBuffer.shift() };
  }
  if (mode === "track_sensor_accel_signal_shuffle") {
    const t = (controllerState.step ?? 1) - 1;
    if (!controllerState.signalHistory) controllerState.signalHistory = [];
    controllerState.signalHistory[t] = {
      tidal: gradient.tidal,
      gradX: gradient.gradX,
      gradY: gradient.gradY,
    };
    const perm = phase14Permutation(controllerState, config, "signal");
    const target = perm[t] ?? t;
    if (target <= t && controllerState.signalHistory[target]) {
      return { gradient: controllerState.signalHistory[target] };
    }
    return { suppressStep: true };
  }
  return {};
}

function phase14ActionAblation(mode, thrust, controllerState, config) {
  if (mode === "track_sensor_accel_sign_flip") return [-thrust[0], -thrust[1]];
  if (mode === "track_sensor_accel_action_shuffle") {
    const t = (controllerState.step ?? 1) - 1;
    if (!controllerState.actionHistory) controllerState.actionHistory = [];
    controllerState.actionHistory[t] = thrust;
    const perm = phase14Permutation(controllerState, config, "action");
    const target = perm[t] ?? t;
    if (target <= t && controllerState.actionHistory[target]) {
      return controllerState.actionHistory[target];
    }
    controllerState.stepWarmup = true;
    return [0, 0];
  }
  return thrust;
}

function computeNaiveLocalThrust(state, config) {
  const positions = state.slice(0, 6);
  const velocities = state.slice(6, 12);
  const [ax, ay] = computeAcceleration(2, positions, config);
  const accelMagnitude = Math.sqrt(ax * ax + ay * ay);
  if (accelMagnitude <= 0.001) return [0, 0];

  const vx3 = velocities[4];
  const vy3 = velocities[5];
  const awayFromAccelerationX = -ax / accelMagnitude;
  const awayFromAccelerationY = -ay / accelMagnitude;
  const dampingX = -0.15 * vx3;
  const dampingY = -0.15 * vy3;

  return limitVector(
    config.thrustLimit * awayFromAccelerationX + dampingX,
    config.thrustLimit * awayFromAccelerationY + dampingY,
    config.thrustLimit,
  );
}

function perturbTidalGradient(gradient, controllerState, config) {
  const rng = controllerRng(controllerState, config);
  const noiseScale = config.tidalNoiseStd;
  return {
    ...gradient,
    tidal: {
      ...gradient.tidal,
      magnitude: Math.max(0, gradient.tidal.magnitude * (1 + noiseScale * normalSample(rng))),
    },
    gradX: gradient.gradX * (1 + noiseScale * normalSample(rng)),
    gradY: gradient.gradY * (1 + noiseScale * normalSample(rng)),
  };
}

function shuffledTidalGradient(controllerState, config) {
  const rng = controllerRng(controllerState, config);
  const period = Math.max(1, config.tidalShufflePeriod);
  const step = controllerState.step ?? 0;
  if (!controllerState.shuffledGradient || step % period === 0) {
    const angle = rng() * 2 * Math.PI;
    controllerState.shuffledGradient = {
      gradX: Math.cos(angle),
      gradY: Math.sin(angle),
    };
  }
  return controllerState.shuffledGradient;
}

function computeOracleThrust(state, config) {
  const candidateDirections = [
    [0, 0],
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
    [Math.SQRT1_2, Math.SQRT1_2],
    [Math.SQRT1_2, -Math.SQRT1_2],
    [-Math.SQRT1_2, Math.SQRT1_2],
    [-Math.SQRT1_2, -Math.SQRT1_2],
  ];
  const candidates = candidateDirections.map(([x, y]) => [
    x * config.thrustLimit,
    y * config.thrustLimit,
  ]);

  const scoreState = (simState, thrustCost) => {
    const positions = simState.slice(0, 6);
    const velocities = simState.slice(6, 12);
    const x3 = positions[4];
    const y3 = positions[5];
    const vx3 = velocities[4];
    const vy3 = velocities[5];
    const radius = Math.sqrt(x3 * x3 + y3 * y3);
    const speed = Math.sqrt(vx3 * vx3 + vy3 * vy3);
    let minPrimaryDistance = Infinity;
    for (let i = 0; i < 2; i += 1) {
      const dx = x3 - positions[i * 2];
      const dy = y3 - positions[i * 2 + 1];
      minPrimaryDistance = Math.min(minPrimaryDistance, Math.sqrt(dx * dx + dy * dy));
    }
    const tidal = computeTidalTensor(simState, config);
    const targetError = Math.abs(tidal.magnitude - config.targetTidal);
    const closePenalty = minPrimaryDistance < config.closeApproachRadius
      ? 1_000
      : 1 / Math.max(minPrimaryDistance, 0.04) ** 2;
    const escapePenalty = radius > config.escapeRadius ? 1_000 : Math.max(0, radius - 1.8) ** 2;
    return (
      8 * closePenalty
      + 3 * escapePenalty
      + 0.12 * speed * speed
      + 0.002 * targetError
      + 0.05 * thrustCost
    );
  };

  let bestThrust = [0, 0];
  let bestScore = Infinity;
  for (const thrust of candidates) {
    let simState = state;
    let score = 0;
    const horizonSteps = 16;
    for (let step = 0; step < horizonSteps; step += 1) {
      simState = integrateStep(simState, config.dt, config, thrust);
      const thrustCost = Math.sqrt(thrust[0] * thrust[0] + thrust[1] * thrust[1]);
      score += scoreState(simState, thrustCost) * (1 + step / horizonSteps);
    }
    if (score < bestScore) {
      bestScore = score;
      bestThrust = thrust;
    }
  }

  return bestThrust;
}

function oracleScoreState(simState, thrustCost, config) {
  const positions = simState.slice(0, 6);
  const velocities = simState.slice(6, 12);
  const x3 = positions[4];
  const y3 = positions[5];
  const vx3 = velocities[4];
  const vy3 = velocities[5];
  const radius = Math.sqrt(x3 * x3 + y3 * y3);
  const speed = Math.sqrt(vx3 * vx3 + vy3 * vy3);
  let minPrimaryDistance = Infinity;
  for (let i = 0; i < 2; i += 1) {
    const dx = x3 - positions[i * 2];
    const dy = y3 - positions[i * 2 + 1];
    minPrimaryDistance = Math.min(minPrimaryDistance, Math.sqrt(dx * dx + dy * dy));
  }
  const tidal = computeTidalTensor(simState, config);
  const targetError = Math.abs(tidal.magnitude - config.targetTidal);
  const closePenalty = minPrimaryDistance < config.closeApproachRadius
    ? 1_000
    : 1 / Math.max(minPrimaryDistance, 0.04) ** 2;
  const escapePenalty = radius > config.escapeRadius ? 1_000 : Math.max(0, radius - 1.8) ** 2;
  return (
    8 * closePenalty
    + 3 * escapePenalty
    + 0.12 * speed * speed
    + 0.002 * targetError
    + 0.05 * thrustCost
  );
}

export function oracleCandidateThrusts(config) {
  return [
    [0, 0],
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
    [Math.SQRT1_2, Math.SQRT1_2],
    [Math.SQRT1_2, -Math.SQRT1_2],
    [-Math.SQRT1_2, Math.SQRT1_2],
    [-Math.SQRT1_2, -Math.SQRT1_2],
  ].map(([x, y]) => [
    x * config.thrustLimit,
    y * config.thrustLimit,
  ]);
}

function stateHasTerminalHazard(state, config) {
  const positions = state.slice(0, 6);
  const x3 = positions[4];
  const y3 = positions[5];
  const r3 = Math.sqrt(x3 * x3 + y3 * y3);
  let minPrimaryDistance = Infinity;
  for (let i = 0; i < 2; i += 1) {
    const dx = x3 - positions[i * 2];
    const dy = y3 - positions[i * 2 + 1];
    minPrimaryDistance = Math.min(minPrimaryDistance, Math.sqrt(dx * dx + dy * dy));
  }
  return r3 > config.escapeRadius || minPrimaryDistance < config.closeApproachRadius;
}

function computeStrictOracleDetails(state, config) {
  const cfg = normalizeConfig(config);
  const candidates = oracleCandidateThrusts(cfg);
  const horizonSteps = 32;
  const substeps = 8;
  const subDt = cfg.dt / substeps;

  let bestThrust = [0, 0];
  let bestScore = Infinity;
  let bestHazardReached = false;

  for (const thrust of candidates) {
    let simState = state;
    let score = 0;
    let hazardReached = false;
    const thrustCost = Math.sqrt(thrust[0] * thrust[0] + thrust[1] * thrust[1]);
    for (let step = 0; step < horizonSteps; step += 1) {
      for (let substep = 0; substep < substeps; substep += 1) {
        simState = integrateStep(simState, subDt, cfg, thrust);
        if (stateHasTerminalHazard(simState, cfg)) hazardReached = true;
      }
      score += oracleScoreState(simState, thrustCost, cfg) * (1 + step / horizonSteps);
    }
    if (score < bestScore) {
      bestScore = score;
      bestThrust = thrust;
      bestHazardReached = hazardReached;
    }
  }

  return {
    thrust: bestThrust,
    score: bestScore,
    hazardReached: bestHazardReached,
  };
}

function computeStrictOracleThrust(state, config) {
  return computeStrictOracleDetails(state, config).thrust;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function computeCounterfactualReceipt(state, thrust, config, oracleStrictThrust) {
  const noopState = integrateStep(state, config.dt, config, [0, 0]);
  const actualState = integrateStep(state, config.dt, config, thrust);
  const oracleState = integrateStep(state, config.dt, config, oracleStrictThrust);
  const noopEnergy = computeSignatures(noopState, config).energy;
  const actualEnergy = computeSignatures(actualState, config).energy;
  const oracleEnergy = computeSignatures(oracleState, config).energy;
  const effectVsNoop = noopEnergy - actualEnergy;
  const rawNormalizer = Math.abs(noopEnergy - oracleEnergy);
  const floor = config.counterfactualNormalizerFloor ?? 1e-9;
  const normalizer = Math.max(rawNormalizer, floor);
  return {
    score: clamp(effectVsNoop / normalizer, -1, 1),
    gapToOracle: actualEnergy - oracleEnergy,
    effectVsNoop,
    absEffectVsNoop: Math.abs(effectVsNoop),
    rawNormalizer,
    normalizer,
    normalizerFloored: rawNormalizer < floor,
  };
}

const MS_AUDIT_HORIZONS = [4, 8, 16, 32];

function computeMultiStepCounterfactualHorizon(state, thrust, oracleStrictThrust, cfg) {
  // Three-arm N-step rollout per eligible step (Phase 15C).
  // actual: thrust at step 1, then thrust for steps 2..N
  // noop:   [0,0] at step 1, then thrust for steps 2..N
  // oracle: oracleStrictThrust at step 1, then thrust for steps 2..N
  const floor = cfg.counterfactualNormalizerFloor ?? 1e-9;
  const hSet = new Set(MS_AUDIT_HORIZONS);
  const actualEnergies = {};
  const noopEnergies = {};
  const oracleEnergies = {};
  let actualState = state;
  let noopState = state;
  let oracleState = state;
  for (let i = 1; i <= 32; i++) {
    const noopThrust = i === 1 ? [0, 0] : thrust;
    const oThrust = i === 1 ? oracleStrictThrust : thrust;
    actualState = integrateStep(actualState, cfg.dt, cfg, thrust);
    noopState   = integrateStep(noopState,   cfg.dt, cfg, noopThrust);
    oracleState = integrateStep(oracleState, cfg.dt, cfg, oThrust);
    if (hSet.has(i)) {
      actualEnergies[i] = computeSignatures(actualState, cfg).energy;
      noopEnergies[i]   = computeSignatures(noopState,   cfg).energy;
      oracleEnergies[i] = computeSignatures(oracleState, cfg).energy;
    }
  }
  const results = {};
  for (const N of MS_AUDIT_HORIZONS) {
    const rawDiff = noopEnergies[N] - actualEnergies[N]; // positive = actual reduces energy vs noop
    const rawNorm = Math.abs(noopEnergies[N] - oracleEnergies[N]);
    const normalizer = Math.max(rawNorm, floor);
    const score = clamp(rawDiff / normalizer, -1, 1);
    const floored = rawNorm < floor;
    results[N] = { rawDiff, rawNorm, score, positive: rawDiff > 0, floored };
  }
  return results;
}

const HAZARD_CF_HORIZONS = [1, 4, 8, 16, 32];

// Phase 17: signed geometric distance to the frozen hazard boundary. Reuses the
// exact geometry of stateHasTerminalHazard (r3 > escapeRadius OR minPrimaryDist
// < closeApproachRadius); hazardMargin <= 0 iff the state is on/beyond a boundary.
function hazardMargins(state, config) {
  const x3 = state[4];
  const y3 = state[5];
  const radius = Math.sqrt(x3 * x3 + y3 * y3);
  let minPrimaryDistance = Infinity;
  for (let i = 0; i < 2; i += 1) {
    const dx = x3 - state[i * 2];
    const dy = y3 - state[i * 2 + 1];
    minPrimaryDistance = Math.min(minPrimaryDistance, Math.sqrt(dx * dx + dy * dy));
  }
  const escapeMargin = config.escapeRadius - radius;
  const closeMargin = minPrimaryDistance - config.closeApproachRadius;
  return { escapeMargin, closeMargin, hazardMargin: Math.min(escapeMargin, closeMargin) };
}

// Phase 17: matched actual-vs-noop rollout (drops the 15C oracle arm + normalizer).
// actual applies `thrust` every step; noop applies [0,0] at step 1 then the same
// `thrust` for steps 2..N. Terminal score is the raw hazard-margin difference.
function computeHazardCounterfactualHorizon(state, thrust, cfg) {
  const hSet = new Set(HAZARD_CF_HORIZONS);
  const maxH = HAZARD_CF_HORIZONS[HAZARD_CF_HORIZONS.length - 1];
  let actualState = state;
  let noopState = state;
  let actualHazardReached = false;
  let noopHazardReached = false;
  const results = {};
  for (let i = 1; i <= maxH; i += 1) {
    actualState = integrateStep(actualState, cfg.dt, cfg, thrust);
    noopState = integrateStep(noopState, cfg.dt, cfg, i === 1 ? [0, 0] : thrust);
    if (stateHasTerminalHazard(actualState, cfg)) actualHazardReached = true;
    if (stateHasTerminalHazard(noopState, cfg)) noopHazardReached = true;
    if (hSet.has(i)) {
      const a = hazardMargins(actualState, cfg);
      const n = hazardMargins(noopState, cfg);
      const hazardAvoided = noopHazardReached && !actualHazardReached ? 1
        : actualHazardReached && !noopHazardReached ? -1
          : 0;
      results[i] = {
        marginEffect: a.hazardMargin - n.hazardMargin,
        escapeMarginEffect: a.escapeMargin - n.escapeMargin,
        closeMarginEffect: a.closeMargin - n.closeMargin,
        hazardAvoided,
      };
    }
  }
  return results;
}

export function computeControlThrust(state, controllerState = {}, config = {}) {
  const cfg = normalizeConfig(config);
  const mode = cfg.controllerMode;
  if (!KNOWN_CONTROLLER_MODES.has(mode)) {
    throw new Error(`unknown controllerMode: ${mode}`);
  }
  controllerState.step = (controllerState.step ?? 0) + 1;
  controllerState.stepWarmup = false;
  if (mode === "off") return [0, 0];

  let thrustX = 0;
  let thrustY = 0;

  if (mode === "scan") {
    const scanFreq = 0.5;
    const scanAmplitude = cfg.thrustLimit * 0.5;
    controllerState.scanPhase = (controllerState.scanPhase ?? 0) + cfg.dt * scanFreq * 2 * Math.PI;
    thrustX = scanAmplitude * Math.cos(controllerState.scanPhase);
    thrustY = scanAmplitude * Math.sin(controllerState.scanPhase * 1.3);
  } else if (mode === "naive") {
    [thrustX, thrustY] = computeNaiveLocalThrust(state, cfg);
  } else if (mode === "oracle") {
    [thrustX, thrustY] = computeOracleThrust(state, cfg);
  } else if (mode === "forward_oracle_strict") {
    [thrustX, thrustY] = computeStrictOracleThrust(state, cfg);
  } else if (
    mode === "seek"
    || mode === "track"
    || mode === "seek_noisy"
    || mode === "track_noisy"
    || mode === "seek_shuffled"
    || mode === "track_shuffled"
    || mode === "seek_sensor_accel"
    || mode === "track_sensor_accel"
    || mode === "track_sensor_accel_guarded"
    || mode === "track_radius_guard"
    || mode === "seek_sensor_delayed"
    || mode === "track_sensor_delayed"
    || mode === "seek_sensor_micro"
    || mode === "track_sensor_micro"
    || PHASE14_ABLATION_MODES.has(mode)
  ) {
    const sensorVariant = controllerSensorVariant(mode);
    let guardedSignature = null;
    let gradient = null;
    if (
      sensorVariant === "accelerometer_array_noisy"
      && (mode === "track_sensor_accel_guarded" || mode === "track_radius_guard" || PHASE14_ABLATION_MODES.has(mode))
    ) {
      const observed = observeGuardedAccelSignatureDetails(state, cfg, controllerState);
      guardedSignature = observed.signature;
      gradient = observed.gradient;
    } else {
      gradient = sensorVariant
        ? computeSensorTidalGradient(state, cfg, sensorVariant, controllerState)
        : computeTidalGradient(state, cfg);
    }
    if (!sensorVariant && mode.endsWith("_noisy")) {
      gradient = perturbTidalGradient(gradient, controllerState, cfg);
    } else if (!sensorVariant && mode.endsWith("_shuffled")) {
      const shuffled = shuffledTidalGradient(controllerState, cfg);
      gradient = {
        ...gradient,
        gradX: shuffled.gradX,
        gradY: shuffled.gradY,
      };
    }
    if (PHASE14_ABLATION_MODES.has(mode)) {
      const ablated = phase14SignalAblation(mode, gradient, controllerState, cfg);
      if (ablated.suppressStep) {
        controllerState.stepWarmup = true;
        return [0, 0];
      }
      if (ablated.gradient) gradient = ablated.gradient;
    }
    const { tidal, gradX, gradY } = gradient;
    if (mode === "track_sensor_accel_guarded" && !guardedSignature.guard) {
      return [0, 0];
    }
    if (mode === "track_radius_guard") {
      const guardX = state[4];
      const guardY = state[5];
      if (Math.sqrt(guardX * guardX + guardY * guardY) < cfg.trackGuardMinRadius) {
        return [0, 0];
      }
    }
    let guardSuppressed = false;
    if (PHASE14_ABLATION_MODES.has(mode) && !shouldRunGuardedTrack(state, tidal, cfg)) {
      guardSuppressed = true;
    }
    const gradMag = Math.sqrt(gradX * gradX + gradY * gradY);
    if (!guardSuppressed && gradMag > 0.001) {
      if (mode.startsWith("seek")) {
        thrustX = -cfg.thrustLimit * gradX / gradMag;
        thrustY = -cfg.thrustLimit * gradY / gradMag;
      } else {
        const error = tidal.magnitude - cfg.targetTidal;
        const thrustMagnitude = Math.min(Math.abs(0.5 * error), cfg.thrustLimit);
        const direction = error > 0 ? -1 : 1;
        thrustX = direction * thrustMagnitude * gradX / gradMag;
        thrustY = direction * thrustMagnitude * gradY / gradMag;
      }
    }
  } else if (mode === "track_radius_inward") {
    const x3 = state[4];
    const y3 = state[5];
    const radius = Math.sqrt(x3 * x3 + y3 * y3);
    if (radius >= cfg.trackGuardMinRadius && radius > 1e-9) {
      const magnitude = cfg.radiusInwardMagnitude;
      thrustX = -magnitude * x3 / radius;
      thrustY = -magnitude * y3 / radius;
    }
  }

  if (PHASE14_ABLATION_MODES.has(mode)) {
    return phase14ActionAblation(mode, [thrustX, thrustY], controllerState, cfg);
  }
  return [thrustX, thrustY];
}

export function classifyEvents(state, signatures, tidal, thrust, config = {}) {
  const cfg = normalizeConfig(config);
  const positions = state.slice(0, 6);
  const x3 = positions[4];
  const y3 = positions[5];
  const r3 = Math.sqrt(x3 * x3 + y3 * y3);
  let minPrimaryDistance = Infinity;
  for (let i = 0; i < 2; i += 1) {
    const dx = x3 - positions[i * 2];
    const dy = y3 - positions[i * 2 + 1];
    minPrimaryDistance = Math.min(minPrimaryDistance, Math.sqrt(dx * dx + dy * dy));
  }

  const thrustMagnitude = Math.sqrt(thrust[0] * thrust[0] + thrust[1] * thrust[1]);
  const lowerBand = cfg.targetTidal * (1 - cfg.targetBandFraction);
  const upperBand = cfg.targetTidal * (1 + cfg.targetBandFraction);

  return {
    invalid: state.some((value) => !Number.isFinite(value)),
    escape: r3 > cfg.escapeRadius,
    closeApproach: minPrimaryDistance < cfg.closeApproachRadius,
    tidalSpike: tidal.magnitude > cfg.tidalSpikeThreshold,
    controllerSaturation: thrustMagnitude >= cfg.thrustLimit * 0.999,
    targetBandLoss: tidal.magnitude < lowerBand || tidal.magnitude > upperBand,
    minPrimaryDistance,
    testParticleRadius: r3,
  };
}

export function summarizeOutcome(eventHistory) {
  const terminal = eventHistory[eventHistory.length - 1];
  const firstTime = (key) => {
    const found = eventHistory.find((entry) => entry.events[key]);
    return found ? found.time : null;
  };
  return {
    terminalOutcome: terminal?.events.invalid ? "invalid"
      : terminal?.events.closeApproach ? "close_approach"
        : terminal?.events.escape ? "escape"
          : "bounded",
    timeToEscape: firstTime("escape"),
    timeToCloseApproach: firstTime("closeApproach"),
    timeToTidalSpike: firstTime("tidalSpike"),
    saturationCount: eventHistory.filter((entry) => entry.events.controllerSaturation).length,
    targetBandLossCount: eventHistory.filter((entry) => entry.events.targetBandLoss).length,
    minPrimaryDistance: Math.min(...eventHistory.map((entry) => entry.events.minPrimaryDistance)),
    maxRadius: Math.max(...eventHistory.map((entry) => entry.events.testParticleRadius)),
  };
}

function ratioOrNull(numerator, denominator) {
  if (denominator <= 0) return null;
  return numerator / denominator;
}

function computeAuroc(samples) {
  const positives = samples.filter((sample) => sample.label);
  const negatives = samples.filter((sample) => !sample.label);
  if (positives.length === 0 || negatives.length === 0) return null;

  let score = 0;
  for (const positive of positives) {
    for (const negative of negatives) {
      if (positive.score > negative.score) score += 1;
      else if (positive.score === negative.score) score += 0.5;
    }
  }
  return score / (positives.length * negatives.length);
}

export function makeEventDiagnosticSamples(eventHistory, config = {}, scoreKey = "tidalMagnitude") {
  const cfg = normalizeConfig(config);
  const firstHazard = eventHistory.find((entry) => entry.events.escape || entry.events.closeApproach);
  const hazardTime = firstHazard?.time ?? null;

  return eventHistory.map((entry) => ({
    time: entry.time,
    label: hazardTime !== null
      && entry.time <= hazardTime
      && hazardTime - entry.time <= cfg.eventWarningHorizon,
    score: entry[scoreKey],
    hazardTime,
  }));
}

export function evaluateWarningThreshold(eventHistory, threshold, config = {}, scoreKey = "tidalMagnitude") {
  const cfg = normalizeConfig(config);
  const samples = makeEventDiagnosticSamples(eventHistory, cfg, scoreKey);
  const firstHazard = eventHistory.find((entry) => entry.events.escape || entry.events.closeApproach);
  const hazardTime = firstHazard?.time ?? null;
  const firstWarning = eventHistory.find((entry) => entry[scoreKey] > threshold);
  const warningTime = firstWarning?.time ?? null;
  const warningLeadTime = hazardTime !== null && warningTime !== null && warningTime <= hazardTime
    ? hazardTime - warningTime
    : null;

  let truePositive = 0;
  let falsePositive = 0;
  let trueNegative = 0;
  let falseNegative = 0;

  for (const sample of samples) {
    const warned = sample.score > threshold;
    if (warned && sample.label) truePositive += 1;
    else if (warned) falsePositive += 1;
    else if (sample.label) falseNegative += 1;
    else trueNegative += 1;
  }

  const precision = ratioOrNull(truePositive, truePositive + falsePositive);
  const recall = ratioOrNull(truePositive, truePositive + falseNegative);
  const f1 = precision !== null && recall !== null && precision + recall > 0
    ? 2 * precision * recall / (precision + recall)
    : null;
  const falseAlarmRate = ratioOrNull(falsePositive, falsePositive + trueNegative);

  return {
    firstHazardTime: hazardTime,
    firstTidalWarningTime: warningTime,
    tidalWarningLeadTime: warningLeadTime,
    tidalWarningThreshold: threshold,
    eventWarningHorizon: cfg.eventWarningHorizon,
    eventSampleCount: samples.length,
    positiveEventSampleCount: truePositive + falseNegative,
    warningCount: truePositive + falsePositive,
    warningTruePositive: truePositive,
    warningFalsePositive: falsePositive,
    warningTrueNegative: trueNegative,
    warningFalseNegative: falseNegative,
    warningPrecision: precision === null ? null : roundNumber(precision, 6),
    warningRecall: recall === null ? null : roundNumber(recall, 6),
    warningF1: f1 === null ? null : roundNumber(f1, 6),
    warningFalseAlarmRate: falseAlarmRate === null ? null : roundNumber(falseAlarmRate, 6),
  };
}

export function evaluateTidalWarningThreshold(eventHistory, threshold, config = {}) {
  const metrics = evaluateWarningThreshold(eventHistory, threshold, config, "tidalMagnitude");
  return {
    firstHazardTime: metrics.firstHazardTime,
    firstTidalWarningTime: metrics.firstTidalWarningTime,
    tidalWarningLeadTime: metrics.tidalWarningLeadTime,
    tidalWarningThreshold: metrics.tidalWarningThreshold,
    eventWarningHorizon: metrics.eventWarningHorizon,
    eventSampleCount: metrics.eventSampleCount,
    positiveEventSampleCount: metrics.positiveEventSampleCount,
    tidalWarningCount: metrics.warningCount,
    tidalWarningTruePositive: metrics.warningTruePositive,
    tidalWarningFalsePositive: metrics.warningFalsePositive,
    tidalWarningTrueNegative: metrics.warningTrueNegative,
    tidalWarningFalseNegative: metrics.warningFalseNegative,
    tidalWarningPrecision: metrics.warningPrecision,
    tidalWarningRecall: metrics.warningRecall,
    tidalWarningF1: metrics.warningF1,
    tidalWarningFalseAlarmRate: metrics.warningFalseAlarmRate,
  };
}

export function evaluateLocalAccelerationWarningThreshold(eventHistory, threshold, config = {}) {
  const metrics = evaluateWarningThreshold(eventHistory, threshold, config, "localAccelerationMagnitude");
  return {
    firstHazardTime: metrics.firstHazardTime,
    firstLocalAccelerationWarningTime: metrics.firstTidalWarningTime,
    localAccelerationWarningLeadTime: metrics.tidalWarningLeadTime,
    localAccelerationWarningThreshold: threshold,
    eventWarningHorizon: metrics.eventWarningHorizon,
    eventSampleCount: metrics.eventSampleCount,
    positiveEventSampleCount: metrics.positiveEventSampleCount,
    localAccelerationWarningCount: metrics.warningCount,
    localAccelerationWarningTruePositive: metrics.warningTruePositive,
    localAccelerationWarningFalsePositive: metrics.warningFalsePositive,
    localAccelerationWarningTrueNegative: metrics.warningTrueNegative,
    localAccelerationWarningFalseNegative: metrics.warningFalseNegative,
    localAccelerationWarningPrecision: metrics.warningPrecision,
    localAccelerationWarningRecall: metrics.warningRecall,
    localAccelerationWarningF1: metrics.warningF1,
    localAccelerationWarningFalseAlarmRate: metrics.warningFalseAlarmRate,
  };
}

export function summarizeEventDiagnostics(eventHistory, config = {}) {
  const cfg = normalizeConfig(config);
  const samples = makeEventDiagnosticSamples(eventHistory, cfg);
  const localAccelerationSamples = makeEventDiagnosticSamples(eventHistory, cfg, "localAccelerationMagnitude");
  const thresholdMetrics = evaluateTidalWarningThreshold(eventHistory, cfg.tidalSpikeThreshold, cfg);
  const auroc = computeAuroc(samples);
  const localAccelerationMetrics = evaluateLocalAccelerationWarningThreshold(
    eventHistory,
    cfg.localAccelerationWarningThreshold,
    cfg,
  );
  const localAccelerationAuroc = computeAuroc(localAccelerationSamples);
  const oracleHazardSamples = cfg.precisionReceipts
    ? eventHistory
      .filter((entry) => Number.isFinite(entry.oracleHazardScore) && typeof entry.oracleHazardLabel === "boolean")
      .map((entry) => ({
        time: entry.time,
        score: entry.oracleHazardScore,
        label: entry.oracleHazardLabel,
      }))
    : [];
  const oracleHazardAuroc = cfg.precisionReceipts ? computeAuroc(oracleHazardSamples) : null;

  return {
    ...thresholdMetrics,
    tidalMagnitudeAuroc: auroc === null ? null : roundNumber(auroc, 6),
    ...localAccelerationMetrics,
    localAccelerationMagnitudeAuroc: localAccelerationAuroc === null
      ? null
      : roundNumber(localAccelerationAuroc, 6),
    ...(cfg.precisionReceipts
      ? {
        oracleHazardAuroc: oracleHazardAuroc === null ? null : roundNumber(oracleHazardAuroc, 6),
        oracleHazardSampleCount: oracleHazardSamples.length,
        oracleHazardPositiveSampleCount: oracleHazardSamples.filter((sample) => sample.label).length,
      }
      : {}),
  };
}

function hazardChannelSample(time, state, signatures, tidal, localAccelerationMagnitude, events, oracleHazard) {
  const vx3 = state[10];
  const vy3 = state[11];
  return {
    time: roundNumber(time, 6),
    label: oracleHazard.hazardReached,
    channels: {
      energy: roundNumber(signatures.energy),
      kineticEnergy: roundNumber(signatures.kineticEnergy),
      potentialEnergy: roundNumber(signatures.potentialEnergy),
      virial: roundNumber(signatures.virial),
      inertia: roundNumber(signatures.inertia),
      tidalMagnitude: roundNumber(tidal.magnitude),
      localAccelerationMagnitude: roundNumber(localAccelerationMagnitude),
      radius: roundNumber(events.testParticleRadius),
      minPrimaryDistance: roundNumber(events.minPrimaryDistance),
      speed: roundNumber(Math.sqrt(vx3 * vx3 + vy3 * vy3)),
    },
  };
}

const EARLY_TRAJECTORY_GRID_STEP = 0.12;
const EARLY_TRAJECTORY_MAX_T = 4.8;

export function runTrial(config = {}) {
  const cfg = normalizeConfig(config);
  const steps = Math.max(1, Math.round(cfg.duration / cfg.dt));
  const controllerState = { scanPhase: 0 };
  let state = initializeState(cfg);
  const initialEnergy = computeSignatures(state, cfg).energy;
  let finalEnergy = initialEnergy;
  let maxAbsEnergyDrift = 0;
  const records = [];
  const eventHistory = [];
  const sensorAudit = [];
  const earlyTrajectory = [];
  const hazardSamples = [];
  const sensorStates = new Map();
  let lastEarlyTrajectoryGridIndex = -1;
  let totalDeltaV = 0;
  let acEligible = 0;
  let acPositive = 0;
  let acSignedSum = 0;
  let cfEligible = 0;
  let cfPositive = 0;
  let cfSignedSum = 0;
  let cfGapSum = 0;
  let cfEffectVsNoopSum = 0;
  let cfAbsEffectVsNoopSum = 0;
  let cfRawNormalizerSum = 0;
  let cfMinRawNormalizer = Infinity;
  let cfNormalizerFloorHits = 0;
  let cfFloorEffectVsNoopSum = 0;
  let cfFloorPositive = 0;
  let cfFloorScoreSum = 0;
  let cfNonFloorEligible = 0;
  let cfNonFloorScoreSum = 0;
  // Phase 15C multi-step audit accumulators (one entry per horizon in MS_AUDIT_HORIZONS)
  const msEffectSum        = [0, 0, 0, 0];
  const msAbsEffectSum     = [0, 0, 0, 0];
  const msPositive         = [0, 0, 0, 0];
  const msRawNormSum       = [0, 0, 0, 0];
  const msFloorHits        = [0, 0, 0, 0];
  const msFloorPositive    = [0, 0, 0, 0];
  const msScoreSum         = [0, 0, 0, 0];
  const msNonFloorEligible = [0, 0, 0, 0];
  const msNonFloorScoreSum = [0, 0, 0, 0];
  let msEligible = 0;
  // Phase 17 hazard-aligned counterfactual accumulators (one entry per HAZARD_CF_HORIZONS)
  const hcfZeros = () => HAZARD_CF_HORIZONS.map(() => 0);
  const hcfMarginEffectSum = hcfZeros();
  const hcfAbsMarginEffectSum = hcfZeros();
  const hcfMarginPositive = hcfZeros();
  const hcfEscapeEffectSum = hcfZeros();
  const hcfEscapePositive = hcfZeros();
  const hcfCloseEffectSum = hcfZeros();
  const hcfClosePositive = hcfZeros();
  const hcfHazardAvoidedSum = hcfZeros();
  const hcfHazardAvoidedCount = hcfZeros();
  const hcfHazardCausedCount = hcfZeros();
  let hcfEligible = 0;

  for (let step = 0; step <= steps; step += 1) {
    const time = step * cfg.dt;
    const thrust = computeControlThrust(state, controllerState, cfg);
    const signatures = computeSignatures(state, cfg);
    const tidal = computeTidalTensor(state, cfg);
    const events = classifyEvents(state, signatures, tidal, thrust, cfg);
    const [localAx, localAy] = computeAcceleration(2, state.slice(0, 6), cfg);
    const localAccelerationMagnitude = Math.sqrt(localAx * localAx + localAy * localAy);
    const thrustMagnitude = Math.sqrt(thrust[0] * thrust[0] + thrust[1] * thrust[1]);
    const shouldAuditStep = step % cfg.sensorAuditEvery === 0
      || events.invalid
      || events.escape
      || events.closeApproach
      || step === steps;
    finalEnergy = signatures.energy;
    if (cfg.precisionReceipts) {
      maxAbsEnergyDrift = Math.max(maxAbsEnergyDrift, Math.abs(signatures.energy - initialEnergy));
      if (!controllerState.stepWarmup && thrustMagnitude > 1e-6) {
        const oracleStrictThrust = cfg.controllerMode === "forward_oracle_strict"
          ? thrust
          : computeStrictOracleThrust(state, cfg);
        const counterfactual = computeCounterfactualReceipt(state, thrust, cfg, oracleStrictThrust);
        cfEligible += 1;
        if (counterfactual.score > 0) cfPositive += 1;
        cfSignedSum += counterfactual.score;
        cfGapSum += counterfactual.gapToOracle;
        if (cfg.counterfactualAudit) {
          cfEffectVsNoopSum += counterfactual.effectVsNoop;
          cfAbsEffectVsNoopSum += counterfactual.absEffectVsNoop;
          cfRawNormalizerSum += counterfactual.rawNormalizer;
          cfMinRawNormalizer = Math.min(cfMinRawNormalizer, counterfactual.rawNormalizer);
          if (counterfactual.normalizerFloored) {
            cfNormalizerFloorHits += 1;
            cfFloorEffectVsNoopSum += counterfactual.effectVsNoop;
            cfFloorScoreSum += counterfactual.score;
            if (counterfactual.effectVsNoop > 0) cfFloorPositive += 1;
          } else {
            cfNonFloorEligible += 1;
            cfNonFloorScoreSum += counterfactual.score;
          }
        }
        if (cfg.multiStepAudit) {
          const msResult = computeMultiStepCounterfactualHorizon(
            state, thrust, oracleStrictThrust, cfg,
          );
          msEligible += 1;
          for (let hi = 0; hi < MS_AUDIT_HORIZONS.length; hi += 1) {
            const N = MS_AUDIT_HORIZONS[hi];
            const r = msResult[N];
            msEffectSum[hi]     += r.rawDiff;
            msAbsEffectSum[hi]  += Math.abs(r.rawDiff);
            if (r.positive) msPositive[hi] += 1;
            msRawNormSum[hi]    += r.rawNorm;
            msScoreSum[hi]      += r.score;
            if (r.floored) {
              msFloorHits[hi] += 1;
              if (r.positive) msFloorPositive[hi] += 1;
            } else {
              msNonFloorEligible[hi] += 1;
              msNonFloorScoreSum[hi] += r.score;
            }
          }
        }
      }
    }
    if (cfg.hazardCounterfactualAudit && !controllerState.stepWarmup && thrustMagnitude > 1e-6) {
      const hcf = computeHazardCounterfactualHorizon(state, thrust, cfg);
      hcfEligible += 1;
      for (let hi = 0; hi < HAZARD_CF_HORIZONS.length; hi += 1) {
        const r = hcf[HAZARD_CF_HORIZONS[hi]];
        hcfMarginEffectSum[hi] += r.marginEffect;
        hcfAbsMarginEffectSum[hi] += Math.abs(r.marginEffect);
        if (r.marginEffect > 0) hcfMarginPositive[hi] += 1;
        hcfEscapeEffectSum[hi] += r.escapeMarginEffect;
        if (r.escapeMarginEffect > 0) hcfEscapePositive[hi] += 1;
        hcfCloseEffectSum[hi] += r.closeMarginEffect;
        if (r.closeMarginEffect > 0) hcfClosePositive[hi] += 1;
        hcfHazardAvoidedSum[hi] += r.hazardAvoided;
        if (r.hazardAvoided === 1) hcfHazardAvoidedCount[hi] += 1;
        if (r.hazardAvoided === -1) hcfHazardCausedCount[hi] += 1;
      }
    }
    if (cfg.trackActionCoupling) {
      const idealGradient = computeTidalGradient(state, cfg);
      const idealGradMagnitude = Math.sqrt(
        idealGradient.gradX * idealGradient.gradX + idealGradient.gradY * idealGradient.gradY,
      );
      if (!controllerState.stepWarmup && thrustMagnitude > 1e-6 && idealGradMagnitude > 1e-6) {
        const signedAlignment = Math.sign(cfg.targetTidal - idealGradient.tidal.magnitude)
          * (thrust[0] * idealGradient.gradX + thrust[1] * idealGradient.gradY)
          / (thrustMagnitude * idealGradMagnitude);
        acEligible += 1;
        if (signedAlignment > 0) acPositive += 1;
        acSignedSum += signedAlignment;
      }
    }
    const oracleHazard = (cfg.precisionReceipts || cfg.hazardChannelAudit) && cfg.controllerMode === "off" && shouldAuditStep
      ? computeStrictOracleDetails(state, cfg)
      : null;
    if (cfg.hazardChannelAudit && cfg.controllerMode === "off" && oracleHazard) {
      hazardSamples.push(hazardChannelSample(
        time,
        state,
        signatures,
        tidal,
        localAccelerationMagnitude,
        events,
        oracleHazard,
      ));
    }
    if (cfg.precisionReceipts && cfg.controllerMode === "off") {
      const gridIndex = Math.round(time / EARLY_TRAJECTORY_GRID_STEP);
      const gridTime = gridIndex * EARLY_TRAJECTORY_GRID_STEP;
      const onGrid = Math.abs(time - gridTime) <= Math.max(1e-9, cfg.dt * 1e-6);
      if (
        gridIndex > lastEarlyTrajectoryGridIndex
        && onGrid
        && gridTime <= EARLY_TRAJECTORY_MAX_T + 1e-9
      ) {
        earlyTrajectory.push({
          step,
          time: gridTime,
          x3: state[4],
          y3: state[5],
          vx3: state[10],
          vy3: state[11],
          terminal: events.invalid || events.escape || events.closeApproach,
        });
        lastEarlyTrajectoryGridIndex = gridIndex;
      }
    }
    eventHistory.push({
      time,
      events,
      tidalMagnitude: tidal.magnitude,
      localAccelerationMagnitude,
      ...(cfg.precisionReceipts
        ? {
          energy: signatures.energy,
          oracleHazardScore: oracleHazard ? signatures.energy : null,
          oracleHazardLabel: oracleHazard ? oracleHazard.hazardReached : null,
        }
        : {}),
    });

    if (
      cfg.sensorAuditVariants.length > 0
      && shouldAuditStep
    ) {
      for (const variant of cfg.sensorAuditVariants) {
        if (!sensorStates.has(variant)) sensorStates.set(variant, {});
        const estimate = computeSensorTidalTensor(state, cfg, variant, sensorStates.get(variant));
        const magnitudeAbsError = Math.abs(estimate.magnitude - tidal.magnitude);
        sensorAudit.push({
          step,
          time: roundNumber(time, 6),
          sensorVariant: variant,
          sensorTier: estimate.sensorTier,
          delayWarmup: estimate.delayWarmup,
          noiseStd: estimate.noiseStd,
          delaySteps: estimate.delaySteps,
          probeDelta: estimate.probeDelta,
          idealMagnitude: roundNumber(tidal.magnitude),
          estimatedMagnitude: roundNumber(estimate.magnitude),
          magnitudeAbsError: roundNumber(magnitudeAbsError),
          magnitudeRelError: tidal.magnitude > 1e-9 ? roundNumber(magnitudeAbsError / tidal.magnitude) : null,
          componentRmse: roundNumber(Math.sqrt((
            (estimate.T_xx - tidal.T_xx) ** 2
            + (estimate.T_xy - tidal.T_xy) ** 2
            + (estimate.T_yx - tidal.T_yx) ** 2
            + (estimate.T_yy - tidal.T_yy) ** 2
          ) / 4)),
          localAccelerationMagnitude: roundNumber(localAccelerationMagnitude),
          terminalOutcomeSoFar: events.invalid ? "invalid"
            : events.closeApproach ? "close_approach"
              : events.escape ? "escape"
                : "running",
        });
      }
    }

    if (step % cfg.logEvery === 0 || step === steps || events.invalid || events.escape || events.closeApproach) {
      records.push({
        step,
        time: roundNumber(time, 6),
        state: roundArray(state),
        thrust: roundArray(thrust),
        signatures: {
          virial: roundNumber(signatures.virial),
          inertia: roundNumber(signatures.inertia),
          energy: roundNumber(signatures.energy),
          kineticEnergy: roundNumber(signatures.kineticEnergy),
          potentialEnergy: roundNumber(signatures.potentialEnergy),
        },
        tidal: {
          magnitude: roundNumber(tidal.magnitude),
          T_xx: roundNumber(tidal.T_xx),
          T_xy: roundNumber(tidal.T_xy),
          T_yx: roundNumber(tidal.T_yx),
          T_yy: roundNumber(tidal.T_yy),
        },
        localAccelerationMagnitude: roundNumber(localAccelerationMagnitude),
        ...(cfg.precisionReceipts
          ? {
            energyDrift: roundNumber(signatures.energy - initialEnergy),
            oracleHazardScore: oracleHazard ? roundNumber(signatures.energy) : null,
            oracleHazardLabel: oracleHazard ? oracleHazard.hazardReached : null,
          }
          : {}),
        events: {
          ...events,
          minPrimaryDistance: roundNumber(events.minPrimaryDistance),
          testParticleRadius: roundNumber(events.testParticleRadius),
        },
      });
    }

    if (step === steps || events.invalid || events.escape || events.closeApproach) break;
    totalDeltaV += thrustMagnitude * cfg.dt;
    state = integrateStep(state, cfg.dt, cfg, thrust);
  }

  return {
    config: cfg,
    records,
    eventHistory,
    sensorAudit,
    ...(cfg.precisionReceipts ? { earlyTrajectory } : {}),
    ...(cfg.hazardChannelAudit ? { hazardSamples } : {}),
    summary: {
      ...summarizeOutcome(eventHistory),
      ...summarizeEventDiagnostics(eventHistory, cfg),
      totalDeltaV: roundNumber(totalDeltaV),
      loggedRecords: records.length,
      simulatedTime: records.at(-1)?.time ?? 0,
      ...(cfg.hazardChannelAudit
        ? {
          hazardChannelSampleCount: hazardSamples.length,
          hazardChannelPositiveSampleCount: hazardSamples.filter((sample) => sample.label).length,
        }
        : {}),
      ...(cfg.precisionReceipts
        ? {
          finalRelEnergyDrift: roundNumber(Math.abs(finalEnergy - initialEnergy) / Math.max(Math.abs(initialEnergy), 1e-9), 10),
          maxAbsEnergyDrift: roundNumber(maxAbsEnergyDrift, 10),
          counterfactualEligibleSteps: cfEligible,
          counterfactualMeanEffect: cfEligible > 0 ? roundNumber(cfSignedSum / cfEligible, 6) : null,
          counterfactualPositiveRate: cfEligible > 0 ? roundNumber(cfPositive / cfEligible, 6) : null,
          meanGapToOracle: cfEligible > 0 ? roundNumber(cfGapSum / cfEligible, 8) : null,
          ...(cfg.counterfactualAudit
            ? {
              counterfactualMeanEffectVsNoop: cfEligible > 0 ? roundNumber(cfEffectVsNoopSum / cfEligible, 12) : null,
              counterfactualMeanAbsEffectVsNoop: cfEligible > 0 ? roundNumber(cfAbsEffectVsNoopSum / cfEligible, 12) : null,
              counterfactualMeanRawNormalizer: cfEligible > 0 ? roundNumber(cfRawNormalizerSum / cfEligible, 12) : null,
              counterfactualMinRawNormalizer: cfEligible > 0 ? roundNumber(cfMinRawNormalizer, 12) : null,
              counterfactualNormalizerFloor: cfg.counterfactualNormalizerFloor,
              counterfactualNormalizerFloorHits: cfNormalizerFloorHits,
              counterfactualNormalizerFloorRate: cfEligible > 0 ? roundNumber(cfNormalizerFloorHits / cfEligible, 6) : null,
              counterfactualFloorMeanEffectVsNoop: cfNormalizerFloorHits > 0
                ? roundNumber(cfFloorEffectVsNoopSum / cfNormalizerFloorHits, 12)
                : null,
              counterfactualFloorPositiveRate: cfNormalizerFloorHits > 0
                ? roundNumber(cfFloorPositive / cfNormalizerFloorHits, 6)
                : null,
              counterfactualFloorMeanScore: cfNormalizerFloorHits > 0
                ? roundNumber(cfFloorScoreSum / cfNormalizerFloorHits, 6)
                : null,
              counterfactualNonFloorEligibleSteps: cfNonFloorEligible,
              counterfactualNonFloorMeanScore: cfNonFloorEligible > 0
                ? roundNumber(cfNonFloorScoreSum / cfNonFloorEligible, 6)
                : null,
            }
            : {}),
          ...(cfg.multiStepAudit
            ? Object.fromEntries(MS_AUDIT_HORIZONS.flatMap((N, hi) => [
              [`counterfactualH${N}EligibleSteps`, msEligible],
              [`counterfactualH${N}MeanEffectVsNoop`, msEligible > 0 ? roundNumber(msEffectSum[hi] / msEligible, 12) : null],
              [`counterfactualH${N}MeanAbsEffectVsNoop`, msEligible > 0 ? roundNumber(msAbsEffectSum[hi] / msEligible, 12) : null],
              [`counterfactualH${N}PositiveRate`, msEligible > 0 ? roundNumber(msPositive[hi] / msEligible, 6) : null],
              [`counterfactualH${N}MeanRawNormalizer`, msEligible > 0 ? roundNumber(msRawNormSum[hi] / msEligible, 12) : null],
              [`counterfactualH${N}NormalizerFloorRate`, msEligible > 0 ? roundNumber(msFloorHits[hi] / msEligible, 6) : null],
              [`counterfactualH${N}MeanScore`, msEligible > 0 ? roundNumber(msScoreSum[hi] / msEligible, 6) : null],
              [`counterfactualH${N}FloorPositiveRate`, msFloorHits[hi] > 0 ? roundNumber(msFloorPositive[hi] / msFloorHits[hi], 6) : null],
              [`counterfactualH${N}NonFloorMeanScore`, msNonFloorEligible[hi] > 0 ? roundNumber(msNonFloorScoreSum[hi] / msNonFloorEligible[hi], 6) : null],
            ]))
            : {}),
        }
        : {}),
      ...(cfg.hazardCounterfactualAudit
        ? Object.fromEntries(HAZARD_CF_HORIZONS.flatMap((N, hi) => [
          [`hazardCfH${N}EligibleSteps`, hcfEligible],
          [`hazardCfH${N}MeanMarginEffect`, hcfEligible > 0 ? roundNumber(hcfMarginEffectSum[hi] / hcfEligible, 8) : null],
          [`hazardCfH${N}MeanAbsMarginEffect`, hcfEligible > 0 ? roundNumber(hcfAbsMarginEffectSum[hi] / hcfEligible, 8) : null],
          [`hazardCfH${N}PositiveRate`, hcfEligible > 0 ? roundNumber(hcfMarginPositive[hi] / hcfEligible, 6) : null],
          [`hazardCfH${N}MeanEscapeMarginEffect`, hcfEligible > 0 ? roundNumber(hcfEscapeEffectSum[hi] / hcfEligible, 8) : null],
          [`hazardCfH${N}EscapePositiveRate`, hcfEligible > 0 ? roundNumber(hcfEscapePositive[hi] / hcfEligible, 6) : null],
          [`hazardCfH${N}MeanCloseMarginEffect`, hcfEligible > 0 ? roundNumber(hcfCloseEffectSum[hi] / hcfEligible, 8) : null],
          [`hazardCfH${N}ClosePositiveRate`, hcfEligible > 0 ? roundNumber(hcfClosePositive[hi] / hcfEligible, 6) : null],
          [`hazardCfH${N}MeanHazardAvoided`, hcfEligible > 0 ? roundNumber(hcfHazardAvoidedSum[hi] / hcfEligible, 6) : null],
          [`hazardCfH${N}HazardAvoidedRate`, hcfEligible > 0 ? roundNumber(hcfHazardAvoidedCount[hi] / hcfEligible, 6) : null],
          [`hazardCfH${N}HazardCausedRate`, hcfEligible > 0 ? roundNumber(hcfHazardCausedCount[hi] / hcfEligible, 6) : null],
        ]))
        : {}),
      ...(cfg.trackActionCoupling
        ? {
          actionCouplingEligibleSteps: acEligible,
          actionCouplingAgreementRate: acEligible > 0 ? roundNumber(acPositive / acEligible) : null,
          actionCouplingSignedEffect: acEligible > 0 ? roundNumber(acSignedSum / acEligible) : null,
        }
        : {}),
    },
  };
}
