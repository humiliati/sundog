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

export function computeControlThrust(state, controllerState = {}, config = {}) {
  const cfg = normalizeConfig(config);
  const mode = cfg.controllerMode;
  controllerState.step = (controllerState.step ?? 0) + 1;
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
    || mode === "seek_sensor_delayed"
    || mode === "track_sensor_delayed"
    || mode === "seek_sensor_micro"
    || mode === "track_sensor_micro"
  ) {
    const sensorVariant = controllerSensorVariant(mode);
    let gradient = sensorVariant
      ? computeSensorTidalGradient(state, cfg, sensorVariant, controllerState)
      : computeTidalGradient(state, cfg);
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
    const { tidal, gradX, gradY } = gradient;
    if (mode === "track_sensor_accel_guarded" && !shouldRunGuardedTrack(state, tidal, cfg)) {
      return [0, 0];
    }
    const gradMag = Math.sqrt(gradX * gradX + gradY * gradY);
    if (gradMag > 0.001) {
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

  return {
    ...thresholdMetrics,
    tidalMagnitudeAuroc: auroc === null ? null : roundNumber(auroc, 6),
    ...localAccelerationMetrics,
    localAccelerationMagnitudeAuroc: localAccelerationAuroc === null
      ? null
      : roundNumber(localAccelerationAuroc, 6),
  };
}

export function runTrial(config = {}) {
  const cfg = normalizeConfig(config);
  const steps = Math.max(1, Math.round(cfg.duration / cfg.dt));
  const controllerState = { scanPhase: 0 };
  let state = initializeState(cfg);
  const records = [];
  const eventHistory = [];
  const sensorAudit = [];
  const sensorStates = new Map();
  let totalDeltaV = 0;

  for (let step = 0; step <= steps; step += 1) {
    const time = step * cfg.dt;
    const thrust = computeControlThrust(state, controllerState, cfg);
    const signatures = computeSignatures(state, cfg);
    const tidal = computeTidalTensor(state, cfg);
    const events = classifyEvents(state, signatures, tidal, thrust, cfg);
    const [localAx, localAy] = computeAcceleration(2, state.slice(0, 6), cfg);
    const localAccelerationMagnitude = Math.sqrt(localAx * localAx + localAy * localAy);
    const thrustMagnitude = Math.sqrt(thrust[0] * thrust[0] + thrust[1] * thrust[1]);
    eventHistory.push({
      time,
      events,
      tidalMagnitude: tidal.magnitude,
      localAccelerationMagnitude,
    });

    if (
      cfg.sensorAuditVariants.length > 0
      && (step % cfg.sensorAuditEvery === 0 || events.invalid || events.escape || events.closeApproach || step === steps)
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
    summary: {
      ...summarizeOutcome(eventHistory),
      ...summarizeEventDiagnostics(eventHistory, cfg),
      totalDeltaV: roundNumber(totalDeltaV),
      loggedRecords: records.length,
      simulatedTime: records.at(-1)?.time ?? 0,
    },
  };
}
