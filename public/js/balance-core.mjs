export const DEFAULT_BALANCE_CONFIG = Object.freeze({
  gravity: 9.81,
  cartMass: 1,
  poleMass: 0.1,
  poleLength: 1.25,
  dt: 1 / 120,
  duration: 20,
  forceLimit: 12,
  railLimit: 2.4,
  fallAngle: 0.65,
  lightElevationDeg: 28,
  lightAzimuthSign: 1,
  shadowSoftness: 0.06,
  shadowConfidenceScale: 0.5,
  sensorNoiseStd: 0,
  sensorDelaySteps: 0,
  sensorQuantization: 0,
  sensorDropoutRate: 0,
  controllerMode: "passive",
  preset: "easy",
  seed: 1,
});

export const BALANCE_PRESETS = Object.freeze({
  easy: Object.freeze({
    label: "Easy lean",
    state: Object.freeze({ x: 0, xDot: 0, theta: 0.08, thetaDot: 0 }),
  }),
  recoverable: Object.freeze({
    label: "Recoverable push",
    state: Object.freeze({ x: -0.2, xDot: 0.15, theta: 0.18, thetaDot: -0.08 }),
  }),
  near_fall: Object.freeze({
    label: "Near fall",
    state: Object.freeze({ x: 0.1, xDot: 0, theta: -0.34, thetaDot: 0.18 }),
  }),
  rail_edge: Object.freeze({
    label: "Rail edge",
    state: Object.freeze({ x: 1.45, xDot: -0.05, theta: 0.12, thetaDot: 0.02 }),
  }),
  noisy_shadow: Object.freeze({
    label: "Noisy shadow",
    state: Object.freeze({ x: 0, xDot: 0, theta: 0.12, thetaDot: 0 }),
    config: Object.freeze({ sensorNoiseStd: 0.025 }),
  }),
  delayed_shadow: Object.freeze({
    label: "Delayed shadow",
    state: Object.freeze({ x: 0, xDot: 0, theta: 0.12, thetaDot: 0 }),
    config: Object.freeze({ sensorDelaySteps: 18 }),
  }),
});

export const BALANCE_CONTROLLER_MODES = Object.freeze({
  passive: Object.freeze({
    label: "Passive",
    status: "implemented",
    informationBudget: "No control force.",
    usesShadow: false,
    usesCartState: false,
    usesBelief: false,
    usesPrivileged: false,
  }),
  naive_cart: Object.freeze({
    label: "Naive cart",
    status: "implemented",
    informationBudget: "Cart position and velocity only; no shadow field or true pole angle.",
    usesShadow: false,
    usesCartState: true,
    usesBelief: false,
    usesPrivileged: false,
  }),
  naive_shadow: Object.freeze({
    label: "Naive shadow",
    status: "implemented",
    informationBudget: "Current shadow residual plus cart proprioception; no dynamics or observability gating.",
    usesShadow: true,
    usesCartState: true,
    usesBelief: false,
    usesPrivileged: false,
  }),
  sundog_shadow: Object.freeze({
    label: "Sundog shadow",
    status: "implemented",
    informationBudget: "Shadow residual, residual velocity, confidence gating, cart proprioception, and bounded reacquire probe.",
    usesShadow: true,
    usesCartState: true,
    usesBelief: false,
    usesPrivileged: false,
  }),
  bayes_floor_shadow_particle: Object.freeze({
    label: "Bayesian floor: shadow particle",
    status: "implemented",
    informationBudget: "Same shadow observation and cart proprioception as the legal controller, with a particle belief over theta/thetaDot. No true theta/thetaDot.",
    usesShadow: true,
    usesCartState: true,
    usesBelief: true,
    usesPrivileged: false,
  }),
  oracle: Object.freeze({
    label: "Privileged oracle",
    status: "implemented",
    informationBudget: "Privileged true pole angle and angular velocity. Diagnostic ceiling only.",
    usesShadow: false,
    usesCartState: true,
    usesBelief: false,
    usesPrivileged: true,
  }),
});

export const IMPLEMENTED_BALANCE_MODES = Object.freeze(
  Object.entries(BALANCE_CONTROLLER_MODES)
    .filter(([, definition]) => definition.status === "implemented")
    .map(([mode]) => mode),
);

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

export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function roundNumber(value, digits = 6) {
  if (!Number.isFinite(value)) return value;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function normalSample(rng) {
  const u1 = Math.max(rng(), 1e-9);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export function normalizeBalanceConfig(config = {}) {
  const presetConfig = BALANCE_PRESETS[config.preset]?.config ?? {};
  return {
    ...DEFAULT_BALANCE_CONFIG,
    ...presetConfig,
    ...config,
  };
}

export function initializeBalanceState(config = {}) {
  const cfg = normalizeBalanceConfig(config);
  const preset = BALANCE_PRESETS[cfg.preset] ?? BALANCE_PRESETS.easy;
  const initial = config.initialState ?? preset.state;
  return {
    x: initial.x,
    xDot: initial.xDot,
    theta: initial.theta,
    thetaDot: initial.thetaDot,
    t: 0,
    fallen: false,
    railHit: false,
  };
}

function addScaledState(state, derivative, scale) {
  return {
    x: state.x + derivative.x * scale,
    xDot: state.xDot + derivative.xDot * scale,
    theta: state.theta + derivative.theta * scale,
    thetaDot: state.thetaDot + derivative.thetaDot * scale,
    t: state.t,
    fallen: state.fallen,
    railHit: state.railHit,
  };
}

function mixDerivatives(k1, k2, k3, k4, dt) {
  return {
    x: (dt / 6) * (k1.x + 2 * k2.x + 2 * k3.x + k4.x),
    xDot: (dt / 6) * (k1.xDot + 2 * k2.xDot + 2 * k3.xDot + k4.xDot),
    theta: (dt / 6) * (k1.theta + 2 * k2.theta + 2 * k3.theta + k4.theta),
    thetaDot: (dt / 6) * (k1.thetaDot + 2 * k2.thetaDot + 2 * k3.thetaDot + k4.thetaDot),
  };
}

export function cartPoleDerivative(state, force, config = {}) {
  const cfg = normalizeBalanceConfig(config);
  const halfLength = cfg.poleLength / 2;
  const totalMass = cfg.cartMass + cfg.poleMass;
  const poleMassLength = cfg.poleMass * halfLength;
  const boundedForce = clamp(force, -cfg.forceLimit, cfg.forceLimit);
  const sinTheta = Math.sin(state.theta);
  const cosTheta = Math.cos(state.theta);
  const temp = (
    boundedForce + poleMassLength * state.thetaDot * state.thetaDot * sinTheta
  ) / totalMass;
  const thetaAcc = (
    cfg.gravity * sinTheta - cosTheta * temp
  ) / (
    halfLength * (4 / 3 - (cfg.poleMass * cosTheta * cosTheta) / totalMass)
  );
  const xAcc = temp - (poleMassLength * thetaAcc * cosTheta) / totalMass;

  return {
    x: state.xDot,
    xDot: xAcc,
    theta: state.thetaDot,
    thetaDot: thetaAcc,
  };
}

export function classifyBalanceState(state, config = {}) {
  const cfg = normalizeBalanceConfig(config);
  return {
    fallen: Math.abs(state.theta) >= cfg.fallAngle,
    railHit: Math.abs(state.x) >= cfg.railLimit,
    timedOut: state.t >= cfg.duration,
  };
}

export function integrateBalanceStep(state, force, config = {}) {
  const cfg = normalizeBalanceConfig(config);
  const dt = cfg.dt;
  const boundedForce = clamp(force, -cfg.forceLimit, cfg.forceLimit);

  const k1 = cartPoleDerivative(state, boundedForce, cfg);
  const k2 = cartPoleDerivative(addScaledState(state, k1, dt / 2), boundedForce, cfg);
  const k3 = cartPoleDerivative(addScaledState(state, k2, dt / 2), boundedForce, cfg);
  const k4 = cartPoleDerivative(addScaledState(state, k3, dt), boundedForce, cfg);
  const delta = mixDerivatives(k1, k2, k3, k4, dt);
  const next = {
    x: state.x + delta.x,
    xDot: state.xDot + delta.xDot,
    theta: state.theta + delta.theta,
    thetaDot: state.thetaDot + delta.thetaDot,
    t: state.t + dt,
    fallen: false,
    railHit: false,
  };

  if (Math.abs(next.x) > cfg.railLimit) {
    next.x = clamp(next.x, -cfg.railLimit, cfg.railLimit);
    next.xDot *= -0.15;
  }

  const terminal = classifyBalanceState(next, cfg);
  next.fallen = terminal.fallen;
  next.railHit = terminal.railHit;
  return next;
}

export function computeShadowGeometry(state, config = {}) {
  const cfg = normalizeBalanceConfig(config);
  const elevationRad = clamp(cfg.lightElevationDeg, 2, 88) * Math.PI / 180;
  const lx = Math.cos(elevationRad) * Math.sign(cfg.lightAzimuthSign || 1);
  const lz = -Math.sin(elevationRad);
  const poleTipX = state.x + cfg.poleLength * Math.sin(state.theta);
  const poleTipZ = Math.max(0, cfg.poleLength * Math.cos(state.theta));
  const baseShadowX = state.x;
  const shadowTipX = poleTipX - poleTipZ * (lx / lz);
  const uprightShadowTipX = state.x - cfg.poleLength * (lx / lz);
  const signedLength = shadowTipX - baseShadowX;
  const uprightSignedLength = uprightShadowTipX - baseShadowX;
  const residual = shadowTipX - uprightShadowTipX;
  const centroid = (baseShadowX + shadowTipX) / 2;
  const length = Math.abs(signedLength);
  const contrast = length / (length + cfg.shadowSoftness + 1e-6);
  const longShadowConfidence = Math.abs(uprightSignedLength)
    / (Math.abs(uprightSignedLength) + cfg.shadowConfidenceScale);
  const confidence = clamp(0.25 * contrast + 0.75 * longShadowConfidence, 0, 1);

  return {
    baseShadowX,
    shadowTipX,
    uprightShadowTipX,
    poleTipX,
    poleTipZ,
    centroid,
    length,
    signedLength,
    uprightSignedLength,
    residual,
    confidence,
  };
}

export function createBalanceRuntime(config = {}) {
  const cfg = normalizeBalanceConfig(config);
  return {
    rng: makeRng(cfg.seed),
    sensorQueue: [],
    previousSensor: null,
    step: 0,
  };
}

function maybeQuantize(value, quantum) {
  if (!quantum) return value;
  return Math.round(value / quantum) * quantum;
}

export function sampleShadowSensor(state, runtime, config = {}) {
  const cfg = normalizeBalanceConfig(config);
  const raw = computeShadowGeometry(state, cfg);
  const rng = runtime.rng ?? makeRng(cfg.seed);
  const dropped = cfg.sensorDropoutRate > 0 && rng() < cfg.sensorDropoutRate;
  const noise = cfg.sensorNoiseStd;
  const noisyTip = maybeQuantize(
    raw.shadowTipX + (noise ? normalSample(rng) * noise : 0),
    cfg.sensorQuantization,
  );
  const noisyCentroid = maybeQuantize(
    raw.centroid + (noise ? normalSample(rng) * noise * 0.5 : 0),
    cfg.sensorQuantization,
  );
  const noisyLength = maybeQuantize(
    Math.abs(noisyTip - raw.baseShadowX),
    cfg.sensorQuantization,
  );
  const noisyResidual = noisyTip - raw.uprightShadowTipX;

  const sample = {
    ...raw,
    shadowTipX: noisyTip,
    centroid: noisyCentroid,
    length: noisyLength,
    residual: noisyResidual,
    valid: !dropped,
  };

  runtime.sensorQueue.push(sample);
  const delay = Math.max(0, Math.round(cfg.sensorDelaySteps));
  while (runtime.sensorQueue.length > delay + 1) {
    runtime.sensorQueue.shift();
  }

  const delayed = runtime.sensorQueue[0] ?? sample;
  const previous = runtime.previousSensor;
  const residualVelocity = previous
    ? (delayed.residual - previous.residual) / cfg.dt
    : 0;
  const lengthVelocity = previous
    ? (delayed.length - previous.length) / cfg.dt
    : 0;

  const output = {
    ...delayed,
    residualVelocity,
    lengthVelocity,
    delaySteps: delay,
    raw,
  };

  if (output.valid) {
    runtime.previousSensor = output;
  }
  runtime.step += 1;
  return output;
}

export function serializeBalanceObservation(state, sensor, config = {}) {
  const cfg = normalizeBalanceConfig(config);
  return {
    x: state.x,
    xDot: state.xDot,
    t: state.t,
    shadowTipX: sensor.shadowTipX,
    shadowCentroidX: sensor.centroid,
    shadowLength: sensor.length,
    shadowResidual: sensor.residual,
    shadowConfidence: sensor.valid ? sensor.confidence : 0,
    shadowValid: sensor.valid,
    residualVelocity: sensor.valid ? sensor.residualVelocity : 0,
    lengthVelocity: sensor.valid ? sensor.lengthVelocity : 0,
    lightElevationDeg: cfg.lightElevationDeg,
    sensorNoiseStd: cfg.sensorNoiseStd,
    sensorDelaySteps: cfg.sensorDelaySteps,
    sensorDropoutRate: cfg.sensorDropoutRate,
    dt: cfg.dt,
    preset: cfg.preset,
    forceLimit: cfg.forceLimit,
    railLimit: cfg.railLimit,
  };
}

function hashText(text) {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function uniqueSortedForces(values, forceLimit) {
  return [...new Set(values
    .map((value) => roundNumber(clamp(value, -forceLimit, forceLimit), 4)))]
    .sort((a, b) => a - b);
}

function estimateShadowAngle(sensor, config = {}) {
  const cfg = normalizeBalanceConfig(config);
  if (!sensor?.valid) return { valid: false, reason: "invalid_sensor" };
  const elevationRad = clamp(cfg.lightElevationDeg, 2, 88) * Math.PI / 180;
  const k = Math.sign(cfg.lightAzimuthSign || 1) / Math.tan(elevationRad);
  const radius = Math.hypot(1, k);
  const phase = Math.atan2(k, 1);
  const y = sensor.residual / Math.max(cfg.poleLength, 1e-9) + k;
  const normalized = y / Math.max(radius, 1e-9);
  const clamped = clamp(normalized, -1, 1);
  const a = Math.asin(clamped);
  const candidates = [
    a - phase,
    Math.PI - a - phase,
  ].map((theta) => {
    let wrapped = theta;
    while (wrapped > Math.PI) wrapped -= Math.PI * 2;
    while (wrapped < -Math.PI) wrapped += Math.PI * 2;
    return wrapped;
  });
  const theta = candidates
    .filter((candidate) => Math.abs(candidate) <= cfg.fallAngle * 1.25)
    .sort((left, right) => Math.abs(left) - Math.abs(right))[0]
    ?? candidates.sort((left, right) => Math.abs(left) - Math.abs(right))[0];
  const derivative = cfg.poleLength * (Math.cos(theta) - k * Math.sin(theta));
  const thetaDot = Math.abs(derivative) > 1e-6
    ? sensor.residualVelocity / derivative
    : 0;
  return {
    valid: Number.isFinite(theta) && Math.abs(normalized) <= 1.08,
    theta: clamp(theta, -cfg.fallAngle * 0.98, cfg.fallAngle * 0.98),
    thetaDot: clamp(thetaDot, -4, 4),
    derivative,
    clamped: clamped !== normalized,
    confidence: sensor.confidence,
  };
}

function ensureBayesState(controllerState, cfg, shadowEstimate = null) {
  if (controllerState.bayes?.particles?.length > 0) return controllerState.bayes;
  const particleCount = Math.max(31, Math.round(cfg.bayesParticleCount ?? 121));
  const seed = ((cfg.seed + 1) * 2654435761) ^ hashText(`${cfg.preset}:balance-bayes-floor`);
  const rng = makeRng(seed >>> 0);
  const thetaRange = shadowEstimate?.valid ? 0.08 : Math.min(cfg.fallAngle * 0.82, 0.52);
  const thetaDotRange = shadowEstimate?.valid ? 0.45 : 0.9;
  const thetaCenter = shadowEstimate?.valid ? shadowEstimate.theta : 0;
  const thetaDotCenter = shadowEstimate?.valid ? shadowEstimate.thetaDot : 0;
  const particles = [];
  for (let i = 0; i < particleCount; i += 1) {
    particles.push({
      theta: clamp(thetaCenter + (rng() * 2 - 1) * thetaRange, -cfg.fallAngle * 0.98, cfg.fallAngle * 0.98),
      thetaDot: clamp(thetaDotCenter + (rng() * 2 - 1) * thetaDotRange, -4, 4),
      weight: 1 / particleCount,
    });
  }
  controllerState.bayes = {
    particles,
    rng,
    lastForce: 0,
    initialized: false,
    diagnostics: null,
  };
  return controllerState.bayes;
}

function injectShadowProposal(bayes, shadowEstimate, cfg) {
  if (!shadowEstimate?.valid || shadowEstimate.confidence < 0.22) return;
  const count = Math.max(1, Math.floor(bayes.particles.length * 0.12));
  const start = bayes.particles.length - count;
  const thetaSpread = Math.max(0.006, (1 - shadowEstimate.confidence) * 0.035);
  const thetaDotSpread = Math.max(0.03, (1 - shadowEstimate.confidence) * 0.22);
  for (let i = start; i < bayes.particles.length; i += 1) {
    bayes.particles[i] = {
      theta: clamp(shadowEstimate.theta + (bayes.rng() - 0.5) * thetaSpread, -cfg.fallAngle * 0.98, cfg.fallAngle * 0.98),
      thetaDot: clamp(shadowEstimate.thetaDot + (bayes.rng() - 0.5) * thetaDotSpread, -4, 4),
      weight: 1 / bayes.particles.length,
    };
  }
  const total = bayes.particles.reduce((sum, particle) => sum + particle.weight, 0);
  if (total > 0) {
    for (const particle of bayes.particles) particle.weight /= total;
  }
}

function predictBayesParticles(bayes, state, cfg) {
  if (!bayes.initialized) {
    bayes.initialized = true;
    return;
  }
  for (const particle of bayes.particles) {
    const predicted = integrateBalanceStep({
      x: state.x,
      xDot: state.xDot,
      theta: particle.theta,
      thetaDot: particle.thetaDot,
      t: state.t,
      fallen: false,
      railHit: false,
    }, bayes.lastForce, cfg);
    const processTheta = (bayes.rng() - 0.5) * 0.002;
    const processThetaDot = (bayes.rng() - 0.5) * 0.01;
    particle.theta = clamp(predicted.theta + processTheta, -cfg.fallAngle * 0.98, cfg.fallAngle * 0.98);
    particle.thetaDot = clamp(predicted.thetaDot + processThetaDot, -4, 4);
  }
}

function weightBayesParticles(bayes, state, sensor, cfg) {
  if (!sensor.valid) return;
  const sigma = Math.max(0.018, cfg.sensorNoiseStd * 2.5, (1 - sensor.confidence) * 0.055);
  const lengthSigma = Math.max(0.035, sigma * 1.6);
  const centroidSigma = Math.max(0.03, sigma * 1.3);
  let total = 0;

  for (const particle of bayes.particles) {
    const predicted = computeShadowGeometry({
      x: state.x,
      xDot: state.xDot,
      theta: particle.theta,
      thetaDot: particle.thetaDot,
      t: state.t,
      fallen: false,
      railHit: false,
    }, cfg);
    const residualErr = (sensor.residual - predicted.residual) / sigma;
    const lengthErr = (sensor.length - predicted.length) / lengthSigma;
    const centroidErr = (sensor.centroid - predicted.centroid) / centroidSigma;
    const confidenceErr = (sensor.confidence - predicted.confidence) / 0.25;
    const exponent = -0.5 * (
      residualErr * residualErr
      + 0.35 * lengthErr * lengthErr
      + 0.2 * centroidErr * centroidErr
      + 0.05 * confidenceErr * confidenceErr
    );
    particle.weight *= Math.max(1e-12, Math.exp(Math.max(-60, exponent)));
    total += particle.weight;
  }

  if (!(total > 0)) {
    const w = 1 / bayes.particles.length;
    for (const particle of bayes.particles) particle.weight = w;
    return;
  }
  for (const particle of bayes.particles) particle.weight /= total;
}

function bayesDiagnostics(particles) {
  let thetaMean = 0;
  let thetaDotMean = 0;
  let weightSquareSum = 0;
  for (const particle of particles) {
    thetaMean += particle.weight * particle.theta;
    thetaDotMean += particle.weight * particle.thetaDot;
    weightSquareSum += particle.weight * particle.weight;
  }
  let thetaVariance = 0;
  let thetaDotVariance = 0;
  for (const particle of particles) {
    thetaVariance += particle.weight * (particle.theta - thetaMean) ** 2;
    thetaDotVariance += particle.weight * (particle.thetaDot - thetaDotMean) ** 2;
  }
  return {
    thetaMean,
    thetaDotMean,
    thetaStd: Math.sqrt(Math.max(0, thetaVariance)),
    thetaDotStd: Math.sqrt(Math.max(0, thetaDotVariance)),
    effectiveSampleSize: weightSquareSum > 0 ? 1 / weightSquareSum : 0,
  };
}

function resampleBayesParticlesIfNeeded(bayes) {
  const diagnostics = bayesDiagnostics(bayes.particles);
  if (diagnostics.effectiveSampleSize >= bayes.particles.length * 0.5) return diagnostics;

  const cumulative = [];
  let sum = 0;
  for (const particle of bayes.particles) {
    sum += particle.weight;
    cumulative.push(sum);
  }
  const n = bayes.particles.length;
  const step = 1 / n;
  let u = bayes.rng() * step;
  let j = 0;
  const next = [];
  for (let i = 0; i < n; i += 1) {
    while (j < cumulative.length - 1 && u > cumulative[j]) j += 1;
    next.push({
      theta: clamp(bayes.particles[j].theta + (bayes.rng() - 0.5) * 0.004, -Math.PI / 2, Math.PI / 2),
      thetaDot: clamp(bayes.particles[j].thetaDot + (bayes.rng() - 0.5) * 0.02, -4, 4),
      weight: step,
    });
    u += step;
  }
  bayes.particles = next;
  return bayesDiagnostics(bayes.particles);
}

function scoreBayesAction(particles, state, action, cfg) {
  const horizonSteps = Math.max(2, Math.round((cfg.bayesHorizonSeconds ?? 0.18) / cfg.dt));
  let score = 0;
  for (const particle of particles) {
    let simulated = {
      x: state.x,
      xDot: state.xDot,
      theta: particle.theta,
      thetaDot: particle.thetaDot,
      t: state.t,
      fallen: false,
      railHit: false,
    };
    for (let i = 0; i < horizonSteps && !simulated.fallen && !simulated.railHit; i += 1) {
      simulated = integrateBalanceStep(simulated, action, cfg);
    }
    const terminalPenalty = simulated.fallen || simulated.railHit ? 30 : 0;
    score += particle.weight * (
      terminalPenalty
      + Math.abs(simulated.theta) * 5
      + Math.abs(simulated.thetaDot) * 0.7
      + Math.abs(simulated.x) * 0.35
      + Math.abs(action) / Math.max(cfg.forceLimit, 1e-6) * 0.08
    );
  }
  return score;
}

function scoreBayesActionPoint(state, theta, thetaDot, action, cfg) {
  const horizonSteps = Math.max(2, Math.round((cfg.bayesHorizonSeconds ?? 0.18) / cfg.dt));
  let simulated = {
    x: state.x,
    xDot: state.xDot,
    theta,
    thetaDot,
    t: state.t,
    fallen: false,
    railHit: false,
  };
  for (let i = 0; i < horizonSteps && !simulated.fallen && !simulated.railHit; i += 1) {
    simulated = integrateBalanceStep(simulated, action, cfg);
  }
  const terminalPenalty = simulated.fallen || simulated.railHit ? 30 : 0;
  return terminalPenalty
    + Math.abs(simulated.theta) * 5
    + Math.abs(simulated.thetaDot) * 0.7
    + Math.abs(simulated.x) * 0.35
    + Math.abs(action) / Math.max(cfg.forceLimit, 1e-6) * 0.08;
}

function bayesObservationStress(sensor, cfg) {
  const delayStress = clamp((cfg.sensorDelaySteps - 12) / 18, 0, 1);
  const noiseStress = clamp(cfg.sensorNoiseStd / 0.08, 0, 1);
  const dropoutStress = clamp((cfg.sensorDropoutRate - 0.1) / 0.25, 0, 1);
  const confidenceStress = sensor?.valid
    ? clamp((0.55 - sensor.confidence) / 0.35, 0, 1)
    : 1;
  return Math.max(delayStress, noiseStress, dropoutStress, confidenceStress);
}

function computeSundogShadowForce(state, sensor, controllerState, cfg) {
  const proxy = sensor.valid ? sensor.residual / Math.max(cfg.poleLength, 1e-6) : 0;
  const proxyVelocity = sensor.valid ? sensor.residualVelocity / Math.max(cfg.poleLength, 1e-6) : 0;
  const confidence = sensor.valid ? sensor.confidence : 0;
  const probe = Math.sin((controllerState.step ?? 0) * 0.11) * (1 - confidence) * 1.2;
  return confidence * (50 * proxy + 8 * proxyVelocity) + 1.0 * state.x + 2.0 * state.xDot + probe;
}

function computeBayesFloorControl(state, sensor, controllerState, cfg) {
  const shadowEstimate = estimateShadowAngle(sensor, cfg);
  const bayes = ensureBayesState(controllerState, cfg, shadowEstimate);
  predictBayesParticles(bayes, state, cfg);
  weightBayesParticles(bayes, state, sensor, cfg);
  let diagnostics = resampleBayesParticlesIfNeeded(bayes);
  injectShadowProposal(bayes, shadowEstimate, cfg);
  diagnostics = bayesDiagnostics(bayes.particles);

  const shadowThetaProxy = sensor.valid ? sensor.residual / Math.max(cfg.poleLength, 1e-6) : diagnostics.thetaMean;
  const shadowThetaDotProxy = sensor.valid ? sensor.residualVelocity / Math.max(cfg.poleLength, 1e-6) : diagnostics.thetaDotMean;
  const inverseWeight = shadowEstimate.valid
    ? clamp((shadowEstimate.confidence - 0.2) / 0.55, 0, 0.85)
    : 0;
  const beliefTheta = inverseWeight * shadowEstimate.theta
    + (1 - inverseWeight) * (0.65 * diagnostics.thetaMean + 0.35 * shadowThetaProxy);
  const beliefThetaDot = inverseWeight * shadowEstimate.thetaDot
    + (1 - inverseWeight) * (0.45 * diagnostics.thetaDotMean + 0.55 * shadowThetaDotProxy);
  const beliefFeedback = 50 * beliefTheta + 8 * beliefThetaDot + 1.0 * state.x + 2.0 * state.xDot;
  const sundogForce = computeSundogShadowForce(state, sensor, controllerState, cfg);
  const candidateForces = uniqueSortedForces([
    sundogForce,
    beliefFeedback,
    -cfg.forceLimit,
    -0.5 * cfg.forceLimit,
    0,
    0.5 * cfg.forceLimit,
    cfg.forceLimit,
  ], cfg.forceLimit);
  const posteriorReady = sensor.valid
    && (shadowEstimate.valid || diagnostics.thetaStd < 0.12)
    && diagnostics.thetaStd < 0.18
    && diagnostics.effectiveSampleSize >= bayes.particles.length * 0.2;
  const blendedBeliefForce = 0.65 * sundogForce + 0.35 * beliefFeedback;
  const sundogAction = clamp(sundogForce, -cfg.forceLimit, cfg.forceLimit);
  const proposalAction = clamp(blendedBeliefForce, -cfg.forceLimit, cfg.forceLimit);
  const sundogScore = scoreBayesActionPoint(state, beliefTheta, beliefThetaDot, sundogAction, cfg);
  const proposalScore = scoreBayesActionPoint(state, beliefTheta, beliefThetaDot, proposalAction, cfg);
  const scoreAdvantage = sundogScore - proposalScore;
  const baseAdvantageThreshold = cfg.bayesAdvantageThreshold ?? 0.06;
  const observationStress = bayesObservationStress(sensor, cfg);
  const advantageThreshold = baseAdvantageThreshold + observationStress * 0.18;
  const degradationReady = observationStress < 0.05
    || (
      shadowEstimate.valid
      && shadowEstimate.confidence >= 0.58
      && diagnostics.thetaStd < 0.08
      && diagnostics.thetaDotStd < 0.55
      && diagnostics.effectiveSampleSize >= bayes.particles.length * 0.4
    );
  const useProposal = posteriorReady && degradationReady && scoreAdvantage > advantageThreshold;
  const bestForce = useProposal ? proposalAction : sundogAction;
  const bestScore = useProposal ? proposalScore : sundogScore;

  bayes.lastForce = bestForce;
  bayes.diagnostics = {
    ...diagnostics,
    particleCount: bayes.particles.length,
    candidateCount: candidateForces.length,
    selectedScore: bestScore,
    posteriorReady,
    inverseValid: shadowEstimate.valid,
    inverseTheta: shadowEstimate.valid ? shadowEstimate.theta : null,
    inverseThetaDot: shadowEstimate.valid ? shadowEstimate.thetaDot : null,
    inverseWeight,
    beliefFeedback,
    blendedBeliefForce,
    sundogCandidateForce: sundogForce,
    sundogScore,
    proposalScore,
    scoreAdvantage,
    baseAdvantageThreshold,
    advantageThreshold,
    observationStress,
    degradationReady,
    selectedCandidate: useProposal ? "bayes_proposal" : "sundog_guard",
  };

  return {
    force: bestForce,
    rawForce: bestForce,
    saturated: Math.abs(bestForce) >= cfg.forceLimit - 1e-9,
    phase: useProposal
      ? "BAYES_TRACK"
      : sensor.valid && sensor.confidence > 0.35
        ? "BAYES_GUARD"
        : "BAYES_SCAN",
    reason: useProposal
      ? "same-shadow inverse proposal cleared advantage gate"
      : "same-information guard kept sundog candidate",
    belief: bayes.diagnostics,
  };
}

export function computeBalanceControl(state, sensor, controllerState = {}, config = {}) {
  const cfg = normalizeBalanceConfig(config);
  const mode = cfg.controllerMode;
  let force = 0;
  let phase = "PASSIVE";
  let reason = "no force";

  if (!BALANCE_CONTROLLER_MODES[mode]) {
    throw new Error(`Unknown balance controller mode: ${mode}`);
  }

  if (mode === "naive_cart") {
    force = -3.2 * state.x - 2.1 * state.xDot;
    phase = "CENTER";
    reason = "cart centering only";
  } else if (mode === "naive_shadow") {
    const proxy = sensor.valid ? sensor.residual / Math.max(cfg.poleLength, 1e-6) : 0;
    force = 18 * proxy + 1.0 * state.x + 2.0 * state.xDot;
    phase = "SHADOW";
    reason = "shadow residual without dynamics";
  } else if (mode === "oracle") {
    force = 50 * state.theta + 8 * state.thetaDot + 1.0 * state.x + 2.0 * state.xDot;
    phase = "ORACLE";
    reason = "privileged theta feedback";
  } else if (mode === "sundog_shadow") {
    const confidence = sensor.valid ? sensor.confidence : 0;
    force = computeSundogShadowForce(state, sensor, controllerState, cfg);
    phase = confidence > 0.35 ? "TRACK" : "SCAN";
    reason = "shadow residual plus history";
  } else if (mode === "bayes_floor_shadow_particle") {
    const control = computeBayesFloorControl(state, sensor, controllerState, cfg);
    controllerState.step = (controllerState.step ?? 0) + 1;
    return control;
  }

  controllerState.step = (controllerState.step ?? 0) + 1;
  const boundedForce = clamp(force, -cfg.forceLimit, cfg.forceLimit);
  return {
    force: boundedForce,
    rawForce: force,
    saturated: Math.abs(force) > cfg.forceLimit,
    phase,
    reason,
  };
}

export function assessBalanceBoundary(config = {}, sensor = null, control = null, state = null) {
  const cfg = normalizeBalanceConfig(config);
  const mechanisms = [];
  const delayMs = cfg.sensorDelaySteps * cfg.dt * 1000;

  const addMechanism = (severity, code, message, value = null) => {
    mechanisms.push({ severity, code, message, value });
  };

  if (cfg.lightElevationDeg >= 80) {
    addMechanism(
      "unsafe",
      "shadow_unobservable",
      "Overhead light collapses shadow length and angle information.",
      `${roundNumber(cfg.lightElevationDeg, 2)} deg`,
    );
  } else if (cfg.lightElevationDeg >= 70) {
    addMechanism(
      "caution",
      "short_shadow",
      "High light elevation shortens the shadow; treat recovery as provisional.",
      `${roundNumber(cfg.lightElevationDeg, 2)} deg`,
    );
  } else if (cfg.lightElevationDeg <= 8) {
    addMechanism(
      "caution",
      "long_shadow_scaling",
      "Very low light makes a long shadow that amplifies scale and rail effects.",
      `${roundNumber(cfg.lightElevationDeg, 2)} deg`,
    );
  }

  if (delayMs >= 200) {
    addMechanism(
      "unsafe",
      "delay_destabilized",
      "Sensor delay is in the pre-registered failure regime.",
      `${roundNumber(delayMs, 1)} ms`,
    );
  } else if (delayMs >= 100) {
    addMechanism(
      "caution",
      "delay_margin",
      "Delay is high enough to hurt recovery before steady balance.",
      `${roundNumber(delayMs, 1)} ms`,
    );
  }

  if (cfg.sensorNoiseStd >= 0.055) {
    addMechanism(
      "unsafe",
      "sensor_noise_floor",
      "Pixel jitter can dominate the shadow residual.",
      roundNumber(cfg.sensorNoiseStd, 4),
    );
  } else if (cfg.sensorNoiseStd >= 0.03) {
    addMechanism(
      "caution",
      "sensor_noise_margin",
      "Noise is high enough that warning timing may jitter.",
      roundNumber(cfg.sensorNoiseStd, 4),
    );
  }

  if (cfg.sensorDropoutRate >= 0.2) {
    addMechanism(
      "unsafe",
      "dropped_frames",
      "Dropped sensor frames are high enough to break residual history.",
      roundNumber(cfg.sensorDropoutRate, 4),
    );
  } else if (cfg.sensorDropoutRate >= 0.08) {
    addMechanism(
      "caution",
      "dropout_margin",
      "Dropped frames may erase short recovery cues.",
      roundNumber(cfg.sensorDropoutRate, 4),
    );
  }

  if (cfg.forceLimit <= 5) {
    addMechanism(
      "unsafe",
      "force_saturated",
      "Actuator authority is too low for the standard recovery impulse.",
      roundNumber(cfg.forceLimit, 3),
    );
  } else if (cfg.forceLimit <= 8) {
    addMechanism(
      "caution",
      "force_margin",
      "Force limit is near the recovery margin.",
      roundNumber(cfg.forceLimit, 3),
    );
  }

  if (Number.isFinite(cfg.disturbanceForce)) {
    if (cfg.disturbanceForce >= 8) {
      addMechanism(
        "unsafe",
        "disturbance_too_large",
        "Disturbance magnitude exceeds the first-pass recovery envelope.",
        roundNumber(cfg.disturbanceForce, 3),
      );
    } else if (cfg.disturbanceForce >= 6.5) {
      addMechanism(
        "caution",
        "disturbance_margin",
        "Disturbance magnitude is high enough to make recovery boundary-dominated.",
        roundNumber(cfg.disturbanceForce, 3),
      );
    }
  }

  if (cfg.railLimit <= 1.2) {
    addMechanism(
      "unsafe",
      "rail_limited",
      "Rail travel is too short for a clean recovery audit.",
      roundNumber(cfg.railLimit, 3),
    );
  } else if (cfg.railLimit <= 1.8) {
    addMechanism(
      "caution",
      "rail_margin",
      "Rail travel may become the limiting mechanism.",
      roundNumber(cfg.railLimit, 3),
    );
  }

  if (sensor) {
    if (!sensor.valid) {
      addMechanism("unsafe", "dropped_frame_now", "Current sensor sample is dropped.", "invalid");
    } else if (sensor.confidence < 0.18) {
      addMechanism(
        "unsafe",
        "shadow_unobservable_now",
        "Current shadow confidence is below the usable band.",
        roundNumber(sensor.confidence, 3),
      );
    } else if (sensor.confidence < 0.35) {
      addMechanism(
        "caution",
        "low_shadow_confidence",
        "Current shadow confidence is in the reacquire band.",
        roundNumber(sensor.confidence, 3),
      );
    }
  }

  if (control?.saturated) {
    addMechanism("caution", "saturation_now", "Controller is currently saturating force.", "saturated");
  }

  if (state && Math.abs(state.x) >= cfg.railLimit * 0.9) {
    addMechanism(
      "unsafe",
      "rail_edge_now",
      "Cart is at the rail boundary.",
      roundNumber(state.x, 3),
    );
  } else if (state && Math.abs(state.x) >= cfg.railLimit * 0.75) {
    addMechanism(
      "caution",
      "rail_margin_now",
      "Cart is close enough to the rail to constrain recovery.",
      roundNumber(state.x, 3),
    );
  }

  const hasUnsafe = mechanisms.some((mechanism) => mechanism.severity === "unsafe");
  const hasCaution = mechanisms.some((mechanism) => mechanism.severity === "caution");
  const status = hasUnsafe ? "do_not_use" : hasCaution ? "watch_boundary" : "likely_readable";
  const label = status === "do_not_use"
    ? "DO NOT USE"
    : status === "watch_boundary"
      ? "WATCH BOUNDARY"
      : "LIKELY READABLE";
  const summary = status === "do_not_use"
    ? "This cell is inside a known or live failure regime; use it to show the boundary, not success."
    : status === "watch_boundary"
      ? "This cell is near a Phase 9 boundary; recovery should be checked against matched seeds."
      : "No Phase 9 boundary warning is active for the current controls.";

  return {
    status,
    label,
    summary,
    delayMs: roundNumber(delayMs, 3),
    mechanisms,
  };
}

export function serializeBalanceSample(state, sensor, control, config = {}) {
  const cfg = normalizeBalanceConfig(config);
  return {
    t: roundNumber(state.t),
    x: roundNumber(state.x),
    xDot: roundNumber(state.xDot),
    theta: roundNumber(state.theta),
    thetaDot: roundNumber(state.thetaDot),
    shadowResidual: roundNumber(sensor.residual),
    shadowLength: roundNumber(sensor.length),
    shadowConfidence: roundNumber(sensor.confidence),
    force: roundNumber(control.force),
    mode: cfg.controllerMode,
    fallen: state.fallen,
    railHit: state.railHit,
  };
}
