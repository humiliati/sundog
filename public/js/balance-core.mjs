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

export function computeBalanceControl(state, sensor, controllerState = {}, config = {}) {
  const cfg = normalizeBalanceConfig(config);
  const mode = cfg.controllerMode;
  let force = 0;
  let phase = "PASSIVE";
  let reason = "no force";

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
    const proxy = sensor.valid ? sensor.residual / Math.max(cfg.poleLength, 1e-6) : 0;
    const proxyVelocity = sensor.valid ? sensor.residualVelocity / Math.max(cfg.poleLength, 1e-6) : 0;
    const confidence = sensor.valid ? sensor.confidence : 0;
    const probe = Math.sin((controllerState.step ?? 0) * 0.11) * (1 - confidence) * 1.2;
    force = confidence * (50 * proxy + 8 * proxyVelocity) + 1.0 * state.x + 2.0 * state.xDot + probe;
    phase = confidence > 0.35 ? "TRACK" : "SCAN";
    reason = "shadow residual plus history";
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
