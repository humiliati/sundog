import assert from "node:assert/strict";
import {
  computeControlThrust,
  initializeState,
  normalizeConfig,
  observeGuardedAccelSignature,
} from "../public/js/threebody-core.mjs";

function approxEqual(actual, expected, label, tolerance = 1e-12) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${label}: expected ${expected}, got ${actual}`,
  );
}

function expectedGuardedTrackThrust(signature, config) {
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

const config = normalizeConfig({
  seed: 11,
  regime: "near_escape",
  controllerMode: "track_sensor_accel_guarded",
  thrustLimit: 0.4,
  sensorNoiseStd: 0,
});
const state = initializeState(config);

const signatureA = observeGuardedAccelSignature(state, config, {});
const signatureB = observeGuardedAccelSignature(state, config, {});
assert.deepEqual(signatureA, signatureB, "noise-free guarded signature should be deterministic");
assert.equal(signatureA.sensorVariant, "accelerometer_array_noisy");
assert.equal(signatureA.sensorTier, "accelerometer_array");
assert.equal(signatureA.sensorNoiseStd, 0);
assert.equal(signatureA.probeDelta, config.tidalProbeDelta);

const expectedThrust = expectedGuardedTrackThrust(signatureA, config);
const actualThrust = computeControlThrust(state, {}, config);
approxEqual(actualThrust[0], expectedThrust[0], "guarded thrust x");
approxEqual(actualThrust[1], expectedThrust[1], "guarded thrust y");

const noisyConfig = { ...config, sensorNoiseStd: 0.03 };
const noisyA = observeGuardedAccelSignature(state, noisyConfig, {});
const noisyB = observeGuardedAccelSignature(state, noisyConfig, {});
assert.deepEqual(noisyA, noisyB, "same seed/noise path should be reproducible from fresh state");
assert.equal(noisyA.seed, noisyConfig.seed);
assert.equal(noisyA.sensorNoiseStd, noisyConfig.sensorNoiseStd);
assert.equal(noisyA.probeDelta, noisyConfig.tidalProbeDelta);

console.log("[threebody-signature-observation] guarded signature parity and noise receipts passed");
