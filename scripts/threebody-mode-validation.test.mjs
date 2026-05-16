import assert from "node:assert/strict";
import {
  computeControlThrust,
  initializeState,
  KNOWN_CONTROLLER_MODES,
  normalizeConfig,
} from "../public/js/threebody-core.mjs";

const baseConfig = normalizeConfig({
  seed: 7,
  regime: "near_escape",
  duration: 0.02,
  dt: 0.01,
  thrustLimit: 0.4,
  sensorNoiseStd: 0,
});
const state = initializeState(baseConfig);

assert.throws(
  () => computeControlThrust(state, {}, { ...baseConfig, controllerMode: "definitely_not_a_mode" }),
  /unknown controllerMode: definitely_not_a_mode/,
);

for (const mode of KNOWN_CONTROLLER_MODES) {
  assert.doesNotThrow(() => {
    const thrust = computeControlThrust(state, {}, { ...baseConfig, controllerMode: mode });
    assert.equal(Array.isArray(thrust), true, `${mode} should return a thrust vector`);
    assert.equal(thrust.length, 2, `${mode} should return a 2D thrust vector`);
  }, `${mode} should be accepted`);
}

console.log(`[threebody-mode-validation] accepted ${KNOWN_CONTROLLER_MODES.size} modes and rejected bogus mode`);
