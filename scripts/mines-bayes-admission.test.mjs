import assert from "node:assert/strict";
import {
  ACTION,
  applyMinesAction,
  getPublicMemory,
  initializeBoardState,
} from "../public/js/mines-core.mjs";
import {
  IMPLEMENTED_MINES_MODES,
  MINES_CONTROLLER_MODES,
  chooseMinesAction,
  frontierIndices,
} from "../public/js/mines-controllers.mjs";
import {
  assertNoMinesBayesObservationLeak,
  forbiddenMinesBayesObservationKeys,
  serializeMinesBayesObservation,
} from "../public/js/mines-bayes-admission.mjs";
import { createSensorRuntime, normalizeSensorConfig } from "../public/js/mines-sensor.mjs";
import { parseArgs } from "./mines-phase4-baselines.mjs";

function withoutMode(observation) {
  const { controllerMode: _controllerMode, ...rest } = observation;
  return rest;
}

function commonAdmissionFields(observation) {
  const {
    controllerMode: _controllerMode,
    budget: _budget,
    activeChannels: _activeChannels,
    pressureGradientField: _pressureGradientField,
    scanReadings: _scanReadings,
    ...common
  } = observation;
  common.actionLedger = common.actionLedger.map((entry) => {
    const { scanReading: _scanReading, ...rest } = entry;
    return rest;
  });
  return common;
}

assert.equal(MINES_CONTROLLER_MODES.bayes_frontier_pressure.status, "pending");
assert.equal(MINES_CONTROLLER_MODES.bayes_frontier_full.status, "pending");
assert.equal(IMPLEMENTED_MINES_MODES.includes("bayes_frontier_pressure"), false);
assert.equal(IMPLEMENTED_MINES_MODES.includes("bayes_frontier_full"), false);

assert.throws(
  () => chooseMinesAction({ mode: "bogus_bayes_lane" }),
  /Unknown mines controller mode/,
  "unknown Mines controller modes must fail loudly",
);

assert.throws(
  () => chooseMinesAction({ mode: "bayes_frontier_pressure" }),
  /pending, not runnable yet/,
  "declared Bayes lanes must remain pending until a posterior policy exists",
);

assert.throws(
  () => parseArgs(["--modes", "bayes_frontier_pressure"]),
  /declared but not implemented yet/,
  "headless harness must reject pending Bayes lanes before rollout",
);

const board = initializeBoardState({
  preset: "easy_sparse",
  seed: 47,
  mineCount: 13,
  scanBudget: 2,
  turnCap: 160,
});
applyMinesAction(board, {
  type: ACTION.REVEAL,
  x: Math.floor(board.config.width / 2),
  y: Math.floor(board.config.height / 2),
});

const scanTarget = frontierIndices(getPublicMemory(board))[0];
assert.equal(Number.isInteger(scanTarget), true, "fixture needs a scan target");
const scanAction = {
  type: ACTION.SCAN,
  x: scanTarget % board.config.width,
  y: Math.floor(scanTarget / board.config.width),
};
const scanResult = applyMinesAction(board, scanAction);
assert.equal(scanResult.applied, true, "fixture scan should apply");

const sensorConfig = normalizeSensorConfig({
  sigma: 1.0,
  sigmaNoise: 0.1,
  dropoutRate: 0.1,
  delaySteps: 0,
  sensorSeed: 123456,
});
const sensorRuntime = createSensorRuntime(sensorConfig);
const scan = sensorRuntime.scan(board, scanAction.x, scanAction.y);
const lastEntry = board.actionLedger[board.actionLedger.length - 1];
lastEntry.scanReading = scan.reading;

const sensor = sensorRuntime.step(board);
const memory = getPublicMemory(board);
const envelopeCell = {
  cellId: "phase10-best-smoke",
  preset: "easy_sparse",
  sensorCell: "doc_default",
  mineDensity: 13 / (board.config.width * board.config.height),
  pressureNoise: 0.1,
  dropoutRate: 0.1,
  scanBudget: 2,
};

const sundogMinimalPhi = serializeMinesBayesObservation({
  controllerMode: "sundog_minimal",
  budget: "pressure",
  memory,
  sensor,
  sensorConfig,
  envelopeCell,
});
const sundogLeanPhi = serializeMinesBayesObservation({
  controllerMode: "sundog_lean",
  budget: "pressure",
  memory,
  sensor,
  sensorConfig,
  envelopeCell,
});
const bayesPressurePhi = serializeMinesBayesObservation({
  controllerMode: "bayes_frontier_pressure",
  budget: "pressure",
  memory,
  sensor,
  sensorConfig,
  envelopeCell,
});
const bayesFullPhi = serializeMinesBayesObservation({
  controllerMode: "bayes_frontier_full",
  budget: "full",
  memory,
  sensor,
  sensorConfig,
  envelopeCell,
});

for (const phi of [sundogMinimalPhi, sundogLeanPhi, bayesPressurePhi, bayesFullPhi]) {
  assert.deepEqual(forbiddenMinesBayesObservationKeys(phi), [], "Phi_t must not contain forbidden keys");
  assert.equal(assertNoMinesBayesObservationLeak(phi), true);
  assert.equal(Object.hasOwn(phi.sensorConfig, "sensorSeed"), false, "sensor seed must be stripped");
  assert.equal(Object.hasOwn(phi, "seed"), false, "board seed must not be serialized");
}

assert.deepEqual(
  withoutMode(sundogMinimalPhi),
  withoutMode(sundogLeanPhi),
  "pressure-budget Phi_t must not depend on the non-Bayes controller lane",
);
assert.deepEqual(
  withoutMode(sundogMinimalPhi),
  withoutMode(bayesPressurePhi),
  "bayes_frontier_pressure must receive the same pressure-budget Phi_t",
);
assert.deepEqual(
  commonAdmissionFields(bayesPressurePhi),
  commonAdmissionFields(bayesFullPhi),
  "pressure and full budgets must share the same public base observation",
);
assert.equal(bayesPressurePhi.pressureGradientField, null, "pressure budget must mask gradients");
assert.deepEqual(bayesPressurePhi.scanReadings, [], "pressure budget must mask scan readings");
assert.equal(Array.isArray(bayesFullPhi.pressureGradientField.x), true, "full budget should include gradient x");
assert.equal(Array.isArray(bayesFullPhi.pressureGradientField.y), true, "full budget should include gradient y");
assert.equal(bayesFullPhi.scanReadings.length, 1, "full budget should include legal scan readings");
assert.equal(bayesFullPhi.actionLedger.some((entry) => Object.hasOwn(entry, "scanReading")), true);
assert.equal(bayesPressurePhi.actionLedger.some((entry) => Object.hasOwn(entry, "scanReading")), false);

const rerunSensorRuntime = createSensorRuntime(sensorConfig);
rerunSensorRuntime.scan(board, scanAction.x, scanAction.y);
const rerunSensor = rerunSensorRuntime.step(board);
const rerunPhi = serializeMinesBayesObservation({
  controllerMode: "bayes_frontier_pressure",
  budget: "pressure",
  memory,
  sensor: rerunSensor,
  sensorConfig,
  envelopeCell,
});
assert.deepEqual(
  withoutMode(bayesPressurePhi),
  withoutMode(rerunPhi),
  "same seed and public state should reproduce the admitted pressure observation",
);

console.log("[mines-bayes-admission] lane guard, no-leak, and Phi_t parity checks passed");
