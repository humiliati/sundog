import {
  ShadowFieldEnv,
  falseBasinField,
} from "../public/js/mesa-core.mjs";

const EPSILON = 1e-10;

function assertClose(actual, expected, label, tolerance = EPSILON) {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`${label}: expected ${expected}, got ${actual}`);
  }
}

function assertVecClose(actual, expected, label, tolerance = EPSILON) {
  if (actual.length !== expected.length) throw new Error(`${label}: length mismatch`);
  actual.forEach((value, index) => assertClose(value, expected[index], `${label}[${index}]`, tolerance));
}

function smokeBasinPositionIntervention(channel) {
  const env = new ShadowFieldEnv({
    seed: 31,
    initialState: {
      x: [2.5, 2.5],
      xGoal: [0, 0],
    },
  });
  const xBefore = env.x.slice();
  const xGoalBefore = env.xGoal.slice();
  const oldFalse = env.config.falseBasinCenter.slice();
  const oldBasinValue = falseBasinField(env.x, env.config);
  env.scheduleIntervention({
    step: 0,
    channel,
    edit: { xFalseNew: [2.5, 2.5] },
  });
  const result = env.step([0, 0]);
  assertVecClose(env.x, xBefore, `${channel} leaves x fixed`);
  assertVecClose(env.xGoal, xGoalBefore, `${channel} leaves xGoal fixed`);
  assertVecClose(env.config.falseBasinCenter, [2.5, 2.5], `${channel} moves false basin`);
  if (result.interventionFlags[0] !== channel) {
    throw new Error(`${channel}: missing intervention flag`);
  }
  if (!(oldBasinValue < 0.01)) {
    throw new Error(`${channel}: expected low old basin value, got ${oldBasinValue}`);
  }
  assertClose(result.rewardChannels.false_basin, 1, `${channel} new basin value`);
  assertClose(result.rewardChannels.false_basin_bonus, env.config.falseBasinBeta, `${channel} new basin bonus`);
  assertVecClose(oldFalse, [-2.5, -2.5], `${channel} default false basin`);
}

function smokeGeometryInterventionStillWorks() {
  const env = new ShadowFieldEnv({
    seed: 32,
    initialState: {
      x: [0, 0],
      xGoal: [1, 1],
    },
  });
  env.scheduleIntervention({
    step: 0,
    channel: "geometry",
    edit: { xGoalNew: [2, 2] },
  });
  const result = env.step([0, 0]);
  assertVecClose(env.xGoal, [2, 2], "geometry moves xGoal");
  if (result.interventionFlags[0] !== "geometry") {
    throw new Error("geometry: missing intervention flag");
  }
}

smokeBasinPositionIntervention("basin-position");
smokeBasinPositionIntervention("basin-position-edit");
smokeGeometryInterventionStillWorks();
console.log("mesa phase4 intervention smoke passed: basin_position=pass geometry=pass");
