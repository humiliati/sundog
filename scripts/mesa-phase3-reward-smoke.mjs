import {
  ShadowFieldEnv,
  falseBasinField,
  normalizeMesaConfig,
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

function distance(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function phase3Formula(env, action) {
  const dense = -distance(env.x, env.xGoal);
  const actionCost = env.config.rewardControlAlpha * (action[0] * action[0] + action[1] * action[1]);
  const falseBasin = falseBasinField(env.x, env.config);
  return {
    dense,
    actionCost,
    falseBasin,
    falseBasinBonus: env.config.falseBasinBeta * falseBasin,
    phase3: dense - actionCost + env.config.falseBasinBeta * falseBasin,
  };
}

function smokeRewardFormula() {
  const env = new ShadowFieldEnv({
    seed: 7,
    initialState: {
      x: [-3, -3],
      xGoal: [1, 1],
    },
  });
  const action = [0.4, -0.2];
  const result = env.step(action);
  const expected = phase3Formula(env, result.action);
  assertClose(result.rewardChannels.dense, expected.dense, "dense");
  assertClose(result.rewardChannels.action_cost, expected.actionCost, "action_cost");
  assertClose(result.rewardChannels.false_basin, expected.falseBasin, "false_basin");
  assertClose(result.rewardChannels.false_basin_bonus, expected.falseBasinBonus, "false_basin_bonus");
  assertClose(result.rewardChannels.phase3_dense_action_basin, expected.phase3, "phase3_dense_action_basin");
}

function smokeFalseBasinFixture() {
  const config = normalizeMesaConfig({
    seed: 11,
    initialState: {
      x: [2, 0],
      xGoal: [0.5, -1.25],
    },
  });
  const probes = [
    { rotate: Math.PI / 2 },
    { translate: [1.25, -0.75] },
    { scale: 1.5 },
    { mirror: "x" },
    { mirror: "y" },
    { rotate: -Math.PI / 4, translate: [-0.5, 0.5], scale: 1.25, mirror: "x" },
  ];
  for (const probe of probes) {
    const env = new ShadowFieldEnv(config);
    const before = env.config.falseBasinCenter.slice();
    env.applyProbe(probe);
    assertVecClose(env.config.falseBasinCenter, before, `falseBasinCenter fixed for ${JSON.stringify(probe)}`);
    assertClose(falseBasinField(before, env.config), 1, `false basin peak fixed for ${JSON.stringify(probe)}`);
  }
}

smokeRewardFormula();
smokeFalseBasinFixture();
console.log("mesa phase3 reward smoke passed: formula=pass x_false_fixture=pass");
