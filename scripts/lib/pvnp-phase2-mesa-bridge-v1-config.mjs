export const V1_RUN_ID = "phase2-mesa-bridge-v1";
export const V1_OUT = "results/pvnp/phase2-mesa-bridge-v1";
export const V1_SMALL_RERUN_ROOT = "results/pvnp/phase2-mesa-bridge-v1-small-rerun/phase4-intervention-battery";

export const SMALL_RAW_BATTERIES = Object.freeze([
  {
    slug: "hc_signature",
    label: "HC-Signature",
    args: [
      "--reference", "hc-signature",
      "--policy-label", "HC-Signature",
      "--out", `${V1_SMALL_RERUN_ROOT}/hc_signature`,
      "--seed-start", "10000",
      "--seeds", "64",
      "--sensor-tier", "local-probe-field",
      "--horizon", "200",
    ],
  },
  {
    slug: "l_signature_canonical_1m",
    label: "L-Signature",
    args: [
      "--policy", "results/mesa/phase2-matched-capacity/policies/signature_ppo_dense_small_seed_0_canonical_1m.policy.json",
      "--policy-label", "L-Signature",
      "--out", `${V1_SMALL_RERUN_ROOT}/l_signature_canonical_1m`,
      "--seed-start", "10000",
      "--seeds", "64",
      "--sensor-tier", "local-probe-field",
      "--horizon", "200",
    ],
  },
  {
    slug: "l_reward_phase3_canonical_1m",
    label: "L-Reward",
    args: [
      "--policy", "results/mesa/phase2-matched-capacity/policies/reward_ppo_phase3_small_seed_0_phase3_canonical_1m.policy.json",
      "--policy-label", "L-Reward",
      "--out", `${V1_SMALL_RERUN_ROOT}/l_reward_phase3_canonical_1m`,
      "--seed-start", "10000",
      "--seeds", "64",
      "--sensor-tier", "local-probe-field",
      "--horizon", "200",
    ],
  },
  {
    slug: "l_mixed_phase3_canonical_1m",
    label: "L-Mixed",
    args: [
      "--policy", "results/mesa/phase2-matched-capacity/policies/mixed_ppo_phase3_lambda_0_5_small_seed_0_phase3_canonical_1m.policy.json",
      "--policy-label", "L-Mixed",
      "--out", `${V1_SMALL_RERUN_ROOT}/l_mixed_phase3_canonical_1m`,
      "--seed-start", "10000",
      "--seeds", "64",
      "--sensor-tier", "local-probe-field",
      "--horizon", "200",
    ],
  },
  {
    slug: "phase5_l_mixed_lambda_0_7_small",
    label: "L-Mixed lambda=0.7",
    args: [
      "--policy", "results/mesa/phase2-matched-capacity/policies/mixed_ppo_phase3_lambda_0_7_small_seed_0_phase5_lambda_0_7.policy.json",
      "--policy-label", "L-Mixed lambda=0.7",
      "--out", `${V1_SMALL_RERUN_ROOT}/phase5_l_mixed_lambda_0_7_small`,
      "--seed-start", "10000",
      "--seeds", "64",
      "--sensor-tier", "local-probe-field",
      "--horizon", "200",
    ],
  },
  {
    slug: "phase5_l_mixed_lambda_0_9_small",
    label: "L-Mixed lambda=0.9",
    args: [
      "--policy", "results/mesa/phase2-matched-capacity/policies/mixed_ppo_phase3_lambda_0_9_small_seed_0_phase5_lambda_0_9.policy.json",
      "--policy-label", "L-Mixed lambda=0.9",
      "--out", `${V1_SMALL_RERUN_ROOT}/phase5_l_mixed_lambda_0_9_small`,
      "--seed-start", "10000",
      "--seeds", "64",
      "--sensor-tier", "local-probe-field",
      "--horizon", "200",
    ],
  },
]);

export function nodeArgsForBattery(battery) {
  return ["scripts/mesa-intervention-battery.mjs", ...battery.args];
}

function quotePowerShellArg(arg) {
  if (/^[A-Za-z0-9_./:=+-]+$/.test(arg)) return arg;
  return `"${arg.replaceAll("`", "``").replaceAll('"', '`"')}"`;
}

export function powerShellCommandForBattery(battery) {
  return ["node", ...nodeArgsForBattery(battery)].map(quotePowerShellArg).join(" ");
}

export function smallRawCommandsPs1() {
  return [
    "# Phase 2 mesa bridge v1 raw-logged Small rerun commands",
    "# Frozen by docs/pvnp/PHASE2_MESA_BRIDGE_V1_SLATE.md",
    "$ErrorActionPreference = 'Stop'",
    "",
    ...SMALL_RAW_BATTERIES.flatMap((battery) => [
      `# ${battery.slug}`,
      powerShellCommandForBattery(battery),
      "",
    ]),
  ].join("\n");
}
