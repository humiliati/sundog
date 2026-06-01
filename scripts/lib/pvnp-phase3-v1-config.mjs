// scripts/lib/pvnp-phase3-v1-config.mjs
//
// Frozen constants for the Phase 3 capacity-relative one-wayness v1 repair
// battery. Source of truth:
//   docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V1_SLATE.md
//
// This module contains only pre-registered contract values. The v1 harness may
// read them, but may not tune them after reading holdout blocks.

export const V1_RUN_ID = "phase3-capacity-one-wayness-v1";
export const V1_SCHEMA = "pvnp-phase3-capacity-one-wayness-v1";
export const V1_OUT = "results/pvnp/phase3-capacity-one-wayness-v1";
export const V1_HOLDOUT_ROOT =
  "results/pvnp/phase3-capacity-one-wayness-v1/phase4-intervention-battery";

export const V1_K = 4;
export const V1_M = 3;
export const V1_HOLDOUT_SEED_STARTS = Object.freeze([60000, 70000, 80000, 90000]);
export const V1_HOLDOUT_SEEDS = 64;
export const V1_HOLDOUT_SENSOR_TIER = "local-probe-field";
export const V1_HOLDOUT_HORIZON = 200;

export const V1_HOLDOUT_SOURCES = Object.freeze([
  {
    slug: "hc_signature",
    label: "HC-Signature",
    sourceKind: "reference",
    reference: "hc-signature",
  },
  {
    slug: "hc_signature_medium",
    label: "HC-Signature",
    sourceKind: "reference",
    reference: "hc-signature",
  },
  {
    slug: "l_signature_canonical_1m",
    label: "L-Signature",
    sourceKind: "policy",
    policy: "results/mesa/phase2-matched-capacity/policies/signature_ppo_dense_small_seed_0_canonical_1m.policy.json",
  },
  {
    slug: "l_signature_medium_10m",
    label: "L-Signature",
    sourceKind: "policy",
    policy: "results/mesa/phase2-matched-capacity/policies/signature_ppo_dense_medium_seed_0_medium_canonical_10m.policy.json",
  },
  {
    slug: "l_reward_phase3_canonical_1m",
    label: "L-Reward",
    sourceKind: "policy",
    policy: "results/mesa/phase2-matched-capacity/policies/reward_ppo_phase3_small_seed_0_phase3_canonical_1m.policy.json",
  },
  {
    slug: "l_reward_phase3_medium_10m",
    label: "L-Reward",
    sourceKind: "policy",
    policy: "results/mesa/phase2-matched-capacity/policies/reward_ppo_phase3_medium_seed_0_medium_phase3_canonical_10m.policy.json",
  },
  {
    slug: "l_mixed_phase3_canonical_1m",
    label: "L-Mixed",
    sourceKind: "policy",
    policy: "results/mesa/phase2-matched-capacity/policies/mixed_ppo_phase3_lambda_0_5_small_seed_0_phase3_canonical_1m.policy.json",
  },
  {
    slug: "l_mixed_phase3_medium_10m",
    label: "L-Mixed",
    sourceKind: "policy",
    policy: "results/mesa/phase2-matched-capacity/policies/mixed_ppo_phase3_lambda_0_5_medium_seed_0_medium_phase3_canonical_10m.policy.json",
  },
  {
    slug: "phase5_l_mixed_lambda_0_7_small",
    label: "L-Mixed lambda=0.7",
    sourceKind: "policy",
    policy: "results/mesa/phase2-matched-capacity/policies/mixed_ppo_phase3_lambda_0_7_small_seed_0_phase5_lambda_0_7.policy.json",
  },
  {
    slug: "phase5_l_mixed_lambda_0_9_small",
    label: "L-Mixed lambda=0.9",
    sourceKind: "policy",
    policy: "results/mesa/phase2-matched-capacity/policies/mixed_ppo_phase3_lambda_0_9_small_seed_0_phase5_lambda_0_9.policy.json",
  },
  {
    slug: "phase5_v4_l_mixed_medium_lambda_0_95",
    label: "L-Mixed-M-lambda-0.95",
    sourceKind: "policy",
    policy: "results/mesa/phase2-matched-capacity/policies/mixed_ppo_phase3_lambda_0_9_medium_seed_0_medium_phase5_v4_lambda_0_95_10m.policy.json",
  },
  {
    slug: "phase5_v4_l_mixed_medium_lambda_0_97",
    label: "L-Mixed-M-lambda-0.97",
    sourceKind: "policy",
    policy: "results/mesa/phase2-matched-capacity/policies/mixed_ppo_phase3_lambda_0_9_medium_seed_0_medium_phase5_v4_lambda_0_97_10m.policy.json",
  },
  {
    slug: "phase5_v4_l_mixed_medium_lambda_0_99",
    label: "L-Mixed-M-lambda-0.99",
    sourceKind: "policy",
    policy: "results/mesa/phase2-matched-capacity/policies/mixed_ppo_phase3_lambda_0_9_medium_seed_0_medium_phase5_v4_lambda_0_99_10m.policy.json",
  },
]);

export const SIGNATURE_FLOOR_GROUPS = Object.freeze([
  {
    group: "hc-signature",
    controllerId: "hc-signature",
    cellIds: Object.freeze(["hc_signature_small", "hc_signature_medium"]),
  },
  {
    group: "l_signature_canonical_1m",
    controllerId: "l_signature_canonical_1m",
    cellIds: Object.freeze(["l_signature_small"]),
  },
  {
    group: "l_signature_medium_10m",
    controllerId: "l_signature_medium_10m",
    cellIds: Object.freeze(["l_signature_medium"]),
  },
]);

export const V0_FALSIFIER_REGRESSION = Object.freeze({
  sourceSlug: "phase5_l_mixed_lambda_0_7_small",
  cellId: "l_mixed_lambda_0_7_small",
  root: "results/pvnp/phase3-capacity-one-wayness-v0-seed-extension/phase4-intervention-battery",
  seedStarts: Object.freeze([20000, 30000, 40000, 50000]),
});

export function holdoutBlockDir(source, seedStart) {
  return `${V1_HOLDOUT_ROOT}/${source.slug}_seedblock_${seedStart}`;
}

export function holdoutArgs(source, seedStart) {
  const args = ["scripts/mesa-intervention-battery.mjs"];
  if (source.sourceKind === "reference") {
    args.push("--reference", source.reference);
  } else {
    args.push("--policy", source.policy);
  }
  args.push(
    "--policy-label", source.label,
    "--out", holdoutBlockDir(source, seedStart),
    "--seed-start", String(seedStart),
    "--seeds", String(V1_HOLDOUT_SEEDS),
    "--sensor-tier", V1_HOLDOUT_SENSOR_TIER,
    "--horizon", String(V1_HOLDOUT_HORIZON),
  );
  return args;
}

function quotePowerShellArg(arg) {
  if (/^[A-Za-z0-9_./:=+-]+$/.test(arg)) return arg;
  return `"${arg.replaceAll("`", "``").replaceAll('"', '`"')}"`;
}

export function holdoutCommandsPs1() {
  const lines = [
    "# Phase 3 capacity-relative one-wayness v1 holdout batteries",
    "# Frozen by docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V1_SLATE.md",
    "# 13 registered source rows x 4 seed starts = 52 source-bound 64-seed blocks.",
    "$ErrorActionPreference = 'Stop'",
    "",
  ];
  for (const source of V1_HOLDOUT_SOURCES) {
    lines.push(`# ${source.slug}`);
    for (const seedStart of V1_HOLDOUT_SEED_STARTS) {
      lines.push(["node", ...holdoutArgs(source, seedStart)].map(quotePowerShellArg).join(" "));
    }
    lines.push("");
  }
  return lines.join("\n");
}
