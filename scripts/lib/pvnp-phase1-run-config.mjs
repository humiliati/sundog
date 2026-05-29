// scripts/lib/pvnp-phase1-run-config.mjs
//
// Shared run-slate configuration for the P-vs-NP Phase 1 harness.

import { BASIN_FAMILIES, PROBE_NOISE_TIERS, PROMISE_BOUNDS } from "./pvnp-phase1-env-core.mjs";

const COMMON = Object.freeze({
  basin_families: BASIN_FAMILIES,
  probe_noise_tiers: PROBE_NOISE_TIERS,
  promise_bounds: PROMISE_BOUNDS,
  horizon: 128,
  max_action_step: 0.025,
  m_min_candidate_grid: [0.02, 0.04, 0.06],
  verifier_access_declaration: {
    may_read: [
      "candidate_policy_id",
      "policy_class",
      "sigma",
      "promise_params",
      "checker_thresholds",
      "trace_commitments",
      "source_observations",
    ],
    may_not_read: [
      "hidden_state",
      "basin_params",
      "latent_field",
      "decoy_params",
      "ground_truth_labels",
      "post_result_thresholds",
    ],
    forbidden_tokens_grep: [
      "ground_truth_labels",
      "B_theta",
      "F_theta",
      "hidden_state",
      "basin_params",
      "latent_field",
      "decoy_params",
    ],
  },
});

export const RUN_CONFIGS = Object.freeze({
  v0: Object.freeze({
    run_id: "phase1-toy-verifier-v0",
    schema_suffix: "v0",
    spec_path: "docs/pvnp/PHASE1_TOY_VERIFIER_SPEC.md",
    slate_path: "docs/pvnp/PHASE1_V0_SLATE.md",
    receipt_path: "docs/pvnp/receipts/2026-05-28_phase1_toy_verifier_v0.md",
    splits: [
      { split: "calibration", count: 64, seedPrefix: "pvnp-v0-cal", inPromise: true },
      { split: "train", count: 256, seedPrefix: "pvnp-v0-train", inPromise: true },
      { split: "verification", count: 256, seedPrefix: "pvnp-v0-verify", inPromise: true },
      { split: "falsifier", count: 256, seedPrefix: "pvnp-v0-fals", inPromise: false },
    ],
    ...COMMON,
  }),
  v1: Object.freeze({
    run_id: "phase1-toy-verifier-v1",
    schema_suffix: "v1",
    spec_path: "docs/pvnp/PHASE1_TOY_VERIFIER_SPEC.md",
    slate_path: "docs/pvnp/PHASE1_V1_SLATE.md",
    receipt_path: "docs/pvnp/receipts/2026-05-28_phase1_toy_verifier_v1.md",
    splits: [
      { split: "calibration", count: 64, seedPrefix: "pvnp-v1-cal", inPromise: true },
      { split: "train", count: 256, seedPrefix: "pvnp-v1-train", inPromise: true },
      { split: "verification", count: 256, seedPrefix: "pvnp-v1-verify", inPromise: true },
      { split: "falsifier", count: 256, seedPrefix: "pvnp-v1-fals", inPromise: false },
    ],
    ...COMMON,
  }),
  v2: Object.freeze({
    run_id: "phase1-toy-verifier-v2",
    schema_suffix: "v2",
    spec_path: "docs/pvnp/PHASE1_TOY_VERIFIER_SPEC.md",
    slate_path: "docs/pvnp/PHASE1_V2_SLATE.md",
    receipt_path: "docs/pvnp/receipts/2026-05-28_phase1_toy_verifier_v2.md",
    splits: [
      { split: "calibration", count: 64, seedPrefix: "pvnp-v2-cal", inPromise: true },
      { split: "train", count: 256, seedPrefix: "pvnp-v2-train", inPromise: true },
      { split: "verification", count: 256, seedPrefix: "pvnp-v2-verify", inPromise: true },
      { split: "falsifier", count: 256, seedPrefix: "pvnp-v2-fals", inPromise: false },
    ],
    ...COMMON,
  }),
  v3: Object.freeze({
    run_id: "phase1-toy-verifier-v3",
    schema_suffix: "v3",
    spec_path: "docs/pvnp/PHASE1_TOY_VERIFIER_SPEC.md",
    slate_path: "docs/pvnp/PHASE1_V3_SLATE.md",
    receipt_path: "docs/pvnp/receipts/2026-05-28_phase1_toy_verifier_v3.md",
    splits: [
      { split: "calibration", count: 64, seedPrefix: "pvnp-v3-cal", inPromise: true },
      { split: "train", count: 256, seedPrefix: "pvnp-v3-train", inPromise: true },
      { split: "verification", count: 256, seedPrefix: "pvnp-v3-verify", inPromise: true },
      { split: "falsifier", count: 256, seedPrefix: "pvnp-v3-fals", inPromise: false },
    ],
    ...COMMON,
  }),
  v4: Object.freeze({
    run_id: "phase1-toy-verifier-v4",
    schema_suffix: "v4",
    spec_path: "docs/pvnp/PHASE1_TOY_VERIFIER_SPEC.md",
    slate_path: "docs/pvnp/PHASE1_V4_SLATE.md",
    receipt_path: "docs/pvnp/receipts/2026-05-28_phase1_toy_verifier_v4.md",
    splits: [
      { split: "calibration", count: 64, seedPrefix: "pvnp-v4-cal", inPromise: true },
      { split: "train", count: 256, seedPrefix: "pvnp-v4-train", inPromise: true },
      { split: "verification", count: 256, seedPrefix: "pvnp-v4-verify", inPromise: true },
      { split: "falsifier", count: 256, seedPrefix: "pvnp-v4-fals", inPromise: false },
    ],
    ...COMMON,
  }),
});

export function inferPhase1VersionFromPath(runDir) {
  const s = String(runDir);
  if (s.includes("v4")) return "v4";
  if (s.includes("v3")) return "v3";
  if (s.includes("v2")) return "v2";
  if (s.includes("v1")) return "v1";
  return "v0";
}

export function getPhase1RunConfig(runDirOrVersion) {
  const version = ["v0", "v1", "v2", "v3", "v4"].includes(runDirOrVersion)
    ? runDirOrVersion
    : inferPhase1VersionFromPath(runDirOrVersion);
  return RUN_CONFIGS[version];
}

export function phase1Schema(prefix, version) {
  return `pvnp-phase1-${prefix}-${version}`;
}
