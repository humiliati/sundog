// scripts/lib/pvnp-phase3-v3-config.mjs
//
// Frozen constants for the Phase 3 capacity-relative one-wayness v3
// disclosure-robustness battery. Source of truth:
//   docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V3_SLATE.md (frozen 2026-06-04 local)
//
// v3 does NOT revise the population, thresholds, K/M, the 0.5 line, or the v2
// per-battery disclosure-consensus rule. It only aggregates the per-cell
// objective_conflict_status across N = 3 fresh disjoint batteries. Nothing here
// may be tuned after reading holdout blocks; doing so voids the run.

import { V1_HOLDOUT_ROOT, V1_HOLDOUT_SEED_STARTS } from "./pvnp-phase3-v1-config.mjs";
import {
  V2B_HOLDOUT_ROOT,
  V2B_HOLDOUT_SEED_STARTS,
  PRE_FREEZE_V2_HOLDOUT_ROOT,
  PRE_FREEZE_V2_HOLDOUT_SEED_STARTS,
} from "./pvnp-phase3-v2-config.mjs";

export const V3_RUN_ID = "phase3-capacity-one-wayness-v3";
export const V3_SCHEMA = "pvnp-phase3-capacity-one-wayness-v3";
export const V3_OUT = "results/pvnp/phase3-capacity-one-wayness-v3";

export const V3_N_FRESH = 3;

// Frozen fresh promotion-eligible batteries (next mechanical disjoint quartets
// after the seen batteries). Pre-registered before any scoring.
export const V3_FRESH_BATTERIES = Object.freeze([
  Object.freeze({
    id: "v3a",
    dataset: "v3a_holdout",
    root: "results/pvnp/phase3-capacity-one-wayness-v3a/phase4-intervention-battery",
    seedStarts: Object.freeze([180000, 190000, 200000, 210000]),
  }),
  Object.freeze({
    id: "v3b",
    dataset: "v3b_holdout",
    root: "results/pvnp/phase3-capacity-one-wayness-v3b/phase4-intervention-battery",
    seedStarts: Object.freeze([220000, 230000, 240000, 250000]),
  }),
  Object.freeze({
    id: "v3c",
    dataset: "v3c_holdout",
    root: "results/pvnp/phase3-capacity-one-wayness-v3c/phase4-intervention-battery",
    seedStarts: Object.freeze([260000, 270000, 280000, 290000]),
  }),
]);

// Already-scored batteries: REGRESSION / DIAGNOSTIC ONLY (slate Anti-P-Hack).
// They have been read and may not support a v3 promotion verdict.
export const V3_SEEN_BATTERIES = Object.freeze([
  Object.freeze({
    id: "v1",
    dataset: "v1_regression",
    root: V1_HOLDOUT_ROOT,
    seedStarts: V1_HOLDOUT_SEED_STARTS,
  }),
  Object.freeze({
    id: "pre_freeze_v2",
    dataset: "pre_freeze_v2_regression",
    root: PRE_FREEZE_V2_HOLDOUT_ROOT,
    seedStarts: PRE_FREEZE_V2_HOLDOUT_SEED_STARTS,
  }),
  Object.freeze({
    id: "v2b",
    dataset: "v2b_regression",
    root: V2B_HOLDOUT_ROOT,
    seedStarts: V2B_HOLDOUT_SEED_STARTS,
  }),
]);

// Frozen cross-battery disclosure-robustness status labels.
export const DISCLOSURE_ROBUSTNESS_STATUSES = Object.freeze([
  "robustly_disclosed",
  "disclosure_robustness_null",
  "not_applicable",
]);

export const V3_VERDICTS = Object.freeze({
  void_run: "void_run",
  falsified: "falsified_registered_cell",
  named_quarantine: "named_quarantine",
  disclosure_robustness_null: "named_quarantine_disclosure_robustness_null",
  posthoc_diagnostic: "posthoc_repair_diagnostic_named_quarantine",
  bounded_positive_strong: "bounded_positive_robust_disclosure_consensus_repair",
  bounded_positive_consensus_only: "bounded_positive_consensus_only_robust_disclosure_repair",
});
