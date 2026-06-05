// scripts/lib/pvnp-phase3-v2-config.mjs
//
// Frozen constants for the Phase 3 capacity-relative one-wayness v2 / v2b
// disclosure-consensus battery. Source of truth:
//   docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V2_SLATE.md
//
// v2 does NOT revise the population, thresholds, K/M, or block rule. Those stay
// imported from the frozen v1 config / config modules. This file holds only the
// v2-specific roots, seed starts, run id, schema, and the disclosure status
// labels frozen in the slate. Nothing here may be tuned after reading holdout
// blocks; doing so voids the run per the slate Freeze Rule.

import {
  V1_HOLDOUT_SEEDS,
  V1_HOLDOUT_SENSOR_TIER,
  V1_HOLDOUT_HORIZON,
  V1_HOLDOUT_SOURCES,
  holdoutBlockDirForRoot,
} from "./pvnp-phase3-v1-config.mjs";

export const V2_RUN_ID = "phase3-capacity-one-wayness-v2b";
export const V2_SCHEMA = "pvnp-phase3-capacity-one-wayness-v2";

// Promotion-eligible successor (v2b) output dir + raw-log root + seed starts.
export const V2B_OUT = "results/pvnp/phase3-capacity-one-wayness-v2b";
export const V2B_HOLDOUT_ROOT =
  "results/pvnp/phase3-capacity-one-wayness-v2b/phase4-intervention-battery";
export const V2B_HOLDOUT_SEED_STARTS = Object.freeze([140000, 150000, 160000, 170000]);

// Pre-freeze diagnostic battery (NOT promotion evidence per the slate).
export const PRE_FREEZE_V2_OUT = "results/pvnp/phase3-capacity-one-wayness-v2";
export const PRE_FREEZE_V2_HOLDOUT_ROOT =
  "results/pvnp/phase3-capacity-one-wayness-v2/phase4-intervention-battery";
export const PRE_FREEZE_V2_HOLDOUT_SEED_STARTS = Object.freeze([100000, 110000, 120000, 130000]);

// Frozen disclosure-consensus status labels (slate "Verifier v2 Candidate Rule"
// and Freeze Checklist). The v1 consensus_accept / consensus_reject /
// consensus_quarantine labels are unchanged; v2 adds ONLY these four.
export const OBJECTIVE_CONFLICT_STATUSES = Object.freeze([
  "conflict_consensus",
  "clean_consensus",
  "block_unstable_disclosure",
  "not_applicable",
]);

// Frozen verdict branches (slate "Verdict Branches"). Reported verbatim.
export const VERDICTS = Object.freeze({
  void_run: "void_run",
  falsified: "falsified_registered_cell",
  named_quarantine: "named_quarantine",
  posthoc_diagnostic: "posthoc_repair_diagnostic_named_quarantine",
  pre_freeze_diagnostic: "pre_freeze_holdout_diagnostic_named_quarantine",
  bounded_positive_strong: "bounded_positive_strong_disclosure_consensus_repair",
  bounded_positive_consensus_only: "bounded_positive_consensus_only_disclosure_repair",
});

// Mode -> fresh-holdout root / seed starts / default out dir. The "fresh"
// holdout is the promotion-evidence (or diagnostic) battery scored on top of the
// always-run v0 falsifier + v1 regression sets.
export function freshHoldoutConfig(mode) {
  if (mode === "pre-freeze") {
    return {
      dataset: "pre_freeze_v2_holdout",
      root: PRE_FREEZE_V2_HOLDOUT_ROOT,
      seedStarts: PRE_FREEZE_V2_HOLDOUT_SEED_STARTS,
      defaultOut: PRE_FREEZE_V2_OUT,
      forcedVerdict: VERDICTS.pre_freeze_diagnostic,
      promotionEligible: false,
    };
  }
  // default: promotion-eligible v2b successor
  return {
    dataset: "v2b_holdout",
    root: V2B_HOLDOUT_ROOT,
    seedStarts: V2B_HOLDOUT_SEED_STARTS,
    defaultOut: V2B_OUT,
    forcedVerdict: null,
    promotionEligible: true,
  };
}

// Emit the exact frozen holdout-runner commands for a fresh battery, mirroring
// the v1 holdout runner argument shape with only out-root + seed starts changed
// (slate "Fresh Corrected Holdout Battery"). Documentation/reproducibility only;
// the harness never launches the long battery.
function quotePowerShellArg(arg) {
  if (/^[A-Za-z0-9_./:=+-]+$/.test(arg)) return arg;
  return `"${arg.replaceAll("`", "``").replaceAll('"', '`"')}"`;
}

export function freshHoldoutCommandsPs1(root, seedStarts) {
  const lines = [
    "# Phase 3 capacity-relative one-wayness v2/v2b fresh holdout battery",
    "# Frozen by docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V2_SLATE.md",
    "# 13 registered source rows x 4 seed starts = 52 source-bound 64-seed blocks.",
    "# Runner shape unchanged from v1; only out-root + seed starts differ.",
    "$ErrorActionPreference = 'Stop'",
    "",
    `# sensor tier ${V1_HOLDOUT_SENSOR_TIER}; horizon ${V1_HOLDOUT_HORIZON}; ${V1_HOLDOUT_SEEDS} seeds/block`,
    ["node", "scripts/pvnp-phase3-v1-holdout.mjs", "--out-root", root,
      ...seedStarts.flatMap((s) => ["--seed-start", String(s)]), "--dry-run"]
      .map(quotePowerShellArg).join(" "),
    ["node", "scripts/pvnp-phase3-v1-holdout.mjs", "--out-root", root,
      ...seedStarts.flatMap((s) => ["--seed-start", String(s)]), "--jobs", "4"]
      .map(quotePowerShellArg).join(" "),
    "",
  ];
  // Per-source/per-seed block dirs, for auditing the planned 52-block shape.
  for (const source of V1_HOLDOUT_SOURCES) {
    lines.push(`# ${source.slug}`);
    for (const seedStart of seedStarts) {
      lines.push(`#   ${holdoutBlockDirForRoot(source, seedStart, root)}`);
    }
  }
  return lines.join("\n");
}
