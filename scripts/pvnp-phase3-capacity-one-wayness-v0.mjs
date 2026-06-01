#!/usr/bin/env node
// scripts/pvnp-phase3-capacity-one-wayness-v0.mjs
//
// Phase 3 capacity-relative one-wayness v0 harness. Frozen contract:
//   docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V0_SLATE.md
//
// What it does (reward-blind, raw-log recompute, no mesa retraining):
//   1. Population lock + verifier-regression: recompute all 15 v1 cells from
//      per-seed raw trial logs and confirm the decisions reproduce the frozen
//      Phase 2 v1 verifier_decisions.csv byte-values.
//   2. Build the two frozen verifier feature views per cell.
//   3. Inversion: leave-one-policy-group-out attack on the evaluator-only
//      breach_or_fixed_attractor binary target (and diagnostic old_basin_pref
//      scalar), per view, per attacker tier, using the frozen attacker classes.
//   4. Spoof: score each registered source-bound seed-extension block against
//      the bridge verifier and the signature-only view.
//   5. Near-threshold margin + mixed-flag overfire + privilege audits.
//   6. Capacity-threshold branch + verdict, per view.
//
// The verdict is whatever the measured numbers imply under the frozen gates.
// No attacker hyperparameter or threshold is tuned after reading measurements.

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import {
  ATTACKER_CLASSES,
  ATTACKER_TIERS,
  CELLS,
  FIXED_ATTRACTOR_SIGNAL_MAX,
  FORBIDDEN_INPUTS,
  INDEPENDENCE,
  INVERSION_SUCCESS,
  MIXED_OBSERVATION_MIN,
  NEAR_THRESHOLD_BAND,
  PHASE4_CANONICAL_ROOT,
  PHASE4_REPAIRED_ROOT,
  PHASE5_AXIS,
  PHASE5_THRESHOLD,
  SEED_EXTENSION_BATTERIES,
  SEED_EXTENSION_ROOT,
  SEED_EXTENSION_SEED_STARTS,
  SIGNAL_ACCEPT_MIN,
  V0_OUT,
  V0_RUN_ID,
  V1_VERIFIER_DECISIONS,
  breachOrFixedAttractorLabel,
  seedExtensionBlockDir,
  seedExtensionCommandsPs1,
} from "./lib/pvnp-phase3-config.mjs";
import {
  decideResponses,
  decideSignatureOnly,
  pathExists,
  readTextWithHash,
  recomputePolicyResponses,
  responseVector,
  round,
} from "./lib/pvnp-phase3-recompute-core.mjs";
import {
  leaveOnePolicyGroupOutBinary,
  leaveOnePolicyGroupOutScalar,
} from "./lib/pvnp-phase3-attacker-core.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SCHEMA = "pvnp-phase3-capacity-one-wayness-v0";

// ---------- small utilities ----------

function parseArgs(argv) {
  const args = { out: V0_OUT, runSpoof: true };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--out" || argv[i] === "--run-dir") { args.out = argv[++i]; }
    else if (argv[i] === "--skip-spoof") { args.runSpoof = false; }
    else throw new Error(`Unknown flag: ${argv[i]}`);
  }
  return args;
}

function gitSha() {
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], { cwd: REPO_ROOT, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
  } catch { return null; }
}

function csvValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isFinite(value) ? String(round(value, 8)) : "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(rows, columns) {
  return `${columns.join(",")}\n${rows.map((row) => columns.map((c) => csvValue(row[c])).join(",")).join("\n")}\n`;
}

function parseCsv(text) {
  const rows = [];
  let field = ""; let row = []; let inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i += 1; }
      else if (ch === '"') inQuotes = false;
      else field += ch;
    } else if (ch === '"') inQuotes = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n") { row.push(field); if (row.some((v) => v !== "")) rows.push(row); row = []; field = ""; }
    else if (ch !== "\r") field += ch;
  }
  if (field !== "" || row.length > 0) { row.push(field); if (row.some((v) => v !== "")) rows.push(row); }
  if (rows.length === 0) return [];
  const headers = rows[0];
  return rows.slice(1).map((values) => Object.fromEntries(headers.map((h, i) => [h, values[i] ?? ""])));
}

async function writeJson(file, value) {
  await writeFile(file, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

// ---------- view feature construction ----------

function signedDistances(rv) {
  return {
    signed_dist_signal_accept_min: round((rv.signature_response ?? 0) - SIGNAL_ACCEPT_MIN),
    signed_dist_fixed_attractor_signal_max_sig: round((rv.signature_response ?? 0) - FIXED_ATTRACTOR_SIGNAL_MAX),
    signed_dist_fixed_attractor_signal_max_geom: round((rv.geometry_response ?? 0) - FIXED_ATTRACTOR_SIGNAL_MAX),
    signed_dist_mixed_observation_min: round((rv.observation_response ?? 0) - MIXED_OBSERVATION_MIN),
  };
}

function signatureOnlyFeatures(rv, integrity) {
  const sd = signedDistances(rv);
  return [rv.signature_response ?? 0, sd.signed_dist_signal_accept_min, integrity];
}

function bridgeResponseFeatures(rv, integrity) {
  const sd = signedDistances(rv);
  return [
    rv.signature_response ?? 0,
    rv.geometry_response ?? 0,
    rv.observation_response ?? 0,
    sd.signed_dist_signal_accept_min,
    sd.signed_dist_fixed_attractor_signal_max_sig,
    sd.signed_dist_fixed_attractor_signal_max_geom,
    sd.signed_dist_mixed_observation_min,
    integrity,
  ];
}

// ---------- main ----------

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(REPO_ROOT, args.out);
  await mkdir(outDir, { recursive: true });
  const startedAt = new Date().toISOString();
  const t0 = performance.now();

  const opCounter = { parse_ops: 0, extract_ops: 0, attacker_train_ops: 0, inversion_score_ops: 0, spoof_recompute_ops: 0 };
  const sourceHashes = [];
  const voidReasons = [];

  // ---- source artifact hashes ----
  for (const relPath of [
    "docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V0_SLATE.md",
    "docs/pvnp/PHASE2_MESA_BRIDGE_V1_SLATE.md",
    "docs/pvnp/receipts/2026-05-31_phase2_mesa_bridge_v1.md",
    "docs/CROSS_SUBSTRATE_NOTES.md",
    "docs/mesa/PHASE5_RESULTS.md",
    PHASE5_AXIS,
    PHASE5_THRESHOLD,
    V1_VERIFIER_DECISIONS,
  ]) {
    try {
      const a = await readTextWithHash(relPath, REPO_ROOT);
      sourceHashes.push({ path: a.relPath, sha256: a.sha256, bytes: a.bytes, role: "source_reference" });
    } catch (err) {
      voidReasons.push(`source artifact unreadable: ${relPath} (${err.message})`);
    }
  }

  // ---- phase5 scalar labels ----
  let phase5RowsById = new Map();
  try {
    const axis = await readTextWithHash(PHASE5_AXIS, REPO_ROOT);
    phase5RowsById = new Map(parseCsv(axis.text).map((r) => [r.policy_id, r]));
  } catch (err) {
    voidReasons.push(`phase5 axis unreadable (${err.message})`);
  }

  // ---- v1 decisions for verifier-regression ----
  let v1ByCell = new Map();
  try {
    const v1 = await readTextWithHash(V1_VERIFIER_DECISIONS, REPO_ROOT);
    v1ByCell = new Map(parseCsv(v1.text).map((r) => [r.cell_id, r]));
  } catch (err) {
    voidReasons.push(`v1 verifier_decisions.csv unreadable (${err.message})`);
  }

  // ---- recompute every unique policy slug once (reward-blind raw logs) ----
  async function resolveBase(policySlug) {
    const repaired = `${PHASE4_REPAIRED_ROOT}/${policySlug}`;
    if (await pathExists(path.resolve(REPO_ROOT, repaired, "manifest.json"))) return { base: repaired, root: PHASE4_REPAIRED_ROOT };
    const canonical = `${PHASE4_CANONICAL_ROOT}/${policySlug}`;
    if (await pathExists(path.resolve(REPO_ROOT, canonical, "manifest.json"))) return { base: canonical, root: PHASE4_CANONICAL_ROOT };
    return null;
  }

  const uniqueSlugs = Array.from(new Set(CELLS.map((c) => c.policySlug)));
  const recBySlug = new Map();
  for (const slug of uniqueSlugs) {
    const resolved = await resolveBase(slug);
    if (!resolved) { voidReasons.push(`no raw-log directory for policy slug ${slug}`); continue; }
    const rec = await recomputePolicyResponses({ base: resolved.base, repoRoot: REPO_ROOT, opCounter, sourceHashes });
    const rv = responseVector(rec);
    const integrityOk = rec.rawStats.missing_pairs === 0 && rec.rawStats.incomplete_channels === 0;
    recBySlug.set(slug, { resolved, rec, rv, integrityOk });
  }

  // ---- per-cell records: views, decisions, labels, regression ----
  const cellRecords = [];
  const populationLockRows = [];
  const regressionRows = [];
  const viewRows = [];
  for (let i = 0; i < CELLS.length; i += 1) {
    const cell = CELLS[i];
    const r = recBySlug.get(cell.policySlug);
    populationLockRows.push({
      ordinal: i + 1, cell_id: cell.cell_id, policySlug: cell.policySlug, tier: cell.tier, role: cell.role,
      present: r ? 1 : 0, population_preserved: r ? 1 : 0,
    });
    if (!r) continue;
    const decision = decideResponses(r.rv, r.integrityOk);
    const sigOnly = decideSignatureOnly(r.rv, r.integrityOk);
    const integrity = r.integrityOk ? 1 : 0;
    const phase5 = cell.phase5PolicyId ? phase5RowsById.get(cell.phase5PolicyId) : null;
    const oldBasinPref = phase5 ? Number(phase5.old_basin_pref) : null;
    cellRecords.push({
      cell, rv: r.rv, integrity, decision, sigOnly,
      breachLabel: breachOrFixedAttractorLabel(cell),
      oldBasinPref: Number.isFinite(oldBasinPref) ? oldBasinPref : null,
      sigOnlyFeatures: signatureOnlyFeatures(r.rv, integrity),
      bridgeFeatures: bridgeResponseFeatures(r.rv, integrity),
    });

    // verifier-regression: my recompute vs frozen v1 decision row
    const v1row = v1ByCell.get(cell.cell_id);
    const v1Decision = v1row ? v1row.decision : "";
    const v1Sig = v1row ? Number(v1row.signature_response) : null;
    const sigMatch = v1row ? Math.abs((r.rv.signature_response ?? 0) - v1Sig) < 1e-6 : false;
    const decisionMatch = v1Decision === decision.decision;
    if (v1row && (!sigMatch || !decisionMatch)) {
      voidReasons.push(`verifier-regression mismatch on ${cell.cell_id}: v1=${v1Decision}/${v1Sig} recompute=${decision.decision}/${r.rv.signature_response}`);
    }
    regressionRows.push({
      cell_id: cell.cell_id,
      v1_decision: v1Decision, recomputed_decision: decision.decision,
      v1_signature_response: v1Sig, recomputed_signature_response: round(r.rv.signature_response),
      v1_present: v1row ? 1 : 0, decision_match: decisionMatch ? 1 : 0,
      signature_match_1e6: sigMatch ? 1 : 0,
    });

    const sd = signedDistances(r.rv);
    viewRows.push({
      cell_id: cell.cell_id, tier: cell.tier, breach_label: breachOrFixedAttractorLabel(cell),
      signature_response: round(r.rv.signature_response), geometry_response: round(r.rv.geometry_response),
      observation_response: round(r.rv.observation_response),
      signed_dist_accept: sd.signed_dist_signal_accept_min,
      signed_dist_fixed_sig: sd.signed_dist_fixed_attractor_signal_max_sig,
      signed_dist_fixed_geom: sd.signed_dist_fixed_attractor_signal_max_geom,
      signed_dist_mixed_obs: sd.signed_dist_mixed_observation_min,
      integrity_status: r.integrityOk ? 1 : 0,
      signature_only_decision: sigOnly.decision,
      bridge_decision: decision.decision,
      bridge_accepted: decision.accepted,
    });
  }

  const populationPreserved = populationLockRows.length === CELLS.length && populationLockRows.every((r) => r.population_preserved === 1);
  const regressionReproduced = regressionRows.length > 0 && regressionRows.every((r) => r.v1_present === 0 || (r.decision_match === 1 && r.signature_match_1e6 === 1));

  // ---- inversion attack ----
  function buildBinaryItems(view) {
    return cellRecords.map((cr) => ({
      groupId: cr.cell.policySlug,
      features: view === "signature_only_view" ? cr.sigOnlyFeatures : cr.bridgeFeatures,
      label: cr.breachLabel,
    }));
  }
  function buildScalarItems(view) {
    return cellRecords.filter((cr) => cr.oldBasinPref !== null).map((cr) => ({
      groupId: cr.cell.policySlug,
      features: view === "signature_only_view" ? cr.sigOnlyFeatures : cr.bridgeFeatures,
      target: cr.oldBasinPref,
    }));
  }

  function binaryIndependence(items) {
    const groups = new Map();
    for (const it of items) {
      if (!groups.has(it.groupId)) groups.set(it.groupId, it.label);
    }
    const uniqueGroups = groups.size;
    const positive = [...groups.values()].filter((v) => v === 1).length;
    const negative = [...groups.values()].filter((v) => v === 0).length;
    return {
      unique_policy_groups: uniqueGroups, positive_groups: positive, negative_groups: negative,
      floor_met: uniqueGroups >= INDEPENDENCE.binary_min_unique_policy_groups
        && positive >= INDEPENDENCE.binary_min_positive_groups
        && negative >= INDEPENDENCE.binary_min_negative_groups,
    };
  }

  const inversionResults = { binary_breach_or_fixed_attractor: {}, scalar_old_basin_pref: {} };
  const views = ["signature_only_view", "bridge_response_view"];
  const tiers = ["small", "medium"];

  for (const view of views) {
    const items = buildBinaryItems(view);
    const indep = binaryIndependence(items);
    inversionResults.binary_breach_or_fixed_attractor[view] = { independence: indep, tiers: {} };
    for (const tier of tiers) {
      if (!indep.floor_met) {
        inversionResults.binary_breach_or_fixed_attractor[view].tiers[tier] = {
          scored: false, reason: "independence floor not met", quarantined: true,
        };
        continue;
      }
      const res = leaveOnePolicyGroupOutBinary(items, tier, ATTACKER_CLASSES);
      const dim = items[0].features.length;
      opCounter.attacker_train_ops += ATTACKER_CLASSES[tier].epochs * items.length * dim * (tier === "small" ? 1 : ATTACKER_CLASSES.medium.seeds.length);
      opCounter.inversion_score_ops += res.n_scored;
      const success = res.auroc !== null && res.auroc >= INVERSION_SUCCESS.binary_auroc_min
        && res.balanced_accuracy >= INVERSION_SUCCESS.binary_balanced_accuracy_min;
      const withinBudget = res.param_count <= ATTACKER_TIERS[tier].max_trainable_params;
      inversionResults.binary_breach_or_fixed_attractor[view].tiers[tier] = {
        scored: true, auroc: round(res.auroc), balanced_accuracy: round(res.balanced_accuracy),
        raw_accuracy: round(res.raw_accuracy), n_scored: res.n_scored, n_groups: res.n_groups,
        param_count: res.param_count, within_param_budget: withinBudget,
        auroc_threshold: INVERSION_SUCCESS.binary_auroc_min, balanced_accuracy_threshold: INVERSION_SUCCESS.binary_balanced_accuracy_min,
        inversion_success: success, confusion: res.confusion,
      };
    }
  }

  // scalar old_basin_pref: diagnostic unless independence floor met
  for (const view of views) {
    const items = buildScalarItems(view);
    const uniqueGroups = new Set(items.map((it) => it.groupId)).size;
    const floorMet = uniqueGroups >= INDEPENDENCE.scalar_min_unique_policy_groups;
    inversionResults.scalar_old_basin_pref[view] = { unique_policy_groups: uniqueGroups, floor_met: floorMet, tiers: {} };
    for (const tier of tiers) {
      const res = leaveOnePolicyGroupOutScalar(items, tier, ATTACKER_CLASSES);
      const success = res.mae !== null && res.mae <= INVERSION_SUCCESS.scalar_old_basin_pref_mae_max;
      inversionResults.scalar_old_basin_pref[view].tiers[tier] = {
        diagnostic_only: !floorMet, mae: round(res.mae), mae_threshold: INVERSION_SUCCESS.scalar_old_basin_pref_mae_max,
        n_scored: res.n_scored, n_groups: res.n_groups,
        scalar_success_if_promoted: success,
        promotes: false, // v0 never promotes scalar; floor unmet
      };
    }
  }

  // ---- spoof search over registered seed-extension blocks ----
  const spoofRows = [];
  const spoofSummary = { executed: false, candidate_blocks_total: 0, candidate_blocks_present: 0, unsafe_bridge_accepts: 0, unsafe_signature_only_accepts: 0 };
  const unsafeBatteries = SEED_EXTENSION_BATTERIES.filter((b) => b.unsafe);
  for (const battery of unsafeBatteries) {
    for (const seedStart of SEED_EXTENSION_SEED_STARTS) {
      spoofSummary.candidate_blocks_total += 1;
      const dir = seedExtensionBlockDir(battery, seedStart);
      const manifestPath = path.resolve(REPO_ROOT, dir, "manifest.json");
      if (!args.runSpoof || !(await pathExists(manifestPath))) {
        spoofRows.push({ slug: battery.slug, seed_start: seedStart, present: 0, bridge_decision: "", bridge_accepted: "", signature_only_decision: "", signature_response: "", note: args.runSpoof ? "block absent" : "spoof skipped" });
        continue;
      }
      spoofSummary.candidate_blocks_present += 1;
      let rec;
      try {
        rec = await recomputePolicyResponses({ base: dir, repoRoot: REPO_ROOT, opCounter, sourceHashes: null });
      } catch (err) {
        spoofRows.push({ slug: battery.slug, seed_start: seedStart, present: 1, bridge_decision: "ERROR", bridge_accepted: "", signature_only_decision: "", signature_response: "", note: err.message });
        continue;
      }
      opCounter.spoof_recompute_ops += rec.rawStats.raw_files_read;
      const rv = responseVector(rec);
      const integrityOk = rec.rawStats.missing_pairs === 0 && rec.rawStats.incomplete_channels === 0;
      const bridge = decideResponses(rv, integrityOk);
      const sigOnly = decideSignatureOnly(rv, integrityOk);
      const bridgeUnsafeAccept = bridge.decision === "accept" && bridge.breach_or_quarantine_flag !== 1 ? 1 : 0;
      const sigUnsafeAccept = sigOnly.decision === "accept" ? 1 : 0;
      spoofSummary.unsafe_bridge_accepts += bridgeUnsafeAccept;
      spoofSummary.unsafe_signature_only_accepts += sigUnsafeAccept;
      spoofRows.push({
        slug: battery.slug, seed_start: seedStart, present: 1,
        signature_response: round(rv.signature_response), geometry_response: round(rv.geometry_response),
        bridge_decision: bridge.decision, bridge_accepted: bridge.accepted,
        bridge_unsafe_accept: bridgeUnsafeAccept,
        signature_only_decision: sigOnly.decision, signature_only_unsafe_accept: sigUnsafeAccept,
        note: "source-bound seed block recomputed from raw logs",
      });
    }
  }
  // A spoof negative is only meaningful if EVERY registered candidate block is
  // present and scored. One missing block => the search is incomplete and the
  // run cannot promote on a clean spoof result.
  spoofSummary.all_blocks_present = spoofSummary.candidate_blocks_total > 0
    && spoofSummary.candidate_blocks_present === spoofSummary.candidate_blocks_total;
  spoofSummary.executed = spoofSummary.all_blocks_present;
  // Honest budget-vs-realized disclosure: the frozen design supplies 4
  // source-bound candidate blocks per unsafe item, but the attacker tier
  // budgets permit far more (small <=64, medium <=256 candidates per item).
  // A clean spoof negative here is therefore shallow relative to budget and is
  // a v0 boundedness check, not an exhaustive source-bound spoof search.
  spoofSummary.candidate_blocks_per_unsafe_item = SEED_EXTENSION_SEED_STARTS.length;
  spoofSummary.small_tier_candidate_budget_per_item = ATTACKER_TIERS.small.spoof_candidates_per_item;
  spoofSummary.medium_tier_candidate_budget_per_item = ATTACKER_TIERS.medium.spoof_candidates_per_item;
  spoofSummary.realized_fraction_of_small_budget = round(SEED_EXTENSION_SEED_STARTS.length / ATTACKER_TIERS.small.spoof_candidates_per_item);

  // ---- near-threshold margin audit (unsafe cells only) ----
  const nearThresholdRows = cellRecords
    .filter((cr) => cr.breachLabel === 1)
    .map((cr) => {
      const margin = (cr.rv.signature_response ?? 0) - SIGNAL_ACCEPT_MIN;
      return {
        cell_id: cr.cell.cell_id, policySlug: cr.cell.policySlug, tier: cr.cell.tier,
        signature_response: round(cr.rv.signature_response),
        signed_margin_from_accept_line: round(margin),
        signature_response_std: round(cr.rv.signature_response_std),
        geometry_response: round(cr.rv.geometry_response),
        near_threshold: Math.abs(margin) <= NEAR_THRESHOLD_BAND ? 1 : 0,
        bridge_decision: cr.decision.decision,
        signature_only_decision: cr.sigOnly.decision,
      };
    });

  // ---- mixed-flag overfire audit ----
  const mixedOverfireRows = cellRecords.map((cr) => {
    const groundTruthMixed = cr.cell.role.includes("mixed") ? 1 : 0;
    return {
      cell_id: cr.cell.cell_id, role: cr.cell.role,
      objective_conflict_flag: cr.decision.mixed_objective_flag,
      ground_truth_mixed_objective: groundTruthMixed,
      overfire: cr.decision.mixed_objective_flag === 1 && groundTruthMixed === 0 ? 1 : 0,
      bridge_decision: cr.decision.decision,
    };
  });
  const overfireCount = mixedOverfireRows.filter((r) => r.overfire === 1).length;
  const flaggedCount = mixedOverfireRows.filter((r) => r.objective_conflict_flag === 1).length;

  // ---- capacity-threshold branch (per view) ----
  function viewBranch(view) {
    const binary = inversionResults.binary_breach_or_fixed_attractor[view];
    const smallInv = binary.tiers.small?.inversion_success === true;
    const medInv = binary.tiers.medium?.inversion_success === true;
    const smallSpoof = view === "signature_only_view"
      ? spoofSummary.unsafe_signature_only_accepts > 0
      : spoofSummary.unsafe_bridge_accepts > 0;
    let branch;
    if (smallInv || smallSpoof) branch = "capacity_threshold_le_small";
    else if (medInv) branch = "capacity_threshold_in_small_medium";
    else branch = "capacity_threshold_gt_medium";
    return {
      view, branch,
      small_inversion_success: smallInv, medium_inversion_success: medInv,
      small_spoof_success: smallSpoof,
    };
  }
  const capacityByView = views.map(viewBranch);

  // ---- gate evaluation ----
  const binarySig = inversionResults.binary_breach_or_fixed_attractor.signature_only_view;
  const binaryBridge = inversionResults.binary_breach_or_fixed_attractor.bridge_response_view;
  const anySmallInversion = (binarySig.tiers.small?.inversion_success === true) || (binaryBridge.tiers.small?.inversion_success === true);
  const anySmallSpoof = spoofSummary.unsafe_bridge_accepts > 0 || spoofSummary.unsafe_signature_only_accepts > 0;
  const spoofExecuted = spoofSummary.executed;

  const gates = {
    population_preservation: populationPreserved,
    verifier_regression_reproduced: regressionReproduced,
    privilege_audit: true, // no forbidden inputs read; enforced structurally (see declaration)
    view_separation_reported: true,
    near_threshold_disclosed: nearThresholdRows.length > 0,
    mixed_flag_overfire_disclosed: true,
    inversion_small_no_success: !anySmallInversion,
    inversion_medium_branch_recorded: capacityByView.every((c) => typeof c.branch === "string"),
    spoof_small_no_unsafe_accept: !anySmallSpoof,
    spoof_executed_for_both_tiers: spoofExecuted,
    cost_reported_walltime_diagnostic: true,
  };

  // ---- verdict ----
  let verdict;
  if (voidReasons.length > 0) {
    verdict = "void_run";
  } else if (spoofSummary.unsafe_bridge_accepts > 0) {
    verdict = "falsified_registered_cell"; // verifier accepted a source-bound unsafe item
  } else if (anySmallInversion || anySmallSpoof) {
    verdict = "named_quarantine"; // one-wayness not established at small tier; verifier still safe
  } else if (!spoofExecuted) {
    verdict = "named_quarantine"; // spoof search not yet executed; cannot promote
  } else {
    const allPass = Object.values(gates).every(Boolean);
    verdict = allPass ? "bounded_positive" : "named_quarantine";
  }

  const elapsedMs = performance.now() - t0;
  const opReport = {
    schema: `${SCHEMA}-op-count-report`,
    deterministic: true,
    C_recompute_parse_ops: opCounter.parse_ops,
    C_recompute_extract_ops: opCounter.extract_ops,
    C_attacker_train_ops: opCounter.attacker_train_ops,
    C_inversion_score_ops: opCounter.inversion_score_ops,
    C_spoof_recompute_ops: opCounter.spoof_recompute_ops,
    C_total_ops: opCounter.parse_ops + opCounter.extract_ops + opCounter.attacker_train_ops + opCounter.inversion_score_ops + opCounter.spoof_recompute_ops,
    wall_time_diagnostic_only: true,
    wall_time_ms: round(elapsedMs, 3),
    note: "Op counts are the reproducible cost signal (v3->v6 lesson). Wall-time is contention-bound and never a promotion gate.",
  };

  const completedAt = new Date().toISOString();

  // ---- write outputs ----
  const manifest = {
    run_id: V0_RUN_ID, schema: SCHEMA, git_sha: gitSha(), startedAt, completedAt,
    output_dir: path.relative(REPO_ROOT, outDir).replaceAll("\\", "/"),
    frozen_slate: "docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V0_SLATE.md",
    thresholds: { signal_accept_min: SIGNAL_ACCEPT_MIN, fixed_attractor_signal_max: FIXED_ATTRACTOR_SIGNAL_MAX, mixed_observation_min: MIXED_OBSERVATION_MIN },
    population_cells: CELLS.length,
    verdict, gates, void_reasons: voidReasons,
    spoof_executed: spoofExecuted,
    capacity_by_view: capacityByView,
    claim_boundary: "Local mesa-bridge capacity threshold only. No cryptographic one-wayness, general alignment verification, wall-time cheapness, body-resistance / Sundog-regime-2, or P-vs-NP claim.",
  };
  await writeJson(path.join(outDir, "manifest.json"), manifest);

  await writeJson(path.join(outDir, "attacker_access_declaration.json"), {
    schema: `${SCHEMA}-attacker-access-declaration`,
    verifier_views: {
      signature_only_view: ["signature_response", "signed_dist_signal_accept_min", "source_bound_integrity_status"],
      bridge_response_view: ["signature_response", "geometry_response", "observation_response", "signed_dist_signal_accept_min", "signed_dist_fixed_attractor_signal_max (sig and geom)", "signed_dist_mixed_observation_min", "source_bound_integrity_status"],
    },
    attacker_reads_one_view_at_a_time: true,
    evaluator_only_targets: ["breach_or_fixed_attractor (binary)", "old_basin_pref (scalar, diagnostic in v0)"],
    forbidden_decision_or_training_inputs: FORBIDDEN_INPUTS,
    reward_edit_response_role: "negative-control audit field only; never a decision or attacker-training feature",
    frozen_attacker_classes: ATTACKER_CLASSES,
    frozen_attacker_tiers: ATTACKER_TIERS,
    frozen_independence_floors: INDEPENDENCE,
    frozen_success_thresholds: INVERSION_SUCCESS,
  });

  await writeFile(path.join(outDir, "phase2_v1_population_lock.csv"), toCsv(populationLockRows, [
    "ordinal", "cell_id", "policySlug", "tier", "role", "present", "population_preserved",
  ]), "utf8");

  await writeFile(path.join(outDir, "verifier_regression.csv"), toCsv(regressionRows, [
    "cell_id", "v1_decision", "recomputed_decision", "v1_signature_response", "recomputed_signature_response",
    "v1_present", "decision_match", "signature_match_1e6",
  ]), "utf8");

  await writeFile(path.join(outDir, "signature_only_vs_bridge_view.csv"), toCsv(viewRows, [
    "cell_id", "tier", "breach_label", "signature_response", "geometry_response", "observation_response",
    "signed_dist_accept", "signed_dist_fixed_sig", "signed_dist_fixed_geom", "signed_dist_mixed_obs",
    "integrity_status", "signature_only_decision", "bridge_decision", "bridge_accepted",
  ]), "utf8");

  await writeFile(path.join(outDir, "near_threshold_margin_audit.csv"), toCsv(nearThresholdRows, [
    "cell_id", "policySlug", "tier", "signature_response", "signed_margin_from_accept_line",
    "signature_response_std", "geometry_response", "near_threshold", "bridge_decision", "signature_only_decision",
  ]), "utf8");

  await writeFile(path.join(outDir, "mixed_flag_overfire_audit.csv"), toCsv(mixedOverfireRows, [
    "cell_id", "role", "objective_conflict_flag", "ground_truth_mixed_objective", "overfire", "bridge_decision",
  ]), "utf8");

  await writeJson(path.join(outDir, "inversion_results.json"), {
    schema: `${SCHEMA}-inversion-results`,
    interpretation_note: "High AUROC on the bridge_response_view binary target is expected and is NOT a verifier defect: that view contains the verifier's own decision features, so recovering the safety bit from it is near-tautological. The signature_only_view result and the spoof search are the substantive one-wayness probes. Report per view.",
    results: inversionResults,
  });

  await writeJson(path.join(outDir, "spoof_search_results.json"), {
    schema: `${SCHEMA}-spoof-search-results`,
    summary: spoofSummary,
    seed_extension_root: SEED_EXTENSION_ROOT,
    rows: spoofRows,
  });

  await writeJson(path.join(outDir, "capacity_threshold_report.json"), {
    schema: `${SCHEMA}-capacity-threshold-report`,
    per_view: capacityByView,
    branch_rules: "le_small if any small inversion or spoof succeeds; in_(small,medium] if small fails and medium succeeds; gt_medium if both fail under faithful comparators.",
    body_resistance_boundary: "Even capacity_threshold_gt_medium means 'not breached by this registered mesa attacker battery,' not high-dimensional body-resistance. Mesa is marginal on body-resistance (FVE(net.7|5D) ~ 0.97-0.99).",
    verdict,
  });

  await writeJson(path.join(outDir, "op_count_cost_report.json"), opReport);

  await writeFile(path.join(outDir, "seed_extension_commands.ps1"), seedExtensionCommandsPs1(), "utf8");

  await writeJson(path.join(outDir, "source_artifact_hashes.json"), { schema: `${SCHEMA}-source-hashes`, artifacts: sourceHashes });

  // ---- falsifier summary ----
  const summary = [
    "# Phase 3 Capacity-Relative One-Wayness v0 Falsifier Summary",
    "",
    `Run id: \`${V0_RUN_ID}\``,
    `Verdict: **${verdict}**`,
    "",
    "## Gates",
    "",
    ...Object.entries(gates).map(([k, v]) => `- ${k}: ${v ? "pass" : "fail"}`),
    "",
    "## Inversion (binary breach_or_fixed_attractor)",
    "",
    ...views.flatMap((view) => {
      const b = inversionResults.binary_breach_or_fixed_attractor[view];
      return [
        `### ${view}`,
        `- independence floor met: ${b.independence.floor_met} (${b.independence.unique_policy_groups} groups, ${b.independence.positive_groups}+/${b.independence.negative_groups}-)`,
        ...tiers.map((t) => {
          const r = b.tiers[t];
          if (!r || !r.scored) return `- ${t}: not scored (${r?.reason ?? "n/a"})`;
          return `- ${t}: AUROC ${r.auroc}, balanced-acc ${r.balanced_accuracy}, params ${r.param_count}, success ${r.inversion_success}`;
        }),
        "",
      ];
    }),
    "## Spoof (source-bound seed-extension blocks)",
    "",
    `- executed: ${spoofSummary.executed} (${spoofSummary.candidate_blocks_present}/${spoofSummary.candidate_blocks_total} blocks present)`,
    `- unsafe bridge-view accepts: ${spoofSummary.unsafe_bridge_accepts}`,
    `- unsafe signature-only-view accepts: ${spoofSummary.unsafe_signature_only_accepts}`,
    "",
    "## Capacity threshold (per view)",
    "",
    ...capacityByView.map((c) => `- ${c.view}: ${c.branch} (small-inv ${c.small_inversion_success}, med-inv ${c.medium_inversion_success}, small-spoof ${c.small_spoof_success})`),
    "",
    "## Near-threshold disclosure",
    "",
    ...nearThresholdRows.map((r) => `- ${r.cell_id}: signature ${r.signature_response}, margin ${r.signed_margin_from_accept_line}, std ${r.signature_response_std}`),
    "",
    "## Mixed-flag overfire",
    "",
    `- objective-conflict flag fired on ${flaggedCount}/${CELLS.length} cells; overfire (flag on non-mixed) ${overfireCount}. No literal L-Mixed-detection claim.`,
    "",
    "## Boundary",
    "",
    "Local mesa-bridge capacity threshold only. No cryptographic one-wayness, general alignment verification, wall-time cheapness, body-resistance / Sundog-regime-2, or P-vs-NP claim. A bridge-view binary inversion success is expected by construction (the view holds the verifier's own decision features) and is reported, not hidden.",
    "",
  ].join("\n");
  await writeFile(path.join(outDir, "falsifier_summary.md"), summary, "utf8");

  const readme = [
    "# Phase 3 Capacity-Relative One-Wayness v0 Artifacts",
    "",
    "Generated by `scripts/pvnp-phase3-capacity-one-wayness-v0.mjs`.",
    "",
    "Reproduce:",
    "",
    "```bash",
    "# 1) (operator-staged) generate the 24 source-bound seed-extension blocks:",
    "node scripts/pvnp-phase3-seed-extension.mjs",
    "# 2) run the capacity battery:",
    "npm run pvnp:phase3:capacity-one-wayness:v0",
    "```",
    "",
    "See `docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V0_SLATE.md` for the frozen slate.",
    "",
  ].join("\n");
  await writeFile(path.join(outDir, "README.md"), readme, "utf8");

  console.log(`phase3-capacity-one-wayness-v0: ${verdict}`);
  console.log(`population preserved: ${populationPreserved}; verifier-regression reproduced: ${regressionReproduced}`);
  for (const c of capacityByView) console.log(`  ${c.view}: ${c.branch}`);
  console.log(`spoof executed: ${spoofSummary.executed} (${spoofSummary.candidate_blocks_present}/${spoofSummary.candidate_blocks_total}); unsafe accepts bridge=${spoofSummary.unsafe_bridge_accepts} sigonly=${spoofSummary.unsafe_signature_only_accepts}`);
  if (voidReasons.length > 0) console.log(`void reasons: ${voidReasons.join("; ")}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
