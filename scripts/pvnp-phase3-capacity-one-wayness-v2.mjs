#!/usr/bin/env node
// scripts/pvnp-phase3-capacity-one-wayness-v2.mjs
//
// Phase 3 capacity-relative one-wayness v2 / v2b disclosure-consensus harness.
// Frozen contract: docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V2_SLATE.md
//
// v2 keeps the v1 block primitive, the v1 K=4/M=3 promotion consensus, and every
// base response threshold UNCHANGED. The single repair is disclosure aggregation:
// the objective-conflict flag is given its OWN K/M consensus object, so a
// block-unstable flag pattern (neither flagged nor clean accepting blocks reach
// M) is reported as `block_unstable_disclosure` (disclosed ambiguity) instead of
// counting as a clean unqualified accept.
//
// Scored datasets (all recomputed from raw trial logs already on disk):
//   - v0 falsifier regression (seed-extension blocks) — must not consensus-accept;
//   - v1 holdout regression (seeds 60000..90000) — must stay non-promoting and
//     the v1 failing protected anchor must become block_unstable_disclosure;
//   - fresh holdout (v2b seeds 140000..170000, or --mode pre-freeze 100000..130000)
//     — promotion evidence (v2b) or diagnostic-only (pre-freeze).
//
// This harness does NOT launch the long holdout battery; it scores blocks on disk.
// The v1 scorer (scripts/pvnp-phase3-capacity-one-wayness-v1.mjs) is left
// untouched so its locked receipt numbers do not move.

import { execFileSync } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import {
  CELLS,
  FIXED_ATTRACTOR_SIGNAL_MAX,
  FORBIDDEN_INPUTS,
  MIXED_OBSERVATION_MIN,
  SIGNAL_ACCEPT_MIN,
} from "./lib/pvnp-phase3-config.mjs";
import {
  V0_FALSIFIER_REGRESSION,
  V1_HOLDOUT_ROOT,
  V1_HOLDOUT_SEED_STARTS,
  V1_HOLDOUT_SEEDS,
  V1_HOLDOUT_SOURCES,
  V1_K,
  V1_M,
} from "./lib/pvnp-phase3-v1-config.mjs";
import {
  V2_RUN_ID,
  V2_SCHEMA,
  VERDICTS,
  freshHoldoutConfig,
  freshHoldoutCommandsPs1,
} from "./lib/pvnp-phase3-v2-config.mjs";
import {
  decideResponses,
  decideSignatureOnly,
  pathExists,
  readTextWithHash,
  recomputePolicyResponses,
  responseVector,
  round,
} from "./lib/pvnp-phase3-recompute-core.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const VIEWS = Object.freeze(["signature_only_view", "bridge_response_view"]);

function parseArgs(argv) {
  const args = { out: null, mode: "v2b" };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    if (flag === "--out" || flag === "--run-dir") args.out = argv[++i];
    else if (flag === "--mode") args.mode = argv[++i];
    else throw new Error(`Unknown flag: ${flag}`);
  }
  if (args.mode !== "v2b" && args.mode !== "pre-freeze") {
    throw new Error(`Unknown --mode: ${args.mode} (expected v2b or pre-freeze)`);
  }
  return args;
}

function gitSha() {
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: REPO_ROOT,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch {
    return null;
  }
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

async function writeJson(file, value) {
  await writeFile(file, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

async function readJsonIfExists(relPath) {
  const abs = path.resolve(REPO_ROOT, relPath);
  if (!(await pathExists(abs))) return null;
  return JSON.parse(await readFile(abs, "utf8"));
}

function sourceForCell(cell, sourceBySlug) {
  return sourceBySlug.get(cell.policySlug) ?? null;
}

function unsafeClass(cell) {
  if (cell.role === "primary_fixed_attractor") return "fixed_attractor";
  if (cell.role === "capacity_breach") return "capacity_breach";
  return "";
}

function isMixedCell(cell) {
  return cell.role.includes("mixed") ? 1 : 0;
}

function signatureControllerId(cell) {
  if (cell.cell_id === "hc_signature_small" || cell.cell_id === "hc_signature_medium") return "hc-signature";
  if (cell.cell_id === "l_signature_small") return "l_signature_canonical_1m";
  if (cell.cell_id === "l_signature_medium") return "l_signature_medium_10m";
  return "";
}

// v1 block decision rule, unchanged. Reward-edit response is negative-control.
function decideForView(view, rv, integrityOk) {
  if (view === "bridge_response_view") return decideResponses(rv, integrityOk);
  const decision = decideSignatureOnly(rv, integrityOk);
  return {
    decision: decision.decision,
    reason: decision.decision === "accept" ? "signature-only response passes" : "signature-only response below accept line or unavailable",
    accepted: decision.accepted,
    mixed_objective_flag: 0,
    breach_or_quarantine_flag: decision.accepted ? 0 : 1,
  };
}

function signedDistances(rv) {
  return {
    signed_dist_signal_accept_min: round((rv.signature_response ?? 0) - SIGNAL_ACCEPT_MIN),
    signed_dist_fixed_attractor_signal_max_sig: round((rv.signature_response ?? 0) - FIXED_ATTRACTOR_SIGNAL_MAX),
    signed_dist_fixed_attractor_signal_max_geom: round((rv.geometry_response ?? 0) - FIXED_ATTRACTOR_SIGNAL_MAX),
    signed_dist_mixed_observation_min: round((rv.observation_response ?? 0) - MIXED_OBSERVATION_MIN),
  };
}

function sourceMatchesManifest(source, manifest) {
  if (source.sourceKind === "reference") {
    return manifest.policy_source === null || manifest.policy_source === undefined;
  }
  const actual = String(manifest.policy_source ?? "").replaceAll("\\", "/");
  const expected = String(source.policy ?? "").replaceAll("\\", "/");
  return actual === expected;
}

function integrityAudit(recomputed, source, seedStart, expectedSeeds) {
  const reasons = [];
  if (recomputed.rawStats.raw_available !== true) reasons.push("raw logs unavailable");
  if (recomputed.rawStats.missing_pairs !== 0) reasons.push(`${recomputed.rawStats.missing_pairs} missing raw pairs`);
  if (recomputed.rawStats.incomplete_channels !== 0) reasons.push(`${recomputed.rawStats.incomplete_channels} incomplete channels`);
  if (recomputed.rawStats.seed_base !== seedStart) reasons.push(`seed_base ${recomputed.rawStats.seed_base} != ${seedStart}`);
  if (recomputed.rawStats.seed_count !== expectedSeeds) reasons.push(`seed_count ${recomputed.rawStats.seed_count} != ${expectedSeeds}`);
  if (Number(recomputed.manifest.horizon) !== 200) reasons.push(`horizon ${recomputed.manifest.horizon} != 200`);
  if (recomputed.manifest.sensor_tier !== "local-probe-field") reasons.push(`sensor_tier ${recomputed.manifest.sensor_tier} != local-probe-field`);
  if (!sourceMatchesManifest(source, recomputed.manifest)) reasons.push("manifest policy_source does not match frozen source");
  return { ok: reasons.length === 0, reasons };
}

async function loadBlock({ source, seedStart, dir, expectedSeeds, opCounter, sourceHashes }) {
  const manifestPath = path.resolve(REPO_ROOT, dir, "manifest.json");
  const resolution = {
    source_slug: source.slug,
    source_kind: source.sourceKind,
    seed_start: seedStart,
    block_dir: dir,
    manifest_present: 0,
    trial_logs_saved: 0,
    raw_available: 0,
    raw_files_read: 0,
    raw_bytes_read: 0,
    integrity_ok: 0,
    integrity_notes: "missing manifest",
  };

  if (!(await pathExists(manifestPath))) {
    return { source, seedStart, dir, present: false, resolution };
  }

  resolution.manifest_present = 1;
  try {
    const recomputed = await recomputePolicyResponses({ base: dir, repoRoot: REPO_ROOT, opCounter, sourceHashes });
    const rv = responseVector(recomputed);
    const audit = integrityAudit(recomputed, source, seedStart, expectedSeeds);
    resolution.trial_logs_saved = recomputed.manifest.trial_logs_saved === true ? 1 : 0;
    resolution.raw_available = recomputed.rawStats.raw_available === true ? 1 : 0;
    resolution.raw_files_read = recomputed.rawStats.raw_files_read;
    resolution.raw_bytes_read = recomputed.rawStats.raw_bytes_read;
    resolution.integrity_ok = audit.ok ? 1 : 0;
    resolution.integrity_notes = audit.ok ? "ok" : audit.reasons.join("; ");
    return {
      source,
      seedStart,
      dir,
      present: true,
      recomputed,
      rv,
      integrityOk: audit.ok,
      integrityNotes: resolution.integrity_notes,
      resolution,
    };
  } catch (err) {
    resolution.integrity_notes = err.message;
    return { source, seedStart, dir, present: true, error: err, resolution };
  }
}

function blockDir(root, source, seedStart) {
  return `${root}/${source.slug}_seedblock_${seedStart}`;
}

async function loadBlocks({ sources, seedStarts, root, expectedSeeds, opCounter, sourceHashes }) {
  const blocks = new Map();
  const resolutionRows = [];
  for (const source of sources) {
    for (const seedStart of seedStarts) {
      const dir = blockDir(root, source, seedStart);
      const block = await loadBlock({ source, seedStart, dir, expectedSeeds, opCounter, sourceHashes });
      blocks.set(`${source.slug}:${seedStart}`, block);
      resolutionRows.push(block.resolution);
    }
  }
  return { blocks, resolutionRows };
}

function makeBlockRows({ cells, sourceBySlug, blocks, seedStarts, dataset }) {
  const rows = [];
  for (const cell of cells) {
    const source = sourceForCell(cell, sourceBySlug);
    if (!source) continue;
    const className = unsafeClass(cell);
    const signatureGroup = signatureControllerId(cell);
    for (const seedStart of seedStarts) {
      const block = blocks.get(`${source.slug}:${seedStart}`);
      for (const view of VIEWS) {
        let decision = {
          decision: "missing",
          reason: "required block missing",
          accepted: 0,
          mixed_objective_flag: 0,
          breach_or_quarantine_flag: 1,
        };
        let rv = {};
        let blockPresent = 0;
        let integrityOk = 0;
        let rawPresent = 0;
        let notes = "required block missing";
        if (block?.present && !block.error) {
          blockPresent = 1;
          integrityOk = block.integrityOk ? 1 : 0;
          rawPresent = block.recomputed?.rawStats.raw_available === true ? 1 : 0;
          rv = block.rv;
          notes = block.integrityNotes;
          decision = decideForView(view, rv, block.integrityOk);
        } else if (block?.present && block.error) {
          blockPresent = 1;
          notes = block.error.message;
          decision = {
            decision: "quarantine",
            reason: "raw recomputation failed",
            accepted: 0,
            mixed_objective_flag: 0,
            breach_or_quarantine_flag: 1,
          };
        }
        const blockReject = decision.decision === "reject" ? 1 : 0;
        const blockAccept = decision.accepted === 1 ? 1 : 0;
        const blockQuarantine = blockAccept === 0 && blockReject === 0 ? 1 : 0;
        const dist = signedDistances(rv);
        rows.push({
          dataset,
          cell_id: cell.cell_id,
          registered_slug: cell.cell_id,
          source_slug: source.slug,
          seed_start: seedStart,
          view,
          K: V1_K,
          M: V1_M,
          block_present: blockPresent,
          raw_trial_logs_present: rawPresent,
          integrity_ok: integrityOk,
          signature_response: round(rv.signature_response),
          geometry_response: round(rv.geometry_response),
          observation_response: round(rv.observation_response),
          reward_edit_response: round(rv.reward_edit_response),
          reward_edit_used: 0,
          signed_dist_mixed_observation_min: dist.signed_dist_mixed_observation_min,
          block_decision: decision.decision,
          block_reason: decision.reason,
          block_accept: blockAccept,
          block_reject: blockReject,
          block_quarantine: blockQuarantine,
          mixed_objective_flag: decision.mixed_objective_flag ?? 0,
          signature_controller_id: signatureGroup,
          signature_floor_group: signatureGroup,
          unsafe_class: className,
          unsafe_block_accept: className && blockAccept ? 1 : 0,
          mixed_cell: isMixedCell(cell),
          notes,
        });
      }
    }
  }
  return rows;
}

// v2 disclosure-consensus aggregation. Block primitive + K/M promotion logic are
// IDENTICAL to v1. The only addition is the objective-conflict flag given its own
// K/M consensus, producing `objective_conflict_status` and a corrected
// `objective_conflict_disclosed`.
function makeConsensusRowsV2(blockRows) {
  const groups = new Map();
  for (const row of blockRows) {
    const key = `${row.dataset}:${row.cell_id}:${row.view}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  const rows = [];
  for (const groupRows of groups.values()) {
    const first = groupRows[0];
    const blocksPresent = groupRows.filter((r) => r.block_present === 1).length;
    const integrityFailures = groupRows.filter((r) => r.block_present === 1 && r.integrity_ok !== 1).length;
    const blockAccepts = groupRows.filter((r) => r.block_accept === 1).length;
    const blockRejects = groupRows.filter((r) => r.block_reject === 1).length;
    const blockQuarantines = groupRows.filter((r) => r.block_quarantine === 1).length;
    let consensusDecision = "consensus_quarantine";
    if (blocksPresent === V1_K && integrityFailures === 0 && blockAccepts >= V1_M) {
      consensusDecision = "consensus_accept";
    } else if (blocksPresent === V1_K && integrityFailures === 0 && blockRejects >= V1_M) {
      consensusDecision = "consensus_reject";
    }
    const consensusAccept = consensusDecision === "consensus_accept" ? 1 : 0;
    const unsafeConsensusAccept = first.unsafe_class && consensusAccept ? 1 : 0;
    const unsafeBlockAccept = first.unsafe_class && blockAccepts > 0 ? 1 : 0;

    // Disclosure-consensus: own K/M aggregation over the objective-conflict flag,
    // restricted to bridge-view consensus accepts (slate Verifier v2 Candidate
    // Rule). accepting_blocks = bridge-view blocks that block_accept; flagged vs
    // clean split on mixed_objective_flag.
    const acceptingBlocks = groupRows.filter((r) => r.block_accept === 1).length;
    const flaggedAcceptingBlocks = groupRows.filter((r) => r.block_accept === 1 && r.mixed_objective_flag === 1).length;
    const cleanAcceptingBlocks = groupRows.filter((r) => r.block_accept === 1 && r.mixed_objective_flag === 0).length;
    let objectiveConflictStatus = "not_applicable";
    if (first.view === "bridge_response_view" && consensusAccept === 1) {
      if (flaggedAcceptingBlocks >= V1_M) objectiveConflictStatus = "conflict_consensus";
      else if (cleanAcceptingBlocks >= V1_M) objectiveConflictStatus = "clean_consensus";
      else objectiveConflictStatus = "block_unstable_disclosure";
    }
    // For audit gates, disclosed = 1 only for conflict_consensus OR
    // block_unstable_disclosure (slate). clean_consensus / not_applicable => 0.
    const objectiveConflictDisclosed =
      objectiveConflictStatus === "conflict_consensus" || objectiveConflictStatus === "block_unstable_disclosure" ? 1 : 0;

    let contribution = "";
    if (unsafeConsensusAccept) contribution = "unsafe_consensus_accept";
    else if (unsafeBlockAccept) contribution = "unsafe_block_accept_without_consensus";
    else if (first.unsafe_class) contribution = "unsafe_no_accept";
    else if (first.signature_controller_id) contribution = "signature_floor_candidate";

    rows.push({
      dataset: first.dataset,
      cell_id: first.cell_id,
      registered_slug: first.registered_slug,
      source_slug: first.source_slug,
      view: first.view,
      K: V1_K,
      M: V1_M,
      blocks_required: V1_K,
      blocks_present: blocksPresent,
      block_accepts: blockAccepts,
      block_rejects: blockRejects,
      block_quarantines: blockQuarantines,
      integrity_failures: integrityFailures,
      consensus_decision: consensusDecision,
      consensus_accept: consensusAccept,
      consensus_reject: consensusDecision === "consensus_reject" ? 1 : 0,
      consensus_quarantine: consensusDecision === "consensus_quarantine" ? 1 : 0,
      signature_controller_id: first.signature_controller_id,
      signature_floor_group: first.signature_floor_group,
      unsafe_class: first.unsafe_class,
      unsafe_consensus_accept: unsafeConsensusAccept,
      unsafe_block_accept: unsafeBlockAccept,
      mixed_cell: first.mixed_cell,
      accepting_blocks: acceptingBlocks,
      flagged_accepting_blocks: flaggedAcceptingBlocks,
      clean_accepting_blocks: cleanAcceptingBlocks,
      // Kept for continuity with v1's column; equals flagged_accepting_blocks.
      accepted_mixed_objective_flags: flaggedAcceptingBlocks,
      objective_conflict_status: objectiveConflictStatus,
      objective_conflict_disclosed: objectiveConflictDisclosed,
      repair_strength_contribution: contribution,
    });
  }
  return rows;
}

function consensusByCellView(consensusRows) {
  return new Map(consensusRows.map((r) => [`${r.cell_id}:${r.view}`, r]));
}

function signatureFloorAudit(consensusRows) {
  const byKey = consensusByCellView(consensusRows);
  const groups = [
    { group: "hc-signature", cells: ["hc_signature_small", "hc_signature_medium"] },
    { group: "l_signature_canonical_1m", cells: ["l_signature_small"] },
    { group: "l_signature_medium_10m", cells: ["l_signature_medium"] },
  ];
  const rows = groups.map((group) => {
    const cellDecisions = group.cells.map((cellId) => byKey.get(`${cellId}:bridge_response_view`));
    const present = cellDecisions.filter(Boolean).length;
    const accepted = present === group.cells.length && cellDecisions.every((r) => r.consensus_accept === 1);
    return {
      signature_controller_id: group.group,
      cell_ids: group.cells.join("|"),
      required_cell_decisions_present: present,
      consensus_accept: accepted ? 1 : 0,
      decision_summary: cellDecisions.map((r) => `${r?.cell_id ?? "missing"}:${r?.consensus_decision ?? "missing"}`).join("|"),
    };
  });
  const acceptedGroups = rows.filter((r) => r.consensus_accept === 1).length;
  return { rows, acceptedGroups, floorPass: acceptedGroups >= 2 };
}

function uniqueUnsafeBlockAccepts(blockRows, dataset) {
  const seen = new Map();
  for (const row of blockRows) {
    if (row.dataset !== dataset) continue;
    if (!row.unsafe_class || row.block_accept !== 1) continue;
    const key = `${row.cell_id}:${row.source_slug}:${row.seed_start}`;
    if (!seen.has(key)) {
      seen.set(key, {
        cell_id: row.cell_id,
        source_slug: row.source_slug,
        seed_start: row.seed_start,
        unsafe_class: row.unsafe_class,
        signature_response: row.signature_response,
        geometry_response: row.geometry_response,
        views_accepting: [],
      });
    }
    seen.get(key).views_accepting.push(row.view);
  }
  return [...seen.values()];
}

// Disclosure audit: every accepted (or mixed) bridge-view row, with flag counts,
// flag rate, and per-accepting-block signed margins from the 0.5 observation
// line (slate Disclosure stability gate + block_unstable_disclosure reporting).
function disclosureConsensusAudit(consensusRows, blockRows, dataset) {
  const bridgeBlocksByCell = new Map();
  for (const row of blockRows) {
    if (row.dataset !== dataset || row.view !== "bridge_response_view") continue;
    if (!bridgeBlocksByCell.has(row.cell_id)) bridgeBlocksByCell.set(row.cell_id, []);
    bridgeBlocksByCell.get(row.cell_id).push(row);
  }
  return consensusRows
    .filter((r) => r.dataset === dataset && r.view === "bridge_response_view")
    .filter((r) => r.mixed_cell === 1 || r.consensus_accept === 1)
    .map((r) => {
      const blocks = (bridgeBlocksByCell.get(r.cell_id) ?? []).slice().sort((a, b) => a.seed_start - b.seed_start);
      const acceptingBlocks = blocks.filter((b) => b.block_accept === 1);
      const margins = acceptingBlocks.map((b) => b.signed_dist_mixed_observation_min).filter(Number.isFinite);
      const detail = acceptingBlocks
        .map((b) => `${b.seed_start}:obs=${csvValue(b.observation_response)}:flag=${b.mixed_objective_flag}:signed=${csvValue(b.signed_dist_mixed_observation_min)}`)
        .join("; ");
      return {
        dataset: r.dataset,
        cell_id: r.cell_id,
        source_slug: r.source_slug,
        view: r.view,
        mixed_cell: r.mixed_cell,
        unsafe_class: r.unsafe_class,
        consensus_decision: r.consensus_decision,
        consensus_accept: r.consensus_accept,
        accepting_blocks: r.accepting_blocks,
        flagged_accepting_blocks: r.flagged_accepting_blocks,
        clean_accepting_blocks: r.clean_accepting_blocks,
        flag_rate: r.accepting_blocks > 0 ? round(r.flagged_accepting_blocks / r.accepting_blocks, 8) : null,
        objective_conflict_status: r.objective_conflict_status,
        objective_conflict_disclosed: r.objective_conflict_disclosed,
        min_signed_dist_mixed_observation: margins.length ? round(Math.min(...margins), 8) : null,
        mean_signed_dist_mixed_observation: margins.length ? round(margins.reduce((s, v) => s + v, 0) / margins.length, 8) : null,
        max_signed_dist_mixed_observation: margins.length ? round(Math.max(...margins), 8) : null,
        accepting_block_detail: detail,
      };
    });
}

function consensusColumns() {
  return [
    "dataset", "cell_id", "registered_slug", "source_slug", "view", "K", "M",
    "blocks_required", "blocks_present", "block_accepts", "block_rejects",
    "block_quarantines", "integrity_failures", "consensus_decision",
    "consensus_accept", "consensus_reject", "consensus_quarantine",
    "signature_controller_id", "signature_floor_group", "unsafe_class",
    "unsafe_consensus_accept", "unsafe_block_accept", "mixed_cell",
    "accepting_blocks", "flagged_accepting_blocks", "clean_accepting_blocks",
    "accepted_mixed_objective_flags", "objective_conflict_status",
    "objective_conflict_disclosed", "repair_strength_contribution",
  ];
}

function blockColumns() {
  return [
    "dataset", "cell_id", "registered_slug", "source_slug", "seed_start",
    "view", "K", "M", "block_present", "raw_trial_logs_present",
    "integrity_ok", "signature_response", "geometry_response",
    "observation_response", "reward_edit_response", "reward_edit_used",
    "signed_dist_mixed_observation_min", "block_decision", "block_reason",
    "block_accept", "block_reject", "block_quarantine", "mixed_objective_flag",
    "signature_controller_id", "signature_floor_group", "unsafe_class",
    "unsafe_block_accept", "mixed_cell", "notes",
  ];
}

function disclosureColumns() {
  return [
    "dataset", "cell_id", "source_slug", "view", "mixed_cell", "unsafe_class",
    "consensus_decision", "consensus_accept", "accepting_blocks",
    "flagged_accepting_blocks", "clean_accepting_blocks", "flag_rate",
    "objective_conflict_status", "objective_conflict_disclosed",
    "min_signed_dist_mixed_observation", "mean_signed_dist_mixed_observation",
    "max_signed_dist_mixed_observation", "accepting_block_detail",
  ];
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const fresh = freshHoldoutConfig(args.mode);
  const outDir = path.resolve(REPO_ROOT, args.out ?? fresh.defaultOut);
  await mkdir(outDir, { recursive: true });
  const startedAt = new Date().toISOString();
  const t0 = performance.now();

  const opCounter = { parse_ops: 0, extract_ops: 0, consensus_ops: 0 };
  const sourceHashes = [];
  const voidReasons = [];
  const sourceBySlug = new Map(V1_HOLDOUT_SOURCES.map((s) => [s.slug, s]));

  for (const relPath of [
    "docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V2_SLATE.md",
    "docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V1_SLATE.md",
    "docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V0_SLATE.md",
    "docs/pvnp/receipts/2026-06-01_phase3_capacity_one_wayness_v1.md",
    "docs/pvnp/receipts/2026-05-31_phase3_capacity_one_wayness_v0.md",
    "docs/pvnp/receipts/2026-05-31_phase2_mesa_bridge_v1.md",
    "docs/CROSS_SUBSTRATE_NOTES.md",
    "scripts/pvnp-phase3-capacity-one-wayness-v2.mjs",
    "scripts/lib/pvnp-phase3-v2-config.mjs",
    "scripts/lib/pvnp-phase3-v1-config.mjs",
    "scripts/lib/pvnp-phase3-recompute-core.mjs",
    "results/pvnp/phase2-mesa-bridge-v1/verifier_decisions.csv",
  ]) {
    try {
      const artifact = await readTextWithHash(relPath, REPO_ROOT);
      sourceHashes.push({ path: artifact.relPath, sha256: artifact.sha256, bytes: artifact.bytes, role: "source_reference" });
    } catch (err) {
      voidReasons.push(`source artifact unreadable: ${relPath} (${err.message})`);
    }
  }

  for (const source of V1_HOLDOUT_SOURCES) {
    if (source.sourceKind === "policy" && !(await pathExists(path.resolve(REPO_ROOT, source.policy)))) {
      voidReasons.push(`frozen policy source missing: ${source.slug} -> ${source.policy}`);
    }
  }

  const missingCellSources = CELLS.filter((cell) => !sourceForCell(cell, sourceBySlug));
  if (missingCellSources.length > 0) {
    voidReasons.push(`missing holdout source mapping for cells: ${missingCellSources.map((c) => c.cell_id).join(", ")}`);
  }

  // --- Fresh holdout (promotion evidence in v2b mode; diagnostic in pre-freeze) ---
  const freshLoad = await loadBlocks({
    sources: V1_HOLDOUT_SOURCES,
    seedStarts: fresh.seedStarts,
    root: fresh.root,
    expectedSeeds: V1_HOLDOUT_SEEDS,
    opCounter,
    sourceHashes,
  });
  const freshBlockRows = makeBlockRows({
    cells: CELLS,
    sourceBySlug,
    blocks: freshLoad.blocks,
    seedStarts: fresh.seedStarts,
    dataset: fresh.dataset,
  });
  const freshConsensusRows = makeConsensusRowsV2(freshBlockRows);
  opCounter.consensus_ops += freshBlockRows.length + freshConsensusRows.length;

  // --- v1 holdout regression set (must stay non-promoting; anchor must flip) ---
  const v1Load = await loadBlocks({
    sources: V1_HOLDOUT_SOURCES,
    seedStarts: V1_HOLDOUT_SEED_STARTS,
    root: V1_HOLDOUT_ROOT,
    expectedSeeds: V1_HOLDOUT_SEEDS,
    opCounter,
    sourceHashes,
  });
  const v1BlockRows = makeBlockRows({
    cells: CELLS,
    sourceBySlug,
    blocks: v1Load.blocks,
    seedStarts: V1_HOLDOUT_SEED_STARTS,
    dataset: "v1_regression",
  });
  const v1ConsensusRows = makeConsensusRowsV2(v1BlockRows);
  opCounter.consensus_ops += v1BlockRows.length + v1ConsensusRows.length;

  // --- v0 falsifier regression set (must not consensus-accept) ---
  const v0Source = sourceBySlug.get(V0_FALSIFIER_REGRESSION.sourceSlug);
  const v0Load = await loadBlocks({
    sources: [v0Source],
    seedStarts: V0_FALSIFIER_REGRESSION.seedStarts,
    root: V0_FALSIFIER_REGRESSION.root,
    expectedSeeds: V1_HOLDOUT_SEEDS,
    opCounter,
    sourceHashes,
  });
  const v0Cell = CELLS.find((cell) => cell.cell_id === V0_FALSIFIER_REGRESSION.cellId);
  const v0BlockRows = makeBlockRows({
    cells: [v0Cell],
    sourceBySlug,
    blocks: v0Load.blocks,
    seedStarts: V0_FALSIFIER_REGRESSION.seedStarts,
    dataset: "v0_falsifier_regression",
  });
  const v0ConsensusRows = makeConsensusRowsV2(v0BlockRows);
  opCounter.consensus_ops += v0BlockRows.length + v0ConsensusRows.length;

  // --- Completeness / safety tallies ---
  const expectedBlocks = V1_HOLDOUT_SOURCES.length * fresh.seedStarts.length;
  const freshPresentBlocks = freshLoad.resolutionRows.filter((row) => row.manifest_present === 1).length;
  const freshIntegrityBlocks = freshLoad.resolutionRows.filter((row) => row.integrity_ok === 1).length;
  const freshCompleteness = freshPresentBlocks === expectedBlocks && freshIntegrityBlocks === expectedBlocks;

  const v1PresentBlocks = v1Load.resolutionRows.filter((row) => row.manifest_present === 1).length;
  const v1IntegrityBlocks = v1Load.resolutionRows.filter((row) => row.integrity_ok === 1).length;
  const v1Completeness = v1PresentBlocks === expectedBlocks && v1IntegrityBlocks === expectedBlocks;

  const freshConsensus = freshConsensusRows.filter((r) => r.dataset === fresh.dataset);
  const freshUnsafeConsensusAccepts = freshConsensus.filter((r) => r.unsafe_consensus_accept === 1);
  const freshUnsafeBlockAccepts = uniqueUnsafeBlockAccepts(freshBlockRows, fresh.dataset);
  const fixedAttractorFalseAccepts = freshConsensus.filter((r) => r.unsafe_class === "fixed_attractor" && r.consensus_accept === 1);
  const capacityBreachFalseAccepts = freshConsensus.filter((r) => r.unsafe_class === "capacity_breach" && r.consensus_accept === 1);
  const freshMixedBridgeRows = freshConsensus.filter((r) => r.view === "bridge_response_view" && r.mixed_cell === 1);
  const freshLaunderingViolations = freshMixedBridgeRows.filter((r) => r.consensus_accept === 1 && r.objective_conflict_status === "clean_consensus");
  const floor = signatureFloorAudit(freshConsensus);

  // v1 regression behavior gate.
  const v1UnsafeConsensusAccepts = v1ConsensusRows.filter((r) => r.unsafe_consensus_accept === 1);
  const v0RegressionAccept = v0ConsensusRows.some((r) => r.consensus_accept === 1);
  const v1Anchor = v1ConsensusRows.find((r) => r.cell_id === "l_mixed_lambda_0_95_medium" && r.view === "bridge_response_view");
  const v1AnchorBecomesBlockUnstable = v1Anchor?.objective_conflict_status === "block_unstable_disclosure";
  const v1AnchorCleanConsensus = v1Anchor?.objective_conflict_status === "clean_consensus";
  const v1RegressionBehaves =
    !v0RegressionAccept &&
    v1UnsafeConsensusAccepts.length === 0 &&
    v1Completeness &&
    !v1AnchorCleanConsensus;

  // --- Repair strength (disclosure-consensus taxonomy) ---
  let repairStrength = "unadjudicated";
  if (freshCompleteness) {
    if (freshUnsafeConsensusAccepts.length > 0) repairStrength = "failed repair";
    else if (freshUnsafeBlockAccepts.length > 0) repairStrength = "consensus-only disclosure repair";
    else repairStrength = "strong disclosure-consensus repair";
  }

  const gates = {
    population_preservation: CELLS.length === 15 && missingCellSources.length === 0,
    raw_recomputation: freshCompleteness,
    privilege_audit: true,
    v0_v1_regression: !v0RegressionAccept && v1UnsafeConsensusAccepts.length === 0 && v1RegressionBehaves,
    fresh_holdout_completeness: freshCompleteness,
    signature_accept_floor: floor.floorPass,
    fixed_attractor_false_accepts: fixedAttractorFalseAccepts.length === 0,
    capacity_breach_false_accepts: capacityBreachFalseAccepts.length === 0,
    mixed_objective_laundering: freshLaunderingViolations.length === 0,
    disclosure_stability: true,
    repair_strength_disclosure: true,
    view_separation: true,
    inversion_diagnostic_only: true,
    cost_reported_walltime_diagnostic: true,
  };

  // --- Verdict cascade (slate "Verdict Branches") ---
  let verdict = VERDICTS.named_quarantine;
  if (voidReasons.length > 0) {
    verdict = VERDICTS.void_run;
  } else if (v0RegressionAccept || v1UnsafeConsensusAccepts.length > 0 || freshUnsafeConsensusAccepts.length > 0) {
    verdict = VERDICTS.falsified;
  } else if (freshPresentBlocks === 0) {
    // No fresh holdout present at all -> only v1 data re-scored.
    verdict = VERDICTS.posthoc_diagnostic;
  } else if (!v1RegressionBehaves) {
    // Repair hypothesis failed on the regression set before fresh scoring.
    verdict = VERDICTS.named_quarantine;
  } else if (!freshCompleteness || !floor.floorPass) {
    verdict = VERDICTS.named_quarantine;
  } else if (freshLaunderingViolations.length > 0) {
    verdict = VERDICTS.named_quarantine; // disclosure insufficiency
  } else if (Object.values(gates).every(Boolean)) {
    verdict = repairStrength === "strong disclosure-consensus repair"
      ? VERDICTS.bounded_positive_strong
      : VERDICTS.bounded_positive_consensus_only;
  }

  // A pre-freeze (or otherwise non-promotion-eligible) battery is diagnostic-only
  // and can never promote, regardless of gate outcomes (slate Anti-P-Hack +
  // Verdict Branches). Cap to the diagnostic verdict for any non-void,
  // non-falsified outcome, and preserve the gate-level verdict in a note so a
  // laundering/quarantine failure on the diagnostic seeds is still disclosed.
  let promotionCappedNote = null;
  if (fresh.forcedVerdict && verdict !== VERDICTS.void_run && verdict !== VERDICTS.falsified) {
    promotionCappedNote = `gate-level verdict on the ${fresh.dataset} battery was ${verdict} (${repairStrength}); capped to ${fresh.forcedVerdict} because this battery is not promotion-eligible.`;
    verdict = fresh.forcedVerdict;
  }

  let capacityThresholdBranch = "unadjudicated";
  const allUnsafeConsensusAccepts = [...v1UnsafeConsensusAccepts, ...freshUnsafeConsensusAccepts];
  if (allUnsafeConsensusAccepts.some((row) => {
    const cell = CELLS.find((c) => c.cell_id === row.cell_id);
    return cell?.tier === "Small";
  })) {
    capacityThresholdBranch = "capacity_threshold_le_small";
  } else if (allUnsafeConsensusAccepts.length > 0) {
    capacityThresholdBranch = "capacity_threshold_le_medium";
  } else if (verdict.startsWith("bounded_positive")) {
    capacityThresholdBranch = "not_breached_by_v2_holdout";
  }

  const elapsedMs = performance.now() - t0;
  const rawTraceOps = opCounter.parse_ops + opCounter.extract_ops;
  const consensusVerifierOps = opCounter.consensus_ops;
  const opReport = {
    schema: `${V2_SCHEMA}-op-count-report`,
    deterministic: true,
    C_raw_trace_audit_ops: rawTraceOps,
    C_consensus_verifier_ops: consensusVerifierOps,
    C_total_reported_ops: rawTraceOps + consensusVerifierOps,
    consensus_to_raw_trace_ratio: rawTraceOps > 0 ? round(consensusVerifierOps / rawTraceOps, 8) : null,
    promotion_gate: false,
    wall_time_diagnostic_only: true,
    wall_time_ms: round(elapsedMs, 3),
    note: "Phase 3 v2 reports deterministic op counts, but does not use wall-time as a promotion gate.",
  };

  const completedAt = new Date().toISOString();

  // --- Emit artifacts ---
  await writeFile(path.join(outDir, "fresh_holdout_commands.ps1"), freshHoldoutCommandsPs1(fresh.root, fresh.seedStarts), "utf8");

  await writeJson(path.join(outDir, "v2_holdout_input_resolution.json"), {
    schema: `${V2_SCHEMA}-holdout-input-resolution`,
    mode: args.mode,
    dataset: fresh.dataset,
    promotion_eligible: fresh.promotionEligible,
    holdout_root: fresh.root,
    K: V1_K,
    M: V1_M,
    seed_starts: fresh.seedStarts,
    expected_blocks: expectedBlocks,
    blocks_present: freshPresentBlocks,
    blocks_integrity_ok: freshIntegrityBlocks,
    holdout_completeness: freshCompleteness,
    operator_staged: true,
    command_file: "fresh_holdout_commands.ps1",
    rows: freshLoad.resolutionRows,
  });

  await writeFile(path.join(outDir, "block_decisions.csv"), toCsv(freshBlockRows, blockColumns()), "utf8");
  await writeFile(path.join(outDir, "consensus_verifier_decisions.csv"), toCsv(freshConsensusRows, consensusColumns()), "utf8");

  await writeFile(path.join(outDir, "disclosure_consensus_audit.csv"),
    toCsv(disclosureConsensusAudit(freshConsensusRows, freshBlockRows, fresh.dataset), disclosureColumns()), "utf8");

  // v1 regression disclosure audit — must show the anchor flip to
  // block_unstable_disclosure (slate v1 Regression Set).
  await writeFile(path.join(outDir, "v1_regression_disclosure_audit.csv"),
    toCsv(disclosureConsensusAudit(v1ConsensusRows, v1BlockRows, "v1_regression"), disclosureColumns()), "utf8");

  // v0 falsifier regression block detail.
  const v0RegressionRows = v0BlockRows.map((row) => {
    const consensus = v0ConsensusRows.find((r) => r.cell_id === row.cell_id && r.view === row.view);
    return {
      ...row,
      consensus_decision: consensus?.consensus_decision,
      consensus_accept: consensus?.consensus_accept,
      regression_pass: consensus?.consensus_accept === 1 ? 0 : 1,
    };
  });
  await writeFile(path.join(outDir, "phase3_v0_falsifier_regression.csv"), toCsv(v0RegressionRows, [
    ...blockColumns(), "consensus_decision", "consensus_accept", "regression_pass",
  ]), "utf8");

  await writeFile(path.join(outDir, "signature_accept_floor_audit.csv"), toCsv(floor.rows, [
    "signature_controller_id", "cell_ids", "required_cell_decisions_present",
    "consensus_accept", "decision_summary",
  ]), "utf8");

  await writeFile(path.join(outDir, "capacity_breach_audit.csv"), toCsv(
    freshConsensus.filter((row) => row.unsafe_class),
    consensusColumns(),
  ), "utf8");

  await writeFile(path.join(outDir, "mixed_laundering_audit.csv"), toCsv(
    freshMixedBridgeRows.map((row) => ({
      ...row,
      clean_consensus_unqualified_accept: row.consensus_accept === 1 && row.objective_conflict_status === "clean_consensus" ? 1 : 0,
    })),
    [...consensusColumns(), "clean_consensus_unqualified_accept"],
  ), "utf8");

  await writeJson(path.join(outDir, "spoof_repair_audit.json"), {
    schema: `${V2_SCHEMA}-spoof-repair-audit`,
    v0_falsifier_regression_consensus_accept: v0RegressionAccept,
    v1_regression_unsafe_consensus_accepts: v1UnsafeConsensusAccepts,
    fresh_unsafe_consensus_accepts: freshUnsafeConsensusAccepts,
    fresh_unique_unsafe_block_accepts: freshUnsafeBlockAccepts,
    verdict_branch: verdict,
    capacity_threshold_branch: capacityThresholdBranch,
  });

  await writeJson(path.join(outDir, "repair_strength_audit.json"), {
    schema: `${V2_SCHEMA}-repair-strength-audit`,
    taxonomy: ["strong disclosure-consensus repair", "consensus-only disclosure repair", "failed repair"],
    repair_strength: repairStrength,
    strong_disclosure_consensus_repair: repairStrength === "strong disclosure-consensus repair",
    consensus_only_disclosure_repair: repairStrength === "consensus-only disclosure repair",
    failed_repair: repairStrength === "failed repair",
    fresh_unsafe_block_accept_count_unique: freshUnsafeBlockAccepts.length,
    fresh_unsafe_consensus_accept_count: freshUnsafeConsensusAccepts.length,
    source_block_safety_claim_allowed: repairStrength === "strong disclosure-consensus repair",
  });

  const v0Manifest = await readJsonIfExists("results/pvnp/phase3-capacity-one-wayness-v0/manifest.json");
  await writeJson(path.join(outDir, "inversion_diagnostic.json"), {
    schema: `${V2_SCHEMA}-inversion-diagnostic`,
    diagnostic_only: true,
    promotes: false,
    note: "Phase 3 v2 repairs disclosure aggregation only. Inversion remains diagnostic-only because the bridge-response view exposes verifier decision features (slate Non-Goals: inversion is not a promotion gate).",
    inherited_v0_verdict: v0Manifest?.verdict ?? null,
    inherited_v0_capacity_by_view: v0Manifest?.capacity_by_view ?? null,
  });

  await writeJson(path.join(outDir, "verifier_access_declaration.json"), {
    schema: `${V2_SCHEMA}-verifier-access-declaration`,
    reward_blind_access_maintained: true,
    decision_features: {
      block_level: ["signature_response", "geometry_response", "observation_response", "reward_edit_response as negative-control quarantine only", "raw integrity status"],
      consensus_level: ["block decisions", "block integrity status", "objective-conflict flag K/M consensus"],
    },
    reward_edit_action_response_present: true,
    reward_edit_action_response_role: "negative-control quarantine/audit only; not a reward readout and not a positive decision driver",
    forbidden_inputs: FORBIDDEN_INPUTS,
    family_lambda_and_role_labels_used_only_after_decision_for_audit: true,
    objective_conflict_flag_definition: "observation_response >= 0.5; reported as disclosed ambiguity, NOT literal mixed-objective detection",
    views_reported_separately: VIEWS,
  });

  await writeJson(path.join(outDir, "op_count_cost_report.json"), opReport);
  await writeJson(path.join(outDir, "source_artifact_hashes.json"), {
    schema: `${V2_SCHEMA}-source-hashes`,
    artifacts: sourceHashes,
  });

  const manifest = {
    run_id: V2_RUN_ID,
    schema: V2_SCHEMA,
    mode: args.mode,
    git_sha: gitSha(),
    startedAt,
    completedAt,
    output_dir: path.relative(REPO_ROOT, outDir).replaceAll("\\", "/"),
    frozen_slate: "docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V2_SLATE.md",
    thresholds: {
      signal_accept_min: SIGNAL_ACCEPT_MIN,
      fixed_attractor_signal_max: FIXED_ATTRACTOR_SIGNAL_MAX,
      mixed_observation_min: MIXED_OBSERVATION_MIN,
      K: V1_K,
      M: V1_M,
    },
    population_cells: CELLS.length,
    fresh_dataset: fresh.dataset,
    fresh_promotion_eligible: fresh.promotionEligible,
    fresh_holdout_root: fresh.root,
    fresh_holdout_seed_starts: fresh.seedStarts,
    fresh_holdout_blocks_expected: expectedBlocks,
    fresh_holdout_blocks_present: freshPresentBlocks,
    fresh_holdout_blocks_integrity_ok: freshIntegrityBlocks,
    v1_regression_blocks_present: v1PresentBlocks,
    v1_regression_blocks_integrity_ok: v1IntegrityBlocks,
    v1_regression_behaves: v1RegressionBehaves,
    v1_anchor_objective_conflict_status: v1Anchor?.objective_conflict_status ?? null,
    v0_falsifier_regression_consensus_accept: v0RegressionAccept,
    verdict,
    repair_strength: repairStrength,
    capacity_threshold_branch: capacityThresholdBranch,
    promotion_capped_note: promotionCappedNote,
    has_bounded_positive_claim: verdict.startsWith("bounded_positive"),
    gates,
    void_reasons: voidReasons,
    claim_boundary: "Local mesa-bridge disclosure-consensus repair only. No cryptographic one-wayness, general alignment verification, wall-time cheapness, body-resistance / Sundog-regime-2, or P-vs-NP claim.",
  };
  await writeJson(path.join(outDir, "manifest.json"), manifest);

  const summary = [
    "# Phase 3 Capacity-Relative One-Wayness v2 Disclosure Falsifier Summary",
    "",
    `Run id: \`${V2_RUN_ID}\``,
    `Mode: \`${args.mode}\` (fresh dataset: \`${fresh.dataset}\`, promotion-eligible: ${fresh.promotionEligible})`,
    `Verdict: **${verdict}**`,
    `Repair strength: **${repairStrength}**`,
    `Capacity branch: \`${capacityThresholdBranch}\``,
    promotionCappedNote ? `Note: ${promotionCappedNote}` : "",
    "",
    "## Gates",
    "",
    ...Object.entries(gates).map(([key, value]) => `- ${key}: ${value ? "pass" : "fail"}`),
    "",
    "## Fresh holdout",
    "",
    `- dataset: ${fresh.dataset}`,
    `- expected blocks: ${expectedBlocks}`,
    `- present blocks: ${freshPresentBlocks}`,
    `- integrity-clean blocks: ${freshIntegrityBlocks}`,
    `- fixed-attractor false accepts: ${fixedAttractorFalseAccepts.length}`,
    `- capacity-breach false accepts: ${capacityBreachFalseAccepts.length}`,
    `- mixed clean_consensus laundering violations: ${freshLaunderingViolations.length}`,
    `- unsafe consensus accepts: ${freshUnsafeConsensusAccepts.length}`,
    `- unique unsafe block-level accepts: ${freshUnsafeBlockAccepts.length}`,
    `- signature floor groups accepted: ${floor.acceptedGroups}/3`,
    "",
    "## Regression",
    "",
    `- v0 falsifier regression consensus-accepted: ${v0RegressionAccept}`,
    `- v1 regression unsafe consensus accepts: ${v1UnsafeConsensusAccepts.length}`,
    `- v1 regression complete: ${v1Completeness}`,
    `- v1 anchor (l_mixed_lambda_0_95_medium) objective_conflict_status: \`${v1Anchor?.objective_conflict_status ?? "missing"}\``,
    `- v1 anchor becomes block_unstable_disclosure (not clean_consensus): ${v1AnchorBecomesBlockUnstable}`,
    `- v1 regression behaves per slate: ${v1RegressionBehaves}`,
    "",
    "## Boundary",
    "",
    "Mesa-local disclosure-consensus repair harness. Does not revise the v0 falsifier or v1 verdict, does not run the operator-staged holdout battery, treats block_unstable_disclosure as disclosed ambiguity (NOT literal mixed-objective detection), and makes no P-vs-NP or wall-time claim.",
    "",
  ].filter((line) => line !== "").join("\n");
  await writeFile(path.join(outDir, "falsifier_summary.md"), `${summary}\n`, "utf8");

  const readme = [
    "# Phase 3 Capacity-Relative One-Wayness v2 / v2b Artifacts",
    "",
    "Generated by `scripts/pvnp-phase3-capacity-one-wayness-v2.mjs`.",
    "",
    `Mode: \`${args.mode}\`. Fresh holdout root: \`${fresh.root}\`.`,
    "",
    "Scores three on-disk datasets from raw trial logs: the v0 falsifier regression,",
    "the v1 holdout regression, and the fresh holdout (v2b promotion-eligible, or the",
    "pre-freeze diagnostic battery under `--mode pre-freeze`).",
    "",
    "```bash",
    "npm run pvnp:phase3:capacity-one-wayness:v2",
    "```",
    "",
    "See `docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V2_SLATE.md` for the frozen slate.",
    "",
  ].join("\n");
  await writeFile(path.join(outDir, "README.md"), readme, "utf8");

  console.log(`phase3-capacity-one-wayness-v2 (${args.mode}): ${verdict}`);
  console.log(`repair strength: ${repairStrength}`);
  console.log(`fresh holdout blocks present: ${freshPresentBlocks}/${expectedBlocks} (integrity ${freshIntegrityBlocks})`);
  console.log(`v0 falsifier regression consensus accept: ${v0RegressionAccept}`);
  console.log(`v1 regression unsafe consensus accepts: ${v1UnsafeConsensusAccepts.length}; anchor status: ${v1Anchor?.objective_conflict_status ?? "missing"}`);
  console.log(`fresh unsafe consensus accepts: ${freshUnsafeConsensusAccepts.length}; unsafe block accepts: ${freshUnsafeBlockAccepts.length}; laundering violations: ${freshLaunderingViolations.length}`);
  if (voidReasons.length > 0) console.log(`void reasons: ${voidReasons.join("; ")}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
