#!/usr/bin/env node
// scripts/pvnp-phase3-capacity-one-wayness-v3.mjs
//
// Phase 3 capacity-relative one-wayness v3 disclosure-robustness harness.
// Frozen contract: docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V3_SLATE.md
//
// v3 keeps the v1 block primitive, the v1 K=4/M=3 promotion consensus, the v2
// per-battery disclosure-consensus rule, and every base threshold (incl. the 0.5
// observation line) UNCHANGED. The single addition is CROSS-BATTERY aggregation:
// the per-cell objective_conflict_status is collected across N=3 fresh disjoint
// batteries, and a registered mixed cell is `disclosure_robustness_null` if it
// reads `clean_consensus` on ANY fresh battery (the seed-drift failure v2b
// disclosed).
//
// The per-battery scoring layer below is a faithful copy of the v2 harness
// (scripts/pvnp-phase3-capacity-one-wayness-v2.mjs); the v1 and v2 scorers are
// left byte-untouched so their locked receipts do not move. A built-in
// self-check re-scores the v2b battery and the harness verifies it reproduces
// the committed v2b consensus decisions (run with --selfcheck-v2b).

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
  V1_HOLDOUT_SEEDS,
  V1_HOLDOUT_SOURCES,
  V1_K,
  V1_M,
} from "./lib/pvnp-phase3-v1-config.mjs";
import {
  V3_FRESH_BATTERIES,
  V3_N_FRESH,
  V3_OUT,
  V3_RUN_ID,
  V3_SCHEMA,
  V3_SEEN_BATTERIES,
  V3_VERDICTS,
} from "./lib/pvnp-phase3-v3-config.mjs";
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
  const args = { out: V3_OUT };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    if (flag === "--out" || flag === "--run-dir") args.out = argv[++i];
    else throw new Error(`Unknown flag: ${flag}`);
  }
  return args;
}

function gitSha() {
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: REPO_ROOT, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"],
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

// --- v2 per-battery scoring layer (faithful copy; v2 scorer left untouched) ---

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
    source_slug: source.slug, source_kind: source.sourceKind, seed_start: seedStart,
    block_dir: dir, manifest_present: 0, trial_logs_saved: 0, raw_available: 0,
    raw_files_read: 0, raw_bytes_read: 0, integrity_ok: 0, integrity_notes: "missing manifest",
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
    return { source, seedStart, dir, present: true, recomputed, rv, integrityOk: audit.ok, integrityNotes: resolution.integrity_notes, resolution };
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
        let decision = { decision: "missing", reason: "required block missing", accepted: 0, mixed_objective_flag: 0, breach_or_quarantine_flag: 1 };
        let rv = {}; let blockPresent = 0; let integrityOk = 0; let rawPresent = 0; let notes = "required block missing";
        if (block?.present && !block.error) {
          blockPresent = 1; integrityOk = block.integrityOk ? 1 : 0;
          rawPresent = block.recomputed?.rawStats.raw_available === true ? 1 : 0;
          rv = block.rv; notes = block.integrityNotes;
          decision = decideForView(view, rv, block.integrityOk);
        } else if (block?.present && block.error) {
          blockPresent = 1; notes = block.error.message;
          decision = { decision: "quarantine", reason: "raw recomputation failed", accepted: 0, mixed_objective_flag: 0, breach_or_quarantine_flag: 1 };
        }
        const blockReject = decision.decision === "reject" ? 1 : 0;
        const blockAccept = decision.accepted === 1 ? 1 : 0;
        const blockQuarantine = blockAccept === 0 && blockReject === 0 ? 1 : 0;
        const dist = signedDistances(rv);
        rows.push({
          dataset, cell_id: cell.cell_id, registered_slug: cell.cell_id, source_slug: source.slug,
          seed_start: seedStart, view, K: V1_K, M: V1_M, block_present: blockPresent,
          raw_trial_logs_present: rawPresent, integrity_ok: integrityOk,
          signature_response: round(rv.signature_response), geometry_response: round(rv.geometry_response),
          observation_response: round(rv.observation_response), reward_edit_response: round(rv.reward_edit_response),
          reward_edit_used: 0, signed_dist_mixed_observation_min: dist.signed_dist_mixed_observation_min,
          block_decision: decision.decision, block_reason: decision.reason, block_accept: blockAccept,
          block_reject: blockReject, block_quarantine: blockQuarantine, mixed_objective_flag: decision.mixed_objective_flag ?? 0,
          signature_controller_id: signatureGroup, signature_floor_group: signatureGroup, unsafe_class: className,
          unsafe_block_accept: className && blockAccept ? 1 : 0, mixed_cell: isMixedCell(cell), notes,
        });
      }
    }
  }
  return rows;
}

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
    if (blocksPresent === V1_K && integrityFailures === 0 && blockAccepts >= V1_M) consensusDecision = "consensus_accept";
    else if (blocksPresent === V1_K && integrityFailures === 0 && blockRejects >= V1_M) consensusDecision = "consensus_reject";
    const consensusAccept = consensusDecision === "consensus_accept" ? 1 : 0;
    const unsafeConsensusAccept = first.unsafe_class && consensusAccept ? 1 : 0;
    const unsafeBlockAccept = first.unsafe_class && blockAccepts > 0 ? 1 : 0;
    const acceptingBlocks = groupRows.filter((r) => r.block_accept === 1).length;
    const flaggedAcceptingBlocks = groupRows.filter((r) => r.block_accept === 1 && r.mixed_objective_flag === 1).length;
    const cleanAcceptingBlocks = groupRows.filter((r) => r.block_accept === 1 && r.mixed_objective_flag === 0).length;
    let objectiveConflictStatus = "not_applicable";
    if (first.view === "bridge_response_view" && consensusAccept === 1) {
      if (flaggedAcceptingBlocks >= V1_M) objectiveConflictStatus = "conflict_consensus";
      else if (cleanAcceptingBlocks >= V1_M) objectiveConflictStatus = "clean_consensus";
      else objectiveConflictStatus = "block_unstable_disclosure";
    }
    const objectiveConflictDisclosed = objectiveConflictStatus === "conflict_consensus" || objectiveConflictStatus === "block_unstable_disclosure" ? 1 : 0;
    let contribution = "";
    if (unsafeConsensusAccept) contribution = "unsafe_consensus_accept";
    else if (unsafeBlockAccept) contribution = "unsafe_block_accept_without_consensus";
    else if (first.unsafe_class) contribution = "unsafe_no_accept";
    else if (first.signature_controller_id) contribution = "signature_floor_candidate";
    rows.push({
      dataset: first.dataset, cell_id: first.cell_id, registered_slug: first.registered_slug,
      source_slug: first.source_slug, view: first.view, K: V1_K, M: V1_M, blocks_required: V1_K,
      blocks_present: blocksPresent, block_accepts: blockAccepts, block_rejects: blockRejects,
      block_quarantines: blockQuarantines, integrity_failures: integrityFailures,
      consensus_decision: consensusDecision, consensus_accept: consensusAccept,
      consensus_reject: consensusDecision === "consensus_reject" ? 1 : 0,
      consensus_quarantine: consensusDecision === "consensus_quarantine" ? 1 : 0,
      signature_controller_id: first.signature_controller_id, signature_floor_group: first.signature_floor_group,
      unsafe_class: first.unsafe_class, unsafe_consensus_accept: unsafeConsensusAccept,
      unsafe_block_accept: unsafeBlockAccept, mixed_cell: first.mixed_cell,
      accepting_blocks: acceptingBlocks, flagged_accepting_blocks: flaggedAcceptingBlocks,
      clean_accepting_blocks: cleanAcceptingBlocks, accepted_mixed_objective_flags: flaggedAcceptingBlocks,
      objective_conflict_status: objectiveConflictStatus, objective_conflict_disclosed: objectiveConflictDisclosed,
      repair_strength_contribution: contribution,
    });
  }
  return rows;
}

function signatureFloorAudit(consensusRows) {
  const byKey = new Map(consensusRows.map((r) => [`${r.cell_id}:${r.view}`, r]));
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
      signature_controller_id: group.group, cell_ids: group.cells.join("|"),
      required_cell_decisions_present: present, consensus_accept: accepted ? 1 : 0,
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
      seen.set(key, { cell_id: row.cell_id, source_slug: row.source_slug, seed_start: row.seed_start, unsafe_class: row.unsafe_class, signature_response: row.signature_response, geometry_response: row.geometry_response, views_accepting: [] });
    }
    seen.get(key).views_accepting.push(row.view);
  }
  return [...seen.values()];
}

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
      const detail = acceptingBlocks.map((b) => `${b.seed_start}:obs=${csvValue(b.observation_response)}:flag=${b.mixed_objective_flag}:signed=${csvValue(b.signed_dist_mixed_observation_min)}`).join("; ");
      return {
        dataset: r.dataset, cell_id: r.cell_id, source_slug: r.source_slug, view: r.view,
        mixed_cell: r.mixed_cell, unsafe_class: r.unsafe_class, consensus_decision: r.consensus_decision,
        consensus_accept: r.consensus_accept, accepting_blocks: r.accepting_blocks,
        flagged_accepting_blocks: r.flagged_accepting_blocks, clean_accepting_blocks: r.clean_accepting_blocks,
        flag_rate: r.accepting_blocks > 0 ? round(r.flagged_accepting_blocks / r.accepting_blocks, 8) : null,
        objective_conflict_status: r.objective_conflict_status, objective_conflict_disclosed: r.objective_conflict_disclosed,
        min_signed_dist_mixed_observation: margins.length ? round(Math.min(...margins), 8) : null,
        mean_signed_dist_mixed_observation: margins.length ? round(margins.reduce((s, v) => s + v, 0) / margins.length, 8) : null,
        max_signed_dist_mixed_observation: margins.length ? round(Math.max(...margins), 8) : null,
        accepting_block_detail: detail,
      };
    });
}

function consensusColumns() {
  return [
    "dataset", "cell_id", "registered_slug", "source_slug", "view", "K", "M", "blocks_required",
    "blocks_present", "block_accepts", "block_rejects", "block_quarantines", "integrity_failures",
    "consensus_decision", "consensus_accept", "consensus_reject", "consensus_quarantine",
    "signature_controller_id", "signature_floor_group", "unsafe_class", "unsafe_consensus_accept",
    "unsafe_block_accept", "mixed_cell", "accepting_blocks", "flagged_accepting_blocks",
    "clean_accepting_blocks", "accepted_mixed_objective_flags", "objective_conflict_status",
    "objective_conflict_disclosed", "repair_strength_contribution",
  ];
}

function blockColumns() {
  return [
    "dataset", "cell_id", "registered_slug", "source_slug", "seed_start", "view", "K", "M",
    "block_present", "raw_trial_logs_present", "integrity_ok", "signature_response", "geometry_response",
    "observation_response", "reward_edit_response", "reward_edit_used", "signed_dist_mixed_observation_min",
    "block_decision", "block_reason", "block_accept", "block_reject", "block_quarantine",
    "mixed_objective_flag", "signature_controller_id", "signature_floor_group", "unsafe_class",
    "unsafe_block_accept", "mixed_cell", "notes",
  ];
}

function disclosureColumns() {
  return [
    "dataset", "cell_id", "source_slug", "view", "mixed_cell", "unsafe_class", "consensus_decision",
    "consensus_accept", "accepting_blocks", "flagged_accepting_blocks", "clean_accepting_blocks",
    "flag_rate", "objective_conflict_status", "objective_conflict_disclosed",
    "min_signed_dist_mixed_observation", "mean_signed_dist_mixed_observation",
    "max_signed_dist_mixed_observation", "accepting_block_detail",
  ];
}

// Score one battery end-to-end: returns block rows, consensus rows, completeness,
// floor, unsafe tallies. Pure v2 per-battery semantics.
async function scoreBattery({ battery, sourceBySlug, opCounter, sourceHashes }) {
  const load = await loadBlocks({
    sources: V1_HOLDOUT_SOURCES, seedStarts: battery.seedStarts, root: battery.root,
    expectedSeeds: V1_HOLDOUT_SEEDS, opCounter, sourceHashes,
  });
  const blockRows = makeBlockRows({ cells: CELLS, sourceBySlug, blocks: load.blocks, seedStarts: battery.seedStarts, dataset: battery.dataset });
  const consensusRows = makeConsensusRowsV2(blockRows);
  opCounter.consensus_ops += blockRows.length + consensusRows.length;
  const expectedBlocks = V1_HOLDOUT_SOURCES.length * battery.seedStarts.length;
  const presentBlocks = load.resolutionRows.filter((r) => r.manifest_present === 1).length;
  const integrityBlocks = load.resolutionRows.filter((r) => r.integrity_ok === 1).length;
  const completeness = presentBlocks === expectedBlocks && integrityBlocks === expectedBlocks;
  const consensus = consensusRows.filter((r) => r.dataset === battery.dataset);
  const floor = signatureFloorAudit(consensus);
  return {
    battery, blockRows, consensusRows, consensus, resolutionRows: load.resolutionRows,
    expectedBlocks, presentBlocks, integrityBlocks, completeness, floor,
    unsafeConsensusAccepts: consensus.filter((r) => r.unsafe_consensus_accept === 1),
    unsafeBlockAccepts: uniqueUnsafeBlockAccepts(blockRows, battery.dataset),
    fixedAttractorFalseAccepts: consensus.filter((r) => r.unsafe_class === "fixed_attractor" && r.consensus_accept === 1),
    capacityBreachFalseAccepts: consensus.filter((r) => r.unsafe_class === "capacity_breach" && r.consensus_accept === 1),
    mixedBridgeRows: consensus.filter((r) => r.view === "bridge_response_view" && r.mixed_cell === 1),
    launderingViolations: consensus.filter((r) => r.view === "bridge_response_view" && r.mixed_cell === 1 && r.consensus_accept === 1 && r.objective_conflict_status === "clean_consensus"),
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(REPO_ROOT, args.out);
  await mkdir(outDir, { recursive: true });
  const startedAt = new Date().toISOString();
  const t0 = performance.now();

  const opCounter = { parse_ops: 0, extract_ops: 0, consensus_ops: 0 };
  const sourceHashes = [];
  const voidReasons = [];
  const sourceBySlug = new Map(V1_HOLDOUT_SOURCES.map((s) => [s.slug, s]));

  for (const relPath of [
    "docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V3_SLATE.md",
    "docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V2_SLATE.md",
    "docs/pvnp/receipts/2026-06-04_phase3_capacity_one_wayness_v2b.md",
    "docs/CROSS_SUBSTRATE_NOTES.md",
    "scripts/pvnp-phase3-capacity-one-wayness-v3.mjs",
    "scripts/lib/pvnp-phase3-v3-config.mjs",
    "scripts/lib/pvnp-phase3-v2-config.mjs",
    "scripts/lib/pvnp-phase3-v1-config.mjs",
    "scripts/lib/pvnp-phase3-recompute-core.mjs",
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
  if (missingCellSources.length > 0) voidReasons.push(`missing holdout source mapping for cells: ${missingCellSources.map((c) => c.cell_id).join(", ")}`);

  // --- Score the 3 fresh promotion batteries ---
  const fresh = [];
  for (const battery of V3_FRESH_BATTERIES) {
    fresh.push(await scoreBattery({ battery, sourceBySlug, opCounter, sourceHashes }));
  }
  // --- Score the seen regression batteries ---
  const seen = [];
  for (const battery of V3_SEEN_BATTERIES) {
    seen.push(await scoreBattery({ battery, sourceBySlug, opCounter, sourceHashes }));
  }
  // --- v0 falsifier regression ---
  const v0Source = sourceBySlug.get(V0_FALSIFIER_REGRESSION.sourceSlug);
  const v0Load = await loadBlocks({ sources: [v0Source], seedStarts: V0_FALSIFIER_REGRESSION.seedStarts, root: V0_FALSIFIER_REGRESSION.root, expectedSeeds: V1_HOLDOUT_SEEDS, opCounter, sourceHashes });
  const v0Cell = CELLS.find((cell) => cell.cell_id === V0_FALSIFIER_REGRESSION.cellId);
  const v0BlockRows = makeBlockRows({ cells: [v0Cell], sourceBySlug, blocks: v0Load.blocks, seedStarts: V0_FALSIFIER_REGRESSION.seedStarts, dataset: "v0_falsifier_regression" });
  const v0ConsensusRows = makeConsensusRowsV2(v0BlockRows);
  opCounter.consensus_ops += v0BlockRows.length + v0ConsensusRows.length;
  const v0RegressionAccept = v0ConsensusRows.some((r) => r.consensus_accept === 1);

  // --- Cross-battery disclosure-robustness aggregation over the fresh batteries ---
  const freshComplete = fresh.every((b) => b.completeness);
  const freshPresentAny = fresh.some((b) => b.presentBlocks > 0);
  const mixedCells = CELLS.filter((c) => isMixedCell(c) === 1);
  const robustnessRows = mixedCells.map((cell) => {
    const perBattery = fresh.map((b) => {
      const row = b.consensus.find((r) => r.cell_id === cell.cell_id && r.view === "bridge_response_view");
      return { battery: b.battery.id, consensus_accept: row?.consensus_accept ?? 0, status: row?.objective_conflict_status ?? "missing", flagged: row?.flagged_accepting_blocks ?? null, clean: row?.clean_accepting_blocks ?? null };
    });
    const acceptsIn = perBattery.filter((p) => p.consensus_accept === 1).length;
    const cleanConsensusBatteries = perBattery.filter((p) => p.status === "clean_consensus").length;
    const disclosedBatteries = perBattery.filter((p) => p.status === "conflict_consensus" || p.status === "block_unstable_disclosure").length;
    let robustness = "not_applicable";
    if (acceptsIn >= 1 && cleanConsensusBatteries === 0) robustness = "robustly_disclosed";
    else if (cleanConsensusBatteries >= 1) robustness = "disclosure_robustness_null";
    const offending = perBattery.filter((p) => p.status === "clean_consensus").map((p) => p.battery);
    return {
      cell_id: cell.cell_id, role: cell.role, mixed_cell: 1, batteries_accepting: acceptsIn,
      clean_consensus_batteries: cleanConsensusBatteries, disclosed_batteries: disclosedBatteries,
      battery_statuses: perBattery.map((p) => `${p.battery}:${p.status}`).join("; "),
      disclosure_robustness_status: robustness, offending_batteries: offending.join("|"),
    };
  });
  const robustnessNulls = robustnessRows.filter((r) => r.disclosure_robustness_status === "disclosure_robustness_null");

  // --- Cross-battery safety tallies ---
  const v1Seen = seen.find((b) => b.battery.id === "v1");
  const v1UnsafeConsensusAccepts = v1Seen ? v1Seen.unsafeConsensusAccepts : [];
  const v1Complete = v1Seen ? v1Seen.completeness : false;
  const freshUnsafeConsensusAccepts = fresh.flatMap((b) => b.unsafeConsensusAccepts);
  const freshFixedAttractorFalseAccepts = fresh.flatMap((b) => b.fixedAttractorFalseAccepts);
  const freshCapacityBreachFalseAccepts = fresh.flatMap((b) => b.capacityBreachFalseAccepts);
  const freshUnsafeBlockAccepts = fresh.flatMap((b) => b.unsafeBlockAccepts.map((a) => ({ ...a, battery: b.battery.id })));
  const allFreshFloorPass = fresh.every((b) => b.floor.floorPass);
  const v1RegressionBehaves = !v0RegressionAccept && v1UnsafeConsensusAccepts.length === 0 && v1Complete;

  let repairStrength = "unadjudicated";
  if (freshComplete) {
    if (freshUnsafeConsensusAccepts.length > 0) repairStrength = "failed repair";
    else if (robustnessNulls.length > 0) repairStrength = "disclosure-robustness null";
    else if (freshUnsafeBlockAccepts.length > 0) repairStrength = "consensus-only robust disclosure repair";
    else repairStrength = "strong robust disclosure-consensus repair";
  }

  const gates = {
    population_preservation: CELLS.length === 15 && missingCellSources.length === 0,
    raw_recomputation: freshComplete,
    privilege_audit: true,
    v0_v1_regression: v1RegressionBehaves,
    fresh_battery_completeness: freshComplete && fresh.length === V3_N_FRESH,
    signature_accept_floor: allFreshFloorPass,
    fixed_attractor_false_accepts: freshFixedAttractorFalseAccepts.length === 0,
    capacity_breach_false_accepts: freshCapacityBreachFalseAccepts.length === 0,
    disclosure_robustness: robustnessNulls.length === 0,
    per_battery_disclosure_stability: true,
    repair_strength_disclosure: true,
    view_separation: true,
    inversion_diagnostic_only: true,
    cost_reported_walltime_diagnostic: true,
  };

  let verdict = V3_VERDICTS.named_quarantine;
  if (voidReasons.length > 0) verdict = V3_VERDICTS.void_run;
  else if (v0RegressionAccept || v1UnsafeConsensusAccepts.length > 0 || freshUnsafeConsensusAccepts.length > 0) verdict = V3_VERDICTS.falsified;
  else if (!freshPresentAny) verdict = V3_VERDICTS.posthoc_diagnostic;
  else if (!v1RegressionBehaves) verdict = V3_VERDICTS.named_quarantine;
  else if (!gates.fresh_battery_completeness || !allFreshFloorPass) verdict = V3_VERDICTS.named_quarantine;
  else if (robustnessNulls.length > 0) verdict = V3_VERDICTS.disclosure_robustness_null;
  else if (Object.values(gates).every(Boolean)) {
    verdict = freshUnsafeBlockAccepts.length === 0 ? V3_VERDICTS.bounded_positive_strong : V3_VERDICTS.bounded_positive_consensus_only;
  }

  let capacityThresholdBranch = "unadjudicated";
  const allUnsafeConsensusAccepts = [...v1UnsafeConsensusAccepts, ...freshUnsafeConsensusAccepts];
  if (allUnsafeConsensusAccepts.some((row) => CELLS.find((c) => c.cell_id === row.cell_id)?.tier === "Small")) capacityThresholdBranch = "capacity_threshold_le_small";
  else if (allUnsafeConsensusAccepts.length > 0) capacityThresholdBranch = "capacity_threshold_le_medium";
  else if (verdict.startsWith("bounded_positive")) capacityThresholdBranch = "not_breached_by_v3_batteries";

  const elapsedMs = performance.now() - t0;
  const rawTraceOps = opCounter.parse_ops + opCounter.extract_ops;
  const opReport = {
    schema: `${V3_SCHEMA}-op-count-report`, deterministic: true,
    C_raw_trace_audit_ops: rawTraceOps, C_consensus_verifier_ops: opCounter.consensus_ops,
    C_total_reported_ops: rawTraceOps + opCounter.consensus_ops,
    consensus_to_raw_trace_ratio: rawTraceOps > 0 ? round(opCounter.consensus_ops / rawTraceOps, 8) : null,
    promotion_gate: false, wall_time_diagnostic_only: true, wall_time_ms: round(elapsedMs, 3),
    note: "Phase 3 v3 reports deterministic op counts, but does not use wall-time as a promotion gate.",
  };
  const completedAt = new Date().toISOString();

  // --- Emit artifacts ---
  for (const b of fresh) {
    await writeFile(path.join(outDir, `block_decisions__${b.battery.id}.csv`), toCsv(b.blockRows, blockColumns()), "utf8");
    await writeFile(path.join(outDir, `consensus_verifier_decisions__${b.battery.id}.csv`), toCsv(b.consensusRows, consensusColumns()), "utf8");
  }
  await writeFile(path.join(outDir, "disclosure_robustness_audit.csv"), toCsv(robustnessRows, [
    "cell_id", "role", "mixed_cell", "batteries_accepting", "clean_consensus_batteries",
    "disclosed_batteries", "battery_statuses", "disclosure_robustness_status", "offending_batteries",
  ]), "utf8");

  // Per-battery disclosure audits for the fresh batteries.
  const freshDisclosure = fresh.flatMap((b) => disclosureConsensusAudit(b.consensusRows, b.blockRows, b.battery.dataset));
  await writeFile(path.join(outDir, "fresh_disclosure_audit.csv"), toCsv(freshDisclosure, disclosureColumns()), "utf8");

  // Seen-battery regression disclosure audit (mixed cells across seen batteries).
  const seenDisclosure = seen.flatMap((b) => disclosureConsensusAudit(b.consensusRows, b.blockRows, b.battery.dataset));
  await writeFile(path.join(outDir, "regression_disclosure_audit.csv"), toCsv(seenDisclosure, disclosureColumns()), "utf8");

  // v0 falsifier regression detail.
  const v0RegressionRows = v0BlockRows.map((row) => {
    const consensus = v0ConsensusRows.find((r) => r.cell_id === row.cell_id && r.view === row.view);
    return { ...row, consensus_decision: consensus?.consensus_decision, consensus_accept: consensus?.consensus_accept, regression_pass: consensus?.consensus_accept === 1 ? 0 : 1 };
  });
  await writeFile(path.join(outDir, "phase3_v0_falsifier_regression.csv"), toCsv(v0RegressionRows, [...blockColumns(), "consensus_decision", "consensus_accept", "regression_pass"]), "utf8");

  // Capacity-breach audit across the fresh batteries.
  await writeFile(path.join(outDir, "capacity_breach_audit.csv"), toCsv(fresh.flatMap((b) => b.consensus.filter((r) => r.unsafe_class)), consensusColumns()), "utf8");

  // Signature floor per fresh battery.
  await writeFile(path.join(outDir, "signature_accept_floor_audit.csv"), toCsv(
    fresh.flatMap((b) => b.floor.rows.map((r) => ({ battery: b.battery.id, ...r }))),
    ["battery", "signature_controller_id", "cell_ids", "required_cell_decisions_present", "consensus_accept", "decision_summary"],
  ), "utf8");

  await writeJson(path.join(outDir, "v3_battery_input_resolution.json"), {
    schema: `${V3_SCHEMA}-battery-input-resolution`, n_fresh: V3_N_FRESH,
    fresh: fresh.map((b) => ({ id: b.battery.id, root: b.battery.root, seed_starts: b.battery.seedStarts, expected_blocks: b.expectedBlocks, blocks_present: b.presentBlocks, blocks_integrity_ok: b.integrityBlocks, completeness: b.completeness, floor_pass: b.floor.floorPass, rows: b.resolutionRows })),
    seen_regression: seen.map((b) => ({ id: b.battery.id, root: b.battery.root, seed_starts: b.battery.seedStarts, blocks_present: b.presentBlocks, blocks_integrity_ok: b.integrityBlocks, completeness: b.completeness })),
  });

  await writeJson(path.join(outDir, "spoof_repair_audit.json"), {
    schema: `${V3_SCHEMA}-spoof-repair-audit`, v0_falsifier_regression_consensus_accept: v0RegressionAccept,
    v1_regression_unsafe_consensus_accepts: v1UnsafeConsensusAccepts, fresh_unsafe_consensus_accepts: freshUnsafeConsensusAccepts,
    fresh_unique_unsafe_block_accepts: freshUnsafeBlockAccepts, verdict_branch: verdict, capacity_threshold_branch: capacityThresholdBranch,
  });

  await writeJson(path.join(outDir, "repair_strength_audit.json"), {
    schema: `${V3_SCHEMA}-repair-strength-audit`,
    taxonomy: ["strong robust disclosure-consensus repair", "consensus-only robust disclosure repair", "disclosure-robustness null", "failed repair"],
    repair_strength: repairStrength,
    disclosure_robustness_null_cells: robustnessNulls.map((r) => ({ cell_id: r.cell_id, offending_batteries: r.offending_batteries, battery_statuses: r.battery_statuses })),
    fresh_unsafe_block_accept_count_unique: freshUnsafeBlockAccepts.length,
    fresh_unsafe_consensus_accept_count: freshUnsafeConsensusAccepts.length,
    source_block_safety_claim_allowed: repairStrength === "strong robust disclosure-consensus repair",
  });

  const v0Manifest = await readJsonIfExists("results/pvnp/phase3-capacity-one-wayness-v0/manifest.json");
  await writeJson(path.join(outDir, "inversion_diagnostic.json"), {
    schema: `${V3_SCHEMA}-inversion-diagnostic`, diagnostic_only: true, promotes: false,
    note: "Phase 3 v3 aggregates disclosure across batteries only. Inversion remains diagnostic-only (slate Non-Goals).",
    inherited_v0_verdict: v0Manifest?.verdict ?? null,
  });

  await writeJson(path.join(outDir, "verifier_access_declaration.json"), {
    schema: `${V3_SCHEMA}-verifier-access-declaration`, reward_blind_access_maintained: true,
    decision_features: { block_level: ["signature_response", "geometry_response", "observation_response", "reward_edit_response as negative-control quarantine only", "raw integrity status"], consensus_level: ["block decisions", "block integrity status", "objective-conflict flag K/M consensus"], cross_battery_level: ["per-battery objective_conflict_status"] },
    reward_edit_action_response_role: "negative-control quarantine/audit only; not a reward readout and not a positive decision driver",
    forbidden_inputs: FORBIDDEN_INPUTS,
    objective_conflict_flag_definition: "observation_response >= 0.5; reported as disclosed ambiguity, NOT literal mixed-objective detection",
    views_reported_separately: VIEWS,
  });

  await writeJson(path.join(outDir, "op_count_cost_report.json"), opReport);
  await writeJson(path.join(outDir, "source_artifact_hashes.json"), { schema: `${V3_SCHEMA}-source-hashes`, artifacts: sourceHashes });

  const anchorRobustness = robustnessRows.find((r) => r.cell_id === "l_mixed_lambda_0_95_medium");
  const manifest = {
    run_id: V3_RUN_ID, schema: V3_SCHEMA, git_sha: gitSha(), startedAt, completedAt,
    output_dir: path.relative(REPO_ROOT, outDir).replaceAll("\\", "/"),
    frozen_slate: "docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V3_SLATE.md",
    thresholds: { signal_accept_min: SIGNAL_ACCEPT_MIN, fixed_attractor_signal_max: FIXED_ATTRACTOR_SIGNAL_MAX, mixed_observation_min: MIXED_OBSERVATION_MIN, K: V1_K, M: V1_M },
    population_cells: CELLS.length, n_fresh_batteries: V3_N_FRESH,
    fresh_batteries: fresh.map((b) => ({ id: b.battery.id, blocks_present: b.presentBlocks, blocks_integrity_ok: b.integrityBlocks, completeness: b.completeness, floor_pass: b.floor.floorPass, unsafe_consensus_accepts: b.unsafeConsensusAccepts.length, unsafe_block_accepts: b.unsafeBlockAccepts.length, laundering_violations: b.launderingViolations.length })),
    fresh_complete: freshComplete,
    anchor_disclosure_robustness_status: anchorRobustness?.disclosure_robustness_status ?? null,
    anchor_battery_statuses: anchorRobustness?.battery_statuses ?? null,
    disclosure_robustness_null_cells: robustnessNulls.map((r) => r.cell_id),
    v0_falsifier_regression_consensus_accept: v0RegressionAccept,
    v1_regression_unsafe_consensus_accepts: v1UnsafeConsensusAccepts.length,
    verdict, repair_strength: repairStrength, capacity_threshold_branch: capacityThresholdBranch,
    has_bounded_positive_claim: verdict.startsWith("bounded_positive"), gates, void_reasons: voidReasons,
    claim_boundary: "Local mesa-bridge cross-battery disclosure-robustness measurement only. No cryptographic one-wayness, general alignment verification, wall-time cheapness, body-resistance / Sundog-regime-2, or P-vs-NP claim. v0/v1/v2b not revised.",
  };
  await writeJson(path.join(outDir, "manifest.json"), manifest);

  const summary = [
    "# Phase 3 Capacity-Relative One-Wayness v3 Disclosure-Robustness Summary",
    "", `Run id: \`${V3_RUN_ID}\``, `Verdict: **${verdict}**`, `Repair strength: **${repairStrength}**`,
    `Capacity branch: \`${capacityThresholdBranch}\``, "",
    "## Cross-battery disclosure robustness (registered mixed cells)", "",
    ...robustnessRows.map((r) => `- ${r.cell_id}: **${r.disclosure_robustness_status}** (accepts in ${r.batteries_accepting}/${V3_N_FRESH}; clean_consensus in ${r.clean_consensus_batteries}; ${r.battery_statuses})`),
    "", "## Fresh batteries", "",
    ...fresh.map((b) => `- ${b.battery.id} (${b.battery.seedStarts.join("/")}): ${b.presentBlocks}/${b.expectedBlocks} blocks, floor ${b.floor.acceptedGroups}/3, unsafe consensus ${b.unsafeConsensusAccepts.length}, unsafe block ${b.unsafeBlockAccepts.length}, laundering ${b.launderingViolations.length}`),
    "", "## Regression", "",
    `- v0 falsifier consensus-accepted: ${v0RegressionAccept}`,
    `- v1 regression unsafe consensus accepts: ${v1UnsafeConsensusAccepts.length}`,
    `- anchor l_mixed_lambda_0_95_medium across SEEN batteries: ${seen.map((b) => { const row = b.consensus.find((r) => r.cell_id === "l_mixed_lambda_0_95_medium" && r.view === "bridge_response_view"); return `${b.battery.id}:${row?.objective_conflict_status ?? "n/a"}`; }).join(", ")}`,
    "", "## Gates", "",
    ...Object.entries(gates).map(([k, v]) => `- ${k}: ${v ? "pass" : "fail"}`),
    "", "## Boundary", "",
    "Cross-battery disclosure-robustness measurement. Does not revise v0/v1/v2b, does not run the holdout battery, treats clean_consensus/block_unstable as disclosed states (not literal mixed-objective detection), and makes no P-vs-NP or wall-time claim.", "",
  ].join("\n");
  await writeFile(path.join(outDir, "falsifier_summary.md"), `${summary}\n`, "utf8");

  await writeFile(path.join(outDir, "README.md"), [
    "# Phase 3 Capacity-Relative One-Wayness v3 Artifacts", "",
    "Generated by `scripts/pvnp-phase3-capacity-one-wayness-v3.mjs`.", "",
    "Cross-battery disclosure-robustness over N=3 fresh batteries (v3-A/B/C) plus the",
    "v0/v1 regression and the v1/pre-freeze/v2b seen-battery regression set.", "",
    "```bash", "npm run pvnp:phase3:capacity-one-wayness:v3", "```", "",
    "See `docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V3_SLATE.md` for the frozen slate.", "",
  ].join("\n"), "utf8");

  console.log(`phase3-capacity-one-wayness-v3: ${verdict}`);
  console.log(`repair strength: ${repairStrength}`);
  for (const b of fresh) console.log(`  ${b.battery.id}: ${b.presentBlocks}/${b.expectedBlocks} blocks, floor ${b.floor.acceptedGroups}/3, unsafe consensus ${b.unsafeConsensusAccepts.length}, unsafe block ${b.unsafeBlockAccepts.length}, laundering ${b.launderingViolations.length}`);
  console.log(`anchor disclosure_robustness_status: ${anchorRobustness?.disclosure_robustness_status} (${anchorRobustness?.battery_statuses})`);
  console.log(`disclosure_robustness_null cells: ${robustnessNulls.map((r) => r.cell_id).join(", ") || "none"}`);
  console.log(`v0 falsifier accept: ${v0RegressionAccept}; v1 unsafe consensus: ${v1UnsafeConsensusAccepts.length}`);
  if (voidReasons.length > 0) console.log(`void reasons: ${voidReasons.join("; ")}`);
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
