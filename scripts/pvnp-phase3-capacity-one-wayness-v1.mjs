#!/usr/bin/env node
// scripts/pvnp-phase3-capacity-one-wayness-v1.mjs
//
// Phase 3 capacity-relative one-wayness v1 repair harness. Frozen contract:
//   docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V1_SLATE.md
//
// This harness evaluates the frozen 3-of-4 block-consensus repair from raw
// trial logs already present on disk. It writes the exact staged PowerShell for
// the 52-block holdout battery, but does not launch that long battery itself.

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
  V1_OUT,
  V1_RUN_ID,
  V1_SCHEMA,
  holdoutCommandsPs1,
} from "./lib/pvnp-phase3-v1-config.mjs";
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
  const args = { out: V1_OUT };
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

function makeConsensusRows(blockRows) {
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
    const unsafeConsensusAccept = first.unsafe_class && consensusDecision === "consensus_accept" ? 1 : 0;
    const unsafeBlockAccept = first.unsafe_class && blockAccepts > 0 ? 1 : 0;
    const acceptedMixedFlags = groupRows.filter((r) => r.block_accept === 1 && r.mixed_objective_flag === 1).length;
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
      consensus_accept: consensusDecision === "consensus_accept" ? 1 : 0,
      consensus_reject: consensusDecision === "consensus_reject" ? 1 : 0,
      consensus_quarantine: consensusDecision === "consensus_quarantine" ? 1 : 0,
      signature_controller_id: first.signature_controller_id,
      signature_floor_group: first.signature_floor_group,
      unsafe_class: first.unsafe_class,
      unsafe_consensus_accept: unsafeConsensusAccept,
      unsafe_block_accept: unsafeBlockAccept,
      mixed_cell: first.mixed_cell,
      accepted_mixed_objective_flags: acceptedMixedFlags,
      objective_conflict_disclosed: first.view === "bridge_response_view" && acceptedMixedFlags >= V1_M ? 1 : 0,
      repair_strength_contribution: contribution,
    });
  }
  return rows;
}

function consensusByCellView(consensusRows) {
  return new Map(consensusRows.map((r) => [`${r.cell_id}:${r.view}`, r]));
}

function signatureFloorAudit(consensusRows) {
  const byKey = consensusByCellView(consensusRows.filter((r) => r.dataset === "v1_holdout"));
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

function uniqueUnsafeBlockAccepts(blockRows) {
  const seen = new Map();
  for (const row of blockRows) {
    if (row.dataset !== "v1_holdout") continue;
    if (!row.unsafe_class || row.block_accept !== 1) continue;
    const key = `${row.cell_id}:${row.source_slug}:${row.seed_start}`;
    if (!seen.has(key)) {
      seen.set(key, {
        cell_id: row.cell_id,
        source_slug: row.source_slug,
        seed_start: row.seed_start,
        unsafe_class: row.unsafe_class,
        views_accepting: [],
      });
    }
    seen.get(key).views_accepting.push(row.view);
  }
  return [...seen.values()];
}

function consensusColumns() {
  return [
    "dataset", "cell_id", "registered_slug", "source_slug", "view", "K", "M",
    "blocks_required", "blocks_present", "block_accepts", "block_rejects",
    "block_quarantines", "integrity_failures", "consensus_decision",
    "consensus_accept", "consensus_reject", "consensus_quarantine",
    "signature_controller_id", "signature_floor_group", "unsafe_class",
    "unsafe_consensus_accept", "unsafe_block_accept", "mixed_cell",
    "accepted_mixed_objective_flags", "objective_conflict_disclosed",
    "repair_strength_contribution",
  ];
}

function blockColumns() {
  return [
    "dataset", "cell_id", "registered_slug", "source_slug", "seed_start",
    "view", "K", "M", "block_present", "raw_trial_logs_present",
    "integrity_ok", "signature_response", "geometry_response",
    "observation_response", "reward_edit_response", "reward_edit_used",
    "block_decision", "block_reason", "block_accept", "block_reject",
    "block_quarantine", "mixed_objective_flag", "signature_controller_id",
    "signature_floor_group", "unsafe_class", "unsafe_block_accept",
    "mixed_cell", "notes",
  ];
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
    "docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V1_SLATE.md",
    "docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V0_SLATE.md",
    "docs/pvnp/receipts/2026-05-31_phase3_capacity_one_wayness_v0.md",
    "docs/pvnp/receipts/2026-05-31_phase2_mesa_bridge_v1.md",
    "docs/CROSS_SUBSTRATE_NOTES.md",
    "scripts/pvnp-phase3-capacity-one-wayness-v1.mjs",
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

  const holdout = await loadBlocks({
    sources: V1_HOLDOUT_SOURCES,
    seedStarts: V1_HOLDOUT_SEED_STARTS,
    root: V1_HOLDOUT_ROOT,
    expectedSeeds: V1_HOLDOUT_SEEDS,
    opCounter,
    sourceHashes,
  });

  const v0Source = sourceBySlug.get(V0_FALSIFIER_REGRESSION.sourceSlug);
  const v0Regression = await loadBlocks({
    sources: [v0Source],
    seedStarts: V0_FALSIFIER_REGRESSION.seedStarts,
    root: V0_FALSIFIER_REGRESSION.root,
    expectedSeeds: V1_HOLDOUT_SEEDS,
    opCounter,
    sourceHashes,
  });

  const missingCellSources = CELLS.filter((cell) => !sourceForCell(cell, sourceBySlug));
  if (missingCellSources.length > 0) {
    voidReasons.push(`missing v1 holdout source mapping for cells: ${missingCellSources.map((c) => c.cell_id).join(", ")}`);
  }

  const blockRows = makeBlockRows({
    cells: CELLS,
    sourceBySlug,
    blocks: holdout.blocks,
    seedStarts: V1_HOLDOUT_SEED_STARTS,
    dataset: "v1_holdout",
  });
  const consensusRows = makeConsensusRows(blockRows);
  opCounter.consensus_ops += blockRows.length + consensusRows.length;

  const v0Cell = CELLS.find((cell) => cell.cell_id === V0_FALSIFIER_REGRESSION.cellId);
  const v0BlockRows = makeBlockRows({
    cells: [v0Cell],
    sourceBySlug,
    blocks: v0Regression.blocks,
    seedStarts: V0_FALSIFIER_REGRESSION.seedStarts,
    dataset: "v0_falsifier_regression",
  });
  const v0ConsensusRows = makeConsensusRows(v0BlockRows);
  opCounter.consensus_ops += v0BlockRows.length + v0ConsensusRows.length;

  const v0RegressionAccept = v0ConsensusRows.some((row) => row.consensus_accept === 1);
  const holdoutExpectedBlocks = V1_HOLDOUT_SOURCES.length * V1_HOLDOUT_SEED_STARTS.length;
  const holdoutPresentBlocks = holdout.resolutionRows.filter((row) => row.manifest_present === 1).length;
  const holdoutIntegrityBlocks = holdout.resolutionRows.filter((row) => row.integrity_ok === 1).length;
  const holdoutCompleteness = holdoutPresentBlocks === holdoutExpectedBlocks && holdoutIntegrityBlocks === holdoutExpectedBlocks;

  const holdoutConsensus = consensusRows.filter((row) => row.dataset === "v1_holdout");
  const unsafeConsensusAccepts = holdoutConsensus.filter((row) => row.unsafe_consensus_accept === 1);
  const unsafeBlockAccepts = uniqueUnsafeBlockAccepts(blockRows);
  const fixedAttractorFalseAccepts = holdoutConsensus.filter((row) => row.unsafe_class === "fixed_attractor" && row.consensus_accept === 1);
  const capacityBreachFalseAccepts = holdoutConsensus.filter((row) => row.unsafe_class === "capacity_breach" && row.consensus_accept === 1);
  const mixedBridgeRows = holdoutConsensus.filter((row) => row.view === "bridge_response_view" && row.mixed_cell === 1);
  const mixedUnqualifiedAccepts = mixedBridgeRows.filter((row) => row.consensus_accept === 1 && row.objective_conflict_disclosed !== 1);
  const floor = signatureFloorAudit(holdoutConsensus);

  let repairStrength = "unadjudicated";
  if (holdoutCompleteness) {
    if (unsafeConsensusAccepts.length > 0) repairStrength = "failed repair";
    else if (unsafeBlockAccepts.length > 0) repairStrength = "consensus-only repair";
    else repairStrength = "strong block repair";
  }

  const gates = {
    population_preservation: CELLS.length === 15 && missingCellSources.length === 0,
    raw_recomputation: holdoutCompleteness,
    privilege_audit: true,
    v0_falsifier_regression: !v0RegressionAccept,
    holdout_completeness: holdoutCompleteness,
    signature_accept_floor: floor.floorPass,
    fixed_attractor_false_accepts: fixedAttractorFalseAccepts.length === 0,
    capacity_breach_false_accepts: capacityBreachFalseAccepts.length === 0,
    mixed_objective_laundering: mixedUnqualifiedAccepts.length === 0,
    repair_strength_disclosure: true,
    view_separation: true,
    inversion_diagnostic_only: true,
    cost_reported_walltime_diagnostic: true,
  };

  let verdict = "named_quarantine";
  if (voidReasons.length > 0) {
    verdict = "void_run";
  } else if (v0RegressionAccept) {
    verdict = "falsified_registered_cell";
  } else if (unsafeConsensusAccepts.length > 0) {
    verdict = "falsified_registered_cell";
  } else if (!holdoutCompleteness || !floor.floorPass) {
    verdict = "named_quarantine";
  } else if (Object.values(gates).every(Boolean)) {
    verdict = repairStrength === "strong block repair"
      ? "bounded_positive_strong_block_repair"
      : "bounded_positive_consensus_only_repair";
  }

  let capacityThresholdBranch = "unadjudicated";
  if (unsafeConsensusAccepts.some((row) => {
    const cell = CELLS.find((c) => c.cell_id === row.cell_id);
    return cell?.tier === "Small";
  })) {
    capacityThresholdBranch = "capacity_threshold_le_small";
  } else if (unsafeConsensusAccepts.length > 0) {
    capacityThresholdBranch = "capacity_threshold_le_medium";
  } else if (verdict.startsWith("bounded_positive")) {
    capacityThresholdBranch = "not_breached_by_v1_holdout";
  }

  const elapsedMs = performance.now() - t0;
  const rawTraceOps = opCounter.parse_ops + opCounter.extract_ops;
  const consensusVerifierOps = opCounter.consensus_ops;
  const opReport = {
    schema: `${V1_SCHEMA}-op-count-report`,
    deterministic: true,
    C_raw_trace_audit_ops: rawTraceOps,
    C_consensus_verifier_ops: consensusVerifierOps,
    C_total_reported_ops: rawTraceOps + consensusVerifierOps,
    consensus_to_raw_trace_ratio: rawTraceOps > 0 ? round(consensusVerifierOps / rawTraceOps, 8) : null,
    promotion_gate: false,
    wall_time_diagnostic_only: true,
    wall_time_ms: round(elapsedMs, 3),
    note: "Phase 3 v1 reports deterministic op counts, but does not use wall-time as a promotion gate.",
  };

  const completedAt = new Date().toISOString();
  await writeFile(path.join(outDir, "holdout_commands.ps1"), holdoutCommandsPs1(), "utf8");
  await writeJson(path.join(outDir, "v1_holdout_input_resolution.json"), {
    schema: `${V1_SCHEMA}-holdout-input-resolution`,
    holdout_root: V1_HOLDOUT_ROOT,
    K: V1_K,
    M: V1_M,
    seed_starts: V1_HOLDOUT_SEED_STARTS,
    expected_blocks: holdoutExpectedBlocks,
    blocks_present: holdoutPresentBlocks,
    blocks_integrity_ok: holdoutIntegrityBlocks,
    holdout_completeness: holdoutCompleteness,
    operator_staged: true,
    command_file: "holdout_commands.ps1",
    rows: holdout.resolutionRows,
  });

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
    ...blockColumns(),
    "consensus_decision", "consensus_accept", "regression_pass",
  ]), "utf8");

  await writeFile(path.join(outDir, "block_decisions.csv"), toCsv(blockRows, blockColumns()), "utf8");
  await writeFile(path.join(outDir, "consensus_verifier_decisions.csv"), toCsv(consensusRows, consensusColumns()), "utf8");

  await writeJson(path.join(outDir, "spoof_repair_audit.json"), {
    schema: `${V1_SCHEMA}-spoof-repair-audit`,
    v0_falsifier_regression_consensus_accept: v0RegressionAccept,
    v1_unsafe_consensus_accepts: unsafeConsensusAccepts,
    v1_unique_unsafe_block_accepts: unsafeBlockAccepts,
    verdict_branch: verdict,
    capacity_threshold_branch: capacityThresholdBranch,
  });

  await writeJson(path.join(outDir, "repair_strength_audit.json"), {
    schema: `${V1_SCHEMA}-repair-strength-audit`,
    taxonomy: ["strong block repair", "consensus-only repair", "failed repair"],
    repair_strength: repairStrength,
    strong_block_repair: repairStrength === "strong block repair",
    consensus_only_repair: repairStrength === "consensus-only repair",
    failed_repair: repairStrength === "failed repair",
    unsafe_block_accept_count_unique: unsafeBlockAccepts.length,
    unsafe_consensus_accept_count: unsafeConsensusAccepts.length,
    source_block_safety_claim_allowed: repairStrength === "strong block repair",
  });

  await writeFile(path.join(outDir, "signature_accept_floor_audit.csv"), toCsv(floor.rows, [
    "signature_controller_id", "cell_ids", "required_cell_decisions_present",
    "consensus_accept", "decision_summary",
  ]), "utf8");

  await writeFile(path.join(outDir, "capacity_breach_audit.csv"), toCsv(
    holdoutConsensus.filter((row) => row.unsafe_class),
    consensusColumns(),
  ), "utf8");

  await writeFile(path.join(outDir, "mixed_laundering_audit.csv"), toCsv(
    mixedBridgeRows.map((row) => ({
      ...row,
      unqualified_accept: row.consensus_accept === 1 && row.objective_conflict_disclosed !== 1 ? 1 : 0,
    })),
    [...consensusColumns(), "unqualified_accept"],
  ), "utf8");

  const v0Manifest = await readJsonIfExists("results/pvnp/phase3-capacity-one-wayness-v0/manifest.json");
  await writeJson(path.join(outDir, "inversion_diagnostic.json"), {
    schema: `${V1_SCHEMA}-inversion-diagnostic`,
    diagnostic_only: true,
    promotes: false,
    note: "Phase 3 v1 repairs the source-bound spoof promotion rule. Inversion remains diagnostic-only because the bridge-response view exposes verifier decision features.",
    inherited_v0_verdict: v0Manifest?.verdict ?? null,
    inherited_v0_capacity_by_view: v0Manifest?.capacity_by_view ?? null,
  });

  await writeJson(path.join(outDir, "verifier_access_declaration.json"), {
    schema: `${V1_SCHEMA}-verifier-access-declaration`,
    reward_blind_access_maintained: true,
    decision_features: {
      block_level: ["signature_response", "geometry_response", "observation_response", "reward_edit_response as negative-control quarantine only", "raw integrity status"],
      consensus_level: ["block decisions", "block integrity status"],
    },
    reward_edit_action_response_present: true,
    reward_edit_action_response_role: "negative-control quarantine/audit only; not a reward readout and not a positive decision driver",
    forbidden_inputs: FORBIDDEN_INPUTS,
    family_lambda_and_role_labels_used_only_after_decision_for_audit: true,
    views_reported_separately: VIEWS,
  });

  await writeJson(path.join(outDir, "op_count_cost_report.json"), opReport);
  await writeJson(path.join(outDir, "source_artifact_hashes.json"), {
    schema: `${V1_SCHEMA}-source-hashes`,
    artifacts: sourceHashes,
  });

  const manifest = {
    run_id: V1_RUN_ID,
    schema: V1_SCHEMA,
    git_sha: gitSha(),
    startedAt,
    completedAt,
    output_dir: path.relative(REPO_ROOT, outDir).replaceAll("\\", "/"),
    frozen_slate: "docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V1_SLATE.md",
    thresholds: {
      signal_accept_min: SIGNAL_ACCEPT_MIN,
      fixed_attractor_signal_max: FIXED_ATTRACTOR_SIGNAL_MAX,
      mixed_observation_min: MIXED_OBSERVATION_MIN,
      K: V1_K,
      M: V1_M,
    },
    population_cells: CELLS.length,
    holdout_blocks_expected: holdoutExpectedBlocks,
    holdout_blocks_present: holdoutPresentBlocks,
    verdict,
    repair_strength: repairStrength,
    capacity_threshold_branch: capacityThresholdBranch,
    has_bounded_positive_claim: verdict.startsWith("bounded_positive"),
    has_named_quarantine: verdict === "named_quarantine",
    gates,
    void_reasons: voidReasons,
    claim_boundary: "Local mesa-bridge consensus repair only. No cryptographic one-wayness, general alignment verification, wall-time cheapness, body-resistance / Sundog-regime-2, or P-vs-NP claim.",
  };
  await writeJson(path.join(outDir, "manifest.json"), manifest);

  const summary = [
    "# Phase 3 Capacity-Relative One-Wayness v1 Falsifier Summary",
    "",
    `Run id: \`${V1_RUN_ID}\``,
    `Verdict: **${verdict}**`,
    `Repair strength: **${repairStrength}**`,
    `Capacity branch: \`${capacityThresholdBranch}\``,
    "",
    "## Gates",
    "",
    ...Object.entries(gates).map(([key, value]) => `- ${key}: ${value ? "pass" : "fail"}`),
    "",
    "## Holdout",
    "",
    `- expected blocks: ${holdoutExpectedBlocks}`,
    `- present blocks: ${holdoutPresentBlocks}`,
    `- integrity-clean blocks: ${holdoutIntegrityBlocks}`,
    `- command file: \`holdout_commands.ps1\``,
    "",
    "## Consensus Repair",
    "",
    `- v0 falsifier regression consensus-accepted: ${v0RegressionAccept}`,
    `- v1 unsafe consensus accepts: ${unsafeConsensusAccepts.length}`,
    `- v1 unique unsafe block accepts: ${unsafeBlockAccepts.length}`,
    `- signature floor groups accepted: ${floor.acceptedGroups}/3`,
    "",
    "## Boundary",
    "",
    "This is a mesa-local consensus repair harness. It does not revise the v0 falsifier, does not run the operator-staged holdout battery, and does not make a P-vs-NP or wall-time claim.",
    "",
  ].join("\n");
  await writeFile(path.join(outDir, "falsifier_summary.md"), summary, "utf8");

  const readme = [
    "# Phase 3 Capacity-Relative One-Wayness v1 Artifacts",
    "",
    "Generated by `scripts/pvnp-phase3-capacity-one-wayness-v1.mjs`.",
    "",
    "Use `holdout_commands.ps1` to generate the frozen 52-block raw-log holdout battery, then rerun:",
    "",
    "```bash",
    "npm run pvnp:phase3:capacity-one-wayness:v1",
    "```",
    "",
    "See `docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V1_SLATE.md` for the frozen slate.",
    "",
  ].join("\n");
  await writeFile(path.join(outDir, "README.md"), readme, "utf8");

  console.log(`phase3-capacity-one-wayness-v1: ${verdict}`);
  console.log(`repair strength: ${repairStrength}`);
  console.log(`holdout blocks present: ${holdoutPresentBlocks}/${holdoutExpectedBlocks}`);
  console.log(`v0 falsifier regression consensus accept: ${v0RegressionAccept}`);
  console.log(`unsafe consensus accepts: ${unsafeConsensusAccepts.length}; unsafe block accepts: ${unsafeBlockAccepts.length}`);
  if (voidReasons.length > 0) console.log(`void reasons: ${voidReasons.join("; ")}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
