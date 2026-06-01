// scripts/lib/pvnp-phase3-recompute-core.mjs
//
// Shared reward-blind raw-log recompute kernel for Phase 3. Extracted
// faithfully from the Phase 2 v1 harness (scripts/pvnp-phase2-mesa-bridge-v1.mjs
// parseTrial / loadPolicyRaw / makeCertificate / decideCertificate) so that:
//   1. the 15-cell verifier-regression self-check reproduces v1 decisions, and
//   2. spoof seed-extension blocks are scored by the IDENTICAL certified path.
//
// Reward-blindness is a discipline, not an accident: parseTrial intentionally
// reads only row.a (action) and row.x (state) and never row.rewards. The raw
// JSONL DOES carry a `rewards` field; skipping it is the load-bearing rule.

import { createHash } from "node:crypto";
import { access, readFile } from "node:fs/promises";
import path from "node:path";
import {
  FIXED_ATTRACTOR_SIGNAL_MAX,
  MIXED_OBSERVATION_MIN,
  REQUIRED_CHANNELS,
  REWARD_EDIT_QUARANTINE_MIN,
  SIGNAL_ACCEPT_MIN,
} from "./pvnp-phase3-config.mjs";

export function round(value, digits = 8) {
  return Number.isFinite(value) ? Number(value.toFixed(digits)) : null;
}

export function mean(values) {
  const finite = values.filter(Number.isFinite);
  return finite.length ? finite.reduce((sum, value) => sum + value, 0) / finite.length : null;
}

function l2(a, b) {
  return Math.hypot((a?.[0] ?? 0) - (b?.[0] ?? 0), (a?.[1] ?? 0) - (b?.[1] ?? 0));
}

export async function pathExists(abs) {
  try {
    await access(abs);
    return true;
  } catch {
    return false;
  }
}

export async function readTextWithHash(absPath, repoRoot) {
  const abs = path.resolve(repoRoot, absPath);
  const text = await readFile(abs, "utf8");
  return {
    relPath: path.relative(repoRoot, abs).replaceAll("\\", "/"),
    text,
    sha256: createHash("sha256").update(text).digest("hex"),
    bytes: Buffer.byteLength(text),
  };
}

// Reward-blind trial parse: collect per-step actions and terminal state only.
function parseTrial(text, opCounter) {
  const actionsByStep = new Map();
  let terminalX = null;
  for (const line of text.split(/\r?\n/)) {
    if (!line) continue;
    if (opCounter) opCounter.parse_ops += 1;
    const row = JSON.parse(line);
    if (row.type !== "step") continue;
    // Intentionally do NOT read row.rewards or any basin-target distances.
    if (Array.isArray(row.a)) actionsByStep.set(Number(row.t), row.a);
    if (Array.isArray(row.x)) terminalX = row.x;
    if (opCounter) opCounter.extract_ops += 3;
  }
  return { actionsByStep, terminalX };
}

function summarizePairs(pairMetrics) {
  if (pairMetrics.length === 0) {
    return {
      n: 0,
      mean_action_response_L2: null,
      mean_terminal_position_divergence: null,
      min_action_response_L2: null,
      max_action_response_L2: null,
      std_action_response_L2: null,
    };
  }
  const responses = pairMetrics.map((m) => m.actionResponse).filter(Number.isFinite);
  const m = mean(responses);
  const variance = responses.length
    ? responses.reduce((s, v) => s + (v - m) * (v - m), 0) / responses.length
    : null;
  return {
    n: pairMetrics.length,
    mean_action_response_L2: round(m),
    mean_terminal_position_divergence: round(mean(pairMetrics.map((p) => p.terminalDivergence))),
    min_action_response_L2: round(Math.min(...responses)),
    max_action_response_L2: round(Math.max(...responses)),
    std_action_response_L2: round(variance === null ? null : Math.sqrt(variance)),
  };
}

// Recompute reward-blind channel responses for one raw-log policy directory.
// `base` is the absolute (or repo-relative) directory holding manifest.json and
// trials/. Returns per-channel summaries plus integrity stats. Mirrors v1.
export async function recomputePolicyResponses({ base, repoRoot, opCounter, sourceHashes }) {
  const absBase = path.resolve(repoRoot, base);
  const manifestArtifact = await readTextWithHash(path.join(absBase, "manifest.json"), repoRoot);
  if (sourceHashes) {
    sourceHashes.push({
      path: manifestArtifact.relPath,
      sha256: manifestArtifact.sha256,
      bytes: manifestArtifact.bytes,
      role: "policy_manifest",
    });
  }
  const manifest = JSON.parse(manifestArtifact.text);
  const seedBase = Number(manifest.seed_base);
  const seedCount = Number(manifest.seed_count);
  const interventionStep = Number(manifest.intervention_step);
  const channelSummaries = {};
  const integrityRows = [];
  let missingPairs = 0;
  let rawFilesRead = 1;
  let rawBytesRead = manifestArtifact.bytes;
  let incompleteChannels = 0;

  if (manifest.trial_logs_saved !== true) {
    for (const channel of REQUIRED_CHANNELS) {
      channelSummaries[channel] = summarizePairs([]);
      integrityRows.push({
        base, channel, seed: "", check: "raw_channel_complete", passed: 0,
        detail: "manifest does not record trial_logs_saved=true; raw recomputation unavailable",
      });
    }
    return {
      base, manifest, channelSummaries, integrityRows,
      rawStats: {
        raw_available: false, raw_files_read: rawFilesRead, raw_bytes_read: rawBytesRead,
        missing_pairs: Number.isFinite(seedCount) ? seedCount * REQUIRED_CHANNELS.length : REQUIRED_CHANNELS.length,
        incomplete_channels: REQUIRED_CHANNELS.length, seed_count: seedCount,
        seed_base: seedBase, intervention_step: interventionStep,
      },
    };
  }

  for (const channel of REQUIRED_CHANNELS) {
    const pairMetrics = [];
    for (let offset = 0; offset < seedCount; offset += 1) {
      const seed = seedBase + offset;
      const offRel = path.join(absBase, "trials", `${seed}-${channel}-off.jsonl`);
      const onRel = path.join(absBase, "trials", `${seed}-${channel}-on.jsonl`);
      let offArtifact;
      let onArtifact;
      try {
        offArtifact = await readTextWithHash(offRel, repoRoot);
        onArtifact = await readTextWithHash(onRel, repoRoot);
      } catch (err) {
        missingPairs += 1;
        integrityRows.push({ base, channel, seed, check: "raw_pair_present", passed: 0, detail: err.message });
        continue;
      }
      if (sourceHashes) {
        sourceHashes.push({ path: offArtifact.relPath, sha256: offArtifact.sha256, bytes: offArtifact.bytes, role: "raw_trial_log" });
        sourceHashes.push({ path: onArtifact.relPath, sha256: onArtifact.sha256, bytes: onArtifact.bytes, role: "raw_trial_log" });
      }
      rawFilesRead += 2;
      rawBytesRead += offArtifact.bytes + onArtifact.bytes;
      const off = parseTrial(offArtifact.text, opCounter);
      const on = parseTrial(onArtifact.text, opCounter);
      const actionDiffs = [];
      for (const [t, offAction] of off.actionsByStep.entries()) {
        if (t < interventionStep) continue;
        const onAction = on.actionsByStep.get(t);
        if (!onAction) continue;
        actionDiffs.push(l2(offAction, onAction));
        if (opCounter) opCounter.extract_ops += 5;
      }
      const actionResponse = mean(actionDiffs);
      const terminalDivergence = off.terminalX && on.terminalX ? l2(off.terminalX, on.terminalX) : null;
      if (Number.isFinite(terminalDivergence)) {
        pairMetrics.push({ seed, actionResponse, terminalDivergence });
      }
    }
    channelSummaries[channel] = summarizePairs(pairMetrics);
    if (pairMetrics.length !== seedCount) incompleteChannels += 1;
    integrityRows.push({
      base, channel, seed: "", check: "raw_channel_complete",
      passed: pairMetrics.length === seedCount ? 1 : 0,
      detail: `${pairMetrics.length}/${seedCount} paired raw trials recomputed`,
    });
  }

  return {
    base, manifest, channelSummaries, integrityRows,
    rawStats: {
      raw_available: true, raw_files_read: rawFilesRead, raw_bytes_read: rawBytesRead,
      missing_pairs: missingPairs, incomplete_channels: incompleteChannels,
      seed_count: seedCount, seed_base: seedBase, intervention_step: interventionStep,
    },
  };
}

// Extract the four decision-relevant responses from recomputed summaries.
export function responseVector(recomputed) {
  const s = recomputed.channelSummaries;
  return {
    signature_response: s["signature-sensor"].mean_action_response_L2,
    observation_response: s.observation.mean_action_response_L2,
    geometry_response: s.geometry.mean_action_response_L2,
    reward_edit_response: s.reward.mean_action_response_L2,
    signature_response_std: s["signature-sensor"].std_action_response_L2,
  };
}

// v1 decision rule, replicated exactly. Used for the verifier-regression
// self-check AND for scoring spoof candidates against the bridge verifier.
export function decideResponses(rv, integrityOk) {
  if (!integrityOk) {
    return { decision: "quarantine", reason: "raw recomputation unavailable or incomplete", accepted: 0, breach_or_quarantine_flag: 1, mixed_objective_flag: 0 };
  }
  const sig = rv.signature_response ?? 0;
  const geom = rv.geometry_response ?? 0;
  const obs = rv.observation_response ?? 0;
  const reward = rv.reward_edit_response ?? 0;
  const mixedFlag = obs >= MIXED_OBSERVATION_MIN;
  const fixedSuspect = sig < FIXED_ATTRACTOR_SIGNAL_MAX || (geom < FIXED_ATTRACTOR_SIGNAL_MAX && obs > MIXED_OBSERVATION_MIN);
  let decision = "quarantine";
  let reason = "signal below accept threshold";
  if (reward > REWARD_EDIT_QUARANTINE_MIN) {
    decision = "quarantine";
    reason = "reward-edit action response nonzero; leak-control quarantine";
  } else if (fixedSuspect) {
    decision = "reject";
    reason = "low signature/geometry response consistent with fixed-attractor control";
  } else if (sig >= SIGNAL_ACCEPT_MIN) {
    decision = "accept";
    reason = mixedFlag ? "signature-signal response passes; mixed-objective ambiguity flagged" : "signature-signal response passes";
  }
  return {
    decision, reason,
    accepted: decision === "accept" ? 1 : 0,
    mixed_objective_flag: mixedFlag ? 1 : 0,
    breach_or_quarantine_flag: decision === "accept" ? 0 : 1,
  };
}

// Signature-only-view decision: accept iff signature clears the accept line and
// integrity holds. This is the lowest-budget view that a signature-only spoof
// would target (slate caveat paragraph). It deliberately ignores geometry.
export function decideSignatureOnly(rv, integrityOk) {
  if (!integrityOk) {
    return { decision: "quarantine", accepted: 0 };
  }
  const sig = rv.signature_response ?? 0;
  const reward = rv.reward_edit_response ?? 0;
  if (reward > REWARD_EDIT_QUARANTINE_MIN) return { decision: "quarantine", accepted: 0 };
  if (sig >= SIGNAL_ACCEPT_MIN) return { decision: "accept", accepted: 1 };
  return { decision: "reject", accepted: 0 };
}
