#!/usr/bin/env node
// scripts/pvnp-phase1-verifier.mjs
//
// Run the signature verifier V on every (policy, env) signature and write
// verifier_decisions.csv for the measurement splits.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { verify, V0_PROMISE, V0_CHECKER_THRESHOLDS } from "./lib/pvnp-phase1-verifier-core.mjs";
import { getPhase1RunConfig } from "./lib/pvnp-phase1-run-config.mjs";
import { loadCacheState, saveCacheState, statsReport } from "./lib/pvnp-phase1-cache.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const args = { runDir: "results/pvnp/phase1-toy-verifier-v0" };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--run-dir") { args.runDir = argv[i + 1]; i += 1; }
    else throw new Error(`Unknown flag: ${argv[i]}`);
  }
  return args;
}

async function readJsonl(p) {
  const text = await readFile(p, "utf8");
  return text.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
}

async function readJsonlIfExists(p) {
  try { return await readJsonl(p); }
  catch (err) {
    if (err.code === "ENOENT") return [];
    throw err;
  }
}

function redactEnv(env) {
  const { hidden_state: _hidden, ...rest } = env;
  return rest;
}

function csvRow(values) {
  return values.map((v) => {
    if (v === null || v === undefined) return "";
    const s = String(v);
    return s.includes(",") || s.includes("\"") ? `"${s.replace(/"/g, '""')}"` : s;
  }).join(",");
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(REPO_ROOT, args.runDir);
  const slate = getPhase1RunConfig(args.runDir);
  const version = slate.schema_suffix;
  const sourceBound = version === "v1" || version === "v2" || version === "v3" || version === "v4";
  const isV2 = version === "v2";
  const isV3 = version === "v3";
  const isV4 = version === "v4";
  const sensorDemoted = isV3 || isV4;
  const usesCache = isV3 || isV4;
  const writesGeometryAudits = isV2 || isV3 || isV4;
  await mkdir(outDir, { recursive: true });

  // v3 and v4 share a source-hash cache between verifier, ablation, and
  // spoof stages. Cold-loaded here (likely empty), warm-written at exit.
  const cachePath = path.join(outDir, "derived_fields_cache.json");
  const cacheState = usesCache ? await loadCacheState(cachePath) : null;

  const sigs = await readJsonl(path.join(outDir, "signatures.jsonl"));
  const commitments = await readJsonlIfExists(path.join(outDir, "trace_commitments.jsonl"));
  const envs = await readJsonl(path.join(outDir, "environments.jsonl"));
  const calibrationManifest = JSON.parse(
    await readFile(path.join(outDir, "calibration_manifest.json"), "utf8"),
  );
  const m_min = calibrationManifest.selected_m_min;

  const envById = new Map(envs.map((e) => [e.id, redactEnv(e)]));
  const commitmentByTrace = new Map();
  const duplicateTraceIds = new Set();
  for (const c of commitments) {
    if (commitmentByTrace.has(c.trace_id)) duplicateTraceIds.add(c.trace_id);
    else commitmentByTrace.set(c.trace_id, c);
  }

  const rows = [[
    "env_id", "policy_id", "split", "decision", "reason",
    "margin_lower_bound", "coverage_touched", "invariance_pass", "noise_std_estimate",
    "integrity_pass", "geometry_pass",
    "verify_wall_ms", "verify_ops",
  ].join(",")];

  const integrityRows = sourceBound ? [[
    "env_id", "policy_id", "split", "check", "decision", "reason",
  ].join(",")] : null;
  const integrityFailureRows = sourceBound ? [[
    "env_id", "policy_id", "split", "check", "registered_behavior", "observed_decision", "observed_reason",
  ].join(",")] : null;
  const geometryBoundaryRows = writesGeometryAudits ? [[
    "env_id", "policy_id", "split", "decision", "reason",
    "geometry_pass", "geometry_reason", "geometry_evidence_coverage",
    "curvature_abs_p95", "center_value_range", "near_boundary_count",
    "scale_interval_lower", "scale_interval_upper", "topology_ambiguity_score",
    "boundary_flags",
  ].join(",")] : null;
  const acceptedOopRows = writesGeometryAudits ? [[
    "env_id", "policy_id", "split", "promise_compliance", "decision", "reason",
    "violation_subtype", "geometry_reason",
  ].join(",")] : null;
  // v3/v4 sensor disposition audit: shadow each measurement decision with
  // the v2-style sensor gate forced ON, recording any decision flips. The
  // v3/v4 primary verifier does NOT gate on sensor_health (per slate);
  // this audit proves no unsafe accept would have been blocked by the old
  // gate.
  const sensorDispositionRows = sensorDemoted ? [[
    "env_id", "policy_id", "split", "v3_decision", "v3_reason",
    "shadow_with_sensor_decision", "shadow_with_sensor_reason",
    "decision_changed_under_old_gate",
  ].join(",")] : null;

  let nAccept = 0, nReject = 0, nQuarantine = 0;
  const verifyCosts = { wall_ms: 0, ops: 0, calls: 0 };

  for (const sigma of sigs) {
    const policyId = sigma.source_observations.policy_id;
    const envId = sigma.source_observations.env_id;
    const publicEnv = envById.get(envId);
    if (publicEnv.split === "calibration") continue; // measurement only

    const expectedTraceId = `${policyId}|${envId}`;
    const traceCommitment = commitmentByTrace.get(expectedTraceId);
    const t0 = performance.now();
    const result = verify({
      sigma,
      expectedTraceId,
      publicEnv,
      m_min,
      promise: V0_PROMISE,
      thresholds: V0_CHECKER_THRESHOLDS,
      traceCommitment,
      commitmentDuplicate: duplicateTraceIds.has(expectedTraceId),
      cacheState,
      stageLabel: "verifier",
    });
    const elapsed = performance.now() - t0;

    if (sensorDemoted) {
      // Shadow verify with the old v2 sensor gate forced ON.
      const shadow = verify({
        sigma,
        expectedTraceId,
        publicEnv,
        m_min,
        promise: V0_PROMISE,
        thresholds: V0_CHECKER_THRESHOLDS,
        traceCommitment,
        commitmentDuplicate: duplicateTraceIds.has(expectedTraceId),
        cacheState,
        stageLabel: "sensor_audit",
        forceSensorGate: true,
      });
      const changed = result.decision !== shadow.decision;
      sensorDispositionRows.push(csvRow([
        envId, policyId, publicEnv.split,
        result.decision, result.reason,
        shadow.decision, shadow.reason,
        changed ? 1 : 0,
      ]));
    }
    // Verifier ops: ~10 constant-time threshold checks.
    const ops = 10;
    verifyCosts.wall_ms += elapsed;
    verifyCosts.ops += ops;
    verifyCosts.calls += 1;
    if (result.decision === "accept") nAccept += 1;
    else if (result.decision === "reject") nReject += 1;
    else nQuarantine += 1;

    rows.push(csvRow([
      envId, policyId, publicEnv.split, result.decision, result.reason,
      sigma.margin_lower_bound.toFixed(6),
      sigma.coverage_digest?.touched_cells ?? Math.round((sigma.geometry_promise_signal_v2?.geometry_evidence_coverage ?? 0) * 256),
      (sigma.invariance_checks_v2 ?? sigma.invariance_checks_v1 ?? sigma.invariance_checks).all_pass ? 1 : 0,
      (sigma.sensor_diagnostics_v3 ?? sigma.sensor_health_v1 ?? sigma.sensor_health).noise_std_estimate.toFixed(6),
      sigma.integrity_checks?.all_pass === false ? 0 : 1,
      (sigma.geometry_promise_signal_v2 ?? sigma.geometry_promise_signal)
        ? ((sigma.geometry_promise_signal_v2 ?? sigma.geometry_promise_signal).all_pass ? 1 : 0)
        : "",
      elapsed.toFixed(3), ops,
    ]));

    if (sourceBound) {
      integrityRows.push(csvRow([
        envId, policyId, publicEnv.split,
        "normal_certificate",
        result.reason && result.reason.includes("integrity") ? "quarantine" : "checked",
        result.reason,
      ]));
    }

    if (writesGeometryAudits) {
      const g = sigma.geometry_promise_signal_v2;
      const failedFlag = g?.boundary_flags
        ? Object.entries(g.boundary_flags).find(([, flag]) => flag)?.[0] ?? ""
        : "";
      geometryBoundaryRows.push(csvRow([
        envId, policyId, publicEnv.split, result.decision, result.reason,
        g?.all_pass ? 1 : 0,
        failedFlag,
        g?.geometry_evidence_coverage?.toFixed(6) ?? "",
        g?.curvature_abs_p95?.toFixed(6) ?? "",
        g?.center_value_range?.toFixed(6) ?? "",
        g?.near_boundary_count ?? "",
        g?.scale_interval?.lower?.toFixed(6) ?? "",
        g?.scale_interval?.upper?.toFixed(6) ?? "",
        g?.topology_ambiguity_score?.toFixed(6) ?? "",
        g?.boundary_flags ? JSON.stringify(g.boundary_flags) : "",
      ]));
      if (publicEnv.promise_compliance !== "in_promise" && result.decision === "accept") {
        const noise = publicEnv.probe_noise_params;
        const subtype = noise.std > V0_PROMISE.probe_noise_max_std
          || noise.dropout_rate > V0_PROMISE.probe_dropout_max_rate
          || noise.delay_steps > V0_PROMISE.probe_delay_max_steps
          ? "probe_noise"
          : "basin_shape";
        acceptedOopRows.push(csvRow([
          envId, policyId, publicEnv.split, publicEnv.promise_compliance,
          result.decision, result.reason, subtype, failedFlag,
        ]));
      }
    }
  }

  if (sourceBound && sigs.length > 0) {
    const target = sigs.find((s) => envById.get(s.source_observations.env_id)?.split !== "calibration") ?? sigs[0];
    const policyId = target.source_observations.policy_id;
    const envId = target.source_observations.env_id;
    const publicEnv = envById.get(envId);
    const expectedTraceId = `${policyId}|${envId}`;
    const traceCommitment = commitmentByTrace.get(expectedTraceId);
    const probes = [
      ["missing_trace_commitment", { sigma: target, traceCommitment: null }],
      ["source_hash_mismatch", { sigma: { ...target, source_hash: `bad-${target.source_hash}` }, traceCommitment }],
      ["derived_field_hash_mismatch", { sigma: { ...target, margin_lower_bound: target.margin_lower_bound + 0.5 }, traceCommitment }],
      ["stale_transform_version", { sigma: { ...target, transform_version: "stale-transform" }, traceCommitment }],
      ["duplicate_trace_id", { sigma: target, traceCommitment, commitmentDuplicate: true }],
    ];
    for (const [check, input] of probes) {
      const result = verify({
        sigma: input.sigma,
        expectedTraceId,
        publicEnv,
        m_min,
        promise: V0_PROMISE,
        thresholds: V0_CHECKER_THRESHOLDS,
        traceCommitment: input.traceCommitment,
        commitmentDuplicate: input.commitmentDuplicate ?? false,
        cacheState,
        stageLabel: "integrity_probe",
      });
      integrityFailureRows.push(csvRow([
        envId, policyId, publicEnv.split, check, "quarantine", result.decision, result.reason,
      ]));
    }
  }

  await writeFile(
    path.join(outDir, "verifier_decisions.csv"),
    rows.join("\n") + "\n",
    "utf8",
  );

  if (sourceBound) {
    await writeFile(path.join(outDir, "integrity_decisions.csv"), integrityRows.join("\n") + "\n", "utf8");
    await writeFile(path.join(outDir, "integrity_failures.csv"), integrityFailureRows.join("\n") + "\n", "utf8");
  }
  if (writesGeometryAudits) {
    await writeFile(path.join(outDir, "geometry_boundary_audit.csv"), geometryBoundaryRows.join("\n") + "\n", "utf8");
    await writeFile(path.join(outDir, "accepted_oop_audit.csv"), acceptedOopRows.join("\n") + "\n", "utf8");
  }
  if (sensorDemoted) {
    await writeFile(path.join(outDir, "sensor_disposition_audit.csv"), sensorDispositionRows.join("\n") + "\n", "utf8");
    await saveCacheState(cachePath, cacheState, "verifier");
    const stats = statsReport(cacheState);
    await writeFile(
      path.join(outDir, "verifier_cache_stats.partial.json"),
      JSON.stringify(stats, null, 2) + "\n",
      "utf8",
    );
  }

  // Roll verifier costs into partial costs file (additive).
  const partialPath = path.join(outDir, "costs.partial.json");
  const existing = JSON.parse(await readFile(partialPath, "utf8"));
  existing.verifier = verifyCosts;
  existing.signature = (() => {
    // sum costs over all sigmas
    let ms = 0, ops = 0;
    for (const s of sigs) { ms += s.cost_signature.wall_ms; ops += s.cost_signature.ops; }
    return { wall_ms: ms, ops, calls: sigs.length };
  })();
  await writeFile(partialPath, JSON.stringify(existing, null, 2) + "\n", "utf8");

  console.log(`m_min=${m_min}; verifier decisions: accept=${nAccept} reject=${nReject} quarantine=${nQuarantine}`);
}

main().catch((err) => { console.error(err); process.exit(1); });
