#!/usr/bin/env node
// scripts/pvnp-phase1-cost-median.mjs
//
// v5 cost-closure measurement (PHASE1_V5_SLATE.md §Cost Measurement Protocol
// + §Required Optimization). Emits the two slate-required artifacts:
//   - cost_multirun_report.json     (median-of-3 cost statistic)
//   - short_circuit_instrumentation_audit.json  (hot-path closure check)
//
// Re-measures C_signature + C_verify + C_full_state over the FROZEN inputs of
// an already-completed v5 run, three times, and reports min/max/mean/median +
// percent spread of C_total_signature plus the derived ratios. A single noisy
// sample gated v4; v5 promotes on the median of three.
//
// Each pass uses a FRESH derived-fields cache so C_verify reflects the same
// cold-start population (unique-source misses) the real verifier stage pays —
// the median is honest, not warm-cache-inflated.
//
// This is a measurement harness, NOT a verifier; it may read privileged hidden
// state for the full-state baseline denominator (same posture as baselines.mjs).
// It is excluded from the privilege-audit target set.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { getPhase1RunConfig } from "./lib/pvnp-phase1-run-config.mjs";
import { computeSignature, buildSourcePayload } from "./lib/pvnp-phase1-signature-core.mjs";
import { verify, V0_PROMISE, V0_CHECKER_THRESHOLDS } from "./lib/pvnp-phase1-verifier-core.mjs";
import { fullStateCheck } from "./lib/pvnp-phase1-evaluator-core.mjs";
import { makeCacheState } from "./lib/pvnp-phase1-cache.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const VERIFIER_CORE = path.join(REPO_ROOT, "scripts", "lib", "pvnp-phase1-verifier-core.mjs");

const PASSES = 3;
const TARGET_C_TOTAL_MS = 1010;
const TARGET_C_TOTAL_MAX_MS = 1250;
const TARGET_FULL_STATE_RATIO = 105;
const TARGET_OP_RATIO = 1.0;
const TARGET_SPREAD_PCT = 25;
const CACHE_REUSE_FLOOR = 0.95;

function parseArgs(argv) {
  const args = { runDir: "results/pvnp/phase1-toy-verifier-v5" };
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

async function readJsonIfExists(p) {
  try { return JSON.parse(await readFile(p, "utf8")); }
  catch (err) { if (err.code === "ENOENT") return null; throw err; }
}

function redactEnv(env) {
  const { hidden_state: _hidden, ...rest } = env;
  return rest;
}

function median(values) {
  const s = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}
function mean(values) { return values.reduce((a, b) => a + b, 0) / values.length; }

// ── short-circuit instrumentation audit (PHASE1_V5_SLATE.md §Required Optimization) ──
// Static check: the hot certificate-integrity path must NOT allocate a
// per-verify closure for short-circuit counting. We require:
//   - the legacy `noteShortCircuit` closure identifier is absent;
//   - short-circuit counting goes through the module-level
//     recordPreIntegrityShortCircuit() called directly.
async function shortCircuitInstrumentationAudit() {
  const src = await readFile(VERIFIER_CORE, "utf8");
  const hasLegacyClosure = /const\s+noteShortCircuit\s*=/.test(src)
    || /noteShortCircuit\s*\(\s*\)/.test(src);
  const hasDirectCounter = src.includes("recordPreIntegrityShortCircuit(cacheState, stageLabel)");
  const directCallCount = (src.match(/recordPreIntegrityShortCircuit\(cacheState, stageLabel\)/g) || []).length;
  const hoistedGuard = src.includes("const hasCacheState = cacheState !== null");
  const passed = !hasLegacyClosure && hasDirectCounter && hoistedGuard;
  return {
    schema: "pvnp-phase1-short-circuit-instrumentation-audit-v5",
    audited_file: "scripts/lib/pvnp-phase1-verifier-core.mjs",
    legacy_closure_present: hasLegacyClosure,
    direct_counter_present: hasDirectCounter,
    direct_counter_call_sites: directCallCount,
    hoisted_cachestate_guard: hoistedGuard,
    no_per_call_closure_allocation: passed,
    passed,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(REPO_ROOT, args.runDir);
  const slate = getPhase1RunConfig(args.runDir);
  const version = slate.schema_suffix;
  if (version !== "v5") {
    console.log(`cost-median is a v5-only artifact; skipping for ${version}`);
    return;
  }
  await mkdir(outDir, { recursive: true });

  // ── Short-circuit instrumentation audit (required for cost repair) ──
  const scAudit = await shortCircuitInstrumentationAudit();
  await writeFile(
    path.join(outDir, "short_circuit_instrumentation_audit.json"),
    JSON.stringify(scAudit, null, 2) + "\n",
    "utf8",
  );

  const traces = await readJsonl(path.join(outDir, "traces.jsonl"));
  const envsRaw = await readJsonl(path.join(outDir, "environments.jsonl"));
  const commitments = await readJsonl(path.join(outDir, "trace_commitments.jsonl"));
  const calibrationManifest = JSON.parse(
    await readFile(path.join(outDir, "calibration_manifest.json"), "utf8"),
  );
  const m_min = calibrationManifest.selected_m_min;

  const envByIdPublic = new Map(envsRaw.map((e) => [e.id, redactEnv(e)]));
  const envByIdFull = new Map(envsRaw.map((e) => [e.id, e]));
  const commitmentByTrace = new Map(commitments.map((c) => [c.trace_id, c]));

  const measurementTraces = traces.filter((tr) => {
    const env = envByIdPublic.get(tr.env_id);
    return env && env.split !== "calibration";
  });

  const samples = [];

  for (let pass = 0; pass < PASSES; pass += 1) {
    // ── C_signature: recompute all signatures from traces (cold) ──
    let cSignatureMs = 0;
    let cSignatureOps = 0;
    const freshSigs = new Map();
    for (const tr of traces) {
      const traceId = `${tr.policy_id}|${tr.env_id}`;
      const publicEnv = envByIdPublic.get(tr.env_id);
      const sourcePayload = buildSourcePayload({
        traceId, publicEnv, positions: tr.positions, probes: tr.probes,
      });
      const t0 = performance.now();
      const sig = computeSignature({
        traceId, publicEnv, positions: tr.positions, probes: tr.probes,
        sourcePayload, version,
      });
      cSignatureMs += performance.now() - t0;
      cSignatureOps += sig.cost_signature.ops;
      freshSigs.set(traceId, sig);
    }

    // ── C_verify: verify all measurement pairs, fresh cache (cold) ──
    const cacheState = makeCacheState();
    let cVerifyMs = 0;
    let cVerifyOps = 0;
    for (const tr of measurementTraces) {
      const traceId = `${tr.policy_id}|${tr.env_id}`;
      const sig = freshSigs.get(traceId);
      const publicEnv = envByIdPublic.get(tr.env_id);
      const traceCommitment = commitmentByTrace.get(traceId);
      const t0 = performance.now();
      verify({
        sigma: sig, expectedTraceId: traceId, publicEnv, m_min,
        promise: V0_PROMISE, thresholds: V0_CHECKER_THRESHOLDS,
        traceCommitment, cacheState, stageLabel: "cost_median_verify",
      });
      cVerifyMs += performance.now() - t0;
      cVerifyOps += 10;
    }

    // ── C_full_state: privileged baseline over measurement pairs ──
    let cFullStateMs = 0;
    let cFullStateOps = 0;
    for (const tr of measurementTraces) {
      const env = envByIdFull.get(tr.env_id);
      const t0 = performance.now();
      fullStateCheck(env, { positions: tr.positions }, m_min);
      cFullStateMs += performance.now() - t0;
      cFullStateOps += tr.positions.length;
    }

    // ── C_rollout: op accounting only (diagnostic denominator) ──
    let cRolloutOps = 0;
    for (const tr of measurementTraces) {
      const T = tr.positions.length - 1;
      cRolloutOps += 5 * T + T;
    }

    const cTotalSignatureMs = cSignatureMs + cVerifyMs;
    const cTotalSignatureOps = cSignatureOps + cVerifyOps;
    samples.push({
      pass: pass + 1,
      c_signature_ms: cSignatureMs,
      c_verify_ms: cVerifyMs,
      c_total_signature_ms: cTotalSignatureMs,
      c_full_state_ms: cFullStateMs,
      full_state_ratio_wall: cFullStateMs > 0 ? cTotalSignatureMs / cFullStateMs : null,
      op_ratio: cRolloutOps > 0 ? cTotalSignatureOps / cRolloutOps : null,
      c_total_signature_ops: cTotalSignatureOps,
      c_rollout_ops: cRolloutOps,
    });
  }

  const totals = samples.map((s) => s.c_total_signature_ms);
  const medTotal = median(totals);
  const minTotal = Math.min(...totals);
  const maxTotal = Math.max(...totals);
  const meanTotal = mean(totals);
  const spreadPct = medTotal > 0 ? ((maxTotal - minTotal) / medTotal) * 100 : 0;
  const medFullStateRatio = median(samples.map((s) => s.full_state_ratio_wall));
  const medOpRatio = median(samples.map((s) => s.op_ratio));

  // Pull cross-artifact inputs needed for the full cost-repair clause set.
  const denomAudit = await readJsonIfExists(path.join(outDir, "cost_denominator_audit.json"));
  const cacheEff = await readJsonIfExists(path.join(outDir, "cache_efficiency_report.json"));
  const rolloutDiagnosticOnly = denomAudit
    ? (denomAudit.rollout?.wall_ms_below_5ms === true && denomAudit.rollout?.stable_enough_for_ratio_denominator === false)
    : null;
  const cacheReuseRate = cacheEff ? cacheEff.cache_eligible_reuse_hit_rate : null;

  const clauses = {
    median_c_total_signature_le_1010: medTotal <= TARGET_C_TOTAL_MS,
    median_full_state_ratio_le_105: medFullStateRatio !== null && medFullStateRatio <= TARGET_FULL_STATE_RATIO,
    median_op_ratio_le_1: medOpRatio !== null && medOpRatio <= TARGET_OP_RATIO,
    max_c_total_signature_le_1250: maxTotal <= TARGET_C_TOTAL_MAX_MS,
    spread_pct_le_25: spreadPct <= TARGET_SPREAD_PCT,
    rollout_diagnostic_only: rolloutDiagnosticOnly === true,
    cache_eligible_reuse_ge_95: cacheReuseRate !== null && cacheReuseRate >= CACHE_REUSE_FLOOR,
    short_circuit_instrumentation_passed: scAudit.passed,
  };
  const costRepairPassed = Object.values(clauses).every(Boolean);

  const report = {
    schema: "pvnp-phase1-cost-multirun-report-v5",
    passes: PASSES,
    samples,
    statistics: {
      c_total_signature_ms: {
        min: minTotal, max: maxTotal, mean: meanTotal, median: medTotal,
        spread_pct: spreadPct,
      },
      full_state_ratio_wall_median: medFullStateRatio,
      op_ratio_median: medOpRatio,
    },
    targets: {
      median_c_total_signature_ms_max: TARGET_C_TOTAL_MS,
      max_c_total_signature_ms_max: TARGET_C_TOTAL_MAX_MS,
      full_state_ratio_wall_max: TARGET_FULL_STATE_RATIO,
      op_ratio_max: TARGET_OP_RATIO,
      spread_pct_max: TARGET_SPREAD_PCT,
      cache_eligible_reuse_min: CACHE_REUSE_FLOOR,
    },
    cross_artifact_inputs: {
      rollout_diagnostic_only: rolloutDiagnosticOnly,
      cache_eligible_reuse_hit_rate: cacheReuseRate,
      short_circuit_instrumentation_passed: scAudit.passed,
    },
    cost_gate_clauses: clauses,
    cost_repair_passed: costRepairPassed,
  };

  await writeFile(
    path.join(outDir, "cost_multirun_report.json"),
    JSON.stringify(report, null, 2) + "\n",
    "utf8",
  );

  console.log(`cost multirun over ${PASSES} passes:`);
  for (const s of samples) {
    console.log(`  pass ${s.pass}: C_total_signature=${s.c_total_signature_ms.toFixed(2)}ms full_state_ratio=${s.full_state_ratio_wall.toFixed(2)}x`);
  }
  console.log(`  MEDIAN C_total_signature=${medTotal.toFixed(2)}ms (<=${TARGET_C_TOTAL_MS}? ${clauses.median_c_total_signature_le_1010})  max=${maxTotal.toFixed(2)}ms (<=${TARGET_C_TOTAL_MAX_MS}? ${clauses.max_c_total_signature_le_1250})  spread=${spreadPct.toFixed(1)}% (<=25? ${clauses.spread_pct_le_25})`);
  console.log(`  MEDIAN full_state_ratio=${medFullStateRatio.toFixed(2)}x (<=${TARGET_FULL_STATE_RATIO}? ${clauses.median_full_state_ratio_le_105})`);
  console.log(`  MEDIAN op_ratio=${medOpRatio.toFixed(4)} (<=1.0? ${clauses.median_op_ratio_le_1})`);
  console.log(`  short_circuit_audit_passed=${scAudit.passed}  cache_reuse=${cacheReuseRate}  cost_repair_passed=${costRepairPassed}`);
}

main().catch((err) => { console.error(err); process.exit(1); });
