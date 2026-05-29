#!/usr/bin/env node
// scripts/pvnp-phase1-ablations.mjs
//
// Vacuity probes: run V with one analytical certificate field dropped at a
// time. Writes ablation_decisions.csv with one row per (env, policy,
// dropped-field) triple. Per spec, ablation lives outside Baselines —
// these are internal sanity checks, not external comparators.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { verify, V0_PROMISE, V0_CHECKER_THRESHOLDS } from "./lib/pvnp-phase1-verifier-core.mjs";
import { getPhase1RunConfig } from "./lib/pvnp-phase1-run-config.mjs";
import { loadCacheState, saveCacheState, statsReport } from "./lib/pvnp-phase1-cache.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// v3 drops sensor_health from the ablation roster per PHASE1_V3_SLATE.md
// §Sensor Disposition Gate ("ablation_decisions.csv must not include
// sensor_health_v1 as a slate-gated load-bearing field"). The remaining
// drop fields are the v3 load-bearing analytical fields.
const DROP_FIELDS_BY_VERSION = Object.freeze({
  v0: ["margin_lower_bound", "coverage_digest", "sensor_health", "invariance_checks"],
  v1: ["margin_lower_bound", "coverage_digest", "sensor_health_v1", "invariance_checks_v1"],
  v2: ["margin_lower_bound", "geometry_promise_signal_v2", "sensor_health_v1", "invariance_checks_v2"],
  v3: ["margin_lower_bound", "geometry_promise_signal_v2", "invariance_checks_v2"],
  v4: ["margin_lower_bound", "geometry_promise_signal_v2", "invariance_checks_v2"],
  v5: ["margin_lower_bound", "geometry_promise_signal_v2", "invariance_checks_v2"],
});

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
  const sourceBound = version === "v1" || version === "v2" || version === "v3" || version === "v4" || version === "v5";
  const usesCache = version === "v3" || version === "v4" || version === "v5";
  const dropFields = DROP_FIELDS_BY_VERSION[version] ?? DROP_FIELDS_BY_VERSION.v0;
  await mkdir(outDir, { recursive: true });

  // v3/v4 share the source-hash cache with the verifier stage. Load the
  // warm cache here; we expect ~100% hits in this stage.
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
  const commitmentByTrace = new Map(commitments.map((c) => [c.trace_id, c]));

  const rows = [[
    "env_id", "policy_id", "split", "dropped_field",
    "decision", "reason", "full_decision_match", "wall_ms", "ops",
  ].join(",")];

  // First, compute full-signature decisions (for the match flag).
  const fullDecisions = new Map();
  for (const sigma of sigs) {
    const policyId = sigma.source_observations.policy_id;
    const envId = sigma.source_observations.env_id;
    const publicEnv = envById.get(envId);
    if (publicEnv.split === "calibration") continue;
    const expectedTraceId = `${policyId}|${envId}`;
    const traceCommitment = commitmentByTrace.get(expectedTraceId);
    const result = verify({
      sigma, expectedTraceId, publicEnv, m_min,
      promise: V0_PROMISE, thresholds: V0_CHECKER_THRESHOLDS,
      traceCommitment: sourceBound ? traceCommitment : null,
      cacheState, stageLabel: "ablation_full",
    });
    fullDecisions.set(`${policyId}|${envId}`, result.decision);
  }

  const ablationCosts = { wall_ms: 0, ops: 0, calls: 0 };
  for (const sigma of sigs) {
    const policyId = sigma.source_observations.policy_id;
    const envId = sigma.source_observations.env_id;
    const publicEnv = envById.get(envId);
    if (publicEnv.split === "calibration") continue;
    const expectedTraceId = `${policyId}|${envId}`;
    const fullDec = fullDecisions.get(`${policyId}|${envId}`);
    const traceCommitment = commitmentByTrace.get(expectedTraceId);

    for (const field of dropFields) {
      const drop = new Set([field]);
      const t0 = performance.now();
      const result = verify({
        sigma, expectedTraceId, publicEnv, m_min,
        promise: V0_PROMISE, thresholds: V0_CHECKER_THRESHOLDS,
        dropFields: drop,
        traceCommitment: sourceBound ? traceCommitment : null,
        cacheState, stageLabel: "ablation",
      });
      const elapsed = performance.now() - t0;
      const ops = 10;
      ablationCosts.wall_ms += elapsed;
      ablationCosts.ops += ops;
      ablationCosts.calls += 1;
      rows.push(csvRow([
        envId, policyId, publicEnv.split, field,
        result.decision, result.reason,
        result.decision === fullDec ? 1 : 0,
        elapsed.toFixed(3), ops,
      ]));
    }
  }

  await writeFile(
    path.join(outDir, "ablation_decisions.csv"),
    rows.join("\n") + "\n",
    "utf8",
  );

  // Add ablation costs to partial costs.
  const partialPath = path.join(outDir, "costs.partial.json");
  const existing = JSON.parse(await readFile(partialPath, "utf8"));
  existing.ablation = ablationCosts;
  await writeFile(partialPath, JSON.stringify(existing, null, 2) + "\n", "utf8");

  if (usesCache) {
    await saveCacheState(cachePath, cacheState, "ablation");
    const stats = statsReport(cacheState);
    await writeFile(
      path.join(outDir, "verifier_cache_stats.partial.json"),
      JSON.stringify(stats, null, 2) + "\n",
      "utf8",
    );
  }

  // Vacuity verdict summary: for each dropped field, what fraction of
  // ablated decisions matched the full decision? High match → potential
  // vacuity (the field added nothing).
  const summary = Object.fromEntries(dropFields.map((f) => [f, { match: 0, total: 0 }]));
  for (let i = 1; i < rows.length; i += 1) {
    const cells = rows[i].split(",");
    const field = cells[3];
    const match = cells[6] === "1";
    summary[field].total += 1;
    if (match) summary[field].match += 1;
  }
  for (const f of dropFields) {
    const s = summary[f];
    const rate = s.total > 0 ? (s.match / s.total).toFixed(4) : "n/a";
    console.log(`  dropped ${f}: match rate = ${rate} (${s.match}/${s.total})`);
  }
}

main().catch((err) => { console.error(err); process.exit(1); });
