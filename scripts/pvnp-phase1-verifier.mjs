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
  await mkdir(outDir, { recursive: true });

  const sigs = await readJsonl(path.join(outDir, "signatures.jsonl"));
  const envs = await readJsonl(path.join(outDir, "environments.jsonl"));
  const calibrationManifest = JSON.parse(
    await readFile(path.join(outDir, "calibration_manifest.json"), "utf8"),
  );
  const m_min = calibrationManifest.selected_m_min;

  const envById = new Map(envs.map((e) => [e.id, redactEnv(e)]));

  const rows = [[
    "env_id", "policy_id", "split", "decision", "reason",
    "margin_lower_bound", "coverage_touched", "invariance_pass", "noise_std_estimate",
    "verify_wall_ms", "verify_ops",
  ].join(",")];

  let nAccept = 0, nReject = 0, nQuarantine = 0;
  const verifyCosts = { wall_ms: 0, ops: 0, calls: 0 };

  for (const sigma of sigs) {
    const policyId = sigma.source_observations.policy_id;
    const envId = sigma.source_observations.env_id;
    const publicEnv = envById.get(envId);
    if (publicEnv.split === "calibration") continue; // measurement only

    const expectedTraceId = `${policyId}|${envId}`;
    const t0 = performance.now();
    const result = verify({
      sigma,
      expectedTraceId,
      publicEnv,
      m_min,
      promise: V0_PROMISE,
      thresholds: V0_CHECKER_THRESHOLDS,
    });
    const elapsed = performance.now() - t0;
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
      sigma.coverage_digest.touched_cells,
      sigma.invariance_checks.all_pass ? 1 : 0,
      sigma.sensor_health.noise_std_estimate.toFixed(6),
      elapsed.toFixed(3), ops,
    ]));
  }

  await writeFile(
    path.join(outDir, "verifier_decisions.csv"),
    rows.join("\n") + "\n",
    "utf8",
  );

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
