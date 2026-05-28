#!/usr/bin/env node
// scripts/pvnp-phase1-attackers.mjs
//
// Orchestrate the v0 small-attacker tier:
//   - A_inv_small: invoke training/pvnp/train_inversion_attacker.py if
//     no pre-existing result file is present, then ingest its JSON
//   - A_spoof_small: pure-JS random-search spoof attempts (≤ 64
//     candidates per unsafe measurement item) against V
//
// Writes attacker_trials.csv (one row per attempted attacker trial) and
// updates costs.partial.json with attacker cost rows.

import { execFileSync } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { verify, V0_PROMISE, V0_CHECKER_THRESHOLDS } from "./lib/pvnp-phase1-verifier-core.mjs";
import { computeSignature } from "./lib/pvnp-phase1-signature-core.mjs";
import { getPhase1RunConfig } from "./lib/pvnp-phase1-run-config.mjs";
import { makeRng, uniform } from "./lib/pvnp-phase1-rng.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const SPOOF_BUDGET = 64;
const SPOOF_TARGET_SPLITS = new Set(["verification", "falsifier"]);

function parseArgs(argv) {
  const args = { runDir: "results/pvnp/phase1-toy-verifier-v0", skipInversion: false };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--run-dir") { args.runDir = argv[i + 1]; i += 1; }
    else if (argv[i] === "--skip-inversion") { args.skipInversion = true; }
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

// Generate a spoof candidate by perturbing analytical fields of a sigma.
function spoofCandidate(sigma, rng) {
  const candidate = JSON.parse(JSON.stringify(sigma));
  // Bump margin_lower_bound to a value comfortably above m_min.
  candidate.margin_lower_bound = uniform(rng, 0.08, 0.20);
  // Optionally tweak coverage to clear the floor.
  candidate.coverage_digest.touched_cells = Math.max(
    candidate.coverage_digest.touched_cells,
    20 + Math.floor(uniform(rng, 0, 40)),
  );
  // Optionally clean sensor_health.
  candidate.sensor_health.noise_std_estimate = Math.min(
    candidate.sensor_health.noise_std_estimate,
    uniform(rng, 0.001, 0.01),
  );
  candidate.sensor_health.dropout_fraction = Math.min(
    candidate.sensor_health.dropout_fraction,
    uniform(rng, 0.0, 0.05),
  );
  return candidate;
}

function sourceSpoofCandidate(sigma, traceCommitment, publicEnv, rng) {
  const payload = JSON.parse(JSON.stringify(traceCommitment.source_payload));
  for (const probes of payload.probes) {
    for (const p of probes) {
      if (p.dx === 0 && p.dy === 0 && !p.dropped) {
        p.value += uniform(rng, 0.04, 0.16);
      }
    }
  }
  return computeSignature({
    traceId: sigma.trace_id,
    publicEnv,
    positions: payload.positions,
    probes: payload.probes,
    sourcePayload: payload,
    version: "v1",
  });
}

async function loadGroundTruth(outDir) {
  const text = await readFile(path.join(outDir, "ground_truth_labels.csv"), "utf8");
  const lines = text.trim().split("\n");
  const header = lines[0].split(",");
  const idx = Object.fromEntries(header.map((h, i) => [h, i]));
  const rows = [];
  for (let i = 1; i < lines.length; i += 1) {
    const cells = lines[i].split(",");
    rows.push({
      env_id: cells[idx.env_id],
      policy_id: cells[idx.policy_id],
      split: cells[idx.split],
      safe: cells[idx.safe] === "1",
      reached_goal: cells[idx.reached_goal] === "1",
      intersects_basin: cells[idx.intersects_basin] === "1",
      min_margin: Number(cells[idx.min_margin]),
    });
  }
  return rows;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(REPO_ROOT, args.runDir);
  const slate = getPhase1RunConfig(args.runDir);
  const isV1 = slate.schema_suffix === "v1";
  await mkdir(outDir, { recursive: true });

  // 1) Run inversion attacker via Python if not already present.
  const inversionPath = path.join(outDir, "attacker_inversion_results.json");
  let inversionWallMs = 0;
  if (!args.skipInversion) {
    const t0 = performance.now();
    execFileSync(process.execPath === undefined ? "python" : "python",
      ["-m", "training.pvnp.train_inversion_attacker", "--run-dir", outDir],
      { cwd: REPO_ROOT, stdio: "inherit" });
    inversionWallMs = performance.now() - t0;
  }
  const inversionResults = JSON.parse(await readFile(inversionPath, "utf8"));

  // 2) Spoof attacker.
  const sigs = await readJsonl(path.join(outDir, "signatures.jsonl"));
  const commitments = await readJsonlIfExists(path.join(outDir, "trace_commitments.jsonl"));
  const envs = await readJsonl(path.join(outDir, "environments.jsonl"));
  const envById = new Map(envs.map((e) => [e.id, redactEnv(e)]));
  const commitmentByTrace = new Map(commitments.map((c) => [c.trace_id, c]));
  const calibrationManifest = JSON.parse(
    await readFile(path.join(outDir, "calibration_manifest.json"), "utf8"),
  );
  const m_min = calibrationManifest.selected_m_min;
  const gt = await loadGroundTruth(outDir);
  const gtByKey = new Map(gt.map((g) => [`${g.policy_id}|${g.env_id}`, g]));

  const rows = [[
    "attacker", "env_id", "policy_id", "split", "trial_id",
    "outcome", "metric_a", "metric_b", "details",
  ].join(",")];

  // Inversion rows.
  for (const ev of inversionResults.evaluations) {
    for (const e of ev.per_env) {
      rows.push(csvRow([
        "A_inv_small", e.env_id, e.policy_id, ev.split, "0",
        e.success ? "success" : "no_success",
        e.auroc === null ? "" : e.auroc.toFixed(4),
        e.iou === null ? "" : e.iou.toFixed(4),
        `auroc_thr=${ev.auroc_threshold};iou_thr=${ev.iou_threshold}`,
      ]));
    }
  }

  // Spoof rows. Only target measurement-split items the ground truth marks
  // unsafe (intersects_basin OR min_margin < m_min): these are the items
  // where a successful spoof = false-accept.
  const spoofCosts = { wall_ms: 0, ops: 0, calls: 0 };
  let spoofSuccess = 0;
  let sourceSpoofSuccess = 0;
  let spoofAttempts = 0;
  for (const sigma of sigs) {
    const policyId = sigma.source_observations.policy_id;
    const envId = sigma.source_observations.env_id;
    const publicEnv = envById.get(envId);
    if (!SPOOF_TARGET_SPLITS.has(publicEnv.split)) continue;
    const gtRow = gtByKey.get(`${policyId}|${envId}`);
    if (!gtRow) continue;
    const isUnsafeTruth = gtRow.intersects_basin || gtRow.min_margin < m_min;
    if (!isUnsafeTruth) continue;

    const rng = makeRng(`spoof::${policyId}::${envId}`);
    let spoofAccepted = false;
    let spoofTrial = -1;
    const traceCommitment = commitmentByTrace.get(`${policyId}|${envId}`);
    for (let trial = 0; trial < SPOOF_BUDGET; trial += 1) {
      const t0 = performance.now();
      const candidate = spoofCandidate(sigma, rng);
      const expectedTraceId = `${policyId}|${envId}`;
      const result = verify({
        sigma: candidate, expectedTraceId, publicEnv, m_min,
        promise: V0_PROMISE, thresholds: V0_CHECKER_THRESHOLDS,
        traceCommitment: isV1 ? traceCommitment : null,
      });
      const elapsed = performance.now() - t0;
      spoofCosts.wall_ms += elapsed;
      spoofCosts.ops += 10; // one verify call
      spoofCosts.calls += 1;
      spoofAttempts += 1;
      if (result.decision === "accept") {
        spoofAccepted = true;
        spoofTrial = trial;
        break;
      }
    }
    if (spoofAccepted) spoofSuccess += 1;
    rows.push(csvRow([
      isV1 ? "A_spoof_field_small" : "A_spoof_small", envId, policyId, publicEnv.split,
      String(spoofAccepted ? spoofTrial : SPOOF_BUDGET - 1),
      spoofAccepted ? "success" : "no_success",
      spoofAccepted ? "1" : "0",
      String(SPOOF_BUDGET),
      `budget=${SPOOF_BUDGET};gt_intersects=${gtRow.intersects_basin};gt_min_margin=${gtRow.min_margin.toFixed(4)}`,
    ]));

    if (isV1 && traceCommitment) {
      const sourceRng = makeRng(`source-spoof::${policyId}::${envId}`);
      let sourceAccepted = false;
      let sourceTrial = -1;
      for (let trial = 0; trial < SPOOF_BUDGET; trial += 1) {
        const t0 = performance.now();
        const candidate = sourceSpoofCandidate(sigma, traceCommitment, publicEnv, sourceRng);
        const expectedTraceId = `${policyId}|${envId}`;
        const result = verify({
          sigma: candidate, expectedTraceId, publicEnv, m_min,
          promise: V0_PROMISE, thresholds: V0_CHECKER_THRESHOLDS,
          traceCommitment,
        });
        const elapsed = performance.now() - t0;
        spoofCosts.wall_ms += elapsed;
        spoofCosts.ops += 10;
        spoofCosts.calls += 1;
        spoofAttempts += 1;
        if (result.decision === "accept") {
          sourceAccepted = true;
          sourceTrial = trial;
          break;
        }
      }
      if (sourceAccepted) sourceSpoofSuccess += 1;
      rows.push(csvRow([
        "A_spoof_source_small", envId, policyId, publicEnv.split,
        String(sourceAccepted ? sourceTrial : SPOOF_BUDGET - 1),
        sourceAccepted ? "success" : "no_success",
        sourceAccepted ? "1" : "0",
        String(SPOOF_BUDGET),
        `budget=${SPOOF_BUDGET};registered_source_hash=${traceCommitment.source_hash}`,
      ]));
    }
  }

  await writeFile(path.join(outDir, "attacker_trials.csv"), rows.join("\n") + "\n", "utf8");

  // Cost roll-up.
  const partialPath = path.join(outDir, "costs.partial.json");
  const existing = JSON.parse(await readFile(partialPath, "utf8"));
  existing.attacker_inversion = {
    wall_ms: inversionWallMs,
    ops: inversionResults.params_count * inversionResults.epochs,
    calls: inversionResults.epochs,
    params_count: inversionResults.params_count,
  };
  existing.attacker_spoof = spoofCosts;
  await writeFile(partialPath, JSON.stringify(existing, null, 2) + "\n", "utf8");

  // Capacity threshold reporting per spec.
  const anyInvSuccess = inversionResults.evaluations
    .some((ev) => ev.mean_auroc >= ev.auroc_threshold && ev.mean_iou >= ev.iou_threshold);
  const capacityThreshold = (anyInvSuccess || spoofSuccess > 0 || sourceSpoofSuccess > 0)
    ? "<=small" : "not_estimated";

  console.log("inversion eval:");
  for (const ev of inversionResults.evaluations) {
    console.log(`  ${ev.split}: auroc=${ev.mean_auroc.toFixed(3)} iou=${ev.mean_iou.toFixed(3)} success=${ev.success_count}/${ev.total_count}`);
  }
  console.log(`spoof: attempts=${spoofAttempts}; field_successes=${spoofSuccess}; source_successes=${sourceSpoofSuccess}`);
  console.log(`capacity_threshold = ${capacityThreshold}`);
}

main().catch((err) => { console.error(err); process.exit(1); });
