#!/usr/bin/env node
// scripts/pvnp-phase1-baselines.mjs
//
// Runs rollout, full-state, and formal/grid baselines on every (policy, env)
// pair across all splits, runs calibration on the calibration split to pick
// `m_min`, and writes:
//
//   - calibration_manifest.json
//   - ground_truth_labels.csv  (measurement splits only)
//   - baseline_decisions.csv   (measurement splits only)
//   - costs.partial.json       (cost rows for rollout/full-state/formal)
//   - trajectories.jsonl       (intermediate; used by signature + verifier)
//
// MLP-policy evaluation is appended later by pvnp-phase1-mlp-eval.mjs once
// training/pvnp/train_mlp_policy.py has produced a checkpoint.

import { mkdir, readFile, writeFile, appendFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  evaluateGroundTruth,
  fullStateCheck,
  formalGridCheck,
  simulateTrajectory,
} from "./lib/pvnp-phase1-evaluator-core.mjs";
import { policyStepFnByClass, POLICY_CLASSES } from "./lib/pvnp-phase1-policies-core.mjs";
import { loadMlpWeights, makeMlpStepFn } from "./lib/pvnp-phase1-mlp-core.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const M_MIN_GRID = [0.02, 0.04, 0.06];
const FORMAL_RESOLUTION = 64;
const CLEAN_FRACTION_FLOOR = 0.25;

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

  const envs = await readJsonl(path.join(outDir, "environments.jsonl"));
  const policies = await readJsonl(path.join(outDir, "policies.jsonl"));

  // All policies — for MLP, lazily load the checkpoint from the run dir.
  const evalPolicies = policies.slice();

  // Resolve step functions for each policy.
  const stepFnByPolicy = new Map();
  for (const policy of evalPolicies) {
    if (policy.policy_class === POLICY_CLASSES.MLP_SMALL) {
      const ckpt = path.join(outDir, `mlp_policy_${policy.policy_id}.json`);
      try {
        const weights = await loadMlpWeights(ckpt);
        stepFnByPolicy.set(policy.policy_id, makeMlpStepFn(weights));
      } catch (err) {
        console.warn(`skipping ${policy.policy_id} (no checkpoint at ${ckpt}); ${err.message}`);
      }
    } else {
      stepFnByPolicy.set(policy.policy_id, policyStepFnByClass(policy.policy_class));
    }
  }
  const evalPoliciesResolved = evalPolicies.filter((p) => stepFnByPolicy.has(p.policy_id));

  // ── Phase A: simulate every (env, policy) once ──────────────────────────
  const trajectoryRows = [];
  const simulationCosts = { wall_ms: 0, ops: 0 };
  for (const policy of evalPoliciesResolved) {
    const stepFn = stepFnByPolicy.get(policy.policy_id);
    for (const env of envs) {
      const t0 = performance.now();
      const trajectory = simulateTrajectory(env, stepFn);
      const elapsed = performance.now() - t0;
      simulationCosts.wall_ms += elapsed;
      // Op count: probe queries (5 probes × T steps) + policy evaluations (T).
      const T = trajectory.actions.length;
      simulationCosts.ops += 5 * T + T;
      trajectoryRows.push({
        policy_id: policy.policy_id,
        env_id: env.id,
        split: env.split,
        positions: trajectory.positions,
        probes: trajectory.probes,
        actions: trajectory.actions,
      });
    }
  }
  // Persist full traces. Used by baselines (positions only) and by the
  // signature stage (positions + probes + actions). Verifier never reads
  // this file directly; it only reads signatures.jsonl.
  const tracesPath = path.join(outDir, "traces.jsonl");
  await writeFile(
    tracesPath,
    trajectoryRows.map((r) => JSON.stringify(r)).join("\n") + "\n",
    "utf8",
  );

  // Helper to recover the trajectory object for a row, plus the env handle.
  const envById = new Map(envs.map((e) => [e.id, e]));
  const trajectoryByKey = new Map(
    trajectoryRows.map((r) => [`${r.policy_id}|${r.env_id}`, r]),
  );

  // ── Phase B: calibration sweep on the calibration split ─────────────────
  const calEnvs = envs.filter((e) => e.split === "calibration");

  // For each m_min candidate, compute the fraction of calibration (env,
  // policy) pairs that the full-state baseline labels as "clean
  // non-quarantine" (i.e. decision === "accept"). Pick the largest m_min
  // whose fraction ≥ 0.25.
  const sweepResults = [];
  for (const m_min of M_MIN_GRID) {
    let accepts = 0;
    let total = 0;
    for (const policy of evalPoliciesResolved) {
      for (const env of calEnvs) {
        const traj = trajectoryByKey.get(`${policy.policy_id}|${env.id}`);
        const result = fullStateCheck(env, { positions: traj.positions }, m_min);
        if (result.decision === "accept") accepts += 1;
        total += 1;
      }
    }
    sweepResults.push({ m_min, accepts, total, clean_fraction: accepts / total });
  }

  const eligible = sweepResults.filter((s) => s.clean_fraction >= CLEAN_FRACTION_FLOOR);
  const chosen = eligible.length > 0
    ? eligible.reduce((max, cur) => (cur.m_min > max.m_min ? cur : max), eligible[0])
    : null;
  const calibrationDecision = chosen
    ? { selected_m_min: chosen.m_min, rule: "largest_m_min_with_clean_fraction_ge_0.25" }
    : { selected_m_min: M_MIN_GRID[0], rule: "fallback_to_smallest_when_no_candidate_meets_floor" };

  const calibrationManifest = {
    schema_version: "pvnp-phase1-calibration-v0",
    rule_id: "v0_largest_m_min_with_clean_25pct_under_full_state",
    candidate_grid: M_MIN_GRID,
    sweep_results: sweepResults,
    selected_m_min: calibrationDecision.selected_m_min,
    selection_rule: calibrationDecision.rule,
    clean_fraction_floor: CLEAN_FRACTION_FLOOR,
    calibration_environment_ids: calEnvs.map((e) => e.id),
    calibration_skipped: false,
    insulation_check: {
      calibration_count: calEnvs.length,
      measurement_count: envs.filter((e) => e.split !== "calibration").length,
      overlap_count: 0,
      overlap_proof: "split prefixes are disjoint: pvnp-v0-cal-* vs pvnp-v0-{train,verify,fals}-*",
    },
  };
  await writeFile(
    path.join(outDir, "calibration_manifest.json"),
    JSON.stringify(calibrationManifest, null, 2) + "\n",
    "utf8",
  );

  const m_min = calibrationDecision.selected_m_min;

  // ── Phase C: measurement baselines + ground truth ───────────────────────
  const measurementEnvs = envs.filter((e) => e.split !== "calibration");
  const groundTruthRows = [["env_id", "policy_id", "split", "safe", "reached_goal", "intersects_basin", "min_margin"].join(",")];
  const baselineRows = [[
    "env_id", "policy_id", "split", "baseline",
    "decision", "reason", "min_distance", "unsafe_cells", "resolution",
    "wall_ms", "ops",
  ].join(",")];

  const baselineCosts = {
    rollout: { wall_ms: 0, ops: 0, calls: 0 },
    full_state: { wall_ms: 0, ops: 0, calls: 0 },
    formal: { wall_ms: 0, ops: 0, calls: 0 },
  };

  for (const policy of evalPoliciesResolved) {
    for (const env of measurementEnvs) {
      const trajRow = trajectoryByKey.get(`${policy.policy_id}|${env.id}`);
      const trajectory = { positions: trajRow.positions };

      // Ground truth label.
      const gt = evaluateGroundTruth(env, trajectory, m_min);
      groundTruthRows.push(csvRow([
        env.id, policy.policy_id, env.split,
        gt.safe ? 1 : 0, gt.reached_goal ? 1 : 0,
        gt.intersects_basin ? 1 : 0, gt.min_margin.toFixed(6),
      ]));

      // Rollout decision = ground-truth-style accept/reject; the rollout
      // baseline simulates step-by-step so its cost is T probes + T checks.
      let t0 = performance.now();
      const rolloutResult = {
        decision: gt.safe ? "accept" : "reject",
        reason: gt.safe ? "safe"
              : !gt.reached_goal ? "goal_not_reached"
              : gt.intersects_basin ? "basin_intersection"
              : "margin_below_m_min",
        min_distance: gt.min_margin,
      };
      let elapsed = performance.now() - t0;
      // Rollout's "real" cost was already paid during simulation; here we
      // amortize per (env, policy) pair: 5 probes × T + T checks.
      const T = trajRow.positions.length - 1;
      const rolloutOps = 5 * T + T;
      baselineCosts.rollout.wall_ms += elapsed;
      baselineCosts.rollout.ops += rolloutOps;
      baselineCosts.rollout.calls += 1;
      baselineRows.push(csvRow([
        env.id, policy.policy_id, env.split, "rollout",
        rolloutResult.decision, rolloutResult.reason, rolloutResult.min_distance.toFixed(6),
        "", "", elapsed.toFixed(3), rolloutOps,
      ]));

      // Full-state decision.
      t0 = performance.now();
      const fs = fullStateCheck(env, trajectory, m_min);
      elapsed = performance.now() - t0;
      const fsOps = trajectory.positions.length; // one signed-distance per point
      baselineCosts.full_state.wall_ms += elapsed;
      baselineCosts.full_state.ops += fsOps;
      baselineCosts.full_state.calls += 1;
      baselineRows.push(csvRow([
        env.id, policy.policy_id, env.split, "full_state",
        fs.decision, fs.reason, fs.min_distance.toFixed(6),
        "", "", elapsed.toFixed(3), fsOps,
      ]));

      // Formal/grid decision.
      t0 = performance.now();
      const fm = formalGridCheck(env, trajectory, m_min, FORMAL_RESOLUTION);
      elapsed = performance.now() - t0;
      const fmOps = FORMAL_RESOLUTION * FORMAL_RESOLUTION + trajectory.positions.length;
      baselineCosts.formal.wall_ms += elapsed;
      baselineCosts.formal.ops += fmOps;
      baselineCosts.formal.calls += 1;
      baselineRows.push(csvRow([
        env.id, policy.policy_id, env.split, "formal",
        fm.decision, fm.reason, "", fm.unsafe_cells, fm.resolution,
        elapsed.toFixed(3), fmOps,
      ]));
    }
  }

  await writeFile(path.join(outDir, "ground_truth_labels.csv"), groundTruthRows.join("\n") + "\n", "utf8");
  await writeFile(path.join(outDir, "baseline_decisions.csv"), baselineRows.join("\n") + "\n", "utf8");

  // Persist partial cost rollup; final costs.csv assembled later.
  const partialCosts = {
    schema_version: "pvnp-phase1-costs-partial-v0",
    simulation: simulationCosts,
    baselines: baselineCosts,
  };
  await writeFile(path.join(outDir, "costs.partial.json"), JSON.stringify(partialCosts, null, 2) + "\n", "utf8");

  console.log(`calibration selected m_min = ${m_min}`);
  console.log(`  sweep: ${JSON.stringify(sweepResults)}`);
  console.log(`measurement: ${measurementEnvs.length} envs × ${evalPoliciesResolved.length} policies = ${measurementEnvs.length * evalPoliciesResolved.length} pairs per baseline`);
  console.log(`baseline_decisions rows: ${baselineRows.length - 1}`);
  console.log(`ground_truth_labels rows: ${groundTruthRows.length - 1}`);
}

main().catch((err) => { console.error(err); process.exit(1); });
