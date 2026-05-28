#!/usr/bin/env node
// scripts/pvnp-phase1-generate-environments.mjs
//
// Generate the v0 environment slate for SUNDOG_V_P_V_NP Phase 1.
// Writes `environments.jsonl` (with `hidden_state` field intact for the
// evaluator) and `manifest.json` (run metadata + freeze hash).
//
// Spec references:
//   docs/pvnp/PHASE1_TOY_VERIFIER_SPEC.md
//   docs/pvnp/PHASE1_V0_SLATE.md

import { execFileSync } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { canonicalize, sha256Hex } from "./lib/canonical-json.mjs";
import { generateSplit, BASIN_FAMILIES, PROBE_NOISE_TIERS, PROMISE_BOUNDS } from "./lib/pvnp-phase1-env-core.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const V0_SLATE = Object.freeze({
  run_id: "phase1-toy-verifier-v0",
  spec_path: "docs/pvnp/PHASE1_TOY_VERIFIER_SPEC.md",
  slate_path: "docs/pvnp/PHASE1_V0_SLATE.md",
  splits: [
    { split: "calibration", count: 64,  seedPrefix: "pvnp-v0-cal",    inPromise: true  },
    { split: "train",       count: 256, seedPrefix: "pvnp-v0-train",  inPromise: true  },
    { split: "verification",count: 256, seedPrefix: "pvnp-v0-verify", inPromise: true  },
    { split: "falsifier",   count: 256, seedPrefix: "pvnp-v0-fals",   inPromise: false },
  ],
  basin_families: BASIN_FAMILIES,
  probe_noise_tiers: PROBE_NOISE_TIERS,
  promise_bounds: PROMISE_BOUNDS,
  horizon: 128,
  max_action_step: 0.025,
  m_min_candidate_grid: [0.02, 0.04, 0.06],
  verifier_access_declaration: {
    may_read: ["candidate_policy_id", "policy_class", "sigma", "promise_params", "checker_thresholds"],
    may_not_read: ["hidden_state", "basin_params", "latent_field", "decoy_params", "ground_truth_labels", "post_result_thresholds"],
    forbidden_tokens_grep: ["ground_truth_labels", "B_theta", "F_theta", "hidden_state", "basin_params", "latent_field", "decoy_params"],
  },
});

function parseArgs(argv) {
  const args = { out: "results/pvnp/phase1-toy-verifier-v0" };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--out") {
      args.out = argv[i + 1];
      i += 1;
    } else throw new Error(`Unknown flag: ${argv[i]}`);
  }
  return args;
}

function commitHash() {
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], { cwd: REPO_ROOT, encoding: "utf8" }).trim();
  } catch {
    return "unknown";
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(REPO_ROOT, args.out);
  await mkdir(outDir, { recursive: true });

  // Generate all splits deterministically.
  const allEnvs = [];
  const perSplitCounts = {};
  for (const split of V0_SLATE.splits) {
    const envs = generateSplit(split);
    allEnvs.push(...envs);
    perSplitCounts[split.split] = envs.length;
  }

  // Write environments.jsonl. Each env is on its own line. The full env
  // (including `hidden_state`) is written; verifier-side loaders MUST call
  // `redactForVerifier(env)` before passing to verifier code.
  const envsPath = path.join(outDir, "environments.jsonl");
  const envsBody = allEnvs.map((e) => canonicalize(e)).join("\n") + "\n";
  await writeFile(envsPath, envsBody, "utf8");

  // Manifest.
  const manifest = {
    schema_version: "pvnp-phase1-manifest-v0",
    run_id: V0_SLATE.run_id,
    slate: V0_SLATE,
    commit: commitHash(),
    generated_at: new Date().toISOString(),
    counts: perSplitCounts,
    total_envs: allEnvs.length,
    environments_path: "environments.jsonl",
    environments_sha256: sha256Hex(envsBody),
  };
  const manifestPath = path.join(outDir, "manifest.json");
  await writeFile(manifestPath, JSON.stringify(manifest, null, 2) + "\n", "utf8");

  // Stdout summary so the run is visible in CI logs.
  console.log(`generated ${allEnvs.length} envs across ${V0_SLATE.splits.length} splits`);
  for (const [split, count] of Object.entries(perSplitCounts)) {
    console.log(`  ${split}: ${count}`);
  }
  console.log(`manifest: ${path.relative(REPO_ROOT, manifestPath)}`);
  console.log(`environments_sha256: ${manifest.environments_sha256.slice(0, 12)}...`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
