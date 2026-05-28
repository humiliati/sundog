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
import { generateSplit } from "./lib/pvnp-phase1-env-core.mjs";
import { getPhase1RunConfig, phase1Schema } from "./lib/pvnp-phase1-run-config.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

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
  const slate = getPhase1RunConfig(args.out);
  await mkdir(outDir, { recursive: true });

  // Generate all splits deterministically.
  const allEnvs = [];
  const perSplitCounts = {};
  for (const split of slate.splits) {
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
    schema_version: phase1Schema("manifest", slate.schema_suffix),
    run_id: slate.run_id,
    slate,
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
  console.log(`generated ${allEnvs.length} envs across ${slate.splits.length} splits`);
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
