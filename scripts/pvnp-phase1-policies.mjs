#!/usr/bin/env node
// scripts/pvnp-phase1-policies.mjs
//
// Register the v0 policy roster: two hand-coded controllers + one small MLP
// slot. Writes `policies.jsonl`. The MLP slot is filled by the trainer
// (training/pvnp/train_mlp_policy.py) writing back the checkpoint path.

import { mkdir, writeFile, readFile, access } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { POLICY_CLASSES } from "./lib/pvnp-phase1-policies-core.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const args = { runDir: "results/pvnp/phase1-toy-verifier-v0" };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--run-dir") { args.runDir = argv[i + 1]; i += 1; }
    else throw new Error(`Unknown flag: ${argv[i]}`);
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(REPO_ROOT, args.runDir);
  await mkdir(outDir, { recursive: true });

  const roster = [
    {
      policy_id: "hc_safe_seeker_v0",
      policy_class: POLICY_CLASSES.HC_SAFE,
      training: { kind: "hand_coded", deterministic: true, params_count: 0 },
      trained_on_split: null,
      checkpoint: null,
    },
    {
      policy_id: "hc_decoy_seeker_v0",
      policy_class: POLICY_CLASSES.HC_DECOY,
      training: { kind: "hand_coded", deterministic: true, params_count: 0 },
      trained_on_split: null,
      checkpoint: null,
    },
    {
      policy_id: "small_mlp_seed_0",
      policy_class: POLICY_CLASSES.MLP_SMALL,
      training: { kind: "mlp_trained", deterministic: false, params_count_max: 20000, seed: 0 },
      trained_on_split: "train",
      checkpoint: null, // filled in by training/pvnp/train_mlp_policy.py
    },
  ];

  const body = roster.map((r) => JSON.stringify(r)).join("\n") + "\n";
  const outPath = path.join(outDir, "policies.jsonl");
  await writeFile(outPath, body, "utf8");
  console.log(`wrote ${roster.length} policies → ${path.relative(REPO_ROOT, outPath)}`);
}

main().catch((err) => { console.error(err); process.exit(1); });
