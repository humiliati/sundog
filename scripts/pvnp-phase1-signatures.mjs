#!/usr/bin/env node
// scripts/pvnp-phase1-signatures.mjs
//
// Compute signature certificates for every (policy, env) trace and write
// signatures.jsonl. Reads only the (policy_id, env_id, positions, probes,
// actions, public env metadata) — no hidden state.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { computeSignature } from "./lib/pvnp-phase1-signature-core.mjs";

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

// Redact hidden_state. Pure function; no imports from env-core to keep the
// privilege boundary explicit (this runner sits on the verifier side).
function redactEnv(env) {
  const { hidden_state: _hidden, ...rest } = env;
  return rest;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(REPO_ROOT, args.runDir);
  await mkdir(outDir, { recursive: true });

  const traces = await readJsonl(path.join(outDir, "traces.jsonl"));
  const envs = await readJsonl(path.join(outDir, "environments.jsonl"));
  const envById = new Map(envs.map((e) => [e.id, redactEnv(e)]));

  const sigs = [];
  for (const trace of traces) {
    const publicEnv = envById.get(trace.env_id);
    if (!publicEnv) throw new Error(`env not found: ${trace.env_id}`);
    const sigma = computeSignature({
      traceId: `${trace.policy_id}|${trace.env_id}`,
      publicEnv,
      positions: trace.positions,
      probes: trace.probes,
    });
    sigs.push(sigma);
  }

  await writeFile(
    path.join(outDir, "signatures.jsonl"),
    sigs.map((s) => JSON.stringify(s)).join("\n") + "\n",
    "utf8",
  );
  console.log(`wrote ${sigs.length} signatures`);
}

main().catch((err) => { console.error(err); process.exit(1); });
