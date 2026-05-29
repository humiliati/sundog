// scripts/mesa-phase6-shard.mjs
//
// Parameterized wrapper for one Phase 6 lambda-control row, per
// docs/proof/PHASE6_LAMBDA_CONTROL.md. Mirrors the threebody-phase15-shard.mjs
// pattern: takes a single shard key (--label) and pins everything else.
//
// Usage:
//   node scripts/mesa-phase6-shard.mjs --label <label> [--thread-cap <N>] [--force]
//
// Examples:
//   node scripts/mesa-phase6-shard.mjs --label phase6_probe_noop_delta_lambda_0_95
//   node scripts/mesa-phase6-shard.mjs --label phase6_noop_delta_lambda_0_95 --thread-cap 2
//
// --thread-cap caps OMP/MKL/OPENBLAS thread pools (default 1, for fan-out).
// Without a cap, PyTorch BLAS will consume every available core and concurrent
// shards will oversubscribe.
//
// Resume-safe: if the row's policy file already exists, this skips the run and
// exits 0 (matching the `Test-Path $policy` guard in the spec's lock loop).
// Pass --force to re-run anyway.

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  ROWS,
  getRow,
  buildTrainArgs,
  buildShardEnv,
  policyPath,
  pythonExec,
} from "./lib/mesa-phase6-rows.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const args = { label: null, threadCap: 1, force: false };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    if (flag === "--label") { args.label = value; i += 1; }
    else if (flag === "--thread-cap") {
      const n = Number(value);
      if (!Number.isInteger(n) || n < 1) {
        throw new Error(`--thread-cap must be a positive integer (got "${value}")`);
      }
      args.threadCap = n;
      i += 1;
    }
    else if (flag === "--force") { args.force = true; }
    else if (flag === "--help" || flag === "-h") {
      printHelpAndExit(0);
    }
    else throw new Error(`Unknown flag: ${flag}`);
  }
  if (!args.label) {
    printHelpAndExit(1);
  }
  return args;
}

function printHelpAndExit(code) {
  const labels = ROWS.map((r) => `  ${r.mode === "probe" ? "[probe]" : "[lock] "} ${r.label}`).join("\n");
  process.stderr.write(
    `usage: node scripts/mesa-phase6-shard.mjs --label <label> [--thread-cap <N>] [--force]\n\n` +
    `Known labels:\n${labels}\n`,
  );
  process.exit(code);
}

function run() {
  const { label, threadCap, force } = parseArgs(process.argv.slice(2));
  const row = getRow(label);
  const policyAbs = path.resolve(repoRoot, policyPath(row));

  console.log(`[shard ${label}] mode=${row.mode} lambda=${row.lambda} compose=${row.compose} scale=${row.scale}` + (row.expect ? ` expect=${row.expect}` : ""));
  console.log(`[shard ${label}] thread-cap=${threadCap} (OMP/MKL/OPENBLAS/NUMEXPR/TORCH)`);
  console.log(`[shard ${label}] policy: ${policyPath(row)}`);

  if (!force && existsSync(policyAbs)) {
    console.log(`[shard ${label}] policy exists, skipping (pass --force to re-run)`);
    process.exit(0);
  }

  const args = buildTrainArgs(row, repoRoot);
  const env = buildShardEnv(process.env, threadCap);
  const exec = pythonExec();
  const started = Date.now();

  console.log(`[shard ${label}] starting at ${new Date(started).toISOString()}`);
  console.log(`[shard ${label}] command: ${exec} ${args.join(" ")}`);

  const child = spawn(exec, args, {
    cwd: repoRoot,
    stdio: "inherit",
    env,
  });

  const onSig = (sig) => {
    if (!child.killed) child.kill(sig);
  };
  process.on("SIGINT", onSig);
  process.on("SIGTERM", onSig);

  child.on("exit", (code, signal) => {
    const wall = (Date.now() - started) / 1000;
    if (signal) {
      console.error(`[shard ${label}] killed by ${signal} after ${wall.toFixed(1)} s`);
      process.exit(1);
    }
    console.log(`[shard ${label}] exit=${code ?? 0} wall=${wall.toFixed(1)} s`);
    process.exit(code ?? 0);
  });
}

run();
