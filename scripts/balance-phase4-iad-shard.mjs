// scripts/balance-phase4-iad-shard.mjs
//
// Single-seed shard of the Phase 4 Balance substrate-leg IAD (Information-
// Accessibility Diagnostic), mirroring the three-body IAD pattern at
// docs/proof/PHASE4_BAYESIAN_FLOOR_BUILDOUT.md ▸ §IA Diagnostic but applied
// to the balance/cart-pole substrate.
//
// Target cell: near_fall × light_elevation=8 (borderline cellClass,
// claim-gate pass, meanRegretVsSundog = 0.267 in the Phase 15 full lock).
// The cell is loaded via --cell-slate phase10-output from a single-row
// envelope.csv at <out-root>/_iad-cell/envelope.csv (written once by the
// orchestrator).
//
// Usage:
//   node scripts/balance-phase4-iad-shard.mjs --seed <S> [--particles <N>]
//        [--horizon-seconds <X>] [--out-root <dir>] [--cell-dir <dir>] [--force]
//
// Output: <out-root>/_iad-shard-seed_<S>/
// Default --out-root: results/balance/phase4-iad
// Default --particles: 512 (spec gold)
// Default --horizon-seconds: 0.5 (10× smoke baseline 0.05)
//
// Resume-safe: skips if manifest.json exists in the shard dir. --force overrides.

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// Pinned Balance IAD args. The cell is loaded from the slate dir; the
// observation-noise/delay/dropout axes default to baseline in that cell row.
const PINNED_ARGS = Object.freeze({
  preset: "near_fall",
  modes: "naive_shadow,sundog_shadow,bayes_floor_shadow_particle,oracle",
  duration: "8",
  cellSlate: "phase10-output",
});

function parseArgs(argv) {
  const args = {
    seed: null,
    particles: 512,
    horizonSeconds: 0.5,
    outRoot: "results/balance/phase4-iad",
    cellDir: "results/balance/phase4-iad/_iad-cell",
    force: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    if (flag === "--seed") {
      const n = Number.parseInt(value, 10);
      if (!Number.isInteger(n) || n < 0) throw new Error(`--seed must be a non-negative integer (got "${value}")`);
      args.seed = n; i += 1;
    } else if (flag === "--particles") {
      const n = Number.parseInt(value, 10);
      if (!Number.isInteger(n) || n < 31) throw new Error(`--particles must be ≥ 31 (got "${value}")`);
      args.particles = n; i += 1;
    } else if (flag === "--horizon-seconds") {
      const n = Number.parseFloat(value);
      if (!Number.isFinite(n) || n <= 0) throw new Error(`--horizon-seconds must be positive (got "${value}")`);
      args.horizonSeconds = n; i += 1;
    } else if (flag === "--out-root") {
      args.outRoot = value; i += 1;
    } else if (flag === "--cell-dir") {
      args.cellDir = value; i += 1;
    } else if (flag === "--force") {
      args.force = true;
    } else if (flag === "--help" || flag === "-h") {
      printHelpAndExit(0);
    } else {
      throw new Error(`Unknown flag: ${flag}`);
    }
  }
  if (args.seed === null) printHelpAndExit(1);
  return args;
}

function printHelpAndExit(code) {
  process.stderr.write(
    `usage: node scripts/balance-phase4-iad-shard.mjs --seed <S> [--particles <N>] [--horizon-seconds <X>] [--out-root <dir>] [--cell-dir <dir>] [--force]\n`,
  );
  process.exit(code);
}

function buildArgs({ seed, particles, horizonSeconds, cellDir, outAbs }) {
  return [
    "scripts/balance-phase15-bayes-floor.mjs",
    "--phase", `phase4-iad-shard-seed_${seed}`,
    "--out", outAbs,
    "--presets", PINNED_ARGS.preset,
    "--modes", PINNED_ARGS.modes,
    "--seeds", "1",
    "--seed-start", String(seed),
    "--duration", PINNED_ARGS.duration,
    "--particle-count", String(particles),
    "--horizon-seconds", String(horizonSeconds),
    "--cell-slate", PINNED_ARGS.cellSlate,
    "--phase10-out", cellDir,
  ];
}

export function shardDir(outRoot, seed) {
  return path.join(outRoot, `_iad-shard-seed_${seed}`);
}

export function shardManifestPath(outRoot, seed) {
  return path.join(shardDir(outRoot, seed), "manifest.json");
}

function run() {
  const opts = parseArgs(process.argv.slice(2));
  const outRel = shardDir(opts.outRoot, opts.seed);
  const outAbs = path.resolve(repoRoot, outRel);
  const manifestAbs = path.resolve(repoRoot, shardManifestPath(opts.outRoot, opts.seed));
  const cellDirAbs = path.resolve(repoRoot, opts.cellDir);

  console.log(`[balance-iad-shard seed=${opts.seed}] particles=${opts.particles} horizon=${opts.horizonSeconds}s out=${outRel}`);

  if (!existsSync(path.join(cellDirAbs, "envelope.csv"))) {
    console.error(`[balance-iad-shard seed=${opts.seed}] cell slate envelope.csv missing at ${opts.cellDir}/envelope.csv — orchestrator must write it first, or invoke balance-phase4-iad-concurrent.mjs to set it up`);
    process.exit(2);
  }

  if (!opts.force && existsSync(manifestAbs)) {
    console.log(`[balance-iad-shard seed=${opts.seed}] manifest exists, skipping (pass --force to re-run)`);
    process.exit(0);
  }

  const args = buildArgs({
    seed: opts.seed,
    particles: opts.particles,
    horizonSeconds: opts.horizonSeconds,
    cellDir: cellDirAbs,
    outAbs,
  });
  const started = Date.now();
  console.log(`[balance-iad-shard seed=${opts.seed}] starting at ${new Date(started).toISOString()}`);

  const child = spawn(process.execPath, args, {
    cwd: repoRoot,
    stdio: "inherit",
    env: { ...process.env, NODE_NO_WARNINGS: "1" },
  });

  const onSig = (sig) => () => { if (!child.killed) child.kill(sig); };
  process.once("SIGINT", onSig("SIGINT"));
  process.once("SIGTERM", onSig("SIGTERM"));

  child.on("exit", (code, signal) => {
    const wall = (Date.now() - started) / 1000;
    if (signal) {
      console.error(`[balance-iad-shard seed=${opts.seed}] killed by ${signal} after ${wall.toFixed(1)} s`);
      process.exit(1);
    }
    console.log(`[balance-iad-shard seed=${opts.seed}] exit=${code ?? 0} wall=${wall.toFixed(1)} s`);
    process.exit(code ?? 0);
  });
}

const __filename = fileURLToPath(import.meta.url);
if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
  run();
}
