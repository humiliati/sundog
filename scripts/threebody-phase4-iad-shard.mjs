// scripts/threebody-phase4-iad-shard.mjs
//
// Single-seed shard of the Phase 4 BF-4b Information-Accessibility Diagnostic
// per docs/proof/PHASE4_BAYESIAN_FLOOR_BUILDOUT.md ▸ §IA Diagnostic.
//
// The spec command runs `--seeds 8` in one invocation (sequential single-
// threaded). This wrapper makes one trial per process so an orchestrator can
// fan out across seeds in parallel — same architectural pattern as
// scripts/threebody-phase15-shard.mjs (3×4 mass×velocity grid sharded into
// 12 parallel wrappers).
//
// Usage:
//   node scripts/threebody-phase4-iad-shard.mjs --seed <S> [--particles <N>] [--out-root <dir>] [--force]
//
// Output: <out-root>/_bf4b-accessibility-shard-seed_<S>/
// Default --out-root: results/proof/phase4
// Default --particles: 512 (spec gold)
//
// Resume-safe: skips run if manifest.json exists in the shard dir. Pass
// --force to re-run.
//
// All other IAD params are PINNED to the spec command and not exposed.

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// Pinned IAD args per PHASE4_BAYESIAN_FLOOR_BUILDOUT.md ▸ §IA Diagnostic spec
// command. Particle count is exposed as a knob (least-sensitive to verdict per
// spec); seed-start is the shard key. Everything else is pinned.
const PINNED_ARGS = Object.freeze({
  regimes: "near_escape",
  massRatios: "1",
  timesteps: "0.01",
  radiusScales: "1.075",
  velocityScales: "1.15",
  thrustLimits: "0.4",
  sensorNoiseSweep: "0.01",
  trackGuardMode: "hazard_quantile",
  trackGuardQuantile: "0.75",
  trackGuardMinRadiusSweep: "1.15",
  trackGuardMaxLocalAccelerationSweep: "2.5",
  trackGuardMaxTidalMagnitudeSweep: "35",
  duration: "16",
  planningHorizonSteps: "800",
  candidateHoldSteps: "800",
  resampleThreshold: "0.5",
  shapeFraction: "0.5",
  signatureAdvantageDtMultiplier: "1",
});

function parseArgs(argv) {
  const args = { seed: null, particles: 512, outRoot: "results/proof/phase4", force: false };
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
      if (!Number.isInteger(n) || n < 1) throw new Error(`--particles must be a positive integer (got "${value}")`);
      args.particles = n; i += 1;
    } else if (flag === "--out-root") {
      args.outRoot = value; i += 1;
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
    `usage: node scripts/threebody-phase4-iad-shard.mjs --seed <S> [--particles <N>] [--out-root <dir>] [--force]\n` +
    `  --seed <S>       seed (and seed-start) for this single-trial shard (required)\n` +
    `  --particles <N>  particle count (default 512 = spec gold)\n` +
    `  --out-root <d>   results dir (default results/proof/phase4)\n` +
    `  --force          re-run even if manifest.json exists in shard dir\n`,
  );
  process.exit(code);
}

function buildArgs({ seed, particles, outAbs }) {
  return [
    "scripts/threebody-phase4-bayes-floor.mjs",
    "--phase", `phase4-bf4b-iad-shard-seed_${seed}`,
    "--out", outAbs,
    "--regimes", PINNED_ARGS.regimes,
    "--mass-ratios", PINNED_ARGS.massRatios,
    "--timesteps", PINNED_ARGS.timesteps,
    "--radius-scales", PINNED_ARGS.radiusScales,
    "--velocity-scales", PINNED_ARGS.velocityScales,
    "--thrust-limits", PINNED_ARGS.thrustLimits,
    "--sensor-noise-sweep", PINNED_ARGS.sensorNoiseSweep,
    "--track-guard-mode", PINNED_ARGS.trackGuardMode,
    "--track-guard-quantile", PINNED_ARGS.trackGuardQuantile,
    "--track-guard-min-radius-sweep", PINNED_ARGS.trackGuardMinRadiusSweep,
    "--track-guard-max-local-acceleration-sweep", PINNED_ARGS.trackGuardMaxLocalAccelerationSweep,
    "--track-guard-max-tidal-magnitude-sweep", PINNED_ARGS.trackGuardMaxTidalMagnitudeSweep,
    "--seed-start", String(seed),
    "--seeds", "1",
    "--duration", PINNED_ARGS.duration,
    "--particle-count", String(particles),
    "--planning-horizon-steps", PINNED_ARGS.planningHorizonSteps,
    "--candidate-hold-steps", PINNED_ARGS.candidateHoldSteps,
    "--resample-threshold", PINNED_ARGS.resampleThreshold,
    "--shape-fraction", PINNED_ARGS.shapeFraction,
    "--signature-advantage-dt-multiplier", PINNED_ARGS.signatureAdvantageDtMultiplier,
  ];
}

export function shardDir(outRoot, seed) {
  return path.join(outRoot, `_bf4b-accessibility-shard-seed_${seed}`);
}

export function shardManifestPath(outRoot, seed) {
  return path.join(shardDir(outRoot, seed), "manifest.json");
}

function run() {
  const opts = parseArgs(process.argv.slice(2));
  const outRel = shardDir(opts.outRoot, opts.seed);
  const outAbs = path.resolve(repoRoot, outRel);
  const manifestAbs = path.resolve(repoRoot, shardManifestPath(opts.outRoot, opts.seed));

  console.log(`[iad-shard seed=${opts.seed}] particles=${opts.particles} out=${outRel}`);

  if (!opts.force && existsSync(manifestAbs)) {
    console.log(`[iad-shard seed=${opts.seed}] manifest exists, skipping (pass --force to re-run)`);
    process.exit(0);
  }

  const args = buildArgs({ seed: opts.seed, particles: opts.particles, outAbs });
  const started = Date.now();
  console.log(`[iad-shard seed=${opts.seed}] starting at ${new Date(started).toISOString()}`);

  const child = spawn(process.execPath, args, {
    cwd: repoRoot,
    stdio: "inherit",
    env: { ...process.env, NODE_NO_WARNINGS: "1" },
  });

  // Single-child wrapper: per-shard signal forwarding is fine here (no listener leak).
  const onSig = (sig) => () => { if (!child.killed) child.kill(sig); };
  process.once("SIGINT", onSig("SIGINT"));
  process.once("SIGTERM", onSig("SIGTERM"));

  child.on("exit", (code, signal) => {
    const wall = (Date.now() - started) / 1000;
    if (signal) {
      console.error(`[iad-shard seed=${opts.seed}] killed by ${signal} after ${wall.toFixed(1)} s`);
      process.exit(1);
    }
    console.log(`[iad-shard seed=${opts.seed}] exit=${code ?? 0} wall=${wall.toFixed(1)} s`);
    process.exit(code ?? 0);
  });
}

// ESM `import.meta.url` check to allow this file to be imported (for shardDir/shardManifestPath)
// without auto-running.
const __filename = fileURLToPath(import.meta.url);
if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
  run();
}
