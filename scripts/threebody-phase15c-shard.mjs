// scripts/threebody-phase15c-shard.mjs
//
// Parameterized wrapper for one Phase 15C full-lock shard, per
// PHASE15C_SPEC.md §3 (12-shard partition by mass-ratio × velocityScale,
// each keeping all 6 modes + 3 radii + 8 seeds intact).
//
// Usage:
//   node scripts/threebody-phase15c-shard.mjs --mass-ratio <X> --velocity-scale <Y>
//   (or via npm:  npm run threebody:phase15c:multistep -- --mass-ratio 1 --velocity-scale 0.95)
//
// Pinned values:
//   <X> ∈ {0.01, 0.3, 1}; <Y> ∈ {0.95, 1.05, 1.1, 1.15}
//
// Per-shard: 6 modes × 1 dt × 3 radii × 8 seeds = 144 trials.
// Lock total: 12 shards × 144 = 1,728 trials.
//
// All other flags are pinned to the Phase 15B lock slate plus --multi-step-audit 1,
// per PHASE15C_SPEC.md §1 Decision Lock.

import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const ALLOWED_MASS_RATIOS = new Set(["0.01", "0.3", "1"]);
const ALLOWED_VELOCITY_SCALES = new Set(["0.95", "1.05", "1.1", "1.15"]);

function parseArgs(argv) {
  const args = { massRatio: null, velocityScale: null };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    if (flag === "--mass-ratio") { args.massRatio = value; i += 1; }
    else if (flag === "--velocity-scale") { args.velocityScale = value; i += 1; }
    else throw new Error(`Unknown flag: ${flag}`);
  }
  if (!args.massRatio || !args.velocityScale) {
    throw new Error("Required: --mass-ratio <X> --velocity-scale <Y> (X ∈ {0.01,0.3,1}; Y ∈ {0.95,1.05,1.1,1.15})");
  }
  if (!ALLOWED_MASS_RATIOS.has(args.massRatio)) {
    throw new Error(`--mass-ratio must be one of ${[...ALLOWED_MASS_RATIOS].join(", ")} (got "${args.massRatio}"); per PHASE15C_SPEC.md §1`);
  }
  if (!ALLOWED_VELOCITY_SCALES.has(args.velocityScale)) {
    throw new Error(`--velocity-scale must be one of ${[...ALLOWED_VELOCITY_SCALES].join(", ")} (got "${args.velocityScale}"); per PHASE15C_SPEC.md §1`);
  }
  return args;
}

function naming(value) {
  // Match the harness's caseId convention: dots become 'p'.
  return String(value).replaceAll(".", "p");
}

function buildHarnessArgs(massRatio, velocityScale) {
  const tag = `mu${naming(massRatio)}-v${naming(velocityScale)}`;
  return [
    "scripts/threebody-operating-envelope.mjs",
    "--phase", `phase15c-multistep-counterfactual-shard-${tag}`,
    "--out",   `results/threebody/phase15c-shard-${tag}`,
    "--regimes", "near_escape",
    "--modes",
    "off,track_sensor_accel_guarded,track_sensor_accel_signal_delay,track_sensor_accel_signal_shuffle,track_sensor_accel_action_shuffle,track_sensor_accel_sign_flip",
    "--mass-ratios", String(massRatio),
    "--velocity-scales", String(velocityScale),
    "--timesteps", "0.004",
    "--radius-scales", "1.025,1.05,1.075",
    "--thrust-limits", "0.4",
    "--sensor-noise-sweep", "0",
    "--track-guard-mode", "hazard_quantile",
    "--track-guard-quantile", "0.75",
    "--track-guard-min-radius-sweep", "1.15",
    "--track-guard-max-local-acceleration-sweep", "2.5",
    "--track-guard-max-tidal-magnitude-sweep", "35",
    "--seeds", "8",
    "--duration", "16",
    "--sensor-audit-every", "240",
    "--track-action-coupling", "1",
    "--precision-receipts", "1",
    "--counterfactual-audit", "1",
    "--counterfactual-normalizer-floor", "1e-9",
    "--multi-step-audit", "1",
  ];
}

function run() {
  const { massRatio, velocityScale } = parseArgs(process.argv.slice(2));
  const tag = `mu${naming(massRatio)}-v${naming(velocityScale)}`;
  const harnessArgs = buildHarnessArgs(massRatio, velocityScale);

  console.log(`[shard ${tag}] mass-ratio=${massRatio} velocity-scale=${velocityScale}`);
  console.log(`[shard ${tag}] out: results/threebody/phase15c-shard-${tag}`);
  console.log(`[shard ${tag}] expected: 6 modes × 1 dt × 3 radii × 8 seeds = 144 trials`);
  console.log(`[shard ${tag}] starting at ${new Date().toISOString()}`);

  const child = spawn(process.execPath, harnessArgs, {
    cwd: repoRoot,
    stdio: "inherit",
  });
  child.on("exit", (code, signal) => {
    if (signal) {
      console.error(`[shard ${tag}] killed by ${signal}`);
      process.exit(1);
    }
    process.exit(code ?? 0);
  });
}

run();
