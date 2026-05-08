import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  normalizeConfig,
  runTrial,
  seededInitialParticle,
} from "../public/js/threebody-core.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function parseArgs(argv) {
  const args = {
    out: "results/threebody/phase5-smoke",
    seedStart: 0,
    seeds: 3,
    regimes: ["stable", "near_escape", "near_collision", "chaotic"],
    modes: ["off"],
    duration: 8,
    dt: 0.01,
    logEvery: 20,
    massRatio: 1,
    thrustLimit: 0.5,
    targetTidal: 2,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    i += 1;

    if (flag === "--out") args.out = value;
    else if (flag === "--seed-start") args.seedStart = Number.parseInt(value, 10);
    else if (flag === "--seeds") args.seeds = Number.parseInt(value, 10);
    else if (flag === "--regimes") args.regimes = parseList(value);
    else if (flag === "--modes") args.modes = parseList(value);
    else if (flag === "--duration") args.duration = Number.parseFloat(value);
    else if (flag === "--dt") args.dt = Number.parseFloat(value);
    else if (flag === "--log-every") args.logEvery = Number.parseInt(value, 10);
    else if (flag === "--mass-ratio") args.massRatio = Number.parseFloat(value);
    else if (flag === "--thrust-limit") args.thrustLimit = Number.parseFloat(value);
    else if (flag === "--target-tidal") args.targetTidal = Number.parseFloat(value);
    else throw new Error(`Unknown flag: ${flag}`);
  }

  if (!Number.isInteger(args.seeds) || args.seeds < 1) {
    throw new Error("--seeds must be a positive integer");
  }
  if (!Number.isFinite(args.duration) || args.duration <= 0) {
    throw new Error("--duration must be positive");
  }
  if (!Number.isFinite(args.dt) || args.dt <= 0) {
    throw new Error("--dt must be positive");
  }
  return args;
}

function trialId({ regime, mode, seed }) {
  return `${regime}_${mode}_seed_${String(seed).padStart(3, "0")}`;
}

function makeTrialConfig(args, seed, regime, mode) {
  return normalizeConfig({
    seed,
    regime,
    controllerMode: mode,
    duration: args.duration,
    dt: args.dt,
    logEvery: args.logEvery,
    massRatio: args.massRatio,
    thrustLimit: args.thrustLimit,
    targetTidal: args.targetTidal,
    initialParticle: seededInitialParticle(seed, regime),
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(repoRoot, args.out);
  const trialDir = path.join(outDir, "trials");
  await mkdir(trialDir, { recursive: true });

  const startedAt = new Date().toISOString();
  const manifest = {
    schema: "sundog.threebody.phase5.v1",
    startedAt,
    purpose: "Phase 5 deterministic harness smoke run; not a benchmark claim.",
    harness: {
      script: "scripts/threebody-harness.mjs",
      core: "public/js/threebody-core.mjs",
    },
    args,
    trials: [],
  };

  for (const regime of args.regimes) {
    for (const mode of args.modes) {
      for (let offset = 0; offset < args.seeds; offset += 1) {
        const seed = args.seedStart + offset;
        const config = makeTrialConfig(args, seed, regime, mode);
        const id = trialId({ regime, mode, seed });
        const trial = runTrial(config);
        const relativePath = `trials/${id}.jsonl`;
        const logPath = path.join(outDir, relativePath);
        const lines = trial.records.map((record) => JSON.stringify(record)).join("\n");
        await writeFile(logPath, `${lines}\n`, "utf8");
        manifest.trials.push({
          id,
          seed,
          regime,
          controllerMode: mode,
          initialParticle: config.initialParticle,
          config: {
            massRatio: config.massRatio,
            masses: config.masses,
            separation: config.separation,
            dt: config.dt,
            duration: config.duration,
            logEvery: config.logEvery,
            thrustLimit: config.thrustLimit,
            targetTidal: config.targetTidal,
          },
          log: relativePath,
          summary: trial.summary,
        });
      }
    }
  }

  manifest.completedAt = new Date().toISOString();
  await writeFile(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

  const outcomeCounts = manifest.trials.reduce((counts, trial) => {
    counts[trial.summary.terminalOutcome] = (counts[trial.summary.terminalOutcome] ?? 0) + 1;
    return counts;
  }, {});

  console.log(`[threebody] wrote ${manifest.trials.length} trials to ${path.relative(repoRoot, outDir)}`);
  console.log(`[threebody] outcomes ${JSON.stringify(outcomeCounts)}`);
}

main().catch((error) => {
  console.error(`[threebody] ${error.message}`);
  process.exitCode = 1;
});
