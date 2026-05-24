// scripts/threebody-phase15-merge.mjs
//
// Merge per-shard Phase 15 results into a unified
// results/threebody/phase15-forward-oracle-precision-lock/ as if a single
// 12,960-trial run produced it. Per PHASE15_SPEC.md §7.
//
// Inputs: results/threebody/phase15-shard-mu<X>-v<Y>/{trials-minimal.jsonl,manifest.json}
// for the 12 shards in the locked partition (3 mass-ratios × 4 velocities).
// Outputs: aggregate CSVs in the merged dir, plus a manifest-summary.json.
//
// Discipline: reconstructs the in-memory `trials` array from per-shard
// trials-minimal.jsonl and runs the EXACT same aggregation pipeline as
// scripts/threebody-operating-envelope.mjs main() does for a single run, so
// the merged aggregate CSVs are byte-identical to what a single-run full lock
// would produce (the binding shard-equivalence gate criterion).

import { mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  makeBestByCellRows,
  makeCellMatrixRows,
  makePairedRows,
  makePrecisionMapRows,
  makeRichardsonOrderRows,
  makeTrialOutcomeRows,
  makeWarningQualityRows,
  rowsToCsv,
  summarizeRows,
} from "./threebody-operating-envelope.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const args = {
    shardsDir: "results/threebody",
    shardPattern: /^phase15-shard-mu[0-9p.]+-v[0-9p.]+$/,
    shardDirs: null, // if set, overrides the glob and uses these dirs exactly
    out: "results/threebody/phase15-forward-oracle-precision-lock",
    strict: true, // error on duplicate trial ids or arg mismatches across shards
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    if (flag === "--shards-dir") { args.shardsDir = value; i += 1; }
    else if (flag === "--shard-dirs") {
      args.shardDirs = value.split(",").map((s) => s.trim()).filter(Boolean);
      i += 1;
    }
    else if (flag === "--out") { args.out = value; i += 1; }
    else if (flag === "--no-strict") { args.strict = false; }
    else throw new Error(`Unknown flag: ${flag}`);
  }
  return args;
}

async function findShardDirs(shardsDir, pattern) {
  const root = path.resolve(repoRoot, shardsDir);
  let entries;
  try { entries = await readdir(root); }
  catch (e) { throw new Error(`Cannot read shards dir ${root}: ${e.message}`); }
  const shards = [];
  for (const name of entries) {
    if (!pattern.test(name)) continue;
    const full = path.join(root, name);
    const s = await stat(full);
    if (s.isDirectory()) shards.push(full);
  }
  shards.sort();
  return shards;
}

async function readJsonl(file) {
  const text = await readFile(file, "utf8");
  return text.split("\n").filter((line) => line.length > 0).map((line) => JSON.parse(line));
}

async function readShard(shardDir) {
  const manifestPath = path.join(shardDir, "manifest.json");
  const trialsPath = path.join(shardDir, "trials-minimal.jsonl");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  const trials = await readJsonl(trialsPath);
  return { shardDir, manifest, trials };
}

function assertConsistentArgs(shards) {
  // Sanity: every shard must have been run with the same canonical Phase 15
  // args, except --mass-ratios / --velocity-scales (which differ per shard by
  // design). Any other axis differing would corrupt the merge.
  if (shards.length === 0) return;
  const ref = shards[0].manifest.args;
  // NOTE: massRatios and velocityScales are intentionally EXCLUDED from this
  // consistency check — they are the per-shard partition axes (each shard has
  // a different value on those axes by design). All OTHER axes must match.
  const keysToCheck = [
    "regimes", "modes", "duration", "timesteps", "radiusScales",
    "thrustLimits", "sensorNoiseSweep", "trackGuardMode",
    "trackGuardQuantiles", "trackGuardMinRadiusSweep",
    "trackGuardMaxLocalAccelerationSweep",
    "trackGuardMaxTidalMagnitudeSweep",
    "candidateMaxWorsenedRate", "candidateMinSurvivalDelta",
    "seeds", "seedStart", "trackActionCoupling", "precisionReceipts",
  ];
  for (const sh of shards.slice(1)) {
    for (const k of keysToCheck) {
      const a = JSON.stringify(ref[k]);
      const b = JSON.stringify(sh.manifest.args[k]);
      if (a !== b) {
        throw new Error(
          `Shard arg mismatch on '${k}' between ${shards[0].shardDir} (${a}) and ${sh.shardDir} (${b}). Merge would corrupt aggregates.`,
        );
      }
    }
  }
}

function assertNoDuplicateTrials(allTrials) {
  // Each (caseId, regime, controllerMode, seed) must appear at most once.
  const seen = new Map();
  for (const t of allTrials) {
    const key = `${t.caseId}\t${t.regime}\t${t.controllerMode}\t${t.seed}`;
    if (seen.has(key)) {
      throw new Error(`Duplicate trial across shards: ${key} (seen in ${seen.get(key)} and again)`);
    }
    seen.set(key, t.caseId);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(repoRoot, args.out);
  await mkdir(outDir, { recursive: true });

  let shardDirs;
  if (args.shardDirs) {
    // Explicit list overrides the glob (used by the equivalence gate to merge
    // smoke-shard dirs whose names don't fit the full-lock pattern).
    shardDirs = args.shardDirs.map((d) => path.resolve(repoRoot, d));
    for (const d of shardDirs) {
      const s = await stat(d).catch(() => null);
      if (!s || !s.isDirectory()) throw new Error(`--shard-dirs entry is not a directory: ${d}`);
    }
  } else {
    shardDirs = await findShardDirs(args.shardsDir, args.shardPattern);
  }
  if (shardDirs.length === 0) {
    throw new Error(args.shardDirs
      ? "--shard-dirs was empty"
      : `No shard dirs matched ${args.shardPattern} under ${args.shardsDir}`);
  }
  console.log(`[merge] found ${shardDirs.length} shard dir(s):`);
  for (const d of shardDirs) console.log(`[merge]   - ${path.relative(repoRoot, d)}`);

  const shards = await Promise.all(shardDirs.map(readShard));
  if (args.strict) assertConsistentArgs(shards);

  // Build the unified trials array (same shape as scripts/threebody-operating-envelope.mjs
  // builds in main(), minus eventHistory which the aggregation functions do not
  // consume — they read trial.summary, trial.sensorAudit, trial.earlyTrajectory).
  const trials = [];
  for (const sh of shards) {
    for (const t of sh.trials) trials.push(t);
  }
  if (args.strict) assertNoDuplicateTrials(trials);
  console.log(`[merge] unified trial count: ${trials.length}`);

  // Use the first shard's args for canonical thresholds (assertConsistentArgs
  // already verified all 12 agree on these).
  const mergeArgs = shards[0].manifest.args;

  // Run the EXACT same pipeline scripts/threebody-operating-envelope.mjs uses:
  const pairedRows = makePairedRows(trials);
  const trialOutcomeRows = makeTrialOutcomeRows(pairedRows);
  const envelopeRows = summarizeRows(pairedRows, mergeArgs, true);
  const aggregateRows = summarizeRows(pairedRows, mergeArgs, false);
  const bestByCellRows = makeBestByCellRows(envelopeRows);
  const cellClassMapRows = makeCellMatrixRows(bestByCellRows, "bestRegionClass");
  const cellDeltaMapRows = makeCellMatrixRows(bestByCellRows, "bestSurvivalDeltaVsPassive");
  const candidateRows = envelopeRows.filter((row) => row.candidateEnvelope);
  const cellPrecisionMapRows = mergeArgs.precisionReceipts ? makePrecisionMapRows(envelopeRows) : null;
  const richardsonOrderRows = mergeArgs.precisionReceipts ? makeRichardsonOrderRows(trials) : null;

  let cellWarningQualityMapRows = null;
  if (mergeArgs.trackActionCoupling || mergeArgs.precisionReceipts) {
    const warningRows = makeWarningQualityRows(envelopeRows, mergeArgs.precisionReceipts);
    cellWarningQualityMapRows = makeCellMatrixRows(warningRows, "meanPassiveWarningAuroc");
  }

  // Unified aggregate writes (same filenames + same write order as single-run main()):
  await writeFile(path.join(outDir, "trial-outcomes.csv"), rowsToCsv(trialOutcomeRows), "utf8");
  await writeFile(path.join(outDir, "paired.csv"), rowsToCsv(pairedRows), "utf8");
  await writeFile(path.join(outDir, "envelope-map.csv"), rowsToCsv(envelopeRows), "utf8");
  await writeFile(path.join(outDir, "aggregate-envelope.csv"), rowsToCsv(aggregateRows), "utf8");
  await writeFile(path.join(outDir, "best-by-cell.csv"), rowsToCsv(bestByCellRows), "utf8");
  await writeFile(path.join(outDir, "cell-class-map.csv"), rowsToCsv(cellClassMapRows), "utf8");
  await writeFile(path.join(outDir, "cell-delta-map.csv"), rowsToCsv(cellDeltaMapRows), "utf8");
  if (cellWarningQualityMapRows) {
    await writeFile(path.join(outDir, "cell-warning-quality-map.csv"), rowsToCsv(cellWarningQualityMapRows), "utf8");
  }
  if (cellPrecisionMapRows) {
    await writeFile(path.join(outDir, "cell-precision-map.csv"), rowsToCsv(cellPrecisionMapRows), "utf8");
  }
  if (richardsonOrderRows) {
    await writeFile(path.join(outDir, "richardson-order-map.csv"), rowsToCsv(richardsonOrderRows), "utf8");
  }
  await writeFile(
    path.join(outDir, "candidate-envelope.csv"),
    rowsToCsv(candidateRows, envelopeRows.length > 0 ? Object.keys(envelopeRows[0]) : []),
    "utf8",
  );

  // Unified manifest summary: names the 12 per-shard sources.
  const manifestSummary = {
    schema: "sundog.threebody.phase15.merge.v1",
    mergedAt: new Date().toISOString(),
    out: path.relative(repoRoot, outDir),
    shardCount: shards.length,
    unifiedTrialCount: trials.length,
    shards: shards.map((sh) => ({
      dir: path.relative(repoRoot, sh.shardDir),
      phase: sh.manifest.args.phase,
      trials: sh.trials.length,
      startedAt: sh.manifest.startedAt,
      completedAt: sh.manifest.completedAt,
      massRatios: sh.manifest.args.massRatios,
      velocityScales: sh.manifest.args.velocityScales,
    })),
    args: mergeArgs,
  };
  await writeFile(
    path.join(outDir, "manifest-summary.json"),
    `${JSON.stringify(manifestSummary, null, 2)}\n`,
    "utf8",
  );

  const outcomeCounts = trials.reduce((counts, t) => {
    counts[t.summary.terminalOutcome] = (counts[t.summary.terminalOutcome] ?? 0) + 1;
    return counts;
  }, {});
  console.log(`[merge] wrote unified aggregates to ${path.relative(repoRoot, outDir)}`);
  console.log(`[merge] unified candidate envelope rows ${candidateRows.length}/${envelopeRows.length}`);
  console.log(`[merge] unified outcomes ${JSON.stringify(outcomeCounts)}`);
}

main().catch((error) => {
  console.error(`[merge] ${error.message}`);
  process.exitCode = 1;
});
