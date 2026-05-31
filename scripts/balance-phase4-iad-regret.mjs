// scripts/balance-phase4-iad-regret.mjs
//
// IAD-style regret reducer for the Phase 4 Balance substrate-leg IAD.
// Reads merged bayes-regret.csv (paired (sundog_shadow vs
// bayes_floor_shadow_particle) per-trial regret already computed by
// balance-phase15-bayes-floor.mjs), aggregates over seeds for the target
// cell, computes mean regret + 95% paired-bootstrap CI, and fires one of
// 3 pre-registered branches:
//
//   • Φ-accessible — CI lower > 0 AND mean ≥ 0.5 × Phase-15 reference (0.267)
//   • Privileged-only — CI includes 0
//   • Partial — CI lower > 0 but mean < 0.5 × Phase-15 reference
//
// Output: phase4-balance-regret-summary.csv + verdict to stdout.
//
// Usage:
//   node scripts/balance-phase4-iad-regret.mjs --in <merged-dir>
//        [--cell-id <id>] [--bootstrap-iterations <N>] [--bootstrap-seed <N>]
//
// Defaults:
//   --in (required)
//   --cell-id <auto> — picks the unique cellId present; errors if >1
//   --bootstrap-iterations 2000
//   --bootstrap-seed 40604 (matches three-body IAD seed for reproducibility-style parity)

import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// Phase-15 reference for verdict thresholds. From
// results/balance/phase15-phase10-full-lock/bayes-regret-summary.csv ▸
// near_fall × light_elevation=8: meanRegretVsSundog=0.267 over 100 seeds.
const PHASE15_REFERENCE = Object.freeze({
  cellId: "light_elevation__8__light_8__delay_0__noise_0__drop_0__force_12__rail_2p4__push_4p5__preset_near_fall",
  meanRegretVsSundog: 0.267,
  n: 100,
});

// Partial vs. Φ-accessible threshold: mean must reach this fraction of the
// Phase-15 reference to count as "approaching" the reference.
const ACCESSIBLE_MEAN_FRACTION = 0.5;

function parseArgs(argv) {
  const args = {
    in: null,
    cellId: null,
    bootstrapIterations: 2000,
    bootstrapSeed: 40604,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    if (flag === "--in") { args.in = value; i += 1; }
    else if (flag === "--cell-id") { args.cellId = value; i += 1; }
    else if (flag === "--bootstrap-iterations") {
      args.bootstrapIterations = Number.parseInt(value, 10);
      if (!Number.isInteger(args.bootstrapIterations) || args.bootstrapIterations < 100) throw new Error("--bootstrap-iterations must be ≥ 100");
      i += 1;
    } else if (flag === "--bootstrap-seed") {
      args.bootstrapSeed = Number.parseInt(value, 10);
      if (!Number.isInteger(args.bootstrapSeed) || args.bootstrapSeed < 0) throw new Error("--bootstrap-seed must be a non-negative integer");
      i += 1;
    } else if (flag === "--help" || flag === "-h") {
      process.stderr.write(`usage: node scripts/balance-phase4-iad-regret.mjs --in <merged-dir> [--cell-id <id>] [--bootstrap-iterations <N>] [--bootstrap-seed <N>]\n`);
      process.exit(0);
    } else {
      throw new Error(`Unknown flag: ${flag}`);
    }
  }
  if (!args.in) throw new Error("--in is required (merged dir from balance-phase4-iad-merge.mjs)");
  return args;
}

function parseCsvRows(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return { header: [], rows: [] };
  const header = lines[0].split(",");
  const rows = lines.slice(1).map((line) => {
    const cells = line.split(",");
    const obj = {};
    for (let i = 0; i < header.length; i += 1) obj[header[i]] = cells[i];
    return obj;
  });
  return { header, rows };
}

// Mulberry32 — fast deterministic PRNG, same family as the threebody regret
// reducer's bootstrap (independent implementation since this is a separate
// script).
function makeRng(seed) {
  let s = seed >>> 0;
  return function next() {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function meanOf(values) {
  let s = 0;
  for (const v of values) s += v;
  return s / values.length;
}

function pairedBootstrapCi(perSeedValues, iterations, rng) {
  const n = perSeedValues.length;
  const means = new Float64Array(iterations);
  for (let it = 0; it < iterations; it += 1) {
    let sum = 0;
    for (let i = 0; i < n; i += 1) {
      const idx = Math.floor(rng() * n);
      sum += perSeedValues[idx];
    }
    means[it] = sum / n;
  }
  const sorted = Array.from(means).sort((a, b) => a - b);
  const lo = sorted[Math.floor(iterations * 0.025)];
  const hi = sorted[Math.floor(iterations * 0.975)];
  return { lower: lo, upper: hi };
}

function classifyVerdict({ meanRegret, ciLower, ciUpper, referenceMean }) {
  if (ciLower <= 0) {
    return {
      branch: "PRIVILEGED-ONLY",
      headline: "CI includes 0 → headroom is NOT recoverable from Φ history.",
      detail: "Per the pre-registered Privileged-only branch, no admissible floor can make the off-set arm fire on this cell. Phase-4-decisive negative for this substrate-cell-Φ triple.",
    };
  }
  const ratio = meanRegret / referenceMean;
  if (ratio >= ACCESSIBLE_MEAN_FRACTION) {
    return {
      branch: "Φ-ACCESSIBLE",
      headline: `CI lower > 0 (${ciLower.toFixed(4)}) AND mean (${meanRegret.toFixed(4)}) ≥ ${ACCESSIBLE_MEAN_FRACTION}× Phase-15 reference (${referenceMean}). Recovered ${(ratio * 100).toFixed(1)}% of reference.`,
      detail: "Off-set arm IS satisfiable from Φ history on this substrate-cell-Φ triple. Phase-4 substrate-empirical leg closes positive on Balance.",
    };
  }
  return {
    branch: "PARTIAL",
    headline: `CI lower > 0 (${ciLower.toFixed(4)}) but mean (${meanRegret.toFixed(4)}) < ${ACCESSIBLE_MEAN_FRACTION}× Phase-15 reference (${referenceMean}). Recovered only ${(ratio * 100).toFixed(1)}% of reference.`,
    detail: "Partial Φ-accessibility — quantify the accessible fraction; decide tractable-floor vs. scope on this number.",
  };
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const inAbs = path.resolve(repoRoot, opts.in);

  const bayesRegretText = await readFile(path.join(inAbs, "bayes-regret.csv"), "utf8");
  const { rows } = parseCsvRows(bayesRegretText);
  if (rows.length === 0) throw new Error("merged bayes-regret.csv has no data rows");

  // Filter to the target cell. Auto-pick if not specified and only one cell exists.
  const cellIds = new Set(rows.map((r) => r.cellId));
  let cellId = opts.cellId;
  if (!cellId) {
    if (cellIds.size === 1) {
      cellId = [...cellIds][0];
    } else {
      throw new Error(`multiple cellIds in merged data (${cellIds.size}); pass --cell-id explicitly:\n  ${[...cellIds].slice(0, 5).join("\n  ")}`);
    }
  } else if (!cellIds.has(cellId)) {
    throw new Error(`--cell-id "${cellId}" not present in merged data; available:\n  ${[...cellIds].slice(0, 5).join("\n  ")}`);
  }

  const cellRows = rows.filter((r) => r.cellId === cellId);
  const perSeed = new Map(); // seed -> regretVsSundog
  for (const r of cellRows) {
    const seed = Number.parseInt(r.seed, 10);
    const regret = Number.parseFloat(r.regretVsSundog);
    if (!Number.isInteger(seed) || !Number.isFinite(regret)) continue;
    if (perSeed.has(seed)) {
      throw new Error(`duplicate seed ${seed} for cellId ${cellId} in merged data`);
    }
    perSeed.set(seed, regret);
  }
  if (perSeed.size === 0) throw new Error("no usable (seed, regretVsSundog) data after filtering");

  const seeds = [...perSeed.keys()].sort((a, b) => a - b);
  const values = seeds.map((s) => perSeed.get(s));
  const meanRegret = meanOf(values);

  const rng = makeRng(opts.bootstrapSeed);
  const { lower: ciLower, upper: ciUpper } = pairedBootstrapCi(values, opts.bootstrapIterations, rng);

  const negativeRegretCount = values.filter((v) => v < 0).length;
  const verdict = classifyVerdict({
    meanRegret,
    ciLower,
    ciUpper,
    referenceMean: PHASE15_REFERENCE.meanRegretVsSundog,
  });

  // Write summary CSV
  const summary = [
    ["cellId", "n_seeds", "mean_regret", "ci_lower_95", "ci_upper_95",
     "negative_regret_count", "phase15_reference_mean", "verdict_branch", "bootstrap_iterations", "bootstrap_seed"].join(","),
    [cellId, seeds.length, meanRegret.toFixed(8), ciLower.toFixed(8), ciUpper.toFixed(8),
     negativeRegretCount, PHASE15_REFERENCE.meanRegretVsSundog, verdict.branch,
     opts.bootstrapIterations, opts.bootstrapSeed].join(","),
  ].join("\n") + "\n";
  const summaryPath = path.join(inAbs, "phase4-balance-regret-summary.csv");
  await writeFile(summaryPath, summary, "utf8");

  // Write per-seed CSV
  const perSeedCsv = [
    "seed,regretVsSundog",
    ...seeds.map((s) => `${s},${perSeed.get(s).toFixed(8)}`),
  ].join("\n") + "\n";
  await writeFile(path.join(inAbs, "phase4-balance-regret-per-seed.csv"), perSeedCsv, "utf8");

  // Stdout report (IAD-style)
  console.log("");
  console.log("─── Balance IAD: Read-Back ────────────────────────────────────");
  console.log(`cellId: ${cellId}`);
  console.log(`seeds: n=${seeds.length} (${seeds.join(",")})`);
  console.log("");
  console.log("Per-seed regret (bayes_floor_shadow_particle − sundog_shadow, normalized survival):");
  for (const s of seeds) {
    const r = perSeed.get(s);
    const sign = r >= 0 ? "+" : "";
    const note = r < 0 ? "  ← sig wins" : r > 0.1 ? "  ← bayes wins" : "";
    console.log(`  seed=${String(s).padStart(2)}  ${sign}${r.toFixed(4)}${note}`);
  }
  console.log("");
  console.log(`Mean regret:           ${meanRegret.toFixed(4)}`);
  console.log(`95% bootstrap CI:      [${ciLower.toFixed(4)}, ${ciUpper.toFixed(4)}]   (${opts.bootstrapIterations} iterations, seed ${opts.bootstrapSeed})`);
  console.log(`Negative-regret rate:  ${negativeRegretCount}/${seeds.length}`);
  console.log(`Phase-15 reference:    ${PHASE15_REFERENCE.meanRegretVsSundog}   (mean over ${PHASE15_REFERENCE.n} seeds in the full lock)`);
  console.log(`% reference recovered: ${(meanRegret / PHASE15_REFERENCE.meanRegretVsSundog * 100).toFixed(1)}%`);
  console.log("");
  console.log("─── Verdict ────────────────────────────────────────────────────");
  console.log(`BRANCH: ${verdict.branch}`);
  console.log(verdict.headline);
  console.log("");
  console.log(verdict.detail);
  console.log("");
  console.log(`Wrote: ${path.relative(repoRoot, summaryPath)}`);
  console.log(`Wrote: ${path.relative(repoRoot, path.join(inAbs, "phase4-balance-regret-per-seed.csv"))}`);
}

main().catch((err) => {
  console.error(`[balance-iad-regret] fatal: ${err.stack || err.message}`);
  process.exit(2);
});
