// scripts/threebody-phase4-iad-sat-slate.mjs
//
// Build a signature-side slate dir for the regret reducer
// (`threebody-phase4-regret.mjs --signature-in <out> --anchor-slate <out>`)
// composed from a base satisfiability dir plus zero-or-more single-seed
// satisfiability shard dirs, filtered to a final slate of seeds.
//
// Use case: the IAD lock's primary slate {0..7} gets a substitute seed (e.g.
// seed=8) when seed=2 stalls. The regret reducer's strict (cell, seed) slate
// match means the sig side must mirror the final IAD slate exactly. This
// script produces that mirror in a separate dir, leaving the base dir intact.
//
// Inputs: each input dir must contain `trial-outcomes.csv` with a `seed`
// column. Headers must match across inputs.
//
// Usage:
//   node scripts/threebody-phase4-iad-sat-slate.mjs --keep-seeds <list>
//        [--base <dir>] [--shards <comma-list-of-dirs>] --out <dir>
//
// Defaults:
//   --base   results/proof/phase4/_bf4b-satisfiability
//   --shards (empty)
//   --keep-seeds REQUIRED — explicit list, e.g. "0,1,3,4,5,6,7,8"
//   --out    REQUIRED

import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const args = {
    base: "results/proof/phase4/_bf4b-satisfiability",
    shards: [],
    keepSeeds: null,
    out: null,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    if (flag === "--base") { args.base = value; i += 1; }
    else if (flag === "--shards") {
      args.shards = value.split(",").map((s) => s.trim()).filter(Boolean);
      i += 1;
    } else if (flag === "--keep-seeds") {
      args.keepSeeds = value.split(",").map((s) => Number.parseInt(s.trim(), 10));
      if (args.keepSeeds.some((n) => !Number.isInteger(n) || n < 0)) throw new Error(`bad --keep-seeds value "${value}"`);
      i += 1;
    } else if (flag === "--out") { args.out = value; i += 1; }
    else if (flag === "--help" || flag === "-h") {
      process.stderr.write(`usage: node scripts/threebody-phase4-iad-sat-slate.mjs --keep-seeds <list> [--base <dir>] [--shards <comma-list>] --out <dir>\n`);
      process.exit(0);
    } else {
      throw new Error(`Unknown flag: ${flag}`);
    }
  }
  if (!args.keepSeeds) throw new Error("--keep-seeds is required (e.g. \"0,1,3,4,5,6,7,8\")");
  if (!args.out) throw new Error("--out is required");
  return args;
}

async function readTrialOutcomes(dir) {
  const csvPath = path.resolve(repoRoot, dir, "trial-outcomes.csv");
  const text = await readFile(csvPath, "utf8");
  const lines = text.split(/\r?\n/).filter((l) => l.length > 0);
  if (lines.length < 1) throw new Error(`empty trial-outcomes.csv at ${csvPath}`);
  const header = lines[0];
  const dataRows = lines.slice(1);
  return { header, dataRows, csvPath };
}

function seedOfRow(row) {
  // The CSV's first column is `seed` (integer). Just parse the first comma-sep field.
  const firstComma = row.indexOf(",");
  if (firstComma < 0) throw new Error(`malformed row: ${row.slice(0, 80)}`);
  const n = Number.parseInt(row.slice(0, firstComma), 10);
  if (!Number.isInteger(n)) throw new Error(`bad seed value at row head: ${row.slice(0, 80)}`);
  return n;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const inputs = [opts.base, ...opts.shards];
  const keepSet = new Set(opts.keepSeeds);

  console.log(`[iad-sat-slate] base=${opts.base}`);
  for (const s of opts.shards) console.log(`[iad-sat-slate] shard=${s}`);
  console.log(`[iad-sat-slate] keep-seeds=${opts.keepSeeds.join(",")}`);
  console.log(`[iad-sat-slate] out=${opts.out}`);

  // Read all inputs; strict header match
  const reads = await Promise.all(inputs.map((d) => readTrialOutcomes(d)));
  const refHeader = reads[0].header;
  for (let i = 1; i < reads.length; i += 1) {
    if (reads[i].header !== refHeader) {
      throw new Error(`header mismatch:\n  base: ${refHeader}\n  input[${i}] (${inputs[i]}): ${reads[i].header}`);
    }
  }

  // Concat, filter to keep-seeds, dedup by (seed, controller_mode) to be safe
  const seenKey = new Set();
  const keptRows = [];
  const droppedSeeds = new Set();
  const seedsFound = new Set();
  for (let i = 0; i < reads.length; i += 1) {
    for (const row of reads[i].dataRows) {
      const seed = seedOfRow(row);
      seedsFound.add(seed);
      if (!keepSet.has(seed)) { droppedSeeds.add(seed); continue; }
      // Dedup key: seed + controller_mode column. Find controller_mode column index.
      const cells = row.split(",");
      const headerCells = refHeader.split(",");
      const idxMode = headerCells.indexOf("controller_mode");
      const mode = idxMode >= 0 ? cells[idxMode] : "?";
      const key = `${seed}:${mode}`;
      if (seenKey.has(key)) {
        console.warn(`[iad-sat-slate] dropping duplicate (seed=${seed}, mode=${mode}) from input[${i}]`);
        continue;
      }
      seenKey.add(key);
      keptRows.push(row);
    }
  }

  // Validate: every keep-seed actually appeared somewhere
  const missing = opts.keepSeeds.filter((s) => !seedsFound.has(s));
  if (missing.length > 0) {
    throw new Error(`keep-seeds missing from all inputs: ${missing.join(",")}`);
  }

  // Sort by (seed, mode) for stable output
  const headerCells = refHeader.split(",");
  const idxMode = headerCells.indexOf("controller_mode");
  keptRows.sort((a, b) => {
    const sa = seedOfRow(a), sb = seedOfRow(b);
    if (sa !== sb) return sa - sb;
    const ma = a.split(",")[idxMode] || "";
    const mb = b.split(",")[idxMode] || "";
    return ma.localeCompare(mb);
  });

  const outAbs = path.resolve(repoRoot, opts.out);
  await mkdir(outAbs, { recursive: true });
  const outCsv = `${refHeader}\n${keptRows.join("\n")}\n`;
  await writeFile(path.join(outAbs, "trial-outcomes.csv"), outCsv, "utf8");

  // Write a small merge manifest for traceability
  const manifest = {
    schema: "sundog.threebody.phase4-iad-sat-slate.v1",
    purpose: "Signature-side slate built from base sat dir + per-seed sat shard dirs, filtered to a final keep-seeds slate. Use as --signature-in / --anchor-slate for threebody-phase4-regret.mjs.",
    inputs: inputs.map((d, i) => ({ index: i, dir: d, role: i === 0 ? "base" : "shard" })),
    keepSeeds: opts.keepSeeds,
    droppedSeeds: [...droppedSeeds].sort((a, b) => a - b),
    seedsFound: [...seedsFound].sort((a, b) => a - b),
    rowsWritten: keptRows.length,
  };
  await writeFile(path.join(outAbs, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

  console.log("");
  console.log(`[iad-sat-slate] wrote ${keptRows.length} rows to ${opts.out}/trial-outcomes.csv`);
  console.log(`[iad-sat-slate] kept seeds: ${opts.keepSeeds.join(",")}`);
  if (droppedSeeds.size > 0) console.log(`[iad-sat-slate] dropped seeds: ${[...droppedSeeds].sort((a, b) => a - b).join(",")}`);
}

main().catch((err) => {
  console.error(`[iad-sat-slate] fatal: ${err.stack || err.message}`);
  process.exit(2);
});
