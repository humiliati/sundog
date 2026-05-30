// scripts/threebody-phase4-iad-merge.mjs
//
// Merge per-shard Phase 4 BF-4b IAD outputs into a unified
// `_bf4b-accessibility/` directory that mirrors what a single 8-seed
// invocation would have produced — the input the regret reducer
// (`threebody-phase4-regret.mjs --bayes-in <merged>`) expects.
//
// Inputs: <shards-root>/_bf4b-accessibility-shard-seed_<S>/{manifest.json,
//   bayes-trial-outcomes.csv, bayes-actions.csv, belief-diagnostics.csv,
//   signature-observations.jsonl}
// Output: <out>/ with the same five files merged.
//
// Strict checks (any failure aborts and writes nothing):
//   - all shards completed (manifest.completedAt present)
//   - pinned IAD args identical across shards (except seedStart, seeds)
//   - no duplicate seed across shards
//
// Usage:
//   node scripts/threebody-phase4-iad-merge.mjs [--shards-root <dir>] [--out <dir>] [--seeds <list-or-count>]
//
// Defaults:
//   --shards-root results/proof/phase4
//   --out results/proof/phase4/_bf4b-accessibility
//   --seeds 8                  (expects shards for seeds 0..7)

import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// Manifest args that are allowed to differ across shards (per-seed-shard
// uniqueness fields). Everything else MUST match exactly across shards.
const ALLOWED_SHARD_DIFFS = new Set(["seedStart", "seeds", "phase", "out"]);

function parseArgs(argv) {
  const args = {
    shardsRoot: "results/proof/phase4",
    out: "results/proof/phase4/_bf4b-accessibility",
    seeds: [0, 1, 2, 3, 4, 5, 6, 7],
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    if (flag === "--shards-root") { args.shardsRoot = value; i += 1; }
    else if (flag === "--out") { args.out = value; i += 1; }
    else if (flag === "--seeds") {
      if (/^\d+$/.test(value)) {
        const n = Number.parseInt(value, 10);
        args.seeds = Array.from({ length: n }, (_, k) => k);
      } else {
        args.seeds = value.split(",").map((s) => Number.parseInt(s.trim(), 10));
      }
      i += 1;
    } else if (flag === "--help" || flag === "-h") {
      process.stderr.write(`usage: node scripts/threebody-phase4-iad-merge.mjs [--shards-root <dir>] [--out <dir>] [--seeds <count|list>]\n`);
      process.exit(0);
    } else {
      throw new Error(`Unknown flag: ${flag}`);
    }
  }
  return args;
}

function shardDirFor(shardsRoot, seed) {
  return path.join(shardsRoot, `_bf4b-accessibility-shard-seed_${seed}`);
}

async function readJsonIfExists(p) {
  try { return JSON.parse(await readFile(p, "utf8")); }
  catch (e) { if (e.code === "ENOENT") return null; throw e; }
}

async function readTextIfExists(p) {
  try { return await readFile(p, "utf8"); }
  catch (e) { if (e.code === "ENOENT") return null; throw e; }
}

function diffArgs(a, b) {
  const keys = new Set([...Object.keys(a || {}), ...Object.keys(b || {})]);
  const diffs = [];
  for (const k of keys) {
    if (ALLOWED_SHARD_DIFFS.has(k)) continue;
    const av = JSON.stringify(a?.[k]);
    const bv = JSON.stringify(b?.[k]);
    if (av !== bv) diffs.push({ key: k, a: av, b: bv });
  }
  return diffs;
}

// Concatenate N CSV bodies sharing a common header. The first shard's full
// content is kept as-is; subsequent shards drop their header line.
function concatCsv(contents) {
  if (contents.length === 0) return "";
  const first = contents[0];
  const firstHeaderEnd = first.indexOf("\n");
  if (firstHeaderEnd < 0) return first;
  const header = first.slice(0, firstHeaderEnd + 1);
  const pieces = [first];
  for (let i = 1; i < contents.length; i += 1) {
    const text = contents[i];
    const headerEnd = text.indexOf("\n");
    const ownHeader = headerEnd < 0 ? text : text.slice(0, headerEnd + 1);
    if (ownHeader !== header) {
      throw new Error(`shard CSV header mismatch:\n  expected: ${header.trim()}\n  shard[${i}]: ${ownHeader.trim()}`);
    }
    const body = headerEnd < 0 ? "" : text.slice(headerEnd + 1);
    if (body.length > 0) pieces.push(body);
  }
  // Ensure trailing newline normalization
  let merged = pieces.join("");
  if (!merged.endsWith("\n")) merged += "\n";
  return merged;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const shardsRootAbs = path.resolve(repoRoot, opts.shardsRoot);
  const outAbs = path.resolve(repoRoot, opts.out);

  console.log(`[iad-merge] shards-root=${opts.shardsRoot} out=${opts.out} seeds=${opts.seeds.join(",")}`);

  // ── Read all shards strictly ───────────────────────────────────────────
  const shards = [];
  for (const seed of opts.seeds) {
    const dirAbs = path.resolve(repoRoot, shardDirFor(opts.shardsRoot, seed));
    const manifest = await readJsonIfExists(path.join(dirAbs, "manifest.json"));
    if (!manifest) throw new Error(`shard seed=${seed}: manifest.json missing at ${dirAbs}`);
    if (!manifest.completedAt) throw new Error(`shard seed=${seed}: manifest.completedAt missing — run incomplete?`);
    const trialOutcomes = await readTextIfExists(path.join(dirAbs, "bayes-trial-outcomes.csv"));
    if (!trialOutcomes) throw new Error(`shard seed=${seed}: bayes-trial-outcomes.csv missing`);
    const actions = await readTextIfExists(path.join(dirAbs, "bayes-actions.csv"));
    const beliefs = await readTextIfExists(path.join(dirAbs, "belief-diagnostics.csv"));
    const obs = await readTextIfExists(path.join(dirAbs, "signature-observations.jsonl"));
    shards.push({ seed, dirAbs, manifest, trialOutcomes, actions, beliefs, obs });
  }

  // ── Strict consistency checks ──────────────────────────────────────────
  // (a) shard args identical (excluding seedStart/seeds/phase/out)
  const refArgs = shards[0].manifest.args;
  for (let i = 1; i < shards.length; i += 1) {
    const diffs = diffArgs(refArgs, shards[i].manifest.args);
    if (diffs.length > 0) {
      const lines = diffs.slice(0, 5).map((d) => `  ${d.key}: shard[0]=${d.a} shard[${i}]=${d.b}`);
      throw new Error(`shard args mismatch (shard[0] vs shard[${i}], seed=${shards[i].seed}):\n${lines.join("\n")}`);
    }
  }
  // (b) seeds in each shard's manifest match the shard's --seed
  for (const s of shards) {
    if (s.manifest.args.seedStart !== s.seed || s.manifest.args.seeds !== 1) {
      throw new Error(`shard seed=${s.seed}: manifest args (seedStart=${s.manifest.args.seedStart}, seeds=${s.manifest.args.seeds}) inconsistent with shard key`);
    }
  }
  // (c) no duplicate seed
  const seedSet = new Set();
  for (const s of shards) {
    if (seedSet.has(s.seed)) throw new Error(`duplicate seed=${s.seed} across shards`);
    seedSet.add(s.seed);
  }

  console.log(`[iad-merge] strict checks passed: ${shards.length} shards, args consistent`);

  // ── Merge CSVs and JSONL ───────────────────────────────────────────────
  const mergedTrialOutcomes = concatCsv(shards.map((s) => s.trialOutcomes));
  const mergedActions = shards.every((s) => s.actions != null)
    ? concatCsv(shards.map((s) => s.actions))
    : null;
  const mergedBeliefs = shards.every((s) => s.beliefs != null)
    ? concatCsv(shards.map((s) => s.beliefs))
    : null;
  const mergedObs = shards.every((s) => s.obs != null)
    ? shards.map((s) => s.obs.endsWith("\n") ? s.obs : s.obs + "\n").join("")
    : null;

  // ── Build merged manifest ──────────────────────────────────────────────
  const startedAts = shards.map((s) => new Date(s.manifest.startedAt).getTime());
  const completedAts = shards.map((s) => new Date(s.manifest.completedAt).getTime());
  const mergedManifest = {
    schema: "sundog.threebody.phase4-bf4b-accessibility.merged.v1",
    mergedFrom: shards.map((s) => ({
      seed: s.seed,
      shardDir: path.relative(repoRoot, s.dirAbs),
      startedAt: s.manifest.startedAt,
      completedAt: s.manifest.completedAt,
      schema: s.manifest.schema,
    })),
    purpose: "Merged IAD shards — output structure mirrors a single 8-seed invocation for the regret reducer.",
    args: {
      ...refArgs,
      phase: "phase4-bf4b-accessibility",
      out: path.relative(repoRoot, outAbs).replaceAll("/", path.sep),
      seedStart: Math.min(...shards.map((s) => s.seed)),
      seeds: shards.length,
    },
    startedAt: new Date(Math.min(...startedAts)).toISOString(),
    completedAt: new Date(Math.max(...completedAts)).toISOString(),
    shardCount: shards.length,
    trialCount: shards.length,
  };

  // ── Write merged output ────────────────────────────────────────────────
  await mkdir(outAbs, { recursive: true });
  await writeFile(path.join(outAbs, "manifest.json"), `${JSON.stringify(mergedManifest, null, 2)}\n`, "utf8");
  await writeFile(path.join(outAbs, "bayes-trial-outcomes.csv"), mergedTrialOutcomes, "utf8");
  if (mergedActions != null) await writeFile(path.join(outAbs, "bayes-actions.csv"), mergedActions, "utf8");
  if (mergedBeliefs != null) await writeFile(path.join(outAbs, "belief-diagnostics.csv"), mergedBeliefs, "utf8");
  if (mergedObs != null) await writeFile(path.join(outAbs, "signature-observations.jsonl"), mergedObs, "utf8");

  console.log(`[iad-merge] wrote merged output to ${opts.out}`);
  console.log(`[iad-merge]   manifest.json (${shards.length} trials)`);
  console.log(`[iad-merge]   bayes-trial-outcomes.csv (${mergedTrialOutcomes.split("\n").length - 2} data rows)`);
  if (mergedActions != null) console.log(`[iad-merge]   bayes-actions.csv (${mergedActions.split("\n").length - 2} data rows)`);
  if (mergedBeliefs != null) console.log(`[iad-merge]   belief-diagnostics.csv (${mergedBeliefs.split("\n").length - 2} data rows)`);
  if (mergedObs != null) console.log(`[iad-merge]   signature-observations.jsonl (${mergedObs.split("\n").filter(Boolean).length} lines)`);
  console.log("");
  console.log("─── next step ─────────────────────────────────────────────────");
  console.log(`node scripts/threebody-phase4-regret.mjs --bayes-in ${opts.out} --signature-in results/proof/phase4/_bf4b-satisfiability --anchor-slate results/proof/phase4/_bf4b-satisfiability --signature-mode track_sensor_accel_guarded`);
}

main().catch((err) => {
  console.error(`[iad-merge] fatal: ${err.stack || err.message}`);
  process.exit(2);
});
