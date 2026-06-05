#!/usr/bin/env node
// scripts/pvnp-phase3-v1-holdout.mjs
//
// Phase 3 holdout battery runner. By default, this generates the frozen v1
// 52 source-bound 64-seed holdout blocks (13 registered sources x 4 seed
// starts) by invoking scripts/mesa-intervention-battery.mjs with the EXACT
// frozen v1 arguments from scripts/lib/pvnp-phase3-v1-config.mjs.
//
// The extra flags are execution conveniences only: they preserve the source
// rows, horizon, seed count, sensor tier, and per-block command shape while
// allowing fresh roots/seed starts, deterministic shards, and bounded local
// parallelism for later slates.
//
// Flags:
//   --source <slug>       restrict to one source slug (repeatable)
//   --seed-start <n>      use one seed start (repeatable; overrides v1 starts)
//   --out-root <path>     holdout root; defaults to the frozen v1 root
//   --jobs <n>            run up to n blocks concurrently; default 1
//   --shard-index <n>     run only blocks where full-plan ordinal % shard_count
//                         equals n; zero-based
//   --shard-count <n>     deterministic shard count
//   --manifest-name <f>   manifest filename; default is shard/smoke aware
//   --dry-run             print the plan and exit; do not spawn
//   --limit <n>           run at most n missing blocks this invocation
//   --smoke               alias for --limit 1 unless a lower limit is supplied

import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import {
  V1_HOLDOUT_ROOT,
  V1_HOLDOUT_SEED_STARTS,
  V1_HOLDOUT_SOURCES,
  holdoutArgsForRoot,
  holdoutBlockDirForRoot,
} from "./lib/pvnp-phase3-v1-config.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function usage() {
  return `Usage: node scripts/pvnp-phase3-v1-holdout.mjs [flags]

Default: frozen Phase 3 v1 holdout population, sequential, v1 output root.

Useful v2 shape:
  node scripts/pvnp-phase3-v1-holdout.mjs --out-root results/pvnp/phase3-capacity-one-wayness-v2/phase4-intervention-battery --seed-start 100000 --seed-start 110000 --seed-start 120000 --seed-start 130000 --jobs 4
`;
}

function requireValue(argv, i, flag) {
  const value = argv[i + 1];
  if (value === undefined || value.startsWith("--")) {
    throw new Error(`Missing value for ${flag}`);
  }
  return value;
}

function parseInteger(value, flag) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) throw new Error(`${flag} must be an integer: ${value}`);
  return parsed;
}

function parseArgs(argv) {
  const args = {
    sources: [],
    seedStarts: [],
    dryRun: false,
    limit: Infinity,
    smoke: false,
    jobs: 1,
    shardIndex: null,
    shardCount: null,
    outRoot: V1_HOLDOUT_ROOT,
    manifestName: null,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    if (flag === "--help" || flag === "-h") {
      console.log(usage());
      process.exit(0);
    } else if (flag === "--source") {
      args.sources.push(requireValue(argv, i, flag));
      i += 1;
    } else if (flag === "--seed-start") {
      args.seedStarts.push(parseInteger(requireValue(argv, i, flag), flag));
      i += 1;
    } else if (flag === "--out-root") {
      args.outRoot = requireValue(argv, i, flag);
      i += 1;
    } else if (flag === "--jobs") {
      args.jobs = parseInteger(requireValue(argv, i, flag), flag);
      i += 1;
    } else if (flag === "--shard-index") {
      args.shardIndex = parseInteger(requireValue(argv, i, flag), flag);
      i += 1;
    } else if (flag === "--shard-count") {
      args.shardCount = parseInteger(requireValue(argv, i, flag), flag);
      i += 1;
    } else if (flag === "--manifest-name") {
      args.manifestName = requireValue(argv, i, flag);
      i += 1;
    } else if (flag === "--dry-run") {
      args.dryRun = true;
    } else if (flag === "--limit") {
      args.limit = parseInteger(requireValue(argv, i, flag), flag);
      i += 1;
    } else if (flag === "--smoke") {
      args.smoke = true;
      args.limit = Math.min(args.limit, 1);
    } else {
      throw new Error(`Unknown flag: ${flag}`);
    }
  }

  if (args.jobs < 1) throw new Error("--jobs must be >= 1");
  if (!Number.isFinite(args.limit) && args.limit !== Infinity) throw new Error("--limit must be finite");
  if (Number.isFinite(args.limit) && args.limit < 0) throw new Error("--limit must be >= 0");
  if ((args.shardIndex === null) !== (args.shardCount === null)) {
    throw new Error("--shard-index and --shard-count must be supplied together");
  }
  if (args.shardCount !== null) {
    if (args.shardCount < 1) throw new Error("--shard-count must be >= 1");
    if (args.shardIndex < 0 || args.shardIndex >= args.shardCount) {
      throw new Error("--shard-index must satisfy 0 <= shard-index < shard-count");
    }
  }
  return args;
}

function unique(items) {
  return [...new Set(items)];
}

function selectedSources(slugs) {
  if (slugs.length === 0) return V1_HOLDOUT_SOURCES;
  const bySlug = new Map(V1_HOLDOUT_SOURCES.map((source) => [source.slug, source]));
  return unique(slugs).map((slug) => {
    const source = bySlug.get(slug);
    if (!source) throw new Error(`Unknown source slug: ${slug}`);
    return source;
  });
}

function selectedSeedStarts(seedStarts) {
  if (seedStarts.length === 0) return V1_HOLDOUT_SEED_STARTS;
  return unique(seedStarts).map((seedStart) => {
    if (seedStart < 0) throw new Error(`Seed start must be >= 0: ${seedStart}`);
    return seedStart;
  });
}

function trimRoot(root) {
  return root.replace(/[\\/]+$/, "");
}

function manifestDirForRoot(root) {
  const trimmed = trimRoot(root);
  if (path.basename(trimmed) === "phase4-intervention-battery") return path.dirname(trimmed);
  return trimmed;
}

function padShard(value, width) {
  return String(value).padStart(width, "0");
}

function defaultManifestName(args) {
  if (args.manifestName) return args.manifestName;
  if (args.shardCount !== null) {
    const width = Math.max(2, String(args.shardCount - 1).length);
    return `holdout_runner_manifest_shard_${padShard(args.shardIndex, width)}_of_${padShard(args.shardCount, width)}.json`;
  }
  if (args.smoke) return "holdout_runner_manifest_smoke.json";
  return "holdout_runner_manifest.json";
}

function quotePowerShellArg(arg) {
  if (/^[A-Za-z0-9_./:=+-]+$/.test(arg)) return arg;
  return `"${arg.replaceAll("`", "``").replaceAll('"', '`"')}"`;
}

function commandText(argv) {
  return ["node", ...argv].map(quotePowerShellArg).join(" ");
}

async function blockComplete(dir) {
  try {
    const manifest = JSON.parse(await readFile(path.resolve(REPO_ROOT, dir, "manifest.json"), "utf8"));
    return manifest.trial_logs_saved === true;
  } catch {
    return false;
  }
}

function runNode(argv) {
  return new Promise((resolve, reject) => {
    const child = spawn("node", argv, { cwd: REPO_ROOT, stdio: "inherit" });
    child.on("error", reject);
    child.on("close", (code, signal) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`node ${argv.join(" ")} exited with ${signal ?? code}`));
      }
    });
  });
}

async function runWithConcurrency(items, jobs, worker) {
  const runs = [];
  let nextIndex = 0;
  const workerCount = Math.min(jobs, Math.max(items.length, 1));

  async function loop(workerIndex) {
    while (nextIndex < items.length) {
      const item = items[nextIndex];
      nextIndex += 1;
      const result = await worker(item, workerIndex);
      runs.push(result);
    }
  }

  await Promise.all(Array.from({ length: workerCount }, (_, workerIndex) => loop(workerIndex)));
  runs.sort((a, b) => a.plan_ordinal - b.plan_ordinal);
  return runs;
}

async function main() {
  const startedAt = new Date().toISOString();
  const t0 = performance.now();
  const args = parseArgs(process.argv.slice(2));
  const sources = selectedSources(args.sources);
  const seedStarts = selectedSeedStarts(args.seedStarts);
  const outRoot = trimRoot(args.outRoot);

  const fullPlan = [];
  for (const source of sources) {
    for (const seedStart of seedStarts) {
      const planOrdinal = fullPlan.length;
      fullPlan.push({
        plan_ordinal: planOrdinal,
        source,
        seedStart,
        dir: holdoutBlockDirForRoot(source, seedStart, outRoot),
      });
    }
  }

  const plan =
    args.shardCount === null
      ? fullPlan
      : fullPlan.filter((item) => item.plan_ordinal % args.shardCount === args.shardIndex);

  let startedMissingBlocks = 0;
  const runs = await runWithConcurrency(plan, args.jobs, async (item, workerIndex) => {
    const argv = holdoutArgsForRoot(item.source, item.seedStart, outRoot);
    const already = await blockComplete(item.dir);
    const base = {
      plan_ordinal: item.plan_ordinal,
      worker_index: workerIndex,
      slug: item.source.slug,
      seed_start: item.seedStart,
      dir: item.dir,
      command: commandText(argv),
      already_complete: already,
    };

    if (args.dryRun) return { ...base, ran: false };
    if (already) return { ...base, ran: false, elapsed_ms: "" };
    if (startedMissingBlocks >= args.limit) {
      return { ...base, ran: false, skipped_reason: "limit reached" };
    }
    startedMissingBlocks += 1;

    await mkdir(path.resolve(REPO_ROOT, item.dir), { recursive: true });
    const blockT0 = performance.now();
    try {
      await runNode(argv);
      return {
        ...base,
        ran: true,
        elapsed_ms: Math.round(performance.now() - blockT0),
      };
    } catch (error) {
      return {
        ...base,
        ran: false,
        failed: true,
        elapsed_ms: Math.round(performance.now() - blockT0),
        error: error.message,
      };
    }
  });

  const manifestOutDir = path.resolve(REPO_ROOT, manifestDirForRoot(outRoot));
  const manifestName = defaultManifestName(args);
  const manifestPath = path.join(manifestOutDir, manifestName);
  const manifest = {
    schema: "pvnp-phase3-holdout-runner-manifest-v2",
    runner_script: "scripts/pvnp-phase3-v1-holdout.mjs",
    v1_default_behavior_preserved: args.outRoot === V1_HOLDOUT_ROOT && args.seedStarts.length === 0,
    output_root: outRoot,
    manifest_path: path.relative(REPO_ROOT, manifestPath).replaceAll("\\", "/"),
    selected_sources: sources.map((source) => source.slug),
    selected_seed_starts: seedStarts,
    source_count: sources.length,
    seed_start_count: seedStarts.length,
    planned_blocks_before_shard: fullPlan.length,
    planned_blocks: plan.length,
    dry_run: args.dryRun,
    smoke: args.smoke,
    limit: Number.isFinite(args.limit) ? args.limit : "none",
    jobs: args.jobs,
    shard:
      args.shardCount === null
        ? null
        : { shard_index: args.shardIndex, shard_count: args.shardCount },
    blocks_run_this_invocation: runs.filter((run) => run.ran).length,
    blocks_already_complete: runs.filter((run) => run.already_complete).length,
    blocks_skipped_limit: runs.filter((run) => run.skipped_reason === "limit reached").length,
    blocks_failed: runs.filter((run) => run.failed).length,
    all_planned_complete: !args.dryRun && runs.every((run) => run.already_complete || run.ran),
    started_at: startedAt,
    completed_at: new Date().toISOString(),
    elapsed_ms: Math.round(performance.now() - t0),
    runs,
  };

  await mkdir(manifestOutDir, { recursive: true });
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

  console.log(
    `phase3-holdout: planned ${plan.length}/${fullPlan.length}, ran ${manifest.blocks_run_this_invocation}, already-complete ${manifest.blocks_already_complete}, failed ${manifest.blocks_failed}`,
  );
  console.log(`manifest: ${manifest.manifest_path}`);
  console.log(`all planned complete: ${manifest.all_planned_complete}`);

  if (manifest.blocks_failed > 0) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
