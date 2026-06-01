#!/usr/bin/env node
// scripts/pvnp-phase3-v1-holdout.mjs
//
// Phase 3 v1 holdout battery runner. Generates the frozen 52 source-bound
// 64-seed holdout blocks (13 registered sources x 4 seed starts) by invoking
// scripts/mesa-intervention-battery.mjs with the EXACT frozen arguments from
// scripts/lib/pvnp-phase3-v1-config.mjs (holdoutArgs / holdoutBlockDir). It does
// not retrain or re-simulate mesa policies; it replays the registered policies
// on the frozen v1 holdout seed blocks.
//
// This is an execution convenience, not a contract change: every command is the
// frozen `holdoutArgs(source, seedStart)`. It mirrors the v0 seed-extension
// runner: blocks already on disk with trial_logs_saved=true are skipped, so the
// batch is idempotent and resumable on a loaded/flaky machine.
//
// Flags:
//   --source <slug>      restrict to one source slug (repeatable)
//   --seed-start <n>     restrict to one seed start (repeatable)
//   --dry-run            print the plan and exit; do not spawn
//   --limit <n>          run at most n blocks this invocation

import { execFileSync } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import {
  V1_HOLDOUT_ROOT,
  V1_HOLDOUT_SEED_STARTS,
  V1_HOLDOUT_SOURCES,
  V1_OUT,
  holdoutArgs,
  holdoutBlockDir,
} from "./lib/pvnp-phase3-v1-config.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const args = { sources: [], seedStarts: [], dryRun: false, limit: Infinity };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === "--source") args.sources.push(argv[++i]);
    else if (a === "--seed-start") args.seedStarts.push(Number(argv[++i]));
    else if (a === "--dry-run") args.dryRun = true;
    else if (a === "--limit") args.limit = Number(argv[++i]);
    else throw new Error(`Unknown flag: ${a}`);
  }
  return args;
}

async function blockComplete(dir) {
  try {
    const manifest = JSON.parse(await readFile(path.resolve(REPO_ROOT, dir, "manifest.json"), "utf8"));
    return manifest.trial_logs_saved === true;
  } catch {
    return false;
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const sources = V1_HOLDOUT_SOURCES.filter((s) => args.sources.length === 0 || args.sources.includes(s.slug));
  const seedStarts = V1_HOLDOUT_SEED_STARTS.filter((s) => args.seedStarts.length === 0 || args.seedStarts.includes(s));

  const plan = [];
  for (const source of sources) {
    for (const seedStart of seedStarts) {
      plan.push({ source, seedStart, dir: holdoutBlockDir(source, seedStart) });
    }
  }

  const runs = [];
  let ran = 0;
  for (const item of plan) {
    const argv = holdoutArgs(item.source, item.seedStart);
    const already = await blockComplete(item.dir);
    if (args.dryRun) {
      runs.push({ slug: item.source.slug, seed_start: item.seedStart, dir: item.dir, command: `node ${argv.join(" ")}`, already_complete: already, ran: false });
      continue;
    }
    if (already) {
      runs.push({ slug: item.source.slug, seed_start: item.seedStart, dir: item.dir, already_complete: true, ran: false, elapsed_ms: "" });
      continue;
    }
    if (ran >= args.limit) {
      runs.push({ slug: item.source.slug, seed_start: item.seedStart, dir: item.dir, already_complete: false, ran: false, skipped_reason: "limit reached" });
      continue;
    }
    await mkdir(path.resolve(REPO_ROOT, item.dir), { recursive: true });
    const t0 = performance.now();
    execFileSync("node", argv, { cwd: REPO_ROOT, stdio: "inherit" });
    const elapsed = performance.now() - t0;
    ran += 1;
    runs.push({ slug: item.source.slug, seed_start: item.seedStart, dir: item.dir, already_complete: false, ran: true, elapsed_ms: Math.round(elapsed) });
  }

  const manifest = {
    schema: "pvnp-phase3-v1-holdout-runner-manifest",
    output_root: V1_HOLDOUT_ROOT,
    source_count: sources.length,
    seed_start_count: seedStarts.length,
    planned_blocks: plan.length,
    blocks_run_this_invocation: ran,
    blocks_already_complete: runs.filter((r) => r.already_complete).length,
    all_planned_complete: runs.every((r) => r.already_complete || r.ran),
    runs,
  };
  const outDir = path.resolve(REPO_ROOT, V1_OUT);
  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "holdout_runner_manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

  console.log(`phase3-v1-holdout: planned ${plan.length}, ran ${ran}, already-complete ${manifest.blocks_already_complete}`);
  console.log(`all planned complete: ${manifest.all_planned_complete}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
