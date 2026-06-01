#!/usr/bin/env node
// scripts/pvnp-phase3-seed-extension.mjs
//
// Phase 3 v0 seed-extension runner. Generates additive source-bound 64-seed
// blocks for the 6 registered unsafe policies x 4 registered seed starts by
// invoking scripts/mesa-intervention-battery.mjs, exactly as frozen in
// docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V0_SLATE.md (Seed-Extension Rule).
//
// It does NOT retrain or re-simulate mesa policies; it replays the registered
// policies on new seed blocks so the spoof search has source-bound candidate
// variation. Blocks already on disk with trial_logs_saved=true are skipped
// (idempotent), so the run can be resumed under the ~10-minute inline rule.
//
// Flags:
//   --battery <slug>     restrict to one battery slug (repeatable)
//   --seed-start <n>     restrict to one seed start (repeatable)
//   --dry-run            print the commands and exit; do not spawn
//   --limit <n>          run at most n blocks this invocation (budget guard)

import { execFileSync } from "node:child_process";
import { mkdir, readFile, writeFile, stat } from "node:fs/promises";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import {
  SEED_EXTENSION_BATTERIES,
  SEED_EXTENSION_SEED_STARTS,
  SEED_EXTENSION_ROOT,
  seedExtensionArgs,
  seedExtensionBlockDir,
} from "./lib/pvnp-phase3-config.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const args = { batteries: [], seedStarts: [], dryRun: false, limit: Infinity };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === "--battery") { args.batteries.push(argv[++i]); }
    else if (a === "--seed-start") { args.seedStarts.push(Number(argv[++i])); }
    else if (a === "--dry-run") { args.dryRun = true; }
    else if (a === "--limit") { args.limit = Number(argv[++i]); }
    else throw new Error(`Unknown flag: ${a}`);
  }
  return args;
}

async function blockComplete(dir) {
  try {
    const manifestPath = path.resolve(REPO_ROOT, dir, "manifest.json");
    const text = await readFile(manifestPath, "utf8");
    const manifest = JSON.parse(text);
    return manifest.trial_logs_saved === true;
  } catch {
    return false;
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const batteries = SEED_EXTENSION_BATTERIES.filter(
    (b) => args.batteries.length === 0 || args.batteries.includes(b.slug),
  );
  const seedStarts = SEED_EXTENSION_SEED_STARTS.filter(
    (s) => args.seedStarts.length === 0 || args.seedStarts.includes(s),
  );

  const plan = [];
  for (const battery of batteries) {
    for (const seedStart of seedStarts) {
      plan.push({ battery, seedStart, dir: seedExtensionBlockDir(battery, seedStart) });
    }
  }

  const runs = [];
  let ran = 0;
  for (const item of plan) {
    const argv = seedExtensionArgs(item.battery, item.seedStart);
    const already = await blockComplete(item.dir);
    if (args.dryRun) {
      runs.push({ slug: item.battery.slug, seed_start: item.seedStart, dir: item.dir, command: `node ${argv.join(" ")}`, already_complete: already, ran: false });
      continue;
    }
    if (already) {
      runs.push({ slug: item.battery.slug, seed_start: item.seedStart, dir: item.dir, already_complete: true, ran: false, elapsed_ms: "" });
      continue;
    }
    if (ran >= args.limit) {
      runs.push({ slug: item.battery.slug, seed_start: item.seedStart, dir: item.dir, already_complete: false, ran: false, skipped_reason: "limit reached" });
      continue;
    }
    await mkdir(path.resolve(REPO_ROOT, item.dir), { recursive: true });
    const t0 = performance.now();
    execFileSync("node", argv, { cwd: REPO_ROOT, stdio: "inherit" });
    const elapsed = performance.now() - t0;
    ran += 1;
    runs.push({ slug: item.battery.slug, seed_start: item.seedStart, dir: item.dir, already_complete: false, ran: true, elapsed_ms: Math.round(elapsed) });
  }

  const manifest = {
    schema: "pvnp-phase3-seed-extension-manifest",
    output_root: SEED_EXTENSION_ROOT,
    battery_count: batteries.length,
    seed_start_count: seedStarts.length,
    planned_blocks: plan.length,
    blocks_run_this_invocation: ran,
    blocks_already_complete: runs.filter((r) => r.already_complete).length,
    all_planned_complete: runs.every((r) => r.already_complete || r.ran),
    runs,
  };
  const outDir = path.resolve(REPO_ROOT, "results/pvnp/phase3-capacity-one-wayness-v0-seed-extension");
  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "seed_extension_manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

  console.log(`phase3-seed-extension: planned ${plan.length}, ran ${ran}, already-complete ${manifest.blocks_already_complete}`);
  console.log(`all planned complete: ${manifest.all_planned_complete}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
