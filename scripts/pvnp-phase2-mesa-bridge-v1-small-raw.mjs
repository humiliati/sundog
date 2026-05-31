#!/usr/bin/env node
// Runs the frozen Phase 2 mesa bridge v1 Small-tier raw-log repair batteries.

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import {
  SMALL_RAW_BATTERIES,
  V1_OUT,
  V1_RUN_ID,
  V1_SMALL_RERUN_ROOT,
  nodeArgsForBattery,
  powerShellCommandForBattery,
  smallRawCommandsPs1,
} from "./lib/pvnp-phase2-mesa-bridge-v1-config.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function gitSha() {
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: REPO_ROOT,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch {
    return null;
  }
}

function rel(file) {
  return path.relative(REPO_ROOT, file).replaceAll("\\", "/");
}

async function sha256File(file) {
  const text = await readFile(file);
  return createHash("sha256").update(text).digest("hex");
}

async function collectRun(battery, elapsedMs) {
  const outArgIndex = battery.args.indexOf("--out");
  const outRel = battery.args[outArgIndex + 1];
  const outDir = path.resolve(REPO_ROOT, outRel);
  const manifestPath = path.join(outDir, "manifest.json");
  const manifestText = await readFile(manifestPath, "utf8");
  const manifest = JSON.parse(manifestText);
  const trialsDir = path.join(outDir, "trials");
  const trialFiles = (await readdir(trialsDir)).filter((name) => name.endsWith(".jsonl"));
  const expectedTrialFiles = Number(manifest.seed_count) * Number(manifest.channels?.length ?? 0) * 2;
  return {
    slug: battery.slug,
    label: battery.label,
    out: outRel,
    command: powerShellCommandForBattery(battery),
    elapsed_ms: Number(elapsedMs.toFixed(3)),
    manifest_path: rel(manifestPath),
    manifest_sha256: await sha256File(manifestPath),
    trial_logs_saved: manifest.trial_logs_saved === true,
    seed_base: manifest.seed_base,
    seed_count: manifest.seed_count,
    horizon: manifest.horizon,
    sensor_tier: manifest.sensor_tier,
    trial_pairs: manifest.trial_pairs,
    trial_file_count: trialFiles.length,
    expected_trial_file_count: expectedTrialFiles,
    trial_file_count_passed: trialFiles.length === expectedTrialFiles,
  };
}

async function main() {
  const outDir = path.resolve(REPO_ROOT, V1_OUT);
  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "small_raw_rerun_commands.ps1"), smallRawCommandsPs1(), "utf8");

  const startedAt = new Date().toISOString();
  const runs = [];
  for (const battery of SMALL_RAW_BATTERIES) {
    const t0 = performance.now();
    execFileSync(process.execPath, nodeArgsForBattery(battery), {
      cwd: REPO_ROOT,
      stdio: "inherit",
    });
    runs.push(await collectRun(battery, performance.now() - t0));
  }
  const completedAt = new Date().toISOString();
  const manifest = {
    schema: "pvnp-phase2-mesa-bridge-v1-small-raw-rerun-manifest",
    run_id: `${V1_RUN_ID}:small-raw`,
    git_sha: gitSha(),
    startedAt,
    completedAt,
    output_root: V1_SMALL_RERUN_ROOT,
    command_file: `${V1_OUT}/small_raw_rerun_commands.ps1`,
    run_count: runs.length,
    all_trial_logs_saved: runs.every((run) => run.trial_logs_saved),
    all_trial_file_counts_passed: runs.every((run) => run.trial_file_count_passed),
    runs,
  };
  await writeFile(
    path.join(outDir, "small_raw_rerun_manifest.json"),
    `${JSON.stringify(manifest, null, 2)}\n`,
    "utf8",
  );

  const totalMs = runs.reduce((sum, run) => sum + run.elapsed_ms, 0);
  console.log(`phase2-mesa-bridge-v1-small-raw: ${runs.length} batteries, ${Number(totalMs.toFixed(3))} ms`);
  console.log(`manifest: ${V1_OUT}/small_raw_rerun_manifest.json`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
