#!/usr/bin/env node
// Generate/check the public non-sovereignty release manifest.

import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

const repoRoot = process.cwd();
const manifestPath = "released/non-sovereignty/MANIFEST.json";

const FILES = [
  "released/non-sovereignty/README.md",
  "released/non-sovereignty/REPRODUCE.md",
  "released/non-sovereignty/TASK_SPEC.md",
  "released/non-sovereignty/METRICS.md",
  "released/non-sovereignty/LICENSE.md",
  "released/non-sovereignty/LICENSE",
  "released/non-sovereignty/CITATION.cff",
  "docs/mesa/NON_SOVEREIGNTY_RELEASED_TASK_FAMILY_SPEC.md",
  "docs/mesa/NS1_CORRIGIBILITY_SHUTDOWN_CHANNEL_SPEC.md",
  "docs/mesa/NS1_C_ARBITER_AUTHORITY_CAP_SPEC.md",
  "docs/mesa/NS2_SHUTDOWN_MANIPULATION_SPEC.md",
  "docs/mesa/NS1_0_SHUTDOWN_CHANNEL_ADMISSION_RESULTS.md",
  "docs/mesa/NS1_B_CORRIGIBILITY_BINDING_RESULTS.md",
  "docs/mesa/NS1_C0_CAP_VALIDITY_RESULTS.md",
  "docs/mesa/NS1_C_ARBITER_AUTHORITY_CAP_RESULTS.md",
  "docs/mesa/NS2_0_ADMISSION_RESULTS.md",
  "docs/mesa/NS2_B_UNIFIED_BOUND_RESULTS.md",
  "scripts/h2-forked-task.mjs",
  "scripts/ns1-shutdown-task.mjs",
  "scripts/mesa-ns1-shutdown-admission.mjs",
  "scripts/mesa-ns1c-cap-validity.mjs",
  "scripts/mesa-ns2-admission.mjs",
  "scripts/mesa-ns1c-binding-eval.mjs",
  "scripts/mesa-ns2-binding-eval.mjs",
  "scripts/mesa-ns1c-aggregate.mjs",
  "scripts/mesa-ns2-aggregate.mjs",
  "scripts/mesa-ns1-b-binding.ps1",
  "scripts/mesa-ns1c-binding.ps1",
  "scripts/mesa-ns2-0-admission.ps1",
  "scripts/mesa-ns2-b-binding.ps1",
  "training/mesa/h2_forked_task.py",
  "training/mesa/ns1_shutdown_task.py",
  "training/mesa/train_ns1_shutdown.py",
  "scripts/mesa-ns-release-manifest.mjs",
  "scripts/mesa-ns-release-parity-smoke.mjs",
  "scripts/mesa-ns-release-smoke.mjs",
  "package.json",
  // shared core rng/util imported by every env (pinned for integrity)
  "public/js/mesa-core.mjs",
  // NS-3 regulator + NS-4 spatial task families (broadened release)
  "scripts/regulator-task.mjs",
  "scripts/spatial-regulator-task.mjs",
  "training/mesa/regulator_task.py",
  "training/mesa/spatial_regulator_task.py",
  "scripts/mesa-ns3-admission.mjs",
  "scripts/mesa-ns4-admission.mjs",
  "training/mesa/train_ns3_presider.py",
  "scripts/mesa-ns3-b-aggregate.mjs",
  "scripts/mesa-ns3-b-binding.ps1",
  "scripts/mesa-ns4-sb0-landscape-scan.mjs",
  "scripts/mesa-ns4-sb1-planner.mjs",
  "scripts/mesa-ns4-sb4-options.mjs",
  "docs/mesa/NS3_REGULATOR_TASK_FAMILY_SPEC.md",
  "docs/mesa/NS3_LEARNED_PRESIDER_REPLICATION_SPEC.md",
  "docs/mesa/NS3_0_REGULATOR_ADMISSION_RESULTS.md",
  "docs/mesa/NS3_B_BINDING_RESULTS.md",
  "docs/mesa/NS4_SPATIAL_REGULATOR_TASK_FAMILY_SPEC.md",
  "docs/mesa/NS4_0_SPATIAL_REGULATOR_ADMISSION_RESULTS.md",
  "docs/mesa/NS3_NS4_MANIPULATION_WALL_SYNTHESIS.md",
  "docs/mesa/NS3_NS4_COMPETENT_SANDBAG_HYPOTHESIS_SLATE.md",
  "docs/mesa/NS4_SB0_LANDSCAPE_RESULTS.md",
  "docs/mesa/NS4_SB1_PLANNER_RESULTS.md",
  "docs/mesa/NS4_SB4_OPTIONS_RESULTS.md",
];

const args = { check: false, out: manifestPath };
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i += 1) {
  const flag = argv[i];
  const value = argv[i + 1];
  if (flag === "--check") args.check = true;
  else if (flag === "--out") { args.out = value; i += 1; }
}

function relPath(file) {
  return file.replaceAll("\\", "/");
}

function hashFile(file) {
  const abs = path.resolve(repoRoot, file);
  if (!existsSync(abs)) throw new Error(`manifest file missing: ${file}`);
  const data = readFileSync(abs);
  return {
    path: relPath(file),
    bytes: data.length,
    sha256: createHash("sha256").update(data).digest("hex"),
  };
}

function currentManifest() {
  return {
    schema: "sundog.non_sovereignty.release_manifest.v1",
    status: "LICENSED_APACHE_2_0",
    generated_at: new Date().toISOString(),
    files: FILES.map(hashFile),
  };
}

function checkManifest() {
  const abs = path.resolve(repoRoot, args.out);
  if (!existsSync(abs)) throw new Error(`manifest not found: ${args.out}`);
  const existing = JSON.parse(readFileSync(abs, "utf8"));
  const expected = currentManifest();
  const byPath = new Map(expected.files.map((row) => [row.path, row]));
  const diffs = [];
  for (const row of existing.files ?? []) {
    const next = byPath.get(row.path);
    if (!next) {
      diffs.push(`unexpected file in manifest: ${row.path}`);
      continue;
    }
    if (row.sha256 !== next.sha256 || row.bytes !== next.bytes) {
      diffs.push(`changed ${row.path}: manifest ${row.sha256}/${row.bytes}, current ${next.sha256}/${next.bytes}`);
    }
    byPath.delete(row.path);
  }
  for (const missing of byPath.keys()) diffs.push(`missing from manifest: ${missing}`);
  if (diffs.length) {
    console.error("Non-sovereignty release manifest check failed:");
    for (const diff of diffs.slice(0, 40)) console.error(`  - ${diff}`);
    if (diffs.length > 40) console.error(`  ... ${diffs.length - 40} more`);
    process.exit(1);
  }
  console.log(`Non-sovereignty release manifest OK (${existing.files.length} files).`);
}

if (args.check) {
  checkManifest();
} else {
  const manifest = currentManifest();
  writeFileSync(path.resolve(repoRoot, args.out), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  console.log(`Wrote ${args.out} (${manifest.files.length} files).`);
}
