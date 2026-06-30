#!/usr/bin/env node
// Release smoke for the non-sovereignty public task family.

import { spawnSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

const repoRoot = process.cwd();
const outDir = "results/mesa/non-sovereignty/release_smoke";

const steps = [
  {
    name: "manifest",
    cmd: ["node", "scripts/mesa-ns-release-manifest.mjs", "--check"],
  },
  {
    name: "js_python_parity",
    cmd: ["node", "scripts/mesa-ns-release-parity-smoke.mjs", "--out", `${outDir}/parity.json`],
  },
  {
    name: "ns1_admission_smoke",
    cmd: [
      "node", "scripts/mesa-ns1-shutdown-admission.mjs",
      "--seeds", "6",
      "--cells", "nominal,wide-fork",
      "--out", `${outDir}/ns1_admission.md`,
      "--json", `${outDir}/ns1_admission.json`,
      "--csv", `${outDir}/ns1_admission.csv`,
    ],
  },
  {
    name: "ns1c_cap_validity_smoke",
    cmd: [
      "node", "scripts/mesa-ns1c-cap-validity.mjs",
      "--seeds", "8",
      "--cells", "nominal,wide-fork,far-lure",
      "--out", `${outDir}/ns1c_cap_validity.md`,
      "--json", `${outDir}/ns1c_cap_validity.json`,
    ],
  },
  {
    name: "ns2_table_regeneration",
    cmd: [
      "node", "scripts/mesa-ns2-aggregate.mjs",
      "--root", "results/mesa/non-sovereignty/ns2_b",
      "--kappas", "0.4,0.6,0.8,1.0",
      "--floors", "0.05,0.10,0.20",
      "--out", `${outDir}/ns2_unified_bound.md`,
      "--json", `${outDir}/ns2_unified_bound.json`,
    ],
  },
  {
    name: "ns3_admission_smoke",
    cmd: [
      "node", "scripts/mesa-ns3-admission.mjs",
      "--seeds", "8",
      "--out", `${outDir}/ns3_admission.md`,
      "--json", `${outDir}/ns3_admission.json`,
    ],
  },
  {
    name: "ns4_admission_smoke",
    cmd: [
      "node", "scripts/mesa-ns4-admission.mjs",
      "--seeds", "8",
      "--out", `${outDir}/ns4_admission.md`,
      "--json", `${outDir}/ns4_admission.json`,
    ],
  },
];

mkdirSync(path.resolve(repoRoot, outDir), { recursive: true });
const started = Date.now();
const rows = [];
for (const step of steps) {
  const t0 = Date.now();
  console.log(`[release-smoke] ${step.name}: ${step.cmd.join(" ")}`);
  const run = spawnSync(step.cmd[0], step.cmd.slice(1), {
    cwd: repoRoot,
    encoding: "utf8",
    shell: false,
  });
  const elapsed = Number(((Date.now() - t0) / 1000).toFixed(3));
  if (run.stdout) process.stdout.write(run.stdout);
  if (run.stderr) process.stderr.write(run.stderr);
  rows.push({
    name: step.name,
    command: step.cmd.join(" "),
    exit_code: run.status,
    elapsed_sec: elapsed,
  });
  if (run.status !== 0) {
    const summary = { phase: "non-sovereignty release smoke", passed: false, failed_step: step.name, steps: rows };
    writeFileSync(path.resolve(repoRoot, outDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf8");
    process.exit(run.status ?? 1);
  }
}
const summary = {
  phase: "non-sovereignty release smoke",
  generated_at: new Date().toISOString(),
  passed: true,
  elapsed_sec: Number(((Date.now() - started) / 1000).toFixed(3)),
  steps: rows,
};
writeFileSync(path.resolve(repoRoot, outDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf8");
console.log(`[release-smoke] PASS in ${summary.elapsed_sec}s. Wrote ${outDir}/summary.json`);
