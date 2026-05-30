// scripts/threebody-phase18-calibrate.mjs
//
// Phase 18 matched-duty calibration (PHASE18_SPEC.md §3/§4). Runs the favorable
// pocket (velocityScale >= 1.05) for `track_sensor_accel_guarded` +
// `track_radius_inward` at each magnitude m in the LOCKED grid {0.1,0.2,0.3,0.4},
// then selects matched-m = argmin_m |meanΔV(inward,m) − meanΔV(guarded)| from
// totalDeltaV ONLY — blind to survival. Prints the table + matched-m and writes
// calibrate-summary.json. The measurement lock then runs at matched-m.

import { spawn } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MAGNITUDE_GRID = [0.1, 0.2, 0.3, 0.4];
const OUT_ROOT = "results/threebody/phase18-radius-control-calibrate";

function tag(m) { return String(m).replaceAll(".", "p"); }

function harnessArgs(m) {
  return [
    "scripts/threebody-operating-envelope.mjs",
    "--phase", `phase18-radius-control-calibrate-m${tag(m)}`,
    "--out", `${OUT_ROOT}-m${tag(m)}`,
    "--regimes", "near_escape",
    "--modes", "track_sensor_accel_guarded,track_radius_inward",
    "--mass-ratios", "0.01,0.3,1",
    "--timesteps", "0.004",
    "--radius-scales", "1.025,1.05,1.075",
    "--velocity-scales", "1.05,1.1,1.15",
    "--thrust-limits", "0.4",
    "--sensor-noise-sweep", "0",
    "--track-guard-mode", "hazard_quantile",
    "--track-guard-quantile", "0.75",
    "--track-guard-min-radius-sweep", "1.15",
    "--track-guard-max-local-acceleration-sweep", "2.5",
    "--track-guard-max-tidal-magnitude-sweep", "35",
    "--seeds", "8",
    "--duration", "16",
    "--sensor-audit-every", "240",
    "--radius-inward-magnitude", String(m),
  ];
}

function run(m) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, harnessArgs(m), { cwd: repoRoot, stdio: "inherit" });
    child.on("exit", (code, signal) => {
      if (signal) reject(new Error(`m=${m} killed by ${signal}`));
      else if (code !== 0) reject(new Error(`m=${m} exited ${code}`));
      else resolve();
    });
  });
}

function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter(Boolean);
  const cols = lines[0].split(",");
  const ci = Object.fromEntries(cols.map((c, i) => [c, i]));
  return lines.slice(1).map((l) => l.split(",")).map((r) => ({
    mode: r[ci.controller_mode],
    velocityScale: Number(r[ci.velocity_scale]),
    totalDeltaV: Number(r[ci.total_delta_v]),
  }));
}

async function meanDeltaV(outDir, mode) {
  const rows = parseCsv(await readFile(path.join(repoRoot, outDir, "trial-outcomes.csv"), "utf8"));
  const sel = rows.filter((r) => r.mode === mode && r.velocityScale >= 1.05 && Number.isFinite(r.totalDeltaV));
  if (sel.length === 0) return null;
  return sel.reduce((s, r) => s + r.totalDeltaV, 0) / sel.length;
}

async function main() {
  for (const m of MAGNITUDE_GRID) {
    console.log(`[phase18-calibrate] running m=${m} (favorable pocket: guarded + inward)`);
    await run(m);
  }
  // guarded ΔV is magnitude-independent; read from the first run, sanity-check across runs.
  const guardedByRun = [];
  const inwardByM = [];
  for (const m of MAGNITUDE_GRID) {
    const outDir = `${OUT_ROOT}-m${tag(m)}`;
    guardedByRun.push(await meanDeltaV(outDir, "track_sensor_accel_guarded"));
    inwardByM.push({ m, deltaV: await meanDeltaV(outDir, "track_radius_inward") });
  }
  const guardedDeltaV = guardedByRun[0];
  const guardedSpread = Math.max(...guardedByRun) - Math.min(...guardedByRun);
  let matched = inwardByM[0];
  for (const row of inwardByM) {
    if (Math.abs(row.deltaV - guardedDeltaV) < Math.abs(matched.deltaV - guardedDeltaV)) matched = row;
  }
  console.log("");
  console.log("=== Phase 18 matched-duty calibration (favorable pocket mean totalDeltaV) ===");
  console.log(`guarded TRACK ΔV: ${guardedDeltaV.toFixed(6)} (cross-run spread ${guardedSpread.toExponential(2)})`);
  for (const row of inwardByM) {
    console.log(`  inward m=${row.m}: ΔV=${row.deltaV.toFixed(6)}  ratio=${(row.deltaV / guardedDeltaV).toFixed(3)}  |Δ|=${Math.abs(row.deltaV - guardedDeltaV).toFixed(6)}`);
  }
  console.log(`MATCHED-M = ${matched.m}  (ΔV ratio ${(matched.deltaV / guardedDeltaV).toFixed(3)}) — selected on ΔV, blind to survival`);
  const summary = {
    schema: "sundog.threebody.phase18-calibrate.v1",
    magnitudeGrid: MAGNITUDE_GRID,
    guardedFavorableMeanDeltaV: guardedDeltaV,
    guardedCrossRunSpread: guardedSpread,
    inward: inwardByM,
    matchedM: matched.m,
    matchedDeltaVRatio: matched.deltaV / guardedDeltaV,
    selectionRule: "argmin |meanDeltaV(inward,m) - meanDeltaV(guarded)| over favorable pocket; blind to survival",
  };
  await writeFile(path.join(repoRoot, `${OUT_ROOT}-m${tag(MAGNITUDE_GRID[0])}`, "calibrate-summary.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  console.log(`[phase18-calibrate] wrote calibrate-summary.json under ${OUT_ROOT}-m${tag(MAGNITUDE_GRID[0])}`);
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
