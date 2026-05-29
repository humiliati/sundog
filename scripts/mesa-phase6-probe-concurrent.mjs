// scripts/mesa-phase6-probe-concurrent.mjs
//
// Fan-out probe runner for Phase 6 lambda-control. Spawns N concurrent
// `train_ppo` shards under per-shard thread caps, captures stdout/stderr to
// per-shard log files, and prints a summary table on completion: wall-clock
// per shard, naive seconds-per-update, and an extrapolated full-lock estimate.
//
// Purpose: empirical sizing for the lock fan-out decision (see PHASE6 spec ▸
// Capped Probe and "Recommended path" in the planning thread). 2-wide and
// 3-wide give different per-shard rates depending on whether mesa env stepping
// or PyTorch BLAS dominates; this script measures it.
//
// Usage:
//   node scripts/mesa-phase6-probe-concurrent.mjs [--fan-out <N>] [--thread-cap <N>]
//                                                 [--rows <labels>] [--out-logs <dir>]
//
// Defaults:
//   --fan-out 2                            (Leg 1 packaging)
//   --thread-cap 1                         (conservative, optimized for fan-out)
//   --rows <PROBE_LABELS joined by ,>      (both probe rows from the spec)
//   --out-logs results/proof/phase6/logs-concurrent-probe
//
// Notes:
//   * Each shard writes its policy to `results/proof/phase6/training-probe/`
//     under a unique filename; rerunning with --force isn't necessary — probes
//     are cheap, but the wrapper's resume-skip will short-circuit completed
//     ones. Pass --force to bypass the per-shard resume guard.
//   * The extrapolation is naive: `(wall / updates) * 305`. It overestimates
//     because probe wall includes one-shot init + an eval pass that doesn't
//     scale linearly with updates. Treat as an upper bound for the lock.

import { spawn } from "node:child_process";
import { createWriteStream, mkdirSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  PROBE_LABELS,
  getRow,
  buildTrainArgs,
  buildShardEnv,
  policyPath,
  pythonExec,
  trackChild,
  META,
} from "./lib/mesa-phase6-rows.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const args = {
    fanOut: 2,
    threadCap: 1,
    rows: PROBE_LABELS.slice(),
    outLogs: "results/proof/phase6/logs-concurrent-probe",
    force: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    if (flag === "--fan-out") {
      const n = Number(value);
      if (!Number.isInteger(n) || n < 1) throw new Error(`--fan-out must be a positive integer (got "${value}")`);
      args.fanOut = n;
      i += 1;
    } else if (flag === "--thread-cap") {
      const n = Number(value);
      if (!Number.isInteger(n) || n < 1) throw new Error(`--thread-cap must be a positive integer (got "${value}")`);
      args.threadCap = n;
      i += 1;
    } else if (flag === "--rows") {
      args.rows = value.split(",").map((s) => s.trim()).filter(Boolean);
      i += 1;
    } else if (flag === "--out-logs") {
      args.outLogs = value;
      i += 1;
    } else if (flag === "--force") {
      args.force = true;
    } else if (flag === "--help" || flag === "-h") {
      printHelpAndExit(0);
    } else {
      throw new Error(`Unknown flag: ${flag}`);
    }
  }
  if (args.rows.length === 0) throw new Error("--rows produced an empty set");
  args.rows.forEach((label) => getRow(label)); // validate
  if (args.fanOut > args.rows.length) {
    console.warn(`[concurrent-probe] --fan-out ${args.fanOut} > rows ${args.rows.length}; capping to ${args.rows.length}`);
    args.fanOut = args.rows.length;
  }
  return args;
}

function printHelpAndExit(code) {
  process.stderr.write(
    `usage: node scripts/mesa-phase6-probe-concurrent.mjs [--fan-out <N>] [--thread-cap <N>] [--rows <labels>] [--out-logs <dir>] [--force]\n` +
    `default rows: ${PROBE_LABELS.join(",")}\n`,
  );
  process.exit(code);
}

function ensureDir(rel) {
  const abs = path.resolve(repoRoot, rel);
  mkdirSync(abs, { recursive: true });
  return abs;
}

// Spawn one shard. Returns a Promise that resolves with the shard's result
// {label, exitCode, signal, startedAt, finishedAt, wallSeconds, skipped}.
// Stdout + stderr are piped to <outLogs>/<label>.log.
function spawnShard({ label, threadCap, outLogsAbs, force }) {
  return new Promise((resolve) => {
    const row = getRow(label);
    const policyAbs = path.resolve(repoRoot, policyPath(row));

    if (!force && existsSync(policyAbs)) {
      const stamp = new Date().toISOString();
      console.log(`[${label}] policy exists, skipping (--force to re-run)`);
      resolve({
        label,
        exitCode: 0,
        signal: null,
        startedAt: stamp,
        finishedAt: stamp,
        wallSeconds: 0,
        skipped: true,
        updates: Number(META.MODE_CONFIG[row.mode].updates),
      });
      return;
    }

    const args = buildTrainArgs(row, repoRoot);
    const env = buildShardEnv(process.env, threadCap);
    const exec = pythonExec();
    const logPath = path.join(outLogsAbs, `${label}.log`);
    const logStream = createWriteStream(logPath, { flags: "w" });
    const startedAt = new Date();
    const startedMs = Date.now();

    logStream.write(`# shard ${label}\n`);
    logStream.write(`# started ${startedAt.toISOString()}\n`);
    logStream.write(`# command: ${exec} ${args.join(" ")}\n`);
    logStream.write(`# env: OMP_NUM_THREADS=${threadCap} MKL_NUM_THREADS=${threadCap} OPENBLAS_NUM_THREADS=${threadCap}\n`);
    logStream.write(`# ─────────────────────────────────────────────────────────────\n`);

    console.log(`[${label}] spawn pid will be set ▸ log=${path.relative(repoRoot, logPath)}`);

    const child = trackChild(spawn(exec, args, {
      cwd: repoRoot,
      stdio: ["ignore", "pipe", "pipe"],
      env,
    }));
    child.stdout.pipe(logStream, { end: false });
    child.stderr.pipe(logStream, { end: false });
    console.log(`[${label}] pid=${child.pid}`);

    child.on("exit", (code, signal) => {
      const finishedAt = new Date();
      const wallSeconds = (Date.now() - startedMs) / 1000;
      logStream.write(`# ─────────────────────────────────────────────────────────────\n`);
      logStream.write(`# finished ${finishedAt.toISOString()} exit=${code} signal=${signal} wall=${wallSeconds.toFixed(2)}s\n`);
      logStream.end();
      console.log(`[${label}] exit=${code ?? "null"} signal=${signal ?? "-"} wall=${wallSeconds.toFixed(1)} s`);
      resolve({
        label,
        exitCode: code,
        signal,
        startedAt: startedAt.toISOString(),
        finishedAt: finishedAt.toISOString(),
        wallSeconds,
        skipped: false,
        updates: Number(META.MODE_CONFIG[row.mode].updates),
      });
    });
  });
}

// Simple worker pool: keep at most `fanOut` shards in flight; pull from queue
// when slots free. Returns array of results in completion order.
async function runPool(rows, fanOut, spawnOpts) {
  const queue = rows.slice();
  const results = [];
  let active = 0;
  return await new Promise((resolveAll, rejectAll) => {
    const launch = () => {
      while (active < fanOut && queue.length > 0) {
        const label = queue.shift();
        active += 1;
        spawnShard({ ...spawnOpts, label }).then((r) => {
          results.push(r);
          active -= 1;
          if (queue.length === 0 && active === 0) {
            resolveAll(results);
          } else {
            launch();
          }
        }).catch(rejectAll);
      }
    };
    launch();
  });
}

function fmtSecs(s) {
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = s / 60;
  if (m < 60) return `${m.toFixed(1)} min`;
  return `${(m / 60).toFixed(2)} h`;
}

function printSummary({ results, fanOut, threadCap, totalWall }) {
  const ran = results.filter((r) => !r.skipped);
  const skipped = results.filter((r) => r.skipped);
  console.log("");
  console.log("─── concurrent probe summary ──────────────────────────────────────────");
  console.log(`fan-out=${fanOut}  thread-cap=${threadCap}  rows=${results.length}  ran=${ran.length}  skipped=${skipped.length}`);
  console.log(`total wall (orchestrator): ${fmtSecs(totalWall)}`);
  console.log("");
  if (ran.length > 0) {
    console.log(" label                                       wall      sec/update  extrap 305 upd (naive)  exit");
    console.log(" ───────────────────────────────────────── ────────── ─────────── ──────────────────────── ────");
    for (const r of ran) {
      const secPerUpdate = r.wallSeconds / r.updates;
      const extrap = secPerUpdate * 305;
      const padLabel = r.label.padEnd(41);
      const padWall = fmtSecs(r.wallSeconds).padStart(10);
      const padSpu = `${secPerUpdate.toFixed(2)} s`.padStart(11);
      const padExtrap = fmtSecs(extrap).padStart(24);
      console.log(` ${padLabel} ${padWall} ${padSpu} ${padExtrap}  ${r.exitCode ?? "?"}`);
    }
  }
  if (skipped.length > 0) {
    console.log("");
    console.log(" skipped (policy exists, use --force to re-run):");
    for (const r of skipped) console.log(`   ${r.label}`);
  }
  if (ran.length > 0) {
    const meanSpu = ran.reduce((a, r) => a + r.wallSeconds / r.updates, 0) / ran.length;
    const fullLockMeanRow = meanSpu * 305;
    const ROWS_IN_LOCK = 6;
    const sequentialLock = fullLockMeanRow * ROWS_IN_LOCK;
    const concurrentLock = fullLockMeanRow * Math.ceil(ROWS_IN_LOCK / fanOut);
    const speedup = sequentialLock / concurrentLock;
    console.log("");
    console.log("─── implied lock planning estimates ───────────────────────────────────");
    console.log(`naive per-row (mean s/update × 305):   ${fmtSecs(fullLockMeanRow)}`);
    console.log(`sequential 6-row lock estimate:        ${fmtSecs(sequentialLock)}`);
    console.log(`${fanOut}-wide concurrent 6-row lock:  ${fmtSecs(concurrentLock)} (~${speedup.toFixed(1)}× speedup)`);
    console.log("note: naive — overcounts init+eval. Real lock will be faster than the row estimate.");
    console.log("note: add per-row probe-slate + intervention-battery time (~few min each) on top.");
  }
  console.log("───────────────────────────────────────────────────────────────────────");
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const outLogsAbs = ensureDir(opts.outLogs);

  console.log(`[concurrent-probe] rows=${opts.rows.join(",")} fan-out=${opts.fanOut} thread-cap=${opts.threadCap}`);
  console.log(`[concurrent-probe] logs dir: ${path.relative(repoRoot, outLogsAbs)}`);

  const orchStart = Date.now();
  const results = await runPool(opts.rows, opts.fanOut, {
    threadCap: opts.threadCap,
    outLogsAbs,
    force: opts.force,
  });
  const totalWall = (Date.now() - orchStart) / 1000;

  printSummary({ results, fanOut: opts.fanOut, threadCap: opts.threadCap, totalWall });

  const worstExit = results.reduce(
    (m, r) => Math.max(m, r.exitCode == null ? 1 : r.exitCode),
    0,
  );
  process.exit(worstExit);
}

main().catch((err) => {
  console.error(`[concurrent-probe] fatal: ${err.stack || err.message}`);
  process.exit(2);
});
