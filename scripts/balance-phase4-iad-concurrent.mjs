// scripts/balance-phase4-iad-concurrent.mjs
//
// Fan-out orchestrator for the Phase 4 Balance substrate-leg IAD. Sharded
// by seed-start across N concurrent shard wrappers. Mirrors
// scripts/threebody-phase4-iad-concurrent.mjs structurally; reuses the
// single-cell envelope.csv slate it writes on startup.
//
// Target cell (pinned, lifted verbatim from
// results/balance/phase10-envelope/envelope.csv):
//   near_fall × light_elevation=8
//   cell_id: light_elevation__8__light_8__delay_0__noise_0__drop_0__force_12__rail_2p4__push_4p5__preset_near_fall
//   cellClass=borderline, claimGatePass=true,
//   meanRegretVsSundog=0.267 in the Phase 15 lock.
//
// Usage:
//   node scripts/balance-phase4-iad-concurrent.mjs [--fan-out <N>] [--seeds <count-or-list>]
//        [--particles <N>] [--horizon-seconds <X>] [--out-root <dir>] [--out-logs <dir>] [--force]
//
// Defaults:
//   --fan-out 3                          (matches the three-body IAD envelope on a 4-core box)
//   --seeds 8                            (0..7; slate-symmetric with three-body IAD)
//   --particles 512
//   --horizon-seconds 0.5
//   --out-root results/balance/phase4-iad
//   --out-logs results/balance/phase4-iad/_iad-shard-logs

import { spawn } from "node:child_process";
import { createWriteStream, mkdirSync, existsSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { shardDir, shardManifestPath } from "./balance-phase4-iad-shard.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// Pinned target-cell envelope.csv content. Single-row slate that the
// balance-phase15-bayes-floor.mjs loader (--cell-slate phase10-output) parses
// into exactly one cell. Header and row are lifted from
// results/balance/phase10-envelope/envelope.csv.
const CELL_HEADER = "cell_id,case_id,axis,axis_value,preset,light_elev_deg,delay_ms,delay_steps,noise_sigma,dropout_rate,rail_limit,force_limit,disturbance_mag,cell_class,static_boundary_mechanisms,survival_passive_mean,survival_sundog_mean,survival_naive_mean,survival_oracle_mean,rms_theta_sundog_mean,rms_theta_naive_mean,rms_theta_oracle_mean,recovery_time_after_impulse,sundog_naive_paired_margin_mean,sundog_naive_survival_ratio,sundog_beats_naive_1p5x,oracle_sundog_paired_margin_mean,seed_count,paired_margin_bootstrap_low,paired_margin_bootstrap_high,sundog_saturation_rate_mean,sundog_force_budget_mean,replay_url,naive_replay_url,oracle_replay_url";
const CELL_ROW = "light_elevation__8__light_8__delay_0__noise_0__drop_0__force_12__rail_2p4__push_4p5__preset_near_fall,light_elevation__8__light_8__delay_0__noise_0__drop_0__force_12__rail_2p4__push_4p5,light_elevation,8,near_fall,8,0,0,0,0,2.4,12,4.5,borderline,long_shadow_scaling,0.393333,5.676,0.441667,8,0.110919,0.286793,0.048638,0.149892,5.234333,12.851312,false,2.324,100,4.607343,5.872666,0.059805,4.420853,,,";

function parseArgs(argv) {
  const args = {
    fanOut: 3,
    seeds: [0, 1, 2, 3, 4, 5, 6, 7],
    particles: 512,
    horizonSeconds: 0.5,
    outRoot: "results/balance/phase4-iad",
    outLogs: "results/balance/phase4-iad/_iad-shard-logs",
    cellDir: "results/balance/phase4-iad/_iad-cell",
    force: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    if (flag === "--fan-out") {
      const n = Number.parseInt(value, 10);
      if (!Number.isInteger(n) || n < 1) throw new Error(`--fan-out must be a positive integer`);
      args.fanOut = n; i += 1;
    } else if (flag === "--seeds") {
      if (/^\d+$/.test(value)) {
        const n = Number.parseInt(value, 10);
        if (n < 1) throw new Error(`--seeds count must be ≥ 1`);
        args.seeds = Array.from({ length: n }, (_, k) => k);
      } else {
        args.seeds = value.split(",").map((s) => {
          const n = Number.parseInt(s.trim(), 10);
          if (!Number.isInteger(n) || n < 0) throw new Error(`bad seed value "${s}"`);
          return n;
        });
      }
      i += 1;
    } else if (flag === "--particles") {
      const n = Number.parseInt(value, 10);
      if (!Number.isInteger(n) || n < 31) throw new Error(`--particles must be ≥ 31`);
      args.particles = n; i += 1;
    } else if (flag === "--horizon-seconds") {
      const n = Number.parseFloat(value);
      if (!Number.isFinite(n) || n <= 0) throw new Error(`--horizon-seconds must be positive`);
      args.horizonSeconds = n; i += 1;
    } else if (flag === "--out-root") {
      args.outRoot = value;
      args.outLogs = path.join(value, "_iad-shard-logs");
      args.cellDir = path.join(value, "_iad-cell");
      i += 1;
    } else if (flag === "--out-logs") {
      args.outLogs = value; i += 1;
    } else if (flag === "--cell-dir") {
      args.cellDir = value; i += 1;
    } else if (flag === "--force") {
      args.force = true;
    } else if (flag === "--help" || flag === "-h") {
      printHelpAndExit(0);
    } else {
      throw new Error(`Unknown flag: ${flag}`);
    }
  }
  if (args.seeds.length === 0) throw new Error("--seeds resolved to empty set");
  if (args.fanOut > args.seeds.length) {
    console.warn(`[balance-iad-concurrent] --fan-out ${args.fanOut} > seeds ${args.seeds.length}; capping`);
    args.fanOut = args.seeds.length;
  }
  return args;
}

function printHelpAndExit(code) {
  process.stderr.write(
    `usage: node scripts/balance-phase4-iad-concurrent.mjs [--fan-out <N>] [--seeds <count|list>] [--particles <N>] [--horizon-seconds <X>] [--out-root <dir>] [--out-logs <dir>] [--cell-dir <dir>] [--force]\n`,
  );
  process.exit(code);
}

function ensureDir(rel) {
  const abs = path.resolve(repoRoot, rel);
  mkdirSync(abs, { recursive: true });
  return abs;
}

function ensureCellSlate(cellDirAbs) {
  const csvPath = path.join(cellDirAbs, "envelope.csv");
  if (existsSync(csvPath)) {
    console.log(`[balance-iad-concurrent] cell slate already present at ${path.relative(repoRoot, csvPath)}`);
    return;
  }
  writeFileSync(csvPath, `${CELL_HEADER}\n${CELL_ROW}\n`, "utf8");
  console.log(`[balance-iad-concurrent] wrote single-cell slate to ${path.relative(repoRoot, csvPath)}`);
}

// One signal handler per process; tracks active children.
const _activeChildren = new Set();
let _signalsRegistered = false;
function _registerSignalHandlers() {
  if (_signalsRegistered) return;
  _signalsRegistered = true;
  const forward = (sig) => () => {
    for (const c of _activeChildren) {
      if (!c.killed) c.kill(sig);
    }
  };
  process.once("SIGINT", forward("SIGINT"));
  process.once("SIGTERM", forward("SIGTERM"));
}
function trackChild(child) {
  _registerSignalHandlers();
  _activeChildren.add(child);
  child.once("exit", () => _activeChildren.delete(child));
  return child;
}

function spawnShard({ seed, particles, horizonSeconds, outRoot, cellDir, outLogsAbs, force }) {
  return new Promise((resolve) => {
    const manifestAbs = path.resolve(repoRoot, shardManifestPath(outRoot, seed));
    if (!force && existsSync(manifestAbs)) {
      console.log(`[seed=${seed}] manifest exists, skipping (--force to re-run)`);
      const stamp = new Date().toISOString();
      resolve({ seed, exitCode: 0, signal: null, wallSeconds: 0, skipped: true, startedAt: stamp, finishedAt: stamp });
      return;
    }

    const logPath = path.join(outLogsAbs, `iad-shard-seed_${seed}.log`);
    const logStream = createWriteStream(logPath, { flags: "w" });
    const startedAt = new Date();
    const startedMs = Date.now();

    const wrapperArgs = [
      "scripts/balance-phase4-iad-shard.mjs",
      "--seed", String(seed),
      "--particles", String(particles),
      "--horizon-seconds", String(horizonSeconds),
      "--out-root", outRoot,
      "--cell-dir", cellDir,
    ];
    if (force) wrapperArgs.push("--force");

    logStream.write(`# balance-iad-shard seed=${seed} particles=${particles} horizon=${horizonSeconds}s\n`);
    logStream.write(`# started ${startedAt.toISOString()}\n`);
    logStream.write(`# command: ${process.execPath} ${wrapperArgs.join(" ")}\n`);
    logStream.write(`# ─────────────────────────────────────────────────────────────\n`);

    console.log(`[seed=${seed}] spawn ▸ log=${path.relative(repoRoot, logPath)}`);

    const child = trackChild(spawn(process.execPath, wrapperArgs, {
      cwd: repoRoot,
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, NODE_NO_WARNINGS: "1" },
    }));
    child.stdout.pipe(logStream, { end: false });
    child.stderr.pipe(logStream, { end: false });

    child.on("exit", (code, signal) => {
      const finishedAt = new Date();
      const wallSeconds = (Date.now() - startedMs) / 1000;
      logStream.write(`# ─────────────────────────────────────────────────────────────\n`);
      logStream.write(`# finished ${finishedAt.toISOString()} exit=${code} signal=${signal} wall=${wallSeconds.toFixed(2)}s\n`);
      logStream.end();
      console.log(`[seed=${seed}] exit=${code ?? "null"} signal=${signal ?? "-"} wall=${(wallSeconds / 60).toFixed(1)} min`);
      resolve({ seed, exitCode: code, signal, wallSeconds, skipped: false, startedAt: startedAt.toISOString(), finishedAt: finishedAt.toISOString() });
    });
  });
}

async function runPool(seeds, fanOut, spawnOpts) {
  const queue = seeds.slice();
  const results = [];
  let active = 0;
  return await new Promise((resolveAll, rejectAll) => {
    const launch = () => {
      while (active < fanOut && queue.length > 0) {
        const seed = queue.shift();
        active += 1;
        spawnShard({ ...spawnOpts, seed }).then((r) => {
          results.push(r);
          active -= 1;
          if (queue.length === 0 && active === 0) resolveAll(results);
          else launch();
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

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  ensureDir(opts.outRoot);
  const cellDirAbs = ensureDir(opts.cellDir);
  const outLogsAbs = ensureDir(opts.outLogs);
  ensureCellSlate(cellDirAbs);

  console.log(`[balance-iad-concurrent] seeds=${opts.seeds.join(",")} fan-out=${opts.fanOut} particles=${opts.particles} horizon=${opts.horizonSeconds}s`);
  console.log(`[balance-iad-concurrent] out-root=${opts.outRoot}`);
  console.log(`[balance-iad-concurrent] logs=${path.relative(repoRoot, outLogsAbs)}`);

  const orchStart = Date.now();
  const results = await runPool(opts.seeds, opts.fanOut, {
    particles: opts.particles,
    horizonSeconds: opts.horizonSeconds,
    outRoot: opts.outRoot,
    cellDir: opts.cellDir,
    outLogsAbs,
    force: opts.force,
  });
  const totalWall = (Date.now() - orchStart) / 1000;

  results.sort((a, b) => a.seed - b.seed);
  const ran = results.filter((r) => !r.skipped);
  const skipped = results.filter((r) => r.skipped);
  const failed = results.filter((r) => r.exitCode !== 0 && !r.skipped);

  console.log("");
  console.log("─── Balance IAD concurrent shard summary ───────────────────────");
  console.log(`fan-out=${opts.fanOut}  particles=${opts.particles}  horizon=${opts.horizonSeconds}s  seeds=${opts.seeds.length}  ran=${ran.length}  skipped=${skipped.length}  failed=${failed.length}`);
  console.log(`total wall (orchestrator): ${fmtSecs(totalWall)}`);
  console.log("");
  for (const r of results) {
    const mark = r.skipped ? "skipped" : r.exitCode === 0 ? "ok" : `EXIT=${r.exitCode}`;
    console.log(`  seed=${String(r.seed).padEnd(2)} ${fmtSecs(r.wallSeconds).padStart(10)}  ${mark}`);
  }
  console.log("");
  console.log("─── next step ─────────────────────────────────────────────────");
  console.log(`node scripts/balance-phase4-iad-merge.mjs --shards-root ${opts.outRoot}`);
  console.log(`(then: balance-phase4-iad-regret.mjs --in ${opts.outRoot}/_iad-merged ...)`);

  process.exit(failed.length > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error(`[balance-iad-concurrent] fatal: ${err.stack || err.message}`);
  process.exit(2);
});
