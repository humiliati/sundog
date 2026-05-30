// scripts/threebody-phase4-iad-concurrent.mjs
//
// Fan-out orchestrator for the Phase 4 BF-4b Information-Accessibility
// Diagnostic. Sharded by seed-start across N concurrent shard wrappers.
//
// Spawns one `threebody-phase4-iad-shard.mjs --seed <S>` child per seed,
// with at most --fan-out children in flight at a time. Captures per-shard
// stdout/stderr to a log file; the orchestrator's own stdout shows lifecycle
// events only.
//
// Usage:
//   node scripts/threebody-phase4-iad-concurrent.mjs [--fan-out <N>] [--seeds <list-or-count>]
//                                                     [--particles <N>] [--out-root <dir>]
//                                                     [--out-logs <dir>] [--force]
//
// Defaults:
//   --fan-out 3                     (4-core box; matches Phase 6 envelope; leaves 1 core for the OS)
//   --seeds 8                       (0..7; matches spec command's `--seeds 8` default)
//                                   pass a comma-list like "0,1,2" for a subset
//   --particles 512                 (spec gold)
//   --out-root results/proof/phase4
//   --out-logs results/proof/phase4/_iad-shard-logs

import { spawn } from "node:child_process";
import { createWriteStream, mkdirSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { shardDir, shardManifestPath } from "./threebody-phase4-iad-shard.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const args = {
    fanOut: 3,
    seeds: [0, 1, 2, 3, 4, 5, 6, 7],
    particles: 512,
    outRoot: "results/proof/phase4",
    outLogs: "results/proof/phase4/_iad-shard-logs",
    force: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    if (flag === "--fan-out") {
      const n = Number.parseInt(value, 10);
      if (!Number.isInteger(n) || n < 1) throw new Error(`--fan-out must be a positive integer (got "${value}")`);
      args.fanOut = n; i += 1;
    } else if (flag === "--seeds") {
      // Either a count "8" (→ 0..7) or a comma-list "0,1,2"
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
      if (!Number.isInteger(n) || n < 1) throw new Error(`--particles must be a positive integer (got "${value}")`);
      args.particles = n; i += 1;
    } else if (flag === "--out-root") {
      args.outRoot = value; i += 1;
    } else if (flag === "--out-logs") {
      args.outLogs = value; i += 1;
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
    console.warn(`[iad-concurrent] --fan-out ${args.fanOut} > seeds ${args.seeds.length}; capping`);
    args.fanOut = args.seeds.length;
  }
  return args;
}

function printHelpAndExit(code) {
  process.stderr.write(
    `usage: node scripts/threebody-phase4-iad-concurrent.mjs [--fan-out <N>] [--seeds <count|list>] [--particles <N>] [--out-root <dir>] [--out-logs <dir>] [--force]\n`,
  );
  process.exit(code);
}

function ensureDir(rel) {
  const abs = path.resolve(repoRoot, rel);
  mkdirSync(abs, { recursive: true });
  return abs;
}

// Process lifecycle tracking — one signal handler at the process level
// (mirror of mesa-phase6-rows.mjs ▸ trackChild but local to keep this script
// self-contained; could be promoted to a generic lib later).
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

// Spawn one shard. Returns Promise<{seed, exitCode, signal, wallSeconds, skipped}>.
function spawnShard({ seed, particles, outRoot, outLogsAbs, force }) {
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
      "scripts/threebody-phase4-iad-shard.mjs",
      "--seed", String(seed),
      "--particles", String(particles),
      "--out-root", outRoot,
    ];
    if (force) wrapperArgs.push("--force");

    logStream.write(`# iad-shard seed=${seed} particles=${particles}\n`);
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
  const outLogsAbs = ensureDir(opts.outLogs);
  ensureDir(opts.outRoot);

  console.log(`[iad-concurrent] seeds=${opts.seeds.join(",")} fan-out=${opts.fanOut} particles=${opts.particles}`);
  console.log(`[iad-concurrent] out-root=${opts.outRoot}`);
  console.log(`[iad-concurrent] logs dir=${path.relative(repoRoot, outLogsAbs)}`);

  const orchStart = Date.now();
  const results = await runPool(opts.seeds, opts.fanOut, {
    particles: opts.particles,
    outRoot: opts.outRoot,
    outLogsAbs,
    force: opts.force,
  });
  const totalWall = (Date.now() - orchStart) / 1000;

  results.sort((a, b) => a.seed - b.seed);
  const ran = results.filter((r) => !r.skipped);
  const skipped = results.filter((r) => r.skipped);
  const failed = results.filter((r) => r.exitCode !== 0 && !r.skipped);

  console.log("");
  console.log("─── IAD concurrent shard summary ───────────────────────────────");
  console.log(`fan-out=${opts.fanOut}  particles=${opts.particles}  seeds=${opts.seeds.length}  ran=${ran.length}  skipped=${skipped.length}  failed=${failed.length}`);
  console.log(`total wall (orchestrator): ${fmtSecs(totalWall)}`);
  console.log("");
  for (const r of results) {
    const mark = r.skipped ? "skipped" : r.exitCode === 0 ? "ok" : `EXIT=${r.exitCode}`;
    console.log(`  seed=${String(r.seed).padEnd(2)} ${fmtSecs(r.wallSeconds).padStart(10)}  ${mark}`);
  }
  if (failed.length > 0) {
    console.log("");
    console.log(`[iad-concurrent] ${failed.length} shard(s) failed; investigate logs before merging`);
  }
  console.log("");
  console.log("─── next step ─────────────────────────────────────────────────");
  console.log(`node scripts/threebody-phase4-iad-merge.mjs --shards-root ${opts.outRoot}`);
  console.log(`(then: threebody-phase4-regret.mjs --bayes-in ${opts.outRoot}/_bf4b-accessibility ...)`);

  process.exit(failed.length > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error(`[iad-concurrent] fatal: ${err.stack || err.message}`);
  process.exit(2);
});
