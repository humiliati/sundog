#!/usr/bin/env node
// JS/Python parity smoke for the released non-sovereignty task family.

import { spawnSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import {
  ShutdownForkedFieldEnv,
  NS1_CELL_DEFS,
  cappedNoRoleController,
  structuralShutdownController,
  rollShutdownEpisode,
} from "./ns1-shutdown-task.mjs";

const repoRoot = process.cwd();
const args = {
  out: "results/mesa/non-sovereignty/release_smoke/parity.json",
  python: process.env.PYTHON ?? "python",
};
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i += 1) {
  const flag = argv[i];
  const value = argv[i + 1];
  if (flag === "--out") { args.out = value; i += 1; }
  else if (flag === "--python") { args.python = value; i += 1; }
}

const fixtures = [
  { cell: "nominal", seed: 10000 },
  { cell: "wide-fork", seed: 10001 },
  { cell: "far-lure", seed: 10002 },
  { cell: "strong-lure", seed: 10003 },
];
const config = {
  reviewBandEnabled: true,
  shutdownEnabled: false,
  shutdownFloorP: 0.1,
  bandYLo: 0.0,
  bandYHi: 0.7,
  bandXHalf: 2.0,
  corrK: 1,
};

const round = (x, n = 12) => (typeof x === "number" && Number.isFinite(x) ? Number(x.toFixed(n)) : x);
const roundArray = (xs) => xs.map((x) => round(x));

function normalizeMetrics(metrics) {
  return Object.fromEntries(Object.entries(metrics).map(([key, value]) => [
    key,
    typeof value === "number" ? round(value) : value,
  ]));
}

function jsFixture({ cell, seed }) {
  const env = new ShutdownForkedFieldEnv(config);
  const obs = env.reset(seed, NS1_CELL_DEFS[cell]);
  const init = {
    cell,
    seed,
    correct: env.correct,
    x: roundArray(env.x),
    t: env.t,
    samples: roundArray(obs.samples),
    sLocal: round(obs.sLocal),
    field_action: roundArray(env.fieldProposal().map((v) => v * env.cfg.actionMax)),
    reward_action: roundArray(env.rewardProposal().map((v) => v * env.cfg.actionMax)),
    shutdown_signal: env.shutdownSignal(),
    in_band_start: env.inBand(env.x) ? 1 : 0,
  };
  const envRoll = new ShutdownForkedFieldEnv(config);
  const make = structuralShutdownController((e) => cappedNoRoleController(e, 0.5));
  const metrics = normalizeMetrics(rollShutdownEpisode(envRoll, make, seed, NS1_CELL_DEFS[cell], { corrK: 1 }));
  return { init, metrics };
}

const pythonProgram = String.raw`
import json
import sys
from training.mesa.ns1_shutdown_task import (
    ShutdownForkedFieldEnv,
    NS1_CELL_DEFS,
    CappedNoRoleController,
    StructuralShutdownController,
    roll_shutdown_episode,
)

payload = json.load(sys.stdin)
fixtures = payload["fixtures"]
config = payload["config"]

def rnd(x, n=12):
    if isinstance(x, float):
        return round(x, n)
    return x

def arr(xs):
    return [rnd(float(x)) for x in xs]

def norm_metrics(m):
    out = {}
    for k, v in m.items():
        out[k] = rnd(v) if isinstance(v, float) else v
    return out

def fixture(cell, seed):
    env = ShutdownForkedFieldEnv(config)
    obs = env.reset(seed, NS1_CELL_DEFS[cell])
    init = {
        "cell": cell,
        "seed": seed,
        "correct": env.correct,
        "x": arr(env.x),
        "t": env.t,
        "samples": arr(obs["samples"]),
        "sLocal": rnd(float(obs["sLocal"])),
        "field_action": arr(env.field_action()),
        "reward_action": arr(env.reward_action()),
        "shutdown_signal": env.shutdown_signal(),
        "in_band_start": 1 if env.in_band(env.x) else 0,
    }
    env_roll = ShutdownForkedFieldEnv(config)
    metrics = norm_metrics(
        roll_shutdown_episode(
            env_roll,
            StructuralShutdownController(CappedNoRoleController(0.5)),
            seed,
            NS1_CELL_DEFS[cell],
            corr_k=1,
        )
    )
    return {"init": init, "metrics": metrics}

print(json.dumps([fixture(f["cell"], int(f["seed"])) for f in fixtures], sort_keys=True))
`;

function compare(left, right, loc = "$", diffs = []) {
  if (typeof left === "number" && typeof right === "number") {
    if (Math.abs(left - right) > 1e-9) diffs.push(`${loc}: ${left} != ${right}`);
    return diffs;
  }
  if (Array.isArray(left) || Array.isArray(right)) {
    if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) {
      diffs.push(`${loc}: array shape mismatch`);
      return diffs;
    }
    for (let i = 0; i < left.length; i += 1) compare(left[i], right[i], `${loc}[${i}]`, diffs);
    return diffs;
  }
  if (left && right && typeof left === "object" && typeof right === "object") {
    const keys = [...new Set([...Object.keys(left), ...Object.keys(right)])].sort();
    for (const key of keys) {
      if (!(key in left) || !(key in right)) diffs.push(`${loc}.${key}: missing key`);
      else compare(left[key], right[key], `${loc}.${key}`, diffs);
    }
    return diffs;
  }
  if (left !== right) diffs.push(`${loc}: ${left} != ${right}`);
  return diffs;
}

const jsRows = fixtures.map(jsFixture);
const py = spawnSync(args.python, ["-c", pythonProgram], {
  cwd: repoRoot,
  input: JSON.stringify({ fixtures, config }),
  encoding: "utf8",
});
if (py.status !== 0) {
  console.error(py.stdout);
  console.error(py.stderr);
  throw new Error(`Python parity fixture failed with exit ${py.status}`);
}
const pyRows = JSON.parse(py.stdout);
const diffs = compare(jsRows, pyRows);
const summary = {
  phase: "non-sovereignty release JS/Python parity smoke",
  generated_at: new Date().toISOString(),
  fixtures,
  config,
  tolerance: 1e-9,
  passed: diffs.length === 0,
  diff_count: diffs.length,
  diffs: diffs.slice(0, 40),
  js_rows: jsRows,
  python_rows: pyRows,
};
mkdirSync(path.resolve(repoRoot, path.dirname(args.out)), { recursive: true });
writeFileSync(path.resolve(repoRoot, args.out), `${JSON.stringify(summary, null, 2)}\n`, "utf8");
if (diffs.length) {
  console.error(`JS/Python parity failed with ${diffs.length} diffs.`);
  for (const diff of diffs.slice(0, 40)) console.error(`  - ${diff}`);
  console.error(`Wrote ${args.out}`);
  process.exit(1);
}
console.log(`JS/Python parity OK (${fixtures.length} fixtures). Wrote ${args.out}`);
