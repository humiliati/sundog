// Generate JS fixtures for Python parity against scripts/h2-multifork-task.mjs.

import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import {
  MultiForkEnv,
  H2_MULTIFORK_DEFAULTS,
  H2_MULTIFORK_CELL_DEFS,
  blindController,
  fieldFollower,
  magGatedController,
  oracleController,
  rewardFollower,
} from "./h2-multifork-task.mjs";
import { makeRng, splitSeed } from "../public/js/mesa-core.mjs";

function parseArgs(argv) {
  const args = {
    out: "results/mesa/h2-frontier/h2_2_parity/fixtures.json",
    seeds: "10000,10001,10002,10003",
    cells: "nominal,spaced,narrow",
    controls: "Oracle-H2.2,P-Field-H2.2,P-Reward-H2.2,Blind-H2.2,Gated-H2.2",
  };
  for (let i = 0; i < argv.length; i += 1) {
    const f = argv[i];
    if (!f.startsWith("--")) continue;
    args[f.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = argv[i + 1];
    i += 1;
  }
  return args;
}

function makeController(label, env, seed) {
  const rng = makeRng(splitSeed(seed, "h2-mf-ctrl"));
  if (label === "Oracle-H2.2") return oracleController(env);
  if (label === "P-Field-H2.2") return fieldFollower(env, rng);
  if (label === "P-Reward-H2.2") return rewardFollower(env);
  if (label === "Blind-H2.2") return blindController(env, rng);
  if (label === "Gated-H2.2") return magGatedController(env, 0.6);
  throw new Error(`unknown control ${label}`);
}

function runTrace({ cell, control, seed }) {
  if (!(cell in H2_MULTIFORK_CELL_DEFS)) throw new Error(`unknown cell ${cell}`);
  const env = new MultiForkEnv();
  const cellOverrides = H2_MULTIFORK_CELL_DEFS[cell];
  const initialObs = env.reset(seed, cellOverrides);
  const initialKey = env.key.slice();
  const ctrl = makeController(control, env, seed);
  const trace = [];
  let done = false;
  while (!done) {
    const obs = env.observe();
    const fieldProposal = env.fieldProposal();
    const rewardProposal = env.rewardProposal();
    const rewardMagnitude = env.rewardMagnitude();
    const phase = env.phase;
    const action = ctrl.act(obs);
    const step = env.step(action);
    done = step.done;
    trace.push({
      t: obs.t,
      x: obs.x,
      phase,
      obs,
      fieldProposal,
      rewardProposal,
      rewardMagnitude,
      action,
      afterX: [env.x, env.y],
      done,
      outcome: env.outcome,
      metrics: env.metrics(),
    });
  }
  return {
    cell,
    cellOverrides,
    control,
    seed,
    initialKey,
    initialObs,
    final: env.metrics(),
    trace,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const seeds = args.seeds.split(",").filter(Boolean).map((v) => Number(v));
  const cells = args.cells.split(",").filter(Boolean);
  const controls = args.controls.split(",").filter(Boolean);
  const episodes = [];
  for (const cell of cells) {
    for (const seed of seeds) {
      for (const control of controls) {
        episodes.push(runTrace({ cell, control, seed }));
      }
    }
  }
  const payload = {
    generatedAt: new Date().toISOString(),
    generator: "scripts/mesa-h2-2-multifork-fixtures.mjs",
    defaults: H2_MULTIFORK_DEFAULTS,
    cells,
    seeds,
    controls,
    episodes,
  };
  await mkdir(path.dirname(args.out), { recursive: true });
  await writeFile(args.out, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");
  console.log(`wrote ${episodes.length} H2.2 multi-fork parity episodes to ${args.out}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
