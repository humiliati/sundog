#!/usr/bin/env node
// Generate JS fixtures for Python parity against scripts/h4-distributed-world-model-task.mjs.

import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import {
  DistributedRelayEnv,
  H4_RELAY_CELL_DEFS,
  H4_RELAY_DEFAULTS,
  makeH4Controller,
  publicObservationHasHiddenLatents,
} from "./h4-distributed-world-model-task.mjs";

function parseArgs(argv) {
  const args = {
    out: "results/mesa/h4-topology/h4_0_parity/fixtures.json",
    seeds: "10000,10001,10002",
    cells: "nominal-relay,stale-relay,decoy-relay",
    controls: "Oracle-H4,Field-H4,Reward-H4,Blind-H4,CurrentObs-H4,FullHistory-H4,Bottleneck-H4,WideMessage-H4",
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    if (!flag.startsWith("--")) continue;
    args[flag.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = argv[i + 1];
    i += 1;
  }
  return args;
}

function runTrace({ cell, control, seed }) {
  if (!(cell in H4_RELAY_CELL_DEFS)) throw new Error(`unknown H4 cell ${cell}`);
  const env = new DistributedRelayEnv();
  const cellOverrides = H4_RELAY_CELL_DEFS[cell];
  const initialObs = env.reset(seed, cellOverrides);
  const initialHidden = env.hiddenState();
  if (publicObservationHasHiddenLatents(initialObs)) {
    throw new Error(`public observation leaks hidden latents in ${cell} seed ${seed}`);
  }
  const ctrl = makeH4Controller(control, env, seed);
  const trace = [];
  let done = false;
  while (!done) {
    const obs = env.observe();
    if (publicObservationHasHiddenLatents(obs)) throw new Error(`hidden latent leaked at t=${obs.t}`);
    const hidden = env.hiddenState();
    const messages1 = env.localMessages(1);
    const messages4 = env.localMessages(4);
    const action = ctrl.act(env, obs);
    const step = env.step(action);
    done = step.done;
    trace.push({
      t: obs.t,
      phase: obs.phase,
      tick_in_gate: obs.tick_in_gate,
      obs,
      hidden,
      messages1,
      messages4,
      action,
      step: {
        done: step.done,
        evaluated: step.evaluated,
        action: step.action,
      },
      after_hidden: env.hiddenState(),
      after_metrics: env.metrics(),
    });
  }
  return {
    cell,
    cellOverrides,
    control,
    seed,
    initialObs,
    initialHidden,
    final: env.metrics(),
    trace,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const seeds = args.seeds.split(",").filter(Boolean).map((value) => Number(value));
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
    generator: "scripts/mesa-h4-topology-fixtures.mjs",
    defaults: H4_RELAY_DEFAULTS,
    cells,
    seeds,
    controls,
    episodes,
  };
  await mkdir(path.dirname(args.out), { recursive: true });
  await writeFile(args.out, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");
  console.log(`wrote ${episodes.length} H4 topology parity episodes to ${args.out}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
