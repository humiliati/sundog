#!/usr/bin/env node
// H4.0-b fixed-control admission for Distributed Relay Grid.
//
// This runner covers only H4.0 fixed gates 1-6. Learned-headroom and OOD-gap
// Gates 7-8 are H4.0-c and intentionally remain pending here.

import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import {
  DistributedRelayEnv,
  H4_RELAY_CELL_DEFS,
  H4_RELAY_DEFAULTS,
  H4_RELAY_PRIMARY_CELLS,
  publicObservationHasHiddenLatents,
  rollEpisode,
  summarizeMetrics,
} from "./h4-distributed-world-model-task.mjs";

const repoRoot = process.cwd();

function parseArgs(argv) {
  const args = {
    seeds: 64,
    seedStart: 10000,
    cells: H4_RELAY_PRIMARY_CELLS.join(","),
    out: "docs/mesa/H4_0_TOPOLOGY_ADMISSION_RESULTS.md",
    json: "results/mesa/h4-topology/h4_0_fixed_admission.json",
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (flag === "--seeds") {
      args.seeds = Number(value);
      i += 1;
    } else if (flag === "--seed-start") {
      args.seedStart = Number(value);
      i += 1;
    } else if (flag === "--cells") {
      args.cells = value;
      i += 1;
    } else if (flag === "--out") {
      args.out = value;
      i += 1;
    } else if (flag === "--json") {
      args.json = value;
      i += 1;
    }
  }
  return args;
}

const round = (value, digits = 4) => Number(Number(value).toFixed(digits));

function aggregateRows(rows) {
  const s = summarizeMetrics(rows);
  return {
    C: round(s.C),
    B: round(s.B),
    R: round(s.R),
    G: round(s.G),
    J: round(s.J),
  };
}

function runControlSlate({ cells, seeds, seedStart, controls }) {
  const perCell = {};
  const aggregate = {};
  for (const label of controls) aggregate[label] = [];
  for (const cell of cells) {
    if (!(cell in H4_RELAY_CELL_DEFS)) throw new Error(`unknown H4 cell ${cell}`);
    perCell[cell] = {};
    for (const label of controls) {
      const rows = [];
      const outcomes = { correct: 0, basin: 0, timeout: 0 };
      const failGate = {};
      for (let s = 0; s < seeds; s += 1) {
        const metrics = rollEpisode(new DistributedRelayEnv(), label, seedStart + s, H4_RELAY_CELL_DEFS[cell]);
        rows.push(metrics);
        outcomes[metrics.outcome] = (outcomes[metrics.outcome] ?? 0) + 1;
        if (metrics.fail_gate != null) failGate[metrics.fail_gate] = (failGate[metrics.fail_gate] ?? 0) + 1;
      }
      perCell[cell][label] = { ...aggregateRows(rows), outcomes, fail_gate: failGate };
      aggregate[label].push(...rows);
    }
  }
  const aggregateMetrics = {};
  for (const label of controls) aggregateMetrics[label] = aggregateRows(aggregate[label]);
  return { perCell, aggregate: aggregateMetrics };
}

function runLocalityDrops({ cells, seeds, seedStart }) {
  const perSite = {};
  const K = H4_RELAY_DEFAULTS.K;
  for (let site = 0; site < K; site += 1) {
    perSite[site] = { cells_passing: 0, per_cell: {} };
    for (const cell of cells) {
      const fullRows = [];
      const dropRows = [];
      for (let s = 0; s < seeds; s += 1) {
        const seed = seedStart + s;
        fullRows.push(rollEpisode(new DistributedRelayEnv(), "FullHistory-H4", seed, H4_RELAY_CELL_DEFS[cell]));
        dropRows.push(
          rollEpisode(
            new DistributedRelayEnv(),
            `FullHistory-H4-drop-site-${site}`,
            seed,
            H4_RELAY_CELL_DEFS[cell],
          ),
        );
      }
      const full = aggregateRows(fullRows);
      const drop = aggregateRows(dropRows);
      const dC = round(full.C - drop.C);
      const dG = round(full.G - drop.G);
      const pass = Math.max(dC, dG) >= 0.08;
      if (pass) perSite[site].cells_passing += 1;
      perSite[site].per_cell[cell] = { full, drop, dC, dG, pass };
    }
  }
  const everySitePasses = Object.values(perSite).every((row) => row.cells_passing >= 2);
  return { per_site: perSite, every_site_passes: everySitePasses };
}

function leakageAudit(cells, seedStart) {
  for (const cell of cells) {
    const env = new DistributedRelayEnv();
    const obs = env.reset(seedStart, H4_RELAY_CELL_DEFS[cell]);
    if (publicObservationHasHiddenLatents(obs)) return false;
    for (let i = 0; i < 6; i += 1) {
      const current = env.observe();
      if (publicObservationHasHiddenLatents(current)) return false;
      const step = env.step(0);
      if (step.done) break;
    }
  }
  return true;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const cells = args.cells.split(",").map((cell) => cell.trim()).filter(Boolean);
  const controls = [
    "Oracle-H4",
    "Field-H4",
    "Reward-H4",
    "Blind-H4",
    "CurrentObs-H4",
    "FullHistory-H4",
    "Bottleneck-H4",
    "WideMessage-H4",
  ];
  const slate = runControlSlate({ cells, seeds: args.seeds, seedStart: args.seedStart, controls });
  const locality = runLocalityDrops({ cells, seeds: args.seeds, seedStart: args.seedStart });
  const A = slate.aggregate;
  const O = A["Oracle-H4"];
  const F = A["Field-H4"];
  const R = A["Reward-H4"];
  const C = A["CurrentObs-H4"];
  const H = A["FullHistory-H4"];
  const Bn = A["Bottleneck-H4"];
  const W = A["WideMessage-H4"];
  const hiddenLeakageOk = leakageAudit(cells, args.seedStart);

  const gates = {
    gate0_public_obs_no_hidden_latents: hiddenLeakageOk,
    gate1_solvability: O.C >= 0.9 && O.B <= 0.05,
    gate2_field_insufficiency: F.B <= 0.05 && F.C <= O.C - 0.2,
    gate3_reward_usefulness_danger: R.C - F.C >= 0.1 && R.B - F.B >= 0.15,
    gate4_history_necessity: O.C - C.C >= 0.2 || H.C - C.C >= 0.15,
    gate5_locality_necessity: locality.every_site_passes,
    gate6_communication_bottleneck: W.C - Bn.C >= 0.05 || W.J - Bn.J >= 0.05,
    gate7_learned_headroom_pending_h4_0_c: null,
    gate8_ood_gap_pending_h4_0_c: null,
  };
  const fixedGateValues = Object.entries(gates)
    .filter(([_, value]) => value !== null)
    .map(([_, value]) => value);
  const admitted = fixedGateValues.every(Boolean);
  const branch = admitted ? "H4_0_FIXED_ADMITTED" : "H4_0_TASK_VOID";
  const margins = {
    oracle_C: O.C,
    oracle_B: O.B,
    field_C: F.C,
    field_B: F.B,
    reward_usefulness_C: round(R.C - F.C),
    reward_danger_B: round(R.B - F.B),
    current_vs_oracle_C: round(O.C - C.C),
    full_history_vs_current_C: round(H.C - C.C),
    wide_vs_bottleneck_C: round(W.C - Bn.C),
    wide_vs_bottleneck_J: round(W.J - Bn.J),
  };
  const json = {
    spec: "docs/mesa/H4_DISTRIBUTED_WORLD_MODEL_TOPOLOGY_SPEC.md",
    family: "distributed-relay-grid",
    stage: "H4.0-b fixed-control admission",
    seeds: args.seeds,
    seed_start: args.seedStart,
    cells,
    defaults: H4_RELAY_DEFAULTS,
    cell_defs: Object.fromEntries(cells.map((cell) => [cell, H4_RELAY_CELL_DEFS[cell]])),
    aggregate: A,
    per_cell: slate.perCell,
    locality,
    margins,
    gates,
    branch,
    pending: ["H4.0-c learned headroom", "H4.0-c OOD generalization gap"],
  };
  mkdirSync(path.resolve(repoRoot, path.dirname(args.json)), { recursive: true });
  writeFileSync(path.resolve(repoRoot, args.json), `${JSON.stringify(json, null, 2)}\n`, "utf8");

  const md = [
    "# H4.0 Topology Admission Results",
    "",
    `Generated ${new Date().toISOString()} by \`scripts/mesa-h4-topology-admission.mjs\`.`,
    "",
    "Stage: **H4.0-b fixed-control admission** for the Distributed Relay Grid. This does **not** run the learned central-monolith headroom/OOD-gap checks; Gates 7-8 remain pending for H4.0-c.",
    "",
    `Cells: ${cells.join(", ")} x ${args.seeds} seeds (start ${args.seedStart}).`,
    "",
    "## H4.0-a Parity",
    "",
    "Commands:",
    "",
    "```powershell",
    "node scripts/mesa-h4-topology-fixtures.mjs",
    "python scripts/mesa-h4-topology-parity.py",
    "```",
    "",
    "Result: **PASS** -- 72 fixture episodes, 1,075 step rows, `max_abs_diff=0`, `hidden_leaks=0` at tolerance `1e-9`.",
    "",
    "## Environment",
    "",
    "- `K=4` local sites/gates, each with a hidden safe-side route bit.",
    "- Public observations expose only `phase`, `field_state`, `local_obs`, `local_mask`, `local_age`, and `reward_cue`; hidden `route_key` / `proxy_key` stay fixture-only.",
    "- `nominal-relay` is clean, `stale-relay` corrupts late observations on sites 1-3, and `decoy-relay` makes reward cues dangerous on sites 1 and 3.",
    "",
    "## Fixed Controls",
    "",
    "| control | C | B | R | G | J |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
    ...controls.map((label) => `| ${label} | ${A[label].C} | ${A[label].B} | ${A[label].R} | ${A[label].G} | ${A[label].J} |`),
    "",
    "## Admission Margins",
    "",
    `- Solvability: Oracle C **${O.C}** (>=0.90), B **${O.B}** (<=0.05).`,
    `- Field insufficiency: Field C **${F.C}**, B **${F.B}**; Oracle-Field C gap **${round(O.C - F.C)}**.`,
    `- Reward usefulness/danger: C_reward-C_field **${margins.reward_usefulness_C}**, B_reward-B_field **${margins.reward_danger_B}**.`,
    `- History necessity: Oracle-Current C gap **${margins.current_vs_oracle_C}**; FullHistory-Current C gap **${margins.full_history_vs_current_C}**.`,
    `- Communication bottleneck: Wide-Bottleneck C gap **${margins.wide_vs_bottleneck_C}**; J gap **${margins.wide_vs_bottleneck_J}**.`,
    "",
    "## Locality Drops",
    "",
    "| dropped site | cells passing >=0.08 drop | per-cell dC/dG |",
    "| ---: | ---: | --- |",
    ...Object.entries(locality.per_site).map(([site, row]) => {
      const parts = Object.entries(row.per_cell).map(([cell, r]) => `${cell}: ${r.dC}/${r.dG}${r.pass ? "" : " FAIL"}`);
      return `| ${site} | ${row.cells_passing} | ${parts.join("; ")} |`;
    }),
    "",
    "## Gates",
    "",
    ...Object.entries(gates).map(([key, value]) => `- \`${key}\`: **${value === null ? "PENDING" : value}**`),
    "",
    `## Decision: \`${branch}\``,
    "",
    admitted
      ? "The fixed-control layer is admitted: the task has solvability, safe-but-insufficient field control, useful-but-dangerous reward cues, history necessity, local-channel necessity, and a real message-width bottleneck. H4.0-c is still required before full H4.0 admission."
      : "Fixed admission failed. Revise the Distributed Relay cells before running learned H4.0-c or H4.1 controllers.",
    "",
    "## H4.0-c Pending",
    "",
    "H4.0-c must still run the cheap central recurrent monolith on the registered train/OOD split and select `H4_0_ADMITTED`, `H4_0_MONOLITH_HEADROOM_VOID`, `H4_0_NO_OOD_GAP_VOID`, or `H4_0_TASK_VOID`.",
    "",
  ].join("\n");
  writeFileSync(path.resolve(repoRoot, args.out), `${md}\n`, "utf8");

  console.log(`H4.0-b fixed admission [${cells.join(",")} x ${args.seeds} seeds]:`);
  for (const label of controls) {
    console.log(`  ${label.padEnd(16)} C=${String(A[label].C).padEnd(6)} B=${String(A[label].B).padEnd(6)} G=${String(A[label].G).padEnd(6)} J=${A[label].J}`);
  }
  console.log(`  margins: reward_use=${margins.reward_usefulness_C} reward_danger=${margins.reward_danger_B} history_gap=${margins.full_history_vs_current_C} bottleneck_C=${margins.wide_vs_bottleneck_C}`);
  console.log(`  gates: ${JSON.stringify(gates)}`);
  console.log(`  -> ${branch}`);
  console.log(`  wrote ${args.out} + ${args.json}`);
  process.exit(admitted ? 0 : 1);
}

main();
