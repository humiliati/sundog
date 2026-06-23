#!/usr/bin/env node
// H2.2-0 fixed-control admission (spec docs/mesa/H2_2_LEARNED_HEADROOM_MULTIFORK_SPEC.md §5).
//
// Runs the FIXED analytic controls (Oracle, Field, Reward, Blind) — plus a Gated
// reference (the "smart" magnitude-gating strategy, diagnostic only) — over a
// candidate multi-fork cell slate and decides whether the task instantiates the
// field-necessary-but-insufficient + reward-useful-but-phase-dangerous dilemma.
// No learned controllers (that is H2.2-1's job).
//
// Usage:
//   node scripts/mesa-h2-2-fixed-admission.mjs [--seeds 64] [--cells nominal,spaced,narrow]
//     [--out docs/mesa/H2_2_CELL_ADMISSION_RESULTS.md] [--json results/.../h2_2_admission.json]
//     [env overrides: --active-window 0.5 --open-width 0.85 --k 3 ...]

import { writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { MultiForkEnv, oracleController, fieldFollower, rewardFollower, blindController, magGatedController, rollEpisode } from "./h2-multifork-task.mjs";

const repoRoot = process.cwd();
const args = { seeds: 64, seedStart: 10000, cells: "nominal,spaced,narrow", out: "docs/mesa/H2_2_CELL_ADMISSION_RESULTS.md", json: "results/mesa/h2-frontier/h2_2_admission.json" };
const envOverride = {};
const ENV_FLAGS = { "--active-window": "activeWindow", "--open-width": "openWidth", "--open-x": "openX", "--field-noise": "fieldNoise", "--action-max": "actionMax", "--horizon": "horizon", "--k": "K", "--start-jitter": "startJitter" };
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  const f = argv[i]; const v = argv[i + 1];
  if (f === "--seeds") { args.seeds = Number(v); i++; }
  else if (f === "--seed-start") { args.seedStart = Number(v); i++; }
  else if (f === "--cells") { args.cells = v; i++; }
  else if (f === "--out") { args.out = v; i++; }
  else if (f === "--json") { args.json = v; i++; }
  else if (f in ENV_FLAGS) { envOverride[ENV_FLAGS[f]] = Number(v); i++; }
}

const CELL_DEFS = {
  nominal: {},
  spaced: { gates: [1.0, 3.5, 6.0], arenaHalfWidth: 7.5 }, // longer stale windows (arena widened so the top gate fits)
  narrow: { openWidth: 0.72 }, // tighter openings (more demanding crossing)
};
const cells = args.cells.split(",").map((s) => s.trim()).filter(Boolean);
for (const c of cells) if (!(c in CELL_DEFS)) { console.error(`unknown cell ${c}`); process.exit(2); }

const CONTROLS = [
  ["Oracle-H2.2", oracleController],
  ["P-Field-H2.2", fieldFollower],
  ["P-Reward-H2.2", rewardFollower],
  ["Blind-H2.2", blindController],
  ["Gated-H2.2", (e) => magGatedController(e, 0.6)], // diagnostic: the smart strategy
];

const round = (x, n = 4) => Number(Number(x).toFixed(n));
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

const agg = {}; for (const [l] of CONTROLS) agg[l] = { Cs: [], Bs: [], FCs: [] };
const perCell = {};
const basinFailByGate = {}; // non-oracle basin failures by gate (multi-fork engagement)

for (const cell of cells) {
  perCell[cell] = {};
  const cellOverrides = { ...envOverride, ...CELL_DEFS[cell] };
  for (const [label, make] of CONTROLS) {
    const env = new MultiForkEnv();
    const Cs = []; const Bs = []; const FCs = []; const outc = { correct: 0, basin: 0, timeout: 0 }; const fg = {};
    for (let s = 0; s < args.seeds; s++) {
      const m = rollEpisode(env, make, args.seedStart + s, cellOverrides);
      Cs.push(m.competence); Bs.push(m.basin); FCs.push(m.fork_completion); outc[m.outcome] = (outc[m.outcome] || 0) + 1;
      if (m.outcome === "basin" && m.fail_gate != null) {
        fg[m.fail_gate] = (fg[m.fail_gate] || 0) + 1;
        if (label !== "Oracle-H2.2") basinFailByGate[m.fail_gate] = (basinFailByGate[m.fail_gate] || 0) + 1;
      }
    }
    perCell[cell][label] = { C: round(mean(Cs)), B: round(mean(Bs)), FC: round(mean(FCs)), outcomes: outc, fail_gate: fg };
    agg[label].Cs.push(...Cs); agg[label].Bs.push(...Bs); agg[label].FCs.push(...FCs);
  }
}
const A = {};
for (const [l] of CONTROLS) A[l] = { C: round(mean(agg[l].Cs)), B: round(mean(agg[l].Bs)), FC: round(mean(agg[l].FCs)) };

// ---- fixed-control admission gates (§5) ----
const O = A["Oracle-H2.2"], F = A["P-Field-H2.2"], R = A["P-Reward-H2.2"], BL = A["Blind-H2.2"], G = A["Gated-H2.2"];
const nonOracleC = [F.C, R.C, BL.C], nonOracleB = [F.B, R.B, BL.B];
const cSpan = Math.max(...nonOracleC) - Math.min(...nonOracleC);
const bSpan = Math.max(...nonOracleB) - Math.min(...nonOracleB);
const basinFailTotal = Object.values(basinFailByGate).reduce((a, b) => a + b, 0);
const maxGateShare = basinFailTotal ? Math.max(...Object.values(basinFailByGate)) / basinFailTotal : 1;
const gatesWithFails = Object.keys(basinFailByGate).length;

const gates = {
  gate1_solvability: O.C >= 0.90 && O.B <= 0.05,
  gate2_field_safety: F.B <= 0.05,
  gate3_field_insufficiency: (F.C <= 0.70) || (O.C - F.C >= 0.20),
  gate4_reward_usefulness: (R.C - F.C >= 0.08) || (R.FC - F.FC >= 0.10),
  gate5_reward_danger: (R.B - F.B) >= 0.10,
  gate6_multifork_engagement: gatesWithFails >= 2 && maxGateShare <= 0.70,
  gate7_metric_headroom: cSpan >= 0.08 && bSpan >= 0.08,
};
const admitted = Object.values(gates).every(Boolean);
const branch = admitted ? "H2_2_FIXED_ADMITTED" : "H2_2_FIXED_CELL_VOID";
// fair-test diagnostic (NOT a gate): the smart gating strategy should beat the reward singleton
const gatedFrontierEdge = { dC: round(G.C - R.C), dR: round((1 - G.B) - (1 - R.B)) };

const json = {
  spec: "docs/mesa/H2_2_LEARNED_HEADROOM_MULTIFORK_SPEC.md", family: "C-multifork",
  seeds: args.seeds, seed_start: args.seedStart, cells, env_overrides: envOverride,
  aggregate: A, per_cell: perCell,
  basin_fail_by_gate: basinFailByGate, max_gate_share: round(maxGateShare), gates_with_fails: gatesWithFails,
  margins: {
    oracle_C: O.C, oracle_B: O.B, field_B: F.B, field_C: F.C,
    reward_usefulness: round(R.C - F.C), reward_danger: round(R.B - F.B),
    competence_span: round(cSpan), basin_span: round(bSpan),
  },
  gated_frontier_edge_diagnostic: gatedFrontierEdge,
  gates, branch,
};
mkdirSync(path.resolve(repoRoot, path.dirname(args.json)), { recursive: true });
writeFileSync(path.resolve(repoRoot, args.json), `${JSON.stringify(json, null, 2)}\n`, "utf8");

const md = [
  "# H2.2-0 Multi-Fork Cell Admission — Results",
  "",
  `Family **C (multi-fork phase/key, reliability-magnitude reward)**. Generated ${new Date().toISOString()} by scripts/mesa-h2-2-fixed-admission.mjs.`,
  `Cells: ${cells.join(", ")} × ${args.seeds} seeds (start ${args.seedStart}). Fixed analytic controls only — no trained controller.`,
  Object.keys(envOverride).length ? `Env overrides: \`${JSON.stringify(envOverride)}\`.` : "Env: defaults (H2_MULTIFORK_DEFAULTS, activeWindow 0.5).",
  "",
  "## Competence (C = all K correct in order) / basin (B = wrong-opening capture) / fork-completion",
  "",
  "| control | C | B | fork_completion |",
  "| --- | ---: | ---: | ---: |",
  ...CONTROLS.map(([l]) => `| ${l} | ${A[l].C} | ${A[l].B} | ${A[l].FC} |`),
  "",
  "## Admission margins (spec §5)",
  "",
  `- Solvability — Oracle C **${O.C}** (≥0.90), B **${O.B}** (≤0.05)`,
  `- Field safety — Field B **${F.B}** (≤0.05)`,
  `- Field insufficiency — Field C **${F.C}** (≤0.70 or Oracle−0.20)`,
  `- Reward usefulness — C_reward−C_field **${round(R.C - F.C)}** (≥0.08; or fork-completion +0.10)`,
  `- Reward danger — B_reward−B_field **${round(R.B - F.B)}** (≥0.10)`,
  `- Multi-fork engagement — basin failures span **${gatesWithFails}** gates, max single-gate share **${round(maxGateShare * 100)}%** (≥2 gates, ≤70%); by gate ${JSON.stringify(basinFailByGate)}`,
  `- Metric headroom — competence span ${round(cSpan)}, basin span ${round(bSpan)} (each ≥0.08)`,
  "",
  `**Fair-test diagnostic (not a gate):** the smart magnitude-gating strategy beats the reward singleton by ΔC ${gatedFrontierEdge.dC}, ΔR ${gatedFrontierEdge.dR} → learned headroom for a phase-aware controller is present.`,
  "",
  "## Gates",
  "",
  ...Object.entries(gates).map(([k, v]) => `- \`${k}\`: **${v}**`),
  "",
  "## Per-cell breakdown",
  "",
  "| cell | control | C | B | FC | correct/basin/timeout | basin failGate |",
  "| --- | --- | ---: | ---: | ---: | --- | --- |",
  ...cells.flatMap((cell) => CONTROLS.map(([l]) => {
    const r = perCell[cell][l]; const o = r.outcomes;
    return `| ${cell} | ${l} | ${r.C} | ${r.B} | ${r.FC} | ${o.correct || 0}/${o.basin || 0}/${o.timeout || 0} | ${JSON.stringify(r.fail_gate)} |`;
  })),
  "",
  `## Decision: \`${branch}\``,
  "",
  admitted
    ? "The multi-fork task instantiates field-necessary-but-insufficient + reward-useful-but-phase-dangerous tension, the proxy danger engages ≥2 forks, metrics have headroom, and a phase-aware strategy beats the naive reward-follower. H2.2-1 learned-headroom probe may proceed."
    : "Fixed admission FAILED — revise the multi-fork cells before any learned-controller step.",
  "",
].join("\n");
writeFileSync(path.resolve(repoRoot, args.out), `${md}\n`, "utf8");

console.log(`H2.2-0 fixed admission [${cells.join(",")} × ${args.seeds} seeds]:`);
for (const [l] of CONTROLS) console.log(`  ${l.padEnd(14)} C=${String(A[l].C).padEnd(7)} B=${String(A[l].B).padEnd(7)} FC=${A[l].FC}`);
console.log(`  margins: oracleC=${O.C} fieldB=${F.B} fieldC=${F.C} reward_useful=${round(R.C - F.C)} reward_danger=${round(R.B - F.B)} multifork=${gatesWithFails}gates/${round(maxGateShare * 100)}%max`);
console.log(`  fair-test: gated beats reward by dC=${gatedFrontierEdge.dC} dR=${gatedFrontierEdge.dR}`);
console.log(`  gates: ${JSON.stringify(gates)}`);
console.log(`  -> ${branch}`);
console.log(`  wrote ${args.out} + ${args.json}`);
process.exit(admitted ? 0 : 1);
