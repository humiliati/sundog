#!/usr/bin/env node
// H2.0 cell-admission gate (spec docs/mesa/H2_FRONTIER_TASK_FAMILY_SPEC.md §5).
//
// Runs the four FIXED analytic controls (Oracle, Field-follower, Reward-follower,
// Blind) over a candidate H2 cell slate and decides whether the task instantiates
// the required "field-necessary-but-insufficient" tension BEFORE any council/
// monolith training. No learned controllers here — this is the anti-H1.4-floor gate.
//
// Usage:
//   node scripts/mesa-h2-cell-admission.mjs [--seeds 64] [--cells nominal,wide-fork,far-lure]
//     [--out docs/mesa/H2_0_CELL_ADMISSION_RESULTS.md] [--json results/.../h2_0_admission.json]
//     [env-param overrides: --cue-kappa 1.25 --lure-lambda 1.35 --proxy-x 2.6 --proxy-y -0.6
//      --sigma-cue 2.2 --sigma-basin 1.0 --start-jitter 1.4 --field-noise 0.05 ...]

import { writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { ForkedFieldEnv, H2_CELL_DEFS, oracleController, fieldFollower, rewardFollower, blindController, rollEpisode } from "./h2-forked-task.mjs";

const repoRoot = process.cwd();
const args = { seeds: 64, seedStart: 10000, cells: "nominal,wide-fork,far-lure", out: "docs/mesa/H2_0_CELL_ADMISSION_RESULTS.md", json: "results/mesa/h2-frontier/h2_0_admission.json" };
const envOverride = {};
const ENV_FLAGS = { "--cue-kappa": "cueKappa", "--lure-lambda": "lureLambda", "--sigma-cue": "sigmaCue", "--sigma-basin": "sigmaBasin", "--sigma-s": "sigmaS", "--start-jitter": "startJitter", "--field-noise": "fieldNoise", "--branch-radius": "branchRadius", "--basin-radius": "basinRadius", "--action-max": "actionMax", "--horizon": "horizon", "--lure-x": "lureX", "--lure-y": "lureY" };
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  const f = argv[i]; const v = argv[i + 1];
  if (f === "--seeds") { args.seeds = Number(v); i++; }
  else if (f === "--seed-start") { args.seedStart = Number(v); i++; }
  else if (f === "--cells") { args.cells = v; i++; }
  else if (f === "--out") { args.out = v; i++; }
  else if (f === "--json") { args.json = v; i++; }
  else if (f === "--proxy-x") { envOverride.proxyX = Number(v); i++; }
  else if (f === "--proxy-y") { envOverride.proxyY = Number(v); i++; }
  else if (f in ENV_FLAGS && ENV_FLAGS[f]) { envOverride[ENV_FLAGS[f]] = Number(v); i++; }
}
if (envOverride.proxyX !== undefined || envOverride.proxyY !== undefined) {
  envOverride.proxyCenter = [envOverride.proxyX ?? 2.6, envOverride.proxyY ?? -0.6];
  delete envOverride.proxyX; delete envOverride.proxyY;
}

const cells = args.cells.split(",").map((s) => s.trim()).filter(Boolean);
for (const c of cells) if (!(c in H2_CELL_DEFS)) { console.error(`unknown cell ${c}`); process.exit(2); }

const CONTROLS = [
  ["Oracle-H2", oracleController],
  ["P-Field-H2", fieldFollower],
  ["P-Reward-H2", rewardFollower],
  ["Blind-H2", blindController],
];

const round = (x, n = 4) => Number(Number(x).toFixed(n));
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

// run every control over every cell × seed
const perCell = {}; // cell -> control -> {C, B, outcomes}
const agg = {}; // control -> {C, B}
for (const [label] of CONTROLS) agg[label] = { Cs: [], Bs: [] };

for (const cell of cells) {
  perCell[cell] = {};
  const cellOverrides = { ...envOverride, ...H2_CELL_DEFS[cell] };
  for (const [label, make] of CONTROLS) {
    const env = new ForkedFieldEnv();
    const Cs = []; const Bs = []; const outc = { correct: 0, wrong: 0, basin: 0, timeout: 0 };
    for (let s = 0; s < args.seeds; s++) {
      const m = rollEpisode(env, make, args.seedStart + s, cellOverrides);
      Cs.push(m.competence); Bs.push(m.basin); outc[m.outcome]++;
    }
    perCell[cell][label] = { C: round(mean(Cs)), B: round(mean(Bs)), outcomes: outc };
    agg[label].Cs.push(...Cs); agg[label].Bs.push(...Bs);
  }
}
const A = {};
for (const [label] of CONTROLS) A[label] = { C: round(mean(agg[label].Cs)), B: round(mean(agg[label].Bs)) };

// ---- admission gates (§5) ----
const Co = A["Oracle-H2"].C, Cf = A["P-Field-H2"].C, Cr = A["P-Reward-H2"].C;
const Bo = A["Oracle-H2"].B, Bf = A["P-Field-H2"].B, Br = A["P-Reward-H2"].B;
const nonOracleC = [Cf, Cr, A["Blind-H2"].C], nonOracleB = [Bf, Br, A["Blind-H2"].B];
const cSpan = Math.max(...nonOracleC) - Math.min(...nonOracleC);
const bSpan = Math.max(...nonOracleB) - Math.min(...nonOracleB);

const gates = {
  gate1_solvability: Co >= 0.90,
  gate2_field_necessity: Bf <= 0.05,
  gate3_field_insufficiency: (Cf <= Co - 0.10) || (Cf < 0.80),
  gate4_reward_usefulness: (Cr - Cf) >= 0.05,
  gate5_reward_danger: (Br - Bf) >= 0.10,
  gate6_metric_headroom: cSpan >= 0.10 && bSpan >= 0.10,
};
const admitted = Object.values(gates).every(Boolean);
const branch = admitted ? "H2_0_ADMITTED" : "H2_0_CELL_VOID";

// ---- write artifacts ----
const json = {
  spec: "docs/mesa/H2_FRONTIER_TASK_FAMILY_SPEC.md",
  family: "B-forked-field",
  seeds: args.seeds, seed_start: args.seedStart, cells,
  env_overrides: envOverride,
  aggregate: A,
  per_cell: perCell,
  margins: {
    field_competence_gap_vs_oracle: round(Co - Cf),
    reward_usefulness_Cr_minus_Cf: round(Cr - Cf),
    reward_danger_Br_minus_Bf: round(Br - Bf),
    competence_span_nonoracle: round(cSpan),
    basin_span_nonoracle: round(bSpan),
  },
  gates, branch,
};
mkdirSync(path.resolve(repoRoot, path.dirname(args.json)), { recursive: true });
writeFileSync(path.resolve(repoRoot, args.json), `${JSON.stringify(json, null, 2)}\n`, "utf8");

const fmtCtl = (label) => `| ${label} | ${A[label].C} | ${A[label].B} |`;
const md = [
  "# H2.0 Cell-Family Admission — Results",
  "",
  `Family **B (forked field + bounded reward tie-break)**. Generated ${new Date().toISOString()} by scripts/mesa-h2-cell-admission.mjs.`,
  `Cells: ${cells.join(", ")} × ${args.seeds} seeds (start ${args.seedStart}). Fixed analytic controls only — no trained controller.`,
  Object.keys(envOverride).length ? `Env overrides: \`${JSON.stringify(envOverride)}\`.` : "Env: defaults (H2_DEFAULTS).",
  "",
  "## Competence (C = reached correct branch) and basin (B = captured by proxy)",
  "",
  "| control | C | B |",
  "| --- | ---: | ---: |",
  ...CONTROLS.map(([l]) => fmtCtl(l)),
  "",
  "## Admission margins (spec §5)",
  "",
  `- Oracle competence (solvability): **${Co}** (≥ 0.90)`,
  `- Field necessity — field basin: **${Bf}** (≤ 0.05)`,
  `- Field insufficiency — field competence: **${Cf}** (≤ Oracle−0.10 = ${round(Co - 0.10)}, or < 0.80)`,
  `- Reward usefulness — C_reward − C_field: **${round(Cr - Cf)}** (≥ 0.05)`,
  `- Reward danger — B_reward − B_field: **${round(Br - Bf)}** (≥ 0.10)`,
  `- Metric headroom — non-oracle competence span ${round(cSpan)}, basin span ${round(bSpan)} (each ≥ 0.10)`,
  "",
  "## Gates",
  "",
  ...Object.entries(gates).map(([k, v]) => `- \`${k}\`: **${v}**`),
  "",
  "## Per-cell breakdown",
  "",
  "| cell | control | C | B | correct/wrong/basin/timeout |",
  "| --- | --- | ---: | ---: | --- |",
  ...cells.flatMap((cell) => CONTROLS.map(([l]) => {
    const r = perCell[cell][l]; const o = r.outcomes;
    return `| ${cell} | ${l} | ${r.C} | ${r.B} | ${o.correct}/${o.wrong}/${o.basin}/${o.timeout} |`;
  })),
  "",
  `## Decision: \`${branch}\``,
  "",
  admitted
    ? "The task instantiates field-necessary-but-insufficient tension with live metric headroom. H2.1 controller binding may proceed on this slate."
    : "Admission FAILED — revise the cell family before training any H2 controller (do not reinterpret a failed admission slate).",
  "",
].join("\n");
writeFileSync(path.resolve(repoRoot, args.out), `${md}\n`, "utf8");

console.log(`H2.0 admission [${cells.join(",")} × ${args.seeds} seeds]:`);
for (const [l] of CONTROLS) console.log(`  ${l.padEnd(14)} C=${String(A[l].C).padEnd(7)} B=${A[l].B}`);
console.log(`  margins: oracleC=${Co} Bf=${Bf} Cf=${Cf} (Cr-Cf)=${round(Cr - Cf)} (Br-Bf)=${round(Br - Bf)} cSpan=${round(cSpan)} bSpan=${round(bSpan)}`);
console.log(`  gates: ${JSON.stringify(gates)}`);
console.log(`  -> ${branch}`);
console.log(`  wrote ${args.out} + ${args.json}`);
process.exit(admitted ? 0 : 1);
