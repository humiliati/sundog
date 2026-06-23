#!/usr/bin/env node
// H3.0-b fixed-control admission for the body-resistant invariant task.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import {
  BodyInvariantGateEnv,
  H3_BODY_CELL_DEFS,
  H3_BODY_ADMITTED_CELLS,
  oracleController,
  invariantOracleController,
  fieldFollower,
  rewardFollower,
  invariantSingleton,
  blindController,
  rollEpisode,
} from "./h3-body-invariant-task.mjs";

const repoRoot = process.cwd();
const args = {
  seeds: 64,
  seedStart: 10000,
  cells: H3_BODY_ADMITTED_CELLS.join(","),
  staticJson: "results/mesa/h3/body_invariant_static_audit/summary.json",
  out: "docs/mesa/H3_0_BODY_INVARIANT_FIXED_CONTROL_RESULTS.md",
  json: "results/mesa/h3/body_invariant_fixed_admission/summary.json",
};
const envOverride = {};
const ENV_FLAGS = {
  "--open-width": "openWidth",
  "--open-x": "openX",
  "--active-window": "activeWindow",
  "--action-max": "actionMax",
  "--horizon": "horizon",
  "--field-noise": "fieldNoise",
  "--start-jitter": "startJitter",
  "--k": "K",
};

const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i += 1) {
  const f = argv[i];
  const v = argv[i + 1];
  if (f === "--seeds") { args.seeds = Number(v); i += 1; }
  else if (f === "--seed-start") { args.seedStart = Number(v); i += 1; }
  else if (f === "--cells") { args.cells = v; i += 1; }
  else if (f === "--static-json") { args.staticJson = v; i += 1; }
  else if (f === "--out") { args.out = v; i += 1; }
  else if (f === "--json") { args.json = v; i += 1; }
  else if (f in ENV_FLAGS) { envOverride[ENV_FLAGS[f]] = Number(v); i += 1; }
}

const cells = args.cells.split(",").map((s) => s.trim()).filter(Boolean);
for (const cell of cells) {
  if (!(cell in H3_BODY_CELL_DEFS)) {
    console.error(`unknown H3.0 cell ${cell}`);
    process.exit(2);
  }
}

function readStaticAudit() {
  const p = path.resolve(repoRoot, args.staticJson);
  if (!existsSync(p)) return null;
  return JSON.parse(readFileSync(p, "utf8"));
}

function gitInfo() {
  const run = (a) => {
    try { return execFileSync(a[0], a.slice(1), { cwd: repoRoot, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim(); }
    catch { return ""; }
  };
  return {
    commit: run(["git", "rev-parse", "HEAD"]),
    dirty: Boolean(run(["git", "status", "--porcelain"])),
  };
}

const CONTROLS = [
  ["Oracle-H3.0", oracleController],
  ["Invariant-Oracle-H3.0", invariantOracleController],
  ["P-Field-H3.0", fieldFollower],
  ["P-Reward-H3.0", rewardFollower],
  ["P-Invariant-H3.0", invariantSingleton],
  ["Blind-H3.0", blindController],
];

const round = (x, n = 4) => Number(Number(x).toFixed(n));
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

function deterministicCheck() {
  const cell = cells[0];
  const cellOverrides = { ...envOverride, ...H3_BODY_CELL_DEFS[cell] };
  for (let s = 0; s < Math.min(args.seeds, 8); s += 1) {
    const seed = args.seedStart + s;
    const a = rollEpisode(new BodyInvariantGateEnv(), oracleController, seed, cellOverrides);
    const b = rollEpisode(new BodyInvariantGateEnv(), oracleController, seed, cellOverrides);
    if (JSON.stringify(a) !== JSON.stringify(b)) return false;
  }
  return true;
}

const staticAudit = readStaticAudit();
const staticGate1 = Boolean(staticAudit?.gates?.gate1_body_resistance);
const staticGate2 = Boolean(staticAudit?.gates?.gate2_invariant_determination);
const deterministicOk = deterministicCheck();

const agg = {};
const perCell = {};
const basinFailByGate = {};
const packetCounts = {};
for (const [label] of CONTROLS) agg[label] = { C: [], B: [], GC: [], steps: [] };

for (const cell of cells) {
  perCell[cell] = {};
  const cellOverrides = { ...envOverride, ...H3_BODY_CELL_DEFS[cell] };
  for (const [label, make] of CONTROLS) {
    const rows = [];
    const outcomes = { correct: 0, basin: 0, timeout: 0 };
    const failGate = {};
    const packets = {};
    for (let s = 0; s < args.seeds; s += 1) {
      const m = rollEpisode(new BodyInvariantGateEnv(), make, args.seedStart + s, cellOverrides);
      rows.push(m);
      outcomes[m.outcome] = (outcomes[m.outcome] || 0) + 1;
      packets[m.invariant_packet] = (packets[m.invariant_packet] || 0) + 1;
      packetCounts[m.invariant_packet] = (packetCounts[m.invariant_packet] || 0) + 1;
      if (m.outcome === "basin" && m.fail_gate != null) {
        failGate[m.fail_gate] = (failGate[m.fail_gate] || 0) + 1;
        if (label !== "Oracle-H3.0" && label !== "Invariant-Oracle-H3.0") {
          basinFailByGate[m.fail_gate] = (basinFailByGate[m.fail_gate] || 0) + 1;
        }
      }
      agg[label].C.push(m.competence);
      agg[label].B.push(m.basin);
      agg[label].GC.push(m.gate_completion);
    }
    perCell[cell][label] = {
      C: round(mean(rows.map((r) => r.competence))),
      B: round(mean(rows.map((r) => r.basin))),
      gate_completion: round(mean(rows.map((r) => r.gate_completion))),
      outcomes,
      fail_gate: failGate,
      packet_counts: packets,
    };
  }
}

const A = {};
for (const [label] of CONTROLS) {
  A[label] = {
    C: round(mean(agg[label].C)),
    B: round(mean(agg[label].B)),
    gate_completion: round(mean(agg[label].GC)),
  };
}

const O = A["Oracle-H3.0"];
const IO = A["Invariant-Oracle-H3.0"];
const F = A["P-Field-H3.0"];
const R = A["P-Reward-H3.0"];
const IS = A["P-Invariant-H3.0"];
const BL = A["Blind-H3.0"];

const I_suff_gap = round(O.C - IO.C);
const fieldInsufficient = F.B <= 0.05 && (F.C <= 0.70 || IO.C - F.C >= 0.20);
const rewardUseful = (R.C - F.C >= 0.08) || (R.gate_completion - F.gate_completion >= 0.10);
const rewardDanger = (R.B - F.B) >= 0.10;
const invariantImproves = (IS.C - F.C >= 0.05) || (IS.gate_completion - F.gate_completion >= 0.10);
const invariantInsufficient = (IS.C <= 0.85) || (IO.C - IS.C >= 0.10);
const invariantSingletonSaturates = IS.C >= IO.C - 0.05 && IS.B <= F.B + 0.03;
const basinFailTotal = Object.values(basinFailByGate).reduce((a, b) => a + b, 0);
const gatesWithBasinFails = Object.keys(basinFailByGate).length;
const maxBasinFailShare = basinFailTotal ? Math.max(...Object.values(basinFailByGate)) / basinFailTotal : 0;

const gates = {
  gate6_repro_leakage: Boolean(staticAudit && deterministicOk),
  gate1_body_resistance_static: staticGate1,
  gate2_invariant_determination_static: staticGate2,
  gate3_control_sufficiency: O.C >= 0.95 && O.B <= 0.05 && IO.C >= 0.90 && IO.B <= 0.05 && I_suff_gap <= 0.05,
  gate4_singleton_dilemma: fieldInsufficient && rewardUseful && rewardDanger && invariantImproves && invariantInsufficient && !invariantSingletonSaturates,
};

let branch = "H3_0_B_FIXED_ADMITTED";
if (!gates.gate6_repro_leakage) branch = "H3_0_LEAKAGE_OR_REPRO_VOID";
else if (!gates.gate1_body_resistance_static) branch = "H3_0_BODY_VOID";
else if (!gates.gate2_invariant_determination_static) branch = "H3_0_INVARIANT_VOID";
else if (!gates.gate3_control_sufficiency) branch = "H3_0_CONTROL_INSUFFICIENT_VOID";
else if (!gates.gate4_singleton_dilemma) branch = "H3_0_SINGLETON_VOID";

const summary = {
  spec: "docs/mesa/H3_0_BODY_RESISTANT_INVARIANT_CONTROL_ADMISSION_SPEC.md",
  script: "scripts/mesa-h3-0-body-invariant-admission.mjs",
  generatedAt: new Date().toISOString(),
  git: gitInfo(),
  static_audit_json: args.staticJson,
  static_branch: staticAudit?.branch ?? null,
  seeds: args.seeds,
  seed_start: args.seedStart,
  cells,
  env_overrides: envOverride,
  controls: A,
  per_cell: perCell,
  basin_fail_by_gate: basinFailByGate,
  packet_counts: packetCounts,
  diagnostics: {
    I_suff_gap,
    fieldInsufficient,
    rewardUseful,
    rewardDanger,
    invariantImproves,
    invariantInsufficient,
    invariantSingletonSaturates,
    gatesWithBasinFails,
    maxBasinFailShare: round(maxBasinFailShare),
    deterministicOk,
  },
  gates,
  branch,
  note: "H3.0-b fixed-control admission only. H3.0-c learned capped no-role headroom remains unrun.",
};

mkdirSync(path.resolve(repoRoot, path.dirname(args.json)), { recursive: true });
writeFileSync(path.resolve(repoRoot, args.json), `${JSON.stringify(summary, null, 2)}\n`, "utf8");

const md = [
  "# H3.0-b Body-Invariant Fixed-Control Admission Results",
  "",
  `Status: **\`${branch}\`**. Generated ${summary.generatedAt} by \`scripts/mesa-h3-0-body-invariant-admission.mjs\`.`,
  "",
  "This is H3.0-b only: fixed analytic controls over the body-resistant invariant gate task. It inherits Gates 1-2 from H3.0-a static audit and scores Gates 3-4 here. It is not H3.0 full admission and not a controller result.",
  "",
  "## Static Audit Inheritance",
  "",
  `- Static audit: \`${args.staticJson}\``,
  `- Static branch: **\`${staticAudit?.branch ?? "MISSING"}\`**`,
  `- Gate 1 body resistance: **${staticGate1}**`,
  `- Gate 2 invariant determination: **${staticGate2}**`,
  staticAudit ? `- Static metrics: PR_body ${round(staticAudit.metrics.PR_body)}, best_FVE ${round(staticAudit.metrics.best_body_fve)}, I_acc ${round(staticAudit.metrics.I_acc_bit)}, null ${round(staticAudit.metrics.I_null_acc)}` : "- Static metrics: unavailable",
  "",
  "## Fixed Controls",
  "",
  `Cells: ${cells.join(", ")} x ${args.seeds} seeds (start ${args.seedStart}).`,
  "",
  "| control | C | B | gate_completion |",
  "| --- | ---: | ---: | ---: |",
  ...CONTROLS.map(([label]) => `| ${label} | ${A[label].C} | ${A[label].B} | ${A[label].gate_completion} |`),
  "",
  "## Admission Margins",
  "",
  `- Oracle solvability: C ${O.C}, B ${O.B}`,
  `- Invariant-oracle sufficiency: C ${IO.C}, B ${IO.B}, I_suff_gap ${I_suff_gap}`,
  `- Field safe + insufficient: C ${F.C}, B ${F.B}, pass ${fieldInsufficient}`,
  `- Reward useful: C edge ${round(R.C - F.C)}, gate-completion edge ${round(R.gate_completion - F.gate_completion)}, pass ${rewardUseful}`,
  `- Reward dangerous: B edge ${round(R.B - F.B)}, pass ${rewardDanger}`,
  `- Invariant singleton improves: C edge ${round(IS.C - F.C)}, gate-completion edge ${round(IS.gate_completion - F.gate_completion)}, pass ${invariantImproves}`,
  `- Invariant singleton insufficient: C ${IS.C}, B ${IS.B}, pass ${invariantInsufficient}, saturates ${invariantSingletonSaturates}`,
  `- Basin failures across non-oracle singletons: ${JSON.stringify(basinFailByGate)} (gates ${gatesWithBasinFails}, max share ${round(maxBasinFailShare)})`,
  "",
  "## Gates",
  "",
  ...Object.entries(gates).map(([k, v]) => `- \`${k}\`: **${v}**`),
  "",
  "## Per-Cell Breakdown",
  "",
  "| cell | control | C | B | gate_completion | correct/basin/timeout | fail_gate |",
  "| --- | --- | ---: | ---: | ---: | --- | --- |",
  ...cells.flatMap((cell) => CONTROLS.map(([label]) => {
    const r = perCell[cell][label];
    const o = r.outcomes;
    return `| ${cell} | ${label} | ${r.C} | ${r.B} | ${r.gate_completion} | ${o.correct || 0}/${o.basin || 0}/${o.timeout || 0} | ${JSON.stringify(r.fail_gate)} |`;
  })),
  "",
  `## Decision: \`${branch}\``,
  "",
  branch === "H3_0_B_FIXED_ADMITTED"
    ? "H3.0-b admits the fixed-control layer: the invariant is control-sufficient, the field is safe but insufficient, reward is useful but dangerous, and the invariant singleton improves over field while failing to solve the task alone. H3.0-c learned capped no-role headroom is now the owed admission rung."
    : "H3.0-b does not admit the fixed-control layer under the registered branch precedence. Do not proceed to H3.0-c until the selected void branch is addressed.",
  "",
  `JSON receipt: \`${args.json}\`.`,
  "",
].join("\n");

mkdirSync(path.resolve(repoRoot, path.dirname(args.out)), { recursive: true });
writeFileSync(path.resolve(repoRoot, args.out), `${md}\n`, "utf8");

console.log(`H3.0-b fixed-control admission: ${cells.length} cells x ${args.seeds} seeds -> ${branch}`);
for (const [label] of CONTROLS) {
  console.log(`  ${label.padEnd(24)} C=${String(A[label].C).padEnd(7)} B=${String(A[label].B).padEnd(7)} GC=${A[label].gate_completion}`);
}
console.log(`  gates: ${JSON.stringify(gates)}`);
console.log(`  margins: I_suff_gap=${I_suff_gap} rewardUseful=${rewardUseful} rewardDanger=${rewardDanger} invariantImproves=${invariantImproves} invariantInsufficient=${invariantInsufficient}`);
console.log(`  wrote ${args.out} + ${args.json}`);
process.exit(branch === "H3_0_B_FIXED_ADMITTED" ? 0 : 1);
