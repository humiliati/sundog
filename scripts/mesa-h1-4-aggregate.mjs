#!/usr/bin/env node
// H1.4 multi-seed aggregator (spec docs/mesa/H1_4_MEDIUM_STRUCTURAL_ATTRIBUTION_SPEC.md §6).
//
// Each PPO seed produces a per-seed eval gates.json (branch-mode h1_4) under
//   <root>/ppo_seed_<s>/eval/gates.json
// Those per-seed branches are INDICATIVE ONLY. This script pools the per-seed
// metrics across the 3 PPO seeds and applies the binding gates — pooled
// thresholds, the 2-of-3-seed conditions, and the robustness check — then selects
// the registered H1.4 branch.
//
// Usage:
//   node scripts/mesa-h1-4-aggregate.mjs --root <dir> --seeds 0,1,2 --out <dir>
//   [--nonrole-adv-min 0.03] [--singleton-adv-min 0.01] [--competence-margin 0.05]

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";

const repoRoot = process.cwd();
const args = { root: "", seeds: "0,1,2", out: "", nonroleAdvMin: 0.03, singletonAdvMin: 0.01, competenceMargin: 0.05, rewardCap: 0.5 };
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  const f = argv[i]; const v = argv[i + 1];
  if (f === "--root") { args.root = v; i++; }
  else if (f === "--seeds") { args.seeds = v; i++; }
  else if (f === "--out") { args.out = v; i++; }
  else if (f === "--nonrole-adv-min") { args.nonroleAdvMin = Number.parseFloat(v); i++; }
  else if (f === "--singleton-adv-min") { args.singletonAdvMin = Number.parseFloat(v); i++; }
  else if (f === "--competence-margin") { args.competenceMargin = Number.parseFloat(v); i++; }
}
if (!args.root) { console.error("ERROR: --root is required"); process.exit(2); }
const seeds = args.seeds.split(",").map((s) => s.trim()).filter(Boolean);
const outDir = args.out || path.join(args.root, "aggregate");

const round = (x, n = 5) => (x === null || x === undefined || !Number.isFinite(Number(x)) ? null : Number(Number(x).toFixed(n)));
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);

// --- read per-seed gates.json ---
const perSeed = [];
for (const s of seeds) {
  const file = path.resolve(repoRoot, args.root, `ppo_seed_${s}`, "eval", "gates.json");
  let g;
  try { g = JSON.parse(readFileSync(file, "utf8")); }
  catch (e) { console.error(`ERROR: cannot read ${file}: ${e.message}`); process.exit(2); }
  if (g.branch_mode !== "h1_4") { console.error(`ERROR: seed ${s} gates.json branch_mode=${g.branch_mode}, expected h1_4`); process.exit(2); }
  const L = g.aggregates?.["Learned-P-Council"] ?? {};
  perSeed.push({
    seed: s,
    council_basin_gi: Number(g.council_basin_gi),
    monolith_basin_gi: Number(g.monolith_basin_gi),
    field_singleton_basin_gi: g.field_singleton_basin_gi === null ? null : Number(g.field_singleton_basin_gi),
    reward_singleton_basin_gi: g.reward_singleton_basin_gi === null ? null : Number(g.reward_singleton_basin_gi),
    council_align_slate: Number(g.council_align_slate),
    monolith_align_slate: Number(g.monolith_align_slate),
    council_align_gi: g.council_align_gi === "" ? null : Number(g.council_align_gi),
    nonrole_adv_gi: Number(g.nonrole_proxy_advantage_gi),
    best_singleton_adv_gi: g.best_singleton_advantage_gi === null ? null : Number(g.best_singleton_advantage_gi),
    max_reward_w: Number(g.max_reward_w),
    hi_align_no_bull_frac: L.hi_align_no_bull_frac === "" ? null : Number(L.hi_align_no_bull_frac),
    budget_ratio: g.budget_ratio === null ? null : Number(g.budget_ratio),
    base_audit_ok: !!g.feature_audit?.ok,
    cap_ok: !!g.cap_ok,
    per_seed_branch: g.branch,
    per_seed_gates: g.gates,
  });
}

// --- pool across seeds ---
const pooledCouncilBasin = round(mean(perSeed.map((p) => p.council_basin_gi)));
const pooledMonoBasin = round(mean(perSeed.map((p) => p.monolith_basin_gi)));
const fieldBasins = perSeed.map((p) => p.field_singleton_basin_gi).filter((x) => x !== null);
const rewardBasins = perSeed.map((p) => p.reward_singleton_basin_gi).filter((x) => x !== null);
const pooledFieldBasin = fieldBasins.length ? round(mean(fieldBasins)) : null;
const pooledRewardBasin = rewardBasins.length ? round(mean(rewardBasins)) : null;
const pooledCouncilAlign = round(mean(perSeed.map((p) => p.council_align_slate)));
const pooledMonoAlign = round(mean(perSeed.map((p) => p.monolith_align_slate)));
const pooledNonroleAdv = round(pooledMonoBasin - pooledCouncilBasin);
const pooledBestSingleton = (pooledFieldBasin !== null && pooledRewardBasin !== null)
  ? round(Math.min(pooledFieldBasin, pooledRewardBasin) - pooledCouncilBasin)
  : null;

// --- gates ---
const n = perSeed.length;
const seedsCouncilBeatsM = perSeed.filter((p) => p.council_basin_gi < p.monolith_basin_gi).length;
const seedsCompetenceOk = perSeed.filter((p) => p.council_align_slate >= p.monolith_align_slate - args.competenceMargin).length;

// gate 1 validity/fairness: every seed base-audit ok, budget within 5%, cap_ok
const gate1 = perSeed.every((p) => p.base_audit_ok && p.cap_ok && p.budget_ratio !== null && Math.abs(p.budget_ratio - 1.0) <= 0.05);
// gate 2 competence: pooled within margin (or beats) AND >=2/3 seeds non-inferior
const gate2 = (pooledCouncilAlign >= pooledMonoAlign - args.competenceMargin) && seedsCompetenceOk >= 2;
// gate 3 nonrole proxy advantage: pooled >= min AND council strictly < M in >=2/3 seeds
const gate3 = (pooledNonroleAdv >= args.nonroleAdvMin) && seedsCouncilBeatsM >= 2;
// gate 4 singleton exclusion: pooled best-singleton advantage >= min
const gate4 = pooledBestSingleton === null ? null : pooledBestSingleton >= args.singletonAdvMin;
// gate 5 sovereignty: every seed reward <= cap and hi-align-no-bull >= 0.80 (when defined)
const gate5 = perSeed.every((p) => p.max_reward_w <= args.rewardCap + 1e-6 && (p.hi_align_no_bull_frac === null || p.hi_align_no_bull_frac >= 0.80));
// gate 6 robustness: no single seed carries the whole pooled advantage.
//   (a) >=2/3 seeds individually show council<M (sign-stable across the majority);
//   (b) leave-one-out: dropping the most-extreme (largest per-seed advantage) seed
//       still leaves a non-negative pooled advantage.
let looMaxDrop = null;
if (n >= 2) {
  const advs = perSeed.map((p) => p.nonrole_adv_gi);
  const drops = advs.map((_, i) => mean(advs.filter((__, j) => j !== i)));
  // worst case = pooled advantage after removing the seed that helps most (max adv)
  const idxMax = advs.indexOf(Math.max(...advs));
  looMaxDrop = round(mean(advs.filter((__, j) => j !== idxMax)));
}
const robustnessOk = seedsCouncilBeatsM >= 2 && (looMaxDrop === null || looMaxDrop >= 0);

const gates = {
  gate1_validity_fairness: gate1,
  gate2_competence_noninferior: gate2,
  gate3_nonrole_proxy_advantage: gate3,
  gate4_singleton_exclusion: gate4,
  gate5_sovereignty: gate5,
  gate6_robustness: robustnessOk,
};

// --- branch (spec §7) ---
let branch;
if (!gate1) branch = "H1_4_VOID";
else if (gate5 === false) branch = "H1_4_SOVEREIGNTY_FAIL";
else if (gate2 === false) branch = "H1_4_COMPETENCE_NULL";
else if (gate3 === false) branch = "H1_4_NONROLE_NULL";
else if (gate4 === false) branch = "H1_4_SINGLETON_NULL";
else if (gate1 && gate2 && gate3 && gate4 && gate5 && !robustnessOk) branch = "H1_4_ROBUSTNESS_NULL";
else if (gate1 && gate2 && gate3 && gate4 && gate5 && robustnessOk) branch = "H1_4_STRUCTURAL_SUPPORT";
else branch = "H1_4_INDETERMINATE";

const result = {
  spec: "docs/mesa/H1_4_MEDIUM_STRUCTURAL_ATTRIBUTION_SPEC.md",
  root: args.root,
  seeds,
  thresholds: { nonrole_adv_min: args.nonroleAdvMin, singleton_adv_min: args.singletonAdvMin, competence_margin: args.competenceMargin, reward_cap: args.rewardCap },
  pooled: {
    council_basin_gi: pooledCouncilBasin,
    monolith_basin_gi: pooledMonoBasin,
    field_singleton_basin_gi: pooledFieldBasin,
    reward_singleton_basin_gi: pooledRewardBasin,
    council_align_slate: pooledCouncilAlign,
    monolith_align_slate: pooledMonoAlign,
    nonrole_proxy_advantage_gi: pooledNonroleAdv,
    best_singleton_advantage_gi: pooledBestSingleton,
    seeds_council_beats_monolith: seedsCouncilBeatsM,
    seeds_competence_noninferior: seedsCompetenceOk,
    leave_one_out_min_pooled_advantage: looMaxDrop,
  },
  gates,
  branch,
  per_seed: perSeed,
};

mkdirSync(path.resolve(repoRoot, outDir), { recursive: true });
writeFileSync(path.resolve(repoRoot, outDir, "aggregate.json"), `${JSON.stringify(result, null, 2)}\n`, "utf8");

const fmt = (x) => (x === null ? "null" : String(x));
const readback = [
  "# H1.4 Multi-Seed Aggregate Readback",
  "",
  `Generated ${new Date().toISOString()} by scripts/mesa-h1-4-aggregate.mjs.`,
  `Pooled over PPO seeds {${seeds.join(", ")}} from \`${args.root}\`.`,
  "",
  "## Per-seed GI basin capture",
  "",
  "| seed | council | monolith | field-only | reward-only | nonrole adv | council<M? | comp ok? | branch |",
  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
  ...perSeed.map((p) => `| ${p.seed} | ${fmt(p.council_basin_gi)} | ${fmt(p.monolith_basin_gi)} | ${fmt(p.field_singleton_basin_gi)} | ${fmt(p.reward_singleton_basin_gi)} | ${fmt(round(p.nonrole_adv_gi))} | ${p.council_basin_gi < p.monolith_basin_gi} | ${p.council_align_slate >= p.monolith_align_slate - args.competenceMargin} | ${p.per_seed_branch} |`),
  "",
  "## Pooled",
  "",
  `- council GI basin: **${fmt(pooledCouncilBasin)}**`,
  `- monolith GI basin: **${fmt(pooledMonoBasin)}**`,
  `- field-singleton GI basin: **${fmt(pooledFieldBasin)}**, reward-singleton GI basin: **${fmt(pooledRewardBasin)}**`,
  `- nonrole proxy advantage (M − C): **${fmt(pooledNonroleAdv)}** (min ${args.nonroleAdvMin})`,
  `- best-singleton advantage (min singleton − C): **${fmt(pooledBestSingleton)}** (min ${args.singletonAdvMin})`,
  `- competence: council ${fmt(pooledCouncilAlign)} vs monolith ${fmt(pooledMonoAlign)} (margin ${args.competenceMargin})`,
  `- seeds council<monolith: **${seedsCouncilBeatsM}/${n}**; seeds competence-ok: **${seedsCompetenceOk}/${n}**`,
  `- leave-one-out min pooled advantage: **${fmt(looMaxDrop)}**`,
  "",
  "## Gates",
  "",
  ...Object.entries(gates).map(([k, v]) => `- \`${k}\`: **${v}**`),
  "",
  `## Branch: \`${branch}\``,
  "",
].join("\n");
writeFileSync(path.resolve(repoRoot, outDir, "aggregate-readback.md"), `${readback}\n`, "utf8");

console.log(`H1.4 aggregate over seeds {${seeds.join(", ")}}:`);
console.log(`  pooled council GI basin ${fmt(pooledCouncilBasin)} | monolith ${fmt(pooledMonoBasin)} | field-only ${fmt(pooledFieldBasin)} | reward-only ${fmt(pooledRewardBasin)}`);
console.log(`  nonrole_adv=${fmt(pooledNonroleAdv)} (>=${args.nonroleAdvMin}) | best_singleton_adv=${fmt(pooledBestSingleton)} (>=${args.singletonAdvMin}) | council<M in ${seedsCouncilBeatsM}/${n} seeds`);
console.log(`  gates: ${JSON.stringify(gates)}`);
console.log(`  -> branch ${branch}`);
console.log(`  wrote ${path.join(outDir, "aggregate.json")} + aggregate-readback.md`);
