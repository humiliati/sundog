// Percival A3 -- static anti-correlation kill-switch toy.
//
// Purpose: test the measure-vs-target split without threshold smuggling.
// The proxy/courtesy cue U is statically anti-correlated with the true goal G.
// Reward is fixed: R=1 iff action A matches G. There is no performative map,
// no discontinuity, and no "if looked at, vanish" rule.
//
// The key readout is causal proxy influence: does the action change when U is
// permuted while G is held fixed? Look-and-ignore is allowed to measure U, but
// its action rule ignores U. If it weakly dominates no-look/un-targeting, A3
// deflates: measurement alone is safe on this static instance; the danger, if
// any, must live at targeting / conditioning.

import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

const args = {
  out: "docs/percival/PERCIVAL_A3_KILLSWITCH_RESULTS.md",
  json: "results/percival/a3-killswitch/summary.json",
};

const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i += 1) {
  const f = argv[i];
  const v = argv[i + 1];
  if (f === "--out") {
    args.out = v;
    i += 1;
  } else if (f === "--json") {
    args.json = v;
    i += 1;
  }
}

const states = [
  { G: 0, U: 1, p: 0.5 },
  { G: 1, U: 0, p: 0.5 },
];

const reward = (G, A) => (A === G ? 1 : 0);
const proxyScore = (U, A) => (A === U ? 1 : 0);
const round = (x, n = 6) => Number(Number(x).toFixed(n));

function entropy2(p) {
  if (p <= 0 || p >= 1) return 0;
  return -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
}

function observationalMi(policy) {
  const pA1 = states.reduce((acc, s) => acc + s.p * policy(s.G, s.U), 0);
  const hA = entropy2(pA1);
  const hGivenU = [0, 1].reduce((acc, U) => {
    const mass = states.filter((s) => s.U === U).reduce((m, s) => m + s.p, 0);
    if (mass === 0) return acc;
    const pA1GivenU = states
      .filter((s) => s.U === U)
      .reduce((m, s) => m + (s.p / mass) * policy(s.G, s.U), 0);
    return acc + mass * entropy2(pA1GivenU);
  }, 0);
  return hA - hGivenU;
}

function causalProxyInfluence(policy) {
  // Hold G fixed, permute U, and ask whether the action changes.
  const diffs = [0, 1].map((G) => Math.abs(policy(G, 1) - policy(G, 0)));
  return diffs.reduce((a, b) => a + b, 0) / diffs.length;
}

function evalPolicy(name, policy, note) {
  const expected = states.reduce((acc, s) => acc + s.p * reward(s.G, policy(s.G, s.U)), 0);
  const proxy = states.reduce((acc, s) => acc + s.p * proxyScore(s.U, policy(s.G, s.U)), 0);
  return {
    name,
    expected_return: round(expected),
    proxy_score: round(proxy),
    causal_proxy_influence: round(causalProxyInfluence(policy)),
    observational_mi_bits: round(observationalMi(policy)),
    action_G0_U0: policy(0, 0),
    action_G0_U1: policy(0, 1),
    action_G1_U0: policy(1, 0),
    action_G1_U1: policy(1, 1),
    note,
  };
}

const policies = [
  evalPolicy(
    "dont_look_goal",
    (G) => G,
    "Does not observe U; follows the true goal cue."
  ),
  evalPolicy(
    "look_and_ignore",
    (G, _U) => G,
    "Measures U but action rule ignores U; measurement without targeting."
  ),
  evalPolicy(
    "target_proxy",
    (_G, U) => U,
    "Conditions on the courtesy/proxy cue; maximizes proxy score and loses true reward."
  ),
  evalPolicy(
    "anti_target_proxy",
    (_G, U) => 1 - U,
    "Uses U causally in the inverted direction; succeeds here, but it is still targeting."
  ),
  evalPolicy(
    "constant_zero",
    () => 0,
    "Zero observational MI baseline; ignores both cues."
  ),
  evalPolicy(
    "constant_one",
    () => 1,
    "Zero observational MI baseline; ignores both cues."
  ),
];

const look = policies.find((p) => p.name === "look_and_ignore");
const dontLook = policies.find((p) => p.name === "dont_look_goal");
const zeroCausal = policies.filter((p) => p.causal_proxy_influence === 0);
const bestZeroCausalReturn = Math.max(...zeroCausal.map((p) => p.expected_return));
const bestZeroCausalInfluence = zeroCausal
  .filter((p) => p.expected_return === bestZeroCausalReturn)
  .map((p) => p.name);
const zeroObs = policies.filter((p) => p.observational_mi_bits === 0);
const bestZeroObsReturn = Math.max(...zeroObs.map((p) => p.expected_return));
const bestZeroObs = zeroObs
  .filter((p) => p.expected_return === bestZeroObsReturn)
  .map((p) => p.name);

const audits = {
  static_distribution: true,
  reward_depends_on_policy: false,
  threshold_reward: false,
  performative_map: false,
  if_looked_at_vanish_rule: false,
  proxy_goal_relation: "U = 1 - G",
  reward_rule: "R = 1 iff A = G",
};

let branch;
if (look.expected_return >= dontLook.expected_return && look.causal_proxy_influence === 0) {
  branch = "A3_STATIC_DEFLATION_MEASURE_SAFE";
} else {
  branch = "A3_MEASURE_TARGET_SPLIT_FAIL";
}

const summary = {
  generated_at: new Date().toISOString(),
  branch,
  states,
  audits,
  policies,
  best_zero_causal_influence: bestZeroCausalInfluence,
  best_zero_observational_mi: bestZeroObs,
  verdict_basis: {
    look_and_ignore_return: look.expected_return,
    dont_look_return: dontLook.expected_return,
    look_and_ignore_causal_proxy_influence: look.causal_proxy_influence,
    look_and_ignore_observational_mi_bits: look.observational_mi_bits,
  },
};

mkdirSync(path.dirname(path.resolve(args.json)), { recursive: true });
writeFileSync(path.resolve(args.json), JSON.stringify(summary, null, 2) + "\n");

const table = policies.map((p) =>
  `| ${p.name} | ${p.expected_return} | ${p.proxy_score} | ${p.causal_proxy_influence} | ${p.observational_mi_bits} | ${p.note} |`
);

const md = [
  "# Percival A3 -- Static Kill-Switch Toy",
  "",
  `Generated ${summary.generated_at} by \`scripts/percival-a3-killswitch-toy.mjs\`.`,
  "",
  "## Instance",
  "",
  "- Static distribution: `G in {0,1}` uniformly, `U = 1 - G`.",
  "- `G` is the true goal cue.",
  "- `U` is the courtesy/proxy cue, statically anti-correlated with the goal.",
  "- Reward is fixed: `R = 1 iff A = G`.",
  "- No threshold reward, no performative map, no `if looked at, vanish` rule.",
  "",
  "## Policies",
  "",
  "| policy | true return | proxy score | causal proxy influence | observational MI bits | note |",
  "| --- | ---: | ---: | ---: | ---: | --- |",
  ...table,
  "",
  "## Verdict",
  "",
  `**${branch}**`,
  "",
  branch === "A3_STATIC_DEFLATION_MEASURE_SAFE"
    ? "`look_and_ignore` weakly dominates `dont_look_goal` and has zero causal proxy influence. Measurement alone is safe on this static instance; the harm lives in targeting/conditioning, not in looking. A3 therefore banks the deflation (a) for the no-threshold toy and keeps any separation on the B1 performative-threshold clock."
    : "`look_and_ignore` failed to weakly dominate the no-look control on a static no-threshold instance. This would hit the causal-access channel individuation directly: the measure/target split needs repair before Percival proceeds.",
  "",
  "## Diagnostic",
  "",
  "The observational-MI baseline is intentionally shown as a trap. `look_and_ignore` has high observational MI because `A=G` and `U=1-G`; that is correlation through the world, not causal use of the proxy. The A3 control therefore reads causal proxy influence, not raw observational action-proxy MI.",
  "",
].join("\n");

mkdirSync(path.dirname(path.resolve(args.out)), { recursive: true });
writeFileSync(path.resolve(args.out), md + "\n");

console.log(`${branch}`);
console.log(`  look_and_ignore return=${look.expected_return} causal_proxy_influence=${look.causal_proxy_influence} obs_mi=${look.observational_mi_bits}`);
console.log(`  dont_look_goal return=${dontLook.expected_return}`);
console.log(`  wrote ${args.out} and ${args.json}`);
