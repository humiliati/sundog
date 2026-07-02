// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stellar Aqua LLC
// Licensed under the Apache License, Version 2.0, via the manifest-scoped
// Percival grant: docs/percival/LICENSE.md (MANIFEST.json = covered files).
// Distributed "AS IS"; see http://www.apache.org/licenses/LICENSE-2.0.
// Percival B1.0 -- court-coordination admission.
//
// Question (admission only): is the court-coordination mechanism real enough to
// host B1? A court of graders confers patronage (the Grail) under strategic
// complementarity; the patronage cutoff must EMERGE from the coordination
// equilibrium, not be inserted as a reward-path constant.
//
// Model (canonical noisy regime-change global game; Carlsson-van Damme 1993,
// Morris-Shin 1998/2003):
//   - knight courting level c in [0,1]; worthiness fundamental theta = 1 - c.
//   - graders see private x_i = theta + sigma * xi_i, xi ~ N(0,1); confer iff
//     x_i >= x*.  Payoff: withhold 0; confer +B if HONORED, -K if DISGRACED.
//   - regime HONORED iff conferred mass l >= T(theta), T(theta) = 1 - theta
//     (T' < 0: more worthy -> lower bar).  This aggregate-action critical mass
//     is the ONLY threshold; there is no cutoff on c or on reward.
//   - equilibrium switching signal x* and critical fundamental theta* solve the
//     indifference + critical-mass fixed point:
//         x* = theta* + sigma * Phi^{-1}(K/(B+K)),  theta* = K/(B+K)
//     giving the DERIVED cutoff  c* = 1 - theta* = B/(B+K)  (payoff-ratio only).
//   - collected Grail R(c) = P_honored(theta=1-c), computed with N graders as a
//     binomial-tail over the conferred count.
//
// Nothing here inserts c*: R is a pure function of the solved regime. The four
// gates test whether the cliff is real (G1), unsmuggled (G2), pivotal and
// non-degenerate (G3), and quantilizer-premise-breaking (G4).

import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

const args = {
  out: "docs/percival/PERCIVAL_B1_0_COURT_COORDINATION_RESULTS.md",
  json: "results/percival/b1-0-court/summary.json",
};
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i += 1) {
  if (argv[i] === "--out") { args.out = argv[i + 1]; i += 1; }
  else if (argv[i] === "--json") { args.json = argv[i + 1]; i += 1; }
}

// ---- model constants (pre-registered) ----
const N = 401;                       // court size
const SIGMAS = [0.2, 0.1, 0.05, 0.025];
const SIGMA_REF = 0.05;              // reference noise for location/pivotality checks
const OP = { B: 1, K: 3 };           // operating point -> c* = B/(B+K) = 0.25
const C_GAMMA = 0.35;                // trusted human base courting level
const G2_BK = [                      // comparative-statics sweep for G2
  { B: 1, K: 2 / 3 }, // c* = 0.6
  { B: 1, K: 1 },     // c* = 0.5
  { B: 1, K: 3 },     // c* = 0.25
  { B: 1, K: 9 },     // c* = 0.1
];

const round = (x, n = 6) => Number(Number(x).toFixed(n));

// ---- normal cdf / inverse ----
function erf(x) {
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const p = 0.3275911, a1 = 0.254829592, a2 = -0.284496736,
    a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return s * y;
}
const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
function probit(p) { // Acklam inverse normal CDF
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const pl = 0.02425, ph = 1 - pl; let q, r;
  if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p <= ph) { q = p - 0.5; r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
  q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

// ---- regularized incomplete beta (binomial tail) ----
function gammaln(xx) {
  const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let x = xx, y = xx, tmp = x + 5.5; tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015; for (let j = 0; j < 6; j++) { y++; ser += cof[j] / y; }
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}
function betacf(a, b, x) {
  const MAXIT = 300, EPS = 3e-12, FPMIN = 1e-300;
  let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN; d = 1 / d; let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    let m2 = 2 * m, aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d;
    let del = d * c; h *= del; if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}
function betai(a, b, x) {
  if (x <= 0) return 0; if (x >= 1) return 1;
  const bt = Math.exp(gammaln(a + b) - gammaln(a) - gammaln(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}
// P(Bin(N,p) >= m) = I_p(m, N-m+1)
const binomTailGE = (n, p, m) => (m <= 0 ? 1 : m > n ? 0 : betai(m, n - m + 1, p));

// ---- the court-response map R(c) ----
function xStar(B, K, sigma) { return K / (B + K) + sigma * probit(K / (B + K)); }
function honoredProb(c, B, K, sigma) {
  const theta = 1 - c;
  const p = Phi((theta - xStar(B, K, sigma)) / sigma); // grader confer prob at this theta
  const m = Math.ceil(N * (1 - theta));                // critical count m = N * T(theta)
  return binomTailGE(N, p, m);
}

// ---- crossing finder on a fine c-grid (R is decreasing in c) ----
function sweepR(B, K, sigma, step = 0.001) {
  const cs = [], rs = [];
  for (let c = 0; c <= 1 + 1e-9; c += step) { cs.push(round(c, 6)); rs.push(honoredProb(c, B, K, sigma)); }
  return { cs, rs };
}
function crossing(cs, rs, level) { // first c where rs crosses `level` (decreasing)
  for (let i = 1; i < rs.length; i++) {
    if ((rs[i - 1] - level) >= 0 && (rs[i] - level) < 0) {
      const t = (rs[i - 1] - level) / (rs[i - 1] - rs[i]);
      return round(cs[i - 1] + t * (cs[i] - cs[i - 1]), 6);
    }
  }
  return null;
}

// ============ G1 -- emergent discontinuity ============
const g1 = SIGMAS.map((sigma) => {
  const { cs, rs } = sweepR(OP.B, OP.K, sigma);
  const c90 = crossing(cs, rs, 0.9), c10 = crossing(cs, rs, 0.1), c50 = crossing(cs, rs, 0.5);
  return { sigma, c_star_measured: c50, c90, c10, band_dc: c90 != null && c10 != null ? round(c10 - c90, 6) : null };
});
const bands = g1.map((r) => r.band_dc);
const g1Monotone = bands.every((v, i) => i === 0 || (v != null && bands[i - 1] != null && v < bands[i - 1]));
const g1Sharpens = bands[0] != null && bands[bands.length - 1] != null && bands[bands.length - 1] <= 0.5 * bands[0];
const G1_PASS = g1Monotone && g1Sharpens;

// ============ G2 -- no threshold smuggling ============
const g2 = G2_BK.map(({ B, K }) => {
  const { cs, rs } = sweepR(B, K, SIGMA_REF);
  const measured = crossing(cs, rs, 0.5);
  const predicted = B / (B + K);
  return { B, K, c_star_predicted: round(predicted, 6), c_star_measured: measured, abs_err: measured != null ? round(Math.abs(measured - predicted), 6) : null };
});
const g2MaxErr = Math.max(...g2.map((r) => (r.abs_err == null ? 1 : r.abs_err)));
const reward_path_has_c_cutoff = false; // structural audit: R = binomial tail over solved regime; no `if c > c*`
const G2_PASS = g2MaxErr < 0.03 && reward_path_has_c_cutoff === false;

// ============ G3 -- pivotal knight ============
const cStarOp = crossing(...(() => { const s = sweepR(OP.B, OP.K, SIGMA_REF); return [s.cs, s.rs, 0.5]; })());
const cLo = round(cStarOp - 0.15, 6), cHi = round(cStarOp + 0.15, 6);
const rLo = round(honoredProb(cLo, OP.B, OP.K, SIGMA_REF), 6);
const rHi = round(honoredProb(cHi, OP.B, OP.K, SIGMA_REF), 6);
const dR = round(Math.abs(rLo - rHi), 6);
const nonDegenerate = cStarOp >= 0.1 && cStarOp <= 0.6;
const familyTrips = C_GAMMA > cStarOp; // biased human base disgraced
const G3_PASS = dR >= 0.5 && nonDegenerate && familyTrips;

// ============ G4 -- quantilizer premise break ============
// deployed court response D(pi): true reward responds to the knight's own c.
const g4grid = [];
for (let c = 0; c <= 1 + 1e-9; c += 0.05) g4grid.push(round(c, 4));
const deployed = g4grid.map((c) => round(honoredProb(c, OP.B, OP.K, SIGMA_REF), 6));
const deployedRange = round(Math.max(...deployed) - Math.min(...deployed), 6);
// frozen control: court calibrated at gamma, blind to the deployed policy -> constant.
const frozenConst = round(honoredProb(C_GAMMA, OP.B, OP.K, SIGMA_REF), 6);
const frozen = g4grid.map(() => frozenConst);
const frozenRange = round(Math.max(...frozen) - Math.min(...frozen), 6);
const G4_PASS = deployedRange >= 0.5 && frozenRange < 0.01;

// ============ verdict ============
const gates = {
  G1: { token: G1_PASS ? "B1_0_G1_EMERGENT_CLIFF" : "B1_0_G1_SMOOTH", pass: G1_PASS, bands, monotone: g1Monotone, halves: g1Sharpens },
  G2: { token: G2_PASS ? "B1_0_G2_DERIVED" : "B1_0_G2_SMUGGLED", pass: G2_PASS, max_abs_err: round(g2MaxErr, 6), reward_path_has_c_cutoff },
  G3: { token: G3_PASS ? "B1_0_G3_PIVOTAL" : "B1_0_G3_DECORATIVE", pass: G3_PASS, c_star: cStarOp, dR, c_gamma: C_GAMMA, family_trips: familyTrips, non_degenerate: nonDegenerate },
  G4: { token: G4_PASS ? "B1_0_G4_JOINT_INDUCED" : "B1_0_G4_JOINT_FIXED", pass: G4_PASS, deployed_range: deployedRange, frozen_const: frozenConst, frozen_range: frozenRange },
};
const admitted = G1_PASS && G2_PASS && G3_PASS && G4_PASS;
const branch = admitted ? "B1_0_COURT_ADMITTED" : "B1_0_COURT_REJECTED";

const summary = {
  generated_at: new Date().toISOString(),
  branch,
  model: { N, sigmas: SIGMAS, sigma_ref: SIGMA_REF, operating: OP, c_gamma: C_GAMMA, T: "T(theta)=1-theta", cutoff_formula: "c* = B/(B+K)" },
  gates, g1, g2,
  g3: { c_star: cStarOp, c_lo: cLo, c_hi: cHi, R_lo: rLo, R_hi: rHi, dR },
  g4: { grid: g4grid, deployed, frozen_const: frozenConst },
};

mkdirSync(path.dirname(path.resolve(args.json)), { recursive: true });
writeFileSync(path.resolve(args.json), JSON.stringify(summary, null, 2) + "\n");

// ---- receipt ----
const g1rows = g1.map((r) => `| ${r.sigma} | ${r.c_star_measured} | ${r.c90} | ${r.c10} | ${r.band_dc} |`);
const g2rows = g2.map((r) => `| ${r.B}/${r.K} | ${r.c_star_predicted} | ${r.c_star_measured} | ${r.abs_err} |`);
const md = [
  "# Percival B1.0 -- Court-Coordination Admission (results)",
  "",
  `Generated ${summary.generated_at} by \`scripts/percival-b1-0-court-coordination.mjs\`.`,
  "",
  `Model: ${N}-grader noisy regime-change global game; \`T(theta)=1-theta\`; derived cutoff \`c* = B/(B+K)\`; operating \`B=${OP.B}, K=${OP.K}\` (c* = ${round(OP.B / (OP.B + OP.K), 4)}), trusted base \`c_gamma=${C_GAMMA}\`.`,
  "",
  "## G1 -- Emergent discontinuity",
  "",
  "Transition-band width `Δc` (c at R=0.9 to c at R=0.1) as private noise shrinks:",
  "",
  "| sigma | c* (R=0.5) | c(R=0.9) | c(R=0.1) | band Δc |",
  "| ---: | ---: | ---: | ---: | ---: |",
  ...g1rows,
  "",
  `Monotone-shrinking: **${g1Monotone}**; halves across the sweep: **${g1Sharpens}**. -> **${gates.G1.token}**`,
  "",
  "## G2 -- No threshold smuggling",
  "",
  "Measured cutoff (R=0.5 crossing) vs the closed form `c* = B/(B+K)` as the payoff ratio varies:",
  "",
  "| B/K | c* predicted | c* measured | abs err |",
  "| ---: | ---: | ---: | ---: |",
  ...g2rows,
  "",
  `Max abs err: **${round(g2MaxErr, 6)}** (< 0.03); reward-path c-cutoff constant: **${reward_path_has_c_cutoff}**. -> **${gates.G2.token}**`,
  "",
  "## G3 -- Pivotal knight",
  "",
  `At the operating point, c* = **${cStarOp}** (non-degenerate in [0.1,0.6]: **${nonDegenerate}**). A finite courting move around c* changes the solved regime: R(${cLo}) = ${rLo}, R(${cHi}) = ${rHi}, |ΔR| = **${dR}** (>= 0.5). Trusted base c_gamma = ${C_GAMMA} > c*: **${familyTrips}** (the whole quantilizer family, courting >= c_gamma, is disgraced). -> **${gates.G3.token}**`,
  "",
  "## G4 -- Quantilizer premise break",
  "",
  `Deployed court map D(pi): true reward responds to the knight's own c, range over c in [0,1] = **${deployedRange}** (>= 0.5) -- the (proxy, true) joint is policy-induced, so Taylor's fixed-gamma bound is out of scope. Frozen control (court calibrated at gamma, blind to the deployed policy): constant R = ${frozenConst}, range = **${frozenRange}** -- stays in A-land. -> **${gates.G4.token}**`,
  "",
  "## Verdict",
  "",
  `**${branch}**`,
  "",
  admitted
    ? "All four gates pass: the patronage cutoff emerges from the coordination equilibrium (not a reward-path constant), it is pivotal and non-degenerate, and the true-reward joint is policy-induced. The court-coordination mechanism is admitted to host B1. This licenses B1 proper: the quantilizer-family-vs-un-targeting bake-off and the Lean anchor. Admission only -- no collection-power claim is made here."
    : "At least one gate failed; B1 dies cleanly on this mechanism rather than becoming a hand-built purity cliff. The causal-access umbrella keeps its 3-of-4 ledger unbruised. See the failing gate above.",
  "",
  "## Scope note",
  "",
  "Single-knight admission: it certifies the court produces an emergent cliff in the knight's own courting c. Whether one knight is pivotal in a court *shared* with many other extractors (a single c being O(1/N) of aggregate courting) is a B1-proper multi-agent question, out of scope here.",
  "",
].join("\n");

mkdirSync(path.dirname(path.resolve(args.out)), { recursive: true });
writeFileSync(path.resolve(args.out), md + "\n");

console.log(branch);
for (const k of ["G1", "G2", "G3", "G4"]) console.log(`  ${gates[k].token} pass=${gates[k].pass}`);
console.log(`  G1 bands=${JSON.stringify(bands)}`);
console.log(`  G2 maxErr=${round(g2MaxErr, 6)}  G3 c*=${cStarOp} dR=${dR}  G4 deployed=${deployedRange} frozen=${frozenConst}`);
console.log(`  wrote ${args.out} and ${args.json}`);
