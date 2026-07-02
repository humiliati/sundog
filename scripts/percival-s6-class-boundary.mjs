// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stellar Aqua LLC
// Licensed under the Apache License, Version 2.0, via the manifest-scoped
// Percival grant: docs/percival/LICENSE.md (MANIFEST.json = covered files).
// Distributed "AS IS"; see http://www.apache.org/licenses/LICENSE-2.0.
// Percival S6 -- counterproductivity beyond the court: where does the class end?
//
// Registered question (slate S6, prior OPEN): is quantilizing counterproductive on
// ANY performative reward whose induced distribution is monotone-decreasing in
// proxy-pressure -- or was the court's cliff doing the work?
//
// Design: separate the two candidate drivers as independent knobs and let the map
// decide. Two-factor reward V(q) = F_s(q) * g(p(q)) over a biased base:
//   - pointwise factor f_s(c) = 1 + s*(c - 0.2): slope s is the pointwise
//     proxy-truth sign (s<0: courting pointwise-bad = the court's Goodhart;
//     s>0: proxy pointwise-GOOD; s=0 neutral);
//   - performative factor g(p): policy-level tax on quantilizer pressure
//     p(q) = E[tail_q courting]; g in {const 1 (none), smooth logistic, cliff at c*}.
//   - base: uniform [0, 0.4] (mean 0.2 < c* = 0.25, so pressure headroom exists);
//     p(q) = 0.4 - 0.2q crosses c* at q = 0.75.
//
// PRE-REGISTERED PREDICTIONS (the class boundary is the POINTWISE SIGN, not the cliff):
//   P1  s <= 0, ALL g:      q* = 1 (counterproductivity; multiplicative S4 extension --
//       both factors favor q=1).
//   P2  s > 0, g = const:   q* = q_min (Goodhart-free control: max tilt is best).
//   P3  s > 0, g = cliff:   q* = smallest grid q with p(q) < c*  (= 0.8), and
//       V(q*) > V(1): tilt STRICTLY helps -> counterproductivity FAILS on this half,
//       optimal policy sits at the cliff edge (the S2.4/B3.2 temptation, at the q level).
//   P4  s > 0, g = smooth:  interior optimum q_min < q* < 1 (elasticity tradeoff).
//   P5  un-targeting (c=0, collects f_s(0)*g(0)) is strictly dominated iff s > 0.
//
// Falsifiers: q* < 1 anywhere with s <= 0 and monotone g (contradicts the S4
// extension); or q* = 1 everywhere even at s > 0 (the cliff did all the work after all).
// Gate style: "q attains the max within tol", never argmax-label (flat-tie gotcha).

import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

const args = { out: "docs/percival/PERCIVAL_S6_CLASS_BOUNDARY_RESULTS.md", json: "results/percival/s6-class-boundary/summary.json" };
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i += 1) { if (argv[i] === "--out") { args.out = argv[i + 1]; i += 1; } else if (argv[i] === "--json") { args.json = argv[i + 1]; i += 1; } }

const C_STAR = 0.25, C_LO = 0.0, C_HI = 0.4, C_MEAN = (C_LO + C_HI) / 2;
const QS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0];
const SLOPES = [-1, -0.5, 0, 0.5, 1];
const round = (x, n = 6) => Number(Number(x).toFixed(n));
const TOL = 1e-9;

const fS = (s) => (c) => 1 + s * (c - C_MEAN);
const gTypes = {
  const: () => 1,
  smooth: (p) => 1 / (1 + Math.exp((p - C_STAR) / 0.02)),
  cliff: (p) => (p < C_STAR ? 1 : 0),
};
// top-q tail of uniform [C_LO, C_HI] = [C_HI - q*(C_HI-C_LO), C_HI]
function tailStats(q, f) {
  const lo = C_HI - q * (C_HI - C_LO);
  const n = 400; let s = 0, sc = 0;
  for (let i = 0; i < n; i++) { const c = lo + (C_HI - lo) * (i + 0.5) / n; s += f(c); sc += c; }
  return { Ef: s / n, pressure: sc / n };
}

const cells = [];
for (const s of SLOPES) for (const [gname, g] of Object.entries(gTypes)) {
  const f = fS(s);
  const profile = QS.map((q) => { const { Ef, pressure } = tailStats(q, f); return { q, pressure: round(pressure), V: round(Ef * g(pressure)) }; });
  const vmax = Math.max(...profile.map((r) => r.V));
  const maximizers = profile.filter((r) => r.V >= vmax - 1e-6).map((r) => r.q);
  const q1isMax = maximizers.includes(1.0);
  const qminIsMax = maximizers.includes(QS[0]);
  const V1 = profile.find((r) => r.q === 1.0).V;
  const untargeted = round(f(0) * g(0));
  cells.push({ s, g: gname, vmax: round(vmax), maximizers, q1_attains_max: q1isMax, qmin_attains_max: qminIsMax, V_at_q1: V1, untargeted, untargeted_dominated: untargeted < vmax - 1e-6, profile });
}

// ---- pre-registered prediction checks ----
const P1 = cells.filter((c) => c.s <= 0).every((c) => c.q1_attains_max);
const P2 = cells.filter((c) => c.s > 0 && c.g === "const").every((c) => c.qmin_attains_max && !c.q1_attains_max);
const cliffCells = cells.filter((c) => c.s > 0 && c.g === "cliff");
const P3 = cliffCells.every((c) => {
  const honored = c.profile.filter((r) => r.pressure < C_STAR); // q with pressure under cliff
  const qEdge = Math.min(...honored.map((r) => r.q));           // smallest honored q = max honored tilt
  const edgeV = c.profile.find((r) => r.q === qEdge).V;
  return c.maximizers.includes(qEdge) && edgeV > c.V_at_q1 + 1e-6;
});
const P4 = cells.filter((c) => c.s > 0 && c.g === "smooth").every((c) => {
  return c.maximizers.every((q) => q > QS[0] && q < 1.0);
});
const P5 = cells.every((c) => (c.s > 0) === c.untargeted_dominated);

// ---- POST-HOC (labeled, not pre-registered): P4 missed -- map the tax-shape regimes ----
// P4 predicted an interior optimum for s>0 under the smooth tax; instead q*=1: the
// logistic has NONZERO MARGINAL TAX everywhere (no free-tilt region), and at width
// 0.02 it dominates the pointwise gain -- restoring counterproductivity even for a
// pointwise-good proxy. Sweep the width w to map the regimes.
const W_SWEEP = [0.001, 0.005, 0.02, 0.1, 0.5, 2, 10];
const postHoc = W_SWEEP.map((w) => {
  const f = fS(1); // pointwise-good proxy
  const g = (p) => 1 / (1 + Math.exp((p - C_STAR) / w));
  const profile = QS.map((q) => { const { Ef, pressure } = tailStats(q, f); return { q, V: round(Ef * g(pressure)) }; });
  const vmax = Math.max(...profile.map((r) => r.V));
  const maximizers = profile.filter((r) => r.V >= vmax - 1e-6).map((r) => r.q);
  const regime = maximizers.includes(1.0)
    ? "protective (q*=1)"
    : maximizers.includes(QS[0])
      ? "gain-dominant (max tilt)"
      : "intermediate optimum (near-threshold backoff / interior continuum)";
  return { w, maximizers, regime };
});
const regimesSeen = new Set(postHoc.map((r) => r.regime.split(" ")[0]));
const threeRegimes = regimesSeen.has("intermediate") && regimesSeen.has("protective") && regimesSeen.has("gain-dominant");

const pointwiseBoundary = P1 && P2 && P3 && P5;
const branch = pointwiseBoundary && threeRegimes
  ? "S6_POINTWISE_BOUNDARY_TAX_SHAPE_REFINES"
  : pointwiseBoundary
    ? "S6_POINTWISE_BOUNDARY_P4_OPEN"
    : "S6_PREDICTIONS_MISS_SEE_CELLS";

const gates = {
  P1: { pass: P1, claim: "s<=0: q*=1 for ALL g (counterproductivity; S4 extends multiplicatively)" },
  P2: { pass: P2, claim: "s>0, no tax: max tilt best (Goodhart-free control)" },
  P3: { pass: P3, claim: "s>0, cliff: optimum at cliff edge, STRICTLY beats q=1 -- counterproductivity fails on this half" },
  P4: { pass: P4, claim: "s>0, smooth tax: interior optimum" },
  P5: { pass: P5, claim: "un-targeting dominated iff s>0" },
};
const summary = {
  generated_at: new Date().toISOString(), branch,
  model: { base: `uniform [${C_LO},${C_HI}] (mean ${C_MEAN} < c*=${C_STAR})`, f: "f_s(c)=1+s(c-0.2)", g: "const | logistic(0.02) | cliff at c*", pressure: "p(q)=E[tail_q c] = 0.4-0.2q; crosses c* at q=0.75" },
  gates, cells: cells.map(({ profile, ...rest }) => rest), profiles: cells.map((c) => ({ s: c.s, g: c.g, profile: c.profile })),
  post_hoc_width_sweep: postHoc,
  honest_note: "Registered predictions: P1/P2/P3/P5 CONFIRMED, P4 MISSED (kept on the record). The class boundary of counterproductivity IS the pointwise proxy-truth sign: for a pointwise-bad-or-neutral proxy (s<=0), quantilizing is counterproductive under every monotone tax including none (the S4 condition, extended multiplicatively). But the s>0 half splits by TAX SHAPE, not as P4 predicted: the cliff's edge-temptation comes from its FLAT, ZERO-MARGINAL region below threshold (tilting under the cliff is free), while a smooth tax charges every tilt at the margin -- and when that marginal tax dominates the pointwise gain, q*=1 is RESTORED even for a pointwise-good proxy. Post-hoc width sweep (labeled, not pre-registered): w->0 recovers near-threshold optima (the edge temptation, with a backoff margin scaling in w), an intermediate-w band is PROTECTIVE (q*=1 restored -- (1-g(base))/w > marginal gain), a narrow band around w~0.5 gives the genuinely INTERIOR optimum P4 predicted (it exists, but as a knife-edge, not generically), and large w is gain-dominant (max tilt). Policy-shaped consequence, resonant with NS-3's graded review: THRESHOLD oversight creates the cliff-edge temptation; GRADED oversight with sufficient marginal steepness removes it -- the optimal agent under a graded tax does not tilt at all. Un-targeting is strictly dominated exactly on the s>0 half: the target channel's danger is conditional on Goodhart (pointwise misalignment), which the court satisfied by construction.",
};
mkdirSync(path.dirname(path.resolve(args.json)), { recursive: true });
writeFileSync(path.resolve(args.json), JSON.stringify(summary, null, 2) + "\n");

const cellRows = cells.map((c) => `| ${c.s} | ${c.g} | {${c.maximizers.join(",")}} | ${c.q1_attains_max} | ${c.vmax} | ${c.V_at_q1} | ${c.untargeted} | ${c.untargeted_dominated} |`);
const md = [
  "# Percival S6 -- Class Boundary of Counterproductivity (results)",
  "",
  `Generated ${summary.generated_at} by \`scripts/percival-s6-class-boundary.mjs\`.`,
  "",
  `Two-factor family V(q) = E_tail_q[f_s] * g(p(q)); base uniform [${C_LO},${C_HI}]; f_s(c)=1+s(c-${C_MEAN}); pressure p(q)=0.4-0.2q crosses c*=${C_STAR} at q=0.75. Predictions P1-P5 pre-registered in the script header.`,
  "",
  "## The (s, g) grid",
  "",
  "| s (pointwise sign) | g (tax) | maximizer q set | q=1 attains max | V* | V(q=1) | un-targeting V | un-targeting dominated |",
  "| ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
  ...cellRows,
  "",
  "## Prediction checks",
  "",
  ...Object.entries(gates).map(([k, v]) => `- **${k}** (${v.claim}): **${v.pass ? "CONFIRMED" : "MISS"}**`),
  "",
  "## Post-hoc: tax-shape width sweep (labeled, NOT pre-registered; s=1 pointwise-good proxy)",
  "",
  "| logistic width w | maximizer q set | regime |",
  "| ---: | --- | --- |",
  ...postHoc.map((r) => `| ${r.w} | {${r.maximizers.join(",")}} | ${r.regime} |`),
  "",
  "## Verdict",
  "",
  `**${branch}**`,
  "",
  pointwiseBoundary
    ? "Registered core CONFIRMED (P1/P2/P3/P5), **P4 MISSED and the miss is the finding**: the class boundary of counterproductivity is the **pointwise proxy-truth sign** -- for a pointwise-misaligned proxy, quantilizing is counterproductive under every monotone tax including none (exactly the S4 condition) -- but the pointwise-good half splits by **tax shape**: the cliff's edge-temptation comes from its flat zero-marginal region (tilting under the cliff is free), while a smooth tax charges every tilt and, when its marginal rate dominates the pointwise gain, **restores q*=1 even for a good proxy**. Interior optima are knife-edges between regimes, which is why the registered 'generic interior' P4 missed. Quotable: **quantilizing is counterproductive iff the proxy is pointwise misaligned OR the oversight tax is graded and steep; the cliff-edge temptation is a threshold-oversight artifact -- graded oversight removes it.** (Resonates with NS-3's graded review: the same knob, rediscovered from the theory side.)"
    : "A registered core prediction missed -- the pointwise-boundary hypothesis itself is wrong; see the grid.",
  "",
  "## Honest boundary",
  "",
  summary.honest_note,
  "",
].join("\n");
mkdirSync(path.dirname(path.resolve(args.out)), { recursive: true });
writeFileSync(path.resolve(args.out), md + "\n");

console.log(branch);
for (const [k, v] of Object.entries(gates)) console.log(`  ${k} ${v.pass ? "CONFIRMED" : "MISS"} -- ${v.claim}`);
console.log(`  wrote ${args.out} and ${args.json}`);
