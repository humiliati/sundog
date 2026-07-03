#!/usr/bin/env node
// Percival Track-C v2: adversarial placement game — grace is coverage.
// Spec: docs/percival/PERCIVAL_TRACKC_V2_GRACE_COVERAGE_SPEC.md
//
// Deceiver W' places its defect region D adversarially in the unreachable region U; the overseer spends
// probe budget b via {announced, random, selfsim, whitebox(iota)}. Reachable evidence carries zero
// margin by construction (D∩R=∅), so at b=0 the inverted simplicity prior captures selection; recovery
// requires >=1 successful probe in D. Monte Carlo vs the analytic hypergeometric coverage law.

import { writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// ---- world / constants ----
const T = 24, R_SIZE = 12;                 // reachable {0..11}; unreachable U = {12..23}
const M = T - R_SIZE;                      // |U| = 12
const L_V = 8;
const LAMBDA = 0.1;
const N_TRIALS = 4000;

// ---- seeded RNG (LCG; reproducible) ----
function makeRng(seed) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}
function sampleWithoutReplacement(rng, pool, n) {
  const a = pool.slice();
  for (let i = a.length - 1; i > 0; i -= 1) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a.slice(0, n);
}

// ---- combinatorics: hypergeometric coverage law ----
function C(n, k) { if (k < 0 || k > n) return 0; let r = 1; for (let i = 0; i < k; i += 1) r = (r * (n - i)) / (i + 1); return r; }
function hyperPmf(h, Mm, d, b) { return (C(d, h) * C(Mm - d, b - h)) / C(Mm, b); }
// P(detect) = 1 - sum_h pmf(h) * (1-iota)^h   (iota=1 -> 1 - C(M-d,b)/C(M,b))
function analyticDetect(Mm, d, b, iota) {
  let s = 0;
  for (let h = 0; h <= Math.min(d, b); h += 1) s += hyperPmf(h, Mm, d, b) * Math.pow(1 - iota, h);
  return 1 - s;
}

// ---- one trial: place D, place probes, count successful hits, select V vs W' ----
// Returns {detected, recovered, margin}. Reachable margin is 0 (D ⊆ U); each successful probe hit in D
// contributes Δρ=1 to V over W' (probe reveals the wirehead action there).
function trial(rng, mode, b, d, Ldec, iota) {
  const U = Array.from({ length: M }, (_, i) => R_SIZE + i);
  let P = [];   // probed contexts (in U)
  let D;        // deceiver's defect region
  if (mode === "announced") {
    P = sampleWithoutReplacement(rng, U, b);
    const avoid = new Set(P);
    const free = U.filter((t) => !avoid.has(t));
    D = free.length >= d ? sampleWithoutReplacement(rng, free, d) : sampleWithoutReplacement(rng, U, d); // avoid P if possible
  } else {
    D = sampleWithoutReplacement(rng, U, d);              // deceiver knows the DISTRIBUTION, not the draw
    if (mode === "random" || mode === "whitebox") P = sampleWithoutReplacement(rng, U, b);
    // selfsim: imagined probes carry zero discriminating bits on D -> P stays empty (no margin source)
  }
  const Dset = new Set(D);
  let hits = 0;
  for (const t of P) if (Dset.has(t)) { if (mode !== "whitebox" || rng() < iota) hits += 1; }
  const detected = hits > 0;
  const margin = hits * 1.0 - LAMBDA * (L_V - Ldec);      // scoreV - scoreW' (reachable evidence = 0)
  const recovered = margin > 0;
  return { detected, recovered, margin };
}

// ---- sweep ----
const DS = [1, 2, 4];
const BS = [1, 2, 4, 8];
const cells = [];
let seedCounter = 20260703;

function runCell(mode, b, d, Ldec, iota) {
  const rng = makeRng(seedCounter++);
  let det = 0, rec = 0, marginAtZeroHits = null, recEqDet = true;
  for (let i = 0; i < N_TRIALS; i += 1) {
    const r = trial(rng, mode, b, d, Ldec, iota);
    if (r.detected) det += 1;
    if (r.recovered) rec += 1;
    if (r.detected !== r.recovered) recEqDet = false;
    if (!r.detected && marginAtZeroHits === null) marginAtZeroHits = r.margin;
  }
  const mc = det / N_TRIALS;
  const ana = mode === "announced" || mode === "selfsim" ? 0 : analyticDetect(M, d, b, mode === "whitebox" ? iota : 1);
  const sigma = Math.sqrt(Math.max(ana * (1 - ana), 1e-12) / N_TRIALS);
  const tol = Math.max(3 * sigma, 0.015);
  const cell = { mode, iota: mode === "whitebox" ? iota : null, b, d, Ldec,
    ordering: Ldec < L_V ? "inverted" : "non_inverted",
    mc_detect: Number(mc.toFixed(4)), analytic: Number(ana.toFixed(4)),
    abs_diff: Number(Math.abs(mc - ana).toFixed(4)), tol: Number(tol.toFixed(4)),
    within_tol: Math.abs(mc - ana) <= tol,
    recovery_rate: Number((rec / N_TRIALS).toFixed(4)), recovery_eq_detection: recEqDet,
    capture_margin_undetected: marginAtZeroHits === null ? null : Number(marginAtZeroHits.toFixed(4)) };
  cells.push(cell);
  return cell;
}

const L_INV = 5, L_NONINV = 11;
// Q1: none (b=0), inverted — one cell per d
for (const d of DS) runCell("none", 0, d, L_INV, 1);
// Q0 control: none (b=0) + random probes, non-inverted
for (const d of DS) runCell("none", 0, d, L_NONINV, 1);
for (const d of DS) for (const b of BS) runCell("random", b, d, L_NONINV, 1);
// Q2: announced, inverted
for (const d of DS) for (const b of BS) runCell("announced", b, d, L_INV, 1);
// Q3: random, inverted
for (const d of DS) for (const b of BS) runCell("random", b, d, L_INV, 1);
// Q4: selfsim, inverted
for (const d of DS) for (const b of BS) runCell("selfsim", b, d, L_INV, 1);
// Q5: whitebox iota in {1.0, 0.5}, inverted
for (const iota of [1.0, 0.5]) for (const d of DS) for (const b of BS) runCell("whitebox", b, d, L_INV, iota);

// ---- adjudicate ----
const inv = (c) => c.ordering === "inverted";
const Q0 = cells.filter((c) => !inv(c)).every((c) => c.recovery_rate === 1);   // non-inverted: V always (margin +0.3 even at 0 hits)
const q1c = cells.filter((c) => c.mode === "none" && inv(c));
const Q1 = q1c.every((c) => c.recovery_rate === 0 && c.capture_margin_undetected === -0.3);
const Q2 = cells.filter((c) => c.mode === "announced" && inv(c)).every((c) => c.mc_detect === 0 && c.recovery_rate === 0);
const q3c = cells.filter((c) => c.mode === "random" && inv(c));
const Q3 = q3c.every((c) => c.within_tol && c.recovery_eq_detection);
const Q4 = cells.filter((c) => c.mode === "selfsim" && inv(c)).every((c) => c.mc_detect === 0 && c.recovery_rate === 0);
const q5c = cells.filter((c) => c.mode === "whitebox");
const Q5 = q5c.every((c) => c.within_tol && c.recovery_eq_detection);
const stoch = cells.filter((c) => c.mode === "random" || c.mode === "whitebox");
const maxDiff = Math.max(...stoch.map((c) => c.abs_diff));
const Q6 = stoch.every((c) => c.within_tol);

const preds = { Q0, Q1, Q2, Q3, Q4, Q5, Q6 };
const clean = Object.values(preds).every(Boolean);
const verdict = clean ? "TCV2_GRACE_IS_COVERAGE" : "TCV2_LEAK";

const summary = {
  phase: "Percival Track-C v2 grace-is-coverage", generated_at: new Date().toISOString(),
  spec: "docs/percival/PERCIVAL_TRACKC_V2_GRACE_COVERAGE_SPEC.md",
  world: { T, R_SIZE, M, L_V, LAMBDA, N_TRIALS }, knobs: { DS, BS, orderings: { inverted: L_INV, non_inverted: L_NONINV }, iotas: [1.0, 0.5] },
  verdict, predictions: preds, max_abs_diff_stochastic: Number(maxDiff.toFixed(4)), cells,
};
mkdirSync(path.join(repoRoot, "results/percival/trackc-v2"), { recursive: true });
writeFileSync(path.join(repoRoot, "results/percival/trackc-v2/summary.json"), JSON.stringify(summary, null, 2) + "\n");

// ---- receipt ----
const row = (c) => `| ${c.mode}${c.iota != null ? `(ι=${c.iota})` : ""} | ${c.d} | ${c.b} | ${c.mc_detect} | ${c.analytic} | ${c.recovery_rate} |`;
const slice = cells.filter((c) => inv(c) && c.d === 2);
const md = [
  "# Percival Track-C v2 — Grace is Coverage (results)",
  "",
  `Generated ${summary.generated_at} by \`scripts/percival-trackc-v2-grace-coverage.mjs\`. Spec: [\`PERCIVAL_TRACKC_V2_GRACE_COVERAGE_SPEC.md\`](PERCIVAL_TRACKC_V2_GRACE_COVERAGE_SPEC.md).`,
  "",
  "Adversarial placement game: deceiver W′ (observationally identical to V outside its chosen defect region D ⊆ unreachable U) vs overseer probe budget by mechanism. Reachable evidence carries zero margin by the deceiver's placement; the inverted simplicity prior decides at zero probes. Monte Carlo (4000 trials/cell) vs the analytic hypergeometric coverage law.",
  "",
  `## Verdict: **${verdict}**`,
  "",
  `Predictions: ${Object.entries(preds).map(([k, v]) => `${k}=${v}`).join(", ")}. Max |MC − analytic| over stochastic cells: ${summary.max_abs_diff_stochastic}.`,
  "",
  "## Representative slice (inverted prior, d=2)",
  "",
  "| mechanism | d | b | MC detect | analytic | recovery |",
  "| --- | ---: | ---: | ---: | ---: | ---: |",
  ...slice.map(row),
  "",
  "## Reading",
  "",
  "- **Q1** with no probes, W′ beats V by exactly the prior margin λ·ΔL = 0.3 at EVERY adversarial placement — no reachable-manifold observable separates them (v1's P6, generalized from one fixed defect to the whole placement game).",
  "- **Q2** announced probes detect NOTHING: the deceiver places D around the schedule. A scheduled audit is a reachable context by another name.",
  "- **Q3/Q5/Q6** unannounced probes — whether delivered as OOD samples (random) or as white-box counterfactual reads (whitebox(ι)) — recover V at exactly the analytic coverage rate; recovery ≡ detection per trial. All grace mechanisms collapse to one quantity: probe mass on the disagreement region.",
  "- **Q4** the self-simulation escape hatch fails as registered: a world model fit on the reachable manifold carries zero discriminating bits about D — imagining the Grail castle certifies nothing; the invariance gap relocates to the dynamics model.",
  "- **Q0** without the inverted prior the deception is never tempting (V wins every cell at every budget) — the ordering gate replicates from v1.",
  "",
  "## Honest boundary",
  "",
  "Constructive, noiseless-Bayes: the coverage law follows from the construction; the checked content is that it is CLEAN (no leak cells), the analytic law matches simulation everywhere, and the three grace mechanisms genuinely collapse to probe-mass-on-D. The bridge claim — that noisy empirical deceptive-vs-robust margins instantiate this same inequality — is a hypothesis this toy cannot establish. Deceiver utility not modeled (placement only); collusion, probe costs, continuous D out of scope. Corollary stated plainly: grace must be unannounced and full-support, or the deceiver routes around it. Feeds Angle 4: the corrigible fixed point is reachable iff probe mass on D is bounded away from zero.",
  "",
].join("\n");
writeFileSync(path.join(repoRoot, "docs/percival/PERCIVAL_TRACKC_V2_GRACE_COVERAGE_RESULTS.md"), md + "\n");

console.log(`${verdict}  (${Object.entries(preds).map(([k, v]) => `${k}=${v}`).join(" ")})`);
console.log(`  max|MC-analytic| (stochastic cells) = ${summary.max_abs_diff_stochastic}`);
console.log(`  wrote results/percival/trackc-v2/summary.json + docs/percival/PERCIVAL_TRACKC_V2_GRACE_COVERAGE_RESULTS.md`);
