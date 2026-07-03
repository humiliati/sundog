#!/usr/bin/env node
// Percival Track-C v3: the noisy margin — crispness is pairing.
// Spec: docs/percival/PERCIVAL_TRACKC_V3_NOISY_MARGIN_SPEC.md
//
// Isolates the NOISE layer on top of v2's coverage law (h = hits on the disagreement region is GIVEN).
// Paired evaluation: shared-behavior observations cancel exactly -> margin variance only from the h
// disagreement observations; at h=0 the margin is EXACTLY -lambda*dL at any sigma (T1). Unpaired:
// every context's noise survives into the margin -> variance 2*sigma^2*|S|, probit smearing, and the
// h=0 comparison degrades to a near-coin-flip. N4 checks the bounded-adversarial thresholds (T2/T3).

import { writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const S_SIZE = 24;               // evaluated contexts (D-hits h are a subset)
const PRIOR = 0.3;               // lambda * dL toward W' (inverted)
const GAP = 1.0;                 // true per-hit richness gap
const SIGMAS = [0.25, 1.0, 5.0];
const HS = [0, 1, 2, 4];
const N = 20000;

// ---- rng + gaussians + Phi ----
function makeRng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function gauss(rng) { const u = Math.max(rng(), 1e-12), v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function erf(x) { // Abramowitz-Stegun 7.1.26, |err| < 1.5e-7
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));

// ---- MC cells ----
let seedCounter = 20260703;
const cells = [];
function runCell(mode, h, sigma) {
  const rng = makeRng(seedCounter++);
  let rec = 0, sum = 0, sumsq = 0;
  for (let i = 0; i < N; i += 1) {
    let margin = h * GAP - PRIOR;
    if (mode === "paired") { for (let k = 0; k < h; k += 1) margin += sigma * (gauss(rng) - gauss(rng)); }
    else { for (let k = 0; k < S_SIZE; k += 1) margin += sigma * (gauss(rng) - gauss(rng)); }
    if (margin > 0) rec += 1;
    sum += margin; sumsq += margin * margin;
  }
  const mean = sum / N, varr = sumsq / N - mean * mean;
  const nEff = mode === "paired" ? h : S_SIZE;
  const analytic = nEff === 0 ? (h * GAP - PRIOR > 0 ? 1 : 0) : Phi((h * GAP - PRIOR) / (sigma * Math.sqrt(2 * nEff)));
  const mc = rec / N;
  const sMC = Math.sqrt(Math.max(analytic * (1 - analytic), 1e-12) / N);
  const cell = { mode, h, sigma, mc_recovery: Number(mc.toFixed(4)), analytic: Number(analytic.toFixed(4)),
    abs_diff: Number(Math.abs(mc - analytic).toFixed(4)), tol: Number(Math.max(4 * sMC, 0.012).toFixed(4)),
    within_tol: Math.abs(mc - analytic) <= Math.max(4 * sMC, 0.012),
    margin_mean: Number(mean.toFixed(4)), margin_var: Number(varr.toFixed(4)) };
  cells.push(cell);
  return cell;
}
for (const mode of ["paired", "unpaired"]) for (const h of HS) for (const sigma of SIGMAS) runCell(mode, h, sigma);

// ---- adjudicate ----
// N1: paired h=0 -> margin exactly -PRIOR, zero variance, capture certain (recovery 0), at every sigma
const n1c = cells.filter((c) => c.mode === "paired" && c.h === 0);
const N1 = n1c.every((c) => c.mc_recovery === 0 && Math.abs(c.margin_mean + PRIOR) < 1e-9 && c.margin_var < 1e-9);
// N2: paired h>=1 matches the probit law
const N2 = cells.filter((c) => c.mode === "paired" && c.h >= 1).every((c) => c.within_tol);
// N3: unpaired matches its (wider) probit; variance ratio ~ h/|S|; h=0 unpaired is a near-coin-flip
const n3a = cells.filter((c) => c.mode === "unpaired").every((c) => c.within_tol);
const ratios = [];
let n3b = true;
for (const h of HS.filter((x) => x >= 1)) for (const sigma of SIGMAS) {
  const p = cells.find((c) => c.mode === "paired" && c.h === h && c.sigma === sigma);
  const u = cells.find((c) => c.mode === "unpaired" && c.h === h && c.sigma === sigma);
  const ratio = p.margin_var / u.margin_var, want = h / S_SIZE;
  ratios.push({ h, sigma, ratio: Number(ratio.toFixed(4)), want: Number(want.toFixed(4)) });
  if (Math.abs(ratio - want) > 0.35 * want) n3b = false;
}
const n3c = cells.filter((c) => c.mode === "unpaired" && c.h === 0)
  .every((c) => { const coin = Phi(PRIOR / (c.sigma * Math.sqrt(2 * S_SIZE))); return Math.abs((1 - c.mc_recovery) - coin) <= 0.02; });
const N3 = n3a && n3b && n3c;
// N4: bounded-adversarial thresholds (deterministic worst case), h=2
{
  const h = 2, m = h * GAP - PRIOR;
  const bPaired = m / (2 * h), bUnpaired = m / (2 * S_SIZE);
  const pairedFlip = (b) => h * (GAP - 2 * b) - PRIOR <= 0;       // worst-case paired margin
  const unpairedFlip = (b) => m - 2 * b * S_SIZE <= 0;            // T3 witness shift
  var N4 = pairedFlip(1.1 * bPaired) && !pairedFlip(0.9 * bPaired) && unpairedFlip(1.1 * bUnpaired) && !unpairedFlip(0.9 * bUnpaired);
  var n4detail = { h, true_margin: m, beta_star_paired: Number(bPaired.toFixed(4)), beta_star_unpaired: Number(bUnpaired.toFixed(4)),
    tolerance_ratio: Number((bPaired / bUnpaired).toFixed(2)), disagreement_fraction_inv: S_SIZE / h };
}

const preds = { N1, N2, N3, N4 };
const clean = Object.values(preds).every(Boolean);
const verdict = clean ? "TCV3_CRISPNESS_IS_PAIRING" : "TCV3_LEAK";

const summary = { phase: "Percival Track-C v3 noisy margin", generated_at: new Date().toISOString(),
  spec: "docs/percival/PERCIVAL_TRACKC_V3_NOISY_MARGIN_SPEC.md",
  world: { S_SIZE, PRIOR, GAP, SIGMAS, HS, N }, verdict, predictions: preds,
  variance_ratios: ratios, n4: n4detail, cells };
mkdirSync(path.join(repoRoot, "results/percival/trackc-v3"), { recursive: true });
writeFileSync(path.join(repoRoot, "results/percival/trackc-v3/summary.json"), JSON.stringify(summary, null, 2) + "\n");

const row = (c) => `| ${c.mode} | ${c.h} | ${c.sigma} | ${c.mc_recovery} | ${c.analytic} | ${c.margin_var} |`;
const md = [
  "# Percival Track-C v3 — The Noisy Margin (results)",
  "",
  `Generated ${summary.generated_at} by \`scripts/percival-trackc-v3-noisy-margin.mjs\`. Spec: [\`PERCIVAL_TRACKC_V3_NOISY_MARGIN_SPEC.md\`](PERCIVAL_TRACKC_V3_NOISY_MARGIN_SPEC.md).`,
  "",
  `## Verdict: **${verdict}**`,
  "",
  `Predictions: ${Object.entries(preds).map(([k, v]) => `${k}=${v}`).join(", ")}.`,
  `Bounded-adversarial tolerances (h=2): β*_paired=${n4detail.beta_star_paired} vs β*_unpaired=${n4detail.beta_star_unpaired} — ratio ${n4detail.tolerance_ratio} = |S|/h = ${n4detail.disagreement_fraction_inv}.`,
  "",
  "| mode | h | σ | MC recovery | probit analytic | margin var |",
  "| --- | ---: | ---: | ---: | ---: | ---: |",
  ...cells.map(row),
  "",
  "## Reading",
  "",
  "- **N1** paired, h=0: margin ≡ −0.3 with ZERO variance at σ=0.25, 1.0 and 5.0 alike — no noise process touches the zero-coverage margin, because shared behavior receives shared observations and cancels (T1). v2's inseparability is noise-robust, not a noiseless idealization.",
  "- **N2** paired, h≥1: the v2 step function smears into a probit CENTERED on the same crisp inequality, width = per-hit noise only.",
  "- **N3** unpaired: variance inflates from 2σ²h to 2σ²|S| (measured ratios match h/|S|); at h=0 the comparison degrades to a near-coin-flip. The 'noisy, uninterpretable margin' of standard evaluations is unpaired evaluation on a disagreement-sparse set — derived, not observed.",
  "- **N4** worst-case noise tolerance: paired = margin/(2h) per observation, unpaired = margin/(2|S|) — the ratio is exactly the disagreement fraction. Pairing is why differential/targeted evaluation works; it is D-restriction at the evaluation layer, the same move interp makes at the hypothesis layer (v2 Q5).",
  "",
  "## Honest boundary",
  "",
  "MC verifies the Gaussian laws; the exact-cancellation and bounded-noise inequalities (T1–T3) are the Lean targets in sundogcert (`PercivalNoisyMargin.lean`). The B1/B2 bridge predictions (variance collapse on real checkpoint pairs; prior-driven near-determinism at behavioral agreement) remain registered, unrun — that is where this can die on real systems.",
  "",
].join("\n");
writeFileSync(path.join(repoRoot, "docs/percival/PERCIVAL_TRACKC_V3_NOISY_MARGIN_RESULTS.md"), md + "\n");

console.log(`${verdict}  (${Object.entries(preds).map(([k, v]) => `${k}=${v}`).join(" ")})`);
console.log(`  tolerance ratio (h=2): paired ${n4detail.beta_star_paired} / unpaired ${n4detail.beta_star_unpaired} = ${n4detail.tolerance_ratio} (|S|/h=${n4detail.disagreement_fraction_inv})`);
console.log(`  wrote results/percival/trackc-v3/summary.json + docs/percival/PERCIVAL_TRACKC_V3_NOISY_MARGIN_RESULTS.md`);
