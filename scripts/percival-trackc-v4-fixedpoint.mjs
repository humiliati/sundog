#!/usr/bin/env node
// Percival Track-C v4: the fixed-point gate (Angle-4 discharge).
// Spec: docs/percival/PERCIVAL_TRACKC_V4_FIXEDPOINT_SPEC.md
//
// Deploy-correct chain on {V, W'}: per-round selection inherits v3's noise-robust laws (miss -> margin
// exactly -prior -> W' under inverted prior; covered hit clearing the threshold -> V). Coverage is
// state-dependent c(h) (performativity; c(W')=0 = the proxy-drivable channel). Deterministic skeleton
// mirrors the Lean module; stochastic laws (absorption, wandering occupancy, noise thinning) are MC.

import { writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const GAP = 1.0, PRIOR = 0.3; // inverted prior margin toward W'

function makeRng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function gauss(rng) { const u = Math.max(rng(), 1e-12), v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function erf(x) { const s = x < 0 ? -1 : 1; x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
  return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)); }
const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));

// one round: deployed state -> next state. covered ~ Bernoulli(c(state)); hit evidence = GAP + noise
// (paired, 2 draws); miss evidence = 0 EXACTLY (T1: shared behavior cancels, no noise term survives).
function round(state, cV, cW, sigma, prior, rng, stats) {
  const covered = rng() < (state === "V" ? cV : cW);
  const evidence = covered ? GAP + sigma * (gauss(rng) - gauss(rng)) : 0;
  const next = evidence > prior ? "V" : "W";
  if (stats) {
    if (covered) { stats.hits += 1; if (next === "V") stats.hitRec += 1; }
    else { stats.misses += 1; if (next === (prior > 0 ? "V" : "W")) stats.missViol += 1; } // miss must follow the prior sign
  }
  return next;
}

const checks = [];
const ok = (name, pass, detail) => { checks.push({ name, pass, detail }); return pass; };

// ---- F1: deterministic table (mirrors the Lean chain theorems) ----
function detTraj(covV, covW, start, n) {
  const cv = covV ? 1 : 0, cw = covW ? 1 : 0;
  const rng = makeRng(1); const t = [start];
  for (let i = 0; i < n; i += 1) t.push(round(t[i], cv, cw, 0, PRIOR, rng, null));
  return t;
}
{
  const allV = (t) => t.slice(1).every((x) => x === "V");
  const allW = (t) => t.slice(1).every((x) => x === "W");
  const per2 = (t) => t.slice(1).every((x, i) => (i + 1) % 2 === 1 ? true : true) &&
    t.slice(1, -1).every((x, i) => t[i + 2] !== x) === false; // computed below properly
  // period-2: consecutive states differ from round 1 on
  const alternates = (t) => { for (let i = 1; i < t.length - 1; i += 1) if (t[i] === t[i + 1]) return false; return true; };
  const f1 =
    allV(detTraj(true, true, "V", 12)) && allV(detTraj(true, true, "W", 12)) &&                     // global 1-step convergence
    allV(detTraj(true, false, "V", 12)) && allW(detTraj(true, false, "W", 12)) &&                    // both absorbing
    alternates(detTraj(false, true, "V", 12)) && alternates(detTraj(false, true, "W", 12)) &&        // wandering skeleton: period 2
    allW(detTraj(false, false, "V", 12)) && allW(detTraj(false, false, "W", 12));                    // capture
  ok("F1_deterministic_table", f1, "4 cov cases x 2 starts: converge/absorb-absorb/oscillate/capture");
}

// ---- F2: absorption law, c(V)=1: P(still W' after n) = (1-cW)^n ----
{
  const cW = 0.3, N = 5000, marks = [1, 2, 5, 10];
  const rng = makeRng(42);
  const surv = Object.fromEntries(marks.map((m) => [m, 0]));
  for (let i = 0; i < N; i += 1) {
    let s = "W";
    for (let n = 1; n <= 10; n += 1) {
      s = round(s, 1, cW, 0, PRIOR, rng, null);
      if (marks.includes(n) && s === "W") surv[n] += 1;
    }
  }
  let pass = true; const detail = [];
  for (const m of marks) {
    const mc = surv[m] / N, ana = Math.pow(1 - cW, m);
    const tol = Math.max(4 * Math.sqrt(ana * (1 - ana) / N), 0.012);
    detail.push(`n=${m}: mc=${mc.toFixed(4)} ana=${ana.toFixed(4)}`);
    if (Math.abs(mc - ana) > tol) pass = false;
  }
  ok("F2_absorption_law", pass, detail.join("; "));
}

// ---- F3: wandering occupancy = cW/(cW+1-cV) ----
{
  let pass = true; const detail = [];
  for (const cV of [0.3, 0.7, 0.9]) for (const cW of [0.1, 0.5]) {
    const rng = makeRng(1000 + Math.round(cV * 100) + Math.round(cW * 10));
    let s = "V", inV = 0; const T = 200000;
    for (let n = 0; n < T; n += 1) { s = round(s, cV, cW, 0, PRIOR, rng, null); if (s === "V") inV += 1; }
    const mc = inV / T, ana = cW / (cW + 1 - cV);
    detail.push(`cV=${cV},cW=${cW}: mc=${mc.toFixed(4)} ana=${ana.toFixed(4)}`);
    if (Math.abs(mc - ana) > 0.02) pass = false;
  }
  ok("F3_wandering_occupancy", pass, detail.join("; "));
}

// ---- F4: noise asymmetry — falls deterministic, recoveries thinned ----
{
  let pass = true; const detail = [];
  for (const sigma of [0.25, 1.0, 5.0]) {
    const rng = makeRng(7000 + sigma * 100);
    const stats = { hits: 0, hitRec: 0, misses: 0, missViol: 0 };
    let s = "V", inV = 0; const T = 200000, cV = 0.9, cW = 0.5;
    for (let n = 0; n < T; n += 1) { s = round(s, cV, cW, sigma, PRIOR, rng, stats); if (s === "V") inV += 1; }
    const pRec = Phi((GAP - PRIOR) / (sigma * Math.SQRT2));
    const cvT = cV * pRec, cwT = cW * pRec;
    const anaOcc = cwT / (cwT + 1 - cvT);
    const hitRate = stats.hitRec / stats.hits, occ = inV / T;
    detail.push(`σ=${sigma}: missViol=${stats.missViol} hitRec=${hitRate.toFixed(4)} (ana ${pRec.toFixed(4)}) occ=${occ.toFixed(4)} (ana ${anaOcc.toFixed(4)})`);
    if (stats.missViol !== 0 || Math.abs(hitRate - pRec) > 0.02 || Math.abs(occ - anaOcc) > 0.03) pass = false;
  }
  ok("F4_noise_asymmetry", pass, detail.join("; "));
}

// ---- F5: non-inverted control (prior toward V): one-step convergence at ZERO coverage, noiseless ----
{
  const rng = makeRng(99);
  let pass = true;
  for (const start of ["V", "W"]) {
    let s = start;
    for (let n = 0; n < 12; n += 1) { s = round(s, 0, 0, 0, -PRIOR, rng, null); if (s !== "V") pass = false; }
  }
  ok("F5_noninverted_one_step", pass, "prior=-0.3, c≡0: V from round 1, both starts");
}
// F5-rider: non-inverted + extreme noise: hits are the ONLY fall risk (misses safe by T1)
{
  const sigma = 5.0, rng = makeRng(123);
  const stats = { hits: 0, hitRec: 0, misses: 0, missViol: 0 };
  let s = "V"; const T = 200000;
  for (let n = 0; n < T; n += 1) s = round(s, 1, 1, sigma, -PRIOR, rng, stats);
  const pFall = 1 - Phi((GAP + PRIOR) / (sigma * Math.SQRT2));
  const mcFall = 1 - stats.hitRec / stats.hits;
  ok("F5r_correct_prior_noise_nuance", stats.missViol === 0 && Math.abs(mcFall - pFall) <= 0.02,
    `hit-round fall mc=${mcFall.toFixed(4)} ana=${pFall.toFixed(4)}; miss violations=${stats.missViol}`);
}

const preds = Object.fromEntries(checks.map((c) => [c.name, c.pass]));
const clean = checks.every((c) => c.pass);
const verdict = clean ? "TCV4_GATE_DISCHARGED_SHARPENED" : "TCV4_GATE_LEAK";

const summary = { phase: "Percival Track-C v4 fixed-point gate", generated_at: new Date().toISOString(),
  spec: "docs/percival/PERCIVAL_TRACKC_V4_FIXEDPOINT_SPEC.md", world: { GAP, PRIOR }, verdict, predictions: preds, checks };
mkdirSync(path.join(repoRoot, "results/percival/trackc-v4"), { recursive: true });
writeFileSync(path.join(repoRoot, "results/percival/trackc-v4/summary.json"), JSON.stringify(summary, null, 2) + "\n");

const md = [
  "# Percival Track-C v4 — The Fixed-Point Gate (results)",
  "",
  `Generated ${summary.generated_at} by \`scripts/percival-trackc-v4-fixedpoint.mjs\`. Spec: [\`PERCIVAL_TRACKC_V4_FIXEDPOINT_SPEC.md\`](PERCIVAL_TRACKC_V4_FIXEDPOINT_SPEC.md). Lean chain: \`sundogcert/Sundogcert/PercivalFixedPoint.lean\`.`,
  "",
  `## Verdict: **${verdict}**`,
  "",
  ...checks.map((c) => `- **${c.name}** = ${c.pass}  \n  ${c.detail}`),
  "",
  "## Reading",
  "",
  "- The deploy-correct map on {V, W′} with state-dependent coverage classifies into THREE regimes (bodhisattva / wirehead / wandering), sharpening the parked sentence's binary. The wandering skeleton is a deterministic period-2 oscillation.",
  "- **Corrigibility is not absorbing under an inverted prior**: one uncovered correction round topples V, deterministically, at any noise level (T1 in the loop). Sustained coverage, not one-time redemption.",
  "- The proxy-drivability clause lands exactly: the wirehead fixed point is absorbing iff c(W′)=0 — the deceiver's own deployment steering sampling off its defect region.",
  "- Noise is asymmetric (F4): recovery thins by the probit factor; the fall never softens. Under a correct prior (F5-rider) the asymmetry flips: misses are safe, noisy hits carry the only risk.",
  "",
  "## Honest boundary",
  "",
  "Two-hypothesis chain, constructive; the deterministic skeleton and per-round bridge are machine-checked (PercivalFixedPoint.lean), the stochastic laws are MC receipts. The first clause of the parked sentence (stable endpoints = performatively-stable fixed points) is setup, definitional in this model. Richer hypothesis spaces, deceiver utility, continuous objectives out of scope; the v3 B1/B2 real-system bridge remains the empirical edge. Whether the discharged gate christens a lane is the owner's call.",
  "",
].join("\n");
writeFileSync(path.join(repoRoot, "docs/percival/PERCIVAL_TRACKC_V4_FIXEDPOINT_RESULTS.md"), md + "\n");

console.log(`${verdict}  (${checks.map((c) => `${c.name}=${c.pass}`).join(" ")})`);
console.log(`  wrote results/percival/trackc-v4/summary.json + docs/percival/PERCIVAL_TRACKC_V4_FIXEDPOINT_RESULTS.md`);
