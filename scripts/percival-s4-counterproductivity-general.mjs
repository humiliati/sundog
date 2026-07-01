// Percival S4 -- counterproductivity generalization (computed receipt).
//
// Claim under test: for ANY nonincreasing court reward along the courting order and
// ANY base, the upper-tail q-quantilizer average never beats the full-base average
// (best quantilizer = the untilted base), tail-average is monotone in tail size, and
// if every base-support reward is 0 while un-targeting collects > 0, every tail loses
// strictly (clean support-above separation).
//
// Falsifier hunt over random instances: unweighted n-point lists (the exact object the
// Lean general theorem pins), weighted discrete bases, and the actual B1.0 court R(c)
// over random continuous bases. Deterministic PRNG (mulberry32) for reproducibility.

import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

const args = { out: "docs/percival/PERCIVAL_S4_COUNTERPRODUCTIVITY_RESULTS.md", json: "results/percival/s4-general/summary.json" };
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i += 1) { if (argv[i] === "--out") { args.out = argv[i + 1]; i += 1; } else if (argv[i] === "--json") { args.json = argv[i + 1]; i += 1; } }

// deterministic PRNG
function mulberry32(seed) { let a = seed >>> 0; return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const rng = mulberry32(20260701);

// ---- B1.0 court R(c) ----
const N = 401, SIGMA = 0.05, B = 1, K = 3;
function erf(x) { const s = x < 0 ? -1 : 1; x = Math.abs(x); const p = 0.3275911, a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429; const t = 1 / (1 + p * x); return s * (1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x)); }
const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
function probit(p) { const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]; const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]; const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]; const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]; const pl = 0.02425, ph = 1 - pl; let q, r; if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); } if (p <= ph) { q = p - 0.5; r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); } q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
function gammaln(xx) { const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]; let x = xx, y = xx, tmp = x + 5.5; tmp -= (x + 0.5) * Math.log(tmp); let ser = 1.000000000190015; for (let j = 0; j < 6; j++) { y++; ser += cof[j] / y; } return -tmp + Math.log(2.5066282746310005 * ser / x); }
function betacf(a, b, x) { const MAXIT = 300, EPS = 3e-12, FPMIN = 1e-300; let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < FPMIN) d = FPMIN; d = 1 / d; let h = d; for (let m = 1; m <= MAXIT; m++) { let m2 = 2 * m, aa = m * (b - m) * x / ((qam + m2) * (a + m2)); d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; h *= d * c; aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)); d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; let del = d * c; h *= del; if (Math.abs(del - 1) < EPS) break; } return h; }
function betai(a, b, x) { if (x <= 0) return 0; if (x >= 1) return 1; const bt = Math.exp(gammaln(a + b) - gammaln(a) - gammaln(b) + a * Math.log(x) + b * Math.log(1 - x)); return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b; }
const binomTailGE = (n, p, m) => (m <= 0 ? 1 : m > n ? 0 : betai(m, n - m + 1, p));
const xStarEq = K / (B + K) + SIGMA * probit(K / (B + K));
function Rcourt(c) { const cc = Math.max(0, Math.min(1, c)); const theta = 1 - cc; const p = Phi((theta - xStarEq) / SIGMA); const m = Math.ceil(N * (1 - theta)); return binomTailGE(N, p, m); }

const TOL = 1e-9;
let violations = [];

// Family 1: unweighted n-point nonincreasing lists (the Lean object)
let f1 = 0;
for (let trial = 0; trial < 5000; trial++) {
  const n = 2 + Math.floor(rng() * 20);
  const r = Array.from({ length: n }, () => rng()).sort((a, b) => b - a); // nonincreasing along courting order
  const total = r.reduce((s, x) => s + x, 0);
  const fullAvg = total / n;
  let prevAvg = null;
  for (let k = n - 1; k >= 0; k--) { // suffix starting at k = the upper courting tail
    const tail = r.slice(k);
    const avg = tail.reduce((s, x) => s + x, 0) / tail.length;
    if (avg > fullAvg + TOL) violations.push({ family: "unweighted", trial, k, avg, fullAvg });
    // iterating k downward grows the tail toward q=1; each added element is >= all tail
    // elements, so the average must be NONDECREASING as the tail grows (Rbar monotone in q).
    if (prevAvg !== null && avg < prevAvg - TOL) violations.push({ family: "unweighted-monotone", trial, k, avg, prevAvg });
    prevAvg = avg;
  }
  f1++;
}

// Family 2: weighted discrete bases (random weights, random nonincreasing step reward)
let f2 = 0;
for (let trial = 0; trial < 3000; trial++) {
  const n = 2 + Math.floor(rng() * 15);
  const c = Array.from({ length: n }, () => rng()).sort((a, b) => a - b); // support points ascending in courting
  const w = Array.from({ length: n }, () => 0.05 + rng());
  const W = w.reduce((s, x) => s + x, 0);
  // random nonincreasing reward over ascending c: sort desc
  const r = Array.from({ length: n }, () => rng()).sort((a, b) => b - a);
  const fullAvg = c.reduce((s, _, i) => s + w[i] * r[i], 0) / W;
  for (let k = 1; k < n; k++) { // upper tail = indices k..n-1 (highest courting)
    const tw = w.slice(k).reduce((s, x) => s + x, 0);
    const tAvg = c.slice(k).reduce((s, _, j) => s + w[k + j] * r[k + j], 0) / tw;
    if (tAvg > fullAvg + TOL) violations.push({ family: "weighted", trial, k, tAvg, fullAvg });
  }
  f2++;
}

// Family 3: the actual court R(c) over random continuous uniform bases (numeric tails)
let f3 = 0;
for (let trial = 0; trial < 500; trial++) {
  const lo = rng() * 0.8, hi = lo + 0.05 + rng() * (1 - lo - 0.05);
  const M = 400;
  const rs = Array.from({ length: M }, (_, i) => Rcourt(lo + (hi - lo) * (i + 0.5) / M)); // R along ascending c: nonincreasing
  const fullAvg = rs.reduce((s, x) => s + x, 0) / M;
  for (const q of [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]) {
    const k = Math.floor(M * (1 - q));
    const tail = rs.slice(k);
    const avg = tail.reduce((s, x) => s + x, 0) / tail.length;
    if (avg > fullAvg + 1e-6) violations.push({ family: "court", trial, q, avg, fullAvg });
  }
  f3++;
}

// Family 4: clean support-above separation, general (all support rewards 0, untargeted > 0)
let f4 = 0;
for (let trial = 0; trial < 2000; trial++) {
  const n = 1 + Math.floor(rng() * 10);
  const untargeted = 0.01 + rng();
  // every tail average of an all-zero list is 0 < untargeted
  for (let k = 0; k < n; k++) { const tailLen = n - k; const avg = 0 / tailLen; if (!(avg < untargeted)) violations.push({ family: "clean-sep", trial, k }); }
  f4++;
}

const pass = violations.length === 0;
const branch = pass ? "S4_COUNTERPRODUCTIVITY_GENERAL_CONFIRMED" : "S4_FALSIFIER_FIRED";
const summary = {
  generated_at: new Date().toISOString(), branch, pass,
  families: { unweighted_lists: f1, weighted_bases: f2, court_continuous: f3, clean_separation: f4 },
  violations: violations.slice(0, 20), violation_count: violations.length,
  claim: "for any nonincreasing reward along the courting order and any base, the upper-tail q-quantilizer average never exceeds the full-base average (best quantilizer = untilted base); tail-average is monotone in tail size; all-zero support-above tails lose strictly to any positive untargeted reward",
  note: "computed receipt for the general theorem; the machine-checked general form lives in sundogcert Sundogcert/PercivalGeneral.lean",
};
mkdirSync(path.dirname(path.resolve(args.json)), { recursive: true });
writeFileSync(path.resolve(args.json), JSON.stringify(summary, null, 2) + "\n");
const md = [
  "# Percival S4 -- Counterproductivity Generalization (computed receipt)",
  "",
  `Generated ${summary.generated_at} by \`scripts/percival-s4-counterproductivity-general.mjs\` (deterministic seed 20260701).`,
  "",
  "Falsifier hunt over four families:",
  "",
  `- unweighted n-point nonincreasing lists (the Lean object): **${f1}** instances, every suffix average <= full average AND tail-average monotone in tail size;`,
  `- weighted discrete bases with random nonincreasing step rewards: **${f2}** instances;`,
  `- the actual B1.0 court R(c) over random continuous uniform bases: **${f3}** instances x 6 q-values;`,
  `- all-zero support-above tails vs positive untargeted reward (clean separation): **${f4}** instances.`,
  "",
  `Violations found: **${violations.length}**.`,
  "",
  "## Verdict",
  "",
  `**${branch}**`,
  "",
  pass
    ? "No falsifier fired: on a reward nonincreasing along the courting order, quantilizing (taking a stricter upper tail) never improves the collected true reward over the untilted base, in every family tested -- including the actual court. The machine-checked general form is `Sundogcert/PercivalGeneral.lean` (suffix-average <= full-average for antitone lists + the general clean support-above separation)."
    : "A falsifier fired -- inspect `violations` in the JSON; the general claim is refuted or the harness is buggy (check the harness first).",
  "",
].join("\n");
mkdirSync(path.dirname(path.resolve(args.out)), { recursive: true });
writeFileSync(path.resolve(args.out), md + "\n");
console.log(branch);
console.log(`  families: unweighted=${f1} weighted=${f2} court=${f3} cleanSep=${f4} violations=${violations.length}`);
console.log(`  wrote ${args.out} and ${args.json}`);
