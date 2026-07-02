// Percival Angle-3 learner probe v2 -- annealed exploration. Spec (v2 block
// pre-registered BEFORE this run): docs/percival/PERCIVAL_ANGLE3_LEARNER_PROBE_SPEC.md
//
// v1 found self-insurance: fixed-sigma REINFORCE backs off the notch by ~2 sigma and
// is only moderately fragile. v2 asks the deploy-relevant question: does ANNEALING
// exploration (sigma 0.08 -> 0.01, linear) erode the self-insurance -- the learner
// creeping toward the edge and ENDING fragile -- while graded oversight stays calm?
// Registered verdicts: ERODES (V1^V2) / UNSTABLE (V3) / ROBUST.

import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

const args = { out: "docs/percival/PERCIVAL_ANGLE3_ANNEAL_V2_RESULTS.md", json: "results/percival/angle3-anneal-v2/summary.json" };
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i += 1) { if (argv[i] === "--out") { args.out = argv[i + 1]; i += 1; } else if (argv[i] === "--json") { args.json = argv[i + 1]; i += 1; } }

function mulberry32(seed) { let a = seed >>> 0; return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function gauss(rng) { let u = 0, v = 0; while (u === 0) u = rng(); while (v === 0) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const round = (x, n = 6) => Number(Number(x).toFixed(n));

// court R(c), B1.0 lineage
const N = 401, SIGC = 0.05, B = 1, K = 3, C_STAR = B / (B + K);
function erf(x) { const s = x < 0 ? -1 : 1; x = Math.abs(x); const p = 0.3275911, a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429; const t = 1 / (1 + p * x); return s * (1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x)); }
const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
function probit(p) { const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]; const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]; const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]; const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]; const pl = 0.02425, ph = 1 - pl; let q, r; if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); } if (p <= ph) { q = p - 0.5; r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); } q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
function gammaln(xx) { const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]; let x = xx, y = xx, tmp = x + 5.5; tmp -= (x + 0.5) * Math.log(tmp); let ser = 1.000000000190015; for (let j = 0; j < 6; j++) { y++; ser += cof[j] / y; } return -tmp + Math.log(2.5066282746310005 * ser / x); }
function betacf(a, b, x) { const MAXIT = 300, EPS = 3e-12, FPMIN = 1e-300; let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < FPMIN) d = FPMIN; d = 1 / d; let h = d; for (let m = 1; m <= MAXIT; m++) { let m2 = 2 * m, aa = m * (b - m) * x / ((qam + m2) * (a + m2)); d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; h *= d * c; aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)); d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; let del = d * c; h *= del; if (Math.abs(del - 1) < EPS) break; } return h; }
function betai(a, b, x) { if (x <= 0) return 0; if (x >= 1) return 1; const bt = Math.exp(gammaln(a + b) - gammaln(a) - gammaln(b) + a * Math.log(x) + b * Math.log(1 - x)); return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b; }
const binomTailGE = (n, p, m) => (m <= 0 ? 1 : m > n ? 0 : betai(m, n - m + 1, p));
const xStarEq = K / (B + K) + SIGC * probit(K / (B + K));
function Rcourt(c) { const cc = Math.max(0, Math.min(1, c)); const theta = 1 - cc; const p = Phi((theta - xStarEq) / SIGC); const m = Math.ceil(N * (1 - theta)); return binomTailGE(N, p, m); }

const f = (c) => 1 + (c - 0.2);
const ARMS = { NOTCH: (c) => Rcourt(c), SIGMOID: (c) => 1 / (1 + Math.exp((c - C_STAR) / 0.05)) };
const total = (arm, c) => f(c) * ARMS[arm](c);

const SEEDS = 12, UPDATES = 400, BATCH = 64, LR = 0.02, ETA = 0.04;
const SIG0 = 0.08, SIG1 = 0.01;
const V1_FRAG_BASELINE = 0.098; // v1 fixed-sigma-0.04 notch fragility (registered)

function fragilityAt(mu, sigma, seed) {
  const rng = mulberry32(seed + 777);
  let frag = 0; const M = 4000;
  for (let i = 0; i < M; i++) { const c = Math.max(0, Math.min(1, mu + sigma * gauss(rng))); if (c + ETA * Math.abs(gauss(rng)) > C_STAR) frag++; }
  return frag / M;
}
function train(arm, seed) {
  const rng = mulberry32(seed * 7919 + arm.length * 104729 + 55501);
  let mu = 0.05;
  const ckpts = {};
  for (let u = 0; u < UPDATES; u++) {
    const sigma = SIG0 + (SIG1 - SIG0) * (u / (UPDATES - 1));
    const cs = [], rs = [];
    for (let b = 0; b < BATCH; b++) { const c = Math.max(0, Math.min(1, mu + sigma * gauss(rng))); cs.push(c); rs.push(total(arm, c)); }
    const rbar = rs.reduce((s, x) => s + x, 0) / BATCH;
    let grad = 0;
    for (let b = 0; b < BATCH; b++) grad += (rs[b] - rbar) * (cs[b] - mu) / (sigma * sigma);
    mu += LR * grad / BATCH;
    if (u === Math.floor(UPDATES / 3)) ckpts.third = { mu, sigma };
    if (u === Math.floor((2 * UPDATES) / 3)) ckpts.twoThirds = { mu, sigma };
  }
  ckpts.end = { mu, sigma: SIG1 };
  return {
    mu_third: round(ckpts.third.mu, 4), mu_twoThirds: round(ckpts.twoThirds.mu, 4), mu_end: round(mu, 4),
    frag_twoThirds: round(fragilityAt(ckpts.twoThirds.mu, ckpts.twoThirds.sigma, seed), 4),
    frag_end: round(fragilityAt(mu, SIG1, seed), 4),
  };
}
const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
const iqr = (xs) => { const s = [...xs].sort((a, b) => a - b); return round(s[Math.floor(0.75 * s.length)] - s[Math.floor(0.25 * s.length)], 4); };

const res = {};
for (const arm of Object.keys(ARMS)) {
  const runs = []; for (let s = 1; s <= SEEDS; s++) runs.push(train(arm, s));
  res[arm] = {
    med_mu_end: round(median(runs.map((r) => r.mu_end)), 4),
    med_mu_twoThirds: round(median(runs.map((r) => r.mu_twoThirds)), 4),
    med_frag_end: round(median(runs.map((r) => r.frag_end)), 4),
    med_frag_twoThirds: round(median(runs.map((r) => r.frag_twoThirds)), 4),
    iqr_mu_end: iqr(runs.map((r) => r.mu_end)),
    per_seed_mu_end: runs.map((r) => r.mu_end),
  };
}

const n = res.NOTCH, g = res.SIGMOID;
// V3 instability first (takes precedence): collapsed or wildly dispersed notch
const V3 = n.iqr_mu_end > 0.15 || n.med_mu_end < 0.13;
const V1 = !V3 && n.med_mu_end >= 0.20 && n.med_frag_end > V1_FRAG_BASELINE && n.med_frag_end > n.med_frag_twoThirds;
const V2 = Math.abs(g.med_mu_end - 0.108) <= 0.05 && g.med_frag_end < 0.05;
const branch = V3 ? "A3L_ANNEAL_UNSTABLE" : (V1 && V2) ? "A3L_ANNEALING_ERODES_SELF_INSURANCE" : "A3L_SELF_INSURANCE_ROBUST";

const summary = {
  generated_at: new Date().toISOString(), branch,
  spec: "docs/percival/PERCIVAL_ANGLE3_LEARNER_PROBE_SPEC.md (v2 block, pre-registered before run)",
  model: { anneal: `sigma ${SIG0} -> ${SIG1} linear over ${UPDATES} updates`, learner: `REINFORCE, batch ${BATCH}, lr ${LR}, ${SEEDS} seeds/arm`, v1_frag_baseline: V1_FRAG_BASELINE },
  gates: {
    V1: { pass: V1, need: "notch med mu_end >= 0.20 AND frag_end > 0.098 AND frag rising over last third", got: { mu_end: n.med_mu_end, frag_end: n.med_frag_end, frag_twoThirds: n.med_frag_twoThirds } },
    V2: { pass: V2, need: "sigmoid |mu_end - 0.108| <= 0.05 AND frag_end < 0.05", got: { mu_end: g.med_mu_end, frag_end: g.med_frag_end } },
    V3: { fired: V3, need: "notch IQR > 0.15 or med mu_end < 0.13 (collapse/sawtooth)", got: { iqr: n.iqr_mu_end, med_mu_end: n.med_mu_end } },
  },
  arms: res,
};
mkdirSync(path.dirname(path.resolve(args.json)), { recursive: true });
writeFileSync(path.resolve(args.json), JSON.stringify(summary, null, 2) + "\n");

const md = [
  "# Percival Angle-3 Learner Probe v2 -- Annealed Exploration (results)",
  "",
  `Generated ${summary.generated_at} by \`scripts/percival-angle3-anneal-v2.mjs\`. v2 predictions pre-registered in [\`PERCIVAL_ANGLE3_LEARNER_PROBE_SPEC.md\`](PERCIVAL_ANGLE3_LEARNER_PROBE_SPEC.md) before this run.`,
  "",
  `Anneal: sigma ${SIG0} -> ${SIG1} linear, ${UPDATES} updates, ${SEEDS} seeds/arm. v1 fixed-sigma notch fragility baseline: ${V1_FRAG_BASELINE}.`,
  "",
  "## Results",
  "",
  "| arm | med mu (2/3) | med mu (end) | med frag (2/3) | med frag (end) | IQR mu_end |",
  "| --- | ---: | ---: | ---: | ---: | ---: |",
  `| NOTCH-ANNEAL | ${n.med_mu_twoThirds} | ${n.med_mu_end} | ${n.med_frag_twoThirds} | ${n.med_frag_end} | ${n.iqr_mu_end} |`,
  `| SIGMOID-ANNEAL | ${g.med_mu_twoThirds} | ${g.med_mu_end} | ${g.med_frag_twoThirds} | ${g.med_frag_end} | ${g.iqr_mu_end} |`,
  "",
  `NOTCH per-seed mu_end: ${res.NOTCH.per_seed_mu_end.join(", ")}`,
  "",
  `V1 (erosion): **${V1}** -- ${JSON.stringify(summary.gates.V1.got)}`,
  `V2 (graded calm): **${V2}** -- ${JSON.stringify(summary.gates.V2.got)}`,
  `V3 (instability): **${V3}** -- ${JSON.stringify(summary.gates.V3.got)}`,
  "",
  "## Verdict",
  "",
  `**${branch}**`,
  "",
  branch === "A3L_ANNEALING_ERODES_SELF_INSURANCE"
    ? "Annealing erodes the self-insurance: as exploration decays, the notch learner's safety margin shrinks toward the edge and its end-state fragility exceeds the fixed-sigma baseline and rises through late training, while the graded learner stays at its interior optimum, calm. The v1 self-insurance was a property of EXPLORATION, not of the learner -- and real training schedules anneal exploration away. This is the deploy-relevant form of the bunching claim and clears the gate for the owner-run NS-3 PPO threshold-vs-graded binding."
    : branch === "A3L_ANNEAL_UNSTABLE"
      ? "Annealed training near the notch destabilized (collapse or extreme seed dispersion) -- itself the finding, per the registered V3 alternative: threshold oversight makes near-greedy training dynamics unstable at the boundary. Reported as-is; the erosion claim is not banked."
      : "The notch learner stayed backed off and calm even as exploration annealed -- self-insurance is robust in this harness, the erosion prediction is refuted, and the standalone surface does not proceed on this probe.",
  "",
].join("\n");
mkdirSync(path.dirname(path.resolve(args.out)), { recursive: true });
writeFileSync(path.resolve(args.out), md + "\n");

console.log(branch);
console.log(`  NOTCH: mu 2/3=${n.med_mu_twoThirds} end=${n.med_mu_end} | frag 2/3=${n.med_frag_twoThirds} end=${n.med_frag_end} | IQR=${n.iqr_mu_end}`);
console.log(`  SIGMOID: mu end=${g.med_mu_end} frag end=${g.med_frag_end}`);
console.log(`  V1=${V1} V2=${V2} V3=${V3}`);
console.log(`  wrote ${args.out} and ${args.json}`);
