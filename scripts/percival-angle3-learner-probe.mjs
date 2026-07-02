// Percival Angle-3 Tier-2 -- learner probe. Spec (pre-registered BEFORE this run):
// docs/percival/PERCIVAL_ANGLE3_LEARNER_PROBE_SPEC.md
//
// Question: do gradient learners bunch just under a review notch, and does grading
// the review make them retreat -- and how far? Four arms: NOTCH (court R(c)),
// SIGMOID (w=0.05, predicted interior backoff ~0.107, NOT zero), FINE (linear rate 2,
// predicted abstention at 0), FREE (no tax, positive control, predicted climb to 1).
// REINFORCE on Gaussian mean, batch baseline, 12 seeds/arm, deterministic PRNG.

import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

const args = { out: "docs/percival/PERCIVAL_ANGLE3_LEARNER_RESULTS.md", json: "results/percival/angle3-learner/summary.json" };
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i += 1) { if (argv[i] === "--out") { args.out = argv[i + 1]; i += 1; } else if (argv[i] === "--json") { args.json = argv[i + 1]; i += 1; } }

function mulberry32(seed) { let a = seed >>> 0; return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function gauss(rng) { let u = 0, v = 0; while (u === 0) u = rng(); while (v === 0) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const round = (x, n = 6) => Number(Number(x).toFixed(n));

// ---- court R(c) (B1.0 lineage) ----
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

// ---- arms ----
const f = (c) => 1 + (c - 0.2);
const ARMS = {
  NOTCH: (c) => Rcourt(c),
  SIGMOID: (c) => 1 / (1 + Math.exp((c - C_STAR) / 0.05)),
  FINE: (c) => Math.max(0, 1 - 2 * c),
  FREE: () => 1,
};
const total = (arm, c) => f(c) * ARMS[arm](c);
// analytic optima on a fine grid (computed, not hardcoded)
function argmaxArm(arm) { let best = -1, bc = 0; for (let c = 0; c <= 1 + 1e-9; c += 0.0005) { const v = total(arm, c); if (v > best + 1e-12) { best = v; bc = c; } } return round(bc, 4); }
const OPT = Object.fromEntries(Object.keys(ARMS).map((a) => [a, argmaxArm(a)]));

// ---- REINFORCE ----
const SEEDS = 12, UPDATES = 400, BATCH = 64, LR = 0.02, ETA = 0.04;
function train(arm, seed, sigma) {
  const rng = mulberry32(seed * 7919 + arm.length * 104729 + Math.round(sigma * 1e4));
  let mu = 0.05; // start low/honest
  const tail = [];
  for (let u = 0; u < UPDATES; u++) {
    const cs = [], rs = [];
    for (let b = 0; b < BATCH; b++) { const c = Math.max(0, Math.min(1, mu + sigma * gauss(rng))); cs.push(c); rs.push(total(arm, c)); }
    const rbar = rs.reduce((s, x) => s + x, 0) / BATCH;
    let grad = 0;
    for (let b = 0; b < BATCH; b++) grad += (rs[b] - rbar) * (cs[b] - mu) / (sigma * sigma);
    mu += LR * grad / BATCH;
    if (u >= UPDATES - 50) tail.push(mu);
  }
  const muBar = tail.reduce((s, x) => s + x, 0) / tail.length;
  // converged-policy metrics under N(muBar, sigma), clipped
  const rng2 = mulberry32(seed + 1);
  let Ec = 0, bunch = 0, low = 0, frag = 0; const M = 4000;
  for (let i = 0; i < M; i++) {
    const c = Math.max(0, Math.min(1, muBar + sigma * gauss(rng2)));
    Ec += c;
    if (c >= C_STAR - 0.08 && c < C_STAR) bunch++;
    if (c < 0.05) low++;
    if (c + ETA * Math.abs(gauss(rng2)) > C_STAR) frag++;
  }
  return { mu: round(muBar, 4), Ec: round(Ec / M, 4), bunch_mass: round(bunch / M, 4), low_mass: round(low / M, 4), fragility: round(frag / M, 4) };
}
const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };

const SIGMA0 = 0.04;
const perArm = {};
for (const arm of Object.keys(ARMS)) {
  const runs = []; for (let s = 1; s <= SEEDS; s++) runs.push(train(arm, s, SIGMA0));
  perArm[arm] = { runs, med_mu: round(median(runs.map((r) => r.mu)), 4), med_Ec: round(median(runs.map((r) => r.Ec)), 4), med_bunch: round(median(runs.map((r) => r.bunch_mass)), 4), med_low: round(median(runs.map((r) => r.low_mass)), 4), med_frag: round(median(runs.map((r) => r.fragility)), 4) };
}
// rider: notch backoff vs exploration sigma
const rider = [0.02, 0.04, 0.08].map((sg) => {
  const runs = []; for (let s = 1; s <= SEEDS; s++) runs.push(train("NOTCH", s, sg));
  return { sigma: sg, med_mu: round(median(runs.map((r) => r.mu)), 4), backoff: round(C_STAR - median(runs.map((r) => r.mu)), 4) };
});

// ---- gates (vs computed optima) ----
const n = perArm.NOTCH, sgm = perArm.SIGMOID, fin = perArm.FINE, fre = perArm.FREE;
const G1 = n.med_mu >= C_STAR - 3 * SIGMA0 && n.med_mu < C_STAR && n.med_low < 0.05
  && n.med_frag > sgm.med_frag && n.med_frag > fin.med_frag && n.med_frag > 0.2;
const G2 = Math.abs(sgm.med_mu - OPT.SIGMOID) <= 0.06 && sgm.med_mu > 0.03 && sgm.med_mu < C_STAR - 0.08;
const G3 = fin.med_Ec < 0.06 && fin.med_frag < 0.05;
const G4 = fre.med_mu > 0.8;
const riderMonotone = rider.every((r, i) => i === 0 || r.backoff >= rider[i - 1].backoff - 0.005);
const allPass = G1 && G2 && G3 && G4;
const branch = allPass ? "A3L_LEARNERS_TRACK_OVERSIGHT_SHAPE" : "A3L_OPTIMA_LEARNER_GAP";

const gates = {
  G1: { pass: G1, claim: "notch bunching with exploration backoff, highest fragility", med_mu: n.med_mu, med_bunch: n.med_bunch, med_frag: n.med_frag },
  G2: { pass: G2, claim: "sigmoid = backed-off interior, NOT abstention", med_mu: sgm.med_mu, analytic_opt: OPT.SIGMOID },
  G3: { pass: G3, claim: "linear fine = abstention", med_Ec: fin.med_Ec, med_frag: fin.med_frag },
  G4: { pass: G4, claim: "positive control climbs", med_mu: fre.med_mu, analytic_opt: OPT.FREE },
  rider: { backoff_monotone_in_sigma: riderMonotone, sweep: rider },
};
const summary = {
  generated_at: new Date().toISOString(), branch,
  spec: "docs/percival/PERCIVAL_ANGLE3_LEARNER_PROBE_SPEC.md (pre-registered before run)",
  model: { f: "1+(c-0.2)", arms: { NOTCH: "court R(c)", SIGMOID: "logistic w=0.05", FINE: "max(0,1-2c)", FREE: "1" }, learner: `REINFORCE Gaussian mu, sigma=${SIGMA0}, batch ${BATCH}, ${UPDATES} updates, lr ${LR}, ${SEEDS} seeds/arm`, analytic_optima: OPT },
  per_arm: Object.fromEntries(Object.entries(perArm).map(([k, v]) => [k, { ...v, runs: undefined }])),
  rider,
  honest_note: "Pre-registered analytic refinement (in the spec, before the run): over a full action line a sigmoid tax can never zero the optimum for a pointwise-good proxy (bottom marginal max ~1.11 < gain 1.25); smoothing only backs the bunching point off. True abstention needs a constant-marginal fine (rate > gain). The probe gates learners against the COMPUTED optima. Notch gate allows the exploration backoff explicitly (stochastic learners pay for mass past the notch); literal edge-sitting was not the prediction.",
};
mkdirSync(path.dirname(path.resolve(args.json)), { recursive: true });
writeFileSync(path.resolve(args.json), JSON.stringify(summary, null, 2) + "\n");

const armRows = Object.entries(perArm).map(([k, v]) => `| ${k} | ${OPT[k]} | ${v.med_mu} | ${v.med_Ec} | ${v.med_bunch} | ${v.med_low} | ${v.med_frag} |`);
const riderRows = rider.map((r) => `| ${r.sigma} | ${r.med_mu} | ${r.backoff} |`);
const md = [
  "# Percival Angle-3 Learner Probe (results)",
  "",
  `Generated ${summary.generated_at} by \`scripts/percival-angle3-learner-probe.mjs\`. Spec pre-registered before the run: [\`PERCIVAL_ANGLE3_LEARNER_PROBE_SPEC.md\`](PERCIVAL_ANGLE3_LEARNER_PROBE_SPEC.md).`,
  "",
  `REINFORCE (Gaussian mean, sigma=${SIGMA0}, ${SEEDS} seeds/arm) on Total(c) = f(c)*g(c), f = 1+(c-0.2), notch at c* = ${C_STAR}.`,
  "",
  "## Per-arm convergence vs computed optima",
  "",
  "| arm | analytic optimum | median mu | median E[c] | bunch mass [c*-0.08, c*) | mass c<0.05 | fragility |",
  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
  ...armRows,
  "",
  `Gate results: G1 notch-bunching **${G1}**; G2 sigmoid backed-off interior **${G2}**; G3 fine abstention **${G3}**; G4 control climbs **${G4}**.`,
  "",
  "## Rider — notch backoff vs exploration scale",
  "",
  "| exploration sigma | median mu | backoff (c* - mu) |",
  "| ---: | ---: | ---: |",
  ...riderRows,
  "",
  `Backoff monotone in sigma: **${riderMonotone}** — the learner analog of the analytic near-threshold backoff.`,
  "",
  "## Verdict",
  "",
  `**${branch}**`,
  "",
  allPass
    ? "Learners track the oversight shape: they **bunch just under the notch** (backed off by their exploration scale, with the highest fragility of any arm), a smoothed sigmoid review only **backs the bunching off** (interior convergence, not abstention — as the pre-registered analytic refinement predicted), a **constant-marginal fine produces abstention**, and the free control climbs (so retreat is not learner weakness). The optima story transfers to learners. This clears the gate for the escalation: the owner-run NS-3 PPO threshold-vs-graded binding, the receipt that would earn the standalone Angle-3 surface."
    : "At least one gate missed: the optima story does not transfer cleanly to learners. The gap is itself the finding; the quantilizing page's oversight-shape section gains a caveat and the standalone surface does not proceed.",
  "",
  "## Honest boundary",
  "",
  summary.honest_note,
  "",
].join("\n");
mkdirSync(path.dirname(path.resolve(args.out)), { recursive: true });
writeFileSync(path.resolve(args.out), md + "\n");

console.log(branch);
for (const g of ["G1", "G2", "G3", "G4"]) console.log(`  ${g} pass=${gates[g].pass} -- ${gates[g].claim}`);
console.log(`  optima: ${JSON.stringify(OPT)}`);
console.log(`  medians: NOTCH mu=${n.med_mu} frag=${n.med_frag} | SIGMOID mu=${sgm.med_mu} (opt ${OPT.SIGMOID}) | FINE Ec=${fin.med_Ec} | FREE mu=${fre.med_mu}`);
console.log(`  rider backoff: ${rider.map((r) => `${r.sigma}->${r.backoff}`).join("  ")} monotone=${riderMonotone}`);
console.log(`  wrote ${args.out} and ${args.json}`);
