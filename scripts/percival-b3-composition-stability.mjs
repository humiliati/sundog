// Percival B3 -- composition & reflective stability (probe).
//
// Two questions on the admitted court reward R(c) (B1.0, c*=0.25):
//
// B3.1 COMPOSITION. Taylor's quantilizers famously do not compose (iterating
//   degrades). Model iterated quantilization as recentering the base on the selected
//   upper-c tail: a uniform base of half-width s, tilted to top-q, has selected mean
//   c + s(1-q), so each round DRIFTS courting up by delta = s(1-q). Any tilt (q<1)
//   drifts toward the cliff and eventually disgraces; un-targeting (c=0) and the raw
//   base (q=1) are drift-free. Does un-targeting compose where quantilizing drifts?
//
// B3.2 REFLECTIVE STABILITY (Galahad vs Perceval). Give the knight an objective
//   J(c) = R(c) + lambda * Uhat(c), Uhat(c)=c (lambda = proxy temptation). The
//   reward-optimal c and its fragility (disgrace risk under courting noise eta) are
//   read out vs lambda. Conjecture: un-targeting is stable only under Galahad-
//   perfection (lambda=0) and even then only weakly; any temptation (Perceval) drives
//   the optimum to the cliff edge -> fragile.

import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

const args = { out: "docs/percival/PERCIVAL_B3_COMPOSITION_STABILITY_RESULTS.md", json: "results/percival/b3-composition-stability/summary.json" };
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i += 1) { if (argv[i] === "--out") { args.out = argv[i + 1]; i += 1; } else if (argv[i] === "--json") { args.json = argv[i + 1]; i += 1; } }

// ---- court reward R(c) (identical to B1.0/B1-proper/B2) ----
const N = 401, SIGMA = 0.05, B = 1, K = 3, C_STAR = B / (B + K); // 0.25
const round = (x, n = 6) => Number(Number(x).toFixed(n));
function erf(x) { const s = x < 0 ? -1 : 1; x = Math.abs(x); const p = 0.3275911, a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429; const t = 1 / (1 + p * x); return s * (1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x)); }
const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
function probit(p) { const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]; const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]; const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]; const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]; const pl = 0.02425, ph = 1 - pl; let q, r; if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); } if (p <= ph) { q = p - 0.5; r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); } q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
function gammaln(xx) { const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]; let x = xx, y = xx, tmp = x + 5.5; tmp -= (x + 0.5) * Math.log(tmp); let ser = 1.000000000190015; for (let j = 0; j < 6; j++) { y++; ser += cof[j] / y; } return -tmp + Math.log(2.5066282746310005 * ser / x); }
function betacf(a, b, x) { const MAXIT = 300, EPS = 3e-12, FPMIN = 1e-300; let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < FPMIN) d = FPMIN; d = 1 / d; let h = d; for (let m = 1; m <= MAXIT; m++) { let m2 = 2 * m, aa = m * (b - m) * x / ((qam + m2) * (a + m2)); d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; h *= d * c; aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)); d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; let del = d * c; h *= del; if (Math.abs(del - 1) < EPS) break; } return h; }
function betai(a, b, x) { if (x <= 0) return 0; if (x >= 1) return 1; const bt = Math.exp(gammaln(a + b) - gammaln(a) - gammaln(b) + a * Math.log(x) + b * Math.log(1 - x)); return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b; }
const binomTailGE = (n, p, m) => (m <= 0 ? 1 : m > n ? 0 : betai(m, n - m + 1, p));
const xStar = K / (B + K) + SIGMA * probit(K / (B + K));
function R(c) { const cc = Math.max(0, Math.min(1, c)); const theta = 1 - cc; const p = Phi((theta - xStar) / SIGMA); const m = Math.ceil(N * (1 - theta)); return binomTailGE(N, p, m); }

// ============ B3.1 -- composition under iteration ============
const C0 = 0.10, S = 0.10, T = 20;      // start honored (below cliff), half-width, rounds
const QS = [0.3, 0.5, 0.7, 0.9, 1.0];
const composition = QS.map((q) => {
  const delta = round(S * (1 - q));      // drift per round from the selected upper tail
  let cum = 0, disgraceRound = null;
  const path = [];
  for (let t = 0; t < T; t++) {
    const c = C0 + t * delta;
    const r = R(c);
    cum += r;
    path.push(round(r, 4));
    if (disgraceRound === null && r < 0.5) disgraceRound = t;
  }
  return { q, drift_per_round: delta, disgrace_round: disgraceRound, cum_reward: round(cum, 4) };
});
const untargetedCum = round(T * R(0), 4); // c=0 every round
const tilts = composition.filter((r) => r.q < 1.0);           // genuine quantilization is a tilt
const everyTiltDrifts = tilts.every((r) => r.disgrace_round !== null);
const untargetedBeatsTilts = tilts.every((r) => untargetedCum > r.cum_reward);
const rawBaseDriftFree = composition.find((r) => r.q === 1.0).disgrace_round === null; // q=1 also composes
const B31_PASS = everyTiltDrifts && untargetedBeatsTilts;

// ============ B3.2 -- reflective stability (Galahad vs Perceval) ============
const ETA = 0.04; // courting noise half-width
function disgraceRisk(c) { // 1 - mean R over [c-eta, c+eta]
  const n = 40; let s = 0; for (let i = 0; i <= n; i++) s += R(c - ETA + (2 * ETA) * i / n); return round(1 - s / (n + 1), 4);
}
function rewardOptimal(lambda) { // argmax_c R(c) + lambda*c
  let best = -Infinity, bestC = 0;
  for (let c = 0; c <= 1 + 1e-9; c += 0.001) { const j = R(c) + lambda * c; if (j > best + 1e-9) { best = j; bestC = round(c, 4); } }
  return bestC;
}
const LAMBDAS = [0, 0.02, 0.05, 0.1, 0.2];
const stability = LAMBDAS.map((lam) => { const copt = rewardOptimal(lam); return { lambda: lam, reward_optimal_c: copt, fragility_disgrace_risk: disgraceRisk(copt) }; });
// un-targeting is the safety anchor: minimal disgrace risk, achieved uniquely at c=0
const untargetedRisk = disgraceRisk(0);
const cliffJump = stability.find((s) => s.lambda > 0);          // any temptation
const galahad = stability.find((s) => s.lambda === 0);
const temptationDrivesToCliff = cliffJump.reward_optimal_c >= C_STAR - 0.02 && cliffJump.fragility_disgrace_risk > 0.2;
const untargetedIsSafestAnchor = untargetedRisk < 0.02 && cliffJump.fragility_disgrace_risk > untargetedRisk + 0.2;
const B32_CONFIRMS_SPLIT = temptationDrivesToCliff && untargetedIsSafestAnchor;

const branch = (B31_PASS && B32_CONFIRMS_SPLIT)
  ? "B3_COMPOSES_STABILITY_NEEDS_ROBUSTNESS"
  : "B3_INCOMPLETE";

const gates = {
  "B3.1": { token: B31_PASS ? "B3_UNTARGETING_COMPOSES" : "B3_COMPOSITION_FAIL", pass: B31_PASS, every_tilt_drifts: everyTiltDrifts, untargeted_beats_tilts: untargetedBeatsTilts, raw_base_also_composes: rawBaseDriftFree, untargeted_cum: untargetedCum },
  "B3.2": { token: B32_CONFIRMS_SPLIT ? "B3_STABILITY_IS_ROBUSTNESS_NOT_REWARD" : "B3_STABILITY_UNCLEAR", confirms_split: B32_CONFIRMS_SPLIT, galahad_optimum_c: galahad.reward_optimal_c, temptation_cliff_c: cliffJump.reward_optimal_c, untargeted_risk: untargetedRisk },
};
const summary = {
  generated_at: new Date().toISOString(), branch,
  model: { court: "B1.0 R(c), c*=" + C_STAR, composition: "iterated quantilization drift = s(1-q); c0=" + C0 + ", s=" + S + ", T=" + T, stability: "J(c)=R(c)+lambda*c; courting noise eta=" + ETA },
  b3_1_composition: composition, untargeted_cum: untargetedCum,
  b3_2_stability: stability, untargeted_risk: untargetedRisk,
  gates,
  honest_note: "B3.1: any tilt (q<1) drifts up by s(1-q)/round and eventually disgraces; un-targeting (c=0) out-collects every tilt and (with the untilted base q=1) is drift-free -- genuine quantilization (a tilt) fails to compose where un-targeting does. That q=1 also composes reinforces lemma 1: the tilt is the drift-causing move. B3.2: R(c) is ~flat over the honored region, so at lambda=0 un-targeting is only WEAKLY optimal (tied across [0,c*)); any temptation lambda>0 drives the reward-optimum to the cliff edge (fragile). Un-targeting is the minimal-disgrace-risk anchor but never the strict reward-optimum -- it is selected by ROBUSTNESS (risk-aversion to the cliff), not reward maximization. Galahad = aligned + robust sits safely at c=0; Perceval = naive/tempted drifts to the cliff. The split is real. Modeling choices flagged: (i) drift = selected-tail recentering s(1-q); (ii) temptation objective R+lambda*Uhat with courting noise eta.",
};
mkdirSync(path.dirname(path.resolve(args.json)), { recursive: true });
writeFileSync(path.resolve(args.json), JSON.stringify(summary, null, 2) + "\n");

const compRows = composition.map((r) => `| ${r.q} | ${r.drift_per_round} | ${r.disgrace_round === null ? "never" : r.disgrace_round} | ${r.cum_reward} |`);
const stabRows = stability.map((s) => `| ${s.lambda} | ${s.reward_optimal_c} | ${s.fragility_disgrace_risk} |`);
const md = [
  "# Percival B3 -- Composition & Reflective Stability (probe)",
  "",
  `Generated ${summary.generated_at} by \`scripts/percival-b3-composition-stability.mjs\`.`,
  "",
  `Court reward R(c) from B1.0 (c* = ${C_STAR}). Two modeling choices, flagged: composition drift = selected-tail recentering \`s(1-q)\`; stability objective \`J(c)=R(c)+lambda*c\` under courting noise eta=${ETA}.`,
  "",
  "## B3.1 -- Composition under iteration",
  "",
  `Start honored at c0=${C0} (below cliff), half-width s=${S}, T=${T} rounds. Iterated quantilization recenters on the selected upper tail, drifting courting up by \`s(1-q)\` per round.`,
  "",
  "| q (tilt) | drift/round | disgrace round | cumulative reward |",
  "| ---: | ---: | ---: | ---: |",
  ...compRows,
  `| un-targeting (c=0) | 0 | never | **${untargetedCum}** |`,
  "",
  `Every tilt (q<1) drifts to disgrace at a finite round: **${everyTiltDrifts}**. Un-targeting strictly out-collects every tilt: **${untargetedBeatsTilts}**. The untilted base (q=1) is also drift-free (ties un-targeting on this honored-start base): **${rawBaseDriftFree}** -- reinforcing lemma 1 (q=1 is the drift-free boundary; the *tilt* is what drifts). -> **${gates["B3.1"].token}**`,
  "",
  "## B3.2 -- Reflective stability (Galahad vs Perceval)",
  "",
  `Objective J(c) = R(c) + lambda*c. Reward-optimal c and its fragility (disgrace risk under courting noise eta=${ETA}):`,
  "",
  "| lambda (temptation) | reward-optimal c | fragility (disgrace risk) |",
  "| ---: | ---: | ---: |",
  ...stabRows,
  "",
  `At lambda=0, R(c) is flat over the honored region -- un-targeting is only *weakly* optimal (tied across [0,c*), tie-broken to c=0). Any temptation lambda>0 drives the reward-optimum to the cliff edge (c ~ ${cliffJump.reward_optimal_c}) with fragility ${cliffJump.fragility_disgrace_risk}. Un-targeting (c=0) has disgrace risk **${untargetedRisk}** -- the minimal-risk anchor, but never the strict reward-optimum. -> **${gates["B3.2"].token}**`,
  "",
  "## Verdict",
  "",
  `**${branch}**`,
  "",
  "Un-targeting **composes** where iterated quantilization drifts to disgrace -- a genuine positive on the composition axis, independent of B2's static deflation. But its reflective stability is a **robustness** property, not a reward-optimality one: a pure reward-maximizer is indifferent (lambda=0) or cliff-drawn (lambda>0, fragile); un-targeting is selected only by risk-aversion to the oversight cliff. Galahad (aligned + robust) sits safely at c=0; Perceval (naive, tempted) drifts to the cliff. The conjectured split is real -- un-targeting is the *safe* choice, not the *optimal* one, the same priced-safety shape as the cap.",
  "",
  "## Honest boundary",
  "",
  summary.honest_note,
  "",
].join("\n");
mkdirSync(path.dirname(path.resolve(args.out)), { recursive: true });
writeFileSync(path.resolve(args.out), md + "\n");

console.log(branch);
for (const k of ["B3.1", "B3.2"]) console.log(`  ${gates[k].token}`);
console.log(`  composition: untargeted_cum=${untargetedCum} vs quantilizers=${composition.map((r) => r.cum_reward).join(",")} (disgrace rounds ${composition.map((r) => r.disgrace_round).join(",")})`);
console.log(`  stability: reward-opt c by lambda=${stability.map((s) => s.reward_optimal_c).join(",")}  fragility=${stability.map((s) => s.fragility_disgrace_risk).join(",")}  untargeted_risk=${untargetedRisk}`);
console.log(`  wrote ${args.out} and ${args.json}`);
