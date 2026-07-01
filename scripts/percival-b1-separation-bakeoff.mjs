// Percival B1 -- separation bake-off (D1-D4), gated on B1_0_COURT_ADMITTED.
//
// Question: over a proxy-biased base gamma, does the un-targeting policy (c=0)
// strictly out-collect every q-quantilizer, and WHERE (base location x spread)?
//
// Reuses the admitted court reward R(c) from B1.0 (Morris-Shin honored prob at
// B=1,K=3,sigma=0.05 -> c* = B/(B+K) = 0.25). The quantilizer family takes upper
// c-tails of gamma (proxy Uhat(c)=c increasing). Because tilting only pushes c up
// and R is nonincreasing, the family's best true-reward member is q=1 (the raw
// base) -- so max_q Rbar(q) = E_gamma[R], and un-targeting must reshape BELOW the
// base to c=0. Clean separation requires the base SUPPORT above the cliff, not the
// mean.

import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

const args = {
  out: "docs/percival/PERCIVAL_B1_SEPARATION_RESULTS.md",
  json: "results/percival/b1-separation/summary.json",
};
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i += 1) {
  if (argv[i] === "--out") { args.out = argv[i + 1]; i += 1; }
  else if (argv[i] === "--json") { args.json = argv[i + 1]; i += 1; }
}

// ---- court reward R(c) (identical operating point to B1.0) ----
const N = 401, SIGMA = 0.05, B = 1, K = 3;
const C_STAR = B / (B + K); // 0.25
const round = (x, n = 6) => Number(Number(x).toFixed(n));

function erf(x) {
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const p = 0.3275911, a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429;
  const t = 1 / (1 + p * x);
  return s * (1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x));
}
const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
function probit(p) {
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const pl = 0.02425, ph = 1 - pl; let q, r;
  if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p <= ph) { q = p - 0.5; r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
  q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}
function gammaln(xx) {
  const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let x = xx, y = xx, tmp = x + 5.5; tmp -= (x + 0.5) * Math.log(tmp); let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) { y++; ser += cof[j] / y; } return -tmp + Math.log(2.5066282746310005 * ser / x);
}
function betacf(a, b, x) {
  const MAXIT = 300, EPS = 3e-12, FPMIN = 1e-300; let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN; d = 1 / d; let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    let m2 = 2 * m, aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; let del = d * c; h *= del; if (Math.abs(del - 1) < EPS) break;
  } return h;
}
function betai(a, b, x) {
  if (x <= 0) return 0; if (x >= 1) return 1;
  const bt = Math.exp(gammaln(a + b) - gammaln(a) - gammaln(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}
const binomTailGE = (n, p, m) => (m <= 0 ? 1 : m > n ? 0 : betai(m, n - m + 1, p));
const xStar = SIGMA * 0 + K / (B + K) + SIGMA * probit(K / (B + K));
function R(c) {
  const theta = 1 - c;
  const p = Phi((theta - xStar) / SIGMA);
  const m = Math.ceil(N * (1 - theta));
  return binomTailGE(N, p, m);
}

// ---- integration helpers ----
function trapzMean(a, b, n = 600) { // mean of R over [a,b]
  if (b <= a) return R(a);
  const h = (b - a) / n; let s = 0.5 * (R(a) + R(b));
  for (let i = 1; i < n; i++) s += R(a + i * h);
  return (s * h) / (b - a);
}

// ---- quantilizer over uniform base [cmin,cmax] ----
const QGRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
function qProfile(cmin, cmax) {
  // top-q by c = upper tail [cmax - q*(cmax-cmin), cmax]
  return QGRID.map((q) => {
    const lo = cmax - q * (cmax - cmin);
    return { q, Rbar: round(trapzMean(lo, cmax)) };
  });
}
function region(cmin, cmax) {
  if (cmax < C_STAR) return "support_below";
  if (cmin > C_STAR) return "support_above";
  return "straddle";
}
function evalBase(mu, s) {
  const cmin = round(Math.max(0, mu - s), 6), cmax = round(Math.min(1, mu + s), 6);
  const prof = qProfile(cmin, cmax);
  const maxRbarVal = Math.max(...prof.map((r) => r.Rbar));
  // R-bar is nondecreasing in q (lemma 1); on ties (flat regions) report the
  // largest q achieving the max, so argmax_q = 1 iff q=1 is a maximizer.
  const maxRow = [...prof].reverse().find((r) => r.Rbar >= maxRbarVal - 1e-9);
  const Egamma = trapzMean(cmin, cmax);
  const R0 = round(R(0));
  return {
    mu, s, c_min: cmin, c_max: cmax, c_gamma: mu, region: region(cmin, cmax),
    E_gamma_R: round(Egamma), max_Rbar: round(maxRbarVal), argmax_q: maxRow.q,
    R0, margin: round(R0 - maxRbarVal), profile: prof,
  };
}

// ============ D1 -- clean separation at a support-above point ============
const d1 = evalBase(0.40, 0.10); // [0.30,0.50], c_min=0.30 > c*=0.25
const EPS = 0.05;
const D1_PASS = d1.region === "support_above" && d1.max_Rbar < EPS && d1.R0 > 1 - EPS;

// ============ D2 -- boundary map (location x spread) ============
const MUS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60];
const SPREADS = [0.05, 0.10, 0.15];
const grid = [];
for (const s of SPREADS) for (const mu of MUS) grid.push(evalBase(mu, s));
const below = grid.filter((g) => g.region === "support_below");
const straddle = grid.filter((g) => g.region === "straddle");
const above = grid.filter((g) => g.region === "support_above");
const belowOK = below.every((g) => g.margin < 0.10);      // deflation cell
const aboveOK = above.every((g) => g.margin > 0.90);      // clean prize cell
const lemma1 = grid.every((g) => g.argmax_q === 1.0);     // best quantilizer is q=1 (weakly) everywhere
// partial-separation identity: max_q Rbar = E_gamma[R], so margin = R0 - E_gamma[R] in every region
const partialIdentity = grid.every((g) => Math.abs(g.margin - (g.R0 - g.E_gamma_R)) < 2e-3);
// straddle is the transition regime: its margins should span (below..above), not be pinned
const straddleMargins = straddle.map((g) => g.margin).sort((a, b) => a - b);
const straddleSpans = straddle.length > 0 && (straddleMargins[straddleMargins.length - 1] - straddleMargins[0]) > 0.2;
const D2_PASS = below.length > 0 && above.length > 0 && straddle.length > 0 && belowOK && aboveOK && lemma1 && partialIdentity && straddleSpans;

// ============ D3 -- competence frontier (margin vs support gap) ============
const frontier = MUS.map((mu) => { const g = evalBase(mu, 0.10); return { c_min: g.c_min, support_gap: round(g.c_min - C_STAR), margin: g.margin, R0: g.R0, max_Rbar: g.max_Rbar }; });
const D3_PASS = d1.R0 > 1 - EPS && d1.profile.every((r) => d1.R0 > r.Rbar);

// ============ D4 -- deflation + straddle cells (reported) ============
const d4 = {
  deflation_below: below.map((g) => ({ mu: g.mu, s: g.s, margin: g.margin })),
  straddle_partial: straddle.map((g) => ({ mu: g.mu, s: g.s, E_gamma_R: g.E_gamma_R, margin: g.margin })),
};

const gates = {
  D1: { token: D1_PASS ? "B1_CLEAN_SEPARATION_CONFIRMED" : "B1_CLEAN_SEPARATION_FAIL", pass: D1_PASS, point: { c_min: d1.c_min, c_max: d1.c_max, max_Rbar: d1.max_Rbar, R0: d1.R0, margin: d1.margin } },
  D2: { token: D2_PASS ? "B1_BOUNDARY_MAPPED" : "B1_BOUNDARY_FAIL", pass: D2_PASS, below_ok: belowOK, above_ok: aboveOK, lemma1_q1_best: lemma1, partial_identity: partialIdentity, straddle_spans: straddleSpans },
  D3: { token: D3_PASS ? "B1_COMPETENT" : "B1_NOT_COMPETENT", pass: D3_PASS, R0: d1.R0 },
};
const allPass = D1_PASS && D2_PASS && D3_PASS;
const branch = allPass ? "B1_SEPARATION_MAPPED_CONDITIONAL" : "B1_SEPARATION_INCOMPLETE";

const summary = {
  generated_at: new Date().toISOString(), branch,
  model: { court_reward: "B1.0 Morris-Shin honored prob", B, K, sigma: SIGMA, c_star: C_STAR, base: "uniform[mu-s, mu+s]", proxy: "Uhat(c)=c", q_grid: QGRID },
  R0: round(R(0)), gates, d1, grid, frontier, d4,
  honest_note: "Conditional on support-above (c_min>c*). Straddle bases give only partial separation (margin=R0-E_gamma[R]). Whether a real base is supported past c* is B2 (private), unresolved here. Admission+separation != prize won.",
};
mkdirSync(path.dirname(path.resolve(args.json)), { recursive: true });
writeFileSync(path.resolve(args.json), JSON.stringify(summary, null, 2) + "\n");

// ---- receipt ----
const mapRows = grid.map((g) => `| ${g.mu} | ${g.s} | ${g.c_min} | ${g.c_max} | ${g.region} | ${g.E_gamma_R} | ${g.max_Rbar} | ${g.argmax_q} | ${g.margin} |`);
const frontRows = frontier.map((f) => `| ${f.c_min} | ${f.support_gap} | ${f.max_Rbar} | ${f.R0} | ${f.margin} |`);
const md = [
  "# Percival B1 -- Separation Bake-Off (results)",
  "",
  `Generated ${summary.generated_at} by \`scripts/percival-b1-separation-bakeoff.mjs\`.`,
  "",
  `Court reward R(c) from B1.0 (B=${B}, K=${K}, sigma=${SIGMA}); cliff c* = ${C_STAR}. Base = uniform[mu-s, mu+s]; proxy Uhat(c)=c; un-targeting collects R(0) = **${round(R(0))}**.`,
  "",
  "## D1 -- Clean separation at a support-above point",
  "",
  `Base [${d1.c_min}, ${d1.c_max}] (c_min = ${d1.c_min} > c* = ${C_STAR}). max_q Rbar(q) = **${d1.max_Rbar}** (< ${EPS}); R(0) = **${d1.R0}**; margin = **${d1.margin}**. -> **${gates.D1.token}**`,
  "",
  "## D2 -- Boundary map (base location x spread)",
  "",
  "| mu | s | c_min | c_max | region | E_gamma[R] | max_q Rbar | argmax q | margin |",
  "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
  ...mapRows,
  "",
  `support-below margins all < 0.10 (deflation): **${belowOK}**; support-above margins all > 0.90 (clean): **${aboveOK}**; best quantilizer is q=1 everywhere (lemma 1): **${lemma1}**; margin = R(0) - E_gamma[R] everywhere (partial-separation identity): **${partialIdentity}**; straddle margins span the transition: **${straddleSpans}**. -> **${gates.D2.token}**`,
  "",
  "The (a)<->(c) boundary is a location x spread surface, not the line c_gamma = c*. Mean-past-cliff with support straddling the cliff is only partial.",
  "",
  "## D3 -- Competence frontier (spread s=0.10)",
  "",
  "| c_min | support gap (c_min - c*) | max_q Rbar | R(0) | margin |",
  "| ---: | ---: | ---: | ---: | ---: |",
  ...frontRows,
  "",
  `Un-targeting collects R(0) = ${d1.R0} (near ceiling) and strictly exceeds every Rbar(q) at the clean point: **${D3_PASS}**. The margin grows with the support gap -- the court's price of grasping. -> **${gates.D3.token}**`,
  "",
  "## D4 -- Deflation and straddle cells (reported)",
  "",
  `Support-below (base entirely honored, no edge): ${d4.deflation_below.length} grid points, margins ~0 -- the honest (a) cell. Straddle (partial): ${d4.straddle_partial.length} points, margin = R(0) - E_gamma[R] in (0,1) -- NOT the clean prize.`,
  "",
  "## Verdict",
  "",
  `**${branch}**`,
  "",
  allPass
    ? "Un-targeting strictly out-collects the entire quantilizer family in the support-above region; the family's best member is q=1 (quantilizing strictly hurts on this court reward); straddle bases give only a partial margin; support-below deflates. The separation is real and its boundary is mapped -- **conditional on the base being supported past the cliff**. This is not the prize won: whether a real base is supported above c* is B2 (private), unresolved. The Lean anchor (B4) is the remaining keeper."
    : "At least one deliverable did not pass; see the failing gate. The separation claim is not banked.",
  "",
  "## Honest boundary",
  "",
  summary.honest_note,
  "",
].join("\n");
mkdirSync(path.dirname(path.resolve(args.out)), { recursive: true });
writeFileSync(path.resolve(args.out), md + "\n");

console.log(branch);
for (const k of ["D1", "D2", "D3"]) console.log(`  ${gates[k].token} pass=${gates[k].pass}`);
console.log(`  R(0)=${round(R(0))}  D1 margin=${d1.margin} (base [${d1.c_min},${d1.c_max}])`);
console.log(`  regions: below=${below.length} straddle=${straddle.length} above=${above.length}  lemma1(q=1 best)=${lemma1} partialId=${partialIdentity} straddleSpans=${straddleSpans}`);
console.log(`  wrote ${args.out} and ${args.json}`);
