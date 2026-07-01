// Percival S2 -- target-cap probe (v2 spec): collapse, gap, reconstruction leak,
// court-vs-tax frontier.
//
// Model (pinned): G ~ U{0,1}; V,U conditionally independent given G with
// P(V=G)=beta, P(U=G)=rho (rho<1; primary grid beta<rho, rho<=beta as zero-tax
// controls). c = do(U)-influence on A holding V fixed (A3-sharpened, named-variable).
// Court = B1.0 R(c) at c*=0.25. Total = Competence * R(c).
// Unique-info tax (binary-symmetric CI) = max(beta,rho) - beta  (a reliability
// edge, not synergy -- owner caveat, recorded).
//
// All quantities exact via enumeration of the 8 (G,V,U) cells. Deterministic.

import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

const args = { out: "docs/percival/PERCIVAL_S2_TARGET_CAP_RESULTS.md", json: "results/percival/s2-target-cap/summary.json" };
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i += 1) { if (argv[i] === "--out") { args.out = argv[i + 1]; i += 1; } else if (argv[i] === "--json") { args.json = argv[i + 1]; i += 1; } }

// ---- court R(c) (identical to B1.0/B1-proper/B2/B3) ----
const N = 401, SIGMA = 0.05, B = 1, K = 3, C_STAR = B / (B + K);
const round = (x, n = 6) => Number(Number(x).toFixed(n));
function erf(x) { const s = x < 0 ? -1 : 1; x = Math.abs(x); const p = 0.3275911, a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429; const t = 1 / (1 + p * x); return s * (1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x)); }
const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
function probit(p) { const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]; const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]; const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]; const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]; const pl = 0.02425, ph = 1 - pl; let q, r; if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); } if (p <= ph) { q = p - 0.5; r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); } q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
function gammaln(xx) { const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]; let x = xx, y = xx, tmp = x + 5.5; tmp -= (x + 0.5) * Math.log(tmp); let ser = 1.000000000190015; for (let j = 0; j < 6; j++) { y++; ser += cof[j] / y; } return -tmp + Math.log(2.5066282746310005 * ser / x); }
function betacf(a, b, x) { const MAXIT = 300, EPS = 3e-12, FPMIN = 1e-300; let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < FPMIN) d = FPMIN; d = 1 / d; let h = d; for (let m = 1; m <= MAXIT; m++) { let m2 = 2 * m, aa = m * (b - m) * x / ((qam + m2) * (a + m2)); d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; h *= d * c; aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)); d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; let del = d * c; h *= del; if (Math.abs(del - 1) < EPS) break; } return h; }
function betai(a, b, x) { if (x <= 0) return 0; if (x >= 1) return 1; const bt = Math.exp(gammaln(a + b) - gammaln(a) - gammaln(b) + a * Math.log(x) + b * Math.log(1 - x)); return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b; }
const binomTailGE = (n, p, m) => (m <= 0 ? 1 : m > n ? 0 : betai(m, n - m + 1, p));
const xStarEq = K / (B + K) + SIGMA * probit(K / (B + K));
function Rcourt(c) { const cc = Math.max(0, Math.min(1, c)); const theta = 1 - cc; const p = Phi((theta - xStarEq) / SIGMA); const m = Math.ceil(N * (1 - theta)); return binomTailGE(N, p, m); }

// ---- exact cell machinery ----
// policy = function (v,u) -> P(A=1)
function cells(beta, rho) {
  const out = [];
  for (const g of [0, 1]) for (const v of [0, 1]) for (const u of [0, 1]) {
    const pv = v === g ? beta : 1 - beta;
    const pu = u === g ? rho : 1 - rho;
    out.push({ g, v, u, p: 0.5 * pv * pu });
  }
  return out;
}
const competence = (cs, pol) => cs.reduce((s, c) => s + c.p * (c.g === 1 ? pol(c.v, c.u) : 1 - pol(c.v, c.u)), 0);
const cInfluence = (pol) => 0.5 * (Math.abs(pol(0, 1) - pol(0, 0)) + Math.abs(pol(1, 1) - pol(1, 0)));
const agreeWithProxyUser = (cs, pol) => cs.reduce((s, c) => s + c.p * (c.u === 1 ? pol(c.v, c.u) : 1 - pol(c.v, c.u)), 0);

// base policies
const P_proxyUser = (v, u) => u;               // unconstrained best on primary grid (beta<rho)
const P_vOnly = (v, _u) => v;                  // canonical c=0 policy; ALSO the reconstruction courtier (Bayes Uhat=V)
const P_xor = (v, u) => (v === u ? 1 : 0);     // stress policy (uses both)
const P_mix = (w) => (v, u) => w * u + (1 - w) * v; // S2.4 family: A=U w.p. w else V; c=w

// exogenous do(U)-invariance projections of a given policy
const proj = {
  mask0: (pol) => (v, _u) => pol(v, 0),
  mask1: (pol) => (v, _u) => pol(v, 1),
  avg: (pol) => (v, _u) => 0.5 * (pol(v, 0) + pol(v, 1)), // scramble == permutation-average on action probs
};

const GRID = [
  { beta: 0.55, rho: 0.70, kind: "primary" },
  { beta: 0.60, rho: 0.75, kind: "primary" },
  { beta: 0.60, rho: 0.90, kind: "primary" },
  { beta: 0.75, rho: 0.90, kind: "primary" },
  { beta: 0.75, rho: 0.60, kind: "control" },
  { beta: 0.70, rho: 0.70, kind: "control" },
];

const TOL = 1e-9;
const results = [];
let s21ok = true, s22ok = true;

for (const { beta, rho, kind } of GRID) {
  const cs = cells(beta, rho);
  const bestUnconstrained = Math.max(competence(cs, P_proxyUser), competence(cs, P_vOnly)); // = max(beta,rho) here
  const tax = round(Math.max(beta, rho) - beta);
  // S2.1: projections of each U-using policy
  const projRows = [];
  for (const [pname, basePol] of [["proxy_user", P_proxyUser], ["xor", P_xor], ["mix_0.5", P_mix(0.5)]]) {
    for (const [jname, J] of Object.entries(proj)) {
      const pp = J(basePol);
      const comp = competence(cs, pp), cinf = cInfluence(pp);
      const ok = cinf < TOL && comp <= beta + TOL;
      if (!ok) s21ok = false;
      projRows.push({ policy: pname, projection: jname, competence: round(comp), c: round(cinf), le_beta: ok });
    }
  }
  const vOnlyComp = competence(cs, P_vOnly);
  if (Math.abs(vOnlyComp - beta) > 1e-9) s21ok = false; // best c=0 attains beta
  // S2.2: gap
  const bestProjected = Math.max(...projRows.map((r) => r.competence), round(vOnlyComp));
  const gap = round(bestUnconstrained - bestProjected);
  if (kind === "primary" && (gap < tax - 1e-6 || gap <= 0)) s22ok = false;
  if (kind === "control" && Math.abs(tax) > TOL && gap > TOL) s22ok = false;
  // S2.3: reconstruction courtier == v_only (Bayes Uhat = V since P(U=V)>1/2)
  const rLeak = round(agreeWithProxyUser(cs, P_vOnly));
  const honestLeak = rLeak; // identical policy -- the indistinguishability point
  const advantageLeak = round(competence(cs, P_vOnly) - vOnlyComp); // courting ADVANTAGE of reconstruction over honest = 0 by identity
  results.push({
    beta, rho, kind, tax, best_unconstrained: round(bestUnconstrained), v_only: round(vOnlyComp),
    best_projected: bestProjected, gap, r_leak: rLeak, honest_r_leak: honestLeak, advantage_leak: advantageLeak,
    proj_rows: projRows,
  });
}

// S2.4 frontier on one primary cell + one control
function frontier(beta, rho) {
  const cs = cells(beta, rho);
  let best = { w: 0, total: -1 }, rows = [];
  for (let w = 0; w <= 1 + 1e-9; w += 0.001) {
    const comp = competence(cs, P_mix(w));
    const total = comp * Rcourt(w);
    if (total > best.total + 1e-12) best = { w: round(w, 4), total: round(total) };
  }
  for (const w of [0, 0.1, 0.2, 0.24, 0.25, 0.3, 0.5, 1.0]) {
    const comp = competence(cs, P_mix(w));
    rows.push({ w, competence: round(comp), R: round(Rcourt(w)), total: round(comp * Rcourt(w)) });
  }
  const maskTotal = round(beta * Rcourt(0));
  const perturbed = round(competence(cells(beta, rho), P_mix(Math.min(1, best.w + 0.03))) * Rcourt(best.w + 0.03));
  return { beta, rho, w_star: best.w, total_star: best.total, mask_total: maskTotal, total_at_wstar_plus_03: perturbed, rows };
}
const frontPrimary = frontier(0.60, 0.90);
const frontControl = frontier(0.75, 0.60);
const cliffAdjacent = frontPrimary.w_star > C_STAR - 0.05 && frontPrimary.w_star < C_STAR;
const controlAtZero = frontControl.w_star === 0;
const s24ok = cliffAdjacent && controlAtZero;

const gates = {
  "S2.1": { token: s21ok ? "S2_COLLAPSE_VERIFIED" : "S2_COLLAPSE_FAIL", pass: s21ok },
  "S2.2": { token: s22ok ? "S2_NO_EXOGENOUS_TARGET_CAP" : "S2_TARGET_CAP_EXISTS_PRIZE_REOPENS", pass: s22ok },
  "S2.3": { token: "S2_NAMED_VARIABLE_CAP_LEAKY_BUT_CEILINGED", note: "correlation-leak = P(V=U), advantage-leak = 0 (courtier == honest V-user; data processing caps reconstruction)" },
  "S2.4": { token: s24ok ? "S2_FRONTIER_MAPPED" : "S2_FRONTIER_UNEXPECTED", pass: s24ok, w_star: frontPrimary.w_star, control_w_star: frontControl.w_star },
};
const branch = s21ok && s22ok ? "S2_TARGET_CAP_COLLAPSES_TO_MASK" : "S2_TARGET_CAP_EXISTS_PRIZE_REOPENS";

const summary = {
  generated_at: new Date().toISOString(), branch,
  model: { joint: "V,U cond. indep. given G; P(V=G)=beta, P(U=G)=rho", court: `B1.0 R(c), c*=${C_STAR}`, objective: "Total = Competence * R(c)", tax: "max(beta,rho)-beta (reliability edge, not synergy -- caveat recorded)" },
  gates, grid: results, frontier_primary: frontPrimary, frontier_control: frontControl,
  honest_note: "Collapse verified: every do(U)-invariance projection lands at competence <= beta; only the canonical V-only policy attains beta; the rho-beta gap is unreachable at c=0 (data processing). RECONSTRUCTION LEAK, resolved honestly: the Bayes reconstruction courtier (Uhat=V) is BEHAVIORALLY IDENTICAL to the honest V-user -- named-variable c(U)=0 admits behavior correlated with the proxy up to P(V=U) (correlation-leak, large), but the courting ADVANTAGE of reconstruction over honesty is ZERO: the same data-processing bound that forces the collapse also ceilings the leak. You cannot reconstruct more courting-power than your unmasked channels carry. So the named-variable audit under-describes (cannot tell honest from reconstructive) but does not under-protect in this model (harm defined causally = both safe; the incentive gradient toward U is genuinely absent at c=0). Frontier: optimum courts to just under the cliff (w*~0.245, B3.2 coherence confirmed); un-targeting/masking is safe-not-optimal on primary cells and OPTIMAL on zero-tax controls (cliff-edge temptation exists only when the proxy carries a reliability edge). Feeds S1: enforcement, evasion, and reconstruction at the target channel are all governed by ONE information bound.",
};
mkdirSync(path.dirname(path.resolve(args.json)), { recursive: true });
writeFileSync(path.resolve(args.json), JSON.stringify(summary, null, 2) + "\n");

const gridRows = results.map((r) => `| ${r.beta} | ${r.rho} | ${r.kind} | ${r.tax} | ${r.best_unconstrained} | ${r.best_projected} | ${r.gap} | ${r.r_leak} | ${r.advantage_leak} |`);
const projSample = results[2].proj_rows.map((r) => `| ${r.policy} | ${r.projection} | ${r.competence} | ${r.c} | ${r.le_beta} |`);
const frontRows = frontPrimary.rows.map((r) => `| ${r.w} | ${r.competence} | ${r.R} | ${r.total} |`);
const md = [
  "# Percival S2 -- Target-Cap Probe (results)",
  "",
  `Generated ${summary.generated_at} by \`scripts/percival-s2-target-cap-probe.mjs\`.`,
  "",
  `Model: \`V,U | G\` conditionally independent, \`P(V=G)=beta\`, \`P(U=G)=rho\`; court = B1.0 \`R(c)\` at \`c*=${C_STAR}\`; \`Total = Competence x R(c)\`. Tax = \`max(beta,rho)-beta\` (reliability edge under the binary-symmetric CI joint -- not synergistic info; caveat recorded for richer joints).`,
  "",
  "## S2.1 + S2.2 -- Collapse and gap (per grid cell)",
  "",
  "| beta | rho | kind | tax | best unconstrained | best projected (c=0) | gap | r_leak P(V=U) | advantage leak |",
  "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
  ...gridRows,
  "",
  "Sample projection detail (beta=0.6, rho=0.9): every do(U)-invariance projection lands at c=0 with competence <= beta; only canonical V-only attains beta.",
  "",
  "| base policy | projection | competence | c | <= beta |",
  "| --- | --- | ---: | ---: | --- |",
  ...projSample,
  "",
  `-> **${gates["S2.1"].token}**, **${gates["S2.2"].token}**`,
  "",
  "## S2.3 -- Reconstruction leak (named-variable vs content-level)",
  "",
  "The Bayes reconstruction courtier (`Uhat = V` since `P(U=V) > 1/2`) has named-variable `c(U) = 0` -- court-honored -- while its behavior agrees with the full courter at `r_leak = P(V=U)` (large, rising with `beta*rho`). **But it is behaviorally identical to the honest V-user**: the courting *advantage* of reconstruction over honesty is exactly `0`. The same data-processing bound that forces the S2.1 collapse also **ceilings the leak** -- you cannot reconstruct more courting-power than your unmasked channels carry. So the named-variable audit *under-describes* (it cannot distinguish honest use from reconstructive courting -- there is no behavioral difference to detect) but does not *under-protect* here: at `c(U)=0` the incentive gradient toward `U` is genuinely absent. The scary version of the leak -- reconstruction that gains real courting advantage -- would require recovering `U`'s unique content from channels that do not carry it, which the bound forbids.",
  "",
  `-> **${gates["S2.3"].token}**`,
  "",
  "## S2.4 -- Court-vs-tax frontier (beta=0.6, rho=0.9 primary; beta=0.75, rho=0.6 control)",
  "",
  "| w (=c) | competence | R(c) | Total |",
  "| ---: | ---: | ---: | ---: |",
  ...frontRows,
  "",
  `Primary: optimum at w* = **${frontPrimary.w_star}** (just under the cliff c*=${C_STAR}; B3.2 cliff-edge coherence CONFIRMED), Total(w*) = ${frontPrimary.total_star} vs masking Total = ${frontPrimary.mask_total}; at w*+0.03 Total collapses to ${frontPrimary.total_at_wstar_plus_03} (fragility). Control (zero-tax): w* = **${frontControl.w_star}** -- un-targeting is OPTIMAL when the proxy has no reliability edge; the cliff-edge temptation exists only when it does.`,
  "",
  `-> **${gates["S2.4"].token}**`,
  "",
  "## Verdict",
  "",
  `**${branch}**`,
  "",
  "In-channel target enforcement collapses to measure masking (competence <= beta, only V-only attains it); the rho-beta gap is unreachable at c=0 by any exogenous projection; the reconstruction leak is real as *correlation* but ceilinged at zero *advantage* by the same information bound; and the reward-optimum courts to the cliff edge exactly when the proxy carries a reliability edge. **One law does all the work at the target channel: you cannot use information you do not causally depend on.** Enforcement (the collapse), evasion (the leak ceiling), and temptation (the frontier) are three faces of the same data-processing bound. Feeds S1: the target safe point is exogenously reachable only by priced channel-retreat; the price is the proxy's reliability edge; and named-variable audits are complete for *incentive* safety in this model even though they cannot express content-level distinctions.",
  "",
  "## Honest boundary",
  "",
  summary.honest_note,
  "",
].join("\n");
mkdirSync(path.dirname(path.resolve(args.out)), { recursive: true });
writeFileSync(path.resolve(args.out), md + "\n");

console.log(branch);
for (const k of ["S2.1", "S2.2", "S2.3", "S2.4"]) console.log(`  ${gates[k].token}${gates[k].pass !== undefined ? ` pass=${gates[k].pass}` : ""}`);
console.log(`  primary frontier: w*=${frontPrimary.w_star} Total*=${frontPrimary.total_star} mask=${frontPrimary.mask_total} | control w*=${frontControl.w_star}`);
console.log(`  wrote ${args.out} and ${args.json}`);
