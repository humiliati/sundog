// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stellar Aqua LLC
// Licensed under the Apache License, Version 2.0, via the manifest-scoped
// Percival grant: docs/percival/LICENSE.md (MANIFEST.json = covered files).
// Distributed "AS IS"; see http://www.apache.org/licenses/LICENSE-2.0.
// Percival B2 -- gamma-provenance (fixed-cliff). PRIVATE / structural.
// [DECLASSIFIED 2026-07-01 by owner decision -- published under the manifest-scoped
//  Percival Apache-2.0 grant; "PRIVATE" above is the historical register. Claim
//  boundary unchanged: structural characterization, no measured real-base claim.]
//
// Primary question (non-manufacturable): at the FIXED admitted cliff c*_0 = 0.25
// (B1.0, B=1,K=3), does a realistic/trusted base clear the clean-prize gate
// inf supp(gamma) > c*_0, or does it straddle? Court strictness is NOT a free knob;
// it appears only as a secondary purism bill (D-B2.3), which cannot upgrade a null.
//
// Registered prior (not a theorem): a quantilizer-trusted base includes restrained,
// low-courting behavior (mass near c=0), so c_min ~ 0, so rho_0 = P_gamma(c<c*_0) > 0.
// The clean prize needs c_min > c*_0 -- a base with NO restraint. Positive surprise =
// "trust without restraint" (a plausibly trusted base with c_min > c*_0).

import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

const args = { out: "docs/percival/PERCIVAL_B2_GAMMA_PROVENANCE_RESULTS.md", json: "results/percival/b2-provenance/summary.json" };
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i += 1) { if (argv[i] === "--out") { args.out = argv[i + 1]; i += 1; } else if (argv[i] === "--json") { args.json = argv[i + 1]; i += 1; } }

// ---- court reward R(c) (identical to B1.0 / B1-proper) ----
const N = 401, SIGMA = 0.05, B = 1, K = 3, C_STAR0 = B / (B + K); // 0.25 fixed
const round = (x, n = 6) => Number(Number(x).toFixed(n));
function erf(x) { const s = x < 0 ? -1 : 1; x = Math.abs(x); const p = 0.3275911, a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429; const t = 1 / (1 + p * x); return s * (1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x)); }
const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
function probit(p) { const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]; const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]; const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]; const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]; const pl = 0.02425, ph = 1 - pl; let q, r; if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); } if (p <= ph) { q = p - 0.5; r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); } q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
function gammaln(xx) { const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]; let x = xx, y = xx, tmp = x + 5.5; tmp -= (x + 0.5) * Math.log(tmp); let ser = 1.000000000190015; for (let j = 0; j < 6; j++) { y++; ser += cof[j] / y; } return -tmp + Math.log(2.5066282746310005 * ser / x); }
function betacf(a, b, x) { const MAXIT = 300, EPS = 3e-12, FPMIN = 1e-300; let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < FPMIN) d = FPMIN; d = 1 / d; let h = d; for (let m = 1; m <= MAXIT; m++) { let m2 = 2 * m, aa = m * (b - m) * x / ((qam + m2) * (a + m2)); d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; h *= d * c; aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)); d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; let del = d * c; h *= del; if (Math.abs(del - 1) < EPS) break; } return h; }
function betai(a, b, x) { if (x <= 0) return 0; if (x >= 1) return 1; const bt = Math.exp(gammaln(a + b) - gammaln(a) - gammaln(b) + a * Math.log(x) + b * Math.log(1 - x)); return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b; }
const binomTailGE = (n, p, m) => (m <= 0 ? 1 : m > n ? 0 : betai(m, n - m + 1, p));
const xStar = K / (B + K) + SIGMA * probit(K / (B + K));
function R(c) { const theta = 1 - c; const p = Phi((theta - xStar) / SIGMA); const m = Math.ceil(N * (1 - theta)); return binomTailGE(N, p, m); }
function EgammaR(cmin, cmax, n = 600) { if (cmax <= cmin) return R(cmin); const h = (cmax - cmin) / n; let s = 0.5 * (R(cmin) + R(cmax)); for (let i = 1; i < n; i++) s += R(cmin + i * h); return (s * h) / (cmax - cmin); }

const R0 = round(R(0));
const region = (cmin, cmax) => (cmax < C_STAR0 ? "support_below" : cmin > C_STAR0 ? "support_above" : "straddle");
function evalBase(cmin, cmax, label) {
  cmin = round(cmin, 6); cmax = round(cmax, 6);
  const rho0 = round(Math.max(0, Math.min(1, (C_STAR0 - cmin) / (cmax - cmin))));
  const Eg = round(EgammaR(cmin, cmax));
  const margin = round(R0 - Eg);           // = R(0) - max_q Rbar(q) by lemma 1
  const cStarNeeded = round(cmin);          // court cliff must drop strictly below c_min
  const payable = cmin > 1e-9;              // a positive stricter court exists only if c_min>0
  return { label, c_min: cmin, c_max: cmax, region: region(cmin, cmax), rho0, E_gamma_R: Eg, margin, c_star_needed: cStarNeeded, purism_payable: payable };
}

// ---- registered probe bases ----
const trusted = evalBase(0.00, 0.40, "trusted (includes restraint at c=0 + grasping past cliff)");
const corrupt = evalBase(0.30, 0.60, "corrupt / trust-without-restraint (uniformly past cliff)");
const restrained = evalBase(0.00, 0.20, "heavily restrained (entirely below cliff)");

// ============ D-B2.1 -- fixed-cliff provenance map ============
const MUS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50], SPREADS = [0.10, 0.20];
const grid = [];
for (const s of SPREADS) for (const mu of MUS) grid.push(evalBase(Math.max(0, mu - s), Math.min(1, mu + s), `mu=${mu},s=${s}`));
const below = grid.filter((g) => g.region === "support_below");
const straddle = grid.filter((g) => g.region === "straddle");
const above = grid.filter((g) => g.region === "support_above");
const cleanConfinedAbove = above.every((g) => g.c_min > C_STAR0) && above.every((g) => g.margin > 0.9);
const trustedInStraddle = trusted.region === "straddle" && trusted.margin < 0.9 && trusted.rho0 > 0;
const D_B21_PASS = below.length > 0 && straddle.length > 0 && above.length > 0 && cleanConfinedAbove && trustedInStraddle;

// ============ D-B2.2 -- trust<->separation tradeoff ============
const W = 0.30;
const tradeoff = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00].map((cmin) => {
  const g = evalBase(cmin, cmin + W, `cmin=${cmin}`);
  return { c_min: g.c_min, rho0: g.rho0, margin: g.margin };
});
let monotone = true;
for (let i = 1; i < tradeoff.length; i++) if (tradeoff[i].margin > tradeoff[i - 1].margin + 1e-6) monotone = false; // margin decreases as rho0 increases
const cleanOnlyAtZeroRho = tradeoff.filter((t) => t.margin > 0.9).every((t) => t.rho0 < 0.05);
const D_B22_PASS = monotone && cleanOnlyAtZeroRho;

// ============ D-B2.3 -- purism bill (context only) ============
const purismBill = {
  trusted: { c_star_needed: trusted.c_star_needed, payable: trusted.purism_payable, note: trusted.purism_payable ? "a stricter court below c_min would clean it" : "c_min=0: NO positive court cliff can make it clean -- structurally unreachable" },
  corrupt: { c_star_needed: corrupt.c_star_needed, already_clean_at_c_star0: corrupt.region === "support_above" },
};

// ============ verdict ============
const cleanIsCorner = D_B21_PASS && D_B22_PASS && corrupt.region === "support_above" && corrupt.margin > 0.9 && !trusted.purism_payable;
const branch = cleanIsCorner ? "B2_CLEAN_IS_CORNER" : "B2_CLEAN_REACHABLE_REALISTIC";

const gates = {
  "D-B2.1": { token: D_B21_PASS ? "B2_FIXED_CLIFF_MAPPED" : "B2_FIXED_CLIFF_FAIL", pass: D_B21_PASS, regions: { below: below.length, straddle: straddle.length, above: above.length }, clean_confined_above: cleanConfinedAbove, trusted_in_straddle: trustedInStraddle },
  "D-B2.2": { token: D_B22_PASS ? "B2_TRADEOFF_MONOTONE" : "B2_TRADEOFF_FAIL", pass: D_B22_PASS, monotone, clean_only_at_zero_rho: cleanOnlyAtZeroRho },
  "D-B2.3": { note: "context only; cannot upgrade a fixed-cliff null", purism_bill: purismBill },
};
const summary = {
  generated_at: new Date().toISOString(), branch, private: true,
  model: { c_star0: C_STAR0, court: "B1.0 fixed (B=1,K=3,sigma=0.05)", base: "uniform[c_min,c_max]", clean_gate: "c_min > c*_0", diagnostic: "rho0 = P_gamma(c<c*_0)" },
  R0, probes: { trusted, corrupt, restrained }, gates, tradeoff,
  honest_note: "Primary verdict at FIXED c*_0=0.25. A trusted base (registered prior: includes restraint, c_min~0) straddles -> only PARTIAL separation (margin=R0-E_gamma[R]). The clean prize needs c_min>c*_0 (no restraint = trust-without-restraint / corrupt base). Because a base with any restraint mass has c_min=0, NO positive court cliff makes it clean -- the clean prize is structurally unreachable for a restraint-bearing base, not merely unlikely. Court-strictness sweep is a purism bill only; it cannot rescue the null. No measured real base is asserted.",
};
mkdirSync(path.dirname(path.resolve(args.json)), { recursive: true });
writeFileSync(path.resolve(args.json), JSON.stringify(summary, null, 2) + "\n");

const gridRows = grid.map((g) => `| ${g.label} | ${g.c_min} | ${g.c_max} | ${g.region} | ${g.rho0} | ${g.E_gamma_R} | ${g.margin} |`);
const tradeRows = tradeoff.map((t) => `| ${t.c_min} | ${t.rho0} | ${t.margin} |`);
const md = [
  "# Percival B2 -- gamma-Provenance (results, PRIVATE)",
  "",
  `Generated ${summary.generated_at} by \`scripts/percival-b2-gamma-provenance.mjs\`.`,
  "",
  `**PRIVATE / structural.** Fixed admitted cliff c*_0 = **${C_STAR0}** (B1.0). Un-targeting collects R(0) = **${R0}**. Clean gate: c_min > c*_0. Diagnostic: rho_0 = P_gamma(c<c*_0).`,
  "",
  "## Registered probe bases",
  "",
  "| base | c_min | c_max | region | rho_0 | E_gamma[R] | margin |",
  "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
  `| ${trusted.label} | ${trusted.c_min} | ${trusted.c_max} | **${trusted.region}** | ${trusted.rho0} | ${trusted.E_gamma_R} | ${trusted.margin} |`,
  `| ${corrupt.label} | ${corrupt.c_min} | ${corrupt.c_max} | **${corrupt.region}** | ${corrupt.rho0} | ${corrupt.E_gamma_R} | ${corrupt.margin} |`,
  `| ${restrained.label} | ${restrained.c_min} | ${restrained.c_max} | **${restrained.region}** | ${restrained.rho0} | ${restrained.E_gamma_R} | ${restrained.margin} |`,
  "",
  `The trusted base (registered prior: includes restraint) **straddles** -> partial margin ${trusted.margin}, not the clean prize. The corrupt / trust-without-restraint base is support-above -> clean margin ${corrupt.margin}.`,
  "",
  "## D-B2.1 -- Fixed-cliff provenance map",
  "",
  "| base | c_min | c_max | region | rho_0 | E_gamma[R] | margin |",
  "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
  ...gridRows,
  "",
  `regions present (below/straddle/above): ${below.length}/${straddle.length}/${above.length}; clean confined to support-above: **${cleanConfinedAbove}**; trusted base in straddle: **${trustedInStraddle}**. -> **${gates["D-B2.1"].token}**`,
  "",
  "## D-B2.2 -- Trust<->separation tradeoff (width 0.30, sliding c_min)",
  "",
  "| c_min | rho_0 | margin |",
  "| ---: | ---: | ---: |",
  ...tradeRows,
  "",
  `margin monotone-decreasing in rho_0: **${monotone}**; clean margin (>0.9) only near rho_0 = 0: **${cleanOnlyAtZeroRho}**. More restraint (more trust) -> less separation. -> **${gates["D-B2.2"].token}**`,
  "",
  "## D-B2.3 -- Purism bill (context only, cannot rescue the null)",
  "",
  `Trusted base: c*_needed = ${purismBill.trusted.c_star_needed}, payable: **${purismBill.trusted.payable}** -- ${purismBill.trusted.note}. Corrupt base: already clean at c*_0 (${purismBill.corrupt.already_clean_at_c_star0}).`,
  "",
  "## Verdict",
  "",
  `**${branch}**`,
  "",
  cleanIsCorner
    ? "At the fixed admitted cliff, the clean prize is confined to the corner: only a corrupt / trust-without-restraint base (c_min > c*_0) is support-above. The registered trusted base straddles -> partial separation only. And because any restraint mass forces c_min = 0, NO positive court cliff makes a restraint-bearing base clean -- the clean unconditional prize is **structurally unreachable for a trusted base**, not merely unlikely. Banked regardless: the partial separation (margin = R0 - E_gamma[R]) and lemma-1 counterproductivity. The unconditional prize is NOT won on trusted bases."
    : "A plausibly trusted base cleared the fixed cliff (trust without restraint) -- the positive surprise. The unconditional prize may be reachable; re-examine the trust prior.",
  "",
  "## Honest boundary",
  "",
  summary.honest_note,
  "",
].join("\n");
mkdirSync(path.dirname(path.resolve(args.out)), { recursive: true });
writeFileSync(path.resolve(args.out), md + "\n");

console.log(branch);
for (const k of ["D-B2.1", "D-B2.2"]) console.log(`  ${gates[k].token} pass=${gates[k].pass}`);
console.log(`  trusted[0,0.4]: region=${trusted.region} rho0=${trusted.rho0} margin=${trusted.margin} c*_needed=${trusted.c_star_needed} payable=${trusted.purism_payable}`);
console.log(`  corrupt[0.3,0.6]: region=${corrupt.region} margin=${corrupt.margin}`);
console.log(`  tradeoff monotone=${monotone} cleanOnlyAtZeroRho=${cleanOnlyAtZeroRho}`);
console.log(`  wrote ${args.out} and ${args.json}`);
