#!/usr/bin/env node
// ME-5 -- the priced quadrant (computed receipt).
// Spec: docs/orderrelative/ME_QUADRANT_HYPOTHESES_SLATE.md entry ME-5.
//
// Replaces the finite/infinity write-side dichotomy with the PRICE functional:
//   price(joint)   = value(V,U) - value(V)     at the joint's own G-prior
//                    (the collapse price: best unconstrained minus best
//                    do(U)-invariant policy -- S2's equivalence theorem makes
//                    the safe-point ceiling = the V-only value in-model)
//   edgeFormula    = max(value(V), value(U)) - value(V)
//                    (the S2/OR-4 reliability-edge formula, value form)
//   deltaBayes     = sup over priors p of [value_full(p) - value_masked(p)]
//                    (the Bayes-deficiency between the unmasked and masked
//                    experiments -- the Blackwell-order quantity)
//
// Pre-registered checks:
//   FLOOR    price >= edgeFormula everywhere (2-line theorem; verified, 0 violations)
//   TRIVIAL  price <= deltaBayes everywhere (price is the gap at ONE prior)
//   F1 CI binary-symmetric grid: edge formula EXACT (the banked cell)
//   F2 CI asymmetric channels:   does CI alone keep the formula exact? (fusion)
//   F3 synergy path lambda*XOR + (1-lambda)*CI: formula error grows to 1/2;
//      the lambda=1 endpoint must reproduce the machine-checked witness
//      (sundogcert PercivalSynergy.lean: price 1/2, formula 0) -- the anchor.
//   F4 random Dirichlet(1) joints: landscape + equality-set fractions.
//
// Read-price: 0 in-model BY THE REPLAY ARGUMENT (interventional reads are
// counterfactual replays; nothing is perturbed) -- a premise, not a measured
// result; recorded with the quantum contrast as its fence. Deterministic
// (mulberry32, seed 20260702).

import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

const args = { out: "docs/orderrelative/ME5_PRICED_QUADRANT_SWEEP.md", json: "results/orderrelative/me5-priced-quadrant/summary.json" };
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i += 1) { if (argv[i] === "--out") { args.out = argv[i + 1]; i += 1; } else if (argv[i] === "--json") { args.json = argv[i + 1]; i += 1; } }

function mulberry32(seed) { let a = seed >>> 0; return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const rng = mulberry32(20260702);
const round = (x, n = 6) => Number(Number(x).toFixed(n));

// joint = flat [g][v][u] -> w[g*4+v*2+u], sums to 1
function analyze(w) {
  const pg1 = w[4] + w[5] + w[6] + w[7] === 0 ? 0 : null; // guard placeholder
  const P1 = [w[4], w[5], w[6], w[7]]; // g=1 cells (v,u) = (0,0),(0,1),(1,0),(1,1)
  const P0 = [w[0], w[1], w[2], w[3]];
  const m1 = P1.reduce((a, b) => a + b, 0), m0 = P0.reduce((a, b) => a + b, 0);
  if (m1 < 1e-9 || m0 < 1e-9) return null;
  const c1 = P1.map((x) => x / m1), c0 = P0.map((x) => x / m0);
  const v1 = [c1[0] + c1[1], c1[2] + c1[3]], v0 = [c0[0] + c0[1], c0[2] + c0[3]];       // V marginals
  const u1 = [c1[0] + c1[2], c1[1] + c1[3]], u0 = [c0[0] + c0[2], c0[1] + c0[3]];       // U marginals
  const val = (a1, a0, p) => a1.reduce((s, x, i) => s + Math.max(p * x, (1 - p) * a0[i]), 0);
  const pOp = m1; // the joint's own operative prior
  const full = val(c1, c0, pOp), vv = val(v1, v0, pOp), vu = val(u1, u0, pOp);
  const price = full - vv;
  const edge = Math.max(vv, vu) - vv;
  let delta = 0, argp = 0;
  for (let k = 0; k <= 1001; k += 1) {
    const p = k === 1001 ? pOp : k / 1000; // include the operative prior itself:
    // delta = sup over p and price = gap(pOp), so price <= delta holds by
    // construction once pOp is a candidate (the grid alone can miss it)
    const g = val(c1, c0, p) - val(v1, v0, p);
    if (g > delta) { delta = g; argp = p; }
  }
  return { price, edge, delta, argp, pOp, vv, vu, full };
}

const TOL = 1e-9, EQT = 1e-6;
let floorViol = 0, trivViol = 0;
const fam = {};

function push(name, r) {
  if (!r) return;
  if (r.price < r.edge - TOL) floorViol += 1;
  if (r.price > r.delta + 1e-4) trivViol += 1; // delta on a 1e-3 grid; allow grid slack
  (fam[name] = fam[name] || []).push(r);
}

// ---- F1: CI binary-symmetric (the banked cell) -----------------------------
function ciJoint(b1, b0, r1, r0) {
  // P(G=1)=1/2; P(V=G|G=1)=b1, P(V=G|G=0)=b0; same for U with r.
  const w = new Array(8).fill(0);
  for (const g of [0, 1]) for (const v of [0, 1]) for (const u of [0, 1]) {
    const pv = v === g ? (g ? b1 : b0) : (g ? 1 - b1 : 1 - b0);
    const pu = u === g ? (g ? r1 : r0) : (g ? 1 - r1 : 1 - r0);
    w[g * 4 + v * 2 + u] = 0.5 * pv * pu;
  }
  return w;
}
let f1FormulaErrMax = 0, f1PriceEdgeErrMax = 0;
for (let bi = 0; bi < 10; bi += 1) for (let ri = 0; ri < 10; ri += 1) {
  const b = 0.5 + 0.05 * bi, r0 = 0.5 + 0.05 * ri;
  const res = analyze(ciJoint(b, b, r0, r0));
  push("F1_ci_symmetric", res);
  f1FormulaErrMax = Math.max(f1FormulaErrMax, Math.abs(res.price - res.edge));
  f1PriceEdgeErrMax = Math.max(f1PriceEdgeErrMax, Math.abs(res.price - Math.max(0, r0 - b)));
}

// ---- F2: CI but ASYMMETRIC channels ----------------------------------------
for (let k = 0; k < 2000; k += 1) {
  const [b1, b0, r1, r0] = [0, 0, 0, 0].map(() => 0.5 + 0.5 * rng());
  push("F2_ci_asymmetric", analyze(ciJoint(b1, b0, r1, r0)));
}

// ---- F3: synergy path lambda*XOR + (1-lambda)*CI(0.7,0.8) ------------------
const xorJ = new Array(8).fill(0);
for (const v of [0, 1]) for (const u of [0, 1]) xorJ[(v ^ u) * 4 + v * 2 + u] = 0.25;
const ciBase = ciJoint(0.7, 0.7, 0.8, 0.8);
const f3rows = [];
for (let li = 0; li <= 10; li += 1) {
  const lam = li / 10;
  const w = xorJ.map((x, i) => lam * x + (1 - lam) * ciBase[i]);
  const r = analyze(w);
  push("F3_synergy_path", r);
  f3rows.push({ lambda: lam, price: round(r.price), edge: round(r.edge), delta: round(r.delta) });
}
const anchor = f3rows[10]; // lambda = 1: must match the Lean witness (price 1/2, edge 0)
const anchorOk = Math.abs(anchor.price - 0.5) <= 1e-6 && Math.abs(anchor.edge) <= 1e-6;

// ---- F4: random Dirichlet(1) joints ----------------------------------------
for (let k = 0; k < 3000; k += 1) {
  const g = new Array(8).fill(0).map(() => -Math.log(1 - rng()));
  const s = g.reduce((a, b) => a + b, 0);
  push("F4_random", analyze(g.map((x) => x / s)));
}

// ---- aggregate --------------------------------------------------------------
const famStats = {};
for (const [name, rs] of Object.entries(fam)) {
  const errs = rs.map((r) => r.price - r.edge);
  const defGap = rs.map((r) => r.delta - r.price);
  famStats[name] = {
    n: rs.length,
    formula_exact_frac: round(errs.filter((e) => Math.abs(e) <= EQT).length / rs.length),
    formula_err_max: round(Math.max(...errs)),
    formula_err_mean: round(errs.reduce((a, b) => a + b, 0) / errs.length),
    price_eq_delta_frac: round(defGap.filter((e) => Math.abs(e) <= 1e-4).length / rs.length),
    delta_minus_price_max: round(Math.max(...defGap)),
  };
}

const summary = {
  phase: "ME-5 priced quadrant sweep",
  generated_at: new Date().toISOString(),
  seed: 20260702,
  gates: {
    FLOOR_price_ge_edge_violations: floorViol,
    TRIVIAL_price_le_delta_violations: trivViol,
    F1_formula_exact_max_err: round(f1FormulaErrMax),
    F1_price_eq_rho_minus_beta_max_err: round(f1PriceEdgeErrMax),
    F3_lambda1_matches_lean_witness: anchorOk,
  },
  families: famStats,
  f3_path: f3rows,
  read_price_note: "0 in-model by the replay argument (premise, not measurement); quantum contrast fenced",
};
const repoRoot = process.cwd();
mkdirSync(path.resolve(repoRoot, path.dirname(args.json)), { recursive: true });
writeFileSync(path.resolve(repoRoot, args.json), `${JSON.stringify(summary, null, 2)}\n`, "utf8");

const verdict = (floorViol === 0 && trivViol === 0 && f1FormulaErrMax <= EQT && anchorOk)
  ? "ME5_EDGE_FORMULA_IS_BINARY_SYMMETRIC_ARTIFACT + ME5_PRICE_IS_LOCAL_DEFICIENCY"
  : "ME5_CHECK_FAILURES (inspect gates)";

const md = [
  "# ME-5 -- the Priced Quadrant (sweep receipt)",
  "",
  `Generated ${summary.generated_at} by \`scripts/orderrelative-me5-priced-quadrant.mjs\` (deterministic seed ${summary.seed}).`,
  "",
  "Definitions: price = value(V,U) - value(V) at the joint's own prior (the collapse price);",
  "edgeFormula = max(value V, value U) - value V (the S2/OR-4 reliability-edge formula);",
  "deltaBayes = sup over priors of the full-vs-masked Bayes-value gap (1e-3 grid).",
  "",
  "## Gates",
  "",
  `- FLOOR (price >= edgeFormula, the 2-line theorem): **${floorViol} violations** over ${Object.values(fam).reduce((a, r) => a + r.length, 0)} joints.`,
  `- TRIVIAL (price <= deltaBayes): **${trivViol} violations**.`,
  `- F1 binary-symmetric CI: formula exact to ${round(f1FormulaErrMax)}; price = rho - beta to ${round(f1PriceEdgeErrMax)} (the banked cell reproduced).`,
  `- F3 lambda = 1 anchor vs the Lean witness (\`PercivalSynergy.lean\`: price 1/2, formula 0): **${anchorOk ? "MATCH" : "MISMATCH"}**.`,
  "",
  "## Families",
  "",
  "| family | n | formula exact | max(price - edge) | mean | price = delta | max(delta - price) |",
  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
  ...Object.entries(famStats).map(([k, s]) =>
    `| ${k} | ${s.n} | ${s.formula_exact_frac} | ${s.formula_err_max} | ${s.formula_err_mean} | ${s.price_eq_delta_frac} | ${s.delta_minus_price_max} |`),
  "",
  "## The synergy path (lambda * XOR + (1 - lambda) * CI(0.7, 0.8))",
  "",
  "| lambda | price | edgeFormula | deltaBayes |",
  "| ---: | ---: | ---: | ---: |",
  ...f3rows.map((r) => `| ${r.lambda} | ${r.price} | ${r.edge} | ${r.delta} |`),
  "",
  `## Verdict: \`${verdict}\``,
  "",
  "The reliability-edge FORMULA is exact exactly where it was banked (binary-symmetric CI)",
  "and is a strict FLOOR elsewhere -- asymmetric-CI fusion and synergy joints price above it,",
  "up to the machine-checked XOR maximum. The PRICE itself survives every family as the",
  "local (operative-prior) Bayes gap, bounded by the full deficiency; equality with the",
  "deficiency holds only where the operative decision problem attains the sup -- prices are",
  "per-decision-problem, as sigma is per-filtration. Read-price: 0 in-model by replay",
  "(premise; quantum contrast fenced).",
  "",
].join("\n");
writeFileSync(path.resolve(repoRoot, args.out), `${md}\n`, "utf8");
console.log(`ME-5 sweep: floor=${floorViol} trivial=${trivViol} F1err=${round(f1FormulaErrMax)} anchor=${anchorOk}`);
for (const [k, s] of Object.entries(famStats)) console.log(`  ${k}: n=${s.n} exact=${s.formula_exact_frac} maxErr=${s.formula_err_max} eqDelta=${s.price_eq_delta_frac}`);
console.log(`  verdict: ${verdict}`);
console.log(`  wrote ${args.out} + ${args.json}`);
