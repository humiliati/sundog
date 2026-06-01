// Faraday Phase 7 - Source / Topology Boundary Audit: numeric support battery.
//
// The hand derivation in docs/faraday/FARADAY_PHASE7_BOUNDARY.md is authoritative.
// This script is a seconds-scale receipt that MIRRORS the hand calculation via
// finite differences on analytic fields. It does not decide any branch.
//
// Three registered cases:
//   Case 1 (A7-clean-control):  electric current source, dF = 0  -> R_F = 0
//   Case 2 (B7-magnetic-source): registered dF = K_m != 0        -> named flux = Q_m
//   Case 3 (B7-topology):        Aharonov-Bohm solenoid          -> F = 0 on loop, holonomy = Phi
//
// Run: node scripts/faraday-phase7-battery.mjs   (or: npm run faraday:phase7)

import { mkdirSync, writeFileSync } from "node:fs";

const H = 1e-5;

// --- vector-calculus helpers on fields V(x) -> [Vx,Vy,Vz], x = [x,y,z,t] ---
const partial = (V, comp, axis, x) => {
  const xp = [...x], xm = [...x];
  xp[axis] += H; xm[axis] -= H;
  return (V(xp)[comp] - V(xm)[comp]) / (2 * H);
};
const curl = (V, x) => [
  partial(V, 2, 1, x) - partial(V, 1, 2, x),
  partial(V, 0, 2, x) - partial(V, 2, 0, x),
  partial(V, 1, 0, x) - partial(V, 0, 1, x),
];
const div = (V, x) => partial(V, 0, 0, x) + partial(V, 1, 1, x) + partial(V, 2, 2, x);
const ddt = (V, x) => [partial(V, 0, 3, x), partial(V, 1, 3, x), partial(V, 2, 3, x)];
const add = (a, b) => a.map((v, i) => v + b[i]);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const amax = (a) => Math.max(...a.map(Math.abs));
const hypot = (a) => Math.hypot(...a);

// ===== Case 1: electric-source control. A = (0,0, x^2 cos t) =====
const A1 = (x) => [0, 0, x[0] * x[0] * Math.cos(x[3])];
const E1 = (x) => ddt(A1, x).map((v) => -v); // E = -dA/dt (phi = 0)
const B1 = (x) => curl(A1, x); // B = curl A
const faraday1 = (x) => add(curl(E1, x), ddt(B1, x)); // curl E + dB/dt
const current1 = (x) => sub(curl(B1, x), ddt(E1, x)); // J = curl B - dE/dt  (c = 1)

const p1 = [0.7, -0.4, 0.3, 0.6];
const case1 = {
  branch: "A7-clean-control",
  sample: p1,
  faradayResidual_R_F: amax(faraday1(p1)), // expect 0
  electricCurrent_Jmag: hypot(current1(p1)), // expect != 0 (source present)
  chargeDensity_divE: div(E1, p1), // expect 0
  magneticDivergence_divB: div(B1, p1), // expect 0 (magnetically clean, dF=0)
};

// ===== Case 2: magnetic-source quarantine. B = (x,y,z) =====
const B2 = (x) => [x[0], x[1], x[2]];
const p2 = [0.5, -0.3, 0.8, 0.0];
// closed-surface (magnetic Gauss) flux over the unit sphere; B.n = r^2 = 1 there
let flux2 = 0;
const Nth = 200, Nph = 400;
for (let i = 0; i < Nth; i++) {
  const th = (i + 0.5) * Math.PI / Nth;
  for (let j = 0; j < Nph; j++) {
    const ph = (j + 0.5) * 2 * Math.PI / Nph;
    const n = [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
    const b = B2([n[0], n[1], n[2], 0]);
    flux2 += (b[0] * n[0] + b[1] * n[1] + b[2] * n[2]) * Math.sin(th) * (Math.PI / Nth) * (2 * Math.PI / Nph);
  }
}
const case2 = {
  branch: "B7-magnetic-source",
  magneticChargeDensity_divB: div(B2, p2), // = 3 = rho_m (matches Phase 4 dF_xyz=3)
  closedSurfaceFlux_oint_BdA: flux2, // = Q_m
  registered_Q_m: 4 * Math.PI,
  faradayLoopResidual: 0, // static, E = 0; survivor is the Gauss-magnetic flux
};

// ===== Case 3: topology / Aharonov-Bohm. Solenoid radius a, flux Phi =====
const PHI = 2 * Math.PI; // chosen registered flux
const a = 1.0, r0 = 2.0; // loop radius outside the solenoid (r0 > a)
const A3 = (x) => {
  const r2 = x[0] * x[0] + x[1] * x[1]; // A = (Phi/2pi) * (-y, x, 0) / r^2  = (Phi/2pi) grad(theta)
  return [-(PHI / (2 * Math.PI)) * x[1] / r2, (PHI / (2 * Math.PI)) * x[0] / r2, 0];
};
const B3 = (x) => curl(A3, x); // expect ~0 for r > 0  (F = 0)
const pL = [r0, 0, 0, 0];
let holonomy = 0;
const Nseg = 2000;
for (let k = 0; k < Nseg; k++) {
  const ph = (k + 0.5) * 2 * Math.PI / Nseg;
  const p = [r0 * Math.cos(ph), r0 * Math.sin(ph), 0, 0];
  const dl = [-r0 * Math.sin(ph) * (2 * Math.PI / Nseg), r0 * Math.cos(ph) * (2 * Math.PI / Nseg), 0];
  const A = A3(p);
  holonomy += A[0] * dl[0] + A[1] * dl[1] + A[2] * dl[2];
}
const case3 = {
  branch: "B7-topology",
  solenoidRadius_a: a,
  loopRadius_r0: r0,
  registered_Phi: PHI,
  pointTier_F_on_loop: amax(B3(pL)), // expect ~0  (control-blind)
  loopTier_holonomy_oint_A: holonomy, // expect Phi  (control-sufficient)
  separation_gap: holonomy - amax(B3(pL)), // = Phi = H^1 period
};

const manifest = {
  experiment: "faraday-phase7-boundary",
  note: "Numeric support only; hand derivation is authoritative.",
  cases: { case1, case2, case3 },
};

const outDir = "results/faraday/phase7-battery";
mkdirSync(outDir, { recursive: true });
writeFileSync(`${outDir}/manifest.json`, JSON.stringify(manifest, null, 2));

const f = (n) => Number(n).toExponential(3);
console.log("=== Faraday Phase 7 support battery ===");
console.log(`Case 1 (A7): R_F=${f(case1.faradayResidual_R_F)}  |J|=${case1.electricCurrent_Jmag.toFixed(4)}  divE=${f(case1.chargeDensity_divE)}  divB=${f(case1.magneticDivergence_divB)}`);
console.log(`Case 2 (B7-mag): divB(rho_m)=${case2.magneticChargeDensity_divB.toFixed(6)}  oint_BdA=${case2.closedSurfaceFlux_oint_BdA.toFixed(6)}  Q_m=${case2.registered_Q_m.toFixed(6)}`);
console.log(`Case 3 (B7-top): F_on_loop=${f(case3.pointTier_F_on_loop)}  holonomy=${case3.loopTier_holonomy_oint_A.toFixed(6)}  Phi=${case3.registered_Phi.toFixed(6)}`);
console.log(`written: ${outDir}/manifest.json`);
