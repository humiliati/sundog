// Faraday Phase 8 - Inhomogeneous (Sourced) Maxwell: dual-shadow closure battery.
//
// The hand derivation in docs/faraday/FARADAY_PHASE8_BOUNDARY.md is authoritative.
// This script mirrors it via finite differences / quadrature on analytic fields.
// It decides no branch.
//
//   Case 1 (A8): electric Gauss closure - the DUAL of the Phase 7 magnetic monopole.
//                E = (x,y,z): div E = 3 = rho, oint.oint E.dA = 4pi = Q_enc.
//   Case 2 (A8): Ampere-Maxwell with displacement current (source-free, J = 0).
//                oint B.dl = d/dt flux_E (the displacement-current term closes the loop).
//   Case 3 (B8): Hodge split - charge q and AB flux Phi read by orthogonal shadows.
//
// Run: node scripts/faraday-phase8-battery.mjs   (or: npm run faraday:phase8)

import { mkdirSync, writeFileSync } from "node:fs";

const H = 1e-5;
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
const sub = (a, b) => a.map((v, i) => v - b[i]);
const amax = (a) => Math.max(...a.map(Math.abs));

// closed-surface flux oint.oint V.n dA over the unit sphere
function sphereFlux(V) {
  let f = 0; const Nth = 200, Nph = 400;
  for (let i = 0; i < Nth; i++) {
    const th = (i + 0.5) * Math.PI / Nth;
    for (let j = 0; j < Nph; j++) {
      const ph = (j + 0.5) * 2 * Math.PI / Nph;
      const n = [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
      const v = V([n[0], n[1], n[2], 0]);
      f += (v[0] * n[0] + v[1] * n[1] + v[2] * n[2]) * Math.sin(th) * (Math.PI / Nth) * (2 * Math.PI / Nph);
    }
  }
  return f;
}
// loop integral oint V.dl around a circle of radius r0 in the z = 0 plane
function loopInt(V, r0) {
  let s = 0; const N = 2000;
  for (let k = 0; k < N; k++) {
    const ph = (k + 0.5) * 2 * Math.PI / N;
    const p = [r0 * Math.cos(ph), r0 * Math.sin(ph), 0, 0];
    const dl = [-r0 * Math.sin(ph) * (2 * Math.PI / N), r0 * Math.cos(ph) * (2 * Math.PI / N), 0];
    const v = V(p);
    s += v[0] * dl[0] + v[1] * dl[1] + v[2] * dl[2];
  }
  return s;
}

// ===== Case 1 (A8): electric Gauss closure. E = (x, y, z) =====
const E1 = (x) => [x[0], x[1], x[2]];
const case1 = {
  branch: "A8-dual-closure",
  chargeDensity_divE: div(E1, [0.5, -0.3, 0.8, 0]),  // = 3 = rho
  gaussFlux_oint_EdA: sphereFlux(E1),                 // = 4pi = Q_enc
  registered_Q: 4 * Math.PI,
  note: "dual of the Phase 7 magnetic monopole - electric charge lives in d*F = J, the normal inhomogeneous side",
};

// ===== Case 2 (A8): Ampere-Maxwell with displacement current. E = t z-hat, B_phi = r/2 =====
const E2 = (x) => [0, 0, x[3]];           // E = t z-hat
const B2 = (x) => [-x[1] / 2, x[0] / 2, 0]; // B_phi = r/2 (azimuthal)
const r0 = 2.0;
const smp = [0.7, -0.4, 0.3, 0.6];
const ampereResidual = amax(sub(curl(B2, smp), ddt(E2, smp))); // = J, expect 0 (source-free)
const loopB = loopInt(B2, r0);                                  // oint B.dl
const dtEz = partial(E2, 2, 3, [0, 0, 0, 0]);                   // d/dt E_z = 1
const displacementCurrent = dtEz * Math.PI * r0 * r0;          // d/dt flux_E
const case2 = {
  branch: "A8-dual-closure",
  conductionCurrent_J: ampereResidual,                // = 0 (source-free displacement region)
  loop_oint_Bdl: loopB,                               // = pi r0^2 = 4pi
  displacementCurrent_ddt_fluxE: displacementCurrent, // = 4pi
  match_residual: Math.abs(loopB - displacementCurrent),
};

// ===== Case 3 (B8): Hodge split. charge q (+) AB flux Phi, read by orthogonal shadows =====
const PHI = 2 * Math.PI;
const A_AB = (x) => {
  const r2 = x[0] * x[0] + x[1] * x[1]; // A = (Phi/2pi) grad(theta) = (Phi/2pi)(-y, x, 0)/r^2
  return [-(PHI / (2 * Math.PI)) * x[1] / r2, (PHI / (2 * Math.PI)) * x[0] / r2, 0];
};
const case3 = {
  branch: "B8-harmonic-survivor",
  dualShadow_chargeFlux_oint_EdA: sphereFlux(E1),  // reads the SOURCE: q = 4pi
  loopShadow_holonomy_oint_A: loopInt(A_AB, r0),   // reads the HARMONIC/AB sector: Phi = 2pi
  registered_q: 4 * Math.PI,
  registered_Phi: 2 * Math.PI,
  decoupled: "charge has no vector potential (oint A_charge = 0); AB has no E-field (oint.oint E_AB.dA = 0) - the two shadows read orthogonal Hodge sectors",
  regime2_location: "harmonic (AB) only; the sourced sector is determined -> no new regime-2",
};

const manifest = {
  experiment: "faraday-phase8-boundary",
  note: "Numeric support only; hand derivation is authoritative.",
  cases: { case1, case2, case3 },
};
const outDir = "results/faraday/phase8-battery";
mkdirSync(outDir, { recursive: true });
writeFileSync(`${outDir}/manifest.json`, JSON.stringify(manifest, null, 2));

const f = (n) => Number(n).toExponential(3);
console.log("=== Faraday Phase 8 support battery ===");
console.log(`Case 1 (A8 Gauss):          divE(rho)=${case1.chargeDensity_divE.toFixed(6)}  oint_EdA=${case1.gaussFlux_oint_EdA.toFixed(6)}  Q=${case1.registered_Q.toFixed(6)}`);
console.log(`Case 2 (A8 Ampere-Maxwell): J=${f(case2.conductionCurrent_J)}  oint_Bdl=${case2.loop_oint_Bdl.toFixed(6)}  ddt_fluxE=${case2.displacementCurrent_ddt_fluxE.toFixed(6)}  match=${f(case2.match_residual)}`);
console.log(`Case 3 (B8 Hodge split):    chargeFlux=${case3.dualShadow_chargeFlux_oint_EdA.toFixed(6)} (q=${case3.registered_q.toFixed(6)})  holonomy=${case3.loopShadow_holonomy_oint_A.toFixed(6)} (Phi=${case3.registered_Phi.toFixed(6)})`);
console.log(`written: ${outDir}/manifest.json`);
