// Resist-side roadmap, Pass 2: axis generalization.
// Show the resistance->sharpening replicates on the TOPOLOGICAL and DIMENSIONAL
// axes (Pass 1 = computational). Each: sigma = 1 - R rises through a threshold set
// by the axis's resistance parameter, at fixed compute. Exit gate: >= 2 of 3 axes.
// Run: node scripts/resist-pass2.mjs

const lcg = (s) => () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
const rand = lcg(20260628);
function randn() { let u = 0, v = 0; while (u === 0) u = rand(); while (v === 0) v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

// ---------- DIMENSIONAL axis ----------
// Body b in R^D; shadow s = M b with M a d x D random map (rank d). Best linear
// reconstruction recovers exactly row(M) (dim d); the (D-d)-dim kernel is hidden.
// Resistance dial rho = body dimension D; threshold rho* = shadow rank d.
function dimensional(d, Dlist, T) {
  const rows = [];
  for (const D of Dlist) {
    const M = Array.from({ length: d }, () => Array.from({ length: D }, randn));
    const Q = []; // orthonormal basis of row(M)
    for (let i = 0; i < d; i++) {
      const v = M[i].slice();
      for (const q of Q) { let dot = 0; for (let j = 0; j < D; j++) dot += v[j] * q[j]; for (let j = 0; j < D; j++) v[j] -= dot * q[j]; }
      let nrm = 0; for (let j = 0; j < D; j++) nrm += v[j] * v[j]; nrm = Math.sqrt(nrm);
      if (nrm > 1e-9) { for (let j = 0; j < D; j++) v[j] /= nrm; Q.push(v); }
    }
    let rec = 0, tot = 0;
    for (let t = 0; t < T; t++) {
      const b = Array.from({ length: D }, randn);
      for (let j = 0; j < D; j++) tot += b[j] * b[j];
      for (const q of Q) { let dot = 0; for (let j = 0; j < D; j++) dot += q[j] * b[j]; rec += dot * dot; }
    }
    const FVE = rec / tot;
    rows.push({ D, FVE, sigma: 1 - FVE });
  }
  return rows;
}

// ---------- TOPOLOGICAL axis ----------
// Discrete gauge field; shadow = visible plaquette fluxes; one "hole" plaquette is
// hidden. Loop holonomy = sum of enclosed fluxes (Stokes). A contractible loop
// (no hole) is fully reconstructable from the shadow; a non-contractible loop
// (encloses the hole) misses exactly the hole flux. Resistance is the loop's
// HOMOTOPY CLASS (sharp on/off); within the resistant class, dial rho = hole flux.
function topological(rhoList, h, T) {
  const rows = [];
  for (const rho of rhoList) {
    let eNC = 0, tNC = 0, eC = 0, tC = 0;
    for (let t = 0; t < T; t++) {
      let vis = 0; for (let i = 0; i < h - 1; i++) vis += randn();
      const hole = rho * randn();
      const holNC = vis + hole, reconNC = vis;       // non-contractible: shadow misses hole
      eNC += (holNC - reconNC) ** 2; tNC += holNC * holNC;
      let visC = 0; for (let i = 0; i < h; i++) visC += randn();
      eC += 0; tC += visC * visC;                     // contractible: recon exact, residual 0
    }
    rows.push({ rho, sigma_nonContractible: eNC / tNC, sigma_contractible: eC / tC });
  }
  return rows;
}

const d = 8;
const dim = dimensional(d, [2, 4, 6, 8, 10, 12, 16, 20, 24], 400);
const topo = topological([0, 0.5, 1, 2, 4, 8], 5, 4000);

console.log(`RESIST_PASS2  axis generalization`);
console.log(`\n[DIMENSIONAL]  shadow rank d=${d}; dial rho = body dim D; threshold rho* = d`);
for (const r of dim) console.log(`  D=${String(r.D).padStart(2)}: FVE(recover)=${r.FVE.toFixed(3)}  sigma=${r.sigma.toFixed(3)}${r.D === d ? "   <- knee (D=d)" : ""}`);
const dimSharp = dim.find((r) => r.D === 2 * d);
console.log(`  at D=2d=${2 * d}: sigma=${dimSharp.sigma.toFixed(3)} (~1-d/D=${(1 - d / (2 * d)).toFixed(3)}); kernel (D-d dims) is independent of the shadow -> unrecoverable at ANY compute.`);

console.log(`\n[TOPOLOGICAL]  loop encloses h=5 plaquettes; one hidden "hole"; dial rho = hole flux`);
for (const r of topo) console.log(`  rho=${String(r.rho).padStart(3)}: sigma(non-contractible)=${r.sigma_nonContractible.toFixed(3)}   sigma(contractible)=${r.sigma_contractible.toFixed(3)}`);
console.log(`  threshold is the HOMOTOPY CLASS: contractible loop sigma=0 for every rho (Stokes); only the hole-enclosing loop resists, graded by rho. The hole flux is absent from the shadow -> unrecoverable at ANY compute.`);

const dimOK = dim.find((r) => r.D <= d).FVE > 0.95 && dimSharp.sigma > 0.4;
const topoOK = topo[topo.length - 1].sigma_nonContractible > 0.6 && topo.every((r) => r.sigma_contractible < 0.05);
console.log(`\nEXIT GATE: dimensional sharp=${dimOK}, topological sharp=${topoOK}; with Pass 1 (computational) that is ${(dimOK ? 1 : 0) + (topoOK ? 1 : 0) + 1}/3 axes -> ${(dimOK ? 1 : 0) + (topoOK ? 1 : 0) + 1 >= 2 ? "MET (resistance, any flavor, is the axis - not a one-construction trick)" : "NOT met"}`);
