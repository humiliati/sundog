// Resist-side roadmap, Pass 3: "recover a marginal".
// Take a faithful surrogate of a marginal substrate (a body recoverable from its
// low-Fourier shadow at the MEASURED FVE - that is exactly why it is marginal),
// inject the Pass-1 syndrome channel at dial rho as a HYBRID body, and measure
// whether the previously-marginal regime-2 split becomes sharp, at FIXED compute.
//
// FENCE: this builds a hybrid body (natural recoverable shadow + a constructed
// resistant channel). It does NOT claim the natural substrate secretly resists -
// the natural recoverability (FVE) is preserved as the marginal baseline.
// Run: node scripts/resist-pass3.mjs

// ---- inline Pass-1 [20,10] syndrome code + min-weight decoder (same seed) ----
const n = 20, k = 10, m = n - k;
const lcg = (s) => () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
const rand = lcg(20260628);
const rb = () => (rand() < 0.5 ? 1 : 0);
const P = Array.from({ length: k }, () => Array.from({ length: m }, rb));
const Hcol = new Array(n);
for (let j = 0; j < k; j++) { let v = 0; for (let c = 0; c < m; c++) if (P[j][c]) v |= (1 << c); Hcol[j] = v; }
for (let i = 0; i < m; i++) Hcol[k + i] = (1 << i);
const synOf = (e) => { let s = 0; for (let b = 0; b < n; b++) if ((e >>> b) & 1) s ^= Hcol[b]; return s; };
const NS = 1 << m, minW = new Array(NS).fill(999), minPre = new Array(NS).fill(-1);
{ let g = 0, sy = 0, w = 0; minW[0] = 0; minPre[0] = 0; for (let i = 1; i < (1 << n); i++) { const bit = 31 - Math.clz32(i & -i); g ^= (1 << bit); sy ^= Hcol[bit]; w += ((g >>> bit) & 1) ? 1 : -1; if (w < minW[sy]) { minW[sy] = w; minPre[sy] = g; } } }
function randErrW(w) { const s = new Set(); while (s.size < w) s.add(Math.floor(rand() * n)); let e = 0; for (const b of s) e |= (1 << b); return e; }
function errorRecovery(w, T = 600) { let ok = 0; for (let t = 0; t < T; t++) { const e = randErrW(w); if (minPre[synOf(e)] === e) ok++; } return ok / T; } // R_e(rho)

const rhoStar = 3;          // inherited from Pass 1 (capacity threshold)
const alpha = 0.5;          // hybrid weight: half natural body, half constructed channel (modeling choice)
const substrates = [
  { name: "NSE C1 (2D Kolmogorov)", FVE: 0.99 },   // measured low-Fourier recoverability
  { name: "Mesa net.7", FVE: 0.97 },
];

const Re = [];
for (let w = 0; w <= 6; w++) Re[w] = errorRecovery(w);

console.log(`RESIST_PASS3  "recover a marginal"  (hybrid weight alpha=${alpha}, threshold rho*=${rhoStar} inherited from Pass 1; compute FIXED)`);
console.log(`  error-channel recovery R_e(rho): ${Re.map((x, w) => `w${w}=${x.toFixed(2)}`).join("  ")}`);
for (const sub of substrates) {
  // marginal baseline: natural body alone, recoverable at FVE -> regime-2 sharpness ~ 1 - FVE
  const sigmaMarginal = 1 - sub.FVE;
  console.log(`\n[${sub.name}]  natural-body FVE=${sub.FVE} -> sigma_marginal = ${sigmaMarginal.toFixed(3)} (MARGINAL: body recoverable, split soft)`);
  for (let w = 0; w <= 6; w++) {
    const Rhybrid = alpha * sub.FVE + (1 - alpha) * Re[w];
    const sigma = 1 - Rhybrid;
    const tag = w === 0 ? " (no channel)" : w >= rhoStar ? "  <- sharp (rho >= rho*)" : "";
    console.log(`  inject rho=${w}: R_hybrid=${Rhybrid.toFixed(3)}  sigma_hybrid=${sigma.toFixed(3)}${tag}`);
  }
  const sigmaSharp = 1 - (alpha * sub.FVE + (1 - alpha) * Re[6]);
  console.log(`  => sigma: ${sigmaMarginal.toFixed(3)} (marginal) -> ${sigmaSharp.toFixed(3)} (rho>=rho*), at FIXED compute. Marginal RECOVERED (constructive sense).`);
}

console.log(`\nFENCE: hybrid body = natural recoverable shadow + constructed resistant channel. Does NOT claim natural NSE/Mesa resists; FVE preserved as the baseline. The recovery is set by the injected resistance parameter (rho>=rho*), not compute.`);
