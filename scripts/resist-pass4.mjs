// Resist-side roadmap, Pass 4 (optional, non-gating): natural-resistance search.
// CHEAP pre-checks only (no simulation; explicitly NOT the compute-bound Sabra
// measurement wall). Question: does a NATURAL (unconstructed) body cross to the
// resist side as a physical parameter grows? Test the obvious candidate -
// high-Reynolds turbulence - by Kolmogorov mode-counting on two axes:
//   ENERGY recoverability (does the low-Fourier shadow capture the body's energy?)
//   DEGREES-OF-FREEDOM count (does the body outgrow the fixed shadow rank?)
// Run: node scripts/resist-pass4.mjs

const K = 8; // fixed low-Fourier shadow rank (fixed compute)
const kd = (Re, dim) => Math.max(K, Math.round(Re ** (dim === 3 ? 0.75 : 0.5))); // dissipation wavenumber
const energyFVE = (Kc, kdc, p) => { // (energy in k<=K) / (total energy), E(k) ~ k^-p
  let lo = 0, tot = 0;
  for (let k = 1; k <= kdc; k++) { const e = k ** (-p); tot += e; if (k <= Kc) lo += e; }
  return lo / tot;
};

console.log(`RESIST_PASS4  natural-resistance search (cheap pre-checks; shadow rank K=${K} fixed)`);

for (const [label, dim, p] of [["3D (E~k^-5/3)", 3, 5 / 3], ["2D / NSE C1 (E~k^-3)", 2, 3]]) {
  console.log(`\n[${label}]`);
  console.log(`  Re        k_d    energy-FVE(<=K)   DOF beyond K = (k_d-K)/k_d`);
  for (const Re of [50, 200, 1000, 10000, 100000]) {
    const d = kd(Re, dim);
    const fve = energyFVE(K, d, p);
    const dof = (d - K) / d;
    console.log(`  ${String(Re).padStart(7)}  ${String(d).padStart(5)}     ${fve.toFixed(3)}            ${dof.toFixed(3)}`);
  }
}

console.log(`\nREAD:`);
console.log(`  - ENERGY stays large-scale: energy-FVE stays HIGH and ~flat as Re grows (the low-Fourier`);
console.log(`    shadow keeps capturing the body's ENERGY). So the naive "high-Re => natural dimensional`);
console.log(`    resistance by energy" is REFUTED by the cheap pre-check - this is why NSE C1 was marginal.`);
console.log(`  - DEGREES OF FREEDOM diverge: the fraction of modes beyond the shadow -> 1 with Re (the body`);
console.log(`    outgrows the fixed shadow in COUNT). But those extra modes are LOW-ENERGY (dissipation /`);
console.log(`    intermittency). Whether they are CONTROL-relevant (=> a real regime-2 split) is exactly`);
console.log(`    the C2 / Sabra intermittency target - the compute-gated measurement wall, NOT cheap.`);
console.log(`\nVERDICT (both exit branches, honest):`);
console.log(`  * Natural HIGH-DIMENSIONAL resistance stays UNREACHED at cheap assessment: the obvious`);
console.log(`    energy candidate fails; the real candidate (low-energy small-scale intermittency) is`);
console.log(`    compute-gated (Sabra wall) and is FLAGGED for a later compute-gated measurement.`);
console.log(`  * One natural EXACT resistance is already in hand: Aharonov-Bohm (topological), but it is`);
console.log(`    LOW-dimensional (one integer per H^1 generator) - does not close the high-dim frontier.`);
console.log(`  * Computational axis: no clean NATURAL example outside constructed codes/crypto.`);
console.log(`  => CONSTRUCTED resistance is the demonstrated path (Passes 0-3). Pass 4 does NOT claim a`);
console.log(`     natural high-stakes body resists; it locates where natural resistance must live`);
console.log(`     (low-energy / high-DOF structure) and confirms that is compute-gated, not cheap.`);
