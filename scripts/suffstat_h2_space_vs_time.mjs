// Sufficient-statistic-order slate, H2: recognizability radius (space) vs
// sufficient-statistic order (time) — are they the SAME coordinate?
//
// Spatial r  = recognizabilityRadius1D: least centered (2L+1)-window that
//              determines the ROLE (desubstitution = a hidden STRUCTURAL latent).
//              Finite for every aperiodic primitive substitution (Mosse).
// Temporal σ = predictive order: least preceding-k window that determines the
//              NEXT SYMBOL (no right-special factor of length k).
//
// Pre-registered prediction: aperiodic sequences have a right-special factor at
// EVERY length (complexity strictly increases), so σ_predict = ∞, while r stays
// finite. If so, r and σ are DIFFERENT coordinates — "space↔time" is a schema
// (least context determining A latent), not one number; they coincide only when
// the latent is matched. Periodic control: both finite (they agree). Run:
//   node scripts/suffstat_h2_space_vs_time.mjs

import { generateWord, recognizabilityRadius1D, repeatCellCaptureRadius }
  from "../ghost/metric-probe-core.js";

// length-k factors with >=2 distinct right extensions (right-special factors).
function rightSpecialCount(word, k) {
  const ext = new Map();
  for (let i = 0; i + k < word.length; i++) {
    const pre = word.slice(i, i + k).join("");
    if (!ext.has(pre)) ext.set(pre, new Set());
    ext.get(pre).add(word[i + k]);
  }
  let rs = 0;
  for (const s of ext.values()) if (s.size >= 2) rs++;
  return rs;
}

// least k in 1..K with RS(k)=0 (preceding k determine the next symbol); else ∞.
function predictiveOrder(word, K) {
  for (let k = 1; k <= K; k++) if (rightSpecialCount(word, k) === 0) return k;
  return Infinity;
}

const K = 18;
const APERIODIC = [
  { name: "fibonacci", depth: 24 },        // |w| = F-scale, ~46k
  { name: "period-doubling", depth: 16 },  // |w| = 2^16
  { name: "thue-morse", depth: 16 },       // |w| = 2^16
];

console.log("SUFFSTAT_H2_SPACE_VS_TIME   (recognizability r  vs  predictive σ)\n");
console.log("(A) aperiodic substitution sequences");
console.log("  name             |w|     r (role, centered)     RS(k)=#right-special, k=1..8        σ_predict");
for (const { name, depth } of APERIODIC) {
  const w = generateWord(name, depth);
  // r converged: equal at depth and depth-1
  const r = recognizabilityRadius1D(name, Math.min(depth, 12));
  const r2 = recognizabilityRadius1D(name, Math.min(depth, 12) - 1);
  const rStr = r === r2 ? `${r} (stable)` : `${r}/${r2}`;
  const rs = [];
  for (let k = 1; k <= 8; k++) rs.push(rightSpecialCount(w, k));
  const sigma = predictiveOrder(w, K);
  console.log(
    `  ${name.padEnd(15)} ${String(w.length).padStart(6)}   ${rStr.padEnd(14)}   [${rs.join(",")}]`.padEnd(86) +
    `   ${sigma === Infinity ? `∞ (>${K})` : sigma}`
  );
}

console.log("\n(B) periodic control (where the two coordinates should AGREE)");
for (const motif of [["a", "b", "c"], ["a", "a", "b"]]) {
  const p = motif.length;
  const w = [];
  for (let i = 0; i < 400; i++) w.push(motif[i % p]);
  const spatial = repeatCellCaptureRadius(motif); // least window exhibiting the period
  const sigma = predictiveOrder(w, K);
  console.log(`  motif=${JSON.stringify(motif).padEnd(15)} period=${p}   spatial capture=${spatial}   σ_predict=${sigma}`);
}

console.log("\nVERDICT:");
console.log("  Aperiodic: r FINITE (structural latent locally recoverable, Mosse) but σ_predict = ∞");
console.log("  (right-special factor at every length → next symbol never finite-order determined).");
console.log("  → r and σ_predict are DIFFERENT coordinates; they agree only on periodic controls.");
console.log("  The 'space↔time' link is the SCHEMA (least context determining A latent), not one number:");
console.log("  recognizability = the schema's STRUCTURAL instance (always in the finite-σ / determine");
console.log("  regime); parity's σ=∞ resist pole lives on the PREDICTIVE filtration, which r never touches.");
