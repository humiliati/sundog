// Resist-side roadmap, Pass 0 anchor: the "one reconstruction metric".
// Demonstrates the RESISTANT end of the recoverability axis on a tiny GF(2)
// syndrome instance: the same secret is (largely) recoverable from the full BODY
// but ~chance from the lossy SHADOW (the syndrome). This anchors the
// reconstruction-accuracy axis [chance=0.5 .. perfect=1.0] on which the marginal
// substrates (NSE C1, Mesa net.7) sit at 0.97-0.99 (recoverable end).
// Theory: z = y H^T = (sG+e) H^T = e H^T  (since G H^T = 0), so the syndrome
// depends ONLY on e, not on s -> the secret is information-theoretically absent
// from the shadow, hence exactly chance to reconstruct. The computation confirms it.
// Run: node scripts/resist-pass0-anchor.mjs

const k = 8, n = 16, m = n - k; // [16,8] systematic code: G=[I_k|P], H=[P^T|I_m]
function lcg(seed) { return () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }; }
const rand = lcg(20260628);
const randbit = () => (rand() < 0.5 ? 1 : 0);

const P = Array.from({ length: k }, () => Array.from({ length: m }, randbit));
function encode(s) {
  const c = new Array(n).fill(0);
  for (let j = 0; j < k; j++) c[j] = s[j];
  for (let col = 0; col < m; col++) { let a = 0; for (let i = 0; i < k; i++) a ^= s[i] & P[i][col]; c[k + col] = a; }
  return c;
}
function syndrome(y) { // z = y H^T, H = [P^T | I_m]
  const z = new Array(m).fill(0);
  for (let col = 0; col < m; col++) { let a = 0; for (let i = 0; i < k; i++) a ^= y[i] & P[i][col]; z[col] = a ^ y[k + col]; }
  return z;
}
function randErr(w) { const e = new Array(n).fill(0); let p = 0; while (p < w) { const i = Math.floor(rand() * n); if (!e[i]) { e[i] = 1; p++; } } return e; }

const N = 6000, w = 1;
const S = [], Z = [], Yk = [];
for (let t = 0; t < N; t++) {
  const s = Array.from({ length: k }, randbit);
  const c = encode(s);
  const e = randErr(w);
  const y = c.map((b, i) => b ^ e[i]);
  S.push(s); Z.push(syndrome(y)); Yk.push(y.slice(0, k)); // body read-off = first k bits of y; shadow = syndrome z
}

// acc(secret | full body): predict s_i by y[:k][i]
let accBody = 0;
for (let i = 0; i < k; i++) { let ok = 0; for (let t = 0; t < N; t++) if (Yk[t][i] === S[t][i]) ok++; accBody += ok / N; }
accBody /= k;

// acc(secret | lossy shadow z): best single z-bit predictor per secret bit (upper bound; best-of-m inflates slightly)
let accShadow = 0;
for (let i = 0; i < k; i++) {
  let best = 0.5;
  for (let j = 0; j < m; j++) { let agree = 0; for (let t = 0; t < N; t++) if (Z[t][j] === S[t][i]) agree++; const a = agree / N; best = Math.max(best, a, 1 - a); }
  accShadow += best;
}
accShadow /= k;

console.log(`RESIST_PASS0_ANCHOR  [${n},${k}] systematic syndrome, N=${N}, err weight=${w}`);
console.log(`  acc(secret | full BODY y[:k]) = ${accBody.toFixed(3)}   (recoverable end of the axis)`);
console.log(`  acc(secret | lossy SHADOW z)  = ${accShadow.toFixed(3)}   (resistant end; chance = 0.5, best-of-${m} predictor)`);
console.log(`  theory: z = e H^T is independent of s, so shadow-reconstruction of the secret is EXACTLY chance.`);
console.log(`  => same secret, ~${accBody.toFixed(2)} from the body vs ~chance from the shadow: the shadow drops the body.`);
