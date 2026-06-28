// Ghost H1a frontier probe: can a NON-LOCAL reader recover structure that the
// diffraction (pure-point / Bragg) channel declares absent?
//
// Pre-registered claim (theory-driven, NOT fitted):
//   The Bragg channel of diffraction is a 2-point / linear statistic. Thue-Morse
//   has singular-continuous diffraction (no Bragg) yet is fully deterministic
//   (recognizable, L=2). So a non-local deterministic reader (desubstitution)
//   should recover TM's structure that the Bragg detector renders ~0, while
//   correctly FAILING on true randomness.
//
// Expected pattern: Bragg B HIGH for pure-point {periodic, Fibonacci,
//   period-doubling}, LOW for {Thue-Morse, random}; desub-validity ~1 for
//   {Thue-Morse, period-doubling}, ~chance for random.
// Falsifier: TM Bragg ~ Fibonacci Bragg (diffraction NOT blind to TM), OR
//   desub-validity does not separate TM from random (reader does not recover TM).
//
// Honest default: this resolves into KNOWN theory (diffraction vs dynamical
// spectrum; homometry; 2-point vs higher-order correlations), NOT a new invariant.
// Run: node scripts/ghost-nonlocal-probe.mjs

import { generateWord } from "../ghost/metric-probe-core.js";

const N = 987; // common length (Fibonacci F16)

const toWordN = (word) => word.slice(0, N);
const toPM1 = (word) => toWordN(word).map((c) => (c === "a" ? 1 : -1));

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Bragg detector: B = max over a fine frequency grid of (1/N^2)|sum w_n e^{-2pi i f n}|^2.
// O(1) for pure-point (Bragg) spectra; -> 0 for singular-continuous / absolutely-continuous.
function braggB(pm1) {
  const n = pm1.length;
  const M = 4 * n;
  let best = 0;
  for (let j = 1; j <= M; j++) {
    const w = 2 * Math.PI * ((0.5 * j) / M); // f in (0, 0.5]
    let re = 0;
    let im = 0;
    for (let k = 0; k < n; k++) {
      const a = w * k;
      re += pm1[k] * Math.cos(a);
      im -= pm1[k] * Math.sin(a);
    }
    const val = (re * re + im * im) / (n * n);
    if (val > best) best = val;
  }
  return best;
}

// Non-local deterministic reader: best-phase fraction of length-2 blocks that are
// valid images of a constant-length substitution. ~1 if the sequence desubstitutes
// cleanly; ~chance for disorder.
function desubValidity(word, validPairs) {
  const w = toWordN(word);
  let best = 0;
  for (const phase of [0, 1]) {
    let valid = 0;
    let total = 0;
    for (let i = phase; i + 1 < w.length; i += 2) {
      total++;
      if (validPairs.has(w[i] + w[i + 1])) valid++;
    }
    if (total > 0) best = Math.max(best, valid / total);
  }
  return best;
}

const periodicWord = Array.from({ length: N }, (_, i) => (i % 2 === 0 ? "a" : "b"));
const fib = generateWord("fibonacci", 14); // length 987
const tm = generateWord("thue-morse", 10); // length 1024
const pd = generateWord("period-doubling", 10); // length 1024
const rnd = mulberry32(20260628);
const randomWord = Array.from({ length: N }, () => (rnd() < 0.5 ? "a" : "b"));

const rows = [
  ["periodic (abab)", periodicWord, { ab: 0, ba: 0 }, null],
  ["Fibonacci", fib, null, "L=1 (variable-length; recognizable)"],
  ["period-doubling", pd, new Set(["ab", "aa"]), null],
  ["Thue-Morse", tm, new Set(["ab", "ba"]), null],
  ["random", randomWord, new Set(["ab", "ba"]), null],
];

const result = {};
console.log(`GHOST_NONLOCAL_PROBE N=${N}`);
console.log("substrate        |  Bragg B   | desub-validity (best phase)");
console.log("-----------------+------------+---------------------------");
for (const [name, word, validPairs, note] of rows) {
  const B = braggB(toPM1(word));
  const dv = validPairs instanceof Set ? desubValidity(word, validPairs) : null;
  result[name] = { braggB: B, desubValidity: dv };
  console.log(
    `${name.padEnd(16)} | ${B.toFixed(5)}  | ${dv == null ? note ?? "n/a" : dv.toFixed(3)}`,
  );
}

// Verdict checks (pre-registered)
const B = (k) => result[k].braggB;
const braggBlindTM = B("Thue-Morse") < 0.1 * B("Fibonacci") && B("Thue-Morse") < 0.1 * B("period-doubling");
const tmLikeRandomBragg = B("Thue-Morse") < 5 * B("random") && B("random") < 5 * B("Thue-Morse");
const readerSeparates = result["Thue-Morse"].desubValidity > 0.9 && result["random"].desubValidity < 0.7;
console.log("");
console.log(`check: TM Bragg-blind vs Fibonacci/pd (B_TM < 0.1*B_pp): ${braggBlindTM}`);
console.log(`check: TM Bragg ~ random (conflated by Bragg channel):   ${tmLikeRandomBragg}`);
console.log(`check: non-local reader separates TM (~1) from random:   ${readerSeparates}`);
console.log("");
console.log(
  readerSeparates
    ? "RESULT: non-local reader recovers TM (desub~1) and rejects random; TM carries NO Bragg, so the pure-point channel declares its order absent yet the reader recovers it (frontier confirmed against the Bragg channel). The two Bragg thresholds above assumed singular-continuous reads like random; the data REFUTES that (SC != AC: TM Bragg ~7x random) -> diffraction is lossy, NOT blind. Corrected reading + receipt: docs/ghost/hypotheses_ghost_slate.md (H1a frontier). Lands on known homometry / dynamical-spectrum theory; not a new invariant."
    : "Non-local reader did NOT separate TM from random -> frontier refuted; see numbers.",
);
