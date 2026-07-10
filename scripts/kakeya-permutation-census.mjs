#!/usr/bin/env node
// PHASE3T - permutation-slice census for the floor's hard regime.
//
// Sub-conjecture (PHASE3S, extreme case): f a permutation of F_q with a
// 4-point fiber in some translate f - s*id  =>  sum_s sigma_s >= q - 2.
// For q <= 17 this is a COROLLARY of the proved floor (exhaustive at q<=7;
// all-orbit exact B&B + PGL transitivity at 11/13/17). This census therefore
// hunts STRUCTURE, not new bounds:
//   (T1) exhaust ALL permutations at q in {7, 11} (5040 / 39,916,800 via
//        Heap's algorithm with incremental fiber stats): min sum-sigma among
//        4-fiber permutations, tightness vs q-2, extremal count;
//   (T2) interpolate extremal permutations as polynomials (Lagrange):
//        degree spectrum = the Hermite fingerprints for the analytic attack;
//   (T3) regime census over ALL functions at q=7: for every (f,d) achieving
//        sacrifice = q-2 with a 4-secant, histogram N_d - does the tight
//        case live at N_d = q (permutation-translate) or spread?
//   (T4) sampled check at q=13 (floor proved there too; consistency).
// Falsifier is instrument-only: any 4-fiber permutation with sum-sigma < q-2
// at q <= 17 would contradict proved results (= a bug), as would
// inconsistency with the PHASE3S sweep. No Euclidean claim.

import fs from "node:fs";
import path from "node:path";

const DEFAULT_OUT = path.join("results", "kakeya", "permutation-census");

function mod(x, q) { return ((x % q) + q) % q; }
function w(m) { return m >= 3 ? ((m - 1) * (m - 2)) / 2 : 0; }

function mulberry(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- incremental translate-stat tracker for permutations ---------------------------
// Tracks, for s = 1..q-1 (s=0 is bijective for permutations, no content):
// cnt[s][c], sigma_s, sumSigma, and the number of cells with m >= 4 (has4)
// plus max fiber overall. Single-position updates in O(q) per swap.
function makeTracker(q) {
  const cnt = [];
  for (let s = 1; s < q; s++) cnt.push(new Uint8Array(q));
  return {
    q, cnt, sumSigma: 0, cells4: 0,
    reset(f) {
      this.sumSigma = 0; this.cells4 = 0;
      for (let s = 1; s < q; s++) {
        const row = this.cnt[s - 1];
        row.fill(0);
        for (let a = 0; a < q; a++) row[mod(f[a] - s * a, q)]++;
        for (let c = 0; c < q; c++) {
          this.sumSigma += w(row[c]);
          if (row[c] >= 4) this.cells4++;
        }
      }
    },
    // remove point (a, y) then later add point (a, y')
    removePoint(a, y) {
      for (let s = 1; s < this.q; s++) {
        const row = this.cnt[s - 1];
        const c = mod(y - s * a, this.q);
        const m = row[c];
        this.sumSigma += w(m - 1) - w(m);
        if (m === 4) this.cells4--;
        row[c] = m - 1;
      }
    },
    addPoint(a, y) {
      for (let s = 1; s < this.q; s++) {
        const row = this.cnt[s - 1];
        const c = mod(y - s * a, this.q);
        const m = row[c];
        this.sumSigma += w(m + 1) - w(m);
        if (m === 3) this.cells4++;
        row[c] = m + 1;
      }
    },
  };
}

// --- T1: exhaust permutations via Heap's algorithm ---------------------------------
function exhaustPermutations(q, keepExtremals) {
  const f = Array.from({ length: q }, (_, i) => i);
  const tr = makeTracker(q);
  tr.reset(f);
  const target = q - 2;
  let perms = 0, with4 = 0, minSigma = Infinity, extremalCount = 0, below = 0;
  const extremals = [];
  const sigmaHist = new Map();

  const record = () => {
    perms++;
    if (tr.cells4 > 0) {
      with4++;
      const ss = tr.sumSigma;
      sigmaHist.set(ss, (sigmaHist.get(ss) ?? 0) + 1);
      if (ss < target) below++;
      if (ss < minSigma) { minSigma = ss; extremals.length = 0; extremalCount = 0; }
      if (ss === minSigma) { extremalCount++; if (extremals.length < keepExtremals) extremals.push([...f]); }
    }
  };

  // Enumerator self-check: at small q, Heap's must visit each permutation
  // exactly once (the v1 run had the even/odd swap convention REVERSED,
  // producing a multiset - caught via duplicate extremal samples).
  const distinct = q <= 8 ? new Set([f.join(",")]) : null;

  record();
  // iterative Heap's algorithm with incremental swap updates
  const c = new Array(q).fill(0);
  let i = 0;
  while (i < q) {
    if (c[i] < i) {
      const j = i % 2 === 0 ? 0 : c[i];
      // swap f[j] <-> f[i] incrementally
      const yj = f[j], yi = f[i];
      tr.removePoint(j, yj); tr.removePoint(i, yi);
      f[j] = yi; f[i] = yj;
      tr.addPoint(j, yi); tr.addPoint(i, yj);
      if (distinct) distinct.add(f.join(","));
      record();
      c[i]++; i = 0;
    } else { c[i] = 0; i++; }
  }
  const factorial = Array.from({ length: q }, (_, k) => k + 1).reduce((x, y) => x * y, 1);
  const enumeratorValid = distinct ? distinct.size === factorial && perms === factorial : perms === factorial;
  return { q, perms, factorial, enumeratorValid, with4, minSigma, target, tight: minSigma === target, belowTarget: below, extremalCount, extremals, sigmaHist: [...sigmaHist.entries()].sort((a, b) => a[0] - b[0]).slice(0, 10) };
}

// --- T2: Lagrange interpolation degree of a function --------------------------------
function inv(x, q) { for (let i = 1; i < q; i++) if ((x * i) % q === 1) return i; throw new Error("noinv"); }
function polyDegree(f, q) {
  // coefficients via Lagrange: sum_a f(a) * L_a(x); compute in O(q^2)
  const coeff = new Array(q).fill(0);
  for (let a = 0; a < q; a++) {
    if (f[a] === 0) continue;
    // L_a(x) = prod_{b != a} (x - b) / (a - b)
    let poly = [1]; // constant 1
    let denom = 1;
    for (let b = 0; b < q; b++) {
      if (b === a) continue;
      const next = new Array(poly.length + 1).fill(0);
      for (let k = 0; k < poly.length; k++) {
        next[k] = mod(next[k] - poly[k] * b, q);
        next[k + 1] = mod(next[k + 1] + poly[k], q);
      }
      poly = next;
      denom = mod(denom * (a - b), q);
    }
    const scale = mod(f[a] * inv(denom, q), q);
    for (let k = 0; k < poly.length; k++) coeff[k] = mod(coeff[k] + poly[k] * scale, q);
  }
  let deg = -1;
  for (let k = q - 1; k >= 0; k--) if (coeff[k] !== 0) { deg = k; break; }
  return { deg, coeff };
}

// --- T3: regime census over all functions at q=7 ------------------------------------
function regimeCensus(q) {
  const f = new Array(q).fill(0);
  const NdHist = new Map(); // N_d at floor-achieving (f,d) with 4-secant
  let achievers = 0;
  for (;;) {
    // fibers
    const Ns = new Array(q).fill(0);
    const fibers = [];
    let sumN = 0;
    for (let s = 0; s < q; s++) {
      const cntRow = new Array(q).fill(0);
      for (let a = 0; a < q; a++) cntRow[mod(f[a] - s * a, q)]++;
      let N = 0;
      for (let cc = 0; cc < q; cc++) if (cntRow[cc] >= 1) N++;
      Ns[s] = N; sumN += N; fibers.push(cntRow);
    }
    for (let d = 0; d < q; d++) {
      const sac = sumN - Ns[d] - (q * (q - 1)) / 2;
      if (sac !== q - 2) continue;
      // 4-secant present?
      let h4 = false;
      for (let s = 0; s < q && !h4; s++) {
        const need = s === d ? 3 : 4;
        for (let cc = 0; cc < q; cc++) if (fibers[s][cc] >= need) { h4 = true; break; }
      }
      if (h4) { achievers++; NdHist.set(Ns[d], (NdHist.get(Ns[d]) ?? 0) + 1); }
    }
    let i = 0;
    while (i < q) { f[i]++; if (f[i] < q) break; f[i] = 0; i++; }
    if (i === q) break;
  }
  return { q, achievers, NdHist: [...NdHist.entries()].sort((a, b) => a[0] - b[0]) };
}

// --- T4: sampled permutations at q=13 ------------------------------------------------
function sampledPerms(q, samples, seed) {
  const rng = mulberry(seed);
  const tr = makeTracker(q);
  let with4 = 0, minSigma = Infinity;
  const f = Array.from({ length: q }, (_, i) => i);
  for (let t = 0; t < samples; t++) {
    // Fisher-Yates
    for (let i = q - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [f[i], f[j]] = [f[j], f[i]];
    }
    tr.reset(f);
    if (tr.cells4 > 0) { with4++; if (tr.sumSigma < minSigma) minSigma = tr.sumSigma; }
  }
  return { q, samples, with4, minSigma, target: q - 2 };
}

function main() {
  const outDir = process.argv.includes("--out") ? process.argv[process.argv.indexOf("--out") + 1] : DEFAULT_OUT;
  const t0 = Date.now();

  console.log("== T1 exhaust q=7 permutations ==");
  const t1a = exhaustPermutations(7, 60);
  console.log(`  perms=${t1a.perms} with4fiber=${t1a.with4} minSigma=${t1a.minSigma} target=${t1a.target} tight=${t1a.tight} below=${t1a.belowTarget} extremals=${t1a.extremalCount}`);
  console.log(`  sigma histogram (low end): ${t1a.sigmaHist.map(([k, v]) => `${k}:${v}`).join(" ")}`);

  console.log("== T2 extremal permutation degrees (q=7) ==");
  const degHist = new Map();
  const samplesOut = [];
  for (const f of t1a.extremals) {
    const { deg } = polyDegree(f, 7);
    degHist.set(deg, (degHist.get(deg) ?? 0) + 1);
    if (samplesOut.length < 6) samplesOut.push({ f: [...f], deg });
  }
  console.log(`  degree histogram over ${t1a.extremals.length} stored extremals: ${[...degHist.entries()].sort((a, b) => a[0] - b[0]).map(([d, n]) => `deg${d}:${n}`).join(" ")}`);
  for (const s of samplesOut) console.log(`    f=[${s.f}] deg=${s.deg}`);

  console.log("== T3 regime census over all f at q=7 (floor-achievers with 4-secant) ==");
  const t3 = regimeCensus(7);
  console.log(`  achievers=${t3.achievers}  N_d histogram: ${t3.NdHist.map(([n, c]) => `N_d=${n}:${c}`).join(" ")}`);

  console.log("== T1b exhaust q=11 permutations (39.9M, incremental) ==");
  const t1b = exhaustPermutations(11, 20);
  console.log(`  perms=${t1b.perms} with4fiber=${t1b.with4} minSigma=${t1b.minSigma} target=${t1b.target} tight=${t1b.tight} below=${t1b.belowTarget} extremals=${t1b.extremalCount}`);
  const degHist11 = new Map();
  for (const f of t1b.extremals) { const { deg } = polyDegree(f, 11); degHist11.set(deg, (degHist11.get(deg) ?? 0) + 1); }
  console.log(`  extremal degree histogram (stored ${t1b.extremals.length}): ${[...degHist11.entries()].sort((a, b) => a[0] - b[0]).map(([d, n]) => `deg${d}:${n}`).join(" ")}`);

  console.log("== T4 sampled q=13 permutations ==");
  const t4 = sampledPerms(13, 2_000_000, 1234567);
  console.log(`  samples=${t4.samples} with4fiber=${t4.with4} minSigmaFound=${t4.minSigma} target=${t4.target}`);

  // Instrument checks: nothing below target at q in {7,11} (proved floors);
  // q=13 sample also must respect the proved floor.
  const pass = t1a.belowTarget === 0 && t1b.belowTarget === 0 && t4.minSigma >= t4.target;
  const manifest = {
    artifactId: "KAK-PHASE3T-PERMUTATION-CENSUS",
    generatedAt: new Date().toISOString(),
    status: "internal extremal-structure census (permutation slice of the floor's hard regime)",
    command: "node scripts/kakeya-permutation-census.mjs",
    note: "For q <= 17 the permutation sub-conjecture is a corollary of the proved floor; this census maps tightness, extremal structure (polynomial degrees), and the N_d regime of floor-achievers. q=19 floor remains open for generic orbits (only harmonic/eq solved exactly there).",
    T1_q7: { ...t1a, extremals: t1a.extremals.slice(0, 10) },
    T2_q7_degrees: [...degHist.entries()],
    T3_regime: t3,
    T1b_q11: { ...t1b, extremals: t1b.extremals.slice(0, 6) },
    T2b_q11_degrees: [...degHist11.entries()],
    T4_q13_sampled: t4,
    falsifier: {
      name: "PERMUTATION_CENSUS_MISMATCH",
      description: "Instrument-only: fires if any 4-fiber permutation at q in {7,11,13-sampled} has sum-sigma below q-2 (contradicting the proved floor) - that would be a bug, not a discovery.",
      status: pass ? "clear" : "fired",
    },
    elapsedSeconds: (Date.now() - t0) / 1000,
  };
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
  console.log(`KAK_PERMUTATION_CENSUS q7tight=${t1a.tight} q11tight=${t1b.tight} falsifier=${manifest.falsifier.status} elapsed=${manifest.elapsedSeconds.toFixed(1)}s`);
  process.exit(pass ? 0 : 1);
}
main();
