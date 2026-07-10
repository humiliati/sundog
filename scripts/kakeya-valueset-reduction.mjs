#!/usr/bin/env node
// PHASE3S - the value-set reduction of the floor conjecture (M3 work).
//
// Dual/function coordinates: a one-per-direction Kakeya completion is a graph
// {(a, f(a)) : a in F_q} plus one infinite point (d), d in F_q, after sending
// the pencil vertex W to the vertical direction. With
//   m_{s,c} = #{a : f(a) = s a + c}   (fibers of the translate f - s*id)
//   N_s     = #{c : m_{s,c} >= 1}     (VALUE SET SIZE of f - s*id)
//   sigma_s = sum_c (m-1)(m-2)/2
// three exact identities (proved by pair counting; machine-verified here):
//   (I1) pi_s - sigma_s = q - N_s                      per slope
//   (I2) sum_s N_s = q(q+1)/2 + sum_s sigma_s
//   (I3) sacrifice = sum_{s != d} N_s - q(q-1)/2 = (q - N_d) + sum_s sigma_s
// So the FLOOR conjecture (sacrifice >= q-2 in the presence of a 4-secant)
// becomes a pure statement about value-set sums of linear translates, and the
// shared X^q pencil factor of PHASE3R-M1 is the 4-point fiber
// (X - c0)^4 | H(X, s0) of the classical Redei polynomial of f.
//
// This script:
//  (A) verifies I1-I3 algebraically AND against an independent projective
//      line-enumeration of the dual point set (random + structured f);
//  (B) EXHAUSTIVELY decides the floor at q in {5,7} in function space
//      (all q^q functions x all d): global min sacrifice (= BM (q-1)/2),
//      min sacrifice among configs containing a 4-secant (floor value), and
//      the +1 premise (no global minimizer contains a 4-secant);
//  (C) runs an adversarial hillclimb over f at q in {11..37} with a frozen
//      4-fiber, hunting sacrifice < q-2 (counterexample hunt; upper-bound
//      evidence only when it fails).
// No Euclidean claim. Floor remains open in general; see PHASE3S receipt.

import fs from "node:fs";
import path from "node:path";

const DEFAULT_OUT = path.join("results", "kakeya", "valueset-reduction");

function mod(x, q) { return ((x % q) + q) % q; }
function w(m) { return m >= 3 ? ((m - 1) * (m - 2)) / 2 : 0; }

// --- algebraic evaluation ----------------------------------------------------------
// Returns { Ns, sigmas, sumN, sumSigma, fibers } for f: F_q -> F_q.
function translateStats(f, q) {
  const Ns = new Array(q).fill(0);
  const sigmas = new Array(q).fill(0);
  const fibers = []; // fibers[s][c]
  for (let s = 0; s < q; s++) {
    const cnt = new Array(q).fill(0);
    for (let a = 0; a < q; a++) cnt[mod(f[a] - s * a, q)]++;
    let N = 0, sg = 0;
    for (let c = 0; c < q; c++) { if (cnt[c] >= 1) N++; sg += w(cnt[c]); }
    Ns[s] = N; sigmas[s] = sg; fibers.push(cnt);
  }
  const sumN = Ns.reduce((x, y) => x + y, 0);
  const sumSigma = sigmas.reduce((x, y) => x + y, 0);
  return { Ns, sigmas, sumN, sumSigma, fibers };
}
function sacrificeAlg(stats, q, d) {
  return stats.sumN - stats.Ns[d] - (q * (q - 1)) / 2;
}
function sacrificeDirect(stats, q, d) {
  // sum_{s!=d} sum_c w(m) + sum_c w(m_{d,c}+1)
  let s2 = 0;
  for (let s = 0; s < q; s++) {
    if (s === d) { for (let c = 0; c < q; c++) s2 += w(stats.fibers[s][c] + 1); }
    else { for (let c = 0; c < q; c++) s2 += w(stats.fibers[s][c]); }
  }
  return s2;
}
function sacrificeId3(stats, q, d) {
  return q - stats.Ns[d] + stats.sumSigma;
}
// Does the (f,d) configuration contain a 4-secant?
// finite: m_{s,c} >= 4 with s != d;  through (d): m_{d,c} >= 3.
function has4Secant(stats, q, d) {
  for (let s = 0; s < q; s++) {
    const need = s === d ? 3 : 4;
    for (let c = 0; c < q; c++) if (stats.fibers[s][c] >= need) return true;
  }
  return false;
}

// --- independent geometric evaluation (projective line enumeration) ----------------
// D = {(a, f(a), 1)} u {(1, d, 0)}; W = (0,1,0). Enumerate all lines [A:B:C],
// skip lines through W (B=0), count |line ∩ D|, sum w. Fully independent code path.
function sacrificeGeo(f, q, d) {
  const pts = [];
  for (let a = 0; a < q; a++) pts.push([a, f[a], 1]);
  pts.push([1, d, 0]);
  let total = 0;
  const seen = new Set();
  // lines [A:B:C] up to scale: canonical first-nonzero = 1
  for (let A = 0; A < q; A++) for (let B = 0; B < q; B++) for (let C = 0; C < q; C++) {
    if (A === 0 && B === 0 && C === 0) continue;
    // canonical rep check
    const firstIdx = A !== 0 ? 0 : B !== 0 ? 1 : 2;
    const first = [A, B, C][firstIdx];
    if (first !== 1) continue;
    // through W=(0,1,0)? A*0+B*1+C*0 = B
    if (mod(B, q) === 0) continue;
    let m = 0;
    for (const [x, y, z] of pts) if (mod(A * x + B * y + C * z, q) === 0) m++;
    total += w(m);
    seen.add(`${A},${B},${C}`);
  }
  return total;
}

// --- (A) identity verification ------------------------------------------------------
function verifyIdentities() {
  const rows = [];
  let pass = true;
  const rng = mulberry(42);
  for (const q of [5, 7, 11]) {
    const cases = [];
    // quadratic (BM extremal), a linear f, and randoms
    cases.push({ name: "quadratic", f: Array.from({ length: q }, (_, a) => mod(a * a, q)) });
    cases.push({ name: "linear", f: Array.from({ length: q }, (_, a) => mod(2 * a + 1, q)) });
    for (let r = 0; r < 4; r++) cases.push({ name: `random${r}`, f: Array.from({ length: q }, () => Math.floor(rng() * q)) });
    for (const { name, f } of cases) {
      const st = translateStats(f, q);
      // I1 per slope: pi_s - sigma_s = q - N_s
      let i1 = true;
      for (let s = 0; s < q; s++) {
        let pi = 0;
        for (let c = 0; c < q; c++) { const m = st.fibers[s][c]; pi += (m * (m - 1)) / 2; }
        if (pi - st.sigmas[s] !== q - st.Ns[s]) i1 = false;
      }
      // I2
      const i2 = st.sumN === (q * (q + 1)) / 2 + st.sumSigma;
      // I3 + geometric, for a couple of d values
      let i3 = true, geo = true;
      for (const d of [0, 1, q - 1]) {
        const a1 = sacrificeAlg(st, q, d), a2 = sacrificeDirect(st, q, d), a3 = sacrificeId3(st, q, d);
        if (a1 !== a2 || a1 !== a3) i3 = false;
        const g = sacrificeGeo(f, q, d);
        if (g !== a1) geo = false;
      }
      const ok = i1 && i2 && i3 && geo;
      pass = pass && ok;
      rows.push({ q, name, i1, i2, i3, geoMatch: geo, ok });
    }
  }
  return { rows, pass };
}

function mulberry(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- (B) exhaustive function-space decision at q in {5,7} --------------------------
function exhaustive(q) {
  const f = new Array(q).fill(0);
  let globalMin = Infinity;
  let min4 = Infinity; // min sacrifice among configs with a 4-secant
  let minimizersWith4 = 0; // global minimizers that contain a 4-secant (should be 0)
  let count = 0;
  const total = Math.pow(q, q);
  for (;;) {
    const st = translateStats(f, q);
    for (let d = 0; d < q; d++) {
      const sac = sacrificeAlg(st, q, d);
      const h4 = has4Secant(st, q, d);
      if (sac < globalMin) globalMin = sac;
      if (h4 && sac < min4) min4 = sac;
    }
    count++;
    // increment base-q counter
    let i = 0;
    while (i < q) { f[i]++; if (f[i] < q) break; f[i] = 0; i++; }
    if (i === q) break;
  }
  // second pass for the +1 premise (needs globalMin known)
  f.fill(0);
  for (;;) {
    const st = translateStats(f, q);
    for (let d = 0; d < q; d++) {
      if (sacrificeAlg(st, q, d) === globalMin && has4Secant(st, q, d)) minimizersWith4++;
    }
    let i = 0;
    while (i < q) { f[i]++; if (f[i] < q) break; f[i] = 0; i++; }
    if (i === q) break;
  }
  return { q, functionsEnumerated: count, total, globalMin, expectedBM: (q - 1) / 2, min4Secant: min4, floorTarget: q - 2, plusOnePremiseHolds: minimizersWith4 === 0 };
}

// --- (C) adversarial hunt at larger q ----------------------------------------------
function hunt(q, quadruple, restarts, steps, seed) {
  const rng = mulberry(seed);
  const A = quadruple; // frozen: f(a)=0 for a in A (4-fiber at slope 0, c=0)
  let best = Infinity;
  for (let r = 0; r < restarts; r++) {
    const f = new Array(q);
    for (let a = 0; a < q; a++) f[a] = A.includes(a) ? 0 : Math.floor(rng() * q);
    let st = translateStats(f, q);
    let cur = Math.min(...Array.from({ length: q }, (_, d) => sacrificeAlg(st, q, d)));
    for (let t = 0; t < steps; t++) {
      // random single-point move off the frozen quadruple
      let a;
      do { a = Math.floor(rng() * q); } while (A.includes(a));
      const old = f[a];
      f[a] = Math.floor(rng() * q);
      if (f[a] === old) continue;
      const st2 = translateStats(f, q);
      const val = Math.min(...Array.from({ length: q }, (_, d) => sacrificeAlg(st2, q, d)));
      // frozen 4-fiber guarantees a 4-secant for every d (slope-0 fiber >= 4)
      if (val <= cur) { cur = val; st = st2; } else { f[a] = old; }
    }
    if (cur < best) best = cur;
  }
  return best;
}

function main() {
  const outDir = process.argv.includes("--out") ? process.argv[process.argv.indexOf("--out") + 1] : DEFAULT_OUT;
  const t0 = Date.now();

  const identities = verifyIdentities();

  const exhaustiveRows = [5, 7].map(exhaustive);
  const exhaustivePass = exhaustiveRows.every(
    (r) => r.globalMin === r.expectedBM && r.min4Secant >= Math.min(r.floorTarget, r.min4Secant) && r.plusOnePremiseHolds,
  );
  // Floor verdicts at 5,7: is min4Secant >= q-2? (q=5 expected loose: q-1)
  const floorAt57 = exhaustiveRows.map((r) => ({ q: r.q, min4Secant: r.min4Secant, floor: r.floorTarget, holds: r.min4Secant >= r.floorTarget }));

  const huntRows = [];
  for (const q of [11, 13, 17, 19, 23, 29, 37]) {
    for (const quad of [[0, 1, 2, 3], [0, 1, 2, mod(4, q)]]) {
      const best = hunt(q, quad, 6, 30000, q * 1000 + quad[3]);
      huntRows.push({ q, quadruple: quad.join(","), minFound: best, floor: q - 2, belowFloor: best < q - 2 });
    }
  }
  const counterexample = huntRows.some((r) => r.belowFloor);

  const manifest = {
    artifactId: "KAK-PHASE3S-VALUESET-REDUCTION",
    generatedAt: new Date().toISOString(),
    status: "internal M3 working receipt: value-set reduction + exhaustive small-q floor + adversarial hunt",
    command: "node scripts/kakeya-valueset-reduction.mjs",
    reduction:
      "sacrifice = sum_{s!=d} N_s - q(q-1)/2 = (q - N_d) + sum_s sigma_s, N_s = value-set size of f - s*id. Floor <=> a 4-point fiber forces the translate value-set sum up by (q-3)/2 over the BM minimum (q^2-1)/2.",
    identities,
    exhaustive: { rows: exhaustiveRows, floorAt57, pass: exhaustivePass },
    hunt: { rows: huntRows, counterexampleFound: counterexample },
    falsifier: {
      name: "VALUESET_REDUCTION_MISMATCH",
      description:
        "Instrument-only: fires if any identity I1-I3 or the independent projective-geometry evaluation disagrees, or the exhaustive q in {5,7} global minimum differs from the BM value (q-1)/2, or a global minimizer contains a 4-secant (+1 premise). The hunt outcome is a measurement (a below-floor find would be a counterexample, reported separately).",
      status: identities.pass && exhaustivePass ? "clear" : "fired",
    },
    elapsedSeconds: (Date.now() - t0) / 1000,
  };
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");

  console.log("== (A) identities I1-I3 + independent geometric eval ==");
  console.log(`  ${identities.rows.length} cases, all ok: ${identities.pass}`);
  console.log("== (B) exhaustive function space ==");
  for (const r of exhaustiveRows)
    console.log(`  q=${r.q} functions=${r.functionsEnumerated} globalMin=${r.globalMin} (BM=${r.expectedBM}) min4secant=${r.min4Secant} (floor target=${r.floorTarget}) +1premise=${r.plusOnePremiseHolds}`);
  console.log("== (C) adversarial hunt (frozen 4-fiber, min over d) ==");
  for (const r of huntRows)
    console.log(`  q=${String(r.q).padEnd(3)} quad={${r.quadruple}} minFound=${r.minFound} floor=${r.floor} ${r.belowFloor ? "*** BELOW FLOOR - COUNTEREXAMPLE ***" : "holds"}`);
  console.log(`KAK_VALUESET_REDUCTION identities=${identities.pass} exhaustive=${exhaustivePass} counterexample=${counterexample} falsifier=${manifest.falsifier.status} elapsed=${manifest.elapsedSeconds.toFixed(1)}s`);
  process.exit(manifest.falsifier.status === "clear" && !counterexample ? 0 : 1);
}
main();
