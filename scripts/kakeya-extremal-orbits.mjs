#!/usr/bin/env node
// PHASE3T step 2 - orbit structure of the floor-extremal permutations.
//
// The corrected census found the 4-fiber permutations with sum-sigma = q-2
// number 1764 at q=7 and 24,200 at q=11 - exactly 1x and 2x the order
// q^2 (q-1)^2 of the natural symmetry group G = {f |-> alpha f(beta x + gamma)
// + delta} (permutation- and fiber-structure-preserving). This script decides
// the orbit structure exhaustively: regenerate ALL extremals, partition them
// under G, and report orbit count, stabilizers, canonical representatives,
// and their anatomy (4-fiber location, sigma decomposition, cycle shape).
// Prediction recorded before running: 1 orbit at q=7, 2 at q=11.

import fs from "node:fs";
import path from "node:path";

const DEFAULT_OUT = path.join("results", "kakeya", "extremal-orbits");

function mod(x, q) { return ((x % q) + q) % q; }
function w(m) { return m >= 3 ? ((m - 1) * (m - 2)) / 2 : 0; }
function inv(x, q) { for (let i = 1; i < q; i++) if ((x * i) % q === 1) return i; throw new Error("noinv"); }

// --- extremal regeneration (fixed Heap's + incremental tracker) --------------------
function collectExtremals(q) {
  const f = Array.from({ length: q }, (_, i) => i);
  const cnt = [];
  for (let s = 1; s < q; s++) cnt.push(new Uint8Array(q));
  let sumSigma = 0, cells4 = 0;
  const resetT = () => {
    sumSigma = 0; cells4 = 0;
    for (let s = 1; s < q; s++) {
      const row = cnt[s - 1]; row.fill(0);
      for (let a = 0; a < q; a++) row[mod(f[a] - s * a, q)]++;
      for (let c = 0; c < q; c++) { sumSigma += w(row[c]); if (row[c] >= 4) cells4++; }
    }
  };
  const removePoint = (a, y) => { for (let s = 1; s < q; s++) { const row = cnt[s - 1]; const c = mod(y - s * a, q); const m = row[c]; sumSigma += w(m - 1) - w(m); if (m === 4) cells4--; row[c] = m - 1; } };
  const addPoint = (a, y) => { for (let s = 1; s < q; s++) { const row = cnt[s - 1]; const c = mod(y - s * a, q); const m = row[c]; sumSigma += w(m + 1) - w(m); if (m === 3) cells4++; row[c] = m + 1; } };
  resetT();
  const target = q - 2;
  const extremals = [];
  const record = () => { if (cells4 > 0 && sumSigma === target) extremals.push(f.join(",")); };
  record();
  const c = new Array(q).fill(0);
  let i = 0;
  while (i < q) {
    if (c[i] < i) {
      const j = i % 2 === 0 ? 0 : c[i];
      const yj = f[j], yi = f[i];
      removePoint(j, yj); removePoint(i, yi);
      f[j] = yi; f[i] = yj;
      addPoint(j, yi); addPoint(i, yj);
      record();
      c[i]++; i = 0;
    } else { c[i] = 0; i++; }
  }
  return extremals;
}

// --- group action -------------------------------------------------------------------
// g = (alpha, beta, gamma, delta), alpha,beta != 0: (g.f)(x) = alpha*f(beta*x+gamma)+delta
function applyG(fArr, q, alpha, beta, gamma, delta) {
  const out = new Array(q);
  for (let x = 0; x < q; x++) out[x] = mod(alpha * fArr[mod(beta * x + gamma, q)] + delta, q);
  return out;
}

function orbitPartition(extremalKeys, q) {
  const extremalSet = new Set(extremalKeys);
  const seen = new Set();
  const orbits = [];
  for (const key of extremalKeys) {
    if (seen.has(key)) continue;
    // BFS the full orbit by applying all group elements to the representative
    const f0 = key.split(",").map(Number);
    const orbit = new Set();
    let escapes = 0; // group images that are NOT extremal (must be 0: G preserves extremality)
    for (let alpha = 1; alpha < q; alpha++)
      for (let beta = 1; beta < q; beta++)
        for (let gamma = 0; gamma < q; gamma++)
          for (let delta = 0; delta < q; delta++) {
            const g = applyG(f0, q, alpha, beta, gamma, delta).join(",");
            orbit.add(g);
            if (!extremalSet.has(g)) escapes++;
          }
    for (const k of orbit) seen.add(k);
    const groupOrder = q * q * (q - 1) * (q - 1);
    orbits.push({ rep: key, size: orbit.size, stabilizerOrder: groupOrder / orbit.size, escapes });
  }
  return orbits;
}

// --- anatomy of a representative ------------------------------------------------------
function anatomy(key, q) {
  const f = key.split(",").map(Number);
  const perSlope = [];
  for (let s = 1; s < q; s++) {
    const row = new Array(q).fill(0);
    for (let a = 0; a < q; a++) row[mod(f[a] - s * a, q)]++;
    const maxFiber = Math.max(...row);
    let sg = 0;
    for (let c = 0; c < q; c++) sg += w(row[c]);
    if (sg > 0 || maxFiber >= 3) perSlope.push({ s, sigma: sg, maxFiber });
  }
  // cycle structure
  const seen = new Array(q).fill(false);
  const cycles = [];
  for (let a = 0; a < q; a++) {
    if (seen[a]) continue;
    let len = 0, x = a;
    while (!seen[x]) { seen[x] = true; x = f[x]; len++; }
    if (len > 1) cycles.push(len);
  }
  const fixed = q - cycles.reduce((x, y) => x + y, 0);
  return { f: key, sigmaSlopes: perSlope, cycles: cycles.sort((a, b) => b - a), fixedPoints: fixed };
}

function main() {
  const outDir = process.argv.includes("--out") ? process.argv[process.argv.indexOf("--out") + 1] : DEFAULT_OUT;
  const t0 = Date.now();
  const results = [];
  for (const q of [7, 11]) {
    console.log(`== q=${q}: regenerating extremals ==`);
    const keys = collectExtremals(q);
    const groupOrder = q * q * (q - 1) * (q - 1);
    console.log(`  extremals=${keys.length}  groupOrder=${groupOrder}  ratio=${(keys.length / groupOrder).toFixed(3)}`);
    const orbits = orbitPartition(keys, q);
    const escapes = orbits.reduce((x, o) => x + o.escapes, 0);
    console.log(`  ORBITS=${orbits.length}  sizes=[${orbits.map((o) => o.size).join(", ")}]  stabilizers=[${orbits.map((o) => o.stabilizerOrder).join(", ")}]  escapes=${escapes}`);
    const reps = orbits.map((o) => anatomy(o.rep, q));
    for (const r of reps) {
      console.log(`  rep f=[${r.f}] cycles=[${r.cycles}] fixed=${r.fixedPoints}`);
      console.log(`      sigma slopes: ${r.sigmaSlopes.map((x) => `s=${x.s}:sigma=${x.sigma},max=${x.maxFiber}`).join("  ")}`);
    }
    results.push({ q, extremals: keys.length, groupOrder, orbits: orbits.map((o) => ({ size: o.size, stabilizerOrder: o.stabilizerOrder })), escapes, representatives: reps });
  }
  const pass = results.every((r) => r.escapes === 0 && r.orbits.reduce((x, o) => x + o.size, 0) === r.extremals);
  const manifest = {
    artifactId: "KAK-PHASE3T-EXTREMAL-ORBITS",
    generatedAt: new Date().toISOString(),
    status: "internal extremal-orbit decomposition (floor extremals mod symmetry)",
    command: "node scripts/kakeya-extremal-orbits.mjs",
    prediction: "recorded pre-run: 1 orbit at q=7 (1764 = |G|), 2 at q=11 (24200 = 2|G|)",
    results,
    falsifier: {
      name: "EXTREMAL_ORBIT_MISMATCH",
      description: "Instrument-only: fires if any group image of an extremal is non-extremal (G must preserve extremality) or the orbit sizes fail to partition the extremal set.",
      status: pass ? "clear" : "fired",
    },
    elapsedSeconds: (Date.now() - t0) / 1000,
  };
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
  console.log(`KAK_EXTREMAL_ORBITS ${results.map((r) => `q${r.q}:${r.orbits.length}orbit(s)`).join(" ")} falsifier=${manifest.falsifier.status} elapsed=${manifest.elapsedSeconds.toFixed(1)}s`);
  process.exit(pass ? 0 : 1);
}
main();
