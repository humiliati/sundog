#!/usr/bin/env node
// PHASE3U step 1 - constructive canonical extremals (classification route).
//
// The banked conjecture: the floor-extremal permutation is unique up to G+
// for all odd q >= 7 (exhaustively true at q in {7,11}). Before uniqueness
// can be attacked at general q, the EXISTENCE leg needs a construction: this
// script extracts extremal permutations from the parabola-optimal completions
// of LOW orbits, deterministically, at q in {7,11,13,17,19,23}:
//   1. build the pure-parabola optimal completion (q+1 lines, sacrifice q-2);
//   2. dualize the lines to points (transversal of the pencil through W);
//   3. for each choice of "infinite" line P: apply the projectivity sending
//      W -> vertical-infinity and P -> (slope-0 infinity), read off the graph
//      f; if some translate f - s*id is bijective, shear so f becomes a
//      PERMUTATION; verify sum-sigma = q-2 with exactly one 4-fiber;
//   4. report which directions admit bijective representations, the 4-fiber
//      cross-ratio class of each extracted model (generic-only law test at
//      13..23), and the cycle shape.
// Instrument-falsified: extracted candidates must verify exactly; the
// existence/absence pattern is a measurement. No Euclidean claim.

import fs from "node:fs";
import path from "node:path";
import * as Core from "../kakeya/kakeya-core.js";

const DEFAULT_OUT = path.join("results", "kakeya", "canonical-extremal");
const INF = -1;

function mod(x, q) { return ((x % q) + q) % q; }
function inv(x, q) { for (let i = 1; i < q; i++) if ((x * i) % q === 1) return i; throw new Error("noinv"); }
function w(m) { return m >= 3 ? ((m - 1) * (m - 2)) / 2 : 0; }

// --- orbit reps + LOW list (from the 3L/3M exact+construction map) ------------------
function crossRatio(a, b, c, d, q) { const diff = (x, y) => (x === INF || y === INF ? 1 : mod(x - y, q)); return (diff(a, c) * diff(b, d) * inv((diff(a, d) * diff(b, c)) % q, q)) % q; }
function sixSet(l, q) { const s = new Set([l, inv(l, q), mod(1 - l, q), inv(mod(1 - l, q), q), mod(l * inv(mod(l - 1, q), q), q), mod(mod(l - 1, q) * inv(l, q), q)]); return [...s].sort((x, y) => x - y); }
function labelOf(l, q) {
  const six = sixSet(l, q);
  const h = [2, mod((q + 1) / 2, q), q - 1].sort((x, y) => x - y);
  if (six.length === 3 && six.join(",") === h.join(",")) return "harmonic";
  if (six.length === 2 && six.every((v) => mod(v * v - v + 1, q) === 0)) return "equianharmonic";
  return "generic";
}
function orbitReps(q) {
  const dc = q + 1, dv = (i) => (i === q ? INF : i), reps = new Map();
  for (let a = 0; a < dc; a++) for (let b = a + 1; b < dc; b++) for (let c = b + 1; c < dc; c++) for (let d = c + 1; d < dc; d++) {
    const key = sixSet(crossRatio(dv(a), dv(b), dv(c), dv(d), q), q).join(",");
    if (!reps.has(key)) reps.set(key, { label: labelOf(crossRatio(dv(a), dv(b), dv(c), dv(d), q), q), quad: [a, b, c, d] });
  }
  const byBase = new Map();
  for (const k of [...reps.keys()].sort()) byBase.set(reps.get(k).label, (byBase.get(reps.get(k).label) ?? 0) + 1);
  const cnt = new Map(), out = [];
  for (const k of [...reps.keys()].sort()) {
    const r = reps.get(k); let label = r.label;
    if (byBase.get(r.label) > 1) { const n = (cnt.get(r.label) ?? 0) + 1; cnt.set(r.label, n); label = `${r.label}-${String.fromCharCode(96 + n)}`; }
    out.push({ label, quad: r.quad });
  }
  return out;
}
// LOW orbits per field (3L map; 11-harmonic excluded: non-parabola optimum).
const LOW_ORBITS = {
  7: ["harmonic"],
  11: ["generic"],
  13: ["equianharmonic", "generic"],
  17: ["harmonic", "generic-a", "generic-b"],
  19: ["generic-a", "generic-b"],
  23: ["harmonic", "generic-a", "generic-b"],
};

// --- pure-parabola optimal completion (3L machinery, alpha = 1) ---------------------
function tangentPoints(q, axis, beta, gamma, mu, out) {
  const t = mod((mu - beta) * inv(2, q), q), c = mod(gamma - t * t, q);
  if (axis === q) for (let x = 0; x < q; x++) out[x] = Core.pointIndex(x, (mu * x + c) % q, q);
  else { const s = axis; for (let x = 0; x < q; x++) { const y = (mu * x + c) % q; out[x] = Core.pointIndex(y, (x + s * y) % q, q); } }
}
function slopePre(q, axis, d) { if (axis === q) return d; if (d === q) return 0; return inv(mod(d - axis, q), q); }
function lineIntercept(q, axis, beta, gamma, dirIndex) {
  const buf = new Array(q);
  tangentPoints(q, axis, beta, gamma, slopePre(q, axis, dirIndex), buf);
  const { x, y } = Core.indexToXY(buf[0], q);
  return dirIndex === q ? x : mod(y - dirIndex * x, q);
}
function parabolaCompletion(q, quad) {
  const dirs = Core.directions(q), n = Core.pointCount(q);
  const starPts = quad.map((i) => [...Core.lineMask(dirs[i], 0, q)]);
  const needed = [];
  for (let i = 0; i < dirs.length; i++) if (!quad.includes(i)) needed.push(i);
  const stamp = new Int32Array(n), cnt = new Uint8Array(n);
  let epoch = 0; const buf = new Array(q);
  let best = { sac: Infinity };
  for (const axis of quad) {
    const mus = needed.map((d) => slopePre(q, axis, d));
    for (let beta = 0; beta < q; beta++) for (let gamma = 0; gamma < q; gamma++) {
      epoch++; let sac = 0;
      for (const pts of starPts) for (const p of pts) { if (stamp[p] !== epoch) { stamp[p] = epoch; cnt[p] = 1; } else { const m = cnt[p]; if (m >= 2) sac += m - 1; cnt[p] = m + 1; } }
      for (const mu of mus) { tangentPoints(q, axis, beta, gamma, mu, buf); for (let x = 0; x < q; x++) { const p = buf[x]; if (stamp[p] !== epoch) { stamp[p] = epoch; cnt[p] = 1; } else { const m = cnt[p]; if (m >= 2) sac += m - 1; cnt[p] = m + 1; } } }
      if (sac < best.sac) best = { sac, axis, beta, gamma };
    }
  }
  // lines: (dirIndex, intercept) for all q+1 directions
  const lines = [];
  for (let i = 0; i < Core.directionCount(q); i++) {
    if (quad.includes(i)) lines.push({ dir: i, b: 0 });
    else lines.push({ dir: i, b: lineIntercept(q, best.axis, best.beta, best.gamma, i) });
  }
  return { lines, sacrifice: best.sac };
}

// --- duality + projectivity ----------------------------------------------------------
function lineDual(dir, b, q) {
  if (dir === q) return [1, 0, mod(-b, q)];
  return [dir, q - 1, b];
}
// 3x3 inverse mod q via adjugate
function inv3(M, q) {
  const [[a, b, c], [d, e, f], [g, h, i]] = M;
  const A = mod(e * i - f * h, q), B = mod(-(d * i - f * g), q), C = mod(d * h - e * g, q);
  const D = mod(-(b * i - c * h), q), E = mod(a * i - c * g, q), F = mod(-(a * h - b * g), q);
  const G = mod(b * f - c * e, q), H = mod(-(a * f - c * d), q), I = mod(a * e - b * d, q);
  const det = mod(a * A + b * B + c * C, q);
  if (det === 0) return null;
  const di = inv(det, q);
  return [[mod(A * di, q), mod(D * di, q), mod(G * di, q)], [mod(B * di, q), mod(E * di, q), mod(H * di, q)], [mod(C * di, q), mod(F * di, q), mod(I * di, q)]];
}
function apply3(M, v, q) {
  return [0, 1, 2].map((r) => mod(M[r][0] * v[0] + M[r][1] * v[1] + M[r][2] * v[2], q));
}

// Extract the (f, shear) representation with "infinite line" = lines[pi].
function extractRep(q, lines, pi) {
  const duals = lines.map((L) => lineDual(L.dir, L.b, q));
  const P = duals[pi]; // -> e1 (on l_inf)
  const W = [0, 0, 1]; // pencil vertex -> e2 (vertical infinity)
  // basis third vector Q independent of P, W
  let Q = null;
  for (const cand of [[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]]) {
    const B = [[P[0], W[0], cand[0]], [P[1], W[1], cand[1]], [P[2], W[2], cand[2]]];
    if (inv3(B, q)) { Q = cand; break; }
  }
  if (!Q) return null;
  const B = [[P[0], W[0], Q[0]], [P[1], W[1], Q[1]], [P[2], W[2], Q[2]]];
  const M = inv3(B, q); // M sends P->e1, W->e2, Q->e3
  // transformed dual points (excluding P) must be affine with distinct x
  const f = new Array(q).fill(null);
  for (let k = 0; k < duals.length; k++) {
    if (k === pi) continue;
    const v = apply3(M, duals[k], q);
    if (v[2] === 0) return null; // unexpected: point on l_inf
    const zi = inv(v[2], q);
    const x = mod(v[0] * zi, q), y = mod(v[1] * zi, q);
    if (f[x] !== null) return null; // not a graph (shouldn't happen)
    f[x] = y;
  }
  return f;
}

function translateStats(f, q) {
  const Ns = new Array(q).fill(0), fibers = [];
  let sumSigma = 0;
  for (let s = 0; s < q; s++) {
    const row = new Array(q).fill(0);
    for (let a = 0; a < q; a++) row[mod(f[a] - s * a, q)]++;
    let N = 0;
    for (let c = 0; c < q; c++) { if (row[c] >= 1) N++; sumSigma += w(row[c]); }
    Ns[s] = N; fibers.push(row);
  }
  return { Ns, fibers, sumSigma };
}

function cycleType(f, q) {
  const seen = new Array(q).fill(false), cycles = [];
  for (let a = 0; a < q; a++) {
    if (seen[a]) continue;
    let len = 0, x = a;
    while (!seen[x]) { seen[x] = true; x = f[x]; len++; }
    if (len > 1) cycles.push(len);
  }
  return { cycles: cycles.sort((a, b) => b - a), fixed: q - cycles.reduce((x, y) => x + y, 0) };
}

function main() {
  const outDir = process.argv.includes("--out") ? process.argv[process.argv.indexOf("--out") + 1] : DEFAULT_OUT;
  const t0 = Date.now();
  const rows = [];
  let instrumentPass = true;

  for (const [qs, lowLabels] of Object.entries(LOW_ORBITS)) {
    const q = Number(qs);
    const reps = orbitReps(q);
    for (const label of lowLabels) {
      const rep = reps.find((r) => r.label === label);
      const { lines, sacrifice } = parabolaCompletion(q, rep.quad);
      if (sacrifice !== q - 2) { instrumentPass = false; rows.push({ q, orbit: label, error: `completion sacrifice ${sacrifice} != ${q - 2}` }); continue; }
      let bijectiveDirs = 0, extracted = null, extractedVia = null;
      for (let pi = 0; pi < lines.length; pi++) {
        const f = extractRep(q, lines, pi);
        if (!f) { instrumentPass = false; continue; }
        const st = translateStats(f, q);
        const smax = st.Ns.indexOf(q);
        if (smax === -1) continue;
        bijectiveDirs++;
        if (!extracted) {
          // shear: g = f - smax * x is a permutation
          const g = f.map((y, x) => mod(y - smax * x, q));
          const gst = translateStats(g, q);
          // verify: permutation, sum sigma = q-2, exactly one 4-fiber
          const isPerm = new Set(g).size === q;
          let four = 0, fourPos = null;
          for (let s = 1; s < q; s++) for (let c = 0; c < q; c++) {
            if (gst.fibers[s][c] === 4) { four++; fourPos = { s, c }; }
            if (gst.fibers[s][c] >= 5) four += 10;
          }
          const ok = isPerm && gst.sumSigma === q - 2 && four === 1;
          if (!ok) { instrumentPass = false; }
          // 4-fiber cross-ratio class
          let crClass = null;
          if (fourPos) {
            const pos = [];
            for (let a = 0; a < q; a++) if (mod(g[a] - fourPos.s * a, q) === fourPos.c) pos.push(a);
            const cr = mod((pos[0] - pos[2]) * (pos[1] - pos[3]) * inv(mod((pos[0] - pos[3]) * (pos[1] - pos[2]), q), q), q);
            crClass = labelOf(cr, q);
          }
          extracted = { g, verified: ok, crClass, ...cycleType(g, q) };
          extractedVia = lines[pi].dir;
        }
      }
      rows.push({
        q, orbit: label, completionSacrifice: sacrifice,
        bijectiveDirections: bijectiveDirs, totalDirections: lines.length,
        extremalPermutationExtracted: extracted !== null && extracted.verified,
        via: extractedVia,
        fourFiberClass: extracted?.crClass ?? null,
        cycles: extracted?.cycles ?? null, fixedPoints: extracted?.fixed ?? null,
        f: extracted ? extracted.g.join(",") : null,
      });
      const r = rows[rows.length - 1];
      console.log(`q=${String(q).padEnd(3)} ${label.padEnd(15)} bijectiveDirs=${r.bijectiveDirections}/${r.totalDirections} extracted=${r.extremalPermutationExtracted} 4fiberClass=${r.fourFiberClass} cycles=[${r.cycles}] fixed=${r.fixedPoints}`);
    }
  }

  const existenceAll = rows.every((r) => r.extremalPermutationExtracted);
  const manifest = {
    artifactId: "KAK-PHASE3U-CANONICAL-EXTREMAL",
    generatedAt: new Date().toISOString(),
    status: "internal classification-route receipt: constructive extremal permutations from parabola completions",
    command: "node scripts/kakeya-canonical-extremal.mjs",
    rows,
    existenceAllFields: existenceAll,
    falsifier: {
      name: "CANONICAL_EXTREMAL_MISMATCH",
      description: "Instrument-only: fires if a parabola completion misses sacrifice q-2 on a LOW orbit, an extraction fails structurally (non-graph), or an extracted candidate fails verification (permutation + sum-sigma = q-2 + exactly one 4-fiber).",
      status: instrumentPass ? "clear" : "fired",
    },
    elapsedSeconds: (Date.now() - t0) / 1000,
  };
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
  console.log(`KAK_CANONICAL_EXTREMAL existence_all=${existenceAll} falsifier=${manifest.falsifier.status} elapsed=${manifest.elapsedSeconds.toFixed(1)}s`);
  process.exit(instrumentPass ? 0 : 1);
}
main();
