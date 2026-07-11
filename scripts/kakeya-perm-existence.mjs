#!/usr/bin/env node
// PHASE3U step 2 - existence of extremal permutations at q >= 13.
//
// Step 1 found the first-found parabola optimum usually has NO bijective
// direction (no permutation representation), while 24,200 extremal
// permutations are KNOWN at q=11. So either other parabola optima carry the
// permutation representations, or the extremal permutations dualize to
// NON-parabola optima (a parabola-family blind spot). Two legs decide:
//  (E1) ALL parabola optima per LOW orbit at q in {11,13,17,19,23}: test all
//       q+1 directions of each for a bijective translate; extract + verify
//       any extremal permutation found.
//  (E2) transposition hillclimb over permutations with a frozen 4-fiber
//       (fixed points a in A: f(a) = a) minimizing sum-sigma; q=11 is the
//       control (existence KNOWN, must reach 9); at q in {13,17,19,23} a hit
//       at q-2 is an existence WITNESS (verified); misses are weak evidence.
// No Euclidean claim. Existence for q >= 17 is the open question under test.

import fs from "node:fs";
import path from "node:path";
import * as Core from "../kakeya/kakeya-core.js";

const DEFAULT_OUT = path.join("results", "kakeya", "perm-existence");
const INF = -1;

function mod(x, q) { return ((x % q) + q) % q; }
function inv(x, q) { for (let i = 1; i < q; i++) if ((x * i) % q === 1) return i; throw new Error("noinv"); }
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

// ---- orbits (as before) ----
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
const LOW_ORBITS = { 11: ["generic"], 13: ["equianharmonic", "generic"], 17: ["harmonic", "generic-a", "generic-b"], 19: ["generic-a", "generic-b"], 23: ["harmonic", "generic-a", "generic-b"] };

// ---- parabola machinery ----
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
function allParabolaOptima(q, quad) {
  const dirs = Core.directions(q), n = Core.pointCount(q);
  const starPts = quad.map((i) => [...Core.lineMask(dirs[i], 0, q)]);
  const needed = [];
  for (let i = 0; i < dirs.length; i++) if (!quad.includes(i)) needed.push(i);
  const stamp = new Int32Array(n), cnt = new Uint8Array(n);
  let epoch = 0; const buf = new Array(q);
  const optima = [];
  for (const axis of quad) {
    const mus = needed.map((d) => slopePre(q, axis, d));
    for (let beta = 0; beta < q; beta++) for (let gamma = 0; gamma < q; gamma++) {
      epoch++; let sac = 0;
      for (const pts of starPts) for (const p of pts) { if (stamp[p] !== epoch) { stamp[p] = epoch; cnt[p] = 1; } else { const m = cnt[p]; if (m >= 2) sac += m - 1; cnt[p] = m + 1; } }
      for (const mu of mus) { tangentPoints(q, axis, beta, gamma, mu, buf); for (let x = 0; x < q; x++) { const p = buf[x]; if (stamp[p] !== epoch) { stamp[p] = epoch; cnt[p] = 1; } else { const m = cnt[p]; if (m >= 2) sac += m - 1; cnt[p] = m + 1; } } }
      if (sac === q - 2) optima.push({ axis, beta, gamma });
    }
  }
  return { optima, needed };
}

// ---- duality extraction (as step 1) ----
function lineDual(dir, b, q) { return dir === q ? [1, 0, mod(-b, q)] : [dir, q - 1, b]; }
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
function apply3(M, v, q) { return [0, 1, 2].map((r) => mod(M[r][0] * v[0] + M[r][1] * v[1] + M[r][2] * v[2], q)); }
function extractRep(q, lines, pi) {
  const duals = lines.map((L) => lineDual(L.dir, L.b, q));
  const P = duals[pi], W = [0, 0, 1];
  let B = null;
  for (const cand of [[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]]) {
    const M0 = [[P[0], W[0], cand[0]], [P[1], W[1], cand[1]], [P[2], W[2], cand[2]]];
    if (inv3(M0, q)) { B = M0; break; }
  }
  if (!B) return null;
  const M = inv3(B, q);
  const f = new Array(q).fill(null);
  for (let k = 0; k < lines.length; k++) {
    if (k === pi) continue;
    const v = apply3(M, duals[k], q);
    if (v[2] === 0) return null;
    const zi = inv(v[2], q);
    const x = mod(v[0] * zi, q), y = mod(v[1] * zi, q);
    if (f[x] !== null) return null;
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
function verifyExtremalPerm(g, q) {
  if (new Set(g).size !== q) return { ok: false };
  const st = translateStats(g, q);
  let four = 0, fourPos = null;
  for (let s = 1; s < q; s++) for (let c = 0; c < q; c++) {
    if (st.fibers[s][c] === 4) { four++; fourPos = { s, c }; }
    if (st.fibers[s][c] >= 5) four += 10;
  }
  if (!(st.sumSigma === q - 2 && four === 1)) return { ok: false };
  const pos = [];
  for (let a = 0; a < q; a++) if (mod(g[a] - fourPos.s * a, q) === fourPos.c) pos.push(a);
  const cr = mod((pos[0] - pos[2]) * (pos[1] - pos[3]) * inv(mod((pos[0] - pos[3]) * (pos[1] - pos[2]), q), q), q);
  return { ok: true, crClass: labelOf(cr, q) };
}

// ---- E2: transposition hillclimb over 4-fiber permutations -------------------------
// Frozen fixed points A (f(a) = a for a in A => 4-fiber at slope 1, c = 0);
// permute the complement; moves = transpositions; minimize sum sigma over s>=1.
function hillclimbPerm(q, A, restarts, steps, seed) {
  const rng = mulberry(seed);
  const rest = [];
  for (let a = 0; a < q; a++) if (!A.includes(a)) rest.push(a);
  let best = Infinity;
  let bestPerm = null;
  for (let r = 0; r < restarts; r++) {
    // random derangement-ish start on rest (values = rest shuffled)
    const vals = [...rest];
    for (let i = vals.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [vals[i], vals[j]] = [vals[j], vals[i]]; }
    const f = new Array(q);
    for (const a of A) f[a] = a;
    rest.forEach((a, i) => { f[a] = vals[i]; });
    const evalSigma = () => {
      let ss = 0;
      for (let s = 1; s < q; s++) {
        const row = new Array(q).fill(0);
        for (let a = 0; a < q; a++) row[mod(f[a] - s * a, q)]++;
        for (let c = 0; c < q; c++) ss += w(row[c]);
      }
      return ss;
    };
    let cur = evalSigma();
    for (let t = 0; t < steps; t++) {
      const i = rest[Math.floor(rng() * rest.length)], j = rest[Math.floor(rng() * rest.length)];
      if (i === j) continue;
      [f[i], f[j]] = [f[j], f[i]];
      const val = evalSigma();
      if (val <= cur) cur = val; else [f[i], f[j]] = [f[j], f[i]];
    }
    if (cur < best) { best = cur; bestPerm = [...f]; }
  }
  return { best, bestPerm };
}

function main() {
  const outDir = process.argv.includes("--out") ? process.argv[process.argv.indexOf("--out") + 1] : DEFAULT_OUT;
  const t0 = Date.now();
  const e1rows = [], e2rows = [];
  let instrumentPass = true;

  console.log("== E1: all parabola optima x all directions ==");
  for (const [qs, labels] of Object.entries(LOW_ORBITS)) {
    const q = Number(qs);
    const reps = orbitReps(q);
    for (const label of labels) {
      const rep = reps.find((r) => r.label === label);
      const { optima } = allParabolaOptima(q, rep.quad);
      let permReps = 0, witness = null;
      for (const opt of optima) {
        const lines = [];
        for (let i = 0; i < Core.directionCount(q); i++) {
          lines.push({ dir: i, b: rep.quad.includes(i) ? 0 : lineIntercept(q, opt.axis, opt.beta, opt.gamma, i) });
        }
        for (let pi = 0; pi < lines.length; pi++) {
          const f = extractRep(q, lines, pi);
          if (!f) continue;
          const st = translateStats(f, q);
          const smax = st.Ns.indexOf(q);
          if (smax === -1) continue;
          const g = f.map((y, x) => mod(y - smax * x, q));
          const v = verifyExtremalPerm(g, q);
          if (v.ok) { permReps++; if (!witness) witness = { f: g.join(","), crClass: v.crClass, axis: opt.axis }; }
          else instrumentPass = false;
        }
      }
      e1rows.push({ q, orbit: label, parabolaOptima: optima.length, permutationReps: permReps, witness });
      console.log(`  q=${String(q).padEnd(3)} ${label.padEnd(15)} optima=${optima.length} permReps=${permReps}${witness ? ` crClass=${witness.crClass}` : ""}`);
    }
  }

  console.log("== E2: transposition hillclimb (frozen fixed-point 4-fiber) ==");
  for (const q of [11, 13, 17, 19, 23]) {
    // fixed-point sets with both CR classes where possible: {0,1,2,3} and {0,1,2,4}
    for (const A of [[0, 1, 2, 3], [0, 1, 2, 4]]) {
      const cr = mod((A[0] - A[2]) * (A[1] - A[3]) * inv(mod((A[0] - A[3]) * (A[1] - A[2]), q), q), q);
      const crClass = labelOf(cr, q);
      const { best, bestPerm } = hillclimbPerm(q, A, 8, 25000, q * 7919 + A[3]);
      let verified = false;
      if (best === q - 2 && bestPerm) verified = verifyExtremalPerm(bestPerm, q).ok;
      e2rows.push({ q, A: A.join(","), crClass, minSigma: best, target: q - 2, hit: best === q - 2, verified, witness: best === q - 2 ? bestPerm.join(",") : null });
      console.log(`  q=${String(q).padEnd(3)} A={${A.join(",")}} (${crClass}) minSigma=${best} target=${q - 2} ${best === q - 2 ? (verified ? "EXISTENCE WITNESS (verified)" : "HIT-unverified") : best < q - 2 ? "*** BELOW (bug?)" : "miss"}`);
      if (best < q - 2) instrumentPass = false;
    }
  }

  const manifest = {
    artifactId: "KAK-PHASE3U-PERM-EXISTENCE",
    generatedAt: new Date().toISOString(),
    status: "internal existence probe: extremal permutations at q >= 13",
    command: "node scripts/kakeya-perm-existence.mjs",
    E1: e1rows, E2: e2rows,
    falsifier: {
      name: "PERM_EXISTENCE_MISMATCH",
      description: "Instrument-only: fires if an extracted/hillclimbed candidate fails verification or sum-sigma drops below q-2 (contradicting proved floors at q <= 17-known fields).",
      status: instrumentPass ? "clear" : "fired",
    },
    elapsedSeconds: (Date.now() - t0) / 1000,
  };
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
  console.log(`KAK_PERM_EXISTENCE falsifier=${manifest.falsifier.status} elapsed=${manifest.elapsedSeconds.toFixed(1)}s`);
  process.exit(instrumentPass ? 0 : 1);
}
main();
