// Kakeya tiny finite-field workbench — pure core logic.
//
// No DOM. Importable by the browser UI (kakeya-workbench.js) and by the Node
// acceptance tests (scripts/kakeya-workbench-tests.mjs). Exact modular
// arithmetic over a prime field F_q. Implements the locked conventions of
// docs/kakeya/PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md.
//
// This is a reader/teaching toy around the KNOWN finite-field Dvir theorem.
// It is not Euclidean Kakeya evidence, not a maximal-function result, and not a
// regime-2 / control-sufficiency claim.

export const SUPPORTED_Q = [5, 7, 11];
export const DEFAULT_Q = 7;

export function isSupportedQ(q) {
  return SUPPORTED_Q.includes(q);
}

// --- coordinates -----------------------------------------------------------
// Canonical point index: point_index(x, y) = y*q + x, with x, y in 0..q-1.
export function pointIndex(x, y, q) {
  return y * q + x;
}
export function pointCount(q) {
  return q * q;
}
export function indexToXY(idx, q) {
  return { x: idx % q, y: Math.floor(idx / q) };
}

// --- directions ------------------------------------------------------------
// Canonical direction order: finite slopes 0..q-1 (slope m, vector (1, m)),
// then the vertical direction "inf" (vector (0, 1)). Exactly q + 1 directions.
export function directions(q) {
  const dirs = [];
  for (let m = 0; m < q; m++) dirs.push({ kind: "slope", m, label: String(m) });
  dirs.push({ kind: "inf", label: "inf" });
  return dirs;
}
export function directionCount(q) {
  return q + 1;
}

// --- lines -----------------------------------------------------------------
// A line is a direction + intercept b. Finite slope m: {(x, (m*x + b) mod q)}.
// Vertical: {(b, y)}. Each line has exactly q points; the q intercept lines of
// one direction partition F_q^2.
export function lineMask(dir, b, q) {
  const mask = new Set();
  if (dir.kind === "slope") {
    for (let x = 0; x < q; x++) {
      const y = (((dir.m * x + b) % q) + q) % q;
      mask.add(pointIndex(x, y, q));
    }
  } else {
    for (let y = 0; y < q; y++) mask.add(pointIndex(b, y, q));
  }
  return mask;
}

function lineSubsetOf(line, K) {
  for (const p of line) if (!K.has(p)) return false;
  return true;
}

// Direction d is covered by body K iff some intercept line in direction d is a
// subset of K.
export function directionCovered(dir, q, K) {
  for (let b = 0; b < q; b++) {
    if (lineSubsetOf(lineMask(dir, b, q), K)) return true;
  }
  return false;
}

// --- the registered primary shadow ----------------------------------------
// Coverage bitset over the canonical direction order. This is the ONLY primary
// shadow; it is q + 1 bits and is deliberately many-to-one (lossy).
export function shadowBitset(q, K) {
  return directions(q).map((dir) => (directionCovered(dir, q, K) ? 1 : 0));
}

// Dvir planar lower-bound floor: binom(q+1, 2) = q(q+1)/2 = the count of
// monomials of total degree <= q-1 in 2 variables. A valid lower bound, NOT the
// exact planar minimum (Dvir's bound is not tight in the plane).
export function dvirFloor(q) {
  return (q * (q + 1)) / 2;
}

// Allowed primary display fields + verdict (spec §4).
export function shadowSummary(q, K) {
  const bits = shadowBitset(q, K);
  const dirs = directions(q);
  const directionsCovered = bits.reduce((a, b) => a + b, 0);
  const dirCount = directionCount(q);
  const missing = dirs.filter((_, i) => bits[i] === 0).map((d) => d.label);
  const complete = directionsCovered === dirCount;
  let verdict;
  if (complete) verdict = "complete finite-field Kakeya set";
  else if (directionsCovered > 0) verdict = "near miss";
  else verdict = "empty shadow";
  return {
    q,
    bodySize: K.size,
    bodyFraction: K.size / pointCount(q),
    directionsCovered,
    directionCount: dirCount,
    coverageFraction: directionsCovered / dirCount,
    missing,
    complete,
    dvirFloor: dvirFloor(q),
    // For any complete set Dvir guarantees this is true; shown as a consistency
    // confirmation, not a discovery.
    dvirFloorConsistent: complete ? K.size >= dvirFloor(q) : null,
    verdict,
    bits,
  };
}

// Shadow export object — ONLY the allowed fields (spec §5.3). No selected
// points, no witness intercepts, no line masks.
export function exportShadow(q, K) {
  const s = shadowSummary(q, K);
  return {
    q,
    directionOrder: directions(q).map((d) => d.label),
    coverageBitset: s.bits,
    bodySize: s.bodySize,
    bodyFraction: s.bodyFraction,
    directionsCovered: s.directionsCovered,
    coverageFraction: s.coverageFraction,
    verdict: s.verdict,
  };
}

// One witness line for a single user-selected covered direction (spec §4/§5.4).
// Returns the point indices of the lowest-intercept covered line, or null.
export function witnessLine(dir, q, K) {
  for (let b = 0; b < q; b++) {
    const line = lineMask(dir, b, q);
    if (lineSubsetOf(line, K)) return { intercept: b, points: [...line] };
  }
  return null;
}

// --- deterministic PRNG (mulberry32) for reproducible random baselines -----
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- baselines (spec §6) — each returns a Set of point indices -------------
export function bEmpty() {
  return new Set();
}
export function bSingleLine(q, dir = directions(q)[0], b = 0) {
  return lineMask(dir, b, q);
}
export function bWholePlane(q) {
  const K = new Set();
  for (let i = 0; i < pointCount(q); i++) K.add(i);
  return K;
}
export function bWholeMinusOne(q, removeIdx = 0) {
  const K = bWholePlane(q);
  K.delete(removeIdx);
  return K;
}
export function bRandomSubset(q, size, seed = 1) {
  const n = pointCount(q);
  const rng = mulberry32(seed);
  const idx = [...Array(n).keys()];
  const take = Math.max(0, Math.min(size, n));
  for (let i = 0; i < take; i++) {
    const j = i + Math.floor(rng() * (n - i));
    const tmp = idx[i];
    idx[i] = idx[j];
    idx[j] = tmp;
  }
  return new Set(idx.slice(0, take));
}
export function bRandomLineCover(q, seed = 1) {
  const rng = mulberry32(seed);
  const K = new Set();
  for (const dir of directions(q)) {
    const b = Math.floor(rng() * q);
    for (const p of lineMask(dir, b, q)) K.add(p);
  }
  return K;
}
// Greedy: process directions in canonical order; for each, add the intercept
// line that minimizes newly-added points; tie-break smallest intercept.
export function bGreedyLineCover(q) {
  const K = new Set();
  for (const dir of directions(q)) {
    let bestLine = null;
    let bestNew = Infinity;
    for (let b = 0; b < q; b++) {
      const line = lineMask(dir, b, q);
      let added = 0;
      for (const p of line) if (!K.has(p)) added++;
      if (added < bestNew) {
        bestNew = added;
        bestLine = line;
      }
    }
    for (const p of bestLine) K.add(p);
  }
  return K;
}
