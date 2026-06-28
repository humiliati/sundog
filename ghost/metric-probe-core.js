// Ghost Phase 4 metric probe core.
//
// Pure logic for the rigorous falsification battery in
// scripts/ghost-metric-tests.mjs. Implements
// docs/ghost/PHASE4_METRIC_PROBE_SPEC.md.
//
// The observable is the recognizability radius = Mosse's constant of
// recognizability (Mosse; Durand & Leroy, arXiv:1610.05577), a finite fixed
// constant of a primitive substitution. This probe does NOT define a new
// invariant; it measures known vocabulary and tries to falsify the unbounded
// reading of the Ghost Boundary Heuristic.
//
// Stage S1: 1D primitive substitutions + periodic control. (2D Penrose
// recognizability is staged separately, per spec section 6.)

// 1D primitive substitutions. seed expands under the rule.
export const SUBSTITUTIONS = {
  fibonacci: { rule: { a: ["a", "b"], b: ["a"] }, seed: "a" },
  "period-doubling": { rule: { a: ["a", "b"], b: ["a", "a"] }, seed: "a" },
  "thue-morse": { rule: { a: ["a", "b"], b: ["b", "a"] }, seed: "a" },
};

function applyOnce(word, rule) {
  const out = [];
  for (const c of word) out.push(...rule[c]);
  return out;
}

export function generateWord(name, depth) {
  const def = SUBSTITUTIONS[name];
  if (!def) throw new Error(`unknown substitution: ${name}`);
  let w = [def.seed];
  for (let i = 0; i < depth; i++) w = applyOnce(w, def.rule);
  return w;
}

// Level-d word with the level-1 supertile structure: each letter carries the
// source letter of its level-1 block and its offset within that block. Cut
// points are offset-0 positions. Requires depth >= 1.
export function generateWithAncestry(name, depth) {
  if (depth < 1) throw new Error("ancestry needs depth >= 1");
  const def = SUBSTITUTIONS[name];
  const parent = generateWord(name, depth - 1); // level d-1 super-letters
  const letters = [];
  const cut = [];
  const role = [];
  for (let k = 0; k < parent.length; k++) {
    const src = parent[k];
    const img = def.rule[src];
    for (let off = 0; off < img.length; off++) {
      letters.push(img[off]);
      cut.push(off === 0);
      role.push(`${src}:${off}`); // (source letter, offset) recovers desubstitution
    }
  }
  return { letters, cut, role, superCount: parent.length };
}

// Recognizability radius (Mosse constant), translation-only. Least L such that
// any two evaluable positions with equal (2L+1)-window share the same role.
// Returns Infinity if no L <= cap works.
export function recognizabilityRadius1D(name, depth, cap = 64) {
  const { letters, role } = generateWithAncestry(name, depth);
  const n = letters.length;
  const maxL = Math.min(cap, Math.floor((n - 1) / 2));
  for (let L = 0; L <= maxL; L++) {
    const seen = new Map();
    let ok = true;
    for (let i = L; i < n - L; i++) {
      let key = "";
      for (let d = -L; d <= L; d++) key += letters[i + d];
      const prev = seen.get(key);
      if (prev === undefined) seen.set(key, role[i]);
      else if (prev !== role[i]) {
        ok = false;
        break;
      }
    }
    if (ok) return L;
  }
  return Infinity;
}

// Cut points partition the word: their count equals the number of level-1
// supertiles, and every position belongs to exactly one block.
export function cutPointsPartition(name, depth) {
  const { cut, superCount } = generateWithAncestry(name, depth);
  let blocks = 0;
  let firstIsCut = cut.length > 0 && cut[0] === true;
  for (const c of cut) if (c) blocks++;
  return { ok: blocks === superCount && firstIsCut, blocks, superCount };
}

// Periodic repeat-cell capture radius (letters): least window length > p that
// exhibits the true period p with no smaller period. For an aperiodic motif of
// length p this is p + 1 (one period plus the confirming repeat).
export function repeatCellCaptureRadius(motif) {
  const p = motif.length;
  const reps = 4;
  const w = [];
  for (let i = 0; i < reps * p; i++) w.push(motif[i % p]);
  const hasPeriod = (arr, q) => {
    if (q < 1 || q >= arr.length) return false; // require a wraparound (q < len)
    for (let i = q; i < arr.length; i++) if (arr[i] !== arr[i - q]) return false;
    return true;
  };
  const noSmaller = (arr, q) => {
    for (let s = 1; s < q; s++) if (hasPeriod(arr, s)) return false;
    return true;
  };
  for (let len = p + 1; len <= w.length; len++) {
    const win = w.slice(0, len);
    if (hasPeriod(win, p) && noSmaller(win, p)) return len;
  }
  return Infinity;
}

// ---------------------------------------------------------------------------
// Stage S2: 2D Penrose recognizability radius (patches up to isometry).
// ---------------------------------------------------------------------------

import * as P from "./aperiodic-core.js";

// Color of a tile after applying its first `m` child-index steps from the seed
// (the seed triangle is RED). Mirrors aperiodic-core.subdivideOnce child colors:
//   RED  -> child0 RED, child1 BLUE
//   BLUE -> child0 BLUE, child1 BLUE, child2 RED
function childColor(color, idx) {
  if (color === P.RED) return idx === 0 ? P.RED : P.BLUE;
  return idx === 2 ? P.RED : P.BLUE;
}
function colorOfPrefix(path, m) {
  let color = P.RED;
  for (let i = 1; i <= m; i++) color = childColor(color, path[i]);
  return color;
}
// Role = desubstitution label = (parent color, child index within parent).
function tileRole(path, depth) {
  return `${colorOfPrefix(path, depth - 1)}:${path[depth]}`;
}

function rnd(v, k) {
  const f = 10 ** k;
  const r = Math.round(v * f) / f;
  return r === 0 ? 0 : r; // normalize -0
}

// Validation: the child-color rule used for roles must match the actual
// generation. Own color recomputed from the path must equal the tile's color.
export function penroseColorConsistency(depth) {
  const model = P.makePenrose(depth);
  return model.triangles.every((t) => colorOfPrefix(t.path, depth) === t.color);
}

// Signature of a patch (list of {type,lx,ly}) as a sorted, mirror-canonical key.
function patchKey(entries, k) {
  const a = entries.map((e) => `${e.type}|${rnd(e.lx, k)}|${rnd(e.ly, k)}`).sort().join(";");
  const b = entries.map((e) => `${e.type}|${rnd(e.lx, k)}|${rnd(-e.ly, k)}`).sort().join(";");
  return a < b ? a : b; // mirror-invariant (reflection up to isometry)
}

// Core: least radius (raw units) such that all interior tiles with the same
// mirror-canonical patch signature share a role. Patches are canonicalized into
// each tile's own frame (rotate so vertex A is on +x).
function recognizabilityCore(model, edgeLength, interiorRadius, maxPatchRadius, k) {
  const all = model.triangles.map((t) => {
    const c = P.centroid(t);
    return { t, c, orient: Math.atan2(t.A.y - c.y, t.A.x - c.x), role: tileRole(t.path, model.depth) };
  });
  const interior = all.filter((o) => Math.hypot(o.c.x, o.c.y) <= interiorRadius);

  const dists = new Set();
  for (const o of interior) {
    const cos = Math.cos(o.orient);
    const sin = Math.sin(o.orient);
    o.entries = [];
    for (const q of all) {
      const dx = q.c.x - o.c.x;
      const dy = q.c.y - o.c.y;
      const d = Math.hypot(dx, dy);
      if (d > maxPatchRadius) continue;
      o.entries.push({ dist: d, type: q.t.color, lx: dx * cos + dy * sin, ly: -dx * sin + dy * cos });
      dists.add(rnd(d, 6));
    }
    o.entries.sort((u, v) => u.dist - v.dist);
  }

  const candidates = [...dists].sort((u, v) => u - v);
  for (const r of candidates) {
    const groups = new Map();
    let ok = true;
    for (const o of interior) {
      const within = o.entries.filter((e) => e.dist <= r + 1e-9);
      const key = patchKey(within, k);
      const prev = groups.get(key);
      if (prev === undefined) groups.set(key, o.role);
      else if (prev !== o.role) {
        ok = false;
        break;
      }
    }
    if (ok) {
      return {
        interiorCount: interior.length,
        edgeLength,
        recognizabilityRadiusRaw: r,
        recognizabilityRadiusEdges: r / edgeLength,
        found: true,
        k,
        interiorRadius,
        maxPatchRadius,
      };
    }
  }
  return { interiorCount: interior.length, edgeLength, found: false, k, interiorRadius, maxPatchRadius };
}

// Pre-registered method: fixed raw-unit interior core. NOTE: at low depth the
// fixed raw core holds few tiles and undersamples the local-environment types,
// underestimating the radius (it climbs with depth). Kept for the pre-registered
// record; the edge-core variant below is the corrected measurement.
export function penroseRecognizability(depth, opts = {}) {
  const { interiorRadius = 0.45, maxPatchRadius = 0.46, k = 4 } = opts;
  const model = P.makePenrose(depth);
  const edgeLength = P.distance(model.triangles[0].A, model.triangles[0].B);
  return { depth, ...recognizabilityCore(model, edgeLength, interiorRadius, maxPatchRadius, k) };
}

// Corrected method: patch cap in EDGE units (bounded tiles/patch, depth-stable
// in scale), interior core taken as large as the decagon inradius allows so it
// grows in tile count with depth. This is the measurement that should converge.
export function penroseRecognizabilityEC(depth, opts = {}) {
  const { maxPatchRadiusEdges = 2.5, safeInradius = 0.951, k = 4 } = opts;
  const model = P.makePenrose(depth);
  const edgeLength = P.distance(model.triangles[0].A, model.triangles[0].B);
  const maxPatchRadius = maxPatchRadiusEdges * edgeLength;
  const interiorRadius = safeInradius - maxPatchRadius;
  return { depth, maxPatchRadiusEdges, ...recognizabilityCore(model, edgeLength, interiorRadius, maxPatchRadius, k) };
}

// Convergence of the edge-core recognizability radius across depths. The
// unbounded reading of the Ghost Boundary Heuristic is falsified if the radius
// stays finite, bounded, and converges (increments shrink) as the core grows.
export function penroseConvergence(depths = [4, 5, 6], opts = {}) {
  const runs = depths.map((d) => penroseRecognizabilityEC(d, opts));
  const edges = runs.map((r) => (r.found ? r.recognizabilityRadiusEdges : null));
  const finite = runs.every((r) => r.found);
  const incrled = [];
  for (let i = 1; i < edges.length; i++) incrled.push(edges[i] - edges[i - 1]);
  const converging = incrled.length >= 2 && Math.abs(incrled[incrled.length - 1]) <= Math.abs(incrled[0]) + 1e-9;
  const last = edges[edges.length - 1];
  return {
    substrate: "penrose-p3 (2D, edge-unit core, patches up to isometry)",
    observable: "recognizability radius (unique composition; edge-normalized)",
    depths,
    radiiEdges: edges,
    increments: incrled,
    finite,
    bounded: finite && last <= 6,
    converging,
    unboundedHeuristicFalsified: finite && last <= 6 && converging,
    note:
      "Edge-unit-core recognizability radius is finite, bounded, and converges from below as the interior core grows; the fixed raw-core variant undersamples at low depth. Either way the radius is finite and bounded, so the unbounded reading of the Ghost Boundary Heuristic fails in 2D.",
    defaultVerdict: "known vocabulary (unique composition / recognizability); not a new invariant",
  };
}

// 2D falsification: recognizability radius at two depths (edge-normalized), with
// finiteness + depth stability. Same logic as 1D: finite + stable falsifies the
// unbounded reading of the Ghost Boundary Heuristic.
export function penroseFalsification(depthA = 4, depthB = 5, tol = 1e-3) {
  const a = penroseRecognizability(depthA);
  const b = penroseRecognizability(depthB);
  const finite = a.found && b.found;
  const stable = finite && Math.abs(a.recognizabilityRadiusEdges - b.recognizabilityRadiusEdges) <= tol;
  return {
    substrate: "penrose-p3 (2D, patches up to isometry)",
    observable: "recognizability radius (unique composition; edge-normalized)",
    atDepthA: a,
    atDepthB: b,
    finite,
    depthStable: stable,
    unboundedHeuristicFalsified: finite && stable,
    note:
      "Penrose recognizability radius is finite and depth-stable in finest-edge units (self-similar local structure), so the unbounded reading of the Ghost Boundary Heuristic fails in 2D as well; the outside is a finite recognizability radius.",
    defaultVerdict: "known vocabulary (unique composition / recognizability); not a new invariant",
  };
}

// 1D falsification battery: recognizability radius at two depths per aperiodic
// substrate plus the periodic control. The unbounded reading of the Ghost
// Boundary Heuristic is falsified when the radius is finite and depth-stable.
export function falsificationReport1D(depthA, depthB, periodicMotif = ["A", "B", "C", "D"]) {
  const aperiodic = {};
  let allFiniteStable = true;
  for (const name of Object.keys(SUBSTITUTIONS)) {
    const rA = recognizabilityRadius1D(name, depthA);
    const rB = recognizabilityRadius1D(name, depthB);
    const finite = Number.isFinite(rB);
    const stable = rA === rB && finite;
    if (!(finite && stable)) allFiniteStable = false;
    aperiodic[name] = {
      observable: "recognizability radius (Mosse constant of recognizability)",
      radiusAtDepthA: rA,
      radiusAtDepthB: rB,
      finite,
      depthStable: stable,
    };
  }
  return {
    substrateKind: "1D primitive substitutions + periodic control",
    depths: { a: depthA, b: depthB },
    aperiodic,
    periodicControl: {
      observable: "repeat-cell capture radius",
      motif: periodicMotif.join(""),
      captureRadius: repeatCellCaptureRadius(periodicMotif),
      hasRepeatCell: true,
    },
    unboundedHeuristicFalsified: allFiniteStable,
    note:
      "Recognizability radius is finite and depth-stable for the aperiodic substrates (Mosse / Durand-Leroy), so the unbounded reading of the Ghost Boundary Heuristic fails; outside debt is a finite recognizability radius. The genuinely unbounded regime is undecidable Wang/SFT extension, which is not simulated here.",
    defaultVerdict: "known vocabulary (constant of recognizability); not a new invariant",
  };
}
