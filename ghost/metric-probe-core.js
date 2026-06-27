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
