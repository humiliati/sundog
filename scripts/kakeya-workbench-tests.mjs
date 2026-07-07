// Acceptance tests for the Kakeya tiny finite-field workbench.
// Implements docs/kakeya/PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md §9, the
// Phase-3 exit criterion (§11.1). Run: `npm run kakeya:test`.

import * as K from "../kakeya/kakeya-core.js";

const Qs = K.SUPPORTED_Q; // [5, 7, 11]
let pass = 0;
const failures = [];
function check(name, cond) {
  if (cond) pass++;
  else failures.push(name);
}

const bitsEqual = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
const sameSet = (a, b) => a.size === b.size && [...a].every((p) => b.has(p));

for (const q of Qs) {
  const dirs = K.directions(q);
  const n = K.pointCount(q);

  // 1. Line cardinality — every precomputed line has exactly q points.
  let card = true;
  for (const d of dirs)
    for (let b = 0; b < q; b++) if (K.lineMask(d, b, q).size !== q) card = false;
  check(`q=${q} T1 line-cardinality`, card);

  // 2. Parallel partition — the q intercept lines of a direction are disjoint
  //    and their union is all of F_q^2.
  let part = true;
  for (const d of dirs) {
    const seen = new Set();
    for (let b = 0; b < q; b++)
      for (const p of K.lineMask(d, b, q)) {
        if (seen.has(p)) part = false;
        seen.add(p);
      }
    if (seen.size !== n) part = false;
  }
  check(`q=${q} T2 parallel-partition`, part);

  // 3. Nonparallel intersection — two lines of different directions meet in
  //    exactly one point.
  let inter = true;
  for (let i = 0; i < dirs.length && inter; i++)
    for (let j = i + 1; j < dirs.length && inter; j++)
      for (let b1 = 0; b1 < q && inter; b1++) {
        const L1 = K.lineMask(dirs[i], b1, q);
        for (let b2 = 0; b2 < q && inter; b2++) {
          const L2 = K.lineMask(dirs[j], b2, q);
          let common = 0;
          for (const p of L1) if (L2.has(p)) common++;
          if (common !== 1) inter = false;
        }
      }
  check(`q=${q} T3 nonparallel-intersection`, inter);

  // 4. Empty set covers zero directions.
  check(`q=${q} T4 empty`, K.shadowSummary(q, K.bEmpty()).directionsCovered === 0);

  // 5. Single line covers exactly its own direction.
  const sl = K.bSingleLine(q, dirs[0], 0);
  const slSum = K.shadowSummary(q, sl);
  check(`q=${q} T5 single-line`, slSum.directionsCovered === 1 && slSum.bits[0] === 1);

  // 6. Whole plane covers all q + 1 directions.
  check(`q=${q} T6 whole-plane`, K.shadowSummary(q, K.bWholePlane(q)).directionsCovered === q + 1);

  // 7. Whole plane minus one point still covers all directions (the
  //    shadow-does-not-reconstruct demonstration).
  let wmo = true;
  for (const idx of [0, Math.floor(n / 2), n - 1])
    if (K.shadowSummary(q, K.bWholeMinusOne(q, idx)).directionsCovered !== q + 1) wmo = false;
  check(`q=${q} T7 whole-minus-one`, wmo);

  // 8. Shadow collision — two distinct bodies produce the same primary shadow.
  const a = K.bSingleLine(q, dirs[0], 0);
  const b = K.bSingleLine(q, dirs[0], 1);
  check(
    `q=${q} T8 shadow-collision`,
    bitsEqual(K.shadowBitset(q, a), K.shadowBitset(q, b)) && !sameSet(a, b),
  );

  // 8b. Shadow collision with different body sizes. Adding one off-line point
  //     to a full line changes the body but does not cover a new full line.
  const linePlusPoint = new Set(a);
  const extraPoint = [...Array(n).keys()].find((p) => !a.has(p));
  linePlusPoint.add(extraPoint);
  check(
    `q=${q} T8b shadow-collision-different-size`,
    bitsEqual(K.shadowBitset(q, a), K.shadowBitset(q, linePlusPoint)) &&
      !sameSet(a, linePlusPoint) &&
      a.size !== linePlusPoint.size,
  );

  // 8c. The complete shadow is also many-to-one and does not recover body size.
  const whole = K.bWholePlane(q);
  const wholeMinus = K.bWholeMinusOne(q, 0);
  check(
    `q=${q} T8c complete-shadow-different-size`,
    bitsEqual(K.shadowBitset(q, whole), K.shadowBitset(q, wholeMinus)) &&
      !sameSet(whole, wholeMinus) &&
      whole.size !== wholeMinus.size,
  );

  // 8d. Structured line-extension count. For each direction, each of q lines
  //     and that line plus one outside point cast the same one-direction shadow.
  const expectedPerDirection = q * (n - q + 1);
  let structuredCounts = true;
  for (let dirIndex = 0; dirIndex < dirs.length; dirIndex++) {
    const sigCounts = new Map();
    const d = dirs[dirIndex];
    for (let intercept = 0; intercept < q; intercept++) {
      const line = K.lineMask(d, intercept, q);
      const lineBits = K.shadowBitset(q, line).join("");
      sigCounts.set(lineBits, (sigCounts.get(lineBits) ?? 0) + 1);
      for (let p = 0; p < n; p++) {
        if (line.has(p)) continue;
        const extended = new Set(line);
        extended.add(p);
        const extendedBits = K.shadowBitset(q, extended).join("");
        sigCounts.set(extendedBits, (sigCounts.get(extendedBits) ?? 0) + 1);
      }
    }
    const expectedBits = dirs.map((_, i) => (i === dirIndex ? 1 : 0)).join("");
    structuredCounts =
      structuredCounts &&
      sigCounts.size === 1 &&
      sigCounts.get(expectedBits) === expectedPerDirection;
  }
  check(`q=${q} T8d structured-line-extension-count`, structuredCounts);

  // 8e. Lemma boundary: a line plus up to q - 2 off-line points still covers
  //     exactly the original direction.
  let safeBoundary = true;
  for (let dirIndex = 0; dirIndex < dirs.length; dirIndex++) {
    const d = dirs[dirIndex];
    const expectedBits = dirs.map((_, i) => (i === dirIndex ? 1 : 0));
    for (let intercept = 0; intercept < q; intercept++) {
      const line = K.lineMask(d, intercept, q);
      const body = new Set(line);
      for (let p = 0; p < n && body.size < q + (q - 2); p++) {
        if (!line.has(p)) body.add(p);
      }
      safeBoundary = safeBoundary && bitsEqual(K.shadowBitset(q, body), expectedBits);
    }
  }
  check(`q=${q} T8e line-plus-q-minus-2-safe`, safeBoundary);

  // 8f. First break: q - 1 off-line points can complete a second direction.
  let firstBreak = true;
  for (let dirIndex = 0; dirIndex < dirs.length; dirIndex++) {
    const d = dirs[dirIndex];
    const otherIndex = (dirIndex + 1) % dirs.length;
    const other = dirs[otherIndex];
    for (let intercept = 0; intercept < q; intercept++) {
      const line = K.lineMask(d, intercept, q);
      const crossing = K.lineMask(other, 0, q);
      const body = new Set(line);
      for (const p of crossing) body.add(p);
      const bits = K.shadowBitset(q, body);
      firstBreak =
        firstBreak &&
        body.size === q + (q - 1) &&
        bits[dirIndex] === 1 &&
        bits[otherIndex] === 1 &&
        bits.reduce((sum, bit) => sum + bit, 0) >= 2;
    }
  }
  check(`q=${q} T8f line-plus-q-minus-1-break`, firstBreak);

  // 8g. Activation floor: for a bare line, every other direction is exactly
  //     q - 1 added points away (min over its q intercept lines).
  const activationCost = (dir, body) => {
    let best = Infinity;
    for (let b = 0; b < q; b++) {
      let missing = 0;
      for (const p of K.lineMask(dir, b, q)) if (!body.has(p)) missing++;
      best = Math.min(best, missing);
    }
    return best;
  };
  const bareLine = K.lineMask(dirs[0], 0, q);
  let activationFloor = true;
  for (let dirIndex = 1; dirIndex < dirs.length; dirIndex++) {
    activationFloor = activationFloor && activationCost(dirs[dirIndex], bareLine) === q - 1;
  }
  check(`q=${q} T8g bare-line-activation-q-minus-1`, activationFloor);

  // 8h. Star ladder: k concurrent lines (k <= q - 1) put every missing
  //     direction exactly q - k points away; the k = q star (pencil minus one
  //     direction) already covers all q + 1 directions.
  let starLadder = true;
  for (const k of [2, q - 1]) {
    const star = new Set();
    for (let i = 0; i < k; i++) for (const p of K.lineMask(dirs[i], 0, q)) star.add(p);
    for (let dirIndex = k; dirIndex < dirs.length; dirIndex++) {
      starLadder = starLadder && activationCost(dirs[dirIndex], star) === q - k;
    }
  }
  const pencilMinusOne = new Set();
  for (let m = 0; m < q; m++) for (const p of K.lineMask(dirs[m], 0, q)) pencilMinusOne.add(p);
  starLadder =
    starLadder && K.shadowSummary(q, pencilMinusOne).directionsCovered === q + 1;
  check(`q=${q} T8h star-ladder-activation`, starLadder);

  // 8i. Joint-vs-marginal pair law: from empty, lighting any two directions
  //     jointly costs exactly 2q - 1 (one shared point), gap = 1.
  const jointPair = (dirA, dirB) => {
    let best = Infinity;
    for (let b1 = 0; b1 < q; b1++) {
      const L1 = K.lineMask(dirA, b1, q);
      for (let b2 = 0; b2 < q; b2++) {
        const union = new Set(L1);
        for (const p of K.lineMask(dirB, b2, q)) union.add(p);
        best = Math.min(best, union.size);
      }
    }
    return best;
  };
  check(
    `q=${q} T8i joint-pair-2q-minus-1`,
    jointPair(dirs[0], dirs[1]) === 2 * q - 1 &&
      jointPair(dirs[0], dirs[q]) === 2 * q - 1,
  );

  // 8j. Parabola witness: the q tangents of y = x^2 (slope 2t, intercept -t^2)
  //     plus one vertical form a complete Kakeya set of size
  //     q(q+1)/2 + (q-1)/2 (the odd-q planar minimum, imported anchor).
  const parabola = new Set();
  for (let t = 0; t < q; t++) {
    const slope = (2 * t) % q;
    const intercept = ((-(t * t)) % q + q) % q;
    for (const p of K.lineMask(dirs[slope], intercept, q)) parabola.add(p);
  }
  for (const p of K.lineMask(dirs[q], 0, q)) parabola.add(p);
  check(
    `q=${q} T8j parabola-minimal-kakeya`,
    parabola.size === (q * (q + 1)) / 2 + (q - 1) / 2 &&
      K.shadowSummary(q, parabola).directionsCovered === q + 1,
  );

  // 9. Greedy line-cover construction covers all directions.
  check(
    `q=${q} T9 greedy-complete`,
    K.shadowSummary(q, K.bGreedyLineCover(q)).directionsCovered === q + 1,
  );

  // 10. Export guard — the shadow export carries no point/intercept/mask list.
  const ex = K.exportShadow(q, K.bWholePlane(q));
  const forbidden = ["point", "selected", "intercept", "mask", "line", "witness", "member"];
  const exportClean = !Object.keys(ex).some((k) =>
    forbidden.some((f) => k.toLowerCase().includes(f)),
  );
  check(`q=${q} T10 export-guard`, exportClean);

  // Cross-check: Dvir floor is consistent for the whole plane (it is a theorem,
  // so this must hold for any complete set).
  check(
    `q=${q} Tx dvir-consistent`,
    K.shadowSummary(q, K.bWholePlane(q)).dvirFloorConsistent === true,
  );
}

console.log(`KAKEYA_WORKBENCH_TESTS q={${Qs.join(",")}} pass=${pass} fail=${failures.length}`);
for (const f of failures) console.log("  FAIL " + f);
process.exit(failures.length === 0 ? 0 : 1);
