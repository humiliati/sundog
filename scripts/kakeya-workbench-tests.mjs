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
