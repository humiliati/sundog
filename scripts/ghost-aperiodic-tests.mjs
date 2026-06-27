// Acceptance tests for the Ghost Phase 3 aperiodic reader (Penrose P3).
// Implements docs/ghost/PHASE3_APERIODIC_READER_SPEC.md section 7.
// Run: `npm run ghost:aperiodic:test`.

import * as G from "../ghost/aperiodic-core.js";

let pass = 0;
const failures = [];

function check(name, cond) {
  if (cond) pass++;
  else failures.push(name);
}

function hasForbiddenKey(obj, forbidden) {
  if (!obj || typeof obj !== "object") return false;
  for (const key of Object.keys(obj)) {
    if (forbidden.some((f) => key.toLowerCase().includes(f))) return true;
    if (hasForbiddenKey(obj[key], forbidden)) return true;
  }
  return false;
}

const phi = G.GOLDEN;
const near = (a, b, tol) => Math.abs(a - b) <= tol;

// T1 seed wheel
const wheel = G.baseWheel();
const wheelCounts = G.countsByType(wheel);
const apexAtOrigin = wheel.every((t) => Math.hypot(t.A.x, t.A.y) < 1e-9);
const farOnUnitCircle = wheel.every(
  (t) => near(Math.hypot(t.B.x, t.B.y), 1, 1e-9) && near(Math.hypot(t.C.x, t.C.y), 1, 1e-9),
);
check("T1 seed wheel is 10 RED triangles", wheel.length === 10 && wheelCounts.red === 10 && wheelCounts.blue === 0);
check("T1 seed apex at origin, far vertices on unit circle", apexAtOrigin && farOnUnitCircle);

// T2 subdivision combinatorics
const oneRed = [{ color: G.RED, A: { x: 0, y: 0 }, B: { x: 1, y: 0 }, C: { x: 0, y: 1 }, path: [0] }];
const oneBlue = [{ color: G.BLUE, A: { x: 0, y: 0 }, B: { x: 1, y: 0 }, C: { x: 0, y: 1 }, path: [0] }];
const redKids = G.countsByType(G.subdivideOnce(oneRed));
const blueKids = G.countsByType(G.subdivideOnce(oneBlue));
check("T2 RED -> {1 RED, 1 BLUE}", redKids.red === 1 && redKids.blue === 1);
check("T2 BLUE -> {1 RED, 2 BLUE}", blueKids.red === 1 && blueKids.blue === 2);

// T3 total count grows by phi^2 per inflation step
let growthOk = true;
let prev = G.makePenrose(4).triangles.length;
for (let d = 5; d <= 6; d++) {
  const cur = G.makePenrose(d).triangles.length;
  if (!near(cur / prev, phi * phi, 0.01)) growthOk = false;
  prev = cur;
}
check("T3 tile count grows by phi^2 per step", growthOk);

// T4 / T5 ratios at depth 6
const model6 = G.makePenrose(6);
const c6 = G.countsByType(model6.triangles);
check("T4 RED:BLUE ratio -> phi", near(c6.blue / c6.red, phi, 0.01));
check("T5 thick:thin rhombus ratio -> phi", near(G.thickThinRatio(model6.triangles), phi, 0.01));

// T6 ancestry path length and supertile partition
const model5 = G.makePenrose(5);
const pathLenOk = model5.triangles.every((t) => t.path.length === model5.depth + 1);
const groups = G.supertilesAtLevel(model5, 3);
const grouped = groups.reduce((s, g) => s + g.tiles.length, 0);
const keysUnique = new Set(groups.map((g) => g.key)).size === groups.length;
check("T6 finest path length == depth + 1", pathLenOk);
check("T6 supertile grouping partitions all finest tiles", grouped === model5.triangles.length && keysUnique);

// T7 window analysis consistency
const a = G.analyzeWindow(model5, { x: 0, y: 0 }, 0.6, 3);
check(
  "T7 contained + crossing == supertiles touching",
  a.supertilesContained + a.supertilesCrossing === a.supertilesTouching && a.supertilesTouching > 0,
);
check("T7 tiles in window is within total", a.tilesInWindow > 0 && a.tilesInWindow <= a.tileCounts.total);

// T8 export discipline
const exported = G.exportReaderAnalysis(model5, { x: 0, y: 0 }, 0.6, 3);
const forbidden = ["theorem", "proof", "invariant", "conjecture", "claim"];
check("T8 export avoids theorem-shaped keys", !hasForbiddenKey(exported, forbidden));
check("T8 export keeps cliff inactive", exported.cliff.active === false);

console.log(`GHOST_APERIODIC_TESTS pass=${pass} fail=${failures.length}`);
for (const failure of failures) console.log("  FAIL " + failure);
process.exit(failures.length === 0 ? 0 : 1);
