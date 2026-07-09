#!/usr/bin/env node
// PHASE3R M1-M2 - floor-mechanism probe (Redei grounding + k-star floor scaling).
//
// M1 (Redei grounding): for a 4-star completion, the Redei polynomial is
//   R(X,Y) = prod_{p=(a,b) in U} (X + a Y - b).
// For a fixed slope y, R(X,y) = prod_p (X - c_p(y)) with c_p(y) = b - a y the
// intercept of the slope-y line through p. A covered direction (a full line of
// slope y in U) forces its intercept value to occur with multiplicity >= q,
// i.e. (X - c(y))^q | R(X,y). The pivot O=(0,0) has intercept 0 in EVERY
// direction, and the four star lines through O all have intercept 0, so the
// root c=0 is shared at high multiplicity across the 4 star directions - the
// "pencil factor". M1 verifies the covered-line divisibility and measures the
// c=0 multiplicity across star vs non-star directions.
//
// M2 (floor scaling): the exact minimal completion excess ex_min of a k-star
// (k concurrent lines) probes how the pencil forces the lower-order term.
// Measured exhaustively (full B&B) over the cross-ratio orbits at small q,
// per k, to (a) confirm the k=4 floor = (q-3)/2 and (b) reveal the q-scaling
// of the relative term. DECISIVE: if ex_min(4,q) = (q-3)/2 and scales linearly
// with slope 1/2, the pencil mechanism is confirmed and the relative-BM target
// is the term q-2; if it scales otherwise, F's mechanism differs.
//
// Scope-support instrument. No proof, no Euclidean claim.

import fs from "node:fs";
import path from "node:path";
import * as Core from "../kakeya/kakeya-core.js";

const DEFAULT_OUT = path.join("results", "kakeya", "floor-mechanism");
const INF = -1;

function mod(x, q) { return ((x % q) + q) % q; }
function inv(x, q) { for (let i = 1; i < q; i++) if ((x * i) % q === 1) return i; throw new Error("noinv"); }

// --- orbits (for the k=4 floor) ---
function crossRatio(a, b, c, d, q) { const diff = (x, y) => (x === INF || y === INF ? 1 : mod(x - y, q)); return (diff(a, c) * diff(b, d) * inv((diff(a, d) * diff(b, c)) % q, q)) % q; }
function sixSet(l, q) { const s = new Set([l, inv(l, q), mod(1 - l, q), inv(mod(1 - l, q), q), mod(l * inv(mod(l - 1, q), q), q), mod(mod(l - 1, q) * inv(l, q), q)]); return [...s].sort((x, y) => x - y); }
function fourStarOrbitReps(q) {
  const dc = q + 1, dv = (i) => (i === q ? INF : i), reps = new Map();
  for (let a = 0; a < dc; a++) for (let b = a + 1; b < dc; b++) for (let c = b + 1; c < dc; c++) for (let d = c + 1; d < dc; d++) {
    const key = sixSet(crossRatio(dv(a), dv(b), dv(c), dv(d), q), q).join(",");
    if (!reps.has(key)) reps.set(key, [a, b, c, d]);
  }
  return [...reps.values()];
}

// --- exact minimal sacrifice of a k-star completion (exhaustive B&B) ---
function exactMinSacrifice(q, starDirs, nodeBudget = Infinity) {
  const dirs = Core.directions(q), n = Core.pointCount(q);
  const star = new Set();
  for (const i of starDirs) for (const p of Core.lineMask(dirs[i], 0, q)) star.add(p);
  const bits = Core.shadowBitset(q, star);
  const targets = [];
  for (let i = 0; i < dirs.length; i++) if (bits[i] === 0) targets.push(i);
  const linePts = targets.map((i) => { const per = []; for (let b = 0; b < q; b++) per.push([...Core.lineMask(dirs[i], b, q)]); return per; });
  const cnt = new Uint8Array(n); let sac = 0;
  const add = (pts) => { for (const p of pts) { const m = cnt[p]; if (m >= 2) sac += m - 1; cnt[p] = m + 1; } };
  const rem = (pts) => { for (const p of pts) { const m = cnt[p]; if (m >= 3) sac -= m - 2; cnt[p] = m - 1; } };
  for (const i of starDirs) add([...Core.lineMask(dirs[i], 0, q)]);
  const k = targets.length; let best = Infinity; let nodes = 0; let exhausted = true;
  const rec = (level) => {
    if (nodes >= nodeBudget) { exhausted = false; return; }
    if (sac >= best) return;
    if (level === k) { best = sac; return; }
    const order = [];
    for (let b = 0; b < q; b++) { nodes++; add(linePts[level][b]); order.push([sac, b]); rem(linePts[level][b]); }
    order.sort((x, y) => x[0] - y[0]);
    for (const [s, b] of order) { if (s >= best) break; add(linePts[level][b]); rec(level + 1); rem(linePts[level][b]); }
  };
  rec(0);
  return { sacrifice: best, exhausted };
}

// --- M1: Redei intercept-multiplicity structure of a completion ---
// Recover a minimal 4-star completion (use the exact B&B's winning assignment).
function exactMinCompletion(q, starDirs) {
  const dirs = Core.directions(q), n = Core.pointCount(q);
  const star = new Set();
  for (const i of starDirs) for (const p of Core.lineMask(dirs[i], 0, q)) star.add(p);
  const bits = Core.shadowBitset(q, star);
  const targets = [];
  for (let i = 0; i < dirs.length; i++) if (bits[i] === 0) targets.push(i);
  const linePts = targets.map((i) => { const per = []; for (let b = 0; b < q; b++) per.push([...Core.lineMask(dirs[i], b, q)]); return per; });
  const cnt = new Uint8Array(n); let sac = 0;
  const add = (pts) => { for (const p of pts) { const m = cnt[p]; if (m >= 2) sac += m - 1; cnt[p] = m + 1; } };
  const rem = (pts) => { for (const p of pts) { const m = cnt[p]; if (m >= 3) sac -= m - 2; cnt[p] = m - 1; } };
  for (const i of starDirs) add([...Core.lineMask(dirs[i], 0, q)]);
  const k = targets.length; let best = Infinity; let bestAssign = null; const choice = new Array(k).fill(0);
  const rec = (level) => {
    if (sac >= best) return;
    if (level === k) { best = sac; bestAssign = [...choice]; return; }
    const order = [];
    for (let b = 0; b < q; b++) { add(linePts[level][b]); order.push([sac, b]); rem(linePts[level][b]); }
    order.sort((x, y) => x[0] - y[0]);
    for (const [s, b] of order) { if (s >= best) break; choice[level] = b; add(linePts[level][b]); rec(level + 1); rem(linePts[level][b]); }
  };
  rec(0);
  // Build the point set U.
  const U = new Set();
  for (const i of starDirs) for (const p of Core.lineMask(dirs[i], 0, q)) U.add(p);
  targets.forEach((i, j) => { for (const p of Core.lineMask(dirs[i], bestAssign[j], q)) U.add(p); });
  return { U, targets, assign: bestAssign, starDirs, sacrifice: best };
}

// intercept of point p=(x,y) under slope s (finite): c = y - s x; vertical: c = x.
function interceptOf(p, s, q) {
  const { x, y } = Core.indexToXY(p, q);
  return s === q ? x : mod(y - s * x, q);
}
function redeiStructure(q, comp) {
  const dirs = Core.directions(q);
  const U = [...comp.U];
  const perDir = [];
  for (let di = 0; di < dirs.length; di++) {
    const s = dirs[di].kind === "inf" ? q : dirs[di].m;
    const multByC = new Map();
    for (const p of U) { const c = interceptOf(p, s, q); multByC.set(c, (multByC.get(c) ?? 0) + 1); }
    const isStar = comp.starDirs.includes(di);
    // covered-line intercept = the value with multiplicity >= q
    let coveredMult = 0, coveredC = null;
    for (const [c, m] of multByC) if (m > coveredMult) { coveredMult = m; coveredC = c; }
    perDir.push({
      dirIndex: di, isStar,
      coveredIntercept: coveredC, coveredMult,
      divisibleQ: coveredMult >= q, // (X - c)^q | R(X,y)
      c0Mult: multByC.get(0) ?? 0, // multiplicity of the shared pencil root c=0
    });
  }
  return perDir;
}

function main() {
  const outDir = process.argv.includes("--out") ? process.argv[process.argv.indexOf("--out") + 1] : DEFAULT_OUT;

  // ---- M1: Redei grounding on a 4-star completion at q in {5,7,11,13} ----
  const m1 = [];
  for (const q of [5, 7, 11, 13]) {
    // canonical harmonic 4-star (or first orbit) - use directions {0,1,2,3}
    const comp = exactMinCompletion(q, [0, 1, 2, 3]);
    const rs = redeiStructure(q, comp);
    const allCoveredDivisible = rs.every((r) => r.divisibleQ);
    const starC0 = rs.filter((r) => r.isStar).map((r) => r.c0Mult);
    const nonStarC0 = rs.filter((r) => !r.isStar).map((r) => r.c0Mult);
    m1.push({
      q, quad: [0, 1, 2, 3], sacrifice: comp.sacrifice,
      allCoveredLinesDivisibleByQ: allCoveredDivisible,
      starDirC0Mult: starC0, // expect >= q at star dirs (line through O)
      nonStarC0MultRange: [Math.min(...nonStarC0), Math.max(...nonStarC0)],
      pencilSharedRoot: starC0.every((m) => m >= q), // the 4 star dirs share c=0 at mult>=q
    });
  }

  // ---- M2: k-star floor scaling ----
  // For k=4, the FLOOR = min over cross-ratio orbits of ex_min (exhaustive).
  // For k=3,5, canonical k-star ex to reveal q-scaling.
  const m2 = { k4Floor: [], kScaling: [] };
  for (const q of [5, 7, 11, 13]) {
    // k=4 floor (min over orbits)
    let floorSac = Infinity;
    for (const quad of fourStarOrbitReps(q)) {
      const { sacrifice } = exactMinSacrifice(q, quad);
      floorSac = Math.min(floorSac, sacrifice);
    }
    const exFloor = floorSac - (q - 1) / 2;
    // The FLOOR is a lower bound ex >= (q-3)/2. It holds (>=) at every q; it is
    // TIGHT (==) once a LOW orbit exists (q >= 7). q=5 has only the harmonic
    // (HIGH) orbit, so the floor is loose there (2 >= 1) - not a violation.
    m2.k4Floor.push({
      q, exFloor, floorBound: (q - 3) / 2,
      holds: exFloor >= (q - 3) / 2,
      tight: exFloor === (q - 3) / 2,
    });
  }
  // canonical k-stars {0,1,...,k-1}, k=3,4,5, per q (skip if depth too large)
  for (const k of [3, 4, 5]) {
    for (const q of [5, 7, 11, 13]) {
      if (k >= q) continue;
      const depth = q + 1 - k;
      const budget = depth > 10 ? 300_000_000 : Infinity; // cap deep cases
      const { sacrifice, exhausted } = exactMinSacrifice(q, [...Array(k).keys()], budget);
      m2.kScaling.push({
        k, q, depth, exhausted,
        sacrifice, ex: sacrifice - (q - 1) / 2,
        pivotCost: ((k - 1) * (k - 2)) / 2,
        exMinusPivotEx: sacrifice - (q - 1) / 2 - ((k - 1) * (k - 2)) / 2 + ((k - 1) * (k - 2)) / 2 - 3 + 3, // keep raw; interpret in receipt
      });
    }
  }

  const m1Pass = m1.every((r) => r.allCoveredLinesDivisibleByQ && r.pencilSharedRoot);
  const k4FloorHolds = m2.k4Floor.every((r) => r.holds);
  const k4FloorTightForLarge = m2.k4Floor.filter((r) => r.q >= 7).every((r) => r.tight);
  // The pencil mechanism: 3-star reaches BM (ex=0), 4-star floor jumps to (q-3)/2.
  const threeStarFree = m2.kScaling.filter((r) => r.k === 3).every((r) => r.ex === 0);
  const k4FloorPass = k4FloorHolds && k4FloorTightForLarge && threeStarFree;
  const manifest = {
    artifactId: "KAK-PHASE3R-M1M2-FLOOR-MECHANISM",
    generatedAt: new Date().toISOString(),
    status: "internal scope-support probe (PHASE3R M1-M2)",
    command: "node scripts/kakeya-floor-mechanism.mjs",
    M1: { note: "Redei grounding: covered-line divisibility + pencil shared-root c=0", results: m1, pass: m1Pass },
    M2: {
      note: "k-star floor scaling: 3-star reaches BM (ex=0); 4-star floor = (q-3)/2 (lower bound, tight for q>=7)",
      ...m2,
      k4FloorHolds, k4FloorTightForLarge, threeStarFree,
      mechanismConfirmed: k4FloorPass,
    },
    falsifier: {
      name: "FLOOR_MECHANISM_MISMATCH",
      description: "Fires if covered lines are not (X-c)^q-divisible, the 4 star directions do not share the c=0 root at mult>=q, the 3-star does not reach BM (ex=0), or the 4-star floor drops below (q-3)/2 / is not tight for q>=7.",
      status: m1Pass && k4FloorPass ? "clear" : "fired",
    },
  };
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");

  console.log("== M1 Redei grounding (4-star {0,1,2,3}) ==");
  for (const r of m1) console.log(`  q=${String(r.q).padEnd(3)} sac=${r.sacrifice} coveredDivisibleByQ=${r.allCoveredLinesDivisibleByQ} starC0mult=[${r.starDirC0Mult}] nonStarC0=${r.nonStarC0MultRange.join("..")} pencilSharedRoot=${r.pencilSharedRoot}`);
  console.log("== M2a k=4 floor (min over orbits) ==");
  for (const r of m2.k4Floor) console.log(`  q=${String(r.q).padEnd(3)} exFloor=${r.exFloor} floorBound(q-3)/2=${r.floorBound} holds=${r.holds}${r.tight ? " TIGHT" : " (loose: no LOW orbit)"}`);
  console.log("== M2b k-star scaling (canonical {0..k-1}) ==");
  for (const r of m2.kScaling) console.log(`  k=${r.k} q=${String(r.q).padEnd(3)} depth=${r.depth} ex=${r.ex} sac=${r.sacrifice} pivotCost=${r.pivotCost} ${r.exhausted ? "" : "(BUDGET-BOUNDED)"}`);
  console.log(`KAK_FLOOR_MECHANISM M1=${m1Pass ? "pass" : "fail"} k4floor=${k4FloorPass ? "pass" : "fail"} falsifier=${manifest.falsifier.status} out=${outDir}`);
  process.exit(manifest.falsifier.status === "clear" ? 0 : 1);
}
main();
