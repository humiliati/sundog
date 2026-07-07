#!/usr/bin/env node
// PHASE3L step 1 - anatomy of optimal 4-star completions (Track 1).
//
// Records the actual optimal completions of one representative 4-star per
// cross-ratio orbit at the cheap exact fields q in {5, 7, 11, 13}, and
// reports their geometry: chosen intercepts, point-multiplicity profile,
// sacrifice breakdown (which concurrencies are burned and where), pivot
// multiplicity, and a dual-conic incidence test on the chosen lines. At
// q in {5, 7} ALL optimal completions are enumerated exhaustively (the
// missing-direction count is tiny); at q in {11, 13} one witness per orbit.
//
// This is an internal analysis instrument feeding the PHASE3L construction
// extraction. Out-of-register discipline as in PHASE3J/3K where applicable;
// all fields here are exact-solver territory. No Euclidean claim.

import fs from "node:fs";
import path from "node:path";
import * as Core from "../kakeya/kakeya-core.js";

const DEFAULT_OUT = path.join("results", "kakeya", "star-completion-anatomy");
const FIELDS = [5, 7, 11, 13];
const EXHAUSTIVE_Q = [5, 7]; // all-optima enumeration
const ALL_OPTIMA_CAP = 2000;
const INF = -1;

function mod(x, q) {
  return ((x % q) + q) % q;
}
function inv(x, q) {
  for (let i = 1; i < q; i++) if ((x * i) % q === 1) return i;
  throw new Error(`no inverse of ${x} mod ${q}`);
}
function bmMinimum(q) {
  return (q * (q + 1)) / 2 + (q - 1) / 2;
}

// --- orbits (PHASE3I/3J machinery) ------------------------------------------------

function crossRatio(a, b, c, d, q) {
  const diff = (x, y) => (x === INF || y === INF ? 1 : mod(x - y, q));
  return (diff(a, c) * diff(b, d) * inv((diff(a, d) * diff(b, c)) % q, q)) % q;
}
function sixSet(l, q) {
  const s = new Set();
  s.add(l);
  s.add(inv(l, q));
  const om = mod(1 - l, q);
  s.add(om);
  s.add(inv(om, q));
  s.add(mod(l * inv(mod(l - 1, q), q), q));
  s.add(mod(mod(l - 1, q) * inv(l, q), q));
  return [...s].sort((x, y) => x - y);
}
function orbitLabelOf(l, q) {
  const six = sixSet(l, q);
  const harmonic = [2, mod((q + 1) / 2, q), q - 1].sort((x, y) => x - y);
  if (six.length === 3 && six.join(",") === harmonic.join(",")) return "harmonic";
  if (six.length === 2 && six.every((v) => mod(v * v - v + 1, q) === 0)) return "equianharmonic";
  return "generic";
}

// One lex-first representative quadruple per orbit.
function orbitReps(q) {
  const dirCount = q + 1;
  const dirValue = (i) => (i === q ? INF : i);
  const reps = new Map();
  for (let a = 0; a < dirCount; a++)
    for (let b = a + 1; b < dirCount; b++)
      for (let c = b + 1; c < dirCount; c++)
        for (let d = c + 1; d < dirCount; d++) {
          const l = crossRatio(dirValue(a), dirValue(b), dirValue(c), dirValue(d), q);
          const key = sixSet(l, q).join(",");
          if (!reps.has(key)) reps.set(key, { label: orbitLabelOf(l, q), quad: [a, b, c, d] });
        }
  // Disambiguate multiple generics by key order.
  const byBase = new Map();
  for (const key of [...reps.keys()].sort()) {
    const base = reps.get(key).label;
    byBase.set(base, (byBase.get(base) ?? 0) + 1);
  }
  const counters = new Map();
  const out = [];
  for (const key of [...reps.keys()].sort()) {
    const r = reps.get(key);
    let label = r.label;
    if (byBase.get(r.label) > 1) {
      const n = (counters.get(r.label) ?? 0) + 1;
      counters.set(r.label, n);
      label = `${r.label}-${String.fromCharCode(96 + n)}`;
    }
    out.push({ label, key, quad: r.quad });
  }
  return out;
}

// --- solver with choice tracking (PHASE3K solver + witness intercepts) ------------

function wordCount(q) {
  return Math.ceil(Core.pointCount(q) / 32);
}
function popcount32(v) {
  v = v - ((v >> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
  return (((v + (v >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24;
}

function starBody(q, dirIndexes) {
  const dirs = Core.directions(q);
  const body = new Set();
  for (const i of dirIndexes) for (const p of Core.lineMask(dirs[i], 0, q)) body.add(p);
  return body;
}

// Exact completion with witness choices. exhaustive=true additionally
// enumerates ALL optimal intercept vectors (capped).
function solveWithWitness(q, K, exhaustive) {
  const dirs = Core.directions(q);
  const words = wordCount(q);
  const bits = Core.shadowBitset(q, K);
  const targets = [];
  for (let i = 0; i < dirs.length; i++) if (bits[i] === 0) targets.push(i);

  const addMasks = targets.map((i) => {
    const perB = [];
    for (let b = 0; b < q; b++) {
      const mask = new Uint32Array(words);
      let size = 0;
      for (const p of Core.lineMask(dirs[i], b, q)) {
        if (!K.has(p)) {
          mask[p >> 5] |= 1 << (p & 31);
          size++;
        }
      }
      perB.push({ mask, size });
    }
    return perB;
  });

  const k = targets.length;
  const unions = [];
  for (let level = 0; level <= k; level++) unions.push(new Uint32Array(words));
  const choice = new Array(k).fill(-1);
  let best = Infinity;
  let bestChoice = null;
  const optima = [];

  function dynBound(level, union) {
    const r = k - level;
    if (r === 0) return 0;
    let sum = 0;
    let max = 0;
    for (let j = level; j < k; j++) {
      let mn = Infinity;
      const perB = addMasks[j];
      for (let b = 0; b < q; b++) {
        let added = 0;
        const m = perB[b].mask;
        for (let w = 0; w < words; w++) added += popcount32(m[w] & ~union[w]);
        if (added < mn) mn = added;
        if (mn === 0) break;
      }
      sum += mn;
      if (mn > max) max = mn;
    }
    return Math.max(max, sum - (r * (r - 1)) / 2);
  }

  function rec(level, count) {
    const cutoff = exhaustive ? best + 1 : best; // allow ties when enumerating
    if (count + dynBound(level, unions[level]) >= cutoff) return;
    if (level === k) {
      if (count < best) {
        best = count;
        bestChoice = [...choice];
        if (exhaustive) {
          optima.length = 0;
          optima.push([...choice]);
        }
      } else if (exhaustive && count === best && optima.length < ALL_OPTIMA_CAP) {
        optima.push([...choice]);
      }
      return;
    }
    const prev = unions[level];
    const next = unions[level + 1];
    for (let b = 0; b < q; b++) {
      let newCount = 0;
      const m = addMasks[level][b].mask;
      for (let w = 0; w < words; w++) {
        next[w] = prev[w] | m[w];
        newCount += popcount32(next[w]);
      }
      if (newCount >= (exhaustive ? best + 1 : best)) continue;
      choice[level] = b;
      rec(level + 1, newCount);
    }
  }

  rec(0, 0);
  return { targets, joint: best, bestChoice, optima: exhaustive ? optima : null };
}

// --- anatomy -----------------------------------------------------------------------

// Homogeneous dual coordinates of a line: y = mx + b -> [m, -1, b]; x = b -> [1, 0, -b].
function dualCoords(dirIndex, b, q) {
  if (dirIndex === q) return [1, 0, mod(-b, q)];
  return [dirIndex, q - 1, b];
}

// Max dual-conic incidence: fit a conic through every 5-subset of the dual
// points, evaluate on the full set, report the best incidence, plus whether
// the maximizing conic is parabola-type (its dual passes through the dual
// point of the line at infinity [0,0,1], i.e. the z^2 coefficient vanishes).
function maxConicIncidence(points, q) {
  if (points.length < 6) return { fitted: false, maxOnConic: points.length, parabola: null };
  let best = { fitted: true, maxOnConic: 0, parabola: null };
  const n = points.length;
  for (let a = 0; a < n; a++)
    for (let b = a + 1; b < n; b++)
      for (let c = b + 1; c < n; c++)
        for (let d = c + 1; d < n; d++)
          for (let e = d + 1; e < n; e++) {
            const fit = fitConic([points[a], points[b], points[c], points[d], points[e]], q);
            if (!fit) continue;
            let on = 0;
            for (const p of points) if (evalConic(fit, p, q) === 0) on++;
            if (on > best.maxOnConic) {
              best = { fitted: true, maxOnConic: on, parabola: fit[5] === 0 };
            }
          }
  return best;
}

function conicRow([x, y, z], q) {
  return [(x * x) % q, (x * y) % q, (y * y) % q, (x * z) % q, (y * z) % q, (z * z) % q];
}

function evalConic(coeff, p, q) {
  const rw = conicRow(p, q);
  let s = 0;
  for (let j = 0; j < 6; j++) s = mod(s + rw[j] * coeff[j], q);
  return s;
}

// Null vector of the 5 x 6 incidence system (one conic through 5 points,
// generically unique up to scale). Returns coeff[6] or null if rank-deficient.
function fitConic(fivePoints, q) {
  const M = fivePoints.map((p) => conicRow(p, q));
  const cols = 6;
  let r = 0;
  const pivots = [];
  for (let c = 0; c < cols && r < M.length; c++) {
    let pr = -1;
    for (let i = r; i < M.length; i++)
      if (M[i][c] % q !== 0) {
        pr = i;
        break;
      }
    if (pr === -1) continue;
    [M[r], M[pr]] = [M[pr], M[r]];
    const invp = inv(mod(M[r][c], q), q);
    for (let j = 0; j < cols; j++) M[r][j] = (M[r][j] * invp) % q;
    for (let i = 0; i < M.length; i++) {
      if (i === r) continue;
      const f = mod(M[i][c], q);
      if (f === 0) continue;
      for (let j = 0; j < cols; j++) M[i][j] = mod(M[i][j] - f * M[r][j], q);
    }
    pivots.push(c);
    r++;
  }
  if (r < 5) return null; // degenerate 5-subset: >1-dim family, skip
  const free = [...Array(cols).keys()].find((c) => !pivots.includes(c));
  if (free === undefined) return null;
  const coeff = new Array(cols).fill(0);
  coeff[free] = 1;
  for (let i = pivots.length - 1; i >= 0; i--) {
    const c = pivots[i];
    let s = 0;
    for (let j = c + 1; j < cols; j++) s = mod(s + M[i][j] * coeff[j], q);
    coeff[c] = mod(-s, q);
  }
  return coeff;
}

// Does a set of dual points lie on a common conic? Fit the conic's 6
// coefficients (x^2, xy, y^2, xz, yz, z^2) from the first 5 points by
// Gaussian elimination mod q, then test the rest. Returns count on conic.
function conicIncidence(points, q) {
  if (points.length < 6) return { fitted: false, onConic: points.length };
  const row = ([x, y, z]) => [
    (x * x) % q,
    (x * y) % q,
    (y * y) % q,
    (x * z) % q,
    (y * z) % q,
    (z * z) % q,
  ];
  // Solve 5 homogeneous equations in 6 unknowns (null space, generically 1-dim).
  const M = points.slice(0, 5).map(row);
  const cols = 6;
  let r = 0;
  const pivots = [];
  for (let c = 0; c < cols && r < M.length; c++) {
    let pr = -1;
    for (let i = r; i < M.length; i++) if (M[i][c] % q !== 0) pr = i >= 0 ? i : pr;
    for (let i = r; i < M.length; i++)
      if (M[i][c] % q !== 0) {
        pr = i;
        break;
      }
    if (pr === -1) continue;
    [M[r], M[pr]] = [M[pr], M[r]];
    const invp = inv(mod(M[r][c], q), q);
    for (let j = 0; j < cols; j++) M[r][j] = (M[r][j] * invp) % q;
    for (let i = 0; i < M.length; i++) {
      if (i === r) continue;
      const f = mod(M[i][c], q);
      if (f === 0) continue;
      for (let j = 0; j < cols; j++) M[i][j] = mod(M[i][j] - f * M[r][j], q);
    }
    pivots.push(c);
    r++;
  }
  // Free column -> null vector.
  const free = [...Array(cols).keys()].find((c) => !pivots.includes(c));
  if (free === undefined) return { fitted: false, onConic: 0 };
  const coeff = new Array(cols).fill(0);
  coeff[free] = 1;
  for (let i = pivots.length - 1; i >= 0; i--) {
    const c = pivots[i];
    let s = 0;
    for (let j = c + 1; j < cols; j++) s = mod(s + M[i][j] * coeff[j], q);
    coeff[c] = mod(-s, q);
  }
  const evalConic = (p) => {
    const rw = row(p);
    let s = 0;
    for (let j = 0; j < cols; j++) s = mod(s + rw[j] * coeff[j], q);
    return s;
  };
  let on = 0;
  for (const p of points) if (evalConic(p) === 0) on++;
  return { fitted: true, onConic: on };
}

function anatomy(q, quad, targets, interceptChoice) {
  const dirs = Core.directions(q);
  // Full q+1-line decomposition: star lines (intercept 0) + chosen lines.
  const family = []; // { dirIndex, b, star }
  for (const i of quad) family.push({ dirIndex: i, b: 0, star: true });
  targets.forEach((dirIndex, j) => family.push({ dirIndex, b: interceptChoice[j], star: false }));

  const mult = new Map();
  const union = new Set();
  for (const { dirIndex, b } of family) {
    for (const p of Core.lineMask(dirs[dirIndex], b, q)) {
      mult.set(p, (mult.get(p) ?? 0) + 1);
      union.add(p);
    }
  }
  let sacrifice = 0;
  const heavy = [];
  const profile = new Map();
  for (const [p, m] of mult) {
    sacrifice += ((m - 1) * (m - 2)) / 2;
    profile.set(m, (profile.get(m) ?? 0) + 1);
    if (m >= 3) {
      const { x, y } = Core.indexToXY(p, q);
      const linesHere = family
        .filter(({ dirIndex, b }) => Core.lineMask(dirs[dirIndex], b, q).has(p))
        .map(({ dirIndex, b, star }) => `${dirs[dirIndex].label}:${b}${star ? "*" : ""}`);
      heavy.push({ point: `(${x},${y})`, mult: m, lines: linesHere });
    }
  }
  heavy.sort((a, b) => b.mult - a.mult);

  // Pivot = origin; chosen lines through it have b = 0.
  const chosenThroughPivot = family.filter((f) => !f.star && f.b === 0).length;

  // Dual-conic test on the non-pivot-concurrent chosen lines + optionally all.
  const chosenDual = family.filter((f) => !f.star).map((f) => dualCoords(f.dirIndex, f.b, q));
  const allDual = family.map((f) => dualCoords(f.dirIndex, f.b, q));

  return {
    size: union.size,
    sacrifice,
    exFromIdentity: sacrifice - (q - 1) / 2,
    multiplicityProfile: [...profile.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([m, n]) => `${m}:${n}`)
      .join(" "),
    pivotMult: mult.get(Core.pointIndex(0, 0, q)) ?? 0,
    chosenThroughPivot,
    heavyPoints: heavy,
    chosenIntercepts: targets.map((dirIndex, j) => `${dirs[dirIndex].label}:${interceptChoice[j]}`),
    conicChosen: maxConicIncidence(chosenDual, q),
    conicAll: maxConicIncidence(allDual, q),
  };
}

// --- main --------------------------------------------------------------------------

function main() {
  const outDir = process.argv.includes("--out")
    ? process.argv[process.argv.indexOf("--out") + 1]
    : DEFAULT_OUT;

  const report = [];
  for (const q of FIELDS) {
    const exhaustive = EXHAUSTIVE_Q.includes(q);
    for (const rep of orbitReps(q)) {
      const body = starBody(q, rep.quad);
      const solved = solveWithWitness(q, body, exhaustive);
      const completion = body.size + solved.joint;
      const ex = completion - bmMinimum(q);
      const witnessAnatomy = anatomy(q, rep.quad, solved.targets, solved.bestChoice);
      // Consistency: identity-derived ex must match solver ex; size must match.
      const consistent =
        witnessAnatomy.size === completion && witnessAnatomy.exFromIdentity === ex;

      // For exhaustive fields: summarize the optimum population.
      let optimaSummary = null;
      if (exhaustive && solved.optima) {
        const pivotCounts = new Map();
        const conicCounts = new Map();
        for (const opt of solved.optima) {
          const a = anatomy(q, rep.quad, solved.targets, opt);
          pivotCounts.set(a.pivotMult, (pivotCounts.get(a.pivotMult) ?? 0) + 1);
          const key = `${a.conicChosen.maxOnConic}/${solved.targets.length}`;
          conicCounts.set(key, (conicCounts.get(key) ?? 0) + 1);
        }
        optimaSummary = {
          count: solved.optima.length,
          capped: solved.optima.length >= ALL_OPTIMA_CAP,
          pivotMultDistribution: [...pivotCounts.entries()]
            .sort((a, b) => a[0] - b[0])
            .map(([m, n]) => `${m}:${n}`)
            .join(" "),
          chosenOnConicDistribution: [...conicCounts.entries()]
            .sort()
            .map(([k, n]) => `${k}:${n}`)
            .join(" "),
        };
      }

      report.push({
        q,
        orbit: rep.label,
        quad: rep.quad.map((i) => Core.directions(q)[i].label).join(" "),
        completion,
        ex,
        level: ex === (q - 1) / 2 ? "HIGH" : ex === (q - 3) / 2 ? "LOW" : "OTHER",
        consistent,
        witness: witnessAnatomy,
        optima: optimaSummary,
      });

      const w = witnessAnatomy;
      console.log(
        [
          `q=${q}`,
          rep.label.padEnd(15),
          `ex=${ex}`,
          report[report.length - 1].level.padEnd(5),
          `sac=${w.sacrifice}`,
          `profile=[${w.multiplicityProfile}]`,
          `pivotMult=${w.pivotMult}`,
          `chosenThruPivot=${w.chosenThroughPivot}`,
          `conicChosen=${w.conicChosen.maxOnConic}/${w.chosenIntercepts.length}${w.conicChosen.parabola === true ? "(parabola)" : w.conicChosen.parabola === false ? "(non-parabola)" : ""}`,
          `conicAll=${w.conicAll.maxOnConic}/${w.chosenIntercepts.length + 4}${w.conicAll.parabola === true ? "(parabola)" : ""}`,
          `consistent=${consistent}`,
          w.optima ? "" : "",
          report[report.length - 1].optima
            ? `optima=${report[report.length - 1].optima.count}${report[report.length - 1].optima.capped ? "+" : ""} pivots[${report[report.length - 1].optima.pivotMultDistribution}]`
            : "",
        ].join(" "),
      );
      for (const h of w.heavyPoints) {
        console.log(`      mult${h.mult} @ ${h.point}: ${h.lines.join(" ")}`);
      }
    }
  }

  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(
    path.join(outDir, "anatomy.json"),
    JSON.stringify({ generatedAt: new Date().toISOString(), report }, null, 2) + "\n",
  );
  const allConsistent = report.every((r) => r.consistent);
  console.log(`ANATOMY consistent=${allConsistent} rows=${report.length} out=${outDir}`);
  process.exit(allConsistent ? 0 : 1);
}

main();
