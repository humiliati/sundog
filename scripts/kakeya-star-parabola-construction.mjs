#!/usr/bin/env node
// PHASE3L step 2 - parabola-tangent construction search for 4-star
// completions (Track 1 main instrument).
//
// Candidate family (from the PHASE3L anatomy: optimal completions at q=13
// harmonic/generic have 12 of 14 lines tangent to one parabola; the pivot is
// always exactly mult-4; all other concurrencies are triples):
//   - choose an AXIS direction a among the 4 star directions;
//   - choose a parabola with axis direction a (3-parameter family: images of
//     y = alpha x^2 + beta x + gamma, alpha != 0, under the axis map);
//   - completion = the 4 star lines + the parabola's tangents in the q-3
//     non-star directions (a parabola has exactly one tangent per non-axis
//     direction, which is why the axis must be a star direction);
//   - sacrifice counted honestly over the q+1-line family;
//     ex = sacrifice - (q-1)/2 (PHASE3G identity, size-verified).
//
// The search minimizes ex over all candidates per orbit representative.
// Constructions are UPPER bounds: they can only refute "high" hypotheses
// (by achieving low), never certify them. Validation gate: at the
// exact-solved fields q in {5,7,11,13,17} the construction minimum is
// compared with the known exact ex per orbit (match = family is optimal
// there; a gap = family incomplete there, reported honestly).
//
// Out-of-register discipline: evaluation fields q in {19,23,29,31,37} are
// sidecars; the workbench register is untouched. No Euclidean claim.

import fs from "node:fs";
import path from "node:path";
import * as Core from "../kakeya/kakeya-core.js";

const ARTIFACT_ID = "KAK-PHASE3L-STAR-PARABOLA-CONSTRUCTION";
const DEFAULT_OUT = path.join("results", "kakeya", "star-parabola-construction");
const VALIDATION_FIELDS = [5, 7, 11, 13, 17];
const EVALUATION_FIELDS = [19, 23, 29, 31, 37];
// Exact per-orbit ex from PHASE3I/3J/3K (solver-certified).
const KNOWN_EXACT = {
  5: { harmonic: 2 },
  7: { harmonic: 2, equianharmonic: 3 },
  11: { harmonic: 4, generic: 4 },
  13: { harmonic: 6, equianharmonic: 5, generic: 5 },
  17: { harmonic: 7, "generic-a": 7, "generic-b": 7 },
};
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

// --- orbits (PHASE3I machinery) -----------------------------------------------

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
  const byBase = new Map();
  for (const key of [...reps.keys()].sort())
    byBase.set(reps.get(key).label, (byBase.get(reps.get(key).label) ?? 0) + 1);
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
    out.push({ label, quad: r.quad });
  }
  return out;
}

// --- construction candidate evaluation ------------------------------------------

// Point list of the standard-frame tangent to y = ax^2 + bx + g with slope mu,
// transformed by the axis map for axis index `axis` (identity when axis = q,
// i.e. inf; else (x,y) -> (y, x + s y) with s = axis).
// Standard tangent: t = (mu - b) / 2a, intercept c = g - a t^2.
function tangentPoints(q, axis, alpha, beta, gamma, mu, out) {
  const t = mod((mu - beta) * inv(mod(2 * alpha, q), q), q);
  const c = mod(gamma - alpha * t * t, q);
  if (axis === q) {
    for (let x = 0; x < q; x++) out[x] = Core.pointIndex(x, (mu * x + c) % q, q);
  } else {
    const s = axis;
    for (let x = 0; x < q; x++) {
      const y = (mu * x + c) % q;
      out[x] = Core.pointIndex(y, (x + s * y) % q, q);
    }
  }
}

// Direction index -> standard-frame slope preimage under the axis map.
// axis = inf: direction d (finite slope) -> mu = d; d = inf unreachable.
// axis = s:   direction inf -> mu = 0; direction v (v != s) -> mu = (v-s)^{-1}.
function slopePreimage(q, axis, dirIndex) {
  if (axis === q) return dirIndex; // dirIndex is a finite slope; inf excluded upstream
  if (dirIndex === q) return 0;
  return inv(mod(dirIndex - axis, q), q);
}

function searchOrbit(q, quad) {
  const dirs = Core.directions(q);
  const n = Core.pointCount(q);
  const starLinePoints = quad.map((i) => [...Core.lineMask(dirs[i], 0, q)]);
  const neededDirs = [];
  for (let i = 0; i < dirs.length; i++) if (!quad.includes(i)) neededDirs.push(i);

  const stamp = new Int32Array(n);
  const count = new Uint8Array(n);
  let epoch = 0;
  const tangentBuf = new Array(q);

  let best = { ex: Infinity };
  let candidates = 0;
  let starts = [];

  for (const axis of quad) {
    // Preimage slopes for the needed directions under this axis map.
    const mus = neededDirs.map((d) => slopePreimage(q, axis, d));
    for (let alpha = 1; alpha < q; alpha++)
      for (let beta = 0; beta < q; beta++)
        for (let gamma = 0; gamma < q; gamma++) {
          candidates++;
          epoch++;
          let sacrifice = 0;
          let size = 0;
          // Star lines first.
          for (const pts of starLinePoints) {
            for (const p of pts) {
              if (stamp[p] !== epoch) {
                stamp[p] = epoch;
                count[p] = 1;
                size++;
              } else {
                const m = count[p];
                if (m >= 2) sacrifice += m - 1;
                count[p] = m + 1;
              }
            }
          }
          // Tangents for the needed directions.
          for (const mu of mus) {
            tangentPoints(q, axis, alpha, beta, gamma, mu, tangentBuf);
            for (let x = 0; x < q; x++) {
              const p = tangentBuf[x];
              if (stamp[p] !== epoch) {
                stamp[p] = epoch;
                count[p] = 1;
                size++;
              } else {
                const m = count[p];
                if (m >= 2) sacrifice += m - 1;
                count[p] = m + 1;
              }
            }
          }
          const ex = sacrifice - (q - 1) / 2;
          if (ex <= best.ex + 2) {
            starts.push({ ex, axisIndex: axis, alpha, beta, gamma });
            if (starts.length > 4000)
              starts = starts.filter((s) => s.ex <= best.ex + 2).slice(0, 2000);
          }
          if (ex < best.ex) {
            best = {
              ex,
              size,
              sacrifice,
              axis: dirs[axis].label,
              alpha,
              beta,
              gamma,
              sizeIdentityOk: size === (q * (q + 1)) / 2 + sacrifice,
            };
          }
        }
  }
  starts = starts
    .filter((s) => s.ex <= best.ex + 2)
    .sort((a, b) => a.ex - b.ex)
    .slice(0, 200);
  return { best, candidates, starts };
}

// --- descent refinement (PHASE3M extension) ---------------------------------------
// Coordinate descent over single-direction line swaps, from the top pure-
// parabola candidates plus seeded random restarts. Closes family gaps where
// the pure family misses the optimum (validation: q=11 harmonic 5 -> 4).
// Upper bounds remain verified completions; epistemics unchanged.

function interceptOfTangent(q, axis, alpha, beta, gamma, dirIndex) {
  // The transformed tangent covering direction dirIndex, as (dirIndex, b).
  const buf = new Array(q);
  tangentPoints(q, axis, alpha, beta, gamma, slopePreimage(q, axis, dirIndex), buf);
  const { x, y } = Core.indexToXY(buf[0], q);
  if (dirIndex === q) return x; // vertical: x = b
  return mod(y - dirIndex * x, q);
}

function descentRefine(q, quad, starts, seeds) {
  const dirs = Core.directions(q);
  const n = Core.pointCount(q);
  const neededDirs = [];
  for (let i = 0; i < dirs.length; i++) if (!quad.includes(i)) neededDirs.push(i);
  // Precompute all line point-lists per (dir, b).
  const linePts = dirs.map((d, i) => {
    const perB = [];
    for (let b = 0; b < q; b++) perB.push([...Core.lineMask(dirs[i], b, q)]);
    return perB;
  });

  const count = new Uint8Array(n);
  let sacrifice = 0;
  const addLine = (i, b) => {
    for (const p of linePts[i][b]) {
      const m = count[p];
      if (m >= 2) sacrifice += m - 1;
      count[p] = m + 1;
    }
  };
  const removeLine = (i, b) => {
    for (const p of linePts[i][b]) {
      const m = count[p];
      if (m >= 3) sacrifice -= m - 2;
      count[p] = m - 1;
    }
  };

  let bestEx = Infinity;
  let bestAssign = null;

  const descendFrom = (assign) => {
    count.fill(0);
    sacrifice = 0;
    for (const i of quad) addLine(i, 0);
    for (const d of neededDirs) addLine(d, assign.get(d));
    let improved = true;
    while (improved) {
      improved = false;
      for (const d of neededDirs) {
        const cur = assign.get(d);
        removeLine(d, cur);
        let bestB = cur;
        let bestSac = Infinity;
        for (let b = 0; b < q; b++) {
          addLine(d, b);
          if (sacrifice < bestSac) {
            bestSac = sacrifice;
            bestB = b;
          }
          removeLine(d, b);
        }
        addLine(d, bestB);
        if (bestB !== cur) improved = true;
        assign.set(d, bestB);
      }
    }
    const ex = sacrifice - (q - 1) / 2;
    if (ex < bestEx) {
      bestEx = ex;
      bestAssign = new Map(assign);
    }
  };

  for (const s of starts) {
    const assign = new Map();
    for (const d of neededDirs)
      assign.set(d, interceptOfTangent(q, s.axisIndex, s.alpha, s.beta, s.gamma, d));
    descendFrom(assign);
  }
  for (let seed = 1; seed <= seeds; seed++) {
    const rng = Core.mulberry32(seed);
    const assign = new Map();
    for (const d of neededDirs) assign.set(d, Math.floor(rng() * q));
    descendFrom(assign);
  }

  // Independent rebuild of the refined winner.
  const body = new Set();
  for (const i of quad) for (const p of Core.lineMask(dirs[i], 0, q)) body.add(p);
  for (const [d, b] of bestAssign) for (const p of Core.lineMask(dirs[d], b, q)) body.add(p);
  const summary = Core.shadowSummary(q, body);
  return {
    ex: bestEx,
    verified: summary.complete && body.size === bmMinimum(q) + bestEx,
    size: body.size,
  };
}

// Independent verification of the winning candidate: rebuild the completion
// as a point set, check size, completeness, and that it contains the star.
function verifyCandidate(q, quad, best) {
  const dirs = Core.directions(q);
  const axis = dirs.findIndex((d) => d.label === best.axis);
  const body = new Set();
  for (const i of quad) for (const p of Core.lineMask(dirs[i], 0, q)) body.add(p);
  const starSize = body.size;
  const buf = new Array(q);
  for (let i = 0; i < dirs.length; i++) {
    if (quad.includes(i)) continue;
    tangentPoints(q, axis, best.alpha, best.beta, best.gamma, slopePreimage(q, axis, i), buf);
    for (const p of buf) body.add(p);
  }
  const summary = Core.shadowSummary(q, body);
  return {
    size: body.size,
    starSize,
    complete: summary.complete,
    sizeMatches: body.size === best.size,
    exVerified: body.size - bmMinimum(q),
  };
}

// --- main ------------------------------------------------------------------------

function main() {
  const outDir = process.argv.includes("--out")
    ? process.argv[process.argv.indexOf("--out") + 1]
    : DEFAULT_OUT;

  const refine = process.argv.includes("--refine"); // PHASE3M descent extension
  const rows = [];
  let validationPass = true;
  let instrumentPass = true;

  for (const q of [...VALIDATION_FIELDS, ...EVALUATION_FIELDS]) {
    for (const rep of orbitReps(q)) {
      const t0 = Date.now();
      const { best, candidates, starts } = searchOrbit(q, rep.quad);
      const verify = verifyCandidate(q, rep.quad, best);
      const refined = refine ? descentRefine(q, rep.quad, starts, 50) : null;
      const finalEx = refined ? Math.min(best.ex, refined.ex) : best.ex;
      const known = KNOWN_EXACT[q]?.[rep.label] ?? null;
      const level =
        finalEx === (q - 1) / 2 ? "HIGH" : finalEx === (q - 3) / 2 ? "LOW" : "OTHER";
      const ok =
        best.sizeIdentityOk &&
        verify.complete &&
        verify.sizeMatches &&
        verify.exVerified === best.ex &&
        (refined === null || refined.verified);
      instrumentPass = instrumentPass && ok;
      if (known !== null && finalEx !== known) validationPass = false;
      rows.push({
        q,
        orbit: rep.label,
        quad: rep.quad.map((i) => Core.directions(q)[i].label).join(" "),
        constructionEx: best.ex,
        refinedEx: refined ? refined.ex : null,
        finalEx,
        level,
        knownExact: known,
        matchesExact: known === null ? null : finalEx === known,
        axis: best.axis,
        params: `a=${best.alpha} b=${best.beta} g=${best.gamma}`,
        candidates,
        seconds: (Date.now() - t0) / 1000,
        verified: ok,
      });
      const r = rows[rows.length - 1];
      console.log(
        `q=${q} ${rep.label.padEnd(15)} pure=${best.ex}${refined ? ` refined=${refined.ex}` : ""} final=${finalEx} ${level.padEnd(5)} ` +
          `${known !== null ? `exact=${known} ${finalEx === known ? "MATCH" : "GAP"}` : "eval"} ` +
          `verified=${ok} (${r.seconds.toFixed(1)}s)`,
      );
    }
  }

  const manifest = {
    artifactId: ARTIFACT_ID,
    generatedAt: new Date().toISOString(),
    status: "internal construction-search receipt (Track 1)",
    command: "node scripts/kakeya-star-parabola-construction.mjs",
    family:
      "axis in star directions; parabola = axis-map image of y = alpha x^2 + beta x + gamma (alpha != 0); completion = 4 star lines + tangents in the q-3 non-star directions; ex = sacrifice - (q-1)/2, size-identity and completeness verified per winner",
    validationFields: VALIDATION_FIELDS,
    evaluationFields: EVALUATION_FIELDS,
    validationPass,
    falsifier: {
      name: "CONSTRUCTION_INSTRUMENT_MISMATCH",
      description:
        "Instrument-only: fires if any winning candidate fails the size identity, completeness, or independent rebuild, or if a construction ex beats a solver-certified exact value (impossible for a valid upper bound). Validation gaps (construction > exact) and evaluation outcomes are measurements.",
      status: instrumentPass && rows.every((r) => r.knownExact === null || r.finalEx >= r.knownExact) ? "clear" : "fired",
    },
    rows,
  };

  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(
    path.join(outDir, "manifest.json"),
    JSON.stringify(manifest, null, 2) + "\n",
  );
  console.log(
    `KAK_STAR_PARABOLA_CONSTRUCTION validation=${validationPass ? "MATCH-ALL" : "GAPS"} falsifier=${manifest.falsifier.status} rows=${rows.length} out=${outDir}`,
  );
  process.exit(manifest.falsifier.status === "clear" ? 0 : 1);
}

main();
