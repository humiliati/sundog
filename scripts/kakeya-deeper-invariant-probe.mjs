#!/usr/bin/env node
// PHASE3N - deeper-invariant probe: the harmonic q = 3 (mod 8) anomaly.
//
// PHASE3M's invariant hunt reduced the ENTIRE 4-star level classification to
// a single collision: sig+type separates all 33 known field-orbits EXCEPT the
// pair {11-harmonic (LOW), 19-harmonic (HIGH)}. The harmonic signature is
// {chi(2), chi(-1), chi(2)}, so sig = -1-1-1 <=> chi(2) = chi(-1) = -1 <=>
// q = 3 (mod 8). Within harmonic the level is a pure function of q (j = 1728
// is fixed), so the whole "deeper invariant" question is exactly:
//
//   For q = 3 (mod 8), is the harmonic 4-star LOW or HIGH?
//   Known: q=11 LOW, q=19 HIGH. Everything else is determined by sig+type.
//
// This probe generates more data in that class via the descent-augmented
// construction (validated exact at 6/6 harmonic fields q<=19, incl. the
// B&B-confirmed q=19), then tests candidate deeper invariants:
//   - higher-power residue symbols of a fixed base against q (the user's
//     "higher-power residue" ask, applied to q since j is fixed);
//   - q mod higher moduli (16, 24, 3, 5, ...);
//   - whether 11 is simply a small-field boundary anomaly.
//
// Epistemics of construction levels: DEF-LOW = a verified completion at
// (q-3)/2 (hard: level is not HIGH); PROB-HIGH = descent could not beat
// (q-1)/2 (formally an upper bound; could be a family gap, though the only
// gap ever seen - q=11 pure - was closed by descent). Controls at other
// residue classes must reproduce the known sig+type verdicts.
//
// Out-of-register: no field here is added to the workbench. No Euclidean claim.

import fs from "node:fs";
import path from "node:path";
import * as Core from "../kakeya/kakeya-core.js";

const ARTIFACT_ID = "KAK-PHASE3N-DEEPER-INVARIANT-PROBE";
const DEFAULT_OUT = path.join("results", "kakeya", "deeper-invariant-probe");
const INF = -1;

// q = 3 (mod 8) primes (the anomaly class) and small controls from other
// classes to confirm the reduction reproduces.
const CLASS_3MOD8 = [11, 19, 43, 59, 67, 83, 107, 131];
const CONTROLS = [5, 7, 13, 17, 29, 37]; // known harmonic levels (5/13/29/37 HIGH, 7/17 LOW)
const KNOWN_HARMONIC = { 5: "HIGH", 7: "LOW", 11: "LOW", 13: "HIGH", 17: "LOW", 19: "HIGH", 29: "HIGH", 37: "HIGH" };

function mod(x, q) {
  return ((x % q) + q) % q;
}
function inv(x, q) {
  for (let i = 1; i < q; i++) if ((x * i) % q === 1) return i;
  throw new Error(`no inverse of ${x} mod ${q}`);
}
function chi(x, q) {
  let base = mod(x, q);
  if (base === 0) return 0;
  let result = 1,
    exp = (q - 1) / 2;
  while (exp > 0) {
    if (exp & 1) result = (result * base) % q;
    base = (base * base) % q;
    exp >>= 1;
  }
  return result === 1 ? 1 : -1;
}
function powResidue(x, q, d) {
  // d-th power residue symbol x^((q-1)/d) mod q, as a representative in [0,q);
  // meaningful only when d | q-1 (else returns null).
  if ((q - 1) % d !== 0) return null;
  let base = mod(x, q);
  if (base === 0) return 0;
  let result = 1,
    exp = (q - 1) / d;
  while (exp > 0) {
    if (exp & 1) result = (result * base) % q;
    base = (base * base) % q;
    exp >>= 1;
  }
  return result;
}
function bmMinimum(q) {
  return (q * (q + 1)) / 2 + (q - 1) / 2;
}

// --- harmonic quadruple + orbit check --------------------------------------------

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
function harmonicQuad(q) {
  const dirCount = q + 1;
  const dv = (i) => (i === q ? INF : i);
  const harmonicKey = [2, mod((q + 1) / 2, q), q - 1].sort((x, y) => x - y).join(",");
  for (let a = 0; a < dirCount; a++)
    for (let b = a + 1; b < dirCount; b++)
      for (let c = b + 1; c < dirCount; c++)
        for (let d = c + 1; d < dirCount; d++) {
          const l = crossRatio(dv(a), dv(b), dv(c), dv(d), q);
          if (sixSet(l, q).join(",") === harmonicKey) return [a, b, c, d];
        }
  throw new Error(`no harmonic quad at q=${q}`);
}

// --- descent-augmented construction (PHASE3L/3M machinery, harmonic only) ---------

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
function slopePreimage(q, axis, dirIndex) {
  if (axis === q) return dirIndex;
  if (dirIndex === q) return 0;
  return inv(mod(dirIndex - axis, q), q);
}
function interceptOfTangent(q, axis, alpha, beta, gamma, dirIndex) {
  const buf = new Array(q);
  tangentPoints(q, axis, alpha, beta, gamma, slopePreimage(q, axis, dirIndex), buf);
  const { x, y } = Core.indexToXY(buf[0], q);
  if (dirIndex === q) return x;
  return mod(y - dirIndex * x, q);
}

// Pure-parabola minimum + top seeds, with alpha normalized to 1 (scaling about
// the pivot fixes every star line and preserves sacrifice, so min over
// (alpha,beta,gamma) = min over (1,beta,gamma)).
function pureSearch(q, quad) {
  const dirs = Core.directions(q);
  const n = Core.pointCount(q);
  const starPts = quad.map((i) => [...Core.lineMask(dirs[i], 0, q)]);
  const needed = [];
  for (let i = 0; i < dirs.length; i++) if (!quad.includes(i)) needed.push(i);
  const stamp = new Int32Array(n);
  const cnt = new Uint8Array(n);
  let epoch = 0;
  const buf = new Array(q);
  let best = Infinity;
  let seeds = [];
  for (const axis of quad) {
    const mus = needed.map((d) => slopePreimage(q, axis, d));
    for (let beta = 0; beta < q; beta++)
      for (let gamma = 0; gamma < q; gamma++) {
        epoch++;
        let sac = 0;
        for (const pts of starPts)
          for (const p of pts) {
            if (stamp[p] !== epoch) {
              stamp[p] = epoch;
              cnt[p] = 1;
            } else {
              const m = cnt[p];
              if (m >= 2) sac += m - 1;
              cnt[p] = m + 1;
            }
          }
        for (const mu of mus) {
          tangentPoints(q, axis, 1, beta, gamma, mu, buf);
          for (let x = 0; x < q; x++) {
            const p = buf[x];
            if (stamp[p] !== epoch) {
              stamp[p] = epoch;
              cnt[p] = 1;
            } else {
              const m = cnt[p];
              if (m >= 2) sac += m - 1;
              cnt[p] = m + 1;
            }
          }
        }
        const ex = sac - (q - 1) / 2;
        if (ex <= best + 1) seeds.push({ ex, axis, beta, gamma });
        if (ex < best) best = ex;
      }
  }
  seeds = seeds.filter((s) => s.ex <= best + 1).sort((a, b) => a.ex - b.ex).slice(0, 120);
  return { pureEx: best, seeds };
}

function descent(q, quad, seeds, randomStarts) {
  const dirs = Core.directions(q);
  const n = Core.pointCount(q);
  const needed = [];
  for (let i = 0; i < dirs.length; i++) if (!quad.includes(i)) needed.push(i);
  const linePts = dirs.map((_, i) => {
    const perB = [];
    for (let b = 0; b < q; b++) perB.push([...Core.lineMask(dirs[i], b, q)]);
    return perB;
  });
  const cnt = new Uint8Array(n);
  let sac = 0;
  const add = (i, b) => {
    for (const p of linePts[i][b]) {
      const m = cnt[p];
      if (m >= 2) sac += m - 1;
      cnt[p] = m + 1;
    }
  };
  const rem = (i, b) => {
    for (const p of linePts[i][b]) {
      const m = cnt[p];
      if (m >= 3) sac -= m - 2;
      cnt[p] = m - 1;
    }
  };
  let bestEx = Infinity;
  let bestAssign = null;
  const run = (assign) => {
    cnt.fill(0);
    sac = 0;
    for (const i of quad) add(i, 0);
    for (const d of needed) add(d, assign.get(d));
    let improved = true;
    while (improved) {
      improved = false;
      for (const d of needed) {
        const cur = assign.get(d);
        rem(d, cur);
        let bb = cur,
          bs = Infinity;
        for (let b = 0; b < q; b++) {
          add(d, b);
          if (sac < bs) {
            bs = sac;
            bb = b;
          }
          rem(d, b);
        }
        add(d, bb);
        if (bb !== cur) improved = true;
        assign.set(d, bb);
      }
    }
    const ex = sac - (q - 1) / 2;
    if (ex < bestEx) {
      bestEx = ex;
      bestAssign = new Map(assign);
    }
  };
  for (const s of seeds) {
    const a = new Map();
    for (const d of needed) a.set(d, interceptOfTangent(q, s.axis, 1, s.beta, s.gamma, d));
    run(a);
  }
  for (let seed = 1; seed <= randomStarts; seed++) {
    const rng = Core.mulberry32(seed * 2654435761);
    const a = new Map();
    for (const d of needed) a.set(d, Math.floor(rng() * q));
    run(a);
  }
  // Verify the winner.
  const body = new Set();
  for (const i of quad) for (const p of Core.lineMask(dirs[i], 0, q)) body.add(p);
  for (const [d, b] of bestAssign) for (const p of Core.lineMask(dirs[d], b, q)) body.add(p);
  const summary = Core.shadowSummary(q, body);
  const verified = summary.complete && body.size === bmMinimum(q) + bestEx;
  return { ex: bestEx, verified };
}

function harmonicLevel(q) {
  const quad = harmonicQuad(q);
  const { pureEx, seeds } = pureSearch(q, quad);
  const ref = descent(q, quad, seeds, 80);
  const ex = Math.min(pureEx, ref.ex);
  const low = (q - 3) / 2,
    high = (q - 1) / 2;
  const level = ex === low ? "LOW" : ex === high ? "HIGH" : "OTHER";
  const epistemics = level === "LOW" ? "DEF-LOW" : "PROB-HIGH";
  return { q, quad, ex, low, high, level, epistemics, verified: ref.verified };
}

// --- deeper-invariant battery on the q=3(mod8) harmonic levels --------------------

function invariantBattery(q) {
  return {
    "q mod 16": mod(q, 16),
    "q mod 24": mod(q, 24),
    "q mod 3": mod(q, 3),
    "q mod 5": mod(q, 5),
    "q mod 7": mod(q, 7),
    "chi3(-1)": chi(-1, q) === -1 ? (powResidue(-1, q, 4) ?? "n/a") : "n/a", // placeholder for finer -1
    "res4(2)": powResidue(2, q, 4), // quartic residue of 2 (defined iff 4|q-1; q=3mod8 => n/a)
    "res8(2)": powResidue(2, q, 8),
    "res3(2)": powResidue(2, q, 3), // cubic residue of 2 (iff q=1 mod 3)
    "chi(3)": chi(3, q),
    "chi(5)": chi(5, q),
    "chi(q-2)": chi(q - 2, q),
    "cls_2_cubefree": powResidue(2, q, 3) === null ? "n/a" : powResidue(2, q, 3),
    "smallfield(q<=11)": q <= 11 ? "small" : "large",
  };
}

function main() {
  const outDir = process.argv.includes("--out")
    ? process.argv[process.argv.indexOf("--out") + 1]
    : DEFAULT_OUT;

  const t0 = Date.now();
  const controlRows = CONTROLS.map((q) => {
    const r = harmonicLevel(q);
    return { ...r, control: true, expected: KNOWN_HARMONIC[q] };
  });
  const classRows = CLASS_3MOD8.map((q) => ({ ...harmonicLevel(q), control: false }));

  const controlsPass = controlRows.every((r) => r.level === r.expected && r.verified);
  const instrumentPass = controlsPass && [...controlRows, ...classRows].every((r) => r.verified);

  // Battery test on the q=3(mod8) class (11,19 known exact; rest construction).
  const battery = {};
  const cand = Object.keys(invariantBattery(11));
  for (const name of cand) {
    const map = new Map();
    let consistent = true;
    const collisions = [];
    for (const r of classRows) {
      const v = invariantBattery(r.q)[name];
      if (v === "n/a" || v === null) continue;
      const key = String(v);
      if (!map.has(key)) map.set(key, r);
      else if (map.get(key).level !== r.level) {
        consistent = false;
        collisions.push(`${map.get(key).q}(${map.get(key).level}) vs ${r.q}(${r.level}) @${key}`);
      }
    }
    battery[name] = { consistent, collisions, definedOn: [...map.keys()].length };
  }

  const manifest = {
    artifactId: ARTIFACT_ID,
    generatedAt: new Date().toISOString(),
    status: "internal deeper-invariant probe receipt (PHASE3N)",
    command: "node scripts/kakeya-deeper-invariant-probe.mjs",
    reduction:
      "PHASE3M sig+type separates all 33 known field-orbits with a UNIQUE collision {11-harmonic LOW, 19-harmonic HIGH}. Harmonic sig={chi(2),chi(-1),chi(2)}=-1-1-1 <=> q=3(mod8); harmonic level is a pure function of q (j=1728 fixed). So the deeper invariant reduces to: for q=3(mod8), harmonic LOW or HIGH?",
    epistemicsNote:
      "DEF-LOW = verified completion at (q-3)/2 (hard). PROB-HIGH = descent could not beat (q-1)/2 (upper bound; the only gap ever seen was q=11 pure, closed by descent). Controls reproduce known harmonic levels at q in {5,7,13,17,29,37}.",
    controlsPass,
    falsifier: {
      name: "DEEPER_INVARIANT_INSTRUMENT_MISMATCH",
      description:
        "Instrument-only: fires if any completion fails verification (completeness + size identity) or a control field's construction level disagrees with its known harmonic level. Battery outcomes are measurements.",
      status: instrumentPass ? "clear" : "fired",
    },
    controlRows,
    classRows,
    battery,
    elapsedSeconds: (Date.now() - t0) / 1000,
    pass: instrumentPass,
  };

  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");

  console.log("Controls (harmonic, must match known):");
  for (const r of controlRows)
    console.log(`  q=${String(r.q).padEnd(3)} ex=${r.ex} ${r.level.padEnd(4)} expected=${r.expected} ${r.level === r.expected ? "OK" : "MISMATCH"} verified=${r.verified}`);
  console.log("q = 3 (mod 8) harmonic class:");
  for (const r of classRows)
    console.log(`  q=${String(r.q).padEnd(3)} ex=${r.ex} ${r.level.padEnd(4)} [${r.epistemics}] verified=${r.verified}`);
  console.log("Deeper-invariant battery (train = q=3(mod8) harmonic levels):");
  for (const [name, b] of Object.entries(battery))
    console.log(`  ${name.padEnd(20)} consistent=${b.consistent} (defined on ${b.definedOn})${b.consistent ? "" : "  " + b.collisions.join(" ; ")}`);
  console.log(
    `KAK_DEEPER_INVARIANT controls=${controlsPass ? "pass" : "fail"} falsifier=${manifest.falsifier.status} elapsed=${manifest.elapsedSeconds.toFixed(1)}s out=${outDir}`,
  );
  process.exit(instrumentPass ? 0 : 1);
}

main();
