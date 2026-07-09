#!/usr/bin/env node
// PHASE3P support - dual-arc verification for the parabola-optimality lemma.
//
// Projective duality turns a one-per-direction 4-star completion (q+1 lines)
// into q+1 points of PG(2,q)*. A primal point of multiplicity m (m completion
// lines through it) dualizes to a "m-rich" line (a line of the dual plane
// meeting the q+1 dual points in exactly m). So:
//   sacrifice = 3 + T,  T = # of 3-rich dual lines,  pivot O = the unique
//   4-rich dual line (the 4 star lines are concurrent at O, hence their duals
//   are collinear on the dual line O*).
// Minimizing T = making the q+1 dual points as arc-like as possible. This
// script verifies, per orbit:
//   (D1) the 4 star lines dualize to 4 points on one line (O*);
//   (D2) # 4-rich lines = 1 (= O*) and # 3-rich lines = T = sacrifice - 3,
//        for the optimal parabola completion;
//   (D3) the q-3 non-star dual points lie on a single conic (dual conic),
//        i.e. the completion lines are tangent to a conic (Segre backbone).
// This is the computational ground for the PHASE3P propositions. No claim
// beyond the workbench. No Euclidean claim.

import fs from "node:fs";
import path from "node:path";
import * as Core from "../kakeya/kakeya-core.js";

const DEFAULT_OUT = path.join("results", "kakeya", "dual-arc-verify");
const FIELDS = [5, 7, 11, 13, 17, 19];
const INF = -1;

function mod(x, q) {
  return ((x % q) + q) % q;
}
function inv(x, q) {
  for (let i = 1; i < q; i++) if ((x * i) % q === 1) return i;
  throw new Error(`no inverse of ${x} mod ${q}`);
}

// --- orbits (reuse) --------------------------------------------------------------
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
function labelOf(l, q) {
  const six = sixSet(l, q);
  const harmonic = [2, mod((q + 1) / 2, q), q - 1].sort((x, y) => x - y);
  if (six.length === 3 && six.join(",") === harmonic.join(",")) return "harmonic";
  if (six.length === 2 && six.every((v) => mod(v * v - v + 1, q) === 0)) return "equianharmonic";
  return "generic";
}
function orbitReps(q) {
  const dc = q + 1;
  const dv = (i) => (i === q ? INF : i);
  const reps = new Map();
  for (let a = 0; a < dc; a++)
    for (let b = a + 1; b < dc; b++)
      for (let c = b + 1; c < dc; c++)
        for (let d = c + 1; d < dc; d++) {
          const l = crossRatio(dv(a), dv(b), dv(c), dv(d), q);
          const key = sixSet(l, q).join(",");
          if (!reps.has(key)) reps.set(key, { label: labelOf(l, q), quad: [a, b, c, d] });
        }
  const byBase = new Map();
  for (const k of [...reps.keys()].sort()) byBase.set(reps.get(k).label, (byBase.get(reps.get(k).label) ?? 0) + 1);
  const cnt = new Map();
  const out = [];
  for (const k of [...reps.keys()].sort()) {
    const r = reps.get(k);
    let label = r.label;
    if (byBase.get(r.label) > 1) {
      const n = (cnt.get(r.label) ?? 0) + 1;
      cnt.set(r.label, n);
      label = `${r.label}-${String.fromCharCode(96 + n)}`;
    }
    out.push({ label, quad: r.quad });
  }
  return out;
}

// --- construction (best parabola completion, alpha=1) ----------------------------
function tangentPoints(q, axis, alpha, beta, gamma, mu, out) {
  const t = mod((mu - beta) * inv(mod(2 * alpha, q), q), q);
  const c = mod(gamma - alpha * t * t, q);
  if (axis === q) for (let x = 0; x < q; x++) out[x] = Core.pointIndex(x, (mu * x + c) % q, q);
  else {
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
function pureBest(q, quad) {
  const dirs = Core.directions(q);
  const n = Core.pointCount(q);
  const starPts = quad.map((i) => [...Core.lineMask(dirs[i], 0, q)]);
  const needed = [];
  for (let i = 0; i < dirs.length; i++) if (!quad.includes(i)) needed.push(i);
  const stamp = new Int32Array(n);
  const cnt = new Uint8Array(n);
  let epoch = 0;
  const buf = new Array(q);
  let best = { ex: Infinity };
  for (const axis of quad) {
    const mus = needed.map((d) => slopePreimage(q, axis, d));
    for (let beta = 0; beta < q; beta++)
      for (let gamma = 0; gamma < q; gamma++) {
        epoch++;
        let sac = 0;
        for (const pts of starPts)
          for (const p of pts) {
            if (stamp[p] !== epoch) { stamp[p] = epoch; cnt[p] = 1; }
            else { const m = cnt[p]; if (m >= 2) sac += m - 1; cnt[p] = m + 1; }
          }
        for (const mu of mus) {
          tangentPoints(q, axis, 1, beta, gamma, mu, buf);
          for (let x = 0; x < q; x++) {
            const p = buf[x];
            if (stamp[p] !== epoch) { stamp[p] = epoch; cnt[p] = 1; }
            else { const m = cnt[p]; if (m >= 2) sac += m - 1; cnt[p] = m + 1; }
          }
        }
        const ex = sac - (q - 1) / 2;
        if (ex < best.ex) best = { ex, sac, axis, beta, gamma };
      }
  }
  const assign = new Map();
  for (const d of needed) assign.set(d, interceptOfTangent(q, best.axis, 1, best.beta, best.gamma, d));
  return { ...best, assign, needed };
}

// --- projective duality ----------------------------------------------------------
// Line dual point (homogeneous, normalized to a canonical representative).
function lineDual(dir, b, q) {
  // finite slope m=dir: y = m x + b -> m x - y + b = 0 -> [m, -1, b]
  // vertical dir=q: x = b -> x - b z = 0 -> [1, 0, -b]
  let v;
  if (dir === q) v = [1, 0, mod(-b, q)];
  else v = [dir, q - 1, b];
  return normalizeProj(v, q);
}
function normalizeProj(v, q) {
  // canonical rep: scale so first nonzero coord = 1
  const a = v.findIndex((x) => mod(x, q) !== 0);
  const s = inv(mod(v[a], q), q);
  return v.map((x) => mod(x * s, q));
}
function collinear(p1, p2, p3, q) {
  // determinant of the 3x3 = 0 mod q
  const [a1, b1, c1] = p1, [a2, b2, c2] = p2, [a3, b3, c3] = p3;
  const det =
    a1 * (b2 * c3 - b3 * c2) - b1 * (a2 * c3 - a3 * c2) + c1 * (a2 * b3 - a3 * b2);
  return mod(det, q) === 0;
}

// Count k-rich lines among a set of distinct projective points: partition all
// C(n,2) pairs by the line they span; a line with t points contributes C(t,2)
// pairs. Recover richness by grouping pairs under a canonical line key.
function richLineProfile(points, q) {
  const lineKey = (p1, p2) => {
    // cross product = the line through p1,p2, normalized
    const [a1, b1, c1] = p1, [a2, b2, c2] = p2;
    const L = [mod(b1 * c2 - b2 * c1, q), mod(c1 * a2 - c2 * a1, q), mod(a1 * b2 - a2 * b1, q)];
    return normalizeProj(L, q).join(",");
  };
  const linePts = new Map(); // lineKey -> Set of point indices
  for (let i = 0; i < points.length; i++)
    for (let j = i + 1; j < points.length; j++) {
      const key = lineKey(points[i], points[j]);
      if (!linePts.has(key)) linePts.set(key, new Set());
      linePts.get(key).add(i).add(j);
    }
  const profile = new Map(); // richness t -> count of lines
  for (const s of linePts.values()) {
    const t = s.size;
    profile.set(t, (profile.get(t) ?? 0) + 1);
  }
  return profile;
}

// Conic through 5 of the points, evaluated on the rest; returns max incidence.
function conicRow([x, y, z], q) {
  return [(x * x) % q, (x * y) % q, (y * y) % q, (x * z) % q, (y * z) % q, (z * z) % q];
}
function fitConic(five, q) {
  const M = five.map((p) => conicRow(p, q));
  let r = 0; const piv = [];
  for (let c = 0; c < 6 && r < M.length; c++) {
    let pr = -1;
    for (let i = r; i < M.length; i++) if (mod(M[i][c], q) !== 0) { pr = i; break; }
    if (pr === -1) continue;
    [M[r], M[pr]] = [M[pr], M[r]];
    const ip = inv(mod(M[r][c], q), q);
    for (let j = 0; j < 6; j++) M[r][j] = mod(M[r][j] * ip, q);
    for (let i = 0; i < M.length; i++) { if (i === r) continue; const f = mod(M[i][c], q); if (!f) continue; for (let j = 0; j < 6; j++) M[i][j] = mod(M[i][j] - f * M[r][j], q); }
    piv.push(c); r++;
  }
  if (r < 5) return null;
  const free = [...Array(6).keys()].find((c) => !piv.includes(c));
  if (free === undefined) return null;
  const co = new Array(6).fill(0); co[free] = 1;
  for (let i = piv.length - 1; i >= 0; i--) { const c = piv[i]; let s = 0; for (let j = c + 1; j < 6; j++) s = mod(s + M[i][j] * co[j], q); co[c] = mod(-s, q); }
  return co;
}
function allOnOneConic(points, q) {
  if (points.length < 5) return true;
  const co = fitConic(points.slice(0, 5), q);
  if (!co) return false;
  const ev = (p) => { const rw = conicRow(p, q); let s = 0; for (let j = 0; j < 6; j++) s = mod(s + rw[j] * co[j], q); return s; };
  return points.every((p) => ev(p) === 0);
}

function verify(q, rep) {
  const pure = pureBest(q, rep.quad);
  const dirs = Core.directions(q);
  const T = pure.sac - 3;

  // Dual points of the q+1 completion lines.
  const starDuals = rep.quad.map((i) => lineDual(i, 0, q));
  const nonStarDuals = [...pure.assign].map(([d, b]) => lineDual(d, b, q));
  const allDuals = [...starDuals, ...nonStarDuals];

  // D1: 4 star duals collinear.
  const d1 =
    collinear(starDuals[0], starDuals[1], starDuals[2], q) &&
    collinear(starDuals[0], starDuals[1], starDuals[3], q);

  // D2: rich-line profile - unique 4-rich (=O*) and #3-rich = T.
  const profile = richLineProfile(allDuals, q);
  const rich4 = profile.get(4) ?? 0;
  const rich3 = profile.get(3) ?? 0;
  const richHigher = [...profile.entries()].filter(([t]) => t >= 5).reduce((s, [, c]) => s + c, 0);
  const d2 = rich4 === 1 && rich3 === T && richHigher === 0;

  // D3: the q-3 non-star duals lie on one conic.
  const d3 = allOnOneConic(nonStarDuals, q);

  return {
    q, orbit: rep.label, sac: pure.sac, ex: pure.ex, T,
    rich4, rich3, richHigher,
    checks: { d1_star_collinear: d1, d2_rich_profile: d2, d3_nonstar_conic: d3 },
    ok: d1 && d2 && d3,
  };
}

function main() {
  const outDir = process.argv.includes("--out") ? process.argv[process.argv.indexOf("--out") + 1] : DEFAULT_OUT;
  const rows = [];
  for (const q of FIELDS) for (const rep of orbitReps(q)) rows.push(verify(q, rep));
  const pass = rows.every((r) => r.ok);
  const manifest = {
    artifactId: "KAK-PHASE3P-DUAL-ARC-VERIFY",
    generatedAt: new Date().toISOString(),
    status: "internal dual-arc verification (PHASE3P support)",
    command: "node scripts/kakeya-dual-arc-verify.mjs",
    statement:
      "For the optimal parabola completion: (D1) the 4 star lines dualize to 4 collinear points; (D2) the q+1 dual points have exactly one 4-rich line (O*), T 3-rich lines (T=sacrifice-3), and no >=5-rich line; (D3) the q-3 non-star dual points lie on one conic. So the completion lines are conic tangents and T = #3-secants of a near-arc, the Segre/Blokhuis-Mazzocca backbone of parabola-optimality.",
    falsifier: {
      name: "DUAL_ARC_MISMATCH",
      description: "Fires if D1/D2/D3 fail on any exact field-orbit.",
      status: pass ? "clear" : "fired",
    },
    rows,
    pass,
  };
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
  for (const r of rows)
    console.log(`q=${String(r.q).padEnd(3)} ${r.orbit.padEnd(14)} sac=${r.sac} T=${r.T} rich4=${r.rich4} rich3=${r.rich3} rich>=5=${r.richHigher} D1${r.checks.d1_star_collinear ? "+" : "!"}D2${r.checks.d2_rich_profile ? "+" : "!"}D3${r.checks.d3_nonstar_conic ? "+" : "!"}`);
  console.log(`KAK_DUAL_ARC_VERIFY pass=${pass} falsifier=${manifest.falsifier.status} out=${outDir}`);
  process.exit(pass ? 0 : 1);
}

main();
