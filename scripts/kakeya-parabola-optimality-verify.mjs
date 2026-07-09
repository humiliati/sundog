#!/usr/bin/env node
// PHASE3Q - parabola-optimality: exhaustive finite-case proof + honest status.
//
// For small q the exact minimum completion is computable by full branch-and-
// bound (exhaustive with pruning => a rigorous minimum, not a heuristic). This
// script freshly re-derives, for q in {5,7,11,13}, per cross-ratio orbit:
//   exactMin  = min sacrifice over ALL completions (exhaustive B&B),
//   parabMin  = min sacrifice over parabola-tangent completions (pure search),
// and decides parabola-optimality (parabMin == exactMin) case by case. This is
// a THEOREM for each finite case (both are exact minima). It confirms:
//   - parabola-optimality HOLDS for every orbit at q in {5,7,13} and for
//     q=11 generic;
//   - it FAILS at q=11 harmonic (exactMin=4 < parabMin=5), the sole exception.
// The general q>=13 statement stays OPEN (see PHASE3Q receipt). No Euclidean
// claim.

import fs from "node:fs";
import path from "node:path";
import * as Core from "../kakeya/kakeya-core.js";

const DEFAULT_OUT = path.join("results", "kakeya", "parabola-optimality-verify");
const FIELDS = [5, 7, 11, 13]; // exhaustive B&B feasible
const INF = -1;

function mod(x, q) { return ((x % q) + q) % q; }
function inv(x, q) { for (let i = 1; i < q; i++) if ((x * i) % q === 1) return i; throw new Error("noinv"); }

// --- orbits ---
function crossRatio(a, b, c, d, q) {
  const diff = (x, y) => (x === INF || y === INF ? 1 : mod(x - y, q));
  return (diff(a, c) * diff(b, d) * inv((diff(a, d) * diff(b, c)) % q, q)) % q;
}
function sixSet(l, q) {
  const s = new Set([l, inv(l, q), mod(1 - l, q), inv(mod(1 - l, q), q), mod(l * inv(mod(l - 1, q), q), q), mod(mod(l - 1, q) * inv(l, q), q)]);
  return [...s].sort((x, y) => x - y);
}
function labelOf(l, q) {
  const six = sixSet(l, q);
  const h = [2, mod((q + 1) / 2, q), q - 1].sort((x, y) => x - y);
  if (six.length === 3 && six.join(",") === h.join(",")) return "harmonic";
  if (six.length === 2 && six.every((v) => mod(v * v - v + 1, q) === 0)) return "equianharmonic";
  return "generic";
}
function orbitReps(q) {
  const dc = q + 1, dv = (i) => (i === q ? INF : i), reps = new Map();
  for (let a = 0; a < dc; a++) for (let b = a + 1; b < dc; b++) for (let c = b + 1; c < dc; c++) for (let d = c + 1; d < dc; d++) {
    const key = sixSet(crossRatio(dv(a), dv(b), dv(c), dv(d), q), q).join(",");
    if (!reps.has(key)) reps.set(key, { label: labelOf(crossRatio(dv(a), dv(b), dv(c), dv(d), q), q), quad: [a, b, c, d] });
  }
  const byBase = new Map();
  for (const k of [...reps.keys()].sort()) byBase.set(reps.get(k).label, (byBase.get(reps.get(k).label) ?? 0) + 1);
  const cnt = new Map(), out = [];
  for (const k of [...reps.keys()].sort()) {
    const r = reps.get(k); let label = r.label;
    if (byBase.get(r.label) > 1) { const n = (cnt.get(r.label) ?? 0) + 1; cnt.set(r.label, n); label = `${r.label}-${String.fromCharCode(96 + n)}`; }
    out.push({ label, quad: r.quad });
  }
  return out;
}

// --- exhaustive exact minimum sacrifice over ALL completions (B&B) ---
function wordCount(q) { return Math.ceil(Core.pointCount(q) / 32); }
function pc32(v) { v = v - ((v >> 1) & 0x55555555); v = (v & 0x33333333) + ((v >> 2) & 0x33333333); return (((v + (v >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24; }

function exactMinSacrifice(q, quad) {
  const dirs = Core.directions(q), words = wordCount(q);
  const star = new Set();
  for (const i of quad) for (const p of Core.lineMask(dirs[i], 0, q)) star.add(p);
  const bits = Core.shadowBitset(q, star);
  const targets = [];
  for (let i = 0; i < dirs.length; i++) if (bits[i] === 0) targets.push(i);
  // masks of each candidate line (full line, not "added points") for sacrifice counting
  // We track multiplicity via incremental sacrifice over the full completion.
  const linePts = targets.map((i) => { const per = []; for (let b = 0; b < q; b++) per.push([...Core.lineMask(dirs[i], b, q)]); return per; });
  const n = Core.pointCount(q);
  const cnt = new Uint8Array(n);
  let sac = 0;
  const add = (pts) => { for (const p of pts) { const m = cnt[p]; if (m >= 2) sac += m - 1; cnt[p] = m + 1; } };
  const rem = (pts) => { for (const p of pts) { const m = cnt[p]; if (m >= 3) sac -= m - 2; cnt[p] = m - 1; } };
  // seed star lines
  const starLines = quad.map((i) => [...Core.lineMask(dirs[i], 0, q)]);
  for (const pts of starLines) add(pts);
  const baseSac = sac; // pivot contributes 3 (four star lines through O)
  const k = targets.length;
  let best = Infinity;
  // greedy upper bound to prune
  const rec = (level) => {
    if (sac >= best) return;
    if (level === k) { best = sac; return; }
    // order intercepts by resulting sacrifice (cheapest first) for pruning
    const order = [];
    for (let b = 0; b < q; b++) {
      add(linePts[level][b]); order.push([sac, b]); rem(linePts[level][b]);
    }
    order.sort((x, y) => x[0] - y[0]);
    for (const [s, b] of order) {
      if (s >= best) break;
      add(linePts[level][b]); rec(level + 1); rem(linePts[level][b]);
    }
  };
  rec(0);
  return best;
}

// --- best parabola completion sacrifice (pure search, alpha=1) ---
function tangentPoints(q, axis, beta, gamma, mu, out) {
  const t = mod((mu - beta) * inv(2, q), q), c = mod(gamma - t * t, q);
  if (axis === q) for (let x = 0; x < q; x++) out[x] = Core.pointIndex(x, (mu * x + c) % q, q);
  else { const s = axis; for (let x = 0; x < q; x++) { const y = (mu * x + c) % q; out[x] = Core.pointIndex(y, (x + s * y) % q, q); } }
}
function slopePre(q, axis, d) { if (axis === q) return d; if (d === q) return 0; return inv(mod(d - axis, q), q); }
function parabolaMinSacrifice(q, quad) {
  const dirs = Core.directions(q), n = Core.pointCount(q);
  const starPts = quad.map((i) => [...Core.lineMask(dirs[i], 0, q)]);
  const needed = []; for (let i = 0; i < dirs.length; i++) if (!quad.includes(i)) needed.push(i);
  const stamp = new Int32Array(n), cnt = new Uint8Array(n); let epoch = 0; const buf = new Array(q);
  let best = Infinity;
  for (const axis of quad) {
    const mus = needed.map((d) => slopePre(q, axis, d));
    for (let beta = 0; beta < q; beta++) for (let gamma = 0; gamma < q; gamma++) {
      epoch++; let sac = 0;
      for (const pts of starPts) for (const p of pts) { if (stamp[p] !== epoch) { stamp[p] = epoch; cnt[p] = 1; } else { const m = cnt[p]; if (m >= 2) sac += m - 1; cnt[p] = m + 1; } }
      for (const mu of mus) { tangentPoints(q, axis, beta, gamma, mu, buf); for (let x = 0; x < q; x++) { const p = buf[x]; if (stamp[p] !== epoch) { stamp[p] = epoch; cnt[p] = 1; } else { const m = cnt[p]; if (m >= 2) sac += m - 1; cnt[p] = m + 1; } } }
      if (sac < best) best = sac;
    }
  }
  return best;
}

function main() {
  const outDir = process.argv.includes("--out") ? process.argv[process.argv.indexOf("--out") + 1] : DEFAULT_OUT;
  const rows = [];
  for (const q of FIELDS) for (const rep of orbitReps(q)) {
    const exact = exactMinSacrifice(q, rep.quad);
    const parab = parabolaMinSacrifice(q, rep.quad);
    rows.push({ q, orbit: rep.label, exactSacrifice: exact, parabolaSacrifice: parab, exactEx: exact - (q - 1) / 2, parabolaEx: parab - (q - 1) / 2, parabolaOptimal: exact === parab });
  }
  // The finite theorem: parabola-optimal everywhere on these fields except q=11 harmonic.
  const exceptions = rows.filter((r) => !r.parabolaOptimal);
  const onlyException11h = exceptions.length === 1 && exceptions[0].q === 11 && exceptions[0].orbit === "harmonic";
  const manifest = {
    artifactId: "KAK-PHASE3Q-PARABOLA-OPTIMALITY-VERIFY",
    generatedAt: new Date().toISOString(),
    status: "internal exhaustive finite-case verification (PHASE3Q)",
    command: "node scripts/kakeya-parabola-optimality-verify.mjs",
    theorem:
      "For q in {5,7,11,13}, both min-sacrifice quantities are exact minima (exhaustive B&B / full parabola search). Parabola-optimality (parabolaSacrifice == exactSacrifice) holds for every orbit EXCEPT q=11 harmonic (exact ex=4 < parabola ex=5). This is a rigorous finite theorem; the general q>=13 statement is open.",
    falsifier: {
      name: "PARABOLA_OPT_FINITE_MISMATCH",
      description: "Fires if any orbit other than q=11 harmonic is parabola-suboptimal on these exhaustively-solved fields, or if a parabola beats the exact minimum (impossible).",
      status: onlyException11h && rows.every((r) => r.parabolaSacrifice >= r.exactSacrifice) ? "clear" : "fired",
    },
    rows,
    onlyException11h,
  };
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
  for (const r of rows) console.log(`q=${String(r.q).padEnd(3)} ${r.orbit.padEnd(14)} exactEx=${r.exactEx} parabolaEx=${r.parabolaEx} ${r.parabolaOptimal ? "PARABOLA-OPTIMAL" : "NON-PARABOLA (exception)"}`);
  console.log(`KAK_PARABOLA_OPT_VERIFY only_exception_11harmonic=${onlyException11h} falsifier=${manifest.falsifier.status} out=${outDir}`);
  process.exit(manifest.falsifier.status === "clear" ? 0 : 1);
}
main();
