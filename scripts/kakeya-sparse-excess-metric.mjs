#!/usr/bin/env node
// H-K4 redemption run: the PHASE4 null said the raw adaptive-fibering gap is a density
// artifact, and named the only candidate metric: EXCESS over a size-matched random control,
// restricted to the sparse regime (|K| <= q^2/2). This script pre-registers that metric and
// tries to kill it with a fresh falsifier, SPARSE_EXCESS_NO_METRIC, which fires if ANY gate
// fails at ANY q:
//   G1 SIGNAL       - structured sparse k-direction unions (k>=2) have excess > 0, strictly
//                     increasing in k.
//   G2 SPECIFICITY  - holdout random bodies (fresh draws, not used to fit the control) exceed
//                     the per-q detection threshold tau_q at rate <= 5% (false-positive rate).
//   G3 STRUCTURE    - broken decoys (same size as a k-union, but every line missing one point,
//                     points replaced randomly; no full line survives) score excess < tau_q.
// tau_q = half the smallest structured excess (k>=2) at that q. Report-only; no Euclidean claim.

import fs from "node:fs";
import path from "node:path";
import * as Core from "../kakeya/kakeya-core.js";

const ARTIFACT_ID = "KAK-HK4B-SPARSE-EXCESS-METRIC";
const OUT_DIR = path.join("results", "kakeya", "sparse-excess-metric");
const QS = [5, 7, 11];
const N_CONTROL = 500;
const N_HOLDOUT = 200;
const FPR_GATE = 0.05;

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function lineSets(q) {
  return Core.directions(q).map((dir) => ({
    label: dir.label,
    lines: Array.from({ length: q }, (_, b) => new Set(Core.lineMask(dir, b, q))),
  }));
}
function subset(line, K) { for (const p of line) if (!K.has(p)) return false; return true; }
function gapOf(LS, K) {
  let fixedBest = 0; const adaptive = new Set(); let dirsWithLine = 0;
  for (const d of LS) {
    const cov = new Set();
    for (const line of d.lines) if (subset(line, K)) for (const p of line) cov.add(p);
    if (cov.size > 0) dirsWithLine++;
    if (cov.size > fixedBest) fixedBest = cov.size;
    for (const p of cov) adaptive.add(p);
  }
  return { gap: adaptive.size - fixedBest, dirsWithLine };
}
function randomBody(n, size, rnd) {
  const K = new Set();
  while (K.size < size) K.add(Math.floor(rnd() * n));
  return K;
}
function kDirUnion(LS, k, q) {
  const K = new Set();
  for (let i = 0; i < k; i++) for (const p of LS[i].lines[i % q]) K.add(p);
  return K;
}
// broken decoy: same size as the k-union, each constituent line loses one non-shared point,
// replacements drawn randomly; redraw until no direction has a surviving full line.
function brokenDecoy(LS, k, q, n, rnd) {
  for (let attempt = 0; attempt < 50; attempt++) {
    const union = kDirUnion(LS, k, q);
    const K = new Set(union);
    for (let i = 0; i < k; i++) {
      const line = LS[i].lines[i % q];
      let removed = false;
      for (const p of line) {
        let shared = false;
        for (let j = 0; j < k; j++) if (j !== i && LS[j].lines[j % q].has(p)) { shared = true; break; }
        if (!shared && K.has(p)) { K.delete(p); removed = true; break; }
      }
      if (!removed) for (const p of line) if (K.has(p)) { K.delete(p); break; }
    }
    while (K.size < union.size) {
      const p = Math.floor(rnd() * n);
      if (!K.has(p)) K.add(p);
    }
    if (gapOf(LS, K).dirsWithLine === 0) return K;
  }
  return null; // could not build a line-free decoy (reported, counts as a gate failure)
}
const mean = (xs) => xs.reduce((s, x) => s + x, 0) / xs.length;

function runQ(q) {
  const LS = lineSets(q);
  const n = Core.pointCount(q);
  const sparseMax = Math.floor(n / 2);
  const rnd = mulberry32(0xb0d1 + q);

  // structured sparse ladder
  const ladder = [];
  for (let k = 1; ; k++) {
    const K = kDirUnion(LS, k, q);
    if (K.size > sparseMax) break;
    ladder.push({ k, size: K.size, gap: gapOf(LS, K).gap });
  }
  const sizes = [...new Set(ladder.map((r) => r.size))];

  // size-matched controls (fit) per structured size
  const control = {};
  for (const s of sizes) control[s] = mean(Array.from({ length: N_CONTROL }, () => gapOf(LS, randomBody(n, s, rnd)).gap));
  for (const r of ladder) r.excess = Number((r.gap - control[r.size]).toFixed(3));

  const structured = ladder.filter((r) => r.k >= 2);
  const tau = structured.length ? Math.min(...structured.map((r) => r.excess)) / 2 : null;

  // G1 signal
  const g1 = structured.length > 0 && structured.every((r) => r.excess > 0)
    && structured.every((r, i) => i === 0 || r.excess > structured[i - 1].excess);

  // G2 specificity on holdout randoms (fresh seed stream)
  const rndHold = mulberry32(0x0d0e + q);
  let fp = 0, held = 0;
  const holdoutMeans = {};
  for (const s of structured.map((r) => r.size)) {
    const ex = Array.from({ length: N_HOLDOUT }, () => gapOf(LS, randomBody(n, s, rndHold)).gap - control[s]);
    holdoutMeans[s] = Number(mean(ex).toFixed(3));
    for (const e of ex) { held++; if (tau != null && e >= tau) fp++; }
  }
  const fpr = held ? fp / held : null;
  const g2 = fpr != null && fpr <= FPR_GATE;

  // G3 broken decoys
  const rndDecoy = mulberry32(0xdec0 + q);
  const decoys = [];
  let g3 = structured.length > 0;
  for (const r of structured) {
    const D = brokenDecoy(LS, r.k, q, n, rndDecoy);
    if (!D) { decoys.push({ k: r.k, size: r.size, excess: null, built: false }); g3 = false; continue; }
    const ex = Number((gapOf(LS, D).gap - control[D.size]).toFixed(3));
    decoys.push({ k: r.k, size: D.size, excess: ex, built: true });
    if (!(tau != null && ex < tau)) g3 = false;
  }

  // density boundary sweep (report-only): first size where control mean leaves ~0
  const sweep = [];
  let boundary = null;
  for (let s = q; s <= n; s += Math.max(1, Math.floor(q / 2))) {
    const m = Number(mean(Array.from({ length: 120 }, () => gapOf(LS, randomBody(n, s, rnd)).gap)).toFixed(3));
    sweep.push({ size: s, control_mean_gap: m });
    if (boundary == null && m > 0.5) boundary = s;
  }

  return { q, pointCount: n, sparseMax, tau: tau != null ? Number(tau.toFixed(3)) : null,
    ladder, holdoutMeans, fpr: fpr != null ? Number(fpr.toFixed(4)) : null, decoys,
    densityBoundary: { firstSizeControlMeanAbove0_5: boundary, heuristic_q2_over_2: sparseMax, sweep },
    gates: { G1_signal: g1, G2_specificity: g2, G3_structure: g3 },
    pass: g1 && g2 && g3 };
}

function main() {
  const results = QS.map(runQ);
  const fired = !results.every((r) => r.pass);
  const manifest = {
    artifactId: ARTIFACT_ID, generatedAt: new Date().toISOString(),
    status: "internal measurement receipt (H-K4 redemption run)",
    metric: "excess(K) = gap(K) - mean gap of size-matched random control; sparse regime |K| <= q^2/2 only",
    preregistered_gates: {
      G1_signal: "structured k>=2 sparse unions: excess > 0, strictly increasing in k",
      G2_specificity: `holdout random false-positive rate <= ${FPR_GATE} at tau_q = min structured excess / 2`,
      G3_structure: "broken decoys (same size, no full lines) score excess < tau_q",
    },
    parameters: { qs: QS, nControl: N_CONTROL, nHoldout: N_HOLDOUT, fprGate: FPR_GATE, deterministicSeeds: true },
    falsifier: {
      name: "SPARSE_EXCESS_NO_METRIC", fired,
      reason: fired
        ? "A pre-registered gate failed: the excess statistic is not a structure metric even in the sparse regime."
        : "All three gates pass at every q: sparse excess is positive and monotone on structure, near-zero on holdout randoms (FPR within gate), and near-zero on same-size broken decoys - a structure-driven metric, not a size artifact.",
    },
    results,
  };
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(path.join(OUT_DIR, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
  const csv = ["q,kind,k,size,gap_or_excess,note",
    ...results.flatMap((R) => [
      ...R.ladder.map((r) => `${R.q},structured,${r.k},${r.size},${r.excess},gap=${r.gap}`),
      ...R.decoys.map((d) => `${R.q},broken-decoy,${d.k},${d.size},${d.excess},built=${d.built}`),
    ])].join("\n");
  fs.writeFileSync(path.join(OUT_DIR, "excess-summary.csv"), csv + "\n");

  for (const R of results) {
    const lad = R.ladder.filter((r) => r.k >= 2).map((r) => `k${r.k}:${r.excess}`).join(" ");
    const dec = R.decoys.map((d) => `k${d.k}:${d.excess}`).join(" ");
    console.log(`KAK_SPARSE_EXCESS q=${R.q} tau=${R.tau} structured[${lad}] decoys[${dec}] `
      + `fpr=${R.fpr} boundary=${R.densityBoundary.firstSizeControlMeanAbove0_5} (heuristic ${R.sparseMax}) `
      + `G1=${R.gates.G1_signal} G2=${R.gates.G2_specificity} G3=${R.gates.G3_structure}`);
  }
  console.log(`KAK_SPARSE_EXCESS_METRIC qs=${QS.join(",")} falsifier=${fired ? "fired" : "clear"} out=${OUT_DIR}`);
  process.exit(fired ? 1 : 0);
}

main();
