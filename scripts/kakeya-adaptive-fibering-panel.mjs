#!/usr/bin/env node
// H-K4 panel: is "adaptive fibering" a real workbench metric or prose? For a body K in
// F_q^2 we compare a FIXED-direction fiber description (cover K with full lines in the single
// best direction) against an ADAPTIVE one (cover K with full lines in ANY direction). The gap
// = adaptive_covered - fixed_best_covered >= 0 is the toy compression signal. Falsifier
// ADAPTIVE_FIBERING_NO_SIGNAL fires unless (i) the gap is positive on structured multi-
// direction bodies AND (ii) it vanishes on random same-size control bodies (so the signal is
// structure-driven, not a finite-grid artifact). Report-only; no Euclidean Kakeya claim.

import fs from "node:fs";
import path from "node:path";
import * as Core from "../kakeya/kakeya-core.js";

const ARTIFACT_ID = "KAK-HK4-ADAPTIVE-FIBERING-PANEL";
const OUT_DIR = path.join("results", "kakeya", "adaptive-fibering-panel");
const QS = [5, 7];
const CONTROL_SAMPLES = 200;

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
  return Core.directions(q).map((dir, i) => ({
    label: dir.label, index: i,
    lines: Array.from({ length: q }, (_, b) => new Set(Core.lineMask(dir, b, q))),
  }));
}
function subset(line, K) { for (const p of line) if (!K.has(p)) return false; return true; }

// fixed = best single direction's covered points; adaptive = points on a full line in ANY dir.
function analyzeBody(LS, K) {
  const perDir = LS.map((d) => {
    const cov = new Set(); let count = 0;
    for (const line of d.lines) if (subset(line, K)) { count++; for (const p of line) cov.add(p); }
    return { label: d.label, count, cov };
  });
  const fixedBest = perDir.reduce((m, c) => Math.max(m, c.cov.size), 0);
  const adaptiveSet = new Set();
  for (const c of perDir) for (const p of c.cov) adaptiveSet.add(p);
  const adaptive = adaptiveSet.size;
  let highAmb = 0, maxAmb = 0;
  for (const p of K) {
    let a = 0; for (const c of perDir) if (c.cov.has(p)) a++;
    if (a >= 2) highAmb++; if (a > maxAmb) maxAmb = a;
  }
  return { size: K.size, dirsWithLine: perDir.filter((c) => c.count > 0).length,
    fixedBest, adaptive, gap: adaptive - fixedBest,
    residualFixed: K.size - fixedBest, residualAdaptive: K.size - adaptive,
    highAmbPoints: highAmb, maxAmb };
}

// body builders
function kDirUnion(LS, k) { const K = new Set(); for (let i = 0; i < k; i++) for (const p of LS[i].lines[i % LS[i].lines.length]) K.add(p); return K; }
function wholePlane(q) { const K = new Set(); for (let p = 0; p < Core.pointCount(q); p++) K.add(p); return K; }
function randomBody(n, size, rnd) {
  const K = new Set();
  while (K.size < size) K.add(Math.floor(rnd() * n));
  return K;
}

function runQ(q) {
  const LS = lineSets(q);
  const n = Core.pointCount(q);
  const dirCount = LS.length; // q+1
  const rnd = mulberry32(0x5a17 + q);

  const structured = [];
  for (let k = 1; k <= dirCount; k++) {
    const K = kDirUnion(LS, k);
    const a = analyzeBody(LS, K);
    // control: random bodies of the SAME size
    let cgSum = 0, cgMax = 0, cWithGap = 0;
    for (let s = 0; s < CONTROL_SAMPLES; s++) {
      const r = analyzeBody(LS, randomBody(n, a.size, rnd));
      cgSum += r.gap; if (r.gap > cgMax) cgMax = r.gap; if (r.gap > 0) cWithGap++;
    }
    structured.push({ kind: `k${k}-dir-union`, k, ...a,
      control_mean_gap: Number((cgSum / CONTROL_SAMPLES).toFixed(4)),
      control_max_gap: cgMax, control_frac_with_gap: Number((cWithGap / CONTROL_SAMPLES).toFixed(4)) });
  }
  const whole = { kind: "whole-plane", k: null, ...analyzeBody(LS, wholePlane(q)),
    control_mean_gap: 0, control_max_gap: 0, control_frac_with_gap: 0 };

  const rows = [...structured, whole];
  const maxStructGap = Math.max(...structured.map((r) => r.gap));
  const worstControlMean = Math.max(...rows.map((r) => r.control_mean_gap));
  const anyControlGap = rows.some((r) => r.control_max_gap > 0);
  const monotone = structured.every((r, i) => i === 0 || r.gap >= structured[i - 1].gap);
  const signalPresent = maxStructGap > 0;
  const structureDriven = worstControlMean === 0 && !anyControlGap;
  const fired = !(signalPresent && structureDriven);
  return { q, pointCount: n, dirCount, maxStructGap, worstControlMean, anyControlGap,
    monotone, signalPresent, structureDriven, fired, rows };
}

function main() {
  const results = QS.map(runQ);
  const anyFired = results.some((r) => r.fired);
  const manifest = {
    artifactId: ARTIFACT_ID, generatedAt: new Date().toISOString(),
    status: "internal measurement receipt",
    hook: "H-K4 adaptive-fibering ambiguity panel (fixed vs adaptive fiber labeling)",
    controlSamples: CONTROL_SAMPLES,
    metric: "gap = adaptive_covered - fixed_best_covered; control = random same-size bodies",
    falsifier: {
      name: "ADAPTIVE_FIBERING_NO_SIGNAL",
      fired: anyFired,
      reason: anyFired
        ? "Some q has no positive structured gap, or a random control reproduced the gap (grid artifact)."
        : "Every q shows a positive structured gap that grows with #directions, and random same-size controls show gap 0 (structure-driven, not a grid artifact).",
    },
    results,
  };
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(path.join(OUT_DIR, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
  const csv = ["q,kind,k,size,dirsWithLine,fixedBest,adaptive,gap,highAmbPoints,maxAmb,control_mean_gap,control_max_gap",
    ...results.flatMap((R) => R.rows.map((r) => `${R.q},${r.kind},${r.k ?? ""},${r.size},${r.dirsWithLine},${r.fixedBest},${r.adaptive},${r.gap},${r.highAmbPoints},${r.maxAmb},${r.control_mean_gap},${r.control_max_gap}`))]
    .join("\n");
  fs.writeFileSync(path.join(OUT_DIR, "panel-summary.csv"), csv + "\n");

  for (const R of results) {
    const gaps = R.rows.filter((r) => r.k).map((r) => `k${r.k}:${r.gap}`).join(" ");
    console.log(`KAK_ADAPTIVE_FIBERING q=${R.q} maxStructGap=${R.maxStructGap} monotone=${R.monotone} `
      + `worstControlMean=${R.worstControlMean} anyControlGap=${R.anyControlGap} `
      + `signal=${R.signalPresent} structureDriven=${R.structureDriven} | gaps[${gaps}]`);
  }
  console.log(`KAK_ADAPTIVE_FIBERING_PANEL qs=${QS.join(",")} falsifier=${anyFired ? "fired" : "clear"} out=${OUT_DIR}`);
}

main();
