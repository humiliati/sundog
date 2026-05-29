#!/usr/bin/env node
// scripts/yang-mills-phase2-su2-3d-ensemble.mjs
//
// Yang-Mills Phase 2 v0 - SU(2) 3D per-beta ensemble runner.

import * as fs from "node:fs";
import * as path from "node:path";
import * as crypto from "node:crypto";
import { execSync } from "node:child_process";
import { performance } from "node:perf_hooks";

import {
  createSU2Lattice,
  cloneSU2Lattice,
  combinedSweep,
  meanPlaquetteByOrientation,
  orientationRelativeSpread,
  computeSignatureV1,
  computeHeldoutV1,
  estimateTauIntSokal,
  mulberry32,
  deriveSubstreamSeed,
  maxLinkUnitarityResidual,
} from "./lib/yang-mills-su2-3d-core.mjs";

const LOCKED_BASE = Object.freeze({
  cell: "SU2_3D",
  latticeSize: "12x12x12",
  boundary: "periodic",
  action: "Wilson",
  generator: "su2_heatbath_overrelax_v1",
  overrelaxPerHeatbath: 4,
  burnIn: 2000,
  pilotSweeps: 512,
  thinning: 32,
  measurements: 32,
  signatureVocab: "v1",
  heldoutVocab: "v1",
  gammaHeldEpsilon: 1e-10,
});

const BETA_LOCKS = Object.freeze({
  "2.0": {
    beta: 2.0,
    seed: 202605290201,
    outRequired: "results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.0_ensemble_v0",
  },
  "2.4": {
    beta: 2.4,
    seed: 202605290202,
    outRequired: "results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.4_ensemble_v0",
  },
  "2.8": {
    beta: 2.8,
    seed: 202605290203,
    outRequired: "results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.8_ensemble_v0",
  },
});

const THRESH = Object.freeze({
  wallClockSeconds: 10 * 60,
  tauIntMax: 16.0,
  thinningTauRatioMin: 2.0,
  heatbathFallbackFractionMax: 0.001,
  unitarityFrobeniusResidualMax: 1e-10,
  orientationRelativeSpreadMax: 5e-2,
});

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) {
      out[key] = true;
    } else {
      out[key] = next;
      i++;
    }
  }
  return out;
}

function asNumber(v, name) {
  const n = Number(v);
  if (!Number.isFinite(n)) throw new Error(`flag --${name} must be finite, got: ${v}`);
  return n;
}

function asInt(v, name) {
  const n = Number(v);
  if (!Number.isInteger(n)) throw new Error(`flag --${name} must be integer, got: ${v}`);
  return n;
}

function betaKey(beta) {
  return beta.toFixed(1);
}

function validateLockedArgs(args) {
  const failures = [];
  const check = (key, expected, actual) => {
    if (actual !== expected) failures.push(`--${key} expected ${expected}, got ${actual}`);
  };
  const beta = asNumber(args.beta, "beta");
  const betaLock = BETA_LOCKS[betaKey(beta)];
  if (!betaLock || betaLock.beta !== beta) {
    failures.push(`--beta expected one of 2.0,2.4,2.8, got ${args.beta}`);
    return { failures, betaLock: null };
  }
  check("cell", LOCKED_BASE.cell, args.cell);
  check("lattice-size", LOCKED_BASE.latticeSize, args["lattice-size"]);
  check("beta", betaLock.beta, beta);
  check("boundary", LOCKED_BASE.boundary, args.boundary);
  check("action", LOCKED_BASE.action, args.action);
  check("generator", LOCKED_BASE.generator, args.generator);
  check("overrelax-per-heatbath", LOCKED_BASE.overrelaxPerHeatbath, asInt(args["overrelax-per-heatbath"], "overrelax-per-heatbath"));
  check("seed", betaLock.seed, asInt(args.seed, "seed"));
  check("burn-in", LOCKED_BASE.burnIn, asInt(args["burn-in"], "burn-in"));
  check("pilot-sweeps", LOCKED_BASE.pilotSweeps, asInt(args["pilot-sweeps"], "pilot-sweeps"));
  check("thinning", LOCKED_BASE.thinning, asInt(args.thinning, "thinning"));
  check("measurements", LOCKED_BASE.measurements, asInt(args.measurements, "measurements"));
  check("signature-vocab", LOCKED_BASE.signatureVocab, args["signature-vocab"]);
  check("heldout-vocab", LOCKED_BASE.heldoutVocab, args["heldout-vocab"]);
  check("gamma-held-epsilon", LOCKED_BASE.gammaHeldEpsilon, asNumber(args["gamma-held-epsilon"], "gamma-held-epsilon"));
  check("out", betaLock.outRequired, (args.out || "").replace(/\\/g, "/"));
  return { failures, betaLock };
}

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function writeText(p, content) {
  ensureDir(path.dirname(p));
  fs.writeFileSync(p, content, "utf8");
}

function writeJSON(p, obj) {
  writeText(p, JSON.stringify(obj, null, 2) + "\n");
}

function formatNumber(v) {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return "0";
  if (Math.abs(v) < 1e-6 || Math.abs(v) >= 1e16) return v.toExponential(12);
  return v.toFixed(12).replace(/0+$/, "").replace(/\.$/, "");
}

function csvEscape(v) {
  const s = typeof v === "number" ? formatNumber(v) : String(v);
  if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function writeCSV(p, header, rows) {
  const lines = [header.join(",")];
  for (const row of rows) lines.push(row.map(csvEscape).join(","));
  writeText(p, lines.join("\n") + "\n");
}

function listFiles(rootDir) {
  const out = [];
  const walk = (d) => {
    for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
      const p = path.join(d, entry.name);
      if (entry.isDirectory()) walk(p);
      else out.push(p);
    }
  };
  walk(rootDir);
  return out;
}

function sha256OfFile(p) {
  return crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex");
}

function collectHashes(outDir, excluded = new Set()) {
  const hashes = {};
  for (const p of listFiles(outDir)) {
    const rel = path.relative(outDir, p).replace(/\\/g, "/");
    if (!excluded.has(rel)) hashes[rel] = sha256OfFile(p);
  }
  return hashes;
}

function finalizeHashes(outDir) {
  writeJSON(path.join(outDir, "hashes.json"), collectHashes(outDir, new Set(["hashes.json"])));
}

function getGitInfo() {
  try {
    const codeCommit = execSync("git rev-parse HEAD", { stdio: ["ignore", "pipe", "ignore"] }).toString().trim();
    const status = execSync("git status --porcelain", { stdio: ["ignore", "pipe", "ignore"] }).toString().trim();
    return { codeCommit, gitDirty: status.length > 0 };
  } catch {
    return { codeCommit: "unknown", gitDirty: null };
  }
}

function reconstructCommandLine() {
  return [process.argv[0], process.argv[1], ...process.argv.slice(2)].join(" ");
}

function makeStats() {
  return { heatbathLinkUpdates: 0, heatbathFallbackCount: 0, heatbathRejectionAttempts: 0 };
}

function fallbackFraction(stats) {
  return stats.heatbathLinkUpdates ? stats.heatbathFallbackCount / stats.heatbathLinkUpdates : 0;
}

function valueStats(values) {
  let mean = 0;
  for (const v of values) mean += v;
  mean /= values.length;
  let variance = 0;
  for (const v of values) {
    const d = v - mean;
    variance += d * d;
  }
  variance /= values.length;
  return { mean, variance };
}

function gammaHeld(heldout, epsilon) {
  const areas = [4, 6, 9];
  const values = [heldout.W14_mean, heldout.W23_mean, heldout.W33_mean];
  const y = values.map((v) => Math.log(Math.max(v, epsilon)));
  const clamped = values.some((v) => v <= epsilon);
  const n = areas.length;
  let sumA = 0, sumY = 0, sumAA = 0, sumAY = 0;
  for (let i = 0; i < n; i++) {
    sumA += areas[i];
    sumY += y[i];
    sumAA += areas[i] * areas[i];
    sumAY += areas[i] * y[i];
  }
  const slope = (n * sumAY - sumA * sumY) / (n * sumAA - sumA * sumA);
  return { gammaHeld: -slope, clamped };
}

function buildManifest({ betaLock, tauResult, wallClockSeconds, codeInfo, stats, maxUnitResidual, orientationMeans, orientationVariances, orientationSpread, artifactHashes }) {
  return {
    phase: "phase2",
    cell: LOCKED_BASE.cell,
    latticeSize: [12, 12, 12],
    beta: betaLock.beta,
    boundary: LOCKED_BASE.boundary,
    action: LOCKED_BASE.action,
    generator: LOCKED_BASE.generator,
    updateMix: { heatbath: 1, overrelaxation: LOCKED_BASE.overrelaxPerHeatbath },
    masterSeed: betaLock.seed,
    burnInSweeps: LOCKED_BASE.burnIn,
    pilotTauIntPlaquette: tauResult ? tauResult.tauInt : null,
    registeredThinning: LOCKED_BASE.thinning,
    retainedConfigurations: LOCKED_BASE.measurements,
    signatureVocabularyVersion: LOCKED_BASE.signatureVocab,
    heldOutTargetVocabularyVersion: LOCKED_BASE.heldoutVocab,
    gammaHeldEpsilonFloor: LOCKED_BASE.gammaHeldEpsilon,
    heatbathFallbackCount: stats.heatbathFallbackCount,
    heatbathFallbackFraction: fallbackFraction(stats),
    heatbathLinkUpdates: stats.heatbathLinkUpdates,
    heatbathRejectionAttempts: stats.heatbathRejectionAttempts,
    linkUnitarityMaxFrobenius: maxUnitResidual,
    perOrientationMeanPlaquette: orientationMeans,
    perOrientationVariancePlaquette: orientationVariances,
    perOrientationRelativeSpread: orientationSpread,
    controlSetDeclared: [
      "CTRL_META",
      "CTRL_RAW",
      "CTRL_RAND",
      "CTRL_RAND_STRAT",
      "CTRL_PERM",
      "CTRL_GAUGE_RAND",
      "CTRL_FINITE_SIZE",
    ],
    controlSetScored: ["ensemble_only_phase2_aggregation_scores_controls"],
    codeCommit: codeInfo.codeCommit,
    gitDirty: codeInfo.gitDirty,
    commandLine: reconstructCommandLine(),
    wallClockSeconds,
    artifactHashes,
  };
}

async function main() {
  const start = performance.now();
  const args = parseArgs(process.argv.slice(2));
  const { failures, betaLock } = validateLockedArgs(args);
  if (failures.length > 0) {
    console.error("[YM-P2-ENS] manifest validation FAILED:");
    for (const f of failures) console.error(`  - ${f}`);
    process.exit(2);
  }

  const outDir = path.resolve(args.out);
  for (const sub of ["autocorr_pilot", "configs", "signatures", "heldout"]) {
    ensureDir(path.join(outDir, sub));
  }

  const L = 12;
  const state = createSU2Lattice(L, deriveSubstreamSeed(betaLock.seed, "init"));
  const rng = mulberry32(deriveSubstreamSeed(betaLock.seed, "sweep"));
  const stats = makeStats();

  console.log(`[YM-P2-ENS] start: beta=${betaLock.beta} L=${L} seed=${betaLock.seed}`);
  console.log(`[YM-P2-ENS] burn-in: ${LOCKED_BASE.burnIn} combined sweeps`);
  for (let s = 0; s < LOCKED_BASE.burnIn; s++) {
    combinedSweep(state, betaLock.beta, LOCKED_BASE.overrelaxPerHeatbath, rng, stats);
  }
  console.log(`[YM-P2-ENS] burn-in done; fallback fraction ${fallbackFraction(stats).toExponential(3)}`);

  console.log(`[YM-P2-ENS] pilot: ${LOCKED_BASE.pilotSweeps} combined sweeps`);
  const pilotSeries = new Float64Array(LOCKED_BASE.pilotSweeps);
  const pilotXY = new Float64Array(LOCKED_BASE.pilotSweeps);
  const pilotXZ = new Float64Array(LOCKED_BASE.pilotSweeps);
  const pilotYZ = new Float64Array(LOCKED_BASE.pilotSweeps);
  for (let s = 0; s < LOCKED_BASE.pilotSweeps; s++) {
    combinedSweep(state, betaLock.beta, LOCKED_BASE.overrelaxPerHeatbath, rng, stats);
    const by = meanPlaquetteByOrientation(state);
    pilotXY[s] = by.xy;
    pilotXZ[s] = by.xz;
    pilotYZ[s] = by.yz;
    pilotSeries[s] = (by.xy + by.xz + by.yz) / 3;
  }
  const tauResult = estimateTauIntSokal(pilotSeries, 6);
  const xyStats = valueStats(pilotXY);
  const xzStats = valueStats(pilotXZ);
  const yzStats = valueStats(pilotYZ);
  const orientationMeans = { xy: xyStats.mean, xz: xzStats.mean, yz: yzStats.mean };
  const orientationVariances = { xy: xyStats.variance, xz: xzStats.variance, yz: yzStats.variance };
  const orientSpread = orientationRelativeSpread(orientationMeans);
  console.log(`[YM-P2-ENS] pilot done; mean ${tauResult.mean.toFixed(6)}, tau_int ${tauResult.tauInt.toFixed(3)}, orientation spread ${orientSpread.toExponential(3)}`);

  writeCSV(path.join(outDir, "autocorr_pilot", "plaquette_series.csv"), ["sweep_index", "mean_plaquette"], Array.from(pilotSeries, (v, i) => [i, v]));
  writeCSV(path.join(outDir, "autocorr_pilot", "plaquette_by_orientation.csv"), ["sweep_index", "xy", "xz", "yz"], Array.from(pilotSeries, (_, i) => [i, pilotXY[i], pilotXZ[i], pilotYZ[i]]));

  const elapsedSoFar = (performance.now() - start) / 1000;
  const sweepsSoFar = LOCKED_BASE.burnIn + LOCKED_BASE.pilotSweeps;
  const predictedTotal =
    elapsedSoFar +
    (elapsedSoFar / sweepsSoFar) * (LOCKED_BASE.thinning * LOCKED_BASE.measurements) +
    LOCKED_BASE.measurements * 0.8;
  if (predictedTotal > THRESH.wallClockSeconds) {
    const wallClockSeconds = (performance.now() - start) / 1000;
    const maxUnitResidual = maxLinkUnitarityResidual(state);
    writeJSON(path.join(outDir, "summary.json"), {
      branch: "Z void_run",
      reason: "pilot timing extrapolation exceeds 10-minute compute cap",
      predictedTotalSeconds: predictedTotal,
      capSeconds: THRESH.wallClockSeconds,
      wallClockSeconds,
    });
    const artifactHashes = collectHashes(outDir, new Set(["manifest.json", "hashes.json"]));
    writeJSON(path.join(outDir, "manifest.json"), buildManifest({ betaLock, tauResult, wallClockSeconds, codeInfo: getGitInfo(), stats, maxUnitResidual, orientationMeans, orientationVariances, orientationSpread: orientSpread, artifactHashes }));
    finalizeHashes(outDir);
    process.exit(3);
  }

  console.log(`[YM-P2-ENS] measurements: ${LOCKED_BASE.measurements} configs at thinning ${LOCKED_BASE.thinning}`);
  const configs = [];
  const signatureRows = [];
  const heldoutRows = [];
  const heldoutSummaryRows = [];
  for (let m = 0; m < LOCKED_BASE.measurements; m++) {
    for (let s = 0; s < LOCKED_BASE.thinning; s++) {
      combinedSweep(state, betaLock.beta, LOCKED_BASE.overrelaxPerHeatbath, rng, stats);
    }
    const cfg = cloneSU2Lattice(state);
    configs.push(cfg);
    const sig = computeSignatureV1(cfg);
    const held = computeHeldoutV1(cfg);
    const gamma = gammaHeld(held, LOCKED_BASE.gammaHeldEpsilon);
    signatureRows.push([m, sig.W11_mean, sig.W11_var, sig.W12_mean, sig.W12_var, sig.W13_mean, sig.W13_var, sig.W22_mean, sig.W22_var]);
    heldoutRows.push([m, held.W14_mean, held.W14_var, held.W23_mean, held.W23_var, held.W33_mean, held.W33_var]);
    heldoutSummaryRows.push([m, held.W14_mean, held.W23_mean, held.W33_mean, gamma.gammaHeld, gamma.clamped ? 1 : 0]);
  }

  writeText(
    path.join(outDir, "configs", "su2_links.jsonl"),
    configs.map((cfg, configIdx) => JSON.stringify({ configIdx, shape: [3, L, L, L, 4], quaternionLinks: Array.from(cfg.links) })).join("\n") + "\n",
  );
  writeCSV(path.join(outDir, "signatures", "signature_vectors.csv"), ["config_idx", "W11_mean", "W11_var", "W12_mean", "W12_var", "W13_mean", "W13_var", "W22_mean", "W22_var"], signatureRows);
  writeCSV(path.join(outDir, "heldout", "heldout_loop_values.csv"), ["config_idx", "W14_mean", "W14_var", "W23_mean", "W23_var", "W33_mean", "W33_var"], heldoutRows);
  writeCSV(path.join(outDir, "heldout", "heldout_summary.csv"), ["config_idx", "W14_mean", "W23_mean", "W33_mean", "gamma_held", "clamped"], heldoutSummaryRows);

  const thinRatio = LOCKED_BASE.thinning / Math.max(tauResult.tauInt, 1e-12);
  const heatbathFallbackFrac = fallbackFraction(stats);
  const maxUnitResidual = maxLinkUnitarityResidual(state);
  const tauOk = tauResult.tauInt <= THRESH.tauIntMax;
  const thinOk = thinRatio >= THRESH.thinningTauRatioMin;
  const fallbackOk = heatbathFallbackFrac <= THRESH.heatbathFallbackFractionMax;
  const unitOk = maxUnitResidual <= THRESH.unitarityFrobeniusResidualMax;
  const orientationOk = orientSpread <= THRESH.orientationRelativeSpreadMax;

  let branch = "P1-A ensemble_health_pass";
  let reason = "all Phase-1-inherited ensemble health thresholds pass";
  if (!unitOk) {
    branch = "YM-P1-QUAR-D unitarity_drift";
    reason = `link unitarity Frobenius residual ${maxUnitResidual} exceeds ${THRESH.unitarityFrobeniusResidualMax}`;
  } else if (!orientationOk) {
    branch = "YM-P1-QUAR-C orientation_anisotropy";
    reason = `per-orientation mean-plaquette relative spread ${orientSpread} exceeds ${THRESH.orientationRelativeSpreadMax}`;
  } else if (!tauOk || !thinOk) {
    branch = "YM-P1-NEG-X autocorrelation_underflow";
    reason = `tau_int=${tauResult.tauInt}, thinning=${LOCKED_BASE.thinning}, thin/tau=${thinRatio.toFixed(3)}`;
  } else if (!fallbackOk) {
    branch = "YM-P1-QUAR-B heatbath_pathology";
    reason = `heatbath fallback fraction ${heatbathFallbackFrac} exceeds ${THRESH.heatbathFallbackFractionMax}`;
  }

  const wallClockSeconds = (performance.now() - start) / 1000;
  const summary = {
    branch,
    reason,
    beta: betaLock.beta,
    pilot: { tauInt: tauResult.tauInt, window: tauResult.window, meanPlaquette: tauResult.mean, variancePlaquette: tauResult.variance, thinningTauRatio: thinRatio },
    orientation: { meanPlaquette: orientationMeans, variancePlaquette: orientationVariances, relativeSpread: orientSpread, thresholdRelativeSpread: THRESH.orientationRelativeSpreadMax },
    measurements: { retained: configs.length },
    heldout: { gammaHeldEpsilonFloor: LOCKED_BASE.gammaHeldEpsilon, clampCount: heldoutSummaryRows.filter((r) => r[5] === 1).length },
    heatbath: { linkUpdates: stats.heatbathLinkUpdates, fallbackCount: stats.heatbathFallbackCount, fallbackFraction: heatbathFallbackFrac, rejectionAttempts: stats.heatbathRejectionAttempts, thresholdFallbackFraction: THRESH.heatbathFallbackFractionMax },
    linkUnitarityMaxFrobenius: maxUnitResidual,
    thresholds: THRESH,
    wallClockSeconds,
  };
  writeJSON(path.join(outDir, "summary.json"), summary);
  const artifactHashes = collectHashes(outDir, new Set(["manifest.json", "hashes.json"]));
  writeJSON(path.join(outDir, "manifest.json"), buildManifest({ betaLock, tauResult, wallClockSeconds, codeInfo: getGitInfo(), stats, maxUnitResidual, orientationMeans, orientationVariances, orientationSpread: orientSpread, artifactHashes }));
  finalizeHashes(outDir);

  console.log(`[YM-P2-ENS] verdict: ${branch}`);
  console.log(`[YM-P2-ENS] ${reason}`);
  console.log(`[YM-P2-ENS] wall clock ${wallClockSeconds.toFixed(2)} s`);
}

main().catch((err) => {
  console.error("[YM-P2-ENS] runner threw:");
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
