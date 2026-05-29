#!/usr/bin/env node
// scripts/yang-mills-phase1-su2-gauge-smoke.mjs
//
// Yang-Mills Phase 1 - SU(2) 2D gauge-invariance smoke runner.
// This runner is scoped to the SU2_2D harness cell only.

import * as fs from "node:fs";
import * as path from "node:path";
import * as crypto from "node:crypto";
import { execSync } from "node:child_process";
import { performance } from "node:perf_hooks";

import {
  createSU2Lattice,
  cloneSU2Lattice,
  combinedSweep,
  meanPlaquette,
  computeSignatureV1,
  computeHeldoutV1,
  computeRawMatrixVector,
  applySU2GaugeTransform,
  randomGaugeQuaternions,
  identityGaugeQuaternions,
  signatureMaxAbsResidual,
  rawMatrixNormalizedL2,
  estimateTauIntSokal,
  mulberry32,
  deriveSubstreamSeed,
  maxLinkUnitarityResidual,
} from "./lib/yang-mills-su2-2d-core.mjs";

const LOCKED = Object.freeze({
  cell: "SU2_2D",
  latticeSize: "16x16",
  beta: 2.0,
  boundary: "periodic",
  action: "Wilson",
  generator: "su2_heatbath_overrelax_v1",
  overrelaxPerHeatbath: 4,
  seed: 202605290102,
  burnIn: 2000,
  pilotSweeps: 512,
  thinning: 32,
  measurements: 32,
  gaugeTransforms: 8,
  signatureVocab: "v1",
  heldoutVocab: "v1",
  outRequired:
    "results/yang-mills/phase1/SU2_2D/2026-05-29_su2_2d_gauge_invariance_smoke_v0",
});

const THRESH = Object.freeze({
  wallClockSeconds: 10 * 60,
  tauIntMax: 16.0,
  thinningTauRatioMin: 2.0,
  identityResidualMax: 1e-12,
  randomGaugeResidualMax: 1e-12,
  rawMedianL2Min: 1e-2,
  rawPerTransformFloor: 1e-6,
  rawPerTransformFracAbove: 0.95,
  heatbathFallbackFractionMax: 0.001,
  unitarityResidualMax: 1e-10,
});

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next === undefined || next.startsWith("--")) {
        out[key] = true;
      } else {
        out[key] = next;
        i++;
      }
    }
  }
  return out;
}

function asNumber(v, name) {
  const n = Number(v);
  if (!Number.isFinite(n)) {
    throw new Error(`flag --${name} must be a finite number, got: ${v}`);
  }
  return n;
}

function asInt(v, name) {
  const n = Number(v);
  if (!Number.isInteger(n)) {
    throw new Error(`flag --${name} must be an integer, got: ${v}`);
  }
  return n;
}

function validateLockedArgs(args) {
  const failures = [];
  const check = (key, expected, actual) => {
    if (actual !== expected) failures.push(`--${key} expected ${expected}, got ${actual}`);
  };
  check("cell", LOCKED.cell, args.cell);
  check("lattice-size", LOCKED.latticeSize, args["lattice-size"]);
  check("beta", LOCKED.beta, asNumber(args.beta, "beta"));
  check("boundary", LOCKED.boundary, args.boundary);
  check("action", LOCKED.action, args.action);
  check("generator", LOCKED.generator, args.generator);
  check(
    "overrelax-per-heatbath",
    LOCKED.overrelaxPerHeatbath,
    asInt(args["overrelax-per-heatbath"], "overrelax-per-heatbath"),
  );
  check("seed", LOCKED.seed, asInt(args.seed, "seed"));
  check("burn-in", LOCKED.burnIn, asInt(args["burn-in"], "burn-in"));
  check("pilot-sweeps", LOCKED.pilotSweeps, asInt(args["pilot-sweeps"], "pilot-sweeps"));
  check("thinning", LOCKED.thinning, asInt(args.thinning, "thinning"));
  check("measurements", LOCKED.measurements, asInt(args.measurements, "measurements"));
  check(
    "gauge-transforms",
    LOCKED.gaugeTransforms,
    asInt(args["gauge-transforms"], "gauge-transforms"),
  );
  check("signature-vocab", LOCKED.signatureVocab, args["signature-vocab"]);
  check("heldout-vocab", LOCKED.heldoutVocab, args["heldout-vocab"]);
  const normOut = (args.out || "").replace(/\\/g, "/");
  check("out", LOCKED.outRequired, normOut);
  return failures;
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

function sha256OfFile(p) {
  return crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex");
}

function listEmittedFilesRecursive(rootDir) {
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

function getGitInfo() {
  try {
    const commit = execSync("git rev-parse HEAD", {
      stdio: ["ignore", "pipe", "ignore"],
    }).toString().trim();
    const status = execSync("git status --porcelain", {
      stdio: ["ignore", "pipe", "ignore"],
    }).toString().trim();
    return { codeCommit: commit || "unknown", gitDirty: status.length > 0 };
  } catch {
    return { codeCommit: "unknown", gitDirty: null };
  }
}

function reconstructCommandLine() {
  return [process.argv[0], process.argv[1], ...process.argv.slice(2)].join(" ");
}

function finalizeHashes(outDir) {
  const hashes = {};
  for (const p of listEmittedFilesRecursive(outDir)) {
    const rel = path.relative(outDir, p).replace(/\\/g, "/");
    if (rel === "hashes.json") continue;
    hashes[rel] = sha256OfFile(p);
  }
  writeJSON(path.join(outDir, "hashes.json"), hashes);
}

function median(values) {
  const sorted = values.slice().sort((a, b) => a - b);
  if (sorted.length % 2 === 1) return sorted[Math.floor(sorted.length / 2)];
  return 0.5 * (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]);
}

function makeStats() {
  return {
    heatbathLinkUpdates: 0,
    heatbathFallbackCount: 0,
    heatbathRejectionAttempts: 0,
  };
}

function fallbackFraction(stats) {
  if (!stats.heatbathLinkUpdates) return 0;
  return stats.heatbathFallbackCount / stats.heatbathLinkUpdates;
}

function buildManifest({ tauResult, wallClockSeconds, codeInfo, stats, maxUnitResidual }) {
  return {
    phase: "phase1",
    cell: LOCKED.cell,
    latticeSize: [16, 16],
    beta: LOCKED.beta,
    boundary: LOCKED.boundary,
    action: LOCKED.action,
    generator: LOCKED.generator,
    updateMix: { heatbath: 1, overrelaxation: LOCKED.overrelaxPerHeatbath },
    masterSeed: LOCKED.seed,
    burnInSweeps: LOCKED.burnIn,
    pilotTauIntPlaquette: tauResult ? tauResult.tauInt : null,
    registeredThinning: LOCKED.thinning,
    retainedConfigurations: LOCKED.measurements,
    signatureVocabularyVersion: LOCKED.signatureVocab,
    heldOutTargetVocabularyVersion: LOCKED.heldoutVocab,
    gammaHeldBinEdgeStatus: "phase1_no_rank_scoring",
    heatbathFallbackCount: stats.heatbathFallbackCount,
    heatbathFallbackFraction: fallbackFraction(stats),
    heatbathLinkUpdates: stats.heatbathLinkUpdates,
    heatbathRejectionAttempts: stats.heatbathRejectionAttempts,
    maxLinkUnitarityResidual: maxUnitResidual,
    controlSetDeclared: [
      "CTRL_META",
      "CTRL_RAW",
      "CTRL_RAND",
      "CTRL_RAND_STRAT",
      "CTRL_PERM",
      "CTRL_GAUGE_RAND",
      "CTRL_FINITE_SIZE",
    ],
    controlSetScored: ["CTRL_RAW", "CTRL_GAUGE_RAND"],
    codeCommit: codeInfo.codeCommit,
    gitDirty: codeInfo.gitDirty,
    commandLine: reconstructCommandLine(),
    wallClockSeconds,
  };
}

async function main() {
  const start = performance.now();
  const args = parseArgs(process.argv.slice(2));
  const validationFailures = validateLockedArgs(args);
  if (validationFailures.length > 0) {
    console.error("[YM-P1-SU2] manifest validation FAILED:");
    for (const f of validationFailures) console.error(`  - ${f}`);
    process.exit(2);
  }

  const outDir = path.resolve(args.out);
  for (const sub of ["autocorr_pilot", "configs", "signatures", "heldout", "gauge_randomization"]) {
    ensureDir(path.join(outDir, sub));
  }

  const L = 16;
  const state = createSU2Lattice(L, deriveSubstreamSeed(LOCKED.seed, "init"));
  const rng = mulberry32(deriveSubstreamSeed(LOCKED.seed, "sweep"));
  const stats = makeStats();

  console.log(`[YM-P1-SU2] start: cell=${LOCKED.cell} L=${L} beta=${LOCKED.beta} seed=${LOCKED.seed}`);
  console.log(`[YM-P1-SU2] burn-in: ${LOCKED.burnIn} combined sweeps`);
  for (let s = 0; s < LOCKED.burnIn; s++) {
    combinedSweep(state, LOCKED.beta, LOCKED.overrelaxPerHeatbath, rng, stats);
  }
  console.log(
    `[YM-P1-SU2] burn-in done; fallback fraction ${fallbackFraction(stats).toExponential(3)}`,
  );

  console.log(`[YM-P1-SU2] pilot: ${LOCKED.pilotSweeps} combined sweeps`);
  const pilotSeries = new Float64Array(LOCKED.pilotSweeps);
  for (let s = 0; s < LOCKED.pilotSweeps; s++) {
    combinedSweep(state, LOCKED.beta, LOCKED.overrelaxPerHeatbath, rng, stats);
    pilotSeries[s] = meanPlaquette(state);
  }
  const tauResult = estimateTauIntSokal(pilotSeries, 6);
  console.log(
    `[YM-P1-SU2] pilot done; mean plaq ${tauResult.mean.toFixed(6)}, tau_int ${tauResult.tauInt.toFixed(3)}, window ${tauResult.window}`,
  );

  writeCSV(
    path.join(outDir, "autocorr_pilot", "plaquette_series.csv"),
    ["sweep_index", "mean_plaquette"],
    Array.from(pilotSeries, (v, i) => [i, v]),
  );

  const elapsedSoFar = (performance.now() - start) / 1000;
  const sweepsSoFar = LOCKED.burnIn + LOCKED.pilotSweeps;
  const sweepsRemaining = LOCKED.thinning * LOCKED.measurements;
  const measurementPasses = LOCKED.measurements * (1 + LOCKED.gaugeTransforms);
  const predictedTotal =
    elapsedSoFar +
    (elapsedSoFar / sweepsSoFar) * sweepsRemaining +
    measurementPasses * 0.15;
  if (predictedTotal > THRESH.wallClockSeconds) {
    const wallClockSeconds = (performance.now() - start) / 1000;
    writeJSON(path.join(outDir, "summary.json"), {
      branch: "Z void_run",
      reason: "pilot timing extrapolation exceeds 10-minute compute cap",
      predictedTotalSeconds: predictedTotal,
      capSeconds: THRESH.wallClockSeconds,
    });
    writeJSON(
      path.join(outDir, "manifest.json"),
      buildManifest({
        tauResult,
        wallClockSeconds,
        codeInfo: getGitInfo(),
        stats,
        maxUnitResidual: maxLinkUnitarityResidual(state),
      }),
    );
    finalizeHashes(outDir);
    process.exit(3);
  }

  console.log(
    `[YM-P1-SU2] measurements: ${LOCKED.measurements} configs at thinning ${LOCKED.thinning}`,
  );
  const configs = [];
  const signatureRows = [];
  const heldoutRows = [];
  for (let m = 0; m < LOCKED.measurements; m++) {
    for (let s = 0; s < LOCKED.thinning; s++) {
      combinedSweep(state, LOCKED.beta, LOCKED.overrelaxPerHeatbath, rng, stats);
    }
    const cloned = cloneSU2Lattice(state);
    configs.push(cloned);
    const sig = computeSignatureV1(cloned);
    const hel = computeHeldoutV1(cloned);
    signatureRows.push([
      m,
      sig.W11_mean,
      sig.W11_var,
      sig.W12_mean,
      sig.W12_var,
      sig.W13_mean,
      sig.W13_var,
      sig.W22_mean,
      sig.W22_var,
    ]);
    heldoutRows.push([
      m,
      hel.W14_mean,
      hel.W14_var,
      hel.W23_mean,
      hel.W23_var,
      hel.W33_mean,
      hel.W33_var,
    ]);
  }

  writeText(
    path.join(outDir, "configs", "su2_links.jsonl"),
    configs
      .map((cfg, configIdx) =>
        JSON.stringify({ configIdx, shape: [2, L, L, 4], quaternionLinks: Array.from(cfg.links) }),
      )
      .join("\n") + "\n",
  );
  writeCSV(
    path.join(outDir, "signatures", "signature_vectors.csv"),
    [
      "config_idx",
      "W11_mean",
      "W11_var",
      "W12_mean",
      "W12_var",
      "W13_mean",
      "W13_var",
      "W22_mean",
      "W22_var",
    ],
    signatureRows,
  );
  writeCSV(
    path.join(outDir, "heldout", "heldout_loop_values.csv"),
    ["config_idx", "W14_mean", "W14_var", "W23_mean", "W23_var", "W33_mean", "W33_var"],
    heldoutRows,
  );
  writeJSON(path.join(outDir, "heldout", "gamma_bin_edges.json"), {
    status: "phase1_no_rank_scoring",
    note:
      "Phase 1 manifest does not score held-out targets; bin edges must be frozen by a Phase 2 manifest before rank-locality scoring.",
  });

  console.log(
    `[YM-P1-SU2] gauge randomization: identity + ${LOCKED.gaugeTransforms} random transforms per config`,
  );
  const sigResRows = [];
  const rawResRows = [];
  let maxGaugeUnitResidual = 0;
  for (let m = 0; m < configs.length; m++) {
    const base = configs[m];
    const baseSig = computeSignatureV1(base);
    const baseRaw = computeRawMatrixVector(base);
    {
      const transformed = applySU2GaugeTransform(base, identityGaugeQuaternions(L));
      maxGaugeUnitResidual = Math.max(maxGaugeUnitResidual, maxLinkUnitarityResidual(transformed));
      sigResRows.push([m, "identity", signatureMaxAbsResidual(baseSig, computeSignatureV1(transformed))]);
      rawResRows.push([m, "identity", rawMatrixNormalizedL2(baseRaw, computeRawMatrixVector(transformed))]);
    }
    for (let t = 0; t < LOCKED.gaugeTransforms; t++) {
      const gaugeRng = mulberry32(deriveSubstreamSeed(LOCKED.seed, "gauge", m, t));
      const transformed = applySU2GaugeTransform(base, randomGaugeQuaternions(L, gaugeRng));
      maxGaugeUnitResidual = Math.max(maxGaugeUnitResidual, maxLinkUnitarityResidual(transformed));
      sigResRows.push([m, `random_${t}`, signatureMaxAbsResidual(baseSig, computeSignatureV1(transformed))]);
      rawResRows.push([m, `random_${t}`, rawMatrixNormalizedL2(baseRaw, computeRawMatrixVector(transformed))]);
    }
  }

  writeCSV(
    path.join(outDir, "gauge_randomization", "signature_residuals.csv"),
    ["config_idx", "transform_id", "max_abs_signature_residual"],
    sigResRows,
  );
  writeCSV(
    path.join(outDir, "gauge_randomization", "raw_link_residuals.csv"),
    ["config_idx", "transform_id", "normalized_l2_raw_residual"],
    rawResRows,
  );

  const identityMax = Math.max(...sigResRows.filter((r) => r[1] === "identity").map((r) => r[2]));
  const randomSigMax = Math.max(...sigResRows.filter((r) => r[1] !== "identity").map((r) => r[2]));
  const randomRawResiduals = rawResRows.filter((r) => r[1] !== "identity").map((r) => r[2]);
  const rawMedian = median(randomRawResiduals);
  const rawFracAbove =
    randomRawResiduals.filter((v) => v > THRESH.rawPerTransformFloor).length /
    randomRawResiduals.length;
  const thinRatio = LOCKED.thinning / Math.max(tauResult.tauInt, 1e-12);
  const heatbathFallbackFrac = fallbackFraction(stats);
  const maxUnitResidual = Math.max(maxLinkUnitarityResidual(state), maxGaugeUnitResidual);

  const identityOk = identityMax <= THRESH.identityResidualMax;
  const randomSigOk = randomSigMax <= THRESH.randomGaugeResidualMax;
  const tauOk = tauResult.tauInt <= THRESH.tauIntMax;
  const thinOk = thinRatio >= THRESH.thinningTauRatioMin;
  const fallbackOk = heatbathFallbackFrac <= THRESH.heatbathFallbackFractionMax;
  const rawOk =
    rawMedian >= THRESH.rawMedianL2Min &&
    rawFracAbove >= THRESH.rawPerTransformFracAbove;
  const unitOk = maxUnitResidual <= THRESH.unitarityResidualMax;

  let branch;
  let reason;
  if (!unitOk) {
    branch = "Z void_run";
    reason = `SU(2) unitarity residual ${maxUnitResidual} exceeds ${THRESH.unitarityResidualMax}`;
  } else if (!identityOk || !randomSigOk) {
    branch = "YM-P1-NEG-A gauge_leakage";
    reason = `identity_max=${identityMax}, random_sig_max=${randomSigMax}`;
  } else if (!tauOk || !thinOk) {
    branch = "YM-P1-NEG-X autocorrelation_underflow";
    reason = `tau_int=${tauResult.tauInt}, thinning=${LOCKED.thinning}, thin/tau=${thinRatio.toFixed(3)}`;
  } else if (!fallbackOk) {
    branch = "YM-P1-QUAR-B heatbath_pathology";
    reason = `heatbath fallback fraction ${heatbathFallbackFrac} exceeds ${THRESH.heatbathFallbackFractionMax}`;
  } else if (!rawOk) {
    branch = "YM-P1-QUAR-A suspicious_raw_invariance";
    reason = `raw_median_L2=${rawMedian}, frac_above_${THRESH.rawPerTransformFloor}=${rawFracAbove.toFixed(4)}`;
  } else {
    branch = "P1-A smoke_pass";
    reason =
      "all burn-in / autocorrelation / signature-invariance / heatbath-fallback / raw-non-invariance thresholds pass";
  }

  const wallClockSeconds = (performance.now() - start) / 1000;
  const summary = {
    branch,
    reason,
    pilot: {
      tauInt: tauResult.tauInt,
      window: tauResult.window,
      meanPlaquette: tauResult.mean,
      variancePlaquette: tauResult.variance,
      thinningTauRatio: thinRatio,
    },
    measurements: { retained: configs.length },
    heatbath: {
      linkUpdates: stats.heatbathLinkUpdates,
      fallbackCount: stats.heatbathFallbackCount,
      fallbackFraction: heatbathFallbackFrac,
      rejectionAttempts: stats.heatbathRejectionAttempts,
      thresholdFallbackFraction: THRESH.heatbathFallbackFractionMax,
    },
    gaugeRandomization: {
      identityResidualMax: identityMax,
      randomSignatureResidualMax: randomSigMax,
      rawMatrixMedianNormalizedL2: rawMedian,
      rawMatrixFractionAboveFloor: rawFracAbove,
      maxLinkUnitarityResidual: maxUnitResidual,
      thresholds: {
        identityResidualMax: THRESH.identityResidualMax,
        randomGaugeResidualMax: THRESH.randomGaugeResidualMax,
        rawMedianL2Min: THRESH.rawMedianL2Min,
        rawPerTransformFloor: THRESH.rawPerTransformFloor,
        rawPerTransformFracAbove: THRESH.rawPerTransformFracAbove,
        unitarityResidualMax: THRESH.unitarityResidualMax,
      },
    },
    wallClockSeconds,
  };
  writeJSON(path.join(outDir, "summary.json"), summary);
  writeJSON(
    path.join(outDir, "manifest.json"),
    buildManifest({
      tauResult,
      wallClockSeconds,
      codeInfo: getGitInfo(),
      stats,
      maxUnitResidual,
    }),
  );
  finalizeHashes(outDir);

  console.log(`[YM-P1-SU2] verdict: ${branch}`);
  console.log(`[YM-P1-SU2] ${reason}`);
  console.log(`[YM-P1-SU2] wall clock ${wallClockSeconds.toFixed(2)} s`);
}

main().catch((err) => {
  console.error("[YM-P1-SU2] runner threw:");
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
