#!/usr/bin/env node
// scripts/yang-mills-phase1-gauge-smoke.mjs
//
// Yang-Mills Phase 1 - U(1) 2D gauge-invariance smoke runner.
//
// Pre-registered by docs/prereg/yang-mills/PHASE1_U1_2D_gauge_invariance_smoke.md
// under P0 lock docs/prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md.
//
// This runner does ONLY the Phase 1 smoke for the cheapest Abelian
// instrumentation cell (U1_2D, 16x16, beta 1.0). It interprets only
// CTRL_GAUGE_RAND and CTRL_RAW per the manifest. It does not do
// nearest-neighbor scoring, smearing, blocking, topological proxies,
// SU(2), or 4D — those are explicitly out of scope at this manifest.
//
// Branch outputs (per manifest §"Pass / Quarantine Thresholds"):
//   - P1-A smoke_pass
//   - YM-P1-NEG-A gauge_leakage
//   - YM-P1-QUAR-A suspicious_raw_invariance
//   - YM-P1-NEG-X autocorrelation_underflow
//   - Z void_run

import * as fs from "node:fs";
import * as path from "node:path";
import * as crypto from "node:crypto";
import { execSync } from "node:child_process";
import { performance } from "node:perf_hooks";

import {
  createU1Lattice,
  cloneU1Lattice,
  metropolisSweep,
  meanPlaquette,
  computeSignatureV1,
  computeHeldoutV1,
  computeRawLinkVector,
  applyU1GaugeTransform,
  randomAlphas,
  identityAlphas,
  signatureMaxAbsResidual,
  rawLinkNormalizedL2,
  estimateTauIntSokal,
  mulberry32,
  deriveSubstreamSeed,
} from "./lib/yang-mills-u1-2d-core.mjs";

// ----------------------------------------------------- locked values ---

const LOCKED = Object.freeze({
  cell: "U1_2D",
  latticeSize: "16x16",
  beta: 1.0,
  boundary: "periodic",
  action: "Wilson",
  generator: "u1_staple_metropolis_v1",
  proposalHalfWidth: 0.75,
  seed: 202605290101,
  burnIn: 2000,
  pilotSweeps: 512,
  thinning: 32,
  measurements: 32,
  gaugeTransforms: 8,
  signatureVocab: "v1",
  heldoutVocab: "v1",
  outRequired:
    "results/yang-mills/phase1/U1_2D/2026-05-29_u1_2d_gauge_invariance_smoke_v0",
});

const THRESH = Object.freeze({
  burnInSweeps: 2000,
  wallClockSeconds: 10 * 60,
  tauIntMax: 16.0,
  thinningTauRatioMin: 2.0,
  identityResidualMax: 1e-12,
  randomGaugeResidualMax: 1e-12,
  rawMedianL2Min: 1e-2,
  rawPerTransformFloor: 1e-6,
  rawPerTransformFracAbove: 0.95,
});

// ----------------------------------------------------- CLI parsing -----

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
    if (actual !== expected) {
      failures.push(`--${key} expected ${expected}, got ${actual}`);
    }
  };
  check("cell", LOCKED.cell, args.cell);
  check("lattice-size", LOCKED.latticeSize, args["lattice-size"]);
  check("beta", LOCKED.beta, asNumber(args.beta, "beta"));
  check("boundary", LOCKED.boundary, args.boundary);
  check("action", LOCKED.action, args.action);
  check("generator", LOCKED.generator, args.generator);
  check(
    "proposal-half-width",
    LOCKED.proposalHalfWidth,
    asNumber(args["proposal-half-width"], "proposal-half-width"),
  );
  check("seed", LOCKED.seed, asInt(args.seed, "seed"));
  check("burn-in", LOCKED.burnIn, asInt(args["burn-in"], "burn-in"));
  check(
    "pilot-sweeps",
    LOCKED.pilotSweeps,
    asInt(args["pilot-sweeps"], "pilot-sweeps"),
  );
  check("thinning", LOCKED.thinning, asInt(args.thinning, "thinning"));
  check(
    "measurements",
    LOCKED.measurements,
    asInt(args.measurements, "measurements"),
  );
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

// ----------------------------------------------------- file helpers ----

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

function csvEscape(v) {
  const s = typeof v === "number" ? formatNumber(v) : String(v);
  if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function formatNumber(v) {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return "0";
  if (Math.abs(v) < 1e-6 || Math.abs(v) >= 1e16) {
    return v.toExponential(12);
  }
  return v.toFixed(12).replace(/0+$/, "").replace(/\.$/, "");
}

function writeCSV(p, header, rows) {
  const lines = [header.join(",")];
  for (const row of rows) {
    lines.push(row.map(csvEscape).join(","));
  }
  writeText(p, lines.join("\n") + "\n");
}

function sha256OfFile(p) {
  const buf = fs.readFileSync(p);
  return crypto.createHash("sha256").update(buf).digest("hex");
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
    })
      .toString()
      .trim();
    let dirty = false;
    try {
      const status = execSync("git status --porcelain", {
        stdio: ["ignore", "pipe", "ignore"],
      })
        .toString()
        .trim();
      dirty = status.length > 0;
    } catch {
      dirty = null;
    }
    return { codeCommit: commit || "unknown", gitDirty: dirty };
  } catch {
    return { codeCommit: "unknown", gitDirty: null };
  }
}

function reconstructCommandLine() {
  const exe = process.argv[0];
  const script = process.argv[1];
  const rest = process.argv.slice(2);
  return [exe, script, ...rest].join(" ");
}

// ----------------------------------------------------- main -----------

async function main() {
  const start = performance.now();
  const args = parseArgs(process.argv.slice(2));

  const validationFailures = validateLockedArgs(args);
  if (validationFailures.length > 0) {
    console.error("[YM-P1] manifest validation FAILED:");
    for (const f of validationFailures) console.error(`  - ${f}`);
    console.error(
      "[YM-P1] runner will not execute; this would be a Z void_run.",
    );
    process.exit(2);
  }

  const outDir = path.resolve(args.out);
  ensureDir(outDir);
  for (const sub of [
    "autocorr_pilot",
    "configs",
    "signatures",
    "heldout",
    "gauge_randomization",
  ]) {
    ensureDir(path.join(outDir, sub));
  }

  const L = 16;
  const beta = LOCKED.beta;
  const halfWidth = LOCKED.proposalHalfWidth;
  const masterSeed = LOCKED.seed;

  const sweepRng = mulberry32(deriveSubstreamSeed(masterSeed, "sweep"));
  const initRng = mulberry32(deriveSubstreamSeed(masterSeed, "init"));

  console.log(
    `[YM-P1] start: cell=${LOCKED.cell} L=${L} beta=${beta} seed=${masterSeed}`,
  );

  // -------------------------------------------------- init + burn-in --

  const state = createU1Lattice(L, deriveSubstreamSeed(masterSeed, "init"));
  // Use sweepRng going forward.

  console.log(`[YM-P1] burn-in: ${LOCKED.burnIn} sweeps`);
  let burnAcceptSum = 0;
  for (let s = 0; s < LOCKED.burnIn; s++) {
    burnAcceptSum += metropolisSweep(state, beta, halfWidth, sweepRng);
  }
  const burnAcceptMean = burnAcceptSum / LOCKED.burnIn;
  console.log(
    `[YM-P1] burn-in done; mean acceptance ${burnAcceptMean.toFixed(4)}`,
  );

  // -------------------------------------------------- pilot ----------

  console.log(`[YM-P1] pilot: ${LOCKED.pilotSweeps} sweeps`);
  const pilotSeries = new Float64Array(LOCKED.pilotSweeps);
  let pilotAcceptSum = 0;
  for (let s = 0; s < LOCKED.pilotSweeps; s++) {
    pilotAcceptSum += metropolisSweep(state, beta, halfWidth, sweepRng);
    pilotSeries[s] = meanPlaquette(state);
  }
  const pilotAcceptMean = pilotAcceptSum / LOCKED.pilotSweeps;
  const tauResult = estimateTauIntSokal(pilotSeries, 6);
  console.log(
    `[YM-P1] pilot done; mean plaq ${tauResult.mean.toFixed(6)}, tau_int ${tauResult.tauInt.toFixed(3)}, window ${tauResult.window}`,
  );

  // Write pilot series CSV.
  {
    const rows = [];
    for (let i = 0; i < pilotSeries.length; i++) {
      rows.push([i, pilotSeries[i]]);
    }
    writeCSV(
      path.join(outDir, "autocorr_pilot", "plaquette_series.csv"),
      ["sweep_index", "mean_plaquette"],
      rows,
    );
  }

  // Wall-clock pilot extrapolation: estimate total runtime now and abort
  // if predicted total > 10 min.
  const elapsedSoFar = (performance.now() - start) / 1000;
  const sweepsSoFar = LOCKED.burnIn + LOCKED.pilotSweeps;
  const sweepsRemaining = LOCKED.thinning * LOCKED.measurements;
  const measurementsCount =
    LOCKED.measurements * (1 + LOCKED.gaugeTransforms);
  const perSweep = elapsedSoFar / sweepsSoFar;
  const predictedTotal =
    elapsedSoFar +
    perSweep * sweepsRemaining +
    measurementsCount * 0.05; // generous slack for signature/transform passes
  if (predictedTotal > THRESH.wallClockSeconds) {
    const summary = {
      branch: "Z void_run",
      reason: "pilot timing extrapolation exceeds 10-minute compute cap",
      predictedTotalSeconds: predictedTotal,
      capSeconds: THRESH.wallClockSeconds,
    };
    writeJSON(path.join(outDir, "summary.json"), summary);
    writeJSON(
      path.join(outDir, "manifest.json"),
      buildManifestSkeleton({
        tauResult,
        wallClockSeconds: elapsedSoFar,
        codeInfo: getGitInfo(),
      }),
    );
    finalizeHashes(outDir);
    console.error(`[YM-P1] VOID: predicted total ${predictedTotal.toFixed(1)} s > cap`);
    process.exit(3);
  }

  // -------------------------------------------------- measurements ---

  console.log(
    `[YM-P1] measurements: ${LOCKED.measurements} configs at thinning ${LOCKED.thinning}`,
  );
  const configs = []; // array of cloned states
  const signatureRows = [];
  const heldoutRows = [];

  for (let m = 0; m < LOCKED.measurements; m++) {
    for (let s = 0; s < LOCKED.thinning; s++) {
      metropolisSweep(state, beta, halfWidth, sweepRng);
    }
    const cloned = cloneU1Lattice(state);
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

  // Write configs as JSONL.
  {
    const lines = [];
    for (let i = 0; i < configs.length; i++) {
      lines.push(
        JSON.stringify({
          configIdx: i,
          shape: [2, L, L],
          links: Array.from(configs[i].links),
        }),
      );
    }
    writeText(path.join(outDir, "configs", "u1_links.jsonl"), lines.join("\n") + "\n");
  }

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
    [
      "config_idx",
      "W14_mean",
      "W14_var",
      "W23_mean",
      "W23_var",
      "W33_mean",
      "W33_var",
    ],
    heldoutRows,
  );

  writeJSON(path.join(outDir, "heldout", "gamma_bin_edges.json"), {
    status: "phase1_no_rank_scoring",
    note:
      "Phase 1 manifest does not score held-out targets; bin edges must be frozen by a Phase 2 manifest before any rank-locality scoring.",
  });

  // -------------------------------------------------- gauge randomization

  console.log(
    `[YM-P1] gauge randomization: identity + ${LOCKED.gaugeTransforms} random transforms per config`,
  );
  const sigResRows = [];
  const rawResRows = [];

  for (let m = 0; m < configs.length; m++) {
    const baseState = configs[m];
    const baseSig = computeSignatureV1(baseState);
    const baseRaw = computeRawLinkVector(baseState);

    // identity transform (index 0)
    {
      const alphas = identityAlphas(L);
      const transformed = applyU1GaugeTransform(baseState, alphas);
      const sig = computeSignatureV1(transformed);
      const raw = computeRawLinkVector(transformed);
      const sigRes = signatureMaxAbsResidual(baseSig, sig);
      const rawRes = rawLinkNormalizedL2(baseRaw, raw);
      sigResRows.push([m, "identity", sigRes]);
      rawResRows.push([m, "identity", rawRes]);
    }

    // 8 random transforms
    for (let t = 0; t < LOCKED.gaugeTransforms; t++) {
      const seed = deriveSubstreamSeed(masterSeed, "gauge", m, t);
      const rng = mulberry32(seed);
      const alphas = randomAlphas(L, rng);
      const transformed = applyU1GaugeTransform(baseState, alphas);
      const sig = computeSignatureV1(transformed);
      const raw = computeRawLinkVector(transformed);
      const sigRes = signatureMaxAbsResidual(baseSig, sig);
      const rawRes = rawLinkNormalizedL2(baseRaw, raw);
      sigResRows.push([m, `random_${t}`, sigRes]);
      rawResRows.push([m, `random_${t}`, rawRes]);
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

  // -------------------------------------------------- branch verdict --

  const identityResiduals = sigResRows
    .filter((r) => r[1] === "identity")
    .map((r) => r[2]);
  const randomSigResiduals = sigResRows
    .filter((r) => r[1] !== "identity")
    .map((r) => r[2]);
  const randomRawResiduals = rawResRows
    .filter((r) => r[1] !== "identity")
    .map((r) => r[2]);

  const identityMax = Math.max(...identityResiduals);
  const randomSigMax = Math.max(...randomSigResiduals);
  const randomRawSorted = randomRawResiduals.slice().sort((a, b) => a - b);
  const median =
    randomRawSorted.length % 2 === 1
      ? randomRawSorted[Math.floor(randomRawSorted.length / 2)]
      : 0.5 *
        (randomRawSorted[randomRawSorted.length / 2 - 1] +
          randomRawSorted[randomRawSorted.length / 2]);
  const fracAboveFloor =
    randomRawResiduals.filter((v) => v > THRESH.rawPerTransformFloor).length /
    randomRawResiduals.length;

  const tauOk = tauResult.tauInt <= THRESH.tauIntMax;
  const thinRatio = LOCKED.thinning / Math.max(tauResult.tauInt, 1e-12);
  const thinOk = thinRatio >= THRESH.thinningTauRatioMin;
  const identityOk = identityMax <= THRESH.identityResidualMax;
  const randomSigOk = randomSigMax <= THRESH.randomGaugeResidualMax;
  const rawMedianOk = median >= THRESH.rawMedianL2Min;
  const rawFracOk = fracAboveFloor >= THRESH.rawPerTransformFracAbove;

  let branch;
  let reason;
  if (!identityOk || !randomSigOk) {
    branch = "YM-P1-NEG-A gauge_leakage";
    reason = `identity_max=${identityMax}, random_sig_max=${randomSigMax}, tolerance=${THRESH.randomGaugeResidualMax}`;
  } else if (!tauOk || !thinOk) {
    branch = "YM-P1-NEG-X autocorrelation_underflow";
    reason = `tau_int=${tauResult.tauInt}, thinning=${LOCKED.thinning}, thin/tau=${thinRatio.toFixed(3)}`;
  } else if (!rawMedianOk || !rawFracOk) {
    branch = "YM-P1-QUAR-A suspicious_raw_invariance";
    reason = `raw_median_L2=${median}, frac_above_${THRESH.rawPerTransformFloor}=${fracAboveFloor.toFixed(4)}`;
  } else {
    branch = "P1-A smoke_pass";
    reason =
      "all burn-in / autocorrelation / signature-invariance / raw-non-invariance thresholds pass";
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
      meanAcceptance: pilotAcceptMean,
      thinningTauRatio: thinRatio,
    },
    measurements: {
      retained: configs.length,
    },
    gaugeRandomization: {
      identityResidualMax: identityMax,
      randomSignatureResidualMax: randomSigMax,
      rawLinkMedianNormalizedL2: median,
      rawLinkFractionAboveFloor: fracAboveFloor,
      thresholds: {
        identityResidualMax: THRESH.identityResidualMax,
        randomGaugeResidualMax: THRESH.randomGaugeResidualMax,
        rawMedianL2Min: THRESH.rawMedianL2Min,
        rawPerTransformFloor: THRESH.rawPerTransformFloor,
        rawPerTransformFracAbove: THRESH.rawPerTransformFracAbove,
      },
    },
    wallClockSeconds,
  };
  writeJSON(path.join(outDir, "summary.json"), summary);

  const codeInfo = getGitInfo();
  const manifestObj = buildManifestSkeleton({
    tauResult,
    wallClockSeconds,
    codeInfo,
  });
  writeJSON(path.join(outDir, "manifest.json"), manifestObj);

  finalizeHashes(outDir);

  console.log(`[YM-P1] verdict: ${branch}`);
  console.log(`[YM-P1] ${reason}`);
  console.log(`[YM-P1] wall clock ${wallClockSeconds.toFixed(2)} s`);
}

function buildManifestSkeleton({ tauResult, wallClockSeconds, codeInfo }) {
  return {
    phase: "phase1",
    cell: LOCKED.cell,
    latticeSize: [16, 16],
    beta: LOCKED.beta,
    boundary: LOCKED.boundary,
    action: LOCKED.action,
    generator: LOCKED.generator,
    updateMix: { metropolis: 1, overrelaxation: 0 },
    masterSeed: LOCKED.seed,
    burnInSweeps: LOCKED.burnIn,
    pilotTauIntPlaquette: tauResult ? tauResult.tauInt : null,
    registeredThinning: LOCKED.thinning,
    retainedConfigurations: LOCKED.measurements,
    signatureVocabularyVersion: LOCKED.signatureVocab,
    heldOutTargetVocabularyVersion: LOCKED.heldoutVocab,
    gammaHeldBinEdgeStatus: "phase1_no_rank_scoring",
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

function finalizeHashes(outDir) {
  const all = listEmittedFilesRecursive(outDir);
  const hashes = {};
  for (const p of all) {
    const rel = path.relative(outDir, p).replace(/\\/g, "/");
    if (rel === "hashes.json") continue;
    hashes[rel] = sha256OfFile(p);
  }
  writeJSON(path.join(outDir, "hashes.json"), hashes);
}

main().catch((err) => {
  console.error("[YM-P1] runner threw:");
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
