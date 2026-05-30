#!/usr/bin/env node
// scripts/yang-mills-phase2-v3-su2-3d-aggregate.mjs
//
// Yang-Mills Phase 2 v3 - SU(2) 3D relative-locality aggregation runner
// with unchanged v1 signatures and a W33 spatial-variance held-out target.

import * as fs from "node:fs";
import * as path from "node:path";
import * as crypto from "node:crypto";
import { execSync } from "node:child_process";
import { performance } from "node:perf_hooks";

import {
  computeSignatureV1,
  computeHeldoutV1,
  computeRawMatrixVector,
  applySU2GaugeTransform,
  randomGaugeQuaternions,
  signatureMaxAbsResidual,
  mulberry32,
  deriveSubstreamSeed,
} from "./lib/yang-mills-su2-3d-core.mjs";

const LOCKED = Object.freeze({
  cell: "SU2_3D",
  latticeSize: "12x12x12",
  betaSlate: "2.0,2.4,2.8",
  signatureVocab: "v1",
  signatureSource: "v0_reread",
  heldoutVocab: "v2",
  heldoutSummary: "spatial_variance_W33",
  heldoutVarianceEstimator: "biased",
  distanceMetric: "euclidean_zscore",
  kSlate: "3,5,10",
  primaryK: 5,
  bootstrapResamples: 1000,
  binConvention: "per_beta_tertile_linear",
  gaugeRandSeedTag: "phase2_v3_aggregation",
  gaugeTransformsPerConfig: 1,
  outRequired: "results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v3",
  inputs: {
    "2.0": "results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.0_ensemble_v0",
    "2.4": "results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.4_ensemble_v0",
    "2.8": "results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.8_ensemble_v0",
  },
});

const BETA_KEYS = Object.freeze(["2.0", "2.4", "2.8"]);
const K_SLATE = Object.freeze([3, 5, 10]);
const PRIMARY_K = 5;
const CHANCE = 1 / 3;
const PERMUTATION_CONTROL_RESAMPLES = 1000;
const MASTER_SEED = 202605290599;
const TARGET_SPREAD_MIN = 1e-12;
const GAUGE_RESIDUAL_TOL = 1e-12;
const AGGREGATION_WALL_CLOCK_MAX = 600;
const SIGNATURE_KEYS = Object.freeze([
  "W11_mean",
  "W11_var",
  "W12_mean",
  "W12_var",
  "W13_mean",
  "W13_var",
  "W22_mean",
  "W22_var",
]);

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

function normalizedPathForCompare(p) {
  return String(p || "").replace(/\\/g, "/");
}

function validateLockedArgs(args) {
  const failures = [];
  const check = (key, expected, actual) => {
    if (actual !== expected) failures.push(`--${key} expected ${expected}, got ${actual}`);
  };
  check("cell", LOCKED.cell, args.cell);
  check("lattice-size", LOCKED.latticeSize, args["lattice-size"]);
  check("beta-slate", LOCKED.betaSlate, args["beta-slate"]);
  check("signature-vocab", LOCKED.signatureVocab, args["signature-vocab"]);
  check("signature-source", LOCKED.signatureSource, args["signature-source"]);
  check("heldout-vocab", LOCKED.heldoutVocab, args["heldout-vocab"]);
  check("heldout-summary", LOCKED.heldoutSummary, args["heldout-summary"]);
  check("heldout-variance-estimator", LOCKED.heldoutVarianceEstimator, args["heldout-variance-estimator"]);
  check("distance-metric", LOCKED.distanceMetric, args["distance-metric"]);
  check("k-slate", LOCKED.kSlate, args["k-slate"]);
  check("primary-k", LOCKED.primaryK, Number(args["primary-k"]));
  check("bootstrap-resamples", LOCKED.bootstrapResamples, Number(args["bootstrap-resamples"]));
  check("bin-convention", LOCKED.binConvention, args["bin-convention"]);
  check("gauge-rand-seed-tag", LOCKED.gaugeRandSeedTag, args["gauge-rand-seed-tag"]);
  check("gauge-transforms-per-config", LOCKED.gaugeTransformsPerConfig, Number(args["gauge-transforms-per-config"]));
  check("out", LOCKED.outRequired, normalizedPathForCompare(args.out));
  for (const betaKey of BETA_KEYS) {
    check(`in-beta-${betaKey}`, LOCKED.inputs[betaKey], normalizedPathForCompare(args[`in-beta-${betaKey}`]));
  }
  return failures;
}

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function ensureCleanDir(p, requiredSuffix) {
  const resolved = path.resolve(p);
  const normalized = resolved.replace(/\\/g, "/");
  if (!normalized.endsWith(requiredSuffix)) throw new Error(`refusing to clean unexpected output dir: ${resolved}`);
  fs.rmSync(resolved, { recursive: true, force: true });
  fs.mkdirSync(resolved, { recursive: true });
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
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function writeCSV(p, header, rows) {
  const lines = [header.join(",")];
  for (const row of rows) lines.push(row.map(csvEscape).join(","));
  writeText(p, lines.join("\n") + "\n");
}

function parseCsv(p) {
  const text = fs.readFileSync(p, "utf8").trim();
  if (!text) return [];
  const lines = text.split(/\r?\n/);
  const header = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    const row = {};
    for (let i = 0; i < header.length; i++) row[header[i]] = cells[i];
    return row;
  });
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

function collectHashes(aggDir, excluded = new Set()) {
  const hashes = {};
  for (const p of listFiles(aggDir)) {
    const rel = path.relative(aggDir, p).replace(/\\/g, "/");
    if (!excluded.has(rel)) hashes[rel] = sha256OfFile(p);
  }
  return hashes;
}

function finalizeHashes(aggDir) {
  writeJSON(path.join(aggDir, "hashes.json"), collectHashes(aggDir, new Set(["hashes.json"])));
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

function percentileLinear(values, q) {
  const sorted = values.slice().sort((a, b) => a - b);
  if (sorted.length === 1) return sorted[0];
  const h = (sorted.length - 1) * q;
  const lo = Math.floor(h);
  const hi = Math.ceil(h);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo]);
}

function assignBin(v, lowEdge, highEdge) {
  if (v <= lowEdge) return 1;
  if (v <= highEdge) return 2;
  return 3;
}

function mean(values) {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function maxAbsVectorDiff(a, b) {
  let maxAbs = 0;
  for (let i = 0; i < a.length; i++) maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
  return maxAbs;
}

function normalizeFeatures(records, getter) {
  const first = getter(records[0]);
  const d = first.length;
  const sums = new Float64Array(d);
  const sumsSq = new Float64Array(d);
  const raw = records.map((record) => getter(record));
  for (const vec of raw) {
    for (let j = 0; j < d; j++) {
      sums[j] += vec[j];
      sumsSq[j] += vec[j] * vec[j];
    }
  }
  const means = new Float64Array(d);
  const stds = new Float64Array(d);
  for (let j = 0; j < d; j++) {
    means[j] = sums[j] / raw.length;
    const variance = Math.max(0, sumsSq[j] / raw.length - means[j] * means[j]);
    const std = Math.sqrt(variance);
    stds[j] = std > 1e-12 ? std : 1;
  }
  const normalized = raw.map((vec) => {
    const out = new Float64Array(d);
    for (let j = 0; j < d; j++) out[j] = (vec[j] - means[j]) / stds[j];
    return out;
  });
  return { normalized, means: Array.from(means), stds: Array.from(stds) };
}

function distanceSq(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

function buildNearestGraph(records, features, candidateFn, maxK) {
  return records.map((query, qi) => {
    const candidates = candidateFn(query, qi).map((ci) => ({
      index: ci,
      distanceSq: distanceSq(features[qi], features[ci]),
    }));
    candidates.sort((a, b) => a.distanceSq - b.distanceSq || records[a.index].globalIndex - records[b.index].globalIndex);
    return {
      query: query.globalIndex,
      queryId: query.id,
      neighbors: candidates.slice(0, maxK).map((n) => ({
        index: records[n.index].globalIndex,
        id: records[n.index].id,
        distanceSq: n.distanceSq,
      })),
    };
  });
}

function buildRandomGraph(records, candidateFn, maxK, seedLabel) {
  return records.map((query, qi) => {
    const rng = mulberry32(deriveSubstreamSeed(MASTER_SEED, seedLabel, query.betaKey, query.configIdx));
    const pool = candidateFn(query, qi).slice();
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    return {
      query: query.globalIndex,
      queryId: query.id,
      neighbors: pool.slice(0, maxK).map((idx) => ({
        index: records[idx].globalIndex,
        id: records[idx].id,
        distanceSq: null,
      })),
    };
  });
}

function makePermutation(records, seedLabel, betaKey = null, labelKind = "withinBin", resampleIndex = 0) {
  const selected = records.filter((r) => betaKey === null || r.betaKey === betaKey).map((r) => r.globalIndex);
  const labels = selected.map((idx) => records[idx][labelKind]);
  const rng = mulberry32(deriveSubstreamSeed(MASTER_SEED, seedLabel, betaKey || "global", resampleIndex));
  for (let i = labels.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [labels[i], labels[j]] = [labels[j], labels[i]];
  }
  const out = {};
  for (let i = 0; i < selected.length; i++) out[selected[i]] = labels[i];
  return out;
}

function makeWithinPermutation(records, resampleIndex) {
  const out = {};
  for (const betaKey of BETA_KEYS) Object.assign(out, makePermutation(records, "CTRL_PERM_within", betaKey, "withinBin", resampleIndex));
  return out;
}

function makeGlobalPermutation(records, resampleIndex) {
  return makePermutation(records, "CTRL_PERM_across", null, "globalBin", resampleIndex);
}

function scoreGraph(graph, labelOf, k) {
  const queryScores = graph.map((row) => {
    const qLabel = labelOf(row.query);
    let hits = 0;
    for (const n of row.neighbors.slice(0, k)) {
      if (labelOf(n.index) === qLabel) hits++;
    }
    return { query: row.query, score: hits / k };
  });
  return { meanBinPurity: mean(queryScores.map((r) => r.score)), queryScores };
}

function scorePermutationControl(graph, permutationMaps, k) {
  const sums = new Map(graph.map((row) => [row.query, 0]));
  for (const labels of permutationMaps) {
    for (const row of graph) {
      const qLabel = labels[row.query];
      let hits = 0;
      for (const n of row.neighbors.slice(0, k)) {
        if (labels[n.index] === qLabel) hits++;
      }
      sums.set(row.query, sums.get(row.query) + hits / k);
    }
  }
  const queryScores = graph.map((row) => ({ query: row.query, score: sums.get(row.query) / permutationMaps.length }));
  return { meanBinPurity: mean(queryScores.map((r) => r.score)), queryScores };
}

function bootstrapCi(records, queryScores, resamples, seedLabel) {
  const scoresByIndex = new Map(queryScores.map((r) => [r.query, r.score]));
  const byBeta = {};
  for (const betaKey of BETA_KEYS) byBeta[betaKey] = records.filter((r) => r.betaKey === betaKey).map((r) => r.globalIndex);
  const rng = mulberry32(deriveSubstreamSeed(MASTER_SEED, "bootstrap", seedLabel));
  const values = [];
  for (let b = 0; b < resamples; b++) {
    let sum = 0;
    let count = 0;
    for (const betaKey of BETA_KEYS) {
      const ids = byBeta[betaKey];
      for (let i = 0; i < ids.length; i++) {
        const sampled = ids[Math.floor(rng() * ids.length)];
        sum += scoresByIndex.get(sampled);
        count++;
      }
    }
    values.push(sum / count);
  }
  return { low: percentileLinear(values, 0.025), high: percentileLinear(values, 0.975) };
}

function kendallTau(x, y) {
  let concordant = 0;
  let discordant = 0;
  for (let i = 0; i < x.length; i++) {
    for (let j = i + 1; j < x.length; j++) {
      const dx = x[i] - x[j];
      const dy = y[i] - y[j];
      const prod = dx * dy;
      if (prod > 0) concordant++;
      else if (prod < 0) discordant++;
    }
  }
  const denom = concordant + discordant;
  return denom ? (concordant - discordant) / denom : 0;
}

function computeKendallRows(records, withinFeatures) {
  const rows = [];
  const perQueryAll = [];
  for (const betaKey of BETA_KEYS) {
    const betaRecords = records.filter((r) => r.betaKey === betaKey);
    const taus = [];
    for (const q of betaRecords) {
      const dists = [];
      const targetDiffs = [];
      for (const other of betaRecords) {
        if (other.globalIndex === q.globalIndex) continue;
        dists.push(Math.sqrt(distanceSq(withinFeatures[q.globalIndex], withinFeatures[other.globalIndex])));
        targetDiffs.push(Math.abs(other.sigma2W33 - q.sigma2W33));
      }
      const tau = kendallTau(dists, targetDiffs);
      taus.push(tau);
      perQueryAll.push(tau);
    }
    rows.push([betaKey, mean(taus), taus.length]);
  }
  rows.push(["all", mean(perQueryAll), perQueryAll.length]);
  return rows;
}

function graphForFile(graph) {
  return graph.map((row) => ({
    query: row.query,
    queryId: row.queryId,
    neighbors: row.neighbors.map((n) => ({ index: n.index, id: n.id, distanceSq: n.distanceSq })),
  }));
}

function signatureObjectFromVector(vec) {
  const out = {};
  for (let i = 0; i < SIGNATURE_KEYS.length; i++) out[SIGNATURE_KEYS[i]] = vec[i];
  return out;
}

function signatureVectorFromObject(sig) {
  return new Float64Array([
    sig.W11_mean,
    sig.W11_var,
    sig.W12_mean,
    sig.W12_var,
    sig.W13_mean,
    sig.W13_var,
    sig.W22_mean,
    sig.W22_var,
  ]);
}

function validateV0EnsembleHealth(betaKey, dir, summary, hashes) {
  const fail = (msg) => {
    throw new Error(`v0 ensemble ${betaKey} failed validation: ${msg}`);
  };
  if (summary.branch !== "P1-A ensemble_health_pass") fail(`branch ${summary.branch}`);
  if ((summary.measurements?.retained ?? 0) !== 32) fail(`retained configs ${summary.measurements?.retained}`);
  if ((summary.pilot?.tauInt ?? Number.POSITIVE_INFINITY) > 16) fail(`tau_int ${summary.pilot?.tauInt}`);
  if ((summary.pilot?.thinningTauRatio ?? 0) < 2) fail(`thinning/tau ${summary.pilot?.thinningTauRatio}`);
  if ((summary.heatbath?.fallbackFraction ?? Number.POSITIVE_INFINITY) > 0.001) fail(`fallback ${summary.heatbath?.fallbackFraction}`);
  if ((summary.linkUnitarityMaxFrobenius ?? Number.POSITIVE_INFINITY) > 1e-10) fail(`unitarity ${summary.linkUnitarityMaxFrobenius}`);
  if ((summary.orientation?.relativeSpread ?? Number.POSITIVE_INFINITY) > 5e-2) fail(`orientation ${summary.orientation?.relativeSpread}`);

  const configsPath = path.join(dir, "configs", "su2_links.jsonl");
  const signaturePath = path.join(dir, "signatures", "signature_vectors.csv");
  const configHash = sha256OfFile(configsPath);
  const signatureHash = sha256OfFile(signaturePath);
  if (hashes["configs/su2_links.jsonl"] !== configHash) {
    fail(`configs hash ${configHash} does not match hashes.json ${hashes["configs/su2_links.jsonl"]}`);
  }
  if (hashes["signatures/signature_vectors.csv"] !== signatureHash) {
    fail(`signature hash ${signatureHash} does not match hashes.json ${hashes["signatures/signature_vectors.csv"]}`);
  }
  return { configHash, signatureHash };
}

function readEnsemble(betaKey, dir, startIndex) {
  const required = [
    "manifest.json",
    "summary.json",
    "configs/su2_links.jsonl",
    "signatures/signature_vectors.csv",
    "hashes.json",
  ];
  for (const rel of required) {
    if (!fs.existsSync(path.join(dir, rel))) throw new Error(`missing required v0 file ${path.join(dir, rel)}`);
  }
  const summary = JSON.parse(fs.readFileSync(path.join(dir, "summary.json"), "utf8"));
  const manifest = JSON.parse(fs.readFileSync(path.join(dir, "manifest.json"), "utf8"));
  const hashes = JSON.parse(fs.readFileSync(path.join(dir, "hashes.json"), "utf8"));
  const { configHash, signatureHash } = validateV0EnsembleHealth(betaKey, dir, summary, hashes);
  const signatures = parseCsv(path.join(dir, "signatures", "signature_vectors.csv"));
  const configs = fs.readFileSync(path.join(dir, "configs", "su2_links.jsonl"), "utf8").trim().split(/\r?\n/);
  if (signatures.length !== 32 || configs.length !== 32) throw new Error(`v0 ensemble ${betaKey} expected 32 signatures/configs`);
  const records = signatures.map((row, localIdx) => {
    const configIdx = Number(row.config_idx);
    const parsedConfig = JSON.parse(configs[localIdx]);
    if (parsedConfig.configIdx !== configIdx) throw new Error(`config index drift for beta ${betaKey}: signature ${configIdx}, jsonl ${parsedConfig.configIdx}`);
    return {
      globalIndex: startIndex + localIdx,
      id: `beta${betaKey}_config${configIdx}`,
      betaKey,
      beta: Number(betaKey),
      configIdx,
      signature: new Float64Array(SIGNATURE_KEYS.map((k) => Number(row[k]))),
      config: { L: 12, links: new Float64Array(parsedConfig.quaternionLinks) },
      sourceDir: dir,
      manifest,
      summary,
    };
  });
  const source = {
    beta: Number(betaKey),
    betaKey,
    resultDir: dir.replace(/\\/g, "/"),
    manifestCodeCommit: manifest.codeCommit,
    manifestGitDirty: manifest.gitDirty,
    summaryBranch: summary.branch,
    configsSha256: configHash,
    hashesJsonConfigsSha256: hashes["configs/su2_links.jsonl"],
    signatureVectorsSha256: signatureHash,
    hashesJsonSignatureVectorsSha256: hashes["signatures/signature_vectors.csv"],
  };
  return { records, source };
}

function computeTargetEdges(records, frozenAtIso, convention, labelPrefix = "") {
  const valueKey = labelPrefix ? `${labelPrefix}Sigma2W33` : "sigma2W33";
  const binDegenerateBetas = [];
  const perBetaEdges = {
    frozenAt: frozenAtIso,
    convention,
    target: "sigma2_W33",
    betas: {},
  };
  for (const betaKey of BETA_KEYS) {
    const betaRecords = records.filter((r) => r.betaKey === betaKey);
    const values = betaRecords.map((r) => r[valueKey]);
    const spread = Math.max(...values) - Math.min(...values);
    if (spread < TARGET_SPREAD_MIN) binDegenerateBetas.push(betaKey);
    perBetaEdges.betas[betaKey] = {
      lowEdge: percentileLinear(values, 1 / 3),
      highEdge: percentileLinear(values, 2 / 3),
      spread,
    };
  }
  const allValues = records.map((r) => r[valueKey]);
  const globalEdges = {
    frozenAt: frozenAtIso,
    convention: "global_tertile_linear",
    target: "sigma2_W33",
    lowEdge: percentileLinear(allValues, 1 / 3),
    highEdge: percentileLinear(allValues, 2 / 3),
    spread: Math.max(...allValues) - Math.min(...allValues),
  };
  return { perBetaEdges, globalEdges, binDegenerateBetas };
}

function assignTargetBinsFromEdges(records, perBetaEdges, globalEdges, labelPrefix = "") {
  const valueKey = labelPrefix ? `${labelPrefix}Sigma2W33` : "sigma2W33";
  const withinKey = labelPrefix ? `${labelPrefix}WithinBin` : "withinBin";
  const globalKey = labelPrefix ? `${labelPrefix}GlobalBin` : "globalBin";
  for (const betaKey of BETA_KEYS) {
    const edges = perBetaEdges.betas[betaKey];
    const betaRecords = records.filter((r) => r.betaKey === betaKey);
    for (const r of betaRecords) r[withinKey] = assignBin(r[valueKey], edges.lowEdge, edges.highEdge);
    edges.assignments = betaRecords.map((r) => ({
      id: r.id,
      configIdx: r.configIdx,
      sigma2W33: r[valueKey],
      bin: r[withinKey],
    }));
  }
  for (const r of records) r[globalKey] = assignBin(r[valueKey], globalEdges.lowEdge, globalEdges.highEdge);
  globalEdges.assignments = records.map((r) => ({
    id: r.id,
    beta: r.beta,
    configIdx: r.configIdx,
    sigma2W33: r[valueKey],
    bin: r[globalKey],
  }));
}

function readScoreMap(scorePath) {
  if (!fs.existsSync(scorePath)) return new Map();
  const rows = parseCsv(scorePath);
  const map = new Map();
  for (const row of rows) map.set(`${row.lane}:${row.control_or_primary}:${row.k}`, Number(row.mean_bin_purity));
  return map;
}

function compareV0V3(scoreRows) {
  const v0ScorePath = "results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v0/aggregation/rank_locality_scores.csv";
  const v0 = readScoreMap(v0ScorePath);
  const rows = [];
  for (const row of scoreRows) {
    const [lane, id, k, v3Mean] = row;
    const key = `${lane}:${id}:${k}`;
    const v0Mean = v0.get(key);
    rows.push({
      lane,
      controlOrPrimary: id,
      k,
      v0GammaHeldMeanBinPurity: v0Mean ?? null,
      v3Sigma2W33MeanBinPurity: v3Mean,
      deltaV3MinusV0: v0Mean === undefined ? null : v3Mean - v0Mean,
    });
  }
  return { v0ScorePath, rows };
}

function assessBranch(inputs) {
  if (inputs.wallClockSeconds > AGGREGATION_WALL_CLOCK_MAX) return ["Z void_run", "aggregation wall clock exceeded 10-minute cap"];
  if (!inputs.v3BinEdgeTimestampOk) return ["Z void_run", "v3 bin-edge timestamp drift"];
  if (inputs.binDegenerateBetas.length > 0) return ["Z bin_degenerate", `degenerate sigma2_W33 distribution for beta ${inputs.binDegenerateBetas.join(",")}`];
  if (Math.abs(inputs.withinCtrlPerm5 - CHANCE) > 0.05) return ["Z graph_contamination", `CTRL_PERM purity ${inputs.withinCtrlPerm5} differs from chance ${CHANCE}`];
  if (
    inputs.gaugeRandPurityDiff > GAUGE_RESIDUAL_TOL ||
    inputs.gaugeSignatureResidualMax > GAUGE_RESIDUAL_TOL ||
    inputs.gaugeTargetResidualMax > GAUGE_RESIDUAL_TOL
  ) {
    return ["YM-P1-NEG-A gauge_leakage", `gauge purity diff ${inputs.gaugeRandPurityDiff}, signature residual ${inputs.gaugeSignatureResidualMax}, target residual ${inputs.gaugeTargetResidualMax}`];
  }
  if (inputs.withinPrimary5 < 0.5 || inputs.withinPrimaryMinusRand5 < 0.10) {
    return ["YM-P2-NEG-A no_rank_local_structure", `primary=${inputs.withinPrimary5}, primary-CTRL_RAND=${inputs.withinPrimaryMinusRand5}`];
  }
  if (inputs.withinPrimaryMinusMeta5 < 0.10) return ["YM-P2-NEG-B metadata_only", `primary-CTRL_META=${inputs.withinPrimaryMinusMeta5}`];
  if (inputs.withinPrimaryMinusRaw5 < 0.10) return ["YM-P2-NEG-D raw_dominates", `primary-CTRL_RAW=${inputs.withinPrimaryMinusRaw5}`];
  if (inputs.acrossPrimaryMinusRandStrat5 < 0.05) return ["YM-P2-NEG-C coupling_triviality", `across primary-CTRL_RAND_STRAT=${inputs.acrossPrimaryMinusRandStrat5}`];
  return ["P2-A bounded_positive", "every Phase 2 v3 point-estimate gate passes"];
}

async function main() {
  const start = performance.now();
  const startWallMs = Date.now();
  const startIso = new Date(startWallMs).toISOString();
  const args = parseArgs(process.argv.slice(2));
  const failures = validateLockedArgs(args);
  if (failures.length > 0) {
    console.error("[YM-P2-V3-AGG] manifest validation FAILED:");
    for (const f of failures) console.error(`  - ${f}`);
    process.exit(2);
  }

  const outDir = path.resolve(args.out);
  const aggDir = path.join(outDir, "aggregation");
  ensureCleanDir(aggDir, "results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v3/aggregation");
  ensureDir(path.join(aggDir, "control_nn_graphs"));

  let records = [];
  const v0EnsembleSources = [];
  for (const betaKey of BETA_KEYS) {
    const read = readEnsemble(betaKey, path.resolve(args[`in-beta-${betaKey}`]), records.length);
    records = records.concat(read.records);
    v0EnsembleSources.push(read.source);
  }
  if (records.length !== 96) throw new Error(`expected 96 records, got ${records.length}`);
  writeJSON(path.join(aggDir, "v0_ensemble_sources.json"), v0EnsembleSources);

  console.log("[YM-P2-V3-AGG] computing sigma2_W33 targets for 96 configs");
  const targetRows = [];
  for (const r of records) {
    const held = computeHeldoutV1(r.config);
    r.muW33 = held.W33_mean;
    r.sigma2W33 = held.W33_var;
    targetRows.push([r.betaKey, r.configIdx, r.muW33, r.sigma2W33, 5184]);
  }
  writeCSV(path.join(aggDir, "v3_target_summary.csv"), ["beta", "config_idx", "mu_W33", "sigma2_W33", "sample_count"], targetRows);

  const { perBetaEdges, globalEdges, binDegenerateBetas } = computeTargetEdges(records, startIso, LOCKED.binConvention);
  const perBetaEdgesPath = path.join(aggDir, "per_beta_v3_bin_edges.json");
  const globalEdgesPath = path.join(aggDir, "global_v3_bin_edges.json");
  writeJSON(perBetaEdgesPath, perBetaEdges);
  writeJSON(globalEdgesPath, globalEdges);
  const writtenEdgeMtimeMs = Math.max(fs.statSync(perBetaEdgesPath).mtimeMs, fs.statSync(globalEdgesPath).mtimeMs);

  const rereadPerBetaEdges = JSON.parse(fs.readFileSync(perBetaEdgesPath, "utf8"));
  const rereadGlobalEdges = JSON.parse(fs.readFileSync(globalEdgesPath, "utf8"));
  assignTargetBinsFromEdges(records, rereadPerBetaEdges, rereadGlobalEdges);
  writeJSON(perBetaEdgesPath, rereadPerBetaEdges);
  writeJSON(globalEdgesPath, rereadGlobalEdges);
  const edgeMtimeMs = Math.max(fs.statSync(perBetaEdgesPath).mtimeMs, fs.statSync(globalEdgesPath).mtimeMs);
  let firstScoringArtifactMtimeMs = Number.POSITIVE_INFINITY;

  const withinCandidate = (q, qi) => records.map((_, i) => i).filter((i) => i !== qi && records[i].betaKey === q.betaKey);
  const acrossCandidate = (_q, qi) => records.map((_, i) => i).filter((i) => i !== qi);
  const maxK = 10;

  const withinFeatures = Array(records.length);
  const withinNormInfo = {};
  for (const betaKey of BETA_KEYS) {
    const betaRecords = records.filter((r) => r.betaKey === betaKey);
    const norm = normalizeFeatures(betaRecords, (r) => r.signature);
    withinNormInfo[betaKey] = { means: norm.means, stds: norm.stds };
    for (let i = 0; i < betaRecords.length; i++) withinFeatures[betaRecords[i].globalIndex] = norm.normalized[i];
  }
  const globalNorm = normalizeFeatures(records, (r) => r.signature);

  const rawWithinFeatures = Array(records.length);
  const rawNormInfo = {};
  for (const betaKey of BETA_KEYS) {
    const betaRecords = records.filter((r) => r.betaKey === betaKey);
    const norm = normalizeFeatures(betaRecords, (r) => computeRawMatrixVector(r.config));
    rawNormInfo[betaKey] = {
      componentCount: norm.means.length,
      meansSha256: crypto.createHash("sha256").update(JSON.stringify(norm.means)).digest("hex"),
      stdsSha256: crypto.createHash("sha256").update(JSON.stringify(norm.stds)).digest("hex"),
    };
    for (let i = 0; i < betaRecords.length; i++) rawWithinFeatures[betaRecords[i].globalIndex] = norm.normalized[i];
  }
  const rawGlobalNorm = normalizeFeatures(records, (r) => computeRawMatrixVector(r.config));

  console.log("[YM-P2-V3-AGG] gauge-randomized v1-signature + sigma2_W33 control");
  const gaugeSignatures = [];
  let gaugeSignatureResidualMax = 0;
  let gaugeTargetResidualMax = 0;
  for (const r of records) {
    const rng = mulberry32(deriveSubstreamSeed(MASTER_SEED, LOCKED.gaugeRandSeedTag, r.betaKey, r.configIdx));
    const transformed = applySU2GaugeTransform(r.config, randomGaugeQuaternions(12, rng));
    const sig = computeSignatureV1(transformed);
    const sigVec = signatureVectorFromObject(sig);
    const held = computeHeldoutV1(transformed);
    gaugeSignatures[r.globalIndex] = sigVec;
    r.gaugeMuW33 = held.W33_mean;
    r.gaugeSigma2W33 = held.W33_var;
    gaugeSignatureResidualMax = Math.max(gaugeSignatureResidualMax, signatureMaxAbsResidual(signatureObjectFromVector(r.signature), sig));
    gaugeTargetResidualMax = Math.max(gaugeTargetResidualMax, Math.abs(r.gaugeSigma2W33 - r.sigma2W33), Math.abs(r.gaugeMuW33 - r.muW33));
  }
  const gaugeEdges = computeTargetEdges(records, startIso, LOCKED.binConvention, "gauge");
  assignTargetBinsFromEdges(records, gaugeEdges.perBetaEdges, gaugeEdges.globalEdges, "gauge");

  const gaugeWithinFeatures = Array(records.length);
  for (const betaKey of BETA_KEYS) {
    const betaRecords = records.filter((r) => r.betaKey === betaKey);
    const norm = normalizeFeatures(betaRecords, (r) => gaugeSignatures[r.globalIndex]);
    for (let i = 0; i < betaRecords.length; i++) gaugeWithinFeatures[betaRecords[i].globalIndex] = norm.normalized[i];
  }
  const gaugeGlobalNorm = normalizeFeatures(records, (r) => gaugeSignatures[r.globalIndex]);

  writeJSON(path.join(aggDir, "signature_normalization.json"), {
    distanceMetric: LOCKED.distanceMetric,
    signatureVocabularyVersion: LOCKED.signatureVocab,
    signatureSource: LOCKED.signatureSource,
    signatureDimension: SIGNATURE_KEYS.length,
    signatureKeys: SIGNATURE_KEYS,
    withinBeta: withinNormInfo,
    acrossBeta: { means: globalNorm.means, stds: globalNorm.stds },
    rawWithinBeta: rawNormInfo,
    rawAcrossBeta: {
      componentCount: rawGlobalNorm.means.length,
      meansSha256: crypto.createHash("sha256").update(JSON.stringify(rawGlobalNorm.means)).digest("hex"),
      stdsSha256: crypto.createHash("sha256").update(JSON.stringify(rawGlobalNorm.stds)).digest("hex"),
    },
  });

  const withinPrimary = buildNearestGraph(records, withinFeatures, withinCandidate, maxK);
  const acrossPrimary = buildNearestGraph(records, globalNorm.normalized, acrossCandidate, maxK);
  const withinGraphPath = path.join(aggDir, "within_beta_nn_graphs.json");
  const acrossGraphPath = path.join(aggDir, "across_beta_nn_graph.json");
  writeJSON(withinGraphPath, { kStored: maxK, graph: graphForFile(withinPrimary) });
  writeJSON(acrossGraphPath, { kStored: maxK, graph: graphForFile(acrossPrimary) });
  firstScoringArtifactMtimeMs = Math.min(fs.statSync(withinGraphPath).mtimeMs, fs.statSync(acrossGraphPath).mtimeMs);

  const controls = {
    CTRL_META: {
      within: buildNearestGraph(records, normalizeFeatures(records, (r) => new Float64Array([r.beta, 12])).normalized, withinCandidate, maxK),
      across: buildNearestGraph(records, normalizeFeatures(records, (r) => new Float64Array([r.beta, 12])).normalized, acrossCandidate, maxK),
    },
    CTRL_RAW: {
      within: buildNearestGraph(records, rawWithinFeatures, withinCandidate, maxK),
      across: buildNearestGraph(records, rawGlobalNorm.normalized, acrossCandidate, maxK),
    },
    CTRL_RAND: {
      within: buildRandomGraph(records, withinCandidate, maxK, "CTRL_RAND_within"),
      across: buildRandomGraph(records, acrossCandidate, maxK, "CTRL_RAND_across"),
    },
    CTRL_RAND_STRAT: {
      within: buildRandomGraph(records, withinCandidate, maxK, "CTRL_RAND_STRAT_within"),
      across: buildRandomGraph(records, withinCandidate, maxK, "CTRL_RAND_STRAT_across"),
    },
    CTRL_GAUGE_RAND: {
      within: buildNearestGraph(records, gaugeWithinFeatures, withinCandidate, maxK),
      across: buildNearestGraph(records, gaugeGlobalNorm.normalized, acrossCandidate, maxK),
    },
  };
  const withinPermutations = [];
  const globalPermutations = [];
  for (let i = 0; i < PERMUTATION_CONTROL_RESAMPLES; i++) {
    withinPermutations.push(makeWithinPermutation(records, i));
    globalPermutations.push(makeGlobalPermutation(records, i));
  }
  controls.CTRL_PERM = { within: withinPrimary, across: acrossPrimary, withinPermutations, globalPermutations };

  for (const [controlId, graphs] of Object.entries(controls)) {
    const payload = controlId === "CTRL_PERM"
      ? {
          kStored: maxK,
          withinGraphSource: "primary",
          acrossGraphSource: "primary",
          permutationControlResamples: PERMUTATION_CONTROL_RESAMPLES,
          seedBase: MASTER_SEED,
          withinPermutationMode: "uniform_within_beta_label_permutation",
          acrossPermutationMode: "uniform_global_label_permutation",
        }
      : { kStored: maxK, within: graphForFile(graphs.within), across: graphForFile(graphs.across) };
    writeJSON(path.join(aggDir, "control_nn_graphs", `${controlId}.json`), payload);
  }

  const labelWithin = (idx) => records[idx].withinBin;
  const labelGlobal = (idx) => records[idx].globalBin;
  const labelGaugeWithin = (idx) => records[idx].gaugeWithinBin;
  const labelGaugeGlobal = (idx) => records[idx].gaugeGlobalBin;
  const scoreRows = [];
  const scoreLookup = {};
  const addScores = (lane, id, graph, labelFn) => {
    for (const k of K_SLATE) {
      const scored = scoreGraph(graph, labelFn, k);
      const ci = bootstrapCi(records, scored.queryScores, LOCKED.bootstrapResamples, `${lane}:${id}:${k}`);
      scoreRows.push([lane, id, k, scored.meanBinPurity, scored.meanBinPurity / CHANCE, ci.low, ci.high]);
      scoreLookup[`${lane}:${id}:${k}`] = scored;
    }
  };
  const addPermutationScores = (lane, id, graph, permutationMaps) => {
    for (const k of K_SLATE) {
      const scored = scorePermutationControl(graph, permutationMaps, k);
      const ci = bootstrapCi(records, scored.queryScores, LOCKED.bootstrapResamples, `${lane}:${id}:${k}`);
      scoreRows.push([lane, id, k, scored.meanBinPurity, scored.meanBinPurity / CHANCE, ci.low, ci.high]);
      scoreLookup[`${lane}:${id}:${k}`] = scored;
    }
  };
  addScores("within_beta", "PRIMARY", withinPrimary, labelWithin);
  addScores("across_beta", "PRIMARY", acrossPrimary, labelGlobal);
  for (const [controlId, graphs] of Object.entries(controls)) {
    if (controlId === "CTRL_PERM") {
      addPermutationScores("within_beta", controlId, graphs.within, graphs.withinPermutations);
      addPermutationScores("across_beta", controlId, graphs.across, graphs.globalPermutations);
    } else if (controlId === "CTRL_GAUGE_RAND") {
      addScores("within_beta", controlId, graphs.within, labelGaugeWithin);
      addScores("across_beta", controlId, graphs.across, labelGaugeGlobal);
    } else {
      addScores("within_beta", controlId, graphs.within, labelWithin);
      addScores("across_beta", controlId, graphs.across, labelGlobal);
    }
  }
  writeCSV(path.join(aggDir, "rank_locality_scores.csv"), ["lane", "control_or_primary", "k", "mean_bin_purity", "discrimination_ratio", "bootstrap_ci_low", "bootstrap_ci_high"], scoreRows);
  writeCSV(path.join(aggDir, "kendall_tau.csv"), ["beta", "mean_kendall_tau", "query_count"], computeKendallRows(records, withinFeatures));
  writeJSON(path.join(aggDir, "v0_v3_comparison.json"), compareV0V3(scoreRows));

  const getScore = (lane, id, k) => scoreLookup[`${lane}:${id}:${k}`].meanBinPurity;
  const wallClockSecondsForBranch = (performance.now() - start) / 1000;
  const branchInputs = {
    v3BinEdgeTimestampOk: writtenEdgeMtimeMs <= firstScoringArtifactMtimeMs && edgeMtimeMs <= firstScoringArtifactMtimeMs,
    binDegenerateBetas,
    withinPrimary5: getScore("within_beta", "PRIMARY", PRIMARY_K),
    withinCtrlRand5: getScore("within_beta", "CTRL_RAND", PRIMARY_K),
    withinCtrlMeta5: getScore("within_beta", "CTRL_META", PRIMARY_K),
    withinCtrlRaw5: getScore("within_beta", "CTRL_RAW", PRIMARY_K),
    withinCtrlPerm5: getScore("within_beta", "CTRL_PERM", PRIMARY_K),
    withinCtrlGaugeRand5: getScore("within_beta", "CTRL_GAUGE_RAND", PRIMARY_K),
    acrossPrimary5: getScore("across_beta", "PRIMARY", PRIMARY_K),
    acrossCtrlRandStrat5: getScore("across_beta", "CTRL_RAND_STRAT", PRIMARY_K),
    gaugeSignatureResidualMax,
    gaugeTargetResidualMax,
    wallClockSeconds: wallClockSecondsForBranch,
  };
  branchInputs.withinPrimaryMinusRand5 = branchInputs.withinPrimary5 - branchInputs.withinCtrlRand5;
  branchInputs.withinPrimaryMinusMeta5 = branchInputs.withinPrimary5 - branchInputs.withinCtrlMeta5;
  branchInputs.withinPrimaryMinusRaw5 = branchInputs.withinPrimary5 - branchInputs.withinCtrlRaw5;
  branchInputs.gaugeRandPurityDiff = Math.max(
    Math.abs(branchInputs.withinCtrlGaugeRand5 - branchInputs.withinPrimary5),
    Math.abs(getScore("across_beta", "CTRL_GAUGE_RAND", PRIMARY_K) - branchInputs.acrossPrimary5),
  );
  branchInputs.acrossPrimaryMinusRandStrat5 = branchInputs.acrossPrimary5 - branchInputs.acrossCtrlRandStrat5;
  branchInputs.thresholds = {
    primaryWithinMeanBinPurity5Min: 0.5,
    primaryMinusCtrlRandMin: 0.10,
    primaryMinusCtrlMetaMin: 0.10,
    primaryMinusCtrlRawMin: 0.10,
    ctrlPermChanceTolerance: 0.05,
    gaugeRandPurityDiffMax: GAUGE_RESIDUAL_TOL,
    gaugeSignatureResidualMax: GAUGE_RESIDUAL_TOL,
    gaugeTargetResidualMax: GAUGE_RESIDUAL_TOL,
    acrossPrimaryMinusRandStratMin: 0.05,
    targetSpreadMin: TARGET_SPREAD_MIN,
    aggregationWallClockSecondsMax: AGGREGATION_WALL_CLOCK_MAX,
  };
  const [branch, reason] = assessBranch(branchInputs);
  branchInputs.branch = branch;
  branchInputs.reason = reason;
  writeJSON(path.join(aggDir, "branch_inputs.json"), branchInputs);

  const wallClockSeconds = (performance.now() - start) / 1000;
  const summary = {
    branch,
    reason,
    phaseVersion: "v3",
    signatureVocabularyVersion: LOCKED.signatureVocab,
    signatureSource: LOCKED.signatureSource,
    signatureDimension: SIGNATURE_KEYS.length,
    heldOutTargetVocabularyVersion: LOCKED.heldoutVocab,
    heldOutTargetSummary: LOCKED.heldoutSummary,
    primary: {
      withinBetaMeanBinPurity5: branchInputs.withinPrimary5,
      acrossBetaMeanBinPurity5: branchInputs.acrossPrimary5,
      discriminationRatioWithin5: branchInputs.withinPrimary5 / CHANCE,
      discriminationRatioAcross5: branchInputs.acrossPrimary5 / CHANCE,
    },
    controls: {
      withinBeta: {
        CTRL_META: branchInputs.withinCtrlMeta5,
        CTRL_RAW: branchInputs.withinCtrlRaw5,
        CTRL_RAND: branchInputs.withinCtrlRand5,
        CTRL_PERM: branchInputs.withinCtrlPerm5,
        CTRL_GAUGE_RAND: branchInputs.withinCtrlGaugeRand5,
      },
      acrossBeta: {
        CTRL_RAND_STRAT: branchInputs.acrossCtrlRandStrat5,
        CTRL_GAUGE_RAND: getScore("across_beta", "CTRL_GAUGE_RAND", PRIMARY_K),
      },
      CTRL_FINITE_SIZE: { notScored: "phase4_reserved", branch: "YM-P4-DEFERRED_FINITE_SIZE" },
    },
    margins: {
      withinPrimaryMinusRand5: branchInputs.withinPrimaryMinusRand5,
      withinPrimaryMinusMeta5: branchInputs.withinPrimaryMinusMeta5,
      withinPrimaryMinusRaw5: branchInputs.withinPrimaryMinusRaw5,
      acrossPrimaryMinusRandStrat5: branchInputs.acrossPrimaryMinusRandStrat5,
      gaugeRandPurityDiff: branchInputs.gaugeRandPurityDiff,
    },
    integrity: {
      v3BinEdgeTimestampOk: branchInputs.v3BinEdgeTimestampOk,
      gaugeSignatureResidualMax,
      gaugeTargetResidualMax,
    },
    sigma2W33: {
      perBetaBinEdgesPath: "aggregation/per_beta_v3_bin_edges.json",
      globalBinEdgesPath: "aggregation/global_v3_bin_edges.json",
      binDegenerateBetas,
    },
    wallClockSeconds,
  };
  writeJSON(path.join(aggDir, "summary.json"), summary);

  const codeInfo = getGitInfo();
  const manifest = {
    phase: "phase2",
    phaseVersion: "v3",
    cell: LOCKED.cell,
    latticeSize: [12, 12, 12],
    betaSlate: [2.0, 2.4, 2.8],
    perBetaConfigurations: 32,
    totalConfigurations: 96,
    signatureVocabularyVersion: LOCKED.signatureVocab,
    signatureSource: LOCKED.signatureSource,
    heldOutTargetVocabularyVersion: LOCKED.heldoutVocab,
    heldOutTargetSummary: LOCKED.heldoutSummary,
    heldOutTargetSampleCountPerConfig: 5184,
    heldOutVarianceEstimator: LOCKED.heldoutVarianceEstimator,
    binConvention: LOCKED.binConvention,
    distanceMetric: LOCKED.distanceMetric,
    kSlate: K_SLATE,
    primaryK: PRIMARY_K,
    bootstrapResamples: LOCKED.bootstrapResamples,
    controlsScored: ["CTRL_META", "CTRL_RAW", "CTRL_RAND", "CTRL_RAND_STRAT", "CTRL_PERM", "CTRL_GAUGE_RAND"],
    controlsDeferred: ["CTRL_FINITE_SIZE"],
    v0EnsembleSources,
    codeCommit: codeInfo.codeCommit,
    gitDirty: codeInfo.gitDirty,
    commandLine: reconstructCommandLine(),
    wallClockSeconds,
  };
  writeJSON(path.join(aggDir, "manifest.json"), manifest);
  finalizeHashes(aggDir);

  console.log(`[YM-P2-V3-AGG] verdict: ${branch}`);
  console.log(`[YM-P2-V3-AGG] ${reason}`);
  console.log(`[YM-P2-V3-AGG] within primary@5 ${branchInputs.withinPrimary5.toFixed(4)}; raw ${branchInputs.withinCtrlRaw5.toFixed(4)}; rand ${branchInputs.withinCtrlRand5.toFixed(4)}`);
  console.log(`[YM-P2-V3-AGG] gauge residuals signature=${gaugeSignatureResidualMax.toExponential(3)} target=${gaugeTargetResidualMax.toExponential(3)}`);
  console.log(`[YM-P2-V3-AGG] wall clock ${wallClockSeconds.toFixed(2)} s`);
}

main().catch((err) => {
  console.error("[YM-P2-V3-AGG] runner threw:");
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
