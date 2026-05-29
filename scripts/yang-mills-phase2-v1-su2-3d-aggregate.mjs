#!/usr/bin/env node
// scripts/yang-mills-phase2-v1-su2-3d-aggregate.mjs
//
// Yang-Mills Phase 2 v1 - SU(2) 3D relative-locality aggregation runner
// with P0-amended APE-smearing signature vocabulary v4.

import * as fs from "node:fs";
import * as path from "node:path";
import * as crypto from "node:crypto";
import { execSync } from "node:child_process";
import { performance } from "node:perf_hooks";

import {
  computeSignatureV1,
  computeRawMatrixVector,
  applySU2GaugeTransform,
  randomGaugeQuaternions,
  signatureMaxAbsResidual,
  meanPlaquetteByOrientation,
  orientationRelativeSpread,
  mulberry32,
  deriveSubstreamSeed,
} from "./lib/yang-mills-su2-3d-core.mjs";
import {
  apeSmearSU2_3D,
  APE_SMEARING_LOCK,
} from "./lib/yang-mills-su2-3d-smearing.mjs";

const LOCKED = Object.freeze({
  cell: "SU2_3D",
  latticeSize: "12x12x12",
  betaSlate: "2.0,2.4,2.8",
  p0Amendment: "P0_AMD_001_APE_SMEARING_2026-05-29",
  smearingAlgorithm: "APE",
  smearingAlpha: 0.5,
  smearingIterations: 10,
  distanceMetric: "euclidean_zscore",
  kSlate: "3,5,10",
  primaryK: 5,
  bootstrapResamples: 1000,
  binConvention: "per_beta_tertile_linear",
  gaugeRandSeedTag: "phase2_v1_aggregation",
  gaugeTransformsPerConfig: 1,
  v0BinEdges: "results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v0/aggregation/per_beta_bin_edges.json",
  outRequired: "results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v1",
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
const MASTER_SEED = 202605290399;
const EDGE_MATCH_TOL = 1e-12;
const SIGNATURE_INTEGRITY_TOL = 1e-12;
const SMEARING_HEALTH_TOL = 1e-10;
const ORIENTATION_SPREAD_MAX = 5e-2;
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
  check("p0-amendment", LOCKED.p0Amendment, args["p0-amendment"]);
  check("smearing-algorithm", LOCKED.smearingAlgorithm, args["smearing-algorithm"]);
  check("smearing-alpha", LOCKED.smearingAlpha, Number(args["smearing-alpha"]));
  check("smearing-iterations", LOCKED.smearingIterations, Number(args["smearing-iterations"]));
  check("v0-bin-edges", LOCKED.v0BinEdges, normalizedPathForCompare(args["v0-bin-edges"]));
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
  if (!normalized.endsWith(requiredSuffix)) {
    throw new Error(`refusing to clean unexpected output dir: ${resolved}`);
  }
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

function signatureObjectFromVector(vec) {
  const out = {};
  for (let i = 0; i < SIGNATURE_KEYS.length; i++) out[SIGNATURE_KEYS[i]] = vec[i];
  return out;
}

function signatureVector(sig) {
  return new Float64Array(SIGNATURE_KEYS.map((k) => Number(sig[k])));
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
  return {
    normalized,
    means: Array.from(means),
    stds: Array.from(stds),
  };
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
  for (const betaKey of BETA_KEYS) {
    Object.assign(out, makePermutation(records, "CTRL_PERM_within", betaKey, "withinBin", resampleIndex));
  }
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
  return {
    meanBinPurity: mean(queryScores.map((r) => r.score)),
    queryScores,
  };
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
  const queryScores = graph.map((row) => ({
    query: row.query,
    score: sums.get(row.query) / permutationMaps.length,
  }));
  return {
    meanBinPurity: mean(queryScores.map((r) => r.score)),
    queryScores,
  };
}

function bootstrapCi(records, queryScores, resamples, seedLabel) {
  const scoresByIndex = new Map(queryScores.map((r) => [r.query, r.score]));
  const byBeta = {};
  for (const betaKey of BETA_KEYS) {
    byBeta[betaKey] = records.filter((r) => r.betaKey === betaKey).map((r) => r.globalIndex);
  }
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
  return {
    low: percentileLinear(values, 0.025),
    high: percentileLinear(values, 0.975),
  };
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
      const gammaDiffs = [];
      for (const other of betaRecords) {
        if (other.globalIndex === q.globalIndex) continue;
        dists.push(Math.sqrt(distanceSq(withinFeatures[q.globalIndex], withinFeatures[other.globalIndex])));
        gammaDiffs.push(Math.abs(other.gammaHeld - q.gammaHeld));
      }
      const tau = kendallTau(dists, gammaDiffs);
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
  const configHash = sha256OfFile(configsPath);
  if (hashes["configs/su2_links.jsonl"] !== configHash) {
    fail(`configs hash ${configHash} does not match hashes.json ${hashes["configs/su2_links.jsonl"]}`);
  }
  return configHash;
}

function readEnsemble(betaKey, dir, startIndex) {
  const required = [
    "manifest.json",
    "summary.json",
    "configs/su2_links.jsonl",
    "signatures/signature_vectors.csv",
    "heldout/heldout_loop_values.csv",
    "heldout/heldout_summary.csv",
    "hashes.json",
  ];
  for (const rel of required) {
    if (!fs.existsSync(path.join(dir, rel))) throw new Error(`missing required v0 file ${path.join(dir, rel)}`);
  }
  const summary = JSON.parse(fs.readFileSync(path.join(dir, "summary.json"), "utf8"));
  const manifest = JSON.parse(fs.readFileSync(path.join(dir, "manifest.json"), "utf8"));
  const hashes = JSON.parse(fs.readFileSync(path.join(dir, "hashes.json"), "utf8"));
  const configHash = validateV0EnsembleHealth(betaKey, dir, summary, hashes);
  const signatures = parseCsv(path.join(dir, "signatures", "signature_vectors.csv"));
  const held = parseCsv(path.join(dir, "heldout", "heldout_summary.csv"));
  const configs = fs.readFileSync(path.join(dir, "configs", "su2_links.jsonl"), "utf8").trim().split(/\r?\n/);
  if (signatures.length !== 32 || held.length !== 32 || configs.length !== 32) {
    throw new Error(`v0 ensemble ${betaKey} expected 32 signatures/held/configs`);
  }
  const heldByIdx = new Map(held.map((r) => [Number(r.config_idx), r]));
  const records = signatures.map((row, localIdx) => {
    const configIdx = Number(row.config_idx);
    const h = heldByIdx.get(configIdx);
    const parsedConfig = JSON.parse(configs[localIdx]);
    if (!h) throw new Error(`missing heldout summary for beta ${betaKey} config ${configIdx}`);
    if (parsedConfig.configIdx !== configIdx) {
      throw new Error(`config index drift for beta ${betaKey}: signature ${configIdx}, jsonl ${parsedConfig.configIdx}`);
    }
    return {
      globalIndex: startIndex + localIdx,
      id: `beta${betaKey}_config${configIdx}`,
      betaKey,
      beta: Number(betaKey),
      configIdx,
      v0Signature: new Float64Array(SIGNATURE_KEYS.map((k) => Number(row[k]))),
      gammaHeld: Number(h.gamma_held),
      clamped: Number(h.clamped) === 1,
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
  };
  return { records, source };
}

function computeBinEdges(records, frozenAtIso, convention) {
  const binDegenerateBetas = [];
  const perBetaEdges = { frozenAt: frozenAtIso, convention, betas: {} };
  for (const betaKey of BETA_KEYS) {
    const betaRecords = records.filter((r) => r.betaKey === betaKey);
    const gammas = betaRecords.map((r) => r.gammaHeld);
    const spread = Math.max(...gammas) - Math.min(...gammas);
    if (spread < 1e-10) binDegenerateBetas.push(betaKey);
    const lowEdge = percentileLinear(gammas, 1 / 3);
    const highEdge = percentileLinear(gammas, 2 / 3);
    for (const r of betaRecords) r.withinBin = assignBin(r.gammaHeld, lowEdge, highEdge);
    perBetaEdges.betas[betaKey] = {
      lowEdge,
      highEdge,
      spread,
      assignments: betaRecords.map((r) => ({ id: r.id, configIdx: r.configIdx, gammaHeld: r.gammaHeld, bin: r.withinBin })),
    };
  }
  const allGammas = records.map((r) => r.gammaHeld);
  const globalLow = percentileLinear(allGammas, 1 / 3);
  const globalHigh = percentileLinear(allGammas, 2 / 3);
  for (const r of records) r.globalBin = assignBin(r.gammaHeld, globalLow, globalHigh);
  const globalEdges = {
    frozenAt: frozenAtIso,
    convention: "global_tertile_linear",
    lowEdge: globalLow,
    highEdge: globalHigh,
    assignments: records.map((r) => ({ id: r.id, beta: r.beta, configIdx: r.configIdx, gammaHeld: r.gammaHeld, bin: r.globalBin })),
  };
  return { perBetaEdges, globalEdges, binDegenerateBetas };
}

function compareEdgesToV0(perBetaEdges, globalEdges, v0PerBetaPath) {
  const v0PerBeta = JSON.parse(fs.readFileSync(v0PerBetaPath, "utf8"));
  const v0GlobalPath = path.join(path.dirname(v0PerBetaPath), "global_bin_edges.json");
  const v0Global = JSON.parse(fs.readFileSync(v0GlobalPath, "utf8"));
  let perBetaMaxAbsDiff = 0;
  for (const betaKey of BETA_KEYS) {
    const current = perBetaEdges.betas[betaKey];
    const locked = v0PerBeta.betas[betaKey];
    perBetaMaxAbsDiff = Math.max(
      perBetaMaxAbsDiff,
      Math.abs(current.lowEdge - locked.lowEdge),
      Math.abs(current.highEdge - locked.highEdge),
    );
  }
  const globalMaxAbsDiff = Math.max(
    Math.abs(globalEdges.lowEdge - v0Global.lowEdge),
    Math.abs(globalEdges.highEdge - v0Global.highEdge),
  );
  return {
    v0PerBetaPath: v0PerBetaPath.replace(/\\/g, "/"),
    v0GlobalPath: v0GlobalPath.replace(/\\/g, "/"),
    v0PerBetaFrozenAt: v0PerBeta.frozenAt,
    v0GlobalFrozenAt: v0Global.frozenAt,
    perBetaMaxAbsDiff,
    globalMaxAbsDiff,
  };
}

function summarizeSmearedOrientation(records) {
  const byBeta = {};
  let maxSpread = 0;
  for (const betaKey of BETA_KEYS) {
    const betaRecords = records.filter((r) => r.betaKey === betaKey);
    const means = { xy: 0, xz: 0, yz: 0 };
    for (const r of betaRecords) {
      means.xy += r.smearedPlaquetteByOrientation.xy;
      means.xz += r.smearedPlaquetteByOrientation.xz;
      means.yz += r.smearedPlaquetteByOrientation.yz;
    }
    means.xy /= betaRecords.length;
    means.xz /= betaRecords.length;
    means.yz /= betaRecords.length;
    const relativeSpread = orientationRelativeSpread(means);
    byBeta[betaKey] = { meanPlaquetteByOrientation: means, relativeSpread };
    if (relativeSpread > maxSpread) maxSpread = relativeSpread;
  }
  return { byBeta, maxSpread };
}

function readV0ScoreMap() {
  const scorePath = path.join(path.dirname(LOCKED.v0BinEdges), "rank_locality_scores.csv");
  const rows = parseCsv(scorePath);
  const map = new Map();
  for (const row of rows) {
    map.set(`${row.lane}:${row.control_or_primary}:${row.k}`, Number(row.mean_bin_purity));
  }
  return { scorePath, map };
}

function compareV0V1(scoreRows) {
  const { scorePath, map } = readV0ScoreMap();
  const rows = [];
  for (const row of scoreRows) {
    const [lane, id, k, v1Mean] = row;
    const key = `${lane}:${id}:${k}`;
    const v0Mean = map.get(key);
    rows.push({
      lane,
      controlOrPrimary: id,
      k,
      v0MeanBinPurity: v0Mean ?? null,
      v1MeanBinPurity: v1Mean,
      deltaV1MinusV0: v0Mean === undefined ? null : v1Mean - v0Mean,
    });
  }
  return { v0ScorePath: scorePath.replace(/\\/g, "/"), rows };
}

function assessBranch(inputs) {
  if (inputs.wallClockSeconds > AGGREGATION_WALL_CLOCK_MAX) return ["Z void_run", "aggregation wall clock exceeded 10-minute cap"];
  if (!inputs.v0BinEdgeTimestampOk) return ["Z void_run", "v0 bin-edge timestamp is not earlier than v1 start"];
  if (inputs.binDegenerateBetas.length > 0) return ["Z bin_degenerate", `degenerate gamma_held distribution for beta ${inputs.binDegenerateBetas.join(",")}`];
  if (inputs.perBetaBinEdgeMaxAbsDiff > EDGE_MATCH_TOL || inputs.globalBinEdgeMaxAbsDiff > EDGE_MATCH_TOL) {
    return ["Z void_run", `v0 bin-edge drift perBeta=${inputs.perBetaBinEdgeMaxAbsDiff}, global=${inputs.globalBinEdgeMaxAbsDiff}`];
  }
  if (inputs.bareSignatureIntegrityMaxAbsResidual > SIGNATURE_INTEGRITY_TOL) {
    return ["Z void_run", `bare signature integrity residual ${inputs.bareSignatureIntegrityMaxAbsResidual}`];
  }
  if (inputs.maxSmearingDetDrift > SMEARING_HEALTH_TOL || inputs.maxSmearingUnitarityResidual > SMEARING_HEALTH_TOL) {
    return ["YM-P2-QUAR-E smearing_drift", `det=${inputs.maxSmearingDetDrift}, unit=${inputs.maxSmearingUnitarityResidual}`];
  }
  if (inputs.perOrientationSmearedSpreadMax > ORIENTATION_SPREAD_MAX) {
    return ["YM-P2-QUAR-C orientation_anisotropy", `smeared orientation spread ${inputs.perOrientationSmearedSpreadMax}`];
  }
  if (Math.abs(inputs.withinCtrlPerm5 - CHANCE) > 0.05) {
    return ["Z graph_contamination", `CTRL_PERM purity ${inputs.withinCtrlPerm5} differs from chance ${CHANCE}`];
  }
  if (inputs.gaugeRandPurityDiff > 1e-12 || inputs.gaugeSignatureResidualMax > 1e-12) {
    return ["YM-P1-NEG-A gauge_leakage", `gauge purity diff ${inputs.gaugeRandPurityDiff}, signature residual ${inputs.gaugeSignatureResidualMax}`];
  }
  if (inputs.withinPrimary5 < 0.5 || inputs.withinPrimaryMinusRand5 < 0.10) {
    return ["YM-P2-NEG-A no_rank_local_structure", `primary=${inputs.withinPrimary5}, primary-CTRL_RAND=${inputs.withinPrimaryMinusRand5}`];
  }
  if (inputs.withinPrimaryMinusMeta5 < 0.10) return ["YM-P2-NEG-B metadata_only", `primary-CTRL_META=${inputs.withinPrimaryMinusMeta5}`];
  if (inputs.withinPrimaryMinusRaw5 < 0.10) return ["YM-P2-NEG-D raw_dominates", `primary-CTRL_RAW=${inputs.withinPrimaryMinusRaw5}`];
  if (inputs.acrossPrimaryMinusRandStrat5 < 0.05) {
    return ["YM-P2-NEG-C coupling_triviality", `across primary-CTRL_RAND_STRAT=${inputs.acrossPrimaryMinusRandStrat5}`];
  }
  return ["P2-A bounded_positive", "every Phase 2 v1 point-estimate gate passes"];
}

async function main() {
  const start = performance.now();
  const startWallMs = Date.now();
  const startIso = new Date(startWallMs).toISOString();
  const args = parseArgs(process.argv.slice(2));
  const failures = validateLockedArgs(args);
  if (failures.length > 0) {
    console.error("[YM-P2-V1-AGG] manifest validation FAILED:");
    for (const f of failures) console.error(`  - ${f}`);
    process.exit(2);
  }

  const outDir = path.resolve(args.out);
  const aggDir = path.join(outDir, "aggregation");
  ensureCleanDir(aggDir, "results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v1/aggregation");
  ensureDir(path.join(aggDir, "control_nn_graphs"));

  const v0BinEdgesPath = path.resolve(args["v0-bin-edges"]);
  const v0GlobalBinEdgesPath = path.join(path.dirname(v0BinEdgesPath), "global_bin_edges.json");
  const v0BinEdgeTimestampOk =
    fs.statSync(v0BinEdgesPath).mtimeMs <= startWallMs &&
    fs.statSync(v0GlobalBinEdgesPath).mtimeMs <= startWallMs;

  let records = [];
  const v0EnsembleSources = [];
  for (const betaKey of BETA_KEYS) {
    const dir = path.resolve(args[`in-beta-${betaKey}`]);
    const read = readEnsemble(betaKey, dir, records.length);
    records = records.concat(read.records);
    v0EnsembleSources.push(read.source);
  }
  if (records.length !== 96) throw new Error(`expected 96 records, got ${records.length}`);
  writeJSON(path.join(aggDir, "v0_ensemble_sources.json"), v0EnsembleSources);

  console.log("[YM-P2-V1-AGG] applying APE smearing to 96 configs");
  const smearingHealthRows = [];
  const bareIntegrityRows = [];
  const smearedSignatureRows = [];
  let maxSmearingDetDrift = 0;
  let maxSmearingUnitarityResidual = 0;
  let bareSignatureIntegrityMaxAbsResidual = 0;
  let bareVsSmearedMaxAbsDelta = 0;
  let bareVsSmearedMeanAbsDeltaSum = 0;
  let bareVsSmearedDeltaCount = 0;

  for (const r of records) {
    const bareSig = computeSignatureV1(r.config);
    const bareVec = signatureVector(bareSig);
    const integrityResidual = signatureMaxAbsResidual(signatureObjectFromVector(r.v0Signature), bareSig);
    bareSignatureIntegrityMaxAbsResidual = Math.max(bareSignatureIntegrityMaxAbsResidual, integrityResidual);
    bareIntegrityRows.push([
      r.betaKey,
      r.configIdx,
      ...SIGNATURE_KEYS.map((k, i) => bareVec[i] - r.v0Signature[i]),
      integrityResidual,
    ]);

    const smeared = apeSmearSU2_3D(r.config, {
      alpha: LOCKED.smearingAlpha,
      iterations: LOCKED.smearingIterations,
    });
    for (const h of smeared.iterationHealth) {
      maxSmearingDetDrift = Math.max(maxSmearingDetDrift, h.maxDetDrift);
      maxSmearingUnitarityResidual = Math.max(maxSmearingUnitarityResidual, h.maxUnitarityResidual);
      smearingHealthRows.push([r.betaKey, r.configIdx, h.iteration, h.maxDetDrift, h.maxUnitarityResidual]);
    }
    const smearedSig = computeSignatureV1(smeared.state);
    r.smearedSignature = signatureVector(smearedSig);
    r.smearedPlaquetteByOrientation = meanPlaquetteByOrientation(smeared.state);
    for (let i = 0; i < SIGNATURE_KEYS.length; i++) {
      const d = Math.abs(r.smearedSignature[i] - r.v0Signature[i]);
      bareVsSmearedMaxAbsDelta = Math.max(bareVsSmearedMaxAbsDelta, d);
      bareVsSmearedMeanAbsDeltaSum += d;
      bareVsSmearedDeltaCount++;
    }
    smearedSignatureRows.push([r.betaKey, r.configIdx, ...Array.from(r.smearedSignature)]);
  }

  writeCSV(
    path.join(aggDir, "smearing_health.csv"),
    ["beta", "config_idx", "iteration", "max_det_drift", "max_unitarity_residual"],
    smearingHealthRows,
  );
  writeCSV(
    path.join(aggDir, "bare_signature_integrity.csv"),
    ["beta", "config_idx", ...SIGNATURE_KEYS.map((k) => `${k}_diff_vs_v0`), "max_abs_residual"],
    bareIntegrityRows,
  );
  writeCSV(
    path.join(aggDir, "smeared_signature_vectors.csv"),
    ["beta", "config_idx", ...SIGNATURE_KEYS],
    smearedSignatureRows,
  );

  const { perBetaEdges, globalEdges, binDegenerateBetas } = computeBinEdges(records, startIso, LOCKED.binConvention);
  const edgeComparison = compareEdgesToV0(perBetaEdges, globalEdges, v0BinEdgesPath);
  perBetaEdges.v0Replay = edgeComparison;
  globalEdges.v0Replay = edgeComparison;
  writeJSON(path.join(aggDir, "per_beta_bin_edges.json"), perBetaEdges);
  writeJSON(path.join(aggDir, "global_bin_edges.json"), globalEdges);

  const orientationSummary = summarizeSmearedOrientation(records);

  const withinCandidate = (q, qi) => records.map((_, i) => i).filter((i) => i !== qi && records[i].betaKey === q.betaKey);
  const acrossCandidate = (_q, qi) => records.map((_, i) => i).filter((i) => i !== qi);
  const sameBetaCandidate = withinCandidate;
  const maxK = 10;

  const withinSigFeatures = Array(records.length);
  const withinNormInfo = {};
  for (const betaKey of BETA_KEYS) {
    const betaRecords = records.filter((r) => r.betaKey === betaKey);
    const norm = normalizeFeatures(betaRecords, (r) => r.smearedSignature);
    withinNormInfo[betaKey] = { means: norm.means, stds: norm.stds };
    for (let i = 0; i < betaRecords.length; i++) withinSigFeatures[betaRecords[i].globalIndex] = norm.normalized[i];
  }
  const sigGlobalNorm = normalizeFeatures(records, (r) => r.smearedSignature);

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

  console.log("[YM-P2-V1-AGG] gauge-randomized smearing control");
  const gaugeSignatures = [];
  let gaugeSignatureResidualMax = 0;
  for (const r of records) {
    const rng = mulberry32(deriveSubstreamSeed(MASTER_SEED, LOCKED.gaugeRandSeedTag, r.betaKey, r.configIdx));
    const transformed = applySU2GaugeTransform(r.config, randomGaugeQuaternions(12, rng));
    const smearedTransformed = apeSmearSU2_3D(transformed, {
      alpha: LOCKED.smearingAlpha,
      iterations: LOCKED.smearingIterations,
    });
    const sig = computeSignatureV1(smearedTransformed.state);
    const sigVec = signatureVector(sig);
    gaugeSignatures[r.globalIndex] = sigVec;
    gaugeSignatureResidualMax = Math.max(
      gaugeSignatureResidualMax,
      signatureMaxAbsResidual(signatureObjectFromVector(r.smearedSignature), sig),
    );
  }
  const gaugeWithinFeatures = Array(records.length);
  for (const betaKey of BETA_KEYS) {
    const betaRecords = records.filter((r) => r.betaKey === betaKey);
    const norm = normalizeFeatures(betaRecords, (r) => gaugeSignatures[r.globalIndex]);
    for (let i = 0; i < betaRecords.length; i++) gaugeWithinFeatures[betaRecords[i].globalIndex] = norm.normalized[i];
  }
  const gaugeGlobalNorm = normalizeFeatures(records, (r) => gaugeSignatures[r.globalIndex]);

  writeJSON(path.join(aggDir, "signature_normalization.json"), {
    distanceMetric: LOCKED.distanceMetric,
    signatureVocabularyVersion: "v4",
    withinBeta: withinNormInfo,
    acrossBeta: { means: sigGlobalNorm.means, stds: sigGlobalNorm.stds },
    rawWithinBeta: rawNormInfo,
    rawAcrossBeta: {
      componentCount: rawGlobalNorm.means.length,
      meansSha256: crypto.createHash("sha256").update(JSON.stringify(rawGlobalNorm.means)).digest("hex"),
      stdsSha256: crypto.createHash("sha256").update(JSON.stringify(rawGlobalNorm.stds)).digest("hex"),
    },
  });

  const withinPrimary = buildNearestGraph(records, withinSigFeatures, withinCandidate, maxK);
  const acrossPrimary = buildNearestGraph(records, sigGlobalNorm.normalized, acrossCandidate, maxK);
  writeJSON(path.join(aggDir, "within_beta_nn_graphs.json"), { kStored: maxK, graph: graphForFile(withinPrimary) });
  writeJSON(path.join(aggDir, "across_beta_nn_graph.json"), { kStored: maxK, graph: graphForFile(acrossPrimary) });

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
      within: buildRandomGraph(records, sameBetaCandidate, maxK, "CTRL_RAND_within"),
      across: buildRandomGraph(records, acrossCandidate, maxK, "CTRL_RAND_across"),
    },
    CTRL_RAND_STRAT: {
      within: buildRandomGraph(records, sameBetaCandidate, maxK, "CTRL_RAND_STRAT_within"),
      across: buildRandomGraph(records, sameBetaCandidate, maxK, "CTRL_RAND_STRAT_across"),
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
    } else {
      addScores("within_beta", controlId, graphs.within, labelWithin);
      addScores("across_beta", controlId, graphs.across, labelGlobal);
    }
  }
  writeCSV(
    path.join(aggDir, "rank_locality_scores.csv"),
    ["lane", "control_or_primary", "k", "mean_bin_purity", "discrimination_ratio", "bootstrap_ci_low", "bootstrap_ci_high"],
    scoreRows,
  );

  writeCSV(path.join(aggDir, "kendall_tau.csv"), ["beta", "mean_kendall_tau", "query_count"], computeKendallRows(records, withinSigFeatures));
  writeJSON(path.join(aggDir, "v0_vs_v1_comparison.json"), compareV0V1(scoreRows));

  const getScore = (lane, id, k) => scoreLookup[`${lane}:${id}:${k}`].meanBinPurity;
  const wallClockSecondsForBranch = (performance.now() - start) / 1000;
  const branchInputs = {
    v0BinEdgeTimestampOk,
    binDegenerateBetas,
    perBetaBinEdgeMaxAbsDiff: edgeComparison.perBetaMaxAbsDiff,
    globalBinEdgeMaxAbsDiff: edgeComparison.globalMaxAbsDiff,
    bareSignatureIntegrityMaxAbsResidual,
    maxSmearingDetDrift,
    maxSmearingUnitarityResidual,
    perOrientationSmearedSpreadMax: orientationSummary.maxSpread,
    smearedOrientationByBeta: orientationSummary.byBeta,
    withinPrimary5: getScore("within_beta", "PRIMARY", PRIMARY_K),
    withinCtrlRand5: getScore("within_beta", "CTRL_RAND", PRIMARY_K),
    withinCtrlMeta5: getScore("within_beta", "CTRL_META", PRIMARY_K),
    withinCtrlRaw5: getScore("within_beta", "CTRL_RAW", PRIMARY_K),
    withinCtrlPerm5: getScore("within_beta", "CTRL_PERM", PRIMARY_K),
    withinCtrlGaugeRand5: getScore("within_beta", "CTRL_GAUGE_RAND", PRIMARY_K),
    acrossPrimary5: getScore("across_beta", "PRIMARY", PRIMARY_K),
    acrossCtrlRandStrat5: getScore("across_beta", "CTRL_RAND_STRAT", PRIMARY_K),
    gaugeSignatureResidualMax,
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
    gaugeRandPurityDiffMax: 1e-12,
    gaugeSignatureResidualMax: 1e-12,
    acrossPrimaryMinusRandStratMin: 0.05,
    perBetaBinEdgeMaxAbsDiff: EDGE_MATCH_TOL,
    globalBinEdgeMaxAbsDiff: EDGE_MATCH_TOL,
    bareSignatureIntegrityMaxAbsResidual: SIGNATURE_INTEGRITY_TOL,
    maxSmearingDetDrift: SMEARING_HEALTH_TOL,
    maxSmearingUnitarityResidual: SMEARING_HEALTH_TOL,
    perOrientationSmearedSpreadMax: ORIENTATION_SPREAD_MAX,
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
    phaseVersion: "v1",
    signatureVocabularyVersion: "v4",
    smearing: {
      algorithm: LOCKED.smearingAlgorithm,
      alpha: LOCKED.smearingAlpha,
      iterations: LOCKED.smearingIterations,
      maxDetDrift: maxSmearingDetDrift,
      maxUnitarityResidual: maxSmearingUnitarityResidual,
      perOrientationSpreadMax: orientationSummary.maxSpread,
    },
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
      bareSignatureIntegrityMaxAbsResidual,
      perBetaBinEdgeMaxAbsDiff: edgeComparison.perBetaMaxAbsDiff,
      globalBinEdgeMaxAbsDiff: edgeComparison.globalMaxAbsDiff,
      gaugeSignatureResidualMax,
    },
    diagnostic: {
      bareVsSmearedMaxAbsDelta,
      bareVsSmearedMeanAbsDelta: bareVsSmearedMeanAbsDeltaSum / Math.max(1, bareVsSmearedDeltaCount),
    },
    gammaHeld: {
      perBetaBinEdgesPath: "aggregation/per_beta_bin_edges.json",
      globalBinEdgesPath: "aggregation/global_bin_edges.json",
      binDegenerateBetas,
    },
    wallClockSeconds,
  };
  writeJSON(path.join(aggDir, "summary.json"), summary);

  const codeInfo = getGitInfo();
  const manifest = {
    phase: "phase2",
    phaseVersion: "v1",
    cell: LOCKED.cell,
    latticeSize: [12, 12, 12],
    betaSlate: [2.0, 2.4, 2.8],
    perBetaConfigurations: 32,
    totalConfigurations: 96,
    signatureVocabularyVersion: "v4",
    heldOutTargetVocabularyVersion: "v1",
    p0Amendment: LOCKED.p0Amendment,
    smearingAlgorithm: LOCKED.smearingAlgorithm,
    smearingAlpha: LOCKED.smearingAlpha,
    smearingIterations: LOCKED.smearingIterations,
    smearingProjection: "complex_sqrt_det_branch_positive_realtr",
    gammaHeldEpsilonFloor: 1e-10,
    binConvention: LOCKED.binConvention,
    distanceMetric: LOCKED.distanceMetric,
    kSlate: K_SLATE,
    primaryK: PRIMARY_K,
    bootstrapResamples: LOCKED.bootstrapResamples,
    controlsScored: ["CTRL_META", "CTRL_RAW", "CTRL_RAND", "CTRL_RAND_STRAT", "CTRL_PERM", "CTRL_GAUGE_RAND"],
    controlsDeferred: ["CTRL_FINITE_SIZE"],
    v0EnsembleSources,
    v0BinEdgesSource: LOCKED.v0BinEdges,
    maxSmearingDetDrift,
    maxSmearingUnitarityResidual,
    perOrientationSmearedSpread: orientationSummary.maxSpread,
    apeSmearingLock: APE_SMEARING_LOCK,
    codeCommit: codeInfo.codeCommit,
    gitDirty: codeInfo.gitDirty,
    commandLine: reconstructCommandLine(),
    wallClockSeconds,
  };
  writeJSON(path.join(aggDir, "manifest.json"), manifest);
  finalizeHashes(aggDir);

  console.log(`[YM-P2-V1-AGG] verdict: ${branch}`);
  console.log(`[YM-P2-V1-AGG] ${reason}`);
  console.log(`[YM-P2-V1-AGG] within primary@5 ${branchInputs.withinPrimary5.toFixed(4)}; raw ${branchInputs.withinCtrlRaw5.toFixed(4)}; rand ${branchInputs.withinCtrlRand5.toFixed(4)}`);
  console.log(`[YM-P2-V1-AGG] smearing det ${maxSmearingDetDrift.toExponential(3)}; unit ${maxSmearingUnitarityResidual.toExponential(3)}; gauge residual ${gaugeSignatureResidualMax.toExponential(3)}`);
  console.log(`[YM-P2-V1-AGG] wall clock ${wallClockSeconds.toFixed(2)} s`);
}

main().catch((err) => {
  console.error("[YM-P2-V1-AGG] runner threw:");
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
