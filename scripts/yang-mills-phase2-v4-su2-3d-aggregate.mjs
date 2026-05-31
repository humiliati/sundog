#!/usr/bin/env node
// scripts/yang-mills-phase2-v4-su2-3d-aggregate.mjs
//
// Yang-Mills Phase 2 v4 - powered-target audit followed, only if admitted,
// by the unchanged SU(2) 3D relative-locality aggregate.

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
  mulberry32,
  deriveSubstreamSeed,
  getLink,
  qMul,
  qConj,
} from "./lib/yang-mills-su2-3d-core.mjs";

const LOCKED = Object.freeze({
  phaseVersion: "v4",
  cell: "SU2_3D",
  latticeSize: "12x12x12",
  ensembleRoot: "results/yang-mills/phase2/SU2_3D",
  outRequired: "results/yang-mills/phase2/SU2_3D/2026-05-31_su2_3d_relative_locality_v4",
  powerIccGate: 0.50,
  powerAgreementGate: 0.50,
  leakageCvr2Gate: 0.25,
  kPrimary: 5,
  bootstrapResamples: 1000,
  seed: 202605310204,
  signatureVocab: "v1",
  signatureSource: "v0_reread",
  heldoutVocab: "v3",
  splitRule: "site_coordinate_parity",
  distanceMetric: "euclidean_zscore",
  kSlate: [3, 5, 10],
  binConvention: "per_beta_tertile_linear",
  gaugeRandSeedTag: "phase2_v4_aggregation",
  gaugeTransformsPerConfig: 1,
});

const BETA_KEYS = Object.freeze(["2.0", "2.4", "2.8"]);
const ENSEMBLE_DIR_BY_BETA = Object.freeze({
  "2.0": "2026-05-29_su2_3d_beta2.0_ensemble_v0",
  "2.4": "2026-05-29_su2_3d_beta2.4_ensemble_v0",
  "2.8": "2026-05-29_su2_3d_beta2.8_ensemble_v0",
});
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
const CANDIDATE_TARGETS = Object.freeze(["mean_W14", "mean_W23", "sigma2_W14", "sigma2_W23", "ratio_W23_W14"]);
const PRIOR_TARGETS = Object.freeze(["gamma_held", "sigma2_W33"]);
const ALL_AUDIT_TARGETS = Object.freeze([...CANDIDATE_TARGETS, ...PRIOR_TARGETS]);
const ORIENTATIONS = Object.freeze([[0, 1], [0, 2], [1, 2]]);
const CHANCE = 1 / 3;
const PERMUTATION_CONTROL_RESAMPLES = 1000;
const TARGET_SPREAD_MIN = 1e-12;
const GAUGE_RESIDUAL_TOL = 1e-12;
const AGGREGATION_WALL_CLOCK_MAX = 600;
const GAMMA_EPSILON = 1e-10;
const LOOP_SAMPLE_COUNT = 12 * 12 * 12 * 3;
const UNDEFINED_FRACTION_MAX = 0.05;
const NUMPY_DEFAULT_RNG_0_PERMUTATION_32 = Object.freeze([
  2, 11, 25, 21, 10, 4, 29, 16,
  23, 6, 18, 26, 3, 30, 8, 0,
  19, 12, 20, 13, 7, 5, 17, 14,
  27, 22, 9, 28, 24, 1, 15, 31,
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
  const checkNumber = (key, expected, actual) => {
    const value = Number(actual);
    if (!Number.isFinite(value) || value !== expected) failures.push(`--${key} expected ${expected}, got ${actual}`);
  };
  check("ensemble-root", LOCKED.ensembleRoot, normalizedPathForCompare(args["ensemble-root"]));
  check("out", LOCKED.outRequired, normalizedPathForCompare(args.out));
  checkNumber("power-icc-gate", LOCKED.powerIccGate, args["power-icc-gate"]);
  checkNumber("power-agreement-gate", LOCKED.powerAgreementGate, args["power-agreement-gate"]);
  checkNumber("leakage-cvr2-gate", LOCKED.leakageCvr2Gate, args["leakage-cvr2-gate"]);
  checkNumber("k-primary", LOCKED.kPrimary, args["k-primary"]);
  checkNumber("bootstrap", LOCKED.bootstrapResamples, args.bootstrap);
  checkNumber("seed", LOCKED.seed, args.seed);
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

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
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

function pearson(x, y) {
  const mx = mean(x);
  const my = mean(y);
  let num = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < x.length; i++) {
    const dx = x[i] - mx;
    const dy = y[i] - my;
    num += dx * dy;
    sx += dx * dx;
    sy += dy * dy;
  }
  const denom = Math.sqrt(sx * sy);
  return denom > 0 ? num / denom : 0;
}

function tertileLabels(values) {
  const low = percentileLinear(values, 1 / 3);
  const high = percentileLinear(values, 2 / 3);
  return values.map((v) => assignBin(v, low, high));
}

function tertileAgreement(a, b) {
  const la = tertileLabels(a);
  const lb = tertileLabels(b);
  let hits = 0;
  for (let i = 0; i < la.length; i++) if (la[i] === lb[i]) hits++;
  return hits / la.length;
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
    const rng = mulberry32(deriveSubstreamSeed(LOCKED.seed, seedLabel, query.betaKey, query.configIdx));
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
  const rng = mulberry32(deriveSubstreamSeed(LOCKED.seed, seedLabel, betaKey || "global", resampleIndex));
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
  const rng = mulberry32(deriveSubstreamSeed(LOCKED.seed, "bootstrap", seedLabel));
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

function computeKendallRows(records, withinFeatures, valueKey) {
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
        targetDiffs.push(Math.abs(other[valueKey] - q[valueKey]));
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

function dirX(mu) {
  return mu === 0 ? 1 : 0;
}

function dirY(mu) {
  return mu === 1 ? 1 : 0;
}

function dirZ(mu) {
  return mu === 2 ? 1 : 0;
}

function wilsonLoopTraceHalfInPlane(state, x, y, z, mu, nu, nMu, nNu) {
  let acc = [1, 0, 0, 0];
  const mx = dirX(mu), my = dirY(mu), mz = dirZ(mu);
  const nx = dirX(nu), ny = dirY(nu), nz = dirZ(nu);
  for (let i = 0; i < nMu; i++) {
    acc = qMul(acc, getLink(state, mu, x + i * mx, y + i * my, z + i * mz));
  }
  for (let j = 0; j < nNu; j++) {
    acc = qMul(
      acc,
      getLink(
        state,
        nu,
        x + nMu * mx + j * nx,
        y + nMu * my + j * ny,
        z + nMu * mz + j * nz,
      ),
    );
  }
  for (let i = 0; i < nMu; i++) {
    acc = qMul(
      acc,
      qConj(
        getLink(
          state,
          mu,
          x + (nMu - 1 - i) * mx + nNu * nx,
          y + (nMu - 1 - i) * my + nNu * ny,
          z + (nMu - 1 - i) * mz + nNu * nz,
        ),
      ),
    );
  }
  for (let j = 0; j < nNu; j++) {
    acc = qMul(
      acc,
      qConj(
        getLink(
          state,
          nu,
          x + (nNu - 1 - j) * nx,
          y + (nNu - 1 - j) * ny,
          z + (nNu - 1 - j) * nz,
        ),
      ),
    );
  }
  return acc[0];
}

function emptyStats() {
  return { n: 0, sum: 0, sumSq: 0 };
}

function addStat(stats, v) {
  stats.n++;
  stats.sum += v;
  stats.sumSq += v * v;
}

function finishStats(stats) {
  const meanValue = stats.sum / stats.n;
  return {
    n: stats.n,
    mean: meanValue,
    variance: Math.max(0, stats.sumSq / stats.n - meanValue * meanValue),
  };
}

function loopMeanVarByParity(state, nMu, nNu) {
  const L = state.L;
  const full = emptyStats();
  const halfA = emptyStats();
  const halfB = emptyStats();
  for (const [mu, nu] of ORIENTATIONS) {
    for (let x = 0; x < L; x++) {
      for (let y = 0; y < L; y++) {
        for (let z = 0; z < L; z++) {
          const v = wilsonLoopTraceHalfInPlane(state, x, y, z, mu, nu, nMu, nNu);
          addStat(full, v);
          addStat(((x + y + z) & 1) === 0 ? halfA : halfB, v);
        }
      }
    }
  }
  return { full: finishStats(full), halfA: finishStats(halfA), halfB: finishStats(halfB) };
}

function computeV4LoopSummaries(state) {
  return {
    W14: loopMeanVarByParity(state, 1, 4),
    W23: loopMeanVarByParity(state, 2, 3),
    W33: loopMeanVarByParity(state, 3, 3),
  };
}

function gammaHeldFromMeans(W14, W23, W33) {
  const areas = [4, 6, 9];
  const values = [W14, W23, W33];
  const y = values.map((v) => Math.log(Math.max(v, GAMMA_EPSILON)));
  const clamped = values.some((v) => v <= GAMMA_EPSILON);
  const n = areas.length;
  let sumA = 0, sumY = 0, sumAA = 0, sumAY = 0;
  for (let i = 0; i < n; i++) {
    sumA += areas[i];
    sumY += y[i];
    sumAA += areas[i] * areas[i];
    sumAY += areas[i] * y[i];
  }
  const slope = (n * sumAY - sumA * sumY) / (n * sumAA - sumA * sumA);
  return { value: -slope, clamped };
}

function targetValueFromSummaries(target, summaries, halfKey = "full") {
  const W14 = summaries.W14[halfKey];
  const W23 = summaries.W23[halfKey];
  const W33 = summaries.W33[halfKey];
  if (target === "mean_W14") return { value: W14.mean, undefined: false };
  if (target === "mean_W23") return { value: W23.mean, undefined: false };
  if (target === "sigma2_W14") return { value: W14.variance, undefined: false };
  if (target === "sigma2_W23") return { value: W23.variance, undefined: false };
  if (target === "sigma2_W33") return { value: W33.variance, undefined: false };
  if (target === "ratio_W23_W14") {
    if (W14.mean === 0 || !Number.isFinite(W14.mean)) return { value: NaN, undefined: true };
    return { value: W23.mean / W14.mean, undefined: false };
  }
  if (target === "gamma_held") {
    return { ...gammaHeldFromMeans(W14.mean, W23.mean, W33.mean), undefined: false };
  }
  throw new Error(`unknown target ${target}`);
}

function maxAbsVectorDiff(a, b) {
  let maxAbs = 0;
  for (let i = 0; i < a.length; i++) maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
  return maxAbs;
}

function olsCoefficients(xRows, y) {
  const m = xRows.length;
  const p = xRows[0].length + 1;
  const columns = Array.from({ length: p }, (_, j) => {
    const col = new Float64Array(m);
    for (let i = 0; i < m; i++) col[i] = j === 0 ? 1 : xRows[i][j - 1];
    return col;
  });
  const qColumns = [];
  const r = Array.from({ length: p }, () => Array(p).fill(0));
  for (let j = 0; j < p; j++) {
    const v = Float64Array.from(columns[j]);
    for (let i = 0; i < j; i++) {
      let dot = 0;
      for (let row = 0; row < m; row++) dot += qColumns[i][row] * v[row];
      r[i][j] = dot;
      for (let row = 0; row < m; row++) v[row] -= dot * qColumns[i][row];
    }
    let normSq = 0;
    for (let row = 0; row < m; row++) normSq += v[row] * v[row];
    const norm = Math.sqrt(normSq);
    if (norm < 1e-14) throw new Error("rank-deficient OLS design matrix");
    r[j][j] = norm;
    for (let row = 0; row < m; row++) v[row] /= norm;
    qColumns[j] = v;
  }
  const qty = Array(p).fill(0);
  for (let j = 0; j < p; j++) {
    for (let row = 0; row < m; row++) qty[j] += qColumns[j][row] * y[row];
  }
  const coef = Array(p).fill(0);
  for (let i = p - 1; i >= 0; i--) {
    let rhs = qty[i];
    for (let j = i + 1; j < p; j++) rhs -= r[i][j] * coef[j];
    coef[i] = rhs / r[i][i];
  }
  return coef;
}

function olsPredict(coef, x) {
  let out = coef[0];
  for (let j = 0; j < x.length; j++) out += coef[j + 1] * x[j];
  return out;
}

function cvFoldPermutation(n) {
  if (n === 32) return Array.from(NUMPY_DEFAULT_RNG_0_PERMUTATION_32);
  const arr = Array.from({ length: n }, (_, i) => i);
  const rng = mulberry32(deriveSubstreamSeed(0, "cv_r2_fallback"));
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function arraySplit(order, folds) {
  const out = [];
  const n = order.length;
  const base = Math.floor(n / folds);
  const extra = n % folds;
  let cursor = 0;
  for (let k = 0; k < folds; k++) {
    const size = base + (k < extra ? 1 : 0);
    out.push(order.slice(cursor, cursor + size));
    cursor += size;
  }
  return out;
}

function cvR2(features, y, folds = 5) {
  const n = y.length;
  const yMean = mean(y);
  const sst = y.reduce((acc, v) => acc + (v - yMean) * (v - yMean), 0);
  if (sst <= 1e-30) return 0;
  const parts = arraySplit(cvFoldPermutation(n), folds);
  let sse = 0;
  for (let k = 0; k < folds; k++) {
    const test = new Set(parts[k]);
    const trainIdx = [];
    const testIdx = [];
    for (let i = 0; i < n; i++) (test.has(i) ? testIdx : trainIdx).push(i);
    const trainX = trainIdx.map((i) => features[i]);
    const trainY = trainIdx.map((i) => y[i]);
    const coef = olsCoefficients(trainX, trainY);
    for (const i of testIdx) {
      const err = y[i] - olsPredict(coef, features[i]);
      sse += err * err;
    }
  }
  return 1 - sse / sst;
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

function evaluateAuditTarget(target, records, gates) {
  const perBeta = {};
  const admissiblePool = CANDIDATE_TARGETS.includes(target);
  for (const betaKey of BETA_KEYS) {
    const betaRecords = records.filter((r) => r.betaKey === betaKey);
    const validRecords = [];
    const fullValues = [];
    const halfAValues = [];
    const halfBValues = [];
    const flagged = [];
    for (const r of betaRecords) {
      const full = targetValueFromSummaries(target, r.v4Loops, "full");
      const halfA = targetValueFromSummaries(target, r.v4Loops, "halfA");
      const halfB = targetValueFromSummaries(target, r.v4Loops, "halfB");
      const bad = full.undefined || halfA.undefined || halfB.undefined ||
        !Number.isFinite(full.value) || !Number.isFinite(halfA.value) || !Number.isFinite(halfB.value);
      if (bad) {
        flagged.push(r.id);
        continue;
      }
      validRecords.push(r);
      fullValues.push(full.value);
      halfAValues.push(halfA.value);
      halfBValues.push(halfB.value);
    }
    const undefinedFraction = flagged.length / betaRecords.length;
    const undefinedStatus = undefinedFraction > UNDEFINED_FRACTION_MAX;
    const icc = validRecords.length >= 3 && !undefinedStatus ? pearson(halfAValues, halfBValues) : 0;
    const agreement = validRecords.length >= 3 && !undefinedStatus ? tertileAgreement(halfAValues, halfBValues) : 0;
    const leakage = validRecords.length >= 12 && !undefinedStatus
      ? cvR2(validRecords.map((r) => r.signature), fullValues, 5)
      : Number.POSITIVE_INFINITY;
    const powered = !undefinedStatus && icc >= gates.icc && agreement >= gates.agreement;
    const disjoint = !undefinedStatus && leakage <= gates.leakage;
    perBeta[betaKey] = {
      beta: Number(betaKey),
      validCount: validRecords.length,
      flaggedCount: flagged.length,
      flaggedIds: flagged,
      undefinedFraction,
      undefinedStatus,
      icc,
      agreement,
      leakageCvR2: leakage,
      powered,
      disjoint,
      admitted: admissiblePool && powered && disjoint,
    };
  }
  const poweredAllBetas = BETA_KEYS.every((betaKey) => perBeta[betaKey].powered);
  const disjointAllBetas = BETA_KEYS.every((betaKey) => perBeta[betaKey].disjoint);
  const admittedAllBetas = admissiblePool && BETA_KEYS.every((betaKey) => perBeta[betaKey].admitted);
  return {
    target,
    admissiblePool,
    perBeta,
    poweredAllBetas,
    disjointAllBetas,
    admittedAllBetas,
    meanIcc: mean(BETA_KEYS.map((betaKey) => perBeta[betaKey].icc)),
    meanLeakageCvR2: mean(BETA_KEYS.map((betaKey) => perBeta[betaKey].leakageCvR2)),
    maxUndefinedFraction: Math.max(...BETA_KEYS.map((betaKey) => perBeta[betaKey].undefinedFraction)),
  };
}

function auditRows(audits) {
  const rows = [];
  for (const audit of audits) {
    for (const betaKey of BETA_KEYS) {
      const p = audit.perBeta[betaKey];
      rows.push([
        audit.target,
        audit.admissiblePool ? 1 : 0,
        betaKey,
        p.validCount,
        p.flaggedCount,
        p.undefinedFraction,
        p.undefinedStatus ? 1 : 0,
        p.icc,
        p.agreement,
        p.leakageCvR2,
        p.powered ? 1 : 0,
        p.disjoint ? 1 : 0,
        p.admitted ? 1 : 0,
      ]);
    }
  }
  return rows;
}

function selectPrimary(audits) {
  const admitted = audits.filter((a) => a.admittedAllBetas);
  admitted.sort((a, b) =>
    b.meanIcc - a.meanIcc ||
    a.meanLeakageCvR2 - b.meanLeakageCvR2 ||
    a.target.localeCompare(b.target));
  return { primary: admitted[0] || null, admitted };
}

function computeTargetEdges(records, valueKey, target, frozenAtIso, convention) {
  const binDegenerateBetas = [];
  const perBetaEdges = {
    frozenAt: frozenAtIso,
    convention,
    target,
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
    target,
    lowEdge: percentileLinear(allValues, 1 / 3),
    highEdge: percentileLinear(allValues, 2 / 3),
    spread: Math.max(...allValues) - Math.min(...allValues),
  };
  return { perBetaEdges, globalEdges, binDegenerateBetas };
}

function assignTargetBinsFromEdges(records, valueKey, perBetaEdges, globalEdges, labelPrefix = "") {
  const withinKey = labelPrefix ? `${labelPrefix}WithinBin` : "withinBin";
  const globalKey = labelPrefix ? `${labelPrefix}GlobalBin` : "globalBin";
  for (const betaKey of BETA_KEYS) {
    const edges = perBetaEdges.betas[betaKey];
    const betaRecords = records.filter((r) => r.betaKey === betaKey);
    for (const r of betaRecords) r[withinKey] = assignBin(r[valueKey], edges.lowEdge, edges.highEdge);
    edges.assignments = betaRecords.map((r) => ({
      id: r.id,
      configIdx: r.configIdx,
      value: r[valueKey],
      bin: r[withinKey],
    }));
  }
  for (const r of records) r[globalKey] = assignBin(r[valueKey], globalEdges.lowEdge, globalEdges.highEdge);
  globalEdges.assignments = records.map((r) => ({
    id: r.id,
    beta: r.beta,
    configIdx: r.configIdx,
    value: r[valueKey],
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

function compareV0V4(scoreRows, primaryTarget) {
  const v0ScorePath = "results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v0/aggregation/rank_locality_scores.csv";
  const v0 = readScoreMap(v0ScorePath);
  const rows = [];
  for (const row of scoreRows) {
    const [lane, id, k, v4Mean] = row;
    const key = `${lane}:${id}:${k}`;
    const v0Mean = v0.get(key);
    rows.push({
      lane,
      controlOrPrimary: id,
      k,
      v0GammaHeldMeanBinPurity: v0Mean ?? null,
      v4PrimaryTarget: primaryTarget,
      v4MeanBinPurity: v4Mean,
      deltaV4MinusV0: v0Mean === undefined ? null : v4Mean - v0Mean,
    });
  }
  return { v0ScorePath, rows };
}

function assessStage2Branch(inputs) {
  if (inputs.wallClockSeconds > AGGREGATION_WALL_CLOCK_MAX) return ["Z void_run", "aggregation wall clock exceeded 10-minute cap"];
  if (!inputs.admittedTargetTimestampOk) return ["Z void_run", "admitted-target timestamp drift"];
  if (!inputs.v4BinEdgeTimestampOk) return ["Z void_run", "v4 bin-edge timestamp drift"];
  if (inputs.binDegenerateBetas.length > 0) return ["Z bin_degenerate", `degenerate ${inputs.primaryTarget} distribution for beta ${inputs.binDegenerateBetas.join(",")}`];
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
  return ["P2-A bounded_positive", "every Phase 2 v4 point-estimate gate passes on a powered target"];
}

function stage1Branch(admittedTarget, gammaAudit) {
  if (gammaAudit.poweredAllBetas) {
    return ["Z void_run", "gamma_held passed the v4 power self-validation gate"];
  }
  if (!admittedTarget.primary) {
    return ["YM-P2-UNDERPOWERED no_powered_target_in_envelope", "no candidate target is powered and disjoint in all three beta values"];
  }
  return [null, null];
}

function writeStage1Summary({
  aggDir,
  branch,
  reason,
  audits,
  admittedTarget,
  v0EnsembleSources,
  wallClockSeconds,
  stage2Scored,
}) {
  const codeInfo = getGitInfo();
  const branchInputs = {
    branch,
    reason,
    stage2Scored,
    gammaHeldPoweredAllBetas: audits.find((a) => a.target === "gamma_held")?.poweredAllBetas ?? null,
    admittedCandidateCount: admittedTarget.admitted.length,
    thresholds: {
      powerIccGate: LOCKED.powerIccGate,
      powerAgreementGate: LOCKED.powerAgreementGate,
      leakageCvR2Gate: LOCKED.leakageCvr2Gate,
      undefinedFractionMax: UNDEFINED_FRACTION_MAX,
      aggregationWallClockSecondsMax: AGGREGATION_WALL_CLOCK_MAX,
    },
    wallClockSeconds,
  };
  writeJSON(path.join(aggDir, "branch_inputs.json"), branchInputs);
  writeJSON(path.join(aggDir, "summary.json"), {
    branch,
    reason,
    phaseVersion: LOCKED.phaseVersion,
    rankLocalityScored: stage2Scored,
    poweredTargetAudit: {
      path: "aggregation/target_power_audit.csv",
      candidatePool: CANDIDATE_TARGETS,
      priorTargetsReportedNotAdmissible: PRIOR_TARGETS,
      admittedCandidateCount: admittedTarget.admitted.length,
      primaryTarget: admittedTarget.primary?.target ?? null,
    },
    wallClockSeconds,
  });
  writeJSON(path.join(aggDir, "manifest.json"), {
    phase: "phase2",
    phaseVersion: LOCKED.phaseVersion,
    cell: LOCKED.cell,
    latticeSize: [12, 12, 12],
    betaSlate: BETA_KEYS.map(Number),
    perBetaConfigurations: 32,
    totalConfigurations: 96,
    signatureVocabularyVersion: LOCKED.signatureVocab,
    signatureSource: LOCKED.signatureSource,
    heldOutTargetVocabularyVersion: LOCKED.heldoutVocab,
    heldOutTargetCandidatePool: CANDIDATE_TARGETS,
    heldOutPriorTargetsReportedNotAdmissible: PRIOR_TARGETS,
    heldOutTargetSampleCountPerConfig: LOOP_SAMPLE_COUNT,
    splitRule: LOCKED.splitRule,
    powerGate: { iccMin: LOCKED.powerIccGate, agreementMin: LOCKED.powerAgreementGate },
    leakageGate: { cvR2Max: LOCKED.leakageCvr2Gate, estimator: "5_fold_cv_ols_r2_numpy_default_rng_seed_0_fold_order" },
    primarySelectionRule: "highest_mean_beta_icc_then_lowest_mean_leakage",
    rankLocalityScored: stage2Scored,
    kSlate: LOCKED.kSlate,
    primaryK: LOCKED.kPrimary,
    bootstrapResamples: LOCKED.bootstrapResamples,
    controlsScored: stage2Scored ? ["CTRL_META", "CTRL_RAW", "CTRL_RAND", "CTRL_RAND_STRAT", "CTRL_PERM", "CTRL_GAUGE_RAND"] : [],
    controlsDeferred: ["CTRL_FINITE_SIZE"],
    v0EnsembleSources,
    codeCommit: codeInfo.codeCommit,
    gitDirty: codeInfo.gitDirty,
    commandLine: reconstructCommandLine(),
    wallClockSeconds,
  });
}

async function main() {
  const start = performance.now();
  const startWallMs = Date.now();
  const startIso = new Date(startWallMs).toISOString();
  const args = parseArgs(process.argv.slice(2));
  const failures = validateLockedArgs(args);
  if (failures.length > 0) {
    console.error("[YM-P2-V4-AGG] manifest validation FAILED:");
    for (const f of failures) console.error(`  - ${f}`);
    process.exit(2);
  }

  const outDir = path.resolve(args.out);
  const aggDir = path.join(outDir, "aggregation");
  ensureCleanDir(aggDir, "results/yang-mills/phase2/SU2_3D/2026-05-31_su2_3d_relative_locality_v4/aggregation");

  let records = [];
  const v0EnsembleSources = [];
  for (const betaKey of BETA_KEYS) {
    const dir = path.resolve(args["ensemble-root"], ENSEMBLE_DIR_BY_BETA[betaKey]);
    const read = readEnsemble(betaKey, dir, records.length);
    records = records.concat(read.records);
    v0EnsembleSources.push(read.source);
  }
  if (records.length !== 96) throw new Error(`expected 96 records, got ${records.length}`);
  writeJSON(path.join(aggDir, "v0_ensemble_sources.json"), v0EnsembleSources);

  console.log("[YM-P2-V4-AGG] Stage 1: computing site-parity loop summaries for powered-target audit");
  const targetRows = [];
  for (const r of records) {
    r.v4Loops = computeV4LoopSummaries(r.config);
    const values = {};
    for (const target of ALL_AUDIT_TARGETS) values[target] = targetValueFromSummaries(target, r.v4Loops, "full").value;
    targetRows.push([
      r.betaKey,
      r.configIdx,
      values.mean_W14,
      values.mean_W23,
      values.sigma2_W14,
      values.sigma2_W23,
      values.ratio_W23_W14,
      values.gamma_held,
      values.sigma2_W33,
      LOOP_SAMPLE_COUNT,
      r.v4Loops.W14.halfA.n,
      r.v4Loops.W14.halfB.n,
    ]);
  }
  writeCSV(
    path.join(aggDir, "v4_target_resummaries.csv"),
    [
      "beta",
      "config_idx",
      "mean_W14",
      "mean_W23",
      "sigma2_W14",
      "sigma2_W23",
      "ratio_W23_W14",
      "gamma_held",
      "sigma2_W33",
      "sample_count",
      "half_A_sample_count",
      "half_B_sample_count",
    ],
    targetRows,
  );

  const gates = {
    icc: Number(args["power-icc-gate"]),
    agreement: Number(args["power-agreement-gate"]),
    leakage: Number(args["leakage-cvr2-gate"]),
  };
  const audits = ALL_AUDIT_TARGETS.map((target) => evaluateAuditTarget(target, records, gates));
  writeCSV(
    path.join(aggDir, "target_power_audit.csv"),
    [
      "target",
      "admissible_pool",
      "beta",
      "valid_count",
      "flagged_count",
      "undefined_fraction",
      "undefined_status",
      "icc",
      "agreement",
      "leakage_cv_r2",
      "powered",
      "disjoint",
      "admitted",
    ],
    auditRows(audits),
  );
  writeJSON(path.join(aggDir, "target_power_audit.json"), {
    frozenAt: startIso,
    splitRule: LOCKED.splitRule,
    sampleCountPerConfig: LOOP_SAMPLE_COUNT,
    gates,
    candidatePool: CANDIDATE_TARGETS,
    priorTargetsReportedNotAdmissible: PRIOR_TARGETS,
    audits,
  });

  const admittedTarget = selectPrimary(audits);
  const admittedTargetPayload = {
    frozenAt: new Date().toISOString(),
    status: admittedTarget.primary ? "admitted" : "no_admitted_target",
    primaryTarget: admittedTarget.primary?.target ?? null,
    selectionRule: "highest mean-over-beta ICC among candidates admitted in all beta values; tie-break lowest mean leakage CV-R2",
    admittedTargets: admittedTarget.admitted.map((a) => ({
      target: a.target,
      meanIcc: a.meanIcc,
      meanLeakageCvR2: a.meanLeakageCvR2,
      perBeta: a.perBeta,
    })),
    rejectedTargets: audits.filter((a) => a.admissiblePool && !a.admittedAllBetas).map((a) => ({
      target: a.target,
      poweredAllBetas: a.poweredAllBetas,
      disjointAllBetas: a.disjointAllBetas,
      meanIcc: a.meanIcc,
      meanLeakageCvR2: a.meanLeakageCvR2,
      perBeta: a.perBeta,
    })),
  };
  const admittedTargetPath = path.join(aggDir, "admitted_target.json");
  writeJSON(admittedTargetPath, admittedTargetPayload);

  const gammaAudit = audits.find((a) => a.target === "gamma_held");
  const [stage1OnlyBranch, stage1OnlyReason] = stage1Branch(admittedTarget, gammaAudit);
  if (stage1OnlyBranch) {
    const wallClockSeconds = (performance.now() - start) / 1000;
    writeStage1Summary({
      aggDir,
      branch: stage1OnlyBranch,
      reason: stage1OnlyReason,
      audits,
      admittedTarget,
      v0EnsembleSources,
      wallClockSeconds,
      stage2Scored: false,
    });
    finalizeHashes(aggDir);
    console.log(`[YM-P2-V4-AGG] verdict: ${stage1OnlyBranch}`);
    console.log(`[YM-P2-V4-AGG] ${stage1OnlyReason}`);
    console.log(`[YM-P2-V4-AGG] wall clock ${wallClockSeconds.toFixed(2)} s`);
    return;
  }

  const primaryTarget = admittedTarget.primary.target;
  for (const r of records) {
    const target = targetValueFromSummaries(primaryTarget, r.v4Loops, "full");
    if (target.undefined || !Number.isFinite(target.value)) throw new Error(`admitted target ${primaryTarget} undefined for ${r.id}`);
    r.primaryTargetValue = target.value;
  }

  const { perBetaEdges, globalEdges, binDegenerateBetas } = computeTargetEdges(
    records,
    "primaryTargetValue",
    primaryTarget,
    startIso,
    LOCKED.binConvention,
  );
  const perBetaEdgesPath = path.join(aggDir, "per_beta_v4_bin_edges.json");
  const globalEdgesPath = path.join(aggDir, "global_v4_bin_edges.json");
  writeJSON(perBetaEdgesPath, perBetaEdges);
  writeJSON(globalEdgesPath, globalEdges);
  const writtenFreezeMtimeMs = Math.max(
    fs.statSync(admittedTargetPath).mtimeMs,
    fs.statSync(perBetaEdgesPath).mtimeMs,
    fs.statSync(globalEdgesPath).mtimeMs,
  );

  const rereadPerBetaEdges = JSON.parse(fs.readFileSync(perBetaEdgesPath, "utf8"));
  const rereadGlobalEdges = JSON.parse(fs.readFileSync(globalEdgesPath, "utf8"));
  assignTargetBinsFromEdges(records, "primaryTargetValue", rereadPerBetaEdges, rereadGlobalEdges);
  writeJSON(perBetaEdgesPath, rereadPerBetaEdges);
  writeJSON(globalEdgesPath, rereadGlobalEdges);
  const edgeMtimeMs = Math.max(fs.statSync(perBetaEdgesPath).mtimeMs, fs.statSync(globalEdgesPath).mtimeMs);
  let firstScoringArtifactMtimeMs = Number.POSITIVE_INFINITY;
  await sleep(25);

  console.log(`[YM-P2-V4-AGG] Stage 2: scoring admitted powered target ${primaryTarget}`);
  ensureDir(path.join(aggDir, "control_nn_graphs"));
  const withinCandidate = (q, qi) => records.map((_, i) => i).filter((i) => i !== qi && records[i].betaKey === q.betaKey);
  const acrossCandidate = (_q, qi) => records.map((_, i) => i).filter((i) => i !== qi);
  const maxK = Math.max(...LOCKED.kSlate);

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

  const gaugeSignatures = [];
  let gaugeSignatureResidualMax = 0;
  let gaugeTargetResidualMax = 0;
  for (const r of records) {
    const rng = mulberry32(deriveSubstreamSeed(LOCKED.seed, LOCKED.gaugeRandSeedTag, r.betaKey, r.configIdx));
    const transformed = applySU2GaugeTransform(r.config, randomGaugeQuaternions(12, rng));
    const sig = computeSignatureV1(transformed);
    const sigVec = signatureVectorFromObject(sig);
    const loops = computeV4LoopSummaries(transformed);
    const gaugeTarget = targetValueFromSummaries(primaryTarget, loops, "full");
    if (gaugeTarget.undefined || !Number.isFinite(gaugeTarget.value)) throw new Error(`gauge target ${primaryTarget} undefined for ${r.id}`);
    gaugeSignatures[r.globalIndex] = sigVec;
    r.gaugePrimaryTargetValue = gaugeTarget.value;
    gaugeSignatureResidualMax = Math.max(gaugeSignatureResidualMax, signatureMaxAbsResidual(signatureObjectFromVector(r.signature), sig));
    gaugeTargetResidualMax = Math.max(gaugeTargetResidualMax, Math.abs(r.gaugePrimaryTargetValue - r.primaryTargetValue));
  }
  const gaugeEdges = computeTargetEdges(records, "gaugePrimaryTargetValue", primaryTarget, startIso, LOCKED.binConvention);
  assignTargetBinsFromEdges(records, "gaugePrimaryTargetValue", gaugeEdges.perBetaEdges, gaugeEdges.globalEdges, "gauge");

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
          seedBase: LOCKED.seed,
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
    for (const k of LOCKED.kSlate) {
      const scored = scoreGraph(graph, labelFn, k);
      const ci = bootstrapCi(records, scored.queryScores, LOCKED.bootstrapResamples, `${lane}:${id}:${k}`);
      scoreRows.push([lane, id, k, scored.meanBinPurity, scored.meanBinPurity / CHANCE, ci.low, ci.high]);
      scoreLookup[`${lane}:${id}:${k}`] = scored;
    }
  };
  const addPermutationScores = (lane, id, graph, permutationMaps) => {
    for (const k of LOCKED.kSlate) {
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
  writeCSV(path.join(aggDir, "kendall_tau.csv"), ["beta", "mean_kendall_tau", "query_count"], computeKendallRows(records, withinFeatures, "primaryTargetValue"));
  writeJSON(path.join(aggDir, "v0_v4_comparison.json"), compareV0V4(scoreRows, primaryTarget));

  const getScore = (lane, id, k) => scoreLookup[`${lane}:${id}:${k}`].meanBinPurity;
  const wallClockSecondsForBranch = (performance.now() - start) / 1000;
  const branchInputs = {
    primaryTarget,
    admittedTargetTimestampOk: writtenFreezeMtimeMs < firstScoringArtifactMtimeMs,
    v4BinEdgeTimestampOk: edgeMtimeMs < firstScoringArtifactMtimeMs,
    binDegenerateBetas,
    withinPrimary5: getScore("within_beta", "PRIMARY", LOCKED.kPrimary),
    withinCtrlRand5: getScore("within_beta", "CTRL_RAND", LOCKED.kPrimary),
    withinCtrlMeta5: getScore("within_beta", "CTRL_META", LOCKED.kPrimary),
    withinCtrlRaw5: getScore("within_beta", "CTRL_RAW", LOCKED.kPrimary),
    withinCtrlPerm5: getScore("within_beta", "CTRL_PERM", LOCKED.kPrimary),
    withinCtrlGaugeRand5: getScore("within_beta", "CTRL_GAUGE_RAND", LOCKED.kPrimary),
    acrossPrimary5: getScore("across_beta", "PRIMARY", LOCKED.kPrimary),
    acrossCtrlRandStrat5: getScore("across_beta", "CTRL_RAND_STRAT", LOCKED.kPrimary),
    gaugeSignatureResidualMax,
    gaugeTargetResidualMax,
    stage1: {
      gammaHeldPoweredAllBetas: gammaAudit.poweredAllBetas,
      primaryTargetMeanIcc: admittedTarget.primary.meanIcc,
      primaryTargetMeanLeakageCvR2: admittedTarget.primary.meanLeakageCvR2,
    },
    wallClockSeconds: wallClockSecondsForBranch,
  };
  branchInputs.withinPrimaryMinusRand5 = branchInputs.withinPrimary5 - branchInputs.withinCtrlRand5;
  branchInputs.withinPrimaryMinusMeta5 = branchInputs.withinPrimary5 - branchInputs.withinCtrlMeta5;
  branchInputs.withinPrimaryMinusRaw5 = branchInputs.withinPrimary5 - branchInputs.withinCtrlRaw5;
  branchInputs.gaugeRandPurityDiff = Math.max(
    Math.abs(branchInputs.withinCtrlGaugeRand5 - branchInputs.withinPrimary5),
    Math.abs(getScore("across_beta", "CTRL_GAUGE_RAND", LOCKED.kPrimary) - branchInputs.acrossPrimary5),
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
  const [branch, reason] = assessStage2Branch(branchInputs);
  branchInputs.branch = branch;
  branchInputs.reason = reason;
  writeJSON(path.join(aggDir, "branch_inputs.json"), branchInputs);

  const wallClockSeconds = (performance.now() - start) / 1000;
  const summary = {
    branch,
    reason,
    phaseVersion: LOCKED.phaseVersion,
    signatureVocabularyVersion: LOCKED.signatureVocab,
    signatureSource: LOCKED.signatureSource,
    signatureDimension: SIGNATURE_KEYS.length,
    heldOutTargetVocabularyVersion: LOCKED.heldoutVocab,
    heldOutTargetSummary: primaryTarget,
    rankLocalityScored: true,
    poweredTargetAudit: {
      path: "aggregation/target_power_audit.csv",
      admittedTargetPath: "aggregation/admitted_target.json",
      primaryTarget,
      primaryTargetMeanIcc: admittedTarget.primary.meanIcc,
      primaryTargetMeanLeakageCvR2: admittedTarget.primary.meanLeakageCvR2,
      gammaHeldPoweredAllBetas: gammaAudit.poweredAllBetas,
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
        CTRL_GAUGE_RAND: getScore("across_beta", "CTRL_GAUGE_RAND", LOCKED.kPrimary),
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
      admittedTargetTimestampOk: branchInputs.admittedTargetTimestampOk,
      v4BinEdgeTimestampOk: branchInputs.v4BinEdgeTimestampOk,
      gaugeSignatureResidualMax,
      gaugeTargetResidualMax,
    },
    target: {
      primaryTarget,
      perBetaBinEdgesPath: "aggregation/per_beta_v4_bin_edges.json",
      globalBinEdgesPath: "aggregation/global_v4_bin_edges.json",
      binDegenerateBetas,
    },
    wallClockSeconds,
  };
  writeJSON(path.join(aggDir, "summary.json"), summary);

  const codeInfo = getGitInfo();
  const manifest = {
    phase: "phase2",
    phaseVersion: LOCKED.phaseVersion,
    cell: LOCKED.cell,
    latticeSize: [12, 12, 12],
    betaSlate: BETA_KEYS.map(Number),
    perBetaConfigurations: 32,
    totalConfigurations: 96,
    signatureVocabularyVersion: LOCKED.signatureVocab,
    signatureSource: LOCKED.signatureSource,
    heldOutTargetVocabularyVersion: LOCKED.heldoutVocab,
    heldOutTargetSummary: primaryTarget,
    heldOutTargetCandidatePool: CANDIDATE_TARGETS,
    heldOutPriorTargetsReportedNotAdmissible: PRIOR_TARGETS,
    heldOutTargetSampleCountPerConfig: LOOP_SAMPLE_COUNT,
    splitRule: LOCKED.splitRule,
    powerGate: { iccMin: LOCKED.powerIccGate, agreementMin: LOCKED.powerAgreementGate },
    leakageGate: { cvR2Max: LOCKED.leakageCvr2Gate, estimator: "5_fold_cv_ols_r2_numpy_default_rng_seed_0_fold_order" },
    primarySelectionRule: "highest_mean_beta_icc_then_lowest_mean_leakage",
    binConvention: LOCKED.binConvention,
    distanceMetric: LOCKED.distanceMetric,
    kSlate: LOCKED.kSlate,
    primaryK: LOCKED.kPrimary,
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

  console.log(`[YM-P2-V4-AGG] verdict: ${branch}`);
  console.log(`[YM-P2-V4-AGG] ${reason}`);
  console.log(`[YM-P2-V4-AGG] admitted target ${primaryTarget}; mean ICC ${admittedTarget.primary.meanIcc.toFixed(4)}; mean leakage ${admittedTarget.primary.meanLeakageCvR2.toFixed(4)}`);
  console.log(`[YM-P2-V4-AGG] within primary@5 ${branchInputs.withinPrimary5.toFixed(4)}; raw ${branchInputs.withinCtrlRaw5.toFixed(4)}; rand ${branchInputs.withinCtrlRand5.toFixed(4)}`);
  console.log(`[YM-P2-V4-AGG] gauge residuals signature=${gaugeSignatureResidualMax.toExponential(3)} target=${gaugeTargetResidualMax.toExponential(3)}`);
  console.log(`[YM-P2-V4-AGG] wall clock ${wallClockSeconds.toFixed(2)} s`);
}

main().catch((err) => {
  console.error("[YM-P2-V4-AGG] runner threw:");
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
