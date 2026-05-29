#!/usr/bin/env node
// scripts/yang-mills-phase2-su2-3d-aggregate.mjs
//
// Yang-Mills Phase 2 v0 - SU(2) 3D relative-locality aggregation runner.

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
} from "./lib/yang-mills-su2-3d-core.mjs";

const LOCKED = Object.freeze({
  cell: "SU2_3D",
  latticeSize: "12x12x12",
  betaSlate: "2.0,2.4,2.8",
  distanceMetric: "euclidean_zscore",
  kSlate: "3,5,10",
  primaryK: 5,
  bootstrapResamples: 1000,
  binConvention: "per_beta_tertile_linear",
  gaugeRandSeedTag: "phase2_aggregation",
  gaugeTransformsPerConfig: 1,
  outRequired: "results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v0",
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

function validateLockedArgs(args) {
  const failures = [];
  const check = (key, expected, actual) => {
    if (actual !== expected) failures.push(`--${key} expected ${expected}, got ${actual}`);
  };
  check("cell", LOCKED.cell, args.cell);
  check("lattice-size", LOCKED.latticeSize, args["lattice-size"]);
  check("beta-slate", LOCKED.betaSlate, args["beta-slate"]);
  check("distance-metric", LOCKED.distanceMetric, args["distance-metric"]);
  check("k-slate", LOCKED.kSlate, args["k-slate"]);
  check("primary-k", LOCKED.primaryK, Number(args["primary-k"]));
  check("bootstrap-resamples", LOCKED.bootstrapResamples, Number(args["bootstrap-resamples"]));
  check("bin-convention", LOCKED.binConvention, args["bin-convention"]);
  check("gauge-rand-seed-tag", LOCKED.gaugeRandSeedTag, args["gauge-rand-seed-tag"]);
  check("gauge-transforms-per-config", LOCKED.gaugeTransformsPerConfig, Number(args["gauge-transforms-per-config"]));
  check("out", LOCKED.outRequired, (args.out || "").replace(/\\/g, "/"));
  for (const betaKey of BETA_KEYS) {
    check(`in-beta-${betaKey}`, LOCKED.inputs[betaKey], (args[`in-beta-${betaKey}`] || "").replace(/\\/g, "/"));
  }
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

function writeCSV(p, header, rows) {
  const lines = [header.join(",")];
  for (const row of rows) {
    lines.push(row.map((v) => {
      const s = typeof v === "number" ? formatNumber(v) : String(v);
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    }).join(","));
  }
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
    const rng = mulberry32(deriveSubstreamSeed(202605290299, seedLabel, query.betaKey, query.configIdx));
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

function makePermutation(records, seedLabel, betaKey = null) {
  const selected = records.filter((r) => betaKey === null || r.betaKey === betaKey).map((r) => r.globalIndex);
  const labels = selected.map((idx) => records[idx].withinBin);
  const rng = mulberry32(deriveSubstreamSeed(202605290299, seedLabel, betaKey || "global"));
  for (let i = labels.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [labels[i], labels[j]] = [labels[j], labels[i]];
  }
  const out = {};
  for (let i = 0; i < selected.length; i++) out[selected[i]] = labels[i];
  return out;
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

function bootstrapCi(records, queryScores, resamples, seedLabel) {
  const scoresByIndex = new Map(queryScores.map((r) => [r.query, r.score]));
  const byBeta = {};
  for (const betaKey of BETA_KEYS) {
    byBeta[betaKey] = records.filter((r) => r.betaKey === betaKey).map((r) => r.globalIndex);
  }
  const rng = mulberry32(deriveSubstreamSeed(202605290299, "bootstrap", seedLabel));
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

function readEnsemble(betaKey, dir, startIndex) {
  if (!fs.existsSync(dir)) throw new Error(`missing ensemble dir ${dir}`);
  const summary = JSON.parse(fs.readFileSync(path.join(dir, "summary.json"), "utf8"));
  if (summary.branch !== "P1-A ensemble_health_pass") {
    throw new Error(`ensemble ${betaKey} did not pass health gate: ${summary.branch}`);
  }
  const manifest = JSON.parse(fs.readFileSync(path.join(dir, "manifest.json"), "utf8"));
  const signatures = parseCsv(path.join(dir, "signatures", "signature_vectors.csv"));
  const held = parseCsv(path.join(dir, "heldout", "heldout_summary.csv"));
  const configs = fs.readFileSync(path.join(dir, "configs", "su2_links.jsonl"), "utf8").trim().split(/\r?\n/);
  const heldByIdx = new Map(held.map((r) => [Number(r.config_idx), r]));
  const records = signatures.map((row, localIdx) => {
    const configIdx = Number(row.config_idx);
    const h = heldByIdx.get(configIdx);
    const parsedConfig = JSON.parse(configs[localIdx]);
    return {
      globalIndex: startIndex + localIdx,
      id: `beta${betaKey}_config${configIdx}`,
      betaKey,
      beta: Number(betaKey),
      configIdx,
      signature: new Float64Array([
        Number(row.W11_mean),
        Number(row.W11_var),
        Number(row.W12_mean),
        Number(row.W12_var),
        Number(row.W13_mean),
        Number(row.W13_var),
        Number(row.W22_mean),
        Number(row.W22_var),
      ]),
      gammaHeld: Number(h.gamma_held),
      clamped: Number(h.clamped) === 1,
      config: { L: 12, links: new Float64Array(parsedConfig.quaternionLinks) },
      sourceDir: dir,
      manifest,
      summary,
    };
  });
  return records;
}

function graphForFile(graph) {
  return graph.map((row) => ({
    query: row.query,
    queryId: row.queryId,
    neighbors: row.neighbors.map((n) => ({ index: n.index, id: n.id, distanceSq: n.distanceSq })),
  }));
}

function assessBranch(inputs) {
  if (inputs.wallClockSeconds > 600) return ["Z void_run", "aggregation wall clock exceeded 10-minute cap"];
  if (inputs.binDegenerateBetas.length > 0) return ["Z bin_degenerate", `degenerate gamma_held distribution for beta ${inputs.binDegenerateBetas.join(",")}`];
  if (!inputs.binFreezeTimestampOk) return ["Z void_run", "bin-edge timestamp drift"];
  if (Math.abs(inputs.withinCtrlPerm5 - CHANCE) > 0.05) return ["Z graph_contamination", `CTRL_PERM purity ${inputs.withinCtrlPerm5} differs from chance ${CHANCE}`];
  if (inputs.gaugeRandPurityDiff > 1e-12) return ["YM-P1-NEG-A gauge_leakage", `CTRL_GAUGE_RAND purity diff ${inputs.gaugeRandPurityDiff}`];
  if (inputs.withinPrimary5 < 0.5 || inputs.withinPrimaryMinusRand5 < 0.10) return ["YM-P2-NEG-A no_rank_local_structure", `primary=${inputs.withinPrimary5}, primary-CTRL_RAND=${inputs.withinPrimaryMinusRand5}`];
  if (inputs.withinPrimaryMinusMeta5 < 0.10) return ["YM-P2-NEG-B metadata_only", `primary-CTRL_META=${inputs.withinPrimaryMinusMeta5}`];
  if (inputs.withinPrimaryMinusRaw5 < 0.10) return ["YM-P2-NEG-D raw_dominates", `primary-CTRL_RAW=${inputs.withinPrimaryMinusRaw5}`];
  if (inputs.acrossPrimaryMinusRandStrat5 < 0.05) return ["YM-P2-NEG-C coupling_triviality", `across primary-CTRL_RAND_STRAT=${inputs.acrossPrimaryMinusRandStrat5}`];
  return ["P2-A bounded_positive", "every Phase 2 v0 point-estimate gate passes"];
}

async function main() {
  const start = performance.now();
  const args = parseArgs(process.argv.slice(2));
  const failures = validateLockedArgs(args);
  if (failures.length > 0) {
    console.error("[YM-P2-AGG] manifest validation FAILED:");
    for (const f of failures) console.error(`  - ${f}`);
    process.exit(2);
  }

  const outDir = path.resolve(args.out);
  const aggDir = path.join(outDir, "aggregation");
  ensureDir(path.join(aggDir, "control_nn_graphs"));

  let records = [];
  for (const betaKey of BETA_KEYS) {
    const dir = path.resolve(args[`in-beta-${betaKey}`]);
    records = records.concat(readEnsemble(betaKey, dir, records.length));
  }
  if (records.length !== 96) throw new Error(`expected 96 records, got ${records.length}`);

  const binDegenerateBetas = [];
  const perBetaEdges = { frozenAt: new Date().toISOString(), convention: LOCKED.binConvention, betas: {} };
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
    frozenAt: perBetaEdges.frozenAt,
    convention: "global_tertile_linear",
    lowEdge: globalLow,
    highEdge: globalHigh,
    assignments: records.map((r) => ({ id: r.id, beta: r.beta, configIdx: r.configIdx, gammaHeld: r.gammaHeld, bin: r.globalBin })),
  };
  const perBetaEdgesPath = path.join(aggDir, "per_beta_bin_edges.json");
  const globalEdgesPath = path.join(aggDir, "global_bin_edges.json");
  writeJSON(perBetaEdgesPath, perBetaEdges);
  writeJSON(globalEdgesPath, globalEdges);
  const edgeMtimeMs = Math.max(fs.statSync(perBetaEdgesPath).mtimeMs, fs.statSync(globalEdgesPath).mtimeMs);
  JSON.parse(fs.readFileSync(perBetaEdgesPath, "utf8"));
  JSON.parse(fs.readFileSync(globalEdgesPath, "utf8"));
  const firstGraphStartedMs = Date.now();

  const withinCandidate = (q, qi) => records.map((_, i) => i).filter((i) => i !== qi && records[i].betaKey === q.betaKey);
  const acrossCandidate = (_q, qi) => records.map((_, i) => i).filter((i) => i !== qi);
  const sameBetaCandidate = withinCandidate;
  const maxK = 10;

  const sigWithinNormByBeta = {};
  const withinSigFeatures = Array(records.length);
  const withinNormInfo = {};
  for (const betaKey of BETA_KEYS) {
    const betaRecords = records.filter((r) => r.betaKey === betaKey);
    const norm = normalizeFeatures(betaRecords, (r) => r.signature);
    withinNormInfo[betaKey] = { means: norm.means, stds: norm.stds };
    for (let i = 0; i < betaRecords.length; i++) {
      sigWithinNormByBeta[betaRecords[i].globalIndex] = norm.normalized[i];
      withinSigFeatures[betaRecords[i].globalIndex] = norm.normalized[i];
    }
  }
  const sigGlobalNorm = normalizeFeatures(records, (r) => r.signature);
  const rawWithinFeatures = Array(records.length);
  const rawNormInfo = {};
  for (const betaKey of BETA_KEYS) {
    const betaRecords = records.filter((r) => r.betaKey === betaKey);
    const norm = normalizeFeatures(betaRecords, (r) => computeRawMatrixVector(r.config));
    rawNormInfo[betaKey] = { componentCount: norm.means.length, meansSha256: crypto.createHash("sha256").update(JSON.stringify(norm.means)).digest("hex"), stdsSha256: crypto.createHash("sha256").update(JSON.stringify(norm.stds)).digest("hex") };
    for (let i = 0; i < betaRecords.length; i++) rawWithinFeatures[betaRecords[i].globalIndex] = norm.normalized[i];
  }
  const rawGlobalNorm = normalizeFeatures(records, (r) => computeRawMatrixVector(r.config));

  const gaugeSignatures = [];
  let gaugeSignatureResidualMax = 0;
  for (const r of records) {
    const rng = mulberry32(deriveSubstreamSeed(202605290299, LOCKED.gaugeRandSeedTag, r.betaKey, r.configIdx));
    const transformed = applySU2GaugeTransform(r.config, randomGaugeQuaternions(12, rng));
    const sig = computeSignatureV1(transformed);
    const sigVec = new Float64Array([sig.W11_mean, sig.W11_var, sig.W12_mean, sig.W12_var, sig.W13_mean, sig.W13_var, sig.W22_mean, sig.W22_var]);
    gaugeSignatures[r.globalIndex] = sigVec;
    gaugeSignatureResidualMax = Math.max(gaugeSignatureResidualMax, signatureMaxAbsResidual({
      W11_mean: r.signature[0], W11_var: r.signature[1], W12_mean: r.signature[2], W12_var: r.signature[3],
      W13_mean: r.signature[4], W13_var: r.signature[5], W22_mean: r.signature[6], W22_var: r.signature[7],
    }, sig));
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
    withinBeta: withinNormInfo,
    acrossBeta: { means: sigGlobalNorm.means, stds: sigGlobalNorm.stds },
    rawWithinBeta: rawNormInfo,
    rawAcrossBeta: { componentCount: rawGlobalNorm.means.length, meansSha256: crypto.createHash("sha256").update(JSON.stringify(rawGlobalNorm.means)).digest("hex"), stdsSha256: crypto.createHash("sha256").update(JSON.stringify(rawGlobalNorm.stds)).digest("hex") },
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
  const withinPermutations = {};
  for (const betaKey of BETA_KEYS) Object.assign(withinPermutations, makePermutation(records, "CTRL_PERM_within", betaKey));
  const globalPermutation = makePermutation(records, "CTRL_PERM_across", null);
  controls.CTRL_PERM = { within: withinPrimary, across: acrossPrimary, withinPermutations, globalPermutation };

  for (const [controlId, graphs] of Object.entries(controls)) {
    const payload = controlId === "CTRL_PERM"
      ? { kStored: maxK, withinGraphSource: "primary", acrossGraphSource: "primary", withinPermutations, globalPermutation }
      : { kStored: maxK, within: graphForFile(graphs.within), across: graphForFile(graphs.across) };
    writeJSON(path.join(aggDir, "control_nn_graphs", `${controlId}.json`), payload);
  }

  const labelWithin = (idx) => records[idx].withinBin;
  const labelGlobal = (idx) => records[idx].globalBin;
  const labelWithinPerm = (idx) => withinPermutations[idx];
  const labelGlobalPerm = (idx) => globalPermutation[idx];
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
  addScores("within_beta", "PRIMARY", withinPrimary, labelWithin);
  addScores("across_beta", "PRIMARY", acrossPrimary, labelGlobal);
  for (const [controlId, graphs] of Object.entries(controls)) {
    if (controlId === "CTRL_PERM") {
      addScores("within_beta", controlId, graphs.within, labelWithinPerm);
      addScores("across_beta", controlId, graphs.across, labelGlobalPerm);
    } else {
      addScores("within_beta", controlId, graphs.within, labelWithin);
      addScores("across_beta", controlId, graphs.across, labelGlobal);
    }
  }
  writeCSV(path.join(aggDir, "rank_locality_scores.csv"), ["lane", "control_or_primary", "k", "mean_bin_purity", "discrimination_ratio", "bootstrap_ci_low", "bootstrap_ci_high"], scoreRows);

  writeCSV(path.join(aggDir, "kendall_tau.csv"), ["beta", "mean_kendall_tau", "query_count"], computeKendallRows(records, withinSigFeatures));

  const getScore = (lane, id, k) => scoreLookup[`${lane}:${id}:${k}`].meanBinPurity;
  const wallClockSecondsForBranch = (performance.now() - start) / 1000;
  const branchInputs = {
    binFreezeTimestampOk: edgeMtimeMs <= firstGraphStartedMs,
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
    acrossPrimaryMinusRandStratMin: 0.05,
    aggregationWallClockSecondsMax: 600,
  };
  const [branch, reason] = assessBranch(branchInputs);
  branchInputs.branch = branch;
  branchInputs.reason = reason;
  writeJSON(path.join(aggDir, "branch_inputs.json"), branchInputs);

  const wallClockSeconds = (performance.now() - start) / 1000;
  const summary = {
    branch,
    reason,
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
    gammaHeld: {
      perBetaBinEdgesPath: "aggregation/per_beta_bin_edges.json",
      globalBinEdgesPath: "aggregation/global_bin_edges.json",
      binDegenerateBetas,
    },
    gaugeRandomization: { signatureResidualMax: gaugeSignatureResidualMax },
    wallClockSeconds,
  };
  writeJSON(path.join(aggDir, "summary.json"), summary);

  const codeInfo = getGitInfo();
  const manifest = {
    phase: "phase2",
    cell: LOCKED.cell,
    latticeSize: [12, 12, 12],
    betaSlate: [2.0, 2.4, 2.8],
    perBetaConfigurations: 32,
    totalConfigurations: 96,
    signatureVocabularyVersion: "v1",
    heldOutTargetVocabularyVersion: "v1",
    gammaHeldEpsilonFloor: 1e-10,
    binConvention: LOCKED.binConvention,
    globalBinConvention: "global_tertile_linear",
    distanceMetric: LOCKED.distanceMetric,
    kSlate: K_SLATE,
    primaryK: PRIMARY_K,
    bootstrapResamples: LOCKED.bootstrapResamples,
    controlsScored: ["CTRL_META", "CTRL_RAW", "CTRL_RAND", "CTRL_RAND_STRAT", "CTRL_PERM", "CTRL_GAUGE_RAND"],
    controlsDeferred: ["CTRL_FINITE_SIZE"],
    perBetaReceiptPaths: BETA_KEYS.map((betaKey) => LOCKED.inputs[betaKey]),
    codeCommit: codeInfo.codeCommit,
    gitDirty: codeInfo.gitDirty,
    commandLine: reconstructCommandLine(),
    wallClockSeconds,
  };
  writeJSON(path.join(aggDir, "manifest.json"), manifest);
  finalizeHashes(aggDir);

  console.log(`[YM-P2-AGG] verdict: ${branch}`);
  console.log(`[YM-P2-AGG] ${reason}`);
  console.log(`[YM-P2-AGG] within primary@5 ${branchInputs.withinPrimary5.toFixed(4)}; raw ${branchInputs.withinCtrlRaw5.toFixed(4)}; rand ${branchInputs.withinCtrlRand5.toFixed(4)}`);
  console.log(`[YM-P2-AGG] wall clock ${wallClockSeconds.toFixed(2)} s`);
}

main().catch((err) => {
  console.error("[YM-P2-AGG] runner threw:");
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
