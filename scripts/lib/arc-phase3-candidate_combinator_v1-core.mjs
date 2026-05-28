import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { ARC_PHASE0_BASELINE_NAMES, arcPhase0BaselinePredictions } from "../arc-phase0-baselines.mjs";

export const FEATURE_SCHEMA_VERSION = "arc-p3-feature-v1";
export const PROTOCOL_VERSION = "arc-p3-protocol-v1";
export const RECEIPT_SCHEMA_VERSION = "arc-p3-receipt-v1";
export const LEARNER_VERSION = "candidate_combinator_v1";
export const DEFAULT_MASTER_SEED = 20260528;
export const DEFAULT_OUT_DIR = "results/arc/phase3-sufficiency-candidate_combinator_v1";
export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

const PHASE2_BASELINES_MANIFEST_HASH = "4968D6F6958C8D40643D09161407252B664665801FCBA3A399780ED7509623E0";
const LEARNER_ARMS = ["signature_only", "signature_palette", "metadata_only", "raw_grid_lowcap"];
const METADATA_DIM = 28;
const SIGNATURE_HASH_DIM = 4096;
const SIGNATURE_VECTOR_DIM = 4124;
const MAX_H = 30;
const MAX_W = 30;
const MASTER_SEED_NAMESPACE = "arc-p3-candidate-combinator-v1";

const D4_TRANSFORMS = [
  { name: "id", fn: (x, y) => [x, y] },
  { name: "rot90", fn: (x, y) => [y, -x] },
  { name: "rot180", fn: (x, y) => [-x, -y] },
  { name: "rot270", fn: (x, y) => [-y, x] },
  { name: "reflect_x", fn: (x, y) => [-x, y] },
  { name: "reflect_y", fn: (x, y) => [x, -y] },
  { name: "transpose", fn: (x, y) => [y, x] },
  { name: "anti_transpose", fn: (x, y) => [-y, -x] }
];

const D4_STENCIL_TRANSFORMS = [
  (grid) => grid,
  rotate90,
  rotate180,
  rotate270,
  reflectHorizontal,
  reflectVertical,
  transpose,
  antiTranspose
];

const PER_INSTANCE_COLUMNS = [
  "instance_id",
  "lane",
  "task_id",
  "primary_prior",
  "arm",
  "stratum",
  "query_index",
  "candidate_pool_size",
  "candidate_pool_size_unique",
  "suppressed_by_discrimination",
  "grid_exact_slot1",
  "grid_exact_any_slot",
  "rep_exact_slot1",
  "rep_exact_any_slot",
  "shape_exact_slot1",
  "palette_exact_slot1",
  "pixel_accuracy_slot1",
  "pixel_accuracy_best",
  "output_rep_distance_slot1",
  "output_rep_distance_best",
  "candidate_pool_contains_target_rep",
  "candidate_pool_contains_target_grid",
  "top2_contains_target_rep",
  "top2_contains_target_grid",
  "failure_label",
  "duplicate_source_pair_indices",
  "slot1_candidate_identity",
  "slot2_candidate_identity"
];

const SCORE_COLUMNS = [
  "lane",
  "arm",
  "stratum",
  "instance_count",
  "suppressed_count",
  "grid_exact_rate_any_slot",
  "rep_exact_rate_slot1",
  "rep_exact_rate_any_slot",
  "shape_exact_rate_slot1",
  "palette_exact_rate_slot1",
  "mean_pixel_accuracy_best",
  "mean_output_rep_distance_slot1",
  "mean_output_rep_distance_best",
  "mean_output_rep_similarity_best",
  "coverage_failure_rate",
  "detection_failure_rate",
  "residual_failure_rate",
  "exact_credit_rate"
];

const DISCRIMINATION_COLUMNS = [
  "task_id",
  "stratum",
  "instance_count",
  "unique_heldout_signatures",
  "majority_signature_rate",
  "collapse_count",
  "learner_task_trivial",
  "trivial_task_count"
];

export function parseArgs(argv, options = {}) {
  const parsed = {
    out: DEFAULT_OUT_DIR,
    masterSeed: DEFAULT_MASTER_SEED,
    allowDirty: false
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") {
      parsed.help = true;
    } else if (arg === "--data-dir") {
      parsed.dataDir = argv[++i];
    } else if (arg === "--register") {
      parsed.register = argv[++i];
    } else if (arg === "--out") {
      parsed.out = argv[++i];
    } else if (arg === "--master-seed") {
      parsed.masterSeed = Number(argv[++i]);
      parsed.seedOverridden = true;
    } else if (arg === "--lodo-manifest") {
      parsed.lodoManifest = argv[++i];
    } else if (arg === "--allow-dirty") {
      parsed.allowDirty = true;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!parsed.dataDir || !parsed.register) {
    parsed.missingRequired = true;
  }
  if (options.requireLodoManifest && !parsed.lodoManifest) {
    parsed.missingLodoManifest = true;
  }
  return parsed;
}

export function printUsage(tool, extra = "") {
  console.log(`Usage:
  node ${tool} --data-dir <ARC-AGI-2/data> --register docs/prereg/arc/P0_TASK_REGISTER.csv [--out ${DEFAULT_OUT_DIR}] [--master-seed 20260528] [--allow-dirty]${extra}`);
}

export async function runLodo(args, argv) {
  const startedAt = new Date().toISOString();
  const context = await prepareContext(args, { includeTestOutputs: false, tool: "scripts/arc-phase3-candidate_combinator_v1-lodo.mjs", argv });
  const discriminationRows = computeDiscriminationRows(context.tasks);
  const trivialTasks = trivialTaskSet(discriminationRows);
  const instances = buildLodoInstances(context.tasks);
  const learnerRows = evaluateInstances(instances, trivialTasks, context.masterSeed);
  const oracleRows = evaluateOracleInstances(instances, trivialTasks, context.masterSeed);
  const perInstanceRows = [...learnerRows, ...oracleRows];
  const scores = [
    ...aggregateScores(perInstanceRows),
    ...await phase0ReferenceScoreRows()
  ];
  const manifest = baseManifest(context, startedAt, "lodo");
  manifest.completedAt = new Date().toISOString();
  manifest.lanesCompleted = ["lodo"];
  manifest.lodoInstanceCount = instances.length;
  manifest.pttestInstanceCount = context.tasks.reduce((sum, task) => sum + task.test.length, 0);
  manifest.discrimination = discriminationSummary(discriminationRows);
  await writeReceipt(context.outDir, {
    manifest,
    scores,
    perInstanceRows,
    discriminationRows
  });
  return { manifest, scores, perInstanceRows, discriminationRows };
}

export async function runPttest(args, argv) {
  const lodoManifestPath = resolve(args.lodoManifest);
  const lodoManifest = JSON.parse(await readFile(lodoManifestPath, "utf8"));
  validateLodoManifest(lodoManifest);

  const startedAt = new Date().toISOString();
  const context = await prepareContext(args, { includeTestOutputs: true, tool: "scripts/arc-phase3-candidate_combinator_v1-pttest.mjs", argv });
  const discriminationRows = computeDiscriminationRows(context.tasks);
  const trivialTasks = trivialTaskSet(discriminationRows);
  const instances = buildPttestInstances(context.tasks);
  const learnerRows = evaluateInstances(instances, trivialTasks, context.masterSeed);
  const oracleRows = evaluateOracleInstances(instances, trivialTasks, context.masterSeed);
  const perInstanceRows = [...learnerRows, ...oracleRows];
  const scores = aggregateScores(perInstanceRows);

  const existingPerInstance = await readCsvIfExists(join(context.outDir, "per_instance.csv"));
  const existingScores = await readCsvIfExists(join(context.outDir, "scores.csv"));
  const existingDiscrimination = await readCsvIfExists(join(context.outDir, "discrimination.csv"));

  const manifest = {
    ...lodoManifest,
    tool: "scripts/arc-phase3-candidate_combinator_v1-pttest.mjs",
    command: context.command,
    completedAt: new Date().toISOString(),
    lanesCompleted: ["lodo", "pttest"],
    pttestStartedAt: startedAt,
    pttestInstanceCount: instances.length,
    allowDirty: context.allowDirty,
    gitCommit: context.gitCommit,
    dataDirHash: context.dataDirHash,
    registerHash: context.registerHash,
    phase2BaselinesManifestHash: context.phase2BaselinesManifestHash,
    phase2BaselinesManifestHashWarning: context.phase2BaselinesManifestHashWarning
  };

  await writeReceipt(context.outDir, {
    manifest,
    scores: [...existingScores, ...scores],
    perInstanceRows: [...existingPerInstance, ...perInstanceRows],
    discriminationRows: existingDiscrimination.length > 0 ? existingDiscrimination : discriminationRows
  });
  return { manifest, scores, perInstanceRows };
}

async function prepareContext(args, { includeTestOutputs, tool, argv }) {
  if (args.help || args.missingRequired) {
    throw new Error("Missing required --data-dir or --register argument.");
  }
  const dataDir = resolve(args.dataDir);
  const registerPath = resolve(args.register);
  const outDir = resolve(args.out ?? DEFAULT_OUT_DIR);
  assertTrainingDataDir(dataDir);
  const git = gitState(args.allowDirty);
  const tasksResult = await loadTasks(dataDir, registerPath, includeTestOutputs);
  const phase2 = await phase2BaselinesHash();
  return {
    ...tasksResult,
    dataDir,
    registerPath,
    outDir,
    masterSeed: Number(args.masterSeed ?? DEFAULT_MASTER_SEED),
    seedOverridden: Boolean(args.seedOverridden),
    allowDirty: Boolean(args.allowDirty),
    gitCommit: git.commit,
    command: ["node", tool, ...argv],
    tool,
    phase2BaselinesManifestHash: phase2.hash,
    phase2BaselinesManifestHashWarning: phase2.warning
  };
}

function assertTrainingDataDir(dataDir) {
  const normalized = dataDir.replaceAll("\\", "/").toLowerCase();
  if (normalized.endsWith("/evaluation")) {
    throw new Error("Phase 3 runners refuse to use an ARC evaluation directory as --data-dir.");
  }
}

async function loadTasks(dataDir, registerPath, includeTestOutputs) {
  const registerRaw = await readFile(registerPath, "utf8");
  const rows = parseCsv(registerRaw).filter((row) => row.status === "include" && row.split === "training");
  if (rows.length === 0) {
    throw new Error("Register has no included training rows.");
  }
  const sorted = [...rows].sort((a, b) => a.task_id.localeCompare(b.task_id));
  if (!sorted.every((row, index) => row.task_id === rows[index].task_id)) {
    throw new Error("P0_TASK_REGISTER.csv is not in task_id-ascending binding order.");
  }

  const tasks = [];
  const fileHashes = [];
  for (const row of rows) {
    const file = `training/${row.task_id}.json`;
    const raw = await readFile(join(dataDir, file), "utf8");
    fileHashes.push({ file, sha256: sha256(raw) });
    const parsed = JSON.parse(raw.replace(/^﻿/, ""));
    tasks.push({
      taskId: row.task_id,
      primaryPrior: row.primary_prior,
      row,
      train: parsed.train.map((pair, index) => ({ index, input: pair.input, output: pair.output })),
      test: parsed.test.map((pair, index) => ({
        index,
        input: pair.input,
        output: includeTestOutputs ? pair.output : undefined
      }))
    });
  }

  return {
    tasks,
    taskCount: tasks.length,
    lodoInstanceCount: tasks.reduce((sum, task) => sum + task.train.length, 0),
    pttestInstanceCount: tasks.reduce((sum, task) => sum + task.test.length, 0),
    registerHash: sha256(registerRaw),
    dataDirHash: sha256(JSON.stringify(fileHashes))
  };
}

function buildLodoInstances(tasks) {
  const instances = [];
  for (const task of tasks) {
    for (const held of task.train) {
      instances.push({
        lane: "lodo",
        instanceId: `lodo:${task.taskId}:${held.index}`,
        task,
        taskId: task.taskId,
        primaryPrior: task.primaryPrior,
        queryIndex: held.index,
        queryInput: held.input,
        targetOutput: held.output,
        conditioning: task.train.filter((pair) => pair.index !== held.index),
        stratum: task.train.length === 2 ? "k_eq_2" : "k_ge_3"
      });
    }
  }
  return instances;
}

function buildPttestInstances(tasks) {
  const instances = [];
  for (const task of tasks) {
    for (const test of task.test) {
      if (!test.output) {
        throw new Error(`Public-training test output missing for ${task.taskId}/${test.index}.`);
      }
      instances.push({
        lane: "pttest",
        instanceId: `pttest:${task.taskId}:${test.index}`,
        task,
        taskId: task.taskId,
        primaryPrior: task.primaryPrior,
        queryIndex: test.index,
        queryInput: test.input,
        targetOutput: test.output,
        conditioning: task.train,
        stratum: task.train.length === 2 ? "k_eq_2" : "k_ge_3"
      });
    }
  }
  return instances;
}

function evaluateInstances(instances, trivialTasks, masterSeed) {
  const rows = [];
  for (const instance of instances) {
    for (const arm of LEARNER_ARMS) {
      rows.push(evaluateLearnerArm(instance, arm, trivialTasks, masterSeed));
    }
  }
  return rows;
}

function evaluateLearnerArm(instance, arm, trivialTasks, masterSeed) {
  const queryRep = representGrid(instance.queryInput, arm);
  const targetRep = representGrid(instance.targetOutput, arm);
  const ranked = [];
  for (const pair of instance.conditioning) {
    const inputRep = representGrid(pair.input, arm);
    const inputDistance = armDistance(arm, queryRep, inputRep);
    for (const candidate of candidateCombinatorCandidates(instance.queryInput, pair, instance.conditioning)) {
      const outputRep = representGrid(candidate.grid, arm);
      ranked.push({
        sourcePairIndex: pair.index,
        candidateKind: candidate.kind,
        candidateRank: candidate.rank,
        inputDistance,
        tie: tieBreak(masterSeed, instance, `${arm}:${candidate.kind}`, pair.index),
        outputRep,
        grid: candidate.grid,
        identity: candidateIdentity(arm, outputRep)
      });
    }
  }
  ranked.sort((a, b) =>
    a.inputDistance - b.inputDistance ||
    a.candidateRank - b.candidateRank ||
    compareBigInt(a.tie, b.tie) ||
    a.sourcePairIndex - b.sourcePairIndex ||
    a.candidateKind.localeCompare(b.candidateKind)
  );

  const { unique, duplicateMap } = dedupeCandidates(ranked);
  const top2 = unique.slice(0, 2);
  return scoreCandidateRow(instance, arm, targetRep, unique, top2, duplicateMap, trivialTasks, {
    candidatePoolSize: ranked.length
  });
}

function candidateCombinatorCandidates(queryInput, pair, allConditioning) {
  const candidates = [];

  // rank -2: output_copy (slot-1 guard preserving nn_output_transfer_v1 baseline)
  // See "candidate_combinator_v1 Clarification: output_copy Rank" amendment.
  candidates.push({ kind: "output_copy", rank: -2, grid: cloneGrid(pair.output) });

  // rank -1: delta_overlay (slot-2 guard preserving nn_delta_transfer_v1 hits)
  // See "candidate_combinator_v1 Clarification: delta_overlay Primitive Addition".
  const deltaGrid = sameShapeDeltaOverlay(queryInput, pair.input, pair.output);
  if (deltaGrid) {
    candidates.push({ kind: "delta_overlay", rank: -1, grid: deltaGrid });
  }

  // rank 0: colormap_fit
  const colormapGrid = sameShapeBijectiveColorMap(queryInput, pair.input, pair.output);
  if (colormapGrid) {
    candidates.push({ kind: "colormap_fit", rank: 0, grid: colormapGrid });
  }

  // rank 1-7: D4 variants of output_i (d4_id is subsumed by output_copy)
  candidates.push({ kind: "d4_rot90", rank: 1, grid: rotate90(pair.output) });
  candidates.push({ kind: "d4_rot180", rank: 2, grid: rotate180(pair.output) });
  candidates.push({ kind: "d4_rot270", rank: 3, grid: rotate270(pair.output) });
  candidates.push({ kind: "d4_reflect_h", rank: 4, grid: reflectHorizontal(pair.output) });
  candidates.push({ kind: "d4_reflect_v", rank: 5, grid: reflectVertical(pair.output) });
  candidates.push({ kind: "d4_transpose", rank: 6, grid: transpose(pair.output) });
  candidates.push({ kind: "d4_anti_transpose", rank: 7, grid: antiTranspose(pair.output) });

  // rank 8+j: cell_union_<j> for each other conditioning pair j
  for (const other of allConditioning) {
    if (other.index === pair.index) continue;
    if (!sameShape(pair.output, other.output)) continue;
    candidates.push({
      kind: `cell_union_${other.index}`,
      rank: 8 + other.index,
      grid: cellUnion(pair.output, other.output)
    });
  }

  return candidates;
}

function sameShapeDeltaOverlay(queryInput, sourceInput, sourceOutput) {
  if (!sameShape(queryInput, sourceInput) || !sameShape(sourceInput, sourceOutput)) {
    return null;
  }
  const out = cloneGrid(sourceOutput);
  for (let y = 0; y < sourceInput.length; y += 1) {
    for (let x = 0; x < sourceInput[0].length; x += 1) {
      if (queryInput[y][x] !== sourceInput[y][x]) {
        out[y][x] = queryInput[y][x];
      }
    }
  }
  return out;
}

function sameShapeBijectiveColorMap(queryInput, sourceInput, sourceOutput) {
  if (!sameShape(sourceInput, sourceOutput)) return null;
  const map = new Map();
  for (let y = 0; y < sourceInput.length; y += 1) {
    for (let x = 0; x < sourceInput[0].length; x += 1) {
      const src = sourceInput[y][x];
      const dst = sourceOutput[y][x];
      if (map.has(src) && map.get(src) !== dst) {
        return null;
      }
      map.set(src, dst);
    }
  }
  // Bijective check: distinct domain entries must map to distinct codomain entries
  const seenDst = new Set();
  for (const dst of map.values()) {
    if (seenDst.has(dst)) {
      return null;
    }
    seenDst.add(dst);
  }
  // Apply M to queryInput; colors not in M's domain pass through unchanged
  return queryInput.map((row) => row.map((value) => map.has(value) ? map.get(value) : value));
}

function cellUnion(a, b) {
  const h = a.length;
  const w = a[0].length;
  const out = Array.from({ length: h }, () => Array(w).fill(0));
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      out[y][x] = a[y][x] !== 0 ? a[y][x] : b[y][x];
    }
  }
  return out;
}

function evaluateOracleInstances(instances, trivialTasks, masterSeed) {
  const rows = [];
  for (const instance of instances) {
    for (const baseline of ARC_PHASE0_BASELINE_NAMES) {
      const pseudoTask = {
        train: instance.conditioning.map((pair) => ({ input: pair.input, output: pair.output })),
        test: [{ input: instance.queryInput, output: instance.targetOutput }]
      };
      const taskRecord = { row: { task_id: instance.taskId }, task: pseudoTask };
      const predictions = arcPhase0BaselinePredictions(baseline, taskRecord, masterSeed)[0] ?? [];
      const ranked = predictions.slice(0, 2).map((grid, index) => ({
        sourcePairIndex: index,
        inputDistance: index,
        tie: BigInt(index),
        outputRep: representGrid(grid, "raw_grid_lowcap"),
        grid,
        identity: JSON.stringify(grid)
      }));
      const { unique, duplicateMap } = dedupeCandidates(ranked);
      rows.push(scoreCandidateRow(
        instance,
        `oracle_copy_floor_lodo_rerun:${baseline}`,
        representGrid(instance.targetOutput, "raw_grid_lowcap"),
        unique,
        unique.slice(0, 2),
        duplicateMap,
        trivialTasks,
        { oracle: true }
      ));
    }
  }
  return rows;
}

function scoreCandidateRow(instance, arm, targetRep, fullPool, top2, duplicateMap, trivialTasks, options = {}) {
  const directArm = arm.startsWith("oracle_copy_floor") ? "raw_grid_lowcap" : arm;
  const gridScorable = directArm === "signature_palette" || directArm === "raw_grid_lowcap";
  const repScorable = !options.oracle;
  const decoded = top2.map((candidate) => decodeCandidate(directArm, candidate.outputRep, candidate.grid));
  const fullDecoded = fullPool.map((candidate) => decodeCandidate(directArm, candidate.outputRep, candidate.grid));
  const targetGrid = instance.targetOutput;

  const slot1 = top2[0];
  const slot2 = top2[1];
  const slot1Decoded = decoded[0];
  const targetIdentity = candidateIdentity(directArm, targetRep);

  const repExactSlot1 = repScorable && slot1 ? slot1.identity === targetIdentity : null;
  const repExactAny = repScorable ? top2.some((candidate) => candidate.identity === targetIdentity) : null;
  const poolContainsTargetRep = repScorable ? fullPool.some((candidate) => candidate.identity === targetIdentity) : null;
  const top2ContainsTargetRep = repScorable ? top2.some((candidate) => candidate.identity === targetIdentity) : null;

  const gridExactSlot1 = gridScorable && slot1Decoded?.grid ? equalsGrid(slot1Decoded.grid, targetGrid) : null;
  const gridExactAny = gridScorable ? decoded.some((item) => item.grid && equalsGrid(item.grid, targetGrid)) : null;
  const poolContainsTargetGrid = gridScorable ? fullDecoded.some((item) => item.grid && equalsGrid(item.grid, targetGrid)) : null;
  const top2ContainsTargetGrid = gridScorable ? decoded.some((item) => item.grid && equalsGrid(item.grid, targetGrid)) : null;

  const shapeExactSlot1 = slot1 ? sameShapeOfRep(directArm, slot1.outputRep, targetRep) : null;
  const paletteExactSlot1 = exposesPalette(directArm) && slot1 ? samePalette(slot1.outputRep, targetRep) : null;
  const pixelAccuracySlot1 = gridScorable && slot1Decoded?.grid ? pixelAccuracy(slot1Decoded.grid, targetGrid) : null;
  const pixelAccuracyBest = gridScorable
    ? maxNumeric(decoded.map((item) => item.grid ? pixelAccuracy(item.grid, targetGrid) : null))
    : null;
  const outputRepDistanceSlot1 = slot1 ? armDistance(directArm, slot1.outputRep, targetRep) : null;
  const outputRepDistanceBest = maxNumeric(top2.map((candidate) =>
    1 - armDistance(directArm, candidate.outputRep, targetRep)
  ));
  const distanceBest = outputRepDistanceBest == null ? null : 1 - outputRepDistanceBest;

  const failureLabel = assignFailureLabel({
    arm: directArm,
    gridScorable,
    repScorable,
    gridExactAny,
    repExactSlot1,
    poolContainsTargetRep,
    poolContainsTargetGrid,
    top2ContainsTargetRep,
    decoderFailed: decoded.some((item) => item.coverageFailure),
    top2Length: top2.length
  });

  return {
    instance_id: instance.instanceId,
    lane: instance.lane,
    task_id: instance.taskId,
    primary_prior: instance.primaryPrior,
    arm,
    stratum: instance.stratum,
    query_index: instance.queryIndex,
    candidate_pool_size: options.candidatePoolSize ?? instance.conditioning?.length ?? fullPool.length,
    candidate_pool_size_unique: fullPool.length,
    suppressed_by_discrimination: trivialTasks.has(instance.taskId),
    grid_exact_slot1: boolOrNa(gridExactSlot1),
    grid_exact_any_slot: boolOrNa(gridExactAny),
    rep_exact_slot1: boolOrNa(repExactSlot1),
    rep_exact_any_slot: boolOrNa(repExactAny),
    shape_exact_slot1: boolOrNa(shapeExactSlot1),
    palette_exact_slot1: boolOrNa(paletteExactSlot1),
    pixel_accuracy_slot1: numberOrNa(pixelAccuracySlot1),
    pixel_accuracy_best: numberOrNa(pixelAccuracyBest),
    output_rep_distance_slot1: numberOrNa(outputRepDistanceSlot1),
    output_rep_distance_best: numberOrNa(distanceBest),
    candidate_pool_contains_target_rep: boolOrNa(poolContainsTargetRep),
    candidate_pool_contains_target_grid: boolOrNa(poolContainsTargetGrid),
    top2_contains_target_rep: boolOrNa(top2ContainsTargetRep),
    top2_contains_target_grid: boolOrNa(top2ContainsTargetGrid),
    failure_label: failureLabel,
    duplicate_source_pair_indices: [...duplicateMap.values()].flat().join(";"),
    slot1_candidate_identity: slot1?.identity ?? "",
    slot2_candidate_identity: slot2?.identity ?? ""
  };
}

function assignFailureLabel(state) {
  if (state.gridScorable) {
    if (state.gridExactAny) return "none";
    if (state.arm === "signature_palette") {
      if (state.decoderFailed || !state.poolContainsTargetRep) return "coverage";
      if (!state.top2ContainsTargetRep) return "detection";
      return "residual";
    }
    if (!state.poolContainsTargetGrid) return "coverage";
    return "detection";
  }
  if (state.repScorable) {
    if (state.repExactSlot1) return "none";
    if (!state.poolContainsTargetRep || state.top2Length === 0) return "coverage";
    return "detection";
  }
  return "coverage";
}

function dedupeCandidates(ranked) {
  const seen = new Map();
  const unique = [];
  const duplicateMap = new Map();
  for (const candidate of ranked) {
    if (!seen.has(candidate.identity)) {
      seen.set(candidate.identity, candidate);
      unique.push(candidate);
    } else {
      const existing = seen.get(candidate.identity);
      const list = duplicateMap.get(existing.sourcePairIndex) ?? [];
      list.push(candidate.sourcePairIndex);
      duplicateMap.set(existing.sourcePairIndex, list);
    }
  }
  return { unique, duplicateMap };
}

function computeDiscriminationRows(tasks) {
  const taskRows = [];
  for (const task of tasks) {
    const signatures = task.train.map((pair) => projectGridShadow(pair.output).canonicalObjectSignature);
    const counts = countBy(signatures);
    const unique = Object.keys(counts).length;
    const maxCount = Math.max(...Object.values(counts));
    const seen = new Set();
    let collapseCount = 0;
    for (const sig of signatures) {
      if (seen.has(sig)) collapseCount += 1;
      seen.add(sig);
    }
    taskRows.push({
      task_id: task.taskId,
      stratum: "task",
      instance_count: signatures.length,
      unique_heldout_signatures: unique,
      majority_signature_rate: round(maxCount / signatures.length, 9),
      collapse_count: collapseCount,
      learner_task_trivial: unique === 1,
      trivial_task_count: unique === 1 ? 1 : 0,
      taskStratum: task.train.length === 2 ? "k_eq_2" : "k_ge_3"
    });
  }
  return [
    ...taskRows.map(({ taskStratum, ...row }) => row),
    aggregateDiscrimination(taskRows, "all_tasks"),
    aggregateDiscrimination(taskRows.filter((row) => row.taskStratum === "k_ge_3"), "k_ge_3"),
    aggregateDiscrimination(taskRows.filter((row) => row.taskStratum === "k_eq_2"), "k_eq_2")
  ];
}

function aggregateDiscrimination(rows, stratum) {
  const instanceCount = sum(rows.map((row) => row.instance_count));
  const trivialCount = rows.filter((row) => row.learner_task_trivial).length;
  return {
    task_id: "__aggregate__",
    stratum,
    instance_count: instanceCount,
    unique_heldout_signatures: sum(rows.map((row) => row.unique_heldout_signatures)),
    majority_signature_rate: instanceCount === 0 ? "NA" : round(sum(rows.map((row) => row.majority_signature_rate * row.instance_count)) / instanceCount, 9),
    collapse_count: sum(rows.map((row) => row.collapse_count)),
    learner_task_trivial: rows.length > 0 && trivialCount / rows.length > 0.3,
    trivial_task_count: trivialCount
  };
}

function trivialTaskSet(discriminationRows) {
  return new Set(
    discriminationRows
      .filter((row) => row.stratum === "task" && String(row.learner_task_trivial) === "true")
      .map((row) => row.task_id)
  );
}

function discriminationSummary(discriminationRows) {
  const aggregate = discriminationRows.find((row) => row.task_id === "__aggregate__" && row.stratum === "all_tasks");
  return {
    trivialTaskCount: Number(aggregate?.trivial_task_count ?? 0),
    learnerTaskTrivialThresholdFired: String(aggregate?.learner_task_trivial) === "true"
  };
}

function aggregateScores(rows) {
  const out = [];
  const arms = [...new Set(rows.map((row) => row.arm))];
  for (const lane of [...new Set(rows.map((row) => row.lane))]) {
    for (const arm of arms) {
      const armRows = rows.filter((row) => row.lane === lane && row.arm === arm);
      if (armRows.length === 0) continue;
      for (const stratum of ["all_tasks", "k_ge_3", "k_eq_2"]) {
        const stratumRows = stratum === "all_tasks" ? armRows : armRows.filter((row) => row.stratum === stratum);
        if (stratumRows.length === 0) continue;
        out.push(scoreAggregate(lane, arm, stratum, stratumRows));
      }
    }
  }
  return out;
}

function scoreAggregate(lane, arm, stratum, rows) {
  const active = rows.filter((row) => String(row.suppressed_by_discrimination) !== "true");
  const denom = active.length;
  const exactCount = active.filter((row) => row.failure_label === "none").length;
  const coverageCount = active.filter((row) => row.failure_label === "coverage").length;
  const detectionCount = active.filter((row) => row.failure_label === "detection").length;
  const residualCount = active.filter((row) => row.failure_label === "residual").length;
  if (denom > 0) {
    const invariant = (exactCount + coverageCount + detectionCount + residualCount) / denom;
    if (Math.abs(invariant - 1) > 1e-9) {
      throw new Error(`Failure-label invariant failed for ${lane}/${arm}/${stratum}: ${invariant}`);
    }
  }
  const exactCreditRate = denom === 0 ? "NA" : round(exactCount / denom, 9);
  const coverageRate = denom === 0 ? "NA" : round(coverageCount / denom, 9);
  const detectionRate = denom === 0 ? "NA" : round(detectionCount / denom, 9);
  const residualRate = denom === 0 ? "NA" : round(residualCount / denom, 9);
  const meanDistBest = meanField(active, "output_rep_distance_best");
  return {
    lane,
    arm,
    stratum,
    instance_count: rows.length,
    suppressed_count: rows.length - active.length,
    grid_exact_rate_any_slot: boolRate(active, "grid_exact_any_slot"),
    rep_exact_rate_slot1: boolRate(active, "rep_exact_slot1"),
    rep_exact_rate_any_slot: boolRate(active, "rep_exact_any_slot"),
    shape_exact_rate_slot1: boolRate(active, "shape_exact_slot1"),
    palette_exact_rate_slot1: boolRate(active, "palette_exact_slot1"),
    mean_pixel_accuracy_best: meanField(active, "pixel_accuracy_best"),
    mean_output_rep_distance_slot1: meanField(active, "output_rep_distance_slot1"),
    mean_output_rep_distance_best: meanDistBest,
    mean_output_rep_similarity_best: meanDistBest === "NA" ? "NA" : round(1 - meanDistBest, 9),
    coverage_failure_rate: coverageRate,
    detection_failure_rate: detectionRate,
    residual_failure_rate: residualRate,
    exact_credit_rate: exactCreditRate
  };
}

async function phase0ReferenceScoreRows() {
  const path = join(ROOT, "results/arc/phase0-baselines/summary.csv");
  try {
    const rows = parseCsv(await readFile(path, "utf8"));
    return rows.map((row) => ({
      lane: "phase0_reference",
      arm: `oracle_copy_floor_phase0_reference:${row.baseline}`,
      stratum: "all_tasks",
      instance_count: row.tasks,
      suppressed_count: 0,
      grid_exact_rate_any_slot: row.task_exact_rate,
      rep_exact_rate_slot1: "NA",
      rep_exact_rate_any_slot: "NA",
      shape_exact_rate_slot1: "NA",
      palette_exact_rate_slot1: "NA",
      mean_pixel_accuracy_best: row.mean_pixel_accuracy,
      mean_output_rep_distance_slot1: "NA",
      mean_output_rep_distance_best: "NA",
      mean_output_rep_similarity_best: "NA",
      coverage_failure_rate: "NA",
      detection_failure_rate: "NA",
      residual_failure_rate: "NA",
      exact_credit_rate: row.task_exact_rate
    }));
  } catch (err) {
    if (err.code === "ENOENT") return [];
    throw err;
  }
}

function representGrid(grid, arm) {
  const projection = projectGridShadow(grid);
  const metadata = metadataVector(grid, projection);
  const suffix = signatureSuffix(projection);
  const rawLabels = rawLabelField(grid);
  return {
    arm,
    grid,
    shape: projection.shape,
    shapeLabel: projection.shape.join("x"),
    palette: projection.palette,
    paletteLabel: projection.palette.join(""),
    nonzeroPalette: projection.nonzeroPalette,
    nonzeroCells: projection.nonZeroCells,
    nonzeroComponents: projection.nonZeroComponents,
    density: projection.density,
    canonicalObjectSignature: projection.canonicalObjectSignature,
    localSignatureBag: projection.localSignatureBag,
    signatureHash: sha256(projection.canonicalObjectSignature),
    localBagHash: sha256(JSON.stringify(projection.localSignatureBag)),
    metadata,
    suffix,
    rawLabels
  };
}

function candidateIdentity(arm, rep) {
  if (arm === "signature_only") return `${rep.signatureHash}|${rep.localBagHash}`;
  if (arm === "signature_palette") {
    return `${rep.shapeLabel}|${rep.paletteLabel}|${rep.nonzeroCells}|${rep.nonzeroComponents}|${rep.density}|${rep.signatureHash}|${rep.localBagHash}`;
  }
  if (arm === "metadata_only") return JSON.stringify(rep.metadata);
  if (arm === "raw_grid_lowcap") return JSON.stringify(rep.grid);
  return JSON.stringify(rep.grid);
}

function armDistance(arm, a, b) {
  if (arm === "signature_only") return signatureCosineDistance(a.suffix, b.suffix);
  if (arm === "signature_palette") return 0.5 * signatureCosineDistance(a.suffix, b.suffix) + 0.5 * meanAbs(a.metadata, b.metadata);
  if (arm === "metadata_only") return meanAbs(a.metadata, b.metadata);
  if (arm === "raw_grid_lowcap") return rawGridDistance(a.rawLabels, b.rawLabels);
  return rawGridDistance(a.rawLabels, b.rawLabels);
}

function decodeCandidate(arm, rep, sourceGrid) {
  if (arm === "raw_grid_lowcap") return { grid: cloneGrid(sourceGrid), coverageFailure: false };
  if (arm === "signature_palette") return topLeftPaletteOrbit(rep);
  return { grid: null, coverageFailure: true };
}

function topLeftPaletteOrbit(rep) {
  const [height, width] = rep.shape;
  if (rep.canonicalObjectSignature === "empty") {
    return { grid: Array.from({ length: height }, () => Array(width).fill(0)), coverageFailure: false };
  }
  const [bbox, cellsText] = rep.canonicalObjectSignature.split("|");
  const [bboxW, bboxH] = bbox.split("x").map(Number);
  if (!Number.isFinite(bboxW) || !Number.isFinite(bboxH) || bboxW > width || bboxH > height) {
    return { grid: null, coverageFailure: true };
  }
  const cells = cellsText.split(";").filter(Boolean).map((token) => {
    const [x, y, role] = token.split(":").map(Number);
    return { x, y, role };
  });
  const roles = [...new Set(cells.map((cell) => cell.role))].sort((a, b) => a - b);
  if (roles.length !== rep.nonzeroPalette.length) {
    return { grid: null, coverageFailure: true };
  }
  const roleToColor = new Map(roles.map((role, index) => [role, rep.nonzeroPalette[index]]));
  const out = Array.from({ length: height }, () => Array(width).fill(0));
  for (const cell of cells) {
    if (cell.y < 0 || cell.y >= height || cell.x < 0 || cell.x >= width) {
      return { grid: null, coverageFailure: true };
    }
    out[cell.y][cell.x] = roleToColor.get(cell.role);
  }
  return { grid: out, coverageFailure: false };
}

function projectGridShadow(grid) {
  const nonZero = nonZeroCells(grid);
  const localBag = nonZero
    .map((cell) => canonicalStencil(grid, cell.x, cell.y, 1))
    .sort();
  const palette = [...new Set(grid.flat())].sort((a, b) => a - b);
  return {
    shape: [grid.length, grid[0].length],
    palette,
    nonzeroPalette: palette.filter((value) => value !== 0),
    nonZeroCells: nonZero.length,
    nonZeroComponents: countComponents(grid),
    density: round(nonZero.length / (grid.length * grid[0].length), 6),
    localSignatureBag: localBag,
    canonicalObjectSignature: canonicalObjectSignature(grid)
  };
}

function canonicalObjectSignature(grid) {
  const variants = objectVariants(grid);
  if (variants.length === 0) return "empty";
  return variants.map((variant) => variant.signature).sort()[0];
}

function objectVariants(grid) {
  const cells = nonZeroCells(grid);
  if (cells.length === 0) return [];
  return D4_TRANSFORMS.map((transform) => {
    const transformed = cells.map((cell) => {
      const [x, y] = transform.fn(cell.x, cell.y);
      return { x, y, color: cell.color };
    });
    const minX = Math.min(...transformed.map((cell) => cell.x));
    const minY = Math.min(...transformed.map((cell) => cell.y));
    const normalized = transformed
      .map((cell) => ({ x: cell.x - minX, y: cell.y - minY, color: cell.color }))
      .sort((a, b) => a.y - b.y || a.x - b.x || a.color - b.color);
    const roleMap = new Map();
    let nextRole = 1;
    const tokens = normalized.map((cell) => {
      if (!roleMap.has(cell.color)) {
        roleMap.set(cell.color, nextRole);
        nextRole += 1;
      }
      return `${cell.x}:${cell.y}:${roleMap.get(cell.color)}`;
    });
    const width = Math.max(...normalized.map((cell) => cell.x)) + 1;
    const height = Math.max(...normalized.map((cell) => cell.y)) + 1;
    return { signature: `${width}x${height}|${tokens.join(";")}` };
  });
}

function canonicalStencil(grid, cx, cy, radius) {
  const cells = [];
  for (let y = cy - radius; y <= cy + radius; y += 1) {
    const row = [];
    for (let x = cx - radius; x <= cx + radius; x += 1) {
      row.push(y < 0 || y >= grid.length || x < 0 || x >= grid[0].length ? 0 : grid[y][x]);
    }
    cells.push(row);
  }
  return D4_STENCIL_TRANSFORMS.map((transform) => roleNormalizeGrid(transform(cells))).sort()[0];
}

function roleNormalizeGrid(grid) {
  const roleMap = new Map([[0, 0]]);
  let nextRole = 1;
  const tokens = [];
  for (const row of grid) {
    for (const value of row) {
      if (!roleMap.has(value)) {
        roleMap.set(value, nextRole);
        nextRole += 1;
      }
      tokens.push(roleMap.get(value));
    }
  }
  return tokens.join("");
}

function metadataVector(grid, projection) {
  const height = grid.length;
  const width = grid[0].length;
  const flat = grid.flat();
  const counts = Array(10).fill(0);
  for (const value of flat) counts[value] += 1;
  return [
    height / 30,
    width / 30,
    (height * width) / 900,
    projection.palette.length / 10,
    projection.nonzeroPalette.length / 9,
    projection.nonZeroCells / 900,
    projection.density,
    projection.nonZeroComponents / 900,
    ...Array.from({ length: 10 }, (_, color) => projection.palette.includes(color) ? 1 : 0),
    ...counts.map((count) => count / (height * width))
  ].map((value) => round(value, 9));
}

function signatureSuffix(projection) {
  const weights = new Map();
  const objectTokens = objectTokensFor(projection.canonicalObjectSignature);
  for (const token of objectTokens) {
    addHashed(weights, "object", token, 1 / objectTokens.length);
  }
  const bagCounts = countBy(projection.localSignatureBag);
  const bagDenom = Math.max(1, projection.localSignatureBag.length);
  for (const [stencil, count] of Object.entries(bagCounts)) {
    addHashed(weights, "bag", `bag:stencil=${stencil}`, count / bagDenom);
  }
  const norm = Math.sqrt([...weights.values()].reduce((sumValue, value) => sumValue + value * value, 0));
  if (norm > 0) {
    for (const [index, value] of weights) {
      weights.set(index, round(value / norm, 9));
    }
  }
  return weights;
}

function objectTokensFor(signature) {
  if (signature === "empty") return ["obj:empty"];
  const [bbox, cellsText] = signature.split("|");
  const [bboxW, bboxH] = bbox.split("x");
  const cells = cellsText ? cellsText.split(";").filter(Boolean) : [];
  const roles = new Set(cells.map((token) => token.split(":")[2]));
  return [
    `obj:bbox_w=${bboxW}`,
    `obj:bbox_h=${bboxH}`,
    `obj:role_count=${roles.size}`,
    `obj:cell_count=${cells.length}`,
    ...cells.map((cell) => `obj:cell=${cell}`)
  ];
}

function addHashed(weights, namespace, token, value) {
  const digest = createHash("sha256").update(`${FEATURE_SCHEMA_VERSION}\0${namespace}\0${token}`).digest();
  const bucket = METADATA_DIM + (digest.readUInt32BE(0) % SIGNATURE_HASH_DIM);
  weights.set(bucket, (weights.get(bucket) ?? 0) + value);
}

function rawLabelField(grid) {
  const labels = [];
  for (let y = 0; y < MAX_H; y += 1) {
    for (let x = 0; x < MAX_W; x += 1) {
      labels.push(y < grid.length && x < grid[0].length ? grid[y][x] : 10);
    }
  }
  return labels;
}

function signatureCosineDistance(a, b) {
  let dot = 0;
  const smaller = a.size < b.size ? a : b;
  const larger = a.size < b.size ? b : a;
  for (const [index, value] of smaller) {
    dot += value * (larger.get(index) ?? 0);
  }
  return Math.max(0, Math.min(1, round(1 - dot, 9)));
}

function meanAbs(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) total += Math.abs(a[i] - b[i]);
  return round(total / a.length, 9);
}

function rawGridDistance(a, b) {
  let diff = 0;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) diff += 1;
  }
  return round(diff / a.length, 9);
}

async function writeReceipt(outDir, { manifest, scores, perInstanceRows, discriminationRows }) {
  await mkdir(outDir, { recursive: true });
  await mkdir(join(outDir, "lodo"), { recursive: true });
  await mkdir(join(outDir, "pttest"), { recursive: true });
  await writeFile(join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  await writeFile(join(outDir, "scores.csv"), toCsv(scores, SCORE_COLUMNS));
  await writeFile(join(outDir, "per_instance.csv"), toCsv(perInstanceRows, PER_INSTANCE_COLUMNS));
  await writeFile(join(outDir, "discrimination.csv"), toCsv(discriminationRows, DISCRIMINATION_COLUMNS));
  const hashes = await hashReceiptFiles(outDir);
  await writeFile(join(outDir, "hashes.json"), `${JSON.stringify(hashes, null, 2)}\n`);
}

function baseManifest(context, startedAt, lane) {
  return {
    generatedAt: startedAt,
    completedAt: null,
    tool: context.tool,
    command: context.command,
    gitCommit: context.gitCommit,
    allowDirty: context.allowDirty,
    dataDir: context.dataDir,
    registerPath: context.registerPath,
    outDir: context.outDir,
    masterSeed: context.masterSeed,
    seedOverridden: context.seedOverridden,
    featureSchemaVersion: FEATURE_SCHEMA_VERSION,
    protocolVersion: PROTOCOL_VERSION,
    receiptSchemaVersion: RECEIPT_SCHEMA_VERSION,
    learnerVersion: LEARNER_VERSION,
    lane,
    taskCount: context.taskCount,
    lodoInstanceCount: context.lodoInstanceCount,
    pttestInstanceCount: context.pttestInstanceCount,
    dataDirHash: context.dataDirHash,
    registerHash: context.registerHash,
    phase2BaselinesManifestHash: context.phase2BaselinesManifestHash,
    phase2BaselinesManifestHashWarning: context.phase2BaselinesManifestHashWarning,
    platform: process.platform,
    nodeVersion: process.version,
    dependencies: {}
  };
}

function validateLodoManifest(manifest) {
  const checks = [
    ["featureSchemaVersion", FEATURE_SCHEMA_VERSION],
    ["protocolVersion", PROTOCOL_VERSION],
    ["receiptSchemaVersion", RECEIPT_SCHEMA_VERSION],
    ["learnerVersion", LEARNER_VERSION]
  ];
  for (const [field, expected] of checks) {
    if (manifest[field] !== expected) {
      throw new Error(`LODO manifest ${field}=${manifest[field]} does not match ${expected}.`);
    }
  }
  if (!Array.isArray(manifest.lanesCompleted) || !manifest.lanesCompleted.includes("lodo")) {
    throw new Error("LODO manifest does not record a completed lodo lane.");
  }
}

async function hashReceiptFiles(outDir) {
  const files = [];
  for await (const file of walk(outDir)) {
    if (file.endsWith("hashes.json")) continue;
    const rel = relative(outDir, file).replaceAll("\\", "/");
    files.push(rel);
  }
  files.sort();
  const hashes = {};
  for (const rel of files) {
    hashes[rel] = sha256(await readFile(join(outDir, rel)));
  }
  return hashes;
}

async function* walk(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      yield* walk(full);
    } else {
      yield full;
    }
  }
}

async function readCsvIfExists(path) {
  try {
    return parseCsv(await readFile(path, "utf8"));
  } catch (err) {
    if (err.code === "ENOENT") return [];
    throw err;
  }
}

async function phase2BaselinesHash() {
  const path = join(ROOT, "results/arc/phase2-baselines/manifest.json");
  try {
    const hash = sha256(await readFile(path));
    return {
      hash,
      warning: hash === PHASE2_BASELINES_MANIFEST_HASH ? "" : `expected ${PHASE2_BASELINES_MANIFEST_HASH}`
    };
  } catch (err) {
    if (err.code === "ENOENT") {
      return { hash: "", warning: "results/arc/phase2-baselines/manifest.json missing" };
    }
    throw err;
  }
}

function gitState(allowDirty) {
  const commit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: ROOT, encoding: "utf8" }).trim().toUpperCase();
  const dirty = execFileSync("git", ["status", "--porcelain", "--untracked-files=no"], { cwd: ROOT, encoding: "utf8" }).trim();
  if (dirty && !allowDirty) {
    throw new Error("Refusing to run on a dirty worktree; commit the freeze marker first or pass --allow-dirty for smoke checks.");
  }
  return { commit };
}

function nonZeroCells(grid) {
  const cells = [];
  for (let y = 0; y < grid.length; y += 1) {
    for (let x = 0; x < grid[0].length; x += 1) {
      if (grid[y][x] !== 0) cells.push({ x, y, color: grid[y][x] });
    }
  }
  return cells;
}

function countComponents(grid) {
  const seen = Array.from({ length: grid.length }, () => Array(grid[0].length).fill(false));
  let count = 0;
  for (let y = 0; y < grid.length; y += 1) {
    for (let x = 0; x < grid[0].length; x += 1) {
      if (grid[y][x] === 0 || seen[y][x]) continue;
      count += 1;
      const stack = [[x, y]];
      seen[y][x] = true;
      while (stack.length > 0) {
        const [cx, cy] = stack.pop();
        for (const [nx, ny] of [[cx + 1, cy], [cx - 1, cy], [cx, cy + 1], [cx, cy - 1]]) {
          if (ny < 0 || ny >= grid.length || nx < 0 || nx >= grid[0].length || seen[ny][nx] || grid[ny][nx] === 0) continue;
          seen[ny][nx] = true;
          stack.push([nx, ny]);
        }
      }
    }
  }
  return count;
}

function rotate90(grid) {
  const h = grid.length;
  const w = grid[0].length;
  return Array.from({ length: w }, (_, y) => Array.from({ length: h }, (_, x) => grid[h - 1 - x][y]));
}

function rotate180(grid) {
  return reflectVertical(reflectHorizontal(grid));
}

function rotate270(grid) {
  return rotate90(rotate180(grid));
}

function reflectHorizontal(grid) {
  return grid.map((row) => [...row].reverse());
}

function reflectVertical(grid) {
  return [...grid].reverse().map((row) => [...row]);
}

function transpose(grid) {
  return Array.from({ length: grid[0].length }, (_, y) => Array.from({ length: grid.length }, (_, x) => grid[x][y]));
}

function antiTranspose(grid) {
  return reflectHorizontal(reflectVertical(transpose(grid)));
}

function equalsGrid(a, b) {
  return sameShape(a, b) && a.every((row, y) => row.every((value, x) => value === b[y][x]));
}

function sameShape(a, b) {
  return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a[0]?.length === b[0]?.length;
}

function pixelAccuracy(a, b) {
  if (!sameShape(a, b)) return 0;
  let correct = 0;
  let total = 0;
  for (let y = 0; y < b.length; y += 1) {
    for (let x = 0; x < b[0].length; x += 1) {
      total += 1;
      if (a[y][x] === b[y][x]) correct += 1;
    }
  }
  return correct / total;
}

function sameShapeOfRep(arm, a, b) {
  if (arm === "signature_only") return null;
  return a.shape[0] === b.shape[0] && a.shape[1] === b.shape[1];
}

function exposesPalette(arm) {
  return arm === "signature_palette" || arm === "metadata_only" || arm === "raw_grid_lowcap";
}

function samePalette(a, b) {
  return a.paletteLabel === b.paletteLabel;
}

function cloneGrid(grid) {
  return grid.map((row) => [...row]);
}

function tieBreak(masterSeed, instance, arm, sourcePairIndex) {
  const text = `${MASTER_SEED_NAMESPACE}\0${masterSeed}\0${instance.taskId}\0${instance.lane}\0${instance.queryIndex}\0${arm}\0${sourcePairIndex}`;
  return createHash("sha256").update(text).digest().readBigUInt64BE(0);
}

function compareBigInt(a, b) {
  return a < b ? -1 : a > b ? 1 : 0;
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex").toUpperCase();
}

function boolOrNa(value) {
  return value === null || value === undefined ? "NA" : Boolean(value);
}

function numberOrNa(value) {
  return value === null || value === undefined || Number.isNaN(value) ? "NA" : round(value, 9);
}

function boolRate(rows, field) {
  const comparable = rows.filter((row) => row[field] !== "NA");
  if (comparable.length === 0) return "NA";
  return round(comparable.filter((row) => String(row[field]) === "true").length / comparable.length, 9);
}

function meanField(rows, field) {
  const values = rows.map((row) => Number(row[field])).filter((value) => Number.isFinite(value));
  if (values.length === 0) return "NA";
  return round(sum(values) / values.length, 9);
}

function rate(rows, predicate) {
  if (rows.length === 0) return "NA";
  return round(rows.filter(predicate).length / rows.length, 9);
}

function maxNumeric(values) {
  const nums = values.filter((value) => value !== null && value !== undefined && Number.isFinite(value));
  return nums.length === 0 ? null : Math.max(...nums);
}

function sum(values) {
  return values.reduce((total, value) => total + Number(value), 0);
}

function countBy(values) {
  const counts = {};
  for (const value of values) counts[value] = (counts[value] ?? 0) + 1;
  return counts;
}

function round(value, places) {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
}

function toCsv(rows, columns) {
  return `${[columns.join(","), ...rows.map((row) => columns.map((column) => csvCell(row[column])).join(","))].join("\n")}\n`;
}

function csvCell(value) {
  const text = String(value ?? "");
  if (/[",\n]/.test(text)) return `"${text.replaceAll("\"", "\"\"")}"`;
  return text;
}

function parseCsv(text) {
  const lines = text.replace(/\r\n/g, "\n").split("\n").filter((line) => line.length > 0);
  if (lines.length === 0) return [];
  const header = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const cells = parseCsvLine(line);
    return Object.fromEntries(header.map((column, index) => [column, cells[index] ?? ""]));
  });
}

function parseCsvLine(line) {
  const cells = [];
  let cell = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === "\"" && line[i + 1] === "\"") {
        cell += "\"";
        i += 1;
      } else if (ch === "\"") {
        inQuotes = false;
      } else {
        cell += ch;
      }
    } else if (ch === "\"") {
      inQuotes = true;
    } else if (ch === ",") {
      cells.push(cell);
      cell = "";
    } else {
      cell += ch;
    }
  }
  cells.push(cell);
  return cells;
}
