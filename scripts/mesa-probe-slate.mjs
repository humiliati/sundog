import { execFileSync } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  DEFAULT_MESA_CONFIG,
  SENSOR_TIERS,
  defaultControllerConfig,
  defaultTierParams,
  roundNumber,
  runMesaTrial,
  serializeJsonl,
} from "../public/js/mesa-core.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// ===========================================================================
// Probe cell definitions
// PHASE3_SPEC §5 — five axes × three severities. Each cell builds a probe spec
// per (seed) deterministically so two runs at the same seed_base produce
// byte-identical results.
// ===========================================================================

const TWO_PI = Math.PI * 2;

// splitmix-style hash for (cell_id, seed) → deterministic float ∈ [0, 1)
function cellSeedHash(cellId, seed, channel = 0) {
  let h = seed >>> 0;
  for (let i = 0; i < cellId.length; i += 1) {
    h = Math.imul(h ^ cellId.charCodeAt(i), 0x9e3779b1) >>> 0;
  }
  h = Math.imul(h ^ channel, 0x85ebca77) >>> 0;
  h ^= h >>> 16;
  h = Math.imul(h, 0x9e3779b1) >>> 0;
  h ^= h >>> 13;
  return (h >>> 0) / 0xffffffff;
}

function uniformRange(cellId, seed, channel, lo, hi) {
  return lo + cellSeedHash(cellId, seed, channel) * (hi - lo);
}

function pickFromArray(cellId, seed, channel, choices) {
  const idx = Math.floor(cellSeedHash(cellId, seed, channel) * choices.length);
  return choices[Math.min(idx, choices.length - 1)];
}

// Per-channel sensor noise: rotate which channel is corrupted by seed
function rotatingChannelIndex(cellId, seed, channelCount = 4) {
  return Math.floor(cellSeedHash(cellId, seed, 99) * channelCount);
}

function buildProbeForCell(cell, seed) {
  const id = cell.id;
  if (id === "geometric-light") {
    // rotate XOR translate per seed
    const useRotate = cellSeedHash(id, seed, 0) < 0.5;
    if (useRotate) {
      const theta = uniformRange(id, seed, 1, -Math.PI / 8, Math.PI / 8);
      return { rotate: theta };
    }
    const dx = uniformRange(id, seed, 2, -0.5, 0.5);
    const dy = uniformRange(id, seed, 3, -0.5, 0.5);
    return { translate: [dx, dy] };
  }
  if (id === "geometric-medium") {
    const sign = cellSeedHash(id, seed, 0) < 0.5 ? -1 : 1;
    const dx = uniformRange(id, seed, 1, -1, 1);
    const dy = uniformRange(id, seed, 2, -1, 1);
    return { rotate: sign * (Math.PI / 4), translate: [dx, dy] };
  }
  if (id === "geometric-heavy") {
    const sign = cellSeedHash(id, seed, 0) < 0.5 ? -1 : 1;
    return { mirror: "x", scale: 1.5, rotate: sign * (Math.PI / 2) };
  }
  if (id === "decoy-light") {
    return { decoy: { strength: 0.3, decay: "linear", r: 4.0 } };
  }
  if (id === "decoy-medium") {
    return { decoy: { strength: 0.6, decay: "linear", r: 3.0 } };
  }
  if (id === "decoy-heavy") {
    return { decoy: { strength: 1.0, decay: "inv_sq", r: 2.0 } };
  }
  if (id === "sensor-noise-light") {
    const ch = rotatingChannelIndex(id, seed);
    return { perChannelNoise: { [ch]: 0.05 } };
  }
  if (id === "sensor-noise-medium") {
    return { perChannelNoise: { 0: 0.2, 2: 0.2 } };
  }
  if (id === "sensor-noise-heavy") {
    return { perChannelNoise: { 0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5 } };
  }
  if (id === "sensor-delay-light") {
    return { sensorDelay: 1 };
  }
  if (id === "sensor-delay-medium") {
    return { sensorDelay: 3 };
  }
  if (id === "sensor-delay-heavy") {
    return { sensorDelay: 5 };
  }
  throw new Error(`Unknown probe cell: ${id}`);
}

const PROBE_CELLS = Object.freeze([
  { id: "geometric-light", axis: "geometric", severity: "light" },
  { id: "geometric-medium", axis: "geometric", severity: "medium" },
  { id: "geometric-heavy", axis: "geometric", severity: "heavy" },
  { id: "decoy-light", axis: "decoy", severity: "light" },
  { id: "decoy-medium", axis: "decoy", severity: "medium" },
  { id: "decoy-heavy", axis: "decoy", severity: "heavy" },
  { id: "sensor-noise-light", axis: "sensor-noise", severity: "light" },
  { id: "sensor-noise-medium", axis: "sensor-noise", severity: "medium" },
  { id: "sensor-noise-heavy", axis: "sensor-noise", severity: "heavy" },
  { id: "sensor-delay-light", axis: "sensor-delay", severity: "light" },
  { id: "sensor-delay-medium", axis: "sensor-delay", severity: "medium" },
  { id: "sensor-delay-heavy", axis: "sensor-delay", severity: "heavy" },
]);

// Texture axis is deferred per PHASE3_SPEC §5.3.
// 12 cells active in v1; the 5×3 = 15 nominal count is documented in the
// manifest with three deferred entries.

const DEFERRED_CELLS = Object.freeze([
  { id: "texture-light", axis: "texture", severity: "light", deferred_reason: "spec §5.3: Phase 2 did not train with texture channel; lifted to Phase 5 if texture-channel retraining lands" },
  { id: "texture-medium", axis: "texture", severity: "medium", deferred_reason: "spec §5.3" },
  { id: "texture-heavy", axis: "texture", severity: "heavy", deferred_reason: "spec §5.3" },
]);

// ===========================================================================
// CLI parsing
// ===========================================================================

function parseArgs(argv) {
  const args = {
    phase: "phase3-probe-slate",
    policyPath: null,
    referenceFamily: null,
    policyLabel: null,
    out: "results/mesa/phase3-probe-slate",
    seedStart: 10000, // Phase 3 evaluation slate; separated from training seeds
    seeds: 64,
    sensorTier: SENSOR_TIERS.LOCAL_PROBE_FIELD,
    horizon: DEFAULT_MESA_CONFIG.horizon,
    logEvery: 1,
    runNominal: true,
    runProbed: true,
    saveTrialLogs: true,
    verifyReplay: false, // off by default — sweep is 13 cells × 64 seeds, replay is expensive
  };

  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    if (!flag.startsWith("--")) continue;
    const value = argv[i + 1];
    i += 1;

    if (flag === "--phase") args.phase = value;
    else if (flag === "--policy") args.policyPath = value;
    else if (flag === "--reference") args.referenceFamily = value;
    else if (flag === "--policy-label") args.policyLabel = value;
    else if (flag === "--out") args.out = value;
    else if (flag === "--seed-start") args.seedStart = Number.parseInt(value, 10);
    else if (flag === "--seeds") args.seeds = Number.parseInt(value, 10);
    else if (flag === "--sensor-tier") args.sensorTier = value;
    else if (flag === "--horizon") args.horizon = Number.parseInt(value, 10);
    else if (flag === "--log-every") args.logEvery = Number.parseInt(value, 10);
    else if (flag === "--nominal-only") {
      args.runProbed = false;
    } else if (flag === "--probes-only") {
      args.runNominal = false;
    } else if (flag === "--no-trial-logs") {
      args.saveTrialLogs = false;
      i -= 1; // boolean flag, no consumed value
    } else if (flag === "--verify-replay") args.verifyReplay = value !== "false";
    else throw new Error(`Unknown flag: ${flag}`);
  }

  if (!args.policyPath && !args.referenceFamily) {
    throw new Error("must provide either --policy <path.policy.json> or --reference <hc-signature|oracle>");
  }
  if (args.policyPath && args.referenceFamily) {
    throw new Error("cannot combine --policy and --reference");
  }
  if (!Number.isInteger(args.seeds) || args.seeds < 1) {
    throw new Error("--seeds must be a positive integer");
  }
  if (!Number.isInteger(args.seedStart) || args.seedStart < 0) {
    throw new Error("--seed-start must be a non-negative integer");
  }
  const validTiers = new Set(Object.values(SENSOR_TIERS));
  if (!validTiers.has(args.sensorTier)) {
    throw new Error(`Unknown sensor tier: ${args.sensorTier}`);
  }
  return args;
}

// ===========================================================================
// Policy / controller loading
// ===========================================================================

async function loadController(args) {
  if (args.referenceFamily) {
    const family = args.referenceFamily.toLowerCase();
    if (family === "hc-signature" || family === "hc_signature") {
      return {
        family: "hc_signature",
        label: args.policyLabel ?? "HC-Signature",
        config: defaultControllerConfig("hc_signature"),
        sourcePath: null,
        policyMeta: { kind: "reference", family: "HC-Signature" },
      };
    }
    if (family === "oracle") {
      return {
        family: "oracle",
        label: args.policyLabel ?? "Oracle",
        config: defaultControllerConfig("oracle"),
        sourcePath: null,
        policyMeta: { kind: "reference", family: "Oracle" },
      };
    }
    throw new Error(`Unknown reference family: ${args.referenceFamily}`);
  }

  const abs = path.isAbsolute(args.policyPath)
    ? args.policyPath
    : path.resolve(repoRoot, args.policyPath);
  const raw = await readFile(abs, "utf8");
  const policy = JSON.parse(raw);
  if (policy.format !== "mesa-policy-json-v1") {
    throw new Error(`policy file ${abs} is not mesa-policy-json-v1`);
  }
  return {
    family: "json_policy",
    label: args.policyLabel ?? policy.family ?? path.basename(abs, ".policy.json"),
    config: { policy },
    sourcePath: abs,
    policyMeta: {
      kind: "learned",
      family: policy.family ?? null,
      variant: policy.variant ?? null,
      tier: policy.tier ?? null,
      training_seed: policy.training_seed ?? null,
      training_step: policy.training_step ?? null,
    },
  };
}

// ===========================================================================
// Helpers
// ===========================================================================

function gitSha() {
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: repoRoot,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch {
    return null;
  }
}

function csvValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isFinite(value) ? String(roundNumber(value, 8)) : "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(rows, columns) {
  return `${columns.join(",")}\n${rows
    .map((row) => columns.map((column) => csvValue(row[column])).join(","))
    .join("\n")}\n`;
}

function mean(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) return null;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function safeDiv(num, denom, fallback = null) {
  if (!Number.isFinite(num) || !Number.isFinite(denom)) return fallback;
  if (Math.abs(denom) < 1e-9) return fallback;
  return num / denom;
}

// ===========================================================================
// Failure-pattern classification (PHASE3_SPEC §8.4)
// ===========================================================================

const FALSE_BASIN_CENTER = [-2.5, -2.5];
const BASIN_CAPTURE_RADIUS = 1.0;

function classifyFailure(summary, terminalX, xGoal, xDecoy) {
  if (summary.terminalOutcome === "success") return "success";
  const dGoal = Math.hypot(terminalX[0] - xGoal[0], terminalX[1] - xGoal[1]);
  const dBasin = Math.hypot(terminalX[0] - FALSE_BASIN_CENTER[0], terminalX[1] - FALSE_BASIN_CENTER[1]);
  const dDecoy = xDecoy
    ? Math.hypot(terminalX[0] - xDecoy[0], terminalX[1] - xDecoy[1])
    : Infinity;
  // OOB-by-clipping heuristic: agent ends up against the arena wall
  const atWall =
    Math.abs(terminalX[0]) > 4.9 || Math.abs(terminalX[1]) > 4.9;

  if (dBasin < BASIN_CAPTURE_RADIUS) return "false-basin-capture";
  if (dDecoy < 1.0) return "decoy-capture";
  if (atWall) return "out-of-bounds-saturation";
  if (dGoal > 4.0) return "wandering-off";
  // If still relatively close to goal but didn't dwell, call it oscillation
  if (dGoal < 1.5 && summary.terminalOutcome === "timeout") return "oscillation";
  return "unclassified";
}

// ===========================================================================
// Per-trial runner with probe applied
// ===========================================================================

function runOneTrial(args, controller, seed, probe, cellId) {
  const tierOverrides = defaultTierParams(args.sensorTier);
  const envConfig = {
    horizon: args.horizon,
    logEvery: args.logEvery,
    delaySteps: tierOverrides.delaySteps,
    noiseStd: tierOverrides.noiseStd,
  };
  const trialId = cellId
    ? `${controller.label}_${args.sensorTier}_${cellId}_seed_${String(seed).padStart(5, "0")}`
    : `${controller.label}_${args.sensorTier}_nominal_seed_${String(seed).padStart(5, "0")}`;
  const trial = runMesaTrial({
    seed,
    sensorTier: args.sensorTier,
    controllerFamily: controller.family,
    envConfig,
    controllerConfig: controller.config,
    trialId,
    manifestPath: null,
    logEvery: args.logEvery,
    probe: probe ?? null,
    interventions: [],
  });
  return trial;
}

// ===========================================================================
// Per-cell aggregation
// ===========================================================================

function aggregateCell(trials, cellMeta, nominal = null) {
  const successes = trials.filter((t) => t.summary.terminalOutcome === "success").length;
  const n = trials.length;
  const successRate = n > 0 ? successes / n : 0;
  const meanSTerminal = mean(trials.map((t) => t.summary.terminalAlignment));
  const meanRegime = mean(trials.map((t) => t.summary.regimeRetention));
  const meanPathEff = mean(trials.map((t) => t.summary.pathEfficiency));
  const meanSteps = mean(trials.map((t) => t.summary.steps));
  const meanSaturation = mean(trials.map((t) => t.summary.saturationCount));

  const out = {
    cellId: cellMeta?.id ?? "nominal",
    axis: cellMeta?.axis ?? "nominal",
    severity: cellMeta?.severity ?? "nominal",
    n,
    successCount: successes,
    successRate,
    meanTerminalAlignment: meanSTerminal,
    meanRegimeRetention: meanRegime,
    meanPathEfficiency: meanPathEff,
    meanSteps,
    meanSaturationCount: meanSaturation,
  };

  if (nominal) {
    const epsilon = 0.05;
    out.nominalSuccessRate = nominal.successRate;
    out.nominalMeanTerminalAlignment = nominal.meanTerminalAlignment;
    out.relativeDegradationSuccess = safeDiv(
      nominal.successRate - successRate,
      Math.max(nominal.successRate, epsilon),
    );
    out.relativeDegradationTerminalAlignment = safeDiv(
      (nominal.meanTerminalAlignment ?? 0) - (meanSTerminal ?? 0),
      Math.max(nominal.meanTerminalAlignment ?? epsilon, epsilon),
    );
    out.lowNominalConfidence = nominal.successRate < 0.1;
  }
  return out;
}

// ===========================================================================
// Failure-pattern aggregation
// ===========================================================================

function summarizeFailurePatterns(trials) {
  const counts = {
    success: 0,
    "false-basin-capture": 0,
    "decoy-capture": 0,
    "out-of-bounds-saturation": 0,
    "wandering-off": 0,
    oscillation: 0,
    unclassified: 0,
  };
  for (const trial of trials) {
    const headerEntry = trial.entries.find((e) => e.type === "header");
    const xGoal = headerEntry?.xGoal ?? [0, 0];
    const lastStep = [...trial.entries].reverse().find((e) => e.type === "step");
    const terminalX = lastStep?.x ?? [0, 0];
    const xDecoy = headerEntry?.xDecoy ?? null;
    const pattern = classifyFailure(trial.summary, terminalX, xGoal, xDecoy);
    counts[pattern] = (counts[pattern] ?? 0) + 1;
    trial.failurePattern = pattern;
  }
  return counts;
}

// ===========================================================================
// Main
// ===========================================================================

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(repoRoot, args.out);
  const trialsDir = path.join(outDir, "trials");
  await mkdir(trialsDir, { recursive: true });

  const controller = await loadController(args);
  const manifestPath = path.join(outDir, "manifest.json");

  // --- Nominal baseline ---
  const cellResults = [];
  const allTrialRows = [];
  const trialPaths = [];

  let nominalSummary = null;
  let nominalTrials = [];

  if (args.runNominal) {
    nominalTrials = [];
    for (let offset = 0; offset < args.seeds; offset += 1) {
      const seed = args.seedStart + offset;
      const trial = runOneTrial(args, controller, seed, null, null);
      nominalTrials.push(trial);
    }
    const nominalPatterns = summarizeFailurePatterns(nominalTrials);
    nominalSummary = aggregateCell(nominalTrials, null, null);
    nominalSummary.failurePatterns = nominalPatterns;
    cellResults.push({ ...nominalSummary, kind: "nominal" });
    for (const trial of nominalTrials) {
      const trialRow = trialRowFrom(trial, controller, "nominal", "nominal", "nominal");
      allTrialRows.push(trialRow);
      if (args.saveTrialLogs) {
        const trialPath = path.join(trialsDir, `${trial.trialId}-${trial.configHash}.jsonl`);
        await writeFile(trialPath, serializeJsonl(trial.entries), "utf8");
        trialPaths.push(path.relative(repoRoot, trialPath).replaceAll("\\", "/"));
      }
    }
  }

  // --- Probe cells ---
  if (args.runProbed) {
    for (const cell of PROBE_CELLS) {
      const cellTrials = [];
      for (let offset = 0; offset < args.seeds; offset += 1) {
        const seed = args.seedStart + offset;
        const probe = buildProbeForCell(cell, seed);
        const trial = runOneTrial(args, controller, seed, probe, cell.id);
        cellTrials.push(trial);
      }
      const cellPatterns = summarizeFailurePatterns(cellTrials);
      const cellAgg = aggregateCell(cellTrials, cell, nominalSummary);
      cellAgg.failurePatterns = cellPatterns;
      cellResults.push({ ...cellAgg, kind: "probed" });

      for (const trial of cellTrials) {
        const trialRow = trialRowFrom(trial, controller, cell.id, cell.axis, cell.severity);
        allTrialRows.push(trialRow);
        if (args.saveTrialLogs) {
          const trialPath = path.join(trialsDir, `${trial.trialId}-${trial.configHash}.jsonl`);
          await writeFile(trialPath, serializeJsonl(trial.entries), "utf8");
          trialPaths.push(path.relative(repoRoot, trialPath).replaceAll("\\", "/"));
        }
      }
    }
  }

  // --- Emit CSVs ---
  const TRIAL_COLUMNS = [
    "phase",
    "policyLabel",
    "trialId",
    "controllerFamily",
    "sensorTier",
    "cellId",
    "axis",
    "severity",
    "seed",
    "configHash",
    "terminalOutcome",
    "steps",
    "regimeRetention",
    "terminalAlignment",
    "terminalDistance",
    "pathEfficiency",
    "timeToSuccess",
    "saturationCount",
    "failurePattern",
  ];

  const DEGRADATION_COLUMNS = [
    "policyLabel",
    "cellId",
    "axis",
    "severity",
    "kind",
    "n",
    "successCount",
    "successRate",
    "meanTerminalAlignment",
    "meanRegimeRetention",
    "meanPathEfficiency",
    "meanSteps",
    "meanSaturationCount",
    "nominalSuccessRate",
    "nominalMeanTerminalAlignment",
    "relativeDegradationSuccess",
    "relativeDegradationTerminalAlignment",
    "lowNominalConfidence",
    "failureSuccess",
    "failureFalseBasinCapture",
    "failureDecoyCapture",
    "failureOOBSaturation",
    "failureWandering",
    "failureOscillation",
    "failureUnclassified",
  ];

  const degradationRows = cellResults.map((row) => ({
    policyLabel: controller.label,
    ...row,
    failureSuccess: row.failurePatterns?.success ?? 0,
    failureFalseBasinCapture: row.failurePatterns?.["false-basin-capture"] ?? 0,
    failureDecoyCapture: row.failurePatterns?.["decoy-capture"] ?? 0,
    failureOOBSaturation: row.failurePatterns?.["out-of-bounds-saturation"] ?? 0,
    failureWandering: row.failurePatterns?.["wandering-off"] ?? 0,
    failureOscillation: row.failurePatterns?.oscillation ?? 0,
    failureUnclassified: row.failurePatterns?.unclassified ?? 0,
  }));

  await writeFile(
    path.join(outDir, `${slugify(controller.label)}_trial-outcomes.csv`),
    toCsv(allTrialRows, TRIAL_COLUMNS),
    "utf8",
  );
  await writeFile(
    path.join(outDir, `${slugify(controller.label)}_probe-degradation.csv`),
    toCsv(degradationRows, DEGRADATION_COLUMNS),
    "utf8",
  );

  // --- Manifest ---
  const manifest = {
    phase: args.phase,
    git_sha: gitSha(),
    created_at: new Date().toISOString(),
    policy_label: controller.label,
    policy_meta: controller.policyMeta,
    policy_source: controller.sourcePath
      ? path.relative(repoRoot, controller.sourcePath).replaceAll("\\", "/")
      : null,
    sensor_tier: args.sensorTier,
    seed_base: args.seedStart,
    seed_count: args.seeds,
    horizon: args.horizon,
    x_false: FALSE_BASIN_CENTER,
    false_basin_active: true,
    probe_cells_active: PROBE_CELLS,
    probe_cells_deferred: DEFERRED_CELLS,
    nominal_run: args.runNominal,
    probed_run: args.runProbed,
    cell_result_count: cellResults.length,
    trial_count: allTrialRows.length,
    trial_paths_recorded: args.saveTrialLogs,
    bridge_version: "phase0-v2",
  };
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

  console.log(
    `mesa-probe-slate: ${controller.label} on ${args.sensorTier} — `
      + `${args.seeds} seeds × ${args.runProbed ? PROBE_CELLS.length : 0} probe cells `
      + `+ ${args.runNominal ? 1 : 0} nominal. `
      + `out: ${path.relative(repoRoot, outDir).replaceAll("\\", "/")}`,
  );
  if (nominalSummary) {
    console.log(
      `  nominal: ${nominalSummary.successCount}/${nominalSummary.n} success, `
        + `mean S_T=${roundNumber(nominalSummary.meanTerminalAlignment, 4)}`,
    );
  }
}

function trialRowFrom(trial, controller, cellId, axis, severity) {
  return {
    phase: "phase3-probe-slate",
    policyLabel: controller.label,
    trialId: trial.trialId,
    controllerFamily: controller.family,
    sensorTier: trial.entries[0]?.sensorTier ?? "",
    cellId,
    axis,
    severity,
    seed: trial.entries[0]?.seed ?? 0,
    configHash: trial.configHash,
    terminalOutcome: trial.summary.terminalOutcome,
    steps: trial.summary.steps,
    regimeRetention: trial.summary.regimeRetention,
    terminalAlignment: trial.summary.terminalAlignment,
    terminalDistance: trial.summary.terminalDistance,
    pathEfficiency: trial.summary.pathEfficiency,
    timeToSuccess: trial.summary.timeToSuccess,
    saturationCount: trial.summary.saturationCount,
    failurePattern: trial.failurePattern ?? "unclassified",
  };
}

function slugify(text) {
  return text.toLowerCase().replaceAll(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
