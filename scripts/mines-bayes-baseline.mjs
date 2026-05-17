#!/usr/bin/env node

import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  ACTION,
  applyMinesAction,
  getPublicMemory,
  initializeBoardState,
  makeRng,
  TERMINAL,
  TILE
} from "../public/js/mines-core.mjs";
import {
  chooseMinesAction,
  MINES_CONTROLLER_MODES
} from "../public/js/mines-controllers.mjs";
import {
  createSensorRuntime,
  normalizeSensorConfig
} from "../public/js/mines-sensor.mjs";
import {
  assertNoMinesBayesObservationLeak,
  forbiddenMinesBayesObservationKeys,
  serializeMinesBayesObservation
} from "../public/js/mines-bayes-admission.mjs";

const __filename = fileURLToPath(import.meta.url);
const REPO_ROOT = path.resolve(path.dirname(__filename), "..");

const BAYES_MODES = new Set(["bayes_frontier_pressure", "bayes_frontier_full"]);
const DEFAULT_MODES = [
  "naive_pressure",
  "sundog_minimal",
  "sundog_lean",
  "bayes_frontier_pressure",
  "oracle_safe"
];

const DEFAULT_ARGS = {
  phase: "phase12-bayes-admission-smoke",
  out: "results/mines/phase12-bayes-admission-smoke",
  cellSlate: "phase10-best-worst",
  phase10Out: "results/mines/phase10-envelope",
  modes: DEFAULT_MODES,
  seeds: 2,
  seedStart: null,
  particleCount: 64,
  turnCap: 160,
  pressureThreshold: 1.2,
  revealRiskThreshold: 0.42,
  flagRiskThreshold: 0.82
};

const CSV_HEADERS = {
  trialOutcomes: [
    "phase",
    "selection",
    "cell_id",
    "cell_class",
    "mode",
    "seed",
    "preset",
    "width",
    "height",
    "mine_count",
    "scan_budget",
    "sigma",
    "sigma_noise",
    "dropout_rate",
    "delay_steps",
    "turns",
    "terminal",
    "won",
    "mine_triggered",
    "revealed_safe_tiles",
    "flags",
    "scans_used",
    "budget_adjusted_safe_tiles",
    "illegal_actions",
    "posterior_decisions",
    "posterior_mean_ess",
    "posterior_min_selected_hazard",
    "posterior_max_selected_hazard"
  ],
  posteriorDiagnostics: [
    "phase",
    "selection",
    "cell_id",
    "mode",
    "seed",
    "turn",
    "budget",
    "phi_hash",
    "frontier_size",
    "particle_count",
    "ess",
    "min_hazard",
    "max_hazard",
    "selected_action",
    "selected_x",
    "selected_y",
    "selected_idx",
    "selected_hazard",
    "observed_pressure_count",
    "scan_reading_count",
    "forbidden_key_count",
    "leak_free"
  ],
  bayesActions: [
    "phase",
    "selection",
    "cell_id",
    "mode",
    "seed",
    "turn",
    "budget",
    "action",
    "x",
    "y",
    "posterior_hazard",
    "ess",
    "frontier_size",
    "legal",
    "terminal_after"
  ],
  regret: [
    "phase",
    "selection",
    "cell_id",
    "cell_class",
    "seed",
    "bayes_mode",
    "target_mode",
    "bayes_budget_adjusted_safe_tiles",
    "target_budget_adjusted_safe_tiles",
    "budget_adjusted_delta",
    "bayes_won",
    "target_won"
  ],
  regretSummary: [
    "phase",
    "selection",
    "cell_id",
    "cell_class",
    "bayes_mode",
    "target_mode",
    "n",
    "mean_budget_adjusted_delta",
    "min_budget_adjusted_delta",
    "max_budget_adjusted_delta",
    "win_delta"
  ]
};

function usage() {
  return [
    "Usage:",
    "  node scripts/mines-bayes-baseline.mjs --phase <name> --out <dir> --cell-slate phase10-best-worst --phase10-out <dir> --modes <csv> --seeds <n> --particle-count <n> --turn-cap <n>",
    "",
    "Example:",
    "  node scripts/mines-bayes-baseline.mjs --phase phase12-bayes-admission-smoke --out results/mines/phase12-bayes-admission-smoke --cell-slate phase10-best-worst --phase10-out results/mines/phase10-envelope --modes naive_pressure,sundog_minimal,sundog_lean,bayes_frontier_pressure,oracle_safe --seeds 2 --particle-count 64 --turn-cap 160"
  ].join("\n");
}

function parseArgs(argv) {
  const args = { ...DEFAULT_ARGS, modes: [...DEFAULT_ARGS.modes] };
  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    if (key === "--help" || key === "-h") {
      console.log(usage());
      process.exit(0);
    }
    if (!key.startsWith("--")) {
      throw new Error(`Unexpected positional argument: ${key}`);
    }
    const value = argv[i + 1];
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`Missing value for ${key}`);
    }
    i += 1;

    switch (key) {
      case "--phase":
        args.phase = value;
        break;
      case "--out":
        args.out = value;
        break;
      case "--cell-slate":
        args.cellSlate = value;
        break;
      case "--phase10-out":
        args.phase10Out = value;
        break;
      case "--modes":
        args.modes = value.split(",").map((mode) => mode.trim()).filter(Boolean);
        break;
      case "--seeds":
        args.seeds = parsePositiveInteger(value, key);
        break;
      case "--seed-start":
        args.seedStart = parseInteger(value, key);
        break;
      case "--particle-count":
        args.particleCount = parsePositiveInteger(value, key);
        break;
      case "--turn-cap":
        args.turnCap = parsePositiveInteger(value, key);
        break;
      case "--pressure-threshold":
        args.pressureThreshold = parseFiniteNumber(value, key);
        break;
      case "--reveal-risk-threshold":
        args.revealRiskThreshold = parseFiniteNumber(value, key);
        break;
      case "--flag-risk-threshold":
        args.flagRiskThreshold = parseFiniteNumber(value, key);
        break;
      default:
        throw new Error(`Unknown option: ${key}`);
    }
  }

  if (args.cellSlate !== "phase10-best-worst") {
    throw new Error(`Unsupported --cell-slate ${args.cellSlate}; expected phase10-best-worst`);
  }
  if (args.modes.length === 0) {
    throw new Error("--modes must include at least one mode");
  }
  for (const mode of args.modes) {
    if (BAYES_MODES.has(mode)) {
      continue;
    }
    const definition = MINES_CONTROLLER_MODES[mode];
    if (!definition) {
      throw new Error(`Unknown Mines controller mode: ${mode}`);
    }
    if (definition.status && definition.status !== "implemented") {
      throw new Error(`Mines controller mode ${mode} is ${definition.status}, not runnable in this smoke`);
    }
  }

  return args;
}

function parsePositiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer; got ${value}`);
  }
  return parsed;
}

function parseInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) {
    throw new Error(`${label} must be an integer; got ${value}`);
  }
  return parsed;
}

function parseFiniteNumber(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${label} must be finite; got ${value}`);
  }
  return parsed;
}

function asNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function asInteger(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isInteger(parsed) ? parsed : fallback;
}

function toRelPath(value) {
  return path.resolve(REPO_ROOT, value);
}

async function readText(filePath) {
  return readFile(filePath, "utf8");
}

function parseCsv(text) {
  const rows = [];
  let field = "";
  let row = [];
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          quoted = false;
        }
      } else {
        field += ch;
      }
      continue;
    }

    if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (ch !== "\r") {
      field += ch;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  if (rows.length === 0) {
    return [];
  }

  const headers = rows[0];
  return rows.slice(1)
    .filter((values) => values.some((value) => value.trim() !== ""))
    .map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])));
}

function csvEscape(value) {
  if (value === null || value === undefined) {
    return "";
  }
  const text = String(value);
  if (/[",\n\r]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

function rowsToCsv(rows, headers) {
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map((header) => csvEscape(row[header])).join(","));
  }
  return `${lines.join("\n")}\n`;
}

async function writeJson(filePath, value) {
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

async function loadPhase10BestWorst(phase10Out) {
  const filePath = path.join(toRelPath(phase10Out), "best-worst-cells.csv");
  const rows = parseCsv(await readText(filePath));
  const selected = rows.filter((row) => row.selection === "best_cell" || row.selection === "worst_cell");
  if (selected.length !== 2) {
    throw new Error(`Expected best_cell and worst_cell in ${filePath}; found ${selected.length}`);
  }
  return selected.map((row) => normalizePhase10Cell(row));
}

function normalizePhase10Cell(row) {
  const replay = parseReplayUrl(row.replay_url);
  const configuredScanBudget = asInteger(row.configured_scan_budget, asInteger(row.scan_budget, 0));
  const replayScanBudget = replay.scanBudget ?? asInteger(row.scan_budget, configuredScanBudget);
  const mineCount = replay.mineCount ?? asInteger(row.mine_count, 0);
  const width = asInteger(row.width, 9);
  const height = asInteger(row.height, 9);

  return {
    selection: row.selection,
    cellId: row.cell_id,
    cellClass: row.cell_class,
    phase10Mode: row.mode,
    representativeSeed: asInteger(row.representative_seed, 0),
    replayUrl: row.replay_url,
    preset: replay.preset || "easy_sparse",
    width,
    height,
    mineCount,
    mineDensity: asNumber(row.mine_density, mineCount / Math.max(1, width * height)),
    clusterStrength: asNumber(row.cluster_strength, 0),
    configuredScanBudget,
    replayScanBudget,
    scanBudget: replayScanBudget,
    sigma: asNumber(row.sigma, 1),
    sigmaNoise: replay.sigmaNoise ?? asNumber(row.sigma_noise, 0),
    dropoutRate: replay.dropoutRate ?? asNumber(row.dropout_rate, 0),
    delaySteps: asInteger(row.delay_steps, 0),
    scanBudgetDeltaVsNaive: asNumber(row.budget_adjusted_delta_vs_naive, 0),
    lowerCiVsNaive: asNumber(row.lower_ci_vs_naive, 0),
    upperCiVsNaive: asNumber(row.upper_ci_vs_naive, 0),
    envelopeCell: {
      selection: row.selection,
      cellId: row.cell_id,
      cellClass: row.cell_class,
      phase10Mode: row.mode,
      width,
      height,
      mineCount,
      mineDensity: asNumber(row.mine_density, mineCount / Math.max(1, width * height)),
      clusterStrength: asNumber(row.cluster_strength, 0),
      configuredScanBudget,
      replayScanBudget,
      sigma: asNumber(row.sigma, 1),
      sigmaNoise: replay.sigmaNoise ?? asNumber(row.sigma_noise, 0),
      dropoutRate: replay.dropoutRate ?? asNumber(row.dropout_rate, 0),
      delaySteps: asInteger(row.delay_steps, 0),
      budgetAdjustedDeltaVsNaive: asNumber(row.budget_adjusted_delta_vs_naive, 0),
      lowerCiVsNaive: asNumber(row.lower_ci_vs_naive, 0),
      upperCiVsNaive: asNumber(row.upper_ci_vs_naive, 0)
    }
  };
}

function parseReplayUrl(value) {
  const parsed = {
    preset: null,
    mineCount: null,
    scanBudget: null,
    sigmaNoise: null,
    dropoutRate: null
  };
  if (!value) {
    return parsed;
  }
  try {
    const url = new URL(value);
    parsed.preset = url.searchParams.get("preset");
    parsed.mineCount = numberParam(url.searchParams, "mine_count");
    parsed.scanBudget = integerParam(url.searchParams, "scan_budget");
    parsed.sigmaNoise = numberParam(url.searchParams, "sigma_noise");
    parsed.dropoutRate = numberParam(url.searchParams, "dropout");
  } catch {
    return parsed;
  }
  return parsed;
}

function numberParam(params, key) {
  const value = params.get(key);
  if (value === null) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function integerParam(params, key) {
  const value = numberParam(params, key);
  return Number.isInteger(value) ? value : null;
}

function buildBoardConfig(cell, mode, seed, args) {
  const bayesFull = mode === "bayes_frontier_full";
  const baseScanBudget = bayesFull ? cell.configuredScanBudget : cell.replayScanBudget;
  const definition = MINES_CONTROLLER_MODES[mode];
  return {
    preset: cell.preset,
    seed,
    turnCap: args.turnCap,
    width: cell.width,
    height: cell.height,
    mineCount: cell.mineCount,
    scanBudget: baseScanBudget,
    generator: {
      clusterStrength: cell.clusterStrength
    },
    ...(definition?.boardOverride ?? {})
  };
}

function buildSensorConfig(cell, mode, seed) {
  return normalizeSensorConfig({
    sigma: cell.sigma,
    sigmaNoise: cell.sigmaNoise,
    dropoutRate: cell.dropoutRate,
    delaySteps: cell.delaySteps,
    sensorSeed: deterministicSeed(`${cell.cellId}:${mode}:${seed}:sensor`)
  });
}

function deterministicSeed(text) {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function hashJson(value) {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex").slice(0, 16);
}

function makeTrialId({ phase, selection, cellId, mode, seed }) {
  return `${phase}__${selection}__${cellId}__${mode}__seed_${seed}`;
}

function centerAction(boardState) {
  return {
    type: ACTION.REVEAL,
    x: Math.floor(boardState.config.width / 2),
    y: Math.floor(boardState.config.height / 2)
  };
}

function seedListForCell(cell, args) {
  const start = args.seedStart === null ? cell.representativeSeed : args.seedStart;
  return Array.from({ length: args.seeds }, (_, index) => start + index);
}

function inferWon(boardState) {
  if (boardState.terminal === TERMINAL.FULL_CLEAR) {
    return true;
  }
  if (boardState.terminal === TERMINAL.MINE_TRIGGERED) {
    return false;
  }
  const safeTiles = boardState.config.width * boardState.config.height - boardState.config.mineCount;
  const revealedSafe = boardState.revealedSafeCount;
  return revealedSafe >= safeTiles;
}

function randomLegalAction(memory, rng) {
  const concealed = [];
  for (let index = 0; index < memory.tiles.length; index += 1) {
    const tile = memory.tiles[index];
    if (tile === TILE.CONCEALED && !memory.flags[index]) {
      concealed.push(index);
    }
  }
  if (concealed.length === 0) {
    return { type: ACTION.REVEAL, x: 0, y: 0 };
  }
  const idx = concealed[Math.floor(rng() * concealed.length)];
  return {
    type: ACTION.REVEAL,
    x: idx % memory.width,
    y: Math.floor(idx / memory.width)
  };
}

function runTrial({ args, cell, mode, seed }) {
  const boardConfig = buildBoardConfig(cell, mode, seed, args);
  const sensorConfig = buildSensorConfig(cell, mode, seed);
  const boardState = initializeBoardState(boardConfig);
  applyMinesAction(boardState, centerAction(boardState));
  const sensorRuntime = createSensorRuntime(sensorConfig);
  const rng = makeRng(deterministicSeed(`${cell.cellId}:${mode}:${seed}:trial`));
  const trialId = makeTrialId({ phase: args.phase, selection: cell.selection, cellId: cell.cellId, mode, seed });

  const bayesDiagnostics = [];
  const bayesActions = [];
  const observationParity = [];
  const posteriorMap = [];

  let illegalActions = 0;
  let posteriorDecisionCount = 0;
  const selectedHazards = [];
  const essValues = [];

  while (!boardState.terminal && boardState.turn < args.turnCap) {
    const sensor = sensorRuntime.step(boardState);
    const memory = getPublicMemory(boardState);
    let action;
    let bayesDecision = null;
    let phi = null;

    if (BAYES_MODES.has(mode)) {
      phi = serializeMinesBayesObservation({
        memory,
        sensor,
        sensorConfig,
        budget: MINES_CONTROLLER_MODES[mode]?.bayesBudget ?? "pressure",
        controllerMode: mode,
        envelopeCell: cell.envelopeCell
      });
      assertNoMinesBayesObservationLeak(phi);
      bayesDecision = chooseBayesAction({ phi, mode, rng, args });
      action = bayesDecision.action;
      posteriorDecisionCount += 1;
      if (Number.isFinite(bayesDecision.selectedHazard)) {
        selectedHazards.push(bayesDecision.selectedHazard);
      }
      if (Number.isFinite(bayesDecision.ess)) {
        essValues.push(bayesDecision.ess);
      }
    } else {
      action = chooseMinesAction({
        mode,
        memory,
        sensor,
        boardState,
        rng,
        options: {
          threshold: args.pressureThreshold
        }
      });
    }

    let result = applyMinesAction(boardState, action);
    if (!result.applied) {
      illegalActions += 1;
      action = randomLegalAction(memory, rng);
      result = applyMinesAction(boardState, action);
    }

    if (result.applied && action.type === ACTION.SCAN) {
      const scanReading = sensorRuntime.scan(boardState, action.x, action.y);
      const latest = boardState.actionLedger[boardState.actionLedger.length - 1];
      if (latest?.type === ACTION.SCAN && latest.index === scanReading.index) {
        latest.scanReading = scanReading.reading;
      }
    }

    if (BAYES_MODES.has(mode) && bayesDecision && phi) {
      const phiHash = hashJson(phi);
      const forbiddenKeys = forbiddenMinesBayesObservationKeys(phi);
      const common = {
        phase: args.phase,
        selection: cell.selection,
        cell_id: cell.cellId,
        mode,
        seed,
        turn: phi.turnIndex,
        budget: phi.budget,
        phi_hash: phiHash,
        frontier_size: bayesDecision.frontierSize,
        particle_count: args.particleCount,
        ess: roundMetric(bayesDecision.ess),
        selected_action: action.type,
        selected_x: action.x,
        selected_y: action.y,
        selected_idx: indexFromPoint(action.x, action.y, phiWidth(phi)),
        selected_hazard: roundMetric(bayesDecision.selectedHazard),
        forbidden_key_count: forbiddenKeys.length,
        leak_free: forbiddenKeys.length === 0
      };
      bayesDiagnostics.push({
        ...common,
        min_hazard: roundMetric(bayesDecision.minHazard),
        max_hazard: roundMetric(bayesDecision.maxHazard),
        observed_pressure_count: bayesDecision.observedPressureCount,
        scan_reading_count: bayesDecision.scanReadingCount
      });
      bayesActions.push({
        phase: args.phase,
        selection: cell.selection,
        cell_id: cell.cellId,
        mode,
        seed,
        turn: phi.turnIndex,
        budget: phi.budget,
        action: action.type,
        x: action.x,
        y: action.y,
        posterior_hazard: roundMetric(bayesDecision.selectedHazard),
        ess: roundMetric(bayesDecision.ess),
        frontier_size: bayesDecision.frontierSize,
        legal: result.applied,
        terminal_after: boardState.terminal ?? ""
      });
      observationParity.push({
        trialId,
        phase: args.phase,
        selection: cell.selection,
        cellId: cell.cellId,
        mode,
        seed,
        turn: phi.turnIndex,
        budget: phi.budget,
        phiHash,
        leakFree: forbiddenKeys.length === 0,
        forbiddenKeys
      });
      if (posteriorMap.length < 80) {
        posteriorMap.push({
          trialId,
          turn: phi.turnIndex,
          selection: cell.selection,
          cellId: cell.cellId,
          mode,
          seed,
          budget: phi.budget,
          frontierSize: bayesDecision.frontierSize,
          candidates: bayesDecision.candidates.slice(0, 8)
        });
      }
    }
  }

  const revealedSafeTiles = boardState.revealedSafeCount;
  const flags = Array.from(boardState.flags).filter(Boolean).length;
  const scansUsed = boardState.actionLedger.filter((entry) => entry.type === ACTION.SCAN).length;
  const budgetAdjustedSafeTiles = revealedSafeTiles - scansUsed;

  const trialOutcome = {
    phase: args.phase,
    selection: cell.selection,
    cell_id: cell.cellId,
    cell_class: cell.cellClass,
    mode,
    seed,
    preset: cell.preset,
    width: boardConfig.width,
    height: boardConfig.height,
    mine_count: boardConfig.mineCount,
    scan_budget: boardConfig.scanBudget,
    sigma: sensorConfig.sigma,
    sigma_noise: sensorConfig.sigmaNoise,
    dropout_rate: sensorConfig.dropoutRate,
    delay_steps: sensorConfig.delaySteps,
    turns: boardState.turn,
    terminal: boardState.terminal ?? "turn_cap",
    won: inferWon(boardState),
    mine_triggered: boardState.terminal === TERMINAL.MINE_TRIGGERED,
    revealed_safe_tiles: revealedSafeTiles,
    flags,
    scans_used: scansUsed,
    budget_adjusted_safe_tiles: budgetAdjustedSafeTiles,
    illegal_actions: illegalActions,
    posterior_decisions: posteriorDecisionCount,
    posterior_mean_ess: roundMetric(mean(essValues)),
    posterior_min_selected_hazard: roundMetric(minFinite(selectedHazards)),
    posterior_max_selected_hazard: roundMetric(maxFinite(selectedHazards))
  };

  return {
    trialOutcome,
    bayesDiagnostics,
    bayesActions,
    observationParity,
    posteriorMap
  };
}

function chooseBayesAction({ phi, mode, rng, args }) {
  const frontier = frontierIndicesFromPhi(phi);
  const budget = phi.budget;
  if (frontier.length === 0) {
    const fallback = firstConcealedUnflagged(phi) ?? 0;
    return {
      action: revealActionFromIndex(fallback, phiWidth(phi)),
      budget,
      frontierSize: 0,
      particleCount: 0,
      ess: 0,
      minHazard: NaN,
      maxHazard: NaN,
      selectedHazard: NaN,
      observedPressureCount: countObservedPressure(phi),
      scanReadingCount: phi.scanReadings?.length ?? 0,
      candidates: []
    };
  }

  const posterior = estimatePosteriorHazards(phi, frontier, args.particleCount, rng);
  const candidates = frontier.map((idx) => {
    const hazard = posterior.hazards.get(idx) ?? posterior.priorMineRate;
    return {
      idx,
      x: idx % phiWidth(phi),
      y: Math.floor(idx / phiWidth(phi)),
      hazard: roundMetric(hazard),
      entropy: roundMetric(binaryEntropy(hazard))
    };
  }).sort((a, b) => {
    const hazardDelta = a.hazard - b.hazard;
    if (Math.abs(hazardDelta) > 1e-12) {
      return hazardDelta;
    }
    return a.idx - b.idx;
  });

  const safest = candidates[0];
  const riskiest = [...candidates].sort((a, b) => b.hazard - a.hazard || a.idx - b.idx)[0];
  const uncertain = [...candidates].sort((a, b) => b.entropy - a.entropy || a.idx - b.idx)[0];
  let action = revealActionFromIndex(safest.idx, phiWidth(phi));
  let selected = safest;

  if (mode === "bayes_frontier_full" && riskiest.hazard >= args.flagRiskThreshold) {
    action = flagActionFromIndex(riskiest.idx, phiWidth(phi));
    selected = riskiest;
  } else if (
    mode === "bayes_frontier_full" &&
    phi.scansRemaining > 0 &&
    uncertain.entropy >= 0.95 &&
    !isScanned(phi, uncertain.idx)
  ) {
    action = scanActionFromIndex(uncertain.idx, phiWidth(phi));
    selected = uncertain;
  } else if (safest.hazard > args.revealRiskThreshold) {
    action = revealActionFromIndex(safest.idx, phiWidth(phi));
    selected = safest;
  }

  return {
    action,
    budget,
    frontierSize: frontier.length,
    particleCount: args.particleCount,
    ess: posterior.ess,
    minHazard: candidates[0]?.hazard ?? NaN,
    maxHazard: riskiest?.hazard ?? NaN,
    selectedHazard: selected?.hazard ?? NaN,
    observedPressureCount: posterior.observedPressureCount,
    scanReadingCount: phi.scanReadings?.length ?? 0,
    candidates
  };
}

function estimatePosteriorHazards(phi, frontier, particleCount, rng) {
  const particles = [];
  let maxLogWeight = -Infinity;
  const observed = observedPressureSamples(phi);

  for (let i = 0; i < particleCount; i += 1) {
    const occupancy = sampleMineOccupancy(phi, rng);
    const predicted = pressureFromOccupancy(occupancy, phiWidth(phi), phiHeight(phi), phi.sensorConfig.sigma);
    const logWeight = pressureLogLikelihood({ phi, observed, predicted });
    particles.push({ occupancy, logWeight });
    if (logWeight > maxLogWeight) {
      maxLogWeight = logWeight;
    }
  }

  let weightSum = 0;
  let weightSquaredSum = 0;
  for (const particle of particles) {
    particle.weight = Math.exp(Math.max(-745, particle.logWeight - maxLogWeight));
    weightSum += particle.weight;
  }
  if (weightSum <= 0 || !Number.isFinite(weightSum)) {
    for (const particle of particles) {
      particle.weight = 1 / particles.length;
    }
  } else {
    for (const particle of particles) {
      particle.weight /= weightSum;
    }
  }

  const hazards = new Map();
  for (const idx of frontier) {
    let hazard = 0;
    for (const particle of particles) {
      hazard += particle.weight * particle.occupancy[idx];
    }
    hazards.set(idx, hazard);
  }
  for (const particle of particles) {
    weightSquaredSum += particle.weight * particle.weight;
  }

  return {
    hazards,
    ess: weightSquaredSum > 0 ? 1 / weightSquaredSum : 0,
    priorMineRate: priorMineRate(phi),
    observedPressureCount: observed.length
  };
}

function sampleMineOccupancy(phi, rng) {
  const n = phiWidth(phi) * phiHeight(phi);
  const occupancy = Array(n).fill(0);
  const available = [];
  let fixedMines = 0;

  for (let idx = 0; idx < n; idx += 1) {
    if (phi.flagState?.[idx]) {
      occupancy[idx] = 1;
      fixedMines += 1;
    } else if (phi.visibleTileState[idx] === "concealed") {
      available.push(idx);
    }
  }

  const remainingMines = Math.max(0, Math.min(available.length, phiMineCount(phi) - fixedMines));
  shuffleInPlace(available, rng);
  for (let i = 0; i < remainingMines; i += 1) {
    occupancy[available[i]] = 1;
  }
  return occupancy;
}

function pressureFromOccupancy(occupancy, width, height, sigma) {
  const field = Array(width * height).fill(0);
  const sigmaSafe = Math.max(0.1, sigma || 1);
  const twoSigmaSq = 2 * sigmaSafe * sigmaSafe;
  const radius = Math.max(1, Math.ceil(sigmaSafe * 3));

  for (let idx = 0; idx < occupancy.length; idx += 1) {
    if (!occupancy[idx]) {
      continue;
    }
    const mx = idx % width;
    const my = Math.floor(idx / width);
    const minX = Math.max(0, mx - radius);
    const maxX = Math.min(width - 1, mx + radius);
    const minY = Math.max(0, my - radius);
    const maxY = Math.min(height - 1, my + radius);
    for (let y = minY; y <= maxY; y += 1) {
      for (let x = minX; x <= maxX; x += 1) {
        const dx = x - mx;
        const dy = y - my;
        const distSq = dx * dx + dy * dy;
        field[indexFromPoint(x, y, width)] += Math.exp(-distSq / twoSigmaSq);
      }
    }
  }
  return field;
}

function pressureLogLikelihood({ phi, observed, predicted }) {
  if (observed.length === 0) {
    return 0;
  }
  const sigmaNoise = Math.max(0.15, phi.sensorConfig.sigmaNoise || 0);
  const variance = sigmaNoise * sigmaNoise + 0.05;
  let error = 0;
  for (const sample of observed) {
    const residual = sample.pressure - predicted[sample.idx];
    const confidenceScale = Math.max(0.05, sample.confidence);
    error += (residual * residual * confidenceScale) / variance;
  }
  return -0.5 * Math.min(1400, error);
}

function observedPressureSamples(phi) {
  const rows = [];
  const field = phi.observedPressureField ?? [];
  const confidence = phi.pressureConfidenceField ?? [];
  for (let idx = 0; idx < field.length; idx += 1) {
    const pressure = field[idx];
    if (!Number.isFinite(pressure)) {
      continue;
    }
    rows.push({
      idx,
      pressure,
      confidence: Number.isFinite(confidence[idx]) ? confidence[idx] : 1
    });
  }
  return rows;
}

function countObservedPressure(phi) {
  return observedPressureSamples(phi).length;
}

function frontierIndicesFromPhi(phi) {
  const frontier = [];
  const width = phiWidth(phi);
  const height = phiHeight(phi);
  for (let idx = 0; idx < phi.visibleTileState.length; idx += 1) {
    if (phi.visibleTileState[idx] !== "concealed" || phi.flagState?.[idx]) {
      continue;
    }
    const x = idx % width;
    const y = Math.floor(idx / width);
    if (neighbors(x, y, width, height).some((neighbor) => phi.visibleTileState[neighbor] === "revealed_safe")) {
      frontier.push(idx);
    }
  }
  if (frontier.length > 0) {
    return frontier;
  }
  const fallback = [];
  for (let idx = 0; idx < phi.visibleTileState.length; idx += 1) {
    if (phi.visibleTileState[idx] === "concealed" && !phi.flagState?.[idx]) {
      fallback.push(idx);
    }
  }
  return fallback;
}

function firstConcealedUnflagged(phi) {
  for (let idx = 0; idx < phi.visibleTileState.length; idx += 1) {
    if (phi.visibleTileState[idx] === "concealed" && !phi.flagState?.[idx]) {
      return idx;
    }
  }
  return null;
}

function neighbors(x, y, width, height) {
  const result = [];
  for (let dy = -1; dy <= 1; dy += 1) {
    for (let dx = -1; dx <= 1; dx += 1) {
      if (dx === 0 && dy === 0) {
        continue;
      }
      const nx = x + dx;
      const ny = y + dy;
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        result.push(indexFromPoint(nx, ny, width));
      }
    }
  }
  return result;
}

function isScanned(phi, idx) {
  return (phi.scanState ?? [])[idx] === true;
}

function revealActionFromIndex(idx, width) {
  return {
    type: ACTION.REVEAL,
    x: idx % width,
    y: Math.floor(idx / width)
  };
}

function flagActionFromIndex(idx, width) {
  return {
    type: ACTION.FLAG,
    x: idx % width,
    y: Math.floor(idx / width)
  };
}

function scanActionFromIndex(idx, width) {
  return {
    type: ACTION.SCAN,
    x: idx % width,
    y: Math.floor(idx / width)
  };
}

function indexFromPoint(x, y, width) {
  return y * width + x;
}

function phiWidth(phi) {
  return phi.boardWidth;
}

function phiHeight(phi) {
  return phi.boardHeight;
}

function phiMineCount(phi) {
  return phi.mineCount;
}

function priorMineRate(phi) {
  const concealed = phi.visibleTileState.filter((state, index) => state === "concealed" && !phi.flagState?.[index]).length;
  const flags = (phi.flagState ?? []).filter(Boolean).length;
  if (concealed <= 0) {
    return 0;
  }
  return Math.max(0, Math.min(1, (phiMineCount(phi) - flags) / concealed));
}

function shuffleInPlace(values, rng) {
  for (let i = values.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [values[i], values[j]] = [values[j], values[i]];
  }
}

function binaryEntropy(p) {
  if (!Number.isFinite(p) || p <= 0 || p >= 1) {
    return 0;
  }
  return -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
}

function roundMetric(value, digits = 6) {
  if (!Number.isFinite(value)) {
    return "";
  }
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function mean(values) {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) {
    return NaN;
  }
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function minFinite(values) {
  const finite = values.filter(Number.isFinite);
  return finite.length ? Math.min(...finite) : NaN;
}

function maxFinite(values) {
  const finite = values.filter(Number.isFinite);
  return finite.length ? Math.max(...finite) : NaN;
}

function buildRegretRows(trialOutcomes, args) {
  const groups = new Map();
  for (const row of trialOutcomes) {
    const key = `${row.selection}\n${row.cell_id}\n${row.seed}`;
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key).push(row);
  }

  const rows = [];
  for (const group of groups.values()) {
    const byMode = new Map(group.map((row) => [row.mode, row]));
    const bayesRows = group.filter((row) => BAYES_MODES.has(row.mode));
    for (const bayes of bayesRows) {
      for (const targetMode of args.modes) {
        if (targetMode === bayes.mode || BAYES_MODES.has(targetMode)) {
          continue;
        }
        const target = byMode.get(targetMode);
        if (!target) {
          continue;
        }
        rows.push({
          phase: args.phase,
          selection: bayes.selection,
          cell_id: bayes.cell_id,
          cell_class: bayes.cell_class,
          seed: bayes.seed,
          bayes_mode: bayes.mode,
          target_mode: target.mode,
          bayes_budget_adjusted_safe_tiles: bayes.budget_adjusted_safe_tiles,
          target_budget_adjusted_safe_tiles: target.budget_adjusted_safe_tiles,
          budget_adjusted_delta: roundMetric(bayes.budget_adjusted_safe_tiles - target.budget_adjusted_safe_tiles),
          bayes_won: bayes.won,
          target_won: target.won
        });
      }
    }
  }
  return rows;
}

function summarizeRegret(regretRows, args) {
  const groups = new Map();
  for (const row of regretRows) {
    const key = [
      row.selection,
      row.cell_id,
      row.cell_class,
      row.bayes_mode,
      row.target_mode
    ].join("\n");
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key).push(row);
  }

  return [...groups.values()].map((rows) => {
    const deltas = rows.map((row) => Number(row.budget_adjusted_delta)).filter(Number.isFinite);
    const winDelta = rows.reduce((sum, row) => sum + (boolish(row.bayes_won) ? 1 : 0) - (boolish(row.target_won) ? 1 : 0), 0);
    const first = rows[0];
    return {
      phase: args.phase,
      selection: first.selection,
      cell_id: first.cell_id,
      cell_class: first.cell_class,
      bayes_mode: first.bayes_mode,
      target_mode: first.target_mode,
      n: rows.length,
      mean_budget_adjusted_delta: roundMetric(mean(deltas)),
      min_budget_adjusted_delta: roundMetric(minFinite(deltas)),
      max_budget_adjusted_delta: roundMetric(maxFinite(deltas)),
      win_delta: winDelta
    };
  });
}

function boolish(value) {
  return value === true || value === "true";
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = toRelPath(args.out);
  const startedAt = new Date();
  const cells = await loadPhase10BestWorst(args.phase10Out);
  await mkdir(outDir, { recursive: true });

  const trialOutcomes = [];
  const posteriorDiagnostics = [];
  const bayesActions = [];
  const observationParity = [];
  const posteriorMap = [];

  const totalTrials = cells.reduce((sum, cell) => sum + seedListForCell(cell, args).length * args.modes.length, 0);
  let completedTrials = 0;

  for (const cell of cells) {
    for (const seed of seedListForCell(cell, args)) {
      for (const mode of args.modes) {
        const result = runTrial({ args, cell, mode, seed });
        trialOutcomes.push(result.trialOutcome);
        posteriorDiagnostics.push(...result.bayesDiagnostics);
        bayesActions.push(...result.bayesActions);
        observationParity.push(...result.observationParity);
        posteriorMap.push(...result.posteriorMap);
        completedTrials += 1;
      }
    }
  }

  const regretRows = buildRegretRows(trialOutcomes, args);
  const regretSummary = summarizeRegret(regretRows, args);
  const completedAt = new Date();
  const wallSeconds = (completedAt.getTime() - startedAt.getTime()) / 1000;
  const leakFree = posteriorDiagnostics.every((row) => Number(row.forbidden_key_count) === 0 && boolish(row.leak_free));

  const summary = {
    phase: args.phase,
    schema: "sundog.mines.phase12-bayes-smoke.summary.v1",
    startedAt: startedAt.toISOString(),
    completedAt: completedAt.toISOString(),
    wallSeconds,
    trials: trialOutcomes.length,
    cells: cells.length,
    modes: args.modes,
    seedsPerCell: args.seeds,
    particleCount: args.particleCount,
    posteriorDecisions: posteriorDiagnostics.length,
    leakFree,
    regretSummary
  };

  const manifest = {
    schema: "sundog.mines.phase12-bayes-smoke.manifest.v1",
    phase: args.phase,
    startedAt: startedAt.toISOString(),
    completedAt: completedAt.toISOString(),
    wallSeconds,
    totalTrials,
    completedTrials,
    source: {
      phase10Out: args.phase10Out,
      cellSlate: args.cellSlate
    },
    args,
    cells: cells.map((cell) => ({
      selection: cell.selection,
      cellId: cell.cellId,
      cellClass: cell.cellClass,
      representativeSeed: cell.representativeSeed,
      seeds: seedListForCell(cell, args),
      preset: cell.preset,
      replayUrl: cell.replayUrl,
      board: {
        width: cell.width,
        height: cell.height,
        mineCount: cell.mineCount,
        replayScanBudget: cell.replayScanBudget,
        configuredScanBudget: cell.configuredScanBudget
      },
      sensor: {
        sigma: cell.sigma,
        sigmaNoise: cell.sigmaNoise,
        dropoutRate: cell.dropoutRate,
        delaySteps: cell.delaySteps
      }
    })),
    artifacts: {
      summary: "summary.json",
      trialOutcomes: "trial-outcomes.csv",
      posteriorDiagnostics: "posterior-diagnostics.csv",
      bayesActions: "bayes-actions.csv",
      observationParity: "observation-parity.jsonl",
      posteriorMap: "frontier-posterior-map.json",
      regret: "bayes-regret.csv",
      regretSummary: "bayes-regret-summary.csv"
    },
    audits: {
      observationAdmissionLeakFree: leakFree,
      bayesModesKeptOutOfImplementedRegistry: args.modes
        .filter((mode) => BAYES_MODES.has(mode))
        .every((mode) => MINES_CONTROLLER_MODES[mode]?.status === "pending")
    }
  };

  await writeJson(path.join(outDir, "summary.json"), summary);
  await writeJson(path.join(outDir, "manifest.json"), manifest);
  await writeFile(path.join(outDir, "trial-outcomes.csv"), rowsToCsv(trialOutcomes, CSV_HEADERS.trialOutcomes), "utf8");
  await writeFile(path.join(outDir, "posterior-diagnostics.csv"), rowsToCsv(posteriorDiagnostics, CSV_HEADERS.posteriorDiagnostics), "utf8");
  await writeFile(path.join(outDir, "bayes-actions.csv"), rowsToCsv(bayesActions, CSV_HEADERS.bayesActions), "utf8");
  await writeFile(path.join(outDir, "bayes-regret.csv"), rowsToCsv(regretRows, CSV_HEADERS.regret), "utf8");
  await writeFile(path.join(outDir, "bayes-regret-summary.csv"), rowsToCsv(regretSummary, CSV_HEADERS.regretSummary), "utf8");
  await writeFile(
    path.join(outDir, "observation-parity.jsonl"),
    `${observationParity.map((row) => JSON.stringify(row)).join("\n")}\n`,
    "utf8"
  );
  await writeJson(path.join(outDir, "frontier-posterior-map.json"), {
    schema: "sundog.mines.phase12-bayes-smoke.frontier-posterior-map.v1",
    phase: args.phase,
    entries: posteriorMap
  });

  console.log(
    `Mines ${args.phase}: ${trialOutcomes.length} trials in ${wallSeconds.toFixed(3)}s (${(trialOutcomes.length / Math.max(0.001, wallSeconds)).toFixed(2)} trials/s)`
  );
  console.log(`Audits: observation admission leak-free ${leakFree ? "pass" : "FAIL"}`);
  console.log(`Wrote ${path.relative(REPO_ROOT, outDir)}`);
  for (const row of regretSummary) {
    console.log(
      `${row.selection} ${row.bayes_mode} vs ${row.target_mode}: mean budget delta ${row.mean_budget_adjusted_delta}, n ${row.n}`
    );
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack : String(error));
  process.exit(1);
});
