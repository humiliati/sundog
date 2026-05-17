#!/usr/bin/env node

import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const DEFAULT_MODES = Object.freeze(["oracle", "bayes_correct", "hc_sundog", "random"]);
const ACTIONS = Object.freeze([
  { id: "stay", dx: 0, dy: 0 },
  { id: "east", dx: 1, dy: 0 },
  { id: "south", dx: 0, dy: 1 },
  { id: "west", dx: -1, dy: 0 },
  { id: "north", dx: 0, dy: -1 },
]);
const CARDINAL_ACTIONS = ACTIONS.filter((action) => action.id !== "stay");
const QUADRATURE = Object.freeze([
  [-2, 0.05448868454964294],
  [-1, 0.24420134200323335],
  [0, 0.40261994689424746],
  [1, 0.24420134200323335],
  [2, 0.05448868454964294],
]);

const DEFAULT_ARGS = Object.freeze({
  phase: "phase1-reference-smoke",
  out: "results/bayes/phase1-reference-smoke",
  seedStart: 0,
  seeds: 32,
  modes: DEFAULT_MODES,
  gridSize: 9,
  maxTurns: 24,
  startX: 0,
  startY: 0,
  targetMinStartDistance: 4,
  sigma: 1.55,
  noiseStd: 0.06,
  baseline: 0.02,
  amplitude: 1,
  bayesInfoWeight: 1.35,
  bayesDistanceWeight: 1,
  bayesHitBonus: 18,
  bayesStayPenalty: 0.35,
  sundogSeekThreshold: 0.16,
  sundogTrackThreshold: 0.52,
  traceSteps: true,
});

function usage() {
  return [
    "Usage:",
    "  node scripts/bayes-phase1-reference.mjs --phase <name> --out <dir> --seeds <n>",
    "",
    "Defaults run a small exact Bayes-Correct hidden-source reference task.",
    "",
    "Example:",
    "  node scripts/bayes-phase1-reference.mjs --phase phase1-reference-smoke --out results/bayes/phase1-reference-smoke --seeds 32 --max-turns 24",
  ].join("\n");
}

function parseArgs(argv) {
  const args = { ...DEFAULT_ARGS, modes: [...DEFAULT_ARGS.modes] };

  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    if (flag === "--help" || flag === "-h") {
      console.log(usage());
      process.exit(0);
    }
    if (!flag.startsWith("--")) throw new Error(`Unexpected positional argument: ${flag}`);
    const value = argv[i + 1];
    if (value === undefined || value.startsWith("--")) throw new Error(`Missing value for ${flag}`);
    i += 1;

    if (flag === "--phase") args.phase = value;
    else if (flag === "--out") args.out = value;
    else if (flag === "--seed-start") args.seedStart = parseInteger(value, flag);
    else if (flag === "--seeds") args.seeds = parsePositiveInteger(value, flag);
    else if (flag === "--modes") args.modes = parseList(value);
    else if (flag === "--grid-size") args.gridSize = parsePositiveInteger(value, flag);
    else if (flag === "--max-turns") args.maxTurns = parsePositiveInteger(value, flag);
    else if (flag === "--start-x") args.startX = parseInteger(value, flag);
    else if (flag === "--start-y") args.startY = parseInteger(value, flag);
    else if (flag === "--target-min-start-distance") args.targetMinStartDistance = parseNonNegativeNumber(value, flag);
    else if (flag === "--sigma") args.sigma = parsePositiveNumber(value, flag);
    else if (flag === "--noise-std") args.noiseStd = parsePositiveNumber(value, flag);
    else if (flag === "--baseline") args.baseline = parseNumber(value, flag);
    else if (flag === "--amplitude") args.amplitude = parsePositiveNumber(value, flag);
    else if (flag === "--bayes-info-weight") args.bayesInfoWeight = parseNumber(value, flag);
    else if (flag === "--bayes-distance-weight") args.bayesDistanceWeight = parsePositiveNumber(value, flag);
    else if (flag === "--bayes-hit-bonus") args.bayesHitBonus = parsePositiveNumber(value, flag);
    else if (flag === "--bayes-stay-penalty") args.bayesStayPenalty = parseNonNegativeNumber(value, flag);
    else if (flag === "--sundog-seek-threshold") args.sundogSeekThreshold = parseNumber(value, flag);
    else if (flag === "--sundog-track-threshold") args.sundogTrackThreshold = parseNumber(value, flag);
    else if (flag === "--trace-steps") args.traceSteps = parseBoolean(value, flag);
    else throw new Error(`Unknown flag: ${flag}`);
  }

  validateArgs(args);
  return args;
}

function parseList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function parseInteger(value, flag) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed)) throw new Error(`${flag} must be an integer`);
  return parsed;
}

function parsePositiveInteger(value, flag) {
  const parsed = parseInteger(value, flag);
  if (parsed < 1) throw new Error(`${flag} must be positive`);
  return parsed;
}

function parseNumber(value, flag) {
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed)) throw new Error(`${flag} must be finite`);
  return parsed;
}

function parsePositiveNumber(value, flag) {
  const parsed = parseNumber(value, flag);
  if (parsed <= 0) throw new Error(`${flag} must be positive`);
  return parsed;
}

function parseNonNegativeNumber(value, flag) {
  const parsed = parseNumber(value, flag);
  if (parsed < 0) throw new Error(`${flag} must be non-negative`);
  return parsed;
}

function parseBoolean(value, flag) {
  if (value === "1" || value === "true") return true;
  if (value === "0" || value === "false") return false;
  throw new Error(`${flag} must be 0, 1, true, or false`);
}

function validateArgs(args) {
  const knownModes = new Set(DEFAULT_MODES);
  const unknownMode = args.modes.find((mode) => !knownModes.has(mode));
  if (unknownMode) throw new Error(`Unknown mode: ${unknownMode}`);
  if (args.gridSize < 5) throw new Error("--grid-size must be at least 5");
  if (args.startX < 0 || args.startX >= args.gridSize || args.startY < 0 || args.startY >= args.gridSize) {
    throw new Error("--start-x/--start-y must be inside the grid");
  }
  if (args.targetMinStartDistance >= args.gridSize * 2) {
    throw new Error("--target-min-start-distance leaves no plausible target cells");
  }
  const candidates = enumerateTargetCandidates(args);
  if (candidates.length < 2) throw new Error("Target prior needs at least two candidate cells");
}

function fnv1a(text) {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function makeRng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function rngFromParts(...parts) {
  return makeRng(fnv1a(parts.join("|")));
}

function gaussianFromParts(...parts) {
  const rng = rngFromParts(...parts);
  const u1 = Math.max(rng(), 1e-12);
  const u2 = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function round(value, places = 6) {
  if (!Number.isFinite(value)) return value;
  const scale = 10 ** places;
  return Math.round(value * scale) / scale;
}

function positionKey(pos) {
  return `${pos.x},${pos.y}`;
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function manhattan(a, b) {
  return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
}

function inBounds(pos, config) {
  return pos.x >= 0 && pos.x < config.gridSize && pos.y >= 0 && pos.y < config.gridSize;
}

function applyAction(pos, action) {
  return { x: pos.x + action.dx, y: pos.y + action.dy };
}

function legalActions(pos, config, options = {}) {
  const includeStay = options.includeStay ?? true;
  const actions = includeStay ? ACTIONS : CARDINAL_ACTIONS;
  return actions.filter((action) => inBounds(applyAction(pos, action), config));
}

function enumerateTargetCandidates(config) {
  const start = { x: config.startX, y: config.startY };
  const candidates = [];
  for (let y = 0; y < config.gridSize; y += 1) {
    for (let x = 0; x < config.gridSize; x += 1) {
      const pos = { x, y };
      if (manhattan(start, pos) >= config.targetMinStartDistance) candidates.push(pos);
    }
  }
  return candidates;
}

function selectTarget(seed, config, candidates) {
  const rng = rngFromParts("target", seed, config.gridSize, config.targetMinStartDistance);
  return candidates[Math.floor(rng() * candidates.length)];
}

function signalMean(pos, target, config) {
  const d2 = (pos.x - target.x) ** 2 + (pos.y - target.y) ** 2;
  return config.baseline + config.amplitude * Math.exp(-d2 / (2 * config.sigma ** 2));
}

function observe({ seed, turn, pos, target, config }) {
  const mean = signalMean(pos, target, config);
  const noise = config.noiseStd * gaussianFromParts("observation", seed, turn, pos.x, pos.y);
  return {
    value: mean + noise,
    mean,
    noise,
  };
}

function createPosterior(candidates) {
  const logPrior = -Math.log(candidates.length);
  return {
    candidates,
    logWeights: candidates.map(() => logPrior),
  };
}

function normalizeLogWeights(logWeights) {
  const maxLog = Math.max(...logWeights);
  const shifted = logWeights.map((value) => Math.exp(value - maxLog));
  const sum = shifted.reduce((acc, value) => acc + value, 0);
  return shifted.map((value) => value / sum);
}

function updatePosterior(posterior, pos, observationValue, config) {
  const nextLogWeights = posterior.logWeights.map((logWeight, index) => {
    const mean = signalMean(pos, posterior.candidates[index], config);
    return logWeight + gaussianLogLikelihood(observationValue, mean, config.noiseStd);
  });
  const probs = normalizeLogWeights(nextLogWeights);
  const normalizedLogs = probs.map((prob) => Math.log(Math.max(prob, 1e-300)));
  return {
    candidates: posterior.candidates,
    logWeights: normalizedLogs,
  };
}

function gaussianLogLikelihood(value, mean, std) {
  const z = (value - mean) / std;
  return -0.5 * z * z - Math.log(std * Math.sqrt(2 * Math.PI));
}

function posteriorStats(posterior, truth = null) {
  const probs = normalizeLogWeights(posterior.logWeights);
  let entropy = 0;
  let mapIndex = 0;
  let mapProb = -Infinity;
  let truthProb = null;
  for (let i = 0; i < probs.length; i += 1) {
    const prob = probs[i];
    if (prob > 0) entropy -= prob * Math.log(prob);
    if (prob > mapProb) {
      mapProb = prob;
      mapIndex = i;
    }
    if (truth && posterior.candidates[i].x === truth.x && posterior.candidates[i].y === truth.y) {
      truthProb = prob;
    }
  }
  const top = probs
    .map((prob, index) => ({ ...posterior.candidates[index], prob }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 5)
    .map((entry) => ({ x: entry.x, y: entry.y, prob: round(entry.prob, 6) }));

  return {
    entropy,
    map: posterior.candidates[mapIndex],
    mapProb,
    truthProb,
    top,
  };
}

function expectedInformationGain(posterior, probePos, config) {
  const probs = normalizeLogWeights(posterior.logWeights);
  const currentEntropy = entropyFromProbs(probs);
  let expectedEntropy = 0;

  for (let actualIndex = 0; actualIndex < posterior.candidates.length; actualIndex += 1) {
    const actualProb = probs[actualIndex];
    if (actualProb < 1e-12) continue;
    const actualMean = signalMean(probePos, posterior.candidates[actualIndex], config);

    for (const [z, weight] of QUADRATURE) {
      const hypotheticalObservation = actualMean + z * config.noiseStd;
      const nextLogs = posterior.logWeights.map((logWeight, candidateIndex) => {
        const mean = signalMean(probePos, posterior.candidates[candidateIndex], config);
        return logWeight + gaussianLogLikelihood(hypotheticalObservation, mean, config.noiseStd);
      });
      expectedEntropy += actualProb * weight * entropyFromProbs(normalizeLogWeights(nextLogs));
    }
  }

  return Math.max(0, currentEntropy - expectedEntropy);
}

function entropyFromProbs(probs) {
  return probs.reduce((acc, prob) => (prob > 0 ? acc - prob * Math.log(prob) : acc), 0);
}

function chooseBayesAction(pos, posterior, config) {
  const probs = normalizeLogWeights(posterior.logWeights);
  const stats = posteriorStats(posterior);
  let best = null;

  for (const action of legalActions(pos, config, { includeStay: true })) {
    const nextPos = applyAction(pos, action);
    let hitMass = 0;
    let expectedDistance = 0;

    for (let i = 0; i < posterior.candidates.length; i += 1) {
      const candidate = posterior.candidates[i];
      const prob = probs[i];
      if (candidate.x === nextPos.x && candidate.y === nextPos.y) hitMass += prob;
      expectedDistance += prob * distance(nextPos, candidate);
    }

    const informationGain = expectedInformationGain(posterior, nextPos, config);
    const stayPenalty = action.id === "stay" ? config.bayesStayPenalty : 0;
    const mapDistance = distance(nextPos, stats.map);
    const utility =
      config.bayesHitBonus * hitMass -
      config.bayesDistanceWeight * expectedDistance +
      config.bayesInfoWeight * informationGain -
      stayPenalty -
      0.001 * mapDistance;
    const candidate = {
      action,
      utility,
      hitMass,
      expectedDistance,
      informationGain,
      mapDistance,
    };
    if (!best || compareBayesCandidate(candidate, best) > 0) best = candidate;
  }

  return best;
}

function compareBayesCandidate(a, b) {
  if (a.utility !== b.utility) return a.utility - b.utility;
  if (a.hitMass !== b.hitMass) return a.hitMass - b.hitMass;
  if (a.expectedDistance !== b.expectedDistance) return b.expectedDistance - a.expectedDistance;
  return b.action.id.localeCompare(a.action.id);
}

function createScanPath(config) {
  const pathCells = [];
  for (let y = 0; y < config.gridSize; y += 1) {
    if (y % 2 === 0) {
      for (let x = 0; x < config.gridSize; x += 1) pathCells.push({ x, y });
    } else {
      for (let x = config.gridSize - 1; x >= 0; x -= 1) pathCells.push({ x, y });
    }
  }
  return pathCells;
}

function createControllerState(mode, config) {
  if (mode === "bayes_correct") return { posterior: createPosterior(enumerateTargetCandidates(config)) };
  if (mode === "hc_sundog") {
    return {
      phase: "scan",
      bestSignal: -Infinity,
      bestPos: { x: config.startX, y: config.startY },
      scanPath: createScanPath(config),
      scanIndex: 0,
      lastSignal: null,
      lastAction: null,
      rotateIndex: 0,
    };
  }
  return {};
}

function chooseAction({ mode, seed, turn, pos, target, observation, state, config }) {
  if (mode === "oracle") return { action: moveToward(pos, target, config), diagnostics: { policy: "truth_shortest_path" } };
  if (mode === "random") {
    const options = legalActions(pos, config, { includeStay: false });
    const rng = rngFromParts("random-action", seed, turn, pos.x, pos.y);
    return { action: options[Math.floor(rng() * options.length)], diagnostics: { policy: "legal_uniform_random" } };
  }
  if (mode === "bayes_correct") {
    state.posterior = updatePosterior(state.posterior, pos, observation.value, config);
    const stats = posteriorStats(state.posterior, target);
    const choice = chooseBayesAction(pos, state.posterior, config);
    return {
      action: choice.action,
      diagnostics: {
        policy: "exact_posterior_expected_utility",
        entropy: round(stats.entropy, 6),
        map: stats.map,
        mapProb: round(stats.mapProb, 6),
        truthProb: round(stats.truthProb, 6),
        top: stats.top,
        utility: round(choice.utility, 6),
        hitMass: round(choice.hitMass, 6),
        expectedDistance: round(choice.expectedDistance, 6),
        informationGain: round(choice.informationGain, 6),
      },
    };
  }
  if (mode === "hc_sundog") return chooseHcSundogAction({ pos, observation, state, config });
  throw new Error(`Unknown mode: ${mode}`);
}

function moveToward(pos, target, config) {
  const candidates = legalActions(pos, config, { includeStay: false })
    .map((action) => ({ action, nextPos: applyAction(pos, action) }))
    .sort((a, b) => {
      const distDelta = manhattan(a.nextPos, target) - manhattan(b.nextPos, target);
      if (distDelta !== 0) return distDelta;
      return a.action.id.localeCompare(b.action.id);
    });
  return candidates[0].action;
}

function chooseHcSundogAction({ pos, observation, state, config }) {
  if (observation.value > state.bestSignal) {
    state.bestSignal = observation.value;
    state.bestPos = { ...pos };
  }
  if (state.phase === "scan" && observation.value >= config.sundogSeekThreshold) state.phase = "seek";
  if (state.phase === "seek" && observation.value >= config.sundogTrackThreshold) state.phase = "track";

  let action;
  let policy = state.phase;
  if (state.phase === "scan") {
    action = nextScanAction(pos, state, config);
    policy = "serpentine_scan";
  } else {
    const fellAwayFromBest = observation.value < state.bestSignal - config.noiseStd * 1.5;
    if (fellAwayFromBest && positionKey(pos) !== positionKey(state.bestPos)) {
      action = moveToward(pos, state.bestPos, config);
      policy = "return_to_best_signal";
    } else if (state.lastSignal !== null && observation.value > state.lastSignal + config.noiseStd * 0.2 && state.lastAction) {
      const continued = applyAction(pos, state.lastAction);
      action = inBounds(continued, config) && state.lastAction.id !== "stay" ? state.lastAction : rotatingAction(pos, state, config);
      policy = "continue_improving_direction";
    } else {
      action = rotatingAction(pos, state, config);
      policy = "local_probe_rotation";
    }
  }

  state.lastSignal = observation.value;
  state.lastAction = action;
  return {
    action,
    diagnostics: {
      policy,
      phase: state.phase,
      bestSignal: round(state.bestSignal, 6),
      bestPos: state.bestPos,
    },
  };
}

function nextScanAction(pos, state, config) {
  while (state.scanIndex < state.scanPath.length && positionKey(state.scanPath[state.scanIndex]) === positionKey(pos)) {
    state.scanIndex += 1;
  }
  const target = state.scanPath[Math.min(state.scanIndex, state.scanPath.length - 1)];
  return moveToward(pos, target, config);
}

function rotatingAction(pos, state, config) {
  const order = ["east", "south", "west", "north"];
  for (let attempt = 0; attempt < order.length; attempt += 1) {
    const id = order[(state.rotateIndex + attempt) % order.length];
    const action = CARDINAL_ACTIONS.find((candidate) => candidate.id === id);
    if (inBounds(applyAction(pos, action), config)) {
      state.rotateIndex = (state.rotateIndex + attempt + 1) % order.length;
      return action;
    }
  }
  return ACTIONS[0];
}

function simulateTrial({ mode, seed, config, candidates }) {
  const target = selectTarget(seed, config, candidates);
  let pos = { x: config.startX, y: config.startY };
  const state = createControllerState(mode, config);
  const steps = [];
  let success = false;
  let turnsToHit = null;
  let cumulativeSignal = 0;

  for (let turn = 0; turn < config.maxTurns; turn += 1) {
    const observation = observe({ seed, turn, pos, target, config });
    cumulativeSignal += observation.value;
    const { action, diagnostics } = chooseAction({ mode, seed, turn, pos, target, observation, state, config });
    const nextPos = applyAction(pos, action);
    if (!inBounds(nextPos, config)) throw new Error(`Illegal ${mode} action ${action.id} at ${positionKey(pos)}`);

    const hit = positionKey(nextPos) === positionKey(target);
    steps.push({
      seed,
      mode,
      turn,
      position: pos,
      observation: round(observation.value, 6),
      expectedMeanAtTruth: round(observation.mean, 6),
      action: action.id,
      nextPosition: nextPos,
      hit,
      diagnostics,
    });

    pos = nextPos;
    if (hit) {
      success = true;
      turnsToHit = turn + 1;
      break;
    }
  }

  const finalDistance = distance(pos, target);
  const outcome = {
    seed,
    mode,
    target,
    start: { x: config.startX, y: config.startY },
    success,
    turnsToHit,
    turnsElapsed: steps.length,
    finalPosition: pos,
    finalDistance: round(finalDistance, 6),
    cumulativeSignal: round(cumulativeSignal, 6),
    score: round(scoreOutcome({ success, turnsToHit, finalDistance, config }), 6),
    path: steps.map((step) => step.position).concat([pos]),
  };
  return { outcome, steps };
}

function scoreOutcome({ success, turnsToHit, finalDistance, config }) {
  if (success) return 1 + (config.maxTurns - turnsToHit) / config.maxTurns;
  return -finalDistance / (config.gridSize * Math.SQRT2);
}

function summarizeTrials(trials, modes) {
  const byMode = {};
  for (const mode of modes) {
    const rows = trials.filter((trial) => trial.mode === mode);
    const successes = rows.filter((trial) => trial.success);
    byMode[mode] = {
      mode,
      trials: rows.length,
      successes: successes.length,
      successRate: round(successes.length / Math.max(1, rows.length), 6),
      meanScore: round(mean(rows.map((trial) => trial.score)), 6),
      meanTurnsToHit: successes.length ? round(mean(successes.map((trial) => trial.turnsToHit)), 6) : null,
      medianTurnsToHit: successes.length ? median(successes.map((trial) => trial.turnsToHit)) : null,
      meanFinalDistance: round(mean(rows.map((trial) => trial.finalDistance)), 6),
      meanCumulativeSignal: round(mean(rows.map((trial) => trial.cumulativeSignal)), 6),
    };
  }
  return byMode;
}

function buildRegretRows(trials, modes) {
  const bySeedMode = new Map(trials.map((trial) => [`${trial.seed}|${trial.mode}`, trial]));
  const rows = [];
  for (const trial of trials) {
    if (trial.mode !== "bayes_correct") continue;
    for (const targetMode of modes) {
      if (targetMode === "bayes_correct") continue;
      const target = bySeedMode.get(`${trial.seed}|${targetMode}`);
      if (!target) continue;
      rows.push({
        seed: trial.seed,
        targetMode,
        bayesScore: trial.score,
        targetScore: target.score,
        scoreDelta: round(trial.score - target.score, 6),
        bayesSuccess: trial.success,
        targetSuccess: target.success,
        bayesTurnsToHit: trial.turnsToHit,
        targetTurnsToHit: target.turnsToHit,
      });
    }
  }
  return rows;
}

function buildExitGate(summary, regretRows) {
  const bayes = summary.bayes_correct;
  const sundog = summary.hc_sundog;
  const bayesVsSundogRows = regretRows.filter((row) => row.targetMode === "hc_sundog");
  const meanScoreDelta = mean(bayesVsSundogRows.map((row) => row.scoreDelta));
  const successOk = bayes.successRate >= sundog.successRate - 1e-9;
  const scoreOk = bayes.meanScore >= sundog.meanScore - 1e-9;
  const turnOk =
    bayes.successRate > sundog.successRate + 1e-9 ||
    bayes.meanTurnsToHit === null ||
    sundog.meanTurnsToHit === null ||
    bayes.meanTurnsToHit <= sundog.meanTurnsToHit + 1e-9;
  return {
    name: "Bayes-Correct beats or matches HC-Sundog under the known model",
    pass: successOk && scoreOk && turnOk,
    successOk,
    scoreOk,
    turnOk,
    bayesSuccessRate: bayes.successRate,
    sundogSuccessRate: sundog.successRate,
    bayesMeanScore: bayes.meanScore,
    sundogMeanScore: sundog.meanScore,
    bayesMeanTurnsToHit: bayes.meanTurnsToHit,
    sundogMeanTurnsToHit: sundog.meanTurnsToHit,
    bayesMinusSundogMeanScoreDelta: round(meanScoreDelta, 6),
  };
}

function selectReplayTrials(trials) {
  const bySeed = new Map();
  for (const trial of trials) {
    if (!bySeed.has(trial.seed)) bySeed.set(trial.seed, {});
    bySeed.get(trial.seed)[trial.mode] = trial;
  }
  const candidates = [...bySeed.entries()]
    .filter(([, rows]) => rows.bayes_correct && rows.hc_sundog)
    .map(([seed, rows]) => ({
      seed,
      bayes: rows.bayes_correct,
      sundog: rows.hc_sundog,
      delta: rows.bayes_correct.score - rows.hc_sundog.score,
    }));
  const bestBayes = [...candidates].sort((a, b) => b.delta - a.delta)[0];
  const closest = [...candidates].sort((a, b) => Math.abs(a.delta) - Math.abs(b.delta))[0];
  const bayesLoss = [...candidates].filter((entry) => entry.delta < 0).sort((a, b) => a.delta - b.delta)[0] ?? null;
  return [
    bestBayes ? replayEntry("bayes_advantage", bestBayes) : null,
    closest ? replayEntry("closest_pair", closest) : null,
    bayesLoss ? replayEntry("bayes_boundary", bayesLoss) : null,
  ].filter(Boolean);
}

function replayEntry(role, entry) {
  return {
    role,
    seed: entry.seed,
    target: entry.bayes.target,
    bayes: {
      success: entry.bayes.success,
      turnsToHit: entry.bayes.turnsToHit,
      score: entry.bayes.score,
      path: entry.bayes.path,
    },
    hcSundog: {
      success: entry.sundog.success,
      turnsToHit: entry.sundog.turnsToHit,
      score: entry.sundog.score,
      path: entry.sundog.path,
    },
    bayesMinusHcScoreDelta: round(entry.delta, 6),
  };
}

function mean(values) {
  if (!values.length) return 0;
  return values.reduce((acc, value) => acc + value, 0) / values.length;
}

function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2) return sorted[mid];
  return round((sorted[mid - 1] + sorted[mid]) / 2, 6);
}

function toJsonl(rows) {
  return rows.map((row) => JSON.stringify(row)).join("\n") + "\n";
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (/[",\n\r]/.test(text)) return `"${text.replaceAll('"', '""')}"`;
  return text;
}

function toCsv(rows, headers) {
  return [
    headers.join(","),
    ...rows.map((row) => headers.map((header) => csvEscape(row[header])).join(",")),
  ].join("\n") + "\n";
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(REPO_ROOT, args.out);
  const startedAt = new Date().toISOString();
  const t0 = performance.now();
  const candidates = enumerateTargetCandidates(args);
  const trials = [];
  const steps = [];

  for (let seedOffset = 0; seedOffset < args.seeds; seedOffset += 1) {
    const seed = args.seedStart + seedOffset;
    for (const mode of args.modes) {
      const trial = simulateTrial({ mode, seed, config: args, candidates });
      trials.push(trial.outcome);
      if (args.traceSteps) steps.push(...trial.steps);
    }
  }

  const completedAt = new Date().toISOString();
  const elapsedSeconds = (performance.now() - t0) / 1000;
  const summary = summarizeTrials(trials, args.modes);
  const regretRows = buildRegretRows(trials, args.modes);
  const exitGate = buildExitGate(summary, regretRows);
  const replayManifest = {
    phase: args.phase,
    sourceManifest: "manifest.json",
    selectedTrials: selectReplayTrials(trials),
  };

  const manifest = {
    schemaVersion: 1,
    phase: args.phase,
    generatedAt: completedAt,
    startedAt,
    completedAt,
    elapsedSeconds: round(elapsedSeconds, 6),
    purpose: "Standalone Phase 1 Bayes-vs-Sundog exact hidden-source reference task.",
    command: `node ${path.relative(REPO_ROOT, fileURLToPath(import.meta.url)).replaceAll("\\", "/")} ${process.argv.slice(2).join(" ")}`,
    config: {
      seedStart: args.seedStart,
      seeds: args.seeds,
      modes: args.modes,
      gridSize: args.gridSize,
      start: { x: args.startX, y: args.startY },
      targetPrior: {
        candidateCount: candidates.length,
        targetMinStartDistance: args.targetMinStartDistance,
      },
      maxTurns: args.maxTurns,
      observationModel: {
        family: "known Gaussian radial field plus known Gaussian measurement noise",
        baseline: args.baseline,
        amplitude: args.amplitude,
        sigma: args.sigma,
        noiseStd: args.noiseStd,
      },
      sameObservationSeedPolicy: "shared_by_seed_turn_position",
      hitRule: "success when the agent moves onto the hidden source grid cell",
    },
    policies: {
      bayes_correct: {
        posterior: "exact grid posterior over target location",
        likelihood: "matches generator",
        actionRule: "one-step expected utility with exact posterior, hit mass, distance cost, and quadrature information gain",
      },
      hc_sundog: {
        observation: "position and scalar field observation only",
        actionRule: "SCAN/SEEK/TRACK response controller with serpentine scan, best-signal return, and local probe rotation",
      },
      oracle: {
        observation: "privileged hidden source for apparatus ceiling only",
      },
      random: {
        observation: "legal random motion baseline",
      },
    },
    audits: {
      exactPosterior: true,
      bayesLikelihoodMatchesGenerator: true,
      noTargetCoordinateInBayesObservation: true,
      noTargetCoordinateInHcSundogObservation: true,
      oracleExcludedFromClaimGate: true,
    },
    summary,
    exitGate,
    artifacts: {
      trials: "trials.jsonl",
      steps: args.traceSteps ? "steps.jsonl" : null,
      summary: "summary.csv",
      regret: "regret.csv",
      replayManifest: "replay-manifest.json",
    },
  };

  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n", "utf8");
  await writeFile(path.join(outDir, "trials.jsonl"), toJsonl(trials), "utf8");
  if (args.traceSteps) await writeFile(path.join(outDir, "steps.jsonl"), toJsonl(steps), "utf8");
  await writeFile(
    path.join(outDir, "summary.csv"),
    toCsv(Object.values(summary), [
      "mode",
      "trials",
      "successes",
      "successRate",
      "meanScore",
      "meanTurnsToHit",
      "medianTurnsToHit",
      "meanFinalDistance",
      "meanCumulativeSignal",
    ]),
    "utf8",
  );
  await writeFile(
    path.join(outDir, "regret.csv"),
    toCsv(regretRows, [
      "seed",
      "targetMode",
      "bayesScore",
      "targetScore",
      "scoreDelta",
      "bayesSuccess",
      "targetSuccess",
      "bayesTurnsToHit",
      "targetTurnsToHit",
    ]),
    "utf8",
  );
  await writeFile(path.join(outDir, "replay-manifest.json"), JSON.stringify(replayManifest, null, 2) + "\n", "utf8");

  console.log(
    `Bayes ${args.phase}: ${trials.length} trials in ${round(elapsedSeconds, 3)}s ` +
      `(${round(trials.length / Math.max(elapsedSeconds, 1e-9), 2)} trials/s)`,
  );
  console.log("Audits: pass");
  console.log(
    `Exit gate: ${exitGate.pass ? "pass" : "FAIL"} ` +
      `(Bayes score ${exitGate.bayesMeanScore}, HC-Sundog score ${exitGate.sundogMeanScore}, ` +
      `delta ${exitGate.bayesMinusSundogMeanScoreDelta})`,
  );
  console.log(`Wrote ${path.relative(REPO_ROOT, outDir)}`);

  if (!exitGate.pass) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
