// Sundog Pressure Mines — Phase 4 baseline/controller lane scaffold.
//
// Pure ES module. No DOM and no file I/O. The browser page and headless
// harness can both consume these lane definitions. Phase 4 keeps cleverness
// deliberately modest: baseline policies and information budgets are declared
// before the Phase 5 Sundog controller tries to win anything.

import { ACTION, TILE } from "./mines-core.mjs";

export const MINES_CONTROLLER_MODES = Object.freeze({
  random_reveal: Object.freeze({
    label: "Passive / random reveal",
    status: "implemented",
    informationBudget: "Public tile/flag/scanned state only; no pressure field, no true occupancy, no exact counts.",
    usesPressure: false,
    usesGradient: false,
    usesConfidence: false,
    usesActionHistory: false,
    usesScan: false,
    usesPrivileged: false,
  }),
  naive_pressure: Object.freeze({
    label: "Naive local pressure",
    status: "implemented",
    informationBudget: "Public frontier plus current observed pressure. Ignores confidence, gradients, scans, and action history.",
    usesPressure: true,
    usesGradient: false,
    usesConfidence: false,
    usesActionHistory: false,
    usesScan: false,
    usesPrivileged: false,
  }),
  threshold_flagger: Object.freeze({
    label: "Naive threshold flagger",
    status: "implemented",
    informationBudget: "Public frontier plus current observed pressure and a fixed pressure threshold. No memory or privileged state.",
    usesPressure: true,
    usesGradient: false,
    usesConfidence: false,
    usesActionHistory: false,
    usesScan: false,
    usesPrivileged: false,
  }),
  naive_pressure_shuffled: Object.freeze({
    label: "Shuffled-pressure ablation",
    status: "implemented",
    informationBudget: "Same public frontier as naive_pressure, but pressure values are deterministically shuffled within the board.",
    usesPressure: true,
    usesGradient: false,
    usesConfidence: false,
    usesActionHistory: false,
    usesScan: false,
    usesPrivileged: false,
    ablation: "shuffled_pressure",
  }),
  naive_pressure_delayed: Object.freeze({
    label: "Delayed-pressure ablation",
    status: "implemented",
    informationBudget: "Same public frontier as naive_pressure, but the harness applies a two-turn observed-field delay.",
    usesPressure: true,
    usesGradient: false,
    usesConfidence: false,
    usesActionHistory: false,
    usesScan: false,
    usesPrivileged: false,
    sensorOverride: Object.freeze({ delaySteps: 2 }),
    ablation: "delayed_pressure",
  }),
  oracle_safe: Object.freeze({
    label: "Privileged oracle",
    status: "implemented",
    informationBudget: "Privileged true occupancy and exact adjacency. Diagnostic ceiling only; not a Sundog-legal input.",
    usesPressure: false,
    usesGradient: false,
    usesConfidence: false,
    usesActionHistory: false,
    usesScan: false,
    usesPrivileged: true,
  }),
  sundog_controller: Object.freeze({
    label: "Sundog controller",
    status: "phase5_pending",
    informationBudget: "Pressure residuals, gradients, confidence, reveal/action history, and bounded scans. No true occupancy or exact counts.",
    usesPressure: true,
    usesGradient: true,
    usesConfidence: true,
    usesActionHistory: true,
    usesScan: true,
    usesPrivileged: false,
  }),
  sundog_no_gradient: Object.freeze({
    label: "Sundog ablation: no gradient",
    status: "phase5_pending",
    informationBudget: "Sundog budget with gradient channels removed.",
    usesPressure: true,
    usesGradient: false,
    usesConfidence: true,
    usesActionHistory: true,
    usesScan: true,
    usesPrivileged: false,
    ablation: "no_gradient",
  }),
  sundog_no_scan: Object.freeze({
    label: "Sundog ablation: no scan",
    status: "phase5_pending",
    informationBudget: "Sundog budget with active scan disabled and scan budget fixed at zero.",
    usesPressure: true,
    usesGradient: true,
    usesConfidence: true,
    usesActionHistory: true,
    usesScan: false,
    usesPrivileged: false,
    ablation: "no_scan",
  }),
  sundog_no_action_history: Object.freeze({
    label: "Sundog ablation: no action history",
    status: "phase5_pending",
    informationBudget: "Sundog sensor channels with prior reveal/flag/scan action history masked.",
    usesPressure: true,
    usesGradient: true,
    usesConfidence: true,
    usesActionHistory: false,
    usesScan: true,
    usesPrivileged: false,
    ablation: "no_action_history",
  }),
  sundog_no_confidence_gate: Object.freeze({
    label: "Sundog ablation: no confidence gate",
    status: "phase5_pending",
    informationBudget: "Sundog channels with confidence/dropout gating ignored.",
    usesPressure: true,
    usesGradient: true,
    usesConfidence: false,
    usesActionHistory: true,
    usesScan: true,
    usesPrivileged: false,
    ablation: "no_confidence_gate",
  }),
});

export const IMPLEMENTED_MINES_MODES = Object.freeze(
  Object.entries(MINES_CONTROLLER_MODES)
    .filter(([, definition]) => definition.status === "implemented")
    .map(([mode]) => mode),
);

function indexOf(width, x, y) {
  return y * width + x;
}

function coordsOf(width, index) {
  return { x: index % width, y: Math.floor(index / width) };
}

function adjacentIndices(width, height, index) {
  const x = index % width;
  const y = Math.floor(index / width);
  const out = [];
  for (let dy = -1; dy <= 1; dy += 1) {
    for (let dx = -1; dx <= 1; dx += 1) {
      if (dx === 0 && dy === 0) continue;
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      out.push(indexOf(width, nx, ny));
    }
  }
  return out;
}

export function concealedUnflaggedIndices(memory) {
  const out = [];
  for (let idx = 0; idx < memory.tiles.length; idx += 1) {
    if (memory.tiles[idx] === TILE.CONCEALED && !memory.flags[idx]) out.push(idx);
  }
  return out;
}

export function frontierIndices(memory) {
  const concealed = concealedUnflaggedIndices(memory);
  const frontier = concealed.filter((idx) => (
    adjacentIndices(memory.width, memory.height, idx)
      .some((neighbor) => memory.tiles[neighbor] === TILE.REVEALED_SAFE)
  ));
  return frontier.length > 0 ? frontier : concealed;
}

function finiteScore(score, fallback = Number.POSITIVE_INFINITY) {
  return Number.isFinite(score) ? score : fallback;
}

function pressureAt(sensor, idx, fallback = Number.POSITIVE_INFINITY) {
  if (!sensor?.observed) return fallback;
  return finiteScore(sensor.observed[idx], fallback);
}

function shuffledPressure(sensor, rng) {
  if (!sensor?.observed) return null;
  const shuffled = Array.from(sensor.observed);
  for (let i = shuffled.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

function pickLowestPressure(indices, scores) {
  let best = null;
  let bestScore = Number.POSITIVE_INFINITY;
  for (const idx of indices) {
    const score = finiteScore(scores[idx]);
    if (score < bestScore || (score === bestScore && (best === null || idx < best))) {
      best = idx;
      bestScore = score;
    }
  }
  return best;
}

function randomReveal(memory, rng) {
  const candidates = concealedUnflaggedIndices(memory);
  if (candidates.length === 0) return { type: ACTION.ABSTAIN };
  const idx = candidates[Math.floor(rng() * candidates.length)];
  return { type: ACTION.REVEAL, ...coordsOf(memory.width, idx) };
}

function revealLowestPressure(memory, sensor, rng, { shuffled = false } = {}) {
  const candidates = frontierIndices(memory);
  if (candidates.length === 0) return { type: ACTION.ABSTAIN };
  const scores = shuffled && sensor?.observed
    ? shuffledPressure(sensor, rng)
    : sensor?.observed;
  const idx = scores ? pickLowestPressure(candidates, scores) : candidates[0];
  return { type: ACTION.REVEAL, ...coordsOf(memory.width, idx ?? candidates[0]) };
}

function thresholdFlagger(memory, sensor, options = {}) {
  const threshold = options.threshold ?? 1.2;
  const candidates = frontierIndices(memory);
  if (candidates.length === 0) return { type: ACTION.ABSTAIN };

  let flagIdx = null;
  let flagScore = Number.NEGATIVE_INFINITY;
  for (const idx of candidates) {
    const score = pressureAt(sensor, idx, Number.NEGATIVE_INFINITY);
    if (score >= threshold && score > flagScore) {
      flagIdx = idx;
      flagScore = score;
    }
  }
  if (flagIdx !== null) {
    return { type: ACTION.FLAG, ...coordsOf(memory.width, flagIdx) };
  }
  return revealLowestPressure(memory, sensor, options.rng ?? (() => 0));
}

function oracleSafe(memory, boardState) {
  const candidates = frontierIndices(memory);
  const safe = candidates
    .filter((idx) => boardState.privileged.occupancy[idx] === 0)
    .sort((a, b) => boardState.privileged.adjacency[a] - boardState.privileged.adjacency[b] || a - b);
  if (safe.length > 0) return { type: ACTION.REVEAL, ...coordsOf(memory.width, safe[0]) };

  const mine = candidates.find((idx) => boardState.privileged.occupancy[idx] === 1);
  if (mine !== undefined) return { type: ACTION.FLAG, ...coordsOf(memory.width, mine) };

  return { type: ACTION.ABSTAIN };
}

export function chooseMinesAction({ mode, memory, sensor, boardState, rng, options = {} }) {
  const definition = MINES_CONTROLLER_MODES[mode];
  if (!definition) throw new Error(`Unknown mines controller mode: ${mode}`);
  if (definition.status !== "implemented") {
    throw new Error(`Mines controller mode ${mode} is ${definition.status}, not runnable yet`);
  }

  switch (mode) {
    case "random_reveal":
      return randomReveal(memory, rng);
    case "naive_pressure":
      return revealLowestPressure(memory, sensor, rng);
    case "threshold_flagger":
      return thresholdFlagger(memory, sensor, { ...options, rng });
    case "naive_pressure_shuffled":
      return revealLowestPressure(memory, sensor, rng, { shuffled: true });
    case "naive_pressure_delayed":
      return revealLowestPressure(memory, sensor, rng);
    case "oracle_safe":
      return oracleSafe(memory, boardState);
    default:
      throw new Error(`No action chooser for mode: ${mode}`);
  }
}
