// Sundog Pressure Mines — Phase 1 board core.
//
// Pure ES module. No DOM, no canvas, no globals. Importable from both the
// browser page (mines.html) and the headless harness (scripts/mines-*.mjs).
//
// Phase 1 scope is the board substrate only: deterministic seeded board
// generation, state transitions for reveal / flag / scan / abstain, terminal
// event detection, presets, and a JSONL-friendly sample serializer.
//
// What this module is NOT:
//   * It does NOT compute pressure values. The pressure sensor model lives in
//     Phase 2 and will consume the privileged adjacency / occupancy grids from
//     this module.
//   * It does NOT expose privileged state to controllers. `getPublicMemory`
//     returns the bounded action ledger; controllers must read from there.
//   * It does NOT run controllers or evaluate metrics. Those land in Phase 4+.
//
// Convention follows public/js/balance-core.mjs: frozen defaults, frozen
// presets, named exports, pure functions, deterministic PRNG, and a sample
// serializer for harness logs.

export const DEFAULT_MINES_CONFIG = Object.freeze({
  width: 9,
  height: 9,
  mineCount: 10,
  seed: 1,
  preset: "easy_sparse",
  // First-click safety: classic Minesweeper guarantees the first reveal is
  // never a mine. We default to true so Sundog must beat first-click-luck on
  // every seeded comparison. Set false for harness ablations.
  firstClickSafe: true,
  // Scan budget. 0 disables scanning entirely (revealed via configuration).
  scanBudget: 0,
  // Turn cap (a "turn" is any action the controller takes, including abstain).
  // null disables. Useful for harness sweeps that need a finite trial length.
  turnCap: null,
  // If true, exhausting the scan budget after attempting another scan is a
  // terminal event. If false (default), scan attempts past zero are rejected
  // as illegal actions and do not terminate the trial.
  terminateOnScanExhaustion: false,
  // Whether the controller is allowed to abstain. Some baselines and harness
  // modes need to disable this to force a decisive action each turn.
  allowAbstain: true,
});

export const MINES_PRESETS = Object.freeze({
  easy_sparse: Object.freeze({
    label: "Easy sparse field",
    config: Object.freeze({ width: 9, height: 9, mineCount: 10, scanBudget: 3 }),
  }),
  clustered: Object.freeze({
    label: "Clustered field",
    config: Object.freeze({
      width: 12,
      height: 12,
      mineCount: 24,
      scanBudget: 4,
    }),
    // Generator hint consumed by initializeBoardState. Clustering biases mine
    // placement toward existing mines to produce deceptive pressure plateaus
    // in Phase 2. Strength 0 = uniform, 1 = strong clustering.
    generator: Object.freeze({ clusterStrength: 0.6 }),
  }),
  ambiguous_overlap: Object.freeze({
    label: "Ambiguous overlap field",
    config: Object.freeze({
      width: 12,
      height: 12,
      mineCount: 28,
      scanBudget: 4,
    }),
    generator: Object.freeze({ clusterStrength: 0.85 }),
  }),
  // Noisy-sensor and delayed-sensor presets shape board topology lightly; the
  // sensor knobs themselves live in the Phase 2 sensor model. We pre-declare
  // the names here so the harness manifest is stable across phases.
  noisy_sensor: Object.freeze({
    label: "Noisy sensor field",
    config: Object.freeze({ width: 10, height: 10, mineCount: 15, scanBudget: 4 }),
    generator: Object.freeze({ clusterStrength: 0.2 }),
  }),
  delayed_sensor: Object.freeze({
    label: "Delayed sensor field",
    config: Object.freeze({ width: 10, height: 10, mineCount: 15, scanBudget: 4 }),
    generator: Object.freeze({ clusterStrength: 0.2 }),
  }),
  dense: Object.freeze({
    label: "Dense field",
    config: Object.freeze({ width: 12, height: 12, mineCount: 36, scanBudget: 6 }),
    generator: Object.freeze({ clusterStrength: 0.1 }),
  }),
});

// Mulberry32 — same PRNG family as balance-core. Stable across browser and
// Node, seedable, deterministic, fast enough for batch sweeps.
export function makeRng(seed) {
  let t = seed >>> 0;
  return function rng() {
    t += 0x6d2b79f5;
    let x = t;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function roundNumber(value, digits = 6) {
  if (!Number.isFinite(value)) return value;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

// Tile state constants. Public to controllers and renderers.
export const TILE = Object.freeze({
  CONCEALED: "concealed",
  REVEALED_SAFE: "revealed_safe",
  REVEALED_MINE: "revealed_mine",
  FLAGGED: "flagged",
});

// Action types. Pure data, no functions attached.
export const ACTION = Object.freeze({
  REVEAL: "reveal",
  FLAG: "flag",
  UNFLAG: "unflag",
  SCAN: "scan",
  ABSTAIN: "abstain",
});

export const TERMINAL = Object.freeze({
  MINE_TRIGGERED: "mine_triggered",
  FULL_CLEAR: "full_clear",
  SCAN_BUDGET_EXHAUSTED: "scan_budget_exhausted",
  TURN_CAP: "turn_cap",
});

// ----- config + validation -----

export function normalizeMinesConfig(config = {}) {
  const presetKey = config.preset ?? DEFAULT_MINES_CONFIG.preset;
  const preset = MINES_PRESETS[presetKey] ?? null;
  const merged = {
    ...DEFAULT_MINES_CONFIG,
    ...(preset?.config ?? {}),
    ...config,
  };
  // Preset name is preserved post-merge so downstream logs can name it.
  merged.preset = presetKey;
  // Generator hints come from preset metadata unless caller overrides.
  merged.generator = Object.freeze({
    clusterStrength: 0,
    ...(preset?.generator ?? {}),
    ...(config.generator ?? {}),
  });

  if (!Number.isInteger(merged.width) || merged.width < 2) {
    throw new Error(`width must be integer >= 2, got ${merged.width}`);
  }
  if (!Number.isInteger(merged.height) || merged.height < 2) {
    throw new Error(`height must be integer >= 2, got ${merged.height}`);
  }
  const tileCount = merged.width * merged.height;
  if (!Number.isInteger(merged.mineCount) || merged.mineCount < 1) {
    throw new Error(`mineCount must be integer >= 1, got ${merged.mineCount}`);
  }
  // Leave room for first-click safety + at least one non-mine tile.
  const maxMines = tileCount - (merged.firstClickSafe ? 1 : 0);
  if (merged.mineCount >= maxMines) {
    throw new Error(
      `mineCount ${merged.mineCount} leaves no safe tiles for a ${merged.width}x${merged.height} board (max ${maxMines - 1})`,
    );
  }
  if (!Number.isInteger(merged.scanBudget) || merged.scanBudget < 0) {
    throw new Error(`scanBudget must be integer >= 0, got ${merged.scanBudget}`);
  }
  if (merged.turnCap !== null && (!Number.isInteger(merged.turnCap) || merged.turnCap < 1)) {
    throw new Error(`turnCap must be null or integer >= 1, got ${merged.turnCap}`);
  }
  if (!Number.isInteger(merged.seed)) {
    merged.seed = Math.trunc(merged.seed) >>> 0;
  }
  return Object.freeze(merged);
}

// ----- board generation -----

function indexOf(width, x, y) {
  return y * width + x;
}

function neighborsOf(width, height, x, y) {
  const out = [];
  for (let dy = -1; dy <= 1; dy += 1) {
    for (let dx = -1; dx <= 1; dx += 1) {
      if (dx === 0 && dy === 0) continue;
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      out.push({ x: nx, y: ny, index: indexOf(width, nx, ny) });
    }
  }
  return out;
}

// Placement is deterministic in (seed, config). Clustering, when configured,
// biases each candidate placement toward tiles adjacent to already-placed
// mines, producing the deceptive-plateau geometry Phase 2 needs.
function placeMines(config, rng, forbiddenIndex) {
  const { width, height, mineCount, generator } = config;
  const tileCount = width * height;
  const occupancy = new Uint8Array(tileCount);
  const clusterStrength = clamp(generator?.clusterStrength ?? 0, 0, 1);

  // Build a weighted candidate list each placement step. With clusterStrength
  // = 0 the weights are uniform and we get standard uniform-random placement.
  const placed = [];
  while (placed.length < mineCount) {
    const weights = new Float64Array(tileCount);
    let total = 0;
    for (let idx = 0; idx < tileCount; idx += 1) {
      if (occupancy[idx]) continue;
      if (forbiddenIndex !== null && idx === forbiddenIndex) continue;
      let w = 1;
      if (clusterStrength > 0 && placed.length > 0) {
        const x = idx % width;
        const y = Math.floor(idx / width);
        let adjacency = 0;
        for (const n of neighborsOf(width, height, x, y)) {
          if (occupancy[n.index]) adjacency += 1;
        }
        // Mix uniform weight with adjacency-proportional weight. Strength
        // controls how steep the bias is.
        w = (1 - clusterStrength) + clusterStrength * adjacency;
        if (w <= 0) w = 1e-6;
      }
      weights[idx] = w;
      total += w;
    }
    if (total <= 0) break;
    let pick = rng() * total;
    for (let idx = 0; idx < tileCount; idx += 1) {
      if (weights[idx] === 0) continue;
      pick -= weights[idx];
      if (pick <= 0) {
        occupancy[idx] = 1;
        placed.push(idx);
        break;
      }
    }
  }
  return occupancy;
}

// Privileged exact adjacency-count grid. Phase 2 will derive the pressure
// field from this. Controllers do not get to read it; the harness audit and
// oracle baseline do.
function computeAdjacency(width, height, occupancy) {
  const counts = new Uint8Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = indexOf(width, x, y);
      if (occupancy[idx]) {
        counts[idx] = 0; // adjacency on a mine tile is undefined; store 0
        continue;
      }
      let count = 0;
      for (const n of neighborsOf(width, height, x, y)) {
        if (occupancy[n.index]) count += 1;
      }
      counts[idx] = count;
    }
  }
  return counts;
}

// ----- state lifecycle -----

export function initializeBoardState(rawConfig = {}) {
  const config = normalizeMinesConfig(rawConfig);
  const rng = makeRng(config.seed);
  // First-click safety is enforced lazily: we generate the board now without
  // a forbidden tile, and on the first reveal we regenerate if the chosen
  // tile is a mine. This keeps placement seed-stable while still guaranteeing
  // a safe opening reveal.
  const occupancy = placeMines(config, rng, null);
  const adjacency = computeAdjacency(config.width, config.height, occupancy);
  const tiles = new Array(config.width * config.height).fill(TILE.CONCEALED);

  return {
    config,
    // Privileged grids — do NOT pass to controllers.
    privileged: Object.freeze({
      occupancy,
      adjacency,
    }),
    // Public state.
    tiles,
    flags: new Uint8Array(config.width * config.height),
    scanned: new Uint8Array(config.width * config.height),
    scansRemaining: config.scanBudget,
    turn: 0,
    revealedSafeCount: 0,
    falseFlagCount: 0,
    // Action ledger: append-only, public, bounded by turn count.
    actionLedger: [],
    terminal: null,
    // Internal RNG state checkpoint so regen-on-first-click is deterministic.
    _generationSeed: config.seed,
    _firstReveal: true,
  };
}

// Pure-ish flood reveal. Mutates the passed state because action application
// is the controller's commit point; callers wanting immutability should clone
// state before calling applyMinesAction.
function floodReveal(state, originIndex) {
  const { width, height } = state.config;
  const { adjacency } = state.privileged;
  const queue = [originIndex];
  while (queue.length > 0) {
    const idx = queue.pop();
    if (state.tiles[idx] !== TILE.CONCEALED) continue;
    if (state.flags[idx]) continue; // flagged tiles are not auto-revealed
    state.tiles[idx] = TILE.REVEALED_SAFE;
    state.revealedSafeCount += 1;
    if (adjacency[idx] === 0) {
      const x = idx % width;
      const y = Math.floor(idx / width);
      for (const n of neighborsOf(width, height, x, y)) {
        if (state.tiles[n.index] === TILE.CONCEALED && !state.flags[n.index]) {
          queue.push(n.index);
        }
      }
    }
  }
}

function regenerateForFirstClickSafety(state, safeIndex) {
  // Re-seed deterministically from the original seed plus a salt so the
  // regeneration is reproducible from the manifest.
  const rng = makeRng((state._generationSeed >>> 0) ^ 0x9e3779b9);
  const newOccupancy = placeMines(state.config, rng, safeIndex);
  const newAdjacency = computeAdjacency(state.config.width, state.config.height, newOccupancy);
  // Replace the frozen privileged object with a fresh frozen copy so external
  // references that captured it earlier do not see a torn write.
  state.privileged = Object.freeze({
    occupancy: newOccupancy,
    adjacency: newAdjacency,
  });
}

// ----- action validation -----

function actionBounds(state, action) {
  const { width, height } = state.config;
  if (action == null || typeof action !== "object") {
    return { ok: false, reason: "action_not_object" };
  }
  if (action.type === ACTION.ABSTAIN) {
    if (!state.config.allowAbstain) return { ok: false, reason: "abstain_disabled" };
    return { ok: true };
  }
  const { x, y } = action;
  if (!Number.isInteger(x) || !Number.isInteger(y)) {
    return { ok: false, reason: "coords_not_integers" };
  }
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return { ok: false, reason: "coords_out_of_bounds" };
  }
  return { ok: true };
}

export function isLegalAction(state, action) {
  if (state.terminal !== null) return { ok: false, reason: "terminal" };
  const bounds = actionBounds(state, action);
  if (!bounds.ok) return bounds;
  if (action.type === ACTION.ABSTAIN) return { ok: true };
  const idx = indexOf(state.config.width, action.x, action.y);
  const tile = state.tiles[idx];
  switch (action.type) {
    case ACTION.REVEAL:
      if (tile !== TILE.CONCEALED) return { ok: false, reason: "tile_not_concealed" };
      if (state.flags[idx]) return { ok: false, reason: "tile_flagged" };
      return { ok: true };
    case ACTION.FLAG:
      if (tile !== TILE.CONCEALED) return { ok: false, reason: "tile_not_concealed" };
      if (state.flags[idx]) return { ok: false, reason: "already_flagged" };
      return { ok: true };
    case ACTION.UNFLAG:
      if (!state.flags[idx]) return { ok: false, reason: "not_flagged" };
      return { ok: true };
    case ACTION.SCAN:
      if (state.scansRemaining <= 0) return { ok: false, reason: "scan_budget_zero" };
      if (state.scanned[idx]) return { ok: false, reason: "tile_already_scanned" };
      if (tile === TILE.REVEALED_SAFE) return { ok: false, reason: "tile_already_revealed" };
      return { ok: true };
    default:
      return { ok: false, reason: "unknown_action_type" };
  }
}

// ----- action application -----

export function applyMinesAction(state, action) {
  const legality = isLegalAction(state, action);
  if (!legality.ok) {
    return { state, applied: false, reason: legality.reason };
  }
  const { width } = state.config;
  state.turn += 1;
  const ledgerEntry = { turn: state.turn, type: action.type };

  if (action.type === ACTION.ABSTAIN) {
    state.actionLedger.push(ledgerEntry);
  } else {
    const idx = indexOf(width, action.x, action.y);
    ledgerEntry.x = action.x;
    ledgerEntry.y = action.y;
    ledgerEntry.index = idx;

    switch (action.type) {
      case ACTION.REVEAL: {
        if (state._firstReveal && state.config.firstClickSafe) {
          if (state.privileged.occupancy[idx] === 1) {
            regenerateForFirstClickSafety(state, idx);
          }
          state._firstReveal = false;
        } else {
          state._firstReveal = false;
        }
        if (state.privileged.occupancy[idx] === 1) {
          state.tiles[idx] = TILE.REVEALED_MINE;
          state.terminal = TERMINAL.MINE_TRIGGERED;
          ledgerEntry.outcome = "mine";
        } else {
          floodReveal(state, idx);
          ledgerEntry.outcome = "safe";
          ledgerEntry.adjacencyHidden = true; // Sundog mode: never write adjacency into ledger
        }
        break;
      }
      case ACTION.FLAG: {
        state.flags[idx] = 1;
        if (state.privileged.occupancy[idx] === 0) {
          state.falseFlagCount += 1;
          ledgerEntry.outcomeAudit = "false_flag"; // privileged audit only — strip before exposing
        }
        break;
      }
      case ACTION.UNFLAG: {
        if (state.flags[idx] && state.privileged.occupancy[idx] === 0) {
          // Decrement the false-flag counter only if the flag was an audited
          // false flag in the first place.
          if (state.falseFlagCount > 0) state.falseFlagCount -= 1;
        }
        state.flags[idx] = 0;
        break;
      }
      case ACTION.SCAN: {
        state.scanned[idx] = 1;
        state.scansRemaining -= 1;
        ledgerEntry.scansRemainingAfter = state.scansRemaining;
        // Scan returns no field data in Phase 1; Phase 2 will inject the
        // sensor response into the ledger entry.
        break;
      }
      default:
        break;
    }
    state.actionLedger.push(ledgerEntry);
  }

  // Terminal checks (skip if we already terminated above via mine_triggered).
  if (state.terminal === null) {
    if (isFullClear(state)) {
      state.terminal = TERMINAL.FULL_CLEAR;
    } else if (
      state.config.terminateOnScanExhaustion &&
      state.scansRemaining <= 0 &&
      action.type === ACTION.SCAN
    ) {
      state.terminal = TERMINAL.SCAN_BUDGET_EXHAUSTED;
    } else if (state.config.turnCap !== null && state.turn >= state.config.turnCap) {
      state.terminal = TERMINAL.TURN_CAP;
    }
  }

  return { state, applied: true, terminal: state.terminal };
}

function isFullClear(state) {
  const total = state.config.width * state.config.height;
  const mines = state.config.mineCount;
  return state.revealedSafeCount >= total - mines;
}

export function getTerminalEvent(state) {
  return state.terminal;
}

// ----- public memory accessor -----

// Returns the bounded, controller-visible state. The privileged grids are not
// included. The action ledger is shallow-copied so the controller cannot
// mutate the canonical record by reference.
export function getPublicMemory(state) {
  const ledger = state.actionLedger.map((entry) => {
    // Strip privileged audit fields before exposing.
    const { outcomeAudit: _drop, ...rest } = entry;
    return rest;
  });
  return Object.freeze({
    width: state.config.width,
    height: state.config.height,
    mineCount: state.config.mineCount,
    preset: state.config.preset,
    turn: state.turn,
    tiles: state.tiles.slice(),
    flags: Array.from(state.flags),
    scanned: Array.from(state.scanned),
    scansRemaining: state.scansRemaining,
    revealedSafeCount: state.revealedSafeCount,
    terminal: state.terminal,
    actionLedger: ledger,
  });
}

// ----- harness sample serializer -----

// Snapshot for JSONL logs. Includes privileged grids only when `includeAudit`
// is true (harness oracle / diagnostic runs). Browser logging should always
// pass false.
export function serializeMinesSample(state, { includeAudit = false } = {}) {
  const sample = {
    turn: state.turn,
    preset: state.config.preset,
    seed: state.config.seed,
    width: state.config.width,
    height: state.config.height,
    mineCount: state.config.mineCount,
    revealedSafeCount: state.revealedSafeCount,
    falseFlagCount: state.falseFlagCount,
    scansRemaining: state.scansRemaining,
    terminal: state.terminal,
  };
  if (includeAudit) {
    sample.audit = {
      occupancy: Array.from(state.privileged.occupancy),
      adjacency: Array.from(state.privileged.adjacency),
    };
  }
  return sample;
}

// ----- runtime helper -----

// Thin convenience wrapper. Returns a `{ state, step }` pair where `step` is
// a closure that applies an action and returns the updated terminal status.
// Mirrors createBalanceRuntime in shape so harness code can stay symmetric.
export function createMinesRuntime(rawConfig = {}) {
  const state = initializeBoardState(rawConfig);
  function step(action) {
    return applyMinesAction(state, action);
  }
  function memory() {
    return getPublicMemory(state);
  }
  return { state, step, memory };
}

// ----- example usage (kept as comments to avoid bloating the bundle) -----
//
//   // Browser (mines.html, future phase wiring):
//   //   import { createMinesRuntime, ACTION } from "./mines-core.mjs?v=...";
//   //   const { state, step, memory } = createMinesRuntime({ preset: "easy_sparse", seed: 7 });
//   //   step({ type: ACTION.REVEAL, x: 4, y: 4 });
//   //   render(memory());
//
//   // Harness (scripts/mines-harness.mjs, Phase 7 deliverable):
//   //   import { createMinesRuntime, ACTION, serializeMinesSample } from "../public/js/mines-core.mjs";
//   //   const { state, step } = createMinesRuntime(trialConfig);
//   //   while (state.terminal === null) {
//   //     const action = controller.choose(getPublicMemory(state));
//   //     step(action);
//   //   }
//   //   logger.write(serializeMinesSample(state, { includeAudit: true }));
