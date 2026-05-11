// Sundog Pressure Mines — Phase 2 sensor model.
//
// This module turns the privileged mine-occupancy grid from mines-core.mjs
// into an indirect, lossy local pressure field. The doc-canonical spec lives
// in docs/sundog_v_minesweeper.md under "Pinned Pressure Field Definition";
// if this implementation diverges from that block, fix the doc *and* preserve
// the prior text as an addendum so the pre-registration record stays
// auditable.
//
// Pure ES module. No DOM, no canvas, no shared mutable globals. Importable
// from both the browser page and the headless harness.
//
// Phase 2 scope:
//   * Gaussian pressure surface over the board
//   * Additive Gaussian noise + Bernoulli dropout + optional quantization
//   * Optional delay via per-runtime history buffer
//   * Local finite-difference gradient
//   * Bounded scan pulse (low-noise active probe)
//   * Privileged audit helpers: AUROC and Pearson correlation
//   * Irreducibility floor enforcement — sub-floor configs throw
//
// Phase 2 does NOT include:
//   * The Phase 3 diagnostic-benchmark sweeps (this module only exposes the
//     audit primitives those sweeps will consume).
//   * Controller logic (Phase 5).
//   * Browser rendering of the field (Phase 6).

import { makeRng } from "./mines-core.mjs";

// ----- pinned defaults and floor (must match the doc) -----

export const DEFAULT_SENSOR_CONFIG = Object.freeze({
  kernel: "gaussian",
  sigma: 1.0,
  sigmaNoise: 0.1,
  dropoutRate: 0.1,
  quantizationLevels: 0, // 0 = disabled
  delaySteps: 0,
  sigmaScan: 0.02,
  // Per-sensor RNG seed. Independent of the board seed so the harness can
  // sweep (board_seed, sensor_seed) pairs.
  sensorSeed: 1,
  // Kernel evaluation truncation. Mines beyond this many σ are skipped to
  // keep computation O(width * height * mineCount_local) instead of O(N²).
  kernelTruncationSigmas: 4,
});

export const SENSOR_FLOOR = Object.freeze({
  sigma: 0.7,
  sigmaNoise: 0.05,
  dropoutRate: 0.05,
});

// Supported kernel families. Phase 2 ships Gaussian; the strategy table is
// here so additional families can plug in for Phase 4 ablations without
// touching the field-computation call sites.
export const KERNEL_FAMILIES = Object.freeze({
  gaussian: Object.freeze({
    label: "Isotropic Gaussian",
    // Returns K(t, m) given squared distance and config.
    contribution(dSquared, config) {
      const sigma2 = config.sigma * config.sigma;
      return Math.exp(-dSquared / (2 * sigma2));
    },
    // Returns the truncation radius (in tiles) beyond which contribution is
    // treated as zero. config.kernelTruncationSigmas σ from each mine.
    truncationRadius(config) {
      return config.kernelTruncationSigmas * config.sigma;
    },
  }),
});

// ----- config + validation -----

export function normalizeSensorConfig(config = {}) {
  const merged = { ...DEFAULT_SENSOR_CONFIG, ...config };
  if (!KERNEL_FAMILIES[merged.kernel]) {
    throw new Error(`unknown kernel family: ${merged.kernel}`);
  }
  if (!(merged.sigma > 0)) {
    throw new Error(`sigma must be > 0, got ${merged.sigma}`);
  }
  if (!(merged.sigmaNoise >= 0)) {
    throw new Error(`sigmaNoise must be >= 0, got ${merged.sigmaNoise}`);
  }
  if (!(merged.dropoutRate >= 0 && merged.dropoutRate < 1)) {
    throw new Error(`dropoutRate must be in [0, 1), got ${merged.dropoutRate}`);
  }
  if (!(merged.sigmaScan >= 0)) {
    throw new Error(`sigmaScan must be >= 0, got ${merged.sigmaScan}`);
  }
  if (!Number.isInteger(merged.quantizationLevels) || merged.quantizationLevels < 0) {
    throw new Error(`quantizationLevels must be integer >= 0, got ${merged.quantizationLevels}`);
  }
  if (!Number.isInteger(merged.delaySteps) || merged.delaySteps < 0) {
    throw new Error(`delaySteps must be integer >= 0, got ${merged.delaySteps}`);
  }
  if (!(merged.kernelTruncationSigmas > 0)) {
    throw new Error(`kernelTruncationSigmas must be > 0, got ${merged.kernelTruncationSigmas}`);
  }
  // Tag before freezing so the hot-path check skips re-normalization.
  Object.defineProperty(merged, "__normalized__", { value: true, enumerable: false });
  return Object.freeze(merged);
}

// Enforces the irreducibility floor. Throws if any of σ, σ_noise, δ would
// place the workbench below the pre-registered floor. Harness sanity-check
// runs (where the operator explicitly wants to *demonstrate* the degenerate
// case) bypass this by calling `assertAboveFloor` with `{ allowSubFloor:
// true }` — but the harness then refuses to emit a public-tier log line.
export function assertAboveFloor(config, { allowSubFloor = false } = {}) {
  const violations = [];
  if (config.sigma < SENSOR_FLOOR.sigma) {
    violations.push(`sigma ${config.sigma} < floor ${SENSOR_FLOOR.sigma}`);
  }
  if (config.sigmaNoise < SENSOR_FLOOR.sigmaNoise) {
    violations.push(`sigmaNoise ${config.sigmaNoise} < floor ${SENSOR_FLOOR.sigmaNoise}`);
  }
  if (config.dropoutRate < SENSOR_FLOOR.dropoutRate) {
    violations.push(`dropoutRate ${config.dropoutRate} < floor ${SENSOR_FLOOR.dropoutRate}`);
  }
  if (violations.length > 0 && !allowSubFloor) {
    throw new Error(
      `sensor config violates irreducibility floor: ${violations.join("; ")}. ` +
        `set allowSubFloor=true to opt in to a degenerate-case sanity run; ` +
        `note that sub-floor runs are not publishable evidence.`,
    );
  }
  return { violations, subFloor: violations.length > 0 };
}

// ----- field computation -----

function neighborMines(boardState, x, y, radius) {
  const { width, height } = boardState.config;
  const occupancy = boardState.privileged.occupancy;
  const out = [];
  const r = Math.ceil(radius);
  const xMin = Math.max(0, x - r);
  const xMax = Math.min(width - 1, x + r);
  const yMin = Math.max(0, y - r);
  const yMax = Math.min(height - 1, y + r);
  for (let my = yMin; my <= yMax; my += 1) {
    for (let mx = xMin; mx <= xMax; mx += 1) {
      if (occupancy[my * width + mx] !== 1) continue;
      const dx = mx - x;
      const dy = my - y;
      const dSq = dx * dx + dy * dy;
      if (dSq <= radius * radius) out.push({ mx, my, dSq });
    }
  }
  return out;
}

// Returns Float64Array of length width*height holding the noiseless true
// pressure surface. Pure: no RNG.
export function computeTruePressure(boardState, rawSensorConfig = {}) {
  const config = rawSensorConfig.__normalized__
    ? rawSensorConfig
    : normalizeSensorConfig(rawSensorConfig);
  const { width, height } = boardState.config;
  const kernel = KERNEL_FAMILIES[config.kernel];
  const truncation = kernel.truncationRadius(config);
  const field = new Float64Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const mines = neighborMines(boardState, x, y, truncation);
      let sum = 0;
      for (const m of mines) {
        sum += kernel.contribution(m.dSq, config);
      }
      field[y * width + x] = sum;
    }
  }
  return field;
}

// Mulberry32 returns uniform in [0, 1). Convert to standard normal via
// Box-Muller. Cache the second draw to avoid wasting half the entropy.
function makeNormalSampler(rng) {
  let cached = null;
  return function normal() {
    if (cached !== null) {
      const value = cached;
      cached = null;
      return value;
    }
    const u1 = Math.max(rng(), 1e-9);
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    cached = r * Math.sin(theta);
    return r * Math.cos(theta);
  };
}

// Stamps noise, dropout, and quantization onto a true pressure surface.
// Returns { observed, confidence } where confidence is a Float64Array of
// {0, 1} per tile (0 = dropped, 1 = observed).
export function sampleObservedPressure(truePressure, rawSensorConfig, rng) {
  const config = rawSensorConfig.__normalized__
    ? rawSensorConfig
    : normalizeSensorConfig(rawSensorConfig);
  if (typeof rng !== "function") {
    throw new Error("sampleObservedPressure requires a rng() function");
  }
  const normal = makeNormalSampler(rng);
  const n = truePressure.length;
  const observed = new Float64Array(n);
  const confidence = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    if (rng() < config.dropoutRate) {
      observed[i] = NaN;
      confidence[i] = 0;
      continue;
    }
    let value = truePressure[i] + config.sigmaNoise * normal();
    if (config.quantizationLevels > 0) {
      // Clip to [0, qMax] and bin. qMax chosen as 4 in normalized units; the
      // smoke check confirms typical pressure rarely exceeds this.
      const qMax = 4;
      const clipped = Math.max(0, Math.min(qMax, value));
      value = Math.round((clipped / qMax) * (config.quantizationLevels - 1)) *
        (qMax / (config.quantizationLevels - 1));
    }
    observed[i] = value;
    confidence[i] = 1;
  }
  return { observed, confidence };
}

// Local finite-difference gradient. Returns two Float64Arrays of length
// width*height for the x and y components. Edge tiles use one-sided
// differences. Dropped-out tiles (NaN) cause the gradient at neighbors to be
// NaN where the central-difference window touches them — that propagation is
// intentional: the controller should treat NaN gradient as "ambiguous".
export function computePressureGradient(field, width, height) {
  const gx = new Float64Array(width * height);
  const gy = new Float64Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      const left = x > 0 ? field[idx - 1] : field[idx];
      const right = x < width - 1 ? field[idx + 1] : field[idx];
      const up = y > 0 ? field[idx - width] : field[idx];
      const down = y < height - 1 ? field[idx + width] : field[idx];
      const dx = x === 0 || x === width - 1 ? right - left : (right - left) / 2;
      const dy = y === 0 || y === height - 1 ? down - up : (down - up) / 2;
      gx[idx] = dx;
      gy[idx] = dy;
    }
  }
  return { gx, gy };
}

// ----- bounded scan pulse -----

// Returns a higher-fidelity reading at one tile. The reading is drawn from
// N(p_true(t), sigmaScan²). The caller is responsible for debiting the scan
// budget in mines-core; this function does not touch state.
//
// Phase 2 scan returns *pressure*, not occupancy probability — the Sundog
// discipline is that the controller has to do inference; scans are a budget-
// cost cleaner-sensor, not a privileged-state leak.
export function scanPulse(boardState, rawSensorConfig, x, y, rng) {
  const config = rawSensorConfig.__normalized__
    ? rawSensorConfig
    : normalizeSensorConfig(rawSensorConfig);
  if (typeof rng !== "function") {
    throw new Error("scanPulse requires a rng() function");
  }
  const normal = makeNormalSampler(rng);
  const { width, height } = boardState.config;
  if (x < 0 || x >= width || y < 0 || y >= height) {
    throw new Error(`scanPulse out of bounds: (${x}, ${y})`);
  }
  const idx = y * width + x;
  // Compute the true pressure at just this tile (avoid recomputing the
  // whole grid).
  const kernel = KERNEL_FAMILIES[config.kernel];
  const truncation = kernel.truncationRadius(config);
  const mines = neighborMines(boardState, x, y, truncation);
  let truePressure = 0;
  for (const m of mines) {
    truePressure += kernel.contribution(m.dSq, config);
  }
  const reading = truePressure + config.sigmaScan * normal();
  return Object.freeze({
    x,
    y,
    index: idx,
    reading,
    truePressure, // privileged audit only; harness strips before logging
  });
}

// ----- privileged audit helpers -----

// Per-tile "true local risk": fraction of the 8-neighborhood that is mined.
// Used as a regression target for "does pressure predict local risk?".
export function computeLocalRisk(boardState) {
  const { width, height } = boardState.config;
  const occupancy = boardState.privileged.occupancy;
  const risk = new Float64Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let count = 0;
      let cells = 0;
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
          cells += 1;
          count += occupancy[ny * width + nx];
        }
      }
      risk[y * width + x] = cells > 0 ? count / cells : 0;
    }
  }
  return risk;
}

// AUROC via the Mann-Whitney U-statistic identity:
//   AUROC = (sum_of_positive_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
// Higher score → predicts the positive class.
//
// NaN scores are treated as "abstained" and excluded from the calculation
// (this lets dropout-out tiles fall out of the audit cleanly).
export function computeAuroc(scores, labels) {
  if (scores.length !== labels.length) {
    throw new Error(`length mismatch: scores=${scores.length}, labels=${labels.length}`);
  }
  const pairs = [];
  for (let i = 0; i < scores.length; i += 1) {
    if (Number.isFinite(scores[i])) {
      pairs.push({ score: scores[i], label: labels[i] ? 1 : 0 });
    }
  }
  pairs.sort((a, b) => a.score - b.score);
  let i = 0;
  let nPos = 0;
  let nNeg = 0;
  let sumPosRanks = 0;
  while (i < pairs.length) {
    let j = i;
    while (j < pairs.length && pairs[j].score === pairs[i].score) j += 1;
    // Average rank for ties; ranks are 1-indexed.
    const avgRank = (i + 1 + j) / 2;
    for (let k = i; k < j; k += 1) {
      if (pairs[k].label === 1) {
        sumPosRanks += avgRank;
        nPos += 1;
      } else {
        nNeg += 1;
      }
    }
    i = j;
  }
  if (nPos === 0 || nNeg === 0) return null;
  return (sumPosRanks - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

// Pearson correlation. NaN pairs are dropped.
export function computeCorrelation(xs, ys) {
  if (xs.length !== ys.length) {
    throw new Error(`length mismatch: xs=${xs.length}, ys=${ys.length}`);
  }
  let n = 0;
  let sumX = 0;
  let sumY = 0;
  for (let i = 0; i < xs.length; i += 1) {
    if (Number.isFinite(xs[i]) && Number.isFinite(ys[i])) {
      sumX += xs[i];
      sumY += ys[i];
      n += 1;
    }
  }
  if (n < 2) return null;
  const meanX = sumX / n;
  const meanY = sumY / n;
  let num = 0;
  let dx2 = 0;
  let dy2 = 0;
  for (let i = 0; i < xs.length; i += 1) {
    if (!Number.isFinite(xs[i]) || !Number.isFinite(ys[i])) continue;
    const dx = xs[i] - meanX;
    const dy = ys[i] - meanY;
    num += dx * dy;
    dx2 += dx * dx;
    dy2 += dy * dy;
  }
  const denom = Math.sqrt(dx2 * dy2);
  return denom > 0 ? num / denom : null;
}

// ----- runtime wrapper -----

// Stateful sensor runtime. Owns its own RNG and a history ring buffer for
// delay. Each `step(boardState)` returns the (possibly delayed) observed
// field, gradient, and confidence at the current turn. The delay buffer is
// keyed off calls, not board turns — callers should call exactly once per
// turn for delay to mean what the doc says.
export function createSensorRuntime(rawSensorConfig = {}) {
  const config = normalizeSensorConfig(rawSensorConfig);
  const rng = makeRng(config.sensorSeed);
  const history = []; // ring of pre-noise true-pressure snapshots
  return {
    config,
    step(boardState) {
      const truePressure = computeTruePressure(boardState, config);
      // Push and trim. delaySteps = 0 means return current; delaySteps = k
      // means return the snapshot from k calls ago, or the current snapshot
      // if the buffer hasn't filled yet (warm-up).
      history.push(truePressure);
      const target = history.length - 1 - config.delaySteps;
      const lagged = target >= 0 ? history[target] : history[0];
      if (history.length > config.delaySteps + 1) history.shift();
      const { observed, confidence } = sampleObservedPressure(lagged, config, rng);
      const { gx, gy } = computePressureGradient(observed, boardState.config.width, boardState.config.height);
      return {
        truePressure: lagged, // privileged — harness logs may include it
        observed,
        confidence,
        gradientX: gx,
        gradientY: gy,
      };
    },
    scan(boardState, x, y) {
      return scanPulse(boardState, config, x, y, rng);
    },
  };
}

