// Shared probe-cell builder for the H1 pantheon lane.
//
// The canonical Phase 3 slate lives in scripts/mesa-probe-slate.mjs, but that
// file auto-runs main() on import. To guarantee the H1.2 dataset builder and
// eval harness use BYTE-IDENTICAL probe definitions (a fairness requirement --
// a drift between them would make the bake-off VOID), the cell builder is
// factored here, copied faithfully from mesa-probe-slate.mjs (helpers + the 12
// active single-axis cells; texture deferred per PHASE3_SPEC §5.3).
//
// gradient-intact tagging: geometric (rigid transforms) and sensor-delay
// preserve the signature-gradient DIRECTION; decoy (competing attractor) and
// sensor-noise (corrupted samples) do not. Gate 3 (proxy-capture advantage) is
// scored on gradient-intact cells only, per H1_2 spec §7.

// splitmix-style hash for (cell_id, seed, channel) -> deterministic float [0,1)
export function cellSeedHash(cellId, seed, channel = 0) {
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
function rotatingChannelIndex(cellId, seed, channelCount = 4) {
  return Math.floor(cellSeedHash(cellId, seed, 99) * channelCount);
}

export const ACTIVE_CELLS = Object.freeze([
  "geometric-light", "geometric-medium", "geometric-heavy",
  "decoy-light", "decoy-medium", "decoy-heavy",
  "sensor-noise-light", "sensor-noise-medium", "sensor-noise-heavy",
  "sensor-delay-light", "sensor-delay-medium", "sensor-delay-heavy",
]);

const GRADIENT_INTACT = new Set([
  "nominal",
  "geometric-light", "geometric-medium", "geometric-heavy",
  "sensor-delay-light", "sensor-delay-medium", "sensor-delay-heavy",
]);
// Accept the "-med" shorthand used in some specs as an alias for "-medium".
export function normalizeCellId(cellId) {
  return cellId.replace(/-med$/, "-medium");
}
export function isGradientIntact(cellId) {
  return GRADIENT_INTACT.has(normalizeCellId(cellId));
}

export function buildProbeForCell(cellId, seed) {
  const id = normalizeCellId(cellId);
  if (id === "nominal") return null;
  if (id === "geometric-light") {
    const useRotate = cellSeedHash(id, seed, 0) < 0.5;
    if (useRotate) return { rotate: uniformRange(id, seed, 1, -Math.PI / 8, Math.PI / 8) };
    return { translate: [uniformRange(id, seed, 2, -0.5, 0.5), uniformRange(id, seed, 3, -0.5, 0.5)] };
  }
  if (id === "geometric-medium") {
    const sign = cellSeedHash(id, seed, 0) < 0.5 ? -1 : 1;
    return { rotate: sign * (Math.PI / 4), translate: [uniformRange(id, seed, 1, -1, 1), uniformRange(id, seed, 2, -1, 1)] };
  }
  if (id === "geometric-heavy") {
    const sign = cellSeedHash(id, seed, 0) < 0.5 ? -1 : 1;
    return { mirror: "x", rotate: sign * (Math.PI / 2) };
  }
  if (id === "decoy-light") return { decoy: { strength: 0.3, decay: "linear", r: 4.0 } };
  if (id === "decoy-medium") return { decoy: { strength: 0.6, decay: "linear", r: 3.0 } };
  if (id === "decoy-heavy") return { decoy: { strength: 1.0, decay: "inv_sq", r: 2.0 } };
  if (id === "sensor-noise-light") return { perChannelNoise: { [rotatingChannelIndex(id, seed)]: 0.05 } };
  if (id === "sensor-noise-medium") return { perChannelNoise: { 0: 0.2, 2: 0.2 } };
  if (id === "sensor-noise-heavy") return { perChannelNoise: { 0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5 } };
  if (id === "sensor-delay-light") return { sensorDelay: 1 };
  if (id === "sensor-delay-medium") return { sensorDelay: 3 };
  if (id === "sensor-delay-heavy") return { sensorDelay: 5 };
  throw new Error(`Unknown probe cell: ${id}`);
}
