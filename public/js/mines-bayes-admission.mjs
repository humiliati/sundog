// Sundog Pressure Mines - Phase 12 Bayesian admission helpers.
//
// This module serializes the legal observation profile (`Phi_t`) for the
// staged Bayesian frontier baselines. It is intentionally policy-free: the
// posterior runner must consume this stream rather than reading board truth
// directly.

const PROFILE_ID = "mines-bayesian-baseline-v1";

const PRESSURE_BUDGET = Object.freeze({
  budget: "pressure",
  usesPressure: true,
  usesConfidence: true,
  usesGradient: false,
  usesScanReadings: false,
  scanBudgetPolicy: "fixed_zero_for_phase10_minimal_comparisons",
});

const FULL_BUDGET = Object.freeze({
  budget: "full",
  usesPressure: true,
  usesConfidence: true,
  usesGradient: true,
  usesScanReadings: true,
  scanBudgetPolicy: "uses_public_bounded_scan_ledger",
});

export const MINES_BAYES_ADMISSION_BUDGETS = Object.freeze({
  pressure: PRESSURE_BUDGET,
  full: FULL_BUDGET,
});

export const MINES_BAYES_FORBIDDEN_KEYS = Object.freeze([
  "occupancy",
  "adjacency",
  "truePressure",
  "pTrue",
  "p_true",
  "seed",
  "boardSeed",
  "sensorSeed",
  "oracle",
  "oracleAction",
  "verdict",
  "verdictLabel",
  "outcomeAudit",
]);

function rounded(value, digits = 9) {
  if (!Number.isFinite(value)) return null;
  return Number.parseFloat(value.toFixed(digits));
}

function numericArray(values, digits = 9) {
  if (!values) return [];
  return Array.from(values, (value) => rounded(value, digits));
}

function booleanArray(values) {
  if (!values) return [];
  return Array.from(values, (value) => Boolean(value));
}

function stringArray(values) {
  if (!values) return [];
  return Array.from(values, (value) => String(value));
}

function actionLedgerEntry(entry, { includeScanReadings }) {
  const out = {
    turn: Number.isInteger(entry.turn) ? entry.turn : null,
    type: entry.type ?? "",
  };
  if (Number.isInteger(entry.x)) out.x = entry.x;
  if (Number.isInteger(entry.y)) out.y = entry.y;
  if (Number.isInteger(entry.index)) out.index = entry.index;
  if (typeof entry.outcome === "string") out.outcome = entry.outcome;
  if (entry.adjacencyHidden === true) out.adjacencyHidden = true;
  if (Number.isInteger(entry.scansRemainingAfter)) out.scansRemainingAfter = entry.scansRemainingAfter;
  if (includeScanReadings && Number.isFinite(entry.scanReading)) out.scanReading = rounded(entry.scanReading, 12);
  return out;
}

function scanReadingsFromLedger(ledger = []) {
  return ledger
    .filter((entry) => Number.isInteger(entry.index) && Number.isFinite(entry.scanReading))
    .map((entry) => ({
      turn: Number.isInteger(entry.turn) ? entry.turn : null,
      index: entry.index,
      x: Number.isInteger(entry.x) ? entry.x : null,
      y: Number.isInteger(entry.y) ? entry.y : null,
      reading: rounded(entry.scanReading, 12),
    }));
}

function sanitizeSensorConfig(config = {}) {
  const {
    sensorSeed: _sensorSeed,
    __normalized__: _normalized,
    ...rest
  } = config ?? {};
  return {
    kernel: rest.kernel ?? "gaussian",
    sigma: rounded(rest.sigma, 6),
    sigmaNoise: rounded(rest.sigmaNoise, 6),
    dropoutRate: rounded(rest.dropoutRate, 6),
    delaySteps: Number.isInteger(rest.delaySteps) ? rest.delaySteps : 0,
    quantizationLevels: Number.isInteger(rest.quantizationLevels) ? rest.quantizationLevels : 0,
    sigmaScan: rounded(rest.sigmaScan, 6),
  };
}

function sanitizeEnvelopeCell(cell = null) {
  if (!cell) return null;
  const allowed = [
    "cellId",
    "preset",
    "sensorCell",
    "cellClass",
    "mode",
    "density",
    "mineDensity",
    "pressureNoise",
    "dropoutRate",
    "delaySteps",
    "clusterStrength",
    "kernelBlur",
    "scanBudget",
  ];
  const out = {};
  for (const key of allowed) {
    if (cell[key] !== undefined) out[key] = cell[key];
  }
  return Object.keys(out).length > 0 ? out : null;
}

function walkKeys(value, visit) {
  if (Array.isArray(value)) {
    value.forEach((item) => walkKeys(item, visit));
    return;
  }
  if (value && typeof value === "object") {
    for (const [key, nested] of Object.entries(value)) {
      visit(key);
      walkKeys(nested, visit);
    }
  }
}

export function forbiddenMinesBayesObservationKeys(observation) {
  const forbidden = new Set(MINES_BAYES_FORBIDDEN_KEYS);
  const present = new Set();
  walkKeys(observation, (key) => {
    if (forbidden.has(key)) present.add(key);
  });
  return [...present].sort();
}

export function assertNoMinesBayesObservationLeak(observation) {
  const present = forbiddenMinesBayesObservationKeys(observation);
  if (present.length > 0) {
    throw new Error(`Mines Bayesian observation leak: ${present.join(", ")}`);
  }
  return true;
}

export function serializeMinesBayesObservation({
  memory,
  sensor,
  sensorConfig,
  budget = "pressure",
  controllerMode = null,
  envelopeCell = null,
} = {}) {
  if (!memory) throw new Error("serializeMinesBayesObservation requires public memory");
  if (!sensor) throw new Error("serializeMinesBayesObservation requires a sensor snapshot");
  const admissionBudget = MINES_BAYES_ADMISSION_BUDGETS[budget];
  if (!admissionBudget) throw new Error(`Unknown Mines Bayesian admission budget: ${budget}`);
  const includeScanReadings = admissionBudget.usesScanReadings;
  const ledger = (memory.actionLedger ?? []).map((entry) => (
    actionLedgerEntry(entry, { includeScanReadings })
  ));

  const observation = {
    schema: "sundog.mines.bayes-admission.phi.v1",
    profileId: PROFILE_ID,
    controllerMode,
    budget: admissionBudget.budget,
    activeChannels: admissionBudget,
    boardWidth: memory.width,
    boardHeight: memory.height,
    mineCount: memory.mineCount,
    mineDensity: rounded(memory.mineCount / Math.max(1, memory.width * memory.height), 9),
    preset: memory.preset ?? null,
    turnIndex: memory.turn,
    terminal: memory.terminal ?? null,
    sensorConfig: sanitizeSensorConfig(sensorConfig),
    visibleTileState: stringArray(memory.tiles),
    flagState: booleanArray(memory.flags),
    scanState: booleanArray(memory.scanned),
    scansRemaining: Number.isInteger(memory.scansRemaining) ? memory.scansRemaining : 0,
    revealedSafeCount: Number.isInteger(memory.revealedSafeCount) ? memory.revealedSafeCount : 0,
    actionLedger: ledger,
    observedPressureField: numericArray(sensor.observed),
    pressureConfidenceField: numericArray(sensor.confidence),
    pressureGradientField: admissionBudget.usesGradient
      ? {
          x: numericArray(sensor.gradientX),
          y: numericArray(sensor.gradientY),
        }
      : null,
    scanReadings: includeScanReadings ? scanReadingsFromLedger(memory.actionLedger) : [],
    envelopeCell: sanitizeEnvelopeCell(envelopeCell),
  };

  assertNoMinesBayesObservationLeak(observation);
  return Object.freeze(observation);
}
