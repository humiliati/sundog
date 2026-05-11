import { normalizeMinesConfig } from "./mines-core.mjs";
import { normalizeSensorConfig } from "./mines-sensor.mjs";
import { MINES_CONTROLLER_MODES } from "./mines-controllers.mjs";

// Boundary vocabulary canonical to both the headless harness and the browser
// workbench. The split into `static` and `live` matters for Phase 10: the
// operating-envelope verdict per cell consumes ONLY the static label
// (config-derived knobs). Live frontier signals belong to per-seed in-game
// warnings, not to the envelope verdict.

function roundMetric(value, digits = 6) {
  if (value === null || value === undefined || Number.isNaN(value)) return null;
  if (!Number.isFinite(value)) return value;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function densityOf(config) {
  return config.mineCount / (config.width * config.height);
}

function mechanismLabel(status) {
  if (status === "do_not_use") return "DO NOT USE";
  if (status === "watch_boundary") return "WATCH BOUNDARY";
  return "LIKELY READABLE";
}

function boundarySummary(status) {
  if (status === "do_not_use") {
    return "This cell is inside a known Phase 9 failure regime; use it to show the boundary, not success.";
  }
  if (status === "watch_boundary") {
    return "This cell is near a Phase 9 boundary; treat any controller lead as provisional until matched seeds agree.";
  }
  return "No Phase 9 boundary warning is active for the current controls.";
}

function statusFromMechanisms(mechanisms) {
  const hasUnsafe = mechanisms.some((mechanism) => mechanism.severity === "unsafe");
  const hasCaution = mechanisms.some((mechanism) => mechanism.severity === "caution");
  return hasUnsafe ? "do_not_use" : hasCaution ? "watch_boundary" : "likely_readable";
}

// Static boundary assessment. Phase 10 uses this exclusively for per-cell
// envelope labels — no live game-state signals leak in. The contract is:
// given the board + sensor + mode config, what is the pre-registered Phase 9
// label for this cell?
export function assessStaticBoundary({
  boardConfig = {},
  sensorConfig = {},
  mode = "sundog_lean",
} = {}) {
  const board = normalizeMinesConfig(boardConfig);
  const sensor = sensorConfig.__normalized__
    ? sensorConfig
    : normalizeSensorConfig(sensorConfig);
  const modeDefinition = MINES_CONTROLLER_MODES[mode] ?? {};
  const mechanisms = [];
  const add = (severity, code, message, value = null) => {
    mechanisms.push({ severity, code, message, value });
  };

  const density = densityOf(board);
  const clusterStrength = board.generator?.clusterStrength ?? 0;
  const tileCount = board.width * board.height;

  if (density >= 0.24) {
    add(
      "unsafe",
      "frontier_ambiguity",
      "Mine density is high enough that many frontier choices collapse into similar local risk.",
      roundMetric(density, 4),
    );
  } else if (density >= 0.18) {
    add(
      "caution",
      "frontier_ambiguity",
      "Mine density is near the first-pass ambiguity boundary.",
      roundMetric(density, 4),
    );
  }

  if (clusterStrength >= 0.75) {
    add(
      "unsafe",
      "frontier_ambiguity",
      "Clustering can create deceptive pressure plateaus around the frontier.",
      roundMetric(clusterStrength, 3),
    );
  } else if (clusterStrength >= 0.45) {
    add(
      "caution",
      "frontier_ambiguity",
      "Clustering is high enough that matched-seed deltas should be checked.",
      roundMetric(clusterStrength, 3),
    );
  }

  if (sensor.sigma >= 6 || sensor.sigmaNoise >= 5 || sensor.dropoutRate >= 0.7) {
    add(
      "unsafe",
      "field_uninformative",
      "Blur, noise, or dropout is high enough to erase local pressure distinction.",
      `sigma=${roundMetric(sensor.sigma, 3)}, noise=${roundMetric(sensor.sigmaNoise, 3)}, dropout=${roundMetric(sensor.dropoutRate, 3)}`,
    );
  } else if (sensor.sigma >= 3 || sensor.sigmaNoise >= 1 || sensor.dropoutRate >= 0.35) {
    add(
      "caution",
      "field_uninformative",
      "The pressure field is degraded enough that controller wins should be treated as narrow.",
      `sigma=${roundMetric(sensor.sigma, 3)}, noise=${roundMetric(sensor.sigmaNoise, 3)}, dropout=${roundMetric(sensor.dropoutRate, 3)}`,
    );
  }

  if (sensor.delaySteps >= 3) {
    add(
      "unsafe",
      "delay_misread",
      "Sensor delay is in the pre-registered failure regime for frontier decisions.",
      sensor.delaySteps,
    );
  } else if (sensor.delaySteps >= 1) {
    add(
      "caution",
      "delay_misread",
      "Delayed pressure can describe the previous frontier rather than the current one.",
      sensor.delaySteps,
    );
  }

  if (tileCount >= 256) {
    add(
      "caution",
      "frontier_ambiguity",
      "Large boards widen the frontier and make single-step pressure ordering less diagnostic.",
      `${board.width}x${board.height}`,
    );
  }

  if (modeDefinition.usesScan && board.scanBudget <= 0) {
    add(
      "unsafe",
      "probe_budget_exhausted",
      "This mode can request scans, but the board exposes no scan budget.",
      board.scanBudget,
    );
  } else if (modeDefinition.usesScan && board.scanBudget <= 1) {
    add(
      "caution",
      "probe_budget_exhausted",
      "Scan budget is thin enough that active probes cannot carry the claim.",
      board.scanBudget,
    );
  }

  const status = statusFromMechanisms(mechanisms);
  return {
    status,
    label: mechanismLabel(status),
    summary: boundarySummary(status),
    mineDensity: roundMetric(density, 6),
    clusterStrength: roundMetric(clusterStrength, 6),
    mechanisms,
    // Pre-computed code list for CSV emission and verdict labelling.
    mechanismCodes: mechanisms.map((mechanism) => mechanism.code),
  };
}

// Live boundary assessment. Overlays per-seed signals on top of the static
// label. The browser panel calls both and merges; the headless Phase 10
// envelope harness MUST NOT call this — its verdict is static-only.
export function assessLiveBoundary(staticAssessment, live = {}) {
  // Defensive copy so we never mutate the static input.
  const baseMechanisms = staticAssessment?.mechanisms
    ? [...staticAssessment.mechanisms]
    : [];
  const mechanisms = baseMechanisms;
  const add = (severity, code, message, value = null) => {
    mechanisms.push({ severity, code, message, value });
  };

  if (Number.isFinite(live.meanFrontierConfidence)) {
    if (live.meanFrontierConfidence < 0.18) {
      add(
        "unsafe",
        "field_uninformative",
        "The current frontier has lost pressure confidence.",
        roundMetric(live.meanFrontierConfidence, 3),
      );
    } else if (live.meanFrontierConfidence < 0.35) {
      add(
        "caution",
        "field_uninformative",
        "The current frontier is in the low-confidence band.",
        roundMetric(live.meanFrontierConfidence, 3),
      );
    }
  }
  if (Number.isFinite(live.frontierSize) && live.frontierSize <= 1 && live.terminal === null) {
    add(
      "caution",
      "frontier_ambiguity",
      "The current frontier has collapsed to a single forced choice.",
      live.frontierSize,
    );
  }
  if (Number.isFinite(live.falseFlagCount)) {
    if (live.falseFlagCount >= 2) {
      add(
        "unsafe",
        "overflagged",
        "False flags are accumulating in this seed.",
        live.falseFlagCount,
      );
    } else if (live.falseFlagCount >= 1) {
      add(
        "caution",
        "overflagged",
        "A false flag has already appeared in this seed.",
        live.falseFlagCount,
      );
    }
  }
  if (live.terminal === "mine_triggered") {
    add(
      "unsafe",
      "controller_overcommitted",
      "This seed ended by revealing a mine.",
      live.terminal,
    );
  }

  const status = statusFromMechanisms(mechanisms);
  return {
    status,
    label: mechanismLabel(status),
    summary: boundarySummary(status),
    mineDensity: staticAssessment?.mineDensity ?? null,
    clusterStrength: staticAssessment?.clusterStrength ?? null,
    mechanisms,
    mechanismCodes: mechanisms.map((mechanism) => mechanism.code),
    staticStatus: staticAssessment?.status ?? null,
  };
}

// Compatibility wrapper preserving the original API shape. Phase 9 harness,
// the browser panel, and any other caller pre-dating the split keep working.
// New code (Phase 10) should call assessStaticBoundary directly.
export function assessMinesBoundary({
  boardConfig = {},
  sensorConfig = {},
  mode = "sundog_lean",
  live = null,
} = {}) {
  const staticAssessment = assessStaticBoundary({ boardConfig, sensorConfig, mode });
  if (!live) return staticAssessment;
  return assessLiveBoundary(staticAssessment, live);
}
