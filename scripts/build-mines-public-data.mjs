import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

const root = process.cwd();

const sourcePaths = Object.freeze({
  phase10Manifest: "results/mines/phase10-envelope/manifest.json",
  phase10BestWorst: "results/mines/phase10-envelope/best-worst-cells.csv",
  phase12Manifest: "results/mines/phase12-phase10-slate-reducer-64/manifest.json",
  phase12Summary: "results/mines/phase12-phase10-slate-reducer-64/summary.json",
  phase12RegretSummary: "results/mines/phase12-phase10-slate-reducer-64/bayes-regret-summary.csv",
  phase12Actions: "results/mines/phase12-phase10-slate-reducer-64/bayes-actions.csv",
  phase12Diagnostics: "results/mines/phase12-phase10-slate-reducer-64/posterior-diagnostics.csv",
  phase12PosteriorMap: "results/mines/phase12-phase10-slate-reducer-64/frontier-posterior-map.json",
});

const outputPath = "public/data/mines-phase13-bayes-floor.json";
const posteriorTurnLimit = 6;
const posteriorCandidateLimit = 10;

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (char === "\"") {
      if (quoted && next === "\"") {
        cell += "\"";
        i += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(cell);
      if (row.some((value) => value.length > 0)) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }

  row.push(cell);
  if (row.some((value) => value.length > 0)) rows.push(row);
  if (rows.length === 0) return [];

  const headers = rows[0];
  return rows.slice(1).map((values) =>
    Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])),
  );
}

async function readText(relativePath) {
  return readFile(join(root, relativePath), "utf8");
}

function asNumber(value) {
  if (value === undefined || value === null || value === "") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function asBool(value) {
  if (typeof value === "boolean") return value;
  if (value === "true") return true;
  if (value === "false") return false;
  return null;
}

function finite(values) {
  return values.filter((value) => Number.isFinite(value));
}

function mean(values) {
  const numbers = finite(values);
  if (numbers.length === 0) return null;
  return numbers.reduce((sum, value) => sum + value, 0) / numbers.length;
}

function min(values) {
  const numbers = finite(values);
  return numbers.length ? Math.min(...numbers) : null;
}

function max(values) {
  const numbers = finite(values);
  return numbers.length ? Math.max(...numbers) : null;
}

function round(value, digits = 6) {
  return Number.isFinite(value) ? Number.parseFloat(value.toFixed(digits)) : value;
}

function countBy(rows, keyFn) {
  const counts = new Map();
  for (const row of rows) {
    const key = keyFn(row) || "unknown";
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return Object.fromEntries([...counts.entries()].sort(([a], [b]) => String(a).localeCompare(String(b))));
}

function cleanRegret(row) {
  return {
    phase: row.phase,
    selection: row.selection,
    cellId: row.cell_id,
    cellClass: row.cell_class,
    bayesMode: row.bayes_mode,
    targetMode: row.target_mode,
    n: asNumber(row.n),
    meanBudgetAdjustedDelta: asNumber(row.mean_budget_adjusted_delta),
    minBudgetAdjustedDelta: asNumber(row.min_budget_adjusted_delta),
    maxBudgetAdjustedDelta: asNumber(row.max_budget_adjusted_delta),
    winDelta: asNumber(row.win_delta),
  };
}

function cleanBestWorst(row) {
  return {
    selection: row.selection,
    cellId: row.cell_id,
    cellClass: row.cell_class,
    mode: row.mode,
    candidate: asBool(row.candidate),
    failureRegime: asBool(row.failure_regime),
    mineDensity: asNumber(row.mine_density),
    mineCount: asNumber(row.mine_count),
    sigmaNoise: asNumber(row.sigma_noise),
    dropoutRate: asNumber(row.dropout_rate),
    targetBudgetAdjustedSafeTilesMean: asNumber(row.target_budget_adjusted_safe_tiles_mean),
    naiveBudgetAdjustedSafeTilesMean: asNumber(row.naive_budget_adjusted_safe_tiles_mean),
    budgetDeltaVsNaiveMean: asNumber(row.budget_delta_vs_naive_mean),
    budgetDeltaVsNaiveCiLow: asNumber(row.budget_delta_vs_naive_ci_low),
    budgetDeltaVsNaiveCiHigh: asNumber(row.budget_delta_vs_naive_ci_high),
    staticBoundaryStatus: row.static_boundary_status,
    mechanismCodes: row.mechanism_codes ? row.mechanism_codes.split("|").filter(Boolean) : [],
    representativeSeed: asNumber(row.representative_seed),
    replayUrl: row.replay_url || null,
  };
}

function cleanAction(row) {
  return {
    phase: row.phase,
    selection: row.selection,
    cellId: row.cell_id,
    mode: row.mode,
    seed: asNumber(row.seed),
    turn: asNumber(row.turn),
    budget: row.budget,
    action: row.action,
    x: asNumber(row.x),
    y: asNumber(row.y),
    posteriorHazard: asNumber(row.posterior_hazard),
    pressureScore: asNumber(row.pressure_score),
    actionRisk: asNumber(row.action_risk),
    riskSource: row.risk_source,
    ess: asNumber(row.ess),
    frontierSize: asNumber(row.frontier_size),
    legal: asBool(row.legal),
    terminalAfter: row.terminal_after || null,
  };
}

function cleanDiagnostic(row) {
  return {
    phase: row.phase,
    selection: row.selection,
    cellId: row.cell_id,
    mode: row.mode,
    seed: asNumber(row.seed),
    turn: asNumber(row.turn),
    budget: row.budget,
    phiHash: row.phi_hash,
    frontierSize: asNumber(row.frontier_size),
    particleCount: asNumber(row.particle_count),
    ess: asNumber(row.ess),
    minHazard: asNumber(row.min_hazard),
    maxHazard: asNumber(row.max_hazard),
    selectedAction: row.selected_action,
    selectedX: asNumber(row.selected_x),
    selectedY: asNumber(row.selected_y),
    selectedIdx: asNumber(row.selected_idx),
    selectedHazard: asNumber(row.selected_hazard),
    selectedPressure: asNumber(row.selected_pressure),
    selectedActionRisk: asNumber(row.selected_action_risk),
    riskSource: row.risk_source,
    observedPressureCount: asNumber(row.observed_pressure_count),
    scanReadingCount: asNumber(row.scan_reading_count),
    forbiddenKeyCount: asNumber(row.forbidden_key_count),
    leakFree: asBool(row.leak_free),
  };
}

function manifestCellKey(cell) {
  return cell.cellId;
}

function summarizeTarget(targetMode, rows) {
  const deltas = rows.map((row) => row.meanBudgetAdjustedDelta);
  return {
    targetMode,
    cells: rows.length,
    meanBudgetAdjustedDelta: round(mean(deltas), 6),
    minBudgetAdjustedDelta: round(min(deltas), 6),
    maxBudgetAdjustedDelta: round(max(deltas), 6),
    negativeCells: rows.filter((row) => Number.isFinite(row.meanBudgetAdjustedDelta) && row.meanBudgetAdjustedDelta < 0).length,
    zeroCells: rows.filter((row) => row.meanBudgetAdjustedDelta === 0).length,
    positiveCells: rows.filter((row) => Number.isFinite(row.meanBudgetAdjustedDelta) && row.meanBudgetAdjustedDelta > 0).length,
  };
}

function targetSummaries(regretRows) {
  const byTarget = new Map();
  for (const row of regretRows) {
    if (!byTarget.has(row.targetMode)) byTarget.set(row.targetMode, []);
    byTarget.get(row.targetMode).push(row);
  }
  const order = ["naive_pressure", "sundog_minimal", "sundog_lean", "oracle_safe"];
  return order
    .filter((target) => byTarget.has(target))
    .map((target) => summarizeTarget(target, byTarget.get(target)));
}

function buildCellRows(phase12Manifest, regretRows) {
  const regretsByCell = new Map();
  for (const row of regretRows) {
    if (!regretsByCell.has(row.cellId)) regretsByCell.set(row.cellId, {});
    regretsByCell.get(row.cellId)[row.targetMode] = {
      meanBudgetAdjustedDelta: row.meanBudgetAdjustedDelta,
      minBudgetAdjustedDelta: row.minBudgetAdjustedDelta,
      maxBudgetAdjustedDelta: row.maxBudgetAdjustedDelta,
      winDelta: row.winDelta,
      n: row.n,
    };
  }

  return (phase12Manifest.cells ?? []).map((cell) => ({
    cellId: manifestCellKey(cell),
    selection: cell.selection,
    cellClass: cell.cellClass,
    preset: cell.preset,
    representativeSeed: cell.representativeSeed,
    replayUrl: cell.replayUrl || null,
    board: cell.board,
    sensor: cell.sensor,
    bayesRegret: regretsByCell.get(manifestCellKey(cell)) ?? {},
  })).sort((a, b) => (
    String(a.selection).localeCompare(String(b.selection))
    || String(a.cellClass).localeCompare(String(b.cellClass))
    || a.cellId.localeCompare(b.cellId)
  ));
}

function findPromotedCandidate(regretRows) {
  const rows = regretRows.filter((row) => row.selection === "phase10_candidate");
  return Object.fromEntries(rows.map((row) => [row.targetMode, {
    meanBudgetAdjustedDelta: row.meanBudgetAdjustedDelta,
    minBudgetAdjustedDelta: row.minBudgetAdjustedDelta,
    maxBudgetAdjustedDelta: row.maxBudgetAdjustedDelta,
    winDelta: row.winDelta,
    n: row.n,
  }]));
}

function pressureFloorGate(targets) {
  const naive = targets.find((target) => target.targetMode === "naive_pressure");
  const minimal = targets.find((target) => target.targetMode === "sundog_minimal");
  return {
    label: "Pressure-floor parity",
    pass: naive?.negativeCells === 0 && minimal?.negativeCells === 0,
    comparators: ["naive_pressure", "sundog_minimal"],
    negativeVsNaiveCells: naive?.negativeCells ?? null,
    negativeVsMinimalCells: minimal?.negativeCells ?? null,
    note:
      "Pass means the pressure-budget Bayes lane is never worse than the pressure-ordering floor on mean budget-adjusted safe tiles. It does not mean Bayes dominates sundog_lean.",
  };
}

function paramsFromReplayUrl(replayUrl) {
  if (!replayUrl) return {};
  const url = new URL(replayUrl);
  return Object.fromEntries(url.searchParams.entries());
}

function makeTurnKey(row) {
  return `${row.cellId}|${row.seed}|${row.turn}`;
}

function compactCandidate(candidate) {
  return {
    idx: candidate.idx,
    x: candidate.x,
    y: candidate.y,
    hazard: candidate.hazard,
    entropy: candidate.entropy,
    pressure: candidate.pressure,
    confidence: candidate.confidence,
    pressureScore: candidate.pressureScore,
    actionRisk: candidate.actionRisk,
  };
}

function selectedCellRegrets(cellId, regretRows) {
  return Object.fromEntries(regretRows
    .filter((row) => row.cellId === cellId)
    .map((row) => [row.targetMode, {
      meanBudgetAdjustedDelta: row.meanBudgetAdjustedDelta,
      minBudgetAdjustedDelta: row.minBudgetAdjustedDelta,
      maxBudgetAdjustedDelta: row.maxBudgetAdjustedDelta,
      winDelta: row.winDelta,
      n: row.n,
    }]));
}

function divergenceCell(regretRows) {
  return regretRows
    .filter((row) => row.targetMode === "sundog_lean")
    .sort((a, b) => (
      Math.abs(b.meanBudgetAdjustedDelta ?? 0) - Math.abs(a.meanBudgetAdjustedDelta ?? 0)
      || String(a.cellId).localeCompare(String(b.cellId))
    ))[0] ?? null;
}

function selectedCellSpecs(bestWorst, regretRows) {
  const best = bestWorst.find((row) => row.selection === "best_cell");
  const worst = bestWorst.find((row) => row.selection === "worst_cell");
  const divergence = divergenceCell(regretRows);
  return [
    best && {
      role: "confirmed_pocket",
      label: "Confirmed pocket",
      reason: "Phase 10 promoted positive pocket.",
      cellId: best.cellId,
    },
    worst && {
      role: "mapped_failure",
      label: "Mapped failure",
      reason: "Phase 10 paired negative region.",
      cellId: worst.cellId,
    },
    divergence && {
      role: "bayes_divergence",
      label: "Bayes-divergence cell",
      reason: "Largest absolute mean delta between bayes_frontier_pressure and sundog_lean in the Phase 10-slate reducer.",
      cellId: divergence.cellId,
    },
  ].filter(Boolean);
}

function buildPosteriorCells({
  cells,
  bestWorst,
  regretRows,
  actions,
  diagnostics,
  posteriorMap,
}) {
  const cellsById = new Map(cells.map((cell) => [cell.cellId, cell]));
  const diagnosticsByTurn = new Map(diagnostics.map((row) => [makeTurnKey(row), row]));
  const actionsByTurn = new Map(actions.map((row) => [makeTurnKey(row), row]));
  const entriesByCellSeed = new Map();
  for (const entry of posteriorMap.entries ?? []) {
    if (entry.mode !== "bayes_frontier_pressure") continue;
    const key = `${entry.cellId}|${entry.seed}`;
    if (!entriesByCellSeed.has(key)) entriesByCellSeed.set(key, []);
    entriesByCellSeed.get(key).push(entry);
  }

  return selectedCellSpecs(bestWorst, regretRows).map((spec) => {
    const cell = cellsById.get(spec.cellId);
    const seed = cell?.representativeSeed ?? 0;
    const entries = [...(entriesByCellSeed.get(`${spec.cellId}|${seed}`) ?? [])]
      .sort((a, b) => a.turn - b.turn)
      .slice(0, posteriorTurnLimit);
    return {
      ...spec,
      seed,
      replayUrl: cell?.replayUrl ?? null,
      replayParams: paramsFromReplayUrl(cell?.replayUrl),
      cell: cell ? {
        cellId: cell.cellId,
        selection: cell.selection,
        cellClass: cell.cellClass,
        preset: cell.preset,
        board: cell.board,
        sensor: cell.sensor,
      } : null,
      bayesRegret: selectedCellRegrets(spec.cellId, regretRows),
      traceSamplePolicy: {
        turns: `first ${posteriorTurnLimit} representative-seed Bayes pressure decisions`,
        candidates: `first ${posteriorCandidateLimit} candidates from each stored posterior map entry`,
      },
      turns: entries.map((entry) => {
        const turnKey = makeTurnKey(entry);
        const diagnostic = diagnosticsByTurn.get(turnKey);
        const action = actionsByTurn.get(turnKey);
        return {
          turn: entry.turn,
          budget: entry.budget,
          frontierSize: entry.frontierSize,
          selected: {
            action: action?.action ?? diagnostic?.selectedAction ?? null,
            x: action?.x ?? diagnostic?.selectedX ?? null,
            y: action?.y ?? diagnostic?.selectedY ?? null,
            hazard: action?.posteriorHazard ?? diagnostic?.selectedHazard ?? null,
            pressure: diagnostic?.selectedPressure ?? null,
            pressureScore: action?.pressureScore ?? null,
            actionRisk: action?.actionRisk ?? diagnostic?.selectedActionRisk ?? null,
            riskSource: action?.riskSource ?? diagnostic?.riskSource ?? null,
            ess: action?.ess ?? diagnostic?.ess ?? null,
            terminalAfter: action?.terminalAfter ?? null,
            leakFree: diagnostic?.leakFree ?? null,
          },
          hazardRange: {
            min: diagnostic?.minHazard ?? min(entry.candidates.map((candidate) => candidate.hazard)),
            max: diagnostic?.maxHazard ?? max(entry.candidates.map((candidate) => candidate.hazard)),
          },
          observedPressureCount: diagnostic?.observedPressureCount ?? null,
          candidates: entry.candidates
            .slice(0, posteriorCandidateLimit)
            .map(compactCandidate),
        };
      }),
    };
  });
}

async function main() {
  const [
    phase10Manifest,
    bestWorstCsv,
    phase12Manifest,
    phase12Summary,
    phase12RegretCsv,
    phase12ActionsCsv,
    phase12DiagnosticsCsv,
    phase12PosteriorMap,
  ] = await Promise.all([
    readText(sourcePaths.phase10Manifest).then(JSON.parse),
    readText(sourcePaths.phase10BestWorst),
    readText(sourcePaths.phase12Manifest).then(JSON.parse),
    readText(sourcePaths.phase12Summary).then(JSON.parse),
    readText(sourcePaths.phase12RegretSummary),
    readText(sourcePaths.phase12Actions),
    readText(sourcePaths.phase12Diagnostics),
    readText(sourcePaths.phase12PosteriorMap).then(JSON.parse),
  ]);

  const bestWorst = parseCsv(bestWorstCsv).map(cleanBestWorst);
  const regretRows = parseCsv(phase12RegretCsv).map(cleanRegret);
  const actions = parseCsv(phase12ActionsCsv).map(cleanAction);
  const diagnostics = parseCsv(phase12DiagnosticsCsv).map(cleanDiagnostic);
  const targetSummary = targetSummaries(regretRows);
  const cells = buildCellRows(phase12Manifest, regretRows);
  const gate = pressureFloorGate(targetSummary);
  const posteriorCells = buildPosteriorCells({
    cells,
    bestWorst,
    regretRows,
    actions,
    diagnostics,
    posteriorMap: phase12PosteriorMap,
  });

  const data = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    purpose:
      "Public Pressure Mines Phase 13 data surface. Generated from Phase 10 operating-envelope receipts and Phase 12 same-field Bayesian pressure-floor receipts; do not hand-edit.",
    sourcePaths,
    claimBoundary:
      "Pressure Mines remains an Operating-Envelope Study. The Bayesian comparator here is a pressure-floor parity receipt versus naive_pressure and sundog_minimal, not a claim that the current posterior policy dominates sundog_lean or clears the board.",
    phase10: {
      phase: phase10Manifest.phase,
      generatedAt: phase10Manifest.generatedAt,
      verdict: phase10Manifest.verdict,
      cellCount: phase10Manifest.cellCount,
      trialCount: phase10Manifest.trialCount,
      bestWorst,
    },
    phase12: {
      phase: phase12Manifest.phase,
      startedAt: phase12Manifest.startedAt,
      completedAt: phase12Manifest.completedAt,
      wallSeconds: phase12Manifest.wallSeconds,
      trials: phase12Manifest.completedTrials,
      cells: cells.length,
      particleCount: phase12Manifest.args?.particleCount,
      sensorSeedPolicy: phase12Manifest.args?.sensorSeedPolicy,
      leakFree: Boolean(phase12Summary.leakFree),
      posteriorDecisions: phase12Summary.posteriorDecisions,
      pressureFloorGate: gate,
      targetSummary,
      promotedCandidate: findPromotedCandidate(regretRows),
      classCounts: countBy(cells, (cell) => cell.cellClass),
    },
    claimCards: [
      {
        id: "phase10-operating-envelope",
        label: "Phase 10 operating envelope",
        status: phase10Manifest.verdict,
        scope:
          "A narrow confirmed density/noise/dropout pocket is published beside its matched failure region.",
      },
      {
        id: "phase12-bayes-pressure-floor",
        label: "Phase 12 Bayesian pressure floor",
        status: gate.pass ? "PRESSURE_FLOOR_RECEIPT" : "REPAIR_REQUIRED",
        scope:
          "Same-field pressure-budget Bayes matches naive_pressure and sundog_minimal floors across the Phase 10 cell slate; sundog_lean remains a separate stronger comparator in the promoted candidate.",
      },
    ],
    posteriorCells,
    cells,
  };

  const absoluteOutput = join(root, outputPath);
  await mkdir(dirname(absoluteOutput), { recursive: true });
  await writeFile(absoluteOutput, `${JSON.stringify(data, null, 2)}\n`, "utf8");
  console.log(`mines public data built: ${outputPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
