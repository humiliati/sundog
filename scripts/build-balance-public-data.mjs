import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

const root = process.cwd();

const sourcePaths = Object.freeze({
  phase10Manifest: "results/balance/phase10-envelope/manifest.json",
  phase10Envelope: "results/balance/phase10-envelope/envelope.csv",
  phase15Manifest: "results/balance/phase15-phase10-full-lock/manifest.json",
  phase15RegretSummary: "results/balance/phase15-phase10-full-lock/bayes-regret-summary.csv",
});

const outputPath = "public/data/balance-phase16-claim-lock.json";

const laneLabels = Object.freeze({
  hard_gate: "Standard hard gate",
  observation_parity_gate: "Observation parity gate",
  reported_only: "Reported-only boundary",
});

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
    const key = keyFn(row);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return Object.fromEntries([...counts.entries()].sort(([a], [b]) => String(a).localeCompare(String(b))));
}

function cleanPhase10Cell(row) {
  return {
    cellId: row.cell_id,
    caseId: row.case_id,
    axis: row.axis,
    axisValue: asNumber(row.axis_value),
    preset: row.preset,
    lightElevationDeg: asNumber(row.light_elev_deg),
    sensorDelaySteps: asNumber(row.delay_steps),
    sensorNoiseStd: asNumber(row.noise_sigma),
    sensorDropoutRate: asNumber(row.dropout_rate),
    forceLimit: asNumber(row.force_limit),
    railLimit: asNumber(row.rail_limit),
    disturbanceForce: asNumber(row.disturbance_mag),
    cellClass: row.cell_class,
    staticBoundaryMechanisms: row.static_boundary_mechanisms
      ? row.static_boundary_mechanisms.split("|").filter(Boolean)
      : [],
    phase10: {
      survival: {
        sundog: asNumber(row.survival_sundog_mean),
        naive: asNumber(row.survival_naive_mean),
        passive: asNumber(row.survival_passive_mean),
        oracle: asNumber(row.survival_oracle_mean),
      },
      sundogNaivePairedMarginMean: asNumber(row.sundog_naive_paired_margin_mean),
      pairedMarginBootstrapLow: asNumber(row.paired_margin_bootstrap_low),
      pairedMarginBootstrapHigh: asNumber(row.paired_margin_bootstrap_high),
      sundogForceBudgetMean: asNumber(row.sundog_force_budget_mean),
      sundogSaturationRateMean: asNumber(row.sundog_saturation_rate_mean),
      seedCount: asNumber(row.seed_count),
      replayUrl: row.replay_url || null,
      naiveReplayUrl: row.naive_replay_url || null,
      oracleReplayUrl: row.oracle_replay_url || null,
    },
  };
}

function cleanPhase15(row) {
  return {
    phase: row.phase,
    preset: row.preset,
    cellId: row.cellId,
    axis: row.axis,
    axisValue: asNumber(row.axisValue),
    cellClass: row.cellClass,
    admissionLane: row.admissionLane,
    admissionLabel: laneLabels[row.admissionLane] ?? row.admissionLane,
    admissionReason: row.admissionReason,
    claimGateRequired: asBool(row.claimGateRequired),
    bayesSanityGateRequired: asBool(row.bayesSanityGateRequired),
    n: asNumber(row.n),
    meanRegretVsSundog: asNumber(row.meanRegretVsSundog),
    negativeRegretRate: asNumber(row.negativeRegretRate),
    sundogParityPass: asBool(row.sundogParityPass),
    meanBayesMinusNaive: asNumber(row.meanBayesMinusNaive),
    bayesWorseThanNaiveRate: asNumber(row.bayesWorseThanNaiveRate),
    bayesSanityPass: asBool(row.bayesSanityPass),
    claimGatePass: asBool(row.claimGatePass),
  };
}

function mergeCells(phase10Rows, phase15Rows) {
  const phase10ByCell = new Map(phase10Rows.map((row) => [row.cellId, row]));
  return phase15Rows.map((phase15) => {
    const phase10 = phase10ByCell.get(phase15.cellId) ?? {};
    return {
      ...phase10,
      cellId: phase15.cellId,
      axis: phase15.axis || phase10.axis || "",
      axisValue: phase15.axisValue ?? phase10.axisValue ?? null,
      preset: phase15.preset || phase10.preset || "",
      cellClass: phase15.cellClass || phase10.cellClass || "",
      phase15,
    };
  });
}

function summarizeLane(lane, cells) {
  const regrets = cells.map((cell) => cell.phase15.meanRegretVsSundog);
  const sanityPass = cells.filter((cell) => cell.phase15.bayesSanityPass).length;
  const claimPass = cells.filter((cell) => cell.phase15.claimGatePass).length;
  const hardGateCells = cells.filter((cell) => cell.phase15.claimGateRequired).length;
  const reportedOnlyCells = cells.length - hardGateCells;
  return {
    lane,
    label: laneLabels[lane] ?? lane,
    cells: cells.length,
    hardGateCells,
    hardGatePassCells: claimPass,
    reportedOnlyCells,
    bayesSanityPassCells: sanityPass,
    bayesSanityCells: cells.length,
    meanRegretVsSundog: round(mean(regrets), 6),
    minRegretVsSundog: round(min(regrets), 6),
    maxRegretVsSundog: round(max(regrets), 6),
    axes: countBy(cells, (cell) => cell.axis),
  };
}

function compactCell(cell) {
  return {
    cellId: cell.cellId,
    preset: cell.preset,
    axis: cell.axis,
    axisValue: cell.axisValue,
    cellClass: cell.cellClass,
    lightElevationDeg: cell.lightElevationDeg ?? null,
    sensorDelaySteps: cell.sensorDelaySteps ?? null,
    sensorNoiseStd: cell.sensorNoiseStd ?? null,
    sensorDropoutRate: cell.sensorDropoutRate ?? null,
    staticBoundaryMechanisms: cell.staticBoundaryMechanisms ?? [],
    phase10: cell.phase10 ?? null,
    phase15: cell.phase15,
  };
}

async function main() {
  const [phase10Manifest, phase15Manifest, phase10Csv, phase15Csv] = await Promise.all([
    readText(sourcePaths.phase10Manifest).then(JSON.parse),
    readText(sourcePaths.phase15Manifest).then(JSON.parse),
    readText(sourcePaths.phase10Envelope),
    readText(sourcePaths.phase15RegretSummary),
  ]);

  const phase10Rows = parseCsv(phase10Csv).map(cleanPhase10Cell);
  const phase15Rows = parseCsv(phase15Csv).map(cleanPhase15);
  const cells = mergeCells(phase10Rows, phase15Rows);
  const hardGateCells = cells.filter((cell) => cell.phase15.claimGateRequired);
  const reportedOnlyCells = cells.filter((cell) => !cell.phase15.claimGateRequired);
  const regrets = cells.map((cell) => cell.phase15.meanRegretVsSundog);
  const laneOrder = ["hard_gate", "observation_parity_gate", "reported_only"];
  const admissionLanes = laneOrder
    .map((lane) => summarizeLane(lane, cells.filter((cell) => cell.phase15.admissionLane === lane)))
    .filter((lane) => lane.cells > 0);

  const data = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    purpose:
      "Public Balance Phase 16 data surface. Generated from Phase 10 operating-envelope receipts and Phase 15 Bayesian-floor claim-lock receipts; do not hand-edit.",
    sourcePaths,
    claimBoundary:
      "Browser cart-pole operating-envelope study only. Same-information Bayesian-floor claim applies inside hard-gate cells; reported-only observation-degradation cells remain explicit boundary diagnostics.",
    phase10: {
      phase: phase10Manifest.phase,
      generatedAt: phase10Manifest.generatedAt,
      verdict: phase10Manifest.verdict,
      trialCount: phase10Manifest.trialCount,
      cellCount: phase10Manifest.cellCount,
      classCounts: countBy(phase10Rows, (cell) => cell.cellClass),
    },
    phase15: {
      phase: phase15Manifest.phase,
      generatedAt: phase15Manifest.generatedAt,
      trialCount: phase15Manifest.trialCount,
      elapsedSeconds: phase15Manifest.elapsedSeconds,
      trialsPerSecond: phase15Manifest.trialsPerSecond,
      auditsPass: Boolean(phase15Manifest.audits?.pass),
      claimGate: phase15Manifest.claimGate,
      cellCount: cells.length,
      hardGateCells: hardGateCells.length,
      reportedOnlyCells: reportedOnlyCells.length,
      hardGatePassCells: hardGateCells.filter((cell) => cell.phase15.claimGatePass).length,
      bayesSanityPassCells: cells.filter((cell) => cell.phase15.bayesSanityPass).length,
      zeroNegativeMeanRegretCells: cells.filter((cell) => (cell.phase15.meanRegretVsSundog ?? -Infinity) >= 0).length,
      meanRegretVsSundog: round(mean(regrets), 6),
      minRegretVsSundog: round(min(regrets), 6),
      maxRegretVsSundog: round(max(regrets), 6),
      admissionLanes,
      classCounts: countBy(cells, (cell) => cell.cellClass),
      axisCounts: countBy(cells, (cell) => cell.axis),
    },
    claimCards: [
      {
        id: "phase10-operating-envelope",
        label: "Phase 10 operating envelope",
        status: phase10Manifest.verdict,
        scope: "Shadow controller versus naive shadow centering across mapped Balance cells.",
      },
      {
        id: "phase15-bayesian-floor",
        label: "Phase 15 Bayesian floor",
        status: phase15Manifest.claimGate?.pass ? "CLAIM_LOCK" : "PENDING",
        scope:
          "Same-information Bayesian-floor receipt with visible hard-gate, observation-parity, and reported-only admission lanes.",
      },
    ],
    cells: cells.map(compactCell).sort((a, b) => (
      a.phase15.admissionLane.localeCompare(b.phase15.admissionLane)
      || a.axis.localeCompare(b.axis)
      || String(a.axisValue ?? "").localeCompare(String(b.axisValue ?? ""), undefined, { numeric: true })
      || a.preset.localeCompare(b.preset)
      || a.cellId.localeCompare(b.cellId)
    )),
  };

  const absoluteOutput = join(root, outputPath);
  await mkdir(dirname(absoluteOutput), { recursive: true });
  await writeFile(absoluteOutput, `${JSON.stringify(data, null, 2)}\n`, "utf8");
  console.log(`balance public data built: ${outputPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
