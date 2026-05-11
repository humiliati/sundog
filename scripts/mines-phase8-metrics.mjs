import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const PRESSURE_THRESHOLDS = Object.freeze([0.5, 0.75, 1.0, 1.2, 1.5, 2.0]);
const GRADIENT_THRESHOLDS = Object.freeze([0.1, 0.25, 0.5, 0.75, 1.0]);

function parseArgs(argv) {
  const args = {
    phase: "phase8-events",
    input: "results/mines/phase7-smoke",
    out: "results/mines/phase8-events",
    pressureWarning: 1.2,
    confidenceWarning: 0.5,
    frontierCollapseSize: 2,
    gradientWarning: 0.5,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    if (!flag.startsWith("--")) continue;
    const value = argv[i + 1];
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`${flag} requires a value`);
    }
    i += 1;

    if (flag === "--phase") args.phase = value;
    else if (flag === "--in") args.input = value;
    else if (flag === "--out") args.out = value;
    else if (flag === "--pressure-warning") args.pressureWarning = Number.parseFloat(value);
    else if (flag === "--confidence-warning") args.confidenceWarning = Number.parseFloat(value);
    else if (flag === "--frontier-collapse-size") args.frontierCollapseSize = Number.parseInt(value, 10);
    else if (flag === "--gradient-warning") args.gradientWarning = Number.parseFloat(value);
    else throw new Error(`Unknown flag: ${flag}`);
  }

  if (!Number.isFinite(args.pressureWarning)) throw new Error("--pressure-warning must be numeric");
  if (!Number.isFinite(args.confidenceWarning)) throw new Error("--confidence-warning must be numeric");
  if (!Number.isInteger(args.frontierCollapseSize) || args.frontierCollapseSize < 0) {
    throw new Error("--frontier-collapse-size must be integer >= 0");
  }
  if (!Number.isFinite(args.gradientWarning)) throw new Error("--gradient-warning must be numeric");

  return args;
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (/[",\r\n]/.test(text)) return `"${text.replaceAll("\"", "\"\"")}"`;
  return text;
}

function toCsv(rows, columns = null) {
  if (rows.length === 0) return "";
  const headers = columns ?? Object.keys(rows[0]);
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map((column) => csvEscape(row[column])).join(","));
  }
  return `${lines.join("\n")}\n`;
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (inQuotes) {
      if (ch === "\"" && next === "\"") {
        field += "\"";
        i += 1;
      } else if (ch === "\"") {
        inQuotes = false;
      } else {
        field += ch;
      }
    } else if (ch === "\"") {
      inQuotes = true;
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
  if (rows.length === 0) return [];
  const headers = rows[0];
  return rows.slice(1)
    .filter((values) => values.some((value) => value !== ""))
    .map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])));
}

function toNumber(value) {
  if (value === "" || value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function toBool(value) {
  return value === true || value === "true";
}

function roundMetric(value, digits = 6) {
  if (value === null || value === undefined || Number.isNaN(value)) return null;
  if (!Number.isFinite(value)) return value;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function mean(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) return null;
  return finite.reduce((acc, value) => acc + value, 0) / finite.length;
}

function sum(values) {
  return values.reduce((acc, value) => acc + (Number.isFinite(value) ? value : 0), 0);
}

function ratio(numerator, denominator) {
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator <= 0) return null;
  return numerator / denominator;
}

function readJsonl(text) {
  return text.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

function numericTrace(row) {
  return {
    ...row,
    seed: toNumber(row.seed),
    turnAfter: toNumber(row.turnAfter),
    controllerTurn: toNumber(row.controllerTurn),
    attemptedX: toNumber(row.attemptedX),
    attemptedY: toNumber(row.attemptedY),
    appliedX: toNumber(row.appliedX),
    appliedY: toNumber(row.appliedY),
    appliedIndex: toNumber(row.appliedIndex),
    illegalActionCount: toNumber(row.illegalActionCount),
    rawSafeTiles: toNumber(row.rawSafeTiles),
    safeTilesAfterOpening: toNumber(row.safeTilesAfterOpening),
    falseFlagCount: toNumber(row.falseFlagCount),
    flagCount: toNumber(row.flagCount),
    scanCount: toNumber(row.scanCount),
    scansRemaining: toNumber(row.scansRemaining),
    scanReading: toNumber(row.scanReading),
    frontierSize: toNumber(row.frontierSize),
    meanFrontierConfidence: toNumber(row.meanFrontierConfidence),
    actionIndex: toNumber(row.actionIndex),
    actionPressure: toNumber(row.actionPressure),
    actionConfidence: toNumber(row.actionConfidence),
    actionGradientMagnitude: toNumber(row.actionGradientMagnitude),
    actionApplied: toBool(row.actionApplied),
    illegalFallback: toBool(row.illegalFallback),
  };
}

function numericTrial(row) {
  return {
    ...row,
    seed: toNumber(row.seed),
    width: toNumber(row.width),
    height: toNumber(row.height),
    mineCount: toNumber(row.mineCount),
    mineDensity: toNumber(row.mineDensity),
    scanBudget: toNumber(row.scanBudget),
    turnCap: toNumber(row.turnCap),
    survived: toBool(row.survived),
    fullClear: toBool(row.fullClear),
    turns: toNumber(row.turns),
    controllerTurns: toNumber(row.controllerTurns),
    revealCount: toNumber(row.revealCount),
    flagCount: toNumber(row.flagCount),
    scanCount: toNumber(row.scanCount),
    illegalActionCount: toNumber(row.illegalActionCount),
    openingSafeCount: toNumber(row.openingSafeCount),
    rawSafeTiles: toNumber(row.rawSafeTiles),
    safeTilesAfterOpening: toNumber(row.safeTilesAfterOpening),
    budgetAdjustedSafeTiles: toNumber(row.budgetAdjustedSafeTiles),
    falseFlagCount: toNumber(row.falseFlagCount),
    pressureThreshold: toNumber(row.pressureThreshold),
    sigma: toNumber(row.sigma),
    sigmaNoise: toNumber(row.sigmaNoise),
    dropoutRate: toNumber(row.dropoutRate),
    delaySteps: toNumber(row.delaySteps),
  };
}

function groupBy(rows, keyFn) {
  const groups = new Map();
  for (const row of rows) {
    const key = keyFn(row);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  return groups;
}

function trialKey(row) {
  return `${row.preset}\t${row.sensorCell}\t${row.seed}\t${row.mode}`;
}

function warningFlags(row, args) {
  const pressureWarning = Number.isFinite(row.actionPressure) && row.actionPressure >= args.pressureWarning;
  const confidenceWarning = Number.isFinite(row.actionConfidence) && row.actionConfidence < args.confidenceWarning;
  const frontierConfidenceWarning = Number.isFinite(row.meanFrontierConfidence)
    && row.meanFrontierConfidence < args.confidenceWarning;
  const gradientWarning = Number.isFinite(row.actionGradientMagnitude)
    && row.actionGradientMagnitude >= args.gradientWarning;
  return {
    pressureWarning,
    confidenceWarning,
    frontierConfidenceWarning,
    gradientWarning,
    anyWarning: pressureWarning || confidenceWarning || frontierConfidenceWarning || gradientWarning,
  };
}

function pushEvent(events, row, eventType, fields = {}) {
  events.push({
    phase: row.phase,
    sourcePhase: row.sourcePhase,
    trialId: row.trialId,
    preset: row.preset,
    sensorCell: row.sensorCell,
    mode: row.mode,
    seed: row.seed,
    turnAfter: row.turnAfter,
    controllerTurn: row.controllerTurn,
    eventType,
    actionType: row.appliedActionType,
    actionIndex: row.appliedIndex,
    actionPressure: row.actionPressure,
    actionConfidence: row.actionConfidence,
    actionGradientMagnitude: row.actionGradientMagnitude,
    frontierSize: row.frontierSize,
    meanFrontierConfidence: row.meanFrontierConfidence,
    rawSafeTiles: row.rawSafeTiles,
    safeTilesAfterOpening: row.safeTilesAfterOpening,
    falseFlagCount: row.falseFlagCount,
    scanCount: row.scanCount,
    terminalAfter: row.terminalAfter,
    ...fields,
  });
}

function labelEvents(traceRows, args) {
  const events = [];
  for (const [, rows] of groupBy(traceRows, (row) => row.trialId)) {
    rows.sort((a, b) => a.turnAfter - b.turnAfter);
    let previous = null;
    let collapsed = false;
    let confidenceLost = false;
    for (const row of rows) {
      const flags = warningFlags(row, args);
      const safeDelta = previous ? row.rawSafeTiles - previous.rawSafeTiles : row.rawSafeTiles;
      const falseFlagDelta = previous ? row.falseFlagCount - previous.falseFlagCount : row.falseFlagCount;
      const isReveal = row.appliedActionType === "reveal";
      const isMineTrigger = isReveal && row.terminalAfter === "mine_triggered";

      if (isReveal && !isMineTrigger && safeDelta > 0) {
        pushEvent(events, row, "safe_reveal", { safeDelta });
      }
      if (isReveal && row.controllerTurn > 0 && flags.anyWarning) {
        pushEvent(events, row, "risky_reveal", {
          pressureWarning: flags.pressureWarning,
          confidenceWarning: flags.confidenceWarning || flags.frontierConfidenceWarning,
          gradientWarning: flags.gradientWarning,
        });
      }
      if (isMineTrigger) {
        pushEvent(events, row, "mine_trigger", { hazardousReveal: true });
      }
      if (falseFlagDelta > 0) {
        pushEvent(events, row, "false_flag", { falseFlagDelta });
      }
      if (row.appliedActionType === "scan" && Number.isFinite(row.scanReading)) {
        pushEvent(events, row, "scan_success", { scanReading: row.scanReading });
      }

      const nowCollapsed = Number.isFinite(row.frontierSize) && row.frontierSize <= args.frontierCollapseSize;
      if (nowCollapsed && !collapsed) {
        pushEvent(events, row, "frontier_collapse", { frontierCollapseSize: args.frontierCollapseSize });
      }
      collapsed = nowCollapsed;

      const nowConfidenceLost = Number.isFinite(row.meanFrontierConfidence)
        && row.meanFrontierConfidence < args.confidenceWarning;
      if (nowConfidenceLost && !confidenceLost) {
        pushEvent(events, row, "confidence_loss", { confidenceWarning: args.confidenceWarning });
      }
      if ((collapsed || confidenceLost) && !nowCollapsed && !nowConfidenceLost && row.controllerTurn > 0) {
        pushEvent(events, row, "recovery", {
          recoveredFromFrontierCollapse: collapsed,
          recoveredFromConfidenceLoss: confidenceLost,
        });
      }
      confidenceLost = nowConfidenceLost;
      previous = row;
    }
  }
  return events;
}

function firstEventTurn(events, trialId, predicate) {
  const turns = events
    .filter((event) => event.trialId === trialId && predicate(event))
    .map((event) => event.controllerTurn)
    .filter(Number.isFinite);
  return turns.length > 0 ? Math.min(...turns) : null;
}

function summarizeTrials(trialRows, events) {
  const eventsByTrial = groupBy(events, (event) => event.trialId);
  return trialRows.map((trial) => {
    const trialEvents = eventsByTrial.get(trial.trialId) ?? [];
    const count = (eventType) => trialEvents.filter((event) => event.eventType === eventType).length;
    const firstWarningTurn = firstEventTurn(trialEvents, trial.trialId, (event) => (
      event.eventType === "risky_reveal"
      || event.eventType === "frontier_collapse"
      || event.eventType === "confidence_loss"
    ));
    const firstForcedRiskTurn = firstEventTurn(trialEvents, trial.trialId, (event) => (
      event.eventType === "risky_reveal"
      || event.eventType === "mine_trigger"
    ));
    return {
      phase: trial.phase,
      sourcePhase: trial.sourcePhase,
      trialId: trial.trialId,
      preset: trial.preset,
      sensorCell: trial.sensorCell,
      mode: trial.mode,
      seed: trial.seed,
      terminal: trial.terminal,
      survived: trial.survived,
      turns: trial.turns,
      rawSafeTiles: trial.rawSafeTiles,
      safeTilesAfterOpening: trial.safeTilesAfterOpening,
      budgetAdjustedSafeTiles: trial.budgetAdjustedSafeTiles,
      falseFlagCount: trial.falseFlagCount,
      scanCount: trial.scanCount,
      scanBudget: trial.scanBudget,
      safeRevealEventCount: count("safe_reveal"),
      riskyRevealCount: count("risky_reveal"),
      mineTriggerCount: count("mine_trigger"),
      falseFlagEventCount: count("false_flag"),
      scanSuccessCount: count("scan_success"),
      frontierCollapseCount: count("frontier_collapse"),
      confidenceLossCount: count("confidence_loss"),
      recoveryCount: count("recovery"),
      hazardousRevealRate: roundMetric(ratio(count("mine_trigger"), trial.revealCount)),
      riskyRevealRate: roundMetric(ratio(count("risky_reveal"), trial.revealCount)),
      scanBudgetEfficiency: roundMetric(ratio(trial.safeTilesAfterOpening, trial.scanCount)),
      firstWarningTurn,
      firstForcedRiskTurn,
      leadTimeBeforeForcedRisk: Number.isFinite(firstWarningTurn) && Number.isFinite(firstForcedRiskTurn)
        ? firstForcedRiskTurn - firstWarningTurn
        : null,
      browserReplayUrl: trial.browserReplayUrl,
      actionTraceHash: trial.actionTraceHash,
    };
  });
}

function summarizeModes(trialSummaries) {
  return Array.from(groupBy(trialSummaries, (row) => `${row.preset}\t${row.sensorCell}\t${row.mode}`).entries())
    .map(([key, rows]) => {
      const [preset, sensorCell, mode] = key.split("\t");
      return {
        phase: rows[0].phase,
        preset,
        sensorCell,
        mode,
        n: rows.length,
        survivalRate: roundMetric(mean(rows.map((row) => row.survived ? 1 : 0))),
        meanRawSafeTiles: roundMetric(mean(rows.map((row) => row.rawSafeTiles))),
        meanBudgetAdjustedSafeTiles: roundMetric(mean(rows.map((row) => row.budgetAdjustedSafeTiles))),
        meanRiskyRevealCount: roundMetric(mean(rows.map((row) => row.riskyRevealCount))),
        meanRiskyRevealRate: roundMetric(mean(rows.map((row) => row.riskyRevealRate))),
        meanMineTriggerCount: roundMetric(mean(rows.map((row) => row.mineTriggerCount))),
        meanFalseFlagCount: roundMetric(mean(rows.map((row) => row.falseFlagCount))),
        meanFrontierCollapseCount: roundMetric(mean(rows.map((row) => row.frontierCollapseCount))),
        meanConfidenceLossCount: roundMetric(mean(rows.map((row) => row.confidenceLossCount))),
        meanRecoveryCount: roundMetric(mean(rows.map((row) => row.recoveryCount))),
        meanScanBudgetEfficiency: roundMetric(mean(rows.map((row) => row.scanBudgetEfficiency))),
        meanLeadTimeBeforeForcedRisk: roundMetric(mean(rows.map((row) => row.leadTimeBeforeForcedRisk))),
      };
    })
    .sort((a, b) => a.preset.localeCompare(b.preset) || a.sensorCell.localeCompare(b.sensorCell) || a.mode.localeCompare(b.mode));
}

function makeMatchedComparisons(trialSummaries, baselineModes = ["random_reveal", "naive_pressure"]) {
  const byKey = new Map(trialSummaries.map((row) => [trialKey(row), row]));
  const rows = [];
  for (const row of trialSummaries) {
    for (const baselineMode of baselineModes) {
      if (row.mode === baselineMode) continue;
      const baseline = byKey.get(`${row.preset}\t${row.sensorCell}\t${row.seed}\t${baselineMode}`);
      if (!baseline) continue;
      rows.push({
        phase: row.phase,
        preset: row.preset,
        sensorCell: row.sensorCell,
        seed: row.seed,
        mode: row.mode,
        baselineMode,
        budgetAdjustedSafeTilesDelta: row.budgetAdjustedSafeTiles - baseline.budgetAdjustedSafeTiles,
        survivalDelta: (row.survived ? 1 : 0) - (baseline.survived ? 1 : 0),
        riskyRevealDelta: row.riskyRevealCount - baseline.riskyRevealCount,
        mineTriggerDelta: row.mineTriggerCount - baseline.mineTriggerCount,
        falseFlagDelta: row.falseFlagCount - baseline.falseFlagCount,
        frontierCollapseDelta: row.frontierCollapseCount - baseline.frontierCollapseCount,
        confidenceLossDelta: row.confidenceLossCount - baseline.confidenceLossCount,
        recoveryDelta: row.recoveryCount - baseline.recoveryCount,
        leadTimeDelta: Number.isFinite(row.leadTimeBeforeForcedRisk) && Number.isFinite(baseline.leadTimeBeforeForcedRisk)
          ? row.leadTimeBeforeForcedRisk - baseline.leadTimeBeforeForcedRisk
          : null,
      });
    }
  }
  return rows;
}

function summarizeMatchedComparisons(rows) {
  return Array.from(groupBy(rows, (row) => `${row.preset}\t${row.sensorCell}\t${row.mode}\t${row.baselineMode}`).entries())
    .map(([key, group]) => {
      const [preset, sensorCell, mode, baselineMode] = key.split("\t");
      return {
        phase: group[0].phase,
        preset,
        sensorCell,
        mode,
        baselineMode,
        n: group.length,
        meanBudgetAdjustedSafeTilesDelta: roundMetric(mean(group.map((row) => row.budgetAdjustedSafeTilesDelta))),
        meanSurvivalDelta: roundMetric(mean(group.map((row) => row.survivalDelta))),
        meanRiskyRevealDelta: roundMetric(mean(group.map((row) => row.riskyRevealDelta))),
        meanMineTriggerDelta: roundMetric(mean(group.map((row) => row.mineTriggerDelta))),
        meanFalseFlagDelta: roundMetric(mean(group.map((row) => row.falseFlagDelta))),
        meanFrontierCollapseDelta: roundMetric(mean(group.map((row) => row.frontierCollapseDelta))),
        meanConfidenceLossDelta: roundMetric(mean(group.map((row) => row.confidenceLossDelta))),
        meanRecoveryDelta: roundMetric(mean(group.map((row) => row.recoveryDelta))),
        meanLeadTimeDelta: roundMetric(mean(group.map((row) => row.leadTimeDelta))),
      };
    })
    .sort((a, b) => a.preset.localeCompare(b.preset) || a.sensorCell.localeCompare(b.sensorCell) || a.mode.localeCompare(b.mode) || a.baselineMode.localeCompare(b.baselineMode));
}

function warningSweep(traceRows) {
  const revealRows = traceRows.filter((row) => row.appliedActionType === "reveal" && row.controllerTurn > 0);
  const out = [];
  for (const [key, rows] of groupBy(revealRows, (row) => `${row.preset}\t${row.sensorCell}\t${row.mode}`)) {
    const [preset, sensorCell, mode] = key.split("\t");
    for (const threshold of PRESSURE_THRESHOLDS) {
      const eligible = rows.filter((row) => Number.isFinite(row.actionPressure));
      out.push(sweepRow({ rows: eligible, threshold, signal: "pressure", preset, sensorCell, mode }));
    }
    for (const threshold of GRADIENT_THRESHOLDS) {
      const eligible = rows.filter((row) => Number.isFinite(row.actionGradientMagnitude));
      out.push(sweepRow({ rows: eligible, threshold, signal: "gradient", preset, sensorCell, mode }));
    }
  }
  return out;
}

function sweepRow({ rows, threshold, signal, preset, sensorCell, mode }) {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;
  for (const row of rows) {
    const value = signal === "pressure" ? row.actionPressure : row.actionGradientMagnitude;
    const warning = value >= threshold;
    const mineTrigger = row.terminalAfter === "mine_triggered";
    if (warning && mineTrigger) tp += 1;
    else if (warning && !mineTrigger) fp += 1;
    else if (!warning && mineTrigger) fn += 1;
    else tn += 1;
  }
  return {
    preset,
    sensorCell,
    mode,
    signal,
    threshold,
    n: rows.length,
    truePositive: tp,
    falsePositive: fp,
    trueNegative: tn,
    falseNegative: fn,
    precision: roundMetric(ratio(tp, tp + fp)),
    recall: roundMetric(ratio(tp, tp + fn)),
    falsePositiveRate: roundMetric(ratio(fp, fp + tn)),
  };
}

function markdownReport({ args, modeSummaries, comparisonSummaries }) {
  const leanRows = comparisonSummaries.filter((row) => row.mode === "sundog_lean" && row.baselineMode === "naive_pressure");
  return [
    "# Sundog Pressure Mines Phase 8 Event Metrics",
    "",
    `Input: \`${args.input}\``,
    `Pressure warning: \`${args.pressureWarning}\``,
    `Confidence warning: \`${args.confidenceWarning}\``,
    `Frontier collapse size: \`${args.frontierCollapseSize}\``,
    "",
    "## Lean Versus Naive",
    "",
    "| preset | sensor cell | safe delta | risky reveal delta | false flag delta | recovery delta |",
    "| --- | --- | ---: | ---: | ---: | ---: |",
    ...leanRows.map((row) => `| ${row.preset} | ${row.sensorCell} | ${row.meanBudgetAdjustedSafeTilesDelta} | ${row.meanRiskyRevealDelta} | ${row.meanFalseFlagDelta} | ${row.meanRecoveryDelta} |`),
    "",
    "## Mode Event Summary",
    "",
    "| preset | sensor cell | mode | survival | risky reveals | frontier collapses | recoveries |",
    "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ...modeSummaries.map((row) => `| ${row.preset} | ${row.sensorCell} | ${row.mode} | ${row.survivalRate} | ${row.meanRiskyRevealCount} | ${row.meanFrontierCollapseCount} | ${row.meanRecoveryCount} |`),
    "",
    "## Scope",
    "",
    "This is event instrumentation over replayable Phase 7 traces. It labels useful and risky decision moments; it is not an operating-envelope verdict.",
    "",
  ].join("\n");
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputDir = path.resolve(repoRoot, args.input);
  const outDir = path.resolve(repoRoot, args.out);
  const sourceTrialRows = parseCsv(await readFile(path.join(inputDir, "trial-rows.csv"), "utf8")).map(numericTrial);
  const sourceTraceRows = readJsonl(await readFile(path.join(inputDir, "step-traces.jsonl"), "utf8")).map(numericTrace);
  const trialRows = sourceTrialRows.map((row) => ({ ...row, sourcePhase: row.phase, phase: args.phase }));
  const traceRows = sourceTraceRows.map((row) => ({ ...row, sourcePhase: row.phase, phase: args.phase }));
  const eventRows = labelEvents(traceRows, args);
  const trialEventSummaries = summarizeTrials(trialRows, eventRows);
  const modeEventSummaries = summarizeModes(trialEventSummaries);
  const matchedEventComparisons = makeMatchedComparisons(trialEventSummaries);
  const matchedEventComparisonSummary = summarizeMatchedComparisons(matchedEventComparisons);
  const warningThresholdSweeps = warningSweep(traceRows);
  const manifest = {
    schema: "sundog.mines.phase8-events.v1",
    generatedAt: new Date().toISOString(),
    phase: args.phase,
    input: path.relative(repoRoot, inputDir),
    out: path.relative(repoRoot, outDir),
    pressureWarning: args.pressureWarning,
    confidenceWarning: args.confidenceWarning,
    frontierCollapseSize: args.frontierCollapseSize,
    gradientWarning: args.gradientWarning,
    trialCount: trialRows.length,
    traceRowCount: traceRows.length,
    eventCount: eventRows.length,
  };

  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "event-rows.csv"), toCsv(eventRows));
  await writeFile(path.join(outDir, "trial-event-summary.csv"), toCsv(trialEventSummaries));
  await writeFile(path.join(outDir, "mode-event-summary.csv"), toCsv(modeEventSummaries));
  await writeFile(path.join(outDir, "matched-event-comparisons.csv"), toCsv(matchedEventComparisons));
  await writeFile(path.join(outDir, "matched-event-comparison-summary.csv"), toCsv(matchedEventComparisonSummary));
  await writeFile(path.join(outDir, "warning-threshold-sweeps.csv"), toCsv(warningThresholdSweeps));
  await writeFile(path.join(outDir, "summary.json"), `${JSON.stringify({
    ...manifest,
    modeEventSummaries,
    matchedEventComparisonSummary,
  }, null, 2)}\n`);
  await writeFile(path.join(outDir, "phase8-events.md"), markdownReport({
    args,
    modeSummaries: modeEventSummaries,
    comparisonSummaries: matchedEventComparisonSummary,
  }));

  console.log(`Mines Phase 8 events read ${path.relative(repoRoot, inputDir)}`);
  console.log(`Trials: ${trialRows.length}; trace rows: ${traceRows.length}; events: ${eventRows.length}`);
  console.log(`Wrote ${path.relative(repoRoot, outDir)}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
