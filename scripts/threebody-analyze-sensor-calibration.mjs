import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const args = {
    input: "results/threebody/phase8-calibration-sweep",
    out: null,
    candidateMaxWorsenedRate: 0.1,
    candidateMinSurvivalDelta: 0,
    candidateMaxMeanRelError: 1,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    i += 1;

    if (flag === "--input") args.input = value;
    else if (flag === "--out") args.out = value;
    else if (flag === "--candidate-max-worsened-rate") args.candidateMaxWorsenedRate = Number.parseFloat(value);
    else if (flag === "--candidate-min-survival-delta") args.candidateMinSurvivalDelta = Number.parseFloat(value);
    else if (flag === "--candidate-max-mean-rel-error") args.candidateMaxMeanRelError = Number.parseFloat(value);
    else throw new Error(`Unknown flag: ${flag}`);
  }

  if (!Number.isFinite(args.candidateMaxWorsenedRate) || args.candidateMaxWorsenedRate < 0) {
    throw new Error("--candidate-max-worsened-rate must be non-negative");
  }
  if (!Number.isFinite(args.candidateMinSurvivalDelta)) {
    throw new Error("--candidate-min-survival-delta must be a finite number");
  }
  if (!Number.isFinite(args.candidateMaxMeanRelError) || args.candidateMaxMeanRelError < 0) {
    throw new Error("--candidate-max-mean-rel-error must be non-negative");
  }

  return args;
}

function parseCsvLine(line) {
  const values = [];
  let current = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    const next = line[i + 1];
    if (char === "\"" && quoted && next === "\"") {
      current += "\"";
      i += 1;
    } else if (char === "\"") {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  values.push(current);
  return values;
}

function parseValue(value) {
  if (value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) && value.trim() !== "" ? number : value;
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length === 0 || lines[0] === "") return [];
  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, parseValue(values[index] ?? "")]));
  });
}

function csvValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll("\"", "\"\"")}"` : text;
}

function rowsToCsv(rows, explicitColumns = null) {
  const columns = explicitColumns ?? [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const lines = [columns.join(",")];
  for (const row of rows) lines.push(columns.map((column) => csvValue(row[column])).join(","));
  return `${lines.join("\n")}\n`;
}

function mean(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) return null;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function sum(values) {
  return values.reduce((total, value) => total + (Number.isFinite(value) ? value : 0), 0);
}

function ratio(numerator, denominator) {
  if (denominator <= 0) return null;
  return numerator / denominator;
}

function roundMetric(value, digits = 6) {
  if (!Number.isFinite(value)) return null;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function groupRows(rows, keyFn) {
  const groups = new Map();
  for (const row of rows) {
    const key = keyFn(row);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  return groups;
}

function settingKey(row, includeRegime = false) {
  const parts = [
    row.sensorTier,
    row.sensorNoiseStd,
    row.sensorDelaySteps,
    row.microManeuverContaminationStd,
    row.controllerMode,
  ];
  if (includeRegime) parts.push(row.regime);
  return parts.join("\t");
}

function errorKey(row, includeRegime = false) {
  const parts = [
    row.sensorTier,
    row.sensorNoiseStd,
    row.sensorDelaySteps,
    row.microManeuverContaminationStd,
    row.controllerMode,
  ];
  if (includeRegime) parts.push(row.regime);
  return parts.join("\t");
}

function summarizeGroup(key, rows, errorRows = [], includeRegime = false, args) {
  const [
    sensorTier,
    sensorNoiseStdText,
    sensorDelayStepsText,
    microManeuverContaminationStdText,
    controllerMode,
    regime,
  ] = key.split("\t");
  const n = rows.length;
  const bounded = rows.filter((row) => row.terminalOutcome === "bounded").length;
  const passiveBounded = rows.filter((row) => row.passiveOutcome === "bounded").length;
  const worsened = rows.filter((row) => row.outcomeDeltaVsPassive < 0).length;
  const improved = rows.filter((row) => row.outcomeDeltaVsPassive > 0).length;
  const survivalRate = ratio(bounded, n);
  const passiveSurvivalRate = ratio(passiveBounded, n);
  const survivalDeltaVsPassive = survivalRate !== null && passiveSurvivalRate !== null
    ? survivalRate - passiveSurvivalRate
    : null;
  const meanMagnitudeRelError = mean(errorRows.map((row) => row.meanMagnitudeRelError));
  const meanMagnitudeAbsError = mean(errorRows.map((row) => row.meanMagnitudeAbsError));
  const worsenedRate = ratio(worsened, n);
  const candidateEnvelope = survivalDeltaVsPassive !== null
    && survivalDeltaVsPassive >= args.candidateMinSurvivalDelta
    && worsenedRate !== null
    && worsenedRate <= args.candidateMaxWorsenedRate
    && (meanMagnitudeRelError === null || meanMagnitudeRelError <= args.candidateMaxMeanRelError);

  return {
    sensorTier,
    sensorNoiseStd: Number.parseFloat(sensorNoiseStdText),
    sensorDelaySteps: Number.parseInt(sensorDelayStepsText, 10),
    microManeuverContaminationStd: Number.parseFloat(microManeuverContaminationStdText),
    controllerMode,
    ...(includeRegime ? { regime } : {}),
    n,
    bounded,
    passiveBounded,
    survivalRate: roundMetric(survivalRate),
    passiveSurvivalRate: roundMetric(passiveSurvivalRate),
    survivalDeltaVsPassive: roundMetric(survivalDeltaVsPassive),
    improvedOutcomeVsPassive: improved,
    worsenedOutcomeVsPassive: worsened,
    worsenedRate: roundMetric(worsenedRate),
    meanOutcomeDeltaVsPassive: roundMetric(mean(rows.map((row) => row.outcomeDeltaVsPassive))),
    meanTimeDeltaVsPassive: roundMetric(mean(rows.map((row) => row.simulatedTimeDeltaVsPassive))),
    meanDeltaV: roundMetric(mean(rows.map((row) => row.totalDeltaV))),
    meanMinPrimaryDistance: roundMetric(mean(rows.map((row) => row.minPrimaryDistance))),
    meanTidalMagnitudeAuroc: roundMetric(mean(rows.map((row) => row.tidalMagnitudeAuroc))),
    meanLocalAccelerationMagnitudeAuroc: roundMetric(mean(rows.map((row) => row.localAccelerationMagnitudeAuroc))),
    meanMagnitudeRelError: roundMetric(meanMagnitudeRelError),
    meanMagnitudeAbsError: roundMetric(meanMagnitudeAbsError),
    candidateEnvelope,
  };
}

function sortEnvelopeRows(a, b) {
  return (
    Number(b.candidateEnvelope) - Number(a.candidateEnvelope)
    || (b.survivalDeltaVsPassive ?? -Infinity) - (a.survivalDeltaVsPassive ?? -Infinity)
    || (a.worsenedRate ?? Infinity) - (b.worsenedRate ?? Infinity)
    || (a.meanMagnitudeRelError ?? Infinity) - (b.meanMagnitudeRelError ?? Infinity)
    || a.sensorTier.localeCompare(b.sensorTier)
    || a.controllerMode.localeCompare(b.controllerMode)
    || a.sensorNoiseStd - b.sensorNoiseStd
    || a.sensorDelaySteps - b.sensorDelaySteps
  );
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputDir = path.resolve(repoRoot, args.input);
  const outDir = path.resolve(repoRoot, args.out ?? args.input);
  await mkdir(outDir, { recursive: true });

  const pairedRows = parseCsv(await readFile(path.join(inputDir, "paired.csv"), "utf8"));
  const sensorErrorRows = parseCsv(await readFile(path.join(inputDir, "sensor-error-summary.csv"), "utf8"));
  const errorsBySetting = groupRows(sensorErrorRows, (row) => errorKey(row, false));
  const errorsByRegime = groupRows(sensorErrorRows, (row) => errorKey(row, true));

  const envelopeRows = Array.from(groupRows(pairedRows, (row) => settingKey(row, false)).entries())
    .map(([key, rows]) => summarizeGroup(key, rows, errorsBySetting.get(key) ?? [], false, args))
    .sort(sortEnvelopeRows);
  const regimeEnvelopeRows = Array.from(groupRows(pairedRows, (row) => settingKey(row, true)).entries())
    .map(([key, rows]) => summarizeGroup(key, rows, errorsByRegime.get(key) ?? [], true, args))
    .sort(sortEnvelopeRows);
  const candidateRows = envelopeRows.filter((row) => row.candidateEnvelope);

  await writeFile(path.join(outDir, "envelope.csv"), rowsToCsv(envelopeRows), "utf8");
  await writeFile(path.join(outDir, "regime-envelope.csv"), rowsToCsv(regimeEnvelopeRows), "utf8");
  await writeFile(
    path.join(outDir, "candidate-envelope.csv"),
    rowsToCsv(candidateRows, envelopeRows.length > 0 ? Object.keys(envelopeRows[0]) : []),
    "utf8",
  );

  console.log(`[threebody] analyzed ${pairedRows.length} paired calibration rows from ${path.relative(repoRoot, inputDir)}`);
  console.log(`[threebody] wrote envelope.csv, regime-envelope.csv, and candidate-envelope.csv`);
  console.log(`[threebody] candidate envelopes ${candidateRows.length}/${envelopeRows.length}`);
}

main().catch((error) => {
  console.error(`[threebody] ${error.message}`);
  process.exitCode = 1;
});
