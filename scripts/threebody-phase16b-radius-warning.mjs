import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const CHANNELS = [
  { name: "radius", direction: 1 },
  { name: "energy", direction: 1 },
];
const FAVORABLE_VELOCITY_MIN = 1.05;
const PASS_BAR = 0.70;
const COVERAGE_BAR = 18;
const EXPECTED_TRIALS = 288;
const EXPECTED_PHASE16_BRANCH = "A_hazard_warnable";
const EXPECTED_PHASE16 = {
  radius: { perCellMeanAuroc: 0.996624, perCellDefinedCount: 27 },
  energy: { perCellMeanAuroc: 0.655508, perCellDefinedCount: 27 },
};

function parseArgs(argv) {
  const args = {
    inDir: "results/threebody/phase16-hazard-channel-audit-lock",
    outDir: "results/threebody/phase16b-radius-warning-repose",
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    i += 1;
    if (flag === "--in") args.inDir = value;
    else if (flag === "--out") args.outDir = value;
    else throw new Error(`Unknown flag: ${flag}`);
  }
  return args;
}

function repoPath(targetPath) {
  return path.isAbsolute(targetPath) ? targetPath : path.join(repoRoot, targetPath);
}

function csvValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "";
  if (typeof value === "boolean") return value ? "true" : "false";
  const text = String(value);
  if (/[",\n]/.test(text)) return `"${text.replaceAll('"', '""')}"`;
  return text;
}

function rowsToCsv(rows) {
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const lines = [columns.join(",")];
  for (const row of rows) lines.push(columns.map((column) => csvValue(row[column])).join(","));
  return `${lines.join("\n")}\n`;
}

function parseCsvLine(line) {
  const values = [];
  let current = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (quoted) {
      if (char === '"' && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        current += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  values.push(current);
  return values;
}

function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) return [];
  const columns = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    return Object.fromEntries(columns.map((column, index) => [column, values[index] ?? ""]));
  });
}

function round(value, digits = 6) {
  if (!Number.isFinite(value)) return null;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function computeAuroc(samples) {
  const clean = samples.filter((sample) => Number.isFinite(sample.score) && typeof sample.label === "boolean");
  const nPos = clean.filter((sample) => sample.label).length;
  const nNeg = clean.length - nPos;
  if (nPos === 0 || nNeg === 0) return null;

  const sorted = clean.slice().sort((a, b) => a.score - b.score);
  let rank = 1;
  let positiveRankSum = 0;
  for (let i = 0; i < sorted.length;) {
    let j = i + 1;
    while (j < sorted.length && sorted[j].score === sorted[i].score) j += 1;
    const averageRank = (rank + rank + (j - i) - 1) / 2;
    for (let k = i; k < j; k += 1) {
      if (sorted[k].label) positiveRankSum += averageRank;
    }
    rank += j - i;
    i = j;
  }

  return (positiveRankSum - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

async function readTrials(inDir) {
  const jsonlPath = path.join(repoPath(inDir), "trials-minimal.jsonl");
  const text = await readFile(jsonlPath, "utf8");
  return text
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function cellKey(trial) {
  return [
    trial.regime,
    trial.massRatio,
    trial.timestep,
    trial.radiusScale,
    trial.velocityScale,
  ].join("\t");
}

function cellObjectFromTrial(trial) {
  return {
    regime: trial.regime,
    massRatio: trial.massRatio,
    timestep: trial.timestep,
    radiusScale: trial.radiusScale,
    velocityScale: trial.velocityScale,
  };
}

function trialSamples(trial, channel) {
  return (trial.hazardSamples ?? [])
    .map((sample) => ({
      label: sample.label,
      score: channel.direction * Number(sample.channels?.[channel.name]),
      cellKey: cellKey(trial),
      velocityScale: trial.velocityScale,
    }))
    .filter((sample) => Number.isFinite(sample.score) && typeof sample.label === "boolean");
}

function summarizeChannel(trials, channel, predicate, scope) {
  const scopeTrials = trials.filter((trial) => trial.controllerMode === "off" && predicate(trial));
  const cells = new Map();
  for (const trial of scopeTrials) {
    if (!cells.has(cellKey(trial))) {
      cells.set(cellKey(trial), { ...cellObjectFromTrial(trial), samples: [] });
    }
    cells.get(cellKey(trial)).samples.push(...trialSamples(trial, channel));
  }

  const cellResults = [...cells.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, cell]) => {
      const positiveCount = cell.samples.filter((sample) => sample.label).length;
      const sampleCount = cell.samples.length;
      const auroc = computeAuroc(cell.samples);
      return {
        scope,
        scoreChannel: channel.name,
        direction: channel.direction > 0 ? "+" : "-",
        cellKey: key,
        regime: cell.regime,
        massRatio: cell.massRatio,
        timestep: cell.timestep,
        radiusScale: cell.radiusScale,
        velocityScale: cell.velocityScale,
        sampleCount,
        positiveCount,
        negativeCount: sampleCount - positiveCount,
        defined: auroc !== null,
        auroc,
      };
    });

  const definedAurocs = cellResults.map((row) => row.auroc).filter(Number.isFinite);
  const samples = [...cells.values()].flatMap((cell) => cell.samples);
  const sampleCount = samples.length;
  const positiveCount = samples.filter((sample) => sample.label).length;
  const sortedAurocs = definedAurocs.slice().sort((a, b) => a - b);
  const meanCellAuroc = definedAurocs.length
    ? definedAurocs.reduce((sum, value) => sum + value, 0) / definedAurocs.length
    : null;
  const medianCellAuroc = definedAurocs.length
    ? sortedAurocs[Math.floor((sortedAurocs.length - 1) / 2)]
    : null;

  return {
    summary: {
      scoreChannel: channel.name,
      direction: channel.direction > 0 ? "+" : "-",
      scope,
      trajectoryCount: scopeTrials.length,
      sampleCount,
      positiveCount,
      negativeCount: sampleCount - positiveCount,
      definedCells: definedAurocs.length,
      totalCells: cells.size,
      meanCellAuroc: round(meanCellAuroc),
      medianCellAuroc: round(medianCellAuroc),
      minCellAuroc: round(sortedAurocs[0]),
      maxCellAuroc: round(sortedAurocs[sortedAurocs.length - 1]),
      pooledAuroc: round(computeAuroc(samples)),
    },
    cellRows: cellResults.map((row) => ({ ...row, auroc: round(row.auroc) })),
  };
}

function readPhase16Rows(inDir) {
  return readFile(path.join(repoPath(inDir), "hazard-channel-audit.csv"), "utf8")
    .then(parseCsv);
}

function verifyPhase16Rows(rows) {
  const byChannel = new Map(rows.map((row) => [row.channel, row]));
  for (const [channel, expected] of Object.entries(EXPECTED_PHASE16)) {
    const row = byChannel.get(channel);
    if (!row) throw new Error(`Missing Phase 16 ${channel} row`);
    const mean = Number(row.perCellMeanAuroc);
    const defined = Number(row.perCellDefinedCount);
    if (round(mean) !== expected.perCellMeanAuroc || defined !== expected.perCellDefinedCount) {
      throw new Error(
        `Phase 16 ${channel} receipt drift: got mean=${mean}, defined=${defined}; expected mean=${expected.perCellMeanAuroc}, defined=${expected.perCellDefinedCount}`,
      );
    }
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const inDir = repoPath(args.inDir);
  const outDir = repoPath(args.outDir);
  const phase16Manifest = JSON.parse(await readFile(path.join(inDir, "hazard-channel-audit-manifest.json"), "utf8"));
  if (phase16Manifest.branch !== EXPECTED_PHASE16_BRANCH) {
    throw new Error(`Expected Phase 16 branch ${EXPECTED_PHASE16_BRANCH}; got ${phase16Manifest.branch}`);
  }
  const phase16Rows = await readPhase16Rows(args.inDir);
  verifyPhase16Rows(phase16Rows);

  const trials = await readTrials(args.inDir);
  if (trials.length !== EXPECTED_TRIALS) {
    throw new Error(`Expected ${EXPECTED_TRIALS} passive trials; got ${trials.length}`);
  }
  if (trials.some((trial) => trial.controllerMode !== "off")) {
    throw new Error("Phase 16B source receipt contains non-passive trials");
  }

  const scopes = [
    { name: "favorable", predicate: (trial) => Number(trial.velocityScale) >= FAVORABLE_VELOCITY_MIN },
    { name: "full_grid", predicate: () => true },
    { name: "v_0p95_control", predicate: (trial) => Number(trial.velocityScale) === 0.95 },
    { name: "v_1p05", predicate: (trial) => Number(trial.velocityScale) === 1.05 },
    { name: "v_1p1", predicate: (trial) => Number(trial.velocityScale) === 1.1 },
    { name: "v_1p15", predicate: (trial) => Number(trial.velocityScale) === 1.15 },
  ];

  const summaries = [];
  const qualityRows = [];
  for (const scope of scopes) {
    for (const channel of CHANNELS) {
      const result = summarizeChannel(trials, channel, scope.predicate, scope.name);
      summaries.push(result.summary);
      qualityRows.push(...result.cellRows);
    }
  }

  const favorableRadius = summaries.find((row) => row.scope === "favorable" && row.scoreChannel === "radius");
  const favorableEnergy = summaries.find((row) => row.scope === "favorable" && row.scoreChannel === "energy");
  if (favorableRadius.meanCellAuroc !== EXPECTED_PHASE16.radius.perCellMeanAuroc
    || favorableRadius.definedCells !== EXPECTED_PHASE16.radius.perCellDefinedCount
    || favorableEnergy.meanCellAuroc !== EXPECTED_PHASE16.energy.perCellMeanAuroc
    || favorableEnergy.definedCells !== EXPECTED_PHASE16.energy.perCellDefinedCount) {
    throw new Error("Phase 16B reducer failed to reproduce Phase 16 per-cell values");
  }

  const branch = favorableRadius.definedCells >= COVERAGE_BAR && favorableRadius.meanCellAuroc >= PASS_BAR
    ? "A_warning_verdict_flips_under_radius"
    : favorableRadius.definedCells >= COVERAGE_BAR
      ? "B_warning_verdict_does_not_flip"
      : "C_mixed_provisional";

  const summaryRows = summaries.map((row) => ({
    ...row,
    passBar: row.scope === "favorable" && row.scoreChannel === "radius" ? PASS_BAR : "",
    coverageBar: row.scope === "favorable" && row.scoreChannel === "radius" ? COVERAGE_BAR : "",
    branch: row.scope === "favorable" && row.scoreChannel === "radius" ? branch : "",
  }));

  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "radius-warning-quality-map.csv"), rowsToCsv(qualityRows), "utf8");
  await writeFile(path.join(outDir, "radius-warning-summary.csv"), rowsToCsv(summaryRows), "utf8");
  const manifest = {
    schema: "sundog.threebody.phase16b-radius-warning-repose.v1",
    generatedAt: new Date().toISOString(),
    source: path.relative(repoRoot, inDir),
    phase16Branch: phase16Manifest.branch,
    sourceTrialCount: trials.length,
    passBar: PASS_BAR,
    coverageBar: COVERAGE_BAR,
    expectedPhase16: EXPECTED_PHASE16,
    reproduced: {
      radius: {
        meanCellAuroc: favorableRadius.meanCellAuroc,
        definedCells: favorableRadius.definedCells,
        totalCells: favorableRadius.totalCells,
      },
      energy: {
        meanCellAuroc: favorableEnergy.meanCellAuroc,
        definedCells: favorableEnergy.definedCells,
        totalCells: favorableEnergy.totalCells,
      },
    },
    branch,
  };
  await writeFile(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  console.log(`[threebody] wrote Phase 16B radius warning re-pose to ${path.relative(repoRoot, outDir)}`);
  console.log(`[threebody] branch ${branch}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
