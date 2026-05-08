import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  normalizeConfig,
  runTrial,
  seededInitialParticle,
} from "../public/js/threebody-core.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const MODE_DEFINITIONS = Object.freeze({
  off: "Passive baseline: no thrust.",
  naive: "Naive local baseline: local acceleration opposition plus velocity damping; no tidal-gradient structure.",
  scan: "SCAN ablation: periodic thrust perturbation only.",
  seek: "Sundog SEEK prototype: descends local tidal-magnitude gradient.",
  track: "Sundog TRACK prototype: follows local tidal-gradient direction to maintain target tidal magnitude.",
  seek_noisy: "Tidal-signal ablation: SEEK with noisy tidal magnitude and gradient observations.",
  track_noisy: "Tidal-signal ablation: TRACK with noisy tidal magnitude and gradient observations.",
  seek_shuffled: "Tidal-signal ablation: SEEK with deterministic shuffled tidal-gradient direction.",
  track_shuffled: "Tidal-signal ablation: TRACK with deterministic shuffled tidal-gradient direction.",
  oracle: "Privileged full-state lookahead guard: scores candidate thrust vectors using simulator state; heuristic, not optimal.",
});

const PRIMARY_METRICS = Object.freeze([
  "terminalOutcome",
  "simulatedTime",
  "totalDeltaV",
  "minPrimaryDistance",
  "saturationCount",
  "targetBandLossCount",
]);

function parseList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function parseArgs(argv) {
  const args = {
    phase: "phase5",
    out: "results/threebody/phase5-smoke",
    seedStart: 0,
    seeds: 3,
    regimes: ["stable", "near_escape", "near_collision", "chaotic"],
    modes: ["off"],
    duration: 8,
    dt: 0.01,
    logEvery: 20,
    massRatio: 1,
    thrustLimit: 0.5,
    targetTidal: 2,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    i += 1;

    if (flag === "--phase") args.phase = value;
    else if (flag === "--out") args.out = value;
    else if (flag === "--seed-start") args.seedStart = Number.parseInt(value, 10);
    else if (flag === "--seeds") args.seeds = Number.parseInt(value, 10);
    else if (flag === "--regimes") args.regimes = parseList(value);
    else if (flag === "--modes") args.modes = parseList(value);
    else if (flag === "--duration") args.duration = Number.parseFloat(value);
    else if (flag === "--dt") args.dt = Number.parseFloat(value);
    else if (flag === "--log-every") args.logEvery = Number.parseInt(value, 10);
    else if (flag === "--mass-ratio") args.massRatio = Number.parseFloat(value);
    else if (flag === "--thrust-limit") args.thrustLimit = Number.parseFloat(value);
    else if (flag === "--target-tidal") args.targetTidal = Number.parseFloat(value);
    else throw new Error(`Unknown flag: ${flag}`);
  }

  if (!Number.isInteger(args.seeds) || args.seeds < 1) {
    throw new Error("--seeds must be a positive integer");
  }
  if (!Number.isFinite(args.duration) || args.duration <= 0) {
    throw new Error("--duration must be positive");
  }
  if (!Number.isFinite(args.dt) || args.dt <= 0) {
    throw new Error("--dt must be positive");
  }
  return args;
}

function trialId({ regime, mode, seed }) {
  return `${regime}_${mode}_seed_${String(seed).padStart(3, "0")}`;
}

function mean(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) return null;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function csvValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function makeSummaryRows(trials) {
  const groups = new Map();
  for (const trial of trials) {
    const key = `${trial.regime}\t${trial.controllerMode}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(trial);
  }

  return Array.from(groups.entries()).map(([key, group]) => {
    const [regime, controllerMode] = key.split("\t");
    const countOutcome = (outcome) => group.filter((trial) => trial.summary.terminalOutcome === outcome).length;
    return {
      regime,
      controllerMode,
      n: group.length,
      bounded: countOutcome("bounded"),
      closeApproach: countOutcome("close_approach"),
      escape: countOutcome("escape"),
      invalid: countOutcome("invalid"),
      meanSimulatedTime: mean(group.map((trial) => trial.summary.simulatedTime)),
      meanTotalDeltaV: mean(group.map((trial) => trial.summary.totalDeltaV)),
      meanSaturationCount: mean(group.map((trial) => trial.summary.saturationCount)),
      meanTargetBandLossCount: mean(group.map((trial) => trial.summary.targetBandLossCount)),
      meanMinPrimaryDistance: mean(group.map((trial) => trial.summary.minPrimaryDistance)),
      meanMaxRadius: mean(group.map((trial) => trial.summary.maxRadius)),
    };
  }).sort((a, b) => (
    a.regime.localeCompare(b.regime) || a.controllerMode.localeCompare(b.controllerMode)
  ));
}

function rowsToCsv(rows, explicitColumns = null) {
  const columns = explicitColumns ?? [
    ...new Set(rows.flatMap((row) => Object.keys(row))),
  ];
  const lines = [columns.join(",")];
  for (const row of rows) {
    lines.push(columns.map((column) => csvValue(row[column])).join(","));
  }
  return `${lines.join("\n")}\n`;
}

function compareOutcome(outcome, baselineOutcome) {
  const rank = {
    invalid: 0,
    close_approach: 1,
    escape: 2,
    bounded: 3,
  };
  return (rank[outcome] ?? 0) - (rank[baselineOutcome] ?? 0);
}

function makePairedRows(trials) {
  const byKey = new Map();
  for (const trial of trials) {
    byKey.set(`${trial.regime}\t${trial.seed}\t${trial.controllerMode}`, trial);
  }

  return trials.map((trial) => {
    const passive = byKey.get(`${trial.regime}\t${trial.seed}\toff`);
    const oracle = byKey.get(`${trial.regime}\t${trial.seed}\toracle`);
    const passiveSummary = passive?.summary;
    const oracleSummary = oracle?.summary;
    return {
      regime: trial.regime,
      seed: trial.seed,
      controllerMode: trial.controllerMode,
      terminalOutcome: trial.summary.terminalOutcome,
      passiveOutcome: passiveSummary?.terminalOutcome,
      oracleOutcome: oracleSummary?.terminalOutcome,
      outcomeDeltaVsPassive: passiveSummary ? compareOutcome(trial.summary.terminalOutcome, passiveSummary.terminalOutcome) : null,
      outcomeDeltaVsOracle: oracleSummary ? compareOutcome(trial.summary.terminalOutcome, oracleSummary.terminalOutcome) : null,
      simulatedTime: trial.summary.simulatedTime,
      passiveSimulatedTime: passiveSummary?.simulatedTime,
      oracleSimulatedTime: oracleSummary?.simulatedTime,
      simulatedTimeDeltaVsPassive: passiveSummary ? trial.summary.simulatedTime - passiveSummary.simulatedTime : null,
      simulatedTimeDeltaVsOracle: oracleSummary ? trial.summary.simulatedTime - oracleSummary.simulatedTime : null,
      totalDeltaV: trial.summary.totalDeltaV,
      passiveTotalDeltaV: passiveSummary?.totalDeltaV,
      oracleTotalDeltaV: oracleSummary?.totalDeltaV,
      minPrimaryDistance: trial.summary.minPrimaryDistance,
      passiveMinPrimaryDistance: passiveSummary?.minPrimaryDistance,
      oracleMinPrimaryDistance: oracleSummary?.minPrimaryDistance,
      log: trial.log,
    };
  }).sort((a, b) => (
    a.regime.localeCompare(b.regime)
    || a.seed - b.seed
    || a.controllerMode.localeCompare(b.controllerMode)
  ));
}

function makeComparisonRows(pairedRows) {
  const groups = new Map();
  for (const row of pairedRows) {
    const key = `${row.regime}\t${row.controllerMode}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }

  return Array.from(groups.entries()).map(([key, group]) => {
    const [regime, controllerMode] = key.split("\t");
    const nonPassive = group.filter((row) => row.controllerMode !== "off");
    const rows = nonPassive.length > 0 ? nonPassive : group;
    return {
      regime,
      controllerMode,
      n: rows.length,
      improvedOutcomeVsPassive: rows.filter((row) => row.outcomeDeltaVsPassive > 0).length,
      worsenedOutcomeVsPassive: rows.filter((row) => row.outcomeDeltaVsPassive < 0).length,
      tiedOutcomeVsPassive: rows.filter((row) => row.outcomeDeltaVsPassive === 0).length,
      meanTimeDeltaVsPassive: mean(rows.map((row) => row.simulatedTimeDeltaVsPassive)),
      meanTimeDeltaVsOracle: mean(rows.map((row) => row.simulatedTimeDeltaVsOracle)),
      meanDeltaV: mean(rows.map((row) => row.totalDeltaV)),
      meanMinPrimaryDistance: mean(rows.map((row) => row.minPrimaryDistance)),
    };
  }).sort((a, b) => (
    a.regime.localeCompare(b.regime) || a.controllerMode.localeCompare(b.controllerMode)
  ));
}

function makeTrialConfig(args, seed, regime, mode) {
  return normalizeConfig({
    seed,
    regime,
    controllerMode: mode,
    duration: args.duration,
    dt: args.dt,
    logEvery: args.logEvery,
    massRatio: args.massRatio,
    thrustLimit: args.thrustLimit,
    targetTidal: args.targetTidal,
    initialParticle: seededInitialParticle(seed, regime),
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(repoRoot, args.out);
  const trialDir = path.join(outDir, "trials");
  await mkdir(trialDir, { recursive: true });

  const startedAt = new Date().toISOString();
  const manifest = {
    schema: `sundog.threebody.${args.phase}.v1`,
    startedAt,
    purpose: `${args.phase.toUpperCase()} deterministic harness run; not a benchmark claim unless separately analyzed.`,
    harness: {
      script: "scripts/threebody-harness.mjs",
      core: "public/js/threebody-core.mjs",
    },
    modeDefinitions: MODE_DEFINITIONS,
    primaryMetrics: PRIMARY_METRICS,
    args,
    trials: [],
  };

  for (const regime of args.regimes) {
    for (const mode of args.modes) {
      for (let offset = 0; offset < args.seeds; offset += 1) {
        const seed = args.seedStart + offset;
        const config = makeTrialConfig(args, seed, regime, mode);
        const id = trialId({ regime, mode, seed });
        const trial = runTrial(config);
        const relativePath = `trials/${id}.jsonl`;
        const logPath = path.join(outDir, relativePath);
        const lines = trial.records.map((record) => JSON.stringify(record)).join("\n");
        await writeFile(logPath, `${lines}\n`, "utf8");
        manifest.trials.push({
          id,
          seed,
          regime,
          controllerMode: mode,
          initialParticle: config.initialParticle,
          config: {
            massRatio: config.massRatio,
            masses: config.masses,
            separation: config.separation,
            dt: config.dt,
            duration: config.duration,
            logEvery: config.logEvery,
            thrustLimit: config.thrustLimit,
            targetTidal: config.targetTidal,
          },
          log: relativePath,
          summary: trial.summary,
        });
      }
    }
  }

  manifest.completedAt = new Date().toISOString();
  const summaryRows = makeSummaryRows(manifest.trials);
  const pairedRows = makePairedRows(manifest.trials);
  const comparisonRows = makeComparisonRows(pairedRows);
  await writeFile(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  await writeFile(path.join(outDir, "summary.csv"), rowsToCsv(summaryRows), "utf8");
  await writeFile(path.join(outDir, "paired.csv"), rowsToCsv(pairedRows), "utf8");
  await writeFile(path.join(outDir, "comparison.csv"), rowsToCsv(comparisonRows), "utf8");

  const outcomeCounts = manifest.trials.reduce((counts, trial) => {
    counts[trial.summary.terminalOutcome] = (counts[trial.summary.terminalOutcome] ?? 0) + 1;
    return counts;
  }, {});

  console.log(`[threebody] wrote ${manifest.trials.length} trials to ${path.relative(repoRoot, outDir)}`);
  console.log(`[threebody] wrote summary.csv with ${summaryRows.length} condition rows`);
  console.log(`[threebody] wrote paired.csv and comparison.csv`);
  console.log(`[threebody] outcomes ${JSON.stringify(outcomeCounts)}`);
}

main().catch((error) => {
  console.error(`[threebody] ${error.message}`);
  process.exitCode = 1;
});
