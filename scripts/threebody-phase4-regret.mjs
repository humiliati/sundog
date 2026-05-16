import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { makeRng, roundNumber } from "../public/js/threebody-core.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const args = {
    bayesIn: "results/proof/phase4/bayes-floor-smoke",
    signatureIn: null,
    out: "results/proof/phase4",
    signatureMode: "track_sensor_accel_guarded",
    bootstrapIterations: 2000,
    bootstrapSeed: 40604,
    fiberMinSamples: 20,
    tidalLogBinWidth: 0.25,
    gradientMagnitudeLogBinWidth: 0.25,
    gradientAngleSectors: 8,
    tMax: null,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    i += 1;

    if (flag === "--bayes-in") args.bayesIn = value;
    else if (flag === "--signature-in") args.signatureIn = value;
    else if (flag === "--out") args.out = value;
    else if (flag === "--signature-mode") args.signatureMode = value;
    else if (flag === "--bootstrap-iterations") args.bootstrapIterations = Number.parseInt(value, 10);
    else if (flag === "--bootstrap-seed") args.bootstrapSeed = Number.parseInt(value, 10);
    else if (flag === "--fiber-min-samples") args.fiberMinSamples = Number.parseInt(value, 10);
    else if (flag === "--tidal-log-bin-width") args.tidalLogBinWidth = Number.parseFloat(value);
    else if (flag === "--gradient-magnitude-log-bin-width") args.gradientMagnitudeLogBinWidth = Number.parseFloat(value);
    else if (flag === "--gradient-angle-sectors") args.gradientAngleSectors = Number.parseInt(value, 10);
    else if (flag === "--t-max") args.tMax = Number.parseFloat(value);
    else if (flag === "--self-test") args.selfTest = true;
    else throw new Error(`Unknown flag: ${flag}`);
  }

  if (args.selfTest) return args;
  if (!args.signatureIn) throw new Error("--signature-in is required");
  if (!Number.isInteger(args.bootstrapIterations) || args.bootstrapIterations < 0) {
    throw new Error("--bootstrap-iterations must be a non-negative integer");
  }
  if (!Number.isInteger(args.fiberMinSamples) || args.fiberMinSamples < 1) {
    throw new Error("--fiber-min-samples must be a positive integer");
  }
  if (!Number.isFinite(args.tidalLogBinWidth) || args.tidalLogBinWidth <= 0) {
    throw new Error("--tidal-log-bin-width must be positive");
  }
  if (!Number.isFinite(args.gradientMagnitudeLogBinWidth) || args.gradientMagnitudeLogBinWidth <= 0) {
    throw new Error("--gradient-magnitude-log-bin-width must be positive");
  }
  if (!Number.isInteger(args.gradientAngleSectors) || args.gradientAngleSectors < 1) {
    throw new Error("--gradient-angle-sectors must be positive");
  }
  return args;
}

function csvValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function rowsToCsv(rows, explicitColumns = null) {
  const columns = explicitColumns ?? [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const lines = [columns.join(",")];
  for (const row of rows) lines.push(columns.map((column) => csvValue(row[column])).join(","));
  return `${lines.join("\n")}\n`;
}

function parseCsvLine(line) {
  const values = [];
  let value = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (quoted) {
      if (char === '"' && line[i + 1] === '"') {
        value += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        value += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      values.push(value);
      value = "";
    } else {
      value += char;
    }
  }
  values.push(value);
  return values;
}

async function readCsv(filePath) {
  const text = await readFile(filePath, "utf8");
  const lines = text.trim().split(/\r?\n/);
  if (lines.length === 0 || !lines[0]) return [];
  const columns = parseCsvLine(lines[0]);
  return lines.slice(1).filter(Boolean).map((line) => {
    const values = parseCsvLine(line);
    const row = {};
    for (let i = 0; i < columns.length; i += 1) row[columns[i]] = values[i] ?? "";
    return row;
  });
}

function asNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function inputFile(inputPath, filename) {
  if (inputPath.endsWith(".csv") || inputPath.endsWith(".jsonl") || inputPath.endsWith(".json")) {
    return path.resolve(repoRoot, inputPath);
  }
  return path.resolve(repoRoot, inputPath, filename);
}

function joinKey(row) {
  return [
    row.case_id,
    row.regime,
    row.seed,
  ].join("\t");
}

function cellKey(row) {
  return [
    row.case_id,
    row.regime,
  ].join("\t");
}

function mean(values) {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return null;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function quantile(values, q) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (sorted.length === 0) return null;
  const index = (sorted.length - 1) * q;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function bootstrapMeanCi(values, iterations, seed) {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return { mean: null, ciLower: null, ciUpper: null };
  if (iterations <= 0 || finite.length === 1) {
    const point = mean(finite);
    return { mean: point, ciLower: point, ciUpper: point };
  }
  const rng = makeRng(seed);
  const draws = [];
  for (let i = 0; i < iterations; i += 1) {
    let sum = 0;
    for (let j = 0; j < finite.length; j += 1) {
      sum += finite[Math.floor(rng() * finite.length)];
    }
    draws.push(sum / finite.length);
  }
  return {
    mean: mean(finite),
    ciLower: quantile(draws, 0.025),
    ciUpper: quantile(draws, 0.975),
  };
}

function logBin(value, width) {
  const safe = Math.max(Math.abs(value), 1e-12);
  const lower = Math.floor(Math.log10(safe) / width) * width;
  const upper = lower + width;
  return `[${roundNumber(lower, 6)},${roundNumber(upper, 6)})`;
}

function angleSector(gradX, gradY, sectors) {
  const magnitude = Math.sqrt(gradX * gradX + gradY * gradY);
  if (magnitude <= 1e-12) return "zero";
  const angle = (Math.atan2(gradY, gradX) + 2 * Math.PI) % (2 * Math.PI);
  return String(Math.min(sectors - 1, Math.floor(angle / (2 * Math.PI / sectors))));
}

function fiberKey(row, args) {
  const gradX = asNumber(row.grad_x) ?? 0;
  const gradY = asNumber(row.grad_y) ?? 0;
  const gradMag = asNumber(row.gradient_magnitude) ?? Math.sqrt(gradX * gradX + gradY * gradY);
  return [
    `guard=${row.guard}`,
    `tidal=${logBin(asNumber(row.tidal_magnitude) ?? 0, args.tidalLogBinWidth)}`,
    `angle=${angleSector(gradX, gradY, args.gradientAngleSectors)}`,
    `grad=${logBin(gradMag, args.gradientMagnitudeLogBinWidth)}`,
    `noise=${row.sensor_noise_std}`,
  ].join("|");
}

async function readJsonl(filePath) {
  const text = await readFile(filePath, "utf8");
  return text.trim().split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

async function classifyCells(args, bayesDir) {
  const observationsPath = inputFile(bayesDir, "signature-observations.jsonl");
  const observations = await readJsonl(observationsPath);
  const fibersByCell = new Map();
  for (const row of observations) {
    const key = cellKey(row);
    const fiber = fiberKey(row, args);
    if (!fibersByCell.has(key)) fibersByCell.set(key, new Map());
    const fibers = fibersByCell.get(key);
    if (!fibers.has(fiber)) fibers.set(fiber, []);
    fibers.get(fiber).push(row);
  }

  const cells = {};
  for (const [key, fibers] of fibersByCell.entries()) {
    const decidableFibers = [];
    for (const [fiber, rows] of fibers.entries()) {
      if (rows.length < args.fiberMinSamples) continue;
      const actionSet = [...new Set(rows.map((row) => row.action_key))].sort();
      decidableFibers.push({
        fiber,
        sampleCount: rows.length,
        actionSet,
        commonAction: actionSet.length === 1 ? actionSet[0] : null,
        status: actionSet.length === 1 ? "common_action" : "conflict",
      });
    }
    const cellClass = decidableFibers.length === 0
      ? "undecidable"
      : decidableFibers.some((fiber) => fiber.status === "conflict")
        ? "off"
        : "on";
    cells[key] = {
      cellKey: key,
      cellClass,
      decidableFiberCount: decidableFibers.length,
      fiberCount: fibers.size,
      fibers: decidableFibers,
    };
  }

  return {
    schema: "sundog.threebody.phase4.cell_fibers.v1",
    classifier: {
      partitionKeys: [
        "guard_t",
        "log_binned_abs_tidal_magnitude",
        "gradient_angle_sector",
        "log_binned_gradient_magnitude",
        "sensor_noise_std",
      ],
      fiberMinSamples: args.fiberMinSamples,
      tidalLogBinWidth: args.tidalLogBinWidth,
      gradientMagnitudeLogBinWidth: args.gradientMagnitudeLogBinWidth,
      gradientAngleSectors: args.gradientAngleSectors,
      commonActionRule: "exact action_key match; zero action is represented by its exact key",
      undecidableRule: "no fiber with at least fiberMinSamples Bayes-reached samples",
    },
    cells,
  };
}

async function inferTMax(args, bayesDir) {
  if (Number.isFinite(args.tMax) && args.tMax > 0) return args.tMax;
  const manifestPath = inputFile(bayesDir, "manifest.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  const duration = Number.parseFloat(manifest.args?.duration);
  if (!Number.isFinite(duration) || duration <= 0) {
    throw new Error("--t-max is required when Bayes manifest args.duration is unavailable");
  }
  return duration;
}

async function reduceRegret(args) {
  const bayesDir = args.bayesIn;
  const signatureRowsRaw = await readCsv(inputFile(args.signatureIn, "trial-outcomes.csv"));
  const bayesRowsRaw = await readCsv(inputFile(bayesDir, "bayes-trial-outcomes.csv"));
  const tMax = await inferTMax(args, bayesDir);
  const cellFibers = await classifyCells(args, bayesDir);
  const signatureByKey = new Map(
    signatureRowsRaw
      .filter((row) => !args.signatureMode || row.controller_mode === args.signatureMode)
      .map((row) => [joinKey(row), row]),
  );
  const regretRows = [];

  for (const bayes of bayesRowsRaw) {
    const signature = signatureByKey.get(joinKey(bayes));
    if (!signature) continue;
    const bayesSafe = asNumber(bayes.simulated_time);
    const signatureSafe = asNumber(signature.simulated_time);
    const bayesDeltaV = asNumber(bayes.total_delta_v);
    const signatureDeltaV = asNumber(signature.total_delta_v);
    const classEntry = cellFibers.cells[cellKey(bayes)] ?? { cellClass: "undecidable" };
    const regret = bayesSafe !== null && signatureSafe !== null ? (bayesSafe - signatureSafe) / tMax : null;
    regretRows.push({
      case_id: bayes.case_id,
      seed: bayes.seed,
      regime: bayes.regime,
      mass_ratio: bayes.mass_ratio,
      timestep: bayes.timestep,
      radius_scale: bayes.radius_scale,
      velocity_scale: bayes.velocity_scale,
      thrust_limit: bayes.thrust_limit,
      sensor_noise_std: bayes.sensor_noise_std,
      cell_class: classEntry.cellClass,
      signature_controller_mode: signature.controller_mode,
      bayes_controller_mode: bayes.controller_mode,
      t_max: tMax,
      T_safe_signature: signatureSafe,
      T_safe_bayes: bayesSafe,
      regret: regret === null ? null : roundNumber(regret, 8),
      total_delta_v_signature: signatureDeltaV,
      total_delta_v_bayes: bayesDeltaV,
      fuel_excess: signatureDeltaV !== null && bayesDeltaV !== null
        ? roundNumber(signatureDeltaV - bayesDeltaV, 8)
        : null,
    });
  }

  const classes = ["on", "off", "undecidable"];
  const negativeRegretCount = regretRows.filter((row) => asNumber(row.regret) < -1e-9).length;
  const negativeRegretRate = regretRows.length > 0 ? negativeRegretCount / regretRows.length : null;
  const floorStatus = negativeRegretRate !== null && negativeRegretRate > 0.05
    ? "non_decisive_floor_repair_required"
    : "floor_sanity_pass";
  const summaryRows = classes.map((cellClass, index) => {
    const group = regretRows.filter((row) => row.cell_class === cellClass);
    const ci = bootstrapMeanCi(
      group.map((row) => asNumber(row.regret)),
      args.bootstrapIterations,
      args.bootstrapSeed + index * 1009,
    );
    return {
      cell_class: cellClass,
      row_count: group.length,
      mean_regret: ci.mean === null ? null : roundNumber(ci.mean, 8),
      ci_lower_95: ci.ciLower === null ? null : roundNumber(ci.ciLower, 8),
      ci_upper_95: ci.ciUpper === null ? null : roundNumber(ci.ciUpper, 8),
      negative_regret_count: group.filter((row) => asNumber(row.regret) < -1e-9).length,
      negative_regret_rate: group.length > 0
        ? roundNumber(group.filter((row) => asNumber(row.regret) < -1e-9).length / group.length, 8)
        : null,
      global_negative_regret_rate: negativeRegretRate === null ? null : roundNumber(negativeRegretRate, 8),
      floor_status: floorStatus,
    };
  });

  return { regretRows, summaryRows, cellFibers };
}

async function runSelfTest() {
  const tmp = await mkdtemp(path.join(os.tmpdir(), "threebody-regret-self-test-"));
  const bayesDir = path.join(tmp, "bayes");
  const sigDir = path.join(tmp, "sig");
  const outDir = path.join(tmp, "out");
  await mkdir(bayesDir, { recursive: true });
  await mkdir(sigDir, { recursive: true });
  await writeFile(path.join(bayesDir, "manifest.json"), `${JSON.stringify({ args: { duration: 10 } })}\n`, "utf8");
  await writeFile(path.join(bayesDir, "bayes-trial-outcomes.csv"), rowsToCsv([
    {
      case_id: "cell_a",
      seed: 0,
      regime: "near_escape",
      mass_ratio: 1,
      timestep: 0.01,
      radius_scale: 1,
      velocity_scale: 1,
      thrust_limit: 0.4,
      sensor_noise_std: 0,
      controller_mode: "bayes_floor_particle_mpc",
      simulated_time: 10,
      total_delta_v: 0.1,
    },
    {
      case_id: "cell_b",
      seed: 1,
      regime: "near_escape",
      mass_ratio: 1,
      timestep: 0.01,
      radius_scale: 1,
      velocity_scale: 1,
      thrust_limit: 0.4,
      sensor_noise_std: 0,
      controller_mode: "bayes_floor_particle_mpc",
      simulated_time: 8,
      total_delta_v: 0.1,
    },
  ]), "utf8");
  await writeFile(path.join(sigDir, "trial-outcomes.csv"), rowsToCsv([
    {
      case_id: "cell_a",
      seed: 0,
      regime: "near_escape",
      controller_mode: "track_sensor_accel_guarded",
      simulated_time: 9,
      total_delta_v: 0.2,
    },
    {
      case_id: "cell_b",
      seed: 1,
      regime: "near_escape",
      controller_mode: "track_sensor_accel_guarded",
      simulated_time: 9,
      total_delta_v: 0.2,
    },
  ]), "utf8");
  const observations = [];
  for (let i = 0; i < 20; i += 1) {
    observations.push({
      case_id: "cell_a",
      seed: 0,
      regime: "near_escape",
      guard: true,
      tidal_magnitude: 2,
      grad_x: 1,
      grad_y: 0,
      gradient_magnitude: 1,
      sensor_noise_std: 0,
      action_key: "0:0",
    });
    observations.push({
      case_id: "cell_b",
      seed: 1,
      regime: "near_escape",
      guard: true,
      tidal_magnitude: 2,
      grad_x: 1,
      grad_y: 0,
      gradient_magnitude: 1,
      sensor_noise_std: 0,
      action_key: i < 10 ? "0:0" : "0.4:0",
    });
  }
  await writeFile(
    path.join(bayesDir, "signature-observations.jsonl"),
    `${observations.map((row) => JSON.stringify(row)).join("\n")}\n`,
    "utf8",
  );

  const result = await reduceRegret({
    ...parseArgs([
      "--bayes-in", bayesDir,
      "--signature-in", sigDir,
      "--out", outDir,
      "--bootstrap-iterations", "20",
    ]),
  });
  const onRow = result.summaryRows.find((row) => row.cell_class === "on");
  const offRow = result.summaryRows.find((row) => row.cell_class === "off");
  if (onRow.row_count !== 1) throw new Error("self-test expected one on row");
  if (offRow.row_count !== 1) throw new Error("self-test expected one off row");
  if (!result.summaryRows.every((row) => row.floor_status === "non_decisive_floor_repair_required")) {
    throw new Error("self-test expected negative-regret sanity failure");
  }
  await rm(tmp, { recursive: true, force: true });
  console.log("[threebody-phase4-regret] self-test passed");
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.selfTest) {
    await runSelfTest();
    return;
  }
  const outDir = path.resolve(repoRoot, args.out);
  await mkdir(outDir, { recursive: true });
  const { regretRows, summaryRows, cellFibers } = await reduceRegret(args);
  await writeFile(path.join(outDir, "phase4-regret.csv"), rowsToCsv(regretRows), "utf8");
  await writeFile(path.join(outDir, "phase4-regret-summary.csv"), rowsToCsv(summaryRows), "utf8");
  await writeFile(path.join(outDir, "cell-fibers.json"), `${JSON.stringify(cellFibers, null, 2)}\n`, "utf8");
  console.log(`[threebody-phase4-regret] wrote ${regretRows.length} joined rows to ${path.relative(repoRoot, outDir)}`);
  console.log("[threebody-phase4-regret] wrote phase4-regret.csv, phase4-regret-summary.csv, cell-fibers.json");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
