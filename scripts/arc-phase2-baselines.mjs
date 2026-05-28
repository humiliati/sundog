// Phase 2 baseline runner.
//
// Computes the same per-task collision-residual and per-train-pair residual
// metrics used by scripts/arc-phase2-projections.mjs, but under three naive
// feature representations, so the shadow operator's numbers have a
// calibrated comparison set:
//
//   - raw_pixel_hash: feature = SHA256 of grid JSON. Pair distance =
//     normalized padded Hamming. Floor for "no projection at all."
//   - shape_palette_density: feature = (h, w, palette_size, density_bucket).
//     Pair distance = mean of three normalized L1 components.
//   - cell_count: feature = nonzero_cell_count. Pair distance = |a - b|
//     normalized by max count in the subset.
//
// Output: results/arc/phase2-baselines/{manifest.json,summary.csv}.
// Also emits a reference row for shadow_operator_v0 by reading the prior
// phase2-projections manifest, so the comparison sits in one CSV.
//
// Discipline: training split only, no answer scoring, no held-split access.
// Leak-checked.

import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";

const args = parseArgs(process.argv.slice(2));
if (args.help || !args.dataDir || !args.register || !args.out) {
  printUsage();
  process.exit(args.help ? 0 : 2);
}

const dataDir = resolve(args.dataDir);
const registerPath = resolve(args.register);
const outDir = resolve(args.out);
const shadowManifestPath = args.shadowManifest
  ? resolve(args.shadowManifest)
  : resolve("results/arc/phase2-projections/manifest.json");

const registerRows = parseCsv(await readFile(registerPath, "utf8"))
  .filter((row) => row.status === "include" && row.split === "training");

if (registerRows.length === 0) {
  fail("Register has no included training rows.");
}

const allGrids = [];
const trainPairs = [];
for (const row of registerRows) {
  const raw = await readFile(join(dataDir, "training", `${row.task_id}.json`), "utf8");
  const task = JSON.parse(raw.replace(/^﻿/, ""));
  for (let i = 0; i < task.train.length; i += 1) {
    allGrids.push({ task_id: row.task_id, role: "train_input", pair_index: i, grid: task.train[i].input });
    allGrids.push({ task_id: row.task_id, role: "train_output", pair_index: i, grid: task.train[i].output });
    trainPairs.push({ task_id: row.task_id, pair_index: i, input: task.train[i].input, output: task.train[i].output });
  }
  for (let i = 0; i < task.test.length; i += 1) {
    allGrids.push({ task_id: row.task_id, role: "test_input", pair_index: i, grid: task.test[i].input });
  }
}

const maxH = Math.max(...allGrids.map((g) => g.grid.length));
const maxW = Math.max(...allGrids.map((g) => g.grid[0].length));
const maxCells = Math.max(...allGrids.map((g) => nonZeroCount(g.grid)));

const baselines = [
  {
    name: "raw_pixel_hash",
    feature: (grid) => sha256(JSON.stringify(grid)),
    pairDistance: paddedHamming
  },
  {
    name: "shape_palette_density",
    feature: (grid) => {
      const h = grid.length;
      const w = grid[0].length;
      const pal = new Set(grid.flat()).size;
      const dens = nonZeroCount(grid) / (h * w);
      return `${h}x${w}|p${pal}|d${Math.floor(dens * 10)}`;
    },
    pairDistance: (a, b) => {
      const ha = a.length;
      const wa = a[0].length;
      const hb = b.length;
      const wb = b[0].length;
      const pa = new Set(a.flat()).size;
      const pb = new Set(b.flat()).size;
      const da = nonZeroCount(a) / (ha * wa);
      const db = nonZeroCount(b) / (hb * wb);
      const shapeTerm = (Math.abs(ha - hb) + Math.abs(wa - wb)) / (maxH + maxW);
      const paletteTerm = Math.abs(pa - pb) / 10;
      const densityTerm = Math.abs(da - db);
      return (shapeTerm + paletteTerm + densityTerm) / 3;
    }
  },
  {
    name: "cell_count",
    feature: (grid) => `c${nonZeroCount(grid)}`,
    pairDistance: (a, b) => maxCells === 0 ? 0 : Math.abs(nonZeroCount(a) - nonZeroCount(b)) / maxCells
  }
];

const baselineResults = [];

for (const baseline of baselines) {
  const taskCollisions = [];
  for (const row of registerRows) {
    const taskGrids = allGrids.filter((g) => g.task_id === row.task_id);
    const features = taskGrids.map((g) => baseline.feature(g.grid));
    const unique = new Set(features);
    taskCollisions.push(1 - unique.size / features.length);
  }
  const pairResiduals = trainPairs.map((p) => baseline.pairDistance(p.input, p.output));
  const allFeatures = allGrids.map((g) => baseline.feature(g.grid));
  const globalUnique = new Set(allFeatures).size;

  baselineResults.push({
    baseline: baseline.name,
    mean_collision_residual: round(mean(taskCollisions), 6),
    mean_train_pair_residual: round(mean(pairResiduals), 6),
    global_unique_features: globalUnique,
    global_total_grids: allFeatures.length
  });
}

let shadowReference = null;
try {
  const shadowManifest = JSON.parse(await readFile(shadowManifestPath, "utf8"));
  shadowReference = {
    baseline: "shadow_operator_v0_reference",
    mean_collision_residual: shadowManifest.aggregate.meanSignatureCollisionResidual,
    mean_train_pair_residual: shadowManifest.aggregate.meanTrainPairResidual,
    global_unique_features: "",
    global_total_grids: shadowManifest.aggregate.gridCount
  };
} catch (err) {
  console.warn(`could not load shadow operator reference from ${shadowManifestPath}: ${err.message}`);
}

const rows = shadowReference ? [shadowReference, ...baselineResults] : baselineResults;

const manifest = {
  generatedAt: new Date().toISOString(),
  tool: "scripts/arc-phase2-baselines.mjs",
  scope: "registered public-training subset; comparison baselines for P_shadow_grid_v0",
  registerPath,
  shadowManifestPath,
  taskCount: registerRows.length,
  gridCount: allGrids.length,
  trainPairCount: trainPairs.length,
  baselines: rows,
  metricNotes: {
    mean_collision_residual: "per-task mean of 1 - unique_features / grids_in_task; lower = more discriminating",
    mean_train_pair_residual: "per-train-pair mean distance under each baseline's pair-distance function",
    distance_caveat: "pair-distance functions are baseline-specific and not directly comparable cell-for-cell; aggregates give a calibrated sense of whether the shadow operator's number is inside, below, or above the naive band"
  }
};

await mkdir(outDir, { recursive: true });
await writeFile(join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
await writeFile(join(outDir, "summary.csv"), toCsv(rows));

console.log(`ARC Phase 2 baselines: ${registerRows.length} task(s), ${allGrids.length} grid(s), ${trainPairs.length} train pair(s)`);
for (const r of rows) {
  console.log(`- ${r.baseline}: collision=${r.mean_collision_residual} pair=${r.mean_train_pair_residual} global_unique=${r.global_unique_features}`);
}

function paddedHamming(a, b) {
  const h = Math.max(a.length, b.length);
  const w = Math.max(a[0].length, b[0].length);
  let mismatch = 0;
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const va = (y < a.length && x < a[0].length) ? a[y][x] : 0;
      const vb = (y < b.length && x < b[0].length) ? b[y][x] : 0;
      if (va !== vb) {
        mismatch += 1;
      }
    }
  }
  return mismatch / (h * w);
}

function nonZeroCount(grid) {
  let n = 0;
  for (const row of grid) {
    for (const v of row) {
      if (v !== 0) {
        n += 1;
      }
    }
  }
  return n;
}

function mean(values) {
  return values.reduce((sum, v) => sum + Number(v), 0) / values.length;
}

function round(value, places) {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function parseCsv(text) {
  const lines = text.replace(/\r\n/g, "\n").split("\n").filter((line) => line.length > 0);
  if (lines.length === 0) return [];
  const header = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const cells = parseCsvLine(line);
    return Object.fromEntries(header.map((column, index) => [column, cells[index] ?? ""]));
  });
}

function parseCsvLine(line) {
  const cells = [];
  let cell = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === "\"" && line[i + 1] === "\"") {
        cell += "\"";
        i += 1;
      } else if (ch === "\"") {
        inQuotes = false;
      } else {
        cell += ch;
      }
    } else if (ch === "\"") {
      inQuotes = true;
    } else if (ch === ",") {
      cells.push(cell);
      cell = "";
    } else {
      cell += ch;
    }
  }
  cells.push(cell);
  return cells;
}

function toCsv(rows) {
  if (rows.length === 0) return "";
  const columns = Object.keys(rows[0]);
  return `${[columns.join(","), ...rows.map((row) => columns.map((column) => csvCell(row[column])).join(","))].join("\n")}\n`;
}

function csvCell(value) {
  const text = String(value ?? "");
  if (/[",\n]/.test(text)) {
    return `"${text.replaceAll("\"", "\"\"")}"`;
  }
  return text;
}

function parseArgs(argv) {
  const parsed = {};
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") {
      parsed.help = true;
    } else if (arg === "--data-dir") {
      parsed.dataDir = argv[++i];
    } else if (arg === "--register") {
      parsed.register = argv[++i];
    } else if (arg === "--out") {
      parsed.out = argv[++i];
    } else if (arg === "--shadow-manifest") {
      parsed.shadowManifest = argv[++i];
    } else {
      fail(`Unknown argument: ${arg}`);
    }
  }
  return parsed;
}

function printUsage() {
  console.log(`Usage:
  node scripts/arc-phase2-baselines.mjs --data-dir <ARC-AGI-2/data> --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase2-baselines [--shadow-manifest results/arc/phase2-projections/manifest.json]`);
}

function fail(message) {
  console.error(message);
  process.exit(1);
}
