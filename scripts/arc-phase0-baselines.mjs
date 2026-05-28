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
const seed = Number(args.seed ?? 20260528);

const registerRows = parseCsv(await readFile(registerPath, "utf8"))
  .filter((row) => row.status === "include" && row.split === "training");

if (registerRows.length === 0) {
  fail("Register has no included training rows.");
}

const tasks = [];
for (const row of registerRows) {
  const raw = await readFile(join(dataDir, "training", `${row.task_id}.json`), "utf8");
  const task = JSON.parse(raw.replace(/^\uFEFF/, ""));
  tasks.push({ row, task, sourceSha256: sha256(raw) });
}

const baselines = [
  ["random_valid", randomValidPredictions],
  ["identity_copy", identityCopyPredictions],
  ["dsl_lite_v0", dslLitePredictions]
];

const predictionRecords = [];
const summaries = [];

for (const [baselineName, predictor] of baselines) {
  const taskRecords = [];
  for (const taskRecord of tasks) {
    const predictions = predictor(taskRecord, seed);
    const scored = scoreTask(taskRecord.task, predictions);
    taskRecords.push({
      task_id: taskRecord.row.task_id,
      primary_prior: taskRecord.row.primary_prior,
      solved: scored.solved,
      pixel_accuracy: scored.pixelAccuracy,
      valid_prediction_count: predictions.reduce((sum, pairPreds) => sum + pairPreds.length, 0),
      prediction_shapes: predictions.map((pairPreds) => pairPreds.map((grid) => shapeLabel(grid)).join("|")).join(";")
    });
    predictionRecords.push({
      baseline: baselineName,
      task_id: taskRecord.row.task_id,
      predictions
    });
  }
  summaries.push(summarizeBaseline(baselineName, taskRecords));
}

const manifest = {
  generatedAt: new Date().toISOString(),
  tool: "scripts/arc-phase0-baselines.mjs",
  dataDir,
  registerPath,
  taskCount: tasks.length,
  seed,
  baselines: summaries,
  notes: [
    "Uses public training tasks only.",
    "Exact task match is primary; pixel accuracy is diagnostic only.",
    "dsl_lite_v0 fits only frozen, simple grid transforms and color maps."
  ]
};

await mkdir(outDir, { recursive: true });
await writeFile(join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
await writeFile(join(outDir, "summary.csv"), toCsv(summaries));
await writeFile(join(outDir, "predictions.json"), `${JSON.stringify(predictionRecords, null, 2)}\n`);

console.log(`ARC Phase 0 baselines: ${tasks.length} task(s)`);
for (const summary of summaries) {
  console.log(`- ${summary.baseline}: ${summary.task_exact}/${summary.tasks} exact (${summary.task_exact_rate})`);
}
console.log(`Wrote ${join(outDir, "manifest.json")}`);
console.log(`Wrote ${join(outDir, "summary.csv")}`);

function randomValidPredictions(taskRecord, baseSeed) {
  const task = taskRecord.task;
  const outputShapes = task.train.map((pair) => [pair.output.length, pair.output[0].length]);
  const palette = [...new Set(task.train.flatMap((pair) => pair.output.flat()))].sort((a, b) => a - b);
  return task.test.map((pair, testIndex) => {
    const attempts = [];
    for (let attempt = 0; attempt < 2; attempt += 1) {
      const [height, width] = outputShapes[(testIndex + attempt) % outputShapes.length];
      const rng = makeRng(seedFor(`${taskRecord.row.task_id}:${baseSeed}:random:${testIndex}:${attempt}`));
      attempts.push(Array.from({ length: height }, () =>
        Array.from({ length: width }, () => palette[Math.floor(rng() * palette.length)])
      ));
    }
    return attempts;
  });
}

function identityCopyPredictions(taskRecord) {
  const colorMap = fitColorMapForTransform(taskRecord.task.train, identity);
  return taskRecord.task.test.map((pair) => {
    const attempts = [cloneGrid(pair.input)];
    if (colorMap) {
      attempts.push(applyColorMap(pair.input, colorMap));
    }
    return uniqueGrids(attempts).slice(0, 2);
  });
}

function dslLitePredictions(taskRecord) {
  const transforms = [
    ["identity", identity],
    ["rot90", rotate90],
    ["rot180", rotate180],
    ["rot270", rotate270],
    ["reflect_h", reflectHorizontal],
    ["reflect_v", reflectVertical],
    ["transpose", transpose],
    ["anti_transpose", antiTranspose],
    ["crop_nonzero", cropNonZero]
  ];
  const candidates = [];
  for (const [name, transform] of transforms) {
    const colorMap = fitColorMapForTransform(taskRecord.task.train, transform);
    if (colorMap) {
      candidates.push({ name, transform, colorMap });
    }
  }
  const constant = fitConstantOutput(taskRecord.task.train);
  return taskRecord.task.test.map((pair) => {
    const attempts = [];
    for (const candidate of candidates) {
      attempts.push(applyColorMap(candidate.transform(pair.input), candidate.colorMap));
    }
    if (constant) {
      attempts.push(cloneGrid(constant));
    }
    if (attempts.length === 0) {
      attempts.push(cloneGrid(pair.input));
    }
    return uniqueGrids(attempts).slice(0, 2);
  });
}

function fitColorMapForTransform(trainPairs, transform) {
  const colorMap = new Map();
  for (const pair of trainPairs) {
    const transformed = transform(pair.input);
    if (!sameShape(transformed, pair.output)) {
      return null;
    }
    for (let y = 0; y < transformed.length; y += 1) {
      for (let x = 0; x < transformed[0].length; x += 1) {
        const src = transformed[y][x];
        const dst = pair.output[y][x];
        if (colorMap.has(src) && colorMap.get(src) !== dst) {
          return null;
        }
        colorMap.set(src, dst);
      }
    }
  }
  return colorMap;
}

function fitConstantOutput(trainPairs) {
  const [first] = trainPairs;
  if (!first) {
    return null;
  }
  return trainPairs.every((pair) => equalsGrid(pair.output, first.output)) ? first.output : null;
}

function scoreTask(task, predictionsByPair) {
  let solved = true;
  let pixelTotal = 0;
  for (let i = 0; i < task.test.length; i += 1) {
    const expected = task.test[i].output;
    const predictions = predictionsByPair[i] ?? [];
    const pairSolved = predictions.some((prediction) => equalsGrid(prediction, expected));
    if (!pairSolved) {
      solved = false;
    }
    pixelTotal += Math.max(0, ...predictions.map((prediction) => pixelAccuracy(prediction, expected)));
  }
  return {
    solved,
    pixelAccuracy: round(pixelTotal / task.test.length, 4)
  };
}

function summarizeBaseline(baseline, taskRecords) {
  const solved = taskRecords.filter((row) => row.solved);
  const pixelMean = taskRecords.reduce((sum, row) => sum + row.pixel_accuracy, 0) / taskRecords.length;
  return {
    baseline,
    tasks: taskRecords.length,
    task_exact: solved.length,
    task_exact_rate: round(solved.length / taskRecords.length, 4),
    mean_pixel_accuracy: round(pixelMean, 4),
    solved_task_ids: solved.map((row) => row.task_id).join(";")
  };
}

function applyColorMap(grid, colorMap) {
  return grid.map((row) => row.map((value) => colorMap.has(value) ? colorMap.get(value) : value));
}

function pixelAccuracy(a, b) {
  if (!sameShape(a, b)) {
    return 0;
  }
  let correct = 0;
  let total = 0;
  for (let y = 0; y < b.length; y += 1) {
    for (let x = 0; x < b[0].length; x += 1) {
      total += 1;
      if (a[y][x] === b[y][x]) {
        correct += 1;
      }
    }
  }
  return correct / total;
}

function shapeLabel(grid) {
  return `${grid.length}x${grid[0].length}`;
}

function sameShape(a, b) {
  return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a[0]?.length === b[0]?.length;
}

function equalsGrid(a, b) {
  return sameShape(a, b) && a.every((row, y) => row.every((value, x) => value === b[y][x]));
}

function uniqueGrids(grids) {
  const seen = new Set();
  const out = [];
  for (const grid of grids) {
    const key = JSON.stringify(grid);
    if (!seen.has(key)) {
      seen.add(key);
      out.push(grid);
    }
  }
  return out;
}

function cloneGrid(grid) {
  return grid.map((row) => [...row]);
}

function identity(grid) {
  return cloneGrid(grid);
}

function rotate90(grid) {
  const h = grid.length;
  const w = grid[0].length;
  return Array.from({ length: w }, (_, y) => Array.from({ length: h }, (_, x) => grid[h - 1 - x][y]));
}

function rotate180(grid) {
  return reflectVertical(reflectHorizontal(grid));
}

function rotate270(grid) {
  return rotate90(rotate180(grid));
}

function reflectHorizontal(grid) {
  return grid.map((row) => [...row].reverse());
}

function reflectVertical(grid) {
  return [...grid].reverse().map((row) => [...row]);
}

function transpose(grid) {
  return Array.from({ length: grid[0].length }, (_, y) => Array.from({ length: grid.length }, (_, x) => grid[x][y]));
}

function antiTranspose(grid) {
  return reflectHorizontal(reflectVertical(transpose(grid)));
}

function cropNonZero(grid) {
  const points = [];
  for (let y = 0; y < grid.length; y += 1) {
    for (let x = 0; x < grid[0].length; x += 1) {
      if (grid[y][x] !== 0) {
        points.push([x, y]);
      }
    }
  }
  if (points.length === 0) {
    return cloneGrid(grid);
  }
  const xs = points.map(([x]) => x);
  const ys = points.map(([, y]) => y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  return grid.slice(minY, maxY + 1).map((row) => row.slice(minX, maxX + 1));
}

function parseCsv(text) {
  const lines = text.replace(/\r\n/g, "\n").split("\n").filter((line) => line.length > 0);
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
  if (rows.length === 0) {
    return "";
  }
  const header = Object.keys(rows[0]);
  return `${[header.join(","), ...rows.map((row) => header.map((column) => csvCell(row[column])).join(","))].join("\n")}\n`;
}

function csvCell(value) {
  const text = String(value ?? "");
  if (/[",\n]/.test(text)) {
    return `"${text.replaceAll("\"", "\"\"")}"`;
  }
  return text;
}

function makeRng(initialSeed) {
  let state = initialSeed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

function seedFor(text) {
  return Number.parseInt(sha256(text).slice(0, 8), 16);
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function round(value, places) {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
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
    } else if (arg === "--seed") {
      parsed.seed = argv[++i];
    } else {
      fail(`Unknown argument: ${arg}`);
    }
  }
  return parsed;
}

function printUsage() {
  console.log(`Usage:
  node scripts/arc-phase0-baselines.mjs --data-dir <ARC-AGI-2/data> --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase0-baselines`);
}

function fail(message) {
  console.error(message);
  process.exit(1);
}
