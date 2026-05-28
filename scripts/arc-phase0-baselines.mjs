import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const IS_MAIN = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (IS_MAIN) {
  await main();
}

export const ARC_PHASE0_BASELINE_NAMES = [
  "random_valid",
  "identity_copy",
  "dsl_lite_v0",
  "dsl_lite_v1",
  "dsl_lite_v2",
  "tiny_learned_v0"
];

export function arcPhase0BaselinePredictions(baselineName, taskRecord, seed = 20260528) {
  if (baselineName === "random_valid") return randomValidPredictions(taskRecord, seed);
  if (baselineName === "identity_copy") return identityCopyPredictions(taskRecord);
  if (baselineName === "dsl_lite_v0") return dslLitePredictions(taskRecord);
  if (baselineName === "dsl_lite_v1") return dslLiteV1Predictions(taskRecord);
  if (baselineName === "dsl_lite_v2") return dslLiteV2Predictions(taskRecord);
  if (baselineName === "tiny_learned_v0") return tinyLearnedV0Predictions(taskRecord);
  throw new Error(`Unknown Phase 0 baseline: ${baselineName}`);
}

export function arcPhase0ScoreTask(task, predictionsByPair) {
  return scoreTask(task, predictionsByPair);
}

async function main() {
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
  ["dsl_lite_v0", dslLitePredictions],
  ["dsl_lite_v1", dslLiteV1Predictions],
  ["dsl_lite_v2", dslLiteV2Predictions],
  ["tiny_learned_v0", tinyLearnedV0Predictions]
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
    "dsl_lite_v0 fits only frozen, simple grid transforms and color maps.",
    "dsl_lite_v1 adds tile, translate, palette_permute (depth 1) per spec line 132 preregistered primitives; v0 logic frozen.",
    "dsl_lite_v2 adds pad, fill_enclosed, component_copy_largest and depth-2 composition over union(v0 ∪ v1 ∪ v2) structural transforms; v0/v1 frozen.",
    "tiny_learned_v0 is per-task nearest-neighbor over train pairs by padded pixel Hamming distance."
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
}

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

function dslLiteV1Predictions(taskRecord) {
  const trainPairs = taskRecord.task.train;
  const candidates = [];

  const tileFactors = fitTileFactors(trainPairs);
  if (tileFactors) {
    const transform = (grid) => tileGrid(grid, tileFactors.ky, tileFactors.kx);
    const colorMap = fitColorMapForTransform(trainPairs, transform);
    if (colorMap) {
      candidates.push((input) => applyColorMap(transform(input), colorMap));
    }
  }

  const shift = fitTranslate(trainPairs);
  if (shift) {
    const transform = (grid) => translateGrid(grid, shift.dy, shift.dx);
    const colorMap = fitColorMapForTransform(trainPairs, transform);
    if (colorMap) {
      candidates.push((input) => applyColorMap(transform(input), colorMap));
    }
  }

  const permutationMaps = fitPalettePermutations(trainPairs);
  for (const map of permutationMaps) {
    candidates.push((input) => applyColorMap(input, map));
  }

  return taskRecord.task.test.map((pair) => {
    const attempts = [];
    for (const fn of candidates) {
      attempts.push(fn(pair.input));
    }
    if (attempts.length === 0) {
      attempts.push(cloneGrid(pair.input));
    }
    return uniqueGrids(attempts).slice(0, 2);
  });
}

function dslLiteV2Predictions(taskRecord) {
  const trainPairs = taskRecord.task.train;
  const candidates = [];

  const v2Structural = enumerateV2StructuralTransforms(trainPairs);
  for (const transform of v2Structural) {
    const colorMap = fitColorMapForTransform(trainPairs, transform);
    if (colorMap) {
      candidates.push((input) => applyColorMap(safeApply(transform, input), colorMap));
    }
  }

  const allStructural = [
    ...enumerateV0StructuralTransforms(),
    ...enumerateV1StructuralTransforms(trainPairs),
    ...v2Structural
  ];

  for (const t1 of allStructural) {
    for (const t2 of allStructural) {
      const composed = (grid) => safeApply(t2, safeApply(t1, grid));
      const colorMap = fitColorMapForTransform(trainPairs, composed);
      if (colorMap) {
        candidates.push((input) => applyColorMap(composed(input), colorMap));
      }
    }
  }

  return taskRecord.task.test.map((pair) => {
    const attempts = [];
    for (const fn of candidates) {
      try {
        attempts.push(fn(pair.input));
      } catch {
        // Some composed transforms may throw on edge-case grid shapes; skip.
      }
    }
    if (attempts.length === 0) {
      attempts.push(cloneGrid(pair.input));
    }
    return uniqueGrids(attempts).slice(0, 2);
  });
}

function enumerateV0StructuralTransforms() {
  return [identity, rotate90, rotate180, rotate270, reflectHorizontal, reflectVertical, transpose, antiTranspose, cropNonZero];
}

function enumerateV1StructuralTransforms(trainPairs) {
  const out = [];
  const tileFactors = fitTileFactors(trainPairs);
  if (tileFactors) {
    out.push((grid) => tileGrid(grid, tileFactors.ky, tileFactors.kx));
  }
  const shift = fitTranslate(trainPairs);
  if (shift) {
    out.push((grid) => translateGrid(grid, shift.dy, shift.dx));
  }
  return out;
}

function enumerateV2StructuralTransforms(trainPairs) {
  const out = [];
  const padding = fitPad(trainPairs);
  if (padding) {
    out.push((grid) => padGrid(grid, padding));
  }
  out.push(fillEnclosed);
  out.push(extractLargestComponent);
  return out;
}

function fitPad(trainPairs) {
  let result = null;
  for (const pair of trainPairs) {
    const ih = pair.input.length;
    const iw = pair.input[0].length;
    const oh = pair.output.length;
    const ow = pair.output[0].length;
    if (oh < ih || ow < iw || (oh === ih && ow === iw)) {
      return null;
    }
    let pairResult = null;
    outer: for (let top = 0; top <= oh - ih; top += 1) {
      for (let left = 0; left <= ow - iw; left += 1) {
        let match = true;
        for (let y = 0; y < ih && match; y += 1) {
          for (let x = 0; x < iw; x += 1) {
            if (pair.output[top + y][left + x] !== pair.input[y][x]) {
              match = false;
              break;
            }
          }
        }
        if (!match) {
          continue;
        }
        const padColors = new Set();
        for (let y = 0; y < oh; y += 1) {
          for (let x = 0; x < ow; x += 1) {
            const inInput = y >= top && y < top + ih && x >= left && x < left + iw;
            if (!inInput) {
              padColors.add(pair.output[y][x]);
            }
          }
        }
        if (padColors.size > 1) {
          continue;
        }
        const padColor = padColors.size === 0 ? 0 : [...padColors][0];
        pairResult = {
          top,
          bottom: oh - ih - top,
          left,
          right: ow - iw - left,
          padColor
        };
        break outer;
      }
    }
    if (!pairResult) {
      return null;
    }
    if (result === null) {
      result = pairResult;
    } else if (
      result.top !== pairResult.top
      || result.bottom !== pairResult.bottom
      || result.left !== pairResult.left
      || result.right !== pairResult.right
      || result.padColor !== pairResult.padColor
    ) {
      return null;
    }
  }
  return result;
}

function padGrid(grid, padding) {
  const ih = grid.length;
  const iw = grid[0].length;
  const oh = ih + padding.top + padding.bottom;
  const ow = iw + padding.left + padding.right;
  const out = Array.from({ length: oh }, () => new Array(ow).fill(padding.padColor));
  for (let y = 0; y < ih; y += 1) {
    for (let x = 0; x < iw; x += 1) {
      out[padding.top + y][padding.left + x] = grid[y][x];
    }
  }
  return out;
}

function fillEnclosed(grid) {
  const h = grid.length;
  const w = grid[0].length;
  const out = cloneGrid(grid);
  const visited = Array.from({ length: h }, () => new Array(w).fill(false));
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      if (grid[y][x] !== 0 || visited[y][x]) {
        continue;
      }
      const region = [[x, y]];
      visited[y][x] = true;
      let touchesBoundary = false;
      const surrounding = new Set();
      for (let i = 0; i < region.length; i += 1) {
        const [cx, cy] = region[i];
        if (cx === 0 || cx === w - 1 || cy === 0 || cy === h - 1) {
          touchesBoundary = true;
        }
        for (const [nx, ny] of [[cx + 1, cy], [cx - 1, cy], [cx, cy + 1], [cx, cy - 1]]) {
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) {
            continue;
          }
          if (grid[ny][nx] === 0 && !visited[ny][nx]) {
            visited[ny][nx] = true;
            region.push([nx, ny]);
          } else if (grid[ny][nx] !== 0) {
            surrounding.add(grid[ny][nx]);
          }
        }
      }
      if (!touchesBoundary && surrounding.size === 1) {
        const color = [...surrounding][0];
        for (const [cx, cy] of region) {
          out[cy][cx] = color;
        }
      }
    }
  }
  return out;
}

function extractLargestComponent(grid) {
  const h = grid.length;
  const w = grid[0].length;
  const visited = Array.from({ length: h }, () => new Array(w).fill(false));
  let best = null;
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      if (grid[y][x] === 0 || visited[y][x]) {
        continue;
      }
      const color = grid[y][x];
      const cells = [[x, y]];
      visited[y][x] = true;
      for (let i = 0; i < cells.length; i += 1) {
        const [cx, cy] = cells[i];
        for (const [nx, ny] of [[cx + 1, cy], [cx - 1, cy], [cx, cy + 1], [cx, cy - 1]]) {
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) {
            continue;
          }
          if (visited[ny][nx] || grid[ny][nx] !== color) {
            continue;
          }
          visited[ny][nx] = true;
          cells.push([nx, ny]);
        }
      }
      if (!best || cells.length > best.cells.length) {
        best = { color, cells };
      }
    }
  }
  if (!best) {
    return cloneGrid(grid);
  }
  const xs = best.cells.map(([x]) => x);
  const ys = best.cells.map(([, y]) => y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const out = Array.from({ length: maxY - minY + 1 }, () => new Array(maxX - minX + 1).fill(0));
  for (const [x, y] of best.cells) {
    out[y - minY][x - minX] = best.color;
  }
  return out;
}

function safeApply(transform, grid) {
  if (!Array.isArray(grid) || grid.length === 0 || !Array.isArray(grid[0]) || grid[0].length === 0) {
    throw new Error("invalid grid for transform");
  }
  return transform(grid);
}

function tinyLearnedV0Predictions(taskRecord) {
  const trainPairs = taskRecord.task.train;
  return taskRecord.task.test.map((pair) => {
    const ranked = trainPairs
      .map((tp, idx) => ({ idx, distance: paddedHammingDistance(pair.input, tp.input) }))
      .sort((a, b) => a.distance - b.distance);
    const attempts = [];
    for (const entry of ranked) {
      attempts.push(cloneGrid(trainPairs[entry.idx].output));
      if (attempts.length >= 2) {
        break;
      }
    }
    if (attempts.length === 0) {
      attempts.push(cloneGrid(pair.input));
    }
    return uniqueGrids(attempts).slice(0, 2);
  });
}

function fitTileFactors(trainPairs) {
  let ky = null;
  let kx = null;
  for (const pair of trainPairs) {
    const ih = pair.input.length;
    const iw = pair.input[0].length;
    const oh = pair.output.length;
    const ow = pair.output[0].length;
    if (ih === 0 || iw === 0 || oh % ih !== 0 || ow % iw !== 0) {
      return null;
    }
    const curKy = oh / ih;
    const curKx = ow / iw;
    if (ky === null) {
      ky = curKy;
      kx = curKx;
    } else if (curKy !== ky || curKx !== kx) {
      return null;
    }
  }
  if (ky === null || ky < 1 || kx < 1 || (ky === 1 && kx === 1)) {
    return null;
  }
  return { ky, kx };
}

function tileGrid(grid, ky, kx) {
  const h = grid.length;
  const w = grid[0].length;
  const out = [];
  for (let by = 0; by < ky; by += 1) {
    for (let y = 0; y < h; y += 1) {
      const row = new Array(w * kx);
      for (let bx = 0; bx < kx; bx += 1) {
        for (let x = 0; x < w; x += 1) {
          row[bx * w + x] = grid[y][x];
        }
      }
      out.push(row);
    }
  }
  return out;
}

function fitTranslate(trainPairs) {
  if (trainPairs.length === 0) {
    return null;
  }
  const h = trainPairs[0].input.length;
  const w = trainPairs[0].input[0].length;
  for (const pair of trainPairs) {
    if (pair.input.length !== h || pair.input[0].length !== w) {
      return null;
    }
    if (!sameShape(pair.input, pair.output)) {
      return null;
    }
  }
  for (let dy = -h + 1; dy < h; dy += 1) {
    for (let dx = -w + 1; dx < w; dx += 1) {
      if (dy === 0 && dx === 0) {
        continue;
      }
      const transform = (grid) => translateGrid(grid, dy, dx);
      if (fitColorMapForTransform(trainPairs, transform)) {
        return { dy, dx };
      }
    }
  }
  return null;
}

function translateGrid(grid, dy, dx) {
  const h = grid.length;
  const w = grid[0].length;
  const out = Array.from({ length: h }, () => new Array(w).fill(0));
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const ny = y + dy;
      const nx = x + dx;
      if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
        out[ny][nx] = grid[y][x];
      }
    }
  }
  return out;
}

function fitPalettePermutations(trainPairs) {
  for (const pair of trainPairs) {
    if (!sameShape(pair.input, pair.output)) {
      return [];
    }
  }
  const palette = [...new Set(
    trainPairs.flatMap((pair) => pair.input.flat().concat(pair.output.flat()))
  )].sort((a, b) => a - b);
  if (palette.length === 0 || palette.length > 5) {
    return [];
  }
  const results = [];
  for (const perm of permutationsOf(palette)) {
    let identical = true;
    for (let i = 0; i < palette.length; i += 1) {
      if (perm[i] !== palette[i]) {
        identical = false;
        break;
      }
    }
    if (identical) {
      continue;
    }
    const map = new Map();
    for (let i = 0; i < palette.length; i += 1) {
      map.set(palette[i], perm[i]);
    }
    let consistent = true;
    for (const pair of trainPairs) {
      if (!equalsGrid(applyColorMap(pair.input, map), pair.output)) {
        consistent = false;
        break;
      }
    }
    if (consistent) {
      results.push(map);
    }
  }
  return results;
}

function permutationsOf(arr) {
  if (arr.length <= 1) {
    return [arr.slice()];
  }
  const out = [];
  for (let i = 0; i < arr.length; i += 1) {
    const rest = arr.slice(0, i).concat(arr.slice(i + 1));
    for (const sub of permutationsOf(rest)) {
      out.push([arr[i], ...sub]);
    }
  }
  return out;
}

function paddedHammingDistance(a, b) {
  const h = Math.max(a.length, b.length);
  const w = Math.max(a[0].length, b[0].length);
  let count = 0;
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const va = (y < a.length && x < a[0].length) ? a[y][x] : 0;
      const vb = (y < b.length && x < b[0].length) ? b[y][x] : 0;
      if (va !== vb) {
        count += 1;
      }
    }
  }
  return count;
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
