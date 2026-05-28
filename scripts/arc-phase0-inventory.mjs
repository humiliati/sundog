import { createHash } from "node:crypto";
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";

const args = parseArgs(process.argv.slice(2));

if (args.help || !args.dataDir) {
  printUsage();
  process.exit(args.help ? 0 : 2);
}

const dataDir = resolve(args.dataDir);
const requestedSplit = args.split ?? "all";
const includeEvaluationTestOutput = Boolean(args.includeEvaluationTestOutput);
const authorizeEvaluationLeak = Boolean(args.authorizeEvaluationLeak);
const outDir = args.out ? resolve(args.out) : null;

if (includeEvaluationTestOutput && !authorizeEvaluationLeak) {
  fail([
    "--include-evaluation-test-output requires the second flag --authorize-evaluation-leak.",
    "Per docs/prereg/arc/PHASE0_TASK_SUBSET_SPEC.md, emitting evaluation test outputs is reserved",
    "for a final post-freeze audit. Both flags must be set together to prevent accidental leaks."
  ].join("\n"));
}

if (includeEvaluationTestOutput && authorizeEvaluationLeak) {
  if (!outDir) {
    fail("Privileged audit run requires --out pointing at a path ending in _PRIVILEGED_AUDIT (e.g. results/arc/phase0-inventory_PRIVILEGED_AUDIT).");
  }
  if (!outDir.endsWith("_PRIVILEGED_AUDIT")) {
    fail(`Privileged audit run must write to a path ending in _PRIVILEGED_AUDIT, refusing to write to ${outDir}. This stops the leaked artifact from overwriting the non-privileged inventory.`);
  }
}

const splitNames = requestedSplit === "all" ? ["training", "evaluation"] : [requestedSplit];
const rows = [];
const sourceFiles = [];

for (const split of splitNames) {
  if (!["training", "evaluation"].includes(split)) {
    fail(`Unknown split "${split}". Use training, evaluation, or all.`);
  }

  const splitDir = join(dataDir, split);
  const files = (await readdir(splitDir)).filter((file) => file.endsWith(".json")).sort();

  for (const file of files) {
    const taskId = file.replace(/\.json$/i, "");
    const path = join(splitDir, file);
    const raw = await readFile(path, "utf8");
    const task = JSON.parse(raw.replace(/^\uFEFF/, ""));
    validateTask(task, `${split}/${file}`);
    const hash = sha256(raw);
    sourceFiles.push({ split, taskId, file: `${split}/${file}`, sha256: hash });
    rows.push(summarizeTask({
      split,
      taskId,
      task,
      sourceSha256: hash,
      includeEvaluationTestOutput
    }));
  }
}

const summary = summarizeRows(rows);
const manifest = {
  generatedAt: new Date().toISOString(),
  tool: "scripts/arc-phase0-inventory.mjs",
  dataDir,
  requestedSplit,
  includeEvaluationTestOutput,
  authorizeEvaluationLeak,
  privilegedAudit: includeEvaluationTestOutput && authorizeEvaluationLeak,
  evaluationPolicy: includeEvaluationTestOutput
    ? "PRIVILEGED AUDIT: evaluation test-output metadata included by explicit double-flag override; this artifact must not be referenced by any non-audit Phase 0/1 verdict"
    : "evaluation test outputs omitted from emitted metadata",
  summary,
  sourceFingerprint: sha256(JSON.stringify(sourceFiles))
};

if (outDir) {
  await mkdir(outDir, { recursive: true });
  await writeFile(join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  await writeFile(join(outDir, "tasks.csv"), toCsv(rows));
  await writeFile(join(outDir, "split-summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
}

console.log(`ARC Phase 0 inventory: ${rows.length} task(s) from ${dataDir}`);
for (const [split, splitSummary] of Object.entries(summary.bySplit)) {
  console.log(`- ${split}: ${splitSummary.tasks} task(s)`);
}
if (outDir) {
  console.log(`Wrote ${join(outDir, "manifest.json")}`);
  console.log(`Wrote ${join(outDir, "tasks.csv")}`);
}

function parseArgs(argv) {
  const parsed = {};
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") {
      parsed.help = true;
    } else if (arg === "--data-dir") {
      parsed.dataDir = argv[++i];
    } else if (arg === "--out") {
      parsed.out = argv[++i];
    } else if (arg === "--split") {
      parsed.split = argv[++i];
    } else if (arg === "--include-evaluation-test-output") {
      parsed.includeEvaluationTestOutput = true;
    } else if (arg === "--authorize-evaluation-leak") {
      parsed.authorizeEvaluationLeak = true;
    } else {
      fail(`Unknown argument: ${arg}`);
    }
  }
  return parsed;
}

function printUsage() {
  console.log(`Usage:
  node scripts/arc-phase0-inventory.mjs --data-dir <ARC-AGI-2/data> [--out <dir>]

Options:
  --split training|evaluation|all       Split to inventory. Default: all.
  --include-evaluation-test-output      Emit evaluation test-output metadata.
                                        Requires --authorize-evaluation-leak.
  --authorize-evaluation-leak           Second-key acknowledgement for the
                                        above. Also forces --out to a path
                                        ending in _PRIVILEGED_AUDIT.
  --help                                Show this help.

Example:
  node scripts/arc-phase0-inventory.mjs --data-dir "$env:USERPROFILE\\Datasets\\ARC-AGI-2\\data" --out results/arc/phase0-inventory`);
}

function validateTask(task, label) {
  if (!task || !Array.isArray(task.train) || !Array.isArray(task.test)) {
    fail(`${label} is not an ARC task with train/test arrays.`);
  }
  for (const [section, pairs] of [["train", task.train], ["test", task.test]]) {
    pairs.forEach((pair, index) => {
      if (!pair || !Array.isArray(pair.input)) {
        fail(`${label} ${section}[${index}] is missing input grid.`);
      }
      validateGrid(pair.input, `${label} ${section}[${index}].input`);
      if (pair.output !== undefined) {
        validateGrid(pair.output, `${label} ${section}[${index}].output`);
      }
    });
  }
}

function validateGrid(grid, label) {
  if (!Array.isArray(grid) || grid.length === 0 || !Array.isArray(grid[0]) || grid[0].length === 0) {
    fail(`${label} must be a non-empty 2D grid.`);
  }
  const width = grid[0].length;
  for (let y = 0; y < grid.length; y += 1) {
    if (!Array.isArray(grid[y]) || grid[y].length !== width) {
      fail(`${label} row ${y} is ragged.`);
    }
    for (const value of grid[y]) {
      if (!Number.isInteger(value) || value < 0 || value > 9) {
        fail(`${label} contains non-ARC color value ${value}.`);
      }
    }
  }
}

function summarizeTask({ split, taskId, task, sourceSha256, includeEvaluationTestOutput }) {
  const trainPairs = task.train;
  const testPairs = task.test;
  const includeTestOutputs = split !== "evaluation" || includeEvaluationTestOutput;
  const emittedGrids = [];
  const trainShapeChanges = [];
  const testShapeChanges = [];

  for (const pair of trainPairs) {
    emittedGrids.push(pair.input, pair.output);
    trainShapeChanges.push(!sameShape(pair.input, pair.output));
  }

  for (const pair of testPairs) {
    emittedGrids.push(pair.input);
    if (includeTestOutputs && pair.output !== undefined) {
      emittedGrids.push(pair.output);
      testShapeChanges.push(!sameShape(pair.input, pair.output));
    }
  }

  const gridStats = emittedGrids.map(gridSummary);
  const colors = [...new Set(gridStats.flatMap((stat) => stat.colors))].sort((a, b) => a - b);
  const heights = gridStats.map((stat) => stat.height);
  const widths = gridStats.map((stat) => stat.width);
  const areas = gridStats.map((stat) => stat.area);
  const densities = gridStats.map((stat) => stat.nonZeroDensity);
  const components = gridStats.map((stat) => stat.nonZeroComponents);
  const symmetryFlags = new Set(gridStats.flatMap((stat) => stat.symmetries));
  const priorHints = priorHintsFor({
    colors,
    gridStats,
    trainShapeChanges,
    testShapeChanges,
    components,
    symmetryFlags
  });
  const rowCore = {
    task_id: taskId,
    split,
    source_sha256: sourceSha256,
    train_pairs: trainPairs.length,
    test_pairs: testPairs.length,
    emitted_test_output_metadata: includeTestOutputs ? "yes" : "no",
    min_height: min(heights),
    max_height: max(heights),
    min_width: min(widths),
    max_width: max(widths),
    max_area: max(areas),
    color_count: colors.length,
    colors: colors.join(""),
    min_nonzero_density: round(min(densities), 4),
    max_nonzero_density: round(max(densities), 4),
    max_nonzero_components: max(components),
    train_shape_changes: trainShapeChanges.filter(Boolean).length,
    test_shape_changes: includeTestOutputs ? testShapeChanges.filter(Boolean).length : "",
    symmetry_hints: [...symmetryFlags].sort().join(";"),
    prior_hints: priorHints.join(";")
  };
  return {
    ...rowCore,
    inventory_row_hash: sha256(JSON.stringify(rowCore))
  };
}

function gridSummary(grid) {
  const height = grid.length;
  const width = grid[0].length;
  const area = height * width;
  const colors = [...new Set(grid.flat())].sort((a, b) => a - b);
  let nonZero = 0;
  for (const row of grid) {
    for (const value of row) {
      if (value !== 0) {
        nonZero += 1;
      }
    }
  }
  return {
    height,
    width,
    area,
    colors,
    nonZeroDensity: nonZero / area,
    nonZeroComponents: countComponents(grid),
    symmetries: symmetryHints(grid)
  };
}

function countComponents(grid) {
  const height = grid.length;
  const width = grid[0].length;
  const seen = Array.from({ length: height }, () => Array(width).fill(false));
  let count = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (grid[y][x] === 0 || seen[y][x]) {
        continue;
      }
      count += 1;
      const color = grid[y][x];
      const stack = [[x, y]];
      seen[y][x] = true;
      while (stack.length > 0) {
        const [cx, cy] = stack.pop();
        for (const [nx, ny] of [[cx + 1, cy], [cx - 1, cy], [cx, cy + 1], [cx, cy - 1]]) {
          if (ny < 0 || ny >= height || nx < 0 || nx >= width || seen[ny][nx] || grid[ny][nx] !== color) {
            continue;
          }
          seen[ny][nx] = true;
          stack.push([nx, ny]);
        }
      }
    }
  }
  return count;
}

function symmetryHints(grid) {
  const hints = [];
  if (grid.length > 1 && grid[0].length > 1 && equalsGrid(grid, reflectHorizontal(grid))) {
    hints.push("reflect_h");
  }
  if (grid.length > 1 && grid[0].length > 1 && equalsGrid(grid, reflectVertical(grid))) {
    hints.push("reflect_v");
  }
  if (equalsGrid(grid, rotate180(grid))) {
    hints.push("rot180");
  }
  return hints;
}

function priorHintsFor({ colors, gridStats, trainShapeChanges, testShapeChanges, components, symmetryFlags }) {
  const hints = new Set();
  if (components.some((value) => value >= 2)) {
    hints.add("objectness");
  }
  if (components.some((value) => value >= 3)) {
    hints.add("counting");
  }
  if (symmetryFlags.size > 0) {
    hints.add("symmetry");
  }
  if (trainShapeChanges.some(Boolean) || testShapeChanges.some(Boolean)) {
    hints.add("spatial_transform");
  }
  if (colors.length >= 3) {
    hints.add("color_role");
  }
  if (gridStats.some((stat) => stat.nonZeroDensity > 0 && stat.nonZeroDensity < 0.35)) {
    hints.add("local_completion");
  }
  return [...hints].sort();
}

function summarizeRows(allRows) {
  const bySplit = {};
  for (const row of allRows) {
    bySplit[row.split] ??= { tasks: 0, trainPairs: 0, testPairs: 0 };
    bySplit[row.split].tasks += 1;
    bySplit[row.split].trainPairs += row.train_pairs;
    bySplit[row.split].testPairs += row.test_pairs;
  }
  return { tasks: allRows.length, bySplit };
}

function sameShape(a, b) {
  return a.length === b.length && a[0].length === b[0].length;
}

function equalsGrid(a, b) {
  return a.length === b.length && a.every((row, y) => row.length === b[y].length && row.every((value, x) => value === b[y][x]));
}

function reflectHorizontal(grid) {
  return grid.map((row) => [...row].reverse());
}

function reflectVertical(grid) {
  return [...grid].reverse().map((row) => [...row]);
}

function rotate180(grid) {
  return reflectVertical(reflectHorizontal(grid));
}

function toCsv(allRows) {
  if (allRows.length === 0) {
    return "";
  }
  const columns = Object.keys(allRows[0]);
  const lines = [columns.join(",")];
  for (const row of allRows) {
    lines.push(columns.map((column) => csvCell(row[column])).join(","));
  }
  return `${lines.join("\n")}\n`;
}

function csvCell(value) {
  const text = String(value ?? "");
  if (/[",\n]/.test(text)) {
    return `"${text.replaceAll("\"", "\"\"")}"`;
  }
  return text;
}

function min(values) {
  return Math.min(...values);
}

function max(values) {
  return Math.max(...values);
}

function round(value, places) {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function fail(message) {
  console.error(message);
  process.exit(1);
}
