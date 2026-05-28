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

const D4_TRANSFORMS = [
  { name: "id", fn: (x, y) => [x, y] },
  { name: "rot90", fn: (x, y) => [y, -x] },
  { name: "rot180", fn: (x, y) => [-x, -y] },
  { name: "rot270", fn: (x, y) => [-y, x] },
  { name: "reflect_x", fn: (x, y) => [-x, y] },
  { name: "reflect_y", fn: (x, y) => [x, -y] },
  { name: "transpose", fn: (x, y) => [y, x] },
  { name: "anti_transpose", fn: (x, y) => [-y, -x] }
];

const D4_STENCIL_TRANSFORMS = [
  (grid) => grid,
  rotate90,
  rotate180,
  rotate270,
  reflectHorizontal,
  reflectVertical,
  transpose,
  antiTranspose
];

const registerRows = parseCsv(await readFile(registerPath, "utf8"))
  .filter((row) => row.status === "include" && row.split === "training");

if (registerRows.length === 0) {
  fail("Register has no included training rows.");
}

const gridRows = [];
const taskRows = [];

for (const row of registerRows) {
  const raw = await readFile(join(dataDir, "training", `${row.task_id}.json`), "utf8");
  const task = JSON.parse(raw.replace(/^\uFEFF/, ""));
  const projections = [];
  const pairResiduals = [];

  task.train.forEach((pair, pairIndex) => {
    const inputProjection = projectionRecord(row.task_id, "train_input", pairIndex, pair.input);
    const outputProjection = projectionRecord(row.task_id, "train_output", pairIndex, pair.output);
    projections.push(inputProjection, outputProjection);
    pairResiduals.push(alignmentResidual(pair.input, pair.output));
  });

  task.test.forEach((pair, pairIndex) => {
    projections.push(projectionRecord(row.task_id, "test_input", pairIndex, pair.input));
  });

  gridRows.push(...projections);

  const uniqueSignatures = new Set(projections.map((projection) => projection.signature_hash));
  const uniqueShapePaletteSignatures = new Set(projections.map((projection) =>
    `${projection.shape}|${projection.palette}|${projection.signature_hash}`
  ));
  const signatureCollisionResidual = 1 - uniqueSignatures.size / projections.length;
  const meanPairResidual = mean(pairResiduals);
  const maxPairResidual = Math.max(...pairResiduals);

  taskRows.push({
    task_id: row.task_id,
    primary_prior: row.primary_prior,
    train_pairs: task.train.length,
    test_inputs: task.test.length,
    grid_count: projections.length,
    unique_signature_count: uniqueSignatures.size,
    signature_collision_residual: round(signatureCollisionResidual, 6),
    unique_shape_palette_signature_count: uniqueShapePaletteSignatures.size,
    mean_train_pair_residual: round(meanPairResidual, 6),
    max_train_pair_residual: round(maxPairResidual, 6),
    mean_density: round(mean(projections.map((projection) => projection.density)), 6),
    signal_label: signalLabel(signatureCollisionResidual, meanPairResidual)
  });
}

const aggregate = {
  taskCount: taskRows.length,
  gridCount: gridRows.length,
  meanSignatureCollisionResidual: round(mean(taskRows.map((row) => row.signature_collision_residual)), 6),
  meanTrainPairResidual: round(mean(taskRows.map((row) => row.mean_train_pair_residual)), 6),
  signalLabels: countBy(taskRows.map((row) => row.signal_label))
};

const manifest = {
  generatedAt: new Date().toISOString(),
  tool: "scripts/arc-phase2-projections.mjs",
  scope: "registered public-training subset only; projection measurement, no answer scoring",
  operator: {
    version: "P_shadow_grid_v0",
    localRadius: 1,
    inheritedSyntheticGate: "docs/prereg/arc/PHASE1_SHADOW_DOMAIN_SPEC.md"
  },
  registerPath,
  taskCount: taskRows.length,
  aggregate
};

await mkdir(outDir, { recursive: true });
await writeFile(join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
await writeFile(join(outDir, "task-summary.csv"), toCsv(taskRows));
await writeFile(join(outDir, "grid-projections.json"), `${JSON.stringify(gridRows, null, 2)}\n`);

console.log(`ARC Phase 2 projections: ${taskRows.length} task(s), ${gridRows.length} grid(s)`);
console.log(`mean signature collision residual: ${aggregate.meanSignatureCollisionResidual}`);
console.log(`mean train-pair residual: ${aggregate.meanTrainPairResidual}`);
for (const [label, count] of Object.entries(aggregate.signalLabels)) {
  console.log(`- ${label}: ${count}`);
}

function projectionRecord(taskId, role, pairIndex, grid) {
  const projection = projectGridShadow(grid);
  return {
    task_id: taskId,
    role,
    pair_index: pairIndex,
    shape: projection.shape.join("x"),
    palette: projection.palette.join(""),
    nonzero_cells: projection.nonZeroCells,
    nonzero_components: projection.nonZeroComponents,
    density: projection.density,
    signature_hash: sha256(projection.canonicalObjectSignature),
    local_bag_hash: sha256(JSON.stringify(projection.localSignatureBag)),
    canonical_object_signature: projection.canonicalObjectSignature
  };
}

function projectGridShadow(grid) {
  const nonZero = nonZeroCells(grid);
  const localBag = nonZero
    .map((cell) => canonicalStencil(grid, cell.x, cell.y, 1))
    .sort();
  return {
    shape: [grid.length, grid[0].length],
    palette: [...new Set(grid.flat())].sort((a, b) => a - b),
    nonZeroCells: nonZero.length,
    nonZeroComponents: countComponents(grid),
    density: round(nonZero.length / (grid.length * grid[0].length), 6),
    localSignatureBag: localBag,
    canonicalObjectSignature: canonicalObjectSignature(grid)
  };
}

function canonicalObjectSignature(grid) {
  const variants = objectVariants(grid);
  if (variants.length === 0) {
    return "empty";
  }
  return variants.map((variant) => variant.signature).sort()[0];
}

function alignmentResidual(left, right) {
  const leftVariants = objectVariants(left);
  const rightVariants = objectVariants(right);
  if (leftVariants.length === 0 && rightVariants.length === 0) {
    return 0;
  }
  if (leftVariants.length === 0 || rightVariants.length === 0) {
    return 1;
  }
  let best = Infinity;
  for (const a of leftVariants) {
    for (const b of rightVariants) {
      best = Math.min(best, normalizedSetDistance(a.tokens, b.tokens));
    }
  }
  return round(best, 6);
}

function objectVariants(grid) {
  const cells = nonZeroCells(grid);
  if (cells.length === 0) {
    return [];
  }
  return D4_TRANSFORMS.map((transform) => {
    const transformed = cells.map((cell) => {
      const [x, y] = transform.fn(cell.x, cell.y);
      return { x, y, color: cell.color };
    });
    const minX = Math.min(...transformed.map((cell) => cell.x));
    const minY = Math.min(...transformed.map((cell) => cell.y));
    const normalized = transformed
      .map((cell) => ({ x: cell.x - minX, y: cell.y - minY, color: cell.color }))
      .sort((a, b) => a.y - b.y || a.x - b.x || a.color - b.color);
    const roleMap = new Map();
    let nextRole = 1;
    const tokens = normalized.map((cell) => {
      if (!roleMap.has(cell.color)) {
        roleMap.set(cell.color, nextRole);
        nextRole += 1;
      }
      return `${cell.x}:${cell.y}:${roleMap.get(cell.color)}`;
    });
    const width = Math.max(...normalized.map((cell) => cell.x)) + 1;
    const height = Math.max(...normalized.map((cell) => cell.y)) + 1;
    return {
      name: transform.name,
      tokens,
      signature: `${width}x${height}|${tokens.join(";")}`
    };
  });
}

function canonicalStencil(grid, cx, cy, radius) {
  const cells = [];
  for (let y = cy - radius; y <= cy + radius; y += 1) {
    const row = [];
    for (let x = cx - radius; x <= cx + radius; x += 1) {
      row.push(y < 0 || y >= grid.length || x < 0 || x >= grid[0].length ? 0 : grid[y][x]);
    }
    cells.push(row);
  }
  const variants = D4_STENCIL_TRANSFORMS.map((transform) => roleNormalizeGrid(transform(cells)));
  return variants.sort()[0];
}

function roleNormalizeGrid(grid) {
  const roleMap = new Map([[0, 0]]);
  let nextRole = 1;
  const tokens = [];
  for (const row of grid) {
    for (const value of row) {
      if (!roleMap.has(value)) {
        roleMap.set(value, nextRole);
        nextRole += 1;
      }
      tokens.push(roleMap.get(value));
    }
  }
  return tokens.join("");
}

function normalizedSetDistance(aTokens, bTokens) {
  const a = new Set(aTokens);
  const b = new Set(bTokens);
  const union = new Set([...a, ...b]);
  if (union.size === 0) {
    return 0;
  }
  let diff = 0;
  for (const token of union) {
    if (a.has(token) !== b.has(token)) {
      diff += 1;
    }
  }
  return diff / union.size;
}

function nonZeroCells(grid) {
  const cells = [];
  for (let y = 0; y < grid.length; y += 1) {
    for (let x = 0; x < grid[0].length; x += 1) {
      if (grid[y][x] !== 0) {
        cells.push({ x, y, color: grid[y][x] });
      }
    }
  }
  return cells;
}

function countComponents(grid) {
  const seen = Array.from({ length: grid.length }, () => Array(grid[0].length).fill(false));
  let count = 0;
  for (let y = 0; y < grid.length; y += 1) {
    for (let x = 0; x < grid[0].length; x += 1) {
      if (grid[y][x] === 0 || seen[y][x]) {
        continue;
      }
      count += 1;
      const stack = [[x, y]];
      seen[y][x] = true;
      while (stack.length > 0) {
        const [cx, cy] = stack.pop();
        for (const [nx, ny] of [[cx + 1, cy], [cx - 1, cy], [cx, cy + 1], [cx, cy - 1]]) {
          if (ny < 0 || ny >= grid.length || nx < 0 || nx >= grid[0].length || seen[ny][nx] || grid[ny][nx] === 0) {
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

function signalLabel(collisionResidual, meanPairResidual) {
  if (collisionResidual <= 0.15 && meanPairResidual <= 0.25) {
    return "compact";
  }
  if (collisionResidual > 0.5 || meanPairResidual > 0.6) {
    return "dispersed";
  }
  return "mixed";
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

function countBy(values) {
  const counts = {};
  for (const value of values) {
    counts[value] = (counts[value] ?? 0) + 1;
  }
  return counts;
}

function mean(values) {
  return values.reduce((sum, value) => sum + Number(value), 0) / values.length;
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
  if (rows.length === 0) {
    return "";
  }
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
    } else {
      fail(`Unknown argument: ${arg}`);
    }
  }
  return parsed;
}

function printUsage() {
  console.log(`Usage:
  node scripts/arc-phase2-projections.mjs --data-dir <ARC-AGI-2/data> --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase2-projections`);
}

function fail(message) {
  console.error(message);
  process.exit(1);
}
