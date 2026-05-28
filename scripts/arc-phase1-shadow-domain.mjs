import { mkdir, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";

const OUT_DIR = resolve("results/arc/phase1-shadow-domain-synthetic");

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

const fixtures = [
  {
    name: "translated_object",
    left: baseGrid(),
    right: translateIntoCanvas(baseGrid(), 8, 8, 3, 2),
    expectSame: true,
    expectZeroResidual: true
  },
  {
    name: "rotated_object",
    left: baseGrid(),
    right: rotate90(baseGrid()),
    expectSame: true,
    expectZeroResidual: true
  },
  {
    name: "reflected_object",
    left: baseGrid(),
    right: reflectHorizontal(baseGrid()),
    expectSame: true,
    expectZeroResidual: true
  },
  {
    name: "color_role_permutation",
    left: baseGrid(),
    right: mapColors(baseGrid(), new Map([[2, 7], [3, 4]])),
    expectSame: true,
    expectZeroResidual: true
  },
  {
    name: "shape_mismatch_negative",
    left: baseGrid(),
    right: negativeGrid(),
    expectSame: false,
    expectZeroResidual: false
  },
  {
    name: "single_cell_flip_negative",
    left: baseGrid(),
    right: addCell(baseGrid(), 4, 1, 2),
    expectSame: false,
    expectZeroResidual: false,
    expectBagMatch: false
  },
  {
    name: "color_collision_negative",
    left: baseGrid(),
    right: mapColors(baseGrid(), new Map([[2, 5], [3, 5]])),
    expectSame: false,
    expectZeroResidual: false,
    expectBagMatch: false
  },
  {
    name: "stencil_bag_translation_positive",
    left: baseGrid(),
    right: translateIntoCanvas(baseGrid(), 8, 8, 3, 2),
    expectSame: true,
    expectZeroResidual: true,
    expectBagMatch: true
  },
  {
    name: "stencil_bag_rotation_positive",
    left: baseGrid(),
    right: rotate90(baseGrid()),
    expectSame: true,
    expectZeroResidual: true,
    expectBagMatch: true
  }
];

const rows = fixtures.map((fixture) => {
  const left = projectGridShadow(fixture.left);
  const right = projectGridShadow(fixture.right);
  const sameSignature = left.canonicalObjectSignature === right.canonicalObjectSignature;
  const residual = alignmentResidual(fixture.left, fixture.right);
  const residualIsZero = residual === 0;
  const bagMatch = JSON.stringify(left.localSignatureBag) === JSON.stringify(right.localSignatureBag);
  const bagCheck = fixture.expectBagMatch === undefined ? true : bagMatch === fixture.expectBagMatch;
  const pass = sameSignature === fixture.expectSame
    && residualIsZero === fixture.expectZeroResidual
    && bagCheck;
  return {
    fixture: fixture.name,
    same_signature: sameSignature,
    alignment_residual: residual,
    bag_match: bagMatch,
    expected_same_signature: fixture.expectSame,
    expected_zero_residual: fixture.expectZeroResidual,
    expected_bag_match: fixture.expectBagMatch ?? "",
    pass
  };
});

const DISCRIMINATION_N = 50;
const DISCRIMINATION_SEED = 20260528;
const DISCRIMINATION_PALETTE = [1, 2, 3, 4, 5];
const rng = makeLcg(DISCRIMINATION_SEED);
const discriminationSignatures = new Map();
for (let i = 0; i < DISCRIMINATION_N; i += 1) {
  const grid = randomGrid(rng, 5, 5, DISCRIMINATION_PALETTE, 0.5);
  const sig = projectGridShadow(grid).canonicalObjectSignature;
  if (!discriminationSignatures.has(sig)) {
    discriminationSignatures.set(sig, []);
  }
  discriminationSignatures.get(sig).push(i);
}
const discriminationDistinct = discriminationSignatures.size;
const discriminationCollisions = [...discriminationSignatures.values()].filter((indices) => indices.length > 1);
const discriminationPass = discriminationDistinct === DISCRIMINATION_N;
const discriminationRow = {
  fixture: "discrimination_50_random_5x5",
  same_signature: `${discriminationDistinct}/${DISCRIMINATION_N} distinct`,
  alignment_residual: "",
  bag_match: "",
  expected_same_signature: `${DISCRIMINATION_N}/${DISCRIMINATION_N} distinct`,
  expected_zero_residual: "",
  expected_bag_match: "",
  pass: discriminationPass
};
rows.push(discriminationRow);

const manifest = {
  generatedAt: new Date().toISOString(),
  tool: "scripts/arc-phase1-shadow-domain.mjs",
  scope: "synthetic grids only; no registered ARC task scoring",
  operator: {
    version: "P_shadow_grid_v0",
    localRadius: 1,
    gauges: ["translation", "rotation", "reflection", "color_role_permutation"]
  },
  summary: {
    fixtures: rows.length,
    passed: rows.filter((row) => row.pass).length,
    failed: rows.filter((row) => !row.pass).length
  },
  discrimination: {
    N: DISCRIMINATION_N,
    seed: DISCRIMINATION_SEED,
    palette: DISCRIMINATION_PALETTE,
    distinctSignatures: discriminationDistinct,
    collisions: discriminationCollisions.map((indices) => ({ indices, count: indices.length })),
    pass: discriminationPass
  },
  rows
};

await mkdir(OUT_DIR, { recursive: true });
await writeFile(join(OUT_DIR, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
await writeFile(join(OUT_DIR, "summary.csv"), toCsv(rows));

for (const row of rows) {
  console.log(`${row.pass ? "PASS" : "FAIL"} ${row.fixture}: same=${row.same_signature} residual=${row.alignment_residual}`);
}

if (manifest.summary.failed > 0) {
  process.exit(1);
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

function baseGrid() {
  return [
    [0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0],
    [0, 2, 0, 0, 0],
    [0, 2, 3, 3, 0],
    [0, 0, 0, 0, 0]
  ];
}

function negativeGrid() {
  return [
    [0, 0, 0, 0, 0],
    [0, 2, 2, 0, 0],
    [0, 0, 2, 0, 0],
    [0, 3, 3, 3, 0],
    [0, 0, 0, 0, 0]
  ];
}

function translateIntoCanvas(grid, height, width, dx, dy) {
  const out = Array.from({ length: height }, () => Array(width).fill(0));
  for (let y = 0; y < grid.length; y += 1) {
    for (let x = 0; x < grid[0].length; x += 1) {
      if (grid[y][x] !== 0) {
        out[y + dy][x + dx] = grid[y][x];
      }
    }
  }
  return out;
}

function mapColors(grid, colorMap) {
  return grid.map((row) => row.map((value) => colorMap.get(value) ?? value));
}

function addCell(grid, x, y, color) {
  const out = grid.map((row) => [...row]);
  out[y][x] = color;
  return out;
}

function randomGrid(rng, height, width, palette, density) {
  return Array.from({ length: height }, () =>
    Array.from({ length: width }, () =>
      rng() < density ? palette[Math.floor(rng() * palette.length)] : 0
    )
  );
}

function makeLcg(seed) {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
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

function toCsv(rows) {
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

function round(value, places) {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
}
