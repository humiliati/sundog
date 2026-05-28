// Phase 0 task register DRAFT helper.
//
// This is NOT a Phase 0 gate. It is a first-pass selector that reads the
// frozen inventory and produces a candidate P0_TASK_REGISTER.csv for human
// review. The binding register is whatever the maintainer commits after
// inspecting these candidates.
//
// Selection rule (priority order; each task lands in the FIRST matching bucket):
//   1. symmetry          -- any pair grid has reflect_h, reflect_v, or rot180.
//   2. spatial_transform -- train_shape_changes >= 1.
//   3. counting          -- max_nonzero_components >= 8.
//   4. local_completion  -- min_nonzero_density <= 0.15.
//   5. color_role        -- color_count >= 5.
//   6. objectness        -- residual.
//
// Within each bucket, tasks are sorted by task_id and the first --per-bucket
// (default 6) are picked. Default total: 36 (matches the spec target).
//
// Predicted boundaries are first-pass guesses from metadata signals only.
// They MUST be refined per task on inspection before the register is committed.
//
// Run:
//   node scripts/arc-phase0-draft-register.mjs \
//     --inventory results/arc/phase0-inventory/tasks.csv \
//     --out docs/prereg/arc/P0_TASK_REGISTER.draft.csv

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

const args = parseArgs(process.argv.slice(2));
if (args.help || !args.inventory || !args.out) {
  printUsage();
  process.exit(args.help ? 0 : 2);
}

const perBucket = Number(args.perBucket ?? 6);
if (!Number.isInteger(perBucket) || perBucket <= 0) {
  fail(`--per-bucket must be a positive integer (got ${args.perBucket}).`);
}

const inventoryPath = resolve(args.inventory);
const outPath = resolve(args.out);
const inventoryCsv = await readFile(inventoryPath, "utf8");
const rows = parseCsv(inventoryCsv);
const trainingRows = rows.filter((row) => row.split === "training");

const BUCKET_ORDER = [
  "symmetry",
  "spatial_transform",
  "counting",
  "local_completion",
  "color_role",
  "objectness"
];

const PRED_BOUNDARY = {
  symmetry: "gauge-breaking ambiguity (symmetry as gauge)",
  spatial_transform: "capacity pressure (shape change carries structural info)",
  counting: "capacity pressure (count is high-entropy)",
  local_completion: "non-local information (global context required to fill)",
  color_role: "gauge-breaking ambiguity (color permutation as gauge)",
  objectness: "full-state-only dependency (residual category)"
};

const buckets = Object.fromEntries(BUCKET_ORDER.map((name) => [name, []]));

for (const row of trainingRows) {
  const bucket = assignBucket(row);
  buckets[bucket].push(row);
}

for (const name of BUCKET_ORDER) {
  buckets[name].sort((a, b) => a.task_id.localeCompare(b.task_id));
}

const selected = [];
const bucketCounts = {};
for (const name of BUCKET_ORDER) {
  const picks = buckets[name].slice(0, perBucket);
  bucketCounts[name] = { available: buckets[name].length, picked: picks.length };
  for (const row of picks) {
    selected.push({ ...row, primary_prior: name });
  }
}

selected.sort((a, b) => a.task_id.localeCompare(b.task_id));

const outRows = selected.map((row) => ({
  task_id: row.task_id,
  split: "training",
  status: "include",
  primary_prior: row.primary_prior,
  secondary_priors: secondaryPriors(row).join(";"),
  inclusion_basis: inclusionBasis(row),
  exclusion_reason: "",
  predicted_boundary: PRED_BOUNDARY[row.primary_prior],
  inventory_row_hash: row.inventory_row_hash,
  manual_inspection: "no",
  notes: "DRAFT auto-selected by arc-phase0-draft-register.mjs; refine after inspecting grids."
}));

const header = [
  "task_id",
  "split",
  "status",
  "primary_prior",
  "secondary_priors",
  "inclusion_basis",
  "exclusion_reason",
  "predicted_boundary",
  "inventory_row_hash",
  "manual_inspection",
  "notes"
];

const csv = [header.join(",")]
  .concat(outRows.map((row) => header.map((col) => csvCell(row[col])).join(",")))
  .join("\n") + "\n";

await mkdir(dirname(outPath), { recursive: true });
await writeFile(outPath, csv);

console.log(`Wrote ${outRows.length} draft row(s) to ${outPath}`);
console.log("Per-bucket availability (training split, n=" + trainingRows.length + "):");
for (const name of BUCKET_ORDER) {
  const c = bucketCounts[name];
  console.log(`  ${name.padEnd(18)} available=${String(c.available).padStart(4)} picked=${c.picked}`);
}

function assignBucket(row) {
  if (row.symmetry_hints) {
    return "symmetry";
  }
  if (Number(row.train_shape_changes) >= 1) {
    return "spatial_transform";
  }
  if (Number(row.max_nonzero_components) >= 8) {
    return "counting";
  }
  if (Number(row.min_nonzero_density) <= 0.15) {
    return "local_completion";
  }
  if (Number(row.color_count) >= 5) {
    return "color_role";
  }
  return "objectness";
}

function secondaryPriors(row) {
  const hints = (row.prior_hints || "").split(";").filter(Boolean);
  return hints.filter((hint) => hint !== row.primary_prior);
}

function inclusionBasis(row) {
  const facts = [];
  facts.push(`grids ${row.min_height}x${row.min_width}..${row.max_height}x${row.max_width}`);
  facts.push(`${row.color_count} colors`);
  facts.push(`up to ${row.max_nonzero_components} components`);
  if (row.symmetry_hints) {
    facts.push(`symmetries: ${row.symmetry_hints}`);
  }
  if (Number(row.train_shape_changes) >= 1) {
    facts.push(`${row.train_shape_changes} train shape-change(s)`);
  }
  facts.push(`density ${row.min_nonzero_density}..${row.max_nonzero_density}`);
  return facts.join("; ");
}

function parseCsv(text) {
  const lines = text.replace(/\r\n/g, "\n").split("\n").filter((line) => line.length > 0);
  const headerCols = parseCsvLine(lines[0]);
  const out = [];
  for (let i = 1; i < lines.length; i += 1) {
    const cols = parseCsvLine(lines[i]);
    const row = {};
    headerCols.forEach((col, j) => { row[col] = cols[j] ?? ""; });
    out.push(row);
  }
  return out;
}

function parseCsvLine(line) {
  const out = [];
  let cell = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === "\"" && line[i + 1] === "\"") { cell += "\""; i += 1; }
      else if (ch === "\"") { inQuotes = false; }
      else { cell += ch; }
    } else if (ch === "\"") { inQuotes = true; }
    else if (ch === ",") { out.push(cell); cell = ""; }
    else { cell += ch; }
  }
  out.push(cell);
  return out;
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
    if (arg === "--help" || arg === "-h") { parsed.help = true; }
    else if (arg === "--inventory") { parsed.inventory = argv[++i]; }
    else if (arg === "--out") { parsed.out = argv[++i]; }
    else if (arg === "--per-bucket") { parsed.perBucket = argv[++i]; }
    else { fail(`Unknown argument: ${arg}`); }
  }
  return parsed;
}

function printUsage() {
  console.log(`Usage:
  node scripts/arc-phase0-draft-register.mjs --inventory <tasks.csv> --out <draft.csv> [--per-bucket N]`);
}

function fail(message) {
  console.error(message);
  process.exit(1);
}
