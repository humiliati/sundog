// Phase 0 context-expansion candidate ordering for Phase 3E fibers.
//
// Frozen by PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md. This is NOT a gate and
// NOT a selector of final tasks. It emits the deterministic, pre-inspection
// candidate queue (per registered prior) so that manual grid inspection happens
// in a fixed order that cannot be post-hoc nearest-neighbor hunted.
//
// Discipline (spec ss"Selection Discipline"):
//   - reads ONLY the input-derived Phase 0 inventory + the existing register;
//   - NEVER computes a Phase 3E context distance, sketch, target-output hash,
//     output-collision group, or any solver result;
//   - reads ONLY the public-training split; never the held-out split.
//
// Candidate ordering (spec ss"Candidate Ordering"):
//   1. start from the public-training inventory;
//   2. drop the already-registered tasks;
//   3. drop invalid tasks and tasks with < 2 train pairs (and < 1 test query);
//   4. partition remaining tasks into the six prior queues by `prior_hints`
//      membership (a task may enter multiple queues);
//   5. within each prior queue, sort by:
//        a. descending matching_hint_count (count of the task's prior_hints
//           tokens -- the documented reading of "number of matching coarse
//           prior hints"; see the freeze-marker amendment),
//        b. ascending abs(max_area - that prior's current median max_area over
//           the originally registered tasks of that prior),
//        c. ascending inventory_row_hash,
//        d. ascending task_id;
//   6. emit the full ordered queue + a sha256 of the queue file.
//
// Run:
//   node scripts/arc-phase0-context-expansion-for-fibers.mjs \
//     --inventory results/arc/phase0-inventory/tasks.csv \
//     --register docs/prereg/arc/P0_TASK_REGISTER.csv \
//     --out results/arc/phase0-context-expansion-for-fibers

import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";

const PRIORS = [
  "objectness",
  "counting",
  "symmetry",
  "spatial_transform",
  "local_completion",
  "color_role"
];

const PER_PRIOR_TARGET = 18;
const MIN_TRAIN_PAIRS = 2;
const MIN_TEST_QUERIES = 1;
const EXPANSION_BATCH = "fiber_context_expansion_v1";

const args = parseArgs(process.argv.slice(2));
if (args.help) {
  printUsage();
  process.exit(0);
}

const inventoryPath = resolve(args.inventory ?? "results/arc/phase0-inventory/tasks.csv");
const registerPath = resolve(args.register ?? "docs/prereg/arc/P0_TASK_REGISTER.csv");
const outDir = resolve(args.out ?? "results/arc/phase0-context-expansion-for-fibers");

const inventoryRows = parseCsv(await readFile(inventoryPath, "utf8"));
const registerRows = parseCsv(await readFile(registerPath, "utf8"));

const registeredIds = new Set(registerRows.map((r) => r.task_id));

// Per-prior median max_area over the originally registered tasks of that prior.
const invByTaskId = new Map(inventoryRows.map((r) => [r.task_id, r]));
const priorMedianArea = {};
for (const prior of PRIORS) {
  const areas = registerRows
    .filter((r) => r.status === "include" && r.primary_prior === prior)
    .map((r) => Number(invByTaskId.get(r.task_id)?.max_area))
    .filter((v) => Number.isFinite(v));
  priorMedianArea[prior] = median(areas);
}

// Eligible candidate pool: training split, not registered, >= 2 train pairs,
// >= 1 test query.
const eligible = inventoryRows.filter((r) =>
  r.split === "training" &&
  !registeredIds.has(r.task_id) &&
  Number(r.train_pairs) >= MIN_TRAIN_PAIRS &&
  Number(r.test_pairs) >= MIN_TEST_QUERIES
);

// Partition into prior queues by prior_hints membership and sort.
const queues = {};
for (const prior of PRIORS) {
  const inQueue = eligible.filter((r) => hintTokens(r).includes(prior));
  const decorated = inQueue.map((r) => {
    const maxArea = Number(r.max_area);
    return {
      task_id: r.task_id,
      matching_hint_count: hintTokens(r).length,
      max_area: maxArea,
      abs_area_diff_from_prior_median: Math.abs(maxArea - priorMedianArea[prior]),
      inventory_row_hash: r.inventory_row_hash,
      prior_hints: r.prior_hints ?? ""
    };
  });
  decorated.sort((a, b) =>
    (b.matching_hint_count - a.matching_hint_count) ||
    (a.abs_area_diff_from_prior_median - b.abs_area_diff_from_prior_median) ||
    a.inventory_row_hash.localeCompare(b.inventory_row_hash) ||
    a.task_id.localeCompare(b.task_id)
  );
  queues[prior] = decorated;
}

const header = [
  "prior",
  "selection_order_rank",
  "task_id",
  "matching_hint_count",
  "max_area",
  "abs_area_diff_from_prior_median",
  "inventory_row_hash",
  "prior_hints"
];

const outRows = [];
for (const prior of PRIORS) {
  queues[prior].forEach((row, idx) => {
    outRows.push({
      prior,
      selection_order_rank: idx + 1,
      task_id: row.task_id,
      matching_hint_count: row.matching_hint_count,
      max_area: row.max_area,
      abs_area_diff_from_prior_median: row.abs_area_diff_from_prior_median,
      inventory_row_hash: row.inventory_row_hash,
      prior_hints: row.prior_hints
    });
  });
}

const csv = [header.join(",")]
  .concat(outRows.map((row) => header.map((col) => csvCell(row[col])).join(",")))
  .join("\n") + "\n";

const queueSha = createHash("sha256").update(csv, "utf8").digest("hex");

await mkdir(outDir, { recursive: true });
await writeFile(join(outDir, "candidate_queue.csv"), csv);
await writeFile(join(outDir, "candidate_queue.sha256"), `${queueSha}  candidate_queue.csv\n`);

console.log(`Wrote ${outRows.length} queue row(s) across ${PRIORS.length} prior(s) to ${join(outDir, "candidate_queue.csv")}`);
console.log(`candidate_queue.sha256: ${queueSha}`);
console.log(`expansion_batch: ${EXPANSION_BATCH}`);
console.log("");
console.log(`Eligible pool: ${eligible.length} task(s) (training, not registered, >=${MIN_TRAIN_PAIRS} train pairs, >=${MIN_TEST_QUERIES} test query).`);
console.log(`Per-prior queue depth (target ${PER_PRIOR_TARGET} includes; queue lists ALL eligible candidates):`);
let shortPriors = 0;
for (const prior of PRIORS) {
  const depth = queues[prior].length;
  const flag = depth >= PER_PRIOR_TARGET ? "OK" : "*** SHORT ***";
  if (depth < PER_PRIOR_TARGET) shortPriors += 1;
  console.log(`  ${prior.padEnd(18)} depth=${String(depth).padStart(4)} medianArea=${String(priorMedianArea[prior]).padStart(6)}  ${flag}`);
}
console.log("");
if (shortPriors > 0) {
  console.log(`PRE-FLIGHT: ${shortPriors} prior(s) have fewer than ${PER_PRIOR_TARGET} eligible candidates.`);
  console.log("Per spec, do NOT rebalance across priors; file phase0_fiber_expansion_hold_insufficient_tasks.");
} else {
  console.log(`PRE-FLIGHT OK: every prior has >= ${PER_PRIOR_TARGET} eligible candidates.`);
}

function hintTokens(row) {
  return (row.prior_hints || "").split(";").filter(Boolean);
}

function median(values) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function parseCsv(text) {
  const lines = text.replace(/\r\n/g, "\n").split("\n").filter((line) => line.length > 0);
  if (lines.length === 0) return [];
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
    else if (arg === "--register") { parsed.register = argv[++i]; }
    else if (arg === "--out") { parsed.out = argv[++i]; }
    else { fail(`Unknown argument: ${arg}`); }
  }
  return parsed;
}

function printUsage() {
  console.log(`Usage:
  node scripts/arc-phase0-context-expansion-for-fibers.mjs \\
    --inventory <tasks.csv> --register <P0_TASK_REGISTER.csv> --out <dir>`);
}

function fail(message) {
  console.error(message);
  process.exit(1);
}
