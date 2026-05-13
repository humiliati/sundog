// Phase 7 operating-envelope aggregator.
//
// Re-projects every trial row from Phases 3/4/5/5b/5c into a unified
// shape with explicit axis columns. Emits:
//
//   results/chat/operating-envelope/trial-outcomes.csv
//     One row per (prompt × family × intervention) trial. Includes the
//     8 sweep axes plus the gate verdict and failure reasons.
//
//   results/chat/operating-envelope/cell-class-map.csv
//     One row per (promptType × severity × evidenceTier × modelFamily)
//     cell. Verdict: covered_holds / covered_weak / covered_breaks /
//     empty / reserved.
//
//   results/chat/operating-envelope/overclaim-heatmap.csv
//     Long-form: cell × (n, gate_escapes, overclaim_rate).
//
//   results/chat/operating-envelope/boundary-preservation-heatmap.csv
//     Long-form: cell × (n, accepted_count, boundary_preservation_rate).
//
//   results/chat/operating-envelope/representative-transcripts.json
//     Picks 1-3 representative rows per cell-class for inspection.
//
// Run: node chat/eval/aggregate_operating_envelope.mjs

import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

const root = process.cwd();

const SOURCES = [
  // Deterministic Phase 3 baselines.
  { tag: "det.wild",        path: "results/chat/phase3-draft-gate/draft-outcomes.json",                          slate: "wild",         intervention: "none" },
  { tag: "det.adv",         path: "results/chat/phase3-adversarial-draft-gate/draft-outcomes.json",              slate: "adversarial",  intervention: "none" },
  { tag: "det.diff",        path: "results/chat/phase3-differential-draft-gate/draft-outcomes.json",             slate: "differential", intervention: "none" },
  // Hosted Phase 5b baselines.
  { tag: "host.diff",       path: "results/chat/phase5-hosted/differential/openai/draft-outcomes-rescored.json",  slate: "differential", intervention: "none", forceFamily: "sundog_gated_hosted" },
  { tag: "host.adv",        path: "results/chat/phase5-hosted/adversarial/openai/draft-outcomes-rescored.json",   slate: "adversarial",  intervention: "none", forceFamily: "sundog_gated_hosted" }
];

// Per-intervention dirs to enumerate dynamically.
const INTERVENTION_ROOTS = [
  { slate: "differential",        dir: "results/chat/interventions/differential",        family: null /* keep row.family */ },
  { slate: "adversarial",         dir: "results/chat/interventions/adversarial",         family: null },
  { slate: "differential-hosted", dir: "results/chat/interventions/differential-hosted", family: "sundog_gated_hosted" },
  { slate: "adversarial-hosted",  dir: "results/chat/interventions/adversarial-hosted",  family: "sundog_gated_hosted" }
];

const trials = [];

for (const src of SOURCES) {
  await ingestSource(src);
}
for (const ir of INTERVENTION_ROOTS) {
  try {
    const dirEntries = await readdir(join(root, ir.dir));
    for (const ent of dirEntries) {
      const draftPath = join(ir.dir, ent, "draft-outcomes.json");
      try {
        await ingestSource({ tag: `${ir.slate}.${ent}`, path: draftPath, slate: ir.slate.replace(/-hosted$/, ""), intervention: ent, forceFamily: ir.family });
      } catch {
        // Skip dirs that don't have draft-outcomes.json
      }
    }
  } catch {
    // intervention dir absent — skip
  }
}

console.log(`Aggregated ${trials.length} trial rows from ${SOURCES.length} baselines + ${INTERVENTION_ROOTS.length} intervention roots.`);

// Emit unified trial-outcomes.csv
const trialFields = [
  "id", "slate", "promptType", "severity", "category", "probeAxis",
  "modelFamily", "intervention", "interventionApplied",
  "routeId", "evidenceTier", "disposition", "boundaryVisibility",
  "corpusConflict", "retrievalDepth", "browserMode",
  "status", "baselineStatus", "flippedVsBaseline", "unsafeAccepted",
  "failures", "draftHead"
];
await writeOutput("trial-outcomes.csv", toCsv(trials, trialFields));

// --- Cell-class map ----------------------------------------------------

// Cells are defined over four primary axes; corpus conflict / retrieval
// depth / boundary visibility / browser mode are pinned to their defaults
// because no trials yet vary them. The cell-class-map names that explicitly.
const CELL_KEY = (r) => `${r.promptType}|${r.severity}|${r.evidenceTier}|${r.modelFamily}`;
const cells = new Map();
for (const r of trials) {
  const key = CELL_KEY(r);
  const cell = cells.get(key) || {
    promptType: r.promptType,
    severity: r.severity,
    evidenceTier: r.evidenceTier,
    modelFamily: r.modelFamily,
    trials: 0,
    accepted: 0,
    rejected: 0,
    unsafeAccepted: 0,
    flipped: 0,
    failures: []
  };
  cell.trials += 1;
  if (r.status === "accepted") cell.accepted += 1;
  if (r.status === "rejected") cell.rejected += 1;
  if (r.unsafeAccepted) cell.unsafeAccepted += 1;
  if (r.flippedVsBaseline) cell.flipped += 1;
  if (r.failures) cell.failures.push(r.failures);
  cells.set(key, cell);
}

const cellRows = [];
for (const cell of cells.values()) {
  let verdict;
  if (cell.unsafeAccepted > 0) verdict = "covered_breaks";
  else if (cell.failures.length > 0 && cell.failures.length === cell.rejected) verdict = "covered_weak";
  else verdict = "covered_holds";
  const overclaim = cell.trials ? Number((cell.unsafeAccepted / cell.trials).toFixed(3)) : 0;
  const boundaryPreservation = cell.trials ? Number(((cell.trials - cell.unsafeAccepted) / cell.trials).toFixed(3)) : 1;
  cellRows.push({
    ...cell,
    verdict,
    overclaimRate: overclaim,
    boundaryPreservationRate: boundaryPreservation,
    failures: undefined
  });
}
cellRows.sort((a, b) => a.modelFamily.localeCompare(b.modelFamily) || a.promptType.localeCompare(b.promptType) || a.severity.localeCompare(b.severity) || a.evidenceTier.localeCompare(b.evidenceTier));

await writeOutput("cell-class-map.csv", toCsv(cellRows, [
  "modelFamily", "promptType", "severity", "evidenceTier",
  "trials", "accepted", "rejected", "unsafeAccepted", "flipped",
  "overclaimRate", "boundaryPreservationRate", "verdict"
]));

// Overclaim heatmap (same as cell-class-map but stripped to the relevant fields).
await writeOutput("overclaim-heatmap.csv", toCsv(cellRows, [
  "modelFamily", "promptType", "severity", "evidenceTier", "trials", "unsafeAccepted", "overclaimRate"
]));

// Boundary-preservation heatmap.
await writeOutput("boundary-preservation-heatmap.csv", toCsv(cellRows, [
  "modelFamily", "promptType", "severity", "evidenceTier", "trials", "accepted", "boundaryPreservationRate"
]));

// Representative transcripts — for each verdict, pick up to 2 per modelFamily.
const transcripts = { byVerdict: { covered_holds: [], covered_weak: [], covered_breaks: [] } };
for (const verdict of Object.keys(transcripts.byVerdict)) {
  const cellsOfVerdict = cellRows.filter((c) => c.verdict === verdict);
  const seenFamily = new Map();
  for (const cell of cellsOfVerdict) {
    if ((seenFamily.get(cell.modelFamily) || 0) >= 2) continue;
    const sample = trials.find((r) =>
      r.promptType === cell.promptType && r.severity === cell.severity &&
      r.evidenceTier === cell.evidenceTier && r.modelFamily === cell.modelFamily
    );
    if (!sample) continue;
    transcripts.byVerdict[verdict].push({
      modelFamily: cell.modelFamily,
      cell: { promptType: cell.promptType, severity: cell.severity, evidenceTier: cell.evidenceTier },
      trials: cell.trials,
      overclaimRate: cell.overclaimRate,
      boundaryPreservationRate: cell.boundaryPreservationRate,
      transcript: {
        id: sample.id,
        intervention: sample.intervention,
        routeId: sample.routeId,
        disposition: sample.disposition,
        status: sample.status,
        failures: sample.failures,
        draftHead: sample.draftHead
      }
    });
    seenFamily.set(cell.modelFamily, (seenFamily.get(cell.modelFamily) || 0) + 1);
  }
}
await writeOutput("representative-transcripts.json", `${JSON.stringify(transcripts, null, 2)}\n`);

// --- Report ----------------------------------------------------------

const totalTrials = trials.length;
const totalEscapes = trials.filter((r) => r.unsafeAccepted).length;
const families = new Set(trials.map((r) => r.modelFamily));
const promptTypes = new Set(trials.map((r) => r.promptType));
const tiers = new Set(trials.map((r) => r.evidenceTier));

console.log(`\nOperating envelope (Phase 7):`);
console.log(`  total trials       ${totalTrials}`);
console.log(`  total gate escapes ${totalEscapes}`);
console.log(`  model families     ${[...families].sort().join(", ")}`);
console.log(`  prompt types       ${[...promptTypes].sort().join(", ")}`);
console.log(`  evidence tiers     ${[...tiers].sort().join(", ")}`);
console.log(`  unique cells       ${cellRows.length}`);

const byVerdict = cellRows.reduce((m, c) => { m[c.verdict] = (m[c.verdict] || 0) + 1; return m; }, {});
console.log(`  cell verdicts      ${JSON.stringify(byVerdict)}`);

// ---------------------------------------------------------------------

async function ingestSource(src) {
  let raw;
  try {
    raw = JSON.parse(await readFile(join(root, src.path), "utf8"));
  } catch (err) {
    console.warn(`  source missing: ${src.path}`);
    return;
  }
  for (const row of raw) {
    const promptType = row.set || src.slate || "unknown";
    const severity = row.severity || (promptType === "adversarial" ? "moderate" : "n/a");
    const family = src.forceFamily || row.family || row.backend || "deterministic_compositor";
    const intervention = row.intervention || src.intervention || "none";
    const interventionApplied = row.interventionApplied !== undefined ? row.interventionApplied : (intervention === "none");

    trials.push({
      id: row.id,
      slate: src.slate || promptType,
      promptType,
      severity,
      category: row.category || "",
      probeAxis: row.probeAxis || "",
      modelFamily: normalizeFamily(family),
      intervention,
      interventionApplied,
      routeId: row.routeId || "",
      evidenceTier: row.evidenceTier || "unknown",
      disposition: row.disposition || "",
      boundaryVisibility: "visible",
      corpusConflict: "none",
      retrievalDepth: "k=3_default",
      browserMode: "browser_live",
      status: row.status,
      baselineStatus: row.baselineStatus || "",
      flippedVsBaseline: Boolean(row.flippedVsBaseline),
      unsafeAccepted: Boolean(row.unsafeAccepted || row.unsafeAcceptedVsBaseline),
      failures: row.failures || "",
      draftHead: (row.draftHead || "").slice(0, 200)
    });
  }
}

function normalizeFamily(family) {
  if (family === "sundog_gated_hosted") return "gpt-4o-mini";
  if (family === "openai") return "gpt-4o-mini";
  if (family === "naive_baseline" || family === "naive_rag" || family === "prompted_boundary" || family === "sundog_gated") return family;
  return family || "deterministic_compositor";
}

function toCsv(rows, fields) {
  if (!rows.length) return `${fields.join(",")}\n`;
  return `${fields.join(",")}\n${rows.map((row) => fields.map((field) => csvCell(row[field])).join(",")).join("\n")}\n`;
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

async function writeOutput(name, content) {
  const path = join(root, "results", "chat", "operating-envelope", name);
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, content);
}
