// Phase 4 probe-slate operating-envelope rollup.
//
// This script aggregates the Phase 3 draft-gate outputs into the Phase 4
// reporting shape. Hosted adapters can later append rows with the same schema.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

const root = process.cwd();
const outDir = join(root, "results", "chat", "probe-slate");
const inputs = [
  {
    slate: "wild",
    promptPath: "chat/prompts/gold-wild.jsonl",
    outcomePath: "results/chat/phase3-draft-gate/draft-outcomes.json",
    summaryPath: "results/chat/phase3-draft-gate/summary.json"
  },
  {
    slate: "adversarial",
    promptPath: "chat/prompts/gold-adversarial.jsonl",
    outcomePath: "results/chat/phase3-adversarial-draft-gate/draft-outcomes.json",
    summaryPath: "results/chat/phase3-adversarial-draft-gate/summary.json"
  },
  {
    slate: "differential",
    promptPath: "chat/prompts/gold-differential.jsonl",
    outcomePath: "results/chat/phase3-differential-draft-gate/draft-outcomes.json",
    summaryPath: "results/chat/phase3-differential-draft-gate/summary.json"
  }
];

const suites = [];
const rows = [];

for (const input of inputs) {
  const outcomes = JSON.parse(await readFile(join(root, input.outcomePath), "utf8"));
  const summary = JSON.parse(await readFile(join(root, input.summaryPath), "utf8"));
  const promptCount = countJsonl(await readFile(join(root, input.promptPath), "utf8"));

  suites.push({
    slate: input.slate,
    promptPath: input.promptPath,
    outcomePath: input.outcomePath,
    summaryPath: input.summaryPath,
    promptCount,
    draftCount: outcomes.length,
    gateEscapes: summary.gateEscapes,
    gateHitRate: summary.gateHitRate,
    addedValue: summary.addedValue
  });

  rows.push(...outcomes.map((row) => ({
    ...row,
    slate: row.set || input.slate,
    sourceOutcomePath: input.outcomePath
  })));
}

const boundaryRows = summarizeBy(rows, ["slate", "family"]);
const overclaimRows = boundaryRows.map((row) => ({
  slate: row.slate,
  family: row.family,
  expectedRejected: row.expectedRejected,
  unsafeAccepted: row.unsafeAccepted,
  rejectedUnsafe: row.rejectedUnsafe,
  overclaimEscapeRate: row.expectedRejected === 0
    ? 0
    : Number((row.unsafeAccepted / row.expectedRejected).toFixed(3))
}));
const axisRows = summarizeBy(rows, ["slate", "category", "family"]);
const transcripts = representativeRows(rows);

const manifest = {
  version: "phase4-probe-slate-v0",
  status: "deterministic_baseline",
  generatedAt: new Date().toISOString(),
  purpose: "Phase 4 operating-envelope baseline over deterministic Ask Sundog draft families before hosted adapters are added.",
  suites,
  totals: {
    prompts: suites.reduce((sum, suite) => sum + suite.promptCount, 0),
    drafts: rows.length,
    gateEscapes: rows.filter((row) => row.unsafeAccepted).length,
    addedValue: rows.filter((row) => row.addedValue).length
  },
  scope: [
    "Deterministic scaffold only; no hosted model adapter is included.",
    "The differential slate is authored to expose route-specific trace failure modes.",
    "Hosted-model parity with the deterministic S1 compositor remains open."
  ],
  outputs: {
    trialOutcomes: "results/chat/probe-slate/trial-outcomes.csv",
    boundaryPreservation: "results/chat/probe-slate/boundary-preservation.csv",
    overclaimRate: "results/chat/probe-slate/overclaim-rate.csv",
    axisBreakdown: "results/chat/probe-slate/axis-breakdown.csv",
    representativeTranscripts: "results/chat/probe-slate/representative-transcripts.json"
  }
};

await writeFileEnsured(join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
await writeFileEnsured(join(outDir, "trial-outcomes.csv"), toCsv(rows, [
  "slate",
  "id",
  "category",
  "probeAxis",
  "family",
  "routeId",
  "disposition",
  "evidenceTier",
  "expectedStatus",
  "status",
  "gateHit",
  "unsafeAccepted",
  "divergence",
  "addedValue",
  "failures",
  "draftHead",
  "finalAnswerHead"
]));
await writeFileEnsured(join(outDir, "boundary-preservation.csv"), toCsv(boundaryRows, summaryFields()));
await writeFileEnsured(join(outDir, "overclaim-rate.csv"), toCsv(overclaimRows, [
  "slate",
  "family",
  "expectedRejected",
  "unsafeAccepted",
  "rejectedUnsafe",
  "overclaimEscapeRate"
]));
await writeFileEnsured(join(outDir, "axis-breakdown.csv"), toCsv(axisRows, [
  "slate",
  "category",
  "family",
  ...summaryFields().filter((field) => field !== "slate" && field !== "family")
]));
await writeFileEnsured(join(outDir, "representative-transcripts.json"), `${JSON.stringify(transcripts, null, 2)}\n`);

console.log(`phase4 probe slate: ${manifest.totals.prompts} prompts, ${manifest.totals.drafts} drafts`);
console.log(`  gate escapes: ${manifest.totals.gateEscapes}`);
console.log(`  addedValue drafts: ${manifest.totals.addedValue}`);

function summarizeBy(items, keys) {
  const groups = new Map();
  for (const row of items) {
    const key = keys.map((field) => row[field] || "").join("\u0000");
    if (!groups.has(key)) {
      groups.set(key, Object.fromEntries(keys.map((field) => [field, row[field] || ""])));
    }

    const group = groups.get(key);
    group.total ||= 0;
    group.accepted ||= 0;
    group.rejected ||= 0;
    group.expectedRejected ||= 0;
    group.unsafeAccepted ||= 0;
    group.rejectedUnsafe ||= 0;
    group.gateHits ||= 0;
    group.addedValue ||= 0;
    group.identical ||= 0;
    group.extends ||= 0;
    group.rewrites ||= 0;
    group.divergenceRejected ||= 0;

    group.total += 1;
    group[row.status] += 1;
    if (row.expectedStatus === "rejected") group.expectedRejected += 1;
    if (row.unsafeAccepted) group.unsafeAccepted += 1;
    if (row.expectedStatus === "rejected" && row.status === "rejected") group.rejectedUnsafe += 1;
    if (row.gateHit) group.gateHits += 1;
    if (row.addedValue) group.addedValue += 1;
    if (row.divergence === "rejected") group.divergenceRejected += 1;
    else group[row.divergence] += 1;
  }

  return [...groups.values()].map((group) => ({
    ...group,
    gateHitRate: rate(group.gateHits, group.total),
    gateEscapeRate: rate(group.unsafeAccepted, group.total),
    addedValueRate: rate(group.addedValue, group.total)
  }));
}

function representativeRows(items) {
  const picked = [];
  const wanted = [
    (row) => row.slate === "differential" && row.family === "sundog_gated" && row.addedValue,
    (row) => row.slate === "differential" && row.family === "prompted_boundary" && row.status === "rejected",
    (row) => row.slate === "adversarial" && row.family === "naive_rag" && row.status === "rejected",
    (row) => row.slate === "wild" && row.family === "sundog_gated" && row.addedValue
  ];

  for (const predicate of wanted) {
    const match = items.find((row) => predicate(row));
    if (match) picked.push(selectTranscriptFields(match));
  }

  return picked;
}

function selectTranscriptFields(row) {
  return {
    slate: row.slate,
    id: row.id,
    category: row.category,
    family: row.family,
    routeId: row.routeId,
    expectedStatus: row.expectedStatus,
    status: row.status,
    divergence: row.divergence,
    addedValue: row.addedValue,
    failures: row.failures,
    draftHead: row.draftHead,
    finalAnswerHead: row.finalAnswerHead
  };
}

function summaryFields() {
  return [
    "slate",
    "family",
    "total",
    "accepted",
    "rejected",
    "expectedRejected",
    "unsafeAccepted",
    "rejectedUnsafe",
    "gateHits",
    "gateHitRate",
    "gateEscapeRate",
    "addedValue",
    "addedValueRate",
    "identical",
    "extends",
    "rewrites",
    "divergenceRejected"
  ];
}

function countJsonl(text) {
  return text.split(/\r?\n/).filter(Boolean).length;
}

function rate(numerator, denominator) {
  if (!denominator) return 0;
  return Number((numerator / denominator).toFixed(3));
}

function toCsv(items, fields) {
  return `${fields.join(",")}\n${items.map((item) => fields.map((field) => csvCell(item[field])).join(",")).join("\n")}\n`;
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

async function writeFileEnsured(path, value) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, value);
}
