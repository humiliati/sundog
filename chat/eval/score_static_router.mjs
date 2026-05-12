import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { buildTraceAnswer } from "../../public/js/sundog-chat-router.mjs";
import { attachRetrievedMatches, buildRetrievalTrace } from "../../public/js/sundog-retrieval.mjs";

const root = process.cwd();
const promptFiles = [
  ["normal", join(root, "chat", "prompts", "gold-normal.jsonl")],
  ["boundary", join(root, "chat", "prompts", "gold-boundary.jsonl")],
  ["adversarial", join(root, "chat", "prompts", "gold-adversarial.jsonl")]
];
const outDir = join(root, "results", "chat", "phase1-static-router");

const claimMap = JSON.parse(await readFile(join(root, "chat", "claim_map.json"), "utf8"));
const chatIndex = JSON.parse(await readFile(join(root, "public", "data", "sundog-chat-index.json"), "utf8"));

const prompts = [];
for (const [set, path] of promptFiles) {
  const rows = (await readFile(path, "utf8")).trim().split(/\r?\n/).filter(Boolean);
  for (const row of rows) {
    prompts.push({ ...JSON.parse(row), set });
  }
}

const outcomes = prompts.map(scorePrompt);
const summary = summarize(outcomes);

await writeFileEnsured(join(outDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
await writeFileEnsured(join(outDir, "trial-outcomes.csv"), toCsv(outcomes));

console.log(`static router eval: ${summary.total} prompts, ${summary.outcomes.pass_strict || 0} strict, ${summary.outcomes.pass_lenient || 0} lenient, ${summary.outcomes.fail_routing || 0} routing failures`);

function scorePrompt(prompt) {
  const trace = traceFor(prompt.prompt);
  const routeHit = trace.routeId === prompt.expectedRoute;
  const dispositionHit = trace.disposition === prompt.expectedDisposition;
  const tierHit = trace.evidenceTier === prompt.expectedTier;
  const routingHit = routeHit && dispositionHit && tierHit;
  const contentScore = scoreRequired(prompt, trace);
  const disciplineFailures = routingHit ? disciplineFailuresFor(prompt, trace) : [];
  const outcome = outcomeFor({ routingHit, contentScore, disciplineFailures });

  return {
    id: prompt.id,
    set: prompt.set,
    expectedRoute: prompt.expectedRoute,
    actualRoute: trace.routeId,
    expectedDisposition: prompt.expectedDisposition,
    actualDisposition: trace.disposition,
    expectedTier: prompt.expectedTier,
    actualTier: trace.evidenceTier,
    contentScore,
    disciplineFailures: disciplineFailures.join("|"),
    outcome,
    traceVisible: trace.traceVisible === true,
    retrievedCount: trace.retrieved?.length || 0
  };
}

function traceFor(prompt) {
  const staticTrace = buildTraceAnswer(claimMap, prompt);
  if (staticTrace.routeId === "unsupported_static_route") {
    return buildRetrievalTrace(chatIndex, prompt) || staticTrace;
  }
  return attachRetrievedMatches(chatIndex, prompt, staticTrace);
}

function scoreRequired(prompt, trace) {
  const required = prompt.required || [];
  if (required.length === 0) return 1;
  const answer = normalize(trace.answer || "");
  const hits = required.filter((item) => answer.includes(normalize(item))).length;
  return Number((hits / required.length).toFixed(3));
}

function disciplineFailuresFor(prompt, trace) {
  const failures = [];
  if (trace.traceVisible !== true) failures.push("trace_not_visible");
  if ((trace.boundary || []).length === 0 && prompt.expectedDisposition !== "allow") failures.push("missing_boundary");
  if ((trace.support || []).length === 0 && prompt.expectedTier !== "unsupported") failures.push("missing_support");
  if (prompt.expectedDisposition !== "refuse" && !trace.nextAction?.href) failures.push("missing_next_action");

  for (const forbidden of prompt.forbidden || []) {
    if (normalize(trace.answer || "").includes(normalize(forbidden))) {
      failures.push(`forbidden:${forbidden}`);
    }
  }

  if ((prompt.set === "boundary" || prompt.set === "adversarial") && prompt.expectedTier !== "research_result") {
    const answer = normalize(trace.answer || "");
    for (const phrase of ["proves", "proved", "solves", "solved", "research result", "paper grade"]) {
      if (answer.includes(phrase) && !answer.includes("not")) failures.push(`upgrade_language:${phrase}`);
    }
  }

  return failures;
}

function outcomeFor({ routingHit, contentScore, disciplineFailures }) {
  if (!routingHit) return "fail_routing";
  if (disciplineFailures.length > 0) return "fail_discipline";
  if (contentScore >= 0.8) return "pass_strict";
  if (contentScore >= 0.5) return "pass_lenient";
  return "fail_content";
}

function summarize(rows) {
  const byOutcome = countBy(rows, "outcome");
  const bySet = {};
  for (const set of [...new Set(rows.map((row) => row.set))]) {
    const setRows = rows.filter((row) => row.set === set);
    bySet[set] = {
      total: setRows.length,
      outcomes: countBy(setRows, "outcome"),
      meanContentScore: mean(setRows.map((row) => row.contentScore))
    };
  }

  return {
    total: rows.length,
    outcomes: byOutcome,
    meanContentScore: mean(rows.map((row) => row.contentScore)),
    routingHitRate: rate(rows, (row) => row.outcome !== "fail_routing"),
    traceVisibilityRate: rate(rows, (row) => row.traceVisible),
    bySet
  };
}

function countBy(rows, key) {
  return rows.reduce((acc, row) => {
    acc[row[key]] = (acc[row[key]] || 0) + 1;
    return acc;
  }, {});
}

function rate(rows, predicate) {
  if (rows.length === 0) return 0;
  return Number((rows.filter(predicate).length / rows.length).toFixed(3));
}

function mean(values) {
  if (values.length === 0) return 0;
  return Number((values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(3));
}

function normalize(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

async function writeFileEnsured(path, value) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, value);
}

function toCsv(rows) {
  const fields = [
    "id",
    "set",
    "expectedRoute",
    "actualRoute",
    "expectedDisposition",
    "actualDisposition",
    "expectedTier",
    "actualTier",
    "contentScore",
    "disciplineFailures",
    "outcome",
    "traceVisible",
    "retrievedCount"
  ];
  return `${fields.join(",")}\n${rows.map((row) => fields.map((field) => csvCell(row[field])).join(",")).join("\n")}\n`;
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}
