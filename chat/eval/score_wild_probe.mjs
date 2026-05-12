// Wild-probe harness: run gold-wild.jsonl (out-of-distribution prompts) through
// the same static-router + retrieval-fallback pipeline that score_static_router.mjs
// uses. Reports where each wild prompt actually lands. No scoring — the goal is
// to surface failure modes, not grade against gold.
//
// Run with: node chat/eval/score_wild_probe.mjs
// Or wire as: "chat:eval:wild": "node chat/eval/score_wild_probe.mjs"

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { buildTraceAnswer } from "../../public/js/sundog-chat-router.mjs";
import { attachRetrievedMatches, buildRetrievalTrace } from "../../public/js/sundog-retrieval.mjs";

const root = process.cwd();
const promptPath = join(root, "chat", "prompts", "gold-wild.jsonl");
const outDir = join(root, "results", "chat", "phase2-wild-probe");

const claimMap = JSON.parse(await readFile(join(root, "chat", "claim_map.json"), "utf8"));
const chatIndex = JSON.parse(await readFile(join(root, "public", "data", "sundog-chat-index.json"), "utf8"));

const rows = (await readFile(promptPath, "utf8")).trim().split(/\r?\n/).filter(Boolean);
const prompts = rows.map((row) => JSON.parse(row));

const outcomes = prompts.map(scorePrompt);
const summary = summarize(outcomes);

await writeFileEnsured(join(outDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
await writeFileEnsured(join(outDir, "wild-outcomes.csv"), toCsv(outcomes));
await writeFileEnsured(join(outDir, "wild-outcomes.json"), `${JSON.stringify(outcomes, null, 2)}\n`);

console.log(`wild probe: ${outcomes.length} prompts`);
console.log(`  unique routes hit: ${Object.keys(summary.byRoute).length}`);
console.log(`  dispositions: ${JSON.stringify(summary.byDisposition)}`);
console.log(`  unsupported_static_route: ${summary.byRoute.unsupported_static_route || 0}`);
console.log(`  retrieval-only fallback fired: ${summary.byDisposition.retrieval_only || 0}`);

function scorePrompt(prompt) {
  const trace = traceFor(prompt.prompt);
  return {
    id: prompt.id,
    category: prompt.category,
    prompt: prompt.prompt,
    expectedBehavior: prompt.expectedBehavior || "",
    notes: prompt.notes || "",
    routeId: trace.routeId,
    disposition: trace.disposition,
    evidenceTier: trace.evidenceTier,
    retrievedCount: trace.retrieved?.length || 0,
    topRetrievedRoute: trace.retrieved?.[0]?.routeId || "",
    topRetrievedDoc: trace.retrieved?.[0]?.doc || "",
    answerHead: (trace.answer || "").slice(0, 120)
  };
}

function traceFor(prompt) {
  const staticTrace = buildTraceAnswer(claimMap, prompt);
  if (staticTrace.routeId === "unsupported_static_route") {
    return buildRetrievalTrace(chatIndex, prompt) || staticTrace;
  }
  return attachRetrievedMatches(chatIndex, prompt, staticTrace);
}

function summarize(rows) {
  const byRoute = {};
  const byDisposition = {};
  const byCategory = {};
  for (const r of rows) {
    byRoute[r.routeId] = (byRoute[r.routeId] || 0) + 1;
    byDisposition[r.disposition] = (byDisposition[r.disposition] || 0) + 1;
    byCategory[r.category] = byCategory[r.category] || {};
    byCategory[r.category][r.routeId] = (byCategory[r.category][r.routeId] || 0) + 1;
  }
  return { total: rows.length, byRoute, byDisposition, byCategory };
}

function toCsv(rows) {
  const fields = [
    "id", "category", "prompt", "routeId", "disposition", "evidenceTier",
    "retrievedCount", "topRetrievedRoute", "topRetrievedDoc",
    "expectedBehavior", "answerHead"
  ];
  return `${fields.join(",")}\n${rows.map((row) => fields.map((field) => csvCell(row[field])).join(",")).join("\n")}\n`;
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

async function writeFileEnsured(path, value) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, value);
}
