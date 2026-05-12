import { readFile, writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { buildTraceAnswer } from "./public/js/sundog-chat-router.mjs";
import { attachRetrievedMatches, buildRetrievalTrace } from "./public/js/sundog-retrieval.mjs";

const root = process.cwd();
const claimMap = JSON.parse(await readFile(join(root, "chat", "claim_map.json"), "utf8"));
const chatIndex = JSON.parse(await readFile(join(root, "public", "data", "sundog-chat-index.json"), "utf8"));

function traceFor(prompt) {
  const staticTrace = buildTraceAnswer(claimMap, prompt);
  if (staticTrace.routeId === "unsupported_static_route") {
    return buildRetrievalTrace(chatIndex, prompt) || staticTrace;
  }
  return attachRetrievedMatches(chatIndex, prompt, staticTrace);
}

const rows = (await readFile(join(root, "chat", "prompts", "gold-wild.jsonl"), "utf8")).trim().split(/\r?\n/).filter(Boolean);
const outcomes = [];
for (const row of rows) {
  const p = JSON.parse(row);
  const trace = traceFor(p.prompt);
  outcomes.push({
    id: p.id, category: p.category, prompt: p.prompt,
    expectedBehavior: p.expectedBehavior,
    routeId: trace.routeId,
    disposition: trace.disposition,
    evidenceTier: trace.evidenceTier,
    retrievedCount: trace.retrieved?.length || 0,
    topRetrieved: trace.retrieved?.[0]?.routeId || ""
  });
}

const outDir = join(root, "results", "chat", "phase2-wild-probe");
await mkdir(outDir, { recursive: true });
await writeFile(join(outDir, "wild-outcomes.json"), JSON.stringify(outcomes, null, 2) + "\n");

const fields = ["id","category","prompt","routeId","disposition","evidenceTier","retrievedCount","topRetrieved","expectedBehavior"];
const csv = fields.join(",") + "\n" + outcomes.map(o => fields.map(f => {
  const v = String(o[f] ?? "");
  return /[",\n]/.test(v) ? `"${v.replaceAll('"','""')}"` : v;
}).join(",")).join("\n") + "\n";
await writeFile(join(outDir, "wild-outcomes.csv"), csv);

const byRoute = {};
const byDispo = {};
const byCat = {};
for (const o of outcomes) {
  byRoute[o.routeId] = (byRoute[o.routeId] || 0) + 1;
  byDispo[o.disposition] = (byDispo[o.disposition] || 0) + 1;
  byCat[o.category] = byCat[o.category] || {};
  byCat[o.category][o.disposition || "(none)"] = (byCat[o.category][o.disposition || "(none)"] || 0) + 1;
}
console.log("total:", outcomes.length);
console.log("\nby routeId:");
for (const [k, v] of Object.entries(byRoute).sort((a,b) => b[1]-a[1])) console.log(`  ${v}  ${k}`);
console.log("\nby disposition:");
for (const [k, v] of Object.entries(byDispo).sort((a,b) => b[1]-a[1])) console.log(`  ${v}  ${k}`);
console.log("\nby category:");
for (const [cat, dispCounts] of Object.entries(byCat)) {
  console.log(`  ${cat}:`, JSON.stringify(dispCounts));
}
