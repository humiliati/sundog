// Phase 5b hosted-draft runner.
//
// For each prompt in the slate:
//   1. Build the trace via the static router + retrieval (same as Phase 3).
//   2. Hand the trace to the hosted adapter, which produces a draft answer.
//   3. Gate the draft via the existing claim gate.
//   4. Record outcome rows in the Phase 3 outcome shape, with one new
//      family — `sundog_gated_hosted` — joining the four deterministic
//      families.
//
// Usage:
//   node chat/eval/run_hosted_drafts.mjs --slate differential --backend mock
//   OPENAI_API_KEY=sk-... node chat/eval/run_hosted_drafts.mjs --slate differential --backend openai
//
// Optional flags:
//   --slate    differential | adversarial | wild | falsification | generality-boundary   (default: differential)
//   --backend  mock | openai                        (default: mock)
//   --limit    integer                              (truncate slate — useful for $ control)
//   --concurrency  integer                          (parallel requests, default 4)
//
// Output:
//   results/chat/phase5-hosted/<slate>/<backend>/draft-outcomes.{csv,json}
//   results/chat/phase5-hosted/<slate>/<backend>/summary.json
//
// The deterministic baseline lives at
//   results/chat/phase3-<slate>-draft-gate/draft-outcomes.json
// — the runner reads it to populate baselineStatus on every row, so the
// Phase 3 deterministic-vs-hosted comparison is one diff query away.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { buildTraceAnswer } from "../../public/js/sundog-chat-router.mjs";
import { attachRetrievedMatches, buildRetrievalTrace, searchChatIndex } from "../../public/js/sundog-retrieval.mjs";
import { gateModelDraft } from "../../public/js/sundog-claim-gate.mjs";
import { categoryFor } from "./lib/draft-families.mjs";
import { createOpenAIAdapter } from "./lib/adapters/openai-adapter.mjs";
import { createMockAdapter } from "./lib/adapters/mock-adapter.mjs";
import { createAnthropicAdapter } from "./lib/adapters/anthropic-adapter.mjs";
import { createGroqAdapter } from "./lib/adapters/groq-adapter.mjs";

const root = process.cwd();
const slate = argValue("--slate") || "differential";
const backend = argValue("--backend") || "mock";
const limit = Number(argValue("--limit") || "0") || 0;
const concurrency = Number(argValue("--concurrency") || "4") || 4;
const delayMs = Number(argValue("--delay-ms") || "0") || 0;
// --retrieval-k overrides the retrieval depth. Unset = use defaults (k≈3
// via buildRetrievalTrace, k≈2 via attachRetrievedMatches). 0 = empty
// retrieved list (router-only). N = re-query the index with limit=N.
const retrievalKArg = argValue("--retrieval-k");
const retrievalK = retrievalKArg === "" ? null : Number(retrievalKArg);

const slateConfig = configForSlate(slate);
const adapter = createAdapter(backend);

const claimMap = JSON.parse(await readFile(join(root, "chat", "claim_map.json"), "utf8"));
const chatIndex = JSON.parse(await readFile(join(root, "public", "data", "sundog-chat-index.json"), "utf8"));

const prompts = (await readFile(join(root, slateConfig.promptPath), "utf8"))
  .trim()
  .split(/\r?\n/)
  .filter(Boolean)
  .map((row) => JSON.parse(row));

const slatePrompts = limit > 0 ? prompts.slice(0, limit) : prompts;
const baselineByPrompt = await loadBaselineByPrompt();

console.log(`phase5-hosted: ${backend} backend, ${slateConfig.label} slate, ${slatePrompts.length} prompts (concurrency=${concurrency})`);
console.log(`adapter info: ${JSON.stringify(adapter.info)}`);

const startedAt = Date.now();
const rows = [];
let errored = 0;

// Concurrent fetch pool (cheap LLM calls benefit from parallelism, but we
// keep it bounded so we don't trip rate limits or burn budget on bugs).
const queue = slatePrompts.slice();
const workers = Array.from({ length: Math.min(concurrency, queue.length) }, () => workerLoop(queue, rows, (e) => { errored += 1; console.error(`  error: ${e.message}`); }));
await Promise.all(workers);

const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
const summary = summarize(rows, errored, elapsed);

const kSuffix = retrievalK === null || Number.isNaN(retrievalK) ? "" : `-k${retrievalK}`;
// For groq, suffix with model so llama and qwen runs don't collide.
const groqModelSuffix = backend === "groq" && adapter.info?.model
  ? "-" + adapter.info.model.replace(/[\/.]/g, "-")
  : "";
const outDir = join(root, "results", "chat", "phase5-hosted", slateConfig.label, `${backend}${groqModelSuffix}${kSuffix}`);
await writeFileEnsured(join(outDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
await writeFileEnsured(join(outDir, "draft-outcomes.csv"), toCsv(rows));
await writeFileEnsured(join(outDir, "draft-outcomes.json"), `${JSON.stringify(rows, null, 2)}\n`);

console.log(`\nDone in ${elapsed}s. ${rows.length} drafts, ${errored} errors.`);
console.log(`  hosted accepted: ${summary.accepted} / ${rows.length}`);
console.log(`  flipped vs deterministic baseline: ${summary.flippedVsBaseline}`);
console.log(`  gate escapes (unsafeAccepted-vs-baseline-rejected): ${summary.gateEscapesVsBaseline}`);
console.log(`Outputs under ${outDir.replace(root + "/", "")}/`);

// ---------------------------------------------------------------------

async function workerLoop(queue, sink, onError) {
  while (queue.length > 0) {
    const prompt = queue.shift();
    if (!prompt) break;
    try {
      const row = await scorePrompt(prompt);
      sink.push(row);
    } catch (err) {
      onError(err);
      sink.push(errorRow(prompt, err));
    }
    // Inter-call pacing for rate-limited backends. --delay-ms 5500 keeps
    // a single worker at ~11 calls/minute (under TPM caps for ~1K-token
    // heavy-trace payloads on Groq's Llama-3.3 tier).
    if (delayMs > 0 && queue.length > 0) {
      await new Promise((r) => setTimeout(r, delayMs));
    }
    continue;
  }
}

async function _workerLoopOLD_DEAD(queue, sink, onError) {
  // Dead branch — kept around so the closing brace below it still pairs
  // with the original workerLoop block during this transitional edit.
  return [queue, sink, onError];
}

async function scorePrompt(prompt) {
  const trace = traceFor(prompt.prompt);

  const draftText = await adapter.draft({
    prompt: prompt.prompt,
    trace,
    context: {
      family: "sundog_gated_hosted",
      category: categoryFor(prompt),
      expectedBehavior: prompt.expectedBehavior,
      forbidden: prompt.forbidden || [],
      probeAxis: prompt.probeAxis || "",
      slate: slateConfig.label
    }
  });

  const gated = gateModelDraft({
    prompt: prompt.prompt,
    trace,
    draft: { answer: draftText, source: `hosted:${backend}` },
    context: {
      family: "sundog_gated_hosted",
      category: categoryFor(prompt),
      expectedBehavior: prompt.expectedBehavior,
      forbidden: prompt.forbidden || [],
      probeAxis: prompt.probeAxis || ""
    }
  });

  const status = gated.draft.status;
  const baseline = baselineByPrompt.get(`${prompt.id}::sundog_gated`);
  const baselineStatus = baseline?.status || null;
  const baselineAnswerHead = baseline?.finalAnswerHead || "";

  return {
    id: prompt.id,
    set: prompt.set || "",
    severity: prompt.severity || "",
    parentId: prompt.parentId || "",
    category: categoryFor(prompt),
    probeAxis: prompt.probeAxis || "",
    family: "sundog_gated_hosted",
    backend,
    routeId: trace.routeId,
    disposition: trace.disposition,
    evidenceTier: trace.evidenceTier,
    expectedStatus: baselineStatus || "",
    status,
    baselineStatus: baselineStatus || "",
    flippedVsBaseline: baselineStatus ? status !== baselineStatus : false,
    unsafeAcceptedVsBaseline: baselineStatus === "rejected" && status === "accepted",
    failures: gated.draft.failures.join("|"),
    draftHead: draftText.slice(0, 200),
    draftFull: draftText,
    finalAnswerHead: gated.answer.slice(0, 200),
    baselineAnswerHead: baselineAnswerHead.slice(0, 200)
  };
}

function errorRow(prompt, err) {
  return {
    id: prompt.id,
    set: prompt.set || "",
    severity: prompt.severity || "",
    parentId: prompt.parentId || "",
    category: categoryFor(prompt),
    probeAxis: prompt.probeAxis || "",
    family: "sundog_gated_hosted",
    backend,
    routeId: "",
    disposition: "",
    evidenceTier: "",
    expectedStatus: "",
    status: "error",
    baselineStatus: "",
    flippedVsBaseline: false,
    unsafeAcceptedVsBaseline: false,
    failures: `adapter_error:${err.message.slice(0, 120)}`,
    draftHead: "",
    finalAnswerHead: "",
    baselineAnswerHead: ""
  };
}

function traceFor(promptText) {
  const staticTrace = buildTraceAnswer(claimMap, promptText);
  let trace;
  if (staticTrace.routeId === "unsupported_static_route") {
    trace = buildRetrievalTrace(chatIndex, promptText) || staticTrace;
  } else {
    trace = attachRetrievedMatches(chatIndex, promptText, staticTrace);
  }
  return applyRetrievalDepthOverride(trace, promptText);
}

function applyRetrievalDepthOverride(trace, promptText) {
  if (retrievalK === null || Number.isNaN(retrievalK)) return trace;
  if (retrievalK === 0) {
    // Router-only: drop all retrieved matches.
    return { ...trace, retrieved: [] };
  }
  // Re-query the index at the requested depth.
  const matches = searchChatIndex(chatIndex, promptText, { limit: retrievalK });
  return { ...trace, retrieved: matches };
}

function summarize(rows, errored, elapsed) {
  const accepted = rows.filter((r) => r.status === "accepted").length;
  const rejected = rows.filter((r) => r.status === "rejected").length;
  const flipped = rows.filter((r) => r.flippedVsBaseline).length;
  const gateEscapesVsBaseline = rows.filter((r) => r.unsafeAcceptedVsBaseline).length;
  const errorRows = rows.filter((r) => r.status === "error").length;
  return {
    backend,
    adapter: adapter.info,
    slate: slateConfig.label,
    promptCount: rows.length,
    elapsedSeconds: Number(elapsed),
    accepted,
    rejected,
    errored: errorRows || errored,
    flippedVsBaseline: flipped,
    gateEscapesVsBaseline,
    byCategory: groupBy(rows, "category"),
    byProbeAxis: groupBy(rows, "probeAxis")
  };
}

function groupBy(rows, field) {
  const out = {};
  for (const row of rows) {
    const k = row[field] || "(none)";
    out[k] ||= { total: 0, accepted: 0, rejected: 0, flipped: 0 };
    out[k].total += 1;
    out[k][row.status] = (out[k][row.status] || 0) + 1;
    if (row.flippedVsBaseline) out[k].flipped += 1;
  }
  return out;
}

async function loadBaselineByPrompt() {
  try {
    const baseline = JSON.parse(await readFile(join(root, slateConfig.baselinePath), "utf8"));
    const map = new Map();
    for (const row of baseline) {
      map.set(`${row.id}::${row.family}`, row);
    }
    return map;
  } catch (err) {
    console.warn(`Could not load Phase 3 baseline at ${slateConfig.baselinePath}: ${err.message}`);
    return new Map();
  }
}

function createAdapter(name) {
  if (name === "openai") return createOpenAIAdapter();
  if (name === "anthropic") return createAnthropicAdapter();
  if (name === "groq") return createGroqAdapter();
  if (name === "mock") return createMockAdapter();
  throw new Error(`Unknown backend "${name}". Expected "openai", "anthropic", "groq", or "mock".`);
}

function configForSlate(name) {
  if (name === "differential") {
    return {
      label: "differential",
      promptPath: join("chat", "prompts", "gold-differential.jsonl"),
      baselinePath: join("results", "chat", "phase3-differential-draft-gate", "draft-outcomes.json")
    };
  }
  if (name === "adversarial") {
    return {
      label: "adversarial",
      promptPath: join("chat", "prompts", "gold-adversarial.jsonl"),
      baselinePath: join("results", "chat", "phase3-adversarial-draft-gate", "draft-outcomes.json")
    };
  }
  if (name === "wild") {
    return {
      label: "wild",
      promptPath: join("chat", "prompts", "gold-wild.jsonl"),
      baselinePath: join("results", "chat", "phase3-draft-gate", "draft-outcomes.json")
    };
  }
  if (name === "falsification") {
    return {
      label: "falsification",
      promptPath: join("chat", "prompts", "gold-falsification.jsonl"),
      // No deterministic Phase 3 baseline for this slate — the falsification
      // slate is hand-authored and runs hosted-only. The runner tolerates a
      // missing baseline file (loadBaselineByPrompt logs a warning and
      // returns an empty map).
      baselinePath: join("results", "chat", "phase11-falsification", "deterministic-baseline-PLACEHOLDER.json")
    };
  }
  if (name === "generality-boundary") {
    return {
      label: "generality-boundary",
      promptPath: join("chat", "prompts", "gold-generality-boundary.jsonl"),
      baselinePath: join("results", "chat", "phase13-generality-boundary", "draft-outcomes.json")
    };
  }
  throw new Error(`Unknown slate "${name}".`);
}

function toCsv(rows) {
  const fields = [
    "id", "set", "severity", "parentId", "category", "probeAxis", "family", "backend",
    "routeId", "disposition", "evidenceTier",
    "expectedStatus", "status", "baselineStatus", "flippedVsBaseline", "unsafeAcceptedVsBaseline",
    "failures", "draftHead", "finalAnswerHead", "baselineAnswerHead"
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

function argValue(name) {
  const index = process.argv.indexOf(name);
  if (index < 0) return "";
  return process.argv[index + 1] || "";
}
