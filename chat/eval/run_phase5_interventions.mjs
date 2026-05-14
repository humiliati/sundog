// Phase 5 causal-intervention runner.
//
// For each (intervention, prompt, family) tuple:
//   1. Build the unmutated trace via the static router + retrieval.
//   2. Apply the intervention mutator to produce a perturbed trace.
//   3. Generate the four family drafts over the perturbed trace.
//   4. Gate each draft.
//   5. Compute Phase-5-specific outcome metrics:
//        - gateHit (same as Phase 3)
//        - unsafeAccepted (gate escape)
//        - flippedVsBaseline (did the intervention change the gate verdict?)
//        - rejectionRate / acceptanceRate (rolled up later by aggregator)
//
// Output:
//   results/chat/interventions/<intervention_id>/draft-outcomes.{csv,json}
//   results/chat/interventions/<intervention_id>/summary.json
//
// Run with:
//   node chat/eval/run_phase5_interventions.mjs --slate differential
//   node chat/eval/run_phase5_interventions.mjs --slate differential --intervention boundary_removed
//
// `--slate` defaults to "differential". `--intervention` defaults to "all",
// which iterates every entry in manifest.plannedInterventions.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { buildTraceAnswer } from "../../public/js/sundog-chat-router.mjs";
import { attachRetrievedMatches, buildRetrievalTrace } from "../../public/js/sundog-retrieval.mjs";
import { gateModelDraft } from "../../public/js/sundog-claim-gate.mjs";
import { FAMILY_DRAFTERS, FAMILY_NAMES, categoryFor } from "./lib/draft-families.mjs";
import { applyIntervention, INTERVENTION_IDS } from "./lib/interventions.mjs";
import { createOpenAIAdapter } from "./lib/adapters/openai-adapter.mjs";
import { createAnthropicAdapter } from "./lib/adapters/anthropic-adapter.mjs";
import { createGroqAdapter } from "./lib/adapters/groq-adapter.mjs";

const root = process.cwd();
const slate = argValue("--slate") || "differential";
const intervention = argValue("--intervention") || "all";
const hostedMode = argv().includes("--hosted");
const hostedBackend = argValue("--backend") || "openai"; // when --hosted, defaults to openai
const hostedConcurrency = Number(argValue("--concurrency") || "4") || 4;
const delayMs = Number(argValue("--delay-ms") || "0") || 0;

const slateConfig = configForSlate(slate);
const interventions = intervention === "all" ? INTERVENTION_IDS : [intervention];

const claimMap = JSON.parse(await readFile(join(root, "chat", "claim_map.json"), "utf8"));
const chatIndex = JSON.parse(await readFile(join(root, "public", "data", "sundog-chat-index.json"), "utf8"));
const chunkById = new Map((chatIndex.chunks || []).map((chunk) => [chunk.id, chunk]));
const allRoutes = [...(claimMap.claims || []), ...(claimMap.nonClaimRoutes || [])];
const routeById = new Map(allRoutes.map((route) => [route.id, route]));

const prompts = (await readFile(join(root, slateConfig.promptPath), "utf8"))
  .trim()
  .split(/\r?\n/)
  .filter(Boolean)
  .map((row) => JSON.parse(row));

const baselineByPrompt = await loadBaselineByPrompt();
const draftCtx = { chunkById, routeById };

// Seeded RNG so swap mutations are deterministic per run.
const rng = makeSeededRng(0xc0ffee);
const interventionCtx = { claimMap, chunkById, routeById, allRoutes, rng };

const allSummaries = {};
// For groq, suffix the slate label and baseline path with the model so
// llama-3.3 / llama-3.1 / qwen runs don't collide on disk.
const groqModel = process.env.GROQ_MODEL || "";
const groqModelSuffix = (hostedBackend === "groq" && groqModel)
  ? "-" + groqModel.replace(/[\/.]/g, "-")
  : "";
const outSlateLabel = hostedMode
  ? `${slateConfig.label}-hosted${hostedBackend === "openai" ? "" : "-" + hostedBackend + groqModelSuffix}`
  : slateConfig.label;
const hostedAdapter = hostedMode
  ? (hostedBackend === "anthropic" ? createAnthropicAdapter()
     : hostedBackend === "groq" ? createGroqAdapter()
     : createOpenAIAdapter())
  : null;

// Load the hosted unmutated-baseline rows (rescored against patched gate).
// Used to compare each post-intervention hosted result against the
// reference no-intervention hosted outcome. Empty map in deterministic mode.
async function loadHostedBaselineMap() {
  // For groq, baselines live at phase5-hosted/<slate>/groq-<model>/
  const hostedBaselineDir = hostedBackend === "groq"
    ? `${hostedBackend}${groqModelSuffix}`
    : hostedBackend;
  // The baseline may be either draft-outcomes-rescored.json (preferred,
  // post-patch) or just draft-outcomes.json. Try the rescored first.
  for (const fname of ["draft-outcomes-rescored.json", "draft-outcomes.json"]) {
    const hostedPath = join(root, "results", "chat", "phase5-hosted", slateConfig.label, hostedBaselineDir, fname);
    try {
      const rows = JSON.parse(await readFile(hostedPath, "utf8"));
      return new Map(rows.map((r) => [r.id, r]));
    } catch {
      // try next
    }
  }
  const hostedPath = join(root, "results", "chat", "phase5-hosted", slateConfig.label, hostedBaselineDir, "draft-outcomes-rescored.json");
  try {
    const rows = JSON.parse(await readFile(hostedPath, "utf8"));
    return new Map(rows.map((r) => [r.id, r]));
  } catch (err) {
    throw new Error(`Hosted baseline not found at ${hostedPath} — run run_hosted_drafts.mjs first and re-score against the patched gate. (${err.message})`);
  }
}
const hostedBaselineByPromptId = hostedMode ? await loadHostedBaselineMap() : new Map();

if (hostedMode) {
  console.log(`Hosted mode: ${JSON.stringify(hostedAdapter.info)}`);
  console.log(`Output dir: results/chat/interventions/${outSlateLabel}/<intervention_id>/`);
  console.log(`Hosted baseline: ${hostedBaselineByPromptId.size} prompt rows`);
}

for (const interventionId of interventions) {
  const rows = [];

  if (hostedMode) {
    // Hosted-only path. Build a queue of (prompt, mutatedTrace) pairs and
    // a concurrency-bounded pool to call the OpenAI adapter on each.
    // Skipped interventions (where meta.applied === false) get a synthetic
    // row that copies the hosted unmutated baseline outcome — no API call.
    const queue = prompts.map((prompt) => {
      const baselineTrace = traceFor(prompt.prompt);
      const { trace: mutatedTrace, meta } = applyIntervention(baselineTrace, interventionId, interventionCtx);
      return { prompt, baselineTrace, mutatedTrace, meta };
    });

    let cursor = 0;
    async function worker() {
      while (cursor < queue.length) {
        const job = queue[cursor++];
        const row = await runHostedRow({ job, interventionId });
        rows.push(row);
        // Inter-call pacing for rate-limited backends. With concurrency=1
        // and --delay-ms 11000, this keeps a single worker under Groq's
        // free-tier TPM caps. Skip the delay on the last job and on
        // skipped (no-API-call) rows.
        if (delayMs > 0 && cursor < queue.length && !row.skipped) {
          await new Promise((r) => setTimeout(r, delayMs));
        }
      }
    }
    await Promise.all(Array.from({ length: Math.min(hostedConcurrency, queue.length) }, worker));

  } else {
    // Deterministic path (original 4-family sweep).
    for (const prompt of prompts) {
      const baselineTrace = traceFor(prompt.prompt);
      const { trace: mutatedTrace, meta } = applyIntervention(baselineTrace, interventionId, interventionCtx);

      for (const family of FAMILY_NAMES) {
        const drafter = FAMILY_DRAFTERS[family];
        const draft = drafter(prompt, mutatedTrace, draftCtx);
        rows.push(buildOutcomeRow({
          family,
          prompt,
          baselineTrace,
          mutatedTrace,
          draft,
          interventionId,
          interventionMeta: meta
        }));
      }
    }
  }

  const summary = summarize(rows, interventionId);
  allSummaries[interventionId] = summary;

  const outDir = join(root, "results", "chat", "interventions", outSlateLabel, interventionId);
  await writeFileEnsured(join(outDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
  await writeFileEnsured(join(outDir, "draft-outcomes.csv"), toCsv(rows));
  await writeFileEnsured(join(outDir, "draft-outcomes.json"), `${JSON.stringify(rows, null, 2)}\n`);

  if (hostedMode) {
    const fam = summary.byFamily["sundog_gated_hosted"] || { accepted: 0, rejected: 0, flipped: 0, unsafeAccepted: 0, applied: 0 };
    const hostedCalls = rows.filter((r) => r.interventionApplied).length;
    console.log(`phase5c hosted ${interventionId} (${slateConfig.label}): ${prompts.length} prompts, ${hostedCalls} API calls, ${rows.length - hostedCalls} skipped (not-applied)`);
    console.log(`  sundog_gated_hosted: ${fam.accepted}A / ${fam.rejected}R, flipped vs baseline: ${fam.flipped}, unsafe accepted: ${fam.unsafeAccepted}`);
  } else {
    console.log(`phase5 ${interventionId} (${slateConfig.label}): ${prompts.length} prompts × ${FAMILY_NAMES.length} families = ${rows.length} drafts`);
    for (const family of FAMILY_NAMES) {
      const fam = summary.byFamily[family] || { accepted: 0, rejected: 0, flipped: 0, unsafeAccepted: 0 };
      console.log(`  ${family}: ${fam.accepted}A / ${fam.rejected}R, flipped vs baseline: ${fam.flipped}, unsafe accepted: ${fam.unsafeAccepted}`);
    }
  }
}

async function runHostedRow({ job, interventionId }) {
  const { prompt, baselineTrace, mutatedTrace, meta } = job;
  const hostedBaseline = hostedBaselineByPromptId.get(prompt.id);

  // If the intervention did not apply (mutation was a no-op), copy the
  // hosted unmutated baseline outcome verbatim — no API call needed.
  if (!meta.applied) {
    return buildHostedOutcomeRow({
      prompt,
      baselineTrace,
      mutatedTrace,
      draft: "",
      gateResult: null,
      interventionId,
      interventionMeta: meta,
      hostedBaseline,
      skipped: true
    });
  }

  let draft = "";
  let gateResult = null;
  let adapterError = null;
  try {
    draft = await hostedAdapter.draft({
      prompt: prompt.prompt,
      trace: mutatedTrace,
      context: {
        family: "sundog_gated_hosted",
        category: categoryFor(prompt),
        expectedBehavior: prompt.expectedBehavior,
        forbidden: prompt.forbidden || [],
        probeAxis: prompt.probeAxis || "",
        slate: slateConfig.label,
        intervention: interventionId
      }
    });
    gateResult = gateModelDraft({
      prompt: prompt.prompt,
      trace: mutatedTrace,
      draft: { answer: draft, source: "openai" },
      context: {
        family: "sundog_gated_hosted",
        category: categoryFor(prompt),
        expectedBehavior: prompt.expectedBehavior,
        forbidden: prompt.forbidden || [],
        probeAxis: prompt.probeAxis || ""
      }
    });
  } catch (err) {
    adapterError = err.message.slice(0, 160);
  }

  return buildHostedOutcomeRow({
    prompt,
    baselineTrace,
    mutatedTrace,
    draft,
    gateResult,
    interventionId,
    interventionMeta: meta,
    hostedBaseline,
    skipped: false,
    adapterError
  });
}

function buildHostedOutcomeRow({ prompt, baselineTrace, mutatedTrace, draft, gateResult, interventionId, interventionMeta, hostedBaseline, skipped, adapterError }) {
  // When skipped: status = hostedBaseline.status (no change), flipped = false.
  // When real call: compare gate verdict against hostedBaseline.status.
  const baselineStatus = hostedBaseline?.status || null;
  let status, failures, finalAnswerHead, draftHead;
  if (skipped) {
    status = baselineStatus || "accepted";
    failures = hostedBaseline?.failures || "";
    draftHead = hostedBaseline?.draftHead || "";
    finalAnswerHead = hostedBaseline?.finalAnswerHead || "";
  } else if (adapterError) {
    status = "error";
    failures = `adapter_error:${adapterError}`;
    draftHead = "";
    finalAnswerHead = "";
  } else {
    status = gateResult.draft.status;
    failures = gateResult.draft.failures.join("|");
    draftHead = draft.slice(0, 200);
    finalAnswerHead = gateResult.answer.slice(0, 200);
  }

  const flippedVsBaseline = baselineStatus && !skipped && status !== baselineStatus;
  const unsafeAccepted = status === "accepted" && baselineStatus === "rejected";

  return {
    id: prompt.id,
    set: prompt.set || "",
    severity: prompt.severity || "",
    parentId: prompt.parentId || "",
    category: categoryFor(prompt),
    probeAxis: prompt.probeAxis || "",
    family: "sundog_gated_hosted",
    intervention: interventionId,
    interventionApplied: interventionMeta.applied,
    interventionReason: interventionMeta.reason || "",
    routeId: mutatedTrace.routeId,
    baselineRouteId: baselineTrace.routeId,
    disposition: mutatedTrace.disposition,
    baselineDisposition: baselineTrace.disposition,
    evidenceTier: mutatedTrace.evidenceTier,
    baselineEvidenceTier: baselineTrace.evidenceTier,
    expectedStatus: baselineStatus || "",
    status,
    baselineStatus: baselineStatus || "",
    flippedVsBaseline: Boolean(flippedVsBaseline),
    unsafeAccepted,
    skipped: Boolean(skipped),
    failures,
    draftHead,
    finalAnswerHead
  };
}

// Cross-intervention top-level summary so the aggregator has a single
// pointer to read.
const overall = {
  slate: slateConfig.label,
  promptCount: prompts.length,
  familyCount: FAMILY_NAMES.length,
  interventionCount: interventions.length,
  baselineSource: slateConfig.baselinePath,
  byIntervention: allSummaries,
  generatedAt: new Date().toISOString()
};
await writeFileEnsured(
  join(root, "results", "chat", "interventions", `slate-${slateConfig.label}-summary.json`),
  `${JSON.stringify(overall, null, 2)}\n`
);

console.log(`\nWrote ${interventions.length} intervention outcome dirs under results/chat/interventions/.`);

// ---------------------------------------------------------------------

function traceFor(promptText) {
  const staticTrace = buildTraceAnswer(claimMap, promptText);
  if (staticTrace.routeId === "unsupported_static_route") {
    return buildRetrievalTrace(chatIndex, promptText) || staticTrace;
  }
  return attachRetrievedMatches(chatIndex, promptText, staticTrace);
}

function buildOutcomeRow({ family, prompt, baselineTrace, mutatedTrace, draft, interventionId, interventionMeta }) {
  const baseline = baselineByPrompt.get(`${prompt.id}::${family}`);
  const gated = gateModelDraft({
    prompt: prompt.prompt,
    trace: mutatedTrace,
    draft: { answer: draft, source: family },
    context: {
      family,
      category: categoryFor(prompt),
      expectedBehavior: prompt.expectedBehavior,
      forbidden: prompt.forbidden || [],
      probeAxis: prompt.probeAxis || ""
    }
  });

  const status = gated.draft.status;
  const baselineStatus = baseline?.status || null;
  const flippedVsBaseline = baselineStatus && status !== baselineStatus;
  const unsafeAccepted = status === "accepted" && shouldHaveBeenRejected({ prompt, mutatedTrace, family });

  return {
    id: prompt.id,
    set: prompt.set || "",
    severity: prompt.severity || "",
    parentId: prompt.parentId || "",
    category: categoryFor(prompt),
    probeAxis: prompt.probeAxis || "",
    family,
    intervention: interventionId,
    interventionApplied: interventionMeta.applied,
    interventionReason: interventionMeta.reason || "",
    routeId: mutatedTrace.routeId,
    baselineRouteId: baselineTrace.routeId,
    disposition: mutatedTrace.disposition,
    baselineDisposition: baselineTrace.disposition,
    evidenceTier: mutatedTrace.evidenceTier,
    baselineEvidenceTier: baselineTrace.evidenceTier,
    expectedStatus: baselineStatus || "",
    status,
    baselineStatus: baselineStatus || "",
    flippedVsBaseline: Boolean(flippedVsBaseline),
    unsafeAccepted,
    failures: gated.draft.failures.join("|"),
    draftHead: draft.slice(0, 160),
    finalAnswerHead: gated.answer.slice(0, 160)
  };
}

function shouldHaveBeenRejected({ prompt, mutatedTrace, family }) {
  if (prompt.set === "differential" && (family === "naive_baseline" || family === "naive_rag")) {
    return true;
  }
  return false;
}

function summarize(rows, interventionId) {
  const byFamily = {};
  for (const row of rows) {
    byFamily[row.family] ||= { total: 0, accepted: 0, rejected: 0, error: 0, flipped: 0, unsafeAccepted: 0, applied: 0 };
    const f = byFamily[row.family];
    f.total += 1;
    f[row.status] = (f[row.status] || 0) + 1;
    if (row.flippedVsBaseline) f.flipped += 1;
    if (row.unsafeAccepted) f.unsafeAccepted += 1;
    if (row.interventionApplied) f.applied += 1;
  }
  return {
    interventionId,
    slate: slateConfig.label,
    promptCount: prompts.length,
    totalDrafts: rows.length,
    byFamily,
    totalFlipped: rows.filter((r) => r.flippedVsBaseline).length,
    totalUnsafeAccepted: rows.filter((r) => r.unsafeAccepted).length,
    mode: hostedMode ? "hosted" : "deterministic"
  };
}

async function loadBaselineByPrompt() {
  // In hosted mode, the baseline is the hosted unmutated rescored outcomes.
  // In deterministic mode, the baseline is the Phase 3 draft-outcomes.
  if (hostedMode) {
    const hostedPath = join(root, "results", "chat", "phase5-hosted", slateConfig.label, "openai", "draft-outcomes-rescored.json");
    try {
      const rows = JSON.parse(await readFile(hostedPath, "utf8"));
      const map = new Map();
      for (const row of rows) map.set(`${row.id}::sundog_gated_hosted`, row);
      return map;
    } catch (err) {
      throw new Error(`Hosted baseline not found at ${hostedPath} — run run_hosted_drafts.mjs --slate ${slateConfig.label} --backend openai first. (${err.message})`);
    }
  }
  const baseline = JSON.parse(await readFile(join(root, slateConfig.baselinePath), "utf8"));
  const map = new Map();
  for (const row of baseline) {
    map.set(`${row.id}::${row.family}`, row);
  }
  return map;
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
  throw new Error(`Unknown slate "${name}".`);
}

function makeSeededRng(seed) {
  let state = seed >>> 0;
  return function rng() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function toCsv(rows) {
  if (rows.length === 0) return "\n";
  const fields = Object.keys(rows[0]);
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

function argv() {
  return process.argv.slice(2);
}

function argValue(name) {
  const index = process.argv.indexOf(name);
  if (index < 0) return "";
  return process.argv[index + 1] || "";
}
