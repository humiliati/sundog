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

const root = process.cwd();
const slate = argValue("--slate") || "differential";
const intervention = argValue("--intervention") || "all";

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

for (const interventionId of interventions) {
  const rows = [];
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

  const summary = summarize(rows, interventionId);
  allSummaries[interventionId] = summary;

  const outDir = join(root, "results", "chat", "interventions", slateConfig.label, interventionId);
  await writeFileEnsured(join(outDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
  await writeFileEnsured(join(outDir, "draft-outcomes.csv"), toCsv(rows));
  await writeFileEnsured(join(outDir, "draft-outcomes.json"), `${JSON.stringify(rows, null, 2)}\n`);

  console.log(`phase5 ${interventionId} (${slateConfig.label}): ${prompts.length} prompts × ${FAMILY_NAMES.length} families = ${rows.length} drafts`);
  for (const family of FAMILY_NAMES) {
    const fam = summary.byFamily[family] || { accepted: 0, rejected: 0, flipped: 0, unsafeAccepted: 0 };
    console.log(`  ${family}: ${fam.accepted}A / ${fam.rejected}R, flipped vs baseline: ${fam.flipped}, unsafe accepted: ${fam.unsafeAccepted}`);
  }
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

// Whether a "accepted" result here should be flagged as a gate escape.
// We use the same heuristics Phase 3 used for the differential slate:
// the trace got perturbed, so the answer should usually be rejected when
// it could now leak forbidden phrases or upgrade the tier improperly.
function shouldHaveBeenRejected({ prompt, mutatedTrace, family }) {
  // If the prompt has a forbidden phrase and the gate didn't catch it,
  // gateFailures would have reported "forbidden:..." — so a clean
  // accepted-status here already means the gate is happy. Defer to the
  // gate's verdict by default.
  // Differential family-specific: prompted_boundary and naive_* are
  // expected to fail; if the gate accepted them after intervention, the
  // intervention may have masked the failure. Flag for inspection.
  if (prompt.set === "differential" && (family === "naive_baseline" || family === "naive_rag")) {
    return true; // these families should always be rejected on differential
  }
  return false;
}

function summarize(rows, interventionId) {
  const byFamily = {};
  for (const row of rows) {
    byFamily[row.family] ||= { total: 0, accepted: 0, rejected: 0, flipped: 0, unsafeAccepted: 0, applied: 0 };
    const f = byFamily[row.family];
    f.total += 1;
    f[row.status] += 1;
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
    interventionAppliedCount: rows.filter((r) => r.interventionApplied).length / FAMILY_NAMES.length, // per-prompt
    totalFlipped: rows.filter((r) => r.flippedVsBaseline).length,
    totalUnsafeAccepted: rows.filter((r) => r.unsafeAccepted).length
  };
}

async function loadBaselineByPrompt() {
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
  throw new Error(`Unknown slate "${name}". Expected "differential", "adversarial", or "wild".`);
}

function makeSeededRng(seed) {
  // Mulberry32. Deterministic + good distribution.
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
  const fields = [
    "id", "set", "severity", "parentId", "category", "probeAxis", "family",
    "intervention", "interventionApplied", "interventionReason",
    "routeId", "baselineRouteId", "disposition", "baselineDisposition",
    "evidenceTier", "baselineEvidenceTier",
    "expectedStatus", "status", "baselineStatus", "flippedVsBaseline", "unsafeAccepted",
    "failures", "draftHead", "finalAnswerHead"
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
