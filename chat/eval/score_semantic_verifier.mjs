// Semantic-verifier bake-off harness (Phase 13 §13).
//
// Measures a layer-2 semantic verifier's contribution ON TOP OF the lexical gate,
// over the off-slate set, with `context.forbidden` disabled (general machinery
// only). Mechanism-agnostic: pass any verifier module that default-exports the
// { name, meta, verify } interface; defaults to the null baseline (control).
//
// Usage:
//   node chat/eval/score_semantic_verifier.mjs                       # null baseline
//   node chat/eval/score_semantic_verifier.mjs ./path/to/verifier.mjs
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { gateFailures } from "../../public/js/sundog-claim-gate.mjs";
import { nullVerifier, wilsonUpper } from "./lib/semantic-verifier.mjs";

const root = process.cwd();
const slatePath = "chat/prompts/offslate-valence-paraphrase.jsonl";
const claimMapPath = "chat/claim_map.json";
const outDir = "results/chat/phase13-semantic-verifier";

const verifierArg = process.argv[2];
let verifier = nullVerifier;
if (verifierArg) {
  const mod = await import(pathToFileURL(verifierArg).href);
  verifier = mod.default || mod.verifier;
  if (!verifier || typeof verifier.verify !== "function") {
    throw new Error(`${verifierArg} must default-export a verifier with a verify() method.`);
  }
}

const claimMap = JSON.parse(readFileSync(claimMapPath, "utf8"));
const routeById = new Map(
  [...(claimMap.claims || []), ...(claimMap.nonClaimRoutes || [])].map((r) => [r.id, r])
);
const rows = readFileSync(slatePath, "utf8").split(/\r?\n/).filter(Boolean).map((l) => JSON.parse(l));

const scored = [];
for (const row of rows) {
  const route = routeById.get(row.route);
  const trace = {
    routeId: route.id,
    disposition: route.disposition || "allow",
    evidenceTier: route.evidenceTier || "unknown",
    failureMode: route.failureMode || null,
    traceVisible: true
  };
  // Layer 1: lexical gate, forbidden DISABLED (general machinery only).
  const lexReject = gateFailures({ prompt: "", trace, draftAnswer: row.draft, context: {} }).length > 0;
  // Layer 2: the plugged semantic verifier (reject-only; may be async).
  const v = await verifier.verify({ draft: row.draft, route });
  const semReject = v.reject === true;
  scored.push({ ...row, lexReject, semReject, stackReject: lexReject || semReject, semReason: v.reason || null });
}

const oc = scored.filter((r) => r.kind === "overclaim");
const honest = scored.filter((r) => r.kind === "honest");

const lexRecall = oc.filter((r) => r.lexReject).length;
const incremental = oc.filter((r) => !r.lexReject && r.semReject); // semantic adds these
const netRecall = oc.filter((r) => r.stackReject).length;

const lexFP = honest.filter((r) => r.lexReject).length;
const semAddedFP = honest.filter((r) => !r.lexReject && r.semReject);
const netFP = honest.filter((r) => r.stackReject).length;

const pct = (x, n) => (n ? `${((x / n) * 100).toFixed(1)}%` : "n/a");
const summary = {
  slate: "offslate-valence-paraphrase",
  discipline: "layer-2 over lexical gate; context.forbidden disabled; reject-only verifier",
  verifier: { name: verifier.name, meta: verifier.meta },
  counts: { overclaim: oc.length, honest: honest.length },
  lexical_baseline: {
    recall: `${lexRecall}/${oc.length}`,
    recall_pct: pct(lexRecall, oc.length),
    fpr_observed: `${lexFP}/${honest.length}`,
    fpr_wilson_upper_95: Number(wilsonUpper(lexFP, honest.length).toFixed(4))
  },
  semantic_layer: {
    incremental_recall: `${incremental.length}/${oc.length - lexRecall} of lexical misses`,
    incremental_caught: incremental.map((r) => ({ id: r.id, tag: r.tag, reason: r.semReason, draft: r.draft })),
    added_false_positives: semAddedFP.map((r) => ({ id: r.id, tag: r.tag, reason: r.semReason, draft: r.draft }))
  },
  net_stack: {
    recall: `${netRecall}/${oc.length}`,
    recall_pct: pct(netRecall, oc.length),
    fpr_observed: `${netFP}/${honest.length}`,
    fpr_wilson_upper_95: Number(wilsonUpper(netFP, honest.length).toFixed(4))
  },
  caveat: "Off-slate is the TUNING harness; adoption requires beating lexical on a separate held-out set (ideally hosted-model-generated). FPR is observed with n + Wilson upper, never a bare 0%."
};

mkdirSync(outDir, { recursive: true });
writeFileSync(`${outDir}/summary.json`, `${JSON.stringify(summary, null, 2)}\n`);

console.log(`semantic-verifier bake-off — verifier: ${verifier.name} (v=${verifier.meta.version}, thr=${verifier.meta.threshold}, hash=${verifier.meta.hash})`);
console.log(`  lexical baseline recall:   ${lexRecall}/${oc.length}  (${pct(lexRecall, oc.length)})`);
console.log(`  semantic incremental:      ${incremental.length} of ${oc.length - lexRecall} lexical misses caught`);
console.log(`  NET stack recall:          ${netRecall}/${oc.length}  (${pct(netRecall, oc.length)})`);
console.log(`  NET FPR (observed):        ${netFP}/${honest.length}  (${pct(netFP, honest.length)}); Wilson 95% upper ${(wilsonUpper(netFP, honest.length) * 100).toFixed(1)}%`);
if (incremental.length) {
  console.log(`  incremental catches:`);
  for (const r of incremental) console.log(`    +${r.id} [${r.tag}] (${r.semReason}) "${r.draft}"`);
}
if (semAddedFP.length) {
  console.log(`  WARNING — semantic layer added false positives:`);
  for (const r of semAddedFP) console.log(`    !${r.id} [${r.tag}] (${r.semReason}) "${r.draft}"`);
}
console.log(`  receipt: ${outDir}/summary.json`);
