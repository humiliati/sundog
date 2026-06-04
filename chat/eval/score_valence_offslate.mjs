// Off-slate Valence robustness scorer (Phase 13 / S1).
//
// Measures whether the gate's GENERAL machinery (UPGRADE_LANGUAGE +
// UNSUPPORTED_CLAIMS + the failure_mode tag-classifier) catches fresh overclaim
// drafts WITHOUT the per-row `forbidden` lists that the gold slate hand-authors.
// Every draft is gated with `context.forbidden = []`, so the per-prompt lists
// cannot help — this is the honest tag-violation recall number (the gate to
// promoting Valence Completeness to a headline metric).
//
// Two arms:
//   recall = overclaim drafts the gate REJECTS / total overclaim drafts
//   false-positive rate = honest drafts the gate WRONGLY rejects / total honest
//
// Usage: node chat/eval/score_valence_offslate.mjs
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { gateFailures } from "../../public/js/sundog-claim-gate.mjs";

const root = process.cwd();
const slatePath = "chat/prompts/offslate-valence-paraphrase.jsonl";
const claimMapPath = "chat/claim_map.json";
const outDir = "results/chat/phase13-valence-offslate";

const claimMap = JSON.parse(readFileSync(claimMapPath, "utf8"));
const routes = [...(claimMap.claims || []), ...(claimMap.nonClaimRoutes || [])];
const routeById = new Map(routes.map((r) => [r.id, r]));

const rows = readFileSync(slatePath, "utf8")
  .split(/\r?\n/)
  .filter(Boolean)
  .map((line) => JSON.parse(line));

const results = rows.map((row) => {
  const route = routeById.get(row.route);
  if (!route) {
    return { ...row, verdict: "error", correct: false, failures: ["route_missing"] };
  }
  // Synthesize the deterministic trace the gate would see for this route.
  const trace = {
    routeId: route.id,
    disposition: route.disposition || "allow",
    evidenceTier: route.evidenceTier || "unknown",
    failureMode: route.failureMode || null,
    traceVisible: true
  };
  // The discipline: EMPTY forbidden — only the general machinery may catch it.
  const failures = gateFailures({ prompt: "", trace, draftAnswer: row.draft, context: {} });
  const verdict = failures.length > 0 ? "reject" : "accept";
  return { ...row, verdict, correct: verdict === row.expectGate, failures };
});

const overclaims = results.filter((r) => r.kind === "overclaim");
const honest = results.filter((r) => r.kind === "honest");
const caught = overclaims.filter((r) => r.verdict === "reject");
const missed = overclaims.filter((r) => r.verdict !== "reject");
const falsePos = honest.filter((r) => r.verdict === "reject");

const recall = overclaims.length ? caught.length / overclaims.length : null;
const fpr = honest.length ? falsePos.length / honest.length : null;

// Per-tag recall
const byTag = {};
for (const r of overclaims) {
  byTag[r.tag] = byTag[r.tag] || { caught: 0, total: 0 };
  byTag[r.tag].total += 1;
  if (r.verdict === "reject") byTag[r.tag].caught += 1;
}

const summary = {
  slate: "offslate-valence-paraphrase",
  discipline: "context.forbidden disabled — general machinery only (UPGRADE_LANGUAGE + UNSUPPORTED_CLAIMS + failure_mode tag-classifier)",
  counts: { total: results.length, overclaim: overclaims.length, honest: honest.length },
  overclaim_recall: recall,
  honest_false_positive_rate: fpr,
  caught: caught.length,
  missed: missed.map((r) => ({ id: r.id, tag: r.tag, draft: r.draft })),
  false_positives: falsePos.map((r) => ({ id: r.id, tag: r.tag, draft: r.draft, failures: r.failures })),
  per_tag_recall: Object.fromEntries(
    Object.entries(byTag).map(([t, v]) => [t, `${v.caught}/${v.total}`])
  ),
  caveat: "Curated-adversarial (internally authored). The gold standard for true off-distribution recall is scoring actual hosted-model drafts with forbidden disabled; this set is the deterministic, no-API proxy."
};

mkdirSync(outDir, { recursive: true });
writeFileSync(`${outDir}/summary.json`, `${JSON.stringify(summary, null, 2)}\n`);

const pct = (x) => (x === null ? "n/a" : `${(x * 100).toFixed(1)}%`);
console.log(`off-slate valence robustness (${results.length} drafts; forbidden DISABLED)`);
console.log(`  overclaim recall:        ${caught.length}/${overclaims.length}  (${pct(recall)})`);
console.log(`  honest false-pos rate:   ${falsePos.length}/${honest.length}  (${pct(fpr)})`);
console.log(`  per-tag recall:          ${JSON.stringify(summary.per_tag_recall)}`);
if (missed.length) {
  console.log(`  MISSED overclaims (general machinery did not catch):`);
  for (const r of missed) console.log(`    ${r.id} [${r.tag}] "${r.draft}"`);
}
if (falsePos.length) {
  console.log(`  FALSE POSITIVES (honest wrongly rejected):`);
  for (const r of falsePos) console.log(`    ${r.id} [${r.tag}] ${JSON.stringify(r.failures)} "${r.draft}"`);
}
console.log(`  receipt: ${outDir}/summary.json`);
