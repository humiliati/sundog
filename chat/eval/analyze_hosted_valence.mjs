// Hosted-draft valence analysis (Phase 13 §13.3 + S1 held-out).
//
// Reads the hosted-draft outcomes for the generality slate, then for every full
// draft computes:
//   (1) deployed gate verdict (WITH per-row forbidden) — the shipped architecture;
//   (2) general-machinery verdict (forbidden DISABLED) — does the gate generalize
//       to real model overclaims without the hand-authored per-row list?
// Banks the gate-caught overclaim drafts as an independent held-out set for the
// §13 semantic-verifier bake-off, and writes a receipt.
//
// Usage: node chat/eval/analyze_hosted_valence.mjs openai anthropic
import { readFileSync, mkdirSync, writeFileSync, existsSync } from "node:fs";
import { gateFailures } from "../../public/js/sundog-claim-gate.mjs";

const backends = process.argv.slice(2);
if (!backends.length) backends.push("openai", "anthropic");

const claimMap = JSON.parse(readFileSync("chat/claim_map.json", "utf8"));
const routeById = new Map(
  [...(claimMap.claims || []), ...(claimMap.nonClaimRoutes || [])].map((r) => [r.id, r])
);
const slateById = new Map(
  readFileSync("chat/prompts/gold-generality-boundary.jsonl", "utf8")
    .split(/\r?\n/).filter(Boolean).map((l) => JSON.parse(l))
    .map((r) => [r.id, r])
);

const heldOut = [];
const perBackend = {};
let totalDrafts = 0;
let totalDeployedReject = 0;
let genCatchesOfDeployedReject = 0;
let acceptedButGenWouldReject = [];

for (const backend of backends) {
  const path = `results/chat/phase5-hosted/generality-boundary/${backend}/draft-outcomes.json`;
  if (!existsSync(path)) { console.warn(`skip ${backend}: ${path} missing`); continue; }
  const rows = JSON.parse(readFileSync(path, "utf8"));
  const stats = { drafts: 0, deployed_reject: 0, gen_reject: 0, gen_catch_of_deployed_reject: 0 };
  for (const row of rows) {
    if (row.status === "error" || typeof row.draftFull !== "string") continue;
    const route = routeById.get(row.routeId);
    const slate = slateById.get(row.id);
    const trace = {
      routeId: row.routeId,
      disposition: row.disposition,
      evidenceTier: row.evidenceTier,
      failureMode: route?.failureMode || null,
      traceVisible: true
    };
    const deployedFails = gateFailures({ prompt: "", trace, draftAnswer: row.draftFull, context: { forbidden: slate?.forbidden || [] } });
    const genFails = gateFailures({ prompt: "", trace, draftAnswer: row.draftFull, context: {} });
    const deployedReject = deployedFails.length > 0;
    const genReject = genFails.length > 0;
    stats.drafts += 1; totalDrafts += 1;
    if (deployedReject) {
      stats.deployed_reject += 1; totalDeployedReject += 1;
      if (genReject) { stats.gen_catch_of_deployed_reject += 1; genCatchesOfDeployedReject += 1; }
      heldOut.push({
        id: row.id, backend, lane: slate?.lane, tag: route?.failureMode, route: row.routeId,
        deployed_failures: deployedFails, gen_failures: genFails, gen_caught: genReject,
        draft: row.draftFull
      });
    }
    if (genReject) stats.gen_reject += 1;
    if (!deployedReject && genReject) {
      acceptedButGenWouldReject.push({ id: row.id, backend, tag: route?.failureMode, gen_failures: genFails, draft: row.draftFull });
    }
  }
  perBackend[backend] = stats;
}

const outDir = "results/chat/phase13-hosted-valence";
mkdirSync(outDir, { recursive: true });
writeFileSync(`${outDir}/held-out-overclaims.jsonl`, heldOut.map((r) => JSON.stringify(r)).join("\n") + (heldOut.length ? "\n" : ""));

const summary = {
  what: "Hosted-draft valence analysis on the generality slate — §13.3 deployed result + forbidden-disabled (general-machinery) recall on real model overclaims",
  backends,
  models: { openai: "gpt-4o-mini", anthropic: "claude-haiku-4-5-20251001" },
  deployed_architecture: {
    note: "gate WITH per-row forbidden + trace handoff (the shipped system). 'reject' = caught an overclaim, fell back to static template.",
    per_backend: perBackend,
    total_drafts: totalDrafts,
    total_deployed_rejects: totalDeployedReject,
    gate_escapes_by_runner: "see each run summary.json (0/0 reported)"
  },
  general_machinery_on_real_overclaims: {
    note: "Of the drafts the deployed gate caught (real model overclaims), how many the GENERAL machinery (forbidden disabled) also catches. This is the honest off-distribution recall on real, model-generated drafts.",
    deployed_rejects: totalDeployedReject,
    general_also_caught: genCatchesOfDeployedReject,
    recall: totalDeployedReject ? Number((genCatchesOfDeployedReject / totalDeployedReject).toFixed(4)) : null,
    accepted_but_general_would_reject: acceptedButGenWouldReject
  },
  held_out_set: { count: heldOut.length, path: `${outDir}/held-out-overclaims.jsonl` },
  caveat: "'Deployed reject' is the gate's own label (forbidden-aware); whether any ACCEPTED draft semantically overclaims is exactly what the §13 semantic verifier would test against this held-out set. Models were given forbidden phrases in the trace handoff (deployed architecture), which suppresses overclaims — so the held-out set is overclaims that slipped past coaching."
};
writeFileSync(`${outDir}/summary.json`, JSON.stringify(summary, null, 2) + "\n");

console.log("hosted-draft valence analysis");
for (const b of backends) if (perBackend[b]) console.log(`  ${b}: ${perBackend[b].drafts} drafts, deployed-reject ${perBackend[b].deployed_reject}, general-machinery caught ${perBackend[b].gen_catch_of_deployed_reject} of those`);
console.log(`  TOTAL real overclaims caught by deployed gate: ${totalDeployedReject}`);
console.log(`  general machinery (forbidden disabled) recall on them: ${genCatchesOfDeployedReject}/${totalDeployedReject}` + (totalDeployedReject ? ` (${((genCatchesOfDeployedReject/totalDeployedReject)*100).toFixed(1)}%)` : ""));
console.log(`  accepted-but-general-would-reject: ${acceptedButGenWouldReject.length}`);
console.log(`  held-out overclaim set: ${heldOut.length} drafts -> ${outDir}/held-out-overclaims.jsonl`);
console.log(`  receipt: ${outDir}/summary.json`);
