#!/usr/bin/env node
// H-K6 audit: are the adversarial cards structurally complete AND do they cover the seeded
// overclaims the lane ledgers already name? The falsifier ADVERSARIAL_CARDS_DO_NOT_STICK has
// two clauses: (B, mechanical here) the cards fail to cover a documented model-committed
// overclaim (a card id where BOTH judges said "overclaim" in the committed Hodge/Kakeya judge
// results); (A, structural proxy here) a card's repair is a bare disclaimer that does not
// preserve the body/shadow hook (no named structure). Clause A's full test is behavioral and
// stays out of scope; the proxy is reported honestly. Report-only; no public claim.

import fs from "node:fs";
import path from "node:path";

const CARDS = path.join("docs", "eval", "hk6-adversarial-cards.jsonl");
const OUT_DIR = path.join("results", "eval", "hk6-card-audit");
const JUDGE_DIRS = [
  path.join("results", "hodge", "register-judge", "cells"),
  path.join("results", "kakeya", "register-judge", "cells"),
];
const REQUIRED = ["id", "lane", "source_row", "prompt", "target_register", "body", "shadow",
  "known_because", "tempting_wrong_answer", "correct_answer", "falsifier_tags",
  "seeded_overclaim_source", "register_changed", "repair", "falsifier_named"];
const COMPOSITION = { hodge: 6, kakeya: 4, cross: 2 };
const FALSIFIER_REGISTRY = new Set([
  "HODGE-CATEGORY-ERROR", "HODGE-LIT-MISMATCH", "HODGE-TOY-LAUNDERING",
  "HODGE-VISUAL-MISCALIBRATED", "REGISTER_PROBLEMS_VACUOUS",
  "KAK-SHADOW-REENCODING", "KAK_SHADOW_REENCODING_EMPIRICAL", "KAK-SIZE-OVERCLAIM",
  "KAK-EUCLIDEAN-LEAP", "KAK-DENSITY-ARTIFACT", "ADAPTIVE_FIBERING_NO_SIGNAL",
  "COLLISION_TABLE_ONLY_RHYME", "ADVERSARIAL_CARDS_DO_NOT_STICK",
]);
// a repair preserves the hook iff it names concrete structure, not just caution
const HOOK = /Lefschetz|Cattani|CDK|Dvir|floor|signature|collision|q\(q\^2-q\+1\)|control|conjecture|cycle|rational|integral|projective|Kaehler|count|rate|judge|line|H\^2|\(p,p\)|\(1,1\)|\(2,2\)|Wang-Zahl|finite field|sparse/i;

function seededOverclaims() {
  const out = new Set();
  for (const dir of JUDGE_DIRS) {
    for (const f of fs.readdirSync(dir)) {
      const detail = JSON.parse(fs.readFileSync(path.join(dir, f), "utf8")).detail;
      for (const r of detail) {
        const vs = Object.values(r.judges).map((j) => j.verdict);
        if (vs.length >= 2 && vs.every((v) => v === "overclaim")) out.add(r.id);
      }
    }
  }
  return out;
}

function main() {
  const cards = fs.readFileSync(CARDS, "utf8").split(/\r?\n/).filter((l) => l.trim())
    .map((l) => JSON.parse(l));
  const rows = cards.map((c) => {
    const checks = {
      fields_complete: REQUIRED.every((k) => c[k] != null && String(c[k]).length > 0
        || Array.isArray(c[k])),
      overclaim_present: Boolean(c.tempting_wrong_answer && c.tempting_wrong_answer.length > 20),
      repair_preserves_hook: Boolean(c.repair && HOOK.test(c.repair) && c.repair.length > 40),
      register_named: Boolean(c.register_changed && c.register_changed.length > 10),
      falsifier_in_registry: FALSIFIER_REGISTRY.has(c.falsifier_named),
      answer_refuses_trap: /^\s*no\b/i.test(c.correct_answer || ""),
    };
    const pass = Object.values(checks).every(Boolean);
    return { id: c.id, lane: c.lane, seeds: c.seeded_overclaim_source, pass, checks };
  });

  const comp = {};
  for (const r of rows) comp[r.lane] = (comp[r.lane] || 0) + 1;
  const compositionOk = Object.entries(COMPOSITION).every(([k, v]) => comp[k] === v);

  const seeds = seededOverclaims();
  const covered = new Set(cards.flatMap((c) => c.seeded_overclaim_source || []));
  const uncovered = [...seeds].filter((s) => !covered.has(s)).sort();
  const staleClaims = [...covered].filter((s) => !seeds.has(s)).sort();

  const structuralFail = rows.filter((r) => !r.pass);
  const fired = structuralFail.length > 0 || uncovered.length > 0 || !compositionOk;
  const manifest = {
    artifactId: "HK6-ADVERSARIAL-CARD-AUDIT", generatedAt: new Date().toISOString(),
    status: "internal structural + coverage audit",
    cards: cards.length, composition: comp, composition_ok: compositionOk,
    structural_pass: rows.length - structuralFail.length,
    seeded_overclaims_documented: [...seeds].sort(),
    seeded_overclaims_covered: [...covered].sort(),
    uncovered_seeds: uncovered, stale_seed_claims: staleClaims,
    falsifier: {
      name: "ADVERSARIAL_CARDS_DO_NOT_STICK", fired,
      clauseB_mechanical: uncovered.length === 0
        ? "clear: every documented model-committed overclaim (both-judge) is covered by a card"
        : `FIRED: uncovered seeds ${uncovered.join(", ")}`,
      clauseA_structural_proxy: structuralFail.length === 0
        ? "proxy clear: every repair names concrete structure (not a bare disclaimer); full behavioral test out of scope"
        : `FIRED: structural failures ${structuralFail.map((r) => r.id).join(", ")}`,
    },
    rows,
  };
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(path.join(OUT_DIR, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
  const csv = ["id,lane,pass,seeds", ...rows.map((r) => `${r.id},${r.lane},${r.pass},"${(r.seeds || []).join(" ")}"`)].join("\n");
  fs.writeFileSync(path.join(OUT_DIR, "card-summary.csv"), csv + "\n");

  for (const r of structuralFail) console.log(`STRUCTURAL-FAIL ${r.id}: ${JSON.stringify(r.checks)}`);
  console.log(`HK6_CARD_AUDIT cards=${cards.length} composition=${JSON.stringify(comp)} `
    + `structural_pass=${manifest.structural_pass}/${cards.length} seeds=${seeds.size} `
    + `uncovered=${uncovered.length} falsifier=${fired ? "fired" : "clear"} out=${OUT_DIR}`);
  process.exit(fired ? 1 : 0);
}

main();
