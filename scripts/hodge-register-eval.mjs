#!/usr/bin/env node
// H-K1 route/fence-fidelity eval over the Hodge register cards (spec section 7; slate
// H-K6 "score only route/fence fidelity"). Deterministic first run: score each card's
// gold answer (correct_answer) and trap answer (tempting_wrong_answer) with a FIXED,
// card-independent route/fence rubric, and check every card is a discriminating eval
// item (gold fences/refuses; trap overclaims). No model, no public claim. The same
// scorer later scores real model responses (the staged model-in-the-loop run).

import fs from "node:fs";
import path from "node:path";

const ARTIFACT_ID = "HODGE-HK1-REGISTER-FIDELITY-EVAL";
const DEFAULT_IN = path.join("docs", "hodge", "register-problem-cards.jsonl");
const DEFAULT_OUT = path.join("results", "hodge", "register-eval");

// Fixed rubric, independent of any card's text. FENCE = boundary/refusal discipline a
// faithful Hodge answer uses; OVERCLAIM = the register/route errors the cards trap.
const FENCE = [
  /\bno\b/i, /representative/i, /conjecture/i, /codimension one/i, /rational only/i,
  /none licensed/i, /\bboundary\b/i, /does not/i, /cannot/i, /not a (body|cycle)/i,
  /not the\b/i, /not because/i, /not through/i, /not safe/i, /\btrivial\b/i,
  /must (say|be labeled)/i, /\brepair\b/i,
];
const OVERCLAIM = [
  /\byes\b/i, /is the (rational class|algebraic cycle)/i, /displays the/i, /constructs/i,
  /by the same mechanism/i, /every integral/i, /exactly the condition/i,
  /has been found/i, /applies unchanged/i, /reading the shadow backward/i,
];

function score(text) {
  const s = String(text || "");
  const fence = FENCE.reduce((n, re) => n + (re.test(s) ? 1 : 0), 0);
  const overclaim = OVERCLAIM.reduce((n, re) => n + (re.test(s) ? 1 : 0), 0);
  return { fence, overclaim, fidelity: fence - overclaim, route_correct: fence > overclaim };
}

function parseArgs(argv) {
  const a = {};
  for (let i = 0; i < argv.length; i++) {
    const r = argv[i];
    if (!r.startsWith("--")) continue;
    const b = r.slice(2), eq = b.indexOf("=");
    if (eq !== -1) { a[b.slice(0, eq)] = b.slice(eq + 1); continue; }
    const next = argv[i + 1];
    if (next && !next.startsWith("--")) { a[b] = next; i++; } else { a[b] = true; }
  }
  return a;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const inPath = args.in || DEFAULT_IN;
  const outDir = args.out || DEFAULT_OUT;
  const cards = fs.readFileSync(inPath, "utf8").split(/\r?\n/).filter((l) => l.trim())
    .map((l) => JSON.parse(l));

  const rows = cards.map((c) => {
    const gold = score(c.correct_answer);
    const trap = score(c.tempting_wrong_answer);
    const separation = gold.fidelity - trap.fidelity;
    // A discriminating eval item: the faithful answer routes correctly (fences) and the
    // tempting answer does not, with a positive fidelity separation.
    const discriminates = gold.route_correct && !trap.route_correct && separation > 0;
    return { id: c.id, gold, trap, separation, discriminates };
  });

  const n = rows.length;
  const discriminating = rows.filter((r) => r.discriminates).length;
  const mean = (f) => Number((rows.reduce((s, r) => s + f(r), 0) / n).toFixed(3));
  const summary = {
    artifact_id: ARTIFACT_ID, date: new Date().toISOString().slice(0, 10),
    input: inPath, n, discriminating, non_discriminating: n - discriminating,
    mean_gold_fidelity: mean((r) => r.gold.fidelity),
    mean_trap_fidelity: mean((r) => r.trap.fidelity),
    mean_separation: mean((r) => r.separation),
    gold_route_correct: rows.filter((r) => r.gold.route_correct).length,
    trap_route_correct: rows.filter((r) => r.trap.route_correct).length,
    eval_ready: discriminating === n,
    rows,
  };
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(summary, null, 2));
  const csv = ["id,gold_fidelity,trap_fidelity,separation,discriminates",
    ...rows.map((r) => `${r.id},${r.gold.fidelity},${r.trap.fidelity},${r.separation},${r.discriminates}`)]
    .join("\n");
  fs.writeFileSync(path.join(outDir, "fidelity-summary.csv"), csv + "\n");

  for (const r of rows.filter((r) => !r.discriminates)) {
    console.log(`NON-DISCRIMINATING ${r.id}: gold_fid=${r.gold.fidelity} trap_fid=${r.trap.fidelity} sep=${r.separation}`);
  }
  console.log(`HODGE_REGISTER_FIDELITY_EVAL cards=${n} discriminating=${discriminating} `
    + `gold_route=${summary.gold_route_correct}/${n} trap_route=${summary.trap_route_correct}/${n} `
    + `mean_sep=${summary.mean_separation} eval_ready=${summary.eval_ready} out=${outDir}`);
  return summary.eval_ready ? 0 : 1;
}

process.exit(main());
