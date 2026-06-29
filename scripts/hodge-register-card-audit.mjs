#!/usr/bin/env node
// H-K1 audit: check the Hodge register-ladder problem cards against the Phase 4 spec
// (docs/hodge/PHASE4_REGISTER_PROBLEM_GENERATOR_SPEC.md). Mechanizes the section-3
// card-generation gate and the section-6 answer-key checklist, and evaluates the
// REGISTER_PROBLEMS_VACUOUS falsifier. Report-only; no model eval, no public page.

import fs from "node:fs";
import path from "node:path";

const ARTIFACT_ID = "HODGE-HK1-REGISTER-CARD-AUDIT";
const DEFAULT_IN = path.join("docs", "hodge", "register-problem-cards.jsonl");
const DEFAULT_OUT = path.join("results", "hodge", "register-card-audit");

const REQUIRED = ["id", "source_row", "prompt", "target_register", "body", "shadow",
  "known_because", "tempting_wrong_answer", "correct_answer", "falsifier_tags"];
const ALLOWED_TAGS = new Set([
  "REGISTER_PROBLEMS_VACUOUS", "HODGE-CATEGORY-ERROR", "HODGE-LIT-MISMATCH",
  "HODGE-TOY-LAUNDERING", "HODGE-VISUAL-MISCALIBRATED"]);
// tags that mark a card as catching a named confusion/boundary (not a lookup)
const CONFUSION_TAGS = new Set([
  "HODGE-CATEGORY-ERROR", "HODGE-LIT-MISMATCH", "HODGE-TOY-LAUNDERING",
  "HODGE-VISUAL-MISCALIBRATED"]);

function parseArgs(argv) {
  const a = {};
  for (let i = 0; i < argv.length; i++) {
    const r = argv[i];
    if (r === "--help" || r === "-h") { a.help = true; continue; }
    if (!r.startsWith("--")) continue;
    const b = r.slice(2), eq = b.indexOf("=");
    if (eq !== -1) { a[b.slice(0, eq)] = b.slice(eq + 1); continue; }
    const next = argv[i + 1];
    if (next && !next.startsWith("--")) { a[b] = next; i++; } else { a[b] = true; }
  }
  return a;
}

function nonEmptyStr(v) { return typeof v === "string" && v.trim().length > 0; }

// Each check returns null on pass or a short failure reason.
function auditCard(c) {
  const checks = {};
  const tags = Array.isArray(c.falsifier_tags) ? c.falsifier_tags : [];
  const hasConfusionTag = tags.some((t) => CONFUSION_TAGS.has(t));
  const tr = String(c.target_register || "");
  const sr = String(c.source_row || "");
  const kb = String(c.known_because || "");
  const sh = String(c.shadow || "");

  // section 3: required fields present and non-empty (no source_row / no tempting
  // wrong answer / unstated answer => generation fails).
  checks.required_fields =
    REQUIRED.every((k) => (k === "falsifier_tags" ? tags.length > 0 : nonEmptyStr(c[k])))
      ? null : "missing or empty required field";
  checks.id_format = /^HODGE-RG-\d{3}$/.test(String(c.id || ""))
    ? null : "id must be HODGE-RG-###";

  // section 6.1: names a target register or arrow.
  checks.names_register = /R[1-4]/.test(tr) ? null : "target_register names no R1-R4 register";
  // section 6.2: source row points to reader ladder / gallery row / boundary.
  checks.source_anchor = /CE[1-3]|G[1-5]|reader fence|hard-excluded|boundary/i.test(sr)
    ? null : "source_row cites no reader/gallery/boundary anchor";
  // section 6.3: body is a known cycle or explicitly 'none licensed'.
  const bodyKind = /^none licensed/i.test(String(c.body || "")) ? "none-licensed" : "cycle";
  checks.body_kind = nonEmptyStr(c.body) ? null : "body empty";
  // section 6.4: shadow is not a bare harmonic representative unless testing that confusion.
  checks.shadow_not_bare_harmonic =
    (/harmonic/i.test(sh) && !/R2/.test(tr) && !/not the harmonic/i.test(sh))
      ? "shadow is a bare harmonic representative" : null;
  // section 6.5: tempting wrong answer maps to a named category error or boundary.
  checks.tempting_named =
    (hasConfusionTag || /boundary/i.test(kb)) ? null
      : "tempting_wrong_answer maps to no named error/boundary";
  // section 6.6 (proxy): answer key auditable from a cited theorem/roster rule or boundary.
  checks.answer_auditable =
    (nonEmptyStr(kb) && /Lefschetz|Cattani|cohomology|hyperplane|point class|boundary|theorem/i.test(kb))
      ? null : "known_because cites no checkable theorem/roster rule";
  // tag hygiene.
  checks.tags_allowed = tags.every((t) => ALLOWED_TAGS.has(t))
    ? null : "falsifier_tags contains an unknown tag";
  // falsifier 4.1 non-vacuity: tests a transition/error/boundary, not definition lookup.
  checks.non_vacuous =
    (/->|vs|fence|admission/i.test(tr) || hasConfusionTag) ? null
      : "card reduces to definition lookup (no transition/error/boundary)";

  const failed = Object.entries(checks).filter(([, v]) => v).map(([k, v]) => ({ check: k, reason: v }));
  // REGISTER_PROBLEMS_VACUOUS fires per spec section 4: lookup-only (non_vacuous) OR
  // answer key not auditable from the named sources (source_anchor / answer_auditable).
  const vacuous = ["non_vacuous", "source_anchor", "answer_auditable"].some((k) => checks[k]);
  return { id: c.id, body_kind: bodyKind, pass: failed.length === 0, failed, vacuous_contrib: vacuous };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log("Usage: node scripts/hodge-register-card-audit.mjs [--in cards.jsonl] [--out dir]");
    return 0;
  }
  const inPath = args.in || DEFAULT_IN;
  const outDir = args.out || DEFAULT_OUT;
  const lines = fs.readFileSync(inPath, "utf8").split(/\r?\n/).filter((l) => l.trim());
  const cards = lines.map((l, i) => {
    try { return JSON.parse(l); }
    catch (e) { throw new Error(`line ${i + 1}: invalid JSON (${e.message})`); }
  });

  const results = cards.map(auditCard);
  const pass = results.filter((r) => r.pass).length;
  const blocked = results.length - pass;
  const vacuousFires = results.some((r) => r.vacuous_contrib);
  const ids = cards.map((c) => c.id);
  const dupIds = ids.length !== new Set(ids).size;
  const bodyKinds = results.reduce((m, r) => ((m[r.body_kind] = (m[r.body_kind] || 0) + 1), m), {});

  const manifest = {
    artifact_id: ARTIFACT_ID, date: new Date().toISOString().slice(0, 10),
    input: inPath, card_count: cards.length, pass, blocked,
    duplicate_ids: dupIds, body_kinds: bodyKinds,
    falsifier: { name: "REGISTER_PROBLEMS_VACUOUS", fires: vacuousFires },
    cards: results,
  };
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2));
  const csv = ["id,body_kind,pass,failed_checks",
    ...results.map((r) => `${r.id},${r.body_kind},${r.pass},${r.failed.map((f) => f.check).join("|")}`)]
    .join("\n");
  fs.writeFileSync(path.join(outDir, "card-summary.csv"), csv + "\n");

  for (const r of results.filter((r) => !r.pass)) {
    console.log(`BLOCKED ${r.id}: ${r.failed.map((f) => `${f.check} (${f.reason})`).join("; ")}`);
  }
  const falsifier = vacuousFires ? "FIRES" : "clear";
  console.log(`HODGE_REGISTER_CARD_AUDIT cards=${cards.length} pass=${pass} blocked=${blocked} `
    + `dup_ids=${dupIds} falsifier=${falsifier} out=${outDir}`);
  return blocked === 0 && !vacuousFires && !dupIds ? 0 : 1;
}

process.exit(main());
