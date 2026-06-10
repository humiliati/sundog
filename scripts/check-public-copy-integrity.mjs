#!/usr/bin/env node
/**
 * check-public-copy-integrity.mjs  —  ship-ticket #10
 *
 * The chat experiment's promise is that the assistant holds the same evidence
 * boundaries the rest of the public copy does (SUNDOG_V_CHAT.md §16). This gate
 * fails the build when that promise could break:
 *
 *   [A] Claim-map routing integrity — every claim_map.json support[].doc resolves
 *       to a real file, and every route's evidenceTier is one the map defines.   (FAIL)
 *   [B] Overclaim scan of the COUPLED SURFACES (index.html elevator pitch +
 *       sundog.html halo atlas).                          (UNSUPPORTED_CLAIMS = FAIL;
 *                                                          UPGRADE_LANGUAGE = WARN)
 *   [C] UNSUPPORTED_CLAIMS across all OTHER public pages — those phrases
 *       ("solves alignment", "llm alignment result", …) must never appear
 *       UNNEGATED anywhere.                                                       (FAIL)
 *   [D] Routes citing no-publish docs as visible sources — the public trace would
 *       name evidence a visitor can't reach.                                      (WARN)
 *
 * Consistency-by-construction: the scan reuses the LIVE claim gate's own
 * `gateFailures()` (public/js/sundog-claim-gate.mjs) — same vocabulary, same
 * negation-aware matching the chat answers are held to. A hedged/negated use
 * ("we do NOT claim this proves X") passes exactly as it does in the gate, and
 * QUOTED adversarial demo prompts (in <blockquote>/<pre>/<code>) are skipped —
 * the chat result page deliberately quotes overclaims to show the gate refusing them.
 *
 * Exits 1 on any FAIL; WARNs are informational. Wired into `postbuild`.
 */

import { readFileSync, existsSync, readdirSync } from "node:fs";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gateFailures } from "../public/js/sundog-claim-gate.mjs";
import { isNoPublish } from "./docs-no-publish.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

// The two dense claim surfaces named by ticket #10 / §16.1.
const COUPLED_SURFACES = ["index.html", "sundog.html"];

// A neutral trace so gateFailures runs ONLY its content checks (no structural
// failures like missing_trace / refusal-route rules) on static copy.
const NEUTRAL_TRACE = { traceVisible: true, routeId: "__public_copy__", disposition: "allow" };

// ---------------------------------------------------------------- helpers ----

/** Strip HTML to visible block-level text chunks (one per paragraph/heading/li). */
function htmlToChunks(html) {
  let s = html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<head[\s\S]*?<\/head>/gi, " ")
    .replace(/<!--[\s\S]*?-->/g, " ")
    // Quoted/example/code blocks are NOT the page's own assertions — drop them.
    .replace(/<blockquote[\s\S]*?<\/blockquote>/gi, " ")
    .replace(/<pre[\s\S]*?<\/pre>/gi, " ")
    .replace(/<code[\s\S]*?<\/code>/gi, " ");
  s = s
    .replace(/<\/(p|li|h[1-6]|div|section|article|td|th|figcaption|blockquote|dd|dt)>/gi, "\n")
    .replace(/<br\s*\/?>/gi, "\n");
  s = s.replace(/<[^>]+>/g, " ").replace(/&[a-z]+;|&#\d+;/gi, " ");
  return s
    .split(/\n+/)
    .map((x) => x.replace(/\s+/g, " ").trim())
    .filter((x) => x.length > 20);
}

/** Run the LIVE claim gate on one chunk; split overclaim failures by class. */
function scanChunk(chunk) {
  const failures = gateFailures({ prompt: "", trace: NEUTRAL_TRACE, draftAnswer: chunk, context: {} });
  return {
    unsupported: failures.filter((f) => f.startsWith("unsupported_claim:")),
    upgrade: failures.filter((f) => f.startsWith("upgrade_language:")),
  };
}

function scanFile(file) {
  const html = readFileSync(join(ROOT, file), "utf8");
  const hits = [];
  for (const chunk of htmlToChunks(html)) {
    const { unsupported, upgrade } = scanChunk(chunk);
    if (unsupported.length || upgrade.length) hits.push({ unsupported, upgrade, chunk });
  }
  return hits;
}

const listRootPages = () => readdirSync(ROOT).filter((f) => f.endsWith(".html")).sort();
const trunc = (s, n = 140) => (s.length > n ? s.slice(0, n) + "…" : s);

// ----------------------------------------------------------- [A] routing ----

function checkRouting() {
  const issues = [];          // FAIL
  const noPublishCites = [];  // WARN
  const map = JSON.parse(readFileSync(join(ROOT, "chat/claim_map.json"), "utf8"));
  // evidenceTiers is an array of { id, label, meaning } — key the set by `id`.
  const tierList = Array.isArray(map.evidenceTiers)
    ? map.evidenceTiers.map((t) => t.id)
    : Object.keys(map.evidenceTiers || {});
  const tiers = new Set(tierList);
  for (const claim of map.claims || []) {
    const id = claim.id || "(unnamed route)";
    if (claim.evidenceTier && !tiers.has(claim.evidenceTier)) {
      issues.push({ id, kind: "invalid-tier", detail: claim.evidenceTier });
    }
    for (const s of claim.support || []) {
      if (s.doc && !existsSync(join(ROOT, s.doc))) issues.push({ id, kind: "missing-doc", detail: s.doc });
      else if (s.doc && isNoPublish(s.doc)) noPublishCites.push({ id, detail: s.doc });
    }
    if (claim.nextAction?.href?.startsWith("/docs/")) {
      const rel = claim.nextAction.href.replace(/^\//, "");
      if (!existsSync(join(ROOT, rel))) issues.push({ id, kind: "missing-nextAction", detail: claim.nextAction.href });
    }
  }
  return { issues, noPublishCites, tierCount: tiers.size, routeCount: (map.claims || []).length, version: map.version };
}

// ----------------------------------------------------------------- report ----

function main() {
  console.log("=== PUBLIC-COPY INTEGRITY ===\n");
  let fails = 0;
  let warns = 0;

  // [A]
  const routing = checkRouting();
  console.log(`[A] CLAIM-MAP ROUTING INTEGRITY  (v${routing.version}, ${routing.routeCount} routes, ${routing.tierCount} tiers)`);
  if (!routing.issues.length) {
    console.log("    clean — every support doc resolves; every tier is defined.");
  } else {
    fails += routing.issues.length;
    for (const i of routing.issues) console.log(`    ✗ FAIL ${i.kind}: ${i.detail}   [route: ${i.id}]`);
  }
  console.log("");

  // [B] coupled surfaces — full scan
  console.log("[B] OVERCLAIM SCAN — COUPLED SURFACES (negation-aware, via live claim-gate)");
  for (const file of COUPLED_SURFACES) {
    const hits = scanFile(file);
    if (!hits.length) { console.log(`    ${file}: clean`); continue; }
    for (const h of hits) {
      if (h.unsupported.length) { fails += h.unsupported.length; console.log(`    ✗ FAIL ${file} [${h.unsupported.join(", ")}]  "${trunc(h.chunk)}"`); }
      if (h.upgrade.length) { warns += h.upgrade.length; console.log(`    ⚠ WARN ${file} [${h.upgrade.join(", ")}]  "${trunc(h.chunk)}"`); }
    }
  }
  console.log("");

  // [C] all OTHER pages — UNSUPPORTED_CLAIMS only
  console.log("[C] UNSUPPORTED-CLAIM SCAN — OTHER PUBLIC PAGES (should never appear unnegated)");
  let cPages = 0;
  for (const file of listRootPages()) {
    if (COUPLED_SURFACES.includes(file)) continue; // already fully scanned in [B]
    const hits = scanFile(file).filter((h) => h.unsupported.length);
    if (!hits.length) continue;
    cPages++;
    for (const h of hits) { fails += h.unsupported.length; console.log(`    ✗ FAIL ${file} [${h.unsupported.join(", ")}]  "${trunc(h.chunk)}"`); }
  }
  if (!cPages) console.log("    clean across all other public pages.");
  console.log("");

  // [D] no-publish source citations
  console.log("[D] NO-PUBLISH SOURCE CITATIONS — routes citing docs withheld from dist/");
  if (!routing.noPublishCites.length) {
    console.log("    none.");
  } else {
    warns += routing.noPublishCites.length;
    for (const c of routing.noPublishCites) console.log(`    ⚠ WARN ${c.detail}   [route: ${c.id}]`);
  }
  console.log("");

  console.log(`--- ${fails} FAIL, ${warns} WARN ---`);
  if (fails) {
    console.log("  ✗ public-copy integrity FAILED.");
    process.exit(1);
  }
  console.log("  ✓ public-copy integrity clean (warnings are informational).");
}

main();
