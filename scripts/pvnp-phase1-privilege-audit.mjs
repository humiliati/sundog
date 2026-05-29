#!/usr/bin/env node
// scripts/pvnp-phase1-privilege-audit.mjs
//
// Static-analysis privilege audit for SUNDOG_V_P_V_NP Phase 1.
// Mirrors the pattern of scripts/mesa-signature-provenance-audit.mjs.
//
// Audit target: verifier and signature code MUST NOT reference forbidden
// tokens. Failures → red verdict, written to audit-report.{json,txt}.
//
// Verifier paths audited:
//   scripts/pvnp-phase1-verifier.mjs
//   scripts/pvnp-phase1-signatures.mjs
//   scripts/pvnp-phase1-ablations.mjs
//   scripts/lib/pvnp-phase1-verifier-core.mjs
//   scripts/lib/pvnp-phase1-signature-core.mjs
//
// Forbidden tokens (string-literal regex; case-sensitive):
//   ground_truth_labels   - never read by verifier code
//   hidden_state          - never read by verifier code
//   basin_params          - never read by verifier code
//   latent_field          - never read by verifier code
//   decoy_params          - never read by verifier code
//   B_theta               - notational shorthand for basin; forbidden
//   F_theta               - notational shorthand for latent field; forbidden
//   signedDistanceToBasin - privileged geometry function
//   evaluator-core        - importing the privileged evaluator module
//
// Allowed exception: the `redactEnv` helper destructures `hidden_state:
// _hidden` to drop it. The pattern `hidden_state: _hidden` appears in the
// allowlist below as an authorized reference (the redactor's job is to
// remove this field on the way in).

import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const AUDIT_VERSION = "pvnp-phase1-privilege-audit-v1";

const AUDIT_TARGETS = Object.freeze([
  "scripts/pvnp-phase1-verifier.mjs",
  "scripts/pvnp-phase1-signatures.mjs",
  "scripts/pvnp-phase1-ablations.mjs",
  "scripts/lib/pvnp-phase1-verifier-core.mjs",
  "scripts/lib/pvnp-phase1-signature-core.mjs",
  "scripts/lib/pvnp-phase1-cache.mjs",
]);

const FORBIDDEN_TOKENS = Object.freeze([
  "ground_truth_labels",
  "basin_params",
  "latent_field",
  "decoy_params",
  "B_theta",
  "F_theta",
  "signedDistanceToBasin",
  "evaluator-core",
]);

// The `hidden_state` token is forbidden EXCEPT inside the redactor pattern.
const REDACTOR_ALLOWED_PATTERN = /hidden_state\s*:\s*_hidden/;

function parseArgs(argv) {
  const args = { runDir: "results/pvnp/phase1-toy-verifier-v0" };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--run-dir") { args.runDir = argv[i + 1]; i += 1; }
    else throw new Error(`Unknown flag: ${argv[i]}`);
  }
  return args;
}

// Scan one file for forbidden tokens. Returns list of {token, line, lineText, ok_under_allowlist}.
function scanFile(text, filePath) {
  const matches = [];
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    // Skip pure comment lines so audit doesn't trip on the spec docstrings.
    // We consider a line "pure comment" if it's whitespace + // ... or starts a /* block.
    const trimmed = line.trim();
    const isCommentLine = trimmed.startsWith("//") || trimmed.startsWith("*");
    if (isCommentLine) continue;

    for (const tok of FORBIDDEN_TOKENS) {
      const idx = line.indexOf(tok);
      if (idx === -1) continue;
      matches.push({ token: tok, line: i + 1, lineText: line, ok_under_allowlist: false });
    }

    // hidden_state: only allowed in the redactor pattern.
    if (line.includes("hidden_state") && !isCommentLine) {
      const okUnderRedactor = REDACTOR_ALLOWED_PATTERN.test(line);
      matches.push({
        token: "hidden_state", line: i + 1, lineText: line,
        ok_under_allowlist: okUnderRedactor,
      });
    }
  }
  return matches;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(REPO_ROOT, args.runDir);
  await mkdir(outDir, { recursive: true });

  const fileResults = [];
  let totalViolations = 0;
  let totalAllowedRedactorHits = 0;

  for (const rel of AUDIT_TARGETS) {
    const abs = path.resolve(REPO_ROOT, rel);
    let text;
    try { text = await readFile(abs, "utf8"); }
    catch (e) {
      fileResults.push({ path: rel, status: "missing", error: String(e) });
      continue;
    }
    const matches = scanFile(text, rel);
    const violations = matches.filter((m) => !m.ok_under_allowlist);
    const allowed = matches.filter((m) => m.ok_under_allowlist);
    totalViolations += violations.length;
    totalAllowedRedactorHits += allowed.length;
    fileResults.push({
      path: rel,
      status: violations.length === 0 ? "clean" : "violation",
      violations,
      allowed_redactor_hits: allowed.length,
    });
  }

  const verdict = totalViolations === 0 ? "green" : "red";
  const report = {
    schema: AUDIT_VERSION,
    generated_at: new Date().toISOString(),
    verdict,
    total_violations: totalViolations,
    total_allowed_redactor_hits: totalAllowedRedactorHits,
    audit_targets: AUDIT_TARGETS,
    forbidden_tokens: FORBIDDEN_TOKENS,
    file_results: fileResults,
  };

  await writeFile(path.join(outDir, "audit-report.json"), JSON.stringify(report, null, 2) + "\n", "utf8");

  const lines = [
    `# pvnp-phase1 privilege audit`,
    `verdict: ${verdict}`,
    `total_violations: ${totalViolations}`,
    `allowed_redactor_hits: ${totalAllowedRedactorHits}`,
    ``,
    `audit targets:`,
    ...AUDIT_TARGETS.map((t) => `  - ${t}`),
    ``,
    `forbidden tokens:`,
    ...FORBIDDEN_TOKENS.map((t) => `  - ${t}`),
    `  - hidden_state (allowed only in redactor pattern)`,
    ``,
    `results:`,
  ];
  for (const r of fileResults) {
    lines.push(`  ${r.path}: ${r.status}`);
    if (r.violations && r.violations.length) {
      for (const v of r.violations) {
        lines.push(`    L${v.line} ${v.token}: ${v.lineText.trim()}`);
      }
    }
  }
  await writeFile(path.join(outDir, "audit-report.txt"), lines.join("\n") + "\n", "utf8");

  console.log(`audit verdict: ${verdict}; violations: ${totalViolations}; allowed redactor hits: ${totalAllowedRedactorHits}`);
  if (verdict !== "green") process.exit(2);
}

main().catch((err) => { console.error(err); process.exit(1); });
