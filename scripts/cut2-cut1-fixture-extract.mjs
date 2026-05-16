#!/usr/bin/env node
// scripts/cut2-cut1-fixture-extract.mjs
//
// One-shot extractor that hashes the Cut-1 fixture: the real, immutable
// objects inside `scripts/structural-failure-p2-harness.mjs` that the
// derived C4 audit must classify as `MACHINERY_LIVE_ROUTE_TEST_VACUOUS`.
//
// Writes:
//   results/structural-failure/cut2-prereg/cut1-fixture-manifest.json
//
// Records, per fixture object: name, role, line range, content sha256.
// Also records the whole-file sha256 + total line count for tamper-evidence.
//
// This is the C4-A "Cut-1 known-vacuous fixture = the actual Cut-1 objects"
// requirement made operational. The line ranges are visually verified at
// extraction time against the function-header grep; if the file is later
// edited (allowed only as new-line append below `runHarness`, never within
// fixture objects), the whole-file hash will move and re-extraction is
// required. Editing any of the fixture-object ranges below voids the C4-B
// self-test by changing the known-vacuous side of the two-sided audit.

import { createHash } from "node:crypto";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { resolve, dirname, relative } from "node:path";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const SCRIPT_PATH = fileURLToPath(import.meta.url);
const SCRIPT_DIR = dirname(SCRIPT_PATH);

const HARNESS_REL = "scripts/structural-failure-p2-harness.mjs";
const OUT_REL = "results/structural-failure/cut2-prereg/cut1-fixture-manifest.json";

// Fixture objects — name, role, [startLine, endLine] (1-indexed, inclusive).
// Line ranges verified 2026-05-16 against the function-header grep.
const FIXTURE_OBJECTS = [
  {
    name: "makeBundle",
    role: "C4-B fixture: bundle generator. Produces f_par = R22/cos(h) — the source of Cut-1's g(h) tautology. The thing that, paired with `routeEstimate`/`analyticInverseEstimate`, makes route ≡ analytic baseline by construction.",
    start: 154,
    end: 159,
  },
  {
    name: "transparentAdapter",
    role: "C4-B fixture: the A1 transparent adapter. Inputs `{f_par, f_cza, f_tan, R22, q}`; abstains on L1-ineligible bundles; emits objective `J = -|f_par - R22/cos(q)|`. Decoys are not inputs (structural exclusion — Cut-1 vacuity factor #2).",
    start: 175,
    end: 201,
  },
  {
    name: "routeEstimate",
    role: "C4-B fixture: the Cut-1 route (grid-search extremum seeker on `J`). Inverts `f_par = R22/cos(h)` by grid search — `g^-1(g(h))`. C4 D1 must show that route construction differs from true `h` on the must-differ band; for this fixture it does not (the famous Cut-1 vacuity).",
    start: 203,
    end: 237,
  },
  {
    name: "analyticInverseEstimate",
    role: "C4-B fixture: matched closed-form baseline `q = arccos(R22 / f_par)`. By design identical to the route on the eligible set — this is the C2-B/D1 'route ≡ analytic baseline' equality that motivates D1 comparing route vs TRUE `h`, not vs this baseline.",
    start: 239,
    end: 250,
  },
  {
    name: "positiveControlEstimate",
    role: "C4-B fixture: the decoy-correlate positive control (raw-bundle least-squares over all 8 features). Did move under decoy edits in Cut-1, yielding `OPAQUE_CORRELATE_POSITIVE_CONTROL_CONFIRMED` — the result whose validity stands even after Cut-1 was reclassified vacuous.",
    start: 282,
    end: 312,
  },
  {
    name: "routeConstructionAudit",
    role: "C4-B fixture: the HARDCODED Cut-1 vacuity audit. Returns `{routeTestVacuous: true, ...}` as constants — *asserted*, not *derived* from live objects. The thing C4 reimplements as a computed predicate set; this fixture is the negative side of the C4-B two-sided self-test.",
    start: 367,
    end: 376,
  },
  {
    name: "classifyRouteOutcome",
    role: "C4-B fixture: Cut-1 verdict classifier. Wraps the audit in the verdict tree (MACHINERY_LIVE_ROUTE_TEST_VACUOUS / CONVERGENCE_NULL / INCONCLUSIVE_DECOY_BATTERY / OPAQUE_CORRELATE / ROUTE_NOT_THE_INVERSE / TRACEABILITY_HARNESS_PASS). For C4-B audit replay, this is how the verdict actually surfaces — and why a hardcoded `routeTestVacuous: true` short-circuits to the vacuous verdict regardless of q1/q2/q3.",
    start: 378,
    end: 427,
  },
];

function repoRoot() {
  return resolve(
    execSync("git rev-parse --show-toplevel", {
      encoding: "utf8",
      cwd: SCRIPT_DIR,
    }).trim()
  );
}

function sha256(text) {
  return createHash("sha256").update(text).digest("hex");
}

async function main() {
  const repo = repoRoot();
  const harnessAbs = resolve(repo, HARNESS_REL);
  const fileText = await readFile(harnessAbs, "utf8");
  const fileBytes = Buffer.byteLength(fileText);
  const fileSha = sha256(fileText);
  const lines = fileText.split(/\r?\n/);

  const fixtures = FIXTURE_OBJECTS.map((f) => {
    if (f.start < 1 || f.end > lines.length || f.start > f.end) {
      throw new Error(
        `fixture ${f.name}: invalid line range ${f.start}-${f.end} (file has ${lines.length} lines)`
      );
    }
    const slice = lines.slice(f.start - 1, f.end).join("\n");
    const firstLine = lines[f.start - 1];
    return {
      name: f.name,
      role: f.role,
      lines_inclusive: [f.start, f.end],
      first_line_marker: firstLine.trim(),
      content_sha256: sha256(slice),
      content_length_bytes: Buffer.byteLength(slice),
      content_line_count: f.end - f.start + 1,
    };
  });

  const manifest = {
    manifest_version: 1,
    frozen_at_pt: "2026-05-16",
    purpose:
      "C4-A operational artifact: hashable Cut-1 known-vacuous fixture manifest. The real `structural-failure-p2-harness.mjs` objects whose collective behavior the derived C4 audit must classify `MACHINERY_LIVE_ROUTE_TEST_VACUOUS`. C4-B is the two-sided self-test: same audit on this fixture must return vacuous, on the minimal-flip fixture must return non-vacuous.",
    spec: "docs/prereg/structural-failure-coincidence/P2_CUT2_C4A_AUDIT_FREEZE.md",
    source_file: HARNESS_REL,
    source_file_sha256: fileSha,
    source_file_byte_count: fileBytes,
    source_file_line_count: lines.length,
    source_file_immutability:
      "The Cut-1 fixture is fixed real artifact (C4-A §6 [G]). The named objects below MUST NOT be edited; doing so voids the C4-B self-test by changing the known-vacuous side of the two-sided audit. Any modification within these line ranges requires an append-only redesign of C4-B, not a manifest amendment.",
    fixture_objects: fixtures,
    fixture_vacuity_summary: {
      route_equals_analytic_baseline_by_construction: true,
      decoys_outside_route_objective: true,
      cza_tangent_do_not_affect_q_estimate: true,
      supralateral_hardcoded_as_non_handle: true,
      route_construction_audit_hardcoded: true,
      decoy_correlate_positive_control_moves: true,
      summary:
        "Cut-1 is g^-1(g(h)) by grid search with a hardcoded vacuity audit; the positive control moves (correlate path exists) but the route side cannot fail the structural test by construction. This is exactly the case the derived C4 audit must catch.",
    },
    extractor_script: "scripts/cut2-cut1-fixture-extract.mjs",
    rerun_command: "node scripts/cut2-cut1-fixture-extract.mjs",
  };

  const outAbs = resolve(repo, OUT_REL);
  await mkdir(dirname(outAbs), { recursive: true });
  await writeFile(outAbs, JSON.stringify(manifest, null, 2) + "\n");

  console.log(`[cut1-fixture-extract] wrote ${relative(repo, outAbs)}`);
  console.log(`[cut1-fixture-extract] source_file_sha256 = ${fileSha}`);
  for (const f of fixtures) {
    console.log(
      `[cut1-fixture-extract]   ${f.name.padEnd(28)} L${f.lines_inclusive[0]}-${f.lines_inclusive[1]}  ${f.content_sha256}`
    );
  }
}

main().catch((err) => {
  console.error(`[cut1-fixture-extract] FAILED: ${err.message}`);
  process.exit(1);
});
