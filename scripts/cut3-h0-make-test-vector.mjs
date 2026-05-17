#!/usr/bin/env node
// scripts/cut3-h0-make-test-vector.mjs
//
// One-shot generator for the cross-runtime canonicalization test vector
// pinned by the H0-2 schema (Finding-1 2026-05-16 audit-notes).
//
// The test vector is one synthetic sidecar object + the canonical string
// it produces + the calib_sha256 it self-pins to. Any alternative
// implementation of the canonicalization algorithm (the H0 measurement
// tool's browser-side canonicalJSON, the restored Node checker, any
// future re-port) MUST produce a byte-identical canonical string and a
// byte-identical sha256 for this sidecar. That equality is the
// load-bearing cross-runtime contract for red-line A's
// `record.provenance.source_sidecar_sha256 == sidecar.calib_sha256`
// mismatch detector.
//
// Output: results/structural-failure/cut3-prereg/h0-canonicalization-test-vector.json

import { mkdir, writeFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { canonicalize, sha256Hex, computeSidecarSelfPin } from "./lib/canonical-json.mjs";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(
  execSync("git rev-parse --show-toplevel", { encoding: "utf8", cwd: SCRIPT_DIR }).trim()
);
const OUT_REL = "results/structural-failure/cut3-prereg/h0-canonicalization-test-vector.json";

// The test sidecar. Deliberately constructed to exercise:
//   - nested objects (operator_decisions)
//   - arrays of objects in insertion order (anchors)
//   - mixed string / number / boolean / null leaves
//   - UTF-8 in a string (the degree symbol)
//   - keys in NON-sorted source order (to verify the canonicalizer sorts)
//
// The sidecar is conformant to §2-A of the H0-2 schema. Its
// `calib_sha256` field is set to "" at fixture-generation time;
// `computeSidecarSelfPin` recomputes the hex.
const TEST_SIDECAR = {
  // Intentionally unsorted top-level keys to exercise the sort:
  sidecar_kind: "h0-measured-sidecar",
  frame_id: "test_vector_v1",
  schema_version: 1,
  frame_path: "test_vector_v1.png",
  render_sha256: "0000000000000000000000000000000000000000000000000000000000000001",
  config_path: null,
  config_sha256: null,
  measurement_method: "operator_manual",
  measured_at_pt: "2026-05-16T00:00:00",
  measured_by: "test-vector",
  sun_px: [400, 400],
  projection: "operator_measured",
  theta_map_kind: "fit2locus",
  anchors: [
    { locus_deg: 22, px_radius: 200, off_ruler: false, measurement_note: "synthetic — 22° anchor at 200 px from sun" },
    { locus_deg: 46, px_radius: 400, off_ruler: false, measurement_note: "synthetic — 46° anchor at 400 px from sun (degree symbol)" },
  ],
  scored_feature_deg: 22,
  operator_decisions: {
    fixture_class: "corpus_candidate",
    compound_code_is_h_leak: "no",
    compound_code_basis: "test vector; no compound code in filename",
    known_pass_selection_basis: null,
  },
  calib_sha256: "",
};

async function main() {
  // Compute the canonical string and self-pin.
  const canonicalString = canonicalize({ ...TEST_SIDECAR, calib_sha256: "" });
  const calibSha256 = sha256Hex(canonicalString);
  const reCheck = computeSidecarSelfPin(TEST_SIDECAR);
  if (reCheck !== calibSha256) {
    throw new Error(`internal inconsistency: computeSidecarSelfPin=${reCheck} vs direct=${calibSha256}`);
  }

  // The pinned test sidecar includes the just-computed calib_sha256.
  const pinnedSidecar = { ...TEST_SIDECAR, calib_sha256: calibSha256 };

  const payload = {
    schema_version: 1,
    artifact: "h0-canonicalization-test-vector",
    purpose:
      "Cross-runtime canonicalization contract for the H0-2 schema sidecar self-pin. Any implementation of the canonicalization algorithm MUST produce byte-identical canonical_string and calib_sha256 for this input. The Node and browser implementations are required to agree; the restored H0 checker's first acceptance test on a Node-restore is `verifySidecarSelfPin(this_sidecar).ok === true`.",
    spec_reference:
      "docs/prereg/structural-failure-coincidence/P2_CUT3_H0_2_SCHEMA.md (Finding-1 audit-notes, 2026-05-16) and scripts/lib/canonical-json.mjs",
    canonicalization_algorithm_summary:
      "Recursive: arrays as `[v0,v1,...]` in order; objects as `{\"k0\":v0,\"k1\":v1,...}` with keys sorted lexicographically (by JS Object.keys().sort() = UTF-16 code-unit order, byte-equivalent for ASCII keys); leaves via JSON.stringify(value). No whitespace. UTF-8 output. See scripts/lib/canonical-json.mjs for the reference implementation.",
    self_pin_convention:
      "Set sidecar.calib_sha256 = \"\" (empty string, NOT null and NOT absent). canonicalize the result. SHA-256 the UTF-8 bytes. Hex-encode lowercase. Write back to sidecar.calib_sha256.",
    input_sidecar: TEST_SIDECAR,
    expected_canonical_string: canonicalString,
    expected_canonical_string_byte_length: Buffer.byteLength(canonicalString, "utf8"),
    expected_calib_sha256: calibSha256,
    pinned_sidecar_with_calib_sha256: pinnedSidecar,
    cross_runtime_consumers: [
      "scripts/cut2-publication-plumbing-guard.mjs canonicalize() (Node; predates this file but identical algorithm)",
      "scripts/cut3-h0-residual-table.mjs canonicalJSON() (Node; predates this file but identical algorithm)",
      "tools/h0-measurement/index.html canonicalJSON() (browser; inline copy of identical algorithm)",
      "scripts/cut3-h0-checker.mjs (when restored to canonical path; MUST adopt scripts/lib/canonical-json.mjs)",
    ],
    s0_acceptance_criterion:
      "The restored Node H0 checker (P2_CUT3_H0_CALIBRATION.md §0/C1 follow-up) MUST pass: import { verifySidecarSelfPin } from scripts/lib/canonical-json.mjs; verifySidecarSelfPin(pinned_sidecar_with_calib_sha256).ok === true. If it does not, red-line A's primary mismatch detector is broken at the canonicalization layer and Cut-3 stays blocked.",
  };

  const absOut = resolve(REPO, OUT_REL);
  await mkdir(dirname(absOut), { recursive: true });
  await writeFile(absOut, JSON.stringify(payload, null, 2) + "\n");

  console.log(`[test-vector] wrote ${OUT_REL}`);
  console.log(`[test-vector] canonical_string bytes : ${Buffer.byteLength(canonicalString, "utf8")}`);
  console.log(`[test-vector] expected_calib_sha256  : ${calibSha256}`);
  console.log("");
  console.log("Cross-runtime consumers MUST reproduce expected_calib_sha256 byte-for-byte.");
}

main().catch((e) => { console.error(`[test-vector] FAILED: ${e.message}`); process.exit(1); });
