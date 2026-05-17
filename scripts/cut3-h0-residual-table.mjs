#!/usr/bin/env node
// scripts/cut3-h0-residual-table.mjs
//
// Wave H0-2 §6.2 — residual-table generator for the Cut-3 H0-2
// measured-sidecar + residual-table schema
// (docs/prereg/structural-failure-coincidence/P2_CUT3_H0_2_SCHEMA.md).
//
// Aggregates artifact A (measured sidecars at h0-sidecars/<frame_id>.json)
// + artifact B (H0 records at h0-records/<frame_id>.json) into artifact C
// (h0-anchor-residual-table.{json,csv}) and the artifact D manifest
// (h0-2-manifest.json).
//
// §3 anti-self-seal at the generator layer
// ------------------------------------------------------------------------
// The verdict-before-measurement hazard at THIS layer is
// "table-rewrites-history": the generator silently "fixes" a row whose
// inputs look wrong. Two structural guards make this impossible:
//
//  (1) NO VERDICT SYNTHESIS. `admit`, `reason_code`, `measured_deg`,
//      `residual_deg`, `off_ruler` are copied byte-for-byte from artifact
//      B (the H0 record produced by the canonical checker). The
//      generator never computes a verdict and never fills in a missing
//      anchor measurement.
//
//  (2) CONSISTENCY-FAIL-LOUD. For each (sidecar, record) pair the
//      generator performs three pin checks:
//        (a) Self-pin: recompute sidecar.calib_sha256 over canonical-JSON
//            with the field reset to "" and verify it matches the stored
//            calib_sha256. (Sidecar tampering detector.)
//        (b) Record-to-sidecar pin: record.provenance.source_sidecar_sha256
//            must equal sidecar.calib_sha256. (Schema §3 PRIMARY
//            mechanical detector per red-line A.)
//        (c) Timestamp ordering: sidecar.measured_at_pt <
//            record.provenance.checker_runtime_pt. (Schema §3 secondary
//            defense-in-depth; lexicographic ISO-8601 compare.)
//      Any failing check ⇒ row emitted with `consistency: false` plus the
//      specific failure code; the generator NEVER silently corrects a
//      row. Orphan sidecars/records (no matching counterpart) emit rows
//      with `reason_code = ORPHAN_RECORD` or `ORPHAN_SIDECAR`.
//
// Self-test mode covers ONLY the generator's plumbing (parsing, schema
// enforcement, row/CSV emission, consistency-detector firing) on minimal
// synthetic conformant inputs. It makes NO claim about H0-2's
// substantive correctness — that comes only after §6.3–§6.4 produce real
// operator-pre-filled sidecars.

import { createHash } from "node:crypto";
import { readFile, readdir, mkdir, writeFile, stat } from "node:fs/promises";
import { resolve, dirname, basename, join, relative, sep } from "node:path";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(
  execSync("git rev-parse --show-toplevel", { encoding: "utf8", cwd: SCRIPT_DIR }).trim()
);

const SIDECARS_DIR_REL = "results/structural-failure/cut3-prereg/h0-sidecars";
const RECORDS_DIR_REL = "results/structural-failure/cut3-prereg/h0-records";
const TABLE_JSON_REL = "results/structural-failure/cut3-prereg/h0-anchor-residual-table.json";
const TABLE_CSV_REL = "results/structural-failure/cut3-prereg/h0-anchor-residual-table.csv";
const MANIFEST_REL = "results/structural-failure/cut3-prereg/h0-2-manifest.json";
const SELF_TEST_OUT_REL = "results/structural-failure/cut3-prereg/h0-residual-table-self-test-result.json";
const DEFAULT_OUT_DIR_REL = "results/structural-failure/cut3-prereg";

const ANCHOR_RESIDUAL_TOL_DEG = 0.5; // [E] inherited from P2_CUT3_RUN_SPEC.md

// CSV column order — pinned to match P2_CUT3_H0_2_SCHEMA.md §2-C verbatim,
// plus the consistency tail added by §3 anti-self-seal at this layer.
const CSV_COLUMNS = [
  "frame_id",
  "frame_class",
  "anchor_locus_deg",
  "measured_deg",
  "residual_deg",
  "off_ruler",
  "within_tolerance",
  "admit",
  "reason_code",
  "operator_decision_codes",
  "png_sha256",
  "sidecar_sha256",
  "h0_record_sha256",
  "consistency",
  "consistency_failure_codes",
];

// ============================================================================
// Hash + canonical JSON utilities
// ============================================================================

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function toPosix(p) {
  return p.split(sep).join("/");
}

function canonicalJSON(value) {
  if (Array.isArray(value)) return "[" + value.map(canonicalJSON).join(",") + "]";
  if (value !== null && typeof value === "object") {
    const keys = Object.keys(value).sort();
    return (
      "{" +
      keys.map((k) => JSON.stringify(k) + ":" + canonicalJSON(value[k])).join(",") +
      "}"
    );
  }
  return JSON.stringify(value);
}

async function hashFile(absPath) {
  const bytes = await readFile(absPath);
  return sha256(bytes);
}

// Self-pin recomputation: canonical-JSON the sidecar with calib_sha256 = ""
// then sha256 the bytes. Compare against sidecar.calib_sha256.
function recomputeSidecarSelfPin(sidecar) {
  const copy = { ...sidecar, calib_sha256: "" };
  return sha256(Buffer.from(canonicalJSON(copy)));
}

// ============================================================================
// Schema validation (mechanical; flags missing/wrong-typed required fields)
// ============================================================================

const SIDECAR_REQUIRED_TOP = [
  "schema_version",
  "sidecar_kind",
  "frame_id",
  "frame_path",
  "render_sha256",
  "measurement_method",
  "measured_at_pt",
  "measured_by",
  "sun_px",
  "projection",
  "theta_map_kind",
  "anchors",
  "operator_decisions",
  "calib_sha256",
];
const SIDECAR_REQUIRED_OPERATOR_DECISIONS = [
  "compound_code_is_h_leak",
  "compound_code_basis",
  "known_pass_selection_basis", // red-line B field
  "fixture_class",
];
const VALID_THETA_MAP_KINDS = new Set(["scale_ticks", "renderer_metadata", "fit2locus"]);
const VALID_FIXTURE_CLASSES = new Set([
  "phase15_known_fail",
  "known_pass_fullspan",
  "corpus_candidate",
]);

function validateSidecar(sidecar) {
  const issues = [];
  if (sidecar.sidecar_kind !== "h0-measured-sidecar") {
    issues.push(`sidecar_kind must be "h0-measured-sidecar", got ${JSON.stringify(sidecar.sidecar_kind)}`);
  }
  for (const k of SIDECAR_REQUIRED_TOP) {
    if (!(k in sidecar)) issues.push(`missing required field: ${k}`);
  }
  if (sidecar.theta_map_kind !== undefined && !VALID_THETA_MAP_KINDS.has(sidecar.theta_map_kind)) {
    issues.push(`theta_map_kind invalid: ${JSON.stringify(sidecar.theta_map_kind)}`);
  }
  if (sidecar.operator_decisions !== undefined) {
    for (const k of SIDECAR_REQUIRED_OPERATOR_DECISIONS) {
      if (!(k in sidecar.operator_decisions)) {
        issues.push(`missing operator_decisions.${k}`);
      }
    }
    const fc = sidecar.operator_decisions.fixture_class;
    if (fc !== undefined && !VALID_FIXTURE_CLASSES.has(fc)) {
      issues.push(`operator_decisions.fixture_class invalid: ${JSON.stringify(fc)}`);
    }
  }
  if (!Array.isArray(sidecar.anchors)) {
    issues.push("anchors must be an array");
  } else {
    for (let i = 0; i < sidecar.anchors.length; i++) {
      const a = sidecar.anchors[i];
      if (typeof a.locus_deg !== "number") issues.push(`anchors[${i}].locus_deg must be a number`);
    }
  }
  return issues;
}

const RECORD_REQUIRED_TOP = [
  "frame_id",
  "valid_angular_span_deg",
  "anchors",
  "admit",
  "reason_code",
];
const RECORD_REQUIRED_PROVENANCE = [
  "source_sidecar_sha256",
  "checker_sha256",
  "checker_runtime_pt",
];

function validateRecord(record) {
  const issues = [];
  for (const k of RECORD_REQUIRED_TOP) {
    if (!(k in record)) issues.push(`missing required field: ${k}`);
  }
  if (record.provenance === undefined || record.provenance === null) {
    issues.push("missing provenance block");
  } else {
    for (const k of RECORD_REQUIRED_PROVENANCE) {
      if (!(k in record.provenance)) issues.push(`missing provenance.${k}`);
    }
  }
  if (!Array.isArray(record.anchors)) {
    issues.push("anchors must be an array");
  }
  return issues;
}

// ============================================================================
// Consistency checks (the §3 anti-self-seal at the generator layer)
// ============================================================================

function checkConsistency(sidecar, record) {
  // Returns { ok, failure_codes }. Never silently corrects.
  const failure_codes = [];

  // (a) sidecar self-pin
  const recomputed = recomputeSidecarSelfPin(sidecar);
  if (recomputed !== sidecar.calib_sha256) {
    failure_codes.push("SIDECAR_SELF_PIN_MISMATCH");
  }

  // (b) record-to-sidecar pin — PRIMARY (red-line A)
  if (record.provenance?.source_sidecar_sha256 !== sidecar.calib_sha256) {
    failure_codes.push("RECORD_SIDECAR_SHA_MISMATCH");
  }

  // (c) timestamp ordering — secondary defense-in-depth
  if (
    typeof sidecar.measured_at_pt === "string" &&
    typeof record.provenance?.checker_runtime_pt === "string"
  ) {
    if (!(sidecar.measured_at_pt < record.provenance.checker_runtime_pt)) {
      failure_codes.push("TIMESTAMP_ORDERING_VIOLATION");
    }
  } else {
    failure_codes.push("TIMESTAMP_FIELD_MISSING_OR_NON_STRING");
  }

  return { ok: failure_codes.length === 0, failure_codes };
}

// ============================================================================
// Row emission — copies verdicts BYTE-FOR-BYTE from record; no synthesis
// ============================================================================

function emitRowsFromPair(sidecar, record, hashes) {
  // Per (frame, anchor) row. Pulls measured_deg / residual_deg / off_ruler /
  // admit / reason_code directly from the record. Operator-decision codes
  // are flattened from sidecar.operator_decisions for the audit column.
  const consistency = checkConsistency(sidecar, record);
  const frame_class = sidecar.operator_decisions?.fixture_class ?? null;
  const operator_decision_codes = sidecar.operator_decisions
    ? [
        `compound_code_is_h_leak=${sidecar.operator_decisions.compound_code_is_h_leak}`,
        `fixture_class=${sidecar.operator_decisions.fixture_class}`,
      ].join(";")
    : "";

  const rows = [];
  // Use record.anchors (which carries the computed measured_deg) as the
  // source of truth for per-anchor numbers. Sidecar anchors describe what
  // the operator measured in pixel space; record anchors describe what
  // the checker computed in degree space.
  for (const a of record.anchors ?? []) {
    rows.push({
      frame_id: record.frame_id ?? sidecar.frame_id,
      frame_class,
      anchor_locus_deg: a.locus_deg,
      measured_deg: a.measured_deg ?? null,
      residual_deg: a.residual_deg ?? null,
      off_ruler: !!a.off_ruler,
      within_tolerance:
        a.residual_deg !== null && a.residual_deg !== undefined
          ? a.residual_deg <= ANCHOR_RESIDUAL_TOL_DEG
          : false,
      admit: record.admit ?? null,
      reason_code: record.reason_code ?? null,
      operator_decision_codes,
      png_sha256: hashes.png_sha256 ?? null,
      sidecar_sha256: sidecar.calib_sha256 ?? null,
      h0_record_sha256: hashes.h0_record_sha256 ?? null,
      consistency: consistency.ok,
      consistency_failure_codes: consistency.failure_codes.join(";"),
    });
  }
  return rows;
}

function emitOrphanSidecarRows(sidecar, hashes) {
  // Sidecar without matching record. One row per declared anchor; admit/
  // reason_code synthesized as ORPHAN_RECORD (this is NOT a verdict claim
  // — it's a structural marker that the record is missing).
  const frame_class = sidecar.operator_decisions?.fixture_class ?? null;
  const operator_decision_codes = sidecar.operator_decisions
    ? `compound_code_is_h_leak=${sidecar.operator_decisions.compound_code_is_h_leak};fixture_class=${sidecar.operator_decisions.fixture_class}`
    : "";
  const rows = [];
  for (const a of sidecar.anchors ?? []) {
    rows.push({
      frame_id: sidecar.frame_id,
      frame_class,
      anchor_locus_deg: a.locus_deg,
      measured_deg: null,
      residual_deg: null,
      off_ruler: !!a.off_ruler,
      within_tolerance: false,
      admit: null,
      reason_code: "ORPHAN_RECORD",
      operator_decision_codes,
      png_sha256: hashes.png_sha256 ?? null,
      sidecar_sha256: sidecar.calib_sha256 ?? null,
      h0_record_sha256: null,
      consistency: false,
      consistency_failure_codes: "RECORD_MISSING",
    });
  }
  return rows;
}

function emitOrphanRecordRows(record, hashes) {
  const rows = [];
  for (const a of record.anchors ?? []) {
    rows.push({
      frame_id: record.frame_id ?? null,
      frame_class: null,
      anchor_locus_deg: a.locus_deg,
      measured_deg: a.measured_deg ?? null,
      residual_deg: a.residual_deg ?? null,
      off_ruler: !!a.off_ruler,
      within_tolerance:
        a.residual_deg !== null && a.residual_deg !== undefined
          ? a.residual_deg <= ANCHOR_RESIDUAL_TOL_DEG
          : false,
      admit: record.admit ?? null,
      reason_code: "ORPHAN_SIDECAR",
      operator_decision_codes: "",
      png_sha256: null,
      sidecar_sha256: null,
      h0_record_sha256: hashes.h0_record_sha256 ?? null,
      consistency: false,
      consistency_failure_codes: "SIDECAR_MISSING",
    });
  }
  return rows;
}

// ============================================================================
// CSV output
// ============================================================================

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "boolean") return value ? "true" : "false";
  const s = String(value);
  if (/[",\r\n]/.test(s)) return '"' + s.replaceAll('"', '""') + '"';
  return s;
}

function rowsToCSV(rows) {
  const lines = [CSV_COLUMNS.join(",")];
  for (const r of rows) {
    lines.push(CSV_COLUMNS.map((c) => csvEscape(r[c])).join(","));
  }
  return lines.join("\n") + "\n";
}

// ============================================================================
// Load directories
// ============================================================================

async function loadJsonDir(absDir) {
  let entries;
  try {
    entries = await readdir(absDir);
  } catch (e) {
    if (e.code === "ENOENT") return [];
    throw e;
  }
  const out = [];
  for (const name of entries.sort()) {
    if (!name.endsWith(".json")) continue;
    const abs = join(absDir, name);
    const bytes = await readFile(abs);
    const parsed = JSON.parse(bytes.toString("utf8"));
    out.push({
      filename: name,
      abs_path: abs,
      json: parsed,
      raw_sha256: sha256(bytes),
    });
  }
  return out;
}

// ============================================================================
// Generate the table
// ============================================================================

async function generate({ sidecarsDir, recordsDir, outDir }) {
  const absSidecars = resolve(REPO, sidecarsDir);
  const absRecords = resolve(REPO, recordsDir);
  const sidecars = await loadJsonDir(absSidecars);
  const records = await loadJsonDir(absRecords);

  // Pair by frame_id
  const sidecarByFrame = new Map();
  const recordByFrame = new Map();
  for (const s of sidecars) {
    const fid = s.json.frame_id;
    if (sidecarByFrame.has(fid)) {
      throw new Error(`duplicate sidecar frame_id: ${fid}`);
    }
    sidecarByFrame.set(fid, s);
  }
  for (const r of records) {
    const fid = r.json.frame_id;
    if (recordByFrame.has(fid)) {
      throw new Error(`duplicate record frame_id: ${fid}`);
    }
    recordByFrame.set(fid, r);
  }

  const allFrameIds = new Set([...sidecarByFrame.keys(), ...recordByFrame.keys()]);
  const sortedFrameIds = [...allFrameIds].sort();

  const allRows = [];
  const inputs_summary = {
    sidecars: sidecars.length,
    records: records.length,
    paired: 0,
    orphan_sidecars: 0,
    orphan_records: 0,
    consistency_failures: 0,
  };

  for (const fid of sortedFrameIds) {
    const s = sidecarByFrame.get(fid);
    const r = recordByFrame.get(fid);
    if (s && r) {
      // Read PNG hash from sidecar's render_sha256 (operator-pinned at
      // measurement time); read record hash from r.raw_sha256
      const hashes = {
        png_sha256: s.json.render_sha256 ?? null,
        h0_record_sha256: r.raw_sha256,
      };
      const validation_issues = [...validateSidecar(s.json), ...validateRecord(r.json).map((x) => "record:" + x)];
      if (validation_issues.length > 0) {
        // Validation is a fatal kind of consistency failure — emit a
        // single placeholder row with the issues; do NOT abort.
        allRows.push({
          frame_id: fid,
          frame_class: null,
          anchor_locus_deg: null,
          measured_deg: null,
          residual_deg: null,
          off_ruler: false,
          within_tolerance: false,
          admit: null,
          reason_code: "SCHEMA_VALIDATION_FAILED",
          operator_decision_codes: "",
          png_sha256: null,
          sidecar_sha256: null,
          h0_record_sha256: null,
          consistency: false,
          consistency_failure_codes: validation_issues.join("|"),
        });
        inputs_summary.consistency_failures++;
        continue;
      }
      const rows = emitRowsFromPair(s.json, r.json, hashes);
      for (const row of rows) {
        if (!row.consistency) inputs_summary.consistency_failures++;
      }
      allRows.push(...rows);
      inputs_summary.paired++;
    } else if (s && !r) {
      const validation_issues = validateSidecar(s.json);
      const hashes = {
        png_sha256: s.json.render_sha256 ?? null,
      };
      if (validation_issues.length > 0) {
        allRows.push({
          frame_id: fid,
          frame_class: null,
          anchor_locus_deg: null,
          measured_deg: null,
          residual_deg: null,
          off_ruler: false,
          within_tolerance: false,
          admit: null,
          reason_code: "SCHEMA_VALIDATION_FAILED",
          operator_decision_codes: "",
          png_sha256: null,
          sidecar_sha256: null,
          h0_record_sha256: null,
          consistency: false,
          consistency_failure_codes: validation_issues.join("|"),
        });
        inputs_summary.consistency_failures++;
        continue;
      }
      allRows.push(...emitOrphanSidecarRows(s.json, hashes));
      inputs_summary.orphan_sidecars++;
    } else if (!s && r) {
      const hashes = { h0_record_sha256: r.raw_sha256 };
      allRows.push(...emitOrphanRecordRows(r.json, hashes));
      inputs_summary.orphan_records++;
    }
  }

  // Sort rows deterministically: (frame_id, anchor_locus_deg)
  allRows.sort((a, b) => {
    const fa = a.frame_id ?? "";
    const fb = b.frame_id ?? "";
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    const la = a.anchor_locus_deg ?? -Infinity;
    const lb = b.anchor_locus_deg ?? -Infinity;
    return la - lb;
  });

  const tablePayload = {
    schema_version: 1,
    artifact: "h0-anchor-residual-table",
    spec_reference: "docs/prereg/structural-failure-coincidence/P2_CUT3_H0_2_SCHEMA.md §2-C",
    generator: "scripts/cut3-h0-residual-table.mjs",
    generator_disclosure:
      "Aggregator only — no verdict synthesis. admit/reason_code/measured_deg/residual_deg are byte-for-byte from artifact B (H0 records). Consistency-fail-loud: pin/timestamp mismatches emit rows with consistency=false and specific failure codes; the generator never silently corrects.",
    anchor_residual_tolerance_deg: ANCHOR_RESIDUAL_TOL_DEG,
    inputs_summary,
    row_count: allRows.length,
    rows: allRows,
  };

  // Write outputs
  const absOutDir = resolve(REPO, outDir ?? "results/structural-failure/cut3-prereg");
  await mkdir(absOutDir, { recursive: true });
  const tableJsonAbs = join(absOutDir, basename(TABLE_JSON_REL));
  const tableCsvAbs = join(absOutDir, basename(TABLE_CSV_REL));
  const manifestAbs = join(absOutDir, basename(MANIFEST_REL));
  const tableJsonRel = toPosix(relative(REPO, tableJsonAbs));
  const tableCsvRel = toPosix(relative(REPO, tableCsvAbs));
  const manifestRel = toPosix(relative(REPO, manifestAbs));
  const tableJsonPretty = JSON.stringify(tablePayload, null, 2) + "\n";
  await writeFile(tableJsonAbs, tableJsonPretty);
  const csvText = rowsToCSV(allRows);
  await writeFile(tableCsvAbs, csvText);

  // Manifest (artifact D)
  const manifest = {
    schema_version: 1,
    artifact: "h0-2-manifest",
    spec_reference: "docs/prereg/structural-failure-coincidence/P2_CUT3_H0_2_SCHEMA.md §2-D",
    sidecars: sidecars.map((s) => ({
      frame_id: s.json.frame_id,
      path: toPosix(relative(REPO, s.abs_path)),
      raw_sha256: s.raw_sha256,
      calib_sha256: s.json.calib_sha256 ?? null,
    })),
    records: records.map((r) => ({
      frame_id: r.json.frame_id,
      path: toPosix(relative(REPO, r.abs_path)),
      raw_sha256: r.raw_sha256,
      source_sidecar_sha256: r.json.provenance?.source_sidecar_sha256 ?? null,
      checker_sha256: r.json.provenance?.checker_sha256 ?? null,
    })),
    table: {
      json_path: tableJsonRel,
      json_sha256: sha256(Buffer.from(tableJsonPretty)),
      json_canonical_sha256: sha256(Buffer.from(canonicalJSON(tablePayload))),
      csv_path: tableCsvRel,
      csv_sha256: sha256(Buffer.from(csvText)),
    },
    inputs_summary,
  };
  const manifestPretty = JSON.stringify(manifest, null, 2) + "\n";
  await writeFile(manifestAbs, manifestPretty);

  return {
    tablePayload,
    manifest,
    tableJsonPretty,
    csvText,
    manifestPretty,
    outputPaths: { tableJsonRel, tableCsvRel, manifestRel },
  };
}

// ============================================================================
// Self-test — schema-mechanical only; synthetic conformant inputs
// ============================================================================
//
// This self-test exercises the generator's PLUMBING, not its substantive
// claims. The synthetic inputs are not H0 records — they are unit-test
// fixtures with the schema shape. PASS here means: the parser, schema
// validator, consistency detector, row emitter, CSV column order, and
// orphan handling all behave as specified. PASS does NOT mean H0-2 is
// closed; that requires §6.3–§6.4 operator pre-fill.

function buildSyntheticPair() {
  // Conformant sidecar and matching record. Used for the "happy path" check.
  const sidecar = {
    schema_version: 1,
    sidecar_kind: "h0-measured-sidecar",
    frame_id: "selftest_pair",
    frame_path: "selftest/pair.png",
    render_sha256: "0000000000000000000000000000000000000000000000000000000000000001",
    config_path: null,
    config_sha256: null,
    measurement_method: "operator_manual",
    measured_at_pt: "2026-05-16T00:00:00",
    measured_by: "selftest",
    sun_px: [400, 400],
    projection: "selftest",
    theta_map_kind: "scale_ticks",
    scale_ticks: [
      { px_radius: 100, deg: 11, source: "operator-eye" },
      { px_radius: 200, deg: 22, source: "operator-eye" },
      { px_radius: 400, deg: 46, source: "operator-eye" },
    ],
    anchors: [
      { locus_deg: 22, px_radius: 200, off_ruler: false, measurement_note: "" },
      { locus_deg: 46, px_radius: 400, off_ruler: false, measurement_note: "" },
    ],
    scored_feature_deg: 22.0,
    operator_decisions: {
      compound_code_is_h_leak: "no",
      compound_code_basis: "no compound code present",
      known_pass_selection_basis: "synthetic selftest fixture",
      fixture_class: "known_pass_fullspan",
    },
    calib_sha256: "",
  };
  sidecar.calib_sha256 = recomputeSidecarSelfPin(sidecar);

  const record = {
    frame_id: "selftest_pair",
    valid_angular_span_deg: 46,
    anchors: [
      { locus_deg: 22, measured_deg: 22, residual_deg: 0, off_ruler: false },
      { locus_deg: 46, measured_deg: 46, residual_deg: 0, off_ruler: false },
    ],
    admit: true,
    reason_code: "OK",
    provenance: {
      source_sidecar_sha256: sidecar.calib_sha256,
      checker_sha256: "0000000000000000000000000000000000000000000000000000000000000002",
      checker_runtime_pt: "2026-05-16T00:01:00",
    },
  };
  return { sidecar, record };
}

function buildShaMismatchPair() {
  // Same as happy-path, but the record's source_sidecar_sha256 has been
  // tampered. Generator MUST emit rows with consistency=false and
  // RECORD_SIDECAR_SHA_MISMATCH.
  const { sidecar, record } = buildSyntheticPair();
  sidecar.frame_id = "selftest_sha_mismatch";
  record.frame_id = "selftest_sha_mismatch";
  sidecar.calib_sha256 = recomputeSidecarSelfPin(sidecar);
  record.provenance.source_sidecar_sha256 = "f".repeat(64); // tampered
  return { sidecar, record };
}

function buildTimestampOrderingViolation() {
  const { sidecar, record } = buildSyntheticPair();
  sidecar.frame_id = "selftest_timestamp";
  record.frame_id = "selftest_timestamp";
  sidecar.calib_sha256 = recomputeSidecarSelfPin(sidecar);
  record.provenance.source_sidecar_sha256 = sidecar.calib_sha256;
  // Set checker_runtime_pt BEFORE measured_at_pt
  record.provenance.checker_runtime_pt = "2025-01-01T00:00:00";
  return { sidecar, record };
}

function buildOrphanSidecar() {
  const { sidecar } = buildSyntheticPair();
  sidecar.frame_id = "selftest_orphan_sidecar";
  sidecar.calib_sha256 = recomputeSidecarSelfPin(sidecar);
  return sidecar;
}

function buildOrphanRecord() {
  const { record } = buildSyntheticPair();
  record.frame_id = "selftest_orphan_record";
  return record;
}

async function selfTest() {
  const tmpDir = resolve(REPO, "results/structural-failure/cut3-prereg/h0-residual-selftest-tmp");
  const sidecarsDir = join(tmpDir, "h0-sidecars");
  const recordsDir = join(tmpDir, "h0-records");
  await mkdir(sidecarsDir, { recursive: true });
  await mkdir(recordsDir, { recursive: true });

  // Build synthetic conformant inputs for the four cases
  const happy = buildSyntheticPair();
  const shaBad = buildShaMismatchPair();
  const tsBad = buildTimestampOrderingViolation();
  const orphanS = buildOrphanSidecar();
  const orphanR = buildOrphanRecord();

  // Write them
  async function writeSidecar(s) {
    await writeFile(join(sidecarsDir, `${s.frame_id}.json`), JSON.stringify(s, null, 2) + "\n");
  }
  async function writeRecord(r) {
    await writeFile(join(recordsDir, `${r.frame_id}.json`), JSON.stringify(r, null, 2) + "\n");
  }

  await writeSidecar(happy.sidecar);
  await writeRecord(happy.record);
  await writeSidecar(shaBad.sidecar);
  await writeRecord(shaBad.record);
  await writeSidecar(tsBad.sidecar);
  await writeRecord(tsBad.record);
  await writeSidecar(orphanS);
  await writeRecord(orphanR);

  // Run the generator over the tmp dirs
  const result = await generateLocal({ sidecarsDir, recordsDir });

  // Verify assertions
  const checks = [];
  function assertPredicate(name, fn) {
    let ok = false;
    let detail = "";
    try {
      ok = !!fn();
    } catch (e) {
      detail = `threw: ${e.message}`;
    }
    checks.push({ name, ok, detail });
  }

  const rows = result.tablePayload.rows;
  const byFrame = new Map();
  for (const r of rows) {
    const list = byFrame.get(r.frame_id) ?? [];
    list.push(r);
    byFrame.set(r.frame_id, list);
  }

  assertPredicate(
    "happy_path_two_rows",
    () => byFrame.get("selftest_pair")?.length === 2,
  );
  assertPredicate(
    "happy_path_both_consistent",
    () => byFrame.get("selftest_pair")?.every((r) => r.consistency === true),
  );
  assertPredicate(
    "happy_path_admit_copied_byte_for_byte",
    () => byFrame.get("selftest_pair")?.every((r) => r.admit === true && r.reason_code === "OK"),
  );

  assertPredicate(
    "sha_mismatch_consistency_false",
    () => byFrame.get("selftest_sha_mismatch")?.every((r) => r.consistency === false),
  );
  assertPredicate(
    "sha_mismatch_emits_RECORD_SIDECAR_SHA_MISMATCH",
    () =>
      byFrame.get("selftest_sha_mismatch")?.every((r) =>
        r.consistency_failure_codes.split(";").includes("RECORD_SIDECAR_SHA_MISMATCH"),
      ),
  );

  assertPredicate(
    "timestamp_ordering_violation_caught",
    () =>
      byFrame.get("selftest_timestamp")?.every((r) =>
        r.consistency_failure_codes.split(";").includes("TIMESTAMP_ORDERING_VIOLATION"),
      ),
  );

  assertPredicate(
    "orphan_sidecar_emits_ORPHAN_RECORD",
    () =>
      byFrame.get("selftest_orphan_sidecar")?.every((r) => r.reason_code === "ORPHAN_RECORD"),
  );
  assertPredicate(
    "orphan_record_emits_ORPHAN_SIDECAR",
    () =>
      byFrame.get("selftest_orphan_record")?.every((r) => r.reason_code === "ORPHAN_SIDECAR"),
  );

  // CSV column-order check
  const csvLines = result.csvText.split("\n");
  assertPredicate(
    "csv_header_matches_spec",
    () => csvLines[0] === CSV_COLUMNS.join(","),
  );

  // No-verdict-synthesis check: the generator never CHANGED an admit value
  // (always copied from record). For happy path the record's admit was true;
  // for orphan/mismatch the generator marked admit appropriately without
  // inventing a verdict.
  assertPredicate(
    "no_verdict_synthesis_on_happy_path",
    () => byFrame.get("selftest_pair")?.every((r) => r.admit === true),
  );
  assertPredicate(
    "orphan_sidecar_admit_null",
    () => byFrame.get("selftest_orphan_sidecar")?.every((r) => r.admit === null),
  );

  const pass_count = checks.filter((c) => c.ok).length;
  const fail_count = checks.length - pass_count;
  const overall_pass = fail_count === 0;

  const payload = {
    self_test: "Residual-table generator plumbing (schema-mechanical only)",
    spec_reference:
      "P2_CUT3_H0_2_SCHEMA.md §2-C, §3, §6.2 — exercises parser, validator, consistency-detector, row emitter, CSV column order, orphan handling on synthetic conformant inputs.",
    disclosure:
      "PASS here means the generator's plumbing behaves as specified. It does NOT mean H0-2 is closed; H0-2 closure requires §6.3–§6.4 operator pre-fill on real Phase-15 frames + a known-PASS fixture. Synthetic test inputs are unit-test fixtures, not fabricated H0 records.",
    tmp_dir_used: toPosix(relative(REPO, tmpDir)),
    pass_count,
    fail_count,
    overall_pass,
    checks,
  };

  const outAbs = resolve(REPO, SELF_TEST_OUT_REL);
  await mkdir(dirname(outAbs), { recursive: true });
  await writeFile(outAbs, JSON.stringify(payload, null, 2) + "\n");

  console.log(`[h0-residual-table self-test] wrote ${SELF_TEST_OUT_REL}`);
  console.log(`[h0-residual-table self-test] ${pass_count}/${checks.length} checks pass`);
  for (const c of checks) {
    const tag = c.ok ? "OK   " : "FAIL ";
    console.log(`  [${tag}] ${c.name}${c.detail ? "  (" + c.detail + ")" : ""}`);
  }
  console.log("");
  console.log(`[h0-residual-table self-test] overall: ${overall_pass ? "PASS (plumbing only)" : "FAIL"}`);
  if (!overall_pass) process.exit(2);
}

// Local generate variant that doesn't touch the real H0-2 output paths
// (used by self-test to avoid polluting real artifacts with test data).
async function generateLocal({ sidecarsDir, recordsDir }) {
  const sidecars = await loadJsonDir(sidecarsDir);
  const records = await loadJsonDir(recordsDir);

  const sidecarByFrame = new Map();
  const recordByFrame = new Map();
  for (const s of sidecars) sidecarByFrame.set(s.json.frame_id, s);
  for (const r of records) recordByFrame.set(r.json.frame_id, r);
  const allFrameIds = new Set([...sidecarByFrame.keys(), ...recordByFrame.keys()]);
  const sortedFrameIds = [...allFrameIds].sort();

  const allRows = [];
  for (const fid of sortedFrameIds) {
    const s = sidecarByFrame.get(fid);
    const r = recordByFrame.get(fid);
    if (s && r) {
      const validation_issues = [...validateSidecar(s.json), ...validateRecord(r.json).map((x) => "record:" + x)];
      if (validation_issues.length > 0) {
        allRows.push({
          frame_id: fid,
          frame_class: null,
          anchor_locus_deg: null,
          measured_deg: null,
          residual_deg: null,
          off_ruler: false,
          within_tolerance: false,
          admit: null,
          reason_code: "SCHEMA_VALIDATION_FAILED",
          operator_decision_codes: "",
          png_sha256: null,
          sidecar_sha256: null,
          h0_record_sha256: null,
          consistency: false,
          consistency_failure_codes: validation_issues.join("|"),
        });
        continue;
      }
      const hashes = { png_sha256: s.json.render_sha256 ?? null, h0_record_sha256: r.raw_sha256 };
      allRows.push(...emitRowsFromPair(s.json, r.json, hashes));
    } else if (s && !r) {
      allRows.push(...emitOrphanSidecarRows(s.json, { png_sha256: s.json.render_sha256 ?? null }));
    } else if (!s && r) {
      allRows.push(...emitOrphanRecordRows(r.json, { h0_record_sha256: r.raw_sha256 }));
    }
  }

  allRows.sort((a, b) => {
    const fa = a.frame_id ?? "";
    const fb = b.frame_id ?? "";
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    const la = a.anchor_locus_deg ?? -Infinity;
    const lb = b.anchor_locus_deg ?? -Infinity;
    return la - lb;
  });

  const tablePayload = { rows: allRows };
  const csvText = rowsToCSV(allRows);
  return { tablePayload, csvText };
}

// ============================================================================
// Validate-single CLI mode
// ============================================================================

async function validateOne(kind, path) {
  const json = JSON.parse(await readFile(resolve(REPO, path), "utf8"));
  const issues = kind === "sidecar" ? validateSidecar(json) : validateRecord(json);
  if (issues.length === 0) {
    console.log(`[h0-residual-table validate] ${kind} OK: ${path}`);
    return;
  }
  console.error(`[h0-residual-table validate] ${kind} INVALID: ${path}`);
  for (const i of issues) console.error(`  - ${i}`);
  process.exit(2);
}

// ============================================================================
// CLI
// ============================================================================

function parseArg(rest, flag) {
  const i = rest.indexOf(flag);
  if (i < 0) return null;
  return rest[i + 1] ?? null;
}

async function main() {
  const [, , cmd, ...rest] = process.argv;
  if (cmd === "generate") {
    const sidecarsDir = parseArg(rest, "--sidecars") ?? SIDECARS_DIR_REL;
    const recordsDir = parseArg(rest, "--records") ?? RECORDS_DIR_REL;
    const outDir = parseArg(rest, "--out-dir") ?? DEFAULT_OUT_DIR_REL;
    const result = await generate({ sidecarsDir, recordsDir, outDir });
    const tableHash = sha256(Buffer.from(result.tableJsonPretty));
    const csvHash = sha256(Buffer.from(result.csvText));
    const manifestHash = sha256(Buffer.from(result.manifestPretty));
    console.log(`[h0-residual-table generate] inputs: ${result.tablePayload.inputs_summary.sidecars} sidecars, ${result.tablePayload.inputs_summary.records} records`);
    console.log(`[h0-residual-table generate] paired: ${result.tablePayload.inputs_summary.paired}, orphan_sidecars: ${result.tablePayload.inputs_summary.orphan_sidecars}, orphan_records: ${result.tablePayload.inputs_summary.orphan_records}, consistency_failures: ${result.tablePayload.inputs_summary.consistency_failures}`);
    console.log(`[h0-residual-table generate] rows: ${result.tablePayload.row_count}`);
    console.log(`[h0-residual-table generate]   ${result.outputPaths.tableJsonRel}  raw=${tableHash}`);
    console.log(`[h0-residual-table generate]   ${result.outputPaths.tableCsvRel}   raw=${csvHash}`);
    console.log(`[h0-residual-table generate]   ${result.outputPaths.manifestRel}    raw=${manifestHash}`);
    return;
  }
  if (cmd === "validate") {
    if (rest.includes("--sidecar")) return validateOne("sidecar", parseArg(rest, "--sidecar"));
    if (rest.includes("--record")) return validateOne("record", parseArg(rest, "--record"));
    console.error("usage: validate --sidecar <path> | validate --record <path>");
    process.exit(64);
  }
  if (cmd === "self-test") return selfTest();
  if (cmd === "hash-file") {
    if (!rest[0]) { console.error("usage: hash-file <path>"); process.exit(64); }
    console.log(await hashFile(resolve(REPO, rest[0])));
    return;
  }
  console.error(
    "usage:\n" +
      "  cut3-h0-residual-table.mjs generate [--sidecars <dir>] [--records <dir>] [--out-dir <dir>]\n" +
      "  cut3-h0-residual-table.mjs validate (--sidecar <path> | --record <path>)\n" +
      "  cut3-h0-residual-table.mjs self-test\n" +
      "  cut3-h0-residual-table.mjs hash-file <path>"
  );
  process.exit(64);
}

main().catch((err) => {
  console.error(`[h0-residual-table] FAILED: ${err.message}`);
  console.error(err.stack);
  process.exit(1);
});
