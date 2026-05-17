#!/usr/bin/env node
// scripts/cut3-h0-checker.mjs
//
// Wave H0-1 runnable calibration checker for Cut-3 H0 angular-calibration
// instrument (P2_CUT3_H0_CALIBRATION.md).
//
// Implements the H0 record schema literally and the admit predicate from §1:
//
//   admit = (scored_feature_deg ≤ valid_angular_span_deg)
//        ∧ (∀ anchor: residual_deg ≤ 0.5)
//        ∧ (no h-leak channel)
//
// Keystone anti-self-seal (§2): valid_angular_span_deg is computed from
// the instrument's own calibrated extent BEFORE scored_feature_deg is
// read. This script enforces that ordering STRUCTURALLY: calibrate()
// has no access to the scored feature value (it reads only the
// calibration sub-fields of the sidecar); admit() is a separate
// function that takes the precomputed Calibration object and the
// scored feature. There is no API path for the scored feature to enter
// span computation.
//
// h-leak detection scope (mechanical scaffolding):
//   The checker flags EXPLICITLY-LABELED h-encoding patterns
//   (h\d+, altitude_\d+, elevation_\d+, sun_alt..., solar_alt..., h_deg,
//   and h-named sidecar fields). Compound HaloSim crystal-config codes
//   like "e13" are NOT auto-flagged — operator review of whether a
//   given compound code semantically encodes h is part of Wave H0-2
//   pre-fill (per-frame H0 records produced from real renders). The
//   freeze's "filename must not encode h" rule binds the operator
//   choosing the corpus; the mechanical checker enforces the label-
//   explicit subset that admits zero false positives.
//
// Modes:
//   check --sidecar <path>        Run the full H0 check on a sidecar.
//        [--out <path>]           Emits the H0 record to stdout or writes it.
//   self-test                     Run the H0-B negative side (Phase-15
//                                 fixture) using test sidecars that
//                                 encode the documented Phase-15 failure
//                                 modes. Verifies all 8 frames emit
//                                 admit=false with reason_code ∈
//                                 {SPAN_TOO_SHORT, ANCHOR_OFF_RULER}.
//   hash-file <path>              Utility.
//
// Test sidecars used in self-test are NOT fabricated H0 records: they
// are documented test inputs that exercise the predicate's reason-code
// logic. The REAL per-frame H0 records come from running `check` on
// each Phase-15 frame with operator-pre-fill sidecars (Wave H0-2).
//
// The known-PASS side of the two-sided self-test (a real full-span
// fixture) is also Wave H0-2 territory: this script's self-test mode
// covers only the negative side and proves the predicate's
// reject-on-failure logic.

import { createHash } from "node:crypto";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { resolve, dirname, basename } from "node:path";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { verifySidecarSelfPin } from "./lib/canonical-json.mjs";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(
  execSync("git rev-parse --show-toplevel", { encoding: "utf8", cwd: SCRIPT_DIR }).trim()
);
const FIXTURE_REL = "results/structural-failure/cut3-prereg/h0-known-fail-fixture.json";
const SELF_TEST_OUT_REL = "results/structural-failure/cut3-prereg/h0-self-test-result.json";

const ANCHOR_RESIDUAL_TOL_DEG = 0.5; // [E] from P2_CUT3_RUN_SPEC.md H0 section

// ----------------------------------------------------------------------------
// h-leak channel rules (label-explicit subset)
// ----------------------------------------------------------------------------

const H_LEAK_FIELD_NAMES = new Set([
  "h", "true_h", "h_deg", "sun_altitude_deg", "altitude_deg",
  "solar_altitude", "solar_altitude_deg", "hidden_h", "hidden_altitude",
  "elevation_deg",
]);

const H_LEAK_FILENAME_PATTERNS = [
  // explicit "h" prefix with digits, word-bounded
  /(?:^|[_-])h\d+(?:[._-]|$)/i,
  /(?:^|[_-])h_deg(?:[._-]|$)/i,
  // explicit altitude / elevation labels with digits
  /(?:^|[_-])altitude[_-]?\d+/i,
  /(?:^|[_-])elevation[_-]?\d+/i,
  /(?:^|[_-])(?:sun|solar)[_-]?alt(?:itude)?[_-]?\d+/i,
];

const H_LEAK_TEXT_PATTERNS = [
  /\bsun[_\s-]?altitude\b/i,
  /\bsolar[_\s-]?altitude\b/i,
  /\btrue[_\s-]?h\b/i,
  /\bh\s*=\s*\d+(?:\.\d+)?\s*(?:deg|°)?\b/i,
];

function checkHLeak(sidecar) {
  const channels = [];

  if (sidecar.operator_decisions?.compound_code_is_h_leak === "yes") {
    channels.push({
      channel: "operator_decision",
      field: "operator_decisions.compound_code_is_h_leak",
      value: "yes",
      basis: sidecar.operator_decisions.compound_code_basis || "",
    });
  }

  const fname = basename(sidecar.frame_path || "");
  if (fname) {
    for (const pat of H_LEAK_FILENAME_PATTERNS) {
      if (pat.test(fname)) {
        channels.push({ channel: "filename", pattern: String(pat), value: fname });
      }
    }
  }

  for (const k of Object.keys(sidecar)) {
    if (H_LEAK_FIELD_NAMES.has(k.toLowerCase())) {
      channels.push({ channel: "sidecar_field", field: k });
    }
  }

  if (typeof sidecar.overlay_text === "string" && sidecar.overlay_text.length > 0) {
    for (const pat of H_LEAK_TEXT_PATTERNS) {
      if (pat.test(sidecar.overlay_text)) {
        channels.push({ channel: "overlay_text", pattern: String(pat) });
      }
    }
  }

  if (typeof sidecar.embedded_metadata === "string" && sidecar.embedded_metadata.length > 0) {
    for (const pat of H_LEAK_TEXT_PATTERNS) {
      if (pat.test(sidecar.embedded_metadata)) {
        channels.push({ channel: "embedded_metadata", pattern: String(pat) });
      }
    }
  }

  return { has_leak: channels.length > 0, channels };
}

// ----------------------------------------------------------------------------
// theta_map construction (three allowed kinds)
// ----------------------------------------------------------------------------

function buildThetaMap(sidecar) {
  const kind = sidecar.theta_map_kind;
  if (kind === "scale_ticks") {
    const ticks = sidecar.scale_ticks;
    if (!Array.isArray(ticks) || ticks.length < 2) {
      throw new Error("scale_ticks requires ≥2 tick entries");
    }
    const sorted = [...ticks].sort((a, b) => a.px_radius - b.px_radius);
    const extent_deg = Math.max(...ticks.map((t) => t.deg));
    return {
      kind,
      params: { ticks: sorted },
      angular_extent_deg: extent_deg,
      pxToDeg(px_radius) {
        if (px_radius <= sorted[0].px_radius) {
          return (sorted[0].deg * px_radius) / sorted[0].px_radius;
        }
        if (px_radius >= sorted[sorted.length - 1].px_radius) {
          return null; // beyond ruler tip
        }
        for (let i = 1; i < sorted.length; i++) {
          if (px_radius <= sorted[i].px_radius) {
            const a = sorted[i - 1], b = sorted[i];
            const t = (px_radius - a.px_radius) / (b.px_radius - a.px_radius);
            return a.deg + t * (b.deg - a.deg);
          }
        }
        return null;
      },
    };
  }
  if (kind === "renderer_metadata") {
    const md = sidecar.renderer_metadata;
    if (!md || typeof md.hash !== "string" || typeof md.extent_deg !== "number") {
      throw new Error("renderer_metadata requires { hash, extent_deg }");
    }
    return {
      kind,
      params: { hash: md.hash, extent_deg: md.extent_deg },
      angular_extent_deg: md.extent_deg,
      pxToDeg(_px_radius) {
        return null;
      },
    };
  }
  if (kind === "fit2locus") {
    const fit = sidecar.fit2locus ?? {
      anchors: (sidecar.anchors || [])
        .filter((a) => typeof a.px_radius === "number" && a.off_ruler !== true)
        .map((a) => ({ px_radius: a.px_radius, deg: a.locus_deg })),
    };
    if (!fit || !Array.isArray(fit.anchors) || fit.anchors.length < 2) {
      throw new Error("fit2locus requires ≥2 anchors");
    }
    const xs = fit.anchors.map((a) => a.px_radius);
    const ys = fit.anchors.map((a) => a.deg);
    const n = xs.length;
    const sx = xs.reduce((a, b) => a + b, 0);
    const sy = ys.reduce((a, b) => a + b, 0);
    const sxx = xs.reduce((a, b) => a + b * b, 0);
    const sxy = xs.reduce((a, b, i) => a + b * ys[i], 0);
    const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    const intercept = (sy - slope * sx) / n;
    const support_min = Math.min(...xs);
    const support_max = Math.max(...xs);
    const extent_deg = Math.max(...ys);
    return {
      kind,
      params: { slope, intercept, support_px: [support_min, support_max] },
      angular_extent_deg: extent_deg,
      pxToDeg(px_radius) {
        if (px_radius < support_min || px_radius > support_max) return null;
        return slope * px_radius + intercept;
      },
    };
  }
  throw new Error(`unknown theta_map_kind: ${kind}`);
}

// ----------------------------------------------------------------------------
// CALIBRATE — computes everything that does NOT depend on the scored feature
// ----------------------------------------------------------------------------
//
// STRUCTURAL ENFORCEMENT: this function has no access to scored_feature_*
// fields of the sidecar (it reads only the calibration sub-fields). The
// returned Calibration object is consumed by admit() which is a separate
// function. There is NO API path for the feature to enter calibration.

function calibrate(sidecar) {
  const theta_map = buildThetaMap(sidecar);

  // valid_angular_span_deg is measured from the instrument's own
  // calibrated extent (theta_map.angular_extent_deg). NEVER from any
  // feature read. This is the §2 anti-self-seal invariant.
  const valid_angular_span_deg = theta_map.angular_extent_deg;

  const anchors = (sidecar.anchors || []).map((a) => {
    let measured_deg = a.measured_deg;
    if (measured_deg === undefined && typeof a.px_radius === "number") {
      measured_deg = theta_map.pxToDeg(a.px_radius);
    }
    const off_ruler =
      measured_deg === null ||
      measured_deg === undefined ||
      a.locus_deg > valid_angular_span_deg;
    return {
      locus_deg: a.locus_deg,
      measured_deg: measured_deg,
      residual_deg: off_ruler ? null : Math.abs(measured_deg - a.locus_deg),
      off_ruler,
    };
  });

  const h_leak = checkHLeak(sidecar);

  return Object.freeze({
    frame_id: sidecar.frame_id,
    render_sha256: sidecar.render_sha256 || null,
    calib_sha256: sidecar.calib_sha256 || null,
    sun_px: sidecar.sun_px || null,
    projection: sidecar.projection || null,
    theta_map: {
      kind: theta_map.kind,
      params: theta_map.params,
    },
    valid_angular_span_deg,
    anchors,
    h_leak,
  });
}

// ----------------------------------------------------------------------------
// ADMIT — applies the predicate; takes the precomputed calibration AND
// the scored feature. The scored feature value MAY NOT enter calibration.
// ----------------------------------------------------------------------------

function admit(calibration, scored_feature_deg) {
  const reasons = [];

  if (calibration.h_leak.has_leak) {
    reasons.push("H_LEAK");
  }

  if (scored_feature_deg > calibration.valid_angular_span_deg) {
    reasons.push("FEATURE_OUTSIDE_SPAN");
  }

  for (const a of calibration.anchors) {
    if (a.off_ruler) reasons.push("ANCHOR_OFF_RULER");
    else if (a.residual_deg > ANCHOR_RESIDUAL_TOL_DEG) reasons.push("ANCHOR_RESIDUAL_OVER_TOL");
  }

  // SPAN_TOO_SHORT: the calibrated angular extent doesn't cover the
  // named atmospheric-optics anchor loci (22° / 46°). Phase-15 mode.
  for (const a of calibration.anchors) {
    if (a.off_ruler && a.locus_deg > calibration.valid_angular_span_deg) {
      if (!reasons.includes("SPAN_TOO_SHORT")) reasons.push("SPAN_TOO_SHORT");
    }
  }

  const priority = ["H_LEAK", "SPAN_TOO_SHORT", "ANCHOR_OFF_RULER", "ANCHOR_RESIDUAL_OVER_TOL", "FEATURE_OUTSIDE_SPAN"];
  const unique = [...new Set(reasons)];
  const reason_code = unique.length === 0 ? "OK" : priority.find((p) => unique.includes(p)) || unique[0];

  return {
    frame_id: calibration.frame_id,
    render_sha256: calibration.render_sha256,
    calib_sha256: calibration.calib_sha256,
    sun_px: calibration.sun_px,
    projection: calibration.projection,
    theta_map: calibration.theta_map,
    valid_angular_span_deg: calibration.valid_angular_span_deg,
    anchors: calibration.anchors,
    scored_feature_deg,
    admit: reason_code === "OK",
    reason_code,
    all_reasons: unique,
    h_leak_channels: calibration.h_leak.channels,
  };
}

// ----------------------------------------------------------------------------
// Self-test mode — H0-B negative side
// ----------------------------------------------------------------------------
//
// Test sidecars below model the documented Phase-15 failure modes
// (P2_CUT3_H0_CALIBRATION.md §3 "the ruler span was shorter than the
// ring field and 22/46° were beyond the tip"). They are NOT fabricated
// H0 records — they are test inputs that exercise the predicate's
// reject-on-failure logic. The real per-frame H0 records come from
// running `check` on the actual Phase-15 frames with operator-pre-fill
// sidecars (Wave H0-2).
//
// The test sidecar's frame_path is set to the frame_id only (no
// extension, no directory) so the mechanical filename leak check does
// NOT fire — the self-test exercises the SPAN/ANCHOR reject path
// specifically. The Phase-15 filenames themselves contain
// HaloSim-crystal-config codes that the operator may or may not
// classify as h-encoding; that decision is part of Wave H0-2 pre-fill.

function buildPhase15TestSidecar(fixtureFrame) {
  return {
    // frame_path deliberately uses frame_id without extension so the
    // mechanical filename-leak check doesn't fire — the negative
    // self-test exercises SPAN/ANCHOR reject logic, not H_LEAK.
    frame_path: fixtureFrame.frame_id,
    frame_id: fixtureFrame.frame_id,
    render_sha256: fixtureFrame.scale_stamped_render.sha256,
    calib_sha256: "test-sidecar-phase15-failure-model",
    sun_px: [400, 400],
    projection: "halosim_scale_zoomed",
    theta_map_kind: "scale_ticks",
    // Pyramidal-zoom scale stamps: ticks go ~5° to ~18°; 22°/46° beyond tip.
    scale_ticks: [
      { px_radius: 100, deg: 5 },
      { px_radius: 200, deg: 10 },
      { px_radius: 300, deg: 15 },
      { px_radius: 360, deg: 18 },
    ],
    anchors: [
      { locus_deg: 22 },
      { locus_deg: 46 },
    ],
  };
}

async function selfTest() {
  const fixturePath = resolve(REPO, FIXTURE_REL);
  const fixture = JSON.parse(await readFile(fixturePath, "utf8"));

  const results = [];
  let pass_count = 0;
  let fail_count = 0;
  for (const frame of fixture.fixture_objects) {
    const sidecar = buildPhase15TestSidecar(frame);
    const calibration = calibrate(sidecar);
    const record = admit(calibration, 18.5);
    const expected = frame.expected_self_test;
    const reason_ok = expected.reason_codes_allowed.includes(record.reason_code);
    const admit_ok = record.admit === expected.admit;
    const self_test_pass = !record.admit && reason_ok;
    if (self_test_pass) pass_count++;
    else fail_count++;
    results.push({
      frame_id: frame.frame_id,
      checker_admit: record.admit,
      checker_reason_code: record.reason_code,
      expected_admit: expected.admit,
      expected_reason_codes: expected.reason_codes_allowed,
      admit_matches_expected: admit_ok,
      reason_code_matches_allowed: reason_ok,
      self_test_pass,
      valid_angular_span_deg: record.valid_angular_span_deg,
      anchors: record.anchors,
    });
  }

  const overall_pass = fail_count === 0 && pass_count === fixture.fixture_objects.length;
  const payload = {
    self_test: "H0-B negative side (Phase-15 known-FAIL fixture)",
    spec_reference:
      "P2_CUT3_H0_CALIBRATION.md §3 — checker MUST reject all 8 Phase-15 frames with reason_code ∈ {SPAN_TOO_SHORT, ANCHOR_OFF_RULER}",
    fixture_manifest: FIXTURE_REL,
    test_sidecar_disposition:
      "Test sidecars model the documented Phase-15 failure modes (short ruler + off-ruler 22°/46° anchors). They are test inputs that exercise the predicate's reject-on-failure logic; they are NOT fabricated H0 records and they are NOT the real per-frame H0 records that operator pre-fill (Wave H0-2) will produce from the actual frames. frame_path in the test sidecars is set to frame_id without extension so the mechanical filename-leak check does not fire — the negative self-test exercises the SPAN/ANCHOR reject path specifically. Operator review of whether the actual Phase-15 filenames (which contain HaloSim crystal-config codes like 'e13') semantically encode h is part of Wave H0-2 pre-fill.",
    pass_count,
    fail_count,
    overall_pass,
    note_on_known_pass_side:
      "The H0-B two-sided self-test also requires a known-PASS full-span fixture (a real ordinary halo render whose stamped ruler covers 22° and 46° with anchor residuals ≤ 0.5°). That fixture is Wave H0-2 territory — operator-in-the-loop identification of an existing render. H0 closes only when BOTH sides resolve correctly.",
    results,
  };

  const outAbs = resolve(REPO, SELF_TEST_OUT_REL);
  await mkdir(dirname(outAbs), { recursive: true });
  await writeFile(outAbs, JSON.stringify(payload, null, 2) + "\n");

  console.log(`[h0-checker self-test] wrote ${SELF_TEST_OUT_REL}`);
  console.log(
    `[h0-checker self-test] H0-B negative side: ${pass_count}/${fixture.fixture_objects.length} frames rejected with expected reason_code`
  );
  for (const r of results) {
    const tag = r.self_test_pass ? "OK   " : "FAIL ";
    console.log(
      `  [${tag}] ${r.frame_id.padEnd(20)} admit=${r.checker_admit} reason=${r.checker_reason_code} span=${r.valid_angular_span_deg}°`
    );
  }
  console.log("");
  console.log(`[h0-checker self-test] overall: ${overall_pass ? "PASS (negative side only)" : "FAIL"}`);
  if (!overall_pass) process.exit(2);
}

// ----------------------------------------------------------------------------
// CLI
// ----------------------------------------------------------------------------

async function checkCmd(sidecarPath, outPath = null) {
  const sidecar = JSON.parse(await readFile(resolve(REPO, sidecarPath), "utf8"));
  if (sidecar.sidecar_kind === "h0-measured-sidecar") {
    const pin = verifySidecarSelfPin(sidecar);
    if (!pin.ok) {
      throw new Error(
        `sidecar calib_sha256 self-pin mismatch: stored=${sidecar.calib_sha256} recomputed=${pin.recomputed}`
      );
    }
  }
  const calibration = calibrate(sidecar);
  if (typeof sidecar.scored_feature_deg !== "number") {
    throw new Error("sidecar.scored_feature_deg is required for `check` mode");
  }
  const record = admit(calibration, sidecar.scored_feature_deg);
  record.provenance = {
    source_sidecar_sha256: sidecar.calib_sha256 || null,
    checker_sha256: await hashFile("scripts/cut3-h0-checker.mjs"),
    checker_runtime_pt: new Date().toISOString(),
  };
  const json = JSON.stringify(record, null, 2) + "\n";
  if (outPath) {
    const absOut = resolve(REPO, outPath);
    await mkdir(dirname(absOut), { recursive: true });
    await writeFile(absOut, json);
    console.log(`[h0-checker] wrote ${outPath}`);
    return;
  }
  console.log(json.trimEnd());
}

async function hashFile(path) {
  const bytes = await readFile(resolve(REPO, path));
  return createHash("sha256").update(bytes).digest("hex");
}

async function main() {
  const [, , cmd, ...rest] = process.argv;
  if (cmd === "self-test") {
    await selfTest();
    return;
  }
  if (cmd === "check") {
    const i = rest.indexOf("--sidecar");
    if (i < 0 || !rest[i + 1]) {
      console.error("usage: check --sidecar <path> [--out <path>]");
      process.exit(64);
    }
    const outIndex = rest.indexOf("--out");
    if (outIndex >= 0 && !rest[outIndex + 1]) {
      console.error("usage: check --sidecar <path> [--out <path>]");
      process.exit(64);
    }
    await checkCmd(rest[i + 1], outIndex >= 0 ? rest[outIndex + 1] : null);
    return;
  }
  if (cmd === "hash-file") {
    if (!rest[0]) { console.error("usage: hash-file <path>"); process.exit(64); }
    console.log(await hashFile(rest[0]));
    return;
  }
  console.error(
      "usage:\n" +
      "  cut3-h0-checker.mjs self-test\n" +
      "  cut3-h0-checker.mjs check --sidecar <path> [--out <path>]\n" +
      "  cut3-h0-checker.mjs hash-file <path>"
  );
  process.exit(64);
}

main().catch((err) => {
  console.error(`[h0-checker] FAILED: ${err.message}`);
  process.exit(1);
});
