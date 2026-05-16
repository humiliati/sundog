#!/usr/bin/env node
// scripts/cut3-h0-known-fail-extract.mjs
//
// Wave H0-1 Phase-15 known-FAIL fixture extractor.
//
// Operationalizes the H0-B negative self-test (P2_CUT3_H0_CALIBRATION.md §3):
// pins the 8 Phase-15 pyramidal scale-stamped frames as the immutable
// known-FAIL fixture by content SHA-256, alongside their paired raw
// renders and .sim configs. The runnable checker (cut3-h0-checker.mjs)
// must reject all 8 with reason_code ∈ {SPAN_TOO_SHORT, ANCHOR_OFF_RULER}.
//
// Pattern mirror: scripts/cut2-cut1-fixture-extract.mjs (Wave-1 C4-A
// Cut-1 fixture). Same immutable-real-artifact discipline lifted to the
// measurement layer for Cut-3.
//
// Output: results/structural-failure/cut3-prereg/h0-known-fail-fixture.json

import { createHash } from "node:crypto";
import { readFile, readdir, mkdir, writeFile } from "node:fs/promises";
import { resolve, dirname, join, relative } from "node:path";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(
  execSync("git rev-parse --show-toplevel", { encoding: "utf8", cwd: SCRIPT_DIR }).trim()
);

const SOURCE_DIR_REL = "docs/calibration/halosim_outputs/phase15_pyrfilter";
const OUT_REL = "results/structural-failure/cut3-prereg/h0-known-fail-fixture.json";

// The 8 Phase-15 scale-stamped frames are the H0-B negative fixture per
// the freeze §3. Each frame's basename is the frame_id; paired files
// (raw render *_4M.png, config *.sim) are recorded for provenance.
async function hashFile(absPath) {
  const bytes = await readFile(absPath);
  return {
    sha256: createHash("sha256").update(bytes).digest("hex"),
    byte_size: bytes.length,
  };
}

async function main() {
  const sourceAbs = resolve(REPO, SOURCE_DIR_REL);
  const entries = await readdir(sourceAbs);
  const scaleFrames = entries
    .filter((n) => /^pyr_.+_scale\.png$/.test(n))
    .sort();

  if (scaleFrames.length !== 8) {
    throw new Error(
      `Expected exactly 8 Phase-15 scale-stamped frames; found ${scaleFrames.length}`
    );
  }

  const frames = [];
  for (const scaleName of scaleFrames) {
    const frame_id = scaleName.replace(/_scale\.png$/, "");
    const scaleAbs = join(sourceAbs, scaleName);
    const scaleHash = await hashFile(scaleAbs);

    const rawName = `${frame_id}_4M.png`;
    const rawAbs = join(sourceAbs, rawName);
    let rawHash = null;
    try {
      rawHash = await hashFile(rawAbs);
    } catch (e) {
      if (e.code !== "ENOENT") throw e;
    }

    const simName = `${frame_id}.sim`;
    const simAbs = join(sourceAbs, simName);
    let simHash = null;
    try {
      simHash = await hashFile(simAbs);
    } catch (e) {
      if (e.code !== "ENOENT") throw e;
    }

    frames.push({
      frame_id,
      scale_stamped_render: {
        path: relative(REPO, scaleAbs).split("\\").join("/"),
        sha256: scaleHash.sha256,
        byte_size: scaleHash.byte_size,
      },
      raw_render: rawHash
        ? {
            path: relative(REPO, rawAbs).split("\\").join("/"),
            sha256: rawHash.sha256,
            byte_size: rawHash.byte_size,
          }
        : null,
      config: simHash
        ? {
            path: relative(REPO, simAbs).split("\\").join("/"),
            sha256: simHash.sha256,
            byte_size: simHash.byte_size,
          }
        : null,
      expected_self_test: {
        admit: false,
        reason_codes_allowed: ["SPAN_TOO_SHORT", "ANCHOR_OFF_RULER"],
        note: "Phase-15 failure: ruler span shorter than the ring field; 22° / 46° anchors off-ruler. The H0 checker MUST emit one of these reason codes for every frame.",
      },
    });
  }

  const manifest = {
    manifest_version: 1,
    frozen_at_pt: "2026-05-16",
    purpose:
      "H0-B known-FAIL fixture — Phase-15 pyramidal scale-stamped frames. The runnable checker (cut3-h0-checker.mjs) MUST reject all 8 with reason_code ∈ {SPAN_TOO_SHORT, ANCHOR_OFF_RULER}; if it does not, H0 is self-sealed and remains UNCLOSED.",
    spec_reference:
      "docs/prereg/structural-failure-coincidence/P2_CUT3_H0_CALIBRATION.md §3 (H0-B self-test) and §4 (provenance: [G] immutable real artifact).",
    fixture_role:
      "The C4-B pattern applied one level out: the H0 instrument is proved on the exact historical negative (Phase-15) that motivated its existence, BEFORE any Cut-3 corpus frame is admitted.",
    fixture_immutability:
      "Real, frozen Phase-15 receipt as-is. Editing any of these eight frames voids the H0-B negative side of the two-sided self-test; doing so requires an append-only redesign of H0-B, not a manifest amendment that retroactively legitimizes the edit.",
    source_directory: SOURCE_DIR_REL,
    fixture_objects: frames,
    extractor_script: "scripts/cut3-h0-known-fail-extract.mjs",
    rerun_command: "node scripts/cut3-h0-known-fail-extract.mjs",
  };

  const outAbs = resolve(REPO, OUT_REL);
  await mkdir(dirname(outAbs), { recursive: true });
  await writeFile(outAbs, JSON.stringify(manifest, null, 2) + "\n");

  console.log(`[h0-known-fail-extract] wrote ${OUT_REL}`);
  console.log(`[h0-known-fail-extract] ${frames.length} frames pinned`);
  for (const f of frames) {
    console.log(
      `  ${f.frame_id.padEnd(20)} scale=${f.scale_stamped_render.sha256.slice(0, 12)}  raw=${f.raw_render?.sha256.slice(0, 12) ?? "n/a"}  sim=${f.config?.sha256.slice(0, 12) ?? "n/a"}`
    );
  }
}

main().catch((err) => {
  console.error(`[h0-known-fail-extract] FAILED: ${err.message}`);
  process.exit(1);
});
