# H0 Measurement Tool How-To

This is the operator guide for the H0-2 measurement helper at
`tools/h0-measurement/index.html` and the companion Node tools described in
`docs/prereg/structural-failure-coincidence/P2_CUT3_H0_2_SCHEMA.md`.

The tool makes measured sidecars. It does not decide whether a frame passes H0,
does not run the checker, and does not make any Cut-3 or traceability claim.

## Tool Map

- `tools/h0-measurement/index.html` — browser/canvas helper for producing one
  `h0-measured-sidecar` JSON file.
- `scripts/lib/canonical-json.mjs` — canonical JSON and `calib_sha256`
  self-pin implementation shared by the Node path.
- `scripts/cut3-h0-checker.mjs` — turns a sealed sidecar into one H0 record.
- `scripts/cut3-h0-residual-table.mjs` — validates sidecars/records and emits
  `h0-anchor-residual-table.{json,csv}` plus `h0-2-manifest.json`.

## Before Measuring

Run these from the repo root:

```powershell
node scripts/cut3-h0-make-test-vector.mjs
node --input-type=module -e "import { readFile } from 'node:fs/promises'; import { verifySidecarSelfPin } from './scripts/lib/canonical-json.mjs'; const tv=JSON.parse(await readFile('results/structural-failure/cut3-prereg/h0-canonicalization-test-vector.json','utf8')); const v=verifySidecarSelfPin(tv.pinned_sidecar_with_calib_sha256); if (!v.ok) process.exit(2); console.log(v.recomputed);"
node scripts/cut3-h0-residual-table.mjs self-test
```

Expected: the first command regenerates the pinned test vector, the second
prints `9c52ab15...c8b6`, and the residual-table self-test reports `11/11`
passing. This is plumbing only, not an H0-2 result.

## Open The Helper

Start the normal repo-root Vite server:

```powershell
npm run dev -- --port 5173
```

Open:

```text
http://127.0.0.1:5173/tools/h0-measurement/
```

Use the repo-root server, not a server rooted at `tools/h0-measurement`, because
the Phase-15 preset dropdown fetches PNGs from `/docs/calibration/...`.

Before clicking any image, check the header badge. It must say the
canonicalizer matches the pinned test vector. If it says drift or error, stop.

## Make A Sidecar

1. Load an image with the Phase-15 preset dropdown, file picker, or drag/drop.
   The helper computes `render_sha256` from the loaded image bytes.
2. Choose the mode:
   - `fit2locus`: click sun center, then the 22 degree halo/anchor, then the
     46 degree halo/anchor.
   - `scale_ticks`: click sun center, click each stamped scale tick, press
     `Done with ticks -> 22° anchor`, then click or mark the 22/46 degree
     anchors.
3. For an off-ruler anchor, use the `Mark 22° off-ruler` or
   `Mark 46° off-ruler` button instead of clicking a guessed point.
4. Fill operator decisions:
   - `frame_id`
   - `fixture_class`
   - `compound_code_is_h_leak`
   - `compound_code_basis`
   - `measured_by`
   - `known_pass_selection_basis` only for `known_pass_fullspan`
5. Edit each anchor `measurement_note` so the sidecar says what the operator
   actually saw.
6. Download the sidecar when the validation panel is clean.

Put real H0-2 sidecars here:

```text
results/structural-failure/cut3-prereg/h0-sidecars/<frame_id>.json
```

## Make Records

For each sealed sidecar, run the checker separately:

```powershell
node scripts/cut3-h0-checker.mjs check --sidecar results/structural-failure/cut3-prereg/h0-sidecars/<frame_id>.json --out results/structural-failure/cut3-prereg/h0-records/<frame_id>.json
```

The checker verifies the sidecar self-pin, computes the H0 record, and writes a
`provenance` block with `source_sidecar_sha256`, `checker_sha256`, and
`checker_runtime_pt`. Do not edit the sidecar after this step; if you need to
fix a measurement, create a new sidecar and rerun the checker.

Optional pre-flight checks:

```powershell
node scripts/cut3-h0-residual-table.mjs validate --sidecar results/structural-failure/cut3-prereg/h0-sidecars/<frame_id>.json
node scripts/cut3-h0-residual-table.mjs validate --record results/structural-failure/cut3-prereg/h0-records/<frame_id>.json
```

## Build The Table

After the real sidecars and records exist:

```powershell
node scripts/cut3-h0-residual-table.mjs generate
```

This writes:

```text
results/structural-failure/cut3-prereg/h0-anchor-residual-table.json
results/structural-failure/cut3-prereg/h0-anchor-residual-table.csv
results/structural-failure/cut3-prereg/h0-2-manifest.json
```

Read the command summary. For a clean paired run, `consistency_failures` should
be `0`. Any `ORPHAN_RECORD`, `ORPHAN_SIDECAR`,
`SIDECAR_SELF_PIN_MISMATCH`, `RECORD_SIDECAR_SHA_MISMATCH`, or
`TIMESTAMP_ORDERING_VIOLATION` is a block to resolve before H0-2 can be
claimed.

For a scratch smoke test, keep generated outputs away from the canonical H0-2
paths:

```powershell
node scripts/cut3-h0-residual-table.mjs generate --sidecars results/structural-failure/cut3-prereg/_tmp-smoke/h0-sidecars --records results/structural-failure/cut3-prereg/_tmp-smoke/h0-records --out-dir results/structural-failure/cut3-prereg/_tmp-smoke/out
```

## Operator Discipline

- The sidecar has no verdict field. The checker owns `admit` and
  `reason_code`.
- The known-PASS fixture must be a real full-span render selected by the
  instrument criterion before the checker runs.
- The Phase-15 known-FAIL side must be measured from the real PNGs, not from
  modeled sidecars.
- A successful tool run is only measurement hygiene. Cut-3 admission remains
  HOLD until the full admission checklist is re-run.
