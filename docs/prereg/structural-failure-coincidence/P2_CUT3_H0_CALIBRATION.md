# Structural Failure Coincidence — Cut 3 H0 Angular-Calibration Instrument (operational freeze)

Pre-registration: [`README.md`](README.md)
Cut-3 run spec: [`P2_CUT3_RUN_SPEC.md`](P2_CUT3_RUN_SPEC.md) (§"H0 — angular-calibration gate", frozen)
Cut-3 admission: [`P2_CUT3_ADMISSION.md`](P2_CUT3_ADMISSION.md) (HOLD; §D item 1, "Allowed next work")
Phase-15 negative fixture (real, immutable): `docs/calibration/halosim_outputs/phase15_pyrfilter/pyr_w*_scale.png`
Filed: **2026-05-16 (PT)**. Status: **H0 INSTRUMENT FILED FOR AUDIT —
EXECUTION HELD.** Like C2-A/C3-A/C4-A, this freezes the **schema, the
keystone invariant, and the self-test obligation**; the runnable checker,
the per-frame H0 records, and the residual table are the maintainer's
pre-run fill (no fabrication). A failing H0 **blocks Cut-3, never
stretches the span**. This is the admission's explicitly *allowed* next
work ("Draft the H0 manifest schema; build a small calibration-manifest
reader/checker"); it runs nothing and admits no corpus.

## Purpose

The Cut-3 run spec froze the H0 *protocol* (the 6-field record, allowed/
disallowed paths, the ≤0.5° tolerance, the Phase-15 pyramidal frames as
the named negative). H0 is now operationalized — turned from a principle
into a concrete, reproducible instrument with a known-negative self-test
— exactly the "a principle is not an artifact" discipline applied to
C2-A/C3-A/C4-A, now upstream of the entire Cut-3 corpus.

## 1. The H0 record, operationalized

One JSON record per (frame, scored-feature), schema fixed here:

```
{ "frame_id", "render_sha256", "calib_sha256",
  "sun_px": [x,y] | "halosim_sun_origin_receipt",
  "projection": "<halosim projection family>",
  "theta_map": { "kind": "scale_ticks" | "renderer_metadata" | "fit2locus",
                 "params": ... },          # deterministic px -> deg-from-sun
  "valid_angular_span_deg": <number>,      # see §2 — measured BEFORE feature
  "anchors": [ { "locus_deg": 22|46|..., "measured_deg", "residual_deg" } ],
  "scored_feature_deg": <number>,
  "admit": <bool>, "reason_code": "<...>" }
```

- **`theta_map`** is a deterministic function of pixel → degrees-from-sun.
  Per the run spec's allowed paths: `scale_ticks` (px-per-degree read from
  HaloSim's *own* stamped graduations, not assumed), `renderer_metadata`
  (a saved+hashed sun-centered angular map), or `fit2locus`
  (least-squares on ≥2 known loci, e.g. 22°+46°, with residuals).
- **`anchors`**: `residual_deg = |measured_deg − locus_deg|` via
  `theta_map`, for every anchor used to admit the frame; the anchor must
  lie inside `valid_angular_span_deg`.
- **Per-feature admissibility predicate** (computable, no human call):
  `admit = (scored_feature_deg ≤ valid_angular_span_deg)
           ∧ (∀ anchor: residual_deg ≤ 0.5)
           ∧ (no h-leak channel)`.
  Admissibility is **per (frame, feature)**: a frame may admit a
  near-sun feature and be inadmissible for a far one.
- **No-h-leak channel check** (run-spec "disallowed"): filename,
  directory, overlay text, embedded metadata, and the calibration
  sidecar must not encode `h`; presence of any ⇒ `admit=false`,
  `reason_code=H_LEAK`.

## 2. H0 keystone load-bearing self-seal (surfaced adversarially)

The C-series pattern, now for H0. The rig hazard is
**`valid_angular_span_deg`**: if the span is measured or declared *after*
the scored feature's position is known, it can be silently stretched to
swallow a feature that is actually off-ruler — which is *exactly the
Phase-15 pyramidal failure* (ring field beyond the ruler tip; the
tempting move is "call the span longer" or "score past the tip").

Frozen invariant: `valid_angular_span_deg` is computed from the
**instrument's own calibrated extent** (Scale ruler tip in degrees /
renderer-metadata coverage / `fit2locus` support interval) by a fixed
procedure, **before `scored_feature_deg` is read into the calibration
step**. The feature-in-span check then only admits or excludes — there is
no span freedom left. A feature outside span is excluded; if post-
exclusion coverage drops below the run-spec corpus minimum, **Cut-3
remains BLOCKED, not forced** (no span-stretching, no anchor
substitution, no "score it anyway"). Span is `[G]` once measured per
frame; it is never re-derived to rescue coverage (A3).

## 3. H0-B self-test against the pre-registered negative fixture

The C4-B pattern: the instrument is proved on the exact historical
negative that motivated it, before it is trusted on any Cut-3 frame.

- **Known-FAIL fixture (real, immutable):** the eight Phase-15 pyramidal
  scale-stamped frames `docs/calibration/halosim_outputs/phase15_pyrfilter/pyr_w*_scale.png`.
  Run on them against the 22°/46° anchors, the H0 checker **must** return
  `admit=false`, `reason_code ∈ {SPAN_TOO_SHORT, ANCHOR_OFF_RULER}` —
  because the ruler span was shorter than the ring field and 22/46° were
  beyond the tip. If the checker does **not** reject these frames, H0 is
  self-sealed and **not closed**.
- **Known-PASS fixture:** one ordinary full-span render whose stamped
  ruler covers both 22° and 46° with anchor residuals ≤ 0.5°. The
  checker **must** return `admit=true`. (If none exists yet, this fixture
  is part of the maintainer pre-run fill; it must be a real render, not a
  synthetic stub.)

Two-sided, with the negative fixture an **immutable real artifact** (no
fixture-design freedom — it is the Phase-15 receipt as-is). H0 is closed
only when both fixtures resolve correctly in the checker's test suite.

## 4. Provenance-tagged freeze (A3)

`[G]` immutable / geometry-or-receipt; `[E]` pre-registered engineering
tolerance (amend-only, justified, never post-results).

| item | role | provenance |
| --- | --- | --- |
| 22° / 46° anchor loci | known-angle cross-checks | **[G]** atmospheric-optics facts |
| span-measured-before-feature ordering | the §2 anti-self-seal invariant | **[G]** (rule; a change is a goalpost move) |
| Phase-15 `pyr_w*_scale.png` known-FAIL fixture | the §3 negative self-test | **[G]** immutable real artifact |
| `≤ 0.5°` anchor-residual tolerance | admit/exclude threshold | **[E]** (vs visual-edge noise; from the frozen run spec) |
| `render_sha256` / `calib_sha256` scheme | reproducibility / no silent swap | **[E]** |
| reason-code vocabulary | exclusion accounting | **[E]** |

No frozen Cut-3 run-spec value is changed; this is its operational
realization, append-only and consistent with the frozen body.

## 5. Honest couplings

- **H0 is upstream of everything in Cut-3.** It gates the corpus
  manifest (admission §D-1), which gates the agent path, baselines, and
  edit operators, which gate any run. A Cut-3 run on an un-H0-checked
  corpus is **void**.
- **No new fixture artifact is created** — the negative fixture is the
  existing Phase-15 `phase15_pyrfilter/*_scale.png` set, reused as-is.
- Ties to admission §D items 1 (manifest), 3 (residual table), 7
  (C5-style write-path guard — H0 records/checker outputs must land only
  under the Cut-3 allowlist).
- The C1 controller, the four quantities, and the Public-Language
  Constraint carry forward from the frozen Cut-3 run spec unchanged.

## Cut-3 H0 binding rules

1. `valid_angular_span_deg` is measured before the feature position is
   read; a post-hoc span ⇒ run **void**.
2. The H0-B self-test must run in the checker suite and return
   FAIL-on-Phase-15-pyramidal / PASS-on-the-full-span-fixture **before**
   any Cut-3 corpus frame is admitted.
3. `≤ 0.5°` residual, the anchor loci, and the no-h-leak rule are frozen;
   relaxing any ⇒ void.
4. Coverage shortfall after honest exclusions ⇒ Cut-3 **BLOCKED**, never
   forced.

## Explicit non-bindings (cannot satisfy H0)

- A global pixels-per-degree constant, or an auto-zoom scale-lock not
  re-measured per frame.
- A center chosen by whichever downstream detector gives the wanted
  answer.
- A span declared/stretched after seeing the scored feature.
- Closing H0 without the Phase-15 pyramidal frames actually failing in
  the checker's test suite.
- Any fabricated or hand-edited H0 record; records are the maintainer's
  pre-run fill from real renders.

## Open items

H0 files the schema + keystone invariant + self-test obligation. Still
the maintainer's pre-run fill: the runnable calibration-manifest
reader/checker, the per-frame H0 records, the anchor-residual table, and
the known-PASS full-span fixture. Cut-3 stays **HELD**; the remaining
admission HOLDs (corpus manifest, agent path, baselines, edit operators,
write-path guard) are unchanged; Cut-3 admission must be re-run once H0
+ those land. Nothing here begins Cut-3.

## Honest prior (unchanged)

H0 is measurement hygiene, not the science. It only decides whether a
rendered frame is angularly trustworthy enough to score — it does not
make the Proxy-Collapse test pass or fail. The open question (a genuine
rendered benchmark result, or an honest null/block) still lives entirely
downstream in an admitted Cut-3 run. No `CONFIRMED` / theorem /
"traceability" / "Cut-3 has begun" language anywhere.

## Audit Notes

*(reviewer space — append-only below)*
