# Phase 10 Follow-on - Parhelic-Belt-Y Replication Spec

This document is the implementation-grade spec for Open Question #1 of
the Phase 10 closeout in
[`RICH_DISPLAY_OVERLAY_NOTES.md`](RICH_DISPLAY_OVERLAY_NOTES.md). It
inherits the mesa-discipline pre-registration shape (`PHASE6_V*_SPEC.md`
lineage; v3.8 used EE; this spec uses FF) and the residual / coverage /
detection three-gate taxonomy ratified in
[`../MESA_CROSSOVER_NOTE.md`](../MESA_CROSSOVER_NOTE.md).

Where this spec and the Phase 10 closeout disagree, the closeout wins.
Where both are silent, this spec is authoritative for the belt-y
replication question.

## 1. Decision Lock

Five pinned calls:

- **Question:** does p13's `+10.4 px` parhelic-belt-y residual against
  the global `--parhelic-y-offset-r22 = -0.05` rule replicate across
  multiple low-h photos, or is it photo-specific?
- **Scope:** primitive-level (parhelic belt vertical placement), not a
  new inversion route. The belt-y residual is recorded under the
  Per-Inversion-Route Residual Table's `Parhelic-belt y residual
  (primitive, not route)` row, with the 12 px primitive threshold.
- **Method:** anchor capture on existing low-h calibration / triage
  photos plus the already-anchored p13 and p27. No new physical data
  acquisition; no controlled-optics work (that is Open Question #3).
- **Decision rule cliff:** "pattern" requires `|belt_y_residual| >= 5
  px` on at least three of the four-or-more low-h photos in the
  expansion set, with same sign and a non-zero correlation with `h`.
  "Photo-specific" is the alternative.
- **Out of scope:** mid-altitude photos (p2 at h = 18.6° is the upper
  edge of the regime we care about; p7 at h = 59.4° is out of regime
  for low-h belt evaluation). No supralateral, CZA, or tangent-route
  work; those questions live elsewhere.

Total Phase 10-FF compute: 0 new PPO, 0 new harness code. Anchor
capture: ~30-60 min including inspection. Analysis: ~15 min.

## 2. Purpose

The current global parhelic-belt rule is a single scalar:
`--parhelic-y-offset-r22 = -0.05`, applied uniformly across the
calibration set. p13's committed anchor (`sun=(543,372)`, `R22=211`,
inferred `h ≈ 6.83°`) leaves a `+10.4 px` belt-y residual — within
the 12 px primitive threshold but on the watch-list per the
`Per-Inversion-Route Residual Table`.

Two competing hypotheses for the residual:

1. **Photo-specific quirk.** p13's parhelia are at a particular spot
   on the parhelic circle that is offset by ~10 px in y from where
   the global rule predicts, for reasons specific to that photo's
   capture geometry or framing.
2. **Low-h pattern.** The global `-0.05 * R22` rule is too tight at
   low sun altitudes; the correct belt-y offset is a function of `h`,
   not a constant. The atmospheric / geometric origin is unimportant
   for this question; we are testing whether the residual replicates.

The two hypotheses make different predictions about other low-h photos.
This spec is the test.

## 3. Scope

Phase 10-FF owns:

- Anchor capture for the low-h replication set (FF anchors).
- Belt-y residual measurement against the current global rule.
- Pre-registered predictions FF1, FF2, FF3 (below).
- Result note at `docs/calibration/PHASE10_BELT_Y_RESULTS.md`.
- A small status update to `RICH_DISPLAY_OVERLAY_NOTES.md` on conclusion.

Phase 10-FF does **not** own:

- Refining the global `--parhelic-y-offset-r22` rule. If FF1 confirms,
  that work is filed as a follow-on (likely a primitive-promotion
  pass).
- CZA-apex or tangent-curvature work.
- Lens-optics or controlled-optics calibration.
- Any inversion-route promotion change.

## 4. Anchor Capture Protocol

### 4.1 Candidate set

From [`PHASE10_EXPANSION_TRIAGE.md`](PHASE10_EXPANSION_TRIAGE.md),
filter to low-h candidates with visible parhelia / belt evidence:

| id | rough sun px | rough h regime | role |
| --- | ---: | --- | --- |
| p13 | `(543, 372)` | h ≈ 6.83° | already anchored; baseline +10.4 px residual under test |
| p27 | `(596, 559)` | h ≈ 0.5° | already anchored; baseline belt residual to compute |
| p18 | `(269, 108)` | low / mid | new anchor — lower-confidence candidate; small image and fuzzy belt, use after stronger candidates |
| p22 | `(646, 449)` | low / mid | new anchor — priority candidate; strong bilateral parhelia / belt reference |
| p25 | `(490, 189)` | low / mid | new anchor — usable parhelia / belt reference, but foreground / flare contamination raises anchor risk |
| p26 | `(465, 203)` | low / mid | new anchor — priority candidate; strong bilateral parhelia and horizontal belt |
| p30 | `(700, 933)` | low | new anchor — priority candidate; rich low-sun 22° halo / parhelia reference |

p7 (h = 59.4°) and p2 (h = 18.6°) are out of the low-h regime for the
test, but p2 may be recorded as an above-regime reference if its belt-y
residual is computable on the existing anchor.

Minimum FF coverage: anchors for p13, p27, and at least three of p18 /
p22 / p25 / p26 / p30. Five anchor cells total at floor; seven if all
listed candidates anchor cleanly.

Visual sanity-check order, 2026-05-13:

1. Anchor **p22**, **p26**, and **p30** first. These are the cleanest belt
   candidates by eye.
2. Anchor **p25** next if one of the first three fails or if broader coverage
   is cheap.
3. Anchor **p18** last among the primary set; it is valid to try, but its
   low resolution and weaker belt make it a lower-confidence cell.
4. Reserve lane if the first five do not produce at least three new low-h
   anchors:
   - **p20** first reserve. It is not CZA-eligible, but it has low-sun
     parhelia / belt evidence; the right side is contaminated by watermark
     and flare, so record single-side or asymmetric confidence if needed.
   - **p19** second reserve. The CZA interpretation was demoted, but the
     R22 ~= 270 px / h ~= 11° provisional anchor makes it a usable low-sun
     halo-separation and belt candidate if the treeline / edge parhelia can
     be measured cleanly.
   - **p28** emergency reserve only; low resolution and haze make it weaker
     than p20 / p19.

If a candidate anchor yields `h > 20°`, drop it from the low-h FF set and
pull the next reserve. The result note may still record the photo as an
above-regime reference, but it does not count toward FF1/FF2/FF3 coverage.

### 4.2 Per-photo capture

Each new anchor records:

- `sun_px` — `(x, y)` of sun centroid in image pixels
- `R22_px` — measured 22° halo radius in pixels (saturation-ring fit
  or visible-arc fit)
- `parhelion_left_px` — `(x, y)` of left parhelion peak
- `parhelion_right_px` — `(x, y)` of right parhelion peak
- `parhelic_belt_y_px` — y-coordinate of the parhelic belt at the
  parhelion azimuth (or symmetric average if both parhelia are visible)

Saved to `docs/calibration/p<NN>-anchor.json` in the existing
per-photo anchor schema. New file per anchor; do not overwrite p13 or
p27.

### 4.3 Anchor-capture guardrails

Per the existing calibration-anchor discipline memory:

- Use saturation-centroid + warm-index + crosshair-crop confirmation
  for sun_px.
- Refine R22_px with the saturation-ring fit, not a visual ring guess.
- If `parhelion_left_y` and `parhelion_right_y` differ by `>= 5 px`,
  record both separately and treat the photo as a candidate for
  parhelic-tilt analysis, not just belt-y.

## 5. Measurement Protocol

For each anchored photo, compute:

- `h_inferred = arccos(R22_px / parhelion_offset_px)` (mean of left/right
  offsets if available; otherwise the single observed offset).
- `predicted_belt_y = sun_y + (-0.05 * R22_px)` (the current global
  rule).
- `belt_y_residual_px = observed_parhelic_belt_y_px - predicted_belt_y`
- `belt_y_residual_pct_R22 = belt_y_residual_px / R22_px`

Tabulate as `belt-y-residuals.csv` with columns:

| photo | h_inferred | R22_px | predicted_belt_y_px | observed_belt_y_px | belt_y_residual_px | belt_y_residual_pct_R22 |

Sort by `h_inferred` ascending.

## 6. Pre-Registered Predictions

### 6.1 (FF1) Belt-y residual replicates as a low-h pattern

`|belt_y_residual_px| >= 5 px` on at least three of the five-or-more
low-h photos in the FF set, with **same sign** across those photos and
a Spearman rank correlation `|ρ(h, belt_y_residual_px)|` outside
`[-0.3, +0.3]`.

**Falsifier (FF1-A, "photo-specific"):** at most one low-h photo
besides p13 has `|belt_y_residual_px| >= 5 px`, OR the residuals
disagree in sign, OR the rank correlation falls inside `[-0.3, +0.3]`.

### 6.2 (FF2) The residual is altitude-dependent

If FF1 confirms (pattern, not photo-specific), then the residual
should also be ordered: belt-y residual at the lowest-h photo should
exceed belt-y residual at the highest-h photo in the FF set by at
least 3 px.

**Falsifier (FF2-A, "constant pattern"):** residuals are uniformly
~5-10 px across the low-h band without monotone ordering by `h`.

FF2 only triggers if FF1 confirms. If FF1 falsifies, FF2 is `not
gated`.

### 6.3 (FF3) The residual does not breach the primitive threshold

`|belt_y_residual_px| < 12 px` on every anchored low-h photo in the FF
set.

**Falsifier (FF3-A, "primitive-threshold breach"):** at least one
anchored photo records `|belt_y_residual_px| >= 12 px`. That
escalates the watch-list flag to a real failure of the parhelic-belt
primitive under the current global rule and forces an immediate
primitive-promotion pass (out of scope for this spec).

FF3 is a guardrail, not an exploratory prediction. Its main role is
catching a worst-case shift before downstream docs are touched.

## 7. Outputs

```
docs/calibration/
  p18-anchor.json       (new, if p18 anchors)
  p22-anchor.json       (new, if p22 anchors)
  p25-anchor.json       (new, if p25 anchors)
  p26-anchor.json       (new, if p26 anchors)
  p30-anchor.json       (new, if p30 anchors)
  PHASE10_BELT_Y_RESULTS.md  (this spec's result note)

results/calibration/phase10-belt-y/
  belt-y-residuals.csv
  belt-y-residual-vs-h.png  (optional; scatter with regression line if FF1 confirms)
  summary.json
```

`summary.json` records FF1 / FF2 / FF3 classification, the rank
correlation, the anchored photo set, and the per-photo residual
table.

## 8. Execution Order

1. **Anchor capture pass.** Anchor at least three of p18 / p22 / p25 /
   p26 / p30. Save per-photo JSON. Add rows to the Anchor Summary
   table in `RICH_DISPLAY_OVERLAY_NOTES.md`. Do NOT update the
   Per-Inversion-Route Residual Table or any verdict surface in this
   step.
2. **Measurement pass.** Compute `belt-y-residuals.csv` and
   `summary.json`. Inspect for sign agreement and rank ordering.
3. **Classification.** Read FF1 / FF2 / FF3 against the table. Record
   conclusions.
4. **Result note.** Draft `PHASE10_BELT_Y_RESULTS.md` with FF
   classifications and the residual table.
5. **Cascade (only if conclusion is non-trivial).**
   - If FF1 falsifies (photo-specific): update the Per-Inversion-Route
     Residual Table's belt row to retire the watch-list flag.
   - If FF1 confirms and FF2 falsifies (constant pattern): file a
     primitive-promotion follow-on to refine the global rule.
   - If FF1 confirms and FF2 confirms (altitude-dependent pattern):
     file a primitive-promotion follow-on for a per-altitude belt-y
     model.
   - If FF3 falsifies: stop, escalate, before any other cascade.

## 9. Exit Criterion

Phase 10-FF (belt-y) is complete when:

- At least five low-h photos have committed anchors (p13 + p27 +
  ≥3 new).
- `belt-y-residuals.csv` and `summary.json` are written.
- FF1, FF2, FF3 are classified.
- `PHASE10_BELT_Y_RESULTS.md` is written with the classification table
  and the per-photo residual table.

## 10. Cross-References

- **Phase 10 closeout:** [`RICH_DISPLAY_OVERLAY_NOTES.md`](RICH_DISPLAY_OVERLAY_NOTES.md)
  Single-handle closeout subsection.
- **Crossover framework:** [`../MESA_CROSSOVER_NOTE.md`](../MESA_CROSSOVER_NOTE.md)
  Geometry Phase 10 closeout section (three-gate taxonomy).
- **Triage source:** [`PHASE10_EXPANSION_TRIAGE.md`](PHASE10_EXPANSION_TRIAGE.md)
  Candidate inventory.
- **Anchor discipline:** project memory
  `feedback_calibration_anchor_discipline.md` (saturation-centroid +
  warm-index + crosshair-crop protocol).
- **Existing low-h anchors:** `p13-anchor.json`, `p27-anchor.json`.

## 11. What Follows

Outcome shapes mapped to next action:

| FF1 | FF2 | FF3 | next action |
| --- | --- | --- | --- |
| confirms | confirms | passes | file primitive-promotion follow-on for per-altitude belt-y model; retire watch-list flag; record altitude-dependent rule as new primitive |
| confirms | falsifies | passes | file primitive-promotion follow-on for a constant-offset refinement of the global rule; retire watch-list flag |
| falsifies | n/a | passes | retire watch-list flag; record p13 as a photo-specific quirk; no primitive change |
| any | any | falsifies | stop, escalate. Primitive-threshold breach forces an out-of-scope primitive review before any other Phase 10 work continues |

Phase 10 Open Questions #2 (tangent-curvature tooling) and #3
(lens-optics test) are independent of FF1/FF2/FF3 outcome. They can
spawn as separate threads at any time.

## 12. Versioning

- **2026-05-13 (FF, initial pin)** — Phase 10 closeout follow-on #1.
  Pre-registered against the closeout's three-open-questions list.
  Inherits mesa v3.8 discipline (pre-registered predictions, decision
  rules before measurement, result-note discipline).
