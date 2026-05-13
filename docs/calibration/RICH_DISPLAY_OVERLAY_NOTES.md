# Rich-Display Overlay Notes

Phase 10 of `docs/SUNDOG_V_GEOMETRY.md` uses these notes to keep the
optional vocabulary layers separate from the calibrated parhelion core.
Image 1 is the label key; images 2, 7, and 13 are the first tuning set.

## Artifacts

| image | overlay | role |
| --- | --- | --- |
| `1.Photometeor-jeff-mod_marked_red.jpg` | source annotation | Label key for CZA, supralateral, suncave Parry, upper tangent, Parry supralateral candidate, infralateral, parhelic circle, lower tangent, parhelia, 22° halo, and 46° halo. |
| `2.Photometeor-jeff_mod_red.jpg` | `overlays/p2.rich-vocabulary.overlay.png` | Clean modern version of the annotated display; strongest rich-vocabulary tuning source. |
| `7.625544777_10241089047341957_4435817776770420050_n.jpg` | `overlays/p7.rich-vocabulary.overlay.png` | High-sun / partial-visibility stress case. |
| `13.480859565_17934474635991868_323320248088719839_n.jpg` | `overlays/p13.rich-vocabulary.overlay.png` | Square social-crop / logo-crop tuning source. |

## Anchor Summary

| image | sun px | R22 px | parhelion offset | inferred h | note |
| --- | ---: | ---: | ---: | ---: | --- |
| p2 | `(567, 496)` | 182 | 192 | 18.6° | Reuses the clean multi-photo calibration anchor; dagger residuals are -1 px / -1 px. |
| p7 | `(1033, 946)` | 200 | 393 | 59.4° | Reuses the high-sun anchor; CZA should be treated as not applicable at this altitude even when the vocabulary overlay draws it. |
| p13 | `(543, 372)` | 211 | 213 / 212 | 6.83° | **Re-anchored 2026-05-13 (Task #52 step 1).** Supersedes rough hand-anchor: sun_x corrected −14 px, offsets refined from assumed-symmetric 220 to measured 213/212, h drops from 17.3° to ~6.8°. p13 is a low-altitude photo, not mid-altitude. Anchor file: `p13-anchor.json`. |

All generated overlays now apply `--parhelic-y-offset-r22 = -0.05`, raising
the parhelic belt and dagger markers by 5% of the observed 22° halo radius.
This fixes the shared vertical belt bias without changing the 22°/46° halo
registration. Track that belt-height residual separately from the
parhelion-offset inversion route: p13's committed anchor clears the x-offset
route, but its JSON `parhelion.y = 351` still leaves a +10.4 px belt-y
residual against the current `-0.05 * R22` overlay rule.

## Phase 10 Measurement Pre-Registration

Geometry resolution, 2026-05-13: Mesa's crossover note makes the residual
table and negative thresholds load-bearing for the next Phase 10 run.

- The `per-inversion-route` residual table lives in this file, alongside the
  per-photo overlay notes. It is a Phase 10 deliverable, not a separate
  document, because the route residuals need to sit next to the photo-level
  visibility classifications that determine whether each route is applicable.
- The quantitative `do not promote` thresholds are a Phase 10 blocker for
  promotion. If the thresholds are not written before an overlay run, that run
  is exploratory only and cannot move a primitive into default/logo language.
- Thresholds are evaluated at the image anchor scale and normalized by each
  photo's measured `R22`.

Pre-registered `do not promote` thresholds:

| candidate class | do not promote if... |
| --- | --- |
| Optional vocabulary primitive | feature residual is `>= 0.06 * R22` or `>= 12 px` on at least two eligible photos |
| Inversion route | route residual is `>= 0.04 * R22` or `>= 8 px` on at least two eligible photos |
| Coverage | fewer than two eligible photos show the primitive/route clearly enough to measure |
| Attribution metric | metric reports a linear "arc X contributes N%" score instead of a route/primitive residual |

`Eligible` means the feature is visible, uncropped enough to measure, and
inside its sun-altitude validity window. Hidden, cropped, or high-sun invalid
features are recorded as `not applicable`, not as failures.

Probation rule, added with the 2026-05-13 verdict: a single eligible photo
touching a cutoff does not fail a route or primitive under the pre-registered
`>= 2 photos` rule, but it does put that item on anchor-probation until the
next anchor-capture pass confirms or clears the residual.

## Per-Inversion-Route Residual Table

This table is the Phase 10 response to the mesa finding that forward and
inverse directions are not symmetric. Do not infer that a good parhelion-offset
fit makes every route good.

| route | p2 residual (L / R) | p7 residual (L / R) | p13 residual (L / R) | promotion status | note |
| --- | ---: | ---: | ---: | --- | --- |
| Parhelion offset → sun altitude | −1 / −1 px (0.5% R₂₂) | 0 / n/a px | 0 / 0 px (new anchor; x-offset only) | **promoted** (calibrated core; **probation cleared 2026-05-13**) | Task #52 step 1 verdict: probation was anchor-driven, not route-driven. Re-anchoring p13 drops x-offset residual from +6/+8 px to ~0/0 px on the new (sun=543,372; R₂₂=211; offsets=213/212) anchor. Route stays in calibrated core; verdict reads as 'bad anchor, route OK'. p13 belt-y residual is tracked separately. |
| CZA apex → sun altitude / visibility | _measurement pending_ | not applicable (h = 59.4° > 32.2° cutoff) | not applicable (cropped; predicted apex y = −50) | **fails coverage gate on current set** | Step 1 re-anchoring moves p13 to h = 6.83°, making the CZA apex fall above the square crop. With p7 above the CZA cutoff, only p2 remains eligible; p2 anchor capture can be recorded, but this route cannot promote without an added CZA-visible photo. |
| Tangent-arc curvature → sun altitude | _measurement pending_ | _measurement pending_ (broad tangent visible at high sun) | _measurement pending_ | pending — needs anchor capture | Curvature is non-monotone in h; do not collapse into total overlay fit. Anchor work in Task #52. |
| Supralateral position → sun altitude | _measurement pending_ | not applicable (cropped) | not applicable (not visible) | **fails coverage gate** | Pre-registered rule: fewer than two eligible photos. Stays as optional vocabulary, but does not promote as a calibrated inversion route on this set. |
| Parhelic-belt y residual (primitive, not route) | not separately measured (Phase 2 set only had x-residuals) | not applicable (high sun) | +10.4 px (4.9% R₂₂) on new anchor | **tracked separately** (below 12 px primitive threshold; under probation watch) | This is a *primitive vertical placement* residual, not a route residual. Comes from the shared `--parhelic-y-offset-r22 = -0.05` setting applied uniformly across the calibration set. At p13's corrected h ~= 6.8°, the parhelion peak sits ~10 px above the predicted belt y; below the 12 px primitive cutoff but worth re-checking after p2 step-2 anchors land, since the global offset may want per-altitude refinement at low h. |

## Vocabulary Classification

| primitive | p2 | p7 | p13 | promote? |
| --- | --- | --- | --- | --- |
| CZA | visible | not applicable / high sun | cropped or not visible | Conditional core: render only inside the CZA sun-altitude validity window; inversion route fails coverage on the current p2/p7/p13 set. |
| Supralateral arc | visible | candidate only | not visible / cropped | Keep as optional vocabulary; p2 carries the strongest evidence. |
| Upper tangent arc | visible | visible / broad | candidate | Promote as stable logo/animation shape language. |
| Lower tangent arc | visible near lower 22° contact | not clear | not clear | Annotation only until another clean low-sun source supports it. |
| Suncave Parry arc | visible/candidate using image 1 key | weak candidate | weak candidate | Candidate label only; useful for education, not logo default. |
| Parry supralateral arc | candidate in image 1 | not visible | not visible | Do not promote beyond optional annotation. |
| Infralateral arcs | visible on p2 periphery | not visible / cropped | not visible / cropped | Good vocabulary label; not a core logo layer unless the design brief wants a rich-display variant. |

## Phase 10 Promotion Verdict *(landed 2026-05-13)*

Verdicts apply the pre-registered thresholds above. Status is one of:
**promoted** (moves into calibrated core / logo defaults), **promoted as
optional vocabulary** (annotation layer, not core), **fails gate** (held
back; reason recorded), or **pending** (cannot be promoted until measured).

| candidate | verdict | reason |
| --- | --- | --- |
| Parhelion offset to h inversion route | **promoted** (calibrated core; probation cleared 2026-05-13) | Task #52 step 1: re-anchored p13 from (557, 372) / R22=210 / offset=220 to (543, 372) / R22=211 / offsets=213/212. New residual is ~0/0 px. The probation finding is recorded as 'bad anchor, route OK' -- the route was never the problem; the rough hand-anchor was. |
| 22 deg halo, parhelia (L/R), parhelic-circle | **promoted** (calibrated core) | Already in core via Phase 2. No regression observed in Phase 10 inspection. |
| Upper tangent arc | **promoted as logo / animation vocabulary** | Visible across p2 and p7; candidate on p13. Two-of-three eligible visibility. Curvature-as-inversion-route remains pending measurement, but visibility is sufficient for shape-language use. |
| CZA primitive (rendered) | **promoted as conditional core** | Render only when `h < 32.2 deg` (atmospheric cutoff). Visible on p2 at h = 18.6 deg; correctly not applicable on p7. p13 is low-altitude but square-cropped above the predicted apex, so the primitive remains core while the inversion route fails coverage. |
| CZA apex to h inversion route | **fails coverage gate on current set** | Task #52 step 1 moves p13 to h = 6.83°, where the predicted apex is y = -50 and therefore outside the image plane. p7 is not applicable above the CZA cutoff, leaving only p2 as an eligible anchor. A p2 measurement is still useful as a single-photo receipt, but the route cannot promote without a new CZA-visible calibration photo. |
| Tangent-arc curvature to h inversion route | **pending** | Same as CZA apex: visible but unmeasured. Curvature is non-monotone in h, so promotion requires sampling at multiple sun altitudes -- Task #52 is the first p2 + p13 pass. |
| Supralateral position to h inversion route | **fails coverage gate** | Visible only on p2 of the Phase 10 set. Pre-registered rule blocks promotion at < 2 eligible photos. Supralateral primitive itself stays in optional vocabulary; the *inversion route* does not promote on this evidence. |
| Lower tangent arc | **promoted as low-sun annotation only** | Visible at p2's lower 22 deg contact; not clear on p7 or p13. One photo of evidence -- fails the two-photo bar for core promotion. Held as low-sun-display annotation; compatible with `low-altitude.json` named-pose examples, but the pose is not additional evidence. |
| Suncave Parry arc | **fails gate** (weak evidence) | Weak/candidate on all three Phase 10 photos. Below the visibility bar for inversion measurement; insufficient for shape-language promotion. Stays in atlas vocabulary as an educational label only. |
| Parry supralateral arc | **fails gate** (weak evidence + coverage) | Candidate on image 1 key only; not visible on p7 or p13. Promotion blocked on coverage. Vocabulary-label-only status confirmed. |
| Infralateral arcs | **promoted as optional vocabulary** (not core) | Visible on p2 periphery; not clear on p7 / p13. Single-photo evidence on the Phase 10 set, but image 1 key labels it. Held as a rich-display vocabulary option; compatible with `forty-six-halo.json` named-pose examples, but the pose is not additional evidence. |
| Any linear arc-importance score (e.g. "arc X = 23% of fit") | **fails gate** (per Mesa crossover rule 5) | Linear additive attribution of fit is mesa's documented failure mode for field-shaped objects. Reject any Phase 11 metric that produces this. |

### Coverage and threshold summary

Pre-registered gate (from the Phase 10 Measurement Pre-Registration section):

- **Visibility coverage**: a primitive or route needs >= 2 eligible photos
  (visible, uncropped, in sun-altitude validity window) to be a candidate.
- **Primitive residual**: >= 0.06 * R22 **or** >= 12 px on >= 2 eligible
  photos -> do not promote.
- **Inversion-route residual**: >= 0.04 * R22 **or** >= 8 px on >= 2
  eligible photos -> do not promote.
- **Attribution form**: any linear "arc X contributes N%" metric -> reject
  outright.

### Do-not-promote list

Concrete output of the Phase 10 gate, in the form the roadmap asks for:

1. **Suncave Parry as logo / animation shape language.** Weak evidence
   across the entire Phase 10 set. Stays in atlas vocabulary as an
   educational label.
2. **Parry supralateral as anything beyond an optional annotation.**
   Coverage fails -- candidate on image 1 only.
3. **Supralateral inversion route on this evidence.** Coverage fails
   (single-photo visibility). The primitive itself stays in optional
   vocabulary; the *route* does not promote.
4. **Lower tangent as core logo geometry.** Single-photo evidence; usable
   only as labeled low-sun annotation or in named-pose examples, not in core
   marks.
5. **Any linear arc-importance attribution metric** at Phase 11 review.
   Per Mesa crossover rule 5.

The Phase 10 gate is **partially open**: visibility-based promotions are
final; inversion-route promotions beyond parhelion-offset are gated on
Task #52 anchor capture.

## Task #52 Anchor-Capture Scope

Task #52 now serves two gates, not one. The original purpose was opening the
CZA-apex / tangent-arc-curvature inversion-route slots in the Phase 10 table.
After landing the Phase 10 verdict, p13's parhelion-offset residual on its
right side touched the 8 px route cutoff exactly, which puts the
parhelion-offset route on anchor-probation under the rule introduced with
this verdict. So Task #52 also has to confirm or clear the p13 measurement
before any downstream work treats it as clean evidence.

Execution order is fixed; do not interleave:

1. **Re-anchor p13 parhelion offset first.** Replace the rough hand-anchor
   with a properly measured (sun_px, R22_px, parhelion_left_px,
   parhelion_right_px) tuple. This is the cleanup pass for the
   anchor-probation flag. Update the Anchor Summary table and the Per-
   Inversion-Route Residual Table in the same commit. Do not touch CZA or
   tangent anchors until this completes.
2. **CZA apex anchor check (Task #53).** Capture the observed CZA apex
   on p2 in photo px. For p13, first record the crop verdict: at h = 6.83°,
   predicted CZA apex y = -50, above the image plane, so p13 is expected to
   be not applicable rather than pending. Add a `CZA apex (px)` column to
   the Anchor Summary table, fill p2 with the measured residual, and mark
   p7 / p13 not applicable. On the current three-photo set this forces the
   CZA-apex inversion route into coverage-gate failure; reopening it needs
   an added CZA-visible calibration photo.
3. **Tangent-arc curvature anchors on p2 and p13.** Sample at least three
   points along each visible tangent arc; fit a local curvature estimate;
   compare to the atlas-predicted curvature at the inferred h. Curvature is
   non-monotone in h, so the residual must be reported with the local
   slope of `dκ/dh` at the photo's altitude, not just the difference.
   Fill the p2, p7 (where visible), and p13 cells.
4. **Supralateral inversion stays failed unless a new visible-supralateral
   reference enters the calibration set.** Coverage rule is what blocks it,
   not residual magnitude. Adding a single additional photo with a clearly
   visible supralateral arc restarts the coverage clock and reopens the
   route; no other action on p2/p7/p13 changes the verdict.

Practical first move: open
`docs/calibration/13.480859565_17934474635991868_323320248088719839_n.jpg`
in an annotation tool and re-measure the four parhelion-offset anchors
before touching CZA or tangent samples. `scripts/overlay_calibrate.py`
now supports both paths needed for Task #52:

- existing CLI anchors: `--sun`, `--r22`, `--parhelion-left`,
  `--parhelion-right`, `--parhelion-y`, `--parhelion-offset`, and
  `--cza-apex X,Y`;
- new Task #52 anchors: `--anchors PATH`, `--tangent-samples
  "X1,Y1;X2,Y2;X3,Y3"`, and `--tangent-kind upper|lower`.

Use `docs/calibration/task52-anchor.example.json` as the JSON-anchor shape
when carrying p13 parhelion anchors, CZA apex crop verdicts, and tangent
samples together.

### Escalation rule (added with the 2026-05-13 verdict)

If step 1 returns a p13 parhelion-offset residual at `>= 12 px` or
`>= 0.06 * R22`, the route still does not formally fail the two-photo
gate because only one photo is implicated. It does, however, escalate from
soft probation to a hard anchor audit: do not use p13 for CZA, tangent, or
supralateral residuals until the sun, R22, and parhelion anchors are
re-measured and the inferred altitude is recomputed.

## Task #52 Step 1 Verdict *(2026-05-13)*

Step 1 (re-anchor p13 parhelion offset) is **complete**. Headline:

**p13 was a low-altitude photo all along.** The rough hand-anchor's
inferred h = 17.3° was wrong by ~10°; the corrected anchor places p13 at
h ≈ 6.8°, closer to the `low-altitude.json` named pose (h = 5°) than to
the mid-altitude regime.

What changed on re-anchoring:
- `sun_x`: 557 -> 543 (delta -14 px; flares biased the original click right)
- `R22`: 210 -> 211 (delta +1 px; essentially unchanged)
- parhelion offsets: assumed-symmetric 220 -> measured 213 (left) and 212 (right); L-R asymmetry -1 px
- inferred h: 17.3 deg -> 6.83 deg (delta -10.5 deg)

Internal consistency check that locked the verdict: the halo's left
extreme at sun_y is at x = 543 - 211 = 332, and the left parhelion peak
sits at x = 330. The 2 px gap matches the prediction R22/cos(h) - R22 =
211/0.9929 - 211 = +1.5 px (parhelion sits just outside the halo's
horizontal extreme at low h). The whole anchor is internally consistent
to within 1-2 px.

Separate belt