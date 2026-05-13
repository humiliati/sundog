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
| p20 | `(1011, 827)` | 455 | 457 / 457 | ~5° | **Task #55 fallback check.** Plausible R22 anchors put the CZA apex above the frame (`y ≈ -83`), so p20 is not CZA-eligible. Anchor file: `p20-anchor.json`. |
| p27 | `(596, 559)` | 219 | 219 / 219 | ~0.5° | **Task #55 CZA expansion check.** Strong visible CZA, but visual apex `(599,142)` sits below predicted `(596,121)`, giving +21 px y residual under the notes' observed-minus-predicted convention. Anchor file: `p27-anchor.json`. |

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

Mesa v3.8 crossover addendum: Phase 11 metric review must keep three
quantities separate: visual/variance salience, route-local residual or
sensitivity, and full-overlay fit. The mesa PC1 result is the warning case:
a component can be variance-heavy and locally sensitive without reproducing
the full multi-component mechanism. For geometry, this means no "arc X
contributes N% of fit" score, no aggregation that hides route-specific
conflicts, and no promotion when a primitive improves one route while
worsening another unless the scope is explicitly narrowed.

Task #54 / Phase 11 measurement schema, inherited from v3.8:

| field | what it records | cannot be used as |
| --- | --- | --- |
| visual salience | how strong / obvious the arc is in the photograph | proof that the arc is load-bearing for inversion |
| route-local residual | residual for the named inversion route at that photo's anchor | total overlay quality |
| route-local sensitivity | whether small anchor / sample changes swing the route residual | primitive importance percentage |
| full-overlay interaction | whether tuning this primitive improves or worsens other routes | an averaged global score |
| partition/conflict flag | shared across photos/routes, photo-specific, or actively conflicting | a reason to hide the conflict in aggregation |

Promotion decisions must name which field they rely on. A primitive that is
visually salient but route-local unstable stays vocabulary; a route that
improves one residual while worsening another is a conflict receipt, not a
mean score to optimize away.

## Per-Inversion-Route Residual Table

This table is the Phase 10 response to the mesa finding that forward and
inverse directions are not symmetric. Do not infer that a good parhelion-offset
fit makes every route good.

| route | p2 residual | p7 / p13 residual | expansion residuals | promotion status | note |
| --- | ---: | ---: | ---: | --- | --- |
| Parhelion offset → sun altitude | −1 / −1 px (0.5% R₂₂) | p7: 0 / n/a px; p13: 0 / 0 px (new anchor; x-offset only) | p20 / p27 are provisional low-sun supports, not promotion measurements | **promoted** (calibrated core; **probation cleared 2026-05-13**) | Task #52 step 1 verdict: probation was anchor-driven, not route-driven. Re-anchoring p13 drops x-offset residual from +6/+8 px to ~0/0 px on the new (sun=543,372; R₂₂=211; offsets=213/212) anchor. Route stays in calibrated core; verdict reads as 'bad anchor, route OK'. p13 belt-y residual is tracked separately. |
| CZA apex → sun altitude / visibility | x: −6.7 px, **y: −19.3 px (−10.6% R₂₂)** | p7 not applicable (h = 59.4° > 32.2° cutoff); p13 cropped (predicted apex y = −50) | p27: x +3 px, **y +21 px (+9.6% R₂₂)**; p19 / p20 cropped | **fails residual gate on expanded set** | Task #55 corrects the earlier interpretation: p2 and p27 both exceed the 8 px / 0.04*R22 route threshold, but their signs are opposite. This is not a stable direction-and-magnitude atlas bias. It is a clean negative for CZA-apex inversion promotion and a weaker Mesa #3 receipt: parhelion-offset remains ~0-1 px while CZA-apex is photo-specific and unreliable. Anchors: `p2-anchor.json`, `p27-anchor.json`, `p20-anchor.json`. |
| Tangent-arc curvature → sun altitude | **detection-degenerate** (fuses with halo top at h=18.6°) | p7 (h=59.4°): column-peak grabs halo outer edge not broad tangent; p13 (h=6.83°): chromatic-haze contamination, no clean smile to fit | p27 (h=0.5°): merges with CZA / sun bloom at horizon transition; p20 not measured | **fails detection across all 4 eligible photos** (v3.8 partition: detection-degenerate, not residual-bounded) | Task #54 first pass 2026-05-13 found the route fails column-peak detection on every photo in the calibration set, with a *different* failure mode at each altitude. Not a residual-gate failure — the residual was never measurable. The earlier 'altitude-regime validity window' framing was too generous; the route's degeneracy is photo-and-feature-specific across the entire altitude range. Promotion blocked until a different detection method (edge-based gradient tracking, template matching, or manual sampling) is built and verified. |
| Supralateral position → sun altitude | _measurement pending_ | p7 cropped; p13 not visible | p27 candidate; p20 weak/cropped | pending expansion | p27 may reopen supralateral coverage, but no residual has been measured. Keep the previous fail verdict for the original p2/p7/p13 set; do not promote the expanded route without p27 anchors. |
| Parhelic-belt y residual (primitive, not route) | not separately measured (Phase 2 set only had x-residuals) | p7 not applicable; p13 +10.4 px (4.9% R₂₂) on new anchor | p20 / p27 not scored for belt promotion | **tracked separately** (below 12 px primitive threshold; under probation watch) | This is a *primitive vertical placement* residual, not a route residual. Comes from the shared `--parhelic-y-offset-r22 = -0.05` setting applied uniformly across the calibration set. At p13's corrected h ~= 6.8°, the parhelion peak sits ~10 px above the predicted belt y; below the 12 px primitive cutoff but worth re-checking after p2 step-2 anchors land, since the global offset may want per-altitude refinement at low h. |

## Tangent-Curvature v3.8 Receipt *(updated 2026-05-13)*

Task #54 measurement pass across all four eligible photos (p2 h=18.6°,
p7 h=59.4°, p13 h=6.83°, p27 h=0.5°) produced a strong Mesa #5 receipt:
**the upper-tangent inversion route is detection-degenerate across the
entire calibration altitude range, with a different failure mode at each
photo.** This is a publishable clean-negative rather than a measurement
backlog item.

Per-photo v3.8 schema:

| photo | h | salience | route-local residual | route-local sensitivity | full-overlay interaction | partition/conflict |
| --- | ---: | --- | --- | --- | --- | --- |
| p2 | 18.6° | high (b-contrast 35.7) | unmeasurable: samples hit halo top within 5 px | n/a — fit degenerate | depends on sun_x as expected | tangent fuses with 22° halo top at this h (compression regime) |
| p7 | 59.4° | broad-tangent visible | unmeasurable: column-peak grabs halo outer edge, not broad tangent above | n/a | n/a | broad-tangent regime; halo's outer edge dominates the detection signal |
| p13 | 6.83° | medium (b-contrast 15.8 = 23% sky) | RMS=41 px on saturation fit; radius 196 vs predicted 1774 | ±5 px wiggle → ±1.2 px Δr (misleadingly low; fit is already stuck) | n/a | chromatic-haze contamination; no clean smile to fit |
| p27 | 0.5° | tangent essentially absent between CZA and halo top | both fits have RMS=22 px with non-physical centers (brightness fit center *below* sun) | n/a | n/a | tangent merges with CZA / sun bloom at horizon transition |

The Mesa #5 reading: visual salience, route-local residual, sensitivity,
and full-overlay interaction are **not the same property**. The tangent
arc is atmospherically real and visible in all four photos. It still
fails to dominate the route-local detection signal in any of them. A
primitive that exists, is visible, and is geometrically meaningful can
still be route-local-unmeasurable under a given detection method. The
v3.8 schema's job was to record those distinctions; it correctly
prevented a "the tangent route works" claim that the data does not
support.

Atmospheric-optics atlas-model implication: the upper tangent arc's
brightness/chromatic peak does NOT generally coincide with its
geometrically-predicted spine. The arc's *visual identity* lives in
combined brightness+chromaticity, not in a single peak that column-wise
detection can grab. Real measurement needs either gradient-based edge
detection (the spine is at a brightness or chromaticity *transition*, not a
peak) or manual sample selection from visual crops.

This receipt also weakens the earlier framing that p2 alone was the
exception. The other three altitudes are degenerate too, just in
different ways. **Three of the four alternative-to-parhelion-offset
inversion routes now have measured failure modes on the current
calibration set**: CZA-apex (opposite-sign residuals exceed threshold),
supralateral (coverage gate), and tangent-curvature (detection-degenerate).
Only parhelion-offset remains as a viable inversion route. That is a
substantively stronger Mesa #3 receipt than the earlier "different routes
have different residuals" — it is "one route works, three others fail at
different layers of the measurement stack."

## Vocabulary Classification

| primitive | p2 | p7 | p13 | promote? |
| --- | --- | --- | --- | --- |
| CZA | visible | not applicable / high sun | cropped or not visible | Conditional core: render only inside the CZA sun-altitude validity window; CZA-apex inversion route fails residual gate on the expanded p2/p27 set. |
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
| CZA primitive (rendered) | **promoted as conditional core** | Render only when `h < 32.2 deg` (atmospheric cutoff). Visible on p2 at h = 18.6 deg and p27 at h ~= 0.5 deg; correctly not applicable on p7. p13 / p19 / p20 are low-altitude but cropped above the predicted apex. The primitive remains core while the CZA-apex inversion route fails residual gate. |
| CZA apex to h inversion route | **fails residual gate on expanded set** | Task #55 added p27 as a second eligible CZA photo. p2 residual is y = -19.3 px; p27 residual is y = +21 px. Both exceed the 8 px route threshold, but the direction does not replicate. Verdict: do not promote; this is a clean negative for a simple CZA-apex inverse. |
| Tangent-arc curvature to h inversion route | **fails detection gate on all four eligible photos** | Task #54 (2026-05-13) measured the route across p2 (h=18.6°), p7 (h=59.4°), p13 (h=6.83°), and p27 (h=0.5°). Column-peak detection produces no measurable curvature fit at any altitude, with a *different* failure mode at each photo: compression into halo top at p2; halo outer edge dominance at p7; chromatic-haze contamination at p13; CZA / sun-bloom merge at p27. The arc is atmospherically real and visible in all four photos, but its brightness/chromatic spine does not dominate any per-column signal. The earlier "altitude-regime validity window" framing was too generous; the route is detection-degenerate across the entire calibration altitude range. Promotion blocked until a non-column-peak detection method (gradient-based edge tracking, template matching, or manual sampling) is built and verified. See the Tangent-Curvature v3.8 Receipt section above for the per-photo v3.8 schema. |
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
3. **Supralateral inversion route on the current committed evidence.**
   Coverage fails (only p2 eligible). The primitive itself stays in
   optional vocabulary. p27 is documented as a candidate but no
   supralateral residual has been measured; the route does not reopen
   without explicit anchor capture and a second eligible residual.
4. **Tangent-arc curvature inversion route under the column-peak detection
   protocol.** Detection-degenerate across all four eligible photos with
   three distinct failure modes (Task #54). Reopening the route requires
   a different detection method, not new photos.
5. **CZA-apex inversion route on the expanded p2/p27 set.** Residual gate
   fails with opposite-sign residuals (p2: y = -19.3 px; p27: y = +21 px),
   so the failure is route-reliability, not a stable systematic.
6. **Lower tangent as core logo geometry.** Single-photo evidence; usable
   only as labeled low-sun annotation or in named-pose examples, not in core
   marks.
7. **Any linear arc-importance attribution metric** at Phase 11 review.
   Per Mesa crossover rule 5.

### Single-handle closeout

The Phase 10 gate closed 2026-05-13. **One inversion route is promoted to
calibrated core; the other three candidate inversion routes fail at three
distinct layers of the measurement stack.**

| route | gate outcome | failure layer |
| --- | --- | --- |
| Parhelion offset → h | **promoted** (calibrated core) | none — passes residual gate |
| CZA apex → h | fails | **residual gate** (p2 and p27 both exceed `0.04 * R22`, opposite signs) |
| Supralateral → h | fails | **coverage gate** (only p2 eligible on the committed set; p27 candidate but unmeasured) |
| Tangent-arc curvature → h | fails | **detection gate** (column-peak protocol fails on all four eligible photos, three distinct degeneracy modes) |

Three independent failure layers is a stronger Mesa #3 receipt than
"different routes have different residuals." It is "one route works, three
others fail at different layers of the measurement stack." The atlas is
**rich in forward generation** (`h → all primitives`) but supports
**one image-recoverable inverse handle** (parhelion offset), not a
redundant four.

Phase 11 can proceed against the visibility-promoted vocabulary now.
Three open questions are filed but not blocking:

1. **Lens-optics test.** Would a controlled-optics calibration set resolve
   CZA-apex's photo-specific direction inconsistency into a stable
   systematic? Mesa #4-style falsification surface.
2. **Tangent-curvature tooling.** Does a gradient-based edge-tracking or
   template-matching detector recover the route on existing photos? This
   is a tooling question, not a physics question — the arc is real and
   visible; the column-peak method is the wrong instrument.
3. **Parhelic-belt-y replication.** Does p13's +10.4 px belt-y residual
   replicate across multiple low-h photos, or is it photo-specific?

Each is scoped well enough to file as a separate task when attacked. None
gate Phase 11.

## Anchor-Capture Closeout (Tasks #52, #53, #54, #55)

All four anchor-capture tasks closed 2026-05-13. The execution-order
discipline below is preserved as historical record; outcomes are recorded
inline.

Closeout summary:

- **Task #52 step 1 — re-anchor p13 parhelion offset:** done. p13
  re-anchored to (sun=543,372; R22=211; offsets=213/212), dropping the
  x-offset residual to ~0/0 px and revealing p13 is a low-altitude photo
  at h ~= 6.83°, not the mid-altitude h = 17.3° the rough hand-anchor
  implied. Parhelion-offset route probation cleared with verdict "bad
  anchor, route OK." The p13 belt-y residual (+10.4 px) is tracked
  separately under the Per-Inversion-Route Residual Table.
- **Task #53 / #55 — CZA apex anchors:** done. p2 residual y = -19.3 px;
  p27 residual y = +21 px. Both exceed the `0.04 * R22` / 8 px route
  threshold, opposite signs. CZA-apex inversion route fails the residual
  gate on the expanded set; not a stable systematic.
- **Task #54 — tangent-arc curvature sampling:** done. Column-peak
  detection fails on all four eligible photos with three distinct
  degeneracy modes (compression at p2, halo-edge dominance at p7,
  chromatic-haze at p13, CZA / sun-bloom merge at p27). See the
  Tangent-Curvature v3.8 Receipt section above. Tangent-arc curvature
  inversion route fails the detection gate.
- **Supralateral inversion:** coverage gate failure on the current
  committed set (only p2 eligible). p27 remains a candidate but no
  supralateral residual has been measured; the route does not reopen
  without explicit anchor capture.

Historical execution order (now retired):

1. **Re-anchor p13 parhelion offset first** — done. Replaced rough
   hand-anchor with measured (sun_px, R22_px, parhelion_left_px,
   parhelion_right_px) tuple; new residual ~0/0 px. Anchor Summary and
   Per-Inversion-Route Residual Table updated.
2. **CZA apex anchor check (Task #53, expanded by Task #55).** p2 apex
   captured (residual y = -19.3 px). Task #55 added p27 (residual
   y = +21 px). Opposite signs convert the verdict from coverage-gate
   failure to residual-gate failure on the expanded set.
3. **Tangent-arc curvature anchors (Task #54).** Four eligible photos
   sampled (p2 / p7 / p13 / p27). No measurable curvature fit at any
   altitude; column-peak detection grabs the wrong feature in a
   photo-specific way. Verdict: detection-gate failure; route blocked
   pending a non-column-peak detector.
4. **Supralateral inversion stays failed unless a new visible-supralateral
   reference enters the calibration set.** Coverage rule is unchanged
   (`>= 2 eligible photos`). p27 is documented as a candidate but
   unmeasured; the route does not reopen without explicit anchor capture
   and a second eligible residual.