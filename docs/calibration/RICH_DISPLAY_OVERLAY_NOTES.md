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
| p13 | `(557, 372)` | 210 | 220 | 17.3° | Rough hand anchor for social-crop morphology; dagger residuals are +6 px / +8 px against rough parhelion marks. |

All generated overlays now apply `--parhelic-y-offset-r22 = -0.05`, raising
the parhelic belt and dagger markers by 5% of the observed 22° halo radius.
This fixes a shared vertical belt residual without changing the 22°/46° halo
registration.

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
| Parhelion offset → sun altitude | −1 / −1 px (0.5% R₂₂) | 0 / n/a px | +6 / +8 px (3.8% R₂₂) | **promoted** (calibrated core; anchor-probation) | Threshold gate passes: only p13 hits ≥ 8 px, and a single failure is below the ≥2 photos rule. Because it touches the cutoff, the route stays on anchor-probation until p13 re-anchoring in Task #52. |
| CZA apex → sun altitude / visibility | _measurement pending_ | not applicable (h = 59.4° > 32.2° cutoff) | _measurement pending_ | pending — needs anchor capture | Valid only below the CZA cutoff. Anchor work in Task #52. |
| Tangent-arc curvature → sun altitude | _measurement pending_ | _measurement pending_ (broad tangent visible at high sun) | _measurement pending_ | pending — needs anchor capture | Curvature is non-monotone in h; do not collapse into total overlay fit. Anchor work in Task #52. |
| Supralateral position → sun altitude | _measurement pending_ | not applicable (cropped) | not applicable (not visible) | **fails coverage gate** | Pre-registered rule: fewer than two eligible photos. Stays as optional vocabulary, but does not promote as a calibrated inversion route on this set. |

## Vocabulary Classification

| primitive | p2 | p7 | p13 | promote? |
| --- | --- | --- | --- | --- |
| CZA | visible | not applicable / high sun | cropped or not visible | Conditional core: render only inside the CZA sun-altitude validity window; inversion route remains pending. |
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
| Parhelion offset to h inversion route | **promoted** (calibrated core; anchor-probation) | Threshold gate: only p13 reaches the 8 px cutoff, and the rule requires >= 2 photos failing. Because p13 touches the cutoff and uses a rough hand-anchor, the route remains on anchor-probation until re-anchoring in Task #52. |
| 22 deg halo, parhelia (L/R), parhelic-circle | **promoted** (calibrated core) | Already in core via Phase 2. No regression observed in Phase 10 inspection. |
| Upper tangent arc | **promoted as logo / animation vocabulary** | Visible across p2 and p7; candidate on p13. Two-of-three eligible visibility. Curvature-as-inversion-route remains pending measurement, but visibility is sufficient for shape-language use. |
| CZA primitive (rendered) | **promoted as conditional core** | Render only when `h < 32.2 deg` (atmospheric cutoff). Visible on p2 at h = 18.6 deg; correctly not applicable on p7. p13 cropped/not visible but not contradictory. Inversion route stays pending. |
| CZA apex to h inversion route | **pending** | No anchor capture yet on p2 / p13. Cannot be promoted to inversion alternative without numbers; cannot be ruled out either. Task #52 captures the anchor. |
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
2. **CZA apex anchors on p2 and p13.** Capture observed CZA apex (x, y)
   in photo px. Add a `CZA apex (px)` column to the Anchor Summary table.
   Compute the CZA-apex → h inversion residual (predicted apex from atlas
   geometry at the inferred h, vs. observed). Fill the p2 and p13 cells in
   the Per-Inversion-Route table. p7 stays not applicable (h = 59.4° is
   above the CZA cutoff).
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
currently accepts the relevant p13 anchor inputs via CLI flags
(`--sun`, `--r22`, `--parhelion-left`, `--parhelion-right`,
`--parhelion-y`, and `--parhelion-offset`) and already supports
`--cza-apex X,Y`. For step 3 it still needs either tangent-sample flags
such as `--tangent-samples X1,Y1;X2,Y2;X3,Y3` or a JSON-anchor mode that
can carry p13 parhelion anchors, CZA apex, and tangent samples together.

### Escalation rule (added with the 2026-05-13 verdict)

If step 1 returns a p13 parhelion-offset residual at `>= 12 px` or
`>= 0.06 * R22`, the route still does not formally fail the two-photo
gate — only one photo is over threshold. But it loses the "rough hand
anchor" excuse that currently keeps the probation status soft.

Consequence: the parhelion-offset route stays in calibrated core for atlas
rendering, but it cannot serve as clean evidence in the perception roadmap
(`SUNDOG_V_PERCEPTION.md`) predicted-then-observed receipt, or in any
public-claim language that depends on Phase 10 evidence, until either
a) a third measurement clears it on a fresh photo, or
b) the p13 photo itself is replaced with one whose anchors are clean.

This is the same shape as mesa's "the negative is the deliverable" rule
applied locally: an unresolved probation flag is its own gate, separate
from the route's promotion status.
