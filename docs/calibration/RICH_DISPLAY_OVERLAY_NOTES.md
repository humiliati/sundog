# Rich-Display Overlay Notes

Phase 10 of `docs/SUNDOG_V_GEOMETRY.md` uses these notes to keep the
optional vocabulary layers separate from the calibrated parhelion core.
Image 1 is the label key; images 2, 7, and 13 are the first tuning set.

> **Post-audit state, 2026-05-13.** This file is in
> mid-campaign rewrite per
> [`../PHASE10_ATTACK_ROADMAP.md`](../PHASE10_ATTACK_ROADMAP.md).
>
> - **Pass B1 landed 2026-05-13** (first technical gate). The
>   parhelion-route residual table now has a per-photo eligibility
>   sub-table with `sec(h) − 1` (geometric lever), `R22-source`
>   (`ring-fit` / `parhelion-derived` / `inferred-other`), and
>   `geometric_validity` columns. Each anchor JSON
>   (`docs/calibration/p*-anchor.json`) carries a top-level
>   `r22_source` + `r22_source_note`; p26's anchor adds a
>   `geometric_validity` block flagging the right-side
>   impossibility (`R22 / right_offset = 1.003 > 1`,
>   `arccos` undefined). Rationale: synthetic optical audit
>   ([`PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md`](PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md))
>   §2 items 4–7 and §2 item 6.
> - **Phase 10 Promotion Verdict and Single-handle closeout below
>   are pre-audit.** They are *pending re-derivation* per the
>   attack roadmap's Pass B2 (parhelion route), Pass A3 (CZA
>   route), and Pass C3 (tangent-arc route). Read the closeout as
>   the state of the verdict *before* the audit; the verdict
>   language is hedge-required until the re-audit gate clears
>   (attack roadmap §5).
> - **CZA-route residual table entries below are formula-bug
>   contaminated.** The atlas hardcodes CZA apex as `sun_y − R46`
>   at `scripts/overlay_calibrate.py:381–384`, geometrically
>   correct only at h ≈ 22°. Pass A1a (formula spec + literature
>   regression test) and Pass A1b (atlas patch) are the next
>   technical work; Pass A2 (p27 re-classification as supralateral /
>   46° halo top, not CZA) and Pass A3 (CZA-route re-verdict)
>   follow.
> - **Tangent-arc detection-degenerate verdict below applies the
>   wrong primitive at p7.** p7 (h = 59.4°) is in the
>   circumscribed-halo regime per atoptics.co.uk and dewbow.co.uk,
>   not the upper-tangent regime. Pass C1 drops p7 from
>   tangent-arc eligibility; Pass C2 (optional) builds a
>   wing-based / Lab b\* detector before the route is called dead.
>
> Do not export the closeout language below to public-framing
> surfaces (gravity ledger, mesa crossover note, homepage elevator
> pitch) without the post-audit hedges already landed there per
> attack roadmap §6.

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
| p22 | `(668, 453)` | 505 | 508 / 507 | ~5.7 deg | **Phase 10 belt-y FF anchor.** Strong bilateral low-sun parhelia; observed belt y `428` lands on the current `-0.05 * R22` rule. Anchor file: `p22-anchor.json`. |
| p26 | `(464, 204)` | 323 | 332 / 322 | ~9.0 deg | **Phase 10 belt-y FF anchor.** Observed belt y `177` sits ~11 px above the current rule; left/right parhelion y differ by 12 px, so this also carries a parhelic-tilt flag. Anchor file: `p26-anchor.json`. |
| p30 | `(701, 934)` | 650 | 666 / 659 | ~11.1 deg | **Phase 10 belt-y FF anchor.** Rich low-sun halo reference; observed belt y `899` is ~3 px above the current rule. Side parhelia are edge-supported and vertically broad, so keep the tilt/edge flag attached. Anchor file: `p30-anchor.json`. |
| p25 | `(489, 186)` | 300 | 305 / 307 | ~11.4 deg | **Phase 10 belt-y robustness anchor.** Observed belt y `176` is about 5 px below the predicted line in image coordinates (script residual -5 px; FF residual +5 px); right-side reference is cleaner than the foreground/flare-contaminated left, so keep the single-side confidence caveat attached. Anchor file: `p25-anchor.json`. |

All generated overlays now apply `--parhelic-y-offset-r22 = -0.05`, raising
the parhelic belt and dagger markers by 5% of the observed 22° halo radius.
This fixes the shared vertical belt bias without changing the 22°/46° halo
registration. Track that belt-height residual separately from the
parhelion-offset inversion route: p13's committed anchor clears the x-offset
route, but its JSON `parhelion.y = 351` left a single-photo belt-y
watch-list residual against the current `-0.05 * R22` overlay rule
(+10.4 px in the script's predicted-minus-observed convention; -10.4 px
in the FF observed-minus-predicted convention).

## Phase 10-FF Belt-Y Verdict

Result note: [`PHASE10_BELT_Y_RESULTS.md`](PHASE10_BELT_Y_RESULTS.md)

The belt-y watch-list flag is retired. The Phase 10-FF replication set
(p27, p22, p13, p26, p30, p25) falsified FF1: three photos reached
`|residual| >= 5 px`, but the signs disagreed and Spearman
`rho(h, residual)` was +0.086. FF2 was not gated. FF3 passed: no low-h
anchor reached the 12 px primitive threshold. Do not promote a low-h
parhelic-belt correction from p13 alone.

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

> **Pass B1 update, 2026-05-13:** the parhelion-route row below is the
> rolled-up summary; the per-photo eligibility breakdown — including
> the new `sec(h) − 1` (geometric lever), `R22-source`, and
> `geometric_validity` columns required by the audit memo §2 items 4–7
> — lives in the **Parhelion-Route Per-Photo Eligibility** sub-table
> immediately below this table. The summary row's "promoted" status
> is *pending re-derivation in Pass B2* against the now-honest
> eligibility set; do not read the rolled-up row as the audit-survived
> verdict on its own.

| route | p2 residual | p7 / p13 residual | expansion residuals | promotion status | note |
| --- | ---: | ---: | ---: | --- | --- |
| Parhelion offset → sun altitude | −1 / −1 px (0.5% R₂₂) | p7: 0 / n/a px; p13: 0 / 0 px (new anchor; x-offset only) | p20 / p27 are provisional low-sun supports, not promotion measurements | **promoted *(pending B2 re-derivation; see eligibility sub-table)*** | Task #52 step 1 verdict: probation was anchor-driven, not route-driven. Re-anchoring p13 drops x-offset residual from +6/+8 px to ~0/0 px on the new (sun=543,372; R₂₂=211; offsets=213/212) anchor. Route stays in calibrated core; verdict reads as 'bad anchor, route OK'. p13 belt-y residual is tracked separately. **Pass B1 caveat 2026-05-13:** the rolled-up "0 / 0 px on every eligible photo" reading does not survive the audit memo §2 items 4–7. The eligibility set this verdict was derived against included photos where R22 was parhelion-derived (p20, p25, p26 — tautological) or where the geometric lever was below 2 % of R22 (p13, p20, p22, p25, p26, p27, p30 — anchor-noise-bounded). See Parhelion-Route Per-Photo Eligibility below. Pass B2 re-derives this row against the eligible-only subset. |

### Pass A1a Spec Results — CZA Literature Formula vs. Legacy Hardcode

Pass A1a deliverable from
[`../PHASE10_ATTACK_ROADMAP.md`](../PHASE10_ATTACK_ROADMAP.md), landed
2026-05-13. Module at `scripts/cza_formula.py`; regression test at
`scripts/test_cza_formula.py`. Reproduce with `python3
scripts/test_cza_formula.py` from the repo root.

**Verified literature formula:**

```
CZA_above_sun_deg(h) = arcsin(sqrt(n^2 - cos^2 h)) - h     for h <= 32.196 deg
                     = (CZA disappears)                     for h >  32.196 deg
where n = 1.31 (refractive index of ice for the CZA wavelength).
```

**Memo formula correction.** The audit memo §2 item 2 *states* the
formula as `90 - h - arcsin(sqrt(n^2 - cos^2 h))`, but that expression
gives 0.27° at h = 22° (memo claims ~46°) and 31.69° at h = 0.5° (memo
claims ~57°). The expression that matches the memo's *own numerical
predictions* is `arcsin(sqrt(n^2 - cos^2 h)) - h`. Pass A1a's
verify-gate caught the memo's stated formula was a transcription error
before it leaked into the atlas patch (§7 of the attack roadmap rule).
The qualitative finding (formula bug exists, p27 visible "CZA" is
mis-identified) survives unchanged; only the *expression* used to land
A1b is corrected. This is the second time in this campaign the verify
gate has caught a load-bearing error in upstream prose (the first was
the Persona 3 p22 over-statement caught by the synthetic memo's own
verification gate).

**Sanity check at h = 22° (the legacy operating point):**

| quantity | value |
| --- | --- |
| literature CZA-above-sun (deg) | 45.734° |
| legacy CZA-above-sun, from `WB_R46 / (WB_R22 / 22)` | 44.000° |
| delta | +1.734° |

The legacy `WB_R46 = 440` encodes 44° (because WB_R22 = 220 px at
10 px/deg), not 46°. The audit memo §2 item 1 flags this as the
second compounding bug (`WB_R46` in pixels is ~9.1 % too small). Even
at h = 22° the literature formula and the legacy disagree by ~17 px
in workbench coords; below h = 22° the gap widens fast.

**Per-photo regression — observed apex y minus predicted apex y:**

| photo | h (°) | r22 (px) | sun_y | observed apex y | literature predicted y | literature residual | legacy predicted y | legacy residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p2  | 18.6 | 182 | 496 | 113 | 114.3 | **−1.3 px** | 132.0 | −19.0 px |
| p27 | 0.5  | 219 | 559 | 142 | **−11.5** (off-frame) | +153.5 px (meaningless — visible feature is not CZA) | 121.0 | +21.0 px |

**Qualitative-direction verdict:**

- **p2:** literature residual (−1.3 px) is closer to zero than legacy
  (−19.0 px) by 17.7 px. **CONFIRMED** the audit memo's prediction
  ("residual collapses to <5 px"; memo also predicts "~0.7 px" but
  flags it as illustrative). Memo §2 items 1–2 stand.
- **p27:** literature CZA apex y = −11.5 sits above the top of the
  frame (off-screen). The visible chromatic arc anchored at y = 142
  is therefore not the CZA at all. **CONFIRMED** the audit memo's
  prediction (memo §2 item 3: "the chromatic arc visible at y = 142
  in p27 sits at ~42° above the sun and is the 46° halo top /
  supralateral merger"). Pass A2 will re-anchor it.

**A1b disposition:** **CLEARED to proceed** with the verified literature
formula (`arcsin(sqrt(n^2 - cos^2 h)) - h`), not the memo's stated
formula. A1b touches `scripts/overlay_calibrate.py:381–384` (replace
`anchored = WB_SUN[1] - WB_R46` with a call into
`cza_formula.cza_apex_y_above_sun_px(h, WB_R22)` subtracted from
`WB_SUN[1]`) and `scripts/overlay_calibrate.py:66` (replace
`WB_R46 = 440` with `WB_R46 = round(2.091 * WB_R22)` ≈ 460). Per the
attack roadmap A1b touch block, the per-overlay `R46_px = r22_obs ×
(46/22)` derivation alternative remains explicitly out-of-scope
(would touch supralateral and other workbench-space consumers).

### Pass A1b Smoke Results — Atlas Formula Patch

Pass A1b deliverable from
[`../PHASE10_ATTACK_ROADMAP.md`](../PHASE10_ATTACK_ROADMAP.md), landed
2026-05-13. Two surgical edits to `scripts/overlay_calibrate.py` (line
66: `WB_R46` constant; lines 381–384: `cza_apex` inner function), plus
the new `cza_formula` import.

Reproduce the p2 smoke test from repo root with:

```bash
python3 scripts/overlay_calibrate.py docs/calibration/2.Photometeor-jeff_mod_red.jpg \
    --anchors docs/calibration/p2-anchor.json \
    --sun-altitude 18.6 --supralateral 0.40 --out /tmp/p2_overlay_a1b.png
```

**p2 smoke (h = 18.6°):**

| quantity | value |
| --- | --- |
| sun pixel | (567, 496) |
| R22 observed | 182 px |
| **R46 predicted** | **380.5 px** *(was 364.0 px before A1b; WB_R46 went 440 → 460)* |
| **CZA apex predicted** | **(567.0, 114.3)** *(was (560.0, 132.0) before A1b)* |
| CZA apex observed | (560, 113) |
| **CZA apex residual (predicted − observed)** | **(+7.0, +1.3) px** *(was (+0.0, −19.3) under the legacy comparison; sign-convention swap from A1a noted)* |
| left/right dagger residual | −0.0 / +0.0 px (unchanged; parhelion route not touched by A1b) |

The y-residual collapsed from −19.3 px (legacy, observed-minus-predicted
convention from p2-anchor.json `cza_apex_residual.y_px`) to +1.3 px
(literature, predicted-minus-observed convention from
overlay_calibrate's residual report). The magnitudes match A1a's
measured target ("literature residual = −1.3 px" under the
observed-minus-predicted convention).

**p7 smoke (h = 59.4°, beyond CZA disappearance):**

The CZA disappears geometrically at h ≈ 32.2° (per
`scripts/cza_formula.py`). The patched `cza_apex` falls back to the
corrected `WB_R46 = 460` anchor when the literature formula returns
`None`. Smoke run on p7 (sun = (1033, 946), R22 = 200, h = 59.4°)
renders without crash; CZA apex predicted at (1033.0, 527.8)
= sun_y − (WB_R46 × scale) = 946 − 460 × 0.9091 = 946 − 418.2.
Downstream visibility classification still marks CZA as "not
applicable" at this altitude; the fallback is only there to keep the
curve renderable rather than crash the script.

**Supralateral spot-check on p2 (roadmap exit criterion):**

The supralateral apex base is `WB_SUN[1] - WB_R46` (line 412), so the
WB_R46 = 440 → 460 change shifts it from 44° to 46° above sun
(+2°, or +4.55% relative). In p2 photo coords (scale 0.8273) this is
a 16.5 px upward shift of the supralateral apex. The roadmap exit
criterion expected "~4% outward from sun-above; if much more, the
workbench-space change has a knock-on effect that needs a follow-up
pass." Observed 4.55% is within tolerance — **no follow-up pass
needed.**

**Knock-on: 46° halo radius.** The 46° halo is drawn at
`WB_R46 * scale` (line 344). Before A1b: 440 * scale; after A1b:
460 * scale. On p2 this is 380.5 px instead of 364.0 px — the 46° halo
is now drawn ~4.5% angularly larger, which is the *correct* direction
(literature R46 = 2.091 × R22, not 2 × R22).

**Out of scope (deferred to Phase 10 backlog):** the per-overlay
`R46_px = r22_obs × (46/22)` derivation alternative. That option would
remove `WB_R46` from the workbench-space load-bearing path entirely
and let each photo derive its own R46 from its measured R22, but it
would require touching every workbench-to-photo consumer of `WB_R46`
(46° halo, supralateral, and any future primitive that uses the
constant). Filed; not blocking Pass A2/A3 or the re-audit gate.

### Parhelion-Route Per-Photo Eligibility

Pass B1 schema rollout, 2026-05-13. Reads
[`r22_source`](#) and [`geometric_validity`](#) from each photo's
anchor JSON; computes `sec(h) − 1` from the inferred altitude.
*Eligible* means **R22-source = ring-fit AND geometric_validity = valid
AND sec(h) − 1 ≥ 2 %**. The 2 % lever threshold comes from the audit
memo §2 item 4 — below it, residuals are at or below typical anchor
noise. Photos with ring-fit R22 but lever &lt; 2 % are *informationally*
useful but cannot route-validate the parhelion verdict on their own.

| photo | h (°) | sec(h) − 1 | R22-source | geom validity | parhelion residual | eligibility |
| --- | ---: | ---: | --- | --- | ---: | --- |
| p2  | 18.6 | **5.52 %** | ring-fit | valid | −1 / −1 px | **eligible** |
| p7  | 59.4 | **96.5 %** | ring-fit (anchor table; no JSON) | valid | 0 / n/a px | **eligible** |
| p13 | 6.83 | 0.71 % | ring-fit | valid | 0 / 0 px | low-lever caveat (anchor-noise-bounded; informational) |
| p20 | ~5.0 | 0.38 % | parhelion-derived | valid | 0 / 0 px | **ineligible** (R22 from parhelion; tautological) |
| p22 | ~5.7 | 0.50 % | ring-fit | valid | 0 / 0 px (presumed) | low-lever caveat (anchor-noise-bounded; informational) |
| p25 | ~11.4 | 2.01 % | parhelion-derived | valid | 0 / 0 px (presumed) | **ineligible** (R22 from parhelion; tautological) |
| p26 | ~9.0 | 1.25 % | parhelion-derived | **invalid (right)** | left: 0 px (presumed); right: undefined (`R22 / offset = 1.003 > 1`) | **ineligible** (R22 from parhelion AND right-side geometric impossibility per memo §2 item 6) |
| p27 | ~0.5 | ≈ 0.00 % | ring-fit (parhelion-derived in opposite direction) | valid | 0 / 0 px (tautological) | **ineligible** (parhelion offsets stipulated as `offset := R22`; memo §2 item 5: "the 0 px residual is tautological") |
| p30 | ~11.1 | 1.90 % | ring-fit | valid | 0 / 0 px (presumed) | low-lever caveat (anchor-noise-bounded; informational; just below 2 %) |

**Pass B1 conclusion against the audit memo's three-photo finding:**
the audit memo §2 item 4 + §4 item 1 says parhelion-offset passes "on
three photos (p2, p7, p13) where the test has meaningful discrimination
and an independently fittable 22° halo." This sub-table's *eligible*
column matches that finding for p2 and p7, and lists p13 as
*low-lever-caveat* rather than fully eligible because its lever is
0.71 % &lt; 2 %. The audit memo's looser 3-photo set hinges on
"unambiguous bilateral peaks AND independently fittable 22° halo"
without a hard lever threshold; this sub-table is the stricter
schema-level reading. **Pass B2 will rule on whether the parhelion
verdict survives on the strict (2-photo) eligible set, the audit's
permissive (3-photo) set, or somewhere in between with the low-lever
photos contributing as informational evidence.** Either way, the
"every eligible photo" framing is gone; the audit-survived
language (memo §6) is "passes residual gate at ~0 px on photos where
sun altitude provides a non-trivial geometric lever and an
independent 22° halo ring is fittable."

`r22_source` is now mirrored in each anchor JSON's top-level field of
the same name plus a `r22_source_note` rationale. p26's anchor adds a
`geometric_validity` block with per-side flags and the audit memo
citation.
| CZA apex → sun altitude / visibility | x: −6.7 px, **y: −19.3 px (−10.6% R₂₂)** | p7 not applicable (h = 59.4° > 32.2° cutoff); p13 cropped (predicted apex y = −50) | p27: x +3 px, **y +21 px (+9.6% R₂₂)**; p19 / p20 cropped | **fails residual gate on expanded set** | Task #55 corrects the earlier interpretation: p2 and p27 both exceed the 8 px / 0.04*R22 route threshold, but their signs are opposite. This is not a stable direction-and-magnitude atlas bias. It is a clean negative for CZA-apex inversion promotion and a weaker Mesa #3 receipt: parhelion-offset remains ~0-1 px while CZA-apex is photo-specific and unreliable. Anchors: `p2-anchor.json`, `p27-anchor.json`, `p20-anchor.json`. |
| Tangent-arc curvature → sun altitude | **detection-degenerate** (fuses with halo top at h=18.6°) | p7 (h=59.4°): column-peak grabs halo outer edge not broad tangent; p13 (h=6.83°): chromatic-haze contamination, no clean smile to fit | p27 (h=0.5°): merges with CZA / sun bloom at horizon transition; p20 not measured | **fails detection across all 4 eligible photos** (v3.8 partition: detection-degenerate, not residual-bounded) | Task #54 first pass 2026-05-13 found the route fails column-peak detection on every photo in the calibration set, with a *different* failure mode at each altitude. Not a residual-gate failure — the residual was never measurable. The earlier 'altitude-regime validity window' framing was too generous; the route's degeneracy is photo-and-feature-specific across the entire altitude range. Promotion blocked until a different detection method (edge-based gradient tracking, template matching, or manual sampling) is built and verified. |
| Supralateral position → sun altitude | _measurement pending_ | p7 cropped; p13 not visible | p27 candidate; p20 weak/cropped | pending expansion | p27 may reopen supralateral coverage, but no residual has been measured. Keep the previous fail verdict for the original p2/p7/p13 set; do not promote the expanded route without p27 anchors. |
| Parhelic-belt y residual (primitive, not route) | not separately measured (Phase 2 set only had x-residuals) | p7 not applicable; p13 -10.4 px under the FF observed-minus-predicted convention | p27 -0.0 px; p22 +0.2 px; p26 -10.8 px; p30 -2.5 px; p25 +5.0 px (caveated) | **watch-list retired 2026-05-13** (FF1 falsified; FF3 passed) | Phase 10-FF tested the shared `--parhelic-y-offset-r22 = -0.05` rule across six low-h anchors. FF1 falsified: the >=5 px residuals do not share a sign (p13 / p26 negative, p25 positive) and Spearman rho is +0.086. FF3 passed: no photo breaches the 12 px primitive threshold. Record p13 as photo-specific / anchor-local, not a promoted low-h correction. See `PHASE10_BELT_Y_RESULTS.md`. |

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
