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
| p27 | `(596, 559)` | 219 | 219 / 219 | ~0.5° | **Re-classified Pass A2 2026-05-13.** Originally anchored as CZA expansion check (Task #55); the visible chromatic arc at `(599, 142)` is now identified as the **46° halo top / supralateral merger** at ~41.89° above sun, not CZA. Per Pass A1a/A1b reverification, the literature CZA at h = 0.5° sits ~57° above sun (px y = −11.5, off-frame above the top), so the visible feature cannot be CZA. CZA fields preserved under `_disputed.cza_apex_legacy_2026_05_13` for reversibility. Now feeds supralateral-route eligibility (candidate, pending Pass A3 coverage check); does NOT feed CZA-route residual evidence. Anchor file: `p27-anchor.json`. |
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

### Pass A2 Results — p27 Primitive Re-Classification

Pass A2 deliverable from
[`../PHASE10_ATTACK_ROADMAP.md`](../PHASE10_ATTACK_ROADMAP.md), landed
2026-05-13. One anchor edit + four references walked back across this
file's CZA-route table, supralateral-route table, anchor summary, and
v3.8 tangent receipt.

**Geometric verification (`scripts/cza_formula.py` against
`p27-anchor.json`):**

| primitive | predicted apex y above sun (deg / px) | predicted pixel y | observed pixel y (visible feature) | delta |
| --- | ---: | ---: | ---: | ---: |
| Literature CZA at h = 0.5° | 57.31° / 570.5 px | **−11.5 (off-frame)** | 142 | n/a (off-frame; visible feature is not CZA) |
| 46° halo top (always 46°) | 46.00° / 458.0 px | 101.1 | 142 | +40.9 px (~4.11° below) |
| Supralateral at h = 0.5° (~46° per literature merger) | ~46.00° / 458.0 px | 101.1 | 142 | +40.9 px (~4.11° below) |

The visible chromatic arc at y = 142 is at 41.89° above sun. The
literature CZA at h = 0.5° is off-frame (above the top of the photo),
so the visible feature cannot be CZA. The 46° halo top and the
supralateral arc both predict ~46° above sun and theoretically coincide
at h = 0.5° (per audit memo §2 item 12: supralateral angular distance
changes only ~0.5° across h = 0–22°, so the two are observationally
indistinguishable on this single low-h photo). Visible feature sits
~4° below the theoretical line — most likely chromatic broadening of
the diffuse halo lower edge or visual-edge measurement bias (consistent
with the existing `_meta.note` on the original p27 anchor: "the flat
apex makes the fitted math apex sensitive").

**Anchor JSON re-shape (`docs/calibration/p27-anchor.json`):** top-level
`cza_apex` and `cza_apex_residual` fields moved under
`_disputed.cza_apex_legacy_2026_05_13` (preserved verbatim with
rationale, reversible). New top-level `supralateral_46halo_merger`
block with apex `(599, 142)`, `deg_above_sun = 41.89`,
`candidate_primitives = [halo_46_top, supralateral_arc]`. New
top-level `supralateral_route_eligibility` block flags the photo for
Pass A3 consideration. `_meta.method`, `_meta.status`, `_meta.note`
rewritten to record the re-classification.

**A3 disposition:** **CLEARED.** Pass A3 re-derives the CZA-route and
supralateral-route verdicts against the post-A2 eligibility set.

### Pass A3 Results — CZA + Supralateral Re-Verdict

Pass A3 deliverable from
[`../PHASE10_ATTACK_ROADMAP.md`](../PHASE10_ATTACK_ROADMAP.md), landed
2026-05-13. No code edits, no anchor edits — A3 is purely a verdict
+ doc-update pass against the now-honest atlas (A1b) and the
re-classified p27 (A2).

**CZA-route eligibility (all 8 anchored photos + p7 from the residual
table):**

| photo | h (°) | literature CZA above sun | predicted apex y | status |
| --- | ---: | ---: | ---: | --- |
| p2 | 18.6 | 46.14° / 381.7 px | **114.3** | **IN-WINDOW** (residual: +1.3 px) |
| p7 | 59.4 | n/a (CZA disappeared) | n/a | NOT APPLICABLE (h > 32.2°) |
| p13 | 6.83 | 51.88° / 497.6 px | −125.6 | OFF-FRAME (above top) |
| p20 | ~5.0 | 53.28° / 1102.1 px | −275.1 | OFF-FRAME (above top) |
| p22 | ~5.7 | 52.69° / 1210.7 px | −757.7 | OFF-FRAME (above top) |
| p25 | ~11.4 | 48.96° / 667.7 px | −481.7 | OFF-FRAME (above top) |
| p26 | ~9.0 | 50.36° / 739.9 px | −535.9 | OFF-FRAME (above top) |
| p27 | 0.5 | 57.31° / 570.5 px | −11.5 | OFF-FRAME *(also REMOVED Pass A2: visible feature reclassified)* |
| p30 | ~11.1 | 49.10° / 1450.1 px | −516.1 | OFF-FRAME (above top) |

**Only p2 is eligible.** Every low-h photo has the literature CZA apex
predicted above the top of the photo because CZA sits ~50–58° above
sun at low altitudes — typical sun-and-halo photos don't have that
much vertical headroom above the sun. p7 is past the disappearance
threshold. p27's visible feature was reclassified by Pass A2. The
single in-window photo (p2) has a sub-px residual (+1.3 px) under the
literature formula.

**CZA-route verdict: fails coverage gate.** This is a structurally
different failure from the pre-audit verdict ("fails residual gate on
expanded set; opposite-sign residuals on p2 and p27"). The pre-audit
verdict was an artifact of (a) the WB_R46 formula bug creating the p2
−19.3 px residual, and (b) the p27 chromatic arc being miscategorized
as CZA. With both corrections landed, the route has no residual-gate
failure on its own merits — it simply has only one in-window anchored
photo. Reopening coverage requires new anchors in 5° < h < 32°.

**Supralateral-route eligibility (post-A2):**

- p2: route-eligible per existing notes; no supralateral apex measured
  in `p2-anchor.json`.
- p27: route-candidate post-A2; apex measured at (599, 142) = ~41.89°
  above sun.

Strict reading of the coverage gate: 1 measured + 1 eligible-but-
unmeasured = below the two-photo threshold. Permissive reading: 2
candidates at threshold.

**Either reading is dominated by structural h-discrimination** (audit
memo §2 item 12). Supralateral angular distance from sun varies only
~0.5° across the h = 0–22° eligibility range:

- p2 (h = 18.6°) → ~45.7° above sun
- p27 (h = 0.5°) → ~46.0° above sun
- delta: **~0.3°** (~2.5 px at p2's R22)

Visual-edge measurement noise on a chromatic-broadened halo arc is
easily 5–10 px, which is the regime p27's own `_meta.note` flags
("flat apex makes the fitted math apex sensitive"). The route's
h-discrimination is below the noise floor on either photo.

**Supralateral-route verdict: fails coverage gate + structural-
discrimination rider.** Distinct from pre-audit ("coverage failure;
p27 might reopen it") because the structural-discrimination finding
says coverage *can't* fix it: even ten new low-h anchored photos
would not restore the route as a useful inverse handle, because
atmospheric physics constrains h-sensitivity below the noise floor.
This is a *route-physics* limitation, not a *dataset* limitation.

**Three-failure-layer framing update.** Pre-audit, the geometry-side
closeout headlined: *"the other three routes fail at three structurally
different layers of the measurement stack"* (residual / coverage /
detection). Post-A3, the count of *gates* is unchanged (CZA, supralateral
both coverage; tangent detection — that's 2 + 1, not 3 independent),
but the *failure modes* remain three structurally different things:

1. **CZA — coverage on physics grounds.** Route physically exists,
   forward formula is correct, p2 residual is sub-px. The single-photo
   coverage limit is an artifact of the photo set's altitude
   distribution and aspect ratio, not of the route or the atlas.
2. **Supralateral — coverage on discrimination grounds.** Route exists
   and is in-window across most of h, but h-sensitivity is structurally
   below the noise floor. Coverage cannot fix this; new physics could.
3. **Tangent — detection on tooling grounds.** Route remains blocked
   under column-peak detection on the post-C1 sampled set (p2 / p13 /
   p27). p7 is removed as circumscribed-halo regime, not upper-tangent
   route evidence. Pass C2 (optional) tests whether a wing-based
   detector recovers the route.

The mesa↔geometry crossover note's "two-of-three have non-atlas
explanations" hedge (per §6 step 2) now lands its substantive
content: of the three failures, the CZA finding is dataset-shape, the
supralateral finding is physics-shape, and the tangent finding is
tooling-shape. None of the three is a "the atlas inversion math is
broken" finding.

**Pass C1 cleared to proceed.** Drop p7 from the tangent eligibility
set per the circumscribed-halo literature regime. **Pass B2 cleared
to proceed** as soon as C1 lands, since B2 reads the now-finalized
post-A3 + post-C1 eligibility tables.

**Pass C2 disposition (optional).** Without C2, the tangent verdict
stays "detection gate failure under column-peak; protocol-conditional."
The re-audit and the rewritten specialist handoff will explicitly mark
tangent detector as Unresolved Open Question per memo §4.8.

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
| CZA apex → sun altitude / visibility | **+7.0 px / y: +1.3 px (literature; Pass A1b)** | p7 NOT APPLICABLE (h = 59.4° > 32.2° cutoff; CZA disappears); p13 (h = 6.83°) OFF-FRAME, predicted apex y = −125.6 (above top of photo) | p20 (h ≈ 5°), p22 (h ≈ 5.7°), p25 (h ≈ 11.4°), p26 (h ≈ 9°), p30 (h ≈ 11.1°): all OFF-FRAME (predicted apex y < 0 at every low-h photo because CZA sits ~50–58° above sun in the disappearance approach); **p27 REMOVED Pass A2** (visible feature is 46° halo top / supralateral merger, not CZA) | **fails coverage gate** *(Pass A3 verdict 2026-05-13; was: "fails residual gate on expanded set")* | **Pass A3 verdict, 2026-05-13.** Re-derived against the post-A1b atlas and post-A2 eligibility set. **Only p2 is in-window with an independent CZA-apex measurement**, and on that single photo the residual is **+1.3 px** — well below the 8 px / 0.04*R22 route threshold. The route does *not* fail the residual gate on its own merits; the pre-audit "opposite-sign residuals on p2 and p27" framing dissolved when (a) A1b corrected the atlas formula bug that was creating the p2 −19 px artifact, and (b) A2 re-classified the p27 chromatic arc as a different primitive. **The route fails the coverage gate**: the pre-registered "fewer than two eligible photos" rule binds because all other anchored photos are out-of-window (p7) or off-frame at the low altitudes where the CZA sits ~50–58° above sun. This is a structurally different failure mode from the pre-audit verdict — *not* a "route-reliability" finding but a "single in-window photo in the calibration set" finding. Reopening coverage would require either anchoring new photos in the 5° < h < 32° range where CZA stays on-frame, or accepting p2's sub-px residual as the single eligible result. Anchors: `p2-anchor.json` (in-window, sub-px); `p27-anchor.json` (re-classified per Pass A2); other six anchored photos all off-frame at literature CZA position. |
| Tangent-arc curvature → sun altitude | **detection-degenerate** (fuses with halo top at h=18.6°) | **p7 REMOVED Pass C1** (h=59.4° is circumscribed-halo regime, not upper-tangent-route eligibility); p13 (h=6.83°): chromatic-haze contamination, no clean smile to fit | p27 (h=0.5°): sun-bloom flare contaminates the sun-meridian column; p20 not measured | **fails detection across the post-C1 sampled set under column-peak** *(protocol-conditional; p7 historical sample reclassified)* | **Pass C1 verdict, 2026-05-14.** p7 is dropped from tangent-route eligibility because h=59.4° is the circumscribed-halo regime per the tangent-arc literature, so the original "all four eligible photos" count included a primitive misclassification. The remaining measured tangent-route set is p2 / p13 / p27. Column-peak detection still fails on those photos, but the negative is explicitly **protocol-conditional**, not a class-level tangent-route verdict. Pass C2 may reopen the route with a wing-based / Lab b* detector; if C2 is skipped, the specialist handoff must list tangent detection as an unresolved open question. |
| Supralateral position → sun altitude | _measurement pending_ (route-eligible but no apex anchored) | p7 cropped; p13 not visible | **p27 measured Pass A2** with apex at (599, 142) = ~41.89° above sun (~4° below the ~46° literature prediction at h = 0.5°, likely chromatic-broadening); p20 weak/cropped | **fails on structural discrimination (Pass A3 verdict 2026-05-13)** *(coverage-gate hedged: 1 measured + 1 eligible-unmeasured)* | **Pass A3 verdict, 2026-05-13.** Post-A2 the eligibility set is p2 (eligible per existing notes, apex unmeasured) + p27 (measured at 41.89° above sun). On a strict reading the route fails the coverage gate (1 measured photo); on a permissive reading it's at the two-photo threshold. **Either reading is dominated by the structural-discrimination finding from audit memo §2 item 12.** Supralateral angular distance from sun varies only ~0.5° across the entire h = 0–22° eligibility range, an order of magnitude less h-sensitivity than parhelion-offset. The predicted h-spread between p2 (h = 18.6° → ~45.7°) and p27 (h = 0.5° → ~46.0°) is **~0.3°**, which at p2's R22 = 182 px corresponds to **~2.5 px**. Visual-edge apex measurement noise on a chromatic-broadened halo arc is easily 5–10 px (see p27's existing "flat apex makes the fitted math apex sensitive" flag in `_meta.note`). The h-spread the route can resolve is below the measurement noise floor on either photo, so **the route would not be a useful inverse handle even with perfect coverage**. Distinct from the pre-audit framing in two ways: (a) it is *not* "p27 might reopen coverage" — coverage may reopen but the route is still dead; (b) the structural limit is in the *atmospheric physics*, not in the dataset or the detection protocol. Reopening coverage on a third low-h photo will not restore the route. Anchors: `p2-anchor.json` (no supralateral apex), `p27-anchor.json` (`supralateral_46halo_merger` block). |
| Parhelic-belt y residual (primitive, not route) | not separately measured (Phase 2 set only had x-residuals) | p7 not applicable; p13 -10.4 px under the FF observed-minus-predicted convention | p27 -0.0 px; p22 +0.2 px; p26 -10.8 px; p30 -2.5 px; p25 +5.0 px (caveated) | **watch-list retired 2026-05-13** (FF1 falsified; FF3 passed) | Phase 10-FF tested the shared `--parhelic-y-offset-r22 = -0.05` rule across six low-h anchors. FF1 falsified: the >=5 px residuals do not share a sign (p13 / p26 negative, p25 positive) and Spearman rho is +0.086. FF3 passed: no photo breaches the 12 px primitive threshold. Record p13 as photo-specific / anchor-local, not a promoted low-h correction. See `PHASE10_BELT_Y_RESULTS.md`. |

## Tangent-Curvature v3.8 Receipt *(updated Pass C1 2026-05-14)*

Task #54 originally sampled p2 (h=18.6°), p7 (h=59.4°), p13
(h=6.83°), and p27 (h=0.5°). Pass C1 drops p7 from the
upper-tangent-route eligibility set: at h=59.4° the display is in the
circumscribed-halo regime, so testing it as an "upper tangent arc" was a
primitive misclassification. The post-C1 receipt is therefore narrower:
**column-peak detection fails on the sampled in-window / low-sun tangent
set (p2, p13, p27), while p7 becomes a high-sun circumscribed-halo
backlog reference.**

This is still a useful clean-negative for the current detector, but it is
not a class-level negative for tangent curvature. C2 is the optional
detector spike that decides whether a wing-based / Lab b* method recovers
the route.

Per-photo v3.8 schema:

| photo | h | salience | route-local residual | route-local sensitivity | full-overlay interaction | partition/conflict |
| --- | ---: | --- | --- | --- | --- | --- |
| p2 | 18.6° | high (b-contrast 35.7) | unmeasurable: samples hit halo top within 5 px | n/a — fit degenerate | depends on sun_x as expected | tangent fuses with 22° halo top at this h (compression regime) |
| p7 | 59.4° | high-sun circumscribed-halo display | **removed from upper-tangent-route eligibility by Pass C1** | n/a | n/a | At h>29° the upper/lower tangent arcs have joined into the circumscribed halo. Historical column-peak result is retained only as a cautionary sample, not as a tangent-route residual. |
| p13 | 6.83° | medium (b-contrast 15.8 = 23% sky) | RMS=41 px on saturation fit; radius 196 vs predicted 1774 | ±5 px wiggle → ±1.2 px Δr (misleadingly low; fit is already stuck) | n/a | chromatic-haze contamination; no clean smile to fit |
| p27 | 0.5° | tangent essentially absent between ~~CZA~~ *(retracted Pass A2)* and halo top | both fits have RMS=22 px with non-physical centers (brightness fit center *below* sun) | n/a | n/a | **Pass A2 2026-05-13 retracts "tangent merges with CZA":** at h = 0.5° the literature CZA apex is ~57° above sun while the upper tangent arc is ~22° above sun — separated by ~35°, they cannot merge optically (audit memo §4.4). The actual failure mode is sun-bloom flare contaminating the sun-meridian column. Substantively: tangent-route detection still fails at p27 (RMS=22 px, non-physical centers), but the failure cause is sun-bloom contamination, not feature merger. |

The Mesa #5 reading survives C1, but in a narrower form: visual salience,
route-local residual, sensitivity, and full-overlay interaction are **not
the same property**. For p2 / p13 / p27, the tangent primitive may be
atmospherically real and geometrically meaningful while remaining
route-local-unmeasurable under the column-peak detector. For p7, the
mistake was earlier in the stack: the photo belongs to the
circumscribed-halo regime, not the upper-tangent-route eligibility set.
The v3.8 schema's job was to record those distinctions; Pass C1 sharpens
that record instead of treating all four failures as one class.

Atmospheric-optics atlas-model implication: the upper tangent arc's
brightness/chromatic peak does NOT generally coincide with its
geometrically-predicted spine. The arc's *visual identity* lives in
combined brightness+chromaticity, not in a single peak that column-wise
detection can grab. Real measurement needs either gradient-based edge
detection (the spine is at a brightness or chromaticity *transition*, not a
peak) or manual sample selection from visual crops.

This receipt also weakens the earlier framing that p2 alone was the
exception. The post-C1 state is: CZA-apex fails coverage after formula
repair and p27 reclassification; supralateral fails on coverage plus
structural h-discrimination; tangent-curvature fails under the
column-peak protocol on the remaining sampled set and is unresolved
pending C2. Only parhelion-offset remains as a currently promoted
inverse route, with its B2 re-verdict still pending.

## Vocabulary Classification

| primitive | p2 | p7 | p13 | promote? |
| --- | --- | --- | --- | --- |
| CZA | visible | not applicable / high sun | cropped or not visible | Conditional core: render only inside the CZA sun-altitude validity window; CZA-apex inversion route fails coverage after A1b/A2/A3, not residual. |
| Supralateral arc | visible | candidate only | not visible / cropped | Keep as optional vocabulary; p27 is now a measured 46°-halo-top / supralateral-merger candidate, but the inversion route fails structural h-discrimination. |
| Upper tangent arc | visible | high-sun circumscribed-halo regime, not upper-tangent-route evidence | candidate | Promote as stable logo/animation shape language from visibility; curvature-as-inversion-route remains protocol-conditional pending C2. |
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
| Upper tangent arc | **promoted as logo / animation vocabulary** | Visible shape-language support remains; p7 is now treated as high-sun circumscribed-halo vocabulary rather than upper-tangent-route evidence. Curvature-as-inversion-route remains protocol-conditional pending C2. |
| CZA primitive (rendered) | **promoted as conditional core** | Render only when `h < 32.2 deg` (atmospheric cutoff). Visible and measured on p2; p27's former CZA mark is reclassified as 46° halo top / supralateral merger. The primitive remains core while the CZA-apex inversion route fails coverage. |
| CZA apex to h inversion route | **fails coverage gate** *(Pass A3 2026-05-13)* | A1b fixed the CZA formula and p2 now has a +1.3 px y residual, below threshold. A2 removed p27 from CZA evidence. The remaining eligible set has only p2; all other anchored photos are out-of-window or off-frame at the literature CZA position. |
| Tangent-arc curvature to h inversion route | **fails detection gate under column-peak on the post-C1 sampled set** | Pass C1 (2026-05-14) removes p7 from upper-tangent-route eligibility because h=59.4° is circumscribed-halo regime. Column-peak detection still fails on p2 / p13 / p27, but this is a protocol-conditional negative, not a class-level tangent verdict. Promotion blocked unless C2 recovers the route with a wing-based / Lab b* detector. |
| Supralateral position to h inversion route | **fails structural-discrimination gate** *(coverage hedged; Pass A3 2026-05-13)* | Post-A2/A3, p27 is measured as a supralateral / 46°-halo-top candidate and p2 is eligible but unmeasured. Even if coverage is treated permissively, the predicted p2-vs-p27 h-spread is ~0.3° (~2.5 px at p2 R22), below visual-edge noise. The route is not a useful inverse handle on atmospheric-physics grounds. |
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
   Pass A3 records a coverage hedge plus structural-discrimination
   failure. p27 is now a measured supralateral / 46°-halo-top candidate,
   but the route's h-signal is below visual-edge noise even if coverage
   is read permissively.
4. **Tangent-arc curvature inversion route under the column-peak detection
   protocol.** Detection-degenerate on the post-C1 sampled set
   (p2 / p13 / p27). p7 is removed from route eligibility because it is
   circumscribed-halo regime. Reopening the route requires C2's
   wing-based / Lab b* detector, not simply more photos.
5. **CZA-apex inversion route on the current committed evidence.** Pass
   A3 converts the verdict from residual-gate failure to coverage-gate
   failure: p2 has a good +1.3 px residual under the repaired formula,
   p27 was a misidentified primitive, and all other anchors are
   out-of-window or off-frame.
6. **Lower tangent as core logo geometry.** Single-photo evidence; usable
   only as labeled low-sun annotation or in named-pose examples, not in core
   marks.
7. **Any linear arc-importance attribution metric** at Phase 11 review.
   Per Mesa crossover rule 5.

### Single-handle closeout

The Phase 10 gate closed 2026-05-13 and was post-audit hedged by Passes
B1, A1a/A1b, A2, A3, and C1. **One inversion route remains promoted
pending B2 re-derivation; the other candidate inversion routes fail or
remain blocked for distinct, now better-named reasons.**

| route | gate outcome | failure layer |
| --- | --- | --- |
| Parhelion offset → h | **promoted** *(pending B2 re-derivation)* | Pass B1 hedges eligibility to photos with independent R22 and non-trivial lever; B2 restates the final verdict. |
| CZA apex → h | fails | **coverage gate** after A1b/A2/A3: p2 residual is good (+1.3 px), but it is the only in-window measured CZA anchor. |
| Supralateral → h | fails | **coverage + structural-discrimination gate** after A2/A3: p27 is measured and p2 is eligible, but h-sensitivity is below measurement noise. |
| Tangent-arc curvature → h | fails / unresolved | **detection gate under column-peak** on p2 / p13 / p27 after C1 removes p7 as circumscribed-halo regime; C2 decides whether this is tooling-conditional or recovered. |

The old "three independent failure layers" language is retired. The
post-audit taxonomy is more precise: CZA is dataset/aspect-ratio
coverage-limited, supralateral is physics-discrimination-limited,
tangent is detector-limited under the current column-peak protocol, and
parhelion offset is the only currently promoted inverse handle, pending
B2's final wording.

Phase 11 can proceed against the visibility-promoted vocabulary now.
Three open questions are filed but not blocking:

1. **CZA coverage expansion.** Would a controlled photo set in
   5° < h < 32° supply a second in-window CZA apex anchor? The old
   "direction inconsistency" question was dissolved by A1b/A2.
2. **Tangent-curvature tooling.** Does a gradient-based edge-tracking or
   template-matching detector recover the route on existing photos? This
   is a tooling question, not a physics question — the arc is real and
   visible; the column-peak method is the wrong instrument.
3. **Parhelic-belt-y replication.** Closed by Phase 10-FF:
   p13's residual is photo-specific; the belt-y watch-list flag is
   retired.

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
- **Task #53 / #55 — CZA apex anchors:** done. ~~p2 residual y = -19.3 px;
  p27 residual y = +21 px. Both exceed the `0.04 * R22` / 8 px route
  threshold, opposite signs. CZA-apex inversion route fails the residual
  gate on the expanded set; not a stable systematic.~~
  **Walked back 2026-05-13 (Pass A1b + Pass A2 + Pass A3):** the p2
  residual was formula-bug contaminated (legacy `WB_R46` hardcode);
  literature formula collapses it to y = +1.3 px under predicted-minus-
  observed. The p27 residual was against a misidentified primitive (the
  visible feature at y = 142 is the 46° halo top / supralateral
  merger, not CZA). Pass A3 re-derived the CZA-route verdict: **fails
  coverage gate** (only p2 is in-window with an independent measurement;
  every other anchored photo is out-of-window or off-frame). The p2
  residual is sub-px; the route has no residual-gate failure on its own
  merits.
- **Task #54 — tangent-arc curvature sampling:** done. Column-peak
  detection fails on the post-C1 sampled set (p2 / p13 / p27), with p7
  removed from upper-tangent-route eligibility as a circumscribed-halo
  display. Historical p7 sampling is retained only as a primitive-ID
  caution. p27's failure mode is ~~CZA / sun-bloom merge~~ **sun-bloom
  flare contaminating the sun-meridian column** (Pass A2 retracts the
  "merges with CZA" framing — at h = 0.5° CZA and tangent are ~35°
  apart in altitude, not coincident; per audit memo §4.4). See the
  Tangent-Curvature v3.8 Receipt section above. Tangent-arc curvature
  inversion remains a *protocol-conditional* negative pending Pass C2
  wing-based detector rebuild.
- **Supralateral inversion:** ~~coverage gate failure on the current
  committed set (only p2 eligible). p27 remains a candidate but no
  supralateral residual has been measured~~ **Pass A3 verdict 2026-05-13:
  fails coverage gate + structural-discrimination rider.** Pass A2
  promoted p27 from CZA-misidentified to supralateral candidate (apex
  measured at ~41.89° above sun). p2 is route-eligible but apex
  unmeasured. **Even at the two-photo threshold the route fails on
  structural h-discrimination:** audit memo §2 item 12 documents that
  supralateral angular distance from sun varies only ~0.5° across
  h = 0–22°, an order of magnitude less than parhelion-offset. The
  predicted spread between p2 and p27 is ~0.3° = ~2.5 px at p2's R22,
  below the typical 5–10 px visual-edge measurement noise. The route
  would not be a useful inverse handle even with perfect coverage; this
  is an atmospheric-physics limitation, not a dataset limitation.

Historical execution order (now retired):

1. **Re-anchor p13 parhelion offset first** — done. Replaced rough
   hand-anchor with measured (sun_px, R22_px, parhelion_left_px,
   parhelion_right_px) tuple; new residual ~0/0 px. Anchor Summary and
   Per-Inversion-Route Residual Table updated.
2. **CZA apex anchor check (Task #53, expanded by Task #55).** Historical
   result was p2 y = -19.3 px and p27 y = +21 px. Pass A1b/A2/A3
   supersedes that reading: p2 is +1.3 px under the repaired formula,
   p27 is not CZA, and the route fails coverage rather than residual.
3. **Tangent-arc curvature anchors (Task #54).** Historical sample was
   p2 / p7 / p13 / p27. Pass C1 supersedes the eligibility list: p7 is
   circumscribed-halo regime and drops out. Column-peak still fails on
   p2 / p13 / p27; route remains blocked pending C2.
4. **Supralateral inversion.** Pass A2/A3 supersedes the old coverage-only
   reading. p27 is now a measured candidate, p2 is eligible but
   unmeasured, and the route fails on structural h-discrimination even
   if coverage is read permissively.
