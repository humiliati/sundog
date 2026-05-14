# Persona: Parhelion promotion + forward-generation auditor

Top-line verdict (one line each):

- **§A (Parhelion promotion):** **Fit artifact + circular dependency.** At every low-h photo in the calibration set the parhelion-offset → h test reduces to "does the offset equal R22?" — and in most low-h photos `R22` is not measured from an independent saturation ring; it is either inferred from the parhelion position itself or pinned to a band where no 22° halo arc is visible. The "~0 px residual on every eligible photo" verdict is mostly tautological. The route is not falsified, but it is not what the closeout claims it is either.
- **§B (Forward generation):** **Over-claimed for at least three primitive classes** — infralateral arcs, 46° halo radius, and the parhelic circle. The atlas draws these primitives on the overlays whether or not they are visible in the photo, and on the rich-vocabulary overlays for p7 and p13 they are predicted in regions where the photo shows nothing. "h → all primitives cleanly" is true geometrically; "h → all primitives visible at the predicted locus in the calibration photos" is not.

---

## §A. Parhelion promotion under audit

Four-way verdict: **fit-artifact + circular-dependency** (with cherry-picked sub-cases).

### A.1 The arithmetic check the closeout never writes down

The route is `h = arccos(R22 / offset)`. Equivalently, `offset = sec(h) · R22`. For low sun altitudes, `sec(h) ≈ 1`, so the test reduces to: **does the parhelion offset equal R22?**

| photo | h_claimed | R22 | offsets (L/R) | offset/R22 (L/R) | sec(h_claimed) | residual interpretation |
|---|---:|---:|---:|---:|---:|---|
| p2 | 18.6° | 182 | 192 / 192 | 1.0549 / 1.0549 | 1.0552 | mid-sun, real lever, residual ~0 px is real |
| p7 | 59.4° | 200 | 393 / n/a | 1.965 / — | 1.965 | high-sun, real lever, **single-sided** (right-side only) |
| p13 | 6.83° | 211 | 213 / 212 | 1.0095 / 1.0047 | 1.0071 | low-sun; lever ≈ 1.5 px; "residual ~0" lives in the noise floor |
| p20 | ~5° | 455 | 457 / 457 | 1.0044 / 1.0044 | 1.0038 | low-sun; lever ≈ 1.7 px; anchor itself is "provisional" |
| p22 | 5.7° | 505 | 508 / 507 | 1.0059 / 1.0040 | 1.0050 | low-sun; lever ≈ 2.5 px |
| p25 | 11.4° | 300 | 305 / 307 | 1.0167 / 1.0233 | 1.0197 | low-sun; right-side has anchor caveat |
| p26 | 9.0° | 323 | 332 / 322 | 1.0279 / **0.9969** | 1.0125 | **right offset is < R22, which is geometrically impossible for any real h** |
| p27 | 0.5° | 219 | 219 / 219 | 1.0000 / 1.0000 | 1.0000 | **identical to four significant figures — degenerate by construction** |
| p30 | 11.15° | 650 | 666 / 659 | 1.0246 / 1.0138 | 1.0193 | low-sun; L/R disagree by 7 px |

Two of these rows are damning.

- **p27** records `R22 = parhelion_offset_left = parhelion_offset_right = 219`. The anchor JSON itself calls this "low-sun dagger estimate from R22 support; not an independent parhelion-core pick." That is a written admission that the parhelion was placed *because R22 said so*, not measured. The "0 px residual" is what you get when you stipulate `offset := R22`.
- **p26** has right-side `offset = 322 < R22 = 323`. `arccos(R22/offset)` is undefined for that ratio. One of the two anchors is wrong. The fact that this is recorded with no flag in the residual table tells you the table is not actually applying the geometric constraint that supposedly defines the route.

### A.2 Is the R22 anchor independent of the parhelion?

The handoff §2.3 says `R22_px` is captured "via saturation-ring fit (not visual ring guess)." That claim does not survive contact with the photos.

- **p2** (h=18.6°), **p7** (h=59.4°), **p13** (h=6.83°), **p30** (h=11.15°): a 22° halo *is* visible as a coherent saturation ring. R22 from a ring fit is genuinely independent of parhelion pick.
- **p20**, **p22**, **p25**, **p26**: **no continuous 22° halo arc is visible in the image at all.** What is visible is two bright parhelia and (sometimes) a faint suggestion of a halo segment. In p22 the dark band that would be the halo is dominated by lens shadow; in p25 and p26 there is no halo arc at all. In these photos R22 cannot have been "fit from the saturation ring" — there is no ring. It was inferred from the parhelion positions, the parhelic-belt geometry, or both. That makes `arccos(R22/offset)` a self-consistency check, not an inversion.
- **p27** (h=0.5°): the 22° halo top is visible above the sun, but the parhelia themselves are absent or indistinct at the horizon. R22 from the upper-halo arc is independent in principle; the *parhelion* is the dependent quantity, anchored to the R22 ring at the parhelic-belt level.

The rough split of the 9-photo anchored set is: 4 photos where the route lever is real (p2, p7, p13, p30), 4 photos where R22 is not independently measurable (p20, p22, p25, p26), 1 photo where the parhelion is anchored to R22 (p27).

### A.3 Where is the parhelion peak actually unambiguous?

Peak unambiguity per photo, from inspection:

- **p2:** unambiguous bilateral peaks with full chromatic saturation. Clean.
- **p7:** unambiguous bilateral peaks; both sit close to the 22° halo rim (consistent with h ≈ 59.4° and sec(h) ≈ 2). Clean, but the table only records one side (`393 / n/a`) — the audit cannot tell why the second side was dropped.
- **p13:** strong right parhelion, **diffuse / weak left parhelion**. Anchor records offsets 213/212 with a discrepancy of 1 px. The left-side pick is centroid-of-haze, not a saturation peak; the 1 px symmetry is too clean for the visible asymmetry of the photo. Re-anchoring this from (337, 777) to (330, 755) shifted the sun by 14 px and the inferred altitude from 17° to 7° — that is a 10° anchor swing on the same photo from a hand-anchor adjustment, which is by itself an unstated route-sensitivity result.
- **p18:** listed as anchored in the handoff but **no p18-anchor.json exists in the directory**. Either the file is missing or the handoff overstates coverage.
- **p19:** listed as anchored (cropped CZA receipt) but **no p19-anchor.json exists**. The sun is on or below the horizon and **no parhelia are visible in the photo at all**, so no parhelion-offset measurement is possible.
- **p20:** parhelia visible, but the right one is contaminated by the Getty watermark + lens-flare ghost. The anchor JSON warns of this. The offsets `457/457` are suspiciously symmetric for a contaminated right side.
- **p22:** strong bilateral parhelia, but **no 22° halo arc** to anchor R22 against. R22 = 505 is a guess, and the residual that follows is downstream of that guess.
- **p25:** bright left, diffuse right, no halo arc. R22 = 300 is inferred. The anchor itself flags the left side as foreground/flare contaminated.
- **p26:** bilateral parhelia, no halo. Left/right y picks differ by 12 px (tilt flag in anchor). Right offset is *less* than R22.
- **p27:** **no discrete parhelia in the photo.** The sun is at the horizon, mostly obscured by mountains. The anchor explicitly says the parhelion is a "low-sun dagger estimate from R22 support; not an independent parhelion-core pick."
- **p30:** strong halo, but parhelia near the image edge and vertically broad ("edge-supported, vertically broad" per anchor); L/R y differ by 19 px.

Tally: clean unambiguous bilateral parhelion peaks in p2, p7, p22, p26, p30 (with edge caveat). Single-sided or compromised in p13, p20, p25. Absent in p19 and p27.

### A.4 Verdict on §A

The closeout sentence — "passes residual gate at ~0 px on every eligible photo" — is technically true and substantively misleading:

1. **Fit-artifact:** for low-h photos (p13, p20, p22, p25, p26, p27, p30 — seven of the nine anchored), the geometric lever `sec(h) − 1` is between 0.4% and 2%, so a "0 px residual" lives well inside any reasonable anchor noise budget (cf. the 14 px sun-x correction on p13 alone). The route is not earning what the gate seems to certify.
2. **Circular dependency:** on p20/p22/p25/p26/p27, R22 is not measured independently of the parhelion position. The "test" is closer to a self-consistency check.
3. **Cherry-picked sub-case:** p27 is recorded in the residual table as a 0/0 px parhelion-offset measurement *and* used as a CZA-route residual receipt, despite the anchor file itself flagging the parhelion pick as derived from R22. This double-counting is not visible in the closeout table.
4. **Genuine signal:** the route *does* work on p2, p7, and p13-with-corrected-anchor — the three photos that have both unambiguous bilateral peaks and a directly fittable 22° halo ring. **Three photos, not seven.** That is not "passes residual gate at ~0 px on every eligible photo"; that is "passes residual gate at ~0 px on the three photos where it is non-trivially testable."

Re-stated honestly, the route is: *one image-recoverable inverse handle when sun altitude is far enough from the horizon that `sec(h) − 1` exceeds anchor noise, and when an independent R22 ring can be fit.* The closeout phrasing strips both of these conditions.

This is the §3 outcome (3) the team flagged as useful: **the promoted-route claim is photo-set-conditional, not structurally clean.** If the calibration set were re-weighted to be h > 15° photos only, you would have three eligible photos (p2, p7, p13) and a coverage-gate problem identical to the supralateral route. The fact that the parhelion route looks "promoted" and the supralateral route looks "coverage-gate-failed" is itself an artifact of how many low-h photos were added under FF1.

---

## §B. Forward-generation under audit

The atlas claim is `h → all primitives` runs cleanly forward through: parhelion offset, 22° halo radius, 46° halo radius, CZA visibility window, tangent positions, supralateral position, parhelic circle, infralateral arcs. Going primitive-by-primitive:

1. **Parhelion offset.** Forward direction is fine *as geometry* (`offset = sec(h) · R22`). Sound.
2. **22° halo radius.** Forward direction is a constant 22°; not a forward "prediction" so much as a fixed property of the atlas. Sound.
3. **46° halo radius.** Forward direction is also a constant. But the *visibility* of the 46° halo is regime-bounded and the atlas overlays draw it whether visible or not. On p2 the upper portion of the 46° halo is visible and the overlay matches. On p7 the rich-vocabulary overlay draws a partial 46° halo where the photo shows nothing identifiable. On p13 the overlay draws the 46° in a sky region where I see no 46° arc. **Over-claim:** drawing the 46° halo on every overlay reads as forward richness; only one of three rich-vocabulary photos actually carries the feature.
4. **CZA visibility window.** Sound as a window claim (`h < 32.2°`); the overlay correctly suppresses CZA on p7. **Position prediction inside the window is exactly the failing CZA-apex inversion route**, so forward and inverse fail together here — a primitive whose inverse fails residual gate cannot be unambiguously forward-clean either. p2 and p27 each carry y-residuals of ~20 px against the forward prediction. **Over-claim:** "h → CZA position runs cleanly forward" requires hedging by `±20 px` at the apex.
5. **Tangent positions.** The detection-gate failure on tangent curvature is also a forward-position failure: on p2, the predicted tangent fuses with the 22° halo top; on p7, the broad tangent is visible at a different y than the column-peak protocol's predicted spine; on p13, no clean tangent visible at predicted locus; on p27, predicted tangent merges with CZA. The atlas does not place the tangent at a stable photo-recoverable locus *in either direction*. The team treats this as "detection-gate," but the forward-position prediction is also not crisp — the brightness/chromatic spine is not where the predicted spine is. **Over-claim:** forward tangent is geometric, but the brightness peak is not at the geometric spine.
6. **Supralateral position.** Visible on p2 only. The forward prediction can be drawn on the other photos but is not testable. **Over-claim:** "h → supralateral position runs cleanly forward" cannot be evidenced from a coverage-failed feature.
7. **Parhelic circle.** Drawn as a horizontal belt at the parhelic-belt y. **Belt-y residual was actively falsified by FF1** (PHASE10_BELT_Y_RESULTS) — the `−0.05 · R22` rule does not replicate sign across low-h photos; Spearman ρ(h, residual) = 0.086. The team retired the watch-list, but the forward prediction the atlas draws is the `−0.05 · R22` rule, which is now known not to match observation in a sign-consistent way. **Over-claim:** the parhelic circle's forward y-position is wrong by up to ~11 px in opposite directions on different low-h photos.
8. **Infralateral arcs.** Visible on p2 periphery only. Drawn on every rich-vocabulary overlay — including p7 and p13, where they are not visible. Literature (atoptics.co.uk infralateral arc page; cloudatlas.wmo.int infralateral-arc page) calls infralateral arcs a *rare* halo and notes their visibility is strongly h-dependent and apparition-set-dependent. **Over-claim:** the atlas predicts a visible infralateral on every photo with h < 50°, but the literature would predict the infralateral as unusual on most photos and the calibration set bears that out (1/9 anchored photos).

### Summary of §B over-claims

| primitive | forward geometric? | forward visible at predicted locus? | over-claim layer |
|---|---|---|---|
| parhelion offset | yes | yes (on 3-of-9 photos with both peaks + ring) | photo-set-conditional |
| 22° halo radius | yes | yes when visible | none |
| 46° halo radius | yes | only on p2 in the rich-vocabulary set | drawn on every overlay, not seen on p7/p13 |
| CZA visibility window | yes (window) | window correct; position misses by ~20 px | position over-claimed |
| CZA position | inverse fails | forward also misses by ~20 px in both signs | yes |
| tangent position | geometric | brightness/chromatic peak ≠ geometric spine | yes, in same way detection gate failed |
| supralateral position | geometric | only on p2 | not evidenceable from coverage-failed feature |
| parhelic circle y | geometric | falsified by FF1 at low h | yes (FF1 falsified the rule the atlas draws) |
| infralateral arcs | geometric | only on p2 | drawn everywhere; literature confirms rarity |

Forward-rich is real in the *geometric* sense: every primitive's locus is a function of `h` alone. Forward-rich is *not* real in the "draw it on the photo and the photo shows it" sense. The closeout's forward/inverse asymmetry receipt collapses these two readings.

---

## Pushback / counterproposal

1. **Rewrite the gravity-language receipt.** Replace "the atlas is rich in forward generation and supports one image-recoverable inverse handle" with: "the atlas's primitives are geometrically parameterized by `h` alone; on a 9-photo calibration set, three photos have both unambiguous parhelia and an independently fittable 22° halo, and on those three photos the parhelion-offset inverse recovers `h` to within anchor noise. The other six photos either have no measurable 22° halo, no discrete parhelia, or both, and the residual table for those photos is closer to a self-consistency check than an inversion test." This is more falsifiable and harder to attack.

2. **Add a route-lever column to the residual table.** Show `sec(h_claimed) − 1` next to each photo's residual. Anything below ~1% lever should be marked as anchor-noise-dominated rather than route-validated.

3. **Add an `R22-independence` column.** Mark each anchor as "ring-fit" (independent), "parhelion-derived" (circular), or "inferred-from-other" (semi-circular). Currently this distinction is buried in per-anchor JSON notes.

4. **Re-test the route on a sub-set restricted to h > 15°.** If the verdict survives on p2 and p7 alone, it survives the audit. If you need to include the low-h photos to make the eligibility count exceed 2, you have a coverage problem identical to supralateral.

5. **Fix the p26 anomaly.** Right-side offset 322 with R22 323 is geometrically impossible and currently sits in the table with no flag. Either the right parhelion x or R22 is wrong; the audit cannot tell which from the JSON.

6. **Drop infralateral and 46° halo from the "h → all primitives" enumeration** in the public-framing language, or hedge them as "h → primitive locus (when visible)". Literature ([atoptics infralateral arc](https://atoptics.co.uk/blog/infralateral-arc/)) classifies infralateral as rare; drawing it on every overlay is forward-over-claim by atmospheric-optics norms.

7. **Hedge the CZA forward position by ±20 px.** The same residuals that fail the inverse residual gate also bound the forward-position accuracy. Forward "cleanly" is incompatible with `y = −19.3 / +21` on the only two CZA-eligible photos.

The audit's verdict is not "the route is wrong." It is: **the route is the only handle that survives**, but only because the calibration set has been weighted toward photos where the route degenerates to a self-consistency check, and where two of the other three routes were never going to be testable in the first place. The single-handle finding is real on three photos and inflated on six.

Sources (used in the §B literature check):

- [Sun dog — Wikipedia](https://en.wikipedia.org/wiki/Sun_dog) — parhelion offset increases with sun altitude; fades above ~60°.
- [Sundogs & Sun Altitude — atoptics.co.uk](https://atoptics.co.uk/blog/sundogs-sun-altitude/) — qualitative offset-vs-altitude curve; at h=0 offset is 22°, at h≈47° offset is ~31°.
- [Infralateral Arc — atoptics.co.uk](https://atoptics.co.uk/blog/infralateral-arc/) — infralateral arcs are rare and altitude-dependent; visibility regime split at h≈50° and h≈68°.
- [Infralateral arc — International Cloud Atlas (WMO)](https://cloudatlas.wmo.int/en/infralateral-arc.html) — classifies infralateral as rare halo.
- [Infralateral arc — Wikipedia](https://en.wikipedia.org/wiki/Infralateral_arc) — formation requires rod-shaped horizontally oriented hexagonal crystals, narrower visibility window than 22° halo / parhelia.
