# Phase 10 Calibration-Set Expansion Triage

Date: 2026-05-13

Purpose: decide whether the new calibration photos should be folded into the
Phase 10 / Task #53 CZA-apex route before Task #54 tangent-curvature sampling.
The load-bearing question is no longer only "can we label more arcs?" It is:
does the p2 CZA-apex residual (~19 px y) repeat on other CZA-visible photos, or
was p2 anomalous?

This is a rough eligibility pass, not a committed anchor file. Sun and R22
values below are visual / saturation-pipeline estimates and must be replaced
with per-photo JSON anchors before promotion-gate scoring.

## Headline

Triage should run before Task #54.

- **Top CZA candidate:** p27.
- **Fallback CZA candidate:** p20, but it has watermark/flare issues.
- **Demoted after diagnostic overlay:** p19. Its visible upper chromatic arc
  aligns with the top of an R22 ~= 270 px halo / upper-tangent region, not a
  separate CZA. The CZA apex predicted from that anchor is off-frame.
- **Best supralateral / rich-display candidate:** p27.
- **Best parhelia / parhelic-belt checks:** p18, p22, p25, p26, p30.
- **Low-value for CZA reopening:** p21, p23, p24, p28, p31.

If p27 and one fallback photo survive anchor capture, the CZA-apex route
reopens for a real promotion-gate evaluation. Three CZA measurements would
tell us whether p2's 19 px y residual is systematic (strong gravity-claim
receipt) or photo-specific (claim narrows to p2 / lens / crop conditions).
p19 no longer counts toward that coverage unless a later manual anchor
overturns the R22 ~= 270 px diagnostic.

## Candidate Inventory

| id | file | rough sun px | rough R22 px | rough h regime | CZA eligibility | supralateral / upper-arc value | triage call |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| p18 | `18.a-solar-sun-halo-sundog-forms-thanks-to-ice-crystals-off-a-steamy-niobrara-river-during-a.webp` | `(269,108)` | TBD | low / mid | none visible | weak | parhelia / belt reference only |
| p19 | `19.cloud-gallery-7010-1-700x525.jpeg` | `(359,430)` | ~270 | h ~ 11° from provisional parhelion offset 275 | not applicable / cropped; predicted CZA apex y ~ -110 | strong upper tangent / 22-halo top; possible non-CZA point feature near y ~ 85 | **demote from CZA coverage**; useful as a low-sun upper-tangent / halo-separation caution |
| p20 | `20.gettyimages-157530019-2048x2048.jpg` | `(1011,827)` | TBD | low / mid | candidate, but flare/watermark contaminate field | weak / candidate | fallback if p27 survives and a second new CZA measurement is needed, or if p27 fails anchor capture |
| p21 | `21.HaloRichard1.jpg` | `(565,284)` | TBD | unknown | not visible / likely cropped | weak | skip for CZA; possible parhelia stress case |
| p22 | `22.i553d1heikh61.jpg` | `(646,449)` | TBD | low / mid | not visible | weak | strong parhelia / belt reference |
| p23 | `23.spectacular-halo-around-sun-airplane-clouds-against-clear-sky-striking-atmospheric-phenomenon-solar-along-404048553.webp` | `(255,221)` | TBD | unknown / high-ish | not visible | weak | skip for CZA; halo-only reference |
| p24 | `24.sundog-mountains-over-blue-ridge-warm-autumn-day-north-carolina-102667537.webp` | `(212,149)` | TBD | unknown | not visible | weak | skip for CZA |
| p25 | `25.Karen-Peck.jpg` | `(490,189)` | TBD | low / mid | not visible / cropped | weak | parhelia / belt reference |
| p26 | `26.Martin-MacFarlane.jpg` | `(465,203)` | TBD | low / mid | not visible / cropped | weak | parhelia / belt reference |
| p27 | `27.Zzhau2Uiospj37zhwHEr4D-1200-80.jpg.webp` | `(596,559)` | TBD; inner 22 halo must be separated from outer arcs | below CZA cutoff | **very strong visible CZA** | **strong supralateral / rich-display value** | **priority anchor capture** |
| p28 | `28.sundog.webp` | `(243,179)` | TBD | low / mid | not visible / likely cropped | weak | skip for CZA |
| p30 | `30.Sundog1.jpg` | `(700,933)` | TBD | low | cropped: predicted CZA likely above frame | rich parhelia / possible outer arcs | good low-sun belt / 22-halo reference, not CZA |
| p31 | `31.sundogs-sun-dogs-parhelia-mock-suns-1.png` | `(508,335)` | TBD | unknown | not visible | weak | skip for CZA |

## Recommended Sequence

1. Create an anchor JSON for p27 first:
   - sun centroid,
   - R22 from a manual 22-halo pick,
   - left/right parhelion or 22-halo support points,
   - CZA apex from a parabola fit or manual lower-edge samples,
   - visibility flags for supralateral and upper tangent.
2. Run `overlay_calibrate.py --anchors` for p27 and record:
   - parhelion-offset residual,
   - CZA apex residual,
   - whether CZA y residual direction matches p2.
3. Pull p20 into the measured set if p27 survives and a third CZA measurement
   is needed for the gravity-claim asymmetry receipt. Treat p20 as lower
   confidence because of watermark / flare contamination.
4. Keep p19 as a negative/cropped CZA receipt unless a manual R22 pick
   overturns the R ~= 270 diagnostic. Its provisional overlay is internally
   consistent with R22 ~= 270, h ~= 11°, parhelion offset ~= 275, and CZA apex
   y ~= -110.
5. Resume Task #54 tangent-curvature sampling after the CZA coverage question
   is resolved or explicitly parked.

## Roadmap Consequence

Task #54 remains useful, but it is no longer the highest-leverage immediate
move. The gravity-side receipt now depends more on whether the p2 CZA-apex
asymmetry repeats on p27 plus a fallback CZA-visible photo than on adding
tangent curvature to the same original three-photo set.
