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

- **Measured CZA expansion photo:** p27. It is CZA-visible, but its residual
  direction is opposite p2.
- **Fallback checked and rejected for CZA:** p20. Plausible R22 anchors put
  the CZA apex above the frame; watermark/flare issues remain secondary.
- **Demoted after diagnostic overlay:** p19. Its visible upper chromatic arc
  aligns with the top of an R22 ~= 270 px halo / upper-tangent region, not a
  separate CZA. The CZA apex predicted from that anchor is off-frame.
- **Best supralateral / rich-display candidate:** p27.
- **Best parhelia / parhelic-belt checks:** p18, p22, p25, p26, p30.
- **Low-value for CZA reopening:** p21, p23, p24, p28, p31.

Task #55 resolves the immediate CZA promotion question: p27 survives anchor
capture, p20 does not, and p19 remains cropped under the R22 ~= 270 px
diagnostic. The CZA-apex route now fails the residual gate on two eligible
photos: p2 y residual = -19.3 px; p27 y residual = +21 px. The opposite signs
mean p2 was not evidence of a stable direction-and-magnitude atlas bias.

## Candidate Inventory

| id | file | rough sun px | rough R22 px | rough h regime | CZA eligibility | supralateral / upper-arc value | triage call |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| p18 | `18.a-solar-sun-halo-sundog-forms-thanks-to-ice-crystals-off-a-steamy-niobrara-river-during-a.webp` | `(269,108)` | TBD | low / mid | none visible | weak | parhelia / belt reference only |
| p19 | `19.cloud-gallery-7010-1-700x525.jpeg` | `(359,430)` | ~270 | h ~ 11° from provisional parhelion offset 275 | not applicable / cropped; predicted CZA apex y ~ -110 | strong upper tangent / 22-halo top; possible non-CZA point feature near y ~ 85 | **demote from CZA coverage**; useful as a low-sun upper-tangent / halo-separation caution |
| p20 | `20.gettyimages-157530019-2048x2048.jpg` | `(1011,827)` | ~455 | low (~5° provisional) | not applicable / cropped; predicted CZA apex y ~ -83 | weak / candidate; watermark + flare contamination | fallback checked and rejected for CZA coverage; keep as low-confidence parhelia/belt reference |
| p21 | `21.HaloRichard1.jpg` | `(565,284)` | TBD | unknown | not visible / likely cropped | weak | skip for CZA; possible parhelia stress case |
| p22 | `22.i553d1heikh61.jpg` | `(646,449)` | TBD | low / mid | not visible | weak | strong parhelia / belt reference |
| p23 | `23.spectacular-halo-around-sun-airplane-clouds-against-clear-sky-striking-atmospheric-phenomenon-solar-along-404048553.webp` | `(255,221)` | TBD | unknown / high-ish | not visible | weak | skip for CZA; halo-only reference |
| p24 | `24.sundog-mountains-over-blue-ridge-warm-autumn-day-north-carolina-102667537.webp` | `(212,149)` | TBD | unknown | not visible | weak | skip for CZA |
| p25 | `25.Karen-Peck.jpg` | `(490,189)` | TBD | low / mid | not visible / cropped | weak | parhelia / belt reference |
| p26 | `26.Martin-MacFarlane.jpg` | `(465,203)` | TBD | low / mid | not visible / cropped | weak | parhelia / belt reference |
| p27 | `27.Zzhau2Uiospj37zhwHEr4D-1200-80.jpg.webp` | `(596,559)` | ~219 | very low (~0.5° provisional) | **visible CZA; apex residual y = +21 px** | **strong supralateral / rich-display value** | **measured; converts CZA-apex verdict to residual-gate failure** |
| p28 | `28.sundog.webp` | `(243,179)` | TBD | low / mid | not visible / likely cropped | weak | skip for CZA |
| p30 | `30.Sundog1.jpg` | `(700,933)` | TBD | low | cropped: predicted CZA likely above frame | rich parhelia / possible outer arcs | good low-sun belt / 22-halo reference, not CZA |
| p31 | `31.sundogs-sun-dogs-parhelia-mock-suns-1.png` | `(508,335)` | TBD | unknown | not visible | weak | skip for CZA |

## Recommended Sequence

1. Keep p27's anchor JSON with the sign-explicit CZA residual:
   observed apex `(599,142)`, predicted `(596,121)`, observed-minus-predicted
   y residual `+21 px`.
2. Keep p20's anchor JSON as a cropped/non-eligible CZA receipt:
   with R22 ~= 455, predicted CZA apex y ~= -83.
3. Keep p19 as a negative/cropped CZA receipt unless a manual R22 pick
   overturns the R ~= 270 diagnostic. Its provisional overlay is internally
   consistent with R22 ~= 270, h ~= 11°, parhelion offset ~= 275, and CZA apex
   y ~= -110.
4. Resume Task #54 tangent-curvature sampling. The CZA route is now parked as
   a residual-gate failure unless a new, clean CZA-visible photo enters the
   calibration set.

## Roadmap Consequence

Task #54 is again the next useful measurement gate. The gravity-side receipt
must be weakened: CZA-apex asymmetry is real as a route-vs-route reliability
gap, but it does **not** repeat as a stable same-direction atlas bias.
