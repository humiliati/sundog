# Persona: CZA residual + Supralateral coverage auditor

Synthetic-audit dry-run, 2026-05-13. Source materials:
`PHASE10_OPTICAL_AUDIT_HANDOFF.md`, `RICH_DISPLAY_OVERLAY_NOTES.md`,
`PHASE10_EXPANSION_TRIAGE.md`, the eleven anchored photos and their
JSON anchor files, plus a scan of `scripts/overlay_calibrate.py`.

## Top-line verdict

- **§A (CZA opposite-sign residuals):** *Pushback.* This is **not** a clean
  route-reliability negative. The −19.3 / +21 px split is an artifact of a
  hardcoded atlas approximation (`CZA apex_y = sun_y − R46`) combined with
  a primitive-misidentification on p27. Re-run the CZA prediction against
  the literature refraction formula and the p2 residual collapses toward
  zero; the p27 measurement should be retracted, not used as evidence of
  route failure.
- **§B (Supralateral coverage):** *Sound with caveat.* The committed set
  genuinely lacks visible supralateral on >1 photo, but supralateral is
  also intrinsically rare in any non-curated dataset (~4 days/yr observable
  vs ~100 for 22° halo; Schaefer / Lynch literature). p27 is the one
  candidate where supralateral may actually be present — and the team is
  currently misreading it as the CZA. Coverage gate is the right verdict
  on the wrong reasoning.

---

## §A — CZA-apex residual gate

**Four-way verdict: not (i)/(ii)/(iii)/(iv) cleanly. It is mostly (ii) +
primitive mis-identification on p27.**

### Atlas-side smoking gun

`scripts/overlay_calibrate.py:381-384`:

```python
# CZA — anchored to 46° halo top
def cza_apex(c):
    anchored = WB_SUN[1] - WB_R46   # 60
    return anchored + (0.85 - max(0.4, min(1.4, c))) * 200
```

The atlas hardcodes the CZA apex at `sun_y − R46`, i.e. at angular
distance 46° above the sun, with no `h`-dependence. This is only true
at h ≈ 22° (where CZA is tangent to the 46° halo top by construction).
At every other h, real CZA sits **above** that point. Literature CZA
zenith distance is `γ` with `sin γ = √(n² − cos²α)`, n ≈ 1.31, α =
sun altitude. Empirical anchor points (atoptics.co.uk, Wikipedia):

| h | CZA altitude | CZA above sun |
| ---: | ---: | ---: |
| 0° | 57.8° | 57.8° |
| 18.6° (p2) | ~64.6° | ~46.0° |
| 22° | ~68° | 46° (tangent to R46) |
| 32.2° | 90° (zenith) | — (degenerate, vanishes) |

### p2 (h = 18.6°, R22 = 182, sun = (567, 496))

- Atlas prediction (hardcoded "sun_y − R46"): y = 496 − 364 = **132**.
- Literature prediction (CZA above sun = 46.4° → 46.4° × 182/22 = 383.7 px):
  y = 496 − 383.7 = **112.3**.
- Observed apex (p2-anchor.json): y = **113**.
- Residual under atlas: 113 − 132 = **−19 px** (matches team).
- Residual under literature formula: 113 − 112.3 = **+0.7 px**.

The −19.3 px residual on p2 is **almost exactly the atlas-vs-literature
prediction gap at h=18.6°**. The team's `sun_px` and `R22_px` anchors
are fine; the visual apex pick at (560, 113) is fine; the overlays
(`p2.overlay.png`, `p2.rich-vocabulary.overlay.png`) visually confirm
the predicted-CZA line (cyan) sitting ~20 px **below** the observed
chromatic arc, which is exactly what the literature/hardcode gap
predicts. **This is a stable, predictable, h-dependent atlas bias on
the CZA primitive, not noise.** Treating p2 as evidence of "route
reliability" was the wrong reading — the route was fine; the
forward-prediction it was being tested against was simplified.

### p27 (h = 0.5°, R22 = 219, sun = (596, 559))

This is the more interesting case. At h ≈ 0.5°:

- Atlas prediction: y = 559 − 438 = **121**.
- Literature prediction: CZA above sun = 57.3° → 57.3° × 219/22 = 570 px;
  y = 559 − 570 = **−11** (off-frame at top).
- "Observed apex" in p27-anchor.json: (599, 142).
- Observed feature is 41.9° above sun (=417 px / 9.95 px/deg).

The bright chromatic arc the team is calling the "CZA apex" in p27 sits
at **~42° above sun**, well below the literature CZA prediction of
~57° above sun (off-frame). The actual CZA at h ≈ 0.5° would be
**above the top of the frame** in p27. The chromatic arc visible in
p27 at y ≈ 142 is therefore **not the CZA** — it is the upper rim of
the 46° halo / supralateral arc complex (which at h ≈ 0 is tangent to
the top of the 46° halo and is geometrically degenerate with the halo
top). Atoptics.co.uk and the Cloud Atlas both note this degeneracy:
at very low sun, supralateral starts "as a pair of arcs tangent to the
sides of a 46° arc" and merges over the halo top; it is well known to
be confused for the 46° halo itself.

p27's `_meta.note` in the anchor JSON already admits the symptom:
*"The CZA band is clear, but the flat apex makes the fitted math apex
sensitive."* A real CZA at h=0.5° is not flat-apexed; it is a narrow,
brilliantly chromatic, near-vertical-cusp smile high in the sky. A
flat-apex chromatic band tangent to the 46° halo top is exactly the
supralateral / 46° halo morphology.

### Reading the team's framing against this

The team writes the +21 / −19 split as a "stable atlas bias has been
falsified — the opposite signs mean the route is unreliable, not
biased." That reading depends on both residuals measuring the same
physical feature against the same atlas formula. **They don't.** p2's
−19 px is the atlas-vs-physics gap on a correctly-identified CZA at
h=18.6°. p27's +21 px is a different primitive (46° halo top /
supralateral) being labelled CZA, against an atlas prediction that
itself is already wrong by ~130 px on the real CZA at h=0.5°. The
"opposite signs" cancel by accident, not by structure.

### Sub-question rejections

- (i) **Anchor error on one photo:** weakly yes on p27, but not in
  `sun_px` / `R22_px` (those are defensible) — the error is in *which
  primitive is being labeled "CZA apex"*. Not the kind of anchor
  error the team's protocol is built to catch.
- (ii) **Genuine atlas bias the team has not recognized:** *yes, this
  is the dominant reading on p2.* The hardcoded `sun_y − R46` is only
  correct at h ≈ 22°. Fix: replace with `sun_y − (90° − h −
  asin(√(n² − cos²(h)) / 1)) × (R22/22)` or equivalent.
- (iii) **Photo-environment / ice-habit sensitivity the atlas
  legitimately ignores:** small contribution at most. Plate-crystal
  wobble (~1° oscillation, Mayor & Tape 1979 / Schaefer 1981 family
  of observations) widens the CZA but does not shift its centroid by
  19 px at R22=182 (which would be ~2.3°). The smudge effect near
  h=32° is also irrelevant for p2 (h=18.6°) and p27 (h=0.5°).
- (iv) **Within real-atmosphere variability:** no. 19 / 21 px on
  R22 = 182 / 219 corresponds to ~2.3° / ~2.1° in sky angle. Plate
  wobble + crystal-quality variation explains ~1° at most. The bulk
  is the atlas formula plus a labelling error.

### What changes the verdict

- Re-run p2's CZA-apex residual against `CZA_above_sun(h) = 90° − h −
  arcsin(√(n²−cos²h))` (use n=1.31). I predict the residual collapses
  to <5 px. If it does, the "CZA route fails residual gate" finding
  retires and CZA-apex becomes a viable second inverse handle on the
  current calibration set (at least on p2).
- For p27, re-do the primitive-ID call. The visible feature at y=142
  is more likely the 46° halo top + supralateral merger than the CZA.
  Either record it as "CZA off-frame / not applicable" or, more
  productively, treat it as the first second-photo of *supralateral*
  visibility (see §B).

---

## §B — Supralateral coverage gate

**Three-way verdict: mostly (i) [intrinsically rare] with a real
(ii) sub-case [p27 is supralateral that the team is misreading as
CZA].**

### What I actually examined

I read every anchored photo in the committed Phase 10 set, looking
specifically for a faint chromatic arc tangent to (and outside of)
the 46° halo on the zenith side, or — at low h — a pair of arcs
tangent to the sides of the 46° halo merging over the top.

| photo | h | 46° halo visible? | supralateral visible? |
| --- | ---: | --- | --- |
| p1 (key) / p2 | 18.6° | yes (faint outer ring) | **yes** — large parabolic arc tangent to 46° halo top is *labelled supralateral* in the p1 key, distinct from the small chromatic CZA above |
| p7 | 59.4° | no — h>32° cutoff | no — supralateral cutoff is also ~32° |
| p13 | 6.83° | no (above frame) | no — top of frame is clean dark blue with no chromatic arc at 46° from sun |
| p18 | low | no | no |
| p19 | ~11° | no (R46 ≈ 540 px > frame) | no — bright feature at top is upper-tangent / suncave-Parry territory, sitting at R22 not R46 |
| p20 | ~5° | no (cropped, R46 ≈ 910 px) | no |
| p22 | ~5.7° | no (cropped) | no |
| p25 | ~11.4° | no (cropped) | no |
| p26 | ~9.0° | no (cropped) | no |
| **p27** | **~0.5°** | **borderline — chromatic arc near R46 visible** | **probable yes — currently mislabelled as CZA** |
| p30 | ~11.1° | no (cropped, top of 22° halo only) | no |

### Why supralateral is intrinsically rare in this dataset

Atmospheric-optics literature is unambiguous: supralateral is rare.
Schaefer / Lynch-family observation counts give ~4 days/year of
visible supralateral against ~100 days/year for the 22° halo (10:1
selection penalty before any cropping or h-window filtering). The
supralateral / 46° halo distinction is itself a famously hard call
that has its own Cloud Atlas entry (*"Differentiation characteristics
of the 46° halo and supralateral arc"*). On a curated atlas-photo set
of 11 anchored frames, finding 1-2 supralateral-eligible photos is
**consistent with literature base rates**, not bad luck.

### Why the team's coverage gate verdict is defensible *for the wrong
reasons*

The team correctly says supralateral is visible only on p2 on the
committed set. They list p27 as a candidate-but-unmeasured. The
correct atmospheric-optics reading is that the chromatic arc on p27
that the team is treating as CZA is, geometrically, *more likely*
to be the supralateral arc (or 46° halo top — they merge at h≈0).
A re-anchor with explicit supralateral-tangency labelling would
plausibly give the team a second eligible photo — at which point the
coverage gate **passes** and supralateral-position → h becomes
measurable on this set.

But the *route* still does not obviously work. Supralateral at h ≈ 0
sits at ~46° above sun (tangent to 46° halo top); at h = 18.6° it
also sits at ~46° above sun on the zenith side. Its angular distance
from the sun changes only ~0.5° over the entire h range from 0° to
22° — much less sensitive to h than parhelion-offset (which changes
~30° over that range). So even with coverage, supralateral-position
→ h is a **structurally weak inverse**: the signal-to-anchor-noise
ratio is order-of-magnitude worse than parhelion offset. The team's
"three layers of failure" framing might survive with a footnote:
supralateral is degenerate-with-46-halo-top *and* has poor
h-discrimination by construction.

### Sub-question dispositions

- (i) **Intrinsically rare:** *yes,* literature-supported. ~4 days/yr
  vs ~100 for 22° halo. Curated atlas sets of 7-11 photos will
  routinely fail the coverage gate.
- (ii) **Actually present beyond p2, missed by detection criterion:**
  *one case, p27.* What the team is calling CZA on p27 is more
  consistent with supralateral / 46° halo top. With a
  primitive-ID correction this *could* push the route to coverage.
- (iii) **Genuinely absent in the calibration set as documented:**
  *substantially yes,* with the p27 caveat. p7 / p13 / p18-30 are
  all either supralateral-cutoff (h>32°) or cropped above the 46°
  halo region. Even an uncropped reshoot would have to thread a
  h<32°, cold-cirrus-or-diamond-dust, plate-crystal-dominant needle
  to add a second supralateral photo.

---

## Pushback / counterproposal

1. **The single-handle verdict's evidentiary chain for CZA-apex is
   compromised.** "Opposite-sign residuals on p2 and p27" is currently
   the load-bearing receipt for "CZA-apex fails the residual gate." On
   inspection, the −19 px is a hardcoded-formula bug at h=18.6° (CZA
   ≠ R46-above-sun outside h≈22°) and the +21 px is a primitive-ID
   error at h=0.5° (the team is measuring something else). Replace the
   atlas's CZA prediction with the refraction formula and re-anchor
   p27's chromatic arc as supralateral / 46° halo top before claiming
   CZA-apex is dead. CZA-apex on p2 alone may pass residual gate;
   the route may still fail coverage (only one eligible photo if
   p27 is reclassified), but the failure layer label changes.

2. **The atlas's forward-generation claim is over-claimed at very low h
   on the CZA primitive.** §2.4 of the handoff says forward generation
   is not under audit. It should be, at least for CZA. The hardcoded
   `sun_y − R46` mispredicts the CZA position by 130+ px at h=0.5°.
   That is forward atlas error, not inverse-route error. If the public
   framing says "rich in forward generation," that claim needs the
   CZA primitive specifically caveated, or the formula fixed.

3. **Supralateral coverage gate is the right verdict, but for a
   different reason than the team thinks.** The team is treating
   supralateral as "almost made coverage; needs one more lucky
   photo." Atmospheric-optics base rates suggest supralateral will
   routinely fail coverage on curated atlas sets of this size, even
   in expansion. A more honest framing is: *"supralateral is rare by
   construction; the coverage gate is a structural finding, not a
   selection-bias one."* This matters for the gravity-side framing —
   "three independent failure layers" implies three independent
   accidents; one of them is a base-rate near-certainty.

4. **The vocabulary on p27 needs reconciliation.** The same chromatic
   arc cannot be simultaneously "CZA" in the per-route residual table
   and "candidate supralateral" in the same row. Pick one. If the
   atlas's degenerate-at-low-h prediction is fixed, the right answer
   may be "neither pure CZA nor pure supralateral — a merged primitive
   that the atlas should explicitly label as such at h<5°."

5. **Concrete next test.** Re-anchor p2 against a refraction-formula
   CZA prediction. If the p2 y-residual drops below 5 px, file an
   addendum: CZA-apex route promotion blocked on *coverage* (single
   photo eligible after p27 reclassification), not residual. The
   single-handle verdict survives but its rationale for the CZA
   row changes from "residual gate" to "coverage gate." That is a
   smaller, cleaner result than the current claim.

---

## Sources retrieved

- [Supralateral arc — Wikipedia](https://en.wikipedia.org/wiki/Supralateral_arc)
- [Circumzenithal arc — Wikipedia](https://en.wikipedia.org/wiki/Circumzenithal_arc)
- [Supralateral & Infralateral Arcs — Atmospheric Optics (Les Cowley)](https://atoptics.co.uk/blog/supralateral-infralateral-arcs/)
- [Is it a 46° halo or a supra/infralateral arc? — Atmospheric Optics](https://www.atoptics.co.uk/blog/is-it-a-46-halo-or-a-supra-infralateral-arc/)
- [Circumzenithal arc — International Cloud Atlas / WMO](https://cloudatlas.wmo.int/en/circumzenithal-arc.html)
- [Differentiation characteristics of the 46° halo and Supralateral arc — Cloud Atlas](https://cloudatlas.wmo.int/en/differentiation-characteristics-of-the-46-degree-halo-and-Supralateral-arc.html)
- [Circumzenithal Arcs vs Solar Elevation — Atmospheric Optics](https://atoptics.co.uk/blog/circumzenithal-arcs-vs-solar-elevation/)
- [CZA — Effect of solar altitude — Atmospheric Optics](https://atoptics.co.uk/blog/cza-effect-of-solar-altitude/)
- [Frequency analysis of the circumzenithal arc: Evidence for the oscillation of ice-crystal plates — JOSA 69(8), 1119 (1979)](https://opg.optica.org/josa/abstract.cfm?uri=josa-69-8-1119)
