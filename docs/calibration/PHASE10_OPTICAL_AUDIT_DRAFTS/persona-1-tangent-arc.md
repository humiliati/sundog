# Persona-1 Tangent-Arc Detection-Gate Audit (Synthetic Dry-Run)

**Persona:** Tangent-arc detection-gate auditor
**Scope:** Phase 10 claim that "tangent-arc-curvature → h fails detection across all four eligible photos with three distinct per-altitude degeneracy modes; column-peak detection is the wrong instrument; the arc is real and visible but its brightness/chromatic spine is not a per-column peak."
**Top-level verdict:** **MIXED — leaning PROTOCOL ARTIFACT for two of four cases, GENUINE DEGENERACY for one, and OUT-OF-TAXONOMY for one.** The team's framing is mostly correct in spirit (column-peak is the wrong instrument) but is muddled by a literature-level misclassification at h=59.4° and an over-generous regime claim at h=0.5°.

The detection-gate failure is real. The per-altitude story the team is selling is sloppier than it needs to be.

---

## Per-case verdicts

### p2 (h = 18.6°) — "compression" / fuses with 22° halo top

**Pushback: this is not a detection failure of the tangent arc; it is the textbook appearance of the upper tangent arc at this h.**

Visual: p2 shows a clean, brilliantly coloured upper-tangent "smile" that *sits on the 22° halo crown* exactly where Cowley's standard description places it — "they touch the 22° halo directly above and below the sun" — and the team's own anchor (sun y=496, R22=182 → halo top at y≈314) puts the tangent's chromatic spine right where the eye finds it in the photograph. Red is sunward, as expected.

The "compression" framing makes it sound like a pathology. It is not. At h=18.6° the upper tangent arc *is* tangent to the 22° halo by construction (zero-angle separation at the sun-meridian, opening into wings on either side). A vertical-column peak detector at the sun-azimuth meridian will see one brightness plateau (halo top + tangent spine, additively combined) and will, correctly, report one peak — but that peak is not a *curvature* signal because the curvature of the tangent arc at its apex on this photo is empirically *near-zero in the vertical direction* (the smile is locally horizontal at the apex, by literature definition of tangency). A column-peak detector tuned to find a curvature spine where the geometry says the spine is locally flat is asking the wrong question.

Counterproposal: the tangent arc's signal at this h lives in the *off-meridian* azimuthal extent of the wings, not at the sun-meridian. A radial profile sampled along arcs at fixed angular offsets from the sun-meridian, or a Frangi/Hessian ridge filter on a 22°-halo-subtracted residual image, will recover the wing-curvature easily. The Lab b\*-channel (yellow-blue) ridge is visible on the photograph all the way out to ~±20° of azimuth. Standard atmospheric-photometry literature (Tape; haloblog.net catalogues) routinely measures tangent-arc opening angle from the wings, not from the apex.

### p7 (h = 59.4°) — "halo-edge dominance" / "broad-tangent regime"

**Pushback: at h = 59.4° this is not an upper tangent arc. It is a circumscribed halo. The team is testing the wrong primitive against the wrong literature.**

Visual: p7 shows a near-circular 22°-radius bright ring, with parhelia clearly displaced *outward* from the ring at the sun-altitude line (consistent with h > 45°). What the team is calling the "broad tangent arc" above the sun is the **upper portion of a circumscribed halo**, which at this altitude is barely distinguishable from the 22° halo proper. Cowley (atoptics.co.uk, *Tangent Arcs*, updated 2024-12-16): "When the sun climbs above 29º both arcs join into a single halo wrapped right around the sun — the 'circumscribed halo'." Fleet (dewbow.co.uk, *High Sun Tangent Arcs*): "The circumscribed halo lies a couple of degrees outside the 22 degree halo at the altitude of the sun, it coincides with the main halo above and below the sun."

At h=59.4°, the geometric separation between the predicted upper-tangent "spine" and the 22° halo crown above the sun is *of order 1–3°*, which on p7 (R22=200 px) is **at most ~10 px**. The team's column-peak detection grabbed the 22° halo's outer edge "because it dominated"; that is correct behaviour by the detector because *at this h the upper-tangent arc and 22° halo crown are physically coincident to within image resolution*. This is not a detection-gate failure of a curvature signal that exists. It is a geometric-degeneracy non-existence-of-signal at this h.

This case should be re-classified out of the "tangent-arc inversion route" entirely. At h ≥ ~30° the inversion route is "circumscribed-halo ellipticity → h", which is a different inversion (the 2D ellipse aspect ratio of the circumscribed halo *is* image-recoverable and is a known atmospheric-optics measurement). The team is not testing that route. They should either drop p7 from the tangent-arc-route eligibility set (out-of-window failure, not detection failure), or replace the test with circumscribed-halo-ellipticity → h.

This is the case where I am most confident the team's reading is wrong. Not because the column-peak failed, but because the team is describing a circumscribed halo as a "broad upper tangent arc."

### p13 (h = 6.83°) — "chromatic-haze contamination"

**Sound with caveat: the failure is real, but it is partially anchor-recovery-driven and partially detector choice.**

Visual: p13 is a low-sun snow-scene with strong forward-scatter haze around the sun; the 22° halo is clearly visible as a circle, and the parhelia are bright, but the upper-tangent region above the sun is washed into a luminous chromatic gradient with no localizable brightness peak. I agree with the team that column-peak detection produces nonsense here (RMS=41 px, radius 196 px vs predicted 1774 px is a fit divergence, not a noisy fit).

That said: the *chromatic* spine is visible to the eye as a faint yellow-to-blue Lab b\*-channel transition arcing across the top of the 22° halo. It is a chromaticity *transition*, not a brightness peak. The team correctly identifies this in the v3.8 receipt summary ("the spine is at a brightness or chromaticity *transition*, not a peak"). So the protocol-artifact reading is correct *for this case* — a Lab b\*-channel ridge filter, or a Sobel-x on the chromaticity image rather than the intensity image, would plausibly recover the arc.

But there is a partial caveat: at h=6.83° the upper-tangent arc *is* near its most-detectable geometric regime (open V-shape, well clear of the 22° halo top by ~3–5°). The fact that it is not cleanly visible on this *specific* photograph is partly an atmospheric/photometric property of the photo (forward-scatter haze, snow albedo back-illumination of the foreground reducing contrast), not a property of the route. A different low-h photo without snow-scene foreground glare would likely have a measurable tangent-arc apex. The team's claim "the route is detection-degenerate across the entire calibration altitude range" overgeneralizes from a small sample.

Counterproposal: chromaticity-channel ridge detection on a 22°-halo-subtracted residual image. Also: this is the case most worth re-testing on an expansion photo (e.g., one of the p18/p22/p25/p26/p30 set if any have a visible upper-tangent region) before declaring h≈5–10° a detection-degenerate regime.

### p27 (h = 0.5°) — "CZA / sun-bloom merge"

**Out of my area as stated, but pushback on the framing: at h≈0.5° the upper tangent arc and the CZA are at nearly the same elevation by accident of geometry, but they are not the same arc and they should not merge in the photograph.**

Visual: p27 shows a striking, near-horizon scene with what reads cleanly as an upper tangent arc above the sun (the colored "smile" with red sunward) and a separate, higher chromatic feature that the team has labeled as CZA. At h=0.5° the upper tangent should be at altitude ≈22.5° above horizon; the CZA should be at altitude ≈90° − ZD_CZA where ZD_CZA depends on h such that at h=0.5° the CZA is at ~57° altitude. These are *not* coincident; they are separated by ~35° of altitude. They cannot "merge" in any literal optical sense.

What the team is probably seeing is that **the sun-bloom column (the vertical bright streak above the sun caused by camera flare / lens scattering / sun-pillar component) crosses the tangent-arc spine vertically, swamping the per-column brightness signal at the sun-meridian**. That is a *photometric artifact of the imaging chain*, not an atmospheric-optical merging of features.

So the route does fail at p27, but the named cause ("CZA / sun-bloom merge") is mixing two things. Strip the sun-bloom artifact (median-filter along the radial-from-sun direction, or use an off-meridian sample) and the tangent spine is visible. The CZA is at a completely different altitude in this photograph.

Counterproposal: off-meridian column sampling, or radial-deflare. Also: the team's "merges with CZA" language should be dropped from the receipt; this is a sun-bloom failure, not a feature-merge failure.

---

## Section-level verdict on "column-peak is the wrong instrument"

**Mostly supported, but the team is overclaiming structural generality.**

The visual evidence supports the *narrow* claim "a per-column-vertical brightness peak in the intensity image, sampled along the sun-meridian, is the wrong detector for the upper tangent arc on any of these four photographs." That is true and the program should adopt it.

The visual evidence does **not** support the broader claim "the upper tangent arc as a feature lacks a brightness/chromatic spine that any peak-based detector could find." Specifically:

- For p2, the apex is locally flat *by tangency definition*; a *curvature* detector applied to the *wings* would work, and the wings are detectable.
- For p7, there is no upper tangent arc in this photograph to detect — it is a circumscribed halo at h=59.4°. The failure is regime-misclassification, not detection.
- For p13, a chromaticity-channel detector would plausibly recover the route, modulo the snow-scene contamination.
- For p27, an off-meridian or deflared column would recover the route.

Three of the four "detection-degenerate" calls are protocol artifacts (wrong meridian, wrong channel, wrong primitive). One (p7) is an out-of-window classification error masquerading as a detection failure. None of the four is, by itself, evidence that the upper-tangent arc is intrinsically un-inverse-mappable across its altitude range.

The team's stronger framing — "this is a publishable clean-negative for the tangent-curvature inversion route as a class" — is **not defensible to an atmospheric-optics reader.** The defensible version is: "column-peak-on-sun-meridian fails on all four photos in our calibration set, with photo-specific causes; whether a better detector recovers the route is a tooling question, not a physics question." Which is, charitably, what the team already says in Open Question #2. The receipt-level language in the *Tangent-Curvature v3.8 Receipt* and the *Single-handle closeout* sections is stronger than that and should be hedged before going into public-framing materials.

---

## Counterproposal summary

If the team wants to know whether the tangent-arc-curvature route is image-recoverable in principle:

1. **Re-classify p7 out of the tangent-arc-route eligibility set.** It is a circumscribed-halo display, not an upper-tangent display. Per Cowley *Tangent Arcs* (https://atoptics.co.uk/blog/tangent-arcs/) and Fleet *High Sun Tangent Arcs* (http://www.dewbow.co.uk/haloes/utan1.html), h ≥ 29° is the circumscribed-halo regime. If the team wants an inverse route at this h, it is "circumscribed-halo aspect ratio → h", a different and well-attested route. The "broad-tangent" framing in the v3.8 receipt is a literature-level misclassification.

2. **For p2, sample the wings, not the apex.** The upper-tangent arc at h=18.6° is locally tangent to the 22° halo at its apex by *definition*. The curvature signal is in the wing opening, ~±15–25° of azimuth from the sun-meridian. A Frangi/Hessian ridge filter on a 22°-halo-subtracted residual image, or radial profiles at fixed azimuth offsets, will recover the wing curvature. Standard tangent-arc opening-angle measurement in the field (see e.g. haloblog.net display catalogues) is wing-based, not apex-based.

3. **For p13, switch detection to the chromaticity channel.** Convert to CIE Lab and apply a ridge filter to the b\* channel (yellow-blue) on a 22°-halo-subtracted residual. The team's own v3.8 receipt language ("the spine is at a brightness or chromaticity *transition*, not a peak") points exactly here. A Sobel-x or Canny edge on the b\* image will plausibly recover the arc spine. Whether this generalizes to a low-h regime call requires one or two more low-h photos without strong foreground forward-scatter — p13's snow-scene haze is partly photo-specific.

4. **For p27, deflare radially and sample off-meridian.** The "merges with CZA" framing in the receipt is wrong (CZA is ~35° higher in altitude than the upper-tangent at h=0.5°). The actual problem is sun-bloom flare contaminating the sun-meridian column. Median-filter or polynomial-detrend along the radial-from-sun direction, then re-sample.

5. **Cap the receipt at "column-peak-on-meridian fails; tooling question filed."** Do not generalize to "the upper tangent arc lacks a brightness/chromatic spine"; the literature describes a clear chromatic spine across the entire low-and-mid-h regime, with red sunward and the apex on the 22° halo crown (Cowley, *Tangent Arcs*, atoptics.co.uk).

If the team adopts (1)–(4), I expect at least p2-wings and p13-chromaticity to come back with measurable curvature; p7 drops out of scope entirely; p27 is a tooling fix. That is at least 2-of-3 in-window photos recovering — enough to clear the two-photo coverage rule. The detection gate fails as stated *only* under the team's column-peak-on-meridian-in-intensity-image protocol; under a literature-standard protocol (wing-based, chromaticity-aware) it is plausibly recoverable on the existing calibration set.

That said, the headline finding "one image-recoverable inverse handle survives audit" may still hold *even if* tangent-arc-curvature recovers, because the team's three-gate taxonomy independently fails CZA-apex (residual) and supralateral (coverage). A successful tangent-curvature recovery would shift the verdict from "one handle" to "two handles," which is outcome (2) in §3 of the handoff and is one of the outcomes the team explicitly wants surfaced.

---

## Sources (retrieved 2026-05-13/14)

- Cowley, L. *Tangent Arcs.* Atmospheric Optics, updated 2024-12-16. https://atoptics.co.uk/blog/tangent-arcs/
  - "On less favourable days they can be just local brightenings of the 22º halo."
  - "When the sun climbs above 29º both arcs join into a single halo wrapped right around the sun — the 'circumscribed halo'."
  - "Red light is refracted less strongly than other colours and so the halo edge closest to the sun is red."
- Fleet, R. *High Sun Tangent Arcs.* dewbow.co.uk, 2004-2008. http://www.dewbow.co.uk/haloes/utan1.html
  - "The circumscribed halo lies a couple of degrees outside the 22 degree halo at the altitude of the sun, it coincides with the main halo above and below the sun."

I did **not** retrieve Greenler (*Rainbows, Halos, and Glories*) or Tape (*Atmospheric Halos*) for this memo. The crystal-orientation and h-regime claims above are sourced only from the two retrieved web references. Treat the wing-based-measurement claim as standard-practice in the halo-observation community as documented on the haloblog.net and atoptics.co.uk sites generally; I have not cited a specific haloblog.net page.
