# Phase 10 Optical Audit — Synthetic Memo

> Synthetic audit dry-run. Three persona auditors (tangent-arc detection / CZA + supralateral / parhelion + forward-generation) examined the Phase 10 claims against the photo set and atmospheric-optics literature. This memo consolidates the findings and is intended as input to the team before any real-specialist outreach. Load-bearing claims have been independently verified against the source files, the photographs, and the atmospheric-optics literature; unverified or contradicted claims have been filtered out and recorded in §3.

## 1. Headline verdict

The audit returns **§3 outcome (3) of the handoff: "A reinterpretation of the parhelion-offset success as photo-set-lucky rather than structurally clean"** — with secondary partial hits on outcomes (2) and (4). The single-handle verdict is not falsified, but its evidentiary chain is **substantially weaker than the closeout claims**: parhelion-offset is a clean recovery on **three** photos (p2, p7, p13 re-anchored), not "every eligible photo," because on six of the nine anchored photos the test is either anchor-noise-dominated (sec(h)−1 lever ≤ 2% on five low-h photos) or measuring `offset = R22` against an `R22` that was itself inferred from the parhelion. The CZA-apex residual-gate failure is partly an artifact of a hardcoded atlas formula that mispredicts CZA position at any h ≠ 22°. The tangent-arc detection-gate failure is real for the column-peak protocol but does not survive as a route-level negative under a literature-standard wing-based or chromaticity-based detector. The supralateral coverage-gate failure is structurally correct but for a different reason than the team thinks (literature base rate, not bad luck). The "rich forward generation" claim is over-claimed for at least three primitive classes. Do not bake the current single-handle language into public framing without the doc changes in §4.

## 2. Verified findings that should change the Phase 10 verdict

- **Atlas hardcodes CZA apex_y = sun_y − R46 (overlay_calibrate.py:381-384), with no `h`-dependence** [Persona 2]. Verified by reading the file. This is correct only at h ≈ 22°. At p2 (h = 18.6°) the literature CZA position (sin γ = √(n²−cos²α), n = 1.31, per iapetus.jb.man.ac.uk/cza/CZA.html and atoptics.co.uk) places the apex ~19 px above the hardcoded prediction, which almost exactly matches the −19.3 px residual the team has filed as a "route-reliability" failure. Compounding bug: `WB_R46 = 440 = 2·WB_R22`, but the true angular ratio is 46/22 ≈ 2.091, so the atlas's "R46" in pixels is ~4% too small even at h = 22°.
- **The p2 −19.3 px CZA residual is mostly an h-dependent atlas-vs-literature gap, not a route-reliability signal** [Persona 2]. Verified by arithmetic. Re-running p2 against `CZA_above_sun(h) = 90° − h − arcsin(√(n²−cos²h))` collapses the residual to <5 px (Persona 2 predicts ~0.7 px).
- **The p27 +21 px CZA residual is a primitive-misidentification, not a CZA measurement** [Persona 2]. At h = 0.5° the literature CZA apex is ~57° above the sun and lies above the top of the frame. The chromatic arc visible at y = 142 in p27 sits at ~42° above the sun and is the 46° halo top / supralateral merger, well known in the literature to be confused for CZA at very low h. The p27-anchor.json `_meta.note` already flags the symptom ("flat apex makes the fitted math apex sensitive") — a true CZA at h = 0.5° is narrow and high; a flat-apex chromatic band tangent to the 46° halo top is supralateral / 46°-halo morphology.
- **The closeout claim "passes residual gate at ~0 px on every eligible photo" is photo-set-conditional, not structural** [Persona 3]. Verified by arithmetic and by anchor inspection. For the parhelion route `offset = sec(h)·R22`, sec(h)−1 = 0.71% at h = 6.83° and 5.52% at h = 18.6°. At h < 11° (five of the nine anchored photos: p13, p20, p22, p25, p26) the lever is below 2% of R22 (~1.5–10 px depending on R22), which is at or below typical anchor noise (the p13 re-anchor itself moved the sun by 14 px). On those photos a 0 px residual is anchor-noise-bounded, not route-validated.
- **p27 parhelion offset is not an independent measurement** [Persona 3]. Verified by reading p27-anchor.json: parhelion `visibility` field reads verbatim *"low-sun dagger estimate from R22 support; not an independent parhelion-core pick."* The recorded `offset_L = offset_R = R22 = 219` is what you get when you stipulate offset := R22; the "0 px residual" is tautological.
- **p26 right-side anchor encodes a geometric impossibility** [Persona 3]. Verified by reading p26-anchor.json: sun_x = 464, right_x = 786, R22 = 323. Right offset = 322 < R22 = 323, so `arccos(R22/offset)` is undefined. The residual table records this with no flag, demonstrating the table does not actively enforce the geometric constraint that defines the route.
- **The `R22 via saturation-ring fit (not visual ring guess)` claim from handoff §2.3 does not survive image inspection on three of the four FF expansion photos** [Persona 3]. Verified by reading the images: **p20, p25, p26 show no continuous 22° halo arc** — only parhelia and (in p20) a sun pillar — so R22 on those photos is not from a ring fit. ⚠ **Partial contradiction:** p22 *does* show a clear 22° halo arc connecting the parhelia (with the central region washed by sun glare), so Persona 3 overstated the p22 case. Three-of-four still stands as a substantive finding.
- **The parhelic-circle y rule `−0.05 · R22` was actively falsified by FF1** [Persona 3]. Verified in PHASE10_BELT_Y_RESULTS.md: residuals on the six FF photos do not share a sign (p13 and p26 negative, p25 positive), Spearman ρ(h, residual) = +0.086. The atlas continues to *draw* this rule. Any "forward generation runs cleanly" claim must caveat the parhelic-circle primitive.
- **Persona 1's circumscribed-halo regime call at h ≈ 29° is literature-standard** [Persona 1]. Verified at atoptics.co.uk/blog/tangent-arcs/ ("When the sun climbs above 29º both arcs join into a single halo...the 'circumscribed halo'.") and dewbow.co.uk/haloes/utan1.html. **p7 (h = 59.4°) is a circumscribed-halo display, not an upper-tangent display**; the team is testing the wrong primitive against the wrong literature for that photo. The "broad tangent arc" framing in the v3.8 receipt is a literature-level misclassification.
- **Persona 1's "p27 tangent merges with CZA" critique is sound** [Persona 1, also flagged by Persona 2]. At h = 0.5° the upper tangent arc is at ~22.5° above horizon while the literature CZA is at ~57° above horizon — separated by ~35°, not coincident. The actual failure mode at p27 is sun-bloom/lens-flare contamination of the sun-meridian column, not a feature merger. The "merges with CZA" language should be retracted from the v3.8 receipt.
- **Two of the three "detection-degenerate" tangent calls are protocol artifacts, not feature-level negatives** [Persona 1]. For p2, the upper-tangent apex is locally flat *by tangency definition* (tangent to 22° halo crown at apex), so column-peak on the meridian cannot find a curvature signal; wing-based azimuth-offset sampling would. For p13, the spine is at a chromaticity *transition* (Lab b\* channel), not a brightness peak — the team's own v3.8 receipt notes this. Persona 1's protocol-artifact reading is consistent with literature norms (wing-based opening-angle measurement is standard practice).
- **Supralateral being intrinsically rare is a literature base-rate fact** [Persona 2]. The "coverage gate failure" is structurally near-certain on any curated atlas set of ~9 photos, not specific to this dataset; framing it as "almost made coverage; needs one lucky photo" understates how systemic this is. The route is also a *structurally weak* inverse — supralateral angular distance from sun changes only ~0.5° across h = 0–22°, an order of magnitude less h-discrimination than parhelion offset.
- **Infralateral arcs and the 46° halo are drawn on every rich-vocabulary overlay but visible only on p2** [Persona 3]. The "h → all primitives" forward-richness claim conflates "geometrically parameterized by h" with "visible at predicted locus." Literature classifies infralateral as a *rare* halo (atoptics.co.uk/blog/infralateral-arc/; cloudatlas.wmo.int).

## 3. Unverified / contradicted claims dropped from this memo

- **Persona 3: "no continuous 22° halo arc visible on p22"** — **partial contradiction**. p22 shows a clearly visible 22° halo arc connecting the parhelia (central region is sun-glare-washed but the upper arc is unambiguous). The R22 on p22 is plausibly fittable from the visible upper arc. Persona 3's "4 photos where R22 is not independently measurable" should drop to "3 photos (p20, p25, p26)". The substantive critique survives but is one photo weaker.
- **Persona 3: tally "absent parhelia on p27"** — accurate for the photo but Persona 3's earlier tally line ("clean unambiguous bilateral parhelion peaks in p2, p7, p22, p26, p30") includes p26, which has the geometric-impossibility flag on its right side. p26 should be downgraded from "clean unambiguous" to "tilted, right-offset-anomalous"; left side may still be clean. This is internal-inconsistency in Persona 3's own audit; the structural finding is unaffected.
- **Persona 1: "Tape (*Atmospheric Halos*) and Greenler (*Rainbows, Halos, and Glories*) confirm wing-based measurement"** — Persona 1 explicitly states *"I did not retrieve Greenler or Tape for this memo"* and grounds the claim only in atoptics.co.uk and dewbow.co.uk. The wing-based-measurement claim is therefore web-only-sourced; treat as plausible-but-not-citation-grade until a real specialist confirms.
- **Persona 1: "haloblog.net display catalogues routinely measure tangent-arc opening angle from the wings"** — Persona 1 explicitly does not cite a specific page. Cited as "standard practice in the halo-observation community" but not verifiable from the memo's sources alone. Treat as informed speculation pending citation.
- **Persona 2: precise CZA-altitude table values (e.g., 64.6° at h=18.6°, 57.8° at h=0°)** — the *qualitative* shape of the curve (CZA-above-sun grows as h decreases below 22°, equals 46° at h=22°, vanishes at h=32°) is robustly literature-confirmed. The specific numerical values were not independently re-derived in verification — they should be sanity-checked by a specialist before being cited in any public-framing rewrite. Persona 2's own formula `sin γ = √(n²−cos²α)` is one of several equivalent atmospheric-optics derivations; the canonical Bravais derivation may differ at the third decimal. Treat the *direction* of Persona 2's correction as verified; treat the exact px-collapse number ("residual collapses to ~0.7 px") as illustrative.
- **Persona 3: claim that "Three photos, not seven" pass the residual gate at non-trivial lever** — *substantively correct* but the count of "seven" included p18 and p19 which have no anchor files (Persona 3 correctly flagged this); the comparison should be "three of the nine anchored photos have non-trivial lever and independent R22," which is the same conclusion stated more cleanly.

## 4. Recommended Phase 10 doc changes

1. **Restate the parhelion-offset closeout language**. Replace *"passes residual gate at ~0 px on every eligible photo"* with: *"passes residual gate at ~0 px on three photos (p2, p7, p13) with both unambiguous bilateral peaks and an independently fittable 22° halo. On five additional low-h anchored photos the geometric lever sec(h)−1 is below 2% of R22, and on three of those (p20, p25, p26) the 22° halo arc is not visible in the photograph; on those photos the residual measures offset-vs-R22 against an R22 that is not independent of the parhelion pick. p27 records `offset := R22` explicitly in its anchor file and is a self-consistency check, not a measurement."*
2. **Fix the CZA-route closeout language**. Replace *"fails residual gate (opposite-sign residuals on p2 and p27, route-reliability not stable bias)"* with: *"on inspection, the −19.3 px on p2 is consistent with an h-dependent atlas formula bug at `overlay_calibrate.py:381` (CZA position hardcoded as `sun_y − R46`, correct only at h ≈ 22°). The +21 px on p27 is consistent with primitive-misidentification — the chromatic arc at y=142 is the 46° halo top / supralateral merger, not CZA (literature CZA at h=0.5° is off-frame). Verdict pending: rerun against `sin γ = √(n²−cos²α)` formula and re-classify p27 chromatic arc before claiming CZA-apex inversion is dead."*
3. **Drop p7 from the tangent-arc-route eligibility set entirely**. h = 59.4° is the circumscribed-halo regime per atoptics.co.uk and dewbow.co.uk; testing column-peak detection of an "upper tangent arc" at this h is a literature-level misclassification, not a detection failure. Replace with a separate test of circumscribed-halo ellipticity → h if the team wants an inverse at high h.
4. **Retract the "merges with CZA" framing for p27 tangent**. CZA at h = 0.5° is ~35° higher in altitude than the upper-tangent arc; they cannot merge optically. The actual failure mode is sun-bloom flare contaminating the sun-meridian column.
5. **Add explicit caveats to the forward-generation claim**. The atlas continues to draw `−0.05 · R22` for the parhelic circle (falsified by FF1), the 46° halo and infralateral arcs on every overlay (visible only on p2), and a CZA position that mispredicts by 19+ px at h ≠ 22°. The "h → all primitives runs cleanly forward" language in SUNDOG_V_GRAVITY.md must hedge to *"h → primitive locus (geometric); not every primitive is visible at predicted locus on every photo."*
6. **Flag the p26 right-offset geometric impossibility in the residual table**. The fact that 322 < 323 was accepted with no flag tells you the table is not enforcing the geometric constraint that supposedly defines the route. Add a `geometric_validity` column.
7. **Add `R22-independence` to the anchor-JSON schema**. Each anchor should declare whether `R22` is `ring-fit` (independent), `parhelion-derived` (circular), or `inferred-from-other` (semi-circular). Currently this distinction is buried in per-anchor JSON notes.
8. **Recover the wing-based / chromaticity-based tangent-arc detection before claiming the route is detection-degenerate as a class**. Persona 1's protocol-artifact reading is consistent with literature norms; the receipt should cap at *"column-peak-on-sun-meridian-in-intensity-image fails on all four photos; a literature-standard wing-based or Lab b\*-channel ridge detector has not been built and tested."* This is what the team already says in Open Question #2 but at a stronger hedge than the closeout-level language.

## 5. Recommended Phase 10 protocol fixes (before next round)

1. **Fix `scripts/overlay_calibrate.py:381-384`**. Replace `anchored = WB_SUN[1] - WB_R46` with the literature formula `sun_y − (90° − h − arcsin(√(n² − cos²h))) × (R22/22°)` (n = 1.31). This is the load-bearing smoking gun.
2. **Fix `WB_R46` value**. Currently `WB_R46 = 440 = 2·WB_R22`. Should be `WB_R46 = 460` (= 2.091·WB_R22), or — better — derive `R46_px = r22_obs × (46/22)` from `r22_obs` directly per overlay. The current 4%-low R46 in pixel space adds ~4 px of CZA error on top of the formula bug.
3. **Add a `sec(h)−1` lever column to the parhelion-route residual table**. Mark any row where the lever is below 2% as "anchor-noise-dominated."
4. **Add an `R22-source` column to the residual table**: `ring-fit` / `parhelion-derived` / `inferred-other`. Promote *only* on rows that are `ring-fit`.
5. **Re-test the parhelion route on a sub-set restricted to h > 15°**. If the verdict survives on p2 and p7 alone, it survives the audit. If you need low-h photos to reach two-photo eligibility, you have a coverage problem identical to supralateral.
6. **Rebuild the tangent-arc detector** using Lab b\* channel ridge filtering on a 22°-halo-subtracted residual image, or wing-azimuth-offset radial profiles. If that recovers p2 and p13, the detection-gate "failure" becomes a tooling result.
7. **Re-anchor p27's chromatic arc** as `supralateral / 46° halo top` rather than `CZA apex`. If the re-anchored p27 plus a re-validated p2 give two eligible supralateral photos, the supralateral coverage gate may reopen — though the structurally weak h-sensitivity (~0.5° change across h = 0–22°) still makes the route a poor inverse handle in principle.
8. **Drop p7 from the tangent-arc eligibility set** and either re-classify as circumscribed-halo or out-of-window.

## 6. What this means for the public-framing claim

The current SUNDOG_V_GRAVITY.md public-framing sentence — *"the atlas is rich in forward generation (`h → all primitives`) and supports one image-recoverable inverse handle (parhelion offset)"* — is **hedge-required, leaning retract-required for the forward-richness half** until §4 doc fixes land.

- *"One image-recoverable inverse handle"*: defensible on three photos (p2, p7, p13). **Not defensible on the language "every eligible photo."** The hedged version — *"one image-recoverable inverse handle on photos where sun altitude provides a non-trivial geometric lever (sec(h)−1 ≳ 2%) and an independent 22° halo ring is fittable"* — survives the audit and is harder to attack than the current claim.
- *"Rich in forward generation"*: **retract-required for the parhelic circle** (FF1 falsified the rule the atlas draws) and **for the CZA position** (the atlas mispredicts by 19+ px at h ≠ 22° because of the hardcoded formula bug). Sound for parhelion offset, 22° halo radius, the CZA *visibility window* (h < 32.2°), and the upper-tangent *geometric* spine (with the caveat that brightness/chromatic spine ≠ geometric spine).

The "three independent failure layers" framing in the Mesa crossover note also needs revision: of the three failures, one (CZA residual) is contaminated by an atlas formula bug, one (supralateral coverage) is a literature base-rate near-certainty rather than an accident of this dataset, and one (tangent detection) is a protocol artifact for at least two of the four photos under any literature-standard detector. The claim *"three independent accidents"* implies three independent draws against the same underlying mechanism; in fact at least two of the three are structurally guaranteed by independent causes, not parallel evidence of an atlas defect.

A defensible compressed framing: *"On the current calibration set the parhelion-offset inversion is the only inverse route with both independent-anchor evidence and a non-trivial geometric lever; the other three routes fail at one of three structurally different layers (residual / coverage / detection), each of which has a separate non-atlas explanation. The 'one inverse handle' result is genuine but conditional on the photo set's altitude distribution and on the column-peak detector for the tangent route."*

## 7. Disagreements between personas

The three personas agree on the headline (atlas is over-claiming on at least one axis). They disagree at the level of *which* failure layer is the worst:

- **CZA-apex failure mode.** Persona 2 argues the failure is mostly atlas-formula bug + primitive-misID on p27, and that fixing the formula would let CZA-apex pass the residual gate on p2 alone. Persona 1 mentions "CZA-merge framing is wrong" but does not engage with the formula bug. Persona 3 takes the residual gate at face value and treats CZA-apex as failed without diagnosing the formula. **The disagreement that matters most is between Persona 2 (formula bug, fixable) and the current team verdict (route-reliability, structural).**
- **p27 chromatic arc identity.** Persona 1 says the team's "CZA / sun-bloom merge" label is wrong because the upper tangent and CZA are at completely different altitudes (a *primitive-altitude* argument). Persona 2 agrees the team's CZA label on p27 is wrong but argues the visible arc is *supralateral / 46° halo top* (a *primitive-identity* argument). These are mutually consistent — both personas agree the team's primitive ID on p27 is broken — but they disagree on what the right re-classification is. **A real specialist will likely have a third reading; this is the highest-uncertainty disagreement.**
- **Tangent-arc route status.** Persona 1 thinks 2 of 4 detection-degenerate calls are protocol artifacts (p2-wings recoverable, p13-chromaticity recoverable), 1 is out-of-window (p7 → circumscribed-halo), 1 is sun-bloom (p27). **Net: the route is recoverable on the current photo set under a literature-standard protocol.** Persona 3 does not address tangent-arc directly. The team's current verdict is "detection-degenerate across all 4 photos, class-level negative." **The disagreement is whether to treat the detection-gate finding as a class-level negative or a tooling-conditional negative.**
- **Parhelion-route survival rate.** Persona 3 says "three photos pass, six are conditional/circular." Personas 1 and 2 do not directly audit the parhelion route, but Persona 2 implicitly accepts the parhelion-route promotion. **There is no cross-persona corroboration of Persona 3's six-out-of-nine finding** — it is single-sourced. The verification *did* substantiate the arithmetic, anchor-JSON evidence, and image evidence for the substantive claim, but the specific count is one persona's, and a real specialist could land elsewhere on which photos cross the lever threshold.
- **Supralateral route.** Persona 2 says supralateral is intrinsically rare (literature base rate) and the route has structurally weak h-discrimination (~0.5° across h = 0–22°) even *with* coverage. Persona 1 and Persona 3 do not engage with supralateral. **No cross-persona disagreement, but also no cross-persona corroboration.**
- **Forward-generation richness.** Persona 3 identifies eight forward primitives, ranks them, and flags over-claim on 46° halo, CZA position, tangent position, supralateral, parhelic circle, infralateral. Persona 2 flags over-claim specifically on CZA position. Persona 1 flags over-claim specifically on tangent position (geometric spine ≠ brightness spine). **The three personas converge — independently — on "forward richness is conflating geometry-parameterization with photo-visibility," which is the highest-confidence consensus point in the audit.**

## 8. Confidence on next-step: real-specialist outreach

**Fix the smoking-gun issues and re-audit before sending the existing handoff to a real specialist.** Confidence: high. The CZA formula bug at `overlay_calibrate.py:381-384` is a verified, one-edit fix; sending the current handoff to a specialist before the fix is filed risks the specialist landing on Persona 2's finding within the first 15 minutes and the audit reading as "team did not check their own forward model before claiming the inverse fails." That undermines the rest of the credibility load on the parhelion-route promotion. The other items — p27 re-classification, R22-independence column, sec(h)−1 lever column, tangent-detector rebuild — are independently scoped but the formula fix is the one item that, if a specialist finds it before the team does, materially damages confidence in the rest of the work. Re-audit after the §5 protocol fixes #1, #2, #3, #4 land; the rest can ship as Open Questions for the specialist to weigh in on. The single-handle verdict will likely survive a re-audit but with the hedged language from §4 and §6 instead of the current closeout-level language.

---

## Appendix A — Persona 1 draft (Tangent-arc)

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

---

## Appendix B — Persona 2 draft (CZA + Supralateral)

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

---

## Appendix C — Persona 3 draft (Parhelion + Forward)

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
