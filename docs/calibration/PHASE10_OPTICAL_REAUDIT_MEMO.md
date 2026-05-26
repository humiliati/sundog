# Phase 10 Optical Re-Audit Memo

Filed: 2026-05-14  
Scope: post-pass state after the Phase 10 attack campaign required passes
(`§6` hedges, handoff stop banner, B1, A1a, A1b, A2, A3, C1, B2). C2
remains optional and unrun.

## Verdict

The re-audit gate clears the technical-pass wave. No new code, anchor, or
route-math blocker was found after the post-pass state was checked against
the synthetic three-persona protocol.

This is a **clearance to rewrite**, not a clearance to send the old
specialist handoff unchanged. The stale handoff and public-framing surfaces
must be ratcheted to the post-pass failure taxonomy before external use.

## Persona Re-Audit

### Persona 1 - Tangent / Primitive Taxonomy

Pass C1 fixed the load-bearing taxonomy error: p7 is no longer counted as
upper-tangent-route evidence because h = 59.4 deg is in the
circumscribed-halo regime. The post-C1 sampled tangent set is p2, p13, and
p27.

The remaining tangent verdict is bounded: the column-peak detector fails on
the post-C1 sampled set, but this is a protocol-conditional negative. With
C2 unrun *(at original filing — see Post-C2, Post-C4, Post-C5, and
Post-C6 Addenda below; all four passes landed later the same day; C2
/ C4 / C6 returned not-recovered; C5 manual sample selection
recovered the route on p2 with R\_uta\_obs / R22 = 0.824 and RMS =
1.23 px but C6 matched-filter on the same b\* substrate falsified the
C5→matched-filter natural extension, putting the route in C5↔C6
substrate tension that recommends specialist re-anchoring as the
verify gate)*, the correct handoff language was "unresolved open
question" for a wing-based or Lab b* ridge detector, not a
class-level tangent failure.

Finding: no new tangent-route blocker. The only remaining risk is stale
surface language that still says "all four eligible photos" or implies a
generic tangent-route failure.

### Persona 2 - CZA / Supralateral Route Semantics

Pass A1a/A1b repaired the CZA atlas formula and verified the repair. On p2,
the CZA apex y residual collapses from the legacy -19 px class of error to
about +1.3 px in the current sign convention. Pass A2 removes p27 from CZA
evidence by reclassifying its visible chromatic feature as 46 deg halo top /
supralateral merger.

The CZA route now fails coverage, not residual reliability: p2 is the only
in-window measured CZA anchor; the other anchored photos are either beyond
the CZA disappearance altitude or have the literature CZA apex off-frame.

The supralateral route remains blocked even if coverage is read
permissively. Its h-sensitivity across the relevant low-altitude range is
below visual-edge measurement noise, so the route is structurally weak as an
inverse handle.

Finding: no new CZA or supralateral blocker. The corrected verdict is
dataset/aspect-ratio coverage for CZA and physics-discrimination for
supralateral.

### Persona 3 - Parhelion Route / Public Claim

Pass B1 and B2 fixed the parhelion overclaim without demoting the route.
The promoted wording is now restricted to p2, p7, and p13: photos with
unambiguous bilateral peaks, valid geometry, non-trivial discrimination, and
an independently fittable 22 deg halo.

The low-h photos remain informational, not promotional. p20, p25, and p26
have parhelion-derived R22 values; p27 stipulates offset := R22; p26 right
side is explicitly invalid because R22 / offset > 1.

Finding: parhelion-offset remains the only promoted inverse handle. The old
"every eligible photo" phrase is retired and should appear only in
retraction or historical-audit contexts.

## Consolidator Verification

- Ran `python scripts\test_cza_formula.py`. The test prints h = 22
  literature CZA above sun = 45.734 deg, CZA disappearance altitude =
  32.196 deg, p2 literature residual = -1.3 px observed-minus-predicted
  (equivalent to +1.3 px predicted-minus-observed in overlay notes), and p27
  literature CZA y = -11.5, confirming the visible p27 feature is not CZA.
- Ran `python scripts\overlay_calibrate.py docs\calibration\2.Photometeor-jeff_mod_red.jpg --anchors docs\calibration\p2-anchor.json --sun-altitude 18.6 --supralateral 0.40 --out $env:TEMP\p2_reaudit_overlay.png`.
  The script reports CZA apex predicted at `(567.0, 114.3)`, residual
  `(dx, dy) = (+7.0, +1.3) px`, and parhelion residuals near zero.
- Checked the post-pass route tables in
  `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`: A3, C1, and B2 are
  present and agree on the current taxonomy.
- Checked `docs/SUNDOG_V_GEOMETRY.md`: the closeout table already uses the
  post-pass route verdicts.
- Checked p26 schema semantics from the B1/B2 table: the right-side
  parhelion remains invalid because `323 / 322 > 1`.

Verification gate result: all load-bearing numeric and schema checks agree
with the post-pass route taxonomy. No persona overstatement survived the
consolidator pass.

## Post-Re-Audit Failure Taxonomy

| route | gate | failure-mode kind |
| --- | --- | --- |
| Parhelion offset -> h | promoted, post-hedged | 3-photo strict eligibility: p2, p7, p13 |
| CZA apex -> h | fails coverage | dataset/aspect-ratio: only p2 is in-window and measured |
| Supralateral -> h | fails structural discrimination | physics-shape: h-spread below visual-edge noise |
| Tangent-arc curvature -> h | partially recovered on p2 under manual sample selection; coverage gate fails; C5<->C6 substrate tension flagged *(see Post-C2 + Post-C4 + Post-C5 + Post-C6 Addenda)* | hybrid coverage + tooling with verify-gate: 4 automated detectors fail (C2/C4/C6 plus column-peak); C5 manual recovers on p2 with methodology hedge; p13/p27 substrate-signal absent; recommended specialist re-anchoring of p2 |

The pre-audit "three independent failure layers" phrase stays retired. The
post-audit phrasing is "three structurally different failure modes":
dataset/aspect-ratio, physics-discrimination, and tooling-protocol.

## Required Next Step

Proceed to the specialist handoff rewrite and public-framing ratchet:

1. Rewrite `docs/calibration/PHASE10_OPTICAL_AUDIT_HANDOFF.md` from the
   post-pass taxonomy; keep C2 as an unresolved open question.
2. Ratchet `docs/MESA_CROSSOVER_NOTE.md` and `docs/SUNDOG_V_GRAVITY.md` out
   of pre-audit hedge mode and replace old residual/failure tables.
3. Keep `index.html` and `chat/claim_map.json` conservative until the public
   claim copy is explicitly approved against this memo.
4. Check `docs/promo/PROMO_HIGHLIGHTS.md` for stale "all four eligible" tangent
   wording before any external reuse.

## Post-C2 Addendum *(2026-05-14)*

The original memo above was filed when Pass C2 was unrun and the tangent
detector was filed as an Unresolved Open Question. C2 has now landed in
the same wave (`scripts/tangent_detector.py` +
`scripts/test_tangent_detector.py`; captured run output at
`docs/calibration/PASS_C2_DETECTOR_OUTPUT.txt`; full receipt in
`docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md` under "### Pass C2
Update"). This addendum records the disposition change for the tangent
row of the verdict table above; the rest of the memo stands.

**Pass C2 result.** A wing-azimuth-offset Lab b\* ridge detector with
22°-halo-radial-profile subtraction (the literature-standard substrate
recommended by `PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md` §4.8 /
Persona 1 §4 item 6) ran on the post-C1 sampled set (p2 / p13 / p27).
Pre-registered coherence gate: a wing is ridge-detected if ≥ 8 / 24
samples have residual b\* amplitude ≥ 3.0 AND radial offset ≤ ±10 px
from the predicted tangent locus. Result on all three photos: 0-2 / ~24
coherent samples per wing, well below the gate. Verdict:
**not-recovered**.

**Methodology hedge that did not save the route.** The pilot run
*without* the halo-radial subtraction step exhibited boundary-pinned
median radial offsets at ±18 px, consistent with the detector finding
the 22° halo's own chromatic ridge inside the search window. Adding
the halo-radial subtraction step un-pinned the offsets (post-subtraction
medians: −7.5 to +10.5 px across the six wings, well inside the ±18 px
window) and the negative survived. The failure cannot be attributed to
halo-ridge contamination of the detection window.

**Taxonomy update.** The "tooling-protocol" classification of the
tangent-route failure survives but narrows: the route fails under
*two* literature-standard detector families (column-peak intensity at
sun meridian; wing-radial Lab b\* with halo-radial subtraction). The
residual open question is whether non-literature-standard detector
designs (wing-slope geometric curvature, matched-filter against a
parameterized arc model, polarization-channel filtering) might recover
the route. These are filed as Phase 10 backlog ideas, not Phase 10
deliverables. The "three structurally different failure modes" framing
otherwise survives unchanged.

**Handoff implication.** `PHASE10_OPTICAL_AUDIT_HANDOFF.md` §2.3 (the
specialist question "is a wing-based / Lab b\* detector worth
building?") was already answered by the team — the detector has been
built and ran negative. The specialist question reframes to: "are the
non-literature-standard detector designs (wing-slope curvature,
matched-filter, polarization) worth building before the route is
called dead?" The handoff was updated in the same wave to carry the
post-C2 verdict; this addendum is the load-bearing pointer for any
reader following the cross-reference back from the handoff to this
memo.

**Single-handle verdict unchanged.** Parhelion-offset remains the sole
promoted inverse handle. The forward-rich / inverse-narrow asymmetric
field-shape pattern survives at the same one-handle resolution; C2
ruled out a *literature-standard* path to two-handle promotion but did
not rule out two-handle promotion via a different detector substrate.

## Post-C4 Addendum *(2026-05-14)*

The Post-C2 Addendum above narrowed the tangent-route open question to
"non-literature-standard detector designs (wing-slope curvature,
matched-filter, polarization)." Pass C4 lands the first of those —
the wing-slope geometric curvature detector Persona 1 §5 explicitly
named — and tests it on the same eligibility set. Implementation:
`scripts/tangent_curvature.py` (regression runner
`scripts/test_tangent_curvature.py`; captured run output at
`docs/calibration/PASS_C4_DETECTOR_OUTPUT.txt`; full receipt in "###
Pass C4 Update" of `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`).

**Pass C4 result.** A wing-slope luminance-gradient edge detector with
circle fit ran on p2 / p13 / p27. Pre-registered gates: gradient
magnitude ≥ 1.5 L\*/px AND radial offset ≤ ±12 px from predicted
tangent locus AND ≥ 12 surviving candidates AND circle-fit RMS ≤ 8 px
AND fitted R\_uta / R22 ∈ [0.7, 1.3]. Result on every photo: 88-100%
of wing samples rejected as weak-gradient, leaving 0-4 surviving
candidates — too few for any circle fit. **Not-recovered** on all
three photos.

**Reclassification of Persona 1's §5 alternative.** The Post-C2
addendum classified wing-slope geometric curvature as a
"non-literature-standard detector design." Reading Persona 1 §5 more
carefully, this is wrong: Persona 1 explicitly named gradient-based
edge detection as the literature-standard alternative to peak-based
ridge detection (*"real measurement needs either gradient-based edge
detection (the spine is at a brightness or chromaticity transition,
not a peak) or manual sample selection from visual crops"*). C4
tested the gradient-based edge detection branch. The route now fails
under **three** literature-standard detector families.

**Taxonomy update.** The "tooling-protocol" classification of the
tangent-route failure survives but narrows further. The remaining
candidate detector designs / substrates that have NOT been tested:

- **Manual sample selection from visual crops** (Persona 1 §5's
  literature-standard alternative #2). Hand-anchoring rather than
  automated detection. `tangent_curvature.fit_circle` already accepts
  arbitrary (x, y) point lists, so if the team hand-anchored arc
  points on each photo, the existing circle-fit machinery could run.
- **Matched-filter detection against a parameterized arc model.** A
  different signal (whole-shape correlation) than the per-sample edge
  or ridge picks tested so far. Audit memo doesn't classify this as
  literature-standard or not.
- **Polarization-channel filtering.** Different *substrate* — requires
  polarizer-equipped follow-up photos. Substrate test, not detector
  test on the same substrate.
- **New calibration photos with stronger tangent display.** Coverage
  expansion rather than detector design. The wing-region L\*-gradient
  rejection rate (88-100%) suggests the calibration set may genuinely
  not carry enough tangent signal for any radial-profile detector at
  the pre-registered amplitude.

**Handoff implication.** `PHASE10_OPTICAL_AUDIT_HANDOFF.md` §2.3 was
updated in the same wave to reflect the post-C4 state: the specialist
question reframes from "is wing-slope curvature worth building?" (now
answered: built and ran negative) to "is manual hand-anchoring,
matched-filter detection, polarization filtering, or new calibration
photos the right next step?" The §2.3 prose carries the four options.

**Single-handle verdict unchanged.** Parhelion-offset remains the sole
promoted inverse handle. The "forward-rich / inverse-narrow" framing
stays at the same one-handle resolution; C4 ruled out the second
literature-standard alternative for two-handle promotion but did not
rule out the remaining four candidates.

## Post-C5 Addendum *(2026-05-14)*

The Post-C4 Addendum above narrowed the tangent-route open question to
"manual sample selection from visual crops" (Persona 1 §5's other
literature-standard alternative), matched-filter detection, polarization
filtering, and new calibration photos. Pass C5 lands the first of those.

**Pass C5 result.** Manual hand-anchored upper-tangent points were
added to `upper_tangent_manual_samples` blocks of each post-C1
eligibility-photo anchor JSON. The runner at
`scripts/test_tangent_manual.py` applies the C4 circle-fit machinery
to the manual points under pre-registered gates: ≥ 5 manual points,
circle-fit RMS ≤ 10 px, R\_uta\_obs / R22 ∈ [0.7, 1.3]. Captured run
output: `docs/calibration/PASS_C5_DETECTOR_OUTPUT.txt`. Per-photo
result:

- **p2: route recovered.** 5 hand-anchored points fit a circle with
  R\_uta\_obs / R22 = **0.824** (within tolerance) and RMS =
  **1.23 px** (well under 10 px). Center fit: (567.5, 160.3) —
  essentially on the sun's vertical axis (sun\_x = 567). The upper
  tangent arc is geometrically real and circular on the cleanest
  display.
- **p13: not recovered.** Only the apex is marginally anchorable
  (faint chromatic brightening at the halo crown); the wing region
  is washed into a luminous chromatic gradient with no localizable
  spine (consistent with Persona 1 §3 p13 entry). Insufficient
  samples for a circle fit.
- **p27: not recovered.** Zero anchorable points. The sun-bloom
  column dominates the halo-crown region where the upper tangent
  arc would sit (consistent with Persona 1 §3 p27 entry); no
  upper-tangent feature is separable from the sun-bloom. The
  chromatic arc visible at the top of the frame is the
  supralateral / 46° halo top merger (per Pass A2 reclassification),
  not upper tangent.

**Coverage gate fails** at 1 / 3 photos < pre-registered ≥ 2 threshold.

**Reclassification of the failure mode.** The Post-C4 Addendum
classified the tangent-route failure as "detection-protocol tooling"
under three literature-standard detectors. C5 refines this to a
**hybrid coverage + detection-tooling failure**:

- On p2, the signal is real and recoverable; the three automated
  detectors (C1 column-peak, C2 wing-radial Lab b\*, C4 wing-slope
  luminance-gradient) all miss what manual hand-anchoring picks up.
  This is *detection-tooling* in a specific sense: the arc's image
  signature is at the resolution of visual chromatic-dome
  identification, not at the resolution of per-sample radial-profile
  peak picks. A matched-filter detector that tests the whole-arc-shape
  hypothesis (rather than per-sample features) should be able to
  recover what C2 / C4 miss; C5's positive result on p2 is the
  strongest evidence for matched-filter being the right next detector.
- On p13 / p27, no detector — automated or manual — recovers the
  signal. This is *substrate-signal absence*, not a detector design
  problem. Better tooling does not help; new substrate (new photos
  or polarization filtering) does.

The "three structurally different failure modes" framing from the
post-A3 + post-C1 taxonomy survives but the tangent classification
sharpens:

| route | failure mode kind |
| --- | --- |
| Parhelion offset -> h | promoted on strict 3-photo subset |
| CZA apex -> h | dataset / aspect-ratio coverage |
| Supralateral -> h | atmospheric-physics discrimination |
| Tangent-arc curvature -> h | hybrid coverage (p13/p27 substrate absence) + detection-tooling (p2 signal present but missed by 3 automated detectors) |

**Methodology hedge.** The p2 manual anchoring was performed by visual
identification at display resolution by the project assistant, with
per-point uncertainty estimated at ±15 px. The RMS = 1.23 px fit
result is suspiciously tight given that uncertainty and could
partially reflect symmetry bias. The points are recorded in
`p2-anchor.json` `upper_tangent_manual_samples.points`; a specialist
re-anchoring is a one-script operation against the existing JSON.
**The C5 verdict should be treated as preliminary pending specialist
re-anchoring** before any public-framing surface relies on the p2
recovery as a confirmed positive.

**Handoff implication.** The handoff `PHASE10_OPTICAL_AUDIT_HANDOFF.md`
§2.3 was updated in the same wave: the specialist question reframes
from "is manual hand-anchoring worth doing?" (now answered — done,
positive on p2) to "is matched-filter detection the right next
literature-standard step?" plus a request for specialist re-anchoring
of p2 as a verify gate.

**Single-handle verdict unchanged.** Parhelion-offset remains the sole
*promoted* inverse handle. The tangent route's verdict shifts from
"fails under three literature-standard detectors" to "partially
recovered on p2 under manual sample selection; coverage gate fails."
The forward-rich / inverse-narrow framing stays at the same one-handle
resolution, but the tangent route now has a published positive result
on one photo — a constructive update rather than a deeper negative.

## Post-C6 Addendum *(2026-05-14)*

The Post-C5 Addendum above named matched-filter detection as the
highest-leverage next candidate ("now MORE promising given p2's
circular fit"). Pass C6 built the matched-filter detector against a
parameterized arc model on the same halo-subtracted Lab b\* substrate
C2 used, ran it on the post-C1 sampled set, and **falsified the
C5→matched-filter natural-extension hypothesis**.

**Pass C6 result.** Detector at `scripts/tangent_matched_filter.py`
(regression runner `scripts/test_tangent_matched_filter.py`; captured
output at `docs/calibration/PASS_C6_DETECTOR_OUTPUT.txt`; full
receipt under "### Pass C6 Update" in
`docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`). Pre-registered
gates: R\_uta\_obs / R22 in [0.7, 1.3]; peak correlation ≥ 0.10; peak
prominence ≥ 1.5× baseline. Per-photo result:

- **p2: not recovered.** The matched-filter correlation is **negative
  across the entire R\_uta scan** (from −0.244 at R\_uta = 91 px up
  to −0.117 at R\_uta = 267 px). There is no positive correlation at
  any R\_uta in [0.5 × R22, 1.5 × R22], including at R\_uta = 150
  consistent with C5's manual fit (R\_uta\_obs = 149.9). The template
  vs. b\*-residual signal-coupling reverses sign over the wing
  region.
- **p13: not recovered.** Positive correlation peak (+0.191) with
  2.05× prominence — but at R\_uta = 106 px = 0.502 × R22, way below
  the [0.7, 1.3] tolerance. Matched-filter is picking up some
  chromatic feature at half the predicted curvature, not the upper
  tangent arc.
- **p27: not recovered.** Score essentially zero (−0.001) at all
  R\_uta. Sun-bloom column dominates the upper region; no coherent
  chromatic structure correlates with the predicted arc template.

**The C5↔C6 contradiction is the load-bearing finding.** C5 manual
sample selection recovered p2 with R\_uta = 149.9 (ratio 0.824) and
RMS = 1.23 px on the same photo where C6 matched-filter returns
correlation = −0.21 at the same R\_uta. Both detectors operate on the
same image and (effectively) the same substrate (C6's halo-subtracted
b\*; C5's visual identification picks up b\* signal among other
things). Yet they reach opposite conclusions.

**Two open interpretations:**

(a) **Substrate mismatch.** C5's manual identification picked up the
arc by *visual gestalt* — combining luminance, chromaticity, spatial
coherence, and likely some prior expectation of arc shape. C6
correlates only halo-subtracted Lab b\*. The arc's gestalt signal may
not be in halo-subtracted b\* alone — could be in absolute b\* (no
subtraction), in L\* magnitude, in chromaticity vector magnitude
(`sqrt(a*² + b*²)`), or in some combined channel. C5 positive + C6
negative narrows the substrate space the route's signal lives in; it
does NOT rule out automated detection on a different substrate.

(b) **Symmetry bias in C5.** The C5 receipt explicitly hedged:
*"RMS = 1.23 px is suspiciously tight given ±15 px per-point
uncertainty, possibly reflecting symmetry bias."* If the manual
hand-anchoring unconsciously favored points consistent with circular
shape, C5's clean fit would be partly artifact. C6's null result on
the same substrate is consistent with this reading — the
matched-filter doesn't share the symmetry bias and reports honestly.

Both interpretations are testable. (a) tests by running matched-filter
on alternative substrates (absolute b\*, L\* magnitude, chromaticity
magnitude). (b) tests by specialist re-anchoring. Both are filed as
**Phase 10 backlog**, neither is a Phase 10 deliverable.

**Taxonomy update.** The "hybrid coverage + detection-tooling"
classification from Post-C5 stands, but the detection-tooling
sub-claim now carries a verify-gate flag:

> *Four automated detectors (C2 wing-radial Lab b\*; C4 wing-slope
> luminance-gradient; C6 matched-filter on halo-subtracted b\*; plus
> the original column-peak) all miss p2's signal. Manual sample
> selection (C5) recovers on p2 but with a methodology hedge that C6
> now amplifies into a substrate-tension verify-gate.*

**Handoff implication.** The handoff `PHASE10_OPTICAL_AUDIT_HANDOFF.md`
§2.3 was updated in the same wave: the specialist question now
includes a request for independent re-anchoring of p2 as the verify
gate, in addition to matched-filter-on-alternative-substrates,
polarization filtering, and new calibration photos.

**Single-handle verdict unchanged.** Parhelion-offset remains the
sole *promoted* inverse handle. The tangent route's "partially
recovered on p2" status holds but now carries the C5↔C6 substrate
tension and a recommended verify-gate before further propagation
into public-framing surfaces. The forward-rich / inverse-narrow
framing stays at one-handle resolution.

