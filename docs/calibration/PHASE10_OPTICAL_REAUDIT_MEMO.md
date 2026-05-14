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
C2 unrun *(at original filing — see Post-C2 Addendum below; C2 landed
later the same day and returned not-recovered)*, the correct handoff
language was "unresolved open question" for a wing-based or Lab b* ridge
detector, not a class-level tangent failure.

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
| Tangent-arc curvature -> h | detection gate under two literature-standard detectors *(see Post-C2 Addendum)* | tooling-shape: C2 detector built and ran; not-recovered |

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
4. Check `docs/PROMO_HIGHLIGHTS.md` for stale "all four eligible" tangent
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

