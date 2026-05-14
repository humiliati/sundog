# Phase 10 Attack Roadmap

> Consolidated post-audit attack plan for the Geometry Phase 10 single-handle
> verdict. The synthetic optical audit
> ([`docs/calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md`](calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md))
> returned §3 outcome (3) — *parhelion-offset reinterpreted as
> photo-set-lucky / fit-driven* — plus partial hits on outcomes (2) and (4).
> Three load-bearing findings survived persona disagreement and the
> consolidator's verification gate. This roadmap turns those findings into
> a sequenced campaign of attack passes, each with entry/exit criteria, so
> the work can be re-audited cleanly before any real-specialist outreach.
>
> **Status as of filing (2026-05-13):** the Phase 10 single-handle verdict
> in [`docs/SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md#phase-10---rich-display-overlay-tuning)
> is *not falsified* but its evidentiary chain is *substantially weaker than
> the closeout claims*. Public-framing surfaces leaning on the verdict
> (`SUNDOG_V_GRAVITY.md`, the homepage `#elevator-pitch` section,
> `MESA_CROSSOVER_NOTE.md`) are hedge-required pending re-audit.

## 1. The Three Load-Bearing Findings

The synthetic memo's §2 lists fourteen verified items; three of them carry
the campaign and gate everything downstream. The full list is the
authoritative source — what follows is the campaign-level summary.

### Finding A — Atlas formula bug at `scripts/overlay_calibrate.py:381–384`

CZA apex is hardcoded as `sun_y − R46`. That expression is geometrically
correct only at h ≈ 22°. At p2 (h = 18.6°) the literature CZA position
sits ~19 px above the hardcoded prediction — which exactly matches the
−19.3 px residual the team filed as a *route-reliability* failure in the
Phase 10 closeout. Compounding bug: `WB_R46 = 440` versus the literature
ratio of 2.091 · `WB_R22` ≈ 460, so even at h = 22° the atlas's R46 in
pixel space is ~4 % too small.

**Net implication:** the CZA-route "failure" is partly atlas bug, not
route failure. The single-handle verdict rests on a *three-route
failure* count that includes a route the atlas is not testing fairly.
([Memo §2 items 1, 2, 3](calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md#2-verified-findings-that-should-change-the-phase-10-verdict))

### Finding B — Parhelion-offset has a circular dependency on six of nine anchored photos

The inversion `h = arccos(R22 / offset)` has discrimination
`sec(h) − 1 = 0.71 %` at h = 6.83° and 5.52 % at h = 18.6°. On the five
low-h photos (p13, p20, p22, p25, p26) the geometric lever is at or
below typical anchor noise; on three of those (p20, p25, p26) the 22°
halo arc is not visible in the photograph at all, so R22 is not from a
ring fit. p27 admits the circularity in its own anchor JSON
(`offset_L = offset_R = R22 = 219` with `_meta.note` flagging it).
p26's right side encodes a literal geometric impossibility
(`R22 / offset = 323 / 322 > 1`, `arccos` undefined) that the residual
table accepted with no flag.

**Net implication:** the closeout's *"passes residual gate at ~0 px on
every eligible photo"* restates honestly as *"passes on three photos
(p2, p7, p13) where the test has meaningful discrimination and an
independently fittable 22° halo."* The verdict survives, but the
hedged version is much harder to attack.
([Memo §2 items 4–7](calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md#2-verified-findings-that-should-change-the-phase-10-verdict))

### Finding C — Tangent-arc detection-gate is misclassified for at least one photo

p7 is at h = 59.4°. Per `atoptics.co.uk/blog/tangent-arcs/` and
`dewbow.co.uk/haloes/utan1.html`, above h ≈ 29° the upper and lower
tangent arcs join into the *circumscribed halo*. Testing column-peak
detection of an "upper tangent arc" at h = 59.4° is a literature-level
primitive misclassification, not a detection failure. Two of the other
three "detection-degenerate" calls (p2, p13) are protocol artifacts
under any literature-standard wing-based or Lab b\*-channel detector,
not feature-level negatives.

**Net implication:** the tangent-arc *route* may be recoverable on the
current photo set under a literature-standard detector. The current
column-peak result is a tooling-conditional negative, not a class-level
negative, and at least one photo (p7) does not belong in the tangent
eligibility set at all.
([Memo §2 items 9, 10, 11](calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md#2-verified-findings-that-should-change-the-phase-10-verdict))

## 2. Cross-Cutting Downstream Impact

The Phase 10 single-handle verdict has been ratcheted into multiple
public-framing surfaces that now over-state what the audit will
support. Each surface needs an explicit hedge or retraction in lockstep
with the §3 attack passes.

| Surface | Current language | Required treatment |
| --- | --- | --- |
| [`docs/SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md#phase-10---rich-display-overlay-tuning) Phase 10 closeout | "passes residual gate at ~0 px on every eligible photo"; three-route failure by residual / coverage / detection | Replace per memo §4.1–4.3; add the audit memo as a substantiation pointer; flag the closeout as "post-audit hedged 2026-05-13" |
| [`docs/MESA_CROSSOVER_NOTE.md`](MESA_CROSSOVER_NOTE.md#geometry-phase-10-closeout-2026-05-13) Phase 10 closeout subsection | "three independent failure layers"; forward-rich/inverse-narrow asymmetry | Walk back the "three independent" framing — the audit shows two of the three failures have non-atlas explanations; preserve the field-shape pattern claim with hedged wording per memo §6 |
| [`docs/SUNDOG_V_GRAVITY.md`](SUNDOG_V_GRAVITY.md) public-framing sentence | "rich in forward generation … one image-recoverable inverse handle" | Hedge-required; lean retract-required for the *forward-richness* half until the parhelic-circle rule, the CZA position bug, and the infralateral / 46° halo visibility caveats land. Memo §6 has the defensible compressed framing. |
| `index.html#elevator-pitch` (homepage v1 pitch) | "first empirical hint that field-not-reward … is describing a real category of object" | Per [`SUNDOG_V_CHAT.md` §16.2](SUNDOG_V_CHAT.md#162-known-integrity-gap-v1-pitch-2026-05-13), this is now logged as audit-driven retraction (not "decide whether to ratchet a route"). Pitch text needs a hedge or paragraph pull until the re-audit lands. |
| `chat/claim_map.json` routes `mesa_roadmap_status`, `framework_pattern`, `unsupported_alignment_overclaim` | Currently treats mesa subspace and field-not-reward as roadmap / unsupported | Do not ratchet. The audit weakens — not strengthens — the case for promoting these routes. Hold at current tier until re-audit. |

## 3. Pass-by-Pass Attack Plan

Each finding gets a sequence of named passes. Passes within a finding are
ordered; passes across findings are coordinated via the §4 sequencing
diagram (B1 is the first technical gate; B2 is last; C2 is optional).

### 3.0 Recommended Execution Order

This is the order to actually run the work in. Pass-detail blocks below
are organized by finding for navigation, but the sequencing here is
authoritative.

1. **Public/doc hedge + handoff stop banner.** **Landed.** Run §6 steps 1–5 in
   parallel with the `PHASE10_OPTICAL_AUDIT_HANDOFF.md` stop banner.
   No technical pass starts until these land.
2. **Pass B1.** **Landed.** Add `r22_source`, `sec(h) − 1`, and
   `geometric_validity` columns to the residual table; flag p26 right
   `invalid`; backfill `r22_source` on every anchor JSON. First
   technical gate.
3. **Pass A1a → A1b.** **Landed.** Write the CZA formula spec + literature
   regression test (A1a); only if A1a confirms the memo's qualitative
   finding, patch `overlay_calibrate.py` (A1b). The patch is gated on
   the test, not on the memo.
4. **Pass A2 → A3.** **Landed.** Re-anchor p27's chromatic arc as supralateral /
   46° halo top (A2); re-derive the CZA-route verdict against the
   fixed atlas and the re-classified p27 (A3).
5. **Pass C1.** **Landed 2026-05-14.** Drop p7 from tangent-arc eligibility per the
   circumscribed-halo regime literature.
6. **Pass C2 → C3.** **Landed 2026-05-14.** Built the wing-azimuth-offset
   Lab b\* ridge detector with 22°-halo-radial-profile subtraction
   (`scripts/tangent_detector.py`, regression runner
   `scripts/test_tangent_detector.py`) and ran it on the post-C1 sampled
   set (p2 / p13 / p27): route **not-recovered** on all three under a
   pre-registered 8 / 24 coherent-sample gate. C3 verdict edit landed
   in same wave: tangent route fails detection under **two**
   literature-standard detector families; tooling-conditional framing
   narrows to non-literature-standard designs (wing-slope curvature,
   matched-filter, polarization), filed as Phase 10 backlog.
7. **Pass B2.** **Landed 2026-05-14.** Re-derive the parhelion verdict against the
   now-finalized eligibility set. Runs last of the technical passes
   so the verdict reads from the post-A3 / post-C3 table.
8. **Re-audit gate (§5).** **Landed 2026-05-14** in
   [`docs/calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md`](calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md).
   The technical-pass wave clears; the gate authorizes the specialist
   handoff rewrite and public-framing ratchet, not reuse of the stale
   pre-audit handoff.
9. **Specialist handoff** (rewrites
   `PHASE10_OPTICAL_AUDIT_HANDOFF.md` with the post-audit verdict
   table) and **public-framing ratchet** (re-tightens the §6 hedges
   based on the re-audit result).

### Finding A — Pass A1a: CZA formula spec + literature regression test

**Status: landed 2026-05-13.** Module at `scripts/cza_formula.py`,
regression test at `scripts/test_cza_formula.py`, result note in
[`docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`](calibration/RICH_DISPLAY_OVERLAY_NOTES.md)
under "Pass A1a Spec Results". Verified results: at h = 22°
literature gives 45.734° vs. legacy 44.000° (the WB_R46 = 440 vs.
460 issue confirmed); at p2 (h = 18.6°) the literature residual is
−1.3 px against the legacy −19.0 px (memo's qualitative direction
**confirmed**); at p27 (h = 0.5°) the literature CZA apex is at
y = −11.5 (above top of frame, **confirmed off-frame**) so the visible
arc at y = 142 is not CZA. **A1b is CLEARED to proceed** with one
correction: the audit memo's *stated* formula
`90 − h − arcsin(√(n²−cos²h))` was a transcription error (gives 0.27°
at h = 22°, not the memo's own claimed ~46°); the formula that
matches the memo's own numerical predictions is
`arcsin(√(n²−cos²h)) − h`. A1b uses the verified expression.

**Goal.** Before changing the atlas, write down the literature CZA
formula as an executable spec and verify it against the calibration
photos. The audit memo's exact px collapse numbers ("~0.7 px on p2")
are *test hypotheses*, not implementation truth — even the memo §3
flags that the specific numerical values were not independently
re-derived. The spec + test answers the question "what does the
literature predict?" before the patch answers "what should the atlas
do about it?"

**Touch.**

- New module `scripts/cza_formula.py` (or a similarly named file in
  the same directory as `overlay_calibrate.py`): one function
  `cza_apex_y_above_sun(h_deg, r22_px, n=1.31)` returning the predicted
  CZA apex y-offset above the sun in the same pixel scale as `r22_px`.
  Implement the literature formula
  `cza_offset_deg = arcsin(√(n² − cos²h)) − h`, scaled by
  `r22_px / 22°`. **Correction 2026-05-13:** the audit memo §2 item 2
  states this formula as `90° − h − arcsin(√(n²−cos²h))`, but that
  expression does *not* match the memo's own numerical predictions
  (gives 0.27° at h = 22°, not ~46°). Pass A1a's verify-gate caught
  the transcription error; A1a's result note documents the corrected
  expression, and A1b above uses it.
- New test file `scripts/test_cza_formula.py`: regression test against
  hand-anchored CZA y-positions on the eligible photos. Measure
  predicted-vs-observed for p2 (h = 18.6°) and p27 (h = 0.5°) at
  minimum; record the actual residual deltas and compare against the
  memo's predicted directions ("p2 should collapse from ~−19 px to
  near-zero"; "p27 should reveal that the visible arc at y = 142 is
  not CZA at all because the literature CZA at h = 0.5° lies off-frame").
  The test does *not* assert specific px values; it asserts the
  qualitative direction of the correction.

**Entry criteria.** Audit memo accepted; this roadmap filed.

**Exit criteria.**

- `cza_formula.py` exists, is importable, and round-trips at h = 22°
  to the legacy `sun_y − R46` value within rounding (sanity check that
  the formula reduces correctly at the legacy operating point).
- The regression test runs and prints predicted-vs-observed deltas for
  p2 and p27. The result is recorded as a note in
  `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md` under "Pass A1a
  spec results", with the actual numbers (not the memo's predicted
  numbers).
- The note explicitly flags whether the test confirms the memo's
  qualitative finding (formula bug, p27 is mis-identified) or
  contradicts it. If the test contradicts the memo, A1b does *not*
  proceed and Pass A1 is reopened with a new hypothesis.

**Out of scope.** Editing `overlay_calibrate.py` (that is A1b).

### Finding A — Pass A1b: Atlas formula patch (gated on A1a)

**Status: landed 2026-05-13.** Two surgical edits to
`scripts/overlay_calibrate.py`: (1) line 66 changed `WB_R46 = 440` to
`WB_R46 = round(2.091 * WB_R22)` (= 460), with maintainer comment
pointing at this roadmap and the audit memo; (2) lines 381–384 the inner
`cza_apex(c)` function now calls `cza_formula.cza_apex_y_above_sun_px(h_deg, WB_R22)`
instead of `WB_SUN[1] - WB_R46`, with a fallback to the legacy anchor
when the CZA disappears (h > ~32.2°). Import added at the top of the
file (line 62–66). **Smoke results:** on p2 the CZA apex y-residual
collapsed from −19.3 px (legacy) to +1.3 px (literature) — matches
A1a's measured target within sign-convention. On p7 (h = 59.4°, beyond
CZA disappearance) the script does not crash; CZA falls back to the
corrected WB_R46 = 460 anchor and renders at the legacy position
(visually classified as "not applicable" downstream per existing
vocabulary rules). **Supralateral spot-check:** apex shifted from 44°
above sun to 46° above sun (+2° = 4.55% relative; matches roadmap's
"~4% outward from sun-above" expectation; no follow-up pass needed).
46° halo radius now drawn correctly at 460 * scale instead of 440 *
scale (~4.5% angularly larger, the right direction). Per-overlay
`R46_px = r22_obs × (46/22)` derivation remains explicitly out-of-scope
per roadmap; deferred as a follow-up Phase 10 backlog item.

**Goal.** Replace the atlas's hardcoded CZA expression with the
A1a-validated formula. Single touch on a workbench-space constant
plus a single touch on the line-381 expression. Do *not* mix
workbench-space and per-overlay pixel-space derivations in this pass.

**Touch.**

- `scripts/overlay_calibrate.py:381–384`: replace
  `anchored = WB_SUN[1] - WB_R46` with a call to
  `cza_formula.cza_apex_y_above_sun(h, WB_R22)` (subtracted from
  `WB_SUN[1]` per the existing convention). Imports added at the top
  of the file.
- `scripts/overlay_calibrate.py:66`: replace `WB_R46 = 440` with
  `WB_R46 = round(2.091 * WB_R22)` (≈ 460). Update the `# holds at
  440 — see SUNDOG_V_GEOMETRY.md "R_46 note"` comment to point at the
  audit memo and this roadmap.
- **Out of scope (intentional):** the audit memo also discusses
  deriving `R46_px = r22_obs × (46/22)` per overlay rather than from
  `WB_R46`. That option would also touch supralateral placement and
  any other workbench-to-photo consumer of `WB_R46`. It is *not* a
  drop-in patch and is filed as a follow-up Phase 10 backlog item.
  Pass A1b commits to the workbench-space constant fix only.

**Entry criteria.** Pass A1a landed and its result note confirms the
memo's qualitative finding.

**Exit criteria.**

- Both edits committed.
- The two affected pixel-position tests, if any, regenerated against
  the new formula and the diff inspected by hand on at least p2 and p27.
- Smoke run: on p2 the predicted CZA apex y collapses from ~19 px
  below the literature locus to within the tolerance recorded in A1a's
  result note (the memo's "~0.7 px" is a hypothesis; A1a's measured
  delta is the actual target).
- Spot-check: render a supralateral overlay on p2 and confirm visually
  that the supralateral position has not shifted unreasonably as a
  side-effect of the `WB_R46` constant change. (Supralateral *should*
  shift ~4 % outward from sun-above; if it shifts much more, the
  workbench-space change has a knock-on effect that needs a follow-up
  pass.)

**Out of scope for this pass.** Re-classifying p27's chromatic arc
(that is Pass A2). Re-running the residual gate (that is Pass A3).
Updating `MESA_CROSSOVER_NOTE.md` (that is the §2 cross-cutting work).
Per-overlay R46_px derivation (deferred; see Touch above).

### Finding A — Pass A2: p27 primitive re-classification

**Status: landed 2026-05-13.** `docs/calibration/p27-anchor.json`
re-anchored: top-level `cza_apex` and `cza_apex_residual` fields moved
under `_disputed.cza_apex_legacy_2026_05_13` (preserved verbatim with
rationale, reversible). New top-level `supralateral_46halo_merger`
block with apex `(599, 142)` = 41.89° above sun, candidate primitives
list `[halo_46_top, supralateral_arc]` per the audit memo's hedge
(observationally indistinguishable at h = 0.5° per literature merger).
New top-level `supralateral_route_eligibility` block flags the photo
for Pass A3 coverage-gate consideration. *Geometric verification:*
literature CZA at h = 0.5° is at y = −11.5 (off-frame above top), 46°
halo top and supralateral both predict y = 101.1 (~46° above sun),
visible feature at y = 142 sits ~4° below either prediction (chromatic
broadening / visual-edge bias, consistent with anchor's existing
`flat apex` flag). *Cross-doc walk-back:*
[`docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`](calibration/RICH_DISPLAY_OVERLAY_NOTES.md)
updated in four places — CZA row drops p27, supralateral row promotes
p27 to measured-candidate, anchor summary row notes re-classification,
v3.8 tangent receipt retracts "merges with CZA" framing per memo §4.4.
Pass A3 inputs are now honest: CZA evidence is p2 alone (sub-px
residual), supralateral eligibility is p2 + p27 (~46° both, weak
discrimination per memo §2 item 12). **A3 cleared to proceed.**

**Goal.** Re-label the p27 chromatic arc currently anchored as CZA
apex. At h = 0.5° the literature CZA apex is ~57° above the sun and
off-frame; the visible arc at y = 142 sits at ~42° above the sun and is
the 46° halo top / supralateral merger.

**Touch.**

- `docs/calibration/p27-anchor.json`: re-anchor the chromatic
  arc as `supralateral / 46° halo top` rather than `cza_apex`. Preserve
  the existing CZA fields under a `_disputed` sub-key so the change is
  reversible.
- `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`: note the
  re-classification with the audit memo as the rationale pointer.

**Entry criteria.** Pass A1b landed (so the residual table is being
read from a fixed atlas).

**Exit criteria.**

- p27's CZA-apex residual is removed from the CZA-route eligibility
  count (it was a misidentified primitive, not a CZA measurement).
- p27 is added to the supralateral-route eligibility set as a
  *candidate* (still pending coverage check).
- The "merges with CZA" framing for p27 in the v3.8 receipt is
  retracted per memo §4.4.

**Out of scope.** Whether p27 plus p2 is enough to reopen the
supralateral coverage gate (Pass A3 question). Whether the
re-classified p27 changes the parhelion-route count (it does not —
parhelion is unaffected by CZA primitive ID).

### Finding A — Pass A3: CZA-route re-verdict

**Status: landed 2026-05-13.** Verdicts re-derived for both CZA and
supralateral against the post-A1b atlas and post-A2 eligibility set.
**CZA: fails coverage gate** *(was: residual gate)*. Only p2 is
in-window with an independent CZA-apex measurement; on that photo the
residual is +1.3 px under the literature formula (well below the
8 px / 0.04*R22 threshold). Every other anchored photo is either
out-of-window (p7, h > 32.2° disappearance) or off-frame (p13, p20,
p22, p25, p26, p27, p30 — at low h the literature CZA sits ~50–58°
above sun, above the top of these photos). Reopening coverage
requires new anchors in 5° &lt; h &lt; 32°. **Supralateral: fails
coverage gate + structural-discrimination rider.** p27 is now a
measured candidate (Pass A2); p2 is route-eligible but apex unmeasured.
The predicted h-spread between p2 and p27 is ~0.3° = ~2.5 px at p2's
R22 (below the typical 5–10 px visual-edge measurement noise), so the
route's h-sensitivity is below the noise floor *even with perfect
coverage*. New photos cannot restore the route — atmospheric physics
constrains the signal below the noise. **Cross-doc updates:** Phase 10
closeout table in `docs/SUNDOG_V_GEOMETRY.md` rewritten for both rows;
prose paragraph about "three structurally different reasons" updated
to reflect the post-A3 failure-mode taxonomy (coverage-physics,
coverage-discrimination, detection-tooling); Open Question #1
reframed from "CZA-apex direction inconsistency" to "CZA coverage
expansion" since the residual-direction question was dissolved by
A1b/A2. **Pass C1 and Pass B2 cleared to proceed** (C1 drops p7
from tangent eligibility; B2 re-derives parhelion verdict against
the post-A3 + post-C1 table). **Pass C2 remains optional** per
roadmap §5.

**Goal.** Re-run the CZA residual gate against the fixed formula and
re-classified p27. Decide whether the route is dead, alive, or pending
more anchored photos.

**Touch.**

- `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`: regenerate the
  CZA-route residual table with the new formula and the re-classified
  p27. If only p2 remains eligible, declare a coverage-gate failure
  honestly rather than residual-gate.
- Phase 10 closeout in `SUNDOG_V_GEOMETRY.md`: rewrite the CZA-route
  row of the table per memo §4.2.

**Entry criteria.** Passes A1b and A2 landed.

**Exit criteria.**

- CZA-route verdict is re-derived and re-stated in the doc with the
  failure layer named correctly (residual / coverage / detection).
- The "three independent failure layers" framing is either preserved
  (with the formula bug call-out) or retracted (if the new count is
  two-failures-plus-tooling-question).

### Finding B — Pass B1: Add lever, R22-source, and geometric-validity columns *(first technical gate)*

**Status: landed 2026-05-13.** Per-photo eligibility sub-table now
lives under "Per-Inversion-Route Residual Table" in
[`docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`](calibration/RICH_DISPLAY_OVERLAY_NOTES.md).
All 8 anchor JSONs (`p2`, `p13`, `p20`, `p22`, `p25`, `p26`, `p27`,
`p30`) carry `r22_source` + `r22_source_note`; `p26-anchor.json` adds
a `geometric_validity` block with the right-side `invalid` flag.
Eligibility split per the audit memo §2 items 4–7: **eligible** (p2,
p7), **low-lever caveat / informational** (p13, p22, p30), **ineligible**
(p20, p25, p26, p27). Pass B2 re-derives the parhelion verdict against
this set.

**Goal.** Make the parhelion-route eligibility honest at the *schema*
level, so future runs cannot accidentally promote a row that fails the
audit's two real eligibility tests, and so the geometric-impossibility
already present in p26 (`right_x − sun_x = 322`, `R22 = 323`,
`arccos(R22/offset)` undefined) is flagged before any other pass
re-runs verdict tables.

**Why this is the first technical gate (post-hedge).** The audit's
parhelion-circularity finding is the least speculative of the three —
the arithmetic is mechanical, the JSON evidence is in
`docs/calibration/p27-anchor.json` verbatim, and the p26 impossibility
is a one-subtraction confirmation. Landing the schema columns first
means every later pass (A3, B2, C3) reads its eligibility from a table
that already enforces the constraints rather than restating them in
prose.

**Touch.**

- `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`: add two columns to
  the parhelion-route residual table: `sec(h) − 1` (geometric lever)
  and `R22-source` (`ring-fit` / `parhelion-derived` /
  `inferred-other`).
- Anchor JSON schema (across `docs/calibration/p*-anchor.json`): add an
  `r22_source` field declaring whether `R22` was independently fit
  from a visible ring or inferred. p20, p25, p26, p27 are
  `parhelion-derived` per the audit; verify the others by image
  inspection.
- Add a `geometric_validity` column for rows where `R22 / offset > 1`
  (p26 right side is the known offender — `R22 = 323`,
  `right_x − sun_x = 322`, so `R22/offset = 1.003 > 1`). Mark the
  offending row `invalid` in the table and add a note pointing at
  the audit memo §2 item 6.

**Entry criteria.** Public/doc hedges (§6) landed; no other technical
pass started.

**Exit criteria.**

- Every anchor JSON has an `r22_source` field.
- The residual table has both new columns and the geometric-validity
  column with p26 right flagged `invalid`.
- The audit memo is cited as the rationale in the doc header.
- The "post-audit hedged 2026-05-13" stamp on
  `RICH_DISPLAY_OVERLAY_NOTES.md` (added by §6 step 1) is updated to
  point at this pass as the first technical gate landed.

### Finding B — Pass B2: Parhelion-route re-verdict on the eligible subset

**Status: landed 2026-05-14.** Route survives promotion against the
Pass B1 eligibility set, with materially weaker language per audit
memo §4.1 / §6. **Audit-survived wording:** *passes residual gate at
~0 px on three photos (p2 h = 18.6°, p7 h = 59.4°, p13 h = 6.83°) with
both unambiguous bilateral peaks and an independently fittable 22°
halo.* The pre-audit "every eligible photo" framing is retired across
`docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md` (rolled-up residual
table row, Phase 10 Promotion Verdict, Single-handle closeout, plus a
new "Pass B2 Results" subsection with the full per-photo verdict
roll-up) and `docs/SUNDOG_V_GEOMETRY.md` Phase 10 closeout table. Of
the three eligible photos: p2 and p7 have meaningful geometric lever
(5.52 % and 96.5 %); p13's lever is 0.71 % but it contributes via
"unambiguous bilateral peaks + ring-fit R22." Five other anchored
photos (p20, p22, p25, p26, p27, p30) are informational only — low
lever, parhelion-derived R22, or geometric impossibility (p26 right).
Downstream public-framing surfaces already landed the matching hedges
in §6 steps 2–4; B2's substantive content matches what those hedges
anticipated. **B2 closes the technical-pass wave.** All required
re-audit-gate passes (§6 hedges, B1, A1a, A1b, A2, A3, C1, B2) are
now landed. Pass C2 then landed in the same 2026-05-14 wave: the
wing-radial Lab b\* ridge detector with 22°-halo-radial-profile
subtraction (`scripts/tangent_detector.py`) ran on the post-C1
sampled set (p2 / p13 / p27) and returned **not-recovered** on all
three under a pre-registered 8 / 24 coherent-sample gate. The
tangent-route open question narrows from "Unresolved Open Question for
a wing-based / Lab b\* detector" to "non-literature-standard detector
designs (wing-slope curvature, matched-filter, polarization filtering)
might recover the route" — filed as Phase 10 backlog.

**Goal.** Re-derive the parhelion promotion verdict against the now-honest
eligibility set. The audit predicts the verdict survives but with
materially weaker language.

**Touch.**

- `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`: under "Phase 10
  Promotion Verdict", restate the parhelion-offset row per memo §4.1.
- `SUNDOG_V_GEOMETRY.md` Phase 10 closeout: replace the *"passes
  residual gate at ~0 px on every eligible photo"* table cell with the
  hedged language from memo §6.

**Entry criteria.** Pass B1 landed.

**Exit criteria.**

- The promotion verdict is restated with the eligibility hedge
  visible.
- The "every eligible photo" language is gone from the geometry doc,
  the crossover note, the gravity ledger, and the elevator pitch.
- If the promotion still survives on the eligible subset (the audit
  predicts it does), the verdict is preserved and the doc gets a
  "post-audit hedged 2026-05-13" stamp. If it does not survive (audit
  did not predict this, but it's a possible outcome), the verdict is
  retracted and Phase 10 reopens.

### Finding C — Pass C1: Drop p7 from the tangent-arc eligibility set

**Status: landed 2026-05-14.** p7 is removed from upper-tangent-route
eligibility in `RICH_DISPLAY_OVERLAY_NOTES.md` because h = 59.4° is the
circumscribed-halo regime, not an upper-tangent regime. The historical
column-peak result remains as a cautionary sample, but no longer counts
as a tangent-route residual. The post-C1 tangent set is p2 / p13 / p27:
column-peak detection still fails there, so the route stays blocked, but
the verdict was **protocol-conditional** pending C2 rather than a
class-level negative. *(Post-C2 update 2026-05-14: Pass C2 ran and also
returned not-recovered; the verdict is now "fails under two
literature-standard detectors", with the tooling-conditional framing
narrowing to non-literature-standard designs. See Pass C2 entry below.)* A circumscribed-halo ellipticity → h test is
left as a Phase 10 backlog idea, not a deliverable.

**Goal.** Remove the literature-level primitive misclassification.

**Touch.**

- `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`: drop p7 from the
  tangent-arc eligibility table; record the reason as
  *circumscribed-halo regime per atoptics.co.uk* with the citation.
- Open a separate placeholder for a *circumscribed-halo ellipticity →
  h* test if the team wants an inverse handle at high h. This is a
  Phase 10 backlog item, not a Phase 10 deliverable.

**Entry criteria.** None — independent of A and B.

**Exit criteria.**

- p7 is gone from the tangent-arc residual table.
- A note in the closeout flags that the original "detection-degenerate
  on all four eligible photos" count was on a list with a primitive
  misclassification at the top.

### Finding C — Pass C2: Wing-based / chromaticity-based tangent detector spike

**Status: landed 2026-05-14.** Detector module at
`scripts/tangent_detector.py`, regression runner at
`scripts/test_tangent_detector.py`, captured run output at
`docs/calibration/PASS_C2_DETECTOR_OUTPUT.txt`, full receipt under
"### Pass C2 Update" in
[`docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`](calibration/RICH_DISPLAY_OVERLAY_NOTES.md).
**Verdict: not-recovered on all three photos in the post-C1 sampled
set.** Coherent ridge samples (residual Lab b\* ≥ 3.0 AND radial offset
≤ ±10 px from predicted tangent locus) are 0-2 / ~24 per wing on every
photo — well below the pre-registered 8 / 24 minimum. The negative
survives the halo-radial subtraction step (median offsets ranged −7.5
to +10.5 px, not boundary-pinned), so the failure cannot be attributed
to 22°-halo-ridge contamination of the detection window. C3 verdict
landed in the same wave (see below). The "tooling-conditional"
framing now narrows from "any wing-based or Lab b\* detector might
work" to "non-literature-standard detector designs (wing-slope
geometric curvature, matched-filter against parameterized arc shape,
polarization-channel filtering) might recover the route" — filed as
Phase 10 backlog ideas, not Phase 10 deliverables.

**Goal.** Build the literature-standard detector and re-test p2 and
p13 (the two photos the audit flags as protocol-artifact failures
under column-peak). This is the "tooling vs. physics" question.

**Touch.**

- New detector code under `scripts/`: Lab b\*-channel ridge filter on a
  22°-halo-subtracted residual image, or wing-azimuth-offset radial
  profiles.
- `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`: append a "Pass C2
  results" subsection with the new detector's verdicts on p2, p13, and
  any other photos in the tangent eligibility set after Pass C1.

**Entry criteria.** Pass C1 landed.

**Exit criteria.**

- The wing-based detector exists, is committed, and has been run
  against the tangent eligibility set.
- The receipt language is updated per memo §4.8: cap the detection
  claim at *"column-peak-on-sun-meridian-in-intensity-image fails on
  all four photos; a literature-standard wing-based or Lab b\*-channel
  ridge detector recovers / does not recover the route on
  {photos}."* The route-level verdict follows the empirical result of
  the new detector.

### Finding C — Pass C3: Tangent-arc route re-verdict

**Status: landed 2026-05-14.** Pass C2 result was **not-recovered** on
p2 / p13 / p27 under the wing-radial Lab b\* ridge detector with
22°-halo-radial-profile subtraction. The verdict landed across coupled
surfaces in the same wave: `RICH_DISPLAY_OVERLAY_NOTES.md` route-residual
table + Promotion Verdict table, `SUNDOG_V_GEOMETRY.md` Phase 10
closeout headline table + post-audit-state block,
`PHASE10_OPTICAL_AUDIT_HANDOFF.md` verdict table,
`SUNDOG_V_GRAVITY.md` forward/inverse asymmetry receipt,
`MESA_CROSSOVER_NOTE.md` Geometry Phase 10 closeout subsection. The
route is **not** classified as class-level negative — the negative is
specific to the two literature-standard detector families tested
(column-peak intensity at sun meridian; wing-radial Lab b\* with
halo-radial subtraction). Non-literature-standard detector designs
(wing-slope geometric curvature, matched-filter, polarization
filtering) remain candidates but are filed as Phase 10 backlog.
The single-handle verdict survives unchanged: parhelion-offset remains
the sole promoted inverse handle on the strict 3-photo subset.

**Goal.** Decide whether the tangent-arc route is *class-level
negative*, *tooling-conditional negative*, or *recovered* after Pass
C2.

**Touch.**

- `SUNDOG_V_GEOMETRY.md` Phase 10 closeout: rewrite the tangent-arc
  row of the table to reflect the C2 outcome.
- If the route is recovered, the single-handle verdict becomes a
  two-handle verdict and the *forward-rich / inverse-narrow* framing
  needs a corresponding edit in `MESA_CROSSOVER_NOTE.md` and
  `SUNDOG_V_GRAVITY.md`.

**Entry criteria.** Pass C2 landed.

**Exit criteria.** Tangent-arc verdict is one of the three named
outcomes above, with the supporting evidence in the residual table
and the language ratcheted in (or out of) the public-framing
surfaces accordingly.

## 4. Sequencing and Dependencies

B1 is the first technical gate. Threads A and C converge after it,
and all three converge at the re-audit gate.

```
§6 Public/doc hedges  (run first, in parallel with handoff stop banner)
            ↓
B1  (first technical gate — schema columns + p26 invalidation)
            ↓
   ┌────────┴────────┐
   ↓                 ↓
Thread A:     Thread C:
A1a → A1b     C1
   ↓          ↓
A2            C2  (optional — see §5)
   ↓          ↓
A3            C3  (gated on C2)
   ↓          ↓
   └────┬─────┘
        ↓
B2  (parhelion-route re-verdict; reads the now-honest table)
        ↓
Re-audit gate (§5)
        ↓
Specialist handoff
        ↓
Public-framing ratchet (§2)
```

Cross-thread coupling worth naming explicitly:

- **B1 lands first** because it makes the residual table enforce
  the audit's two real eligibility tests (lever + R22-source) and
  flags the p26 geometric impossibility as a column rather than a
  prose footnote. A3, B2, and C3 all read from this table, so
  landing B1 first means each later pass restates a verdict whose
  eligibility set is already correct.
- **B2 lands last** of the technical passes because its job is to
  re-derive the parhelion verdict against the now-finalized
  eligibility set. Running B2 before A3 / C3 risks restating the
  parhelion verdict and then having to restate it again if A3 or
  C3 changes the surrounding table.
- **A2 may unlock supralateral.** If the re-classified p27 plus p2
  gives two eligible supralateral measurements, the supralateral
  coverage gate may reopen — though memo §2 item 12 notes the route
  has structurally weak h-discrimination (~0.5° change across h =
  0–22°) and is unlikely to be a useful inverse handle in principle.
  Treat any reopening as a Phase 10 backlog item, not a blocker.
- **None of the passes block the public-framing hedges** in §6.
  Those should land *first*, in parallel with the handoff stop
  banner, before any technical pass begins. Waiting for the
  re-audit to land before hedging the public copy keeps a known
  integrity gap open longer than necessary.

## 5. Re-Audit Gate (Before Real-Specialist Outreach)

Per memo §8: do not send the existing handoff to a real specialist
before the §3 passes land. The smoking-gun formula bug at
`overlay_calibrate.py:381` is a fifteen-minute find for any halo
specialist; if they land on it before the team does, the rest of the
audit credibility takes damage.

**Re-audit entry criteria.** All passes labelled "Required for
re-audit" below are committed:

- **§6 hedges** (public/doc) — required, must land first.
- **Handoff stop banner** on `PHASE10_OPTICAL_AUDIT_HANDOFF.md` —
  required, in parallel with §6 hedges.
- **B1** (schema columns + p26 invalidation) — required, first
  technical gate.
- **A1a** (formula spec + literature regression test) — required.
- **A1b** (atlas formula patch) — required, gated on A1a.
- **A2** (p27 re-classification) — required, gated on A1b.
- **A3** (CZA-route re-verdict) — required, gated on A1b and A2.
- **C1** (drop p7 from tangent eligibility) — required, **landed
  2026-05-14**.
- **C2** (new tangent detector) — *landed 2026-05-14*. Wing-radial
  Lab b\* ridge detector with 22°-halo-radial-profile subtraction
  (`scripts/tangent_detector.py`) ran on the post-C1 sampled set
  (p2 / p13 / p27) and did **not** recover the route. The receipt
  language now resolves the §4.8 template to: *"column-peak fails on
  p2 / p13 / p27; a literature-standard wing-based Lab b\* ridge
  detector with 22°-halo-radial-profile subtraction does not recover
  the route on p2 / p13 / p27 either."* Non-literature-standard
  detector designs (wing-slope geometric curvature, matched-filter,
  polarization filtering) are filed as Phase 10 backlog.
- **C3** (tangent-arc re-verdict) — *landed 2026-05-14*, gated on
  C2's not-recovered outcome. Verdict edits propagated to the coupled
  surfaces in the same wave (RICH_DISPLAY_OVERLAY_NOTES.md route-residual
  + Promotion tables, SUNDOG_V_GEOMETRY.md Phase 10 closeout, handoff
  verdict table, gravity ledger, mesa crossover note).
- **B2** (parhelion-route re-verdict) — required, runs *last* of the
  technical passes so it reads from the post-A3/post-C3 table.

**Re-audit shape.** Re-run the synthetic three-persona protocol
against the post-pass state of the docs and code. The consolidator's
verification gate is mandatory; treat any persona over-statement as
disqualifying for that finding (see [§7](#7-meta-treat-the-verify-gate-as-load-bearing)
on the verify gate).

**Status 2026-05-14:** landed in
[`docs/calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md`](calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md).
No new load-bearing code, anchor, or route-math blocker was found. The
technical-pass wave clears with the post-pass taxonomy: parhelion promoted
on the strict 3-photo subset; CZA coverage-gated; supralateral
physics-discrimination-gated; tangent detector unresolved under C2.
**Post-C2 update 2026-05-14:** Pass C2 then landed, with verdict
**not-recovered** on p2 / p13 / p27 under the wing-radial Lab b\*
detector with halo-radial subtraction. The re-audit memo's tangent
disposition is amended via its Post-C2 addendum; the taxonomy
otherwise survives.

**Specialist handoff entry criteria.** Re-audit completes with no new
load-bearing findings, or with new findings that are bounded to known
open questions in `PHASE10_OPTICAL_AUDIT_HANDOFF.md`. The handoff doc
itself is updated to reflect the post-audit state of the verdict
language.

## 6. Public-Framing Hedge Sequencing

Independent of the §3 passes, the surfaces in §2 should be hedged
*now* and re-ratcheted (or further retracted) after the re-audit. The
order minimizes window-of-inconsistency between coupled surfaces. All
six steps run before any §3 technical pass starts.

0. **`docs/calibration/PHASE10_OPTICAL_AUDIT_HANDOFF.md` STOP banner.**
   Add a `⛔ DO NOT SEND` block at the top of the handoff naming the
   audit, the three findings, and pointing at this roadmap's re-audit
   gate (§5). The handoff doc itself is rewritten *after* the
   re-audit; the banner is what prevents the stale verdict table from
   being shipped to a specialist in the meantime. **Status: landed
   2026-05-13.**
1. **`docs/SUNDOG_V_GEOMETRY.md` Phase 10 closeout.** Add a
   "Post-audit hedge 2026-05-13" sub-block at the top of the closeout
   that names the audit memo, the three load-bearing findings, and
   the fact that the closeout language below is pending re-derivation
   per this roadmap. **Status: landed 2026-05-13; superseded
   2026-05-14 by the post-pass state block and re-audit memo.**
2. **`docs/MESA_CROSSOVER_NOTE.md` Phase 10 closeout subsection.** Add
   the same hedge stamp; walk back "three independent failure
   layers" to the audit-survived language ("three structurally
   different failure layers, two of which have non-atlas
   explanations"). **Status: landed 2026-05-13.**
3. **`docs/SUNDOG_V_GRAVITY.md` public-framing sentence.** Hedge the
   forward-richness half per memo §6. **Status: landed 2026-05-13.**
4. **`index.html#elevator-pitch` v1 pitch.** Hedge in place: keep
   the third paragraph but add an inline audit-driven hedge clause at
   the load-bearing sentence. Bump `data-version` to v1.1,
   `data-revised` to today, update the visible stamp, and add a
   `data-audit-hedge` attribute naming the controlling roadmap.
   **Status: landed 2026-05-13 as v1.1.** (Pull-paragraph option from
   the original step 4 was considered and rejected in favor of
   hedge-in-place to preserve structure for the re-audit re-ratchet.)
5. **`docs/SUNDOG_V_CHAT.md` §16.2.** Update the third bullet
   (mesa↔geometry crossover) to flip the resolution path from
   "decide whether to ratchet a route" to "audit-driven retraction
   pending re-audit; do not ratchet `claim_map.json` routes."
   **Status: landed 2026-05-13.**

## 7. Meta: Treat the Verify Gate as Load-Bearing

The synthetic audit's consolidator verification step has now paid for
itself three times: it caught one Persona 3 over-statement (the p22 halo
claim) plus several uncited Persona invocations in the first audit; Pass
A1a's executable spec caught the audit memo's CZA formula transcription
error before the atlas patch landed; and the re-audit gate forced the
post-pass taxonomy back through numeric, schema, and stale-surface checks
before specialist handoff. These are not "courtesy steps"; they are the
load-bearing part of the audit protocol.

Going forward, the protocol is:

- A persona-style audit produces draft findings.
- The consolidator independently re-verifies each load-bearing
  finding against source files, photographs, and atmospheric-optics
  literature.
- Findings that fail verification go into the consolidated memo's
  *§3 Unverified / contradicted* bin, not the §2 verified bin.
- The consolidated memo is the input to action; the persona drafts
  are appendices.
- The same protocol applies to the post-pass re-audit before
  specialist handoff.

## 8. References

- [`docs/calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md`](calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md)
  — the verified findings, the §3 dropped claims, the §4 doc-change
  recommendations, the §5 protocol fixes, the §6 public-framing
  treatment, the §7 persona disagreements, and the §8 confidence on
  next-step. **This roadmap is the action plan for that memo's §4 and
  §5; the memo is authoritative on findings.**
- [`docs/calibration/PHASE10_OPTICAL_AUDIT_DRAFTS/persona-1-tangent-arc.md`](calibration/PHASE10_OPTICAL_AUDIT_DRAFTS/persona-1-tangent-arc.md)
- [`docs/calibration/PHASE10_OPTICAL_AUDIT_DRAFTS/persona-2-cza-supralateral.md`](calibration/PHASE10_OPTICAL_AUDIT_DRAFTS/persona-2-cza-supralateral.md)
- [`docs/calibration/PHASE10_OPTICAL_AUDIT_DRAFTS/persona-3-parhelion-forward.md`](calibration/PHASE10_OPTICAL_AUDIT_DRAFTS/persona-3-parhelion-forward.md)
- [`docs/calibration/PHASE10_OPTICAL_AUDIT_HANDOFF.md`](calibration/PHASE10_OPTICAL_AUDIT_HANDOFF.md)
  — the existing handoff doc; do not send to a specialist before the
  §5 re-audit gate.
- [`docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`](calibration/RICH_DISPLAY_OVERLAY_NOTES.md)
  — the canonical residual tables and per-photo notes; primary edit
  surface for Passes A3, B1, B2, C1, C2, C3.
- [`docs/SUNDOG_V_GEOMETRY.md` Phase 10](SUNDOG_V_GEOMETRY.md#phase-10---rich-display-overlay-tuning)
  — the Phase 10 section; receives the post-audit hedge per §6.
- [`docs/SUNDOG_V_MESA.md`](SUNDOG_V_MESA.md) — mesa roadmap; the
  field-not-reward thesis it stages now depends on this roadmap's
  outcome via the crossover note.
- [`docs/MESA_CROSSOVER_NOTE.md`](MESA_CROSSOVER_NOTE.md) — the
  crossover note; receives the hedge per §6.
- [`docs/SUNDOG_V_GRAVITY.md`](SUNDOG_V_GRAVITY.md) — gravity ledger;
  the public-framing sentence is hedge-required.
- [`docs/SUNDOG_V_CHAT.md` §16](SUNDOG_V_CHAT.md#16-coupled-public-copy-surfaces-integrity-coordination)
  — coupled-public-copy integrity protocol; §16.2 third item is
  updated by §6 of this roadmap.
- `scripts/overlay_calibrate.py:381–384` — the line-381 expression
  edit for Pass A1b. `scripts/overlay_calibrate.py:66` — the
  workbench-space `WB_R46` constant edit for Pass A1b.
