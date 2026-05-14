# Phase 10 Optical Specialist Audit Handoff

> ## ✅ CLEARED FOR EXTERNAL HANDOFF — 2026-05-14
>
> **Status:** This handoff was rewritten 2026-05-14 against the
> post-re-audit state of the Phase 10 campaign. The previous version of
> this file (carrying a `⛔ DO NOT SEND` banner) is preserved in git
> history; readers wanting the pre-audit verdict and audit-ask should
> diff against that revision.
>
> **Provenance chain:**
>
> 1. Original Phase 10 closeout: 2026-05-13 (single-handle verdict).
> 2. Synthetic three-persona optical audit:
>    [`PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md`](PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md)
>    (2026-05-13). Found three load-bearing issues: CZA formula bug,
>    parhelion-route eligibility overclaim, p7 tangent
>    misclassification.
> 3. Attack roadmap to address them:
>    [`../PHASE10_ATTACK_ROADMAP.md`](../PHASE10_ATTACK_ROADMAP.md).
>    Eight required passes (§6 hedges, B1, A1a, A1b, A2, A3, C1, B2)
>    plus one optional (C2). All required passes landed 2026-05-13/14.
> 4. Re-audit gate:
>    [`PHASE10_OPTICAL_REAUDIT_MEMO.md`](PHASE10_OPTICAL_REAUDIT_MEMO.md)
>    (2026-05-14). Three-persona re-audit + consolidator verification.
>    Verdict: technical-pass wave clears; no new code, anchor, or
>    route-math blocker. The campaign authorizes this rewrite and the
>    public-framing ratchet, not reuse of the pre-audit verdict
>    language.
>
> The audit-ask below is the *narrowed* set of questions the campaign
> did not internally resolve. Questions about the CZA formula, p27
> primitive ID, p7 tangent eligibility, and the parhelion-route
> "every eligible photo" framing have all been internally addressed
> and are documented in the attack roadmap rather than being asked
> here.

## 1. Why we're asking

The Sundog program has a halo-overlay atlas calibrated against a small
set of sun + halo + parhelion photographs. The atlas claims that all
visible halo primitives derive forward from a single parameter `h` (sun
altitude). Phase 10 of the geometry roadmap is the empirical test of
whether the atlas inverts — whether `signature → h` is recoverable from
photographs without the per-photo hand-anchor that currently bootstraps
each fit.

After the post-re-audit campaign, the verdict is:

| route | gate outcome | failure-mode kind |
| --- | --- | --- |
| Parhelion offset → h | **promoted (post-hedged)** | 3-photo strict eligibility (p2 h = 18.6°, p7 h = 59.4°, p13 h = 6.83°): unambiguous bilateral peaks, valid geometry, ring-fit R22, non-trivial discrimination (for p2, p7) or low-lever-but-supported (for p13) |
| CZA apex → h | **fails coverage gate** | dataset / aspect-ratio: only p2 is in-window with an independent residual (+1.3 px under the post-A1b literature formula); other anchored photos are either past the CZA disappearance threshold (h > 32.2°) or have the literature CZA apex predicted above the top of the frame |
| Supralateral position → h | **fails structural-discrimination gate** | atmospheric physics: supralateral angular distance from sun varies only ~0.5° across h = 0–22°, below the typical 5–10 px visual-edge measurement noise even at perfect coverage |
| Tangent-arc curvature → h | **detection gate under two literature-standard detectors** *(C1 + C2 landed 2026-05-14)* | tooling protocol: column-peak fails on the post-C1 sampled set (p2, p13, p27); p7 dropped as circumscribed-halo regime at h = 59.4°. **Pass C2 then built a wing-radial Lab b\* ridge detector with 22°-halo-radial-profile subtraction (`scripts/tangent_detector.py`) and ran it on p2 / p13 / p27: not-recovered on all three photos under a pre-registered 8 / 24 coherent-sample gate (amplitude ≥ 3.0 in residual b\*, radial offset ≤ ±10 px from predicted locus).** The halo-radial subtraction explicitly addresses Persona 1's methodological recommendation, so the failure is not detection-window contamination. The "tooling-conditional" framing now narrows to non-literature-standard detector designs (wing-slope geometric curvature, matched-filter against parameterized arc, polarization-channel filtering) — Phase 10 backlog. |

The audit-survived public-framing sentence in
[`../SUNDOG_V_GRAVITY.md`](../SUNDOG_V_GRAVITY.md) is:

> *Parhelion-offset → h passes the residual gate at ~0 px on three
> photos (p2, p7, p13) with both unambiguous bilateral peaks and an
> independently fittable 22° halo. The three other tested routes fail
> for three different reasons: dataset / aspect-ratio coverage (CZA),
> atmospheric-physics discrimination (supralateral), and
> detection-protocol tooling (tangent fails under two literature-standard
> detectors — column-peak intensity and wing-radial Lab b\* with
> halo-radial subtraction — after Pass C2 landed 2026-05-14;
> tooling-conditional narrows to non-literature-standard designs).*

We want pushback on this verdict from atmospheric-optics specialists.
The audit is most useful if it ends in a counterexample we can act on.

## 2. What we want pushed on

The campaign internally resolved the load-bearing pre-audit issues
(formula bug, primitive ID, eligibility framing). The questions below
are the ones a single domain specialist is best-positioned to answer
that the team is not.

### 2.1 Are these the right four routes? *(unchanged)*

We tested parhelion-offset, CZA-apex, supralateral-position, and
tangent-arc curvature. Atmospheric-optics literature lists more
candidate features:

- 46° halo radius vs 22° halo radius — currently both treated as
  anchor-derived, not inverse routes.
- Parhelic-circle visible extent.
- Sun pillar length (when present).
- Helic / sub-parhelic features.
- Lower tangent arc curvature (mirror of upper-tangent test).
- Circumscribed-halo ellipticity at h > 29° (filed as Phase 10 backlog
  after Pass C1 dropped p7 from upper-tangent eligibility).

Question: are there `signature → h` inverse routes we are not even
attempting that we should be? Conversely, is one of the four we tested
not actually expected to invert cleanly under any protocol (i.e., were
we testing a route that atmospheric optics knows is degenerate by
construction)?

### 2.2 Is the supralateral structural-discrimination finding correctly bounded?

Pass A3's verdict: supralateral angular distance from sun varies only
~0.5° across the h = 0–22° eligibility range, which puts the available
h-signal below the typical 5–10 px visual-edge measurement noise. We
read this as a *route-physics* limitation, not a *dataset* limitation:
no realistic photo set fixes it.

Questions: (i) is the ~0.5° figure (audit memo §2 item 12) actually
the right number under the literature's standard supralateral
parameterization, or is there a better one? (ii) Are there sub-cases
(specific crystal habits, viewing azimuths, low-h-only regimes) where
the h-spread is materially larger? (iii) Is the supralateral arc
genuinely structurally weak as an inverse handle in atmospheric optics,
or is the team interpreting a non-canonical formulation?

### 2.3 Tangent-arc curvature: now negative under two literature-standard detectors — is a non-literature-standard detector worth building?

**Pass C2 update 2026-05-14.** The wing-based / Lab b\* detector has
now been built and run. Implementation:
[`scripts/tangent_detector.py`](../../scripts/tangent_detector.py) (a
wing-azimuth-offset Lab b\* ridge detector with 22°-halo-radial-profile
subtraction — the literature-standard substrate audit memo §4.8 /
Persona 1 §4 item 6 explicitly recommended). On the post-C1 sampled
set (p2 / p13 / p27), the detector returns **not-recovered** on every
photo: 0-2 / ~24 coherent ridge samples per wing under a pre-registered
gate (amplitude ≥ 3.0 in residual b\*, radial offset ≤ ±10 px from
predicted tangent locus, ≥ 8 / 24 samples coherent). The negative
survives the halo-radial-profile subtraction step that addresses the
audit memo's methodological concern; it cannot be attributed to
22°-halo-ridge contamination of the detection window. Full receipt:
"### Pass C2 Update" in
[`RICH_DISPLAY_OVERLAY_NOTES.md`](RICH_DISPLAY_OVERLAY_NOTES.md);
captured run output at [`PASS_C2_DETECTOR_OUTPUT.txt`](PASS_C2_DETECTOR_OUTPUT.txt);
re-audit memo addendum in
[`PHASE10_OPTICAL_REAUDIT_MEMO.md`](PHASE10_OPTICAL_REAUDIT_MEMO.md).

The route is now **detection-gate negative under two literature-standard
detector families**: column-peak intensity at sun meridian, and
wing-radial Lab b\* with halo-radial subtraction. The "tooling-conditional"
framing narrows to non-literature-standard detector designs.

Reframed questions for the specialist:

(i) The wing-slope **geometric** curvature detector (Persona 1 §3 p2
entry recommended this for the locally-flat-apex case — C2 tested
chromatic ridge, not geometric curvature). Is this a literature-standard
approach the team missed, or is it a non-literature-standard design that
needs new tooling work?

(ii) Matched-filter detection against a parameterized upper-tangent-arc
model (template correlation, which uses the full arc geometry rather
than per-sample ridge picks). Standard practice in halo-observation
literature, or a non-literature-standard escalation?

(iii) Polarization-channel filtering — the upper tangent arc is
partially polarized. Is a polarizer-equipped follow-up photo at
controlled altitude (cf. [`../SUNDOG_V_PERCEPTION.md`](../SUNDOG_V_PERCEPTION.md)
Phase 1) a different *substrate test* the specialist would recommend
before declaring the route dead?

(iv) Is "upper tangent arc curvature → h" even the right inverse-route framing
at the literature level, or is the canonical route something like
"upper-tangent opening angle"?

### 2.4 Anchor-capture protocol soundness *(sharpened post-B1)*

Our per-photo anchor protocol is now documented at the schema level in
[`RICH_DISPLAY_OVERLAY_NOTES.md`](RICH_DISPLAY_OVERLAY_NOTES.md)
"Parhelion-Route Per-Photo Eligibility" sub-table, plus
top-level `r22_source` and `geometric_validity` fields in each
`p<NN>-anchor.json`. The schema records:

- `sun_px` via saturation-centroid + warm-index + crosshair-crop
  confirmation
- `R22_px` via saturation-ring fit (`ring-fit`) or stipulated from
  parhelion (`parhelion-derived`) or inherited (`inferred-other`),
  with the source declared per anchor
- `parhelion_left_px` / `parhelion_right_px` via peak detection
- `parhelic_belt_y_px` via belt-region inspection
- `geometric_validity` (per parhelion side) flagging when
  `R22 / offset > 1` (geometric impossibility)

Questions: (i) do these protocols match atmospheric-optics literature
norms for halo geometry measurement? (ii) Are there well-established
alternative methods that would reduce anchor noise materially? (iii)
Would controlled-optics conditions (calibrated camera, known lens
distortion, flat-field correction) change which routes pass the
residual gate?

### 2.5 Forward-generation claim *(refined post-A1b)*

Pass A1b corrected the CZA atlas formula (the legacy `sun_y − WB_R46`
hardcode was geometrically correct only at h ≈ 22°; replaced with the
literature `arcsin(√(n² − cos²h)) − h` formula and `WB_R46` corrected
from 440 to 460 in workbench coords). Forward generation now runs
cleanly for:

- Parhelion offset (`R22 / cos(h)`)
- 22° halo radius (anchor)
- 46° halo radius (post-A1b correction)
- CZA visibility window (h < 32.2°) and CZA apex position when in-window
- Upper tangent geometric spine (with the caveat that brightness/chromatic
  spine ≠ geometric spine)

The atlas still draws but should hedge for:

- Parhelic-circle vertical offset (`−0.05 · R22`): the Phase 10-FF
  belt-y replication study (PHASE10_BELT_Y_RESULTS.md) falsified this
  as a generic rule; p13's residual is photo-specific.
- 46° halo and infralateral arcs: drawn on every rich-vocabulary
  overlay but visible only on p2 in the calibration set.

Questions: (i) are there forward-direction primitives we are
over-claiming for in the post-A1b atlas? (ii) Is the literature CZA
formula we adopted (Bravais-derivation form) the canonical one, or
does atmospheric-optics literature use a different formulation that
would give slightly different numbers? Pass A1a's regression test
confirmed the formula direction qualitatively at p2 and p27, but the
test does not certify the formula's third-decimal precision.

### 2.6 Three-failure-mode taxonomy

Pre-audit, we framed the three non-promoted routes as failing "three
structurally different layers of the measurement stack" (residual /
coverage / detection). Post-audit + post-re-audit, the taxonomy is
refined: the four routes fail (or pass) along three *structurally
different failure modes*, not three independent gates:

1. **Parhelion offset:** promoted under post-hedged 3-photo
   eligibility.
2. **CZA:** fails on **dataset / aspect-ratio** grounds (single
   in-window photo).
3. **Supralateral:** fails on **atmospheric-physics discrimination**
   grounds (h-signal below measurement noise).
4. **Tangent:** fails on **tooling-protocol** grounds (current
   detector wrong; literature detector not yet built).

Question: does this match standard atmospheric-optics measurement-stack
vocabulary, or are we reinventing a taxonomy that already exists under
different names? If the field has language for these distinctions, we
should adopt it. (The crossover note
[`../MESA_CROSSOVER_NOTE.md`](../MESA_CROSSOVER_NOTE.md) ports this
taxonomy back to the mesa-side controller-substrate work; the
vocabulary's portability is also a consideration.)

### 2.7 Public-framing claim

Audit-survived sentence (replacing the pre-audit "rich in forward
generation + one image-recoverable inverse handle" framing):

> *The atlas is forward-rich for primitives whose geometry the
> literature parameterizes from `h` alone, with hedges on the
> parhelic-circle rule and the 46° halo / infralateral visibility
> windows. On the inverse direction, the parhelion-offset route is
> promoted on the strict 3-photo eligibility set (p2, p7, p13); the
> three other tested routes fail at three structurally different
> failure modes: dataset / aspect-ratio coverage (CZA), atmospheric-
> physics discrimination (supralateral), and detection-protocol tooling
> (tangent fails under two literature-standard detectors after C2 landed
> 2026-05-14; tooling-conditional narrows to non-literature-standard
> designs).*

Question: is this defensible to atmospheric-optics readers? Where it
isn't, what would the right hedged phrasing be?

## 3. What would change the verdict

We are explicitly trying to surface counterexamples. The audit is most
useful if it ends in any of:

1. **A fifth route we did not consider** that does invert cleanly on
   the current photo set. (Verdict shifts from "one handle (post-hedged
   3-photo)" to "two-or-more handles.")
2. **A wing-based or Lab b\* ridge detector that recovers the
   tangent-arc route** on at least p2 and p13. (Verdict shifts to "two
   handles, with tangent at protocol-conditional confidence.") The
   campaign's C2 pass would build and test such a detector; the audit's
   pointer at the right detector is itself useful.
3. **A reframing of the supralateral structural-discrimination
   finding** that says it is correctly bounded in some sub-regime not
   the team considered. (Verdict shifts from "supralateral is
   structurally dead" to "supralateral is dead under our
   parameterization, alive under literature-standard X.")
4. **A pointer to literature that says forward generation is itself
   overclaimed** in some primitive class we haven't already hedged.
5. **No counterexamples found; the verdict is structurally sound under
   known atmospheric-optics frameworks.** (Verdict baked; public
   framing ratchets to the audit-survived sentence in §2.7.)

Outcomes (1)–(3) are most actionable. Outcome (4) is a publication
hygiene check. Outcome (5) is the post-audit ratchet trigger.

The pre-audit outcome (3) ("a reinterpretation of the parhelion-offset
success as photo-set-lucky rather than structurally clean") is now
internally resolved by Passes B1 and B2: the parhelion verdict is
already restricted to the 3-photo strict eligibility set with the
audit-survived wording. A specialist finding *additional* photo-set
issues is still useful but is no longer the main risk.

## 4. Materials provided

### 4.1 Photo set and overlays

`docs/calibration/` and `docs/calibration/overlays/` contain:

- The full image set (p1 through p31; not all anchored — anchored set
  is p2, p7 (per Anchor Summary; no JSON), p13, p20, p22, p25, p26, p27,
  p30).
- Rich-vocabulary overlays for p2 / p7 / p13 (`p2.rich-vocabulary.overlay.png` etc.).
- Per-photo anchor JSONs at `p<NN>-anchor.json`. Each carries a
  top-level `r22_source` and `r22_source_note`; `p26-anchor.json`
  adds a `geometric_validity` block; `p27-anchor.json` carries a
  `supralateral_46halo_merger` block (post-A2 re-classification) and
  preserves the original CZA fields under `_disputed.cza_apex_legacy_2026_05_13`
  for reversibility.

### 4.2 Canonical project docs (read in this order)

1. **[`PHASE10_OPTICAL_REAUDIT_MEMO.md`](PHASE10_OPTICAL_REAUDIT_MEMO.md)**
   *(2026-05-14)* — the re-audit verdict + post-pass failure taxonomy.
   Treat as the authoritative state for what the campaign accomplished.
2. **[`../PHASE10_ATTACK_ROADMAP.md`](../PHASE10_ATTACK_ROADMAP.md)** —
   the eight required passes (§6 hedges, B1, A1a, A1b, A2, A3, C1, B2)
   plus the optional C2. Each pass carries entry/exit criteria, status
   stamps, and cross-doc walk-back inventory.
3. **[`PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md`](PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md)**
   *(2026-05-13)* — the synthetic three-persona pre-pass audit. Reads
   as the historical record of what the campaign was responding to.
4. **[`RICH_DISPLAY_OVERLAY_NOTES.md`](RICH_DISPLAY_OVERLAY_NOTES.md)** —
   Phase 10 source-of-truth: anchor summary, Per-Inversion-Route
   Residual Table (with Pass A1a/A1b/A2/A3 result sub-sections),
   Parhelion-Route Per-Photo Eligibility sub-table, Phase 10 Promotion
   Verdict, Single-handle closeout, Tangent-Curvature v3.8 Receipt.
5. **[`PHASE10_EXPANSION_TRIAGE.md`](PHASE10_EXPANSION_TRIAGE.md)** —
   candidate photo inventory and triage calls.
6. **[`../SUNDOG_V_GEOMETRY.md`](../SUNDOG_V_GEOMETRY.md)** Phase 10 +
   Phase 11 sections — overall geometry roadmap; the Phase 10 closeout
   table is post-pass / post-re-audit.
7. **[`../MESA_CROSSOVER_NOTE.md`](../MESA_CROSSOVER_NOTE.md)** —
   Geometry Phase 10 closeout section + three-failure-mode taxonomy;
   the cross-substrate framing the post-audit work confirms.
8. **[`../SUNDOG_V_GRAVITY.md`](../SUNDOG_V_GRAVITY.md)** forward/inverse
   asymmetry receipt — the public-framing language under audit.
9. **[`PHASE10_BELT_Y_SPEC.md`](PHASE10_BELT_Y_SPEC.md)** + **[`PHASE10_BELT_Y_RESULTS.md`](PHASE10_BELT_Y_RESULTS.md)** —
   parhelic-belt-y replication cycle (FF1 falsified, photo-specific).
   Example of the pre-registration / decision-rule discipline applied
   to a sub-question; not load-bearing for the main audit question.

### 4.3 Code referenced by the audit

- **[`../../scripts/cza_formula.py`](../../scripts/cza_formula.py)** —
  literature CZA-above-sun formula module. Authoritative source for
  the post-A1b atlas math.
- **[`../../scripts/test_cza_formula.py`](../../scripts/test_cza_formula.py)** —
  regression test against p2 and p27 anchors. Runs via
  `python scripts/test_cza_formula.py` from repo root.
- **[`../../scripts/overlay_calibrate.py`](../../scripts/overlay_calibrate.py)** —
  the atlas overlay renderer. Lines 62–80 carry the Pass A1b
  WB_R46 + cza_formula edits; lines ~395–410 carry the cza_apex
  inner-function literature call.

### 4.4 What the audit can ignore

The mesa-side documents (`SUNDOG_V_MESA.md`, `PHASE6_V*_RESULTS.md`)
are not under audit. They are a parallel in-vitro result that the
atlas-side single-handle finding is being compared against in the
public-framing language. The audit does not need to evaluate the
mesa-side; it only needs to flag if the comparison is being overclaimed
in `SUNDOG_V_GRAVITY.md` or `MESA_CROSSOVER_NOTE.md`.

## 5. Deliverable shape

A short memo (2–4 pages) addressing the questions in §2 with verdicts
of the form:

- "Sound" / "Sound with caveat: …" / "Pushback: …" / "Out of my area"

Plus, where the answer is pushback, a pointer to literature or a
counterproposal we can test.

We do not need a paper or a comprehensive review. We need the kind of
memo a domain specialist would send a non-specialist team that has
done careful work in a neighboring area and is about to publish.
Brutally short is fine; brutally honest is better.

## 6. Out of scope for this audit

- Mesa-side in-vitro results (`SUNDOG_V_MESA.md`).
- The Sundog brand / UI / app surfaces.
- The perception roadmap (`SUNDOG_V_PERCEPTION.md`) — separate
  downstream roadmap, not part of the Phase 10 verdict.
- Detailed redesign of the anchor-capture pipeline — pointers to
  better methods are useful; we will implement them ourselves.
- Implementation of any proposed alternative detector for the
  tangent-arc-curvature route — pointing us at the right method is
  enough.
- Re-litigation of the campaign's internal findings (formula bug, p27
  primitive ID, p7 tangent eligibility, parhelion eligibility
  framing). The campaign already resolved these; if the audit
  disagrees with a specific resolution, surface that as a §3 outcome,
  not as a §2 question.

## 7. Audience-tier placement

This handoff is the **specialist tier** of a three-tier outreach
framework documented in
[`PHASE11_OUTREACH_BRIEF.md`](PHASE11_OUTREACH_BRIEF.md). The other two
tiers — technically-literate science-communications editors, and
Wikipedia-adjacent / external-link reviewers — receive different
artifacts pitched at different levels of literature/jargon density;
this handoff is the version pitched at an atmospheric-optics specialist
on a ~30-minute time budget. The brief's §15 deployment methodology
covers how the three-tier outreach is meant to be synthesised before
external deployment (same synthetic-persona protocol as the Phase 10
attack campaign, with external-reviewer-impersonator personas instead
of internal-audit personas).

The brief intentionally holds the two-substrate field-shape pattern
claim (documented in [`../MESA_CROSSOVER_NOTE.md`](../MESA_CROSSOVER_NOTE.md)
and [`../SUNDOG_V_GRAVITY.md`](../SUNDOG_V_GRAVITY.md)) out of Phase 11
outreach. If a specialist asks about the cross-substrate framing in
follow-up, treat it as in-scope; do not lead with it in any deliverable
built from this handoff.

This handoff is Persona A's primary input artifact in the Phase 11
synthetic dispatch at
[`PHASE11_OUTREACH_SYNTHETIC_MEMO.md`](PHASE11_OUTREACH_SYNTHETIC_MEMO.md)
(scaffolded 2026-05-14). Specialist-bounce-test findings against this
handoff are recorded in the dispatch memo's §3 verified-findings bin;
external send happens only after the dispatch's verification gate
clears.

## 8. Contact for clarifications

This document and the canonical docs in §4.2 are intended to be
self-contained, but the program team is available for clarifying
questions. Address questions to the geometry-side lead by including a
pointer to the section of this handoff or the linked canonical docs.
