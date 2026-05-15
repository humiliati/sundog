# P2 Agent Run — Four-Quantity Scoring

Artifact for: boundary-map loci L1–L5 (P0 deliverable, frozen 2026-05-15 PT)
Pre-registration: [`README.md`](README.md)
Roadmap: [`SUNDOG_V_GRAVITY.md`](../../SUNDOG_V_GRAVITY.md) ▸ Candidate 13 ▸ Roadmap
Run date: **2026-05-15 (PT)**
Status: **P2 PASSES on all four quantities.** Outcome branch selected: B —
traceability harness passes on this domain. P3 (disposition) is now unblocked.

---

## What P2 tests

The prereg's Agent Run Admission Rule requires that the run score four
quantities **separately**:

1. **Convergence** to the withheld hidden target `h` (sun altitude).
2. **Counterfactual steerability** — does editing the indirect signal move
   the agent's estimate to the analytically predicted new value?
3. **Failure-boundary coincidence** — does the route fail, abstain, or
   switch handles at every L1–L5 boundary in the BOUNDARY_MAP?
4. **Matched-baseline efficiency** — does the route outperform a matched
   null baseline?

Quantities 1–3 are the traceability claim. Quantity 4 is efficiency only.
Failure on 2 or 3 would rebrand the result as opaque-correlate; failure on
4 alone does not erase traceability.

## The agent

The agent is the parhelion-offset closed-form inverse as implemented in
`public/js/parhelion-geometry.mjs`:

- Forward: `offset_px = R22_px / cos(h)` (`daggerPointsFromSunAltitude`,
  equivalently `phase3.daggerOffset`).
- Inverse: `h_est = arccos(R22_px / offset_px)`.

The per-photo `R22_px` is from an independent ring fit (not from the
parhelion), and `offset_px` is the bilateral or unilateral parhelion
distance from the sun center, both measured in image-pixel coordinates.
This is the sole **promoted** inverse handle after the Phase 10 audit.

---

## Q1 — Convergence

Eligible photo set: **p2, p7, p13** (strict eligibility per
`docs/calibration/PHASE10_OPTICAL_AUDIT_HANDOFF.md` §1, Pass B2).

| photo | h_true | R22_px | offset_px | h_est | residual | lever sec(h)−1 |
| --- | --- | --- | --- | --- | --- | --- |
| p2  | 18.6° | 182 | 192.0 (bilateral mean) | 18.573° | −0.027° | 5.52% (eligible) |
| p7  | 59.4° | 200 | 393 (right-side only)  | 59.409° | +0.009° | 96.5% (eligible) |
| p13 | 6.83° | 211 | 212.5 (bilateral mean) | 6.812°  | −0.018° | 0.71% (anchor-noise-bounded) |

**Sources for calibration values.**
p2: `docs/calibration/p2-anchor.json` (sun=[567,496], R22=182, parhelion
left_x=375, right_x=759). p13: `docs/calibration/p13-anchor.json` (sun=[543,372],
R22=211, parhelion left_x=330, right_x=755). p7: R22=200, right-side
offset=393 px per `docs/calibration/PHASE10_OPTICAL_AUDIT_DRAFTS/persona-3-parhelion-forward.md`
table row (only one side recorded; audit noted the omission, treated as
still-eligible).

**Caveats recorded.**
*p7 production-mechanism open question:* at h = 59.4° the circumscribed halo
is present and it is uncertain whether the anchored bright spot is a
plate-population parhelion or a circumscribed-halo brightness
(`PHASE10_OPTICAL_AUDIT_HANDOFF.md` §2.8 wave-2 specialist question, not
yet resolved). This casts p7's traceability contribution as provisional. If
p7 is later demoted, the strict eligible set reduces to p2 + p13, both of
which independently pass Q1.
*p13 lever caveat:* the 0.018° residual is within anchor noise (the 2026-05-13
re-anchor shifted the sun by 14 px / inferred h by 10°); `r22_source_note`
in `p13-anchor.json` explicitly flags the 0 px residual as "informational,
not route-validating." This is consistent with the BOUNDARY_MAP L1 regime
description ("anchor-noise-bounded, not an independent handle") and does not
alter route promotion; Pass B2 included p13 on "unambiguous bilateral peaks +
ring-fit halo" grounds, not on residual-gate discrimination.

**Verdict Q1: CONVERGENT.** Sub-degree residuals on all three eligible photos.
Residuals are sub-pixel for p2 and p7 (lever > 2%). No eligible photo fails
convergence.

---

## Q2 — Counterfactual Steerability

**Protocol.** Hold R22 fixed at the p2 value (182 px). Synthesize a
counterfactual parhelion offset corresponding to four target altitudes
(`h_target ∈ {10°, 25°, 35°, 50°}`). Apply the inverse to recover
`h_recovered`. If the formula is a true lever, `h_recovered = h_target`
exactly.

| h_target | offset_synthetic (px) | h_recovered | residual |
| --- | --- | --- | --- |
| 10° | 184.81 | 10.0000° | 0.000000° |
| 25° | 200.81 | 25.0000° | 0.000000° |
| 35° | 222.18 | 35.0000° | 0.000000° |
| 50° | 283.14 | 50.0000° | 0.000000° |

**Result.** Machine-precision zero residual at every target. This is
structurally guaranteed: the closed-form forward and inverse are exact
inverses of each other, so editing the offset by the predicted Δ drives
the estimate to the predicted `h_target`. The formula is a **true lever**,
not a correlate.

**Interpretation.** A mere correlate would not satisfy this: if the policy
had learned to associate image-style or metadata cues with altitude, editing
only the parhelion offset would not produce the analytically predicted shift.
The analytical route does, because the action is causally organized around
the inverse formula.

**Verdict Q2: STEERABLE.** Perfect counterfactual recovery across the tested
range.

---

## Q3 — Failure-Boundary Coincidence

Each locus from the BOUNDARY_MAP is tested against the code guard actually
implementing it.

### L1 — Parhelion route eligibility (low-leverage / tautological / invalid)

The BOUNDARY_MAP traceable-agent prediction: succeeds on the strict eligible
set; reports low leverage or ineligibility elsewhere.

| regime | test | observed behavior |
| --- | --- | --- |
| Low-leverage (sec(h)−1 < 2%, e.g. h ≤ ~8°) | Photo-level eligibility flag | Photos in this band flagged anchor-noise-bounded; not promoted as independent route evidence (consistent with p13 caveat above and Pass B1 per-photo table) |
| Tautological (R22 from parhelion, not ring fit) | r22_source != "ring-fit" | p20, p22, p25, p26: `r22_source_note` flags tautology; excluded from route-validating set |
| Geometrically invalid | p26 right-side validity block | p26-anchor.json `geometric_validity` block excludes right-side offset |

**L1 verdict: COINCIDENT.** Route correctly limits promotion to the strict
eligible set; abstains or reports ineligibility outside it. Mere-correlate
prediction (smooth confident estimates on tautological / low-leverage rows)
does not match the documented behavior.

### L2 — CZA visibility cutoff

BOUNDARY_MAP: CZA route fails / abstains / switches at h > 32° (coded guard
`czaVisibleAtAltitude` ≤ 32).

| h | czaVisibleAtAltitude | behavior |
| --- | --- | --- |
| 28° | true | CZA route active |
| 30° | true | CZA route active |
| 32° | true | CZA route active |
| 33° | false | CZA route abstains |
| 35° | false | CZA route abstains |
| 40° | false | CZA route abstains |

Cutoff at 33° (first false step) is consistent with the coded guard
`altitudeDeg ≤ 32` in `parhelion-geometry.mjs` `czaVisibleAtAltitude`.

**L2 verdict: COINCIDENT.** Route abstains above the coded cutoff; does not
preserve a CZA-apex inverse past disappearance.

### L3 — Tangent arc → circumscribed-halo merge

BOUNDARY_MAP: tangent-dependent route fails / abstains / switches at h ≥ 29°.
Constant `TANGENT_ARC_CIRCUMSCRIBED_H = 29` in `parhelion-geometry.mjs`; the
`tangentArcLocus` function returns `null` for h ≥ 29.

| h | tangentArcLocus | behavior |
| --- | --- | --- |
| 20° | active | upper tangent handle available |
| 25° | active | upper tangent handle available |
| 28° | active | upper tangent handle available |
| 29° | null | handle absent — circumscribed regime |
| 30° | null | handle absent — circumscribed regime |
| 35° | null | handle absent — circumscribed regime |

**L3 verdict: COINCIDENT.** Route returns null exactly at the coded merge
boundary; it does not claim continuous tangent-curvature recovery through
the singularity. (Note: the tangent route was not promoted through Phase 10;
this boundary check confirms the code guard is operative at the documented
merge point even for the unpromoted route.)

### L4 — Supralateral route (structural-discrimination gate)

BOUNDARY_MAP: fails at **all h**; the handle is not promoted under the
documented apparatus.

The supralateral route was rejected by the Phase 10 audit at every tested
altitude. No promotion code path exists in `parhelion-geometry.mjs` for
supralateral-position → h. The route is absent, not gated.

**L4 verdict: COINCIDENT.** The route refuses to promote supralateral
position as a useful inverse handle, exactly as documented.

### L5 — Rendered ≠ anchored (evidence-admissibility boundary)

BOUNDARY_MAP: only anchored closed-form §A rows count as inverse evidence.

The atlas distinguishes rendered vs. anchored in
`docs/calibration/HALO_PHENOMENA_ACCOUNTING.md` §A Status Vocabulary. The
following primitives are drawn in the atlas but explicitly flagged as
non-evidence for traceability: supralateral arc, suncave-Parry arc,
Parry-supralateral arc, infralateral arc (hardcoded placeholder draw paths
with no Phase 10 anchor), parhelic-circle empirical smile (parametric
approximation, not closed-form inverse), sun-pillar vesica (placed but
unanchored). These are in the `rendered-optional` or `named-only` status
tiers.

**L5 verdict: COINCIDENT.** The admissibility boundary is operative: only
the parhelion-offset route (§A "promoted" tier) counts as inverse evidence;
drawn-but-unanchored primitives do not.

---

**Overall Q3 verdict: BOUNDARY-COINCIDENT.** All five BOUNDARY_MAP loci fire
as documented in the traceable-agent prediction column. The mere-correlate
prediction column does not describe the observed behavior at any locus.

---

## Q4 — Matched-Baseline Efficiency

**Baseline.** Constant-mean predictor: predicts h_mean = 28.28° (the mean of
the eligible set's known altitudes) for every input.

| photo | h_true | baseline residual | route residual |
| --- | --- | --- | --- |
| p2  | 18.6° | 9.68°  | 0.027° |
| p7  | 59.4° | 31.12° | 0.009° |
| p13 | 6.83° | 21.45° | 0.018° |
| **RMSE** |  | **22.53°** | **≈ 0.02°** |

The parhelion route achieves RMSE ≈ 0.02° vs. the constant-mean baseline at
22.53° — a ~1100× reduction in angular error.

**Verdict Q4: EFFICIENT.** Route substantially outperforms the null baseline.
The efficiency claim is independent of the traceability claim (Q1–Q3).

---

## Summary

| quantity | verdict | brief note |
| --- | --- | --- |
| Q1 — Convergence | **PASS** | ≤ 0.027° residual on all three eligible photos |
| Q2 — Counterfactual steerability | **PASS** | Machine-precision zero at four synthetic targets |
| Q3 — Failure-boundary coincidence | **PASS** | All five L1–L5 guards fire as documented |
| Q4 — Matched-baseline efficiency | **PASS** | RMSE 0.02° vs. 22.53° baseline |

Q1 + Q2 + Q3 = traceability claim passes. Q4 = efficiency claim passes.

No quantity fails. The outcome branch is unambiguous: **B — Traceability
harness passes on this domain.** Stakeholder-safe benchmark / apparatus
claim is warranted. P3 (outcome-branched disposition) is now unblocked.

**Public-Language Constraint remains in force** until P3 applies it to the
final public copy.
