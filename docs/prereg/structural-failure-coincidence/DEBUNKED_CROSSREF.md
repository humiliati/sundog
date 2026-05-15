# Cross-Reference: BOUNDARY_MAP.md ↔ docs/Debunked.md

Context-integration step for the structural-failure-coincidence falsification
path. Produced as part of the P1 admission review preparation, per the issue
*"Reality-check and roadmap docs/prereg/structural-failure-coincidence/BOUNDARY_MAP.md
against docs/debunked.md"*.

Cross-referenced documents:
- [`BOUNDARY_MAP.md`](BOUNDARY_MAP.md) — frozen P0 deliverable 2026-05-15
- [`docs/Debunked.md`](../../Debunked.md) — prior best-failure analysis,
  Pushable Occluder / PATH domain

---

## 1. Domain Separation

The two documents operate in **different experimental domains**. They do not
compete; they test different failure modes of different aspects of the Sundog
project.

| | BOUNDARY_MAP.md | docs/Debunked.md |
| --- | --- | --- |
| **Experiment domain** | Halo-geometry altitude inference (geometry / gravity lane) | Photometric-controller beam alignment (PATH lane) |
| **Hidden cause** | Sun altitude `h`, withheld from the controller | Occluder / beam-cone geometry |
| **Indirect signal** | Halo image feature bundle | Detector intensities (photometric ring) |
| **Falsification mode** | Traceability — does the agent use the closed-form inverse, or a learned correlate? | Operational limit — can a flat controller discover a two-stage plan? |
| **Agent type** | Inference / estimation agent; scored on convergence + steerability + boundary coincidence | Photometric extremum-seeking controller; scored on alignment success and push utilisation |
| **Parent roadmap** | `SUNDOG_V_GRAVITY.md` Candidate 13 | `SUNDOG_V_PATH.md`, Pushable Occluder slate |
| **Phase status** | P0 passes; at P1 (falsifier admission review) | Identified as best failure candidate; PATH implementation pending |

---

## 2. Cross-Reference Map

### 2a. Claims in BOUNDARY_MAP.md

| Locus | Core claim | Status in docs/Debunked.md |
| --- | --- | --- |
| **L1 — Parhelion offset route** | Eligible set is strictly p2/p7/p13; low-h photos are anchor-noise-bounded; parhelion-derived-R22 photos are tautological; p26 right side is geometrically invalid | Not addressed — different domain |
| **L2 — CZA visibility cutoff** | CZA usable only below ~32°; a traceable agent must fail, abstain, or switch at the cutoff; a mere correlate continues through it | Not addressed |
| **L3 — Tangent arc merge** | No separate tangent-curvature handle at h ≥ 29°; traceable agent must degrade, abstain, or switch at merge; correlate maintains a continuous estimate | Not addressed |
| **L4 — Supralateral route** | h-spread ~0.5° over h = 0–22°, below visual-edge measurement noise; hard structural-discrimination fail at all altitudes | Not addressed |
| **L5 — Rendered ≠ anchored** | Only anchored closed-form rows count as inverse evidence; drawn-but-unanchored primitives are non-evidence | Not addressed |

### 2b. Claims in docs/Debunked.md

| Claim | Status in BOUNDARY_MAP.md |
| --- | --- |
| Flat photometric controller fails to discover two-stage plan when occluder is present | Not addressed — different domain; orthogonal failure mode |
| Best failure verdict is BOUNDARY FOUND (not BUSTED) — the method hits its honest upper limit | Orthogonal; BOUNDARY_MAP uses a compatible verdict vocabulary (traceable / correlate / fail-and-switch) |
| Mesa Optimization proxy collapse is the deeper scientific falsification (less visual) | Not addressed in BOUNDARY_MAP; both are on the gravity/traceability roadmap as separate candidates |
| Score Aliasing (Occluded Code) is the gimmick-type failure, least important | Not addressed |
| A pushable-occluder failure is visually legible and theorem-legible | Not addressed; BOUNDARY_MAP operates on the analytic/receipt level, not the visual demonstration level |

---

## 3. Does BOUNDARY_MAP.md Rely on Assumptions Challenged or Invalidated in docs/Debunked.md?

**No.**

The five loci in BOUNDARY_MAP.md are grounded entirely in:

- The geometry receipts from the Phase 10 optical audit
  (`docs/calibration/PHASE10_OPTICAL_AUDIT_HANDOFF.md`)
- Coded guards in `public/js/parhelion-geometry.mjs`
- The `docs/calibration/HALO_PHENOMENA_ACCOUNTING.md` §A accounting matrix and
  "Honesty ratchet" vocabulary
- Literature values from *Atmospheric Halos* Ch. 6

None of these foundations are addressed, challenged, or superseded in
docs/Debunked.md. The debunking analysis concerns the photometric controller in
the beam-alignment / occluder domain, which shares no codebase overlap with the
parhelion-geometry inverse.

**Verdict: clean — no assumption inheritance.**

---

## 4. Does BOUNDARY_MAP.md Introduce a Materially New Falsification Path?

**Yes — and it is more foundational than the Pushable Occluder path.**

The distinction is:

- **Pushable Occluder** tests whether a *given controller* can handle a
  multi-stage planning demand — an operational limit assuming the controller
  is already doing something alignment-relevant.
- **Structural Failure Coincidence** (BOUNDARY_MAP.md) tests whether *any*
  controller is traceable at all — whether its behavior is causally organized
  around the closed-form inverse, or is an opaque correlate that happens to
  converge.

A controller that succeeds at Pushable Occluder could still be an opaque
correlate. The structural-failure-coincidence test is the prerequisite check:
if the agent does not fail at precisely the documented inverse singularities, a
positive alignment result is epistemically ambiguous — it could be a
distributional shortcut, not a traceable route.

BOUNDARY_MAP.md is materially new because it provides a **pre-agent
falsification map** written entirely from existing receipts, before any agent
is trained. The Pushable Occluder prereg does not have an equivalent
receipt-cited locus map at P0. That gap is the key structural difference.

---

## 5. Enrichment Opportunity: Combining the Two Paths

The geometry team's structural approach can **enrich** the Pushable Occluder
path, not merely coexist with it:

> Once an agent is trained on the Pushable Occluder task, run the
> structural-failure-coincidence check from BOUNDARY_MAP.md against it.
> If the agent fails to abstain or switch at L1–L5 boundaries while succeeding
> at pushable-occluder alignment, the result is interesting: a correlate can
> solve the operational task. If it coincides with the boundaries, the result
> is stronger: the agent is both operationally capable *and* traceable.

This enrichment makes the two paths additive rather than competing. Recommended
order: structural-failure-coincidence first (more foundational, less training
cost at P0/P1), Pushable Occluder second (more training-heavy, inherits
the traceability check as a supplementary test).

---

## 6. Context Missing from BOUNDARY_MAP.md

BOUNDARY_MAP.md currently does not:

1. Reference docs/Debunked.md or explain that its path is orthogonal to the
   Pushable Occluder path.
2. Note that the verdict vocabulary (BOUNDARY FOUND, STALLED, OPERATING
   ENVELOPE) used in Debunked.md is compatible with, but not the same as,
   the BOUNDARY_MAP traceable / correlate / fail-and-switch vocabulary.
3. Cross-reference the mesa-optimization falsification mode (proxy collapse),
   which is separately identified in Debunked.md as the deeper scientific
   failure and is on the gravity roadmap as a distinct candidate.
4. Explicitly state the enrichment opportunity: the structural-failure-
   coincidence test could be applied post-hoc to any Pushable Occluder
   controller.

This cross-reference document (DEBUNKED_CROSSREF.md) and the amendment added
to BOUNDARY_MAP.md together address items 1–3. Item 4 is noted as a
follow-up action.

---

## 7. Unresolved Validity Risks

| # | Risk | Severity | Notes |
| --- | --- | --- | --- |
| 1 | P1 admission review not yet complete | Medium | BOUNDARY_MAP.md passed P0 (all 5 loci receipt-cited, no row BLOCKED); P1 (independent reviewer pass) is still pending. P2 (agent run) remains blocked on P1. |
| 2 | Coded-guard vs. literature reconciliation | Low | Two recorded reconciliations (L2: coded 32° vs. literature ~32.2°; L4: receipt ~0.5°/h=0–22° vs. prereg ~0.3° span). Neither changes a regime. P1 reviewer should confirm both. |
| 3 | Orthogonality claim first asserted here | Low | The claim that BOUNDARY_MAP.md paths do not depend on any assumption in Debunked.md is made here for the first time. A second reviewer should confirm this is the correct reading during P1. |
| 4 | Mesa-optimization path not cross-referenced | Low | Debunked.md identifies proxy collapse as the deeper scientific failure. BOUNDARY_MAP.md does not address this because it is a separate candidate on the gravity roadmap. No blocking risk, but should be noted in P1 review so a reviewer does not confuse them. |
| 5 | Pushable Occluder results do not validate L1–L5 | Low | The two domains are independent. A successful Pushable Occluder result does not constitute traceability evidence for L1–L5. The two results are additive, not substitutable. |

---

## 8. Prioritization Recommendation

**Primary — advance BOUNDARY_MAP.md to P1 (falsifier admission review)
immediately.**

Rationale:

1. **More foundational.** Traceability (is the agent using the documented
   inverse?) must be established before any alignment result is scientifically
   defensible. The Pushable Occluder result is operationally interesting but
   epistemically ambiguous without it.
2. **P0 already passes.** The boundary map is written from existing receipts
   with no new work. P1 can begin with this document as context integration.
3. **Fastest B-or-D path.** A bad failure-shape prediction at P1 stops the
   program before expensive training. A boundary-coincident agent result at P2
   is stronger than any probe-readout claim.
4. **No assumption dependency on debunked paths.** There is no risk that
   advancing this path reinstates anything that was correctly debunked.

**Status of Pushable Occluder path (from Debunked.md):**

Treat as **provisional / conditional**. The Pushable Occluder is the best
visual failure for the rail and the best operational boundary, but it is
secondary to traceability. Re-examine after structural-failure-coincidence P2
completes. Apply the L1–L5 traceability test to the Pushable Occluder
controller as a supplementary enrichment check.

---

## 9. Follow-Up Roadmap

### 9a. If advancing Structural Failure Coincidence (BOUNDARY_MAP.md)

| step | action | owner |
| --- | --- | --- |
| **P1-a** | Run independent reviewer pass over frozen BOUNDARY_MAP.md; confirm every locus carries all five fields and a named receipt; record reconciliations | reviewer |
| **P1-b** | Confirm L2 (32° vs. 32.2°) and L4 (~0.5° vs. ~0.3°) reconciliations do not change regime classifications | reviewer |
| **P1-c** | Sign off on the orthogonality claim (BOUNDARY_MAP path does not depend on debunked assumptions) with a second-reader confirmation | reviewer |
| **P1-d** | Record P1 outcome as an amendment to BOUNDARY_MAP.md; update phase ladder in SUNDOG_V_GRAVITY.md Candidate 13 | maintainer |
| **P2** | Agent run — score four quantities separately: convergence, steerability, boundary coincidence, efficiency | experimenter |
| **enrichment** | After P2, apply L1–L5 structural-failure-coincidence check to the Pushable Occluder controller as a supplementary traceability audit | experimenter |

### 9b. If documentation / context integration comes first

| step | action | status |
| --- | --- | --- |
| Cross-reference map (BOUNDARY_MAP ↔ Debunked.md) | This document | **done** |
| Amendment to BOUNDARY_MAP.md citing Debunked.md | Added to Amendments section | **done** |
| Footer in docs/Debunked.md pointing to BOUNDARY_MAP path | Added | **done** |
| SUNDOG_V_GEOMETRY.md — note the orthogonality of the structural-failure-coincidence path relative to PATH domain | Open; can be done as a minor note in the Candidate 13 summary | pending |

---

## 10. Conclusion

BOUNDARY_MAP.md introduces a **genuinely distinct, non-duplicated
falsification path**. It does not rely on any assumption challenged or
invalidated in docs/Debunked.md. The two documents operate in different
experimental domains and test different failure modes.

The geometry team's structural contribution is more foundational than the
Pushable Occluder path: it tests whether the agent is traceable at all, before
any controller is trained. The Pushable Occluder tests operational limits of a
controller assumed to be traceable.

**Recommended path: PRIMARY.** Advance BOUNDARY_MAP.md to P1 (falsifier
admission review) immediately, using this document as the context-integration
step. Complete the Pushable Occluder path afterward, enriched by the
traceability check.

---

*Cross-reference first checked 2026-05-15 (PT). Domains confirmed orthogonal;
no overlap found.*
