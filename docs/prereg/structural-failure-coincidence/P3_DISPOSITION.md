# P3 Outcome-Branched Disposition

Artifact for: P2 result mapping
Pre-registration: [`README.md`](README.md)
P2 results: [`P2_AGENT_RUN.md`](P2_AGENT_RUN.md)
Roadmap: [`SUNDOG_V_GRAVITY.md`](../../SUNDOG_V_GRAVITY.md) ▸ Candidate 13 ▸ Roadmap
Date: **2026-05-15 (PT)**
Status: **P3 complete.** Outcome branch B selected. Public-Language
Constraint applied. Benchmark / apparatus claim is warranted.

---

## P2 result summary

| quantity | result |
| --- | --- |
| Q1 — Convergence | PASS — ≤ 0.027° residual on all three eligible photos |
| Q2 — Counterfactual steerability | PASS — machine-precision zero at four synthetic targets |
| Q3 — Failure-boundary coincidence | PASS — all five L1–L5 guards fire as documented |
| Q4 — Matched-baseline efficiency | PASS — RMSE 0.02° vs. 22.53° baseline |

---

## Outcome Branching

Mapping P2 against the prereg's Outcome Branching table
(`README.md` §Outcome Branching):

| row | test | P2 result |
| --- | --- | --- |
| Cannot write the boundary map | — | Not applicable — boundary map was written (P0) |
| Agent fails to converge inside eligible regimes | Q1 | **Not selected** — Q1 passes |
| Agent converges but is not steerable | Q2 | **Not selected** — Q2 passes |
| Agent converges and is steerable but crosses failure boundaries without degradation / abstention / switch | Q3 | **Not selected** — Q3 passes |
| **Agent converges, is steerable, and its failure boundary coincides with the closed-form identifiability boundary** | Q1+Q2+Q3 | **SELECTED** |

**Selected outcome:** *Agent converges, is steerable, and its failure
boundary coincides with the closed-form identifiability boundary.*

**Interpretation:** Traceability harness passes on this domain.

**Publication stance:** Stakeholder-safe **B path**: benchmark / apparatus
claim, not universal theorem.

---

## Caveats that do not alter the branch selection

The following recorded caveats are honest nuances within the B path. They
do not move the outcome to a different branch.

1. **p7 production-mechanism question** (`PHASE10_OPTICAL_AUDIT_HANDOFF.md`
   §2.8). The p7 bright spot may be circumscribed-halo brightness rather
   than a plate-population parhelion. If p7 is later demoted by a
   specialist, the strict eligible set reduces to p2 + p13. Q1 still passes
   on p2 alone (lever 5.52%, sub-pixel residual). Branch selection is
   unchanged.

2. **p13 anchor-noise caveat.** p13's 0.018° residual is within anchor
   noise (lever 0.71%). This was noted in P0 and confirmed in P1. Pass B2
   included p13 on bilateral-peak + ring-fit grounds. Q1 remains passing on
   p2 + p7 (both lever > 2%).

3. **Single inverse handle promoted.** Only the parhelion-offset route was
   promoted after the Phase 10 audit. The CZA and tangent-arc routes both
   failed coverage or detection gates. Q3 confirms the documented failure
   boundaries for all three routes, but only the parhelion-offset route
   contributes to Q1/Q2. The benchmark apparatus is therefore
   *parhelion-offset-specific*, not a general multi-route traceability claim.

4. **Forward-richness / inverse-narrowness asymmetry.** The atlas is
   forward-rich on the primitive classes the literature parameterizes from
   `h` alone, and inverse-narrow on the strict 3-photo parhelion-offset
   eligibility set. The traceability claim is scoped to that narrow eligible
   set. This is an honest result, not a weakness to be hidden.

---

## Public-Language Constraint: applied

Per `README.md` §Public-Language Constraint, the following language is now
**permitted** for the parhelion-offset–altitude route on the strict eligible
photo set:

**Use:**
- traceability harness
- indirect-inference alignment benchmark
- hidden-cause recovery from indirect signals
- falsifiable apparatus
- identifiability boundary
- parhelion-offset → altitude inverse on the strict eligible set
- benchmark / apparatus claim scoped to the documented photo set
- failure boundary coincides with the documented closed-form singularities

**Continue to avoid:**
- theorem (this is not a universal theorem; it is a domain result)
- universal alignment proof
- "probe decoded it" as route evidence (not tested in this harness)
- any claim that indirect signals generally beat direct state
- any claim that the traceability result extends to unpromoted routes or
  photos outside the strict eligible set
- language implying p7's production mechanism is confirmed

---

## Stakeholder-safe claim text (B path)

The following statement is warranted by the P0–P3 sequence and obeys the
Public-Language Constraint:

> Sundog is a traceability harness for indirect-inference alignment. On the
> parhelion-offset route, the closed-form inverse `h = arccos(R22 / offset)`
> recovers sun altitude from halo-photo measurements with sub-degree
> residuals on the strict eligible photo set, and fails, abstains, or
> switches handles at every identifiability boundary documented before
> training: the CZA disappearance cutoff (h > 32°), the tangent-arc merge
> (h ≥ 29°), the structural-discrimination gate on the supralateral route
> (all h), and the evidence-admissibility boundary between anchored and
> rendered-but-unanchored atlas primitives. The failure boundary of the
> route coincides with the closed-form identifiability boundary written in
> advance.

This is a benchmark / apparatus claim, not a theorem. It holds on the
tested domain (the documented inverse, the strict eligible photo set, the
five BOUNDARY_MAP loci). It makes no claim about unpromoted routes, new
photos outside the eligible set, or general indirect-signal superiority.

---

## Phase ladder update

P3 completes the program for this pre-registration. The full phase ladder
for Candidate 13 (structural-failure-coincidence benchmark) is:

| phase | status |
| --- | --- |
| P0 — Boundary map (frozen) | **complete — 2026-05-15** |
| P1 — Falsifier admission review | **complete — 2026-05-15** |
| P2 — Agent run, four quantities | **complete — 2026-05-15** |
| P3 — Outcome-branched disposition | **complete — 2026-05-15** |

Program conclusion: **B path — traceability harness passes on this domain.**
