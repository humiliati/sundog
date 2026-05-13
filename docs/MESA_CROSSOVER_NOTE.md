# Mesa ↔ Geometry Crossover Note

> _Filed 2026-05-13 from Mesa team to Geometry team. Status: load-bearing
> for Phase 10/11 overlay tuning kickoff and for gravity-claim public
> framing in `SUNDOG_V_GRAVITY.md`._

The mesa-optimization roadmap (`SUNDOG_V_MESA.md`) hit its v3.1/v3.2 gate
in the in-vitro lane and produced a methodology surface that transfers
cleanly into geometry. This note captures the transfer in a single place
so Phase 10/11 doesn't have to rediscover it under time pressure.

## The one-liner

The controller mesa cracked open has the same shape your sky does —
**small-dimensional, entangled, holistically read, asymmetric under
inversion** — and the methodological discipline that earned mesa its
anchors is the same discipline that will earn the overlay protocol its
strongest public claim.

---

## Five transferable findings

### 1. The field-shape has now been observed in two substrates

Mesa headline after v3.1/v3.2: the basin-attractor mechanism lives in an
**entangled 5D subspace at `net.7`** that resists every attempt to factor
it — not into top SAE features (v2), not into top-k neurons by L2 (v3.2),
not into PC1-alone or PCs-2-5-alone (v3.1). It is **read holistically or
not at all.**

That is structurally the same object Geometry's atlas committed to when
it stopped treating arcs as independent features and started treating
them as visible portions of a small set of complete implied circles.

Both projects converged, from opposite ends, on the same shape:
**small handful of generators, irreducibly entangled, only legible as a
whole.**

**Why it matters for gravity-language**: the field-not-reward framing in
`SUNDOG_V_GRAVITY.md` now has both an **in-vitro receipt** (mesa) and an
**in-the-wild receipt** (parhelion overlays). That is a quiet but real
ratchet for the public claim — same shape, two substrates, independent
methods.

### 2. Variance is not mechanism

Mesa's PC1 carries **38.8% of the diff variance and almost no patch
effect**. The most visually obvious component was not the load-bearing
one. The team chased SAE features with `|corr| = 0.89` that produced
zero patch effect.

**Geometry analog** for Phase 10/11: the **brightest, easiest-to-overlay
arc is not necessarily the most diagnostic** for the inverse inference
`h = arccos(R₂₂ / parhelion_offset)`. A rich-display vocabulary pass that
ranks primitives by visual contribution will mis-rank them by inferential
contribution.

**Action for Phase 10**: track overlay-fit residuals **per-primitive,
not by total fit**. Explicitly record which optional arcs do inversion
work versus which are decorative. The `do not promote` list in the
current Phase 10 spec is the right shape; this finding sharpens what
"do not promote" should mean.

### 3. Directional asymmetry is structural, not a quirk

Mesa v3.1 found that the **basin-inducing direction generalizes** across
held-out Medium pairs while the **basin-resisting direction is weaker
and more policy-specific**. Forward and inverse are not symmetric.

**Geometry analog**: today, `h → signature` runs cleanly forward
(parhelion offset, halo radius, CZA visibility window all derived from
sun altitude); the inverse `signature → h` runs through **one preferred
route** — `arccos(R₂₂ / parhelion_offset)`.

The mesa lesson is to **suspect that other inversion paths carry
different residual profiles**:

- CZA apex position → h (cutoff geometry; only valid for h < 32.2°)
- Tangent-arc curvature → h (curvature is a non-monotone function of h)
- Supralateral position → h (tangent to 46° halo top; geometric, not refractive)

These are not equivalent for inversion accuracy. **Measure each
explicitly** rather than assuming a unified inversion route.

### 4. Smoke gates and pre-registered negatives

Mesa got **disproportionate value from spec'ing what failure looks like
before running**. v3.2 deliberately stopped at the smoke gate and
recorded a clean negative. That is a publishable result, not a wasted
run.

**Action for Phase 10**: pre-register, **for each candidate vocabulary
primitive, the residual threshold below which it does not get promoted
into the calibrated core**.

The `do not promote` list in the current Phase 10 spec is already
pointing this direction. The mesa pattern is to:

1. Make the failure condition **quantitative** (residual ≥ N pixels at
   anchor scale, across ≥ M of the calibration photos).
2. **Commit to honoring it** before the overlay run, in writing.
3. Treat the negative as a deliverable, not a non-event.

### 5. Non-linear attribution is mandatory for field-shaped objects

The hardest-earned mesa lesson: if geometry ever tries to decompose
the overlay fit into "which arcs contribute how much" via any **linear
additive scheme**, mesa's v3.2 result predicts it will fail in a
specific way — **partial delivery of the right basis still produces
wrong downstream behavior.**

**The atlas already dodges this by construction** (it does not claim
linear decomposition into arc-importance scores; every visible arc is
the upper portion of one full implied circle, all derived from a small
shared parameter set).

**Action**: bank this so future overlay metrics do not accidentally
reintroduce the assumption. If a Phase 10/11 metric ever wants to
report "arc X contributes 23% of the fit", flag it: this is the mesa
trap surfacing in geometry.

---

## What lands where

| Finding | Phase 10/11 deliverable | Gravity claim language |
|---|---|---|
| Same shape in two substrates | — | Add "in-vitro + in-the-wild receipt" pair to `SUNDOG_V_GRAVITY.md` field-shape section |
| Variance ≠ mechanism | Per-primitive residual reporting in `RICH_DISPLAY_OVERLAY_NOTES.md` | — |
| Directional asymmetry | New deliverable: per-inversion-route residual table at Phase 10 gate | Cite as structural property in falsification surface |
| Pre-registered negatives | Quantitative `do not promote` threshold, committed before overlay run | — |
| No linear attribution | Phase 11 metric review: reject any arc-importance score | Cite as why atlas is the right model class |

---

## Open questions for Geometry

1. Should Phase 10 add a `per-inversion-route` residual deliverable
   alongside the existing per-photo overlays?
2. Is the pre-registration of `do not promote` thresholds a Phase 10
   blocker or a Phase 10 gate?
3. Does `SUNDOG_V_GRAVITY.md` want to absorb the "two substrates" framing
   as a separate appendix, or fold it into the existing field-shape
   section?

## Geometry resolution, 2026-05-13

1. **Yes: add the per-inversion-route residual deliverable.** It lives in
   `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md` beside the per-photo
   notes rather than as a standalone document. The table measures parhelion
   offset, CZA apex, tangent-arc curvature, and supralateral position
   separately wherever each route is visible/applicable.
2. **Threshold pre-registration is a Phase 10 blocker for promotion.** An
   overlay run without written thresholds may still be exploratory, but it
   cannot promote primitives into calibrated defaults or logo language.
3. **Keep two-substrate convergence folded into the Gravity Claim for now.**
   It becomes a separate appendix only if Phase 10/11 produce enough
   quantitative route-residual evidence to make the two-substrate framing a
   public proof surface rather than a claim-section support beam.

---

## v3.8 addendum, 2026-05-13

Mesa Phase 6 v3.8 sharpens the crossover note rather than changing its
direction. The per-PC decomposition and signed-effect map add three
constraints Geometry should inherit during Phase 10/11:

1. **Do not equate variance, single-route sensitivity, and full mechanism.**
   PC1 is variance-heavy and now has the largest single-PC P->C max mean
   ablation cost, but PC1 alone still does not reproduce the K=5 patch.
   Geometry should report visual salience, route-local residual/sensitivity,
   and full-overlay fit as separate columns, never as one primitive
   importance score.
2. **Directional asymmetry has internal structure.** P->C is more
   component-partitioned and pair-specific; C->P is more component-shared
   and family-wide. Geometry's per-inversion-route residual table should
   preserve this possibility: one route can be photo- or primitive-specific
   while another route generalizes across the atlas.
3. **Active interference belongs in the `do not promote` logic.** Axis U's
   signed-effect map shows that some neurons help one direction while hurting
   the other. The geometry analogue is a primitive or route that improves one
   overlay target while worsening another; that case should be recorded as a
   conflict, not averaged into a global score.

This addendum lands directly in
`docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md` as a Phase 11 metric-review
rule: no arc-importance percentages, no route aggregation that hides
directional conflicts, and no promotion without per-route residuals.

---

## Follow-on, 2026-05-13: perception roadmap promoted

The "structural transition" thread in this note — mesa's Phase 5→6 move
from describing the cliff behaviorally to localizing it causally — has
now seeded a structurally matched roadmap on the geometry side:
[`SUNDOG_V_PERCEPTION.md`](SUNDOG_V_PERCEPTION.md). That doc reframes
the atlas's "what it does NOT yet model" list as a prediction surface
and instruments three candidate predictions ranked by apparatus cost:

1. Sub-visible parhelic circle continuation (smartphone + polarizer; Phase 1–4).
2. Intersection enhancement at predicted circle crossings (narrowband + long-exposure; Phase 5, reach).
3. Color-order inversion at arc tangencies (spectrometer; Phase 6, explicit reach).

Phase 0 is locked at the smartphone-only apparatus tier with the
predicted-curve software and pre-registration template inherited from
mesa Phase 6 discipline. Phase 7 is the earned-language ratchet that
would add a third receipt — *in-the-wild predictive* — alongside the
existing in-vitro (mesa) and in-the-wild observational (atlas
overlays) ones, on a Phase 4 positive verdict only.

The five transferable findings above remain authoritative for Phase
10/11 of geometry; perception is a separate roadmap, not a successor.

## Reference

- `docs/SUNDOG_V_MESA.md` — the mesa roadmap, v3.1/v3.2 results
- `docs/SUNDOG_V_GRAVITY.md` — gravity claim staging ledger
- `docs/SUNDOG_V_GEOMETRY.md` Phase 10/11 — overlay tuning + logo toolkit
- `docs/SUNDOG_V_PERCEPTION.md` — atlas-as-instrument roadmap (filed 2026-05-13)
