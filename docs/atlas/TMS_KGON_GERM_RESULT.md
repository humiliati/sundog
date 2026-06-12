# RESULT — S3-A5: TMS k-gon transitions germ-classified

**Registered run 2026-06-12 · VERDICT: O2 — MAXWELL LEVEL-CROSSING (a pre-registered SUCCESS
branch; the O1 headline does not fire).** Prereg `TMS_KGON_GERM_PREREG.md` (frozen 2026-06-12,
before any classification); H0 receipt `TMS_H0_HARVEST.md`; scripts `tms_potential.py` (gates) +
`tms_germ_classify.py` (E0/classify/chart) + frozen test `test_tms_germ_classify.py` (**20/20**).
Substrate: the published closed form of arXiv:2310.06301 v1 (Lemma 3.1), validated against an
independent integration path to 1e-15 (V0) and under the rule-2 importance deformation to 1e-15
(V1; the deformation survived its withdraw clause). All kills were live; none fired.

## Headline

**The only transition event the published TMS closed form admits along a potential-deforming
control axis is a Maxwell level-crossing — provably not an elementary catastrophe.** On the
lab-derived importance axis u = I₆ (rule-2-declared; the H0 harvest established the published
potential carries NO continuous knob of its own), the c=6 inventory contains exactly one event in
u ∈ [0.5, 2.0]: the global minimizer switches **5⁺-gon → 6-gon at u\* = 0.658197813**, with both
competing critical points present, essential-corank 0, and λ_min/scale ≥ θ_M = 10⁻³ on both sides
and AT the crossing. No critical point is born, destroyed, or degenerate anywhere in the window
(E0 lift-off probe negative everywhere; chart: **zero Y=0-coincident caustic cells** at full
pinned resolution).

## The rigid-scaling mechanism (why O1 cannot fire on this axis)

Because the importance weight multiplies the ENTIRE output-6 piece of the per-feature sum, and a
dead-feature member's live configuration is u-independent, each k<6 member's landscape **scales
rigidly without deforming**: its relative Hessian spectrum is u-invariant (measured: the 5⁺'s
λ_min/scale = 0.22883 IDENTICAL at u = 0.5 and 2.0 to <1e-9 — pinned in the frozen test) and its
loss is exactly linear in u. The lift-off curvature at the 5⁺ is I₆ × (positive constant), so no
degeneracy is reachable: the axis can only move the Maxwell competition between members, never
fold their critical points. The numerics confirm what the structure forces — the right shape for
a defensible clean null.

## Certificates (all registered, all pinned in the frozen test)

- **Gates:** K1a–c (k-gon roots, 12-gon both system roots, non-existence c∈{9,10,11,13}, all 8
  published loss cross-checks, Cor.-D.2 constants, the exact 1/144 σ⁺ gap); V0 = 1.03e-15;
  V1 = 9.0e-16 + machine identity at I≡1.
- **E0:** one event in-window (no widen needed); lift-off probe collapses at every u.
- **θ_M:** both members corank-0, λ_min/scale ∈ [0.064, 0.229] ≥ 10⁻³ across the window.
- **K2 controls (pinned ng=420 protocol):** A₄ dives to 0.000 < 0.25; A₃ members 0.510 ≥ 0.25;
  D₄ fires corank-2 at 0.0036 < 0.05; A₄ stays corank-1 (0.823); θ_M clearance 0.317 ≥ 0.01.
- **Chart (420×420, x∈[−0.45,0.45] × u∈[0.5,2.0]):** smooth-cell caustic exists (1064 crossings,
  tilt-family structure at Y≠0) but **0 cells coincide with Y=0** — no fold of actual critical
  points; kink cells 4861/176400 (2.8%, chamber boundaries, O4-routed); curl certificate
  **3.36e-9** at refined step (≤1e-7; the 5.9e-5 grid-step FD tail is 4th-order truncation in the
  sharp-curvature region — measured, documented, info-only); symmetric-subspace energy audit
  1.1e-10 (the asymmetric minimizer never finds lower H).

## Answer to the anchor (owner-gated surface)

Timaeus board, "Can we classify further transitions in toy models?" — answered on the null-success
branch for the importance axis: **the 5⁺→6 transition is a level-crossing between coexisting
nondegenerate critical points (the Bayesian-phase-transition picture of the source paper extends
to potential deformation), not an A-series catastrophe; and the rigid-scaling mechanism shows no
per-feature importance axis can ever produce one at a dead-feature member.** Deliverable shape if
unlocked: short note + classification table into their ecosystem. Outreach owner-gated; internal
provenance stays out of any external artifact.

## Honesty notes

1. The control axis is LAB-DERIVED (rule-2, declared everywhere): the published potential has no
   continuous knob, so "no catastrophe along the importance axis" is a statement about the most
   natural published-anchored deformation, not about all conceivable deformations. Axes that
   deform WITHIN-piece structure (e.g. finite sparsity, which couples features) are future work
   and could behave differently — the rigid-scaling argument does not cover them.
2. 4-gon-involving events were not adjudicated (none occurred in-window); they are C¹-not-C²
   (O4-routed by H0 fact) had they appeared.
3. The chart's 1064-crossing caustic at Y≠0 classifies the tilt-extended family; its germ
   structure was not classified (no Y=0 coincidence → not k-gon-transition events; kept as
   context). The scope amendment k∈{4,5} (no 3-gons exist) was declared pre-freeze.
4. Paper findings banked: the 8-gon (l\*,b\*) internal inconsistency resolves to the TABLE value
   (1.5504497, −1.2911926); the paper's second 12-gon system root VIOLATES its own gon-constraint;
   Theorem F.2's 2π/c vs 2π/k typo noted.
5. Owed pre-publication sweep (prereg §6) before any external note: TMS/superposition ×
   {equivariant bifurcation, Golubitsky, symmetry breaking, Γ-invariant unfolding} ∪
   {catastrophe, germ, Thom, cusp, swallowtail}.

*Provenance: H0 harvest (full LaTeX source) → prereg frozen → gates (2 constructor bugs caught by
K1c before any result existed) → E0 (event found where the envelope estimate predicted, 0.665 vs
0.658) → certificates → controls → chart (batched, Newton-polished, kink-audited, refined-step
curl) → frozen test 20/20. One implementation evolution during Stage B, all pre-verdict: batched
evaluator validated vs the gated path at 1.4e-14; energy-based symmetry audit replacing
coordinate-based; refined-step curl certificate (step size unpinned by prereg) after diagnosing
the grid-step tail as truncation.*
