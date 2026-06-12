# PREREG — S3-A5: TMS k-gon transitions germ-classified (FROZEN 2026-06-12, before any classification run)

**Header declaration (binding):** substrate = transcription of the arXiv:2310.06301 v1 published
closed form (Lemma 3.1 / eq. (4); polar eq. (17)); K1-gated; **any term not in the paper is flagged
lab-derived per H0.** **RULE-2 DEVIATION (declared):** the published potential carries NO continuous
deforming knob (H0 receipt `TMS_H0_HARVEST.md`), so the control axis is the LAB-DERIVED
feature-importance reweighting H_I = Σᵢ Iᵢ·[published i-th pieces] (Elhage et al. 2022's importance
concept applied to the paper's eq.-(2) one-hot distribution; exact linear reweighting, no new
integrals), anchored at the published object (I≡1 ⇒ H identically). Validation V0/V1 below;
failure ⇒ entry WITHDRAWN per the slate clause. The control WINDOW is likewise lab-chosen (no
published phase diagram sweeps any potential knob): primary u₁ = I₆ ∈ [0.5, 2.0]; E0 widen-once
enlargement = [0.25, 3.0]. **H0-driven scope amendment (pre-freeze):** the published inventory at
c=6 provides the 4→5 and 5→6 transition families (+ σ⁺ variants); the slate headline's k∈{3,4,5}
narrows to k∈{4,5} (no 3-gons exist in the published inventory).

**Standing discipline:** pre-registered KILL; clean null = SUCCESS; forward-generate only;
deterministic + frozen tests; headless CPU; nearest priors named (2310.06301; Watanabe-school RLCT;
Golubitsky–Schaeffer–Stewart for O3a; banked H8/H4/Lowitz).

## §1 Substrate and configuration

r=2, c=6 (the paper's headline configuration; richest inventory: 18 critical-point families).
Implementation in the paper's polar coordinates (lᵢ, θᵢ, bᵢ), Σθᵢ=2π — the O(2) orbit already
quotiented; feature-permutation symmetry handled by fixed wedge ordering and fixed-point-subspace
restriction (never counted as degeneracy). All code numpy/scipy, deterministic, no RNG except fixed
integer seeds for multi-start lists.

## §2 Gates (in order; nothing downstream runs on a failed gate)

- **K1 (reproduction):** `scripts/tms_potential.py` must reproduce the published inventory:
  (l\*,b\*) for k=4..8 to ≤1e-4 abs (8-gon: resolve the paper's 4e-5 internal inconsistency by
  high-precision solution of the published polynomial system; report which value matches); both
  12-gon roots; non-existence spot-checks c∈{9,10,11,13}; L(k,c)=(h_k+c−k)/(3c) cross-check table
  (8 values, ≤1e-5 abs); b⁺=1/(2c) and the 1/144 gap L(5)−L(5⁺). Failure ⇒ K1 ABORT (fix or
  withdraw, never classify).
- **V0 (potential-vs-definition):** H/(3c) at ≥200 fixed pseudo-random w (fixed seed) must match
  deterministic Gauss–Legendre quadrature (≥64 nodes/coordinate) of the integral definition of L(w)
  under eq. (2) to rel ≤1e-6. Catches transcription errors the inventory alone could miss.
- **V1 (deformation-vs-definition):** H_I/(3c) at u₁∈{0.5, 1.5, 2.0} × ≥50 fixed w must match
  quadrature of the importance-weighted eq.-(2) loss to rel ≤1e-6; and H_I(u₁=1) ≡ H to machine
  precision. Failure of V0/V1 ⇒ **WITHDRAWN** (the rule-2 clause).
- **E0 (existence):** the u₁ window must contain ≥1 transition EVENT — global-minimizer identity
  change OR critical-point inventory change (birth/death/merge), established during the K1-style
  inventory continuation stage BEFORE any germ classification unblinds. Zero events in [0.5,2.0] ⇒
  widen ONCE to [0.25,3.0]; still zero ⇒ E0 ABORT, banked as "no transition in the feasible
  closed-form window" (a lesser receipt, not an O-branch success).

## §3 Chart construction (the category-error guard, enforced by construction)

Per event: (a) Hessian eigh of H_I at the event's critical orbit in polar coordinates; isotypic
decomposition under the residual symmetry group; O(2)/permutation zero modes projected out
(NEVER counted); **essential corank** = dim of the remaining near-kernel. (b) If essential corank
= 1: reduced potential by exact partial minimization (Lyapunov–Schmidt via envelope): x = the
near-kernel coordinate; V_red(x; u₁) = min over the complement of H_I, valid only while the
complement block stays positive definite with λ_min/scale ≥ θ_M (else route §4 O3/K5).
**Reduced gradient by the envelope theorem** (∂V_red/∂x = ∂H_I/∂x at the partial minimizer —
analytic, no FD). (c) Chart = the Whitney family map F(x,u₁) = (u₁, ∂V_red/∂x) on a grid, fed to
the frozen `atlas_jet_classify.jet_from_chart` / `cusp_c3` / `corank_from_chart`. (d) **Curl/
potential certificate per chart:** envelope gradient vs 4th-order finite difference of V_red,
rel ≤1e-7 — the chart is a genuine gradient map of an actual potential or it never reaches the
classifier. (e) **Smoothness audit:** indicator-region (chamber) crossings located along all
reduction paths; any event within the kink tolerance (below) of a chamber boundary is an O4
candidate; 4-gon-involving events are O4-audit-first by H0 fact (C¹-not-C² there).

**Chart-intrinsic pinned protocol (frozen now, applied identically to controls and TMS charts):**
ng=420; window-to-fold-pair-separation ratio = 4.26 and ≥98 grid points across the separation at
the reference member (the numbers of the in-context A4 control at h=−0.40: cusps at ±√(−h/6) =
±0.2582, window 2L=2.2). Controls re-run in-context under this exact protocol: Morin-A4 trajectory
(synthetic_swallowtail, h→0 dive) + its h=−0.40 A₃ members + synthetic_umbilic D₄.

## §4 Frozen thresholds and the O-partition (disjoint, exhaustive)

THRESHOLDS (frozen now): **A4-dive ratio 0.25** (|c₃|/|c₃|_generic < 0.25 ⇒ A₄-class dive);
**corank2_rel 0.05**; **√-opening slope window 0.5 ± 0.07 over ≥1 decade of u−u\***
(continuation-topology PRIMARY); **θ_M = 10⁻³** (reduced-Hessian λ_min relative to median |eig|),
with the K2 calibration requirement that control nondegenerate points clear θ_M by ≥10×;
**kink tolerance = 2 grid cells** of the smoothness-audit chart.

- **O1** = ≥1 event adjudicates A₂/A₃/A₄ with BOTH instruments agreeing (continuation topology
  primary, jet confirmatory); reported with sub-label **O1-A₃** (headline-expected: fold-pair
  creation/annihilation, √-opening in window, |c₃| bounded vs the A4 control) vs **O1-other**
  (A₂/A₄ — still a positive elementary answer; keeps the headline's truth value adjudicated).
- **O2** = Maxwell level-crossing: global-minimizer identity changes while BOTH competing critical
  points exist and stay nondegenerate (λ_min/scale ≥ θ_M) on both sides. (The inference axis n is
  a priori Maxwell-only and is NOT a chart axis.)
- **O3a** = symmetry-forced AND equivariantly classifiable: kernel isotypic decomposition reported,
  fixed-point-subspace reduction performed, equivariant class labelled at least to
  pitchfork/transcritical/Z₂-hysteresis granularity (Golubitsky–Schaeffer–Stewart). Null language
  mandated: "not A–D elementary; equivariantly classified as X."
- **O3b** = genuinely codim>3 / unclassified after the equivariant pass.
- **O4** = event within kink tolerance of a chamber boundary (outside Thom's smooth classification;
  named informative null; expected home of 4-gon events per H0).
- **UNADJUDICATED** = fails every adjudicator → reported and COUNTS AGAINST the entry, never dropped.
O2/O3a/O3b/O4 are SUCCESSES (every branch answers the anchor's posted question).

## §5 Kills

K1/V0/V1/E0 as §2. **K2 instrument-transfer kill** (pinned protocol): A₄ control ratio fails to
dive <0.25, A₃ control dips <0.25, D₄ control fails corank-2 at 0.05, A₄ falsely fires corank-2,
or θ_M clearance <10× ⇒ entry DEAD, banked as an instrument finding (bundle with the HS14 ticket).
**K3 headline kill = clean-null SUCCESS:** no O1 event anywhere ⇒ "TMS k-gon transitions are not
A–D elementary; equivariant classification table attached." **K4 consistency kill:** continuation
topology and jet label disagree after grid-refinement doubling ⇒ jet leg quarantined (H4 lesson),
topology-only label banked. **K5 ill-posedness guard:** essential corank ≥2 before 1-D reduction ⇒
D-series/O3 routing only; an A-label may never be forced. DETERMINISM ABORT: reruns not
byte-identical.

## §6 Files, budget, provenance

`scripts/tms_potential.py` (closed form + H_I + gradients/Hessians + projectors + smoothness audit
+ K1/V0/V1), `scripts/tms_germ_classify.py` (continuation, reduction, charts, controls, O-partition),
`scripts/test_tms_germ_classify.py` (frozen: K1 numbers, V0/V1 spot values, control readouts, one
end-to-end event verdict). Result → `TMS_KGON_GERM_RESULT.md`. Budget: 2.5–4 d + 1 d (deformation
clause). External anchor (Timaeus board "Can we classify further transitions in toy models?") —
all outreach owner-gated; internal/ provenance never leaks into external artifacts. Priors named in
the slate entry; the owed pre-publication sweep extends to TMS/superposition × {equivariant
bifurcation, Golubitsky, symmetry breaking, Γ-invariant unfolding} ∪ {catastrophe, germ, Thom,
cusp, swallowtail}.
