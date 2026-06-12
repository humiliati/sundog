# TMS H0 HARVEST RECEIPT — arXiv:2310.06301 v1 (S3-A5 gate 0)

**Executed 2026-06-12, BEFORE the prereg froze** (the binding order). Source of record: the arXiv
e-print LaTeX source (v1, the only version; submitted to ICLR 2024, not accepted/published —
treat as preprint). Full source archived at `internal/harvests/slttms_2310.06301_v1.tex` (gitignored).
Paper: Chen*, Lau*, Mendel, Wei, Murfet, "Dynamical versus Bayesian Phase Transitions in a Toy Model
of Superposition." Full harvest report archived in session task output (agent, 25 tool uses, entire
2,915-line source read incl. appendices).

## The closed form (Lemma 3.1, eq. (4))

Model: f(x,w) = ReLU(WᵀW x + b), W ∈ M_{r,c}(ℝ), b ∈ ℝᶜ, x ∈ [0,1]ᶜ; data = eq. (2): uniform feature
choice 1/c × uniform magnitude U[0,1] (one-hot; "the high sparsity limit of the TMS input
distribution of Elhage et al. 2022"). Population loss L(w) = H(w)/(3c) with H = Σᵢ [δ(bᵢ≤0)H⁻ᵢ +
δ(bᵢ>0)H⁺ᵢ], the piecewise-cubic per-feature pieces of eq. (4) (indicator regions P_{i,j}, P_i,
Q_{i,j}; Nᵢ = (1−‖Wᵢ‖²)² − 3(1−‖Wᵢ‖²)bᵢ + 3bᵢ²; full transcription in the archived harvest +
source). Polar r=2 form: appendix eq. (17) in (lᵢ, θᵢ, bᵢ) with Σθ = 2π. Each dead feature adds
exactly +1 to H (Corollary D.2).

## Knob verdict (the load-bearing H0 question)

**The published potential carries NO continuous deforming knob.** Discrete only: c (features; theory
at c∈{4,5,6} experiments, k=c∈{5..8}, c=12, non-existence checked 9≤c≤203 ∉ 4ℤ) and r (=2
throughout). ALL continuous parameters in the paper (sample size n, prior σ, SGD/SGLD/MCMC
hyperparameters, training time) are INFERENCE-side: the landscape L(w) does not depend on any of
them. The paper's phase diagrams sweep n only (100→1200/2000 at c=6; 100→600 at c=4,5).
**Consequence: the slate's no-published-knob fallback clause triggers** — lab-derived deformation,
rule-2 declaration, validation-or-withdraw (see prereg §2).

## Smoothness/symmetry facts that bind the design

- Generic symmetries: O(2) on the hidden plane (every critical point is a 1-parameter orbit;
  the 5⁺ Hessian's single zero eigenvalue is the O(2) mode) + joint feature permutation.
- H is ANALYTIC at the 5-gon (proved via an explicit open-neighborhood form) and at k=c gons;
  H is C¹ but NOT C² at 4-gons (direction-dependent Hessians; chamber boundaries) → 4-gon events
  are O4/kink-audit candidates by default, never forced into A-labels.
- k<c gons are minimally singular with FLAT directions (dead-feature bᵢ<0 cones contribute the
  constant 1) — the Lyapunov–Schmidt reduction must treat flat directions explicitly.

## K1 reproduction gate (a reimplementation must hit ALL of these; no free parameters)

- (l\*, b\*): 4-gon (1, 0); 5 (1.17046, −0.28230); 6 (1.32053, −0.61814); 7 (1.44839, −0.96691);
  8 (1.55045, −1.29119) ⚠ paper-internal 4e-5 inconsistency vs (1.55041, −1.29122) — RESOLUTION
  PROCEDURE: solve the published polynomial system (x⁶G + x²y²H + 2x⁴yM + y² − x⁴ = 0;
  2x⁸F + 3x⁶yG − x²y³H − 2x⁶ − 2y³ = 0, F/G/H/M as published, α=2π/k, s = unique integer in
  [k/4−1, k/4)) at high precision and report which published value matches — never silently pick.
  12-gon: BOTH roots (1.03322, −0.46654), (1.24975, −0.85483). Non-existence for 9≤c≤203 ∉ 4ℤ
  (spot-check c=9,10,11,13).
- Losses: L(k,c) = (h_k + c − k)/(3c), h₅=0.23738, h₆=0.86746, h₇=1.74870, h₈=2.77311; cross-checks
  L(5,6)=0.06874, L(5⁺,6)=0.06180 (gap exactly 1/144 via b⁺=1/(2c)), L(6,6)=0.04819, L(4,5)=0.06667,
  L(4⁺,5)=0.05667, L(5,5)=0.01583, L(4⁺,6)=0.10417, L(4,4)=0.
- λ: (3c−1)/2 for k=c; 7/8.5/10/11.5 for k=5..8; 5⁺(c=6)=8.5; 4-gon direction-dependent lists
  {4,4.5,5,5.5}, φ=1 {3,...}, etc. (λ is NOT a primary gate quantity for us — listed for context.)
- Bayesian anchors (bonus K1 items, Maxwell-side): n_cr(5→6, c=6)=601 from ΔL=−0.02055, Δλ=1.5,
  Δc=2.7535; n_cr(4⁺→5)≈380; observed MCMC window 600≤n≤700.
- Transcription flags: Theorem F.2 writes θ=2π/c where 2π/k is meant everywhere else (apparent v1
  typo — note, don't silently correct); the polynomial system's first appearance has a "−zy³H(s)"
  typo (later restatements read −x²y³H(s)).

## Deformation decision (rule-2 deviation, declared here and in the prereg header)

Candidates per the harvest verdict: (i) finite-sparsity S (closest to the paper's text, but NO
pre-limit object exists in the paper — it would be derived in-lab with an all-zeros-sample
convention ambiguity, heavy validation); (ii) **feature-importance reweighting (CHOSEN)** — Elhage
et al. 2022's published importance concept, applied to THIS paper's eq.-(2) distribution: the loss is
a per-output-feature sum, so the importance-weighted potential is EXACTLY H_I = Σᵢ Iᵢ·[the published
i-th pieces] — a linear reweighting of Lemma 3.1's own terms, no new integrals, anchored at the
published object (I≡1 ⇒ H exactly). Control axis u₁: I = (1,…,1, u₁) (the importance of one dead-at-
5-gon feature), the axis along which the k-gon ↔ (k+1)-gon competition genuinely deforms.
Validation gates in the prereg (§2): V0 (H itself vs deterministic quadrature of the integral
definition) and V1 (H_I at u₁≠1 vs quadrature of the importance-weighted eq.-(2) loss). Failure ⇒
entry WITHDRAWN per the slate clause.
