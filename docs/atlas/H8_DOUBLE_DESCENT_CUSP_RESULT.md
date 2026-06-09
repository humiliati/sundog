# H8 Result — is double-descent removal a Whitney A3 cusp? (a clean, pre-registered NULL)

> **2026-06-08.** Rank-2 survivor of fresh slate #2 (`wm13nclfe`). Tests whether the regularization-induced
> disappearance of double descent is a Whitney A3 cusp (fold-pair annihilation), using the H4/H5-validated
> jet classifier on a genuine gradient map. NOT public-eligible. Attribution: Belkin 2019 / Mei–Montanari /
> Hastie–Montanari–Rosset–Tibshirani 2019 (double descent); Nakkiran 2020 (regularization removal);
> Thom/Whitney (cusp); the Atlas jet classifier.

## Headline verdict — NULL (informative)

**The regularization-induced disappearance of double descent is NOT a Whitney cusp.** It is a SINGLE
critical point (the bump's max) whose location **slides to γ→∞ while its amplitude → 0** — the peak
escapes/flattens, it does not annihilate against a min at a finite point. There is no max+min fold-pair,
hence no cusp. The recast first *forbade the category error* (the ridgeless peak is a variance pole, not a
fold), and the real test then came out a clean null.

## Stage 1 — the forward model (validated)
Isotropic ridge regression, well-specified, proportional asymptotics (Hastie et al 2019). Effective ridge
`κ` and excess test risk, both closed-form and **smooth** (no training loop):
```
κ(γ,λ) = [(λ+γ−1) + √((1−λ−γ)² + 4λ)] / 2            (the κ>0 root; Σ=I self-consistent ridge)
R(γ,λ; σ²,r²) = (r²·κ² + σ²·γ) / ((1+κ)² − γ)        (excess risk; r²=‖β*‖², σ²=label noise)
```
- **Validated vs Monte-Carlo ridge** (n=300, 40 draws): rel. error ≤ ~1.4% across γ∈{0.5,0.8,1.1,2.0},
  λ∈{0.02,0.2}.
- **Double descent reproduced:** a γ≈1 bump (peak R 15.3 → 4.6 → 1.9 → 1.1) that SHRINKS and VANISHES as λ
  grows (Nakkiran's "optimal regularization removes double descent"); annihilation λ* ∈ (0.2, 0.5).

## Stage 2 — the catastrophe classification (the test)
**Chart `F(γ,λ) = (λ, R(γ;λ))`** — a genuine Lagrangian map (`γ` the state, `R` the potential, `λ` the
control), so `det DF = −∂R/∂γ` and the caustic `φ=0` is the risk-critical-point locus; a cusp would be
`φ=0 ∧ c2=0` = the peak-annihilation. This mirrors the H5 mirage chart exactly. (The vetter's required fix:
NOT the ad-hoc `(p, dR/dp)` map, which is a 2nd-derivative test, not a caustic map.)

| diagnostic | result |
|---|---|
| #critical points of `R(γ)` vs λ (away from the pole) | **1 → 0** (single max vanishing) — NOT 2→0 (no max+min fold-pair) |
| peak γ as λ→λ* | **1.26 → 1.81 → 3.01 → 4.59 → ∞** (slides to infinity) while bump height 0.385 → 0.0016 |
| jet classifier on the chart | **caustic present, 0 cusps, corank-1** |
| A4 calibration control | the Morin swallowtail collapses (|c3|=1.39 near-merge) — our chart has no cusp to compare |

So the disappearance is a **peak-amplitude shrinkage / escape to γ→∞**, not a fold-pair annihilation — no
A3 cusp, no A4, no D4.

## Pre-registered scorecard
| Gate | Result |
|---|---|
| (recast) the ridgeless peak is a POLE, forbidden as a fold-test | honored |
| Stage-1 model validity (closed form vs MC; double descent) | **PASS** |
| **HEADLINE: the annihilation is an A3 cusp** | **NULL** — no max+min fold-pair; single peak slides to ∞ |
| kill criterion "monotone shrink, no fold-pair = clean null" | **TRIGGERED** (the confirmed outcome) |

## Honest boundaries
- This is the **well-specified isotropic ridge** model (the cleanest, validated instance). It has no
  "classical-regime" bias-variance min, so the double descent is a single max — which is exactly why the
  removal is a slide-to-∞, not a fold. A misspecified/anisotropic model with a genuine classical min AND a
  modern-regime min *could* in principle produce a max+min fold (→ cusp); that is a separate, owed test,
  and engineering it would risk constructing the answer. The pre-registered test on the natural model is a
  **null**.
- Forward-only; the chart is a genuine gradient map (not the ad-hoc 2nd-derivative test); the catastrophe
  labels are *computed* by the classifier, not asserted.

## Files
- `scripts/double_descent_cusp.py` — closed-form risk + MC validation + the chart classification.
- `scripts/test_double_descent_cusp.py` — frozen test (7/7 PASS).
- `docs/atlas/FRESH_HYPOTHESES_SLATE_2.md` — the slate this came from (the pre-registration).
