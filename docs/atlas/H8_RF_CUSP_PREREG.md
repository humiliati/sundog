# H8-RF pre-registration — double-descent removal as Whitney A3 cusp: the owed random-features leg

> **2026-06-11. FROZEN BEFORE THE FIRST FULL-GRID RUN.** Slate entry: S3-A2-H8RF
> (`internal/slates/HYP_SLATE_3_EXTERNAL_ANCHORS_2026-06-11.md`, post-fix Fa3/N3/Fe3/I2). Banked
> predecessor: `docs/atlas/H8_DOUBLE_DESCENT_CUSP_RESULT.md` (isotropic ridge, clean NULL: single peak
> escapes to γ→∞; its Honest Boundaries names THIS substrate as the owed follow-up). NOT public-eligible.
> Standing discipline: pre-registered KILL, **a clean null is a SUCCESS**; forward-generate only;
> deterministic seeded runs + frozen tests; cheap headless CPU; name the nearest prior, state the delta.

## 0. External anchor (ADJACENT — archived at write time)

Timaeus project board, "Learning Coefficient Analysis of Double Descent Phenomena" (Hard / Unstarted /
Applied), captured live 2026-06-11 from timaeus.co/projects/double-descent — **five central questions
verbatim**: (1) "How does the learning coefficient vary with model size and training time in double descent
scenarios?" (2) "Can we recover a typical bias-variance tradeoff by plotting loss against the effective
parameter count derived from the learning coefficient?" (3) "Are there detectable differences in learning
coefficient behavior between EWDD and MWDD?" (4) "How does label noise affect the learning coefficient
trajectory in EWDD scenarios?" (5) "Can learning coefficient analysis provide evidence for or against the
hypothesis that EWDD and MWDD have distinct causes?"
Earlier discrepant captures (both carried per the slate fix): writer 2026-06-11 "Can we see differences
between these two kinds of double descent in the learning coefficient?" (a near-paraphrase of Q3, not found
verbatim today); refuter 2026-06-11 = Q1 verbatim.
**ADJACENT: this experiment is a complementary classification layer; it does NOT address the LLC questions
or the epoch-wise axis.** Any outreach is owner-gated and leads with the germ table, never implying the
board item done.

## 1. Claim (binding form, from the slate entry)

In the Mei–Montanari closed-form random-features ridge risk (arXiv:1908.05355, Definition 1 + Theorem 2),
on the pinned slices of §3, the regularization-removal of the model-wise double-descent bump is a fold-pair
annihilation terminating in a Whitney A3 cusp — operationally ALL of:
(i) a bracketed interior max+min pair annihilates 2→0 at finite λ̄_c(τ²) at finite interior ψ₁;
(ii) on the chart F(ψ₁,λ̄)=(λ̄_norm, R_norm) the jet classifier reads ≥1 cusp at the merge with corank-1
(s1_min_rel ≥ 0.05) and |c₃| adjudicated per §6 against the locus-median denominator;
(iii) the fold-pair scaling battery passes (§7);
(iv) every cusp call survives grid doubling and edge-parameter perturbation (§5 K4).
Binding recast preserved verbatim from Slate-2: catastrophe framing applies ONLY to the regularized risk —
**classification domain λ̄ ≥ 0.02**; the ridgeless interpolation peak is a variance POLE, never evaluated.
Clean null = K1 fires → banked as "regularization-removal of double descent is mechanism-class peak-escape,
not a catastrophe, in BOTH canonical closed forms" — a SUCCESS that completes the mechanism-class answer.

## 2. Apparatus transcription (verbatim from arXiv:1908.05355 v[current], extracted 2026-06-11)

**Estimator (Eq. 2):** â(λ) = argmin_a { (1/n) Σ_j (y_j − Σ_i a_i σ(⟨θ_i,x_j⟩/√d))² + (Nλ/d)‖a‖² },
x_i, θ_a ~ Unif(S^{d−1}(√d)), y = ⟨β,x⟩ + ε, ‖β‖² = F₁², ε ~ N(0,τ²).

**Activation coefficients (Assumption 1, Eq. 8–9):** μ₀=E[σ(G)], μ₁=E[Gσ(G)], μ★²=E[σ(G)²]−μ₀²−μ₁²,
**ζ ≡ μ₁/μ★ (Eq. 9)**. ReLU pinned: μ₀=1/√(2π), μ₁=1/2, μ★²=1/4−1/(2π)=0.0908451…, ζ²=2.751938….
**TRANSCRIPTION NOTE (named):** the paper's intro summary (Eq. 4) prints ζ ≡ μ₁²/μ★², conflicting with
Assumption 1 (Eq. 9) ζ ≡ μ₁/μ★ under which Definition 1 is stated. **Pinned: Eq. 9.** The MC gate (§4)
arbitrates: if the Eq.-9 transcription fails MC at the pinned cells, ABORT-A-APPARATUS fires and the
transcription (not the result) is revisited, with the change logged here as an amendment.

**Definition 1 (Eq. 15–20):** ν₁,ν₂ : ℂ₊→ℂ₊ analytic, the unique solution of
ν₁ = ψ₁(−ξ − ν₂ − ζ²ν₂/(1−ζ²ν₁ν₂))⁻¹, ν₂ = ψ₂(−ξ − ν₁ − ζ²ν₁/(1−ζ²ν₁ν₂))⁻¹
with |ν_j(ξ)| ≤ ψ_j/ℑ(ξ) for ℑ(ξ) > C. χ ≡ ν₁(i√(ψ₁ψ₂λ̄))·ν₂(i√(ψ₁ψ₂λ̄)).
E₀ ≡ −χ⁵ζ⁶ + 3χ⁴ζ⁴ + (ψ₁ψ₂−ψ₂−ψ₁+1)χ³ζ⁶ − 2χ³ζ⁴ − 3χ³ζ² + (ψ₁+ψ₂−3ψ₁ψ₂+1)χ²ζ⁴ + 2χ²ζ² + χ² + 3ψ₁ψ₂χζ² − ψ₁ψ₂
E₁ ≡ ψ₂χ³ζ⁴ − ψ₂χ²ζ² + ψ₁ψ₂χζ² − ψ₁ψ₂
E₂ ≡ χ⁵ζ⁶ − 3χ⁴ζ⁴ + (ψ₁−1)χ³ζ⁶ + 2χ³ζ⁴ + 3χ³ζ² + (−ψ₁−1)χ²ζ⁴ − 2χ²ζ² − χ²
B = E₁/E₀, V = E₂/E₀. **Theorem 2 (Eq. 21, linear target F★=0): R_RF = F₁²·B(ζ,ψ₁,ψ₂,λ/μ★²) +
τ²·V(ζ,ψ₁,ψ₂,λ/μ★²).** We sweep **λ̄ ≡ λ/μ★²** directly (the argument B/V take); estimator-λ = λ̄·μ★².

**Solver (branch selection de-fiddled):** at ξ = i·s, s=√(ψ₁ψ₂λ̄) > 0, the ℂ₊ branch is ν_j = i·b_j with
b_j > 0 real, satisfying b₁ = ψ₁/(s + b₂ + ζ²b₂/(1+ζ²b₁b₂)) and symmetrically; χ = −b₁b₂ < 0 real. Damped
(½) fixed-point iteration from b_j = ψ_j/s; convergence tol 1e-14 relative; residual of Eq. 15 checked
< 1e-10 at every evaluated point (violation → ABORT-B, apparatus).

**Chart (banked construction):** F(ψ₁,λ̄) = (λ̄_norm, R_norm), the exact `double_descent_cusp.cusp_chart`
shape (NOT the killed ad-hoc (p,dR/dp) map); fed to the frozen `atlas_jet_classify.{jet_from_chart,
cusp_c3, corank_from_chart}`. Zero new instrument code.

## 3. Pinned slices, grids, windows, constants

- **ψ₂ = 10 (primary geometry; the Fig-3 anchor). F₁² = 1 throughout.**
- **Locus slices: τ² ∈ {0.4, 0.8, 1.2, 1.6, 2.4}. PRIMARY (headline) slice: τ² = 0.8.**
- **Slice-validity rule (pre-pinned):** a slice participates iff (a) the interior max+min pair exists at
  λ̄=0.02 and (b) its λ̄_c ≥ 0.0286 (so the §7 approach window stays ≥ the 0.02 floor). Slices failing
  (a)/(b) are excluded-with-report. **≥3 valid locus slices required for the |c₃| locus-median denominator;
  if <3, the |c₃|-ratio adjudication is VOID and the A3 call rests on corank-1 + K3 alone, |c₃| reported
  descriptively** (the slate already demotes |c₃| to locus-level evidence).
- **Recorded window-validity probe (pre-freeze, 2026-06-11, coarse census only — no classification, no
  scaling, no charts):** pair-at-floor EXISTS at ψ₂=10 for τ² ∈ {0.2: min@4.32/max@6.50, 0.4, 0.8:
  min@2.18/max@8.13, (0.05, 0.1: NO pair — bump already gone at the floor)}; coarse λ̄_c brackets:
  τ²=0.2→(0.02,0.03) **floor-clipped, hence excluded**, 0.4→(0.03,0.04), 0.8→(0.06,0.08), 1.2→(0.08,0.10),
  1.6→(0.10,0.14). The high-SNR floor-clipping (τ²≤0.2) is itself reported: at high SNR the bump dies
  below the pole-recast floor, where germ adjudication is forbidden. ψ₂=3, τ²=0.2 pair-at-floor confirmed
  (robustness slice, report-only).
- **Census grid (per slice, per λ̄):** ψ₁ ∈ [0.25, 60], 12000 uniform points, edges excluded 3 cells;
  critical points = sign changes of the finite-difference dR/dψ₁. Escape tie-break: ONE-SHOT 4× domain
  extension to ψ₁ ∈ [0.25, 240] at the same density.
- **λ̄ sweep:** 400 log-spaced points in [0.02, 12] per slice for the census trace; λ̄_c by bisection on the
  2→0 count transition to relative 1e-8.
- **Chart windows (rule pinned):** ψ₁ ∈ [0.4·ψ₁*, 2.5·ψ₁*], λ̄ ∈ [max(0.02, 0.5·λ̄_c), 1.6·λ̄_c], where
  (ψ₁*, λ̄_c) is the slice's measured merge; ng = 260 (doubling escalation: 520).
- **Shared normalization (the "ONE shared normalization" with constants from the designated reference
  slice):** m_ref, s_ref = mean, std of R over the PRIMARY slice's chart window; λ̄-normalization by the
  PRIMARY slice's window edges. ALL slices' charts use these SAME constants (recorded in output, pinned by
  the frozen test). Y_norm = (R−m_ref)/s_ref, X_norm = (λ̄−lo_ref)/(hi_ref−lo_ref).
- **Approach window (escape diagnostics):** λ̄ ∈ [0.7·λ̄_c_est, λ̄_c_est), 24 points.
- **Edge-parameter perturbation (K4):** all four chart-window factors {0.4, 2.5, 0.5, 1.6} perturbed ±10%
  (one at a time); every cusp call must persist.

## 4. Gates (aborts, not results)

**ABORT-A-APPARATUS** (fallback substrate Hastie et al. 2019 §5 anisotropic ridge ALLOWED, as a new prereg):
- **MC gate:** closed form vs Monte-Carlo of Eq. (2) at the 10 pinned cells, each |closed−MC|/MC < 12%
  (mean over 24 seeded instances; rng(2026); n_test=4000):
  - ψ₂=3, d=100, n=300, τ²=0.5 (the published-agreement geometry, Figs 5/7): (ψ₁,λ̄) ∈
    {0.5, 6.0}×{0.11, 1.1} ∪ {1.0, 12.0}×{0.55} — 6 cells, all away from ψ₁=ψ₂=3.
  - ψ₂=10, d=60, n=600, τ²=0.8 (the primary slice, validated AT the classification regime): (ψ₁,λ̄) ∈
    {3, 25}×{0.06, 0.10} — 4 cells, away from ψ₁=ψ₂=10.
- **Reference-curve gate (Fig 3-left invariants, ρ=5 ⇒ τ²=0.2, ψ₂=10):** (a) at λ̄=0.02 an interior local
  max exists in ψ₁∈(2,20); (b) at λ̄ ≥ 3 zero interior critical points AND dR/dψ₁ < 0 throughout [0.25,60];
  (c) R(ψ₁=0.01; λ̄=0.1, τ²=0.2) within 2% of F₁²=1. **Fig 1 invariant (ψ₂=3, τ²=0, λ=10⁻³ ⇒ λ̄≈0.011,
  evaluated below the classification floor as an APPARATUS check only):** interior local max in ψ₁∈(1,6).
**ABORT-A-RESULT** = an MC-validated solver showing no interior max+min pair anywhere on the pre-registered
grid → this FEEDS K1 as the mechanism verdict. **Fallback firing on the RESULT branch is PROHIBITED BY
NAME: no substrate swap, no slice re-pinning, no domain extension beyond the one-shot tie-break.**
**ABORT-B (controls out of band → NO verdict may be reported):**
- Morin A4 near-merge (h=−0.02) ratio to its own generic (h=−0.40): must be < 0.25 (the dive fires);
- column A3 (prism60, h=30): top |c₃| > 2.0;
- synthetic D4 umbilic (w=0): corank-2 (s1_min_rel < 0.05); A4 swallowtail chart (h=0): corank-1;
- banked isotropic chart ((1.05,4.0)×(0.05,0.6)) re-run through the NEW pipeline: caustic=True, n_cusps=0;
- scaling-battery calibration control (§7) slope outside 0.5±0.05;
- any ν-fixed-point residual ≥ 1e-10 on an evaluated grid point.
All controls at matched ng (260; 520 on escalation).

## 5. Kill lattice (adjudication order: ABORT-A-APPARATUS → ABORT-B → K1 → K4 → K2 → K3)

- **K1 (exhaustive, fires the clean null):** no finite-interior-point 2→0 annihilation of a bracketed
  max+min pair on the PRIMARY slice, for ANY reason — (a) either-member boundary exit (including the
  lower/classical member through ψ₁=0.25+3 cells) before merge; (b) upper-member escape = peak-ψ₁ growth
  ≥2× across the approach window (tie-break: the one-shot 4× extension, then final); (c) interior count ≠ 2
  anywhere in the approach window; (d) merge outside the pinned domain. → cusp claim DEAD; banked as the
  second mechanism-class null; the anchor question answered.
- **K4 (grid-stability void, the H4 lesson):** any cusp call that does not survive ng-doubling AND all
  ±10% edge perturbations → VOID (no verdict from that chart; if the primary slice voids twice, report
  "instrument-unstable on this substrate", no germ claim).
- **K2 (wrong germ):** 0 cusps grid-stably at the merge, OR corank-2 (s1_min_rel < 0.05), OR |c₃| locus
  ratio < 0.5 after the mandated escalation — **the band [0.25, 0.5) after ng-doubling FIRES K2 as
  germ-indeterminate; the A3 claim is dead as stated; banked as "finite merge of unresolved germ class";
  no softer verdict may be reported from the band.** Ratio < 0.25 = A4-dive (also K2, labeled A4-suspect),
  with the dive adjudicated ALONG the λ̄_c(τ²) locus exactly as Morin h is swept.
- **K3 (scaling kill, independent):** battery fail (§7) → any passing K2 cusp call is RETRACTED, not
  softened; verdict "cusp-like merge failing the normal-form scaling".
- Headline A3 = (i)+(ii)+(iii)+(iv) all pass. Single-slice |c₃| support is locus-level only (§3).

## 6. |c₃| adjudication

Generic denominator ≡ **median of top-|c₃| at the merge cusp over all VALID locus slices, every chart under
the §3 shared normalization**. Per-slice ratio = top-|c₃|(slice)/median. Thresholds: ≥0.5 A3-consistent;
[0.25,0.5) escalate-then-K2; <0.25 A4-dive. (Calibration anchors, in-context: Morin A4 dives <0.25 of its
own generic; column A3 sits ≫1 absolute.)

## 7. Scaling battery (K3) + tol_cal

On the PRIMARY slice: bisect λ̄_c (rel 1e-8); measure s(λ̄) = |ψ₁_max−ψ₁_min| at ≥12 log-spaced
ε = (λ̄_c−λ̄)/λ̄_c ∈ [0.02, 0.3]; fit log s vs log ε by least squares. PASS = slope ∈ 0.5 ± tol_cal AND
R² ≥ 0.99. **tol_cal earned on the SAME census root-finder and SAME fitting code path:** run the identical
pipeline on the calibration family V(x;h) = x⁴ + h·x² + 0.15·x (critical points of a generic quartic tilt;
exact ½ law at its fold-pair annihilation), x-grid [−2,2]×12000, h bisected identically; control slope must
land in 0.5±0.05 (else ABORT-B); **tol_cal ≡ max(0.05, 3·|slope_ctrl − 0.5|)**.

## 8. Outputs + frozen test

`scripts/double_descent_cusp_rf.py` (single deterministic run, prints every gate verdict + the §5
adjudication; exit 0 only on headline A3, exit 1 otherwise — mirroring the isotropic leg). Frozen test
`scripts/test_double_descent_cusp_rf.py` pins byte-stably: ζ², χ and R at 3 pinned cells, the 10 MC cell
verdicts (seeded), all ABORT-B control numbers, λ̄_c + ψ₁* (primary), the scaling slope + R², the |c₃|
locus table, and the final verdict token. Result doc `docs/atlas/H8_RF_CUSP_RESULT.md`.

## 9. Priors / citations (named)

Nakkiran–Venkat–Kakade–Ma 2020 (arXiv:2003.01897; proves monotonicity under optimal ridge, never
classifies the removal's germ); Mei–Montanari (arXiv:1908.05355; the closed form, no catastrophe
classification); Hastie–Montanari–Rosset–Tibshirani 2019 (isotropic predecessor's source); banked
in-house: H8_DOUBLE_DESCENT_CUSP_RESULT.md (isotropic null), H5 (chart pattern), H4 (grid-stability
lesson). Prior-art zero (double descent × catastrophe taxonomy) verified at title+abstract tier by two
independent parties 2026-06-11; **full-text re-verification owed before any external priority claim.**
