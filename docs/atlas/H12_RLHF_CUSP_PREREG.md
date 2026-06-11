# H12 PREREG — KL-bandit symmetry breaking despite a representable analytic optimum

**Frozen:** 2026-06-11, before any H12 code exists. Lane: cat-learning / RLHF-toy dynamics.
Internal slate ref: HS12 in `internal/slates/HYP_SLATE_2026-06-10.md` (gitignored).

**Standing discipline:** pre-registered KILL — a clean null is a SUCCESS; forward-generate only;
deterministic seeded runs + frozen tests; CPU-only, exact expected gradients (no sampling), no
external data; name the nearest prior, state the delta.

**Honest scope, stated up front:** a minimal mechanism-attribution model — a single-state bandit,
a tiny trunk, exact deterministic gradient flow. No claim about real RLHF systems; the question is
whether deterministic function approximation ALONE can sustain certified symmetry-broken attractors
when the objective's optimum is symmetric, analytic, and REPRESENTABLE. Both outcomes bank: a
positive locates collapse in the parameterization's gradient-flow landscape (not the objective,
not sampling noise); the clean null says deterministic approximation alone is insufficient at this
scale, implicating sampling noise / multi-state structure for real collapse.

---

## §0 Spec sharpenings at prereg time (recorded before any run)

1. **Hessian certificate, implementable form.** The slate's "λ_max(Hessian) < −δ" is unsatisfiable
   verbatim for a trunk with a final-layer bias (the softmax global-shift direction is an exact
   zero mode). Fix, pinned: the trunk has **no final-layer bias** (tanh MLP has no other generic
   continuous gauge), and the certificate is **λ_max(∇²J) ≤ −1e-8** (strict local max), computed
   with torch-exact autograd Hessian in float64, cross-validated against the numpy apparatus.
2. **Width vs area exponents.** The slate's "ε-hysteresis ... wedge-width/loop-area scales
   ~ (β*−β)^{3/2}" conflates two observables. The cusp normal form `V = a·m²/2 + m⁴/4 − ε·m`,
   a ∝ −(β*−β), gives fold locations ε_f = ±(2/(3√3))|a|^{3/2}: **wedge WIDTH ∝ (β*−β)^{3/2}**,
   while branch separation ~ 2√|a| makes **loop AREA ∝ (β*−β)²**. The kill binds on the WIDTH
   exponent (bisection-measured, cleanest); the area is reported against its own prediction (2).
3. β-hysteresis stays excluded (supercritical normal form has none — already the slate's fix);
   β-continuation is for branch-following and the exponent fit only.

## §1 Claim

Under exact-expected-gradient KL-regularized reward maximization
`J(θ) = E_{a~π_θ}[r(a)] − β·KL(π_θ ‖ uniform)` on a Z2-symmetric two-good-mode bandit, a small
shared-trunk MLP policy sustains **certified** symmetry-broken attractors below a finite β*:

1. **(Breaking)** at ε = 0, certified equilibria (§2e battery) reached from fresh seeded inits
   with order parameter |m| ≥ 0.05, where m = π(a₊) − π(a₋) and the Gibbs optimum has m ≡ 0;
2. **(Pitchfork)** along the β-continuation branch, |m| ~ (β*−β)^(1/2 ± 0.15) on an
   adaptively-refined window (≥ 12 certified points, (β*−β)/β* ∈ [0.02, 0.3], log-log R² ≥ 0.99);
3. **(Cusp wedge)** at fixed β < β*, ε-direction hysteresis: a bistable wedge whose
   bisection-measured width scales as (β*−β)^(3/2 ± 0.4) over 5 pinned β values (R² ≥ 0.98), with
   a perturbed-continuation control excluding zero-gradient freezing;

— even though the analytic optimum π* ∝ exp(r/β) is symmetric (m = 0 at ε = 0), unique, and the
SAME trunk demonstrably represents it (§2f representability check), and the tabular control (§2g)
converges globally to it (Mei et al. 2020 regime).

## §2 First leg (the only leg)

**(a) Files:** `scripts/cat_rlhf_cusp.py` (driver; numpy float64 hot path with manual gradients,
torch float64 for the Hessian certificate — cross-validated) + `scripts/test_cat_rlhf_cusp.py`
(frozen test) + `results/atlas/h12/`. Receipt `docs/atlas/H12_RLHF_CUSP_RESULT.md`.

**(b) Bandit (pinned):** n = 6 actions. Embeddings e_a ∈ ℝ²: good pair a₊ = (1, +1),
a₋ = (1, −1); distractors (−1, 0), (−0.5, 0), (0, 0), (0.5, 0) — all on the mirror axis. Rewards
r(a±) = 1 ± ε/2, r(distractor) = 0. π_ref = uniform. The Z2 (y → −y) swaps a₊ ↔ a₋, fixes
distractors; at ε = 0 the problem is exactly equivariant, realized in parameter space by negating
the y-column of the first layer (frozen-test check).

**(c) Policy trunk (pinned):** logit(a) = f_θ(e_a); f = MLP 2→8→8→1, tanh, biases on hidden
layers only, **no final bias** (§0.1). 104 parameters. Inits: per-layer N(0, 1/fan_in), seeds 0–7.

**(d) Flow / convergence (pinned):** maximize J by Armijo backtracking gradient ascent (η₀ = 0.5,
shrink 0.5, growth 1.5 capped at 1.0, c = 1e-4), max 5e5 steps, to ‖∇J‖∞ ≤ 1e-8; then Newton
polish (only when λ_max(H) < 0) to ‖∇J‖∞ ≤ 1e-10. Non-converged runs are reported UNCERTIFIED and
carry no weight.

**(e) Certification battery (every point that enters a verdict):**
   - ‖∇J‖∞ ≤ 1e-10;
   - λ_max(∇²J) ≤ −1e-8 (torch-exact, float64);
   - step-halving invariance: plain-GD re-convergence from the endpoint at halved step leaves
     |Δm| ≤ 1e-8;
   - numpy/torch gradient agreement at the endpoint ≤ 1e-10 (apparatus cross-check);
   - (hysteresis steps + spot checks) perturb-return: θ̂ + seeded N(0, 1e-4²) re-converges to
     |Δm| ≤ 1e-6 (2 draws) — the zero-gradient-freezing exclusion (the H4 lesson).

**(f) Representability check (control):** distill the SAME architecture to the Gibbs logits
(r − mean r)/β by full-batch MSE on the 6 actions; PASS iff TV(π_distill, π*) ≤ 1e-3, checked at
the smallest grid β and at the midpoint of the exponent window. FAIL ⇒ any positive verdict
downgrades to "capacity collapse" (named outcome, NOT the claim).

**(g) Tabular control (gate):** same J, policy = 6 raw logits (no trunk). All 8 inits × all coarse
β must reach |m − m_gibbs| ≤ 1e-6 (global convergence regime). FAIL ⇒ apparatus bug — QUARANTINE
(abort, not result).

**(h) Sweep plan:** coarse β grid = geomspace(0.05, 1.5, 16) × 8 inits × ε = 0 (with
m_gibbs(β, ε) computed closed-form alongside everywhere). If breaking is found: β-continuation
from the deepest broken point upward (warm start, adaptive step halving on branch loss) to bracket
β*; β̂* from a linear fit of m² vs β on the upper window half; exponent from log|m| vs
log(β̂*−β) OLS on the §1.2 window. Hysteresis at β_hyst = {0.60, 0.70, 0.80, 0.85, 0.90}·β̂*:
fold locations by warm-started bisection in ε (tol 1e-5, perturb-return at every accepted point);
width w(β) = ε_f⁺ − ε_f⁻; width-exponent OLS over the 5 points; loop area from an 81-point ε
up/down sweep across [−1.2, +1.2]·w/2 (report-only, predicted exponent 2).

**(i) Smoke (pre-registered as NON-VERDICT):** reduced grid/inits/tolerances to validate plumbing;
no number from it enters the verdict.

## §3 Pre-registered outcomes (complementary; no gray zone)

- **KILL (i) — clean informative null:** no certified breaking anywhere on the coarse grid from
  fresh inits (all certified |m| < 0.05 at ε = 0). Banked as: deterministic function approximation
  alone is insufficient for symmetry-broken collapse at this scale — sampling noise / multi-state
  capacity are implicated for real collapse.
- **KILL (ii) — broken but not cusp-class:** breaking exists but the battery fails — pitchfork
  exponent ∉ [0.35, 0.65] or window R² < 0.99 or no measurable wedge (width < 10× bisection tol at
  every β_hyst) or width exponent ∉ [1.1, 1.9]. Banked H4-style as a named null ("symmetry-broken
  attractors of non-cusp class"), with the measured exponents.
- **POSITIVE:** breaking + full battery passes ⇒ certified stable broken attractors under exact
  deterministic gradient flow despite a representable symmetric optimum, with pitchfork + cusp
  signatures — the mechanism-attribution headline.
- **Gate (iii):** tabular control breaks ⇒ QUARANTINE (bug, not result).
- **Downgrade (iv):** representability fails ⇒ "capacity collapse" (named outcome), verdict
  re-labeled, not banked as the claim.

## §4 Honesty constraints

1. The catastrophe call rests on **bistability topology of densely-continuated equilibrium
   charts** (certified branches, folds by bisection); no jet/|c₃| instrument anywhere in the kill
   path (the H4 banked confound: interpolated |c₃| on coarse charts produced spurious ~29000
   readings).
2. Every verdict-bearing number comes from a CERTIFIED point (§2e); uncertified runs are listed in
   the receipt but decide nothing.
3. ΔJ = J(Gibbs) − J(attractor) reported for every broken attractor (the competence cost of the
   landscape artifact).
4. Scope: this scale, this architecture, this budget; mechanism attribution for the minimal model
   only.

## §5 Nearest priors (named), and the delta

**In-house:** H4 grokking-catastrophe (the calibrated two-basin discipline + the
interpolated-|c₃| confound; null), H8 family (closed-form-anchored charts; nulls), HS10/H10 (the
prereg→gate→verdict pattern this run reuses). **External:** Korbak et al. 2022 (RLHF objective =
reverse-KL to the Gibbs target — fixes what "the objective wants"); Mei et al. 2020 (tabular
softmax PG global convergence — the control's theorem); Bishop PRML §10 (mode-seeking VI collapse
folklore); janus 2022 (informal objective-vs-trained collapse gap). **Delta:** certified stable
broken attractors under EXACT deterministic gradient flow despite a REPRESENTABLE analytic
optimum, adjudicated by a calibrated pitchfork+cusp battery with the H4 lessons built in —
neither the known mode-seeking-collapse story (objective-level) nor the known
capacity-collapse story (representation-level), but the landscape level in between.

## §6 Receipt plan

`docs/atlas/H12_RLHF_CUSP_RESULT.md`: outcome (KILL-i / KILL-ii / POSITIVE / quarantine /
downgrade), the (β, init) breaking chart, exponent fits with windows and R², wedge table, control
results, certification statistics (how many points attempted/certified), deviations with reasons,
frozen-test pin list. Memory + slate status update after banking.
