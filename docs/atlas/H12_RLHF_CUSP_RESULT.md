# H12 RESULT — KL-bandit symmetry breaking: KILL(i), the clean informative null

**Run:** 2026-06-12 (prereg frozen 2026-06-11 at `5d52071d`, before any code; deviations D1–D4 all
committed pre-verdict: `4e4cff6c`, `633e159f`, `fd5cf8ac`, `c7a816ad`, `0e2aa32d`).
**Prereg:** `docs/atlas/H12_RLHF_CUSP_PREREG.md`. Internal slate ref: HS12 of slate 2026-06-10
(gitignored internal document).
**Outcome: KILL(i) — no certified symmetry breaking anywhere.** Both controls passed; the verdict
run (16 β × 8 inits, full battery, wall 5929 s of compute) decided.

---

## Headline

**Under exact deterministic gradient flow, the shared-trunk policy does not break the Z2 symmetry —
anywhere, at any β, from any init.** Across all **128 fresh-init endpoints** on
β ∈ geomspace(0.05, 1.5, 16) × 8 seeds at ε = 0, the order parameter never leaves zero:
max |m| = **2.8e-5 over every endpoint** (certified or not), and **6.9e-11 over the 41 certified
attractors**. Stronger still: every certified endpoint sits ON the analytic Gibbs optimum —
J(Gibbs) − J(attractor) ≤ **2.2e-16** (machine epsilon). The trunk doesn't just approach the
symmetric optimum; where the apparatus can certify at all, it *is* the Gibbs solution.

The banked mechanism-attribution sentence: **deterministic function approximation alone is
insufficient for symmetry-broken collapse at this scale** — the minimal landscape-level mechanism
(a tiny shared trunk under exact KL-regularized policy-gradient flow, with the symmetric optimum
representable and the tabular control globally convergent) produces no broken attractors.
Real-world collapse phenomena must implicate what this model deliberately excludes: sampling
noise, multi-state/sequential structure, finite-step optimizers, or scale. Per the prereg, this
clean null is a SUCCESS; the pitchfork and cusp-wedge stages were never engaged (no breaking), and
the calibrated fold-bisection instrument remains banked via the frozen test for future use.

## Controls (the dichotomy's load-bearing legs — both green)

- **Tabular control (gate iii):** raw-logit softmax policy converges to Gibbs at every coarse β
  (|m − m_gibbs| ≤ 1e-6 across the grid) — the Mei-et-al.-2020 regime reproduced; the trunk is the
  only variable.
- **Representability (control iv):** the same 104-param trunk distills the Gibbs logits at the
  hardest grid point (β = 0.05, logit gap 20 nats) to TV = **2.1e-8** — so the null is NOT
  capacity-limited: the symmetric optimum is representable *and* reached.

## Certification coverage (the honest boundary)

41/128 points certified (grad ≤ 1e-10, λ_max ≤ +1e-9, step-halving, perturb-return). All 87
failures are **'grad' only** — convergence-rate limits, never an unstable Hessian and never a
failed dynamical certificate. The structure: β ≤ 0.195 certifies nothing (the tanh-saturation
slow-manifold regime — gradient flow crawls toward large-weight Gibbs solutions); certification
density rises from β = 0.245 to 6/8 by β ≥ 0.48; seeds 3 and 4 ride slow manifolds at every β
(uncertified 16/16 each). Why the null still stands on the uncertified regime: **every uncertified
endpoint is also symmetric** (|m| ≤ 2.8e-5 — three orders below the 0.05 breaking floor), all
drifting toward the symmetric optimum that the representability control proves finite and
reachable. The certification hole is an instrument boundary (rate, not existence), disclosed here;
KILL(i)'s formal quantifier ("no *certified* breaking") and the substantive picture agree.

## Deviations (all pre-verdict, all committed before the verdict run)

- **D1 (`4e4cff6c`, caught by the frozen test):** the prereg's λ_max ≤ −1e-8 certificate is
  unsatisfiable at functional optima — the parameter Hessian is J^T H_L J + O(gL), rank ≤ 6, so
  98/104 eigenvalues are numerically zero (measured 7.5e-16). Amended: λ_max ≤ +1e-9 (no unstable
  direction) + perturb-return promoted to every verdict-bearing point (the dynamical attractor
  certificate); pinv-Newton polish (gradients to ~1e-16).
- **D2 (`633e159f`, caught by the smoke):** Armijo step cap 1.0 → 256 — the pinned cap made
  small-β runs crawl; Armijo validates every step, so the cap is power, not physics. The smoke's
  uncertified m = −1.000 at β = 0.05 evaporated under the empowered apparatus (a slow transient,
  the H4 25k-step-ambiguity analog).
- **D3 (`fd5cf8ac`) + D4 (`c7a816ad`):** early stall-exit (< 25%-improvement per 60k steps) and
  η-growth only after clean steps — wall-clock only; uncertified-stall reporting is exactly the
  prereg's "non-converged runs carry no weight" category.
- **Checkpointing (`0e2aa32d`, operational):** three run attempts died by external process
  termination (no traceback, no crash event; cause unidentified — plausibly cross-session process
  cleanup on a busy machine). Per-row checkpointing + a relaunch watchdog made the verdict run
  kill-proof; completed rows resume byte-identically.

No deviation touches a kill threshold, a band, a grid, a seed, or the battery's semantics.

## What this banks (and what it does not)

Banked: the clean null with unusual sharpness (certified endpoints AT Gibbs to machine epsilon;
all 128 endpoints symmetric); the validated pitchfork/cusp battery (fold-walker calibrated on the
analytic cusp normal form to 3e-6 — unused here, reusable); the controls pattern (tabular +
representability) that makes the null mechanism-attributing rather than merely negative. NOT
banked: any claim beyond this scale/architecture/flow (a single-state bandit, 104 params, exact
expected gradients, 8 inits); any claim about real RLHF systems — the model's exclusions (sampling
noise, multi-state structure) are now the *implicated* ingredients, which is the point.

In-house priors extended: H4 (grokking, two-basin null) and H8 (closed-form charts, nulls) gain a
third calibrated-battery null; the cat-learning lane's record is now three-for-three "the
catastrophe reading fails at minimal scale under honest instruments".

## Frozen test

`scripts/test_cat_rlhf_cusp.py` — **18/18** (14 apparatus/calibration checks + 4 banked pins
against the committed verdict JSON). Verdict artifact:
`results/atlas/h12/cat_rlhf_cusp_result.json` (full 128-point table with per-point certificates
and failure modes); smoke record at `results/atlas/h12/cat_rlhf_cusp_result_smoke.json`; run logs
+ row checkpoint alongside.

## Follow-on (owner-gated, not this run)

The slate's named escalations, now sharpened by the null: add sampling noise (REINFORCE instead of
exact gradients) at the SAME scale — if breaking appears, noise is the ingredient; add multi-state
sequential structure; scale the trunk. Each is a one-variable step off this clean baseline.
