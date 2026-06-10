# H9 pre-registration — a LOAD-BEARING determine latent on a trajectory shadow (the arrow of time)

> **DESIGN LOCKED 2026-06-09, before the frozen run.** The principled frontier opened by the H8 capstone
> theorem (`H8_SHADOW_GEOMETRICITY_THEOREM.md`): H8 proved no load-bearing *resist* on snapshot shadows
> (resist ⟹ phase ⟹ geometric) and flagged (R2) that the only escape — trajectory-irreducible invariants —
> are *determine*-type. H9 tests whether such a **determine-type latent on a trajectory shadow is genuinely
> LOAD-BEARING**: recoverable from the real worldline but not from a natural finite-order matched-statistics
> surrogate. The candidate is the **arrow of time / non-equilibrium probability current** — the cleanest
> trajectory-irreducible determine invariant. A crux (recon + my own verification) already supports it; this
> pre-reg locks the claim, the honest scope, and the kills for a full frozen run + adversarial review.
> NOT public-eligible; a clean result either way is bankable. Attribution: the Shadow/charFun laws + H8
> theorem; Risken (rotational OU / non-equilibrium currents); Schreiber–Schmitz (IAAFT surrogates);
> Crutchfield (causal-states, for the strong-notion follow-on).

## The latent & substrate — the non-equilibrium current (rotational OU)
`dv = −(kI + φJ)v dt + σ dW`, `J` the 90° rotation (rotational Ornstein–Uhlenbeck in ℝ²). The latent
`xc = φ` (SIGNED) is the steady **rotational probability current** — a directed, time-irreversible quantity.
Recovered as the **trajectory-average angular momentum** `⟨v_t × v_{t+1}⟩` (an ergodic time-average ⟹
finite-mean ⟹ determine-type). **Shadow** = a trajectory window of length `W`; lossiness = ensemble over
windows (the current estimator concentrates by LLN — *determine*, not resist).
- **Why the foil is provably fair (the v2/v3 confound, fixed by a symmetry):** time reversal preserves
  *every* symmetric order-2 statistic (per-channel spectrum, amplitude histogram, equal-time + symmetric
  two-time covariance) **exactly**, and flips *only* the arrow. The rotational OU's stationary covariance is
  **φ-independent** (Risken). So φ is **orthogonal to all symmetric order-2 by construction** — not by
  empirical luck. The SIGN of φ is provably invisible to every time-symmetric statistic.

## The claim & the valid load-bearing test (own-R² within-distribution — the v3 fix)
**Claim (weak/order-k notion, honest):** φ is a determine-type latent **load-bearing against the
time-symmetric (equilibrium / order-2 / IAAFT) surrogate class** — the standard time-series null — because it
is precisely the irreversibility content time-symmetric models provably lack.
- **LB := arrow-feature own-R²(signed φ) ≥ 0.70  AND  symmetric-order-2 own-R²(signed φ) ≤ 0.20  AND
  IAAFT-surrogate own-R²(signed φ) ≤ 0.20**, all within-distribution at matched n/W/noise/probe.
- **Crux already obtained** (recon + my verification): arrow own-R²=0.947; symmetric-order-2=0.000;
  IAAFT=0.000; determine concentration std∝1/√W (0.014→0.001); trivial-FAIL (shuffled φ)=0.000. *(Self-caught:
  a "time-reversed trajectory" control I first wrote is NOT a valid foil — the reverse carries −φ, which the
  regression recovers; only the symmetric-feature and IAAFT foils are valid. Reported honestly.)*

### Controls (the H8 battery)
- **Determine-check:** the current estimator must concentrate (std∝1/√W) — the determine signature, the
  *opposite* of the resist gate; if it RESISTED it would be the H8 case, not H9.
- **Anti-vacuity pair:** trivial-PASS (the arrow feature carries φ → high own-R²) AND trivial-FAIL (φ-label
  shuffled → own-R²≈0).
- **NEGATIVE CONTROL (the apparatus is not rigged to always find load-bearing):** the **Hurst exponent of
  fractional Gaussian noise** — a determine-type latent that is FULLY in the power spectrum (Gaussian), so a
  matched-spectrum surrogate RECOVERS it → **NOT load-bearing**. This must come out NULL, proving the test can
  say "geometric" too.

## Honest scope (pre-committed — weak vs strong)
- **Weak (this pre-reg, the frozen claim):** load-bearing vs the **time-symmetric** surrogate class. Genuine
  and non-tautological (the standard IAAFT null is the natural foil; the latent is real non-equilibrium
  content), but the arrow is itself a low-order *antisymmetric 2-time* statistic — so it is **not** load-bearing
  vs a surrogate that also matches the antisymmetric structure. State this plainly; do not overclaim "vs all
  finite order."
- **Strong (the follow-on, NOT this pre-reg):** load-bearing vs **all finite-order** surrogates — needs a
  **causal-state / ε-machine** latent (a process with no finite-order sufficient statistic, e.g. the even
  process; φ = statistical complexity Cμ) tested against an order-k surrogate ladder. Harder (CSSR
  reconstruction, data-hungry); pursued only if the weak result is clean.

## Kill criteria (each bankable)
- **KILL-GEOMETRIC:** symmetric-order-2 or IAAFT own-R²(φ) > 0.20 → φ leaks into the time-symmetric structure
  → not load-bearing (e.g. the discretization O(dt) leak — mitigated by SIGNED φ + small dt + the symmetric-
  order-2 orthogonality witness).
- **KILL-DETERMINE-FAIL:** the estimator does not concentrate / the latent RESISTS → it is not a determine
  latent (it would belong to the H8 resist program) → out of scope.
- **KILL-VACUITY:** the negative control (fGn Hurst) comes out "load-bearing" → the apparatus is rigged →
  invalid, fix it.

## Honest boundaries
- Decisive evidence is the **dissection** (arrow-feature own-R² vs symmetric-order-2 own-R²) + the corr(φ,
  current) witness, NOT the raw-window own-R² (≈0 for both — weak MLP probe, the v5 lesson); lead with the
  dissection.
- The fair-foil rests on the φ-independent stationary covariance (Risken) + signed φ; report the symmetric-
  order-2 own-R² as the standing orthogonality witness at the frozen scale.

## Files (to be produced)
- `scripts/trajectory_determine_shadow.py` — the H9 probe (rotational OU; arrow vs symmetric-order-2 vs IAAFT
  own-R²; determine-concentration; the fGn negative control; anti-vacuity controls). *(verification draft exists)*
- `scripts/test_*` + `results/atlas/h9/` + `docs/atlas/H9_LOADBEARING_DETERMINE_RESULT.md`.
