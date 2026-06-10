# H8 v5 pre-registration — is the obstacle SUBSTRATE-GENERAL? (substrate S5: slow-fast chaos)

> **DESIGN LOCKED 2026-06-09, before any v5 run.** The H8 obstacle is robust across four CGL-spiral attempts
> (v1–v4): every charFun-resist latent was **geometric** (template/superposition-reproducible) → not
> load-bearing. v5 tests whether this is CGL-specific or **substrate-general / fundamental**, on a
> genuinely non-CGL substrate (deterministic chaos), with a SHARP geometric/non-geometric discriminator.
> NOT public-eligible; honest NULL (obstacle substrate-general) is the most-likely outcome and a success;
> a clean escape (a load-bearing non-geometric charFun-resist) would BREAK the conjecture. Attribution: the
> Shadow-Invertibility / charFun laws; v1's matched-spectrum null; the H8 trilogy+v4 receipts.

## The reframed question (what "geometric" means, precisely)
A charFun-resist latent is a **phase**. It is **GEOMETRIC** iff it lives in the **linear/spectral** structure
of the observation — reproducible by a static template OR, equivalently, **recoverable from a
matched-power-spectrum phase-randomized surrogate**. It is **NON-GEOMETRIC** (the escape) iff its recovery
lives in the **nonlinear dynamical** structure that matched-spectrum surrogacy DESTROYS. The escape the saga
needs = a phase that (a) charFun-resists AND (b) is non-geometric (survives matched-spectrum randomization)
AND (c) is load-bearing (requires the dynamics). The conjecture says (a)+(b) cannot co-occur.

## Substrate S5 — slow-fast deterministic chaos (genuinely non-CGL, non-spatial)
A fast chaotic signal whose **nonlinear character** is slowly modulated by a phase φ. Concretely: a
slowly-driven chaotic map/flow `x_{n+1}=f(x_n; r(n))`, `r(n)=r0 + A·cos(φ + ω_slow·n)` sweeping the
nonlinear regime (period-doubling ↔ chaos), with `ω_slow ≪` the fast rate so φ is ~constant over a window.
The latent `xc = φ` (the slow modulation phase, bounded, no wrap). Shadow = a **window** of the fast signal
(covering enough fast cycles to expose the nonlinear structure). Lossiness = mean over `K` subunits at
**jittered φ** (`φ+λ·ξ`). Library-amortized.

## THE DECISIVE TEST (the geometric/non-geometric discriminator) — within-distribution own-R²
The valid test (v3 lesson): own-R² WITHIN each distribution, never cross-R².
- **REAL:** recover φ from the fast-chaos window → own-R²_real.
- **SUP2-SPECTRAL (the geometric surrogate):** each window **phase-randomized** (FFT → randomize phases →
  IFFT), preserving the power spectrum, destroying nonlinear structure → recover φ → own-R²_spec.
- **GEOMETRIC iff** own-R²_spec ≈ own-R²_real (the spectrum carries φ; a static spectral template suffices).
- **NON-GEOMETRIC iff** own-R²_real ≥ 0.70 AND own-R²_spec ≤ 0.30 (a large GAP — φ-recovery NEEDS the
  nonlinear dynamical structure that matched-spectrum surrogacy cannot reproduce).
- Plus **SUP2-SINE** (a clean slow sinusoid at φ — the trivial geometric template) and **A-STATIC** (no
  dynamics) as in v4; and the **corrected anti-vacuity controls** (trivial-PASS: a feature that carries φ
  geometrically scores high; trivial-FAIL: φ randomized → own-R²→0).

## G-KINV (BOTH forms + finite-mean control, no substitution) + broadband-charFun prerequisite
φ+λξ jitter: FORM-1 half-life-vs-K must saturate; FORM-2 fixed-λ cont(λ_test) vs K must stay ≤0.15/not rise;
side-by-side finite-mean LLN control must show the determine signature. Confirm the φ→feature map is
broadband (else single-tone → charFun recurs → φ DETERMINES → KILL-PERIODIC).

## Honest prior (pre-committed)
- **~70% NULL (obstacle substrate-general):** own-R²_spec ≈ own-R²_real → φ is spectral/geometric. Banks the
  strongest version: *the obstacle is about the OBSERVATION (any shadow-recoverable charFun-resist phase is
  geometric), not RD/CGL — load-bearing charFun-resist is impossible for observation-based shadows.*
- **~20% the ESCAPE:** own-R²_real ≫ own-R²_spec, φ charFun-resists, load-bearing → the conjecture BREAKS;
  a load-bearing non-geometric charFun-resist exists → proceed to the full apparatus + adversarial review.
- **~10% confounded** (φ not cleanly recoverable / not charFun) → fix or scope.

## Kill criteria
- **KILL-GEOMETRIC:** own-R²_spec > 0.30 (matched-spectrum recovers φ) → φ spectral/geometric → NULL.
- **KILL-PERIODIC:** φ determines (G-KINV finite-mean / single-tone) → NULL.
- **KILL-VACUITY:** anti-vacuity controls fail → invalid test, fix the probe.

## Crux go/no-go (before the full build)
The decisive matched-spectrum head-to-head (own-R²_real vs own-R²_spec) on a careful slow-fast-chaos setup —
NOT confounded like the first v5 sketch (φ modulates the nonlinear regime; recovery tested via the window,
matched-spectrum surrogate as the geometric foil). If own-R²_spec is high → fast NULL (substrate-general).
Only a clean GAP (real high, spectral low) warrants the full build + adversarial review.

## Files (to be produced)
- `scripts/reaction_diffusion_chaos_shadow.py` — the S5 probe (slow-fast chaos; matched-spectrum + sine +
  static surrogates; own-R² head-to-head; both G-KINV forms + controls).
- `scripts/test_*` + `results/atlas/h8v5/` + `docs/atlas/H8V5_SUBSTRATE_GENERALITY_RESULT.md`.
