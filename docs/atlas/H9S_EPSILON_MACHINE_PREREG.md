# H9-strong pre-registration — load-bearing determine vs ALL finite-order surrogates (a causal-state / ε-machine latent)

> **DESIGN LOCKED 2026-06-09, before the frozen run.** The strong-notion follow-on to the banked H9 weak
> positive (`H9_LOADBEARING_DETERMINE_RESULT.md`). The weak result was load-bearing only vs the **time-symmetric**
> class (the arrow is itself a low-order antisymmetric 2-time statistic). H9-strong asks the decisive question:
> can a determine latent be load-bearing vs **EVERY finite-order** surrogate — i.e. live in a genuinely
> infinite-order (causal-state) statistic with **no finite-order sufficient statistic**? NOT public-eligible; a
> clean result either way is bankable. Attribution: Crutchfield & Young (ε-machines / causal states / statistical
> complexity); the parity/sofic process (the canonical strictly-sofic, finite-state, infinite-Markov-order
> example); the Shadow / H8 / H9 lineage.

## The substrate — a PARITY causal state (strictly sofic: 2 causal states, infinite Markov order)
Observable two-channel binary sequence `(b_t, c_t)`, `t = 1..L`:
- `b_t ~ Bernoulli(1/2)` i.i.d. (a fair-coin driver, observed).
- **Hidden causal state** = the running parity `P_t = b_1 ⊕ b_2 ⊕ ... ⊕ b_t ∈ {0,1}` — a **2-state ε-machine**
  (P flips by `b_t` each step) that nonetheless requires the **entire history** to compute (no finite Markov
  order reproduces it: the canonical strictly-sofic process).
- `c_t` = a noisy readout of the causal state with fidelity `φ`: `c_t = P_t` w.p. `(1+φ)/2`, else `1−P_t`.
- **Latent** `φ ∈ [0,1]` = the parity-readout fidelity. As a ±1 correlation, `corr(c_t, P_t) = φ` exactly.

**Why φ is load-bearing vs ALL finite order (the analytic core, not just empirics).** For any fixed finite set
`S` of past indices, the parity over `S`, `⊕_{i∈S} b_i`, is **statistically independent of `c_t`'s φ-signal**:
`P_t = (⊕_{i∈S} b_i) ⊕ (⊕_{i∉S} b_i)`, and the complementary parity `⊕_{i∉S} b_i` is a fair coin independent of
`S`, which **randomizes** the relation. So `c_t` is **uncorrelated with every finite-order function of the
sequence**; the φ-signal is readable **only** by computing the full-history parity (the causal state). This is
the precise sense in which the latent has **no finite-order sufficient statistic**.

## The claim & the order-k surrogate ladder (the strong load-bearing test)
**Claim (strong notion):** `φ` is a determine-type latent **load-bearing against the entire order-k surrogate
class for every k** — recoverable from the real sequence via the causal state, but from **no** order-k
matched-statistics surrogate, for all k (analytically), confirmed empirically across a ladder `k = 1..K`.
- **Order-k surrogate** = a `k`-th order Markov model fit to the real joint sequence (alphabet `{0,1}² = 4`
  symbols) and **resampled** — preserves every `(k+1)`-block joint statistic in expectation, destroys all
  longer-range structure. As `k→∞` it converges to the real process (so the ladder is honest: only a finite-K
  empirical claim + the analytic all-k argument).
- **Feature** (same extraction on real and surrogate): the parity-correlation vector
  `[ corr(c_t, ⊕_{last d} b) for d = 1..D ]  ++  corr(c_t, full running parity P_t)`. The regressor recovers φ
  from whichever entry carries it.
- **LB := real own-R²(φ) ≥ 0.70  AND  order-k-surrogate own-R²(φ) ≤ 0.20 for every k in 1..K.**

### Controls
- **NEGATIVE CONTROL (the ladder is not broken — it CAN detect finite order):** an **order-d latent** `φ_d`
  read off the order-d parity `c_t = (⊕_{last d} b) readout`. The order-k surrogate **must** recover `φ_d` once
  `k ≥ d` (recovery RISES with k, crossing at k=d), while staying low for `k < d`. If the ladder never detects
  the order-d control, the surrogate is too lossy ⟹ invalid. (Run d=1 and d=2.)
- **Sofic witness:** a k-th order Markov model of the REAL sequence fails to reproduce the full-parity
  correlation for every tested k (it would only as k→∞) — the strictly-sofic / infinite-order signature.
- **Anti-vacuity:** trivial-PASS (real recovers φ); trivial-FAIL (shuffled φ-label → own-R²≈0).
- **Determine signature:** the causal-state correlation estimator concentrates (std ∝ 1/√L) — a determine
  latent, the H9 lineage (not a resist).

## Kill criteria (each bankable)
- **KILL-FINITE-ORDER-LEAK:** some order-k surrogate (k in the ladder) recovers φ above 0.20 ⟹ φ leaks at finite
  order k ⟹ load-bearing only vs `< k`, NOT vs all finite order ⟹ downgrade the claim to "order-k" honestly.
- **KILL-LADDER-BROKEN:** the negative control (order-d latent) is NOT recovered by the surrogate at k≥d ⟹ the
  surrogate destroys too much ⟹ the test is invalid (the surrogate isn't a faithful order-k match), fix it.
- **KILL-NOT-DETERMINE / KILL-VACUITY:** the estimator doesn't concentrate, or trivial-FAIL > 0.20.

## Honest scope & boundaries (pre-committed)
- **All-k is ANALYTIC; the empirics test k = 1..K.** The "load-bearing vs every finite order" rests on the
  complementary-parity independence argument above (holds for all k); the ladder `k≤K` is the empirical
  confirmation + the negative control showing the ladder CAN detect finite order. State both; don't claim the
  empirics alone prove all-k.
- **Markov-estimation ceiling:** the order-k surrogate needs `~4^{k+1}` contexts estimated; at fixed L the high-k
  surrogate is data-limited (it eventually memorizes the sequence). Keep K modest (≤ ~4) with L large enough
  that the negative control still resolves; report the context-count vs data margin.
- **This is a designed/synthetic substrate** (a constructed ε-machine), like every Shadow-lane probe — it tests
  whether the framework's strong notion is *instantiable and detectable*, not a claim about natural systems.
- **Lead with the dissection** (real vs the surrogate-ladder curve + the negative-control crossing), not the raw
  recovery number — the H9 discipline.

## Files (to be produced)
- `scripts/epsilon_machine_shadow.py` — the H9-strong probe (parity ε-machine; order-k Markov surrogate ladder;
  parity-correlation feature; order-d negative control; sofic witness; determine concentration).
- `scripts/test_epsilon_machine_shadow.py` + `docs/atlas/H9S_EPSILON_MACHINE_RESULT.md`.
