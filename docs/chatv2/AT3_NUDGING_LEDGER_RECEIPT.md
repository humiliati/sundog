# AT-3 — Nudging-Ledger Receipt (the maintained-ledger form, first measurement)

> 2026-07-04. Owner-run sweeps per `AT3_NUDGING_LEDGER_SPEC.md` v1.1 (42 configs × 2
> regimes; artifacts `results/proof/at3-g{300,200}/`). **Non-promotional; 32×32
> truncation; licensed grammar only — "the maintained ledger carries the registered
> decision at a budget below its own synchronization threshold, on this cell."**

## Verdicts (per regime, frozen table)

- **G=200: `AT3_LEDGER_SPLIT_CONFIRMED`** — split-positive at **(K_obs=1, μ=10)** and
  **(K_obs=1, μ=50)**: state-insufficient (median tail err 1.200 / 0.745 ≥ 0.1,
  non-transient) ∧ decision acc **1.000 / 0.997** ≥ max(majority 0.698, scrambled 0.698)
  + 0.10 ∧ twin gap **0.001** ≤ 0.05 ∧ K_obs = 1 < K_sync(μ) = 2. The μ=1 cell fails
  only the twin-match (gap 0.117 — see oddity below).
- **G=300: sync table only — decision readout unpowered** (single-class labels even in
  the 500k window; the envelope wander AT-3's own smoke measured at 20k scales persists
  beyond 500k at this regime). Typed honestly as a power report, not forced into a
  branch. K_sync structure replicates exactly.

## The K_sync table (both regimes, all μ): a cliff at 2

| K_obs | μ=1 | μ=10 | μ=50 |
|---|---|---|---|
| 1 | err 1.42–1.49 (fails) | err 1.17–1.20 (fails) | err 0.75–1.41 (fails) |
| 2–6 | **synchronized (err ~10⁻⁵)** | synchronized | synchronized |

**K_sync = 2 at every μ, both regimes; the transition is `AT3_SHARP`** (a cliff, not a
grade). One observed mode cannot synchronize the field at any tested gain; two can, at
every tested gain.

## The finding: the gap the static gauge could not see

**K_dec = 1 < K_sync = 2 ≤ K\* > 4.** At G=200 the decision is carried at an observation
budget *strictly below* the ledger's own synchronization threshold — the §3.6 vacuity
does **NOT** complete at the ledger level. This overturns the AT-2b gauge specifically:
the static R²-proxy said state "reconstructs" at K=1 (R² 0.96) and called the collapse
vacuous; the actual data-assimilation gauge says synchronization needs K=2. **AT-2b's
collapse was gauge-relative; under the proper dynamical gauge, the decision budget sits
strictly below the state budget.** The three-budget ladder on this cell now reads:
decide = 1, synchronize = 2, twin-certificate wall > 4.

## Mechanism, typed honestly (what the controls certify)

- **The carry is observation-relay + temporal pairing, not emergent ledger computation.**
  AT-2b showed the action is statically readable from the single observed mode; the
  ledger relays that mode while its field stays unsynchronized. This is the split *as
  specified* — but it is the relay form, and the receipt says so.
- **Maintenance is load-bearing:** the scrambled ledger (same marginal observation
  statistics, broken temporal order) reads exactly majority (0.698) — the floor. The
  temporal pairing of the stream, not its values, carries the decision.
- **The carry is decision-typed:** the 9-mode closure-free twin matches the full ledger
  to 0.001 — no high-state reconstruction is involved (that is what the control was
  built to certify).
- **Reported oddity (μ=1):** the crude twin *beats* the full ledger (1.000 vs 0.883) —
  at weak gain, the full ledger's unsynchronized chaos contaminates its own K3
  signature; the 9-mode twin has nothing to contaminate with. An instrument note with
  design value for AT-4's carrier choice.

## Disposition

- The slate's central question has its **first maintained-ledger positive**, in relay
  form, on the C1 cell at G=200: a dynamically maintained shadow carries the registered
  decision below its own synchronization threshold, with the carry certified
  decision-typed and maintenance certified necessary.
- **AT-4 rides this estimator** (as scoped): the crossover transplant now has a live
  carrier, a measured K_sync, and the AT-6-mandated SNR-aware surface family waiting.
- The `navierstokes.html` "shadow (a maintained ledger)" object now exists as a built,
  measured instrument. **No public surface changes from here** (no-publish inherited);
  promo implications are a separate owner-gated decision.
- G=300 readout power: any future G=300 decision read needs either a longer window or a
  drift-aware calibration — recorded as the regime's instrument requirement, not pursued.

Cross-refs: `AT3_NUDGING_LEDGER_SPEC.md` v1.1, `AT2B_GROWTH_LAW_RECEIPT.md` (the static
gauge this overturns), `AT2_GROWTH_LAW_RECEIPT.md` (K\* wall), `AT6_CHARFUN_TYPING_RECEIPT.md`
(envelope wander + AT-4 surface mandate), `NSE_ATTRACTOR_TAIL_HYPOTHESES.md` AT-3/F3.
