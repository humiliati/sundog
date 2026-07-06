# AT-4 — Crossover-Transplant Receipt

> 2026-07-04. Run of the frozen `AT4_CROSSOVER_TRANSPLANT_SPEC.md` (G=200, carrier =
> AT-3's confirmed split cell (K_obs=1, μ=10); artifact
> `results/proof/at4-g200/at4_summary.json`). **Non-promotional; licensed grammar;
> relay-form caveat inherited.**

## Verdict: `AT4_SURFACE_SUFFICIENT` — the pre-planned recorded branch

**The label is window-statistic-determined on-slice.** surface_max = **1.000** = ledger
1.000 (ceiling 0.991; scrambled 0.931 ≈ the slice majority exactly; liveness 1.000;
slice n = 2,251 ≥ 800). And the verdict survives the strictest reading: the *purely
permutation-invariant* arms alone reach 1.000 (W=500 moments; quantiles 0.996) — no
appeal to the within-window order of the w-gram arms is needed. At bulk the picture is
the same in miniature: ledger 1.000 vs best surface 0.974 — margin 0.026 < δ everywhere.

**On this cell, the crossover form does not separate — not because the carrier fails,
but because the surface never does.**

## Two honest catches (recorded as findings, not excuses)

1. **The margin-band slice inherited label imbalance: slice-damp = 0.930.** Below-
   threshold samples sit far below e_max (quiet epochs); above-threshold samples cluster
   near it — so the "ambiguous band" is 93/7 and every slice accuracy floats on a 0.930
   majority. Arithmetically the +0.10 crossover gate was nearly unreachable on this slice
   (max possible margin over the majority floor: 0.070). **Transplant gap, owned:** the
   LLM slice discipline always carried a balance gate ([0.40, 0.60]); the AT-4 spec
   omitted it. Any AT-4b would re-register with a balance-gated slice — a new
   registration, not a rescue of this one; the verdict stands as filed (and note the
   bulk read, which is balanced at 0.299, also shows no crossover).
2. **J_q(500) at G=200 is trailing-window-determined.** The W=500 trailing mean reads
   the τ=500 forward-max at 1.000 — the label's horizon sits inside the cell's
   near-regular autocorrelation structure (AT-6: E_low autocorr ≈ 261 steps; no noise
   floor). This is the third empirical mirror of `AveragingDecodability`'s theorem on
   this cell: on a noiseless near-regular substrate, windowed statistics lose amplitude,
   never decodability. **AT-6 taxonomy row added: J_q(500)@G=200 = component/slow-type,
   surface-readable at matched W.**

## Controls (all behaved)

Scrambled = slice majority exactly (0.931) and bulk majority exactly (0.698) — the
temporal-pairing signal AT-3 measured is invisible here *because the surface task is
already saturated*; permutation sanity confirmed moments/quantiles/w1 are strictly
order-blind (perm ≡ slice) while w ≥ 2 grams are local-order statistics (perm < slice),
exactly as designated; liveness 1.000.

## What the slate now says, whole

With AT-4 filed, every runnable entry has a receipt, and the two target forms have
opposite, complementary answers on the C1 cell:

- **The maintained-ledger form separates** (AT-3): decide = 1 < synchronize = 2 — a
  real, controls-certified gap in the proper dynamical gauge, relay-typed.
- **The crossover form does not** (AT-4): no registered label was surface-blocked —
  the surface never fails on this near-regular, noiseless cell, even on the ambiguity
  slice, even restricted to strictly order-blind statistics.

That is precisely the LLM-side arc, mirrored: existence of the split at low dimension
in the right gauge (H2 ↔ AT-3), and no surface-blocked label bank on natural data
(V3 ↔ AT-4) — with the same root cause each time (natural distributions are
surface-dominated; wash-out needs a noise floor). The cross-substrate symmetry the
slate conjectured in F3 is now measured on both sides.

**Remaining:** AT-5 (compute-free Lean module, the σ_traj = ∞ symbolic anchor — open,
buildable any time); AT-7 parked by design. AT-4b has a scoped fork in
`AT4B_ROLLOUT_CROSSOVER_SCOPE.md`: balance-gate-only AT-4b is
`AT4B_BALANCE_ONLY_NOT_RECOMMENDED`, but a mechanism-bearing successor is available as
`AT4B_ROLLOUT_DETRENDED_G300` (sub-sync ledger rollout forecast, horizon beyond
autocorrelation, drift-aware G=300 objective, balance-gated slice). This is a new
registration if commissioned, not a rescue of AT-4.

Cross-refs: `AT4_CROSSOVER_TRANSPLANT_SPEC.md`, `AT3_NUDGING_LEDGER_RECEIPT.md` (carrier),
`AT6_CHARFUN_TYPING_RECEIPT.md` (taxonomy + noise-floor lesson),
`AT1_BOUNDARY_LAYER_RECEIPT.md` (margin-band idiom), `sundogcert/AveragingDecodability.lean`
(the theorem this mirrors), `AT4B_ROLLOUT_CROSSOVER_SCOPE.md`,
`NSE_ATTRACTOR_TAIL_HYPOTHESES.md` AT-4/F3.
