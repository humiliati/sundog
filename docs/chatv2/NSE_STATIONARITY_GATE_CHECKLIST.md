# The Stationarity Input Gate (reusable checklist; H4 of the post-AT slate)

> 2026-07-06. Deliverable of H4 in `NSE_POST_AT_HYPOTHESES_SLATE.md`. This is
> process doctrine distilled from `AT4B_UNPOWERED_INPUT` (three formation
> failures, receipted in `AT4B0_ADMISSION_RECEIPT.md`) and the AT-1/AT-2 atom
> discoveries. It is not a new AT run and not a route around any closed verdict.
> Any future natural-label or crossover registration in this lane imports this
> checklist **verbatim** and freezes its own constants inside the stated windows
> before any number is read.

## The checklist

1. **Blocked split, not contiguous.** Train/calibration and test/eval data are
   separated by guard gaps of at least `max(W, tau)` (largest trailing window,
   largest lookahead horizon). Contiguous 70/30 splits are not accepted for
   decision processes with envelope structure (AT-4b v1.1: the test block
   sampled one phase transition and died).
2. **Blockwise damp table before anything else.** The held-out segment is cut
   into contiguous blocks; the per-block positive-fraction (damp) table is
   computed and reported **before** any probe, surface, or adjudicator read.
3. **Input stop on collapse or drift.** If overall held-out damp falls outside
   the importing spec's registered window (which must sit inside `[0.20, 0.80]`),
   or any block collapses to 0 or 1 (or leaves the spec's registered per-block
   window), the run stops at input with a typed `*_UNPOWERED_INPUT`-style
   verdict. No downstream number is read after an input failure — ever.
4. **Atom clearance beside every power gate.** For every quantile- or
   margin-thresholded functional: mass within the registered epsilon of the
   threshold must not exceed the registered cap (house: `±1e-6`, cap `0.05`),
   with the ball-straddle diagnostic reported (AT-1: the power gate provably
   misses the near-atom degeneracy; AT-2: atom mass can be 0, 0.5, and 1.0 at
   three horizons of the same functional).
5. **Drift-aware is not drift-proof.** A rolling or drift-aware threshold does
   not license skipping items 1–3 (AT-4b v1.1: the envelope wander survived a
   rolling threshold; AT-4b v1.2: it also survived a 4x scale lift).
6. **Scale is measured, not guessed.** If a formation scale must be raised, it
   is imported from a measured anchor (AT-2's powering-block scale), committed
   as a bounded number of amendments with a pre-committed stop, and the stop
   fires without a rescue round.

## Status

`NSE-H4-STATIONARITY-GATE-LANDED` — landed when the first importing spec
cross-links it. **First importer: `NSE_H1_JSELECTOR_SPEC.md` (2026-07-06).**

Cross-refs: `NSE_POST_AT_HYPOTHESES_SLATE.md` (H4),
`AT4B0_ADMISSION_RECEIPT.md`, `AT4B_ROLLOUT_DETRENDED_SPEC.md`,
`AT1_BOUNDARY_LAYER_RECEIPT.md`, `AT2_GROWTH_LAW_RECEIPT.md`,
`NSE_ATTRACTOR_TAIL_SYNTHESIS.md` §6.
