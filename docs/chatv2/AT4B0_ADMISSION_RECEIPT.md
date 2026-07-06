# AT-4b Rung 0 — Admission Receipt

> 2026-07-05. Two runs of `AT4B_ROLLOUT_DETRENDED_SPEC.md` §1 (truth-only, G=300;
> artifacts `results/proof/at4b0-g300/`). **Non-promotional. No surface or crossover
> number was read in either run — both stops are at input formation.**

## Verdict: `AT4B_UNPOWERED_INPUT` (slice stage) — after a v1.1 label fix that worked

| run | label shape | outcome |
| --- | --- | --- |
| v1 | rolling q0.70 of E_low **values** (the scope's literal example) | damp 0.977–0.979 at every τ — degenerate: a max-over-τ almost surely exceeds a value-quantile at beyond-autocorr horizons. `UNPOWERED_INPUT` (horizon stage) |
| v1.1 | rolling q0.70 of the **matched functional** (trailing lookahead-maxes; no future leakage) | **τ=1500 clears: damp 0.395, beyond-autocorr (detrended autocorr ≈ 500 steps)** — but no balanced slice can be formed: `UNPOWERED_INPUT` (slice stage) |

The v1→v1.1 amendment was made at the horizon-formation stage with zero downstream
numbers read (documented in the spec); it did its job — the label powered up.

## The slice-stage diagnosis (post-verdict diagnostics; verdict unchanged)

**Quintile damp across the 500k window: 0.684 → 0.794 → 0.496 → 0.000 → 0.000.**
Train-block damp 0.564; **test-block damp 0.000** — every candidate band, every β, has
test balance 0.000. The excursion process at G=300 is **non-stationary at window scale
even under drift-aware calibration**: sustained zero damp four epochs into the quiet
phase means the rolling quantile keeps chasing a *still-decaying* envelope. This is the
envelope wander's fourth appearance (AT-3 smoke → AT-3 G=300 readout → AT-2's τ=2000
saturation, retroactively → here), now shown to survive a rolling threshold.

**Scale note (why this is a window fact, not a regime impossibility):** AT-2's G=300
discriminator blocks were 2.5M steps and pinned damp at 0.30 — the label powers at
block scales that average over the phase structure. A 500k contiguous window samples
one phase transition and dies.

## Disposition (owner's call — none taken)

1. **Close AT-4b at `UNPOWERED_INPUT`.** Combined with AT-4 (`SURFACE_SUFFICIENT` at
   G=200), the crossover form then closes across both regimes: surface-saturated where
   the dynamics are regular, comparison-unformable where they are not. A coherent
   terminal state for the form.
2. **One bundled v1.2 formation amendment, with a pre-committed stop.** Window
   500k → **2,000,000 steps** (the scale AT-2 measured as label-powering at G=300) +
   **blocked alternating split** (50k train/test blocks, 5,000-step guard gaps =
   max(W, τ) excised at every boundary against window/lookahead leakage; fixed
   assignment, no shuffling). Honest flags: this is the *third* formation iteration —
   each was pre-read and each fixed a measured mechanism, but the discipline cost is
   real; if commissioned, **no v1.3** — a v1.2 formation failure files
   `UNPOWERED_INPUT` as final. Costs: rung 0 ≈ 15 min truth-only; rung-1 rollout
   configs scale ~4× (owner-overnight-sized).

Recommendation, held loosely: option 2 — because the 2M scale is *measured* (AT-2), not
guessed, and because rung 0's stop rule still protects the ledger compute either way.

Cross-refs: `AT4B_ROLLOUT_DETRENDED_SPEC.md` (v1.1), `AT4B_ROLLOUT_CROSSOVER_SCOPE.md`,
`AT4_CROSSOVER_TRANSPLANT_RECEIPT.md`, `AT2_GROWTH_LAW_RECEIPT.md` (the 2.5M-block damp
anchor), `AT3_NUDGING_LEDGER_RECEIPT.md` (the wander's prior appearances).
