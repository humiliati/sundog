# AT-4b Rung 0 — Admission Receipt

> 2026-07-05. Three formation runs of `AT4B_ROLLOUT_DETRENDED_SPEC.md` (truth-only,
> G=300; artifacts `results/proof/at4b0-g300/` and
> `results/proof/at4b0-g300-v12/`). **Non-promotional. No surface or crossover
> number was read in any run — every stop is at input formation.**

## Final verdict: `AT4B_UNPOWERED_INPUT` (horizon stage, v1.2)

| run | label shape | outcome |
| --- | --- | --- |
| v1 | rolling q0.70 of E_low **values** (the scope's literal example) | damp 0.977–0.979 at every τ — degenerate: a max-over-τ almost surely exceeds a value-quantile at beyond-autocorr horizons. `UNPOWERED_INPUT` (horizon stage) |
| v1.1 | rolling q0.70 of the **matched functional** (trailing lookahead-maxes; no future leakage) | **τ=1500 clears: damp 0.395, beyond-autocorr (detrended autocorr ≈ 500 steps)** — but no balanced slice can be formed: `UNPOWERED_INPUT` (slice stage) |
| v1.2 | v1.1 label, **2,000,000-step** truth window, blocked alternating split (50k train/test blocks, 5k guards) | no horizon clears: τ=1500 damp 0.091, τ=2500 damp 0.107, τ=5000 damp 0.103, all beyond-autocorr. `UNPOWERED_INPUT` (horizon stage; final, no v1.3) |

The v1→v1.1 amendment was made at the horizon-formation stage with zero downstream
numbers read (documented in the spec); it did its job — the label powered up. The
owner-commissioned v1.2 amendment was the pre-committed final formation attempt:
measured AT-2 scale plus leakage-safe blocked split. It failed before slice formation.

## v1.1 slice-stage diagnosis (post-verdict diagnostics; superseded by v1.2 final)

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

## v1.2 final diagnosis

The v1.2 run consumed 2,005,001 truth steps in 872s. Detrended autocorrelation first
zero stayed ≈ 500 steps, so all registered horizons were beyond persistence. None
powered the label: 0.091 / 0.107 / 0.103 are all below the registered `[0.20, 0.40]`
damp window. Because the horizon gate failed, no slice, surface, or rollout-carrier
number was read.

Disposition: the pre-committed stop fires. AT-4b closes at `AT4B_UNPOWERED_INPUT`; no
v1.3 and no rung-1 ledger rollout. Combined with AT-4 (`SURFACE_SUFFICIENT` at G=200),
the crossover form is terminal on this C1 substrate: surface-saturated where the dynamics
are regular, input-unformable where the dynamics wander.

Cross-refs: `AT4B_ROLLOUT_DETRENDED_SPEC.md` (v1.2), `AT4B_ROLLOUT_CROSSOVER_SCOPE.md`,
`AT4_CROSSOVER_TRANSPLANT_RECEIPT.md`, `results/proof/at4b0-g300-v12/at4b0_summary.json`,
`AT2_GROWTH_LAW_RECEIPT.md` (the 2.5M-block damp anchor),
`AT3_NUDGING_LEDGER_RECEIPT.md` (the wander's prior appearances).
