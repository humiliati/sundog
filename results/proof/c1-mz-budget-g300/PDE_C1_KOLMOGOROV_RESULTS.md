# PDE C1 Mori-Zwanzig Energy-Budget Receipt (Level 1)

**Status:** MZ_BUDGET_DIAGNOSTIC  (explanatory, NOT promotion-bearing)
**Preset:** `lock_v7_g300`
**Adjudicator:** `mz-budget`

## Decomposition `dE_low/dt = g(Phi_K) + R`

`g` = low dissipation + forcing + band-closed transfer (Phi_K-only); `R` = nonlinear transfer involving >=1 high mode (the Mori-Zwanzig coupling).

## Coupling magnitude (population, robust)

- `rms(R) / rms(dE_low/dt)`: `2.922`
- `rms(R) / rms(g)`: `0.9226`
- `rms(R) / rms(T_low)`: `1`
- samples with `|R| > |g|`: `0.4802`

## Per-sample coupling fraction `rho = |R|/(|g|+|R|)`

- median / mean / p90: `0.4932` / `0.5267` / `0.6932`

## Validation / context

- dissipation `D_low` mean / max: `-0.129` / `-0.1177` (max must be <= 0)
- forcing input `F_low` mean: `0.1713` (should be > 0)
- `damp_fraction`: `0.2688`  samples: `50000`

## Reading

The unresolved coupling `R` is **not** a small minority of the tendency (`rms(R)/rms(dE_low/dt) = 2.92`): instantaneous control-sufficiency must come from cancellation/averaging over the lookahead (Level 2), a subtler story. Explanatory only; C1 status unchanged.

## Files

- `manifest.json`
- `mz-budget-samples.csv`
