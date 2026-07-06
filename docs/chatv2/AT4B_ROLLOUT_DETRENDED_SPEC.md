# AT-4b — Rollout-Crossover Spec (frozen; both rungs; `AT4B_ROLLOUT_DETRENDED_G300`)

> 2026-07-05. Lift of `AT4B_ROLLOUT_CROSSOVER_SCOPE.md`'s recommended design into its
> frozen pre-registration — **all rung-0 and rung-1 numbers frozen here, before any
> curve exists.** New registration, not a rescue of AT-4 (`AT4_SURFACE_SUFFICIENT`
> stands). `AT4B_BALANCE_ONLY_NOT_RECOMMENDED` is inherited as closed. Non-promotional;
> the scope's claim boundary binds verbatim.

## 1. Rung 0 — data/surface admission (truth-only; agent-run, ~15 min)

- **Cell:** G=300, seed 0, burn-in 100k, truth stream 500,000 steps + 5,001-step
  lookahead tail. No ledgers at this rung.
- **Drift-aware label (the new registered objective, frozen):** threshold_s = rolling
  q = 0.70 quantile of E_low_K3(u) over the trailing epoch [s − 50,000, s);
  y(s; τ) = max E_low_K3(u) over (s, s + τ] > threshold_s. Margin_s = that max −
  threshold_s.
- **Horizon (ordered list, first-clearing rule):** τ ∈ {1500, 2500, 5000}; the primary
  horizon = the first τ with label damp ∈ [0.20, 0.40] over post-warmup eval instants.
  None clearing ⇒ `AT4B_UNPOWERED_INPUT`. The detrended autocorrelation time (E_low
  minus rolling median) is measured and reported; the chosen τ must exceed it (report,
  and if violated the horizon list continues — beyond-persistence is the mechanism).
- **Eval instants:** every 50 steps, s ≥ 50,000 (first full epoch); contiguous 70/30
  split, 2,500-step gap, seed 0.
- **Balanced slice (gated by construction):** candidate margin bands |margin| ≤
  P_β(|margin|), β ∈ {0.30, 0.50}, then whole-set; a candidate is admitted iff held-out
  (test-block) balance ∈ [0.40, 0.60] and test mass ≥ 400. If no candidate clears
  as-is, one registered repair: majority-subsample within the band to 50/50 (seed 0).
  Nothing clears ⇒ `AT4B_UNPOWERED_INPUT` (typed, no gate-weakening).
- **Surface suite (strongest registered shot):** AT-4's family verbatim — moments,
  quantiles, binned grams (8 calibration-quantile bins from the first epoch,
  w ∈ {1, 2, 4, 8}, hash ≤ 4,096) — on trailing windows **W ∈ {1000, 2500, 5000}**
  (the horizon grid sits inside the window grid, per AT-6). surface_max = max over
  probes × W on the balanced slice. Liveness: window-mean axis ≥ 0.95 on-slice.
  Reference (reported): Φ_K3(u) snapshot read.
- **Admission branch:** `AT4B_SURFACE_SUFFICIENT_ADMISSION` iff surface_max ≥ **0.90**
  on the balanced slice (the deterministic truth-rollout ceiling is ≈ 1.0, so ≥ 0.90
  leaves < δ of room — stop before ledger compute; the clean terminal negative).
  Else **room exists** → rung 1 unblocked.

## 2. Rung 1 — rollout carrier (only on rung-0 room; owner-run or long background)

- **Carrier:** AOT ledger (AT-3 machinery verbatim) at **K_obs = 1, μ = 10** (primary;
  μ = 50 = the one registered sensitivity); v(0) fresh field seed 1; same 500k window.
  **Carrier read = direct rollout forecast:** from v(s), free-run (no observations)
  τ_primary steps; ŷ = max E_low_K3(rollout) > threshold_s. No training, no probe.
- **Comparators on the same balanced slice/test block:** surface_max (rung 0, frozen);
  **persistence floor** ŷ_pers = max E_low_K3(u) over [s − τ, s] > threshold_s;
  **scrambled-ledger rollout** (seed 2); **K_obs = 2 synced control** (the
  classical-DA reference); truth-rollout ceiling (reported; ≈ 1 by determinism).
  Sync state of the K_obs = 1 carrier must be `state_insufficient` (AT-3 criterion)
  for any crossover claim. Eval instants subsampled every 250 steps for rollouts
  (≥ 1,600 points; slice-intersected).
- **Branch table:** the scope's seven branches verbatim (`AT4B_CROSSOVER_CONFIRMED`
  requires: on the balanced slice, acc_rollout ≥ surface_max + 0.10 ∧ ≥ persistence +
  0.10 ∧ ≥ scrambled + 0.10 ∧ liveness ∧ carrier state-unsynchronized;
  `AT4B_SYNC_VACUOUS` if the advantage exists only at K_obs ≥ 2; `AT4B_ROLLOUT_JOINT_
  INSUFFICIENT` if the sub-sync rollout beats no floor; plus UNPOWERED_INPUT /
  DEAD_APPARATUS / NEG_B as scoped).

## 3. Deliverables

Rung 0: `scripts/at4b0_admission.py` → `AT4B0_ADMISSION_RECEIPT.md` (horizon table,
detrended autocorr, slice construction trail, per-probe surface table, branch).
Rung 1 (conditional): `scripts/at4b1_rollout.py` → `AT4B1_ROLLOUT_RECEIPT.md`.

## 4. Does not claim

The scope's claim boundary verbatim; additionally: the rolling-quantile objective is a
new registration whose relation to frozen J_q is measured, never assumed; the
deterministic truth-rollout ceiling ≈ 1 is a property of a noiseless proxy, named as
such; a rung-0 stop is a first-class outcome, not a failure to launch.
