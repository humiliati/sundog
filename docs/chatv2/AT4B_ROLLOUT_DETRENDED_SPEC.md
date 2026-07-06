# AT-4b — Rollout-Crossover Spec (frozen; both rungs; `AT4B_ROLLOUT_DETRENDED_G300`)

> 2026-07-05. Lift of `AT4B_ROLLOUT_CROSSOVER_SCOPE.md`'s recommended design into its
> frozen pre-registration — **all rung-0 and rung-1 numbers frozen here, before any
> curve exists.** New registration, not a rescue of AT-4 (`AT4_SURFACE_SUFFICIENT`
> stands). `AT4B_BALANCE_ONLY_NOT_RECOMMENDED` is inherited as closed. Non-promotional;
> the scope's claim boundary binds verbatim.

## 1. Rung 0 — data/surface admission (truth-only; agent-run, ~15 min)

- **Cell:** G=300, seed 0, burn-in 100k, truth stream 500,000 steps + 5,001-step
  lookahead tail. No ledgers at this rung.
- **Drift-aware label (the new registered objective; v1.1):** threshold_s = rolling
  q = 0.70 quantile of the **matched functional** — the trailing lookahead-maxes
  {m_τ(p) = max E_low_K3 over (p, p+τ] : p ∈ [s − 50,000, s − τ)} (windows entirely in
  s's past — no future leakage); y(s; τ) = m_τ(s) > threshold_s; margin = m_τ(s) −
  threshold_s. *(v1.1 amendment, 2026-07-05, made at the horizon-formation stage BEFORE
  any surface or slice number was read: the v1 value-quantile shape — the scope's literal
  example — is degenerate for max-functionals at beyond-autocorr horizons: rung 0 v1
  measured damp 0.977–0.979 at every τ and filed `AT4B_UNPOWERED_INPUT`. The
  matched-functional quantile is the AT-3-proven portable-quantile idiom, made rolling.
  The v1 receipt row is retained in the rung-0 receipt.)*
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

---

> **Post-run status (2026-07-05): Rung 0 filed —
> `AT4B_UNPOWERED_INPUT` (slice stage)** in `AT4B0_ADMISSION_RECEIPT.md`.
> The v1.1 matched-functional rolling threshold powered the label at `tau = 1500`
> (`damp = 0.395`) beyond the detrended autocorrelation scale, but no balanced
> held-out slice could be formed: the test block had zero positives. No surface
> or crossover number was read. Live owner fork: close here, or commission one
> bundled v1.2 formation amendment (2M window + blocked alternating split; no
> v1.3).

## 5. v1.2 Formation Amendment (commissioned; pre-committed stop)

> 2026-07-05. Owner selected the one bundled v1.2 formation amendment after the
> v1.1 slice-stage `AT4B_UNPOWERED_INPUT` receipt. This amendment changes only
> the Rung 0 formation scale and split. It is frozen before any v1.2 surface or
> crossover number is read.

**Overrides to §1:**

- Truth stream: `500,000 -> 2,000,000` steps, plus the same `5,001`-step
  lookahead tail. The scale is imported from AT-2's measured G=300 powering
  block, not guessed.
- Split: contiguous 70/30 is replaced by a fixed blocked alternating split:
  50,000-step train block, 5,000-step guard gap, 50,000-step test block,
  5,000-step guard gap, repeated. Even blocks are train, odd blocks are test;
  no shuffling. The guard gap is `max(W, tau) = 5,000`, protecting both trailing
  windows and forward lookahead labels against train/test leakage.
- Everything else in §1 stays unchanged: matched-functional rolling threshold,
  `tau` list `{1500, 2500, 5000}`, damp gate `[0.20, 0.40]`, balanced-slice
  ladder, surface suite, liveness, and admission branch.

**Pre-committed stop:** if v1.2 fails at horizon or slice formation, file
`AT4B_UNPOWERED_INPUT` as final for AT-4b. No v1.3.

**Staged owner command (do not agent-run; expected >10 min):**

```powershell
python scripts/at4b0_admission.py --formation-version v1.2 --out results/proof/at4b0-g300-v12
```

Expected wall-clock: roughly 4x the 500k truth-only rung, around 60 minutes on
this box. If it reaches the surface suite, the receipt should be
`AT4B0_V12_ADMISSION_RECEIPT.md` or an appended v1.2 section in
`AT4B0_ADMISSION_RECEIPT.md`, with the exact branch and a direct statement of
whether any surface numbers were read.
