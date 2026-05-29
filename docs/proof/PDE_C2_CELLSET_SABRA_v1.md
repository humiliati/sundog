# PDE C2 Cell Set v1 — Stationarity-Gated Sabra Burst Cell

> **Design proposal for sign-off**, filed 2026-05-29. Re-pose of
> [`PDE_C2_CELLSET_SABRA_v0.md`](PDE_C2_CELLSET_SABRA_v0.md), whose
> headline run returned `PDE-C2-DEFERRED-BASERATE` with a *block-dependent*
> burst rate (train 0.138, val 0, test 0) → the trajectory was not
> stationary / not representatively sampled across the labelled span.
> v1 fixes stationarity rather than retuning the burst threshold (a v0
> retune would be `PDE-C2-NEG-B`). **Status: proposed, not built, not
> run.** §8 lists the sign-off decisions; build/run only after sign-off.

## 1. Root-cause read of the v0 deferral

The v0 base rates ran **0.138 → 0 → 0** across temporally-ordered
blocks (calib → train → val → test). The monotonic decline to zero is
the signature of **energy drift**, not random rare-cluster scatter
(which would give non-monotonic noise). Most likely the constant
*additive* forcing `f_1 = const` on shell 1 does not drive a
statistically steady cascade on the labelled timescale: the large-scale
energy drifts, so a calib-derived `E_burst` (set early, high) is rarely
reached in later blocks. The integrator is verified correct (inviscid
energy conservation), so this is a forcing/stationarity issue, not an
integration bug.

Two fixes are needed, and "longer blocks" alone is **not** sufficient if
the cause is drift — the forcing must produce a stationary state first.

## 2. Fix A — forcing scheme for a statistically steady cascade

**Default (recommended): fixed-amplitude forcing on shell 1.** Hold
`|u_1|` at a fixed value each step (renormalise the forced shell's
modulus, let its phase evolve), the standard shell-model device for a
statistically steady energy input. This reliably removes large-scale
energy drift. The inviscid energy-conservation self-test is unaffected
(it tests the nonlinear term with forcing off).

**Retained alternative (sign-off option):** keep v0's constant additive
`f_1` but with a verified-long warmup (only if the §3 diagnostic shows
it actually plateaus). Constant-power (fixed energy-input-rate) forcing
is a third option.

The forcing scheme is the **load-bearing sign-off decision** (§8.1): it
is a model change, so it is named, not chosen unilaterally.

## 3. Fix B — stationarity diagnostic (cheap, run FIRST)

Before any labelled cell, run a **diagnostic trajectory** under the
chosen forcing and report, downsampled, the time series of:

- total energy `Σ_n |u_n(t)|^2`;
- `max_n |u_n(t)|^2` (the burst observable).

From these, pre-registered estimates:

- **equilibration time `T_eq`**: the time after which a sliding-window
  mean of total energy is flat to a pinned tolerance;
- **burst recurrence time `T_burst`**: mean inter-exceedance interval of
  `max_n|u_n|^2` above a provisional high quantile.

This is the cheap analogue of the base-rate-first discipline: diagnose
the stationarity and burst timescales *before* committing a full cell.
It also distinguishes drift (energy never flattens) from rare-cluster
intermittency (flattens, but `T_burst` is long).

## 4. Cell parameters set by a pre-registered rule

The labelled cell's warmup and block lengths are then pinned by a rule
fixed here (concrete numbers go in a v1 patch *after* the diagnostic,
before the verdict-bearing run):

```text
warmup      >= 3 * T_eq
each block  >= 50 * T_burst   (so each of train/val/test contains
                               O(50) burst events)
calib block >= 50 * T_burst
gaps        >= 5 * T_burst
```

Everything else (Sabra `N=22`, `λ=2`, `k0=2^-4`, `ε=1/2`, `ν=1e-7`,
`dt`, `q_burst=0.98`, channels, log-signature L2, the four
matched-budget baselines, leakage-safe contiguous splits) is **inherited
unchanged from v0**.

## 5. Fix C — per-block stationarity gate (new)

The v0 base-rate gate (test base rate ∈ [0.05, 0.40]) is **necessary but
not sufficient** — it missed the block-dependence. v1 adds, checked
before any detector-vs-baseline read:

```text
base_rate(train), base_rate(val), base_rate(test) each in [0.05, 0.40]
AND  pairwise |base_rate(i) - base_rate(j)| <= 0.10
```

- Fail the band → `PDE-C2-DEFERRED-BASERATE` (as v0).
- Pass the band but fail pairwise consistency → new
  **`PDE-C2-DEFERRED-NONSTATIONARY`** (the v0 failure mode, now named).
- Both pass → proceed to the matched-budget comparison.

No retune of `q_burst`/`E_burst`/blocks after reading a gate result
(that is `PDE-C2-NEG-B`); a further re-pose is a v2 cell.

The receipt reports **per-block diagnostics** (mean/median max-energy,
burst count, base rate) so a deferral is interpretable (drift vs
under-sampling).

## 6. Inherited from v0

Channel tiers (Tier 0 shell log-energies headline; Tier 1 +transfers;
Tier 2 +hidden-self-similarity), log-signature level 2, the four
matched-budget baselines (DMD / CSD / lacunarity / Rényi) with the
matched-budget rule, the Pareto lead-time vs. false-positive evaluation,
the two-sided negative (`PDE-C2-NEG-A` / `PDE-C2-NEG-B`), and the
`PDE-C1-NEG` re-framing rule. The matched-budget 4-baseline comparison
remains the next increment, now gated on **both** the base-rate band and
the per-block stationarity gate.

## 7. Harness changes (for build — not yet built)

In `scripts/pde_c2_sabra_cell.py`:

- a `--forcing {additive, fixed-amplitude, constant-power}` option
  (default `fixed-amplitude` per §2);
- a `--diagnostic` mode that runs §3 and writes the energy time series +
  `T_eq` / `T_burst` estimates, filing no verdict;
- per-block base-rate + energy/burst diagnostics in the receipt;
- the §5 per-block stationarity gate and the new
  `PDE-C2-DEFERRED-NONSTATIONARY` branch.

The inviscid energy-conservation self-test stays the integrator gate and
must still pass.

## 8. Open sign-off decisions

1. **Forcing scheme (load-bearing).** Default `fixed-amplitude` |u_1|
   forcing (recommended — guarantees a steady cascade), vs retaining v0
   additive forcing with a long verified warmup, vs constant-power.
2. **Stationarity gate tolerance.** Pairwise base-rate consistency
   `≤ 0.10` and band `[0.05, 0.40]` — confirm.
3. **Diagnostic-then-cell staging.** Run the §3 diagnostic first (cheap),
   pin warmup/blocks from `T_eq`/`T_burst`, then the full cell — vs pin
   conservative large values up front and skip the diagnostic.
4. **Compute budget.** Long blocks (O(50) bursts each) + 4 baselines +
   hyperparameter search will be heavier than v0's ~32 min; confirm a
   trajectory length / target wall-clock.
5. **Build trigger.** On sign-off I add the forcing option + diagnostic
   mode + stationarity gate, re-run the energy self-test, run the §3
   diagnostic, then the verdict-bearing cell.

## 9. Cross-references

- [`PDE_C2_CELLSET_SABRA_v0.md`](PDE_C2_CELLSET_SABRA_v0.md) — the
  deferred v0 cell (§12 disposition) this re-poses.
- [`PDE_C2_SHELL_SIGNATURE_SCOPING.md`](PDE_C2_SHELL_SIGNATURE_SCOPING.md)
  — research object, channel taxonomy, baselines, two-sided negative.
- [`PDE_C1_REGIME_GENERALITY_v1.md`](PDE_C1_REGIME_GENERALITY_v1.md) —
  the C1 portable-objective + gate discipline whose lineage this follows
  (deferral → diagnose → new pinned cell, never a same-cell retune).
- [`../SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md) — ledger.
