# PDE C1 Regime Generality v1 — Portable Objective

> **Pre-registration and result**, filed 2026-05-29. Successor to
> [`PDE_C1_REGIME_GENERALITY_v0.md`](PDE_C1_REGIME_GENERALITY_v0.md),
> whose G=300 probe returned `PDE-C1-RG-DEFERRED_VACUITY`: the
> kNN/twin-state machinery stayed usable, but the *registered overshoot
> proxy went near-vacuous* (`damp_fraction ≈ 0.004`) under
> heavy-tailed/intermittent energy statistics. The regime-generality
> question was therefore **not adjudicated** — the failure was objective
> construction, not fiber-locality coverage or support-level
> non-injectivity. This artifact replaces the fixed-percentile objective
> with a **regime-portable** one and re-poses the Grashof-axis generality
> test cleanly.
>
> **Status: executed.** §12 records `PDE-C1-RG-POS`: the portable
> objective passes its portability gate at G=200 and G=300, the control
> half is POSITIVE at both regimes, and the G=300 twin-state companion
> is CERTIFIED. C1 remains unpromoted because proxy faithfulness and
> external PDE review remain open.

## 1. The fix in one line

Replace "exceed the **burn-in** 95th-percentile `E_K` in lookahead"
with "be in the top `(1−q)` fraction of **own-regime** τ-excursions,"
calibrated on a **held-out post-burn-in** window. This pins
`damp_fraction = 1−q` by construction at every regime, removing the
v0 vacuity failure mode.

## 2. Why the v0 objective was not portable

`E_max` was the 95th percentile of the (last-25%) burn-in `E_K`. Two
coupled defects under intermittency:

- **Calibration/sampling mismatch.** A single burst in the burn-in
  calibration window shifts the 95th percentile far out; the
  post-burn-in sampling window is comparatively quiescent → almost no
  look-ahead window reaches `E_max`.
- **Heavy-tailed shape-dependence.** At G=200 the energy distribution is
  tight, so the 95th percentile sits near the bulk (damp_fraction 0.30);
  at G=300 it is intermittent, so the 95th percentile sits on a
  rare-burst tail (damp_fraction 0.004). The *same* percentile rule
  means different things at different regimes.

Confirmed regime property, not artifact: v2 (G=300, full-burnin E_max)
also gave sub-1% damp. The objective, not the machinery, fails to
transfer.

## 3. Portable objective (pinned)

For a state `u` with no-op τ-lookahead trajectory, define the
**lookahead-max excursion**

```text
M(u) = max_{t in [0, tau]} E_K(u(t)),     tau = 5.0 time units.
```

Calibrate the threshold on a held-out post-burn-in calibration window
`C` (disjoint from the adjudication sample `A`, see §4):

```text
E_max = quantile_q( { M(u) : u in C } ),     q = 0.70.
```

The proxy selector, applied to the adjudication sample `A`:

```text
pi_hat(u) = damp_low_band   iff   M(u) > E_max
          = no_op           otherwise,        for u in A.
```

By construction `damp_fraction` on `C` is exactly `1 − q = 0.30`; on
the disjoint stationary `A` it is `≈ 0.30` (the **portability gate**,
§6, verifies this at both regimes). `q = 0.70` is chosen so the new
objective's damp scale matches v5's emergent `0.30`, making the G=200
re-run a clean comparison to the existing witness.

Everything else is inherited from v5/v6 unchanged: `k_f = 2`, `K = 3`,
`d_K = 18`, `dt = 0.01`, `tau = 500` steps, `epsilon_K = 0.05·sqrt(2
E_max)` (now with the portable `E_max`), kNN sweep `k ∈
{10,15,20,25,30,40,50}` with `a_mm` thresholds (≤0.005 POSITIVE,
≥0.015 NEG-A), twin-state `k_twin=50`, `delta_H = max(1e-6,
0.05·median‖Q_K‖)`, gates `0.01` / `100`.

## 4. Held-out split (pinned)

The post-burn-in trajectory is partitioned into two disjoint blocks
with a decorrelation gap:

```text
calibration window C : 50,000 samples at interval 50 steps
decorrelation gap    : 5,000 steps (~a few Lyapunov times)
adjudication sample A: 50,000 samples at interval 50 steps
```

`E_max` is computed from `C` only; labelling, kNN convergence, and
twin-state run on `A` only. `A` is held at 50,000 to match v5's
adjudication N for apples-to-apples comparison. Total post-burn-in
integration roughly doubles vs v5 (~40 min/run expected; see §7 cost).
Rationale for held-out rather than in-sample: although `E_max` is a
single global scalar (so in-sample calibration would not obviously leak
*fiber* structure into the local kNN test), held-out removes all doubt
at low cost and matches the agreed design.

*Cost-reduced alternative (not pinned; a sign-off option):* `C = A =
30,000` with the same gap, ~1.2× v5 cost. Decide at sign-off.

## 5. Harness objective-mode

A new objective mode, selected by a flag, leaves the v0–v6 fixed-
percentile path untouched:

```text
--objective {overshoot-burnin (default), portable-quantile}
```

`portable-quantile` config additions (pinned in the preset, pre-
registered): `objective_quantile q = 0.70`, `calibration_sample_count =
50000`, `calibration_gap_steps = 5000`. New presets `lock_v7_g200`
(`k_f=2, G=200, K=3`) and `lock_v7_g300` (`k_f=2, G=300, K=3`), both
with `e_max_burnin_fraction` irrelevant under this objective (E_max no
longer from burn-in) and `objective = portable-quantile`.

Implemented by extending `run_cell` to integrate the extra calibration
block + gap before the adjudication block; compute `M(u)` for
calibration samples; set `E_max = quantile(M_C, q)`; then
label/adjudicate the adjudication block exactly as today. All
adjudicators (`knn`, `knn-sweep`, `twin-state`) consume the adjudication
block unchanged. Smoke parity and overshoot-burnin regression were
checked before the lock runs.

## 6. Portability gate (new, pre-registered)

**Before** interpreting any control-sufficiency verdict, confirm the
objective is actually portable:

```text
damp_fraction(A) in [0.20, 0.40]   at BOTH G=200 and G=300.
```

- If both pass → the objective is portable; proceed to read the kNN
  verdicts as a genuine regime comparison.
- If either fails → **`PDE-C1-RG-PORTABILITY-FAIL`**: even the
  held-out-quantile objective does not transfer (e.g. severe
  non-stationarity between `C` and `A`). This is itself a finding;
  do **not** retune `q` or the split to rescue it (that would be
  `PDE-C1-RG-NEG-B`). A new construction (burst-onset, §9) would be a
  separate v2 pre-registration.

`[0.20, 0.40]` is `0.30 ± 0.10`; by construction `A` should sit very
near `0.30`, so this gate mainly catches stationarity/leakage failures.

## 7. Program and run order (pinned)

```text
1. lock_v7_g200 --adjudicator knn-sweep   (positive control + de-confound)
2. lock_v7_g300 --adjudicator knn-sweep   (the generality test)
   [portability gate checked on both before interpreting either]
3. IF both return STRICTNESS_WITNESS_POSITIVE:
      lock_v7_g300 --adjudicator twin-state   (support companion)
      (lock_v7_g200 twin-state optional, for a complete re-witness)
```

The **G=200 re-run is mandatory and runs first**: the portable objective
is a *new* objective, so the v5 witness must be re-established under it
before G=300 generality means anything. If G=200 flips (does **not**
return POSITIVE under the portable objective), that is a critical
finding — it would mean the v5 POSITIVE was partly an artifact of the
old fixed-percentile threshold — and the generality program pauses to
reconcile before G=300 is interpreted.

Expected wall-clock ~40 min per kNN-sweep run (≈2× v5 from the doubled
post-burn-in integration), ~35 min per twin-state run. Up to ~4 runs.
None inline under the ~10-minute rule.

## 8. Branches and interpretation

Per-regime kNN verdict feeds the existing
[`PDE_C1_REGIME_GENERALITY_v0.md`](PDE_C1_REGIME_GENERALITY_v0.md) §6
branch family, gated by §6 portability here:

| G=200 kNN | G=300 kNN | program outcome |
|---|---|---|
| POSITIVE | POSITIVE | `PDE-C1-RG-POS` (control-suff replicates across G under a portable objective); proceed to twin-state |
| POSITIVE | NEG-A | `PDE-C1-RG-NEG-A` (regime-2 is a localized window on the Grashof axis — the informative failure) |
| POSITIVE | deferral | `PDE-C1-RG-INCONCLUSIVE_CONTROL` |
| not POSITIVE | — | **program pauses**: portable objective overturns the v5 witness; reconcile before any generality claim |

`PDE-C1-RG-POS` here means: the v5 Reading-2 regime-2 witness has a
higher-Grashof replication **under an objective that is portable by
construction** — strictly stronger than a same-objective coincidence.
It still does **not** mean generality across all `G`, across `k_f`, an
infinite-dimensional NSE theorem, `J`-optimality, or promotion without
external review.

## 9. Pre-registration discipline

- All values in §3–§6 are fixed before any verdict-bearing run.
- Post-hoc change to `q`, the split, the gap, `epsilon_K`, the kNN/
  twin-state thresholds, or the regime after reading a receipt →
  `PDE-C1-RG-NEG-B`.
- The portability gate is a **precondition**, not a tunable: a
  `PORTABILITY-FAIL` is filed, not rescued.
- Burst-onset (a `dE_K/dt` or relative-jump onset criterion) is the
  **documented v2 alternative**, not part of this artifact; it carries
  more free parameters and would need its own pre-registration.

## 10. Build Decisions Closed

The sign-off choices landed as:

1. **Split sizes:** 50k calibration / 50k adjudication, preserving
   apples-to-apples comparison with v5.
2. **Quantile:** `q = 0.70`, targeting `damp_fraction ≈ 0.30` to match
   the v5 action scale.
3. **Build:** `--objective portable-quantile` plus `lock_v7_g200` /
   `lock_v7_g300` presets, with regression and smoke checks before the
   verdict-bearing runs.

## 12. Result (2026-05-29) — PDE-C1-RG-POS

The full program executed; all three verdict-bearing runs landed. The
portable objective fixed the v6 vacuity and the complete regime-2
witness **replicates at G=300**.

| run | portability gate (adj damp ∈ [0.20,0.40]) | verdict |
|---|---|---|
| `lock_v7_g200` kNN (positive control) | 0.3003 ✓ (calib 0.300) | `STRICTNESS_WITNESS_POSITIVE`, `a_mm = −0.00079`, slope 0.736 |
| `lock_v7_g300` kNN (generality test) | 0.2688 ✓ (calib 0.300) | `STRICTNESS_WITNESS_POSITIVE`, `a_mm = +0.00058`, slope 0.564 |
| `lock_v7_g300` twin-state | — | `TWIN_STATE_CERTIFIED` (100% witness coverage, 942,834 unique pairs, `δ_H = 0.0111` from real median ‖Q_K‖ = 0.222) |

Receipts: `results/proof/c1-rg-v1-g200-knn-sweep/`,
`results/proof/c1-rg-v1-g300-knn-sweep/`,
`results/proof/c1-rg-v1-g300-twin-state/`.

### What it establishes

- **Portability gate passed at both regimes.** Held-out
  `damp_fraction` = 0.300 (G=200) and 0.269 (G=300), both in
  [0.20,0.40]. The portable objective is genuinely regime-portable:
  v6's `damp_fraction = 0.004` vacuity at G=300 is gone (0.269 now). The
  small G=300 calib-vs-adj gap (0.300 → 0.269) is a mild, honest
  signature of intermittency non-stationarity, well inside the gate.
- **Positive control (G=200) re-establishes the v5 witness under a
  different, sounder objective.** `a_mm = −0.00079`, slope 0.736 vs v5's
  −0.00078 / 0.737 — near-identical despite a different `E_max`
  construction (held-out look-ahead-max quantile 0.7344 vs burn-in
  95th-percentile). The v5 control-sufficiency result is **not** an
  artifact of the old threshold rule.
- **Generality test (G=300) POSITIVE.** `mean_minority` extrapolates to
  ~zero (`a_mm = +0.00058 ≤ 0.005`), through-origin, 10–22× below the
  random-label floor → clean decision surface, control-sufficient.
- **Twin-state CERTIFIED at G=300**, non-degenerately, at the same
  `ε_K = 0.0664` as the control read, so the two halves **compose** into
  a complete regime-2 witness at G=300.

**Net: regime-2 (state-insufficient AND control-sufficient) is now a
two-regime result on the Grashof axis — (k_f=2, G=200) and
(k_f=2, G=300) — under a portable objective, no longer cell-local.**
The control half is also dimension-robust (v4/v5 across d=32/d=18) and
now objective-robust (fixed-percentile and portable-quantile both
POSITIVE at G=200).

### What it does NOT establish (scope held)

- **Two points on ONE axis.** Grashof only; `k_f = 2` fixed throughout.
  The objective family now has two members but both are
  energy-overshoot-type. This is not full substrate-generality.
- **Finite-Galerkin, sampled-support, numerical** — not a theorem about
  the infinite-dimensional NSE attractor.
- **Proxy faithfulness strengthened, not closed.** Two independent
  objective constructions agreeing helps, but `π̂` is still a proxy, not
  a derived `J`-optimal selector with an explicit action cost.
- **External PDE review (criterion c) open.** **C1 remains UNPROMOTED** —
  but this is the strongest the candidate has been: a two-regime,
  dimension-robust, objective-robust, end-to-end regime-2 witness.

### Next firm-up axes (none required, none urgent)

`k_f`-axis generality (vary forcing geometry at fixed G); a genuinely
different objective family (enstrophy / burst-onset, the documented v2
alternative); proxy-faithfulness via a derived `J`-optimal selector;
external review. Any of these would further harden C1; none is needed
to bank the present two-regime result.

## 11. Cross-References

- [`PDE_C1_REGIME_GENERALITY_v0.md`](PDE_C1_REGIME_GENERALITY_v0.md) —
  the G=300 fixed-percentile probe that deferred for objective vacuity;
  this artifact is its portable-objective successor.
- [`PDE_C1_KNN_CONVERGENCE_CHECK.md`](PDE_C1_KNN_CONVERGENCE_CHECK.md) /
  [`PDE_C1_TWIN_STATE_CERTIFICATE.md`](PDE_C1_TWIN_STATE_CERTIFICATE.md)
  — the adjudicators reused unchanged on the adjudication block.
- [`PDE_C1_CELLSET_KOLMOGOROV_v5.md`](PDE_C1_CELLSET_KOLMOGOROV_v5.md) —
  the completed `G=200, K=3` witness the G=200 re-run must re-establish
  under the portable objective.
- [`../SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md) — the
  ledger; §12's `PDE-C1-RG-POS` is reflected there as a two-regime,
  still-unpromoted C1 witness.
