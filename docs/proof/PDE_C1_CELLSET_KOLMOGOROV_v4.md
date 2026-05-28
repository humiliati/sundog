# PDE C1 Cell Set v4 - Kolmogorov-Flow Instance at k_f = 2, G = 200, E_max from steady-state burn-in

> Methodology sibling to
> [`PDE_C1_CELLSET_KOLMOGOROV_v3.md`](PDE_C1_CELLSET_KOLMOGOROV_v3.md),
> filed 2026-05-28 in response to the v3 lock disposition and the
> same-day fiber-protocol E_max windowing amendment. v4 holds the v3
> regime constant (`k_f = 2`, `G = 200`) and pins
> `e_max_burnin_fraction = 0.25` (last quarter of burn-in only for
> the 95th-percentile threshold). This isolates the methodology
> question from the regime question: does fixing E_max alone unlock
> the discriminative regime at v3's settings?
>
> This is the first cell-set in the C1 chain whose delta is a
> *methodology* parameter rather than a *regime* parameter. Same
> instance-of-pre-registration discipline applies.
>
> Status: drafted, desk-auditable, unreviewed, no numerical verdict
> yet. The harness exposes this cell as `--preset lock_v4` (and
> `--preset fallback_v4` for the coverage-gate fall-back).

## 1. Methodology Delta from v3

The single change is the `E_max` windowing:

```text
v0..v3: e_max_burnin_fraction = 1.0   (95th pct of full burn-in,
                                        includes transient peaks)
v4:     e_max_burnin_fraction = 0.25  (95th pct of last 25% of burn-in,
                                        steady-state portion only)
```

For v3's burn-in length of `10^5` steps, the last 25% is the final
`25,000` steps (250 time units at `dt = 0.01`). For 2D Kolmogorov at
`G = 200`, the viscous-mode timescale is `1 / (nu k_f^2) ≈ 3.5` time
units; 250 time units is ~70 viscous turnovers, well past the
transient-decay window. The last 25% should be a faithful sample of
the steady-state attractor.

No other parameter changes.

## 2. Inherited from v3 (and transitively v0/v1/v2)

All v3 settings are inherited unchanged:

- **Regime.** 2D NSE on `T^2 = [0, 2*pi]^2`, `N = 16` modes per axis,
  `k_f = 2`, `G = 200`, `nu = sqrt(1/200) ≈ 0.0707`, forcing amplitude
  `1`, decision domain `B_abs`, evaluation measure SRB-like.
- **Observation.** `K = 4`, `d_K = 32`, `F_K = sigma(Phi_K)`.
- **Objective.** Low-band energy safety trigger; look-ahead horizon
  `tau = 5` time units; action space `{no_op, damp_low_band}`;
  deterministic proxy `\hat{pi}`. `E_max` is **the only objective
  parameter whose computation changes**: it is now the 95th
  percentile of the *last 25%* of burn-in.
- **Comparator.** Finite-Galerkin non-injectivity on `B_abs` (§4.1);
  literature determining-mode reference as sufficiency ceiling only.
- **Fiber classifier.** Unchanged.
- **Negative branch.** Unchanged; `PDE-C1-NEG-A` and `PDE-C1-NEG-B`
  apply.
- **§7 Parameter instantiation.** All values inherited except the new
  E_max windowing parameter: `dt = 0.01`, `T_burnin = 10^5` steps,
  `epsilon_K = 0.05 sqrt(2 E_max)` (recomputed under the windowed
  E_max), `h_K = epsilon_K / sqrt(d_K)`, `n_min = 30`, `delta_action
  = 0.10`, `S_pos = 0.50`, `delta_proxy_min = 0.01`,
  **`e_max_burnin_fraction = 0.25`**, `N_sample = 50,000`, sampling
  interval `0.5` time units, fall-back to `N_sample = 200,000` on
  coverage deferral, action tie-break favouring `no_op`.

## 3. New v4 Pin

- **E_max windowing.** `e_max_burnin_fraction = 0.25`.

That is the only spec-level change. The harness preset `lock_v4`
encodes it; all other parameters resolve identically to `lock_v3`.

## 4. Discriminative-Power Expectation

The v3 lock execution gave concrete numbers we can project from:

- Burn-in 95th percentile = `1.0674` (current `E_max`, contaminated
  by transient peak at `3.1263`).
- Post-burn-in (steady-state) 95th percentile is not directly
  reported, but `sample_energy_max = 0.7843`. The steady-state 95th
  percentile is somewhere in `[0.78, 0.79]`.
- Projected v4 `E_max` ≈ `0.78–0.79`. Then `epsilon_K = 0.05 sqrt(2
  × 0.78) ≈ 0.0625` (vs v3's `0.0731`) and `h_K ≈ 0.011` (vs v3's
  `0.0129`).
- With `E_max ≈ 0.78`, samples crossing the threshold in lookahead
  should approximate 5% by construction (95th percentile of the same
  distribution we're testing against). Expected `damp_fraction ≈ 0.05`,
  comfortably inside `[delta_proxy_min, 1 - delta_proxy_min] =
  [0.01, 0.99]`.
- Coverage: `h_K` shrinks slightly (~15% smaller), so bin count may
  rise slightly above v3's 44,065. Still likely too many bins for
  `n_min = 30` at `N_sample = 50,000`. **Coverage failure is a real
  possibility**, in which case the pre-registered fall-back to
  `N_sample = 200,000` becomes admissible (statistical coverage gap,
  not structural vacuity).

This pre-registers two predictions:

1. The structural-vacuity precedence rule does **not** fire on v4
   (because the steady-state-windowed E_max is close to the
   steady-state attractor's typical excursion, so the proxy is not
   structurally constant).
2. v4 may land `STRICTNESS_WITNESS_POSITIVE`, `PDE-C1-NEG-A`, or
   `DEFERRED_COVERAGE`. Coverage failure would be statistical
   (insufficient samples per bin), not the high-dimensional curse of
   v2.

If v4 *does* fire structural vacuity, the E_max-only amendment was
insufficient; further methodology rethinking (smaller K, different
objective) is the next move.

## 5. What v4 Closes / Does Not Close

**Same closures as v0..v3 methodology surface, plus the E_max windowing
clarification:** cells can now pre-register a burn-in sub-window for
threshold computation. This addresses the transient-contamination
pattern observed across v0/v1/v3.

**Open via execution.** A non-vacuous v4 verdict (any of POSITIVE /
NEG-A / DEFERRED_COVERAGE) would be the first substantive C1 read.

**Not closed by v0/v1/v2/v3/v4 alone:**

- Criterion (a) Front-A vacuity rebuttal (sidecar-level).
- Criterion (c) named external PDE reviewer.
- Attractor-support twin-state certificate.

## 6. Status

- **Drafted.** 2026-05-28.
- **Desk-auditable.** A reviewer can read v3 + the v3 disposition +
  this v4 methodology delta and audit the change without running code.
- **Unreviewed.** No external PDE reviewer has signed off on the
  `e_max_burnin_fraction = 0.25` choice.
- **Unrun.** The `lock_v4` preset is present in the harness; no
  execution has been attempted.

## 7. Cross-references

- [`PDE_C1_CELLSET_KOLMOGOROV_v3.md`](PDE_C1_CELLSET_KOLMOGOROV_v3.md)
  - v3 cell-set and its Lock Execution Disposition that motivates v4.
- [`PDE_C1_FIBER_PROTOCOL.md`](PDE_C1_FIBER_PROTOCOL.md) -
  methodology; the E_max windowing rule was added 2026-05-28 to §3
  Proxy Selector.
- [`PDE_C1_CELLSET_KOLMOGOROV.md`](PDE_C1_CELLSET_KOLMOGOROV.md) /
  [`PDE_C1_CELLSET_KOLMOGOROV_v1.md`](PDE_C1_CELLSET_KOLMOGOROV_v1.md)
  / [`PDE_C1_CELLSET_KOLMOGOROV_v2.md`](PDE_C1_CELLSET_KOLMOGOROV_v2.md)
  - prior cell-set siblings.
- [`PDE_DETERMINING_MODES_POSTULATE1.md`](PDE_DETERMINING_MODES_POSTULATE1.md)
  - the C1 sidecar.
- [`../SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md) - the
  ledger.
- [`../../scripts/pde_c1_kolmogorov_cell.py`](../../scripts/pde_c1_kolmogorov_cell.py)
  - harness. `--preset lock_v4` runs this cell at
  `(k_f = 2, G = 200, e_max_burnin_fraction = 0.25)`.
