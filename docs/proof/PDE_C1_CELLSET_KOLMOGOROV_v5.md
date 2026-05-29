# PDE C1 Cell Set v5 - Kolmogorov-Flow Instance at k_f = 2, G = 200, K = 3

> Methodology sibling to
> [`PDE_C1_CELLSET_KOLMOGOROV_v4.md`](PDE_C1_CELLSET_KOLMOGOROV_v4.md),
> filed 2026-05-28 in response to the v4 lock + fall-back combined
> disposition. v4 confirmed that the E_max amendment unlocks proxy
> discrimination (`damp_fraction ≈ 0.30`, stable between 50,000 and
> 200,000 samples), but uniform binning at `K = 4` fails on this
> attractor regardless of sample budget (45k occupied bins at v4 lock,
> 139k at fall-back, `evaluated_bin_count = 0` in both). v5 holds the
> v4 regime constant (`k_f = 2`, `G = 200`,
> `e_max_burnin_fraction = 0.25`) and pins `K = 3` (signature dim 18
> instead of 32) per the same-day fiber-protocol K-parameter
> amendment. Forced mode `(0, 2)` remains in the K=3 signature
> (it sorts ahead of higher-`k^2` modes in the selection).
>
> Status: drafted, desk-auditable, unreviewed, no numerical verdict
> yet. Harness exposes this cell as `--preset lock_v5` and
> `--preset fallback_v5`.

## 1. Methodology Delta from v4

The single change is the signature mode count:

```text
v4: K = 4   (signature dim d_K = 32, 16 Fourier modes)
v5: K = 3   (signature dim d_K = 18,  9 Fourier modes)
```

Bin count scales as `(R_attractor / h_K)^{d_K}`, so dropping
`d_K = 32 → 18` reduces the dimensional cost by a factor of
`(R/h)^{14}`. For `R/h ≈ 5` (rough estimate from v4), that's
`5^14 ≈ 6×10^9` fewer bins in the worst case — overkill by orders of
magnitude relative to v4's coverage gap.

**Forced-mode inclusion check.** At `k_f = 2`, the forced mode is
`(0, 2)` with `max(|kx|,|ky|) = 2`. The harness `select_low_modes`
picks `K² = 9` modes for `K = 3` by sorting on
`(max(|kx|,|ky|), k², ix, iy)`. Modes with `max = 1`: four
half-plane representatives `(0,1), (1,1), (-1,1), (1,0)`. Modes
with `max = 2`: sorted first by `k²`, so `(0,2)` and `(2,0)`
(both `k² = 4`) come ahead of any `k² = 5` or `k² = 8` modes. The
first 9 modes include `(0,2)`. **Forced mode retained.** ✓

## 2. Inherited from v0–v4

All v4 settings are inherited unchanged except `K`:

- **Regime.** 2D NSE on `T^2 = [0, 2*pi]^2`, `N = 16` modes per axis,
  `k_f = 2`, `G = 200`, `nu = sqrt(1/200) ≈ 0.0707`, forcing
  amplitude `1`, decision domain `B_abs`, evaluation measure SRB-like.
- **Observation.** **`K = 3`**, `d_K = 18`, `F_K = sigma(Phi_K)`.
  The signature still includes the forced mode `(0, 2)`.
- **Objective.** Low-band energy safety trigger; look-ahead horizon
  `tau = 5` time units; action space `{no_op, damp_low_band}`;
  deterministic proxy `\hat{pi}`. `E_max` = 95th percentile of last
  25% of burn-in (v4 amendment).
- **Comparator.** Finite-Galerkin non-injectivity on `B_abs` (the
  algebraic construction is dimension-independent); literature
  determining-mode reference as sufficiency ceiling only.
- **Fiber classifier.** Unchanged.
- **Negative branch.** Unchanged.
- **§7 Parameter instantiation.** All values inherited except
  `K = 3` and the derived `d_K = 18`, `epsilon_K = 0.05 sqrt(2 E_max)`
  (recomputed at v5 burn-in), `h_K = epsilon_K / sqrt(18) ≈
  epsilon_K / 4.24` (vs v4's `epsilon_K / 5.66`).

## 3. New v5 Pin

- **Signature mode count.** `K = 3`.
- **Signature dimension (derived).** `d_K = 2K^2 = 18`.

That is the only spec-level change. The harness preset `lock_v5`
encodes it; all other parameters resolve identically to `lock_v4`.

## 4. Discriminative-Power and Coverage Expectations

**Proxy discrimination expectation.** At the v4 regime, the
forced mode `(0, 2)` carries most of the energy and the safety
trigger fires on look-ahead windows where `E_K` (the K-truncated
low-band energy) crosses `E_max`. Dropping K=4→K=3 reduces the
signature components from 32 to 18; the safety predicate now
"sees" energy in 9 Fourier modes instead of 16. The forced mode is
preserved, so the dominant energy contribution is intact.

`damp_fraction` is expected to remain in the discriminative band
`[0.01, 0.99]`. It may shift up or down by a moderate factor
(perhaps `±2×`) depending on how much energy lives in the dropped
modes `(0, 3), (3, 0), (1, 3), (-1, 3), (3, 1), (-3, 1)` — that's 6
of the 16 K=4 modes that v5 drops. If they carry significant energy
at `G = 200`, `damp_fraction` could shift. v4 lock had `damp_fraction
≈ 0.30`; v5 expected range `[0.05, 0.60]`.

**Coverage expectation.** Bin count should drop dramatically. v4
fall-back gave 139,361 bins at `N = 200,000`. With `d_K` dropping
by 14 dimensions and `h_K` increasing by a factor of `sqrt(32/18) ≈
1.33`, the projected v5 bin count is **at least an order of
magnitude smaller**, probably 5k–20k bins at `N = 50,000`. If
correct, `samples_per_bin_avg = N / bins` would be 2.5–10, with
many bins reaching `n_min = 30` on the high-density part of the
attractor. **`S_eval ≥ 0.5` is the pre-registered expectation.**

If v5 still defers for coverage, the next step is **not** another K
drop (K=2 loses the forced mode at k_f=2). Options would include
larger `epsilon_K` (adaptive `h_K` amendment) or a regime adjustment.

If `damp_fraction` falls outside `[0.01, 0.99]` on v5 (unlikely
given the v4 evidence but possible if the dropped modes were
load-bearing), the appropriate filing is `DEFERRED_VACUITY` and a
re-examination of the K=3 mode selection.

## 5. What v5 Closes / Does Not Close

**Same closures as v0–v4 methodology surface, plus the K-parameter
clarification.** Cells can now pin `K` to trade signature richness
against bin coverage, with explicit forced-mode inclusion
verification per cell.

**Open via execution.** A non-vacuous, non-deferred v5 verdict
(`STRICTNESS_WITNESS_POSITIVE` or `PDE-C1-NEG-A`) would be the
**first substantive C1 read** across five cell-set iterations.

**Not closed by v0/v1/v2/v3/v4/v5 alone:**

- Criterion (a) Front-A vacuity rebuttal (sidecar-level).
- Criterion (c) named external PDE reviewer.
- Attractor-support twin-state certificate.

## 6. Status

- **Drafted.** 2026-05-28.
- **Desk-auditable.** A reviewer can read v4 + the v4 combined
  disposition + this v5 methodology delta and audit the change without
  running code.
- **Unreviewed.** No external PDE reviewer has signed off on the
  `K = 3` choice.
- **Unrun.** The `lock_v5` preset is present in the harness; no
  execution has been attempted.

## 7. Cross-references

- [`PDE_C1_CELLSET_KOLMOGOROV_v4.md`](PDE_C1_CELLSET_KOLMOGOROV_v4.md)
  - v4 cell-set and its Lock Execution Disposition (lock + fall-back)
  that motivates v5.
- [`PDE_C1_FIBER_PROTOCOL.md`](PDE_C1_FIBER_PROTOCOL.md) -
  methodology; the K-parameter rule was added 2026-05-28 to §2 Local
  Symbols + Binning Instance.
- [`PDE_C1_CELLSET_KOLMOGOROV.md`](PDE_C1_CELLSET_KOLMOGOROV.md) /
  [v1](PDE_C1_CELLSET_KOLMOGOROV_v1.md) /
  [v2](PDE_C1_CELLSET_KOLMOGOROV_v2.md) /
  [v3](PDE_C1_CELLSET_KOLMOGOROV_v3.md) — prior cell-set siblings.
- [`PDE_DETERMINING_MODES_POSTULATE1.md`](PDE_DETERMINING_MODES_POSTULATE1.md)
  - the C1 sidecar.
- [`../SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md) - the
  ledger.
- [`../../scripts/pde_c1_kolmogorov_cell.py`](../../scripts/pde_c1_kolmogorov_cell.py)
  - harness. `--preset lock_v5` runs this cell at
  `(k_f = 2, G = 200, e_max_burnin_fraction = 0.25, K = 3)`.
