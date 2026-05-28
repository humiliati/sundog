# PDE C1 Cell Set v2 - Kolmogorov-Flow Instance at k_f = 2, G = 300

> Sibling cell-set instance to
> [`PDE_C1_CELLSET_KOLMOGOROV.md`](PDE_C1_CELLSET_KOLMOGOROV.md) (v0)
> and [`PDE_C1_CELLSET_KOLMOGOROV_v1.md`](PDE_C1_CELLSET_KOLMOGOROV_v1.md)
> (v1), filed 2026-05-28 in response to the v1 lock disposition. v1
> escaped v0's laminar-vacuity but found a new structural vacuity:
> the post-transient attractor at `k_f = 2`, `G = 100` is non-trivial
> in signature space (1397 occupied bins) but **`E_K` is exactly
> conserved** on it (sample energy constant to 13 decimal places),
> so the safety-trigger objective is inaccessible. v2 triples the
> Grashof number to `G = 300` while keeping `k_f = 2` to inject
> enough energy that the attractor symmetry breaks and `E_K`
> fluctuates non-trivially. This is a new cell-set instance, not a
> retune of v0 or v1.
>
> Status: drafted, desk-auditable, unreviewed, no numerical verdict
> yet. The harness exposes this cell as `--preset lock_v2` (and
> `--preset fallback_v2` for the coverage-gate fall-back).

## 1. Regime Delta from v1

The single regime change is the Grashof number:

```text
v1: G = 100   (supercritical at k_f = 2 but with E_K-conserving attractor)
v2: G = 300   (3x energy injection, expected to break the conservation)
```

The viscosity scales accordingly: `nu = sqrt(forcing_amplitude / G) =
sqrt(1/300) ≈ 0.0577` (v1 had `nu = 0.1`). All other parameters are
inherited from v1 unchanged.

`G = 300` is chosen as the smallest tripling that comfortably stays
within the `N = 16` Galerkin truncation's resolution envelope. The
viscous-mode wavenumber `k_d ~ 1/sqrt(nu) ~ 4.16` sits inside the
dealiased cutoff (`N/3 ≈ 5.3`), so spectral pile-up is not a concern
at this regime. Higher `G` (e.g. `G = 1000`) would require a larger
truncation and is deferred to a possible v3 if v2 is also vacuous.

## 2. Inherited from v0 and v1

All of the following are inherited unchanged from v0 / v1:

- **§1 Regime (excluding `G`).** 2D NSE on `T^2 = [0, 2*pi]^2`,
  `N = 16` modes per axis (`32 × 32` grid), `k_f = 2` (from v1),
  dimensionless forcing amplitude `1`, decision domain `B_abs` of the
  truncated system after burn-in, evaluation measure the SRB-like
  distribution.
- **§2 Observation.** `K = 4`, signature dimension `d_K = 32`,
  σ-algebra `F_K`. The signature includes the forced mode `(0, 2)`.
- **§3 Objective.** Low-band energy safety trigger; `E_max` =
  95th percentile of `E_K` along the v2 burn-in (recomputed for v2
  because the dynamics differ); look-ahead horizon `τ = 5` time units;
  action space `{no_op, damp_low_band}`; deterministic provisional
  selector / proxy `\hat{pi}` per the fiber protocol section 3.
- **§4 Comparator.** Finite-Galerkin non-injectivity witness on
  `B_abs` (regime-independent); literature determining-mode reference
  as sufficiency ceiling only. The attractor-support twin-state
  certificate remains a deferred bridge.
- **§5 Fiber classifier.** Unchanged.
- **§6 Negative branch.** Unchanged; `PDE-C1-NEG-A` and `PDE-C1-NEG-B`
  receipts apply.
- **§7 Parameter instantiation.** All section 7 values are inherited:
  `dt = 0.01`, `T_burnin = 10^5` steps, `epsilon_K = 0.05 sqrt(2 E_max)`
  (numerically re-pinned at v2 burn-in completion since `E_max` is
  regime-dependent), `h_K = epsilon_K / sqrt(d_K)`, `n_min = 30`,
  `delta_action = 0.10`, `S_pos = 0.50`, `delta_proxy_min = 0.01`,
  `N_sample = 50,000`, sampling interval `0.5` time units, fall-back
  to `N_sample = 200,000` on coverage deferral, action tie-break
  favouring `no_op`.

## 3. New v2 Pin

- **Grashof number.** `G = 300`.
- **Viscosity (derived).** `nu = sqrt(1/300) ≈ 0.0577`.

That is the only spec-level change. The harness preset `lock_v2`
encodes it; everything else in `build_config` resolves identically to
`lock_v1`.

## 4. Discriminative-Power Expectation

The v1 lock disposition identified energy-exact conservation as the
v1 failure mode. The v2 hypothesis: at higher Grashof, the
forced-dissipative balance breaks energy conservation on the
attractor, producing non-trivial `E_K` fluctuations that exercise
the safety trigger.

Expected v2 behaviour:

- `burnin_energy_max - burnin_energy_min` should be O(`burnin_energy_mean`)
  rather than O(0.0001) (as v1 showed post-burn-in). Burn-in trajectory
  reaches statistical steady state with non-trivial spread.
- `sample_energy_max - sample_energy_min` should also be O(mean): the
  attractor itself has energy fluctuations, not just transients.
- `damp_fraction` should land in `[0.01, 0.99]`. The 95th-percentile
  threshold gives, by construction, a `damp_fraction ~ 0.05` at
  steady state if the look-ahead window is comparable to the
  energy-fluctuation autocorrelation time.
- If `damp_fraction` again falls outside `[0.01, 0.99]`, v2 also
  fires `DEFERRED_VACUITY`. The next sibling v3 would raise `G` further
  (e.g. `G = 1000` with `N >= 32` to maintain resolution) or change
  the objective.
- If `0.01 ≤ damp_fraction ≤ 0.99`, the predicate has discriminative
  content and the verdict is `STRICTNESS_WITNESS_POSITIVE`,
  `PDE-C1-NEG-A`, or `DEFERRED_COVERAGE` (and on coverage deferral,
  the fall-back at `N_sample = 200,000` is admissible because the
  vacuity isn't structural).

This expectation is **pre-registered**, not a result.

## 5. What v2 Closes / Does Not Close

**Same closures as v0 / v1 methodology surface:** the fiber
protocol's binning predicate, the proxy selector substitution rule,
the parameter-pinning discipline, the structural-vacuity precedence
rule (added 2026-05-28).

**Open via execution.** A non-vacuous v2 verdict would be the first
substantive C1 read. If v2 also defers, the next step is either a v3
at higher `G` with a larger truncation, or a re-objective at the
sidecar level (e.g. a different `J` not based on low-band energy).

**Not closed by v0 / v1 / v2:**

- Criterion (a) Front-A vacuity rebuttal (sidecar-level).
- Criterion (c) named external PDE reviewer.
- Attractor-support twin-state certificate.

## 6. Status

- **Drafted.** 2026-05-28.
- **Desk-auditable.** A reviewer can read v0 + v1 + this v2 delta and
  audit the regime change without running code.
- **Unreviewed.** No external PDE reviewer has signed off on the
  `G = 300` choice as a faithful symmetry-breaking instance.
- **Unrun.** The `lock_v2` preset is present in the harness; no
  execution has been attempted.

## 7. Cross-references

- [`PDE_C1_CELLSET_KOLMOGOROV.md`](PDE_C1_CELLSET_KOLMOGOROV.md) - v0
  cell-set and "Lock Execution Disposition (2026-05-28)" section
  recording the laminar-vacuity outcome.
- [`PDE_C1_CELLSET_KOLMOGOROV_v1.md`](PDE_C1_CELLSET_KOLMOGOROV_v1.md)
  - v1 cell-set and "Lock Execution Disposition (2026-05-28)" section
  recording the energy-conservation-vacuity outcome that motivates v2.
- [`PDE_C1_FIBER_PROTOCOL.md`](PDE_C1_FIBER_PROTOCOL.md) -
  methodology; the structural-vacuity-precedence rule (section 5
  step 8, added 2026-05-28) is what filed v1 as `DEFERRED_VACUITY`
  and would catch a same-mode failure on v2.
- [`PDE_DETERMINING_MODES_POSTULATE1.md`](PDE_DETERMINING_MODES_POSTULATE1.md)
  - the C1 sidecar; v0, v1, v2 are concrete instances of the
  pre-registered cell set described there.
- [`../SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md) - the
  ledger.
- [`../../scripts/pde_c1_kolmogorov_cell.py`](../../scripts/pde_c1_kolmogorov_cell.py)
  - harness. The `--preset lock_v2` switch runs this v2 cell with
  the inherited v0/v1 parameter values plus `G = 300`.
