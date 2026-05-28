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

## Lock Execution Disposition (2026-05-28)

The v2 lock cell was executed on 2026-05-28 via
[`../../scripts/pde_c1_kolmogorov_cell.py`](../../scripts/pde_c1_kolmogorov_cell.py)
with `--preset lock_v2`. Receipt and manifest at
`results/proof/c1-kolmogorov-v2-lock/`. The harness completed in
1336 seconds (~22 min) with `steps_per_second ≈ 1946` and reported a
procedural `DEFERRED_COVERAGE` verdict.

### Energy conservation broken — partial success

The v1 disposition's diagnosis was correct: tripling Grashof broke the
energy-conserved attractor.

- `burnin_energy_min / max / mean = 0.79 / 4.69 / 1.13`. Burn-in spans
  a 5.9× range — substantive chaotic excursions, not just transients.
- `sample_energy_min / max / mean = 0.78 / 1.52 / 0.89`. Post-burn-in
  itself has a ~95%-of-mean energy spread, qualitatively different
  from v1's 13-decimal-locked constant.
- `damp_low_band_count = 430`, `no_op_count = 49,570`. The safety
  trigger actually fires — 430 of 50,000 sample windows cross
  `E_max = 1.40`. `damp_fraction = 0.0086` is structurally non-zero.
- The structural-vacuity precedence rule correctly does **not** fire.

### Two new failure modes

But v2 surfaces two issues v1 didn't reach:

1. **Coverage collapse.** `occupied_bin_count = 45,103`,
   `evaluated_bin_count = 0`, `S_eval = 0`. With 50,000 samples spread
   across 45,103 bins, the average bin has ~1.1 samples; no bin
   reaches `n_min = 30`. The pre-registered binning resolution at
   `h_K ≈ 0.0148` is too fine for the attractor extent in 32-dim
   signature space — the curse of dimensionality, not a methodological
   bug.
2. **Near-vacuity.** `damp_fraction = 0.0086 < delta_proxy_min = 0.01`.
   Even if coverage cleared, the statistical vacuity gate would fire
   (`DEFERRED_VACUITY` for proxy near-constancy on the sampled
   support). The safety predicate is barely discriminative.

### Disposition

The procedural verdict `DEFERRED_COVERAGE` is honest at face value. The
pre-registered fall-back (`N_sample = 200,000`) would give ~4.4
samples/bin on average — still well below `n_min = 30` for most bins
— so it would almost certainly defer again. The deeper diagnosis is
that v2's regime is *too* discriminative-by-binning: the attractor
spreads across more of signature space than the fixed `h_K` can
resolve at 50k–200k samples.

This is not a `PDE-C1-NEG-B` retune candidate. The cell-set v0/v1/v2
binning prescription `h_K = ε_K / √d_K` adapts to `E_max` but not to
attractor *extent*; the prescription is what it is. A coverage
failure on a high-dimensional attractor is an honest receipt under
the pinned methodology.

### Bridge to v3

v0 was sub-critical (laminar). v1 was super-critical but
energy-conserved. v2 broke conservation but landed in a curse-of-
dimensionality coverage regime. **v3 backs off the Grashof number
to `G = 200`** — between v1's invariant set and v2's high-dimensional
chaos — looking for the sweet spot where the attractor is
discriminative *and* bin-resolvable at the pre-registered budget.
Staged at
[`PDE_C1_CELLSET_KOLMOGOROV_v3.md`](PDE_C1_CELLSET_KOLMOGOROV_v3.md);
harness exposes `--preset lock_v3`.

If v3 also fails for coverage or near-vacuity, the natural next move
is **not** another G knob — it is to consider a methodology
amendment that lets `h_K` adapt to attractor extent (or a smaller K
that reduces the signature space dimensionally). Both are
substantive changes warranting their own sign-off.

## 6. Status

- **Drafted.** 2026-05-28.
- **Desk-auditable.** A reviewer can read v0 + v1 + this v2 delta and
  audit the regime change without running code.
- **Unreviewed.** No external PDE reviewer has signed off on the
  `G = 300` choice as a faithful symmetry-breaking instance.
- **Executed.** v2 lock ran 2026-05-28 in 1336 sec (~22 min); receipt
  at `results/proof/c1-kolmogorov-v2-lock/`.
- **Disposition.** Procedural `DEFERRED_COVERAGE` recorded honestly;
  diagnosis is two-fold (coverage collapse + near-vacuity on
  high-dimensional chaos). v3 at `G = 200` is the next cell-set
  instance to test the discriminative-and-resolvable hypothesis.

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
