# PDE C1 Cell Set v3 - Kolmogorov-Flow Instance at k_f = 2, G = 200

> Sibling cell-set instance to v0, v1, and v2, filed 2026-05-28 in
> response to the v2 lock disposition. v2 escaped v1's
> energy-conservation vacuity by tripling Grashof to `G = 300`, but
> landed in a curse-of-dimensionality coverage regime: 45,103
> occupied bins from 50,000 samples (~1.1 samples/bin) with
> `damp_fraction = 0.0086` just below `delta_proxy_min = 0.01`.
> v3 backs off to `G = 200` to find a sweet spot where the attractor
> is discriminative *and* bin-resolvable at the pre-registered budget.
> Everything else inherited from v0/v1/v2. This is a new cell-set
> instance, not a retune of v0/v1/v2.
>
> Status: drafted, desk-auditable, unreviewed, no numerical verdict
> yet. The harness exposes this cell as `--preset lock_v3` (and
> `--preset fallback_v3` for the coverage-gate fall-back).

## 1. Regime Delta from v2

The single regime change is the Grashof number:

```text
v2: G = 300   (chaotic with broken conservation, but coverage collapses)
v3: G = 200   (intended middle ground: chaotic + bin-resolvable)
```

Derived parameters:

- `nu = sqrt(forcing_amplitude / G) = sqrt(1/200) ≈ 0.0707` (between
  v1's `0.1` and v2's `0.0577`).
- Viscous-dissipation scale `k_d ~ 1/sqrt(nu) ~ 3.76`, comfortably
  inside the `N/3 ≈ 5.3` dealiased cutoff at `N = 16`.

`G = 200` is a 2× step from v1 and a 1.5× step *below* v2. The
expectation is that the linear-stability threshold for `k_f = 2` is
already exceeded at `G = 100` (v1), so `G = 200` remains supercritical;
but the chaotic attractor at `G = 200` should be less spread out in
signature space than v2's, since attractor dimension typically grows
with Reynolds number.

## 2. Inherited from v0, v1, and v2

All of the following are inherited unchanged from v0 / v1 / v2:

- **§1 Regime (excluding `G`).** 2D NSE on `T^2 = [0, 2*pi]^2`,
  `N = 16` modes per axis (`32 × 32` grid), `k_f = 2` (from v1),
  dimensionless forcing amplitude `1`, decision domain `B_abs` of the
  truncated system after burn-in, evaluation measure the SRB-like
  distribution.
- **§2 Observation.** `K = 4`, signature dimension `d_K = 32`,
  σ-algebra `F_K`. The signature includes the forced mode `(0, 2)`.
- **§3 Objective.** Low-band energy safety trigger; `E_max` =
  95th percentile of `E_K` along the v3 burn-in (recomputed for v3
  because the dynamics differ); look-ahead horizon `τ = 5` time units;
  action space `{no_op, damp_low_band}`; deterministic provisional
  selector / proxy `\hat{pi}` per the fiber protocol section 3.
- **§4 Comparator.** Finite-Galerkin non-injectivity witness on
  `B_abs` (regime-independent); literature determining-mode reference
  as sufficiency ceiling only. The attractor-support twin-state
  certificate remains a deferred bridge.
- **§5 Fiber classifier.** Unchanged.
- **§6 Negative branch.** Unchanged.
- **§7 Parameter instantiation.** All section 7 values inherited:
  `dt = 0.01`, `T_burnin = 10^5` steps, `epsilon_K = 0.05 sqrt(2 E_max)`
  (numerically re-pinned at v3 burn-in completion), `h_K = epsilon_K
  / sqrt(d_K)`, `n_min = 30`, `delta_action = 0.10`, `S_pos = 0.50`,
  `delta_proxy_min = 0.01`, `N_sample = 50,000`, sampling interval
  `0.5` time units, fall-back to `N_sample = 200,000` on coverage
  deferral, action tie-break favouring `no_op`.

## 3. New v3 Pin

- **Grashof number.** `G = 200`.
- **Viscosity (derived).** `nu = sqrt(1/200) ≈ 0.0707`.

That is the only spec-level change. The harness preset `lock_v3`
encodes it; everything else in `build_config` resolves identically
to `lock_v2`.

## 4. Discriminative-Power Expectation

The v3 hypothesis is that `G = 200` sits in a Goldilocks zone:

- **Above v1's structural-vacuity floor.** Energy conservation should
  be broken; `damp_fraction` should be substantively non-zero. Order
  of magnitude expectation: 0.01 to 0.2.
- **Below v2's coverage cliff.** The attractor extent in signature
  space should be moderate enough that 50,000 samples occupy O(1000)
  bins, with O(50) samples/bin on average — comfortably above
  `n_min = 30` for most bins.

Expected v3 behaviour:

- `burnin_energy_mean` somewhere between v1's `0.64` and v2's `1.13`
  (rough scaling argument; not load-bearing).
- `sample_energy_max - sample_energy_min` should be O(`sample_energy_mean`),
  matching v2's qualitative behaviour but at smaller amplitude.
- `damp_fraction` should land in `[0.01, 0.99]`. If `damp_fraction`
  falls below `0.01` again, the **near-vacuity** gate fires
  `DEFERRED_VACUITY` (not structural, but statistical).
- `occupied_bin_count` should be in the low thousands (between v1's
  1397 and v2's 45,103), giving `S_eval ≥ 0.5` at `N_sample = 50,000`.
- If coverage fails, the pre-registered fall-back at
  `N_sample = 200,000` is admissible (this is statistical
  insufficiency, not structural — unlike v1's case).

If v3 still defers (for either gate), the natural next steps are
either:

- a methodology amendment letting `h_K` adapt to attractor extent
  (e.g. `h_K = ε_K * (R_eff / R_ref)` for some adaptive scale), or
- a smaller `K` to reduce signature-space dimensionality
  (e.g. v4 at `K = 3` or `K = 2` keeping `k_f = 2`, `G = 200`).

Both would be substantive methodology changes warranting
amendment-level discussion.

This expectation is **pre-registered**, not a result.

## 5. What v3 Closes / Does Not Close

**Same closures as v0 / v1 / v2 methodology surface.**

**Open via execution.** A non-vacuous, non-deferred v3 verdict would
be the first substantive C1 read. If v3 also defers, the diagnosis
sharpens (which gate fires) and the next-move options above narrow.

**Not closed by v0 / v1 / v2 / v3 alone:**

- Criterion (a) Front-A vacuity rebuttal (sidecar-level).
- Criterion (c) named external PDE reviewer.
- Attractor-support twin-state certificate.

## 6. Status

- **Drafted.** 2026-05-28.
- **Desk-auditable.** A reviewer can read v0 + v1 + v2 + this v3
  delta and audit the regime change without running code.
- **Unreviewed.** No external PDE reviewer has signed off on the
  `G = 200` choice as a faithful discriminative-and-resolvable
  instance.
- **Unrun.** The `lock_v3` preset is present in the harness; no
  execution has been attempted.

## 7. Cross-references

- [`PDE_C1_CELLSET_KOLMOGOROV.md`](PDE_C1_CELLSET_KOLMOGOROV.md) - v0.
- [`PDE_C1_CELLSET_KOLMOGOROV_v1.md`](PDE_C1_CELLSET_KOLMOGOROV_v1.md)
  - v1.
- [`PDE_C1_CELLSET_KOLMOGOROV_v2.md`](PDE_C1_CELLSET_KOLMOGOROV_v2.md)
  - v2, including its Lock Execution Disposition that motivates this
  v3 sibling.
- [`PDE_C1_FIBER_PROTOCOL.md`](PDE_C1_FIBER_PROTOCOL.md) - methodology.
- [`PDE_DETERMINING_MODES_POSTULATE1.md`](PDE_DETERMINING_MODES_POSTULATE1.md)
  - the C1 sidecar.
- [`../SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md) - the
  ledger.
- [`../../scripts/pde_c1_kolmogorov_cell.py`](../../scripts/pde_c1_kolmogorov_cell.py)
  - harness. `--preset lock_v3` runs this cell at
  `(k_f = 2, G = 200)`.
