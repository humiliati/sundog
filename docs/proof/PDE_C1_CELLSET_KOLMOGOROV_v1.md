# PDE C1 Cell Set v1 - Kolmogorov-Flow Instance at k_f = 2

> Sibling cell-set instance to
> [`PDE_C1_CELLSET_KOLMOGOROV.md`](PDE_C1_CELLSET_KOLMOGOROV.md), filed
> 2026-05-28 in response to the v0 lock disposition. The v0 cell was
> executed and deferred under the fiber-protocol vacuity gate because
> 2D Kolmogorov flow at `G = 100`, `k_f = 4` is linearly stable: the
> proxy selector never fires `damp_low_band` on the sampled support.
> v1 re-pins `k_f = 2` to enter a supercritical regime where the
> strictness predicate has discriminative content. All other v0
> parameters are inherited unchanged. This is a new cell-set instance,
> not a retune of v0 (per
> [`PDE_C1_FIBER_PROTOCOL.md`](PDE_C1_FIBER_PROTOCOL.md) section 7
> parameter-pinning discipline).
>
> Status: drafted, desk-auditable, unreviewed, no numerical verdict
> yet. The harness exposes this cell as `--preset lock_v1` (and
> `--preset fallback_v1` for the coverage-gate fall-back).

## 1. Regime Delta from v0

The single regime change is the forcing wavenumber:

```text
v0: k_f = 4   (laminar / linearly stable at G = 100)
v1: k_f = 2   (supercritical at G = 100)
```

The 2D Kolmogorov-flow linear-stability threshold decreases as `k_f`
decreases (the Meshalkin-Sinai / Iudovich tradition). At `k_f = 4`
the laminar solution is stable for moderate `G`; at `k_f = 2` the
threshold sits well below `G = 100` for standard parameterisations,
so the laminar Kolmogorov solution is expected to be unstable and the
attractor non-trivial. This is the load-bearing physical hypothesis
for v1's discriminative power.

If `k_f = 2` is also non-discriminative on the lock_v1 run, a
subsequent cell-set v2 at `k_f = 1` (or `G ≥ 1000`) is the natural
next sibling, filed as a separate artifact.

## 2. Inherited from v0

All of the following are inherited unchanged from
[`PDE_C1_CELLSET_KOLMOGOROV.md`](PDE_C1_CELLSET_KOLMOGOROV.md):

- **§1 Regime (excluding `k_f`).** 2D NSE on `T^2 = [0, 2*pi]^2`,
  `N = 16` modes per axis (`32 × 32` grid), `G = 100`, dimensionless
  forcing amplitude `1`, decision domain `B_abs` of the truncated
  system after burn-in, evaluation measure the SRB-like distribution.
- **§2 Observation.** `K = 4`, signature dimension `d_K = 32`,
  σ-algebra `F_K`. The signature includes the forced mode `(0, k_f)`
  for both v0 (`k_f = 4`) and v1 (`k_f = 2`).
- **§3 Objective.** Low-band energy safety trigger; `E_max` =
  95th percentile of `E_K` along the v1 burn-in (recomputed for v1
  because the dynamics differ); look-ahead horizon `τ = 5` time units;
  action space `{no_op, damp_low_band}`; deterministic provisional
  selector / proxy `\hat{pi}` per the fiber protocol section 3.
- **§4 Comparator.** Finite-Galerkin non-injectivity witness on
  `B_abs` (the algebraic construction is regime-independent); literature
  determining-mode reference as sufficiency ceiling only. The
  attractor-support twin-state certificate remains a deferred bridge.
- **§5 Fiber classifier.** Unchanged.
- **§6 Negative branch.** Unchanged; `PDE-C1-NEG-A` and `PDE-C1-NEG-B`
  receipts are the v1 negatives as well.
- **§7 Parameter instantiation.** All section 7 values are inherited:
  `dt = 0.01`, `T_burnin = 10^5` steps, `epsilon_K = 0.05 sqrt(2 E_max)`
  (numerically re-pinned at v1 burn-in completion since `E_max` is
  regime-dependent), `h_K = epsilon_K / sqrt(d_K)`, `n_min = 30`,
  `delta_action = 0.10`, `S_pos = 0.50`, `delta_proxy_min = 0.01`,
  `N_sample = 50,000`, sampling interval `0.5` time units, fall-back
  to `N_sample = 200,000` on coverage deferral, action tie-break
  favouring `no_op`.

## 3. New v1 Pin

- **Forcing wavenumber.** `k_f = 2`.

That is the only spec-level change. The harness preset `lock_v1`
encodes this; everything else in `build_config` resolves identically
to `lock`.

## 4. Discriminative-Power Expectation

The vacuity gate (`delta_proxy_min = 0.01`) requires `damp_fraction`
to lie in `[0.01, 0.99]` before a verdict is interpreted. Expected v1
behaviour:

- Under `E_max` = 95th percentile of burn-in `E_K`, the analytic
  expectation is that the post-burn-in trajectory crosses `E_max`
  on roughly 5% of look-ahead windows in a stationary chaotic regime
  (the exact rate depends on autocorrelation, but should sit well
  inside `[0.01, 0.99]`).
- If `damp_fraction < 0.01` on lock_v1, `k_f = 2` is also
  insufficiently supercritical at `G = 100`; file `DEFERRED_VACUITY`
  and proceed to v2 (`k_f = 1` or higher `G`).
- If `0.01 ≤ damp_fraction ≤ 0.99` on lock_v1, the predicate has
  discriminative content and the verdict is interpreted as
  `STRICTNESS_WITNESS_POSITIVE`, `PDE-C1-NEG-A`, `DEFERRED_COVERAGE`,
  or one of the fall-back paths per the fiber protocol section 5.

This expectation is **pre-registered**, not a result. It is named here
so a v1 lock that falls outside `[0.01, 0.99]` is filed cleanly as
`DEFERRED_VACUITY` rather than as a surprise.

## 5. What v1 Closes / Does Not Close

**Same closures as v0 § "What This v0 Closes And Does Not Close":**
the fiber protocol's binning predicate, the proxy selector substitution
rule, the parameter-pinning discipline. v0 and v1 jointly close the
methodology surface; only the regime differs.

**v0 was deferred for vacuity.** v1 is the discriminative-regime
sibling. A non-vacuous v1 verdict would be the first substantive read
on C1.

**Not closed by v0 or v1 alone:**

- Criterion (a) Front-A vacuity rebuttal (sidecar-level).
- Criterion (c) named external PDE reviewer.
- Attractor-support twin-state certificate.

## 6. Status

- **Drafted.** 2026-05-28.
- **Desk-auditable.** A reviewer can read v0 plus this v1 delta and
  audit the regime change without running code.
- **Unreviewed.** No external PDE reviewer has signed off on the
  `k_f = 2` choice as a faithful supercritical-regime instance.
- **Unrun.** The `lock_v1` preset is present in the harness; no
  execution has been attempted.

## 7. Cross-references

- [`PDE_C1_CELLSET_KOLMOGOROV.md`](PDE_C1_CELLSET_KOLMOGOROV.md) - v0
  cell-set, including the "Lock Execution Disposition (2026-05-28)"
  section that motivates this v1 sibling.
- [`PDE_C1_FIBER_PROTOCOL.md`](PDE_C1_FIBER_PROTOCOL.md) - methodology;
  the vacuity gate (section 5 step 8, added 2026-05-28) is what filed
  v0 as `DEFERRED_VACUITY` and would catch a same-mode failure on v1.
- [`PDE_DETERMINING_MODES_POSTULATE1.md`](PDE_DETERMINING_MODES_POSTULATE1.md)
  - the C1 sidecar; both v0 and v1 are concrete instances of the
  pre-registered cell set described there.
- [`../SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md) - the
  ledger. Promotion of C1 depends on a clean adjudication of the
  Reading-2 fiber predicate on some discriminative cell.
- [`../../scripts/pde_c1_kolmogorov_cell.py`](../../scripts/pde_c1_kolmogorov_cell.py)
  - harness. The `--preset lock_v1` switch runs this v1 cell with
  the inherited v0 parameter values plus `k_f = 2`.
