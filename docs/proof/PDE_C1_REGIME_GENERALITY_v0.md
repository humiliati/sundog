# PDE C1 Regime Generality v0 - G=300 Kolmogorov Cell

> Pre-registration for the first regime-generality probe after the v5
> end-to-end Reading-2 regime-2 witness. Filed 2026-05-28 before any
> verdict-bearing run at this cell. This artifact pins exactly one new
> regime: hold the objective, forcing geometry, signature dimension,
> tolerance, kNN convergence adjudicator, and twin-state certificate
> rules fixed; raise the Grashof number from `G = 200` to `G = 300`.

## 1. Question

The v5 cell established the complete empirical regime-2 pattern:

```text
state-insufficient: Phi_K non-injective on sampled SRB support
control-sufficient: proxy action locally determined by Phi_K up to a
                    measure-zero decision surface
```

But that result is cell-local. The open question here is:

```text
Does the same Reading-2 regime-2 pattern survive a higher-Grashof
Kolmogorov-flow regime, or was (k_f = 2, G = 200) a sweet spot?
```

This is a regime-generality test, not a dimension check and not an
objective-dependence test.

## 2. Pinned Cell

Primary new cell:

```text
name: lock_v6
k_f = 2
G = 300
K = 3
d_K = 18
N_sample = N_twin = 50,000
```

Full parameter pin:

| parameter | value | relation to v5 |
|---|---:|---|
| grid size | 32 | unchanged |
| active Galerkin cutoff | 16 | unchanged |
| forcing wavenumber `k_f` | 2 | unchanged |
| forcing amplitude | 1.0 | unchanged |
| Grashof `G` | 300 | changed from 200 |
| viscosity `nu = sqrt(1/G)` | 0.057735 | changed from 0.070711 |
| signature mode count `K` | 3 | unchanged |
| signature dimension `d_K` | 18 | unchanged |
| burn-in steps | 100,000 | unchanged |
| sample count | 50,000 | unchanged |
| sample interval | 50 steps | unchanged |
| lookahead horizon | 500 steps / 5.0 time units | unchanged |
| integration step `dt` | 0.01 | unchanged |
| `E_max` rule | 95th percentile of last 25% burn-in | unchanged from v4/v5 |
| `epsilon_K` | `0.05 * sqrt(2 E_max)` | unchanged formula |
| random seed | 20260528 | unchanged |

There is no pre-registered fall-back sample count for this v0
regime-generality probe. A deferral is a real receipt, not permission
to tune this same cell after reading.

## 3. Why This Cell

`G = 300` is the smallest already-explored upward regime move with
evidence of nontrivial post-burn-in dynamics. The earlier v2 run at
`G = 300, K = 4` broke the energy-conservation symmetry but was nearly
vacuous under the old full-burn-in `E_max` rule (`damp_fraction =
0.0086`) and failed hard-bin coverage. This v6 cell applies the lessons
that were learned after v2:

- keep `k_f = 2`, so the forcing geometry and forced-mode inclusion are
  unchanged;
- keep `K = 3`, matching the v5 completed witness and retaining the
  forced mode `(0, 2)`;
- keep the v4/v5 `E_max` amendment (`last 25%` of burn-in), so
  transient contamination is not allowed to silently decide vacuity;
- use the kNN convergence check and twin-state certificate rather than
  hard bins.

If the v6 cell passes, the C1 result is no longer a one-regime
phenomenon. If it fails, the failure is informative: regime-2 may be a
localized window rather than a substrate-wide pattern.

## 4. What Is Held Fixed

No objective retune:

```text
proxy pi_hat(u) = damp_low_band iff no-op lookahead carries E_K above E_max
```

No adjudicator retune:

```text
kNN sweep: k in {10, 15, 20, 25, 30, 40, 50}
primary statistic: mean_minority vs r_k_median
POSITIVE threshold: a_mm <= 0.005
NEG-A threshold: a_mm >= 0.015
```

No twin-state retune:

```text
k_twin = 50
delta_H = max(1e-6, 0.05 * median ||Q_K||)
witness_sample_fraction gate = 0.01
unique_witness_pairs gate = 100
```

Post-hoc changes to the regime, objective, `K`, `E_max` window,
neighbourhood sweep, `delta_H`, or thresholds after reading a receipt
are filed as `PDE-C1-RG-NEG-B` for this artifact.

## 5. Run Order

Run the control-sufficiency half first, because it is the at-risk half:

```powershell
python scripts\pde_c1_kolmogorov_cell.py --preset lock_v6 --adjudicator knn-sweep --out results\proof\c1-regime-g300-v6-knn-sweep
```

If and only if that returns `STRICTNESS_WITNESS_POSITIVE`, run the
support-level state-insufficiency companion:

```powershell
python scripts\pde_c1_kolmogorov_cell.py --preset lock_v6 --adjudicator twin-state --out results\proof\c1-regime-g300-v6-twin-state
```

The twin-state run may still be executed for diagnostic curiosity after
a control failure, but it is not required for the regime-generality
decision: without control-sufficiency, the cell is not a regime-2
replication.

Expected wall-clock: approximately 25-40 minutes for the kNN sweep and
20-35 minutes for the twin-state run on the local CPU, based on the
v5 receipts and the higher `G`. Do not run the verdict-bearing commands
inline under the repo's ~10-minute rule.

## 6. Branches

After the kNN sweep:

| kNN result | regime-generality branch | meaning |
|---|---|---|
| `STRICTNESS_WITNESS_POSITIVE` | proceed to twin-state | control-sufficiency replicated at `G = 300` |
| `PDE-C1-NEG-A` | `PDE-C1-RG-NEG-A` | higher regime is control-insufficient under the proxy |
| `DEFERRED_VACUITY` | `PDE-C1-RG-DEFERRED_VACUITY` | proxy is not discriminative at this cell |
| `DEFERRED_FIDELITY_COVERAGE` | `PDE-C1-RG-DEFERRED_COVERAGE` | kNN radius fidelity failed |
| `INCONCLUSIVE_CONVERGENCE` | `PDE-C1-RG-INCONCLUSIVE_CONTROL` | zero-radius trend is ambiguous |

After the twin-state companion, if reached:

| twin-state result | regime-generality branch | meaning |
|---|---|---|
| `TWIN_STATE_CERTIFIED` | `PDE-C1-RG-POS` | complete regime-2 witness replicated at `G = 300` |
| `TWIN_STATE_DEFERRED_*` | `PDE-C1-RG-INCONCLUSIVE_SUPPORT` | control replicated, support bridge not closed |
| `TWIN_STATE_NO_CERTIFICATE` | `PDE-C1-RG-INCONCLUSIVE_SUPPORT` | no support-level certificate; not proof of injectivity |

## 7. Interpretation Rules

`PDE-C1-RG-POS` means:

```text
The v5 Reading-2 regime-2 witness has one higher-Grashof replication
under the same objective and same epsilon_K construction.
```

It does **not** mean:

- generality across all `G`;
- a theorem about the infinite-dimensional NSE attractor;
- proof that the proxy is `J`-optimal;
- promotion without external PDE review.

`PDE-C1-RG-NEG-A` means:

```text
The higher-Grashof cell does not preserve the proxy control-sufficiency
pattern. The v5 witness remains valid but should be read as a localized
cell result, not substrate-general evidence.
```

Any deferral or inconclusive branch preserves the current C1 status and
does not move the ledger.

## 8. Harness

The harness exposes the pinned cell as:

```text
--preset lock_v6
```

implemented in
[`../../scripts/pde_c1_kolmogorov_cell.py`](../../scripts/pde_c1_kolmogorov_cell.py).
`lock_v6` is verdict-bearing; manual overrides force `SMOKE_ONLY` as in
the earlier cells.

Smoke commands for plumbing only:

```powershell
python scripts\pde_c1_kolmogorov_cell.py --preset lock_v6 --adjudicator knn-sweep --allow-unregistered-overrides --sample-count 120 --burnin-steps 300 --sample-interval-steps 3 --lookahead-steps 20 --out results\proof\c1-regime-g300-v6-smoke-knn

python scripts\pde_c1_kolmogorov_cell.py --preset lock_v6 --adjudicator twin-state --allow-unregistered-overrides --sample-count 120 --burnin-steps 300 --sample-interval-steps 3 --lookahead-steps 20 --out results\proof\c1-regime-g300-v6-smoke-twin
```

Smoke receipts, run 2026-05-28:

| receipt | status | elapsed seconds | purpose |
|---|---|---:|---|
| `results/proof/c1-regime-g300-v6-smoke-knn/` | `SMOKE_ONLY` | 2.436 | verifies `lock_v6` kNN-sweep plumbing |
| `results/proof/c1-regime-g300-v6-smoke-twin/` | `SMOKE_ONLY` | 2.525 | verifies `lock_v6` high-mode capture and twin-state plumbing |

These smokes use manual overrides and file no scientific result. The
short burn-in is intentionally non-representative; only parser/config,
integrator, adjudicator, manifest, and receipt wiring are checked.

## 9. Cross-References

- [`PDE_C1_KNN_CONVERGENCE_CHECK.md`](PDE_C1_KNN_CONVERGENCE_CHECK.md) -
  the control-sufficiency adjudicator and thresholds reused here.
- [`PDE_C1_TWIN_STATE_CERTIFICATE.md`](PDE_C1_TWIN_STATE_CERTIFICATE.md) -
  the support-level state-insufficiency certificate reused here.
- [`PDE_C1_CELLSET_KOLMOGOROV_v2.md`](PDE_C1_CELLSET_KOLMOGOROV_v2.md) -
  the earlier `G = 300` hard-bin attempt whose limitations motivate the
  cleaned v6 retest.
- [`PDE_C1_CELLSET_KOLMOGOROV_v5.md`](PDE_C1_CELLSET_KOLMOGOROV_v5.md) -
  the completed `G = 200, K = 3` witness this cell tests against.
- [`../SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md) - the
  ledger that tracks C1 promotion boundaries.
