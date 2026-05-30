# PDE C1 Twin-State Certificate

> Pre-registration and execution harness for the C1 support-level
> state-insufficiency bridge. Filed 2026-05-28 after the v4/v5 kNN
> convergence checks delivered a provisional, dimension-robust
> control-sufficiency positive at `(k_f = 2, G = 200)`. This artifact
> tests the remaining state-insufficiency-on-attractor half: whether
> `Phi_K` is non-injective on the sampled SRB-like support, not merely
> on the absorbing ball `B_abs`.

## 1. Claim Boundary

This certificate does **not** adjudicate proxy-control sufficiency. That
was the role of the kNN convergence check
([`PDE_C1_KNN_CONVERGENCE_CHECK.md`](PDE_C1_KNN_CONVERGENCE_CHECK.md)).
It also does not prove a theorem about the exact infinite-dimensional
Navier-Stokes attractor.

It claims only this, if the positive branch fires:

```text
On the finite-Galerkin sampled support for the pinned Kolmogorov cell,
there is positive-mass numerical evidence of distinct states with the
same low signature Phi_K and separated complementary high-mode state Q_K.
```

That closes the C1 regime-2 state-insufficiency bridge at the sampled
support level for the tested cell. C1 still remains unpromoted until
proxy faithfulness and external PDE review are handled.

## 2. Target Cell

Primary target:

```text
--preset lock_v5
k_f = 2
G = 200
K = 3
d_K = 18
N_twin = 50,000
```

Why v5 first: v5 is the lower-dimensional replication of the v4
positive, with near-identical boundary-shell slope and intercept. A
support certificate at v5 is therefore the tightest direct companion to
the dimension-robust control-sufficiency read.

Optional cross-check:

```text
--preset lock_v4
k_f = 2
G = 200
K = 4
d_K = 32
N_twin = 50,000
```

The v4 run is a replication check, not a precondition for the v5
certificate.

## 3. State Coordinates

The low signature `Phi_K` is exactly the existing harness signature:
half-plane Fourier representatives, normalized as

```text
omega_hat(k) / (M^2 |k|)
```

with real and imaginary parts split into Euclidean coordinates.

The complementary high-mode projection `Q_K` is implemented by the same
normalization over all active de-aliased half-plane Galerkin modes not
included in `Phi_K`. This keeps the low-signature and high-mode
separation norms in the same coordinate convention.

For v5, the harness reports:

```text
high_mode_count = 211
high_mode_dimension = 422
```

for the active de-aliased complement. For v4, the complement is smaller
because `Phi_K` includes more low modes.

## 4. Pair Search

For each sampled state `u_i`, query a `BallTree` in signature space for
the `k_twin = 50` nearest neighbours, including self. Discard self. A
directed candidate edge `(i, j)` is signature-near iff:

```text
||Phi_K(u_i) - Phi_K(u_j)|| <= epsilon_K
```

where `epsilon_K` is inherited from the C1 cell:

```text
epsilon_K = 0.05 * sqrt(2 E_max)
```

and `E_max` is computed exactly as in the selected preset. Coverage is
measured at the sample level:

```text
candidate_sample_fraction =
  |{ i : i has at least one non-self signature-near neighbour }| / N_twin
```

The coverage gate is inherited:

```text
candidate_sample_fraction >= S_pos = 0.50
```

## 5. High-Mode Separation Threshold

Before reading the full lock result, pin:

```text
delta_H = max(1e-6, 0.05 * median_i ||Q_K(u_i)||)
```

Rationale: the certificate should not count machine-noise high-mode
differences as state separation, but the threshold should scale with the
actual high-mode amplitude of the sampled attractor. The formula uses
only the sample's marginal high-mode norm distribution, not the
candidate-pair separations.

A witness edge is a signature-near candidate edge satisfying:

```text
||Q_K(u_i) - Q_K(u_j)|| >= delta_H
```

## 6. Branches

Pre-registered branches, in order:

1. **`SMOKE_ONLY`** if the preset is not verdict-bearing or manual
   overrides are used.
2. **`TWIN_STATE_DEFERRED_HIGH_MODE_FLOOR`** if
   `median_i ||Q_K(u_i)|| <= 1e-6`. The sampled support is numerically
   flat in complementary modes at this scale.
3. **`TWIN_STATE_DEFERRED_COVERAGE`** if
   `candidate_sample_fraction < 0.50`. The sample does not provide enough
   signature-near candidate pairs.
4. **`TWIN_STATE_CERTIFIED`** iff both hold:

```text
witness_sample_fraction >= 0.01
unique_witness_pairs >= 100
```

where `witness_sample_fraction` is the fraction of all samples with at
least one witness edge, and `unique_witness_pairs` canonicalizes
directed edges to unordered pairs.

5. **`TWIN_STATE_NO_CERTIFICATE`** otherwise. This is explicitly a
   no-certificate receipt, not a proof of injectivity.

## 7. Harness

Implemented in
[`../../scripts/pde_c1_kolmogorov_cell.py`](../../scripts/pde_c1_kolmogorov_cell.py)
as:

```powershell
python scripts\pde_c1_kolmogorov_cell.py --preset lock_v5 --adjudicator twin-state --out results\proof\c1-kolmogorov-v5-twin-state
```

Receipt files:

```text
manifest.json
PDE_C1_KOLMOGOROV_RESULTS.md
twin-state-witnesses.csv
```

The witness CSV records a capped set of witness examples for audit:
sample ids, signature distance, high-mode distance, high-distance over
`delta_H`, and the two high-mode norms.

## 8. Smoke Receipt

Smoke command, run 2026-05-28:

```powershell
python scripts\pde_c1_kolmogorov_cell.py --preset smoke --adjudicator twin-state --allow-unregistered-overrides --sample-count 80 --burnin-steps 200 --sample-interval-steps 3 --lookahead-steps 10 --out results\proof\c1-twin-state-smoke
```

Receipt: `results/proof/c1-twin-state-smoke/`.

Readout:

```text
status = SMOKE_ONLY
elapsed_seconds = 2.030
candidate_sample_fraction = 1.0
witness_sample_fraction = 1.0
```

The smoke validates plumbing only: high-mode capture, BallTree query,
candidate/witness aggregation, manifest writing, receipt writing, and
witness CSV output. It files no support-level certificate.

## 9. Full-Run Commands

Do not run these inline under the repo's ~10-minute rule. Expected
wall-clock is approximately 20-30 minutes for v5, based on the v5 lock
and v5 kNN-sweep receipts.

Primary:

```powershell
python scripts\pde_c1_kolmogorov_cell.py --preset lock_v5 --adjudicator twin-state --out results\proof\c1-kolmogorov-v5-twin-state
```

Optional replication:

```powershell
python scripts\pde_c1_kolmogorov_cell.py --preset lock_v4 --adjudicator twin-state --out results\proof\c1-kolmogorov-v4-twin-state
```

If v5 returns `TWIN_STATE_CERTIFIED`, the C1 ledger can mark the
state-insufficiency-on-attractor bridge closed for the tested regime.
If it returns any deferral or no-certificate receipt, C1 remains at the
current honest boundary: non-injectivity certified on `B_abs`, not yet
on `supp(mu_SRB)`.

## 11. Result (2026-05-28) — TWIN_STATE_CERTIFIED at v5

Primary v5 run (`results/proof/c1-kolmogorov-v5-twin-state/`, ~21 min):
**`TWIN_STATE_CERTIFIED`**.

| quantity | value | gate |
|---|---:|---|
| `delta_H` | 0.01167 | `0.05 × median‖Q_K‖`, floor `1e-6` |
| median ‖Q_K‖ / min / max | 0.2333 / 0.2202 / 0.2442 | — |
| signature-near coverage | 1.00 | ≥ `S_pos = 0.50` |
| candidate pairs (unique) | 1,263,121 | — |
| witness sample fraction | **1.00** | ≥ 0.01 |
| witness pairs (unique) | **693,795** | ≥ 100 |
| witness high-dist p50 / p95 | 0.0154 / 0.0210 | ≥ `delta_H` |

**Robustness / non-degeneracy checks:**

- **Not machine noise.** `delta_H` is set by the real high-mode
  amplitude (median ‖Q_K‖ = 0.23 ≈ 27% of the ‖Φ_K‖ scale ~0.85), far
  above the `1e-6` floor.
- **Not "any two points."** Candidates are within `ε_K = 0.0606`; the
  50-NN radius at this regime is ~0.045 (~5% of the signature norm), so
  signature-near is genuinely signature-local.
- **Overwhelmingly above the gates** (100% witness coverage; 694k
  unique witness pairs vs the 100 threshold), not marginal.

**What this closes.** `Phi_K` is non-injective on the sampled SRB-like
support for the v5 cell — the state-insufficiency-on-attractor half
that §4.1 of the cell-set had only established algebraically on
`B_abs`. Tested at the **same `ε_K`** as the kNN control-sufficiency
read, so the two halves compose into a complete Reading-2 **regime 2**
witness at this cell: within `ε_K` signature-balls, Q_K varies
(certified here) while the proxy action stays constant up to
measure-zero (the kNN POSITIVE). The signature collapses distinct full
states yet suffices for the control objective.

**Sharpened 2026-05-29 (paired fiber-constancy).** The composition above
is at a *matched radius* — two population statistics over the same
`ε_K`-balls. [`PDE_C1_PAIRED_FIBER_CONSTANCY.md`](PDE_C1_PAIRED_FIBER_CONSTANCY.md)
tightens it to a **paired** test on the *same* witness pairs: among the
certified `Q_K`-separated pairs, the proxy-action disagreement is
`D_witness = 0.0367` (G=200) / `0.0382` (G=300), both well under the
`delta_action = 0.10` line and within ~1 point of the candidate-pair rate
(`0.0319` / `0.0290`). High-mode separation adds almost nothing to action
disagreement — the residual is a signature-space boundary layer, not
`Q_K`-driven. Both runs reproduced this certificate bit-for-bit
(`PAIRED_FIBER_CONSTANCY_POSITIVE`), composing state-insufficiency and
control-sufficiency on the same pairs in both regimes.

**Honest calibration.** This was the *expected-easy* half — on a
chaotic attractor with a 422-dim high-mode complement at ~0.23
amplitude, signature-local states with separated high modes are nearly
automatic. The certificate's contribution is that it is now *measured*
on the actual support with pre-registered thresholds, not argued.

**What it does NOT do.** Finite-Galerkin, sampled-support, numerical —
not a theorem about the infinite-dimensional NSE attractor. One cell
(k_f=2, G=200, K=3); not regime-generality. Does not touch proxy
faithfulness or external review. **C1 stays unpromoted.**

## 10. Cross-References

- [`PDE_C1_CELLSET_KOLMOGOROV.md`](PDE_C1_CELLSET_KOLMOGOROV.md) section
  4.1 - the original absorbing-ball non-injectivity witness and
  attractor-support caveat.
- [`PDE_C1_FIBER_PROTOCOL.md`](PDE_C1_FIBER_PROTOCOL.md) section 6 - the
  one-line deferred support-level certificate spec.
- [`PDE_C1_KNN_CONVERGENCE_CHECK.md`](PDE_C1_KNN_CONVERGENCE_CHECK.md) -
  the control-sufficiency result this state-insufficiency bridge
  complements.
- [`../SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md) - the
  staging ledger that keeps this certificate as an open firm-up item
  until the full run returns.
