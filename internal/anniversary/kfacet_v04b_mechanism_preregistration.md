# v0.4b Mechanism Predictor Pre-Registration

Status: pre-registered, not yet run. 2026-05-22.
Audience: operator who executes the staged sweep; future paper-side writer;
v0.4c comparison reader.
Companions:
- `kfacet_v04a_domain_map_preregistration.md` (verdict-landed domain audit
  that locked the v0.4 body).
- `kfacet_v04a0_o434_anatomy.md` (methodology pre-mortem for gauge
  classifier).
- `kfacet_v03h_writeup.md` (v0.3 closure + projection-language framing).
- `docs/threebody/CROSS_SUBSTRATE_NOTES.md` §6-§7 (projection vocabulary).

## One-Line Read

With the v0.4a verdict locking all 273 supplementary-B piano-trios into
the `Z2_clean` domain, v0.4b asks the **next non-circular question**:
*Does the Z_2 neutral/tangent shadow of each orbit contain enough
structure to predict that orbit's stability flag, stratified by m_3?*
The primary predictor `gamma_3` uses only Z_2 tangent-structure
invariants (kernel dimensions, isotypic dim split, neutral-sector
conditioning) and explicitly **excludes** anything that inspects
off-unit-circle Floquet multipliers. The falsifier is a single
full-table chi-squared / likelihood-ratio test over m_3 bins with
`N >= 5`. The non-circularity is load-bearing; a pre-registered
no-variation retirement clause retires `gamma_3` as non-informative
(rather than over-fitting) if the chosen features are
nearly-constant across the catalog.

## Locked From v0.4a

```text
Body:        primary Z_2 piano-trio orbit
Projection:  orbit -> Z_2 neutral/tangent shadow
Domain:      all 273 supplementary-B rows (Z2_clean under the two-pass
             gauge classifier)
```

The v0.4a verdict simplifies v0.4b's scope: no domain triage, no
smaller-symmetry side-channel. Every catalog row contributes to the
S/U observable.

## Observable to Predict (Locked)

```text
table[m_3][stability] = (S_count, U_count) per m_3

Observed (from v0.4a manifest):
  m_3   S    U   total
  0.4  35   20    55
  0.5  15   16    31
  0.6   6   16    22
  0.7   4   14    18
  0.8   6   12    18
  0.9   8   17    25
  1.0   7   31    38
  1.1   1   22    23
  1.2   2    5     7
  1.3   2    2     4   <- N < 5, descriptive only
  1.4   0    4     4   <- N < 5, descriptive only
  1.5   4    4     8
  1.6   7    1     8
  1.7   0   10    10
  1.9   0    2     2   <- N < 5, descriptive only

Gating bins (N >= 5):  {0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 1.6, 1.7}
Descriptive-only:      {1.3, 1.4, 1.9}
```

## Primary Predictor: gamma_3 (Non-Circular Structural)

`gamma_3` predicts the per-m_3 stable fraction `P_S(m_3)` from
Z_2 tangent-structure invariants computed at each row's IC. The
prediction is then compared row-by-row against the observed stability
flag, aggregated per m_3, and tested against the binomial observation
`(S_count, U_count)`.

### Allowed gamma_3 features (Z_2 neutral/tangent invariants)

```text
1. kernel_dim:          dim ker(M_i - I) at the row's adaptive floor
                        (after neutral quotient if used).
2. F_beta_even_dim:     dim of F_beta = +I isotypic in ker(M_i - I).
3. F_beta_odd_dim:      dim of F_beta = -I isotypic in ker(M_i - I).
4. F_beta_even_odd_ratio: F_beta_even_dim / max(F_beta_odd_dim, 1).
5. bridge_band_count:   count of (M_i - I) singular values in (1e-7, 1e-3),
                        i.e., near-1 bridge-band SVs.
6. F_beta_leakage_inf:  ||(I - P_ker) F_beta P_ker||_inf, the F_beta
                        leakage outside the recovered kernel.
7. neutral_sector_conditioning:  condition number of the standard 8-dim
                        Hamiltonian/translation/boost neutral block at y0.
8. period_T:            stratification variable (NOT a fitted feature).
9. m_3:                 stratification variable.
```

### Disallowed for prediction (would make gamma_3 circular)

```text
- "has off-unit-circle Floquet pair"
- spectral radius of M_i
- unit-circle count / number of eigenvalues with |lambda| > 1
- max |lambda_i| over Floquet spectrum
- Krein signature
- any function of M_i eigenvalues (real or complex) beyond what is
  carried by the allowed kernel/isotypic-dim counts above
- the stability label itself, or anything derivable from it without
  the orbit IC
```

If `gamma_3` is constructed from any feature in this disallowed list,
the v0.4b registration is invalidated and must be re-issued.

### gamma_3 form (pre-registered, paper-side closure required)

`gamma_3` is a function

```text
gamma_3 : (allowed features) -> P_S in [0, 1]
```

The functional form is **registered paper-side before any compute** in a
separate `kfacet_v04b_gamma3_form.md` sub-document. Acceptable forms:
- Linear or low-order polynomial in the allowed features.
- Logistic on a linear combination.
- A pre-registered piecewise / threshold rule (e.g., "predict S iff
  `F_beta_even_dim >= k`").

Adaptive / fitted forms (fit the model to v0.4a's table and then test
on the same table) are **disallowed**. The pre-registration must commit
to a fixed functional form before reading the per-row table.

## Falsification Surface

### Primary falsifier (gating)

A single full-table likelihood-ratio test over the **12 gating m_3 bins
with `N >= 5`**:

```text
For each gating m_3 bin:
  N(m_3) = total rows
  S_observed(m_3) = observed stable count
  P_S_predicted(m_3) = mean of gamma_3 over the rows in that m_3 bin

H_0:  S_observed(m_3) ~ Binomial(N(m_3), P_S_predicted(m_3))

Test statistic:  -2 log Lambda  (likelihood ratio)
              =  sum over gating bins of
                 -2 [ log L(S_obs | P_S_pred) - log L(S_obs | S_obs/N) ]

Reference distribution:  chi-squared with df = 12 (number of gating bins)
                         minus the number of free parameters in gamma_3
                         (registered paper-side; e.g., for a 3-feature
                         linear model, df = 9)

Falsifier:  p-value < 0.01  <-->  gamma_3 prediction fails the table
                                  at the 99% level.
```

The test threshold (`p < 0.01`) is registered. Multiple-comparison
correction is built into the single-table aggregation; no per-bin
adjustment.

### Diagnostics (non-gating)

For each m_3 bin (including the descriptive-only `{1.3, 1.4, 1.9}`):

```text
P_S_observed(m_3) = S_count / N
P_S_predicted(m_3) = mean(gamma_3 over rows in that m_3)
binomial residual = (P_S_observed - P_S_predicted) / sqrt(P_S_pred (1-P_S_pred) / N)
2-sigma flag = |residual| > 2.0
```

Flags are reported for inspection but **do not gate the verdict**.
Per-bin 2-sigma flags on small bins are descriptive context, not
falsification.

### Pre-Registered No-Variation Retirement Clause

If any of the following holds when computed across all 273 rows:

```text
For every allowed feature f in gamma_3's input set:
    coefficient_of_variation(f) = std(f) / |mean(f)|  <  0.01
```

then `gamma_3` is **retired as non-informative** and the v0.4b verdict
becomes:

```text
v0.4b verdict (retirement):
  The Z_2 neutral/tangent shadow features chosen for gamma_3 do not
  contain enough structural variation across the 273-row catalog to
  predict stability stratification. The projection is well-defined
  but the shadow does not preserve enough information to predict the
  observed S/U distribution at the granularity of m_3 bins.
  
  This is a clean projection-limit result, not a falsification of
  v0.4. The Z_2 mechanism may still apply; it just does not carry
  stability information at this body.
```

Retirement is preferable to over-fitting. The clause prevents
post-hoc feature selection from rescuing a non-informative predictor.

### Compute-Reuse Sidecar: gamma_1 Decomposition (Non-Gating)

After `gamma_3` runs and M_i is computed for all 273 rows, an
**explicitly non-predictive** sidecar uses `gamma_1` features (the F_beta
even/odd Floquet spectrum) to attribute instability to one F_beta block:

```text
For each U row:
  Compute F_beta-even and F_beta-odd block restrictions of M_i.
  Count off-unit-circle eigenvalue pairs per block.
  Report which block carries the instability for this row.

Aggregate:
  table[m_3][F_beta_even_carries_U vs F_beta_odd_carries_U]
```

The sidecar is **not a predictor**. It is a mechanism-decomposition
diagnostic that reuses M_i computed for `gamma_3`. Its result feeds
discussion in v0.4c paper-side write-up but does not enter the
falsifier.

## Covariates (Recorded, Non-Gating)

For each row, the receipt records:

```text
- z_0  (continuous IC parameter; not used in gamma_3 fit)
- T    (period; not used in gamma_3 fit)
- m_3  (stratification only)
- gamma_3 features (allowed list above)
- stability flag (observed)
- gamma_3 prediction P_S for this row
```

Non-gating exploratory readout: residuals of `gamma_3` prediction vs
`z_0` and vs `T` within each m_3 bin. Useful as a mechanism-discovery
follow-up if the primary test passes; not part of the falsifier.

## Stratification Discipline

```text
Primary stratification:  m_3 only (15 discrete values, 12 gating bins)
Covariates recorded:     z_0 (continuous), T, stability flag
Non-gating readout:      residuals vs z_0 and T inside each m_3 bin
```

Conditioning on `(m_3, z_0)` joint bins is rejected as too sparse for
the catalog (some `(m_3, z_0)` cells have 1-2 rows). The single-axis
m_3 stratification provides the statistical power; finer-grained
analyses are exploratory.

## Compute Discipline

```text
gamma_3 needs:
  - M_i for all 273 rows (variational integration, 18-dim system over
    one period).
  - F_beta operator (catalog-style, IC-independent; reused from
    kfacet-sentinel construction).
  - Kernel basis at adaptive floor + F_beta isotypic projection.

Per-row time estimate:
  - M_i variational integration alone at rtol=1e-12: ~10-30 sec/row
  - Kernel + isotypic post-processing: ~1 sec/row
  - Total: ~15-35 sec/row

Total catalog estimate:
  273 rows * ~25 sec average = ~115 min = ~2 hours staged.

Compute is bounded by M_i computation; gamma_3 feature extraction is
trivial relative to integration. gamma_1 sidecar reuses M_i with no
additional integration.

Above the inline ~10-minute rule. Stage as operator-driven sweep.
```

## PowerShell Run Commands (Staged)

```powershell
# v0.4b primary sweep: M_i + Z_2 isotypic decomposition for all 273 supp-B rows
$m3_values = @("0.4","0.5","0.6","0.7","0.8","0.9","1.0","1.1","1.2","1.3","1.4","1.5","1.6","1.7","1.9")
foreach ($m3 in $m3_values) {
  python scripts/isotrophy_workbench.py kfacet-row-z2-sweep `
    --source B `
    --path docs/isotrophy/supplementary-B_piano-init-condit-3d.txt `
    --m3 $m3 `
    --limit 0 `
    --rtol 1e-12 `
    --atol 1e-12 `
    --closure-floor 1e-7 `
    --out "results/isotrophy/k-facet-v04b-mechanism/per_m3/m3eq$($m3)"
}

# v0.4b aggregator: emits per-row receipt + gamma_3 feature table + chi-squared
# verdict + gamma_1 sidecar attribution. Pre-registered, no per-row knobs.
python scripts/v04b_aggregator.py `
  --gamma3-form internal/anniversary/kfacet_v04b_gamma3_form.md
```

**Note:** `kfacet-row-z2-sweep` is a new workbench subcommand to be
written at v0.4b implementation time. Its job is to compute M_i for a
row and emit the `gamma_3` allowed-feature row plus the M_i.npy for
the sidecar. Spec is in **Aggregator + sweep-command schema** below.

## Aggregator + Sweep-Command Schema

```text
kfacet-row-z2-sweep:
  Input:  catalog row (source + path + m_3 + index/indices)
  Computes:
    - M_i via variational integration
    - kernel basis at --closure-floor
    - F_beta-even/odd isotypic dim split
    - bridge_band_count (SVs of (M_i-I) in (1e-7, 1e-3))
    - F_beta_leakage_inf
    - neutral_sector_conditioning
  Saves:
    M_i.npy, F_beta_isotypic_basis.npy, gamma_3_features.json
  Receipt schema (per row):
    {row_index, label, m_3, z_0, period, stability,
     gamma_3_features: {kernel_dim, F_beta_even_dim, F_beta_odd_dim,
                        F_beta_even_odd_ratio, bridge_band_count,
                        F_beta_leakage_inf, neutral_sector_conditioning},
     M_i_norm_inf, integration_seconds}

v04b_aggregator:
  Input:  per-row receipts under per_m3/m3eq*/{O*.json}
          + paper-side gamma_3 form spec
  Computes:
    - gamma_3 prediction P_S for each row
    - per-m_3 (P_S_pred, P_S_obs, residual)
    - likelihood-ratio test over gating bins
    - no-variation retirement check
    - gamma_1 sidecar: F_beta block Floquet attribution per U row
  Emits:
    manifest.json with verdict in
      {pass, fail, retired_no_variation}
    per_row_table.csv
    per_m3_table.csv (for diagnostics)
    gamma_1_sidecar_attribution.csv (non-gating)
```

## Sequencing After v0.4b

```text
v0.4a (DONE):  domain map; 273/273 Z2_clean.
v0.4b (THIS):  gamma_3 structural predictor + chi-squared falsifier.
v0.4c:         read verdict.
               - pass:  v0.4 mechanism positive; paper-side write-up.
               - fail:  gamma_3 form is wrong; propose gamma_3' for v0.4b'.
               - retired_no_variation: v0.4 projection-limit; paper-side
                 record that the shadow does not carry stability info.
```

After v0.4c, regardless of outcome:

- The gamma_1 sidecar attribution informs paper-side discussion of
  which F_beta block (even or odd) carries instability where present.
- Recorded covariates (z_0, T) enable follow-up analyses if the
  primary verdict is `pass` or `retired_no_variation`.

## What This Pre-Registration Does NOT Do

- **No on-the-fly fitting of gamma_3.** The functional form is locked
  in `kfacet_v04b_gamma3_form.md` paper-side before any per-row
  prediction is computed.
- **No use of stability information for prediction.** Stability is
  observed, not predicted-from.
- **No per-bin gating.** The chi-squared test aggregates over all
  gating bins; per-bin 2-sigma flags are diagnostic only.
- **No revision of v0.4a.** The Z2_clean domain is locked.
- **No revisit of v0.3.** The v0.3 epilogue's domain-of-applicability
  finding is unaffected.

## What Falsifies the v0.4b Registration Itself

If at implementation time we discover any of:

- A feature listed in "Allowed" is structurally equivalent to a
  forbidden feature (e.g., `bridge_band_count` is shown to be
  one-to-one with off-unit-circle pair count).
- The gamma_3 form cannot be expressed without using a disallowed
  feature.
- The chi-squared test's degrees of freedom calculation is wrong.

then the pre-registration is re-issued. The verdict is not read against
a pre-registration that itself is invalidated.

## How To Verify This Pre-Registration Is Still Current

1. Re-parse supp-B: `npm run isotrophy:parse:b` returns 273 rows;
   per-m_3 counts match the observed table above.
2. Confirm v0.4a verdict: open
   `results/isotrophy/k-facet-v04a-domain-map/manifest.json` and check
   `verdict = outcome_A_all_Z2_clean`, `class_counts.Z2_clean = 273`.
3. Confirm `kfacet_v04b_gamma3_form.md` exists and registers the
   functional form before any per-row prediction is generated.

If any of (1)-(3) drifts, this pre-registration should be re-issued
before any v0.4b sweep launch.
