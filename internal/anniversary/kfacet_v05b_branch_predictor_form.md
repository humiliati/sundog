# v0.5b Branch-Predictor Registration (Held-Out)

Status: **VERDICT LANDED 2026-05-23**. The held-out predictor fails:
`accuracy_model = 0.6198` vs always-`U` `0.6388`, with McNemar
`win = 28`, `loss = 33`, `p = 1.0`. Verdict:
`branch_predictor_fails_heldout`.

Audience: v0.5b runner; paper-side reviewer of the v0.5 transition from
audit to prediction.

Companions:

- `kfacet_v05a_branch_map_form.md` -- v0.5a audit verdict and active
  branch hash.
- `kfacet_v04_writeup.md` -- v0.4 chapter close that opened the branch
  program.
- `kfacet_v04a_domain_map_preregistration.md` -- supp-B domain map source.

Frame: v0.5a showed that the branch shadow `(m_3 < 1, z_0 < 0.3)` carries
stability information. v0.5b asks whether that shadow predicts held-out
stability better than the catalog-skew baseline.

## Verdict (Landed, 2026-05-23)

The primary leave-one-`m_3`-bin-out gate fails:

```text
gating rows:       263
gating S/U:        95 S / 168 U
model accuracy:    0.619772
always-U accuracy: 0.638783
accuracy delta:   -0.019011

McNemar:
  win:             28
  loss:            33
  n_discordant:    61
  p-value:         1.0  [win <= loss short-circuit]

verdict:           branch_predictor_fails_heldout
```

Structural reading:

> v0.5a's branch hash stratifies stability in-sample, but the branch-majority
> predictor does not generalize across held-out mass bins. The branch shadow
> is a descriptive catalog partition, not a mass-bin-held-out predictive
> mechanism under this first registered rule.

The fold table localizes the failure. The `m_3 = 0.4` bin is the dominant
stable fold (`35 S / 20 U`). When it is held out, the remaining low-mass,
low-`z_0` rows train that branch as `U` (`28 S / 33 U`), so the model misses
the large `m_3 = 0.4` stable block. In the other low-mass folds, the same
branch trains as `S`, but the stable-row wins and unstable-row false positives
nearly cancel (`28` wins vs `33` losses pooled).

Sidecars:

```text
single-rule sidecar:
  rule:        predict S iff (m_3 < 1 and z_0 < 0.3)
  accuracy:    0.688213 vs always-U 0.638783
  McNemar:     win = 63, loss = 50, p = 0.129431
  verdict:     fails sidecar threshold

random-half sidecar:
  seed:        20260523
  accuracy:    0.688213 vs always-U 0.638783
  McNemar:     win = 63, loss = 50, p = 0.129431
  verdict:     fails sidecar threshold
```

Receipt paths:

```text
results/isotrophy/k-facet-v05b-branch-predictor/manifest.json
results/isotrophy/k-facet-v05b-branch-predictor/per_fold_table.csv
results/isotrophy/k-facet-v05b-branch-predictor/per_row_predictions.csv
results/isotrophy/k-facet-v05b-branch-predictor/branch_training_table.csv
results/isotrophy/k-facet-v05b-branch-predictor/sidecar_single_rule.csv
results/isotrophy/k-facet-v05b-branch-predictor/sidecar_random_half.csv
scripts/v05b_branch_predictor.py
```

## Four Decisions Locked By This Form

1. **Held-out partition**: leave-one-`m_3`-bin-out is the primary gate.
   The gating folds are the 12 `m_3` bins with `N >= 5`. This respects the
   bifurcation/cross-mass structure. A deterministic random half split is
   reserved as a non-gating robustness sidecar.

2. **Predictor form**: keep the full active 4-bucket `(b1, b2)` hash from
   v0.5a. Do NOT narrow to the single rule `S iff (m_3 < 1 and z_0 < 0.3)`
   as the primary form. The single rule is recorded as a sidecar only.

3. **Falsifier**: compare held-out branch-majority predictions against the
   always-`U` baseline using a one-sided exact McNemar/binomial test on
   discordant rows, with `p <= 0.01` and positive accuracy delta required
   for a pass.

4. **Diagnostic surface**: continuous quantities (`T`, `abs(v_z)`, `E`,
   `|L|`, and derived ratios) remain diagnostic-only. They are candidate
   v0.5c features, not v0.5b features.

## Body / Projection / Observable

```text
Body:
  supplementary-B piano-trio orbit as primary branch object
  domain: 273 rows, already verified Z2_clean by v0.4a

Projection:
  catalog branch shadow
    row -> active branch label from (m_3 < 1, z_0 < 0.3)

Prediction target:
  stability label in {S, U}

Primary observable:
  held-out row-level prediction accuracy vs always-U, pooled across
  leave-one-m_3-bin-out folds
```

The v0.5b predictor is intentionally catalog-side and orbit-input-only. It
does not use Floquet spectra, monodromy data, Z2 tangent decompositions, or
post-integration stability information to construct branch labels.

## Inputs

Primary input:

```text
results/isotrophy/k-facet-v05a-branch-map/per_row_table.csv
```

This table supplies, per row:

```text
label, index, m3, z0, period, stability, vz, abs_vz,
m3_z0_squared, branch_label, b1_m3_lt_1, b2_z0_lt_0p3,
b3_abs_vz_lt_1e_minus_6, b4_m3_z0_sq_lt_2
```

The v0.5b runner may also read:

```text
results/isotrophy/k-facet-v05a-branch-map/manifest.json
```

only to verify that the active bits are exactly:

```text
b1 = (m_3 < 1)
b2 = (z_0 < 0.3)
```

No new orbit integration, tangent integration, or gauge minimization is
authorized by this form.

## Gating Folds

Primary leave-one-`m_3`-bin-out folds are the 12 bins with `N >= 5`:

```text
m_3   N    S    U
0.4   55   35   20
0.5   31   15   16
0.6   22    6   16
0.7   18    4   14
0.8   18    6   12
0.9   25    8   17
1.0   38    7   31
1.1   23    1   22
1.2    7    2    5
1.5    8    4    4
1.6    8    7    1
1.7   10    0   10
```

Total gating rows: `263`.

The three small bins are report-only and excluded from the primary gate:

```text
m_3   N    S    U
1.3    4    2    2
1.4    4    0    4
1.9    2    0    2
```

They may be scored in a supplemental table using the predictor trained on
all 263 gating rows, but they do not enter the primary p-value, accuracy, or
verdict.

## Predictor Algorithm

For each gating fold `f`:

```text
test_f  = rows with m_3 == f
train_f = rows in the 12 gating bins with m_3 != f
```

For each active branch label `b` in the training set:

```text
S_train(b) = number of training rows in branch b with stability S
U_train(b) = number of training rows in branch b with stability U
```

Class prediction for a test row in branch `b`:

```text
if S_train(b) > U_train(b):
    predict S
else:
    predict U
```

Ties therefore predict `U`. This is the conservative direction against the
always-`U` baseline. It avoids creating an `S` call when the training fold
does not support one.

Absent-branch fallback:

```text
if branch b has no training rows:
    predict the global training majority class
    ties again predict U
```

The absent-branch fallback is not expected on supp-B, but it is specified to
make the runner total.

Probability diagnostics, not gating:

```text
p_hat_S(b) = (S_train(b) + 0.5) / (S_train(b) + U_train(b) + 1.0)
```

This Jeffreys-smoothed branch probability is emitted for log-loss/Brier
diagnostics only. The primary verdict uses the class prediction above.

## Primary Baseline

The baseline predicts `U` for every held-out row:

```text
baseline(row) = U
```

This is the catalog-skew baseline. On the full v0.5a catalog it achieves
`176 / 273 = 64.47%`; on the 263-row gating subset the runner must compute
and record the exact baseline accuracy before scoring v0.5b.

## Primary Test: Exact McNemar Against Always-U

Pool predictions across the 12 leave-one-`m_3` folds.

Define paired discordance counts:

```text
win  = # rows where v0.5b predictor is correct and always-U is wrong
loss = # rows where v0.5b predictor is wrong   and always-U is correct
```

Rows where both are correct or both are wrong do not enter McNemar's
discordant-pair test.

The one-sided exact McNemar p-value is:

```text
n_discordant = win + loss
p_value = P[ Binomial(n_discordant, 0.5) >= win ]
```

If `win <= loss`, set `p_value = 1.0`.

Primary verdicts:

```text
if n_discordant < 10:
    verdict = branch_predictor_inconclusive_low_discordance

elif (accuracy_model > accuracy_always_U) and (win > loss) and (p_value <= 0.01):
    verdict = branch_predictor_passes_heldout

else:
    verdict = branch_predictor_fails_heldout
```

The p-value threshold is locked at `0.01`. No multiple-comparison correction
is needed for the primary gate because there is exactly one primary
held-out test.

## Secondary Metrics (Report-Only)

The runner records, but does not gate on:

```text
accuracy_model
accuracy_always_U
accuracy_delta
per_fold_accuracy_delta
balanced_accuracy_model
balanced_accuracy_always_U
Brier score using p_hat_S
log loss using p_hat_S
per_branch training S fractions by fold
```

Balanced accuracy is useful because supp-B is U-skewed, but it is not the
primary gate. The primary gate remains paired row-level improvement over
always-`U`.

## Sidecar A: Fixed Single-Rule Predictor

The following rule is recorded as a non-gating sidecar:

```text
predict S iff (m_3 < 1 and z_0 < 0.3)
predict U otherwise
```

This sidecar is NOT the v0.5b primary predictor because it bakes in the
v0.5a observed direction of the effect. It is useful as a readability check:
if it agrees with the fold-trained 4-bucket predictor on most rows, the
branch-shadow effect has a simple public explanation; if it diverges, the
full four-bucket form was necessary.

## Sidecar B: Deterministic Random-Half Robustness Split

The random split is not primary. If implemented, it is a robustness sidecar
only:

```text
seed:              20260523
split:             deterministic 50/50 row split over the 263 gating rows
train/test:        train branch-majority predictor on half A, test on half B;
                   train on half B, test on half A; pool both directions
primary-like test: same exact McNemar calculation, report-only
```

This sidecar probes whether the branch-majority rule survives a row-level
split. It does not replace leave-one-`m_3`-bin-out because a random split can
place near-duplicate mass-bin structure in both train and test.

## Continuous Diagnostics Are Held Back

The following are emitted but not used:

```text
T
abs(v_z)
sign(v_z)
E
|L|
m_3 * z_0^2
K / |E|
T-normalized quantities
```

Reason: v0.5a's pass was produced by the branch hash alone. Promoting
continuous quantities in v0.5b would turn the first held-out predictor into a
feature-search step. They are reserved for v0.5c if v0.5b fails or if the
per-fold residuals suggest a specific next mechanism.

Before any continuous ratio becomes gating, the reference scales must be
registered explicitly (`T_kepler`, `mu_eff`, energy normalization, and any
log/ratio transforms).

## Edge Cases

1. **Branch absent in training**: use global training majority, tie -> `U`;
   record `absent_branch_fallback_used = true`.

2. **Fold has only one stability class**: score normally. McNemar pooling
   absorbs this; the per-fold table records the one-class fold.

3. **Small `m_3` bins (`N < 5`)**: excluded from the primary gate, reported
   only. They must not be quietly added to the gate after seeing the result.

4. **Exact 50/50 branch training split**: branch predicts `U` by the tie rule.

5. **Metric disagreement**: if McNemar passes but accuracy delta is not
   positive, verdict is fail. If accuracy delta is positive but McNemar does
   not pass, verdict is fail. Both are required.

## Receipt Schema

Planned output:

```text
results/isotrophy/k-facet-v05b-branch-predictor/
  manifest.json
  per_fold_table.csv
  per_row_predictions.csv
  branch_training_table.csv
  sidecar_single_rule.csv
  sidecar_random_half.csv       (optional if implemented)
```

`manifest.json` fields:

```text
mode
version = "v0.5b-branch-predictor-heldout"
form_lock = "internal/anniversary/kfacet_v05b_branch_predictor_form.md"
input_per_row_table
gating_m3_bins
excluded_report_only_m3_bins
active_bits
primary_partition = "leave_one_m3_bin_out"
predictor_form = "fold_trained_branch_majority_on_active_b1_b2"
tie_rule = "U"
baseline = "always_U"
N_gate
S_gate
U_gate
accuracy_model
accuracy_always_U
accuracy_delta
win
loss
n_discordant
mcnemar_exact_one_sided_p
verdict
absent_branch_fallback_used
sidecar_single_rule_summary
sidecar_random_half_summary
diagnostics_summary
```

## Implementation Plan

Suggested script:

```text
scripts/v05b_branch_predictor.py
```

Suggested command:

```powershell
python scripts\v05b_branch_predictor.py `
  --input results\isotrophy\k-facet-v05a-branch-map\per_row_table.csv `
  --out results\isotrophy\k-facet-v05b-branch-predictor
```

Expected runtime: seconds. No staged compute is required.

## Interpretation

If v0.5b passes:

> The catalog branch shadow is not only associated with stability in-sample;
> it predicts held-out mass-bin stability better than the always-unstable
> catalog baseline under a pre-registered branch-majority rule.

This would license v0.5c to ask *why* the `(m_3, z_0)` branch boundary carries
stability, with continuous diagnostics promoted one at a time.

If v0.5b fails:

> v0.5a found real in-sample stratification, but the branch shadow does not
> generalize across mass bins. The branch hash is a descriptive catalog
> partition, not a predictive mechanism.

This would be a clean projection-limit result, not a failed run.

## Lock-In Statement

Any change to the held-out partition, active bits, branch-majority rule,
tie rule, primary baseline, McNemar threshold, gating-bin set, or continuous
feature exclusion after the v0.5b runner is executed is a re-registration,
not a refinement.

No v0.5b predictor claim is licensed until this form is signed off and the
held-out runner writes its receipt.
