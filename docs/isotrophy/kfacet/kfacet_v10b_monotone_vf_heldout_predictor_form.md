# v0.10b Monotone vf Held-Out Predictor Form Lock

Status: **OPERATOR LOCK 2026-05-30.** Reviewed for self-consistency — the folds
reconcile (234 primary rows = 82 S / 152 U across 10 `m_3` bins with N≥5, plus 16
report-only rows across 5 bins with N<5 = 250 total; zone counts match the frozen
v0.9a/v0.10a receipt) — and design soundness (the AUC/risk-ranking primary gate
sidesteps the sub-50%-stable-everywhere base-rate trap that would break a hard
always-`U` classifier; the within-`m_3` stratified permutation null with full CV
refit isolates the zone signal from the `m_3` base-rate confound). No v0.10b runner
has been written and no v0.10b command has been run at lock time. This document
locks the held-out partition, predictor family, scoring metric, permutation
falsifier, pass/fail tree, and claim boundary before any v0.10b prediction is
computed.

**Naming.** This is **v0.10b**, the held-out predictor rung licensed by the
v0.10a ordered-trend pass. Earlier v0.9/v0.10a documents reserved the label
`v0.9b` for "the held-out predictor." This file consumes that reserved slot under
the post-pause v0.10 numbering: `v0.10b` is the actual artifact; `v0.9b` remains
an alias in older prose, not a separate future chapter.

## Verdict (Landed 2026-05-30)

Receipt: `results/isotrophy/k-facet-v10b-monotone-vf-heldout/manifest.json`.

```text
verdict:                monotone_vf_predictor_fails_heldout
frozen-input check:     PASSED (250 rows, 87 S/163 U; zones 19/165/66; v10a registered)
primary partition:      234 gating rows (82 S/152 U), 10 m_3 folds (N>=5)

PRIMARY GATE (binding):
  pooled held-out AUC:  0.4125   (delta vs constant 0.5  =  -0.0875)   -> AUC <= 0.5
  within-m3 permutation p (n=10000, seed 20260523): 9.999e-5 (perm-null mean 0.265)
  gate = (AUC > 0.5) AND (p <= 0.01):  AUC>0.5 FAILS  ->  FAIL

per-fold held-out AUC (diagnostic, non-gating):
  m3   N   S  U   foldAUC
  0.4  55 35 20   0.671
  0.5  30 15 15   0.591
  0.6  22  6 16   0.438
  0.7  18  4 14   0.464
  0.8  18  6 12   0.840
  0.9  21  6 15   0.900
  1.0  35  7 28   0.811
  1.1  23  1 22   0.818
  1.2   7  2  5   0.550
  1.7   5  0  5   n/a (no S)
  mean (computable): ~0.65

hard-label sidecar (NON-GATING): accuracy delta vs always-U = -0.167,
                                 McNemar one-sided p = 0.99999 (base-rate-trapped, as anticipated)
```

**Reading — projection-limit, the audit-vs-predictor pattern again.** The pooled
held-out AUC is **below 0.5**, so the registered gate FAILS. But the mandated
per-fold diagnostic shows the failure is a **cross-bin calibration artifact, not a
within-bin signal failure**: within most held-out mass bins the monotone zone
predictor ranks stable above unstable (per-fold AUC mostly 0.55–0.90, mean ~0.65),
and the permutation p is tiny (the observed 0.41 beats the within-m_3-shuffled null
of 0.265, so genuine zone structure exists). What fails is the **pooled** cross-bin
ranking: the zone-only score is not comparable across `m_3` bins because the mass-bin
base rate — which the predictor cannot use — dominates global ranking.

Per the locked verdict tree this is a clean **FAIL**; the goalposts stay where they
were locked (pooled AUC primary), and the encouraging per-fold AUCs are a
diagnostic, **not** a post-hoc rescue. The v0.10a in-sample / within-bin
velocity-fraction → stability trend is real but does **not** yield a
globally-calibrated held-out risk score across mass bins. This closes the predictor
rung as a **projection-limit** result (cf. v0.5); a within-`m_3` or `m_3`-conditional
predictor would be a **fresh** pre-registered chapter, not a v0.10b refinement. No
K_facet change; isotrophy stays paused with one in-sample positive and a registered
held-out null.

## Frame

v0.10a registered an **in-sample**, post-hoc-discovered but properly aligned
ordered trend:

```text
positional-dominant  -> mixed       -> velocity-heavy
S_fraction 0.1053       0.3394         0.4394
J-T exact p = 0.007304
```

v0.10b asks the next, stricter question:

> Does the monotone velocity-fraction signal survive when each `m_3` bin is held
> out and must be scored by a predictor trained only on the other mass bins?

This is a held-out **risk/ranking** predictor, not a hard majority classifier.
That distinction is load-bearing: even the frozen velocity-heavy zone is below a
50% stable majority globally (`29 S / 37 U`). A hard accuracy/McNemar classifier
against always-`U` would mostly test whether any zone crosses a majority-S
threshold, which is not what v0.10a established. The primary held-out test
therefore scores whether a monotone risk model ranks held-out stable rows above
held-out unstable rows.

## Integrity Caveat

The direction and zone cutpoints are inherited from v0.9a/v0.10a after the
monotone pattern was observed. v0.10b is held-out across `m_3` bins, but it is not
an independent external-catalog validation. A pass would mean:

> the v0.10a monotone vf signal is held-out predictive across mass bins inside
> the frozen 250-row analyzable supp-B domain.

A pass would **not** make isotrophy theorem-facing, would **not** change the
v0.3h K_facet structural-null, and would **not** claim catalog-wide prediction
over the 23 v0.7a integration-blocked rows.

## Frozen Inputs

Primary input:

```text
results/isotrophy/k-facet-v09a-signed-vf-three-zone/per_row_table.csv
```

Receipt cross-check input:

```text
results/isotrophy/k-facet-v10a-jt-trend/manifest.json
```

The runner must assert:

```text
row count:                    250
stability totals:             87 S / 163 U
zone counts:
  positional-dominant:        N=19,  S=2,  U=17
  mixed:                      N=165, S=56, U=109
  velocity-heavy:             N=66,  S=29, U=37
v0.10a verdict:               jt_trend_monotone_registered
v0.10a exact p:               < 0.01
zone recompute:               recompute from velocity_fraction under {0.25,0.50}
                              and require zero mismatches with stored zone
```

Any mismatch aborts. No new orbit integration, variational integration, gauge
minimization, or re-derivation of `velocity_fraction` is authorized.

## Gating Folds

Primary partition: leave-one-`m_3`-bin-out over frozen analyzable rows with
`N >= 5`. The fold threshold inherits the v0.5b discipline and avoids making tiny
folds verdict-bearing.

```text
m_3   N    S    U
0.4   55   35   20
0.5   30   15   15
0.6   22    6   16
0.7   18    4   14
0.8   18    6   12
0.9   21    6   15
1.0   35    7   28
1.1   23    1   22
1.2    7    2    5
1.7    5    0    5
```

Primary gating rows: `234` (`82 S / 152 U`).

Report-only folds (`N < 5`) are scored by a model trained on all 234 gating rows
but do not enter the primary metric or p-value:

```text
m_3   N    S    U
1.3    4    2    2
1.4    4    0    4
1.5    4    1    3
1.6    2    2    0
1.9    2    0    2
```

## Predictor

For each gating fold `f`:

```text
test_f  = rows with m_3 == f
train_f = rows in the primary gating set with m_3 != f
```

Feature:

```text
zone_index:
  positional-dominant = 0
  mixed               = 1
  velocity-heavy      = 2
```

Training:

1. Compute raw training stable fraction per zone:

```text
raw_p_z = S_train(zone=z) / N_train(zone=z)
```

2. Fit a nondecreasing three-level score by weighted isotonic regression
   (Pool Adjacent Violators Algorithm) on `zone_index`, weighted by
   `N_train(zone=z)`.

3. Assign each held-out row the fitted score for its zone:

```text
score(row) = pava_score_train_f[zone_index(row)]
```

No smoothing, threshold fitting, logistic regression, raw `vf` thresholding,
`m_3` feature, branch-label feature, or per-test-fold tuning is allowed. If a
training fold somehow has an empty zone, the empty zone is merged with the
nearest populated neighbor before PAVA; this condition must be reported and
would be a review flag, but it is not expected under the frozen folds.

## Primary Metric

Primary held-out score: pooled cross-validated AUC over the 234 primary gating
rows.

```text
AUC = P(score(S_row) > score(U_row))
      + 0.5 * P(score(S_row) = score(U_row))
```

This is the rank analogue of the v0.10a ordered trend. The constant-score
baseline has `AUC = 0.5`.

The runner must also report:

- per-fold AUC where both S and U are present;
- pooled AUC excluding no-S/no-U folds as a diagnostic only;
- mean held-out score by stability;
- Brier score vs a fold-trained constant baseline (`p_train_global`) as a
  secondary calibration diagnostic.

## Primary Falsifier

Binding p-value: stratified Monte Carlo label permutation with full
cross-validation re-fit.

Procedure:

1. Preserve each row's `m_3`, `zone`, and fold assignment.
2. Shuffle stability labels **within each `m_3` bin** among the 234 primary gating
   rows. This preserves the mass-bin S/U skew, including the load-bearing
   `m_3 = 0.4` fold, while destroying any within-bin relation between zone and
   stability.
3. For each permutation, rerun the entire leave-one-`m_3`-bin-out training and
   scoring pipeline, including PAVA, and compute pooled held-out AUC.
4. One-sided upper-tail p:

```text
p = (1 + #{AUC_perm >= AUC_obs}) / (1 + n_permutations)
```

Locked:

```text
seed = 20260523
n_permutations = 10000
alpha = 0.01
```

This is Monte Carlo, not exact. Exact enumeration is infeasible because the model
is refit under each folded, stratified label allocation. The seed and permutation
count are locked before runner implementation.

## Verdict Tree

```text
if frozen-input cross-check fails:
    ABORT  (not the frozen v0.9a/v0.10a receipt)
elif any primary gating fold has N < 5:
    ABORT  (partition mismatch)
elif AUC_obs <= 0.5:
    verdict = monotone_vf_predictor_fails_heldout
elif permutation_p <= 0.01:
    verdict = monotone_vf_predictor_passes_heldout
else:
    verdict = monotone_vf_predictor_fails_heldout
```

The pass requires both positive AUC delta and the one-sided permutation floor.
There is no "partial" branch for p between 0.01 and 0.05; such a result is a
descriptive held-out hint and a formal fail.

## Required Diagnostics

The receipt must expose the likely failure mode rather than hiding it inside the
pooled statistic:

- `m_3 = 0.4` fold row count, S/U, observed AUC, and score distribution.
- Fold table for all 10 primary folds: `m_3`, N, S, U, trained zone scores,
  fold AUC, fold Brier, and whether the fold has both classes.
- Training zone rates per fold before and after PAVA.
- Report-only fold scores for `m_3 in {1.3,1.4,1.5,1.6,1.9}`.
- Alignment diagnostic against v0.5a branch labels, report-only. Branch labels
  are never features.

## Hard-Label Sidecar (Non-Gating)

For continuity with v0.5b, the runner also reports a hard-classification sidecar:

```text
candidate thresholds:
  T1: predict S iff zone_index >= 1  (mixed or velocity-heavy)
  T2: predict S iff zone_index >= 2  (velocity-heavy only)
```

For each training fold, select `T1` or `T2` by higher training balanced accuracy;
ties choose `T2`. Score held-out predictions against always-`U` using accuracy
delta and exact one-sided McNemar/binomial on discordant rows.

This sidecar is **not gating**. If it passes, it licenses a future classifier
claim review. If it fails, it does not override a primary AUC pass, because the
registered v0.10b claim is held-out monotone risk/ranking, not majority-S
classification.

## Disallowed Moves

v0.10b may NOT use:

- raw `velocity_fraction` as a continuous predictor;
- v0.7a′ quartile labels or quartile cutpoints;
- branch labels, `m_3`, `z0`, `E`, `|L|`, period, or any catalog-coordinate
  feature as predictor input;
- eigenvalue magnitudes, spectral radius, unit-circle status, unstable-pair
  count, or any threshold that defines the published S/U label;
- test-fold labels for threshold choice, calibration, smoothing, or model
  selection;
- post-run threshold changes to rescue a fail.

## Claim Boundary

A PASS:

- says the v0.10a monotone velocity-fraction trend is **held-out predictive as a
  rank/risk score** across left-out `m_3` bins inside the 250-row analyzable
  supp-B domain;
- does NOT say there is a hard classifier unless the non-gating sidecar is
  separately promoted after review;
- does NOT cover the 23 v0.7a integration-blocked rows;
- does NOT make K_facet theorem-facing or revise the v0.3h structural-null;
- does NOT claim a general three-body stability predictor beyond Li-Liao
  supplementary-B under this frozen ansatz.

A FAIL:

- says the monotone trend is real in-sample (v0.10a) but does not survive the
  registered leave-one-mass-bin-out risk-prediction gate;
- should close the predictor rung as another projection-limit result unless a
  fresh, explicitly different domain or predictor family is registered.

## Receipt Schema

```text
results/isotrophy/k-facet-v10b-monotone-vf-heldout/
  manifest.json
    - schema = "sundog.isotrophy.v0.10b-monotone-vf-heldout.v1"
    - mode = "v0.10b-monotone-vf-heldout"
    - form_lock = "docs/isotrophy/kfacet/kfacet_v10b_monotone_vf_heldout_predictor_form.md"
    - input_v09a_per_row_table
    - input_v10a_manifest
    - frozen_input_crosscheck
    - primary_gating_folds
    - report_only_folds
    - predictor = { feature: zone_index, model: weighted_pava_by_training_fold }
    - primary_metric = { auc_observed, auc_delta_vs_constant, permutation_p,
                         seed, n_permutations, alpha }
    - verdict
  per_fold_table.csv
  per_row_predictions.csv
  training_zone_scores.csv
  permutation_summary.csv
  hard_label_sidecar.csv
```

## Lock-In Statement

This form is committed before any v0.10b prediction, permutation, fold score, hard
sidecar, or Brier diagnostic is computed. Any change to the fold threshold,
predictor family, PAVA rule, primary metric, permutation null, pass/fail tree, or
claim boundary after runner execution is a re-registration, not a refinement.

Implementation may proceed against the frozen v0.9a per-row table and v0.10a
manifest only. Expected runtime: seconds to low minutes.

