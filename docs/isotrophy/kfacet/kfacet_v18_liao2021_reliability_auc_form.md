# v0.18 liao2021 Reliability-Conditioned Per-Cell AUC Form

Status: **OPERATOR LOCK 2026-06-03; PREPARED, NOT MEASURED.** At lock time no
v0.18 runner had been written, no v0.18 sample had been drawn, no v0.18 D5 rows
had been integrated, and no v0.18 AUC or reliability statistic had been
computed. After lock, the runner was implemented and `npm run
isotrophy:v18:prepare` completed the source/sample-only stage:
`reliability_auc_prepared_not_measured`.

Prepared receipt:

```text
command:        npm run isotrophy:v18:prepare
out:            results/isotrophy/k-facet-v18-liao2021-reliability-auc/
grid scan:      4x4=7, 5x5=10, 6x6=12, 7x7=16, 8x8=18,
                10x10=24, 12x12=34, 16x16=46 (all matched lock)
holdout:        v14+v15+v16+v17 = 4640 rows, disjoint
supported:      18 cells
sample:         2880 rows (80 S + 80 U per cell)
next:           run the 16 staged D5 shards, then merge + analyze
```

## Frame

v0.16 established the first clean Tier-2 external transfer of the tail-resolved
velocity-fraction score. v0.17 replicated that transfer and showed that the
seven-cell heterogeneity pattern is reproducible, not one-sample texture:

```text
v0.16 AUC_cond = 0.6469866, p_perm = 1.0e-5
v0.17 AUC_cond = 0.6468750, p_perm = 1.0e-5
v0.17 heterogeneity Spearman vs v0.16 = 1.0, exact p = 1/5040
```

The v0.17 anatomy sidecar then suggested a post-hoc explanation: per-cell AUC
looks like an instrument-reliability map. The strongest low cell
(`qA2_qB1`) is the known frame-fragile pocket; the other low cell
(`qA3_qB0`) is close to a near-null rather than a strong competing direction.
Everywhere the four-frame read is stable and the score has usable separation,
the v0.11 direction holds.

v0.18 turns that explanation into a fresh, wider-cell test:

> Across a wider liao2021 mass-cell grid, do frame-reliable cells carry higher
> per-cell AUC, and are low/reversed cells concentrated in instrument-fragile or
> near-null regions rather than stable counter-law regions?

This is a **mechanism/anatomy chapter**, not a new feature search and not a
promotion of the theorem claim.

Expectation calibration: `wider_transfer_replicates_reliability_unresolved` is
a live, informative outcome. The primary statistic tests the **frame-fragility**
half of the reliability story only. v0.17 had one low cell that was clearly
frame-fragile (`qA2_qB1`) and one low cell that was frame-stable but near-null
(`qA3_qB0`). If the wider 8 x 8 low cells are dominated by near-null structure
rather than frame fragility, the label-blind reliability statistic can honestly
miss while the report-only anatomy still explains the pattern.

## Integrity Caveat

```text
post-hoc hypothesis disclosed:
  The reliability-map hypothesis was motivated by the v0.17 anatomy sidecar.
  v0.18 tests it only on fresh rows and a wider source-defined grid.

feature frozen:
  The v0.16/v0.17 score is reused exactly:
  score = median(vf_0, vf_37, vf_90, vf_211).

domain changed only to widen the cell map:
  v0.18 uses a nested sorted-mass 8 x 8 grid instead of the v0.14-v0.17 4 x 4
  grid. This is a registered domain change for the reliability question, not a
  rescue of any prior verdict.

fresh quadruple holdout:
  Exclude every liao2021 orbit used in v0.14, v0.15, v0.16, or v0.17 before
  drawing v0.18 rows.

non-circularity boundary:
  The primary reliability predictor is label-blind frame spread. Label-aware
  score-split anatomy is reported separately and cannot rescue the primary
  frame-reliability statistic.
```

Forbidden:

```text
changing the score, frame set, D5 gates, stability labels, or permutation seed
after seeing v0.18 rows
selecting cells or rows by vf, score, zone, frame spread, D5 output, or AUC
dropping frame-fragile rows from the primary statistic
lowering the pooled transfer floor or the reliability bars after execution
using label-aware anatomy to claim a non-circular reliability pass
promoting to Tier-3, full-catalog prevalence, or theorem-facing status
```

## Frozen Inputs

Carried forward unchanged unless explicitly listed as v0.18-specific:

```text
target:                  liao2021 non-hierarchical catalog
target tier:             Tier 2, Li/Liao lineage
target SHA-256:          9c06eedc41b537b2ed926217cdf149d9f808a21866148ea3d89b9eb3c6069fd4
D5 measurement:          v0.7/v0.13a DOP853 pipeline
rtol/atol:               1e-12 / 1e-12
max_step:                0.02*T
D5 gates:                symplecticity <= 1e-4; reciprocal-pair residual <= 1e-4
frames:                  {0, 37, 90, 211} degrees
score:                   median(vf_0, vf_37, vf_90, vf_211)
success rule:            all four frames integrate and pass D5 gates
stability label:         source-provided linear S/U label
primary row sampling:    uniform without replacement within (mass_cell, source_label)
seed:                    20260523
```

## v0.18 Domain

v0.18 widens the mass-cell grid but keeps the sampling discipline:

```text
mass-cell construction:  `build_mass_cells` generalized to 8 parts:
                         sort by normalized mass a, split_evenly(..., 8);
                         within each a-slice sort by normalized mass b,
                         split_evenly(..., 8)
holdout:                 v0.14 sample_frame.csv
                       + v0.15 sample_frame.csv
                       + v0.16 sample_frame.csv
                       + v0.17 sample_frame.csv
supported cell:          eligible stable >= 80 AND eligible unstable >= 80
rows per cell:           160 = 80 stable + 80 unstable
expected supported:      18 cells
expected rows:           2880
source-support gate:     exactly 18 supported cells; otherwise blocked_for_lock_review
```

Source-only feasibility scan (no vf, score, D5, or AUC computed) after excluding
the v0.14-v0.17 sample frames:

```text
grid parts   supported cells with >=80 S and >=80 U   rows at 80/80
4 x 4        7                                       1120
5 x 5        10                                      1600
6 x 6        12                                      1920
7 x 7        16                                      2560
8 x 8        18                                      2880   <-- proposed v0.18 default
10 x 10      24                                      3840
12 x 12      34                                      5440
16 x 16      46                                      7360
```

The 8 x 8 grid is the proposed lock default: wide enough for a cell-level
reliability test, but still a staged sharded run rather than a full-catalog
weeklong sweep.

Lock verification: the privileged-truth source scan reproduced the table above
exactly, including the 4 x 4 anchor (`7` supported cells) and the 8 x 8 default
(`18` supported cells) with a disjoint quadruple holdout of 4640 rows. The 4 x 4
anchor confirms the generalized grid builder matches the v0.14-v0.17
construction before widening.

## Required Prior Receipts

The runner must assert before drawing rows:

```text
v0.13 target inventory includes liao2021 as Tier 2
v0.13a leakage bounded 0.0 and adapter parity passed
v0.13b verdict = coarse_zone_rule_frame_stable_enough_to_test
v0.14 verdict = sample_transfer_undecidable_coverage
v0.15 verdict = stable_support_transfer_directional_weak
v0.16 verdict = tail_resolved_transfer_passes_clean
v0.17 verdict = heterogeneous_transfer_replicates_clean
all v0.14-v0.17 target SHA values match this form's target SHA
all v0.14-v0.17 sample_frame.csv files exist for the quadruple holdout
```

Any mismatch aborts.

## Primary Measurement

For every successful orbit:

```text
vf_frames:       [vf_0, vf_37, vf_90, vf_211]
score:           median(vf_frames)
frame_spread:    max(vf_frames) - min(vf_frames)
zone:            report-only coarse zone under {0.25, 0.50}; non-gating
```

Within each primary cell:

```text
AUC_cell:        Mann-Whitney AUC, stable ranked above unstable
D_cell:          S_success * U_success
frame_p50/p75/p90/p95/max over successful rows
score quantiles by source label
S_p50_minus_U_p50 report-only
zone-2 fraction report-only
```

Primary cell coverage:

```text
cell enters primary iff S_success >= 50 AND U_success >= 50
coverage pass iff primary_cells >= 16 AND primary_success_rows >= 2400
```

Attrition policy:

```text
clean:     attrition <= 0.05
warning:   attrition <= 0.10 AND Wilson95_high <= 0.20
blocked:   attrition > 0.10 OR Wilson95_high > 0.20
```

Frame-spread is **not** used to drop rows or cells. It is the object under test.
Only a catastrophic global failure blocks:

```text
blocked_by_global_frame_instability iff
  median(frame_spread) > 0.20 OR p90(frame_spread) > 0.75
```

## Baseline Transfer Check

Before the reliability question is interpreted, v0.18 repeats the pooled
within-cell transfer statistic on the wider grid:

```text
AUC_cond = sum(J_cell) / sum(D_cell)
permutation: 100000 within-cell S/U label shuffles, seed 20260523
pass: AUC_cond >= 0.55 AND p_perm <= 0.01
directional weak: 0.50 < AUC_cond < 0.55 AND p_perm <= 0.01
fail: AUC_cond <= 0.50 OR p_perm > 0.01
```

This check does not need to pass for the reliability map to be reported, but a
pooled fail limits the chapter to "why the wider grid failed," not confirmation.
Because v0.18 uses a finer 8 x 8 conditioning grid, its pooled AUC is not
numerically interchangeable with the v0.16/v0.17 4 x 4 AUC of ~0.647, even
though it keeps the same `>=0.55` floor.

## Primary Reliability Statistic

The non-circular reliability predictor is cell-level frame stability:

```text
cell_reliability = -log10(frame_p90 + 1e-9)
outcome          = AUC_cell
statistic        = Spearman rho(cell_reliability, AUC_cell)
null             = 100000 random pairings of reliability values to AUC values
                   over the primary cells, seed 20260523
                   (Monte Carlo; 18! is too large for the exact v0.17-style
                   enumeration)
direction        = higher reliability -> higher AUC
pass bar         = rho >= 0.45 AND one-sided p_reliability <= 0.05
```

This is the only verdict-bearing "reliability drives AUC" statistic. It is
label-blind on the predictor side: frame spread is computed without using S/U
labels or score splits.

## Reversal Guard

The reliability interpretation must not hide a strong, frame-stable competing
signal. Define:

```text
frame_fragile_cell       = frame_p90 >= 0.10
decisive_negative_cell   = AUC_cell <= 0.45 AND reverse-direction cell permutation
                           p_reverse <= 0.05
stable_decisive_negative = decisive_negative_cell AND NOT frame_fragile_cell
```

Reversal guard passes iff `stable_decisive_negative_count == 0`.

`p_reverse` is a per-cell within-cell S/U-label permutation, one-sided for
`AUC_cell < 0.5`, preserving the cell's score multiset and S/U counts. It uses
100000 Monte Carlo shuffles with seed 20260523 offset deterministically by cell
order.

Cells with AUC below 0.5 but not decisive by this rule are reported as near-null
or weak-reversal anatomy, not as a competing law.

## Label-Aware Anatomy (Report-Only)

For interpretation, but not for the primary reliability pass, report:

```text
per-cell S/U score quantiles
S_p50_minus_U_p50
score IQR by label and pooled
zone saturation
period / energy / angular momentum summaries if already available in the row receipts
band_A / band_B membership from v0.13b, if the orbit index lies in the known fragile bands
```

This panel may explain a low cell as "near-null" or "fragile," but it cannot
change the primary reliability statistic or any verdict branch.

Readback must explicitly surface the frame-p90 distribution across the 18
primary cells (min, median, p75, p90, max, plus the sorted per-cell table). If
only a few cells populate the low-reliability tail, the report must say so; the
Spearman power then comes from a small number of detector cells, not a broad
continuous spread.

## Verdict Tree

```text
blocked_by_receipt
  required prior receipts / target SHA / sample-frame files missing or inconsistent

blocked_for_lock_review
  8 x 8 source-support census yields anything other than the locked 18 supported cells

blocked_by_attrition
  attrition_fraction > 0.10 OR attrition_wilson95_high > 0.20

blocked_by_global_frame_instability
  median(frame_spread) > 0.20 OR p90(frame_spread) > 0.75

blocked_by_coverage
  primary_cells < 16 OR primary_success_rows < 2400

wider_transfer_fails
  AUC_cond <= 0.50 OR p_perm > 0.01

wider_transfer_directional_weak
  0.50 < AUC_cond < 0.55 AND p_perm <= 0.01

wider_transfer_replicates_reliability_unresolved
  AUC_cond >= 0.55 AND p_perm <= 0.01
  AND (rho < 0.45 OR p_reliability > 0.05)

frame_reliability_supported_but_reversal_guard_fails
  AUC_cond >= 0.55 AND p_perm <= 0.01
  AND rho >= 0.45 AND p_reliability <= 0.05
  AND stable_decisive_negative_count > 0

reliability_drives_per_cell_auc_supported
  AUC_cond >= 0.55 AND p_perm <= 0.01
  AND rho >= 0.45 AND p_reliability <= 0.05
  AND stable_decisive_negative_count == 0
```

Attrition warning without hard block appends `_with_attrition_warning`.

## Required Outputs

```text
results/isotrophy/k-facet-v18-liao2021-reliability-auc/
  manifest.json
  source_support_census.csv
  sample_frame.csv
  sample_shard_XX_of_NN.csv
  per_row_sample.csv
  per_cell_auc.csv
  frame_reliability_map.csv
  reliability_auc_test.json
  reversal_guard.json
  anatomy_sidecar.csv
  permutation_summary.json
  operator_commands.md
```

`manifest.json` must include verdict, startedAt/completedAt, gitCommit,
target_sha256, excluded sample-frame counts, sample_seed, grid parts,
supported_cells, sample_rows_requested/success, attrition + Wilson interval,
primary_cells, primary_success_rows, AUC_cond, J_cond, D_cond, p_perm,
reliability_rho, reliability_p, stable_decisive_negative_count, frame-spread
global median/p90, target_tier=2, and the claim boundary.

## Implementation Commands (to add after lock)

The run is above the inline agent budget. Implementation should stage the exact
operator commands and shard the D5 measurement:

```powershell
npm run isotrophy:v18:prepare
npm run isotrophy:v18:shard -- --shard 0 --shards 16
npm run isotrophy:v18:shard -- --shard 1 --shards 16
# ...
npm run isotrophy:v18:shard -- --shard 15 --shards 16
npm run isotrophy:v18:merge
npm run isotrophy:v18:analyze
```

Expected cost is roughly 2.5-3x the 1120-row v0.16/v0.17 measurement, because
the proposed sample is 2880 rows and uses the same four-frame D5 score. The
operator command file must record measured per-row rate after the first shard
finishes and revise the wall-clock estimate.

## Claim Boundary

Allowed if `reliability_drives_per_cell_auc_supported`:

```text
Across a fresh wider liao2021 mass-cell map, per-cell transfer strength tracks
the frame reliability of the tail-resolved velocity-fraction read. Low or
reversed cells are not evidence of a stable competing law unless they survive
the reversal guard; the supported claim is an instrument-reliability explanation
of the reproducible v0.16/v0.17 heterogeneity.
```

Bounded exactly as before: Tier-2 / Li-Liao lineage / liao2021 stable-support
sampling / within-cell rank signal / tail-resolved continuous score / not
coarse-zone / not full-catalog prevalence / not Tier-3 independent / not
theorem-facing.

If reliability fails, the allowed claim is only that the v0.17 reliability-map
interpretation did not survive a wider fresh-cell test. That does **not** erase
the v0.16/v0.17 external transfer receipts.

## Lock-In Statement

Committed as a draft before any v0.18 runner or v0.18 sample exists. Any change
to the grid, holdout, sample size, score, D5 path, reliability statistic, bars,
reversal guard, verdict tree, or claim boundary after execution is a
re-registration.
