# v0.16 liao2021 Tail-Resolved Transfer Form Draft

## Result (2026-06-02)

**Verdict: `tail_resolved_transfer_passes_clean`.** The first clean external (Tier-2,
fresh doubly-held-out, outcome-balanced) confirmation in the arc -- the tail-resolved
continuous score clears the 0.55 floor where the coarse zone could not.

```text
coverage           7/7 supported cells primary, 1120 rows
attrition          0.0000  Wilson95 [0.0000, 0.0034]   (conservative all-4-frame rule, zero loss)
frame-spread gate  median 0.00036, p90 0.0184  -> CLEAN (bounds 0.10 / 0.50)
AUC_cond           0.6470  (J_cond 28985 / D_cond 44800)
p_perm             1.0e-5  (100k within-cell score permutation; 0 of 100k >= observed; null mean 0.500, max 0.582)
```

**The clean triangulation (same 1120 rows, same direction).** The report-only coarse
zone AUC on these exact rows is `0.510` (saturated, ~chance) -- reproducing v0.15's
coarse-zone result -- while the continuous tail-resolved score is `0.647`. The only
difference is binning vs continuous: the coarse {0.25,0.50} zone WAS discarding the
signal, demonstrated directly on one row set. v0.16's hypothesis is confirmed.

**Honest texture -- the effect is real but mass-cell-heterogeneous.** Per-cell AUC
ranges 0.43-0.97:

```text
qA0_qB0 0.972  qA3_qB3 0.745  qA3_qB1 0.696  qA3_qB2 0.639  qA1_qB0 0.575   (v0.11 direction)
qA2_qB1 0.432  qA3_qB0 0.470                                                (direction reversed)
```

The v0.11 direction (stable -> higher vf) holds strongly in 5 of 7 cells and reverses in
2. The pooled effect is significant against the within-cell permutation null (which
respects cell structure), so it is not a pooling artifact -- but it is NOT a uniform
per-cell law. Leave-one-out robustness: dropping the strongest cell (qA0_qB0) still gives
pooled AUC 0.593 > 0.55, so the pass is not a single-cell artifact. The qA0_qB0
separation (S median vf 0.9965 vs U 0.9904) is far above DOP853 numerical noise, not an
integration artifact.

**Ensemble vs single-frame.** The secondary single-frame `vf_0` AUC is 0.636, within
~0.01 of the 4-frame ensemble 0.647 -- as the pre-lock frame-spread preview predicted
(ensemble approx single-frame for ~90% of orbits). The frame ensemble bought a little
robustness on the fragile tail; the headline lever was continuous-vs-coarse
(0.51 -> 0.65), not ensemble-vs-single-frame.

**Allowed claim (locked):** a tail-resolved 4-frame ensemble velocity-fraction score
transfers as a Tier-2 within-cell stability-ranking signal on fresh held-out liao2021
stable-support rows; the underlying velocity-fraction projection carries external
stability information that its coarse-zone v0.11 form does not (v0.15). Does NOT claim
the coarse zone transfers, full-catalog transfer, prevalence, Tier-3 / independent
confirmation, or theorem-facing status.

**Independent verification.** A standalone brute-force pair-count recompute (#(S>U) +
0.5 ties, independent of the runner's midrank path) reproduced AUC_cond
0.6469866071428572, J 28985, D 44800, the secondary vf_0 AUC 0.635546875, the zone AUC
0.50984375, and all seven per-cell AUCs BIT-FOR-BIT, with 0 zone_index_vf0 re-derivation
mismatches. Receipt + verifiers:
`results/isotrophy/k-facet-v16-liao2021-tail-resolved-transfer/` (`manifest.json`,
`per_cell_rank.csv`, `permutation_summary.json`, `coarse_zone_relationship.csv`,
`_independent_check.{py,json}`, `_sample_verify.{py,json}`). git_commit 4bc2b7a7.

**Next-chapter boundary.** Per the locked tree, a PASS may justify a larger
supported-region or multi-feature liao2021 chapter (which must re-lock scope / sharding /
attrition / frame / floor / claim). The mass-cell heterogeneity is the natural next
question: which mass regions carry the transfer and why two reverse. PASS does not
promote beyond Tier-2 or make isotrophy theorem-facing.

---

Status: **OPERATOR LOCK 2026-06-01; VERDICT LANDED `tail_resolved_transfer_passes_clean` 2026-06-02.**
No v0.16 runner had been written, no v0.16
sample drawn, no v0.16 D5 rows integrated, and no v0.16 transfer statistic computed
at lock time. Runner implementation began after lock. This form locks the
feature-resolution follow-up to v0.15.

Decision adjustments before lock:

```text
frame stability:        generous frame-spread gate on the four ensemble frames
ensemble-frame failure: conservative -- any missing ensemble frame counts as attrition
zone/score relation:    report-only diagnostic, never a verdict gate
```

Pre-lock frame-spread preview (marginal only, no stability split, v0.15 rows) found
the proposed gate clean by a wide margin: median spread 0.0003 and p90 0.0205 against
clean bounds 0.10 / 0.50. This informed no score threshold and no stability read.

## Frame

v0.15 defeated the v0.14 coverage wall (7/7 supported cells primary, 1120 rows, zero
attrition) and produced a decidable result: `stable_support_transfer_directional_weak`
(AUC_cond 0.5125, p_perm 0.00033). The direction transferred and was permutation-
significant, but the effect sat below the 0.55 floor. The mechanism was diagnosed: the
coarse `{0.25, 0.50}` zone feature is **saturated** in liao2021's stable-support region
-- 98.4% of orbits fall in zone 2, so the 3-zone binning discards nearly all of the
velocity-fraction information.

The marginal velocity-fraction distribution in that region (measured on v0.15 success
rows, **no stability split**, firewall-clean) confirms the binning -- not the underlying
projection -- is the bottleneck:

```text
vf quantiles (v0.15 stable-support region):  p5=0.75  p25=0.94  p50=0.97  p75=0.99
                                              16% below 0.9,  5% at/below 0.75,  min 0.075
                                              zone-2 internal IQR = 0.054
```

98.4% of orbits are collapsed to one zone, yet the continuous vf beneath them spans
`[0.5, 1.0]` with a real lower tail. A tail-resolved score has resolution to exploit
that the coarse zone destroys.

v0.16 asks the resolution question directly:

> On a fresh, doubly-held-out, outcome-balanced sample from liao2021's stable-support
> cells, does a **tail-resolved, frame-ensemble velocity-fraction score** rank stable
> rows above unstable rows at the pre-registered external-transfer effect floor?

A PASS would show the underlying velocity-fraction projection (continuous) carries
external stability information even though its coarse-zone v0.11 form does not (v0.15).
A FAIL would show the projection itself -- not just its binning -- does not transfer
with magnitude to this region.

## Integrity Caveat

v0.16 breaks **two** axes relative to v0.11, deliberately and disclosed:

```text
feature:  NEW. v0.11/v0.14/v0.15 used the coarse {0.25,0.50} zone of a single-frame vf.
          v0.16 uses a continuous 4-frame ensemble median of vf. This is not the v0.11
          feature; it is a new tail-resolved, frame-aggregate projection.
domain:   FRESH. Same 7 stable-support cells, but excluding every v0.14 AND v0.15
          sampled orbit -- no row that informed the saturation finding is reused.
```

Hypothesis provenance (disclosed): the tail-resolution hypothesis was motivated by the
v0.15 saturation result and the **marginal** vf distribution above. It was NOT derived
from any vf-vs-stability inspection on v0.15, held-out, or fresh rows. The feature is
fully specified here with zero free parameters before any v0.16 measurement.

Frozen from the prior chapters (unchanged):

```text
target file SHA-256:     9c06eedc41b537b2ed926217cdf149d9f808a21866148ea3d89b9eb3c6069fd4
target tier:             Tier 2 only
mass-cell construction:  v0.14 nested sorted-mass 4 x 4 cells
D5 measurement:          v0.7/v0.13a DOP853 pipeline
frame rotations:         v0.13a/b SO(2) {37, 90, 211} degrees (the ensemble frames)
within-cell statistic:   pooled conditional rank AUC
effect floor:            AUC_cond >= 0.55
permutation p gate:      p_perm <= 0.01
permutation seed:        20260523
```

**The effect floor is HELD at 0.55. It is not lowered to chase v0.15's 0.5125.**

Forbidden:

```text
lowering the effect floor or the p gate
adding free parameters to the score (the 4-frame median is fully fixed)
reporting whichever of several candidate features clears the floor (the primary score
  is the single 4-frame ensemble median; nothing else may set the verdict)
selecting sampled rows by any vf, score, zone, or D5 output (draw on mass_cell +
  source stability label only)
including any v0.14 OR v0.15 sampled orbit in the v0.16 primary sample
dropping frame-fragile rows from the primary statistic
using coarse-zone/score agreement, disagreement, or report-only zone AUC to set or
  change the verdict
using the balanced sample to estimate liao2021 stability prevalence
claiming this rescues or transfers the v0.11 COARSE-ZONE feature (v0.15 settled that)
re-reading or revising v0.15's directional_weak verdict
promoting to Tier-3 / independent confirmation
revising the v0.3h K_facet structural null
```

## Required Prior Receipts

The runner must assert all of the following before drawing a sample:

```text
v0.13 target inventory includes liao2021 as Tier 2
v0.13a leakage bounded at 0.0
v0.13b verdict = coarse_zone_rule_frame_stable_enough_to_test
v0.13 rate-probe attrition <= 0.10 and Wilson95_high <= 0.20
v0.14 verdict = sample_transfer_undecidable_coverage
v0.15 verdict = stable_support_transfer_directional_weak
v0.14 + v0.15 target SHA match this form's target SHA
v0.14 sample_frame.csv AND v0.15 sample_frame.csv exist (for the double holdout)
```

Any mismatch aborts.

## Source-Support Census (fresh held-out)

Before any v0.16 D5 measurement, recompute source S/U counts in the frozen v0.14 mass
cells **after excluding both v0.14 and v0.15 sampled orbits**. Signal-blind: source
masses + source stability labels only; no vf, score, zone, monodromy, or transfer
statistic.

The census reads (independently reproduced from the source file with v0.14's
`build_mass_cells`; eligible = source minus the v0.14 ∪ v0.15 holdout):

| mass cell | eligible S | eligible U | disposition |
|---|---:|---:|---|
| `mass_qA0_qB0` | 402 | 7,824 | supported |
| `mass_qA1_qB0` | 568 | 7,658 | supported |
| `mass_qA2_qB1` | 2,664 | 5,561 | supported |
| `mass_qA3_qB0` | 2,514 | 5,712 | supported |
| `mass_qA3_qB1` | 3,078 | 5,147 | supported |
| `mass_qA3_qB2` | 2,341 | 5,884 | supported |
| `mass_qA3_qB3` | 1,038 | 7,187 | supported |

Locked stable-support rule (fresh):

```text
supported cell iff eligible S >= 80 AND eligible U >= 80 after excluding v0.14 ∪ v0.15
```

Expected supported cells: 7 (smallest eligible stable pool 402, > 5x the 80 draw). The
nine zero-stable / marginal cells from v0.14/v0.15 remain report-only and out of scope.

## Sample Design

Primary sample:

```text
supported cells:      expected 7
rows per cell:        160
stable rows/cell:      80
unstable rows/cell:    80
expected sample rows: 1120
seed:                 20260523
sampling:             uniform without replacement within (mass_cell, source_label)
holdout rule:         exclude every orbit_index in BOTH v0.14 and v0.15 sample_frame.csv
```

If a supported cell has fewer than 80 eligible stable or 80 eligible unstable rows after
the double exclusion, the run aborts for lock review. Deficits are not redistributed.

The draw is outcome-balanced by source label and blind to vf / score / zone. As in
v0.15, the balanced design keeps the rank-AUC valid (AUC is prevalence-invariant under
case-control) and forbids any prevalence estimate.

## New Feature: 4-Frame Ensemble Median Velocity-Fraction

This is the chapter's one new object. It is fully fixed here.

For each sampled orbit, on its CoM-centered initial state `x0`:

```text
frame set F = { 0deg (identity), 37deg, 90deg, 211deg }   in-plane SO(2) rotations
              (the v0.13a/b rotation set; translation is dropped -- it is absorbed by
               CoM-centering, so vf(translation) == vf(identity) and would only
               double-count the identity in the median)

for each f in F:
    rotate x0 by f, integrate through the frozen D5 path, compute monodromy,
    select_gamma_1, velocity_fraction  ->  vf_f         (continuous, in [0,1])

score = median( vf_0, vf_37, vf_90, vf_211 )            (median of 4 = mean of the two
                                                          middle order statistics)
```

Rationale (answers v0.13a directly): a single-frame vf is frame-relative because
`select_gamma_1`'s largest-real-part argmax flips under rotation when Floquet real-parts
cluster. The **median over the frame ensemble** is a robust central estimate that
averages out single-frame argmax flips, while retaining continuous resolution the coarse
zone discards. It is frame-aggregate by construction, not a single-frame reading.

Cost note: v0.15 already integrates exactly these 4 frames per orbit (base + the 3
rotations of its frame audit). v0.16's primary score is therefore computed from the same
4 integrations -- same runtime; the runner persists the continuous per-frame vf and
medians it rather than deriving zones.

Secondary scores (reported, NON-gating):

```text
vf_0                         single-frame continuous velocity_fraction
zone_index(vf_0; 0.25,0.50)  original coarse v0.11-style zone, identity frame only
```

Report both alongside the primary ensemble score. `vf_0` shows whether the ensemble
bought frame stability over a single-frame read. The coarse-zone sidecar shows the
relationship between the new tail-resolved score and the old saturated instrument. Neither
secondary score can set or change the verdict.

## D5 Measurement

For each frame integration, the frozen v0.7/v0.13a D5 path:

```text
integrator:           DOP853
rtol:                 1e-12
atol:                 1e-12
max_step_fraction:    0.02
symplecticity_gate:   1e-4
reciprocal_pair_gate: 1e-4
```

An orbit is a measurement success only if **all four ensemble frames**
`{0,37,90,211}` integrate and pass both D5 sanity gates. Rows that are
integration-blocked or sanity-failed in the identity frame or any non-identity ensemble
frame are excluded from the primary statistic and counted in attrition. A partial
ensemble is a different feature and is forbidden.

Attrition policy:

```text
attrition_fraction <= 0.05:                                   clean
0.05 < attrition_fraction <= 0.10 AND Wilson95_high <= 0.20:   attrition warning
attrition_fraction > 0.10 OR Wilson95_high > 0.20:            blocked_by_attrition
```

## Primary Domain

After D5 attrition, a supported cell enters the primary statistic only if:

```text
N_success >= 120
S_success >= 50
U_success >= 50
```

Coverage gate:

```text
primary_supported_cells >= 6
primary_success_rows >= 900
```

If the coverage gate fails, verdict is `tail_resolved_transfer_undecidable_coverage`.
Cells that fail primary gates are report-only.

## Primary Statistic

Within each primary supported cell, compare every stable row to every unstable row on
the continuous ensemble score (Mann-Whitney form):

```text
J_cell = #[(S,U): score(S) > score(U)]  +  0.5 * #[(S,U): score(S) = score(U)]
D_cell = S_cell * U_cell
AUC_cell = J_cell / D_cell

J_cond = sum J_cell over primary cells
D_cond = sum D_cell over primary cells
AUC_cond = J_cond / D_cond
```

Direction is pre-registered (the v0.11 direction): **stable rows should have higher
ensemble score than unstable rows.** With a continuous score, exact ties are rare and
take the 0.5 weight; there is no zone-degeneracy regime and no v0.14-style discreteness
ceiling -- the null is smooth.

## Permutation Test

```text
within-cell S/U permutation
preserve each primary cell's score multiset
preserve each primary cell's S and U counts
permutations = 100000
seed = 20260523
p_perm = (1 + # permuted J_cond >= observed J_cond) / (1 + permutations)
one-sided in the pre-registered direction
```

## Frame-Stability Audit

The primary score is frame-aggregate by construction, but that robustness must be
priced rather than assumed. The lock uses a **generous internal frame-spread gate** on
the same four frames that define the score, avoiding an extra held-out-rotation sweep
while still blocking a pass if the ensemble is visibly unstable.

For each successful orbit:

```text
frame_spread = max(vf_0, vf_37, vf_90, vf_211) - min(vf_0, vf_37, vf_90, vf_211)
```

Report the distribution overall, by supported mass cell, and in fragile bands A/B:

```text
median, p75, p90, p95, max
```

Frame-spread policy:

```text
median(frame_spread) <= 0.10 AND p90(frame_spread) <= 0.50:
  clean

median(frame_spread) <= 0.20 AND p90(frame_spread) <= 0.75:
  frame warning; transfer may pass only with warning language

median(frame_spread) > 0.20 OR p90(frame_spread) > 0.75:
  verdict = tail_resolved_transfer_blocked_by_frame_instability
```

These are deliberately generous because v0.16's score is a robust central estimate, not
a claim that every individual frame is stable. The gate asks only whether the ensemble
is so frame-variable that a pass would be uninterpretable.

## Coarse-Zone Relationship Diagnostic

The original v0.11 coarse zone is saturated in liao2021 and cannot gate v0.16. It is
reported only, using the identity-frame `vf_0`:

```text
zone_index_vf0 = 0 if vf_0 < 0.25
zone_index_vf0 = 1 if 0.25 <= vf_0 < 0.50
zone_index_vf0 = 2 if vf_0 >= 0.50
```

Report:

```text
zone_auc_cond_report_only        within-cell AUC using zone_index_vf0
score_quantiles_by_zone          score p25/p50/p75 within each coarse zone
zone2_internal_score_quantiles   score p5/p25/p50/p75/p95 inside saturated zone 2
```

This diagnostic can support interpretation (for example, showing that the continuous
score resolves variation inside zone 2), but it cannot set or change the verdict.

## Verdict Tree

Run prior-receipt, target, sample, attrition, coverage, and frame gates before
interpreting the rank statistic.

```text
tail_resolved_transfer_blocked_by_receipt
  required prior receipts missing or inconsistent
  OR target SHA mismatch
  OR v0.14/v0.15 sample exclusion unavailable
  OR supported-cell census deviates from the locked rule and cannot draw 80/80

tail_resolved_transfer_blocked_by_attrition
  attrition_fraction > 0.10 OR attrition_wilson95_high > 0.20

tail_resolved_transfer_blocked_by_frame_instability
  median(frame_spread) > 0.20 OR p90(frame_spread) > 0.75

tail_resolved_transfer_undecidable_coverage
  primary_supported_cells < 6 OR primary_success_rows < 900

tail_resolved_transfer_passes_clean
  attrition_fraction <= 0.05
  AND median(frame_spread) <= 0.10
  AND p90(frame_spread) <= 0.50
  AND AUC_cond >= 0.55
  AND p_perm <= 0.01

tail_resolved_transfer_passes_with_warning
  AUC_cond >= 0.55
  AND p_perm <= 0.01
  AND no hard block fired
  AND (
        0.05 < attrition_fraction <= 0.10
        OR 0.10 < median(frame_spread) <= 0.20
        OR 0.50 < p90(frame_spread) <= 0.75
      )

tail_resolved_transfer_directional_weak
  AUC_cond > 0.50
  AND AUC_cond < 0.55
  AND p_perm <= 0.01

tail_resolved_transfer_fails
  AUC_cond <= 0.50 OR p_perm > 0.01
```

## Required Outputs

Expected additive runner:

```text
scripts/v16_liao2021_tail_resolved_transfer.py
```

Expected command shape:

```powershell
python scripts/v16_liao2021_tail_resolved_transfer.py prepare `
  --target docs/isotrophy/external_targets/liao2021_nonhierarchical.txt `
  --v14 results/isotrophy/k-facet-v14-liao2021-sampled-transfer `
  --v15 results/isotrophy/k-facet-v15-liao2021-stable-support-transfer `
  --out results/isotrophy/k-facet-v16-liao2021-tail-resolved-transfer `
  --stable-per-cell 80 `
  --unstable-per-cell 80 `
  --seed 20260523

python scripts/v16_liao2021_tail_resolved_transfer.py shard `
  --out results/isotrophy/k-facet-v16-liao2021-tail-resolved-transfer `
  --shard-index 0 `
  --shard-count 14

python scripts/v16_liao2021_tail_resolved_transfer.py merge `
  --out results/isotrophy/k-facet-v16-liao2021-tail-resolved-transfer `
  --shard-count 14

python scripts/v16_liao2021_tail_resolved_transfer.py analyze `
  --out results/isotrophy/k-facet-v16-liao2021-tail-resolved-transfer `
  --permutations 100000 `
  --seed 20260523
```

Convenience npm aliases mirror the locked command shape:

```powershell
npm run isotrophy:v16:prepare
npm run isotrophy:v16:shard -- --shard-index 0 --shard-count 14
npm run isotrophy:v16:merge
npm run isotrophy:v16:analyze
```

```text
results/isotrophy/k-facet-v16-liao2021-tail-resolved-transfer/
```

Required files:

```text
manifest.json
source_support_census.csv
sample_frame.csv
sample_shard_XX_of_NN.csv
per_row_sample.csv
per_cell_rank.csv
frame_dispersion.csv          (frame-spread gate + diagnostics, all success rows)
coarse_zone_relationship.csv  (report-only zone/score sidecar)
permutation_summary.json
operator_commands.md
```

`per_row_sample.csv` must record, per orbit: the four per-frame `vf_0/vf_37/vf_90/vf_211`,
the ensemble `score`, the secondary `vf_0`, `zone_index_vf0`, `frame_spread`,
`mass_cell`, `stability`, `status`, and the D5 residuals. `per_cell_rank.csv` must
record per cell: `N_success`, `S`, `U`, `J_cell`, `D_cell`, `AUC_cell`, the primary
flag, and score quantiles (p25/p50/p75) split by stability for the readback.

`manifest.json` must include: `verdict`, `startedAt`, `completedAt`, `gitCommit`,
`target_sha256`, `v14_sample_excluded_count`, `v15_sample_excluded_count`, `sample_seed`,
`supported_cells`, `sample_rows_requested`, `sample_rows_success`, `attrition_fraction`
(+ Wilson), `primary_supported_cells`, `primary_success_rows`, `frame_spread_median`,
`frame_spread_p90`, `AUC_cond`, `J_cond`, `D_cond`, `p_perm`, `secondary_vf0_AUC_cond`,
`zone_auc_cond_report_only`, `target_tier = 2`.

## Claim Boundary

If v0.16 passes, the allowed claim is:

> On a fresh, doubly-held-out, outcome-balanced sample from liao2021's source-supported
> mass cells, a tail-resolved 4-frame ensemble velocity-fraction score transfers as a
> Tier-2 within-cell stability-ranking signal. The underlying velocity-fraction
> projection carries external stability information in this region even though its
> coarse-zone v0.11 form does not (v0.15).

Forbidden upgrades:

```text
does not claim the v0.11 COARSE-ZONE feature transfers (v0.15 settled: it does not)
does not claim full-catalog liao2021 transfer
does not claim confirmation outside the stable-support cells
does not estimate stable-orbit prevalence
does not claim independent Tier-3 confirmation
does not make a globally calibrated predictor
does not revise v0.15, v0.10b, or the v0.3h K_facet structural null
does not make isotrophy theorem-facing by itself
```

If v0.16 fails:

> In liao2021's stable-support region, even a tail-resolved frame-ensemble velocity-
> fraction score does not transfer at the registered floor: the velocity-fraction
> projection itself -- not merely its coarse binning -- does not carry external
> stability magnitude here. The v0.11 effect is supp-B-specific in magnitude.

If v0.16 is directional weak:

> The tail-resolved score detected a directional trace below the pre-registered
> external-transfer effect floor -- finer resolution did not lift the effect over the bar.

If v0.16 blocks or is undecidable, the allowed claim is limited to the blocking
condition.

## Next Chapter Boundary

Only a v0.16 PASS may justify a larger supported-region or multi-feature liao2021
chapter (re-locking scope, sharding, attrition, frame diagnostics, floor, claim).

If v0.16 fails or is directional-weak, the liao2021 velocity-fraction external-transfer
path should **close**: both the coarse zone (v0.15) and its tail-resolved frame-ensemble
relative (v0.16) will have been tested at the registered floor on held-out external
rows. Any further work would need a genuinely different substrate or a different
projection, not another vf variant on this catalog.

## Lock-In Statement

Committed before the v0.16 runner is written. Any change to the feature definition, the
frame set, the sample, the double-holdout rule, the decision bars, the effect floor, the
permutation, the firewall, the frame-spread gate, the report-only zone/score sidecar, or
the claim boundary after runner execution is a re-registration. The runner imports the
frozen v0.14/v0.15 D5 + zone + frame + AUC + permutation symbols unchanged and adds only
the ensemble-median scoring, the double holdout, the continuous-score AUC, the
frame-spread gate, and the coarse-zone relationship sidecar.
