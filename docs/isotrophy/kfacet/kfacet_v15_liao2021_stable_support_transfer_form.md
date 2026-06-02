# v0.15 liao2021 Stable-Support Transfer Form Draft

## Result (2026-06-01)

**Verdict: `stable_support_transfer_directional_weak`.** The coverage wall is defeated
and the test was decidable; the signal direction transfers and is permutation-significant,
but the effect size sits below the pre-registered 0.55 external-transfer floor.

```text
coverage           7/7 supported cells primary, 1120 primary rows   (>= 6 / >= 900)
attrition          0.0000  Wilson95 [0.0000, 0.0034]   (zero; stable-support region integrates cleanly)
frame-zone change  0.0170  Wilson95 [0.0109, 0.0263]   (band_A 0.0%, band_B 3.8%, outside 1.7%)
AUC_cond           0.5125  (J_cond 22960.5 / D_cond 44800)
p_perm             0.00033 (100k within-cell S/U permutation, seed 20260523)
```

**Mechanism (the licensed reading + why it is weak).** liao2021's stable-support region
is **zone-2-saturated**: every supported cell has 154-160 of 160 sampled orbits in the
velocity-heavy zone (one cell, qA3_qB3, is pure zone-2 -> `zone_degenerate`, contributing
exactly 0.5 under the locked retain-not-drop rule). The coarse {0.25,0.50} zone feature
therefore has almost no variance to discriminate on. Where the few low-zone orbits DO
appear they are disproportionately unstable -- the v0.11 direction (stable rows occupy
higher zones) -- so the pooled AUC is a real, consistent, significant 0.5125, but tiny
next to supp-B's internal 0.678. This is NOT the v0.14 discreteness artifact: the
balanced 80/80 design produced a smooth null (observed 0.51251 < null max 0.51433; null
mean 0.50000; 32 of 100k permutations >= observed). Per-cell AUC tracks zone spread
exactly -- the two cells with the most low-zone orbits (qA3_qB0 with 6, AUC 0.5375;
qA3_qB2 with 4, AUC 0.5250) carry the signal; the near-saturated cells sit at ~0.500-0.506.

**Allowed claim (locked):** "The stable-support liao2021 test detected a directional trace
below the pre-registered external-transfer effect floor." No PASS-level transfer claim is
licensed; the supp-B effect size does not reproduce externally.

**Independent verification.** A standalone brute-force recompute from `per_row_sample.csv`
(re-derive zone from recorded vf with the frozen cutpoints, apply the locked
N>=120/S>=50/U>=50 primary rule, re-pair S/U within each cell, re-pool) reproduced
AUC_cond 0.5125111607142857, J_cond 22960.5, D_cond 44800, 7 primary cells, 1120 rows
BIT-FOR-BIT, with 0 zone re-derivation mismatches. Receipt + verifiers:
`results/isotrophy/k-facet-v15-liao2021-stable-support-transfer/` (`manifest.json`,
`per_cell_rank.csv`, `permutation_summary.json`, `_independent_check.{py,json}`,
`_census_verify.{py,json}`, `_sample_verify.{py,json}`). git_commit d7045c2a.

**Next-chapter boundary.** Per the locked tree, `directional_weak` is neither a PASS (does
not justify a larger supported-region chapter) nor a `fails` (does not auto-close the
liao2021 path). The diagnosis is specific: the coarse 3-zone projection saturates where
vf -> 1, which is most of liao2021's stable-support region. Any continuation must register
a feature with resolution inside the velocity-heavy band (e.g. a finer or continuous vf
statistic) -- a new feature and a fresh pre-registration, not a v0.15 re-read.

---

Status: **OPERATOR LOCK 2026-06-01; VERDICT LANDED `stable_support_transfer_directional_weak` 2026-06-01.**
Reviewed for self-consistency, non-circularity,
and integrity. The source-support census was independently reproduced from the source
file with v0.14's own `build_mass_cells` (0 mismatches; 7 supported cells; the 80/80
case-control draw is feasible in every supported cell after v0.14 exclusion -- smallest
post-exclusion stable pool 482). Three additive hardening adjustments were applied at
lock (explicit feature-blind-draw firewall; zone-degeneracy retain-not-drop clarification;
per-cell zone-composition reporting requirement); none changes the sample, mass cells,
D5 path, effect floor, permutation, or claim boundary. Runner implementation began
after lock (`scripts/v15_liao2021_stable_support_transfer.py`), but no registered
v0.15 sample has been drawn, no v0.15 D5 rows integrated, and no v0.15 transfer
statistic computed at implementation time. This form locks the coverage-wall follow-up
to v0.14. Any change after this lock is a re-registration.

## Frame

v0.14 was a clean undecidable result, not a negative:

```text
verdict              sample_transfer_undecidable_coverage
primary_mass_cells   7   < 10
primary_success_rows 560 < 800
attrition            1.33% clean
frame-zone change    1.35% clean
report-only AUC      0.5239, p=0.00503
```

The block was structural: the registered 16 sorted-mass cells were not all capable of
hosting a within-cell S-vs-U comparison. Nine cells contained zero stable rows in the
v0.14 sample, and the liao2021 source population itself concentrates stable orbits in
a small stable-support region.

v0.15 asks the narrower, actually hostable question:

> Within liao2021 mass cells that source labels show can host both stable and unstable
> rows, does the frozen v0.11 coarse velocity-fraction zone order rank stable rows
> above unstable rows on a fresh outcome-balanced sample?

This is a coverage-rescue chapter. It does not erase v0.14; it changes the domain from
"all mass cells sampled uniformly" to "source-supported mass cells sampled
case-control."

## Integrity Caveat

v0.15 breaks one v0.14 condition deliberately:

```text
v0.14 sample: uniform within all 16 mass cells, outcome-blind.
v0.15 sample: outcome-balanced within stable-support mass cells, using source S/U
              labels only to make the rank comparison estimable.
```

That is a real design change. It is licensed only by the v0.14 coverage wall and is
locked before any v0.15 D5 or zone measurement.

Still frozen:

```text
target file SHA-256:     9c06eedc41b537b2ed926217cdf149d9f808a21866148ea3d89b9eb3c6069fd4
target tier:             Tier 2 only
mass-cell construction:  v0.14 nested sorted-mass 4 x 4 cells
D5 measurement:          v0.7/v0.13a DOP853 pipeline
zone cutpoints:          {0.25, 0.50}
zone order:              positional < mixed < velocity-heavy
primary statistic:       within-cell conditional rank AUC
effect floor:            AUC_cond >= 0.55
permutation p gate:      p_perm <= 0.01
frame-zone audit:        v0.13b/v0.14 frame perturbations
```

Forbidden:

```text
changing zone cutpoints
using raw velocity_fraction as the primary score
training a classifier or calibrator
choosing cells from v0.15 zones, vf, AUCs, or p-values
including v0.14 sample rows in the v0.15 primary sample
dropping frame-fragile rows from the primary statistic
selecting v0.15 sampled rows by velocity_fraction, zone_index, or any D5 output (the draw uses mass_cell + source stability label only)
using the balanced sample to estimate liao2021 stability prevalence
promoting any result to Tier-3 / independent confirmation
rewriting v0.10b's mass-marginal held-out null
rewriting v0.14's undecidable verdict
```

## Required Prior Receipts

The runner must assert all of the following before drawing a sample:

```text
v0.13 target inventory includes liao2021 as Tier 2
v0.13a leakage is bounded at 0.0
v0.13b verdict = coarse_zone_rule_frame_stable_enough_to_test
v0.13 liao2021 rate probe attrition <= 0.10 and Wilson95_high <= 0.20
v0.14 verdict = sample_transfer_undecidable_coverage
v0.14 target SHA matches this form's target SHA
v0.14 sample_frame.csv exists so v0.15 can exclude those rows
```

Any mismatch aborts.

## Source-Support Census

Before any v0.15 D5 measurement, compute source S/U counts in the frozen v0.14 mass
cells. This uses source masses and source stability labels only; it does not read or
compute vf, zones, monodromy, or any transfer statistic.

The census reads (independently reproduced from the source file with v0.14's
`build_mass_cells`; 0 mismatches; per-cell totals and the 13,315 stable total both
reconcile to the 135,445-row catalog):

| mass cell | source S | source U | disposition |
|---|---:|---:|---|
| `mass_qA0_qB0` | 486 | 7,980 | supported |
| `mass_qA0_qB1` | 0 | 8,466 | report-only |
| `mass_qA0_qB2` | 0 | 8,465 | report-only |
| `mass_qA0_qB3` | 0 | 8,465 | report-only |
| `mass_qA1_qB0` | 654 | 7,812 | supported |
| `mass_qA1_qB1` | 29 | 8,436 | marginal report-only |
| `mass_qA1_qB2` | 0 | 8,465 | report-only |
| `mass_qA1_qB3` | 0 | 8,465 | report-only |
| `mass_qA2_qB0` | 0 | 8,466 | report-only |
| `mass_qA2_qB1` | 2,765 | 5,700 | supported |
| `mass_qA2_qB2` | 0 | 8,465 | report-only |
| `mass_qA2_qB3` | 0 | 8,465 | report-only |
| `mass_qA3_qB0` | 2,621 | 5,845 | supported |
| `mass_qA3_qB1` | 3,186 | 5,279 | supported |
| `mass_qA3_qB2` | 2,444 | 6,021 | supported |
| `mass_qA3_qB3` | 1,130 | 7,335 | supported |

Locked stable-support rule:

```text
supported cell iff source S >= 80 AND source U >= 80 after excluding v0.14 sampled rows
```

Expected supported cells: 7.

The marginal `mass_qA1_qB1` cell is report-only in v0.15. It has only 29 stable rows
before v0.14 exclusion and cannot support the locked 80/80 case-control draw.

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
holdout rule:          exclude every orbit_index in v0.14 sample_frame.csv
```

If a supported cell has fewer than 80 stable or 80 unstable rows after v0.14 exclusion,
the run aborts for lock review. Deficits are not redistributed.

The draw is outcome-balanced by source label. This is acceptable for a rank-AUC test
because AUC is invariant to class prevalence under case-control sampling, but it means
v0.15 cannot estimate how common stable orbits are in liao2021.

## D5 Measurement

For sampled rows, compute the frozen v0.7/v0.13a D5 measurement:

```text
integrator:           DOP853
rtol:                 1e-12
atol:                 1e-12
max_step_fraction:    0.02
symplecticity_gate:   1e-4
reciprocal_pair_gate: 1e-4
```

Primary score:

```text
zone_index = 0 if velocity_fraction < 0.25
zone_index = 1 if 0.25 <= velocity_fraction < 0.50
zone_index = 2 if velocity_fraction >= 0.50
```

Rows that are integration-blocked or fail sanity gates are excluded from the primary
rank statistic and counted in attrition.

Attrition policy:

```text
attrition_fraction <= 0.05:
  clean sample

0.05 < attrition_fraction <= 0.10 AND Wilson95_high <= 0.20:
  attrition warning; transfer may pass only with warning language

attrition_fraction > 0.10 OR Wilson95_high > 0.20:
  verdict = stable_support_transfer_blocked_by_attrition
```

## Primary Domain

After D5 attrition, a supported mass cell enters the primary statistic only if:

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

If the coverage gate fails, verdict is `stable_support_transfer_undecidable_coverage`.

Cells that fail primary gates are report-only.

## Primary Statistic

Within each primary supported mass cell, compare every stable row to every unstable row:

```text
score(row) = zone_index(row)

J_cell = #[(S,U): score(S) > score(U)]
       + 0.5 * #[(S,U): score(S) = score(U)]

D_cell = S_cell * U_cell
AUC_cell = J_cell / D_cell

J_cond = sum J_cell over primary cells
D_cond = sum D_cell over primary cells
AUC_cond = J_cond / D_cond
```

Direction is pre-registered:

```text
stable rows should have higher zone_index than unstable rows
```

This is the same monotone coarse-zone direction used by v0.11/v0.14.

Zone-degenerate primary cells are retained, not dropped. A supported cell whose sampled
rows all share one zone contributes `AUC_cell = 0.5` by construction (no rank
information) and stays in the pooled `AUC_cond`. This dilutes the pooled statistic toward
0.5 and can only make the test HARDER to pass -- it is an anti-inflation property, not a
correction to undo (v0.14 had two such pure-zone-2 supported cells). `per_cell_rank.csv`
records per-cell zone composition so any degeneracy is visible in the readback.

## Permutation Test

Primary p-value:

```text
within-cell S/U permutation
preserve each primary cell's zone multiset
preserve each primary cell's S and U counts
permutations = 100000
seed = 20260523
p_perm = (1 + # permuted J_cond >= observed J_cond) / (1 + permutations)
```

The permutation test is one-sided in the pre-registered direction.

## Frame-Zone Audit

Run the same frame perturbations as v0.13b/v0.14 on sampled rows with successful base
D5:

```text
rotations SO(2) by {37, 90, 211} degrees
translation by the v0.13a fixed offset, absorbed by CoM-centering
```

Record:

```text
base_zone
rotated_zone
zone_changed
distance_to_nearest_cutpoint
fragile_band in {band_A, band_B, outside}
```

Frame policy:

```text
zone_change_fraction <= 0.05:
  clean

0.05 < zone_change_fraction <= 0.15:
  frame warning; transfer may pass only with warning language

zone_change_fraction > 0.15:
  verdict = stable_support_transfer_blocked_by_frame_instability
```

The primary statistic includes frame-fragile rows. Dropping them is forbidden.

## Runtime Plan

The v0.14 sharded measurement completed the same target and same D5 path with clean
attrition. v0.15 has fewer base rows than v0.14 but remains a long measurement because
the frame-zone audit multiplies successful rows by up to four additional D5-equivalent
passes.

Expected command shape after implementation:

```powershell
python scripts/v15_liao2021_stable_support_transfer.py prepare `
  --target docs/isotrophy/external_targets/liao2021_nonhierarchical.txt `
  --v14 results/isotrophy/k-facet-v14-liao2021-sampled-transfer `
  --out results/isotrophy/k-facet-v15-liao2021-stable-support-transfer `
  --stable-per-cell 80 `
  --unstable-per-cell 80 `
  --seed 20260523

python scripts/v15_liao2021_stable_support_transfer.py shard `
  --out results/isotrophy/k-facet-v15-liao2021-stable-support-transfer `
  --shard-index 0 `
  --shard-count 14

python scripts/v15_liao2021_stable_support_transfer.py merge `
  --out results/isotrophy/k-facet-v15-liao2021-stable-support-transfer `
  --shard-count 14

python scripts/v15_liao2021_stable_support_transfer.py analyze `
  --out results/isotrophy/k-facet-v15-liao2021-stable-support-transfer `
  --permutations 100000 `
  --seed 20260523
```

The 14 shard commands may be run concurrently by the operator. The full measurement
must not be run inline as one agent command.

Convenience npm aliases mirror the locked command shape:

```powershell
npm run isotrophy:v15:prepare
npm run isotrophy:v15:shard -- --shard-index 0 --shard-count 14
npm run isotrophy:v15:merge
npm run isotrophy:v15:analyze
```

## Verdict Tree

```text
stable_support_transfer_blocked_by_receipt
  required prior receipts missing or inconsistent
  OR target SHA mismatch
  OR v0.14 sample exclusion unavailable
  OR supported-cell census deviates from the locked rule and cannot draw 80/80

stable_support_transfer_blocked_by_attrition
  attrition_fraction > 0.10 OR attrition_wilson95_high > 0.20

stable_support_transfer_blocked_by_frame_instability
  zone_change_fraction > 0.15

stable_support_transfer_undecidable_coverage
  primary_supported_cells < 6 OR primary_success_rows < 900

stable_support_transfer_passes_clean
  attrition_fraction <= 0.05
  AND zone_change_fraction <= 0.05
  AND AUC_cond >= 0.55
  AND p_perm <= 0.01

stable_support_transfer_passes_with_warning
  AUC_cond >= 0.55
  AND p_perm <= 0.01
  AND no hard block fired
  AND (0.05 < attrition_fraction <= 0.10 OR 0.05 < zone_change_fraction <= 0.15)

stable_support_transfer_directional_weak
  AUC_cond > 0.50
  AND AUC_cond < 0.55
  AND p_perm <= 0.01

stable_support_transfer_fails
  AUC_cond <= 0.50 OR p_perm > 0.01
```

## Required Outputs

Expected output directory:

```text
results/isotrophy/k-facet-v15-liao2021-stable-support-transfer/
```

Required files:

```text
manifest.json
source_support_census.csv
sample_frame.csv
sample_shard_XX_of_14.csv
per_row_sample.csv
per_cell_rank.csv
frame_zone_audit.csv
permutation_summary.json
operator_commands.md
```

`per_cell_rank.csv` must record, per supported cell: `N_success`, `S`, `U`, the
`zone0`/`zone1`/`zone2` counts and the S/U split by zone, `J_cell`, `D_cell`, `AUC_cell`,
and the primary flag -- so zone-degenerate cells are visible without recomputation.

`manifest.json` must include:

```text
verdict
startedAt
completedAt
gitCommit
target_sha256
v14_sample_excluded_count
sample_seed
supported_cells
sample_rows_requested
sample_rows_success
attrition_fraction
attrition_wilson95_low
attrition_wilson95_high
primary_supported_cells
primary_success_rows
zone_change_fraction
AUC_cond
J_cond
D_cond
p_perm
target_tier = 2
```

## Claim Boundary

If v0.15 passes, the allowed claim is:

> On a fresh outcome-balanced sample from liao2021's source-supported mass cells,
> excluding v0.14 sampled rows, the frozen supp-B coarse velocity-fraction zone order
> transfers as a Tier-2 within-cell stability-ranking signal.

Forbidden upgrades:

```text
does not claim full-catalog liao2021 transfer
does not claim confirmation in the nine non-stable-support cells
does not estimate stable-orbit prevalence
does not claim independent Tier-3 confirmation
does not make a globally calibrated predictor
does not overturn v0.10b
does not revise v0.14 from undecidable to pass/fail
does not revise the v0.3h K_facet structural null
does not make isotrophy theorem-facing by itself
```

If v0.15 fails, the allowed claim is:

> In the liao2021 stable-support region, the frozen supp-B coarse zone signal did not
> transfer under the registered outcome-balanced within-cell test.

If v0.15 is directional weak, the allowed claim is:

> The stable-support liao2021 test detected a directional trace below the
> pre-registered external-transfer effect floor.

If v0.15 blocks or is undecidable, the allowed claim is limited to the blocking
condition.

## Next Chapter Boundary

Only a v0.15 PASS may justify drafting a larger supported-region or full supported-cell
liao2021 chapter. That later chapter must re-lock sample/full-sweep scope, sharding,
attrition policy, frame-zone diagnostics, effect floor, and claim boundary.

If v0.15 fails, the liao2021 external-transfer path should close unless a new target or
a genuinely new, pre-registered feature replaces the coarse zone projection.
