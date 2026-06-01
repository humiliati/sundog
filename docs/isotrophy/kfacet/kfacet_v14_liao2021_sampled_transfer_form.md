# v0.14 liao2021 Sampled Zone Transfer Form Draft

## Result (2026-06-01)

**Verdict: `sample_transfer_undecidable_coverage`.** The coverage gate fired exactly
as locked — BOTH sub-gates fail:

```text
primary_mass_cells   = 7   < 10   (FAIL)
primary_success_rows = 560 < 800  (FAIL)
```

**Blocking condition (the licensed reading).** Of the 16 registered sorted-mass
quantile cells, **9 contained zero stable orbits** and were report-only; only 7
cleared the `N_success>=40 AND S>=4 AND U>=4` primary criterion. liao2021's
non-hierarchical population is both overwhelmingly **unstable** (stable orbits
concentrate in the high-`qA` corner of mass space — qA3 cells) and overwhelmingly
**zone-2 / velocity-heavy** (most cells are 64–80 of 80 rows in zone 2; two primary
cells, qA3_qB2 and qA3_qB3, are pure zone-2). A within-cell stable-vs-unstable
conditional comparison simply cannot be hosted in most registered cells. Per the
locked claim boundary, **no transfer reading is licensed** — the supp-B coarse zone
signal is neither confirmed nor refuted on liao2021 by this test.

**Report-only diagnostics (NOT a transfer reading; recorded per Output Contract):**

```text
attrition_fraction  = 0.0133  Wilson95 [0.0083, 0.0212]   (clean; no supp-A wall)
zone_change_fraction= 0.0135  Wilson95 [0.0084, 0.0215]   (band_A 2.0%, band_B 1.5%, outside 1.3%)
AUC_cond            = 0.5239  (J_cond 3668 / D_cond 7001, 7 primary cells, 560 rows)
p_perm              = 0.00503 (100k within-cell S/U permutation, seed 20260523)
sample              = 1280 requested -> 1263 success (1 integration_blocked, 16 sanity_failed)
```

The AUC/p above are pinned to 0.5 by structure in 2 of the 7 primary cells
(pure-zone-2 -> mechanically 0.5) and the p sits at the permutation discreteness
ceiling (observed AUC == null max; 502 of 100k ties, 0 strictly exceed). These are
diagnostics only and do not promote.

**Independent verification.** A standalone brute-force recompute from
`per_row_sample.csv` (re-derive zone from recorded vf with the frozen {0.25,0.50}
cutpoints, re-pair S/U within each cell, re-pool) reproduced AUC_cond 0.523925153549493,
J_cond 3668.0, D_cond 7001, 7 primary cells, 560 rows **bit-for-bit**, with **0 zone
re-derivation mismatches** (the runner's `zone_index` column is faithful to the v0.11
rule). Receipt + verifier: `results/isotrophy/k-facet-v14-liao2021-sampled-transfer/`
(`manifest.json`, `per_cell_rank.csv`, `permutation_summary.json`,
`_v14_independent_check.{py,json}`). git_commit df699390.

**Next-chapter boundary (unchanged).** Per the locked tree, only a PASS justifies a
larger liao2021 chapter; an undecidable result does not. A future attempt must re-lock
the design to defeat the coverage problem — e.g. **outcome-balanced or stability-
stratified cell construction** so cells are guaranteed to host both classes, or a
**stable-orbit-targeted draw** within mass cells (signal-blind on vf, balanced on the
stability label). v0.14 remains the honest first transfer test, not the final claim.

---

Status: **OPERATOR LOCK 2026-06-01; VERDICT LANDED `sample_transfer_undecidable_coverage`.**
Runner implementation began after lock
(`scripts/v14_liao2021_sampled_transfer.py`), but no registered v0.14 sample has
been drawn, no transfer statistic computed, and no v0.14 D5 rows integrated as a
measurement receipt at implementation time. This document locks the first sampled
transfer test from supp-B to the Tier-2 Li/Liao 2021 non-hierarchical catalog.

Reviewed for self-consistency, non-circularity, and integrity:

- **Conditioning-design break disclosed + locked pre-sample.** v0.14 conditions on
  sorted-mass quantile cells (not m3 strata) because liao2021 fixes m3=1 and varies
  (m1,m2); the Integrity Caveat names this a real design choice, not a mechanical
  replay. Cells are mass-only, deterministic-tie, signal-blind. The vf-zone feature,
  the {0.25,0.50} cutpoints, the within-cell conditional-AUC statistic, and the D5
  pipeline stay FROZEN from v0.11/v0.13a -- only the conditioning variable changes,
  forced by the target. The claim is scoped to "conditional within-mass-cell," not
  "the identical rule."
- **Signal-blind sample** (mass-stratified, outcome-blind, seed 20260523); the
  forbidden-list bans cutpoint changes, raw-vf scoring, classifier training, post-hoc
  cell tuning, dropping frame-fragile rows, Tier-3 promotion, and v0.10b rewriting.
- **Anti-overclaim:** the `AUC_cond >= 0.55` effect floor + graded verdicts
  (clean / warning / directional-weak / fail) stop a large sample from turning a tiny
  effect into a claim; the frame diagnostic reports bands A/B WITHOUT dropping
  frame-fragile rows from the primary statistic (the v0.13b discipline).
- Honest Tier-2 (near-the-source) claim boundary; runtime sharded (16 x 80).

Adjustment at lock (one): added a target-SHA-256-match assertion -- v0.14 must score
the EXACT file the v0.13a / v0.13b / feasibility gates were measured on (the v0.13a
manifest `download_sha256`); a different or updated file silently invalidates every
preflight gate.

## Frame

The v0.13/v0.13a sequence has now cleared the target-feasibility question:

```text
target:          liao2021_nonhierarchical_unequal_mass_135445
independence:    Tier 2 (same Li/Liao lineage, different paper / orbit construction)
leakage:         bounded, leakage = 0.0
adapter:         expansion-only; P1 + parity clean
frame zones:     coarse_zone_rule_frame_stable_enough_to_test
D5 feasibility:  97 success / 0 blocked / 3 sanity-fail
                 attrition = 3.0%, Wilson95 [1.0%, 8.45%]
```

v0.14 asks the first actual transfer question:

> On a pre-registered sample from liao2021, does the frozen v0.11 coarse
> velocity-fraction zone order rank stable rows above unstable rows after
> conditioning on liao2021 mass-ratio cells?

This is not a full-catalog lock. It is a bounded-cost sampled transfer test.

## Integrity Caveat

v0.14 breaks one thing relative to v0.11 because the target forces it:

```text
v0.11 conditioning: within m3 strata, because supp-B has m1=m2=1 and variable m3.
v0.14 conditioning: within registered sorted-mass cells, because liao2021 has
                    m3=1 fixed and variable (m1,m2).
```

That is a real design choice, not a mechanical replay. It is locked here before
any v0.14 sample is drawn or scored.

Still frozen from v0.11/v0.13a:

```text
zone cutpoints:        {0.25, 0.50}
zone order:            positional < mixed < velocity-heavy
D5 measurement:        v0.7/v0.12/v0.13a inherited pipeline
target claim tier:     Tier 2 only
```

Forbidden in v0.14:

```text
changing zone cutpoints
using raw velocity_fraction as the primary score
training a classifier or calibrator
tuning mass cells after seeing zones or transfer statistics
dropping frame-fragile rows from the primary statistic
promoting any result to Tier-3 / independent confirmation
rewriting v0.10b's mass-marginal held-out null
```

## Frozen Inputs

Target:

```text
docs/isotrophy/external_targets/liao2021_nonhierarchical.txt
```

If the promoted tracked target file is not present yet, the runner must fetch the
exact source URL from the v0.13 inventory, write it first to `_staging`, record its
SHA-256, and require operator approval before promoting or scoring it:

```text
https://raw.githubusercontent.com/sjtu-liao/three-body/main/non-hierarchical-3b-supplementary_data.txt
```

Required prior receipts:

```text
docs/isotrophy/kfacet/kfacet_v13_target_inventory_snapshot.md
docs/isotrophy/kfacet/kfacet_v13a_liao2021_adapter_leakage_preflight_form.md
results/isotrophy/k-facet-v13a-liao2021-preflight/manifest.json
results/isotrophy/k-facet-v13b-frame-zone-stability/manifest.json
results/isotrophy/k-facet-v13-external-target-search/manifest.json
```

The runner must assert:

```text
v0.13 target verdict:    external_target_locked
selected slug:           liao2021_nonhierarchical_unequal_mass_135445
target tier:             2
leakage bounded:         true
leakage fraction:        0.0
v0.13b verdict:          coarse_zone_rule_frame_stable_enough_to_test
D5 feasibility attrition: <= 0.10
D5 feasibility Wilson hi: <= 0.20
target SHA-256:          == download_sha256 in the v0.13a preflight manifest
                         (results/isotrophy/k-facet-v13a-liao2021-preflight/manifest.json)
                         -- v0.14 must score the EXACT file the leakage / frame /
                         feasibility gates were measured on; a different or updated
                         file invalidates every preflight gate.
```

Any mismatch aborts.

## Mass Conditioning

For every liao2021 row, compute the sorted normalized mass triple:

```text
mu_i = m_i / (m1 + m2 + m3)
(a,b,c) = sort(mu_1, mu_2, mu_3), with a <= b <= c
```

Use only source masses. Do not use stability labels, D5 outputs, zones, periods,
energy, or angular momentum to define cells.

Primary cell design: **nested 4 x 4 mass quantile cells**.

```text
1. On the post-leakage eligible target population, split rows into four a-quartile
   bands by source-only quantiles of a.
2. Within each a-quartile band, split rows into four b-quartile bands by
   source-only quantiles of b.
3. The cell key is mass_qA{0..3}_qB{0..3}; expected cells = 16.
4. Quantile ties are resolved by stable catalog order, not by labels or features.
```

This gives mass-conditioned comparisons without trying to pretend liao2021 has
the same `m3` strata as supp-B.

## Sample Design

Primary sample:

```text
sample rows:       1280
cells:             16 mass cells
rows per cell:     80
seed:              20260523
sampling rule:     uniform without replacement within each mass cell
```

If a mass cell contains fewer than 80 rows, take all rows in that cell and do not
redistribute the deficit. This condition is expected not to occur in the 135,445-row
target; if it does, it is a review flag but not a tuning opportunity.

The sample is mass-stratified and outcome-blind. Source S/U labels are used later
for the registered rank statistic and coverage gates, not to draw the sample.

## Runtime Plan

The feasibility probe measured approximately:

```text
seconds_per_row = 6.4
```

Projected v0.14 sample cost:

```text
1280 rows base D5 only    ~= 2.3 hours
16 shards x 80 rows       ~= 8.5 minutes per shard, plus overhead, for base D5 only
frame-zone audit          ~= up to 4 additional D5-equivalent passes per successful row
```

The implemented shard runner is faithful to the full frame-zone audit, so practical
shard wall-clock can approach roughly 5x the base-only estimate (still staged, not
inline). The full 1280-row sample must not be run inline by an agent as one command.
The intended execution is 16 shard commands plus a merge/analyze step. If any single
shard is projected over the repository inline budget, stage it for the operator
instead of running it.

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

Rows that are integration-blocked or fail sanity gates are excluded from the
primary rank statistic and counted in attrition.

Attrition policy:

```text
attrition_fraction <= 0.05:
  clean sample

0.05 < attrition_fraction <= 0.10 AND Wilson95_high <= 0.20:
  attrition warning; transfer may pass only with warning language

attrition_fraction > 0.10 OR Wilson95_high > 0.20:
  verdict = sample_transfer_blocked_by_attrition
```

## Primary Domain

After D5 attrition, a mass cell enters the primary statistic only if:

```text
N_success >= 40
S >= 4
U >= 4
```

Coverage gate:

```text
primary_mass_cells >= 10
primary_success_rows >= 800
```

If the coverage gate fails, verdict is `sample_transfer_undecidable_coverage`.
Cells that fail primary gates are report-only.

## Primary Statistic

Within each primary mass cell, compare every stable row to every unstable row:

```text
J_c = #{(S_i, U_j): zone_index(S_i) > zone_index(U_j)}
      + 0.5 * #{(S_i, U_j): zone_index(S_i) = zone_index(U_j)}

D_c = S_c * U_c
```

Pool across primary cells:

```text
J_cond   = sum_c J_c
D_cond   = sum_c D_c
AUC_cond = J_cond / D_cond
```

`AUC_cond = 0.5` is the constant-zone conditional baseline. Larger values mean
stable rows tend to occupy higher frozen vf zones than unstable rows **within the
same registered mass cell**.

## Binding Null

Binding p-value: stratified label permutation within primary mass cells.

```text
preserve each cell's zone_index values
preserve each cell's S and U counts
shuffle S/U labels within each cell
seed = 20260523
permutations = 100000
p_perm = (1 + count(AUC_perm >= AUC_observed)) / (1 + permutations)
```

No exact-enumeration kernel is required for v0.14. It may be reported as a sidecar
only if cheap, but the binding p-value is the permutation p.

## Frame-Zone Diagnostic

v0.13b established that the coarse zone rule is frame-stable enough to test:

```text
supp-B zone-change fraction:   4.35% <= 15%
liao2021 pooled zone-change:   about 1.3% <= 5%
```

v0.14 must still report frame stability on the sampled rows.

For every sampled row with a successful D5 measurement, compute zone changes under
the registered frame perturbations from v0.13b and record:

```text
base_zone
rotated_zone
zone_changed = base_zone != rotated_zone
distance_to_nearest_cutpoint
fragile_band = none | A | B
```

Fragile catalog bands, using 1-based liao2021 catalog row position:

```text
band A: 74697..84708
band B: 109861..116560
```

Frame diagnostic branches:

```text
zone_change_fraction <= 0.05:
  clean frame diagnostic

0.05 < zone_change_fraction <= 0.15:
  frame warning; transfer may pass only with warning language

zone_change_fraction > 0.15:
  verdict = sample_transfer_blocked_by_frame_instability
```

The primary statistic is always computed on the full primary domain, not on a
post-hoc frame-robust subset. The frame-robust subset may be reported as a
diagnostic sidecar only.

## Verdict Tree

Run prior-receipt, target, sample, attrition, coverage, and frame gates before
interpreting the rank statistic.

```text
sample_transfer_blocked_by_receipt
  required v0.13/v0.13a/v0.13b receipts are missing or inconsistent

sample_transfer_blocked_by_attrition
  attrition_fraction > 0.10 OR Wilson95_high > 0.20

sample_transfer_blocked_by_frame_instability
  sampled zone_change_fraction > 0.15

sample_transfer_undecidable_coverage
  primary_mass_cells < 10 OR primary_success_rows < 800

sample_transfer_passes_clean
  attrition_fraction <= 0.05
  AND zone_change_fraction <= 0.05
  AND AUC_cond >= 0.55
  AND p_perm <= 0.01

sample_transfer_passes_with_warning
  AUC_cond >= 0.55
  AND p_perm <= 0.01
  AND no hard block fired
  AND (0.05 < attrition_fraction <= 0.10 OR 0.05 < zone_change_fraction <= 0.15)

sample_transfer_directional_weak
  AUC_cond > 0.50
  AND AUC_cond < 0.55
  AND p_perm <= 0.01

sample_transfer_fails
  AUC_cond <= 0.50 OR p_perm > 0.01
```

Effect floor `AUC_cond >= 0.55` is inherited from the v0.12 external-transfer
posture to avoid letting large samples turn tiny effects into claim upgrades.

## Required Outputs

Expected output directory:

```text
results/isotrophy/k-facet-v14-liao2021-sampled-transfer/
```

Required files:

```text
manifest.json
source_profile.csv
mass_cells.csv
sample_frame.csv
sample_shard_XX_of_16.csv
per_row_sample.csv
per_cell_rank.csv
frame_zone_audit.csv
permutation_summary.json
operator_commands.md
```

`manifest.json` must include:

```text
verdict
startedAt
completedAt
gitCommit
target_sha256
sample_seed
sample_rows_requested
sample_rows_success
attrition_fraction
attrition_wilson95_low
attrition_wilson95_high
primary_mass_cells
primary_success_rows
zone_change_fraction
AUC_cond
J_cond
D_cond
p_perm
target_tier = 2
```

## Command Shape After Implementation

Expected additive runner:

```text
scripts/v14_liao2021_sampled_transfer.py
```

Implementation note: if the promoted tracked target file is absent locally, the
runner may use the existing `_staging` copy only when its SHA-256 matches the v0.13a
`download_sha256` receipt. A different file aborts before sampling.

Expected command shapes:

```powershell
python scripts/v14_liao2021_sampled_transfer.py prepare `
  --target docs/isotrophy/external_targets/liao2021_nonhierarchical.txt `
  --out results/isotrophy/k-facet-v14-liao2021-sampled-transfer `
  --sample-rows-per-cell 80 `
  --seed 20260523

python scripts/v14_liao2021_sampled_transfer.py shard `
  --out results/isotrophy/k-facet-v14-liao2021-sampled-transfer `
  --shard-index 0 `
  --shard-count 16

python scripts/v14_liao2021_sampled_transfer.py merge `
  --out results/isotrophy/k-facet-v14-liao2021-sampled-transfer `
  --shard-count 16

python scripts/v14_liao2021_sampled_transfer.py analyze `
  --out results/isotrophy/k-facet-v14-liao2021-sampled-transfer `
  --permutations 100000 `
  --seed 20260523
```

Convenience npm aliases mirror these commands:

```powershell
npm run isotrophy:v14:prepare
npm run isotrophy:v14:shard -- --shard-index 0 --shard-count 16
npm run isotrophy:v14:merge
npm run isotrophy:v14:analyze
```

The 16 shard commands may be run concurrently by the operator. A future runner
may add a convenience launcher, but the receipt must still preserve per-shard CSVs.

## Claim Boundary

If v0.14 passes, the allowed claim is:

> On a pre-registered 1,280-row mass-stratified sample from the Tier-2 Li/Liao
> 2021 non-hierarchical catalog, the frozen v0.11 coarse velocity-fraction zone
> order transfers as a conditional within-mass-cell stability-ranking signal.

Forbidden upgrades:

```text
does not claim full-catalog liao2021 transfer
does not claim independent Tier-3 confirmation
does not make a globally calibrated predictor
does not overturn v0.10b
does not revise the v0.3h K_facet structural null
does not make isotrophy theorem-facing by itself
does not claim anything about attrited rows, report-only cells, or untested catalogs
```

If v0.14 fails, the allowed claim is:

> The frozen supp-B coarse zone signal did not transfer to the registered liao2021
> sampled mass-cell test.

If v0.14 is directional weak, the allowed claim is:

> The sampled liao2021 test detected a directional trace below the pre-registered
> external-transfer effect floor.

If v0.14 blocks or is undecidable, the allowed claim is limited to the blocking
condition. No transfer reading is licensed.

## Next Chapter Boundary

Only a v0.14 PASS may justify drafting a full-catalog or larger-sample liao2021
chapter. That later chapter must re-lock:

```text
whether it is full-catalog or sampled
mass-cell design
sharding plan
attrition policy
frame-zone diagnostic
effect floor
claim boundary
```

v0.14 is the first transfer test, not the final liao2021 claim.
