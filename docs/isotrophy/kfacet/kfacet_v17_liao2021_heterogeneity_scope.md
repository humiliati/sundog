# v0.17 liao2021 Tail-Resolved Heterogeneity Replication Form

## Result (2026-06-02)

**Verdict: `heterogeneous_transfer_replicates_clean`** -- the top of the tree. On fresh
triple-held-out rows the Tier-2 tail-resolved transfer replicates AND its mass-cell
heterogeneity reproduces.

```text
pooled replication: AUC_cond 0.646875 (J 28980 / D 44800)  vs v0.16 0.6469866  -> clears 0.55
                    p_perm 1.0e-5   attrition 0.0000   frame-spread median 0.00037 p90 0.0256 (clean)
heterogeneity:      Spearman rho 1.0  exact-perm p_rho 1/5040 = 0.000198 (only the correct
                    pairing of 5040 reaches rho=1)  -> heterogeneity replicates
```

**The pooled transfer is not a v0.16 fluke.** A pre-registered replication on fresh rows
(excluding all v0.14/v0.15/v0.16 orbits, 0/0/0 overlap) lands AUC_cond 0.646875 -- within
0.0001 of v0.16's 0.6469866 -- at the permutation floor. The held-out external Tier-2
transfer is confirmed twice.

**The mass-cell heterogeneity is reproducible structure, not one-sample texture.** All
seven fresh per-cell AUCs preserve v0.16's exact rank order (rho=1.0), and all seven
preserve their direction relative to 0.5 -- including both sub-0.5 cells staying sub-0.5
(`qA2_qB1` 0.432->0.453, `qA3_qB0` 0.470->0.474). The local pattern is real and
reproducible. The fresh AUCs are genuinely different from v0.16 (max abs diff 0.031), so
rho=1.0 is a true rank replication, not a re-read.

**Honest bound (n=7):** rho=1.0 is a seven-point rank concordance; the exact-perm
p=1/5040 is genuinely significant (a perfect ordering is unlikely by chance), but the
claim is "reproducible seven-cell heterogeneity," not a mechanism. WHY `qA2_qB1` and
`qA3_qB0` sit below 0.5 is the open question the non-gating anatomy sidecar opens.

**Post-hoc anatomy (interpretation, not a registered mechanism).** Ordered by fresh
v0.17 AUC, the sidecar reads the replicated pattern as an instrument-reliability map:

```text
cell        AUC    masses(a,b,c)      zone2  S_p50   U_p50   S-U     frame_p90
qA2_qB1    0.453   .311 .327 .362     158    .9366   .9687  -0.0322   0.321
qA3_qB0    0.474   .320 .325 .355     154    .9461   .9540  -0.0079   0.025
qA1_qB0    0.563   .302 .325 .376     157    .9937   .9938  -0.0000   0.022
qA3_qB2    0.653   .326 .333 .341     158    .9548   .9198  +0.0351   0.004
qA3_qB1    0.700   .324 .330 .346     156    .9558   .9286  +0.0272   0.009
qA3_qB3    0.744   .324 .336 .339     159    .9517   .8472  +0.1044   0.005
qA0_qB0    0.941   .289 .327 .383     157    .9966   .9905  +0.0061   0.001
```

`qA2_qB1` is the frame-fragile cell: its frame-spread p90 is 0.321, over an order of
magnitude above the well-calibrated cells and exactly the localized fragility already
flagged by the v0.13b frame audit. `qA3_qB0` is closer to a near-null than to a competing
physics: fresh AUC 0.474 with a median score gap of only -0.008. Everywhere the ensemble
is frame-stable and the cell has score resolution, the v0.11 direction holds. Zone
saturation is uniform (154-159/160 rows in zone 2), so the coarse zone is not the
discriminator. This anatomy is coherent with the receipts, but it remains descriptive:
confirming "reliability drives per-cell AUC" would require its own wider-cell chapter.

**Independent verification.** A standalone brute-force pair-count pooled AUC + a
hand-rolled rank/Pearson exact-permutation Spearman (independent of the runner's midrank
path) reproduced AUC_cond 0.646875, J 28980, D 44800, rho 1.0, p_rho 1/5040 BIT-FOR-BIT,
and confirmed the v0.17 vector differs from v0.16 (not a re-read). Receipt + verifier:
`results/isotrophy/k-facet-v17-liao2021-heterogeneity/` (`manifest.json`,
`per_cell_rank.csv`, `heterogeneity_replication.json`, `anatomy_sidecar.csv`,
`_independent_check.{py,json}`).

**Claim (locked):** the tail-resolved velocity-fraction projection transfers on fresh
held-out liao2021 samples AND its mass-cell heterogeneity is reproducible enough to report
as structure. Bounded: Tier-2 / Li-Liao lineage / stable-support region / within-cell rank
signal / not coarse-zone / not full-catalog prevalence / not Tier-3 independent / not
theorem-facing. The v0.16 pass stands; v0.17 strengthens it from a single external pass to
a replicated one with reproducible per-cell structure.

---

Status: **OPERATOR LOCK 2026-06-02; VERDICT LANDED `heterogeneous_transfer_replicates_clean` 2026-06-02.**
At lock time no v0.17 runner had been written, no v0.17 sample had been drawn, no v0.17
D5 rows had been integrated, and no v0.17 statistic had been computed. The completed run
followed the locked form: fresh-row REPLICATION of v0.16, not a new feature search.

Prepared receipt:

```text
command:      npm run isotrophy:v17:prepare
out:          results/isotrophy/k-facet-v17-liao2021-heterogeneity/
supported:    7/7 locked cells
sample:       1120 rows (80 S + 80 U per cell)
holdout:      excluded 3520 prior sample rows (v0.14 + v0.15 + v0.16)
overlap:      v17 vs v14/v15/v16 = 0 / 0 / 0
status:       tail_resolved_heterogeneity_prepared_not_measured
```

Lock-review adjustments folded in (2026-06-02):

```text
heterogeneity stat:   recast from a bare Spearman rho>=0.60 threshold to an EXACT
                      permutation Spearman test (7! = 5040 enumeration, p <= 0.05);
                      rho>=0.60 alone is ~p 0.07 at n=7, not significant.
sign concordance:     demoted to a report-only per-cell panel with per-cell SE; only the
                      ~3-SE-separated cells carry a real directional prediction.
verdict gate:         the POOLED replication is the sole verdict-bearing gate; the
                      heterogeneity test only chooses between the two pooled-pass labels.
open decisions:       sample 80/80 (parity); exact-perm rho bar; lock the exact reference
                      vector (rho consumes its rank order); add Option C anatomy sidecar
                      non-gating on existing v0.16 rows.
verified pre-lock:    pinned v0.16 vector == receipt (0.0 diff); triple-holdout 80/80
                      feasible in all 7 cells (smallest eligible stable pool 322).
```

## Frame

v0.16 landed the first clean Tier-2 external transfer of the tail-resolved score
(`AUC_cond = 0.6470`, `p_perm = 1.0e-5`, 1120 fresh double-holdout rows). It is real at
the pooled within-cell level but not a uniform per-cell law:

```text
in-direction cells: qA0_qB0, qA1_qB0, qA3_qB1, qA3_qB2, qA3_qB3
reversed cells:     qA2_qB1, qA3_qB0
range:              AUC_cell 0.432 -> 0.972
```

v0.17 asks the replication question:

> Does the tail-resolved velocity-fraction signal replicate as a pooled external transfer
> at the locked v0.16 floor on fresh rows, and does the v0.16 mass-cell heterogeneity
> pattern repeat well enough to report as structure rather than one-sample texture?

## Integrity Caveat

```text
replication, not discovery: v0.17 may use v0.16 to DEFINE the replication target (the
   per-cell reference vector) precisely because it replicates a discovered pattern on
   FRESH rows -- it does not discover a new law.
feature:  FROZEN. The v0.16 4-frame ensemble-median score is reused exactly. No new
          feature, kernel, cutpoint, or mass partition.
domain:   FRESH (triple-holdout). The seven v0.16 supported cells, excluding every
          v0.14, v0.15, AND v0.16 sampled orbit.
firewall: the heterogeneity test MUST NOT rescue a pooled failure; a heterogeneity
          failure MUST NOT demote the v0.16 pass.
```

Frozen from v0.14 / v0.15 / v0.16 (unchanged):

```text
target file SHA-256:     9c06eedc41b537b2ed926217cdf149d9f808a21866148ea3d89b9eb3c6069fd4
target tier:             Tier 2 only
mass-cell construction:  v0.14 nested sorted-mass 4 x 4 cells
D5 measurement:          v0.7/v0.13a DOP853 pipeline (rtol=atol=1e-12, max_step 0.02,
                         symplecticity/reciprocal gates 1e-4)
score:                   median(vf_0, vf_37, vf_90, vf_211)   (frames {0,37,90,211} deg)
all-four-frames rule:    success requires all four frames to integrate AND pass D5 gates;
                         any failing frame counts as attrition (no partial medians)
within-cell statistic:   pooled conditional rank AUC (Mann-Whitney, ties 0.5)
effect floor:            AUC_cond >= 0.55
permutation p gate:      p_perm <= 0.01
permutation:             100000 within-cell S/U label shuffle, seed 20260523
frame-spread gate:       median(frame_spread) <= 0.10 & p90 <= 0.50 clean; <= 0.20 & 0.75
                         warning; > 0.20 or > 0.75 blocked
```

**The effect floor is HELD at 0.55. It is not lowered to v0.16's 0.647 nor to any other
value.**

Forbidden:

```text
introducing any new feature, kernel, cutpoint, or mass partition
lowering the effect floor or the p gate
using the heterogeneity test to rescue a pooled failure
using a heterogeneity failure to demote or revise the v0.16 pass
selecting sampled rows by any vf, score, zone, or D5 output (draw on mass_cell +
  source stability label only)
including any v0.14, v0.15, OR v0.16 sampled orbit in the v0.17 primary sample
dropping frame-fragile rows from the primary statistic
tuning the reference vector, the rho bar, or the sign panel after seeing v0.17 rows
promoting to Tier-3 / independent confirmation or theorem-facing status
```

## Required Prior Receipts

The runner must assert all of the following before drawing a sample:

```text
v0.13 target inventory includes liao2021 as Tier 2; v0.13a leakage bounded 0.0;
v0.13b verdict = coarse_zone_rule_frame_stable_enough_to_test
v0.14 verdict = sample_transfer_undecidable_coverage
v0.15 verdict = stable_support_transfer_directional_weak
v0.16 verdict = tail_resolved_transfer_passes_clean
v0.14 + v0.15 + v0.16 target SHA all match this form's target SHA
v0.14, v0.15, AND v0.16 sample_frame.csv all exist (for the triple holdout)
v0.16 per_cell_rank.csv exists and supplies the reference vector
```

Any mismatch aborts.

## Pinned v0.16 Reference Vector (verified == receipt, 0.0 diff)

```text
mass_qA0_qB0  0.97203125     mass_qA3_qB0  0.47046875
mass_qA1_qB0  0.57515625     mass_qA3_qB1  0.69593750
mass_qA2_qB1  0.43156250     mass_qA3_qB2  0.63921875
                             mass_qA3_qB3  0.74453125
```

The heterogeneity Spearman test consumes the RANK ORDER of this vector; the exact values
are pinned for transparency and the report-only sign panel.

## Source-Support Census (fresh triple-holdout)

Signal-blind (source masses + stability labels only). Eligible = source minus the
v0.14 ∪ v0.15 ∪ v0.16 holdout (3520 orbits excluded). Independently reproduced with
v0.14's `build_mass_cells`:

| mass cell | eligible S | eligible U | disposition |
|---|---:|---:|---|
| `mass_qA0_qB0` | 322 | 7,744 | supported |
| `mass_qA1_qB0` | 488 | 7,578 | supported |
| `mass_qA2_qB1` | 2,584 | 5,481 | supported |
| `mass_qA3_qB0` | 2,434 | 5,632 | supported |
| `mass_qA3_qB1` | 2,998 | 5,067 | supported |
| `mass_qA3_qB2` | 2,261 | 5,804 | supported |
| `mass_qA3_qB3` | 958 | 7,107 | supported |

Locked rule: supported iff eligible S >= 80 AND eligible U >= 80 after the triple
exclusion. Expected supported cells: 7 (smallest eligible stable pool 322, 4x the draw).

## Sample Design

```text
domain:        the seven v0.16 supported cells
rows per cell: 160  (80 stable + 80 unstable)   -- v0.16 parity
expected rows: 1120
seed:          20260523
sampling:      uniform without replacement within (mass_cell, source_label)
holdout rule:  exclude every orbit_index in v0.14 AND v0.15 AND v0.16 sample_frame.csv,
               applied BEFORE the seeded draw
```

If any supported cell has fewer than 80 eligible stable or unstable rows after the triple
exclusion, the run aborts for lock review. The draw is outcome-balanced by source label
and blind to vf / score / zone (case-control; AUC is prevalence-invariant; no prevalence
estimate is licensed).

## D5 Measurement + Score

Per orbit, the frozen four-frame ensemble (identical to v0.16): integrate the
{0,37,90,211}-degree rotations of the CoM-centered state through the frozen D5 path,
compute vf at each, `score = median(vf_0, vf_37, vf_90, vf_211)`. An orbit is a success
only if all four frames integrate and pass both D5 gates; otherwise it is attrited.

Attrition policy (frozen): `<=0.05` clean; `0.05-0.10` with Wilson95_high `<=0.20`
warning; `>0.10` or Wilson95_high `>0.20` -> `blocked_by_attrition`.

## Primary Domain

A supported cell enters the primary statistic iff `N_success >= 120`, `S_success >= 50`,
`U_success >= 50`. Coverage gate: `primary_supported_cells >= 6` AND
`primary_success_rows >= 900`; else `blocked_by_coverage`.

## Primary Statistic -- Pooled Replication (the sole verdict gate)

Within each primary cell, Mann-Whitney `J_cell = #[score(S)>score(U)] + 0.5*ties`,
`D_cell = S*U`; pool `AUC_cond = sum J_cell / sum D_cell`. Direction pre-registered:
stable ranks above unstable. Binding null: 100000 within-cell S/U label shuffle
preserving each cell's score multiset and S/U counts, seed 20260523, one-sided,
`p_perm = (1 + #perm >= observed) / (1 + 100000)`.

Pooled replication passes iff `AUC_cond >= 0.55 AND p_perm <= 0.01` (the exact v0.16
floor, unchanged).

## Heterogeneity Replication Check (secondary; chooses between pooled-pass labels only)

Computed ONLY if the pooled replication passes. Compares the fresh v0.17 per-cell AUC
vector to the pinned v0.16 reference vector over the matched primary cells.

```text
statistic:   Spearman rho( v0.16 reference AUC_cell , v0.17 AUC_cell ) over matched cells
null:        EXACT permutation -- enumerate all k! pairings of the k matched cells
             (k=7 -> 5040), p_rho = #{rho_perm >= rho_obs} / k!
bar:         heterogeneity replicates iff p_rho <= 0.05
```

At n=7, `p_rho <= 0.05` corresponds to `rho >= ~0.64` (Sigma d^2 <= 20); a bare
`rho >= 0.60` is only ~`p 0.07` and is NOT used as the bar. The n=7 power limit is
disclosed: this is a low-power rank check, not a high-resolution map test.

Report-only per-cell sign panel (NON-gating): for each cell, report
`v0.17 AUC_cell`, `v0.16 reference`, direction-vs-0.5 match, `|AUC-0.5|`, and an
approximate per-cell AUC SE (`~0.032` near 0.5 at 80/80). Only the cells with
`|reference - 0.5| > ~0.10` (`qA0_qB0`, `qA3_qB3`, `qA3_qB1`, `qA3_qB2`) carry a real
directional prediction; `qA1_qB0` (+0.075), `qA2_qB1` (-0.068) sit ~2 SE out
(lean-but-uncertain) and `qA3_qB0` (-0.030, ~1 SE) is sampling-indeterminate. A sign flip
in a sub-2-SE cell is expected and is NOT a heterogeneity failure on its own.

## Frame-Stability Gate

Carried forward from v0.16: `frame_spread = max(vf)-min(vf)` over the four frames;
report median/p75/p90/p95/max overall + by cell + by fragile band; gate
`median<=0.10 & p90<=0.50` clean / `<=0.20 & <=0.75` warning / else
`blocked_by_frame_instability`.

## Verdict Tree

```text
blocked_by_receipt
  required prior receipts / SHA / triple-holdout exclusion missing or inconsistent
  OR supported-cell census deviates from the locked rule and cannot draw 80/80

blocked_by_attrition
  attrition_fraction > 0.10 OR attrition_wilson95_high > 0.20

blocked_by_frame_instability
  median(frame_spread) > 0.20 OR p90(frame_spread) > 0.75

blocked_by_coverage
  primary_supported_cells < 6 OR primary_success_rows < 900

tail_resolved_transfer_not_replicated
  AUC_cond <= 0.50 OR p_perm > 0.01      (clear null on fresh rows)

tail_resolved_transfer_replication_directional_weak
  0.50 < AUC_cond < 0.55 AND p_perm <= 0.01   (directional but below the v0.16 floor)

pooled_transfer_replicates_heterogeneity_unresolved
  AUC_cond >= 0.55 AND p_perm <= 0.01 AND attrition/frame not hard-blocked
  AND heterogeneity p_rho > 0.05

heterogeneous_transfer_replicates_clean
  AUC_cond >= 0.55 AND p_perm <= 0.01 AND attrition clean AND frame clean
  AND heterogeneity p_rho <= 0.05
```

A pooled pass under an attrition/frame WARNING band (not a hard block) appends a
`_with_warning` note to the pooled-pass label; it does not change the pass/fail.

## Required Outputs

```text
results/isotrophy/k-facet-v17-liao2021-heterogeneity/
  manifest.json                source_support_census.csv     sample_frame.csv
  sample_shard_XX_of_NN.csv     per_row_sample.csv            per_cell_rank.csv
  frame_dispersion.csv          heterogeneity_replication.json (rho, p_rho, matched
                                cells, v0.16 vs v0.17 per-cell AUC, sign panel + SE)
  anatomy_sidecar.csv (Option C, non-gating)
  permutation_summary.json      operator_commands.md
```

`manifest.json` must include: verdict, startedAt/completedAt, gitCommit, target_sha256,
v14/v15/v16 excluded counts, sample_seed, supported_cells, sample_rows_requested/success,
attrition (+Wilson), primary_supported_cells, primary_success_rows, frame_spread
median/p90, AUC_cond, J_cond, D_cond, p_perm, heterogeneity_rho, heterogeneity_p_rho,
target_tier=2.

## Anatomy Sidecar (Option C -- non-gating, existing v0.16 rows)

On the ALREADY-MEASURED v0.16 rows only (no new D5), report per cell -- especially the
two reversed cells `qA2_qB1`, `qA3_qB0`: mass coordinates, score quantiles, frame spread,
zone saturation, and any already-present period / energy / angular-momentum summaries.
Pure interpretation; it cannot set or change any v0.17 verdict.

## Claim Boundary

Even a clean v0.17 pass remains: Tier-2 / Li-Liao-lineage external; liao2021
stable-support region only; within-cell rank signal only; tail-resolved continuous score
only; not coarse-zone transfer; not full-catalog prevalence; not Tier-3 independent; not
theorem-facing.

```text
heterogeneous_transfer_replicates_clean -> allowed claim:
  "The tail-resolved velocity-fraction projection transfers on fresh held-out liao2021
   samples and its mass-cell heterogeneity is reproducible enough to report as structure,
   not merely v0.16 texture." (Tier-2, bounded as above.)

pooled_transfer_replicates_heterogeneity_unresolved -> allowed claim:
  "The pooled within-cell transfer replicates on fresh rows; local per-cell signs remain
   descriptive only (heterogeneous in the discovery sample, n=7 rank check inconclusive)."

tail_resolved_transfer_not_replicated / _directional_weak -> allowed claim:
  limited to the non-replication; does NOT retroactively demote v0.16 (a single
  fresh-sample miss bounds the claim's robustness, it does not erase the v0.16 receipt).

blocked_* -> claim limited to the blocking condition.
```

## Lock-In Statement

Committed before the v0.17 runner is written. Any change to the feature, frame set,
sample, triple-holdout rule, reference vector, decision bars, effect floor, permutation,
heterogeneity statistic, firewall, or claim boundary after runner execution is a
re-registration. The runner imports the frozen v0.14/v0.15/v0.16 D5 + frame + ensemble +
AUC + permutation symbols unchanged and adds only the triple holdout, the exact-permutation
Spearman heterogeneity check, and the non-gating anatomy sidecar.
