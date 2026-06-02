# v0.17 liao2021 Heterogeneity Scope Draft

Status: **SCOPE DRAFT ONLY -- not locked.** No v0.17 runner, sample, D5
measurement, transfer statistic, or claim update is authorized by this memo.

## Why v0.17 Exists

v0.16 landed the first clean Tier-2 external transfer:

```text
tail-resolved 4-frame ensemble score
AUC_cond = 0.6470
p_perm   = 1.0e-5
rows     = 1120 fresh double-holdout liao2021 stable-support rows
```

It also exposed the next live question. The signal is real at the pooled
within-cell level, but it is not a uniform per-cell law:

```text
in-direction cells: qA0_qB0, qA1_qB0, qA3_qB1, qA3_qB2, qA3_qB3
reversed cells:     qA2_qB1, qA3_qB0
range:              AUC_cell 0.432 -> 0.972
```

The v0.16 pass is robust to dropping the strongest cell, so this is not a
single-cell artifact. But the claim profile now needs a sharper answer:

> Does the tail-resolved velocity-fraction signal replicate as a pooled external
> transfer with a stable mass-cell heterogeneity pattern, or was v0.16 a pooled
> transfer whose local signs were sample texture?

That is the v0.17 problem.

The exact v0.16 reference vector, if Option A is promoted to a lock form, is:

```text
mass_qA0_qB0  0.97203125
mass_qA1_qB0  0.57515625
mass_qA2_qB1  0.43156250
mass_qA3_qB0  0.47046875
mass_qA3_qB1  0.69593750
mass_qA3_qB2  0.63921875
mass_qA3_qB3  0.74453125
```

## Scope Principle

v0.17 should not introduce a new feature. The v0.16 feature is already the
first successful external instrument:

```text
score = median(vf_0, vf_37, vf_90, vf_211)
```

The honest next move is replication and localization, not another feature
search. New kernels, new cutpoints, and stability-peeking mass partitions should
stay out of v0.17.

## Candidate Designs

### Option A -- Fresh Heterogeneity Replication (Recommended)

Draw a new liao2021 supported-cell sample, excluding all v0.14, v0.15, and v0.16
orbits. Reuse the v0.16 measurement stack and score exactly.

Proposed primary sample:

```text
domain:       the same seven v0.16 primary supported mass cells
draw:         80 stable + 80 unstable per cell
exclusions:   v0.14, v0.15, v0.16 sampled orbit ids
rows:         1120 if all cells retain support
feature:      unchanged v0.16 tail-resolved ensemble score
statistic:    within-cell Mann-Whitney AUC_cond + 100k within-cell permutation
```

Primary questions:

1. Does the pooled external transfer replicate at the locked v0.16 floor?
2. Does the v0.16 per-cell AUC pattern replicate well enough to describe
   mass-cell heterogeneity as a result rather than a one-sample texture?

Proposed gates for lock review:

```text
pooled replicate:
  AUC_cond >= 0.55
  AND p_perm <= 0.01

heterogeneity replicate:
  pooled replicate passes
  AND Spearman rho(v0.16 AUC_cell, v0.17 AUC_cell) >= 0.60
  AND at least 5/7 cells preserve their v0.16 direction relative to 0.5
```

Possible verdicts:

```text
heterogeneous_transfer_replicates_clean
  pooled replicate passes
  AND heterogeneity replicate passes
  AND attrition/frame gates clean

pooled_transfer_replicates_heterogeneity_unresolved
  pooled replicate passes
  AND heterogeneity replicate fails or is weak

tail_resolved_transfer_not_replicated
  AUC_cond < 0.55 OR p_perm > 0.01

blocked_by_coverage_or_receipt
  any required supported cell cannot supply the fresh rows, or frozen receipts fail
```

Interpretation if Option A passes: v0.16 was not a one-off external pass; the
tail-resolved projection transfers on fresh liao2021 rows, and the mass-cell
heterogeneity is reproducible enough to become part of the claim language.

Interpretation if pooled passes but heterogeneity fails: the transfer survives,
but local sign claims remain descriptive only. The public posture should say
"pooled within-cell transfer, heterogeneous in the discovery sample" rather than
"reproducible mass-cell map."

### Option B -- Denser Per-Cell Replication

Same design as Option A, but increase to 120 or 160 stable + 120 or 160 unstable
per cell if the post-exclusion support allows it.

This improves per-cell precision but roughly scales runtime by 1.5x to 2x. It
is useful only if v0.17 is intended to publish a per-cell map rather than simply
test whether the v0.16 heterogeneity repeats.

Recommendation: do not choose this first unless the source support census shows
large clean reserves in all seven cells and the operator wants a long-budget
chapter.

### Option C -- Existing-Row Anatomy Audit

Analyze only v0.16 rows and source metadata:

```text
mass coordinates
score quantiles
frame spread
zone saturation
period / energy / angular-momentum summaries if already present
```

This is cheap and useful for interpretation, especially around the two reversed
cells, but it is not a confirmation chapter. It should be labeled an anatomy
sidecar, not a v0.17 transfer result.

## Recommended v0.17 Shape

Use Option A as the lock candidate:

```text
v0.17 title: liao2021 Tail-Resolved Heterogeneity Replication
role:        fresh-row replication of v0.16 pooled transfer and per-cell pattern
feature:     frozen v0.16 score
domain:      seven v0.16 supported cells
sample:      80/80 per cell, excluding v0.14/v0.15/v0.16 rows
runtime:     same order as v0.16; stage for operator/sharded runner, do not run inline
```

The v0.16 run recorded `startedAt = 2026-06-02T05:17:58+00:00` and
`completedAt = 2026-06-02T15:13:21+00:00`, so an 80/80 v0.17 parity run should
be treated as a long-budget staged run, not an inline agent run.

The only new statistical object would be the cell-pattern replication check:
the v0.16 seven-cell AUC vector becomes the pre-registered reference vector,
and v0.17 tests whether a fresh sample preserves its rank/order and direction
well enough to make heterogeneity claimable.

## Integrity Guardrails

- v0.17 may use v0.16 to define the replication target because it is explicitly
  a replication of a discovered pattern, not a discovery of a new law.
- The v0.16 feature, frame set, D5 gates, attrition accounting, AUC machinery,
  and permutation seed should remain frozen unless the lock form names a reason.
- No new feature may be promoted inside v0.17. If a new mass-conditioned kernel
  or transformed score is desired, it should be v0.18 or a separate feature
  chapter.
- The heterogeneity gate must not rescue a pooled failure. If the pooled transfer
  fails, the chapter records `tail_resolved_transfer_not_replicated` even if a few
  cells look interesting.
- A heterogeneity failure must not demote the existing v0.16 pass. It only limits
  the local-cell claim language.

## Claim Boundaries

Even a clean v0.17 pass would still be:

```text
Tier-2 / Li-Liao-lineage external
liao2021 stable-support region only
within-cell rank signal only
tail-resolved continuous score only
not coarse-zone transfer
not full-catalog prevalence
not Tier-3 independent
not theorem-facing
```

The maximum honest upgrade would be:

> The tail-resolved velocity-fraction projection transfers on fresh held-out
> liao2021 samples and its mass-cell heterogeneity is reproducible enough to be
> reported as structure, not merely as v0.16 texture.

## Open Decisions Before Lock

1. **Sample size:** 80/80 per cell for v0.16-parity, or a larger long-budget
   precision sample.
2. **Heterogeneity bar:** whether Spearman `rho >= 0.60` plus 5/7 sign concordance
   is the right balance, or whether the sign pattern should be report-only.
3. **Reference vector:** whether to lock the exact v0.16 per-cell AUC vector as the
   replication target, or only the five-positive / two-reversed sign pattern.
4. **Anatomy sidecar:** whether to add a cheap source-metadata readback alongside
   the fresh replication, explicitly non-gating.
