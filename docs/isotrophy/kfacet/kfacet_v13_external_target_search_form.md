# v0.13 External Target Search Form Draft

Status: **OPERATOR LOCK 2026-05-30.** No v0.13 runner has been written, no candidate
inventory has been computed, no external file has been downloaded for this chapter,
and no D5 velocity-fraction value has been scored. This document locks the
target-search procedure, eligibility gates, feasibility probes, selection rule,
verdict tree, and claim boundary for finding a clean external transfer target after
v0.12 blocked on attrition.

Reviewed for self-consistency, non-circularity, and integrity:

- **The anti-target-shopping firewall is airtight** across four layers: the Integrity
  Caveat forbidden-list, the D5 probe not recording/using `velocity_fraction`, the
  signal-blind selection rule, and the Next Chapter Boundary re-lock (a target lock
  is permission to DRAFT a transfer, never to run or interpret one).
- **Feasibility bars correctly encode the v0.12 lesson:** `attrition <= 0.10` AND
  `Wilson95 upper <= 0.20`, so a target whose CI still plausibly crosses the 0.20
  transfer-block gate cannot be locked. Internally consistent (at n=100 a 0.10 point
  gives Wilson upper ~0.17 < 0.20; 30-row probes are correctly too wide to lock). The
  96 h / 24 h-sharded runtime bars would have flagged supp-A (161 h) on speed alone.
- **Independence tiers prevent overclaiming:** Tier 0 (supp-A/B / repo-derived) cannot
  be selected; only Tier 1 -> "near-external," Tier 2/3 -> "independent external."
- Verdict tree complete (locked / near / none / access-blocked / operator-probe);
  claim boundary honestly scoped (no vf-signal claim, no v0.10b / K_facet revision).

Two surgical adjustments at lock (both strengthen the firewall or the eventual
transfer's validity; neither changes the chapter's substance):

1. Discovery-blindness clause added to the Integrity Caveat forbidden-list: candidate
   DISCOVERY, not just selection, must be signal-blind.
2. Stability-label commensurability added to the Next Chapter Boundary re-lock list:
   the external S/U label must denote the same linear (monodromy/Floquet) stability as
   supp-B, or the transfer is invalid.

Implementation note (already mandated by the D5 Feasibility Probe section; flagged
because the v0.12 attrition-probe did the opposite): the v0.13 feasibility probe
reuses `per_row_pipeline` but MUST NOT emit `velocity_fraction` or `zone_index` to any
receipt -- only success/blocked/sanity, runtime, period, mass key, failure reason.

Operational note: the inventory/discovery stage is an operator/agent-curated literature
and data-store search (it is not algorithmic); the pre-registration firewall binds that
search to be signal-blind per adjustment 1, and binds selection to be feature-blind per
the Target Selection Rule.

## Frame

The isotrophy reopening now has a four-step ledger:

```text
v0.10a: marginal ordered trend registered in-sample
v0.10b: mass-marginal held-out predictor failed
v0.11:  within-m3 conditional velocity-fraction rank passed on supp-B
v0.12:  same-paper supp-A transfer blocked by D5 attrition
```

v0.12 did not falsify the velocity-fraction hypothesis. It showed that the
frozen v0.7 D5 measurement cannot support a supplementary-A transfer:

```text
uniform attrition_fraction = 0.3433
Wilson 95% CI              = [0.2919, 0.3987]
locked block threshold     = 0.20
verdict                    = external_transfer_blocked_by_attrition
```

v0.13 is therefore not a rescue of v0.12. It is a source-selection chapter:

> Find and lock an external target catalog where the frozen v0.11 rule could be
> tested without leakage and without a v0.12-style D5 attrition wall.

If v0.13 succeeds, it selects a target for a later transfer chapter. It does not
itself test the velocity-fraction signal.

## Integrity Caveat

The danger in a target search is silent target shopping. v0.13 prevents that by
separating three activities:

```text
allowed in v0.13:
  source discovery
  schema/profile checks
  overlap/leakage checks
  D5 feasibility probes that report only runtime and attrition
  target selection by pre-feature source quality and feasibility

forbidden in v0.13:
  inspecting velocity_fraction zone counts by stability
  computing AUC_cond, J_cond, chi-squared, J-T, or any S/U-vs-feature statistic
  selecting a target because it appears likely to pass the v0.11 rule
  searching for, prioritizing, or excluding candidate sources by any known or
    suspected velocity_fraction-vs-stability behavior -- discovery itself must be
    signal-blind (you may search for "3-body periodic catalog with linear-stability
    labels"; you may not search for catalogs known to exhibit the vf/stability link)
  changing v0.11 cutpoints {0.25, 0.50}
  relaxing the frozen D5 numerical controls
  restricting a target post-hoc to the rows that integrate cleanly
```

Any candidate touched by v0.13 must carry a manifest that records what was
observed and what was deliberately not observed.

## Frozen Transfer Rule

The eventual transfer rule remains v0.11:

```text
zone_index:
  positional-dominant = 0   if velocity_fraction < 0.25
  mixed               = 1   if 0.25 <= velocity_fraction < 0.50
  velocity-heavy      = 2   if velocity_fraction >= 0.50
```

`m3` or the target's exact mass-tuple key is used only as a conditioning
stratum. No target is eligible for a v0.11-style transfer unless it supports
within-stratum S/U comparisons.

## Candidate Intake

Candidate sources may come from:

```text
local repo files
paper supplementary files
public project pages
repository releases
Zenodo / Figshare / institutional data stores
author-provided catalog mirrors
```

Every candidate gets a stable slug:

```text
<source_family>__<table_name>__<access_date>
```

The intake record must include:

```text
slug
source URL or local path
access date
download hash, if downloaded
license / citation note, if present
source family and authors
relationship to supp-B and supp-A
orbit type and dimensionality
body count
potential / equations
mass parameters
initial-condition fields
period field
stability-label definition
row count
label counts, if labels are provided in source metadata
known transformations needed for integration
```

Downloading or copying a candidate file is allowed only into a clearly named
staging path under:

```text
docs/isotrophy/external_targets/_staging/
```

No staged target file is promoted into tracked docs until it passes the source
profile gate and the operator approves retaining it.

## Hard Eligibility Gates

A candidate is eligible for v0.13 feasibility work only if it passes all hard
gates before any D5 computation:

```text
three-body Newtonian or explicitly transformable to the same Newtonian substrate
periodic-orbit initial conditions sufficient to integrate the orbit
period or closure time provided
mass tuple or mass parameter provided per row
linear stability label or equivalent stable/unstable label provided by source
at least 200 rows before overlap quarantine
at least 5 candidate conditioning strata before D5 attrition
at least 5 S and 5 U rows in at least 5 candidate strata before D5 attrition
not supplementary-A
not supplementary-B
not generated from this repo's v0.3-v0.12 outputs
source can be cited and re-fetched or preserved with hash
```

Report-only candidates may be logged if they fail one gate, but they cannot be
selected for the next transfer chapter.

## Independence Tiers

Independence is scored before feasibility probes:

```text
Tier 3: independent source family, independent search or publication, independent
        catalog construction

Tier 2: same broad research lineage or same public site, but different paper,
        different table, and not derived from supp-A/supp-B

Tier 1: same paper/source family but different supplement or representation

Tier 0: supp-A, supp-B, duplicate, or repo-derived artifact
```

Tier 0 cannot be selected. Tier 1 can be inventoried for completeness, but v0.13
must prefer any viable Tier 2 or Tier 3 target over a viable Tier 1 target. If
only Tier 1 targets are viable, the verdict must say "near-external target
locked," not "independent external target locked."

## Overlap Quarantine

Every candidate is compared against prior isotrophy evidence sources:

```text
docs/isotrophy/supplementary-B_piano-init-condit-3d.txt
docs/isotrophy/supplementary-A_periodic-3d_mirror.txt
results/isotrophy/k-facet-v03-freeze-supplementary-b-comparison/
results/isotrophy/k-facet-v11-m3-conditional-vf-rank/
results/isotrophy/k-facet-v12-external-frozen-transfer/
```

Use the strongest identity key the schema supports:

```text
preferred: same mass tuple and max_abs_delta(normalized IC fields, T) <= 1e-9
fallback:  source-native orbit identifiers plus invariants, marked weaker
```

If the candidate uses the same mirrored ansatz, also test the ansatz reflection
image:

```text
(z0, vx, vy, vz) -> (-z0, vx, vy, -vz) at same mass tuple and T
```

Overlap policy:

```text
overlap rows are excluded from feasibility row counts
candidate is leakage-blocked if overlap_fraction > 0.05
candidate is report-only if identity matching is too weak to bound leakage
```

## D5 Feasibility Probe

The D5 measurement is frozen to v0.7/v0.12:

```text
integrator:           DOP853
rtol:                 1e-12
atol:                 1e-12
max_step_fraction:    0.02
symplecticity_gate:   1e-4
reciprocal_pair_gate: 1e-4
```

The feasibility probe may run the D5 pipeline, but it must not expose or use the
computed `velocity_fraction` values. It records only:

```text
success / integration_blocked / sanity_failed
runtime seconds
period
mass key
failure reason
```

It must not tabulate success, failure, or runtime by stability label unless the
candidate is already rejected; the selected-target decision must be based on
measurement feasibility, not S/U-feature relationships.

### Probe Sampling

For each candidate that passes hard eligibility:

```text
seed:              20260523
probe rows:        100 uniform rows after overlap quarantine
minimum probe:     30 rows if the candidate has fewer than 100 eligible rows
stop condition:    stage, do not run inline, if projected probe exceeds 10 minutes
```

If a 100-row probe would exceed the inline budget, v0.13 may run a capped
12-row rate probe and stage the full 100-row feasibility command for the
operator. The 12-row probe cannot lock a target by itself.

### Feasibility Bars

A candidate is D5-viable only if the 100-row uniform probe satisfies:

```text
attrition_fraction <= 0.10
Wilson 95% upper bound <= 0.20
projected full transfer wall-clock <= 96 hours single-threaded
projected full transfer wall-clock <= 24 hours under the proposed shard count
```

`attrition_fraction` counts both integration-blocked rows and sanity-failed rows.
The Wilson upper-bound rule is the direct v0.12 lesson: do not lock a target whose
uncertainty still plausibly crosses the 0.20 transfer-block gate.

## Target Selection Rule

After all eligible candidates are profiled and feasibility-probed, select the
highest-ranked viable target by this ordered rule:

1. Highest independence tier.
2. Lowest Wilson 95% upper bound on attrition.
3. Largest projected primary-domain row count.
4. Most stable source/citation and easiest reproducibility.
5. Shortest projected full transfer runtime.

No S/U-vs-feature statistic may enter this ordering.

If two candidates remain tied, choose the one with the larger number of
conditioning strata. If still tied, choose the lexicographically first slug and
record the tie.

## Verdict Tree

```text
external_target_locked
  at least one Tier 2 or Tier 3 candidate passes hard gates, leakage gate, and
  D5 feasibility bars; the highest-ranked target is locked for a later transfer
  chapter

near_external_target_locked
  no Tier 2 or Tier 3 candidate is viable, but a Tier 1 non-duplicate candidate
  passes all gates; it may support a near-external chapter only

no_viable_external_target_found
  inventory completed and no candidate passes the hard gates plus D5 feasibility
  bars

target_search_blocked_by_access
  candidate sources could not be accessed or preserved well enough to make a
  reproducible inventory

target_search_operator_probe_needed
  a candidate passes source gates, but the required 100-row feasibility probe
  exceeds the inline budget and has not yet been run by the operator
```

## Required Outputs

Expected output directory:

```text
results/isotrophy/k-facet-v13-external-target-search/
```

Required files:

```text
manifest.json
target_inventory.csv
source_profiles.json
overlap_audits.csv
feasibility_probes.csv
rejected_targets.csv
selection_readback.md
operator_commands.md
```

`manifest.json` must include:

```text
verdict
startedAt
completedAt
gitCommit
candidate_count
eligible_candidate_count
viable_candidate_count
selected_slug, if any
selected_independence_tier, if any
selected_attrition_fraction, if any
selected_wilson95_high, if any
selected_projected_primary_rows, if any
selected_projected_runtime_hours, if any
```

## Command Shape After Implementation

The implementation should be additive:

```text
scripts/v13_external_target_search.py
```

Expected command shapes:

```powershell
python scripts/v13_external_target_search.py inventory `
  --out results/isotrophy/k-facet-v13-external-target-search

python scripts/v13_external_target_search.py profile `
  --inventory results/isotrophy/k-facet-v13-external-target-search/target_inventory.csv `
  --out results/isotrophy/k-facet-v13-external-target-search

python scripts/v13_external_target_search.py probe `
  --inventory results/isotrophy/k-facet-v13-external-target-search/target_inventory.csv `
  --probe-rows 100 `
  --seed 20260523 `
  --rtol 1e-12 `
  --atol 1e-12 `
  --out results/isotrophy/k-facet-v13-external-target-search

python scripts/v13_external_target_search.py select `
  --out results/isotrophy/k-facet-v13-external-target-search
```

The inventory/profile stages are cheap and may run inline if implemented. The
probe stage must obey the repository's long-run rule: if expected wall-clock
exceeds about 10 minutes, stage the exact PowerShell command for the operator and
stop.

## Claim Boundary

If v0.13 locks a target, the allowed claim is:

> A pre-feature search identified a reproducible external target that is
> independent enough, low-leakage enough, and numerically tractable enough to
> support a later frozen v0.11 transfer test.

Forbidden upgrades:

```text
does not confirm the v0.11 velocity-fraction signal
does not overturn v0.10b
does not revise the v0.3h K_facet structural null
does not make isotrophy theorem-facing
does not score any S/U-vs-velocity-fraction relationship
does not permit relaxing D5 precision in the later transfer chapter
```

If v0.13 finds no target, the allowed claim is:

> The current external-transfer path remains unconfirmed because no reproducible,
> low-leakage, D5-tractable target was found under the locked search rules.

If v0.13 blocks on operator probe, the allowed claim is:

> A candidate target remains possible, but target lock is pending the required
> 100-row D5 feasibility probe.

## Next Chapter Boundary

Only after `external_target_locked` or `near_external_target_locked` may a later
chapter draft a frozen transfer statistic. That later chapter must re-lock:

```text
selected target path and hash
overlap exclusion list
primary conditioning strata
stability-label commensurability with supp-B's linear (monodromy/Floquet) stability
  convention -- the transfer is valid only if the external S/U label denotes the same
  kind of stability; a non-commensurable label blocks the transfer
attrition policy
permutation null
effect floor
claim boundary matching the target's independence tier
```

v0.13 target lock is permission to draft the transfer test, not permission to run
or interpret it.
