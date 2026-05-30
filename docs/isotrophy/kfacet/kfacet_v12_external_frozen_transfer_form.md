# v0.12 External Frozen Transfer Form Lock

Status: **OPERATOR LOCK 2026-05-30.** No v0.12 runner has been written, no v0.12
command has been run, and no supplementary-A row has been scored for this chapter
at lock time. This document locks the frozen transfer domain, overlap quarantine,
attrition policy, statistic, verdict tree, diagnostics, and claim boundary before
any v0.12 scoring.

Reviewed for self-consistency, non-circularity, and integrity:

- **Strongest non-circularity in the program — breaks the DOMAIN.** The v0.11 rule
  (cutpoints {0.25,0.50}, zone order, S-up-zone direction) is frozen and pinned by
  asserting the v0.11 receipt before any scoring; supp-A is genuine out-of-sample.
  The disallowed-feature list is comprehensive (no cutpoint tuning, no classifier,
  no raw vf, no period/energy/|L|/branch/z0/eigenvalue features).
- **Convention check (done at lock):** supp-A and supp-B use the byte-identical
  mirrored-velocity ansatz and both canonicalize `z0 > 0`, so a shared orbit lands
  at identical ICs and the exact 1e-9 quarantine is the correct identity key. The
  feature (Floquet vf) is well-defined on supp-A (same orbit family, same v0.7 D5
  monodromy/gauge conventions) — this is a true transfer, not a domain mismatch.
- **Effect floor (AUC_cond >= 0.55) defeats the large-N significance trap:** on a
  10,059-row table a tiny AUC could clear a permutation p with no real effect; the
  0.55 floor + the `directional_weak` branch (0.50 < AUC < 0.55, p <= 0.01) keep
  that from masquerading as a transfer while preserving the information.
- **Null is appropriate:** stratified within-`m3` label permutation (preserve
  S_m/U_m), n=100000, seed 20260523 — the same null v0.11 used as its sanity
  sidecar, made binding here to avoid depending on a novel large-support exact
  kernel; exact enumeration kept as an optional cheap sidecar (support <= 250000),
  never binding.
- **Attrition is evidence, not housekeeping:** <=5% clean / <=20% warning (PASS
  reworded to "analyzable-domain transfer") / >20% blocked, plus floor blocks
  (primary < 500 rows or < 5 both-class strata). Inherits the v0.7 D5 integration
  lessons; smoke/projection is required before any long run and the agent may not
  run the full pass inline.
- **Claim boundary is the most carefully bounded in the program:** a PASS is a
  same-paper external-supplement transfer only — not universe-wide, not
  independent-catalog, no v0.10b overturn, no global predictor, no K_facet revision,
  no theorem-facing promotion, nothing about excluded rows.

Adjustment made at lock (one): hardened the overlap quarantine with an
ansatz-reflection-image check — `(z0,vx,vy,vz) -> (-z0,vx,vy,-vz)` at the same
`(m3,T)` — folded into the combined overlap fraction and the same `> 0.05` leakage
abort, so the external claim is robust to the two supplements having picked opposite
vz-sign representatives of a shared orbit (referee-proof de-duplication; expected
near-zero given the shared `z0 > 0` canonicalization).

## Frame

v0.11 registered the frozen velocity-fraction zone order as a conditional
within-`m3` rank signal on the analyzable supplementary-B piano-trio table:

```text
AUC_cond: 0.6782549421
exact p:  2.046197217e-7
verdict:  m3_conditional_vf_rank_passes
```

That was still an in-sample conditional result. v0.12 asks the next cleaner
evidence-upgrade question:

> If the v0.11 rule is frozen exactly, does it transfer to a different
> Li-Liao catalog table that was not used to discover, tune, or score v0.11?

The proposed target is supplementary-A:

```text
docs/isotrophy/supplementary-A_periodic-3d_mirror.txt
```

This is a 10,059-row 3D periodic-orbit catalog with the same compact mirrored
ansatz and S/U stability labels. It is external to the supplementary-B v0.11
table, but it is still a same-paper / same-source-family transfer. A PASS would
therefore upgrade the evidence from "supp-B in-sample conditional signal" to
"same-paper external supplement transfer." It would not be a universe-wide
validation and would not license a claim over unrelated astronomical catalogs.

## Integrity Caveat

The rule is frozen because v0.11 has already looked at supplementary-B:

```text
zone_index:
  positional-dominant = 0   if velocity_fraction < 0.25
  mixed               = 1   if 0.25 <= velocity_fraction < 0.50
  velocity-heavy      = 2   if velocity_fraction >= 0.50
```

v0.12 may not tune those cutpoints, fit mass-specific thresholds, train a
classifier, use raw `velocity_fraction` as a continuous score, add period,
energy, angular momentum, branch labels, `z0`, eigenvalue magnitudes, spectral
radius, unit-circle counts, or any other orbital feature to the primary score.

`m3` is allowed only as a conditioning stratum, exactly as in v0.11. The
question is not "can we build the best predictor?" It is:

```text
Within fixed m3 strata in a new table, does the frozen v0.11 zone order rank
stable rows above unstable rows?
```

## Frozen Inputs

Discovery / rule source:

```text
docs/isotrophy/supplementary-B_piano-init-condit-3d.txt
results/isotrophy/k-facet-v11-m3-conditional-vf-rank/manifest.json
```

Primary transfer target:

```text
docs/isotrophy/supplementary-A_periodic-3d_mirror.txt
```

The runner must assert the v0.11 receipt before doing any v0.12 scoring:

```text
v0.11 verdict:       m3_conditional_vf_rank_passes
v0.11 AUC_cond:      0.6782549421
v0.11 exact p:       2.046197217e-7
v0.11 zone rule:     {0.25, 0.50}
```

Any mismatch aborts. No v0.12 branch may reinterpret v0.11.

## Source Profile Gate

Before integration or scoring, parse the target and write a source profile:

```text
expected data rows:       10059
required numeric columns: m3, z0, vx, vy, vz, T
required label column:    stability in {S, U}
fixed ansatz:             m1=m2=1, m3 variable, mirrored velocities
```

Abort if the target does not parse, if the data row count is not 10,059, if any
required numeric column is missing, or if any stability label is outside `{S,U}`.

The source profile must report:

```text
row count
m3 strata
S/U counts by m3
period min/median/max by m3
candidate primary strata before overlap and attrition gates
```

## Overlap Quarantine

v0.12 is a transfer test, so rows already consumed by prior isotrophy evidence
must not enter the primary statistic.

Quarantine sources:

```text
docs/isotrophy/supplementary-B_piano-init-condit-3d.txt
results/isotrophy/m3eq1-sigma3-precondition-fixed-inverse-orientation-25/
results/isotrophy/k-facet-v03-freeze-supplementary-b-comparison/
results/isotrophy/k-facet-v11-m3-conditional-vf-rank/
```

Row identity is numeric, not by orbit label. A target row is marked overlap if:

```text
same m3 and max_abs_delta(z0, vx, vy, vz, T) <= 1e-9
```

Labels, row order, and `O_index` are not used for matching.

### Ansatz reflection image (added at lock)

Both supplements share the **identical** mirrored-velocity ansatz —
`r1=(-1,0,0), r2=(1,0,0), r3=(0,0,z0)`; `v1=(vx,vy,vz), v2=(vx,vy,-vz),
v3=(-(m1+m2)vx/m3, -(m1+m2)vy/m3, 0)`; `m1=m2=1` — and both canonicalize to
`z0 > 0`. A shared physical orbit therefore lands at identical numeric ICs and
the exact rule above catches it. The one residual leakage channel is the
ansatz's own z-reflection symmetry, under which the same physical orbit maps to

```text
(z0, vx, vy, vz)  ->  (-z0, vx, vy, -vz)     at the same (m3, T)
```

If the two supplements ever picked opposite vz-sign representatives for the same
orbit, the exact match would miss it. To make the external claim robust to that,
a target row is ALSO marked overlap if a discovery/quarantine row matches its
reflection image within `1e-9` (same `m3`, same `T`). Exact and reflection-image
matches are tallied separately in `overlap_audit.csv`; the **combined** overlap
fraction feeds the same `> 0.05 -> external_transfer_blocked_by_leakage` gate. A
near-zero reflection-image count (expected, since both supplements canonicalize
`z0 > 0`) is itself a clean-separation receipt rather than noise.

Overlap policy:

```text
overlap rows are excluded from the primary statistic
overlap rows are written to report_only_rows.csv
if overlap_fraction_of_candidate_primary > 0.05, abort as leakage-prone
```

The strict `m3=1` sigma3 rows from supplementary-A are also excluded from the
primary transfer statistic and reported separately. They have already carried
K_facet structural-null work and should not quietly double-count as new
transfer evidence.

## Velocity-Fraction Receipt

The v0.12 feature is the v0.7a D5 velocity-fraction receipt, with the same
operational meaning:

```text
velocity_fraction = fraction of the selected Floquet direction's physical
                    displacement norm carried by velocity coordinates
```

Selection rule:

```text
use the largest-real-part nontrivial Floquet direction under the existing v0.7
monodromy / gauge conventions
```

Numerical controls inherit v0.7a/v0.7a-prime:

```text
integrator:          DOP853
rtol:                1e-12
atol:                1e-12
max_step_fraction:   0.02
symplecticity gate:  1e-4
reciprocal-pair gate: 1e-4
```

Rows that fail integration or sanity gates are not silently dropped. They are
counted in attrition and written with a machine-readable failure reason.

## Probe Before Full Run

The full supplementary-A velocity-fraction pass is expected to exceed the
agent inline-run budget. v0.12 therefore has a required probe stage.

Probe command shape, after implementation:

```powershell
python scripts/v12_external_frozen_transfer.py profile `
  --target docs/isotrophy/supplementary-A_periodic-3d_mirror.txt `
  --discovery docs/isotrophy/supplementary-B_piano-init-condit-3d.txt `
  --out results/isotrophy/k-facet-v12-external-frozen-transfer

python scripts/v12_external_frozen_transfer.py smoke `
  --target docs/isotrophy/supplementary-A_periodic-3d_mirror.txt `
  --discovery docs/isotrophy/supplementary-B_piano-init-condit-3d.txt `
  --out results/isotrophy/k-facet-v12-external-frozen-transfer `
  --smoke-rows 30 `
  --rtol 1e-12 `
  --atol 1e-12
```

The smoke selection is deterministic:

```text
choose up to 5 rows from each of 6 m3 bins spanning the target's m3 range
within each selected m3 bin, choose period quantiles {min, q25, median, q75, max}
after overlap quarantine
cap total smoke rows at 30
```

The smoke writes:

```text
seconds_per_row
integration_success_fraction
sanity_fail_fraction
projected_full_wall_clock
projected_analyzable_rows
```

If projected full wall-clock exceeds about 10 minutes, the full transfer command
must be staged for the operator or a long-budget runner. The agent must not run
the full pass inline.

## Probe Results (steps 1-4, executed 2026-05-30)

Receipt: `results/isotrophy/k-facet-v12-external-frozen-transfer/`
(`manifest.json`, `source_profile.csv`, `overlap_audit.csv`, `integration_smoke.csv`).
Runner `scripts/v12_external_frozen_transfer.py` reuses the v0.7a D5
`per_row_pipeline` verbatim -- identical monodromy / gauge / gate conventions, no
reimplementation.

**Profile (integration-free):**

```text
rows:                       10,059   (source gate PASS; all labels in {S,U})
m3 strata:                  20 over [0.1, 2.0]   S = 1,996   U = 8,063
overlap vs supp-B:          exact 235 / reflection 0  =  2.75%  (< 5% leakage abort)
  -> reflection 0 = clean-separation receipt (both supplements canonicalize z0>0)
strict m3=1 excluded:       1,504   (conservative "all m3=1"; see decision A)
nonoverlap candidate rows:  8,320    (>> 500 floor)
candidate primary strata:   17 pre-vf (N>=30 / S>=5 / U>=5)   (>> 5 floor)
```

**Smoke (30 rows, reused D5 pipeline, 34.9 min wall):**

```text
seconds_per_row:            69.77   (median 67, max 214 -- long-period dominated)
integration success:        22/30 = 73.3%   (6 blocked + 2 sanity-fail)
projected full wall-clock:  161 h  ~= 6.7 days single-threaded (8,320 rows)
projected analyzable rows:  ~6,101
```

**Two findings that gate the full run:**

1. **Cost: 6.7 days single-threaded.** The vf pass is row-independent, so the full
   run must be SHARDED (Phase-15 pattern; the v0.7a runner already carries R2.C
   append-per-row resume). 8x concurrent -> ~20 h; 16x -> ~10 h.

2. **Attrition risk vs the locked >0.20 BLOCK gate.** Smoke attrition = 26.7%, and
   it is period-biased: 7 of 8 failures are long-period (T = 38-201) hitting the
   integrator step-size wall (6 blocked) or the symplecticity gate (2 sanity). The
   smoke is period-STRATIFIED by design ({min,q25,median,q75,max} per m3 bin), so it
   over-weights the hard long-period tail and OVER-ESTIMATES the uniform-catalog
   `attrition_fraction` the Attrition Policy actually scores. The true full-run rate
   is uncertain and could land in either the 0.05-0.20 "analyzable-domain transfer"
   band or the >0.20 "external_transfer_blocked_by_attrition" band. An UNBIASED
   uniform-random attrition probe (not period-stratified) is the cheap de-risk
   before committing 6.7 days: if it reads >0.20, the locked policy pre-blocks the
   full run and the compute is saved.

**Staging (step 6) -- the full transfer pass is NOT run inline.** Three operator
decisions precede authorizing the full run (step 7):

```text
A. m3=1 exclusion: conservative all-m3=1 (drops 1,504; safe over-exclusion) vs
   strict-sigma3-only (recovers ~1,479 rows; needs a sigma3 pass on supp-A m3=1).
B. Attrition de-risk: a uniform-random ~150-row attrition probe for an unbiased
   attrition_fraction before the full run (recommended).
C. Shard count for the full vf pass (8x ~20 h / 16x ~10 h), then build step 7
   (stratified-permutation transfer statistic + verdict) on the analyzable output.
```

### Attrition de-risk probe result (decision B, executed 2026-05-30)

A sharded uniform-random SRS (NOT period-stratified) of 300 candidate rows --
6 shards x 50, seed 20260523, reusing `per_row_pipeline`; the merge reconciles the
shards to the exact deterministic sample (`reconcile_abort = None`).

```text
uniform attrition_fraction = 0.3433   (103/300: 66 blocked + 37 sanity-fail)
Wilson 95% CI              = [0.2919, 0.3987]   -- entire interval ABOVE 0.20
vs locked 0.20 BLOCK gate  = decisive_above (CI low 0.292 >> 0.20; no straddle)
verdict implication        = external_transfer_blocked_by_attrition
```

Per-m3 the loss is SYSTEMIC, not localized (so no domain restriction rescues it,
and restriction would be a disallowed post-hoc move regardless): m3=0.4 is 0% but
m3 = 0.5-1.8 run 23-67%, rising with m3 (0.7 -> 35%, 0.8 -> 41%, 1.2 -> 44%,
1.3 -> 53%, 1.6 -> 67%).

**Honest correction to the step-5 readback.** It predicted the period-stratified
smoke (26.7%) would OVER-estimate the uniform rate. It UNDER-estimated: uniform is
34.3%, higher. The smoke forces each bin's {min,q25} short-period rows (which
succeed) into the sample, so it under-represents the catalog's long-period bulk.
The directional guess was wrong; the probe was decisive regardless -- which is the
whole reason we measured the unbiased rate instead of assuming it.

**Verdict: the full transfer run is policy-BLOCKED and is NOT run.** The locked D5
measurement (DOP853, rtol/atol 1e-12, max_step 0.02 T) cannot integrate ~1/3 of
supp-A's candidate orbits at the frozen precision (long-period / high-m3 step-size
wall + symplecticity failures). Per the Attrition Policy this is
`external_transfer_blocked_by_attrition`: NOT a failure of the velocity-fraction
hypothesis, but a statement that the frozen measurement cannot support the external
transfer question on this target. Both rescue paths are closed by the form ---
relaxing precision breaks the "identical frozen v0.7 measurement" requirement (it
would no longer be the same feature), and restricting to the short-period analyzable
subset is a disallowed post-hoc fail-rescue that would bias the transfer. The ~2.5 h
probe saved the projected 6.7-day full run.

The supp-B conditional-positive (v0.10a in-sample trend + v0.11 within-m3 rank,
exact p = 2.05e-7) stands unchanged; it simply does not receive an external-catalog
confirmation from supp-A, because supp-A is numerically intractable for the frozen
D5 integrator. A genuinely external confirmation would need either a different
target catalog whose orbits integrate cleanly at 1e-12, or a separately-registered
measurement chapter -- not a v0.12 refinement.

Steps 1-4 plus the decision-B de-risk probe are complete; the unbiased probe
decisively blocks the full run under the locked attrition policy, so step 7 (the
full 8,320-row pass + transfer statistic) is NOT run.

## Primary Transfer Domain

Start from target rows that pass all of:

```text
not overlap-quarantined
not strict m3=1 sigma3 prior-evidence row
velocity_fraction computed
symplecticity gate passed
reciprocal-pair gate passed
```

Primary `m3` strata must satisfy:

```text
N >= 30
S >= 5
U >= 5
```

Rows in one-class or tiny strata are report-only. They are scored when possible,
but they do not enter the primary statistic or p-value.

## Attrition Policy

Attrition is evidence, not housekeeping.

```text
attrition_fraction = blocked_or_sanity_failed_rows / nonoverlap_candidate_rows
```

Branches:

```text
attrition_fraction <= 0.05:
  clean domain

0.05 < attrition_fraction <= 0.20:
  attrition warning; scoring may proceed, but any PASS is explicitly
  "analyzable-domain transfer" rather than clean catalog transfer

attrition_fraction > 0.20:
  verdict = external_transfer_blocked_by_attrition
```

Also block if, after attrition and primary-stratum gates:

```text
primary rows < 500
primary both-class m3 strata < 5
```

These blocks are not failures of the velocity-fraction hypothesis. They mean the
locked D5 measurement could not support the external transfer question on this
target.

## Fixed Score

For every analyzable row:

```text
zone_index = 0 if velocity_fraction < 0.25
zone_index = 1 if 0.25 <= velocity_fraction < 0.50
zone_index = 2 if velocity_fraction >= 0.50
```

No training, smoothing, calibration, fold fitting, or mass-specific adjustment is
allowed.

## Primary Statistic

Within each primary `m3` stratum, compare every stable row to every unstable row:

```text
J_m = #{(S_i, U_j): zone_index(S_i) > zone_index(U_j)}
      + 0.5 * #{(S_i, U_j): zone_index(S_i) = zone_index(U_j)}

D_m = S_m * U_m
```

Pool across primary strata:

```text
J_cond   = sum_m J_m
D_cond   = sum_m D_m
AUC_cond = J_cond / D_cond
```

`AUC_cond = 0.5` is the frozen conditional baseline. Larger values mean stable
rows tend to occupy higher velocity-fraction zones than unstable rows within the
same `m3` stratum.

## Binding Null

Binding p-value: stratified label permutation.

Permutation rule:

```text
within each primary m3 stratum, shuffle S/U labels while preserving S_m and U_m
recompute J_cond and AUC_cond
seed = 20260523
permutations = 100000
p_perm = (1 + count(AUC_perm >= AUC_observed)) / (1 + permutations)
```

The exact v0.11 enumeration kernel may be reported as a sidecar only if the
combined support is small enough to compute comfortably:

```text
exact sidecar allowed if total doubled-J support <= 250000
exact sidecar never replaces the binding permutation p in v0.12
```

This avoids making v0.12 depend on a novel large-support exact kernel while
still permitting the exact receipt when it is cheap.

## Verdict Tree

Run source, overlap, smoke, attrition, and primary-domain gates first.

```text
external_transfer_blocked_by_source
  source profile fails

external_transfer_blocked_by_leakage
  overlap_fraction_of_candidate_primary > 0.05

external_transfer_blocked_by_attrition
  attrition_fraction > 0.20
  OR primary rows < 500
  OR primary both-class m3 strata < 5

external_transfer_passes_clean
  attrition_fraction <= 0.05
  AND AUC_cond >= 0.55
  AND p_perm <= 0.01

external_transfer_passes_attrition_warning
  0.05 < attrition_fraction <= 0.20
  AND AUC_cond >= 0.55
  AND p_perm <= 0.01

external_transfer_directional_weak
  AUC_cond > 0.50
  AND p_perm <= 0.01
  AND AUC_cond < 0.55

external_transfer_fails
  AUC_cond <= 0.50
  OR p_perm > 0.01
```

The `0.55` effect floor is deliberately above a barely-positive rank signal.
This chapter is meant to upgrade evidence, not merely re-detect a tiny
directional trace in a much larger table.

## Required Diagnostics

The results readback must include:

```text
source row count and S/U totals
overlap count and overlap fraction
strict m3=1 prior-evidence exclusion count
smoke seconds/row and projected full wall-clock
attrition by failure reason
attrition by m3 and period quintile
primary rows, S/U totals, and primary m3 strata count
per-stratum N/S/U, zone counts, J_m, D_m, AUC_m
pooled J_cond, D_cond, AUC_cond
p_perm, seed, and permutation count
comparison to v0.11 AUC_cond = 0.6782549421
comparison to v0.10b global held-out AUC = 0.4125
m3=0.4 diagnostic if that stratum survives primary gates
```

Report-only tables:

```text
overlap rows
strict sigma3 prior-evidence rows
tiny / one-class strata
blocked or sanity-failed rows
exact-enumeration sidecar, if computed
```

## Output Contract

Expected output directory:

```text
results/isotrophy/k-facet-v12-external-frozen-transfer/
```

Required files:

```text
manifest.json
source_profile.csv
overlap_audit.csv
integration_smoke.csv
per_row_transfer_table.csv
per_stratum_transfer_table.csv
permutation_summary.json
report_only_rows.csv
```

`manifest.json` must include:

```text
verdict
startedAt
completedAt
gitCommit
nodeVersion or pythonVersion
target_path
discovery_path
v0.11_manifest_path
zone_cutpoints
seed
permutations
attrition_fraction
primary_row_count
primary_strata_count
AUC_cond
p_perm
```

## Claim Boundary

If v0.12 passes, the allowed claim is:

> The v0.11 frozen velocity-fraction zone order transfers from the
> supplementary-B piano-trio table to the distinct supplementary-A periodic-orbit
> table as a conditional within-`m3` stability-ranking signal.

If v0.12 passes with attrition warning, append:

> on the analyzable domain that passed the locked v0.7 D5 numerical gates.

Forbidden upgrades:

```text
does not overturn v0.10b's mass-marginal held-out null
does not create a globally calibrated predictor
does not revise the v0.3h K_facet structural null
does not promote isotrophy to theorem-facing status by itself
does not claim anything about rows excluded by overlap, prior-evidence, attrition,
or tiny/one-class strata
does not claim independent-catalog validation beyond the same-paper supplement
```

If v0.12 fails, the allowed claim is:

> The v0.11 conditional velocity-fraction signal is not shown to transfer under
> the frozen same-paper external supplement test.

If v0.12 blocks, the allowed claim is:

> The locked D5 velocity-fraction measurement did not support a decisive external
> transfer test on supplementary-A under the pre-registered attrition/source gates.

## Implementation Boundary

This draft authorizes no code. On lock, the next implementation should be
additive:

```text
scripts/v12_external_frozen_transfer.py
npm script alias, if desired
no edits to v0.11 result files
no edits to prior manifests
```

The implementation sequence after lock:

1. Write parser/profile/overlap code.
2. Run profile only.
3. Write smoke path and velocity-fraction measurement wrapper.
4. Run the 30-row smoke.
5. Record measured seconds/row and projected full cost in this file or the
   results readback.
6. Stop if the full run exceeds the inline budget; stage the exact PowerShell
   command for operator or long-budget runner.
7. Run full transfer only after operator confirmation.
