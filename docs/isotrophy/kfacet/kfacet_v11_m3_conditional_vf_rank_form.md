# v0.11 m3-Conditional vf Rank Form Lock

Status: **OPERATOR LOCK 2026-05-30.** No v0.11 runner has been written and no v0.11
command has been run at lock time. This document locks the conditional domain, fixed
rank score, exact stratified null, pass/fail tree, diagnostics, and claim boundary
before any v0.11 statistic is computed.

Reviewed for self-consistency, non-circularity, and integrity:

- **Arithmetic reconciles** against the v0.10b receipt: the 9 primary strata sum to
  229 rows = 82 S / 147 U; +21 report-only (5 one-class + 16 tiny) = 250 analyzable;
  +23 v0.7a integration-blocked = 273 supp-B catalog. The 9 strata are exactly
  v0.10b's both-class folds.
- **The m3=1.7 exclusion is mathematically inert, not performance-motivated:** a
  one-class stratum contributes `D_m = S_m*U_m = 0` pairs to `AUC_cond` and nothing
  to the exact null, so including or excluding it yields identical `J_cond` and
  `p_exact`. No "drop a stratum to pass" risk.
- **Non-circularity breaks the statistic (the allowed one):** v0.10b was a global,
  trained, leave-one-bin-out AUC with a permutation null; v0.11 is an in-sample,
  frozen-rule, within-stratum rank AUC with an exact stratified null. The
  within-stratum construction removes the `m3` base-rate confound by construction.
  The test is not rigged to pass — v0.10b already showed m3=0.6/0.7 anti-rank
  within-bin, so the pooled conditional statistic genuinely aggregates working
  strata (0.8/0.9/1.0/1.1) against failing ones.
- **Honest claim boundary:** a PASS registers in-sample within-`m3` conditional rank
  information (post-hoc direction disclosed — the same footing that licensed v0.10a),
  NOT a global predictor, NOT a v0.10b overturn, NOT theorem-facing.
- **Exact null is sound and cheap:** stratified Mann-Whitney with tied zone-groups;
  `D_cond = sum S_m*U_m = 1467`, convolution support ~2934 integer bins, seconds.

Adjustment made at lock (one): added a non-binding within-stratum permutation sanity
cross-check (seed 20260523, n=10000) guarding the hand-rolled enumeration +
convolution kernel, with an ABORT-on-gross-divergence branch. The binding p-value
remains `p_exact`; the sanity sidecar mirrors v0.10a's exact+permutation discipline.

## Frame

v0.10a registered a monotone in-sample trend:

```text
positional-dominant  -> mixed       -> velocity-heavy
S_fraction 0.1053       0.3394         0.4394
J-T exact p = 0.007304
```

v0.10b then tested whether that trend becomes a **globally calibrated held-out
risk score across mass bins**. It failed cleanly:

```text
pooled leave-one-m3-bin-out held-out AUC = 0.4125 <= 0.5
verdict = monotone_vf_predictor_fails_heldout
```

The mandated v0.10b diagnostics showed the failure mode: several held-out
`m3` folds had within-bin AUC above 0.5, but the pooled cross-bin score collapsed
because `m3` base rates dominate global ordering. v0.11 is therefore a **fresh
conditional chapter**, not a v0.10b refinement:

> Given a fixed mass-ratio bin, does the frozen vf zone order rank stable rows
> above unstable rows?

This is a fixed-rule conditional rank test. It is not a calibrated global
predictor and not a hard classifier.

## Integrity Caveat

The v0.11 hypothesis is diagnostic-driven: v0.10b exposed the within-bin signal
after its global gate failed. A v0.11 pass would register a **conditional
within-`m3` rank signal** on the frozen analyzable supplementary-B domain. It would
not overturn v0.10b, would not rescue the global predictor, and would not be an
independent external-catalog validation.

The key non-p-hack boundary:

```text
v0.10b claim failed: one global zone-only risk score transfers across mass bins.
v0.11 claim, if passed: within fixed m3 strata, the frozen zone order carries
                       stability ranking information.
```

Those are different claims. v0.11 may not be used to rewrite v0.10b's verdict.

## Frozen Inputs

Primary input:

```text
results/isotrophy/k-facet-v09a-signed-vf-three-zone/per_row_table.csv
```

Required receipt cross-checks:

```text
results/isotrophy/k-facet-v10a-jt-trend/manifest.json
results/isotrophy/k-facet-v10b-monotone-vf-heldout/manifest.json
```

The runner must assert:

```text
v0.9a/v0.10a frozen row count:        250
stability totals:                     87 S / 163 U
zone counts:
  positional-dominant:                N=19,  S=2,  U=17
  mixed:                              N=165, S=56, U=109
  velocity-heavy:                     N=66,  S=29, U=37
zone recompute:                       recompute from velocity_fraction under
                                      {0.25,0.50}; require zero mismatches
v0.10a verdict:                       jt_trend_monotone_registered
v0.10b verdict:                       monotone_vf_predictor_fails_heldout
v0.10b pooled AUC condition:           AUC <= 0.5
```

Any mismatch aborts. No new orbit integration, variational integration, gauge
minimization, or re-derivation of `velocity_fraction` is authorized.

## Primary Conditional Domain

The primary domain uses frozen analyzable rows in `m3` strata with:

```text
N >= 5
S >= 1
U >= 1
```

This includes every non-tiny, both-class mass bin and excludes no bin based on
the observed v0.10b fold AUC.

```text
m3    N    S    U
0.4   55   35   20
0.5   30   15   15
0.6   22    6   16
0.7   18    4   14
0.8   18    6   12
0.9   21    6   15
1.0   35    7   28
1.1   23    1   22
1.2    7    2    5
```

Primary rows: `229` (`82 S / 147 U`).

Report-only strata:

```text
one-class N>=5:
  m3=1.7   N=5   S=0   U=5

tiny N<5:
  m3=1.3   N=4   S=2   U=2
  m3=1.4   N=4   S=0   U=4
  m3=1.5   N=4   S=1   U=3
  m3=1.6   N=2   S=2   U=0
  m3=1.9   N=2   S=0   U=2
```

The report-only strata are scored and printed, but do not enter the primary
statistic, exact null, p-value, or verdict.

## Fixed Rank Score

Feature and score are frozen from v0.9a/v0.10a:

```text
zone_index:
  positional-dominant = 0
  mixed               = 1
  velocity-heavy      = 2
```

No training is performed. No mass-specific threshold, smoothing, calibration,
logistic regression, raw `velocity_fraction` threshold, branch label, `z0`, `E`,
`|L|`, period, or `m3` value may enter the score. `m3` is used only to define the
conditional comparison strata.

## Primary Statistic

Within each primary `m3` stratum, compare every stable row to every unstable row
in the same stratum:

```text
J_m = #{(S_i, U_j): zone_index(S_i) > zone_index(U_j)}
      + 0.5 * #{(S_i, U_j): zone_index(S_i) = zone_index(U_j)}

D_m = S_m * U_m
```

Pool across primary strata:

```text
J_cond = sum_m J_m
D_cond = sum_m D_m
AUC_cond = J_cond / D_cond
```

`AUC_cond = 0.5` is the constant-score conditional baseline. Larger values mean
stable rows tend to occupy higher vf zones than unstable rows **within the same
mass-ratio bin**.

## Exact Stratified Null

Binding p-value: deterministic exact fixed-stratum enumeration.

For each primary `m3` stratum:

1. Preserve the zone counts in that stratum.
2. Preserve the total `S_m` and `U_m` in that stratum.
3. Enumerate every feasible allocation of `S` counts across the occupied zones.
4. Weight each allocation by the stratum's multivariate-hypergeometric
   probability.
5. Compute the stratum `J_m` for that allocation.

The runner combines the per-stratum exact distributions by convolution over
`2 * J_m` integer scores, yielding the exact null distribution of `J_cond` under
within-stratum exchangeability of S/U labels.

Binding one-sided p-value:

```text
p_exact = P_null(J_cond >= J_cond_observed)
```

Sanity gates:

```text
per-stratum probability mass within 1e-10 of 1
combined probability mass within 1e-9 of 1
```

If either mass check fails, abort rather than fall back to Monte Carlo.

### Permutation sanity cross-check (non-binding)

The per-stratum enumeration and the cross-stratum convolution are the novel
kernel of this chapter; the mass-sanity gates prove the enumeration is *complete*
but not that `J_m` and the convolution score-axis are aligned correctly. As an
independent guard (mirroring v0.10a's exact+permutation discipline), the runner
also estimates the same one-sided p-value by within-stratum label permutation:
shuffle the S/U labels within each primary `m3` stratum (preserving `S_m`, `U_m`,
and the zone counts), recompute `J_cond`, repeat with the program-locked seed
`20260523` for `n = 10000` draws, and report

```text
p_perm_sanity = (1 + #{J_cond_perm >= J_cond_observed}) / (1 + 10000)
```

This is a SANITY sidecar; the binding p-value remains `p_exact`. The two must
agree within Monte-Carlo error (`p_exact` inside `p_perm_sanity +/- 5` standard
errors, accounting for the `1/n` resolution floor in the deep tail). A gross
divergence indicates an enumeration/convolution kernel bug and INVALIDATES the
run (ABORT, fix, re-run) — it never silently changes the verdict.

## Verdict Tree

```text
if frozen-input cross-check fails:
    ABORT  (not the frozen v0.9a/v0.10a/v0.10b receipt chain)
elif primary conditional domain != the 9 strata / 229 rows listed above:
    ABORT  (domain mismatch)
elif exact-null probability-mass sanity fails:
    ABORT  (exact enumeration invalid)
elif exact-vs-permutation-sanity divergence beyond ~5 Monte-Carlo SE:
    ABORT  (enumeration/convolution kernel bug)
elif AUC_cond <= 0.5:
    verdict = m3_conditional_vf_rank_fails
elif p_exact <= 0.01:
    verdict = m3_conditional_vf_rank_passes
else:
    verdict = m3_conditional_vf_rank_fails
```

No partial branch exists. Results with `0.01 < p <= 0.05` are reported as
descriptive hints and formal failures.

## Required Diagnostics

The receipt must report:

- per-stratum `N`, `S`, `U`, zone counts, `J_m`, `D_m`, and `AUC_m`;
- `m3 = 0.4` diagnostic block, because prior chapters identified it as
  load-bearing;
- pair-weight contribution by stratum (`D_m / D_cond`);
- exact-null support size per stratum and after convolution;
- comparison against v0.10b pooled AUC:
  - v0.10b global pooled held-out AUC = `0.4125` (failed);
  - v0.11 conditional AUC (this run);
- report-only stratum table for one-class and tiny bins;
- alignment diagnostic against v0.5a branch labels, report-only. Branch labels
  are never features.

## Disallowed Moves

v0.11 may NOT use:

- raw continuous `velocity_fraction`;
- v0.7a-prime quartile labels or quartile cutpoints;
- `m3` as a score feature or calibration input;
- mass-specific thresholds;
- branch labels, `z0`, `E`, `|L|`, period, or catalog-coordinate features;
- eigenvalue magnitudes, spectral radius, unit-circle status, unstable-pair
  count, or any threshold that defines the published S/U label;
- post-run domain changes to drop weak strata or rescue a fail.

## Claim Boundary

A PASS:

- says the frozen vf zone order carries **within-`m3` conditional rank
  information** about stability on the 229-row primary analyzable supp-B domain;
- does NOT claim a globally calibrated predictor across mass bins;
- does NOT overturn v0.10b's held-out global null;
- does NOT cover the 21 report-only rows or the 23 v0.7a integration-blocked
  rows;
- does NOT make isotrophy theorem-facing or revise the v0.3h K_facet
  structural-null.

A FAIL:

- says the encouraging v0.10b per-fold diagnostics do not survive the registered
  exact conditional rank gate;
- should close the conditional-predictor rung unless a fresh external catalog or
  genuinely different physical mechanism is registered.

## Receipt Schema

```text
results/isotrophy/k-facet-v11-m3-conditional-vf-rank/
  manifest.json
    - schema = "sundog.isotrophy.v0.11-m3-conditional-vf-rank.v1"
    - mode = "v0.11-m3-conditional-vf-rank"
    - form_lock = "docs/isotrophy/kfacet/kfacet_v11_m3_conditional_vf_rank_form.md"
    - input_v09a_per_row_table
    - input_v10a_manifest
    - input_v10b_manifest
    - frozen_input_crosscheck
    - primary_strata
    - report_only_strata
    - score = { feature: zone_index, values: {positional:0, mixed:1, velocity_heavy:2} }
    - observed = { J_cond, D_cond, AUC_cond }
    - exact_null = { per_stratum_support, combined_support,
                     per_stratum_probability_mass, combined_probability_mass,
                     p_value_one_sided }
    - permutation_sanity = { seed: 20260523, n: 10000, p_perm_sanity,
                             standard_error, consistent_with_exact }
    - verdict
  per_stratum_table.csv
  report_only_strata.csv
  exact_null_summary.csv
```

## Lock-In Statement

This form is committed before any v0.11 conditional statistic, exact null,
stratum AUC, or report-only score is computed. Any change to the primary domain,
fixed rank score, exact-null construction, significance floor, verdict tree, or
claim boundary after runner execution is a re-registration, not a refinement.

Implementation may proceed against the frozen v0.9a per-row table plus v0.10a and
v0.10b manifests only. Expected runtime: seconds.
