# v0.8a Purity-Quartile Audit Form Lock

Status: **VERDICT LANDED 2026-05-24**. Verdict:
**`purity_quartile_fails_audit`** at `chi^2 = 4.94`, p = 0.176 (df=3,
critical 11.34). The unsigned purity transform `abs(vf - 0.5)` does
NOT capture the v0.7a' signal at the registered floor. The
diagnostic purity_signed contingency reproduces v0.7a' (chi^2 =
16.43; expected because purity_signed = vf - 0.5 is a linear
transform of vf, preserving the quartile ordering). The asymmetry
diagnostic (`|S_fraction(Q1_signed) - S_fraction(Q4_signed)|` =
0.064) is **below the 0.2 flag threshold**, so the asymmetric U-shape
flag did NOT fire -- the S-fractions at the two pure ends are
similar. The asymmetry that breaks the purity transform is in
**row density**, not S-fraction: 184/250 rows have vf < 0.5 (max vf
= 0.87, no rows near vf = 1), so the unsigned purity folding does
not align with the v0.7a' U-shape structure.

This document locks the v0.8a purity-quartile chi-squared audit on
the 250-row analyzable supp-B subset under the parent registration's
candidate slot **A** (purity-quartile audit). Candidate forms B
(monotone purity-threshold), C (Spearman correlation), and D
(median-split) are held back.

## Verdict (Landed, 2026-05-24)

Receipt:
`results/isotrophy/k-facet-v08a-purity-audit/manifest.json`.

```text
verdict:               purity_quartile_fails_audit
domain:                250 analyzable rows  (87 S / 163 U)
purity_sd:             0.0961  (>> 0.01 retirement floor;
                                purity has real variation, signal
                                just isn't captured by quartile
                                chi-squared)
chi^2_purity:          4.943205
df:                    3
critical (p=0.01):     11.34
p_value (chi-sq(3)):   0.1760
alignment_tightness:   0.5873  (well below 0.8 warning threshold)
test_branch_taken:     chi_squared
```

Primary purity quartile contingency (re-binned within the 250-row
analyzable subset):

```text
Q_purity  N    S    U   S_fraction  chi2_contrib  approx_vf_range
1         63   21   42   0.3333       0.06         vf in [0.44, 0.56]   mid-vf
2         62   18   44   0.2903       0.91         vf in [0.39, 0.44] U [0.56, 0.61]
3         62   19   43   0.3065       0.47         vf in [0.28, 0.39] U [0.61, 0.72]
4         63   29   34   0.4603       3.50         vf in [0, 0.28] U [0.72, 0.87]
                                      -----
                            chi^2 =   4.94
```

Q4 (highest purity; vf far from 0.5) shows 46% S -- elevated relative
to the catalog mean 34.8%, but Q1, Q2, Q3 are essentially flat around
30%. The signal is concentrated at the high-purity end but does not
distribute across the quartiles in a way that reaches chi-squared(3)
> 11.34.

Diagnostic purity_signed contingency (signed direction; report-only):

```text
Q_signed  N    S    U   S_fraction  chi2_contrib
1         63   31   32   0.4921       5.76    vf in [0, 0.30]    pos pure
2         62   11   51   0.1774       7.95    vf in [0.30, 0.42]
3         62   18   44   0.2903       0.91    vf in [0.42, 0.50]
4         63   27   36   0.4286       1.80    vf in [0.50, 0.87]  vel pure
                                       -----
                            chi^2 =  16.43

asymmetry: |S_fraction(Q1) - S_fraction(Q4)| = |0.4921 - 0.4286|
                                              = 0.064
                                              < 0.2 (flag threshold)
asymmetric_u_shape_flag: False
```

The diagnostic reproduces v0.7a's chi^2 = 16.43 exactly. This
confirms the v0.7a' signal is real and lives in the **signed** vf
direction; the unsigned purity transform loses access to it because
the vf distribution is density-asymmetric (74% of rows have vf <
0.5).

vf distribution diagnostic (250 analyzable rows):

```text
vf_min:        0.0426
vf_q25:        0.2972
vf_q50:        0.4188   (median below 0.5)
vf_q75:        0.5036
vf_max:        0.8653
vf < 0.5:      184 / 250 (73.6%)
vf > 0.5:       66 / 250 (26.4%)
```

The vf distribution is heavily skewed toward low values (positional-
leaning gamma_1). The "Q4 of vf" rows (vf > 0.5) are actually vf in
[0.5, 0.87], not symmetric around vf = 0.5. The purity transform
`abs(vf - 0.5)` treats vf = 0 and vf = 1 as equivalent, but
the data doesn't have many vf-near-1 rows, so the symmetry assumption
of "purity" is violated by the catalog's actual distribution.

Per-m_3 analyzable counts (for v0.8b partition design, recorded but
v0.8b is NOT licensed under the fails verdict):

```text
m_3=0.4: 55     m_3=0.9: 21     m_3=1.4: 4
m_3=0.5: 30     m_3=1.0: 35     m_3=1.5: 4
m_3=0.6: 22     m_3=1.1: 23     m_3=1.6: 2
m_3=0.7: 18     m_3=1.2: 7      m_3=1.7: 5
m_3=0.8: 18     m_3=1.3: 4      m_3=1.9: 2
```

High-m_3 bins (1.4-1.9) are very thin after attrition (2-5 rows
each), confirming the v0.6/v0.7 pattern: attrition concentrates at
long-period high-m_3, where the variational equation hits the
precision wall.

**Structural reading:**

> v0.8a tested the hypothesis that v0.7a's U-shaped vf-quartile
> signal is captured by an unsigned direction-purity scalar
> `purity = abs(vf - 0.5)`. The hypothesis fails at the registered
> floor: chi-squared(3) = 4.94 vs critical 11.34, p = 0.176. The
> failure is NOT explained by feature constant-retirement (purity
> has sd = 0.096) or alignment-tightness with the v0.5a branch hash
> (alignment 0.587, well below 0.8). The failure is explained by
> the **density asymmetry** of the vf distribution: 74% of rows lie
> below vf = 0.5, max vf = 0.87, so the "vf near 1" end of the
> "purity" interpretation is structurally under-populated. The
> diagnostic purity_signed quartile contingency reproduces v0.7a's
> chi^2 = 16.43 exactly (since vf - 0.5 is a linear transform of
> vf), confirming the v0.7a' positive is a real signed-direction
> signal that the unsigned purity transform discards.

Per the locked outcome interpretation:

> Direction-purity does not stratify stability on the analyzable
> sub-catalog at the registered floor. The v0.7a' U-shape signature
> was not purity-driven in the structural sense; the v0.8 chapter
> closes on the purity shadow.

The asymmetric_u_shape diagnostic flag did NOT fire (asymmetry 0.06
< 0.2), so the U-shape is not asymmetric in S-fraction terms. The
asymmetry that broke the purity transform is in catalog row density,
not stability outcome.

v0.8b (held-out purity predictor) is NOT licensed under the fails
verdict.

**Receipts:**

```text
docs/isotrophy/kfacet/kfacet_v08a_purity_quartile_audit_form.md  (this document)
results/isotrophy/k-facet-v08a-purity-audit/manifest.json
results/isotrophy/k-facet-v08a-purity-audit/per_row_table.csv
results/isotrophy/k-facet-v08a-purity-audit/contingency_table_purity.csv
results/isotrophy/k-facet-v08a-purity-audit/contingency_table_purity_signed.csv
results/isotrophy/k-facet-v08a-purity-audit/per_m3_analyzable_counts.csv
scripts/v08a_purity_audit.py
```

Audience: v0.8a runner; v0.8b form-lock author (conditional on v0.8a
passing); paper-side reviewer of the direction-purity transition.

Companions:

- `kfacet_v08_mechanism_preregistration.md` -- v0.8 parent
  registration. Locks body / projection / observable, operational
  definition of purity, non-circularity audit, disallowed-feature
  list, and inheritance discipline.
- `kfacet_v07a_prime_restricted_scope_form.md` -- v0.7a' PASS
  verdict on the velocity-fraction quartile audit (source of the
  U-shape signature that licenses v0.8).
- `kfacet_v07a_velocity_fraction_audit_form.md` -- v0.7a parent
  audit (source of the vf operational definition; non-circularity
  sentence carries forward).
- `kfacet_v06b_within_branch_energy_audit_form.md` -- sparse-cell
  fallback tree (inheritance source).
- `kfacet_v06a_energy_quartile_audit_form.md` -- alignment-tightness
  guard (inheritance source).

Frame: v0.7a' produced a non-monotone U-shape on velocity-fraction
quartiles (Q1 49% S, Q2 18% S, Q3 29% S, Q4 43% S). The U-shape
suggests **direction-purity** -- the distance of gamma_1's
velocity-fraction from the maximally-mixed point at vf = 0.5 -- as
the load-bearing axis. v0.8a tests whether the purity quartile
stratifies S/U cleanly under the same statistical pipeline that
v0.7a' used.

## Why Purity-Quartile (Form A) Over Alternatives

Per the parent registration:

```text
A. purity-quartile audit (4 x 2 chi-squared):     SELECTED.
   - Most-direct analogue of v0.7a's B audit form (which v0.7a'
     also used). Inherits exactly the same statistical pipeline
     (4-bin chi-squared, df = 3, critical 11.34, alignment-
     tightness guard, sparse-cell fallback).
   - Preserves enough granularity to detect monotone-in-purity
     structure AND any residual asymmetry between Q1-pure and
     Q4-pure rows (recorded via the purity_signed diagnostic).
   - Cleanest extension of v0.7a' under inheritance discipline.

B. monotone purity-threshold (2 x 2):     held back.
   - Coarser test (df = 1, critical 6.63) with lower power if
     the effect is graded rather than discrete-threshold.
   - Threshold-choice flexibility (which quantile?) requires extra
     pre-registration work.

C. Spearman rank correlation:     held back.
   - Different test statistic; breaks the chi-squared lineage
     inherited from v0.6/v0.7.
   - Continuous; avoids binning but loses the chi^2 contribution
     breakdown that revealed the U-shape in v0.7a'.

D. median-split + binomial fallback:     held back.
   - Even coarser (2 x 2 with 125/125 split); same df-power
     tradeoff as B.
   - Power-conservative but loses the quartile-pattern visibility.
```

The locked choice prioritizes **inheritance fidelity** with v0.7a'
plus **diagnostic granularity** (the four quartiles let us see the
shape inside the analyzable subset, not just a monotone summary).

## What v0.8a Is

v0.8a is a **catalog-side audit on the v0.7a analyzable subset**,
NOT a predictor (continues v0.5a/v0.6a/v0.6b/v0.7a/v0.7a'
audit-not-predictor framing):

```text
Domain:           250 analyzable rows from v0.7a (rows where
                  compute_monodromy_vectorized completed at the
                  R1-amended sanity gates). Attrition (23 blocked +
                  11 R1-sanity-failed) is carried as a permanent
                  domain restriction.

Feature:          purity(row) = abs(vf(row) - 0.5)
                  where vf is the v0.7a velocity-fraction (locked
                  selection rule + tie-break cascade + reference
                  frame + non-circularity provenance).

Binning:          quartiles over the 250-row analyzable subset's
                  purity distribution.

Test:             chi-squared independence of Q_purity vs S/U.
                  df = 3 (or occupied_bins - 1 under sparse fallback).
                  critical 11.34 at p = 0.01 (chi-squared(3)).

Alignment guard:  max over Q_purity of (fraction in any single
                  v0.5a branch_label bucket). Thresholds 0.8
                  (warning) / 0.95 (severe).

Sparse fallback:  v0.6b discipline (seed 20260523, n_permutations
                  10000, min_occupied_bin_count >= 2,
                  expected_cell >= 5).

Constant-feat:    if sd(purity) < 0.01, retire.

Verdicts:
  Pass:           chi^2 > 11.34 AND alignment <= 0.8
                  -> purity_quartile_passes_audit
  Warn pass:      chi^2 > 11.34 AND alignment in (0.8, 0.95]
                  -> purity_quartile_passes_audit_alignment_warning
  Severe pass:    chi^2 > 11.34 AND alignment > 0.95
                  -> purity_quartile_passes_audit_severe_alignment
  Fail:           chi^2 <= 11.34
                  -> purity_quartile_fails_audit
  Sparse:         min_occupied_bin_count < 2
                  -> purity_quartile_inconclusive_sparse
  Retired:        sd(purity) < 0.01
                  -> purity_quartile_retired_near_constant
```

A clean Pass licenses v0.8b: a separately-registered held-out
purity predictor with the v0.5b discipline. A Warn/Severe Pass
licenses only an alignment-breaking v0.8b. A Fail closes the
purity-shadow sub-question; an asymmetric U-shape diagnostic
(see below) may motivate a non-purity feature re-registration as
v0.8a'.

## The Locked Form

### Operational definition (inherited from v0.8 parent)

```text
For each row r in the 250-row analyzable subset:

  purity(r) = abs(vf(r) - 0.5)

  range: [0, 0.5]

  purity(r) = 0    iff  vf(r) = 0.5  (maximally mixed)
  purity(r) = 0.5  iff  vf(r) in {0, 1} (perfectly pure end)
```

### Binning (locked)

```text
cutpoints:    q25, q50, q75 of the 250-row analyzable subset's
              purity distribution.

method:       numpy.quantile(purity_values, [0.25, 0.50, 0.75],
                             method='linear').

assignment:   right-closed lower intervals; ties to lower bin
              (matches v0.6a/v0.7a/v0.7a' convention).

                Q_purity(r) = 1  iff  purity(r) <= q25
                Q_purity(r) = 2  iff  q25 < purity(r) <= q50
                Q_purity(r) = 3  iff  q50 < purity(r) <= q75
                Q_purity(r) = 4  iff  purity(r) >  q75
```

### Expected direction (pre-registered)

If v0.7a's U-shape is purity-driven, the expected ordering is
**monotone increasing in Q_purity**:

```text
expected:  S_fraction(Q_purity=1) < S_fraction(Q_purity=2) <
           S_fraction(Q_purity=3) < S_fraction(Q_purity=4)
```

If the observed pattern is NOT monotone (e.g., Q4 has lower S
than Q3), the U-shape is asymmetric and the purity transform
may not be the cleanest mechanism. The purity_signed diagnostic
(below) addresses this.

## Diagnostic: purity_signed Quartile (Asymmetry Check)

The v0.8 parent registration recorded `purity_signed = vf - 0.5`
as a diagnostic-only quantity. v0.8a's receipt MUST emit the
purity_signed quartile contingency to check whether Q1-pure
(vf near 0) and Q4-pure (vf near 1) orbits behave the same:

```text
purity_signed quartiles (4-bin over [-0.5, 0.5]):
  Q1: vf in [0, q25_signed]    near-positional pure end
  Q2: vf in (q25_signed, q50_signed]
  Q3: vf in (q50_signed, q75_signed]
  Q4: vf in (q75_signed, 1]    near-velocity pure end

asymmetry diagnostic:
  abs(S_fraction(Q1) - S_fraction(Q4))

If asymmetry > 0.2 (locked threshold):
  flag asymmetric_u_shape on receipt; v0.8b may need
  re-registration with a non-purity feature.
```

The diagnostic is **not gating**. The primary verdict is determined
by the unsigned purity quartile audit. The diagnostic is recorded
for v0.8b form-lock design.

## Per-m_3 Analyzable Row Counts (Required Diagnostic)

The v0.8a receipt MUST report per-m_3-bin analyzable row counts
on the 250-row subset, so v0.8b's leave-one-m_3-bin-out partition
design is informed:

```text
m_3 bin     analyzable count
0.4         (count after attrition)
0.5
0.6
...
1.9
```

This is non-gating but mandatory on the receipt. Long-period
high-m_3 bins (1.5, 1.6, 1.7) are expected to be thin after
attrition; v0.8b's partition may need to drop bins with N < 5 to
report-only status.

## Pre-Audit Sanity Surface

Before v0.8a runs:

```text
1. v0.7a per_row_table.csv must be present and readable.
2. Filter to analyzable rows (integration_blocked == False);
   expected count = 250.
3. v0.5a branch_label must be present per row (already joined in
   v0.7a per_row_table).
4. Verify vf values are in [0, 1] for all 250 rows; flag any out-
   of-range row as a parser/computation bug.
5. Compute purity = abs(vf - 0.5) for all 250 rows.
6. Constant-feature retirement check: if sd(purity) < 0.01,
   verdict = purity_quartile_retired_near_constant, no audit run.
```

## Non-Circularity Audit

purity is a monotone deterministic function of vf; non-circularity
provenance is inherited verbatim from v0.7a:

```text
- purity depends on vf only, which depends on gamma_1, which is
  the eigenvector of the largest-real-part Floquet eigenvalue.
- The eigenvalue ORDERING is used to disambiguate; the eigenvalue
  MAGNITUDE never enters purity.
- The 250-row subset is filtered by integration_blocked, a runtime
  status of the integrator, NOT a function of stability.
- The S/U label is the held-back test column, joined at the
  chi-squared step.
- The v0.5a branch_label is joined ONLY for the alignment-tightness
  diagnostic, NOT used to construct purity.
```

The locked non-circularity sentence from v0.7a is re-asserted for
v0.8a.

## Free Parameter Count

```text
Number of free parameters fitted to the data: 0
```

The feature (purity = abs(vf - 0.5)), the binning rule (quartiles),
the quantile method (numpy linear interpolation), the bin assignment
convention (right-closed lower), the df formula (occupied_bins - 1),
the critical value (11.34), the alignment-tightness thresholds
(0.8, 0.95), the constant-feature retirement threshold (sd < 0.01),
the sparse-cell fallback parameters (min_expected = 5,
min_occupied_bin_count = 2, permutation_seed = 20260523,
n_permutations = 10000), and the asymmetry-diagnostic threshold
(0.2) are all locked pre-data.

## Sparse-Cell Fallback Logic (Inherited from v0.6b)

```text
1. Compute expected cell counts E_{kj} = N_k * C_j / 250.
2. Compute min_occupied_bin_count = min over occupied bins of N_k.
3. Branch:
   a) If min_occupied_bin_count < 2:
        verdict = purity_quartile_inconclusive_sparse
        no p-value emitted.
   b) Elif min over occupied cells of E_{kj} < 5:
        permutation test (seed = 20260523, n_permutations = 10000);
        p_value = empirical tail.
   c) Else:
        asymptotic chi-squared(df = occupied_bins - 1) at p = 0.01.
```

For quartile binning on 250 rows with 87 S / 163 U, expected counts
per cell are approximately {22, 41} -- well above the asymptotic
floor. The fallback is documented for completeness.

## Alignment-Tightness Guard (Inherited from v0.6a / v0.7a)

```text
For each Q_purity in {1, 2, 3, 4}:
  - identify the v0.5a branch_label that dominates the bin.
  - compute dominant_fraction = (rows in bin in dominant branch)
                                / (rows in bin).

alignment_tightness_scalar = max over Q_purity of dominant_fraction.

Thresholds:    0.8 (warning), 0.95 (severe).

If chi^2 > 11.34:
  alignment <= 0.8     -> purity_quartile_passes_audit
  alignment in (0.8, 0.95] -> purity_quartile_passes_audit_alignment_warning
  alignment > 0.95     -> purity_quartile_passes_audit_severe_alignment
```

## Edge Cases (Pre-Registered)

1. **Out-of-range vf**: any row with vf outside [0, 1] is a
   parser/computation bug; block v0.8a and inspect.

2. **Constant-feature retirement**: sd(purity) < 0.01 triggers
   `purity_quartile_retired_near_constant`. Pre-mortem estimate:
   v0.7a' vf_sd was 0.144 over 250 rows; purity sd should be
   similar order (>= 0.05); retirement is structurally unlikely.

3. **Empty quartile**: structurally impossible for quartile binning
   over 250 rows; not in scope.

4. **All-S or all-U quartile**: chi-squared still computes;
   expected cell counts handle the case.

5. **Asymmetric U-shape**: if `|S_fraction(purity_signed_Q1) -
   S_fraction(purity_signed_Q4)| > 0.2`, the receipt flags
   `asymmetric_u_shape = True`. The verdict is unchanged (still
   reflects unsigned purity quartile chi-squared); v0.8b form-lock
   author is informed to consider non-purity features.

## Pre-Mortem Expectation

The per-row purity distribution has NOT been computed before
lock-in. The audit runs blind.

If the v0.7a' U-shape is purity-driven and roughly symmetric:
the v0.8a verdict is **Pass** (chi^2 > 11.34) with alignment
similar to v0.7a's 0.698 (below 0.8); the quartile S-fractions are
expected to be monotone increasing in Q_purity, with Q4 (high
purity) approximately matching v0.7a's combined Q1+Q4 S-fraction
(about 46%) and Q1 (low purity, near vf=0.5) approximately matching
v0.7a's Q2 S-fraction (about 18%).

If the U-shape is asymmetric (e.g., Q4-pure has much higher S than
Q1-pure), the purity transform conflates two mechanisms; the
asymmetric_u_shape flag fires and v0.8b form-lock author is
expected to register a non-purity feature.

If purity does not stratify (Fail verdict), the v0.7a' positive
was an artifact of the quartile boundary rather than a structural
direction-purity effect; the chapter closes on the purity shadow.

## Lock-In Statement

This audit form is committed before:

- per-row purity has been computed,
- the 250-row subset's purity distribution has been examined,
- the quartile cutpoints have been calculated,
- the alignment-tightness scalar has been computed,
- the purity_signed diagnostic contingency has been computed.

Any change to the feature, binning, df formula, critical value,
alignment-tightness thresholds, constant-feature retirement
threshold, sparse-cell fallback parameters, or asymmetry-diagnostic
threshold after the v0.8a runner is executed is a re-registration,
not a refinement.

Implementation may proceed against this form lock using
`results/isotrophy/k-facet-v07a-velocity-fraction-audit/per_row_table.csv`
as the sole input. No new variational compute is required.

## Receipt Schema (Planned)

```text
results/isotrophy/k-facet-v08a-purity-audit/
  manifest.json
    - mode = "v0.8a-purity-quartile-audit"
    - form_lock = "docs/isotrophy/kfacet/kfacet_v08a_purity_quartile_audit_form.md"
    - input_v07a_per_row_table
    - domain_row_count = 250
    - domain_S_count, domain_U_count
    - attrition_disclosure: {v07a_full_catalog_row_count,
                              v07a_integration_blocked_count,
                              v07a_sanity_failed_count}
    - cutpoints_purity: {q25, q50, q75}
    - cutpoints_purity_signed: {q25, q50, q75}
    - constant_feature_sd: sd(purity)
    - constant_feature_retired: bool
    - contingency_purity:        4 x 2 (Q_purity, S/U)
    - chi_squared:               test statistic
    - test_branch_taken:         "chi_squared" | "permutation" |
                                 "inconclusive_sparse" | "retired_near_constant"
    - df:                        3 (or occupied_bins - 1)
    - critical:                  11.34
    - p_value:                   chi-squared tail OR permutation tail
    - alignment_tightness_scalar
    - alignment_records:         per-quartile dominant branch + counts
    - verdict:                   purity_quartile_{passes_audit |
                                                  passes_audit_alignment_warning |
                                                  passes_audit_severe_alignment |
                                                  fails_audit |
                                                  inconclusive_sparse |
                                                  retired_near_constant}
    - diagnostic_purity_signed_contingency:  4 x 2 (Q_purity_signed, S/U)
    - asymmetry_diagnostic:      abs(S_fraction(Q1_signed) - S_fraction(Q4_signed))
    - asymmetric_u_shape_flag:   asymmetry > 0.2
    - per_m3_analyzable_counts:  dict {m3_key: N}
  per_row_table.csv
  contingency_table_purity.csv
  contingency_table_purity_signed.csv
  per_m3_analyzable_counts.csv
```

## Implementation Plan

Suggested script: `scripts/v08a_purity_audit.py`.

Suggested command:

```powershell
python scripts\v08a_purity_audit.py `
  --v07a-per-row results\isotrophy\k-facet-v07a-velocity-fraction-audit\per_row_table.csv `
  --out          results\isotrophy\k-facet-v08a-purity-audit
```

Expected runtime: seconds. No new variational compute.

## Interpretation

If v0.8a passes (chi^2 > 11.34, alignment <= 0.8):

> The Floquet direction-purity scalar stratifies stability on the
> 250-row analyzable supp-B subset under a pre-registered quartile
> chi-squared audit, independent of the v0.5a branch hash. This
> confirms the v0.7a' U-shape signature is purity-driven and
> licenses v0.8b: a separately-registered held-out predictor with
> the v0.5b discipline.

If v0.8a passes with alignment warning (chi^2 > 11.34, alignment
in (0.8, 0.95]):

> Pass-but-suspect. The purity quartiles track the v0.5a branch
> hash tightly; v0.8b's partition must be re-registered alignment-
> breaking (within-branch test mirroring v0.6b).

If v0.8a fails (chi^2 <= 11.34):

> Direction-purity does not stratify stability on the analyzable
> sub-catalog. The v0.7a' U-shape signature was not purity-driven
> in the structural sense; the chapter closes on the purity
> shadow. If the asymmetric_u_shape flag fired, the U-shape's
> asymmetry suggests a non-purity feature mechanism; the v0.8a'
> form-lock author may register a signed-direction or sub-quartile
> feature with a fresh circularity audit.

If purity retires near-constant: structurally unlikely given
v0.7a' vf_sd; the chapter closes on the purity-feature shadow.

## Doc Trail

- `kfacet_v08_mechanism_preregistration.md` -- parent registration.
- `kfacet_v07a_prime_restricted_scope_form.md` -- v0.7a' PASS
  source of the U-shape signature.
- `kfacet_v07a_velocity_fraction_audit_form.md` -- v0.7a parent
  audit (vf operational definition source).
- `kfacet_v06a_energy_quartile_audit_form.md` -- alignment-
  tightness guard pattern.
- `kfacet_v06b_within_branch_energy_audit_form.md` -- sparse-cell
  fallback tree pattern.

---

Purity-quartile audit on the 250-row analyzable subset. df = 3,
critical 11.34. Inherits v0.7a non-circularity sentence verbatim.
Diagnostic purity_signed contingency for U-shape asymmetry check.
Per-m_3 analyzable row counts emitted to inform v0.8b partition
design.
