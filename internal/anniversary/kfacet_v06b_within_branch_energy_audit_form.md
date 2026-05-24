# v0.6b Within-Branch Energy Audit Form Lock (Alignment-Breaking)

Status: **VERDICT LANDED 2026-05-24**. The within-branch
energy-quartile audit failed the registered p <= 0.01 threshold under
the sparse-cell fallback's exact permutation test. Verdict:
`within_branch_energy_fails_audit` at `chi^2 = 6.904`,
`permutation_p = 0.0292` (`n_permutations = 10000`, `seed = 20260523`,
`min_expected = 2.655 < 5`, `min_occupied_bin_count = 6 > 2`). The
|L| sidecar likewise fell short of an independent loud-signal trigger
(`chi^2 = 4.465`, `permutation_p = 0.0741`, report-only). The energy
shadow does NOT carry stability information beyond labeling the
v0.5a `(m_3 < 1, z_0 < 0.3)` branch on supp-B at the registered floor.

Audience: v0.6b runner; v0.6c form-lock author (if v0.6b passes);
paper-side reviewer of the alignment-breaking transition.

Companions:

- `kfacet_v06a_energy_quartile_audit_form.md` -- v0.6a parent audit
  with `energy_quartile_passes_audit_alignment_warning` verdict that
  licensed this alignment-breaking child.
- `kfacet_v06_mechanism_preregistration.md` -- v0.6 parent registration
  (operational definitions of E and |L|, disallowed-feature list,
  inheritance discipline).
- `kfacet_v05a_branch_map_form.md` -- v0.5a branch hash; the
  alignment-broken stratification axis.
- `kfacet_v05b_branch_predictor_form.md` -- v0.5b held-out predictor
  pattern; inheritance source for the sparse-cell fallback discipline.

Frame: v0.6a passed chi-squared (33.70 vs 11.34) but the
alignment-tightness scalar fired hard (0.956 vs 0.8 threshold). Q4
holds 65 of 68 rows in the v0.5a `(m_3 < 1, z_0 < 0.3)` branch
(95.6%), so the in-sample energy stratification is plausibly the
same bin-locality phenomenon v0.5b exposed dressed in energy clothing.
v0.6b breaks the alignment by **conditioning on branch**: within the
v0.5a stable branch alone (113 rows / 63 S / 50 U), does Q_E carry
stability information?

## Verdict (Landed, 2026-05-24)

Receipt:
`results/isotrophy/k-facet-v06b-within-branch-energy/manifest.json`.

The within-branch audit cleared all sanity gates and triggered the
sparse-cell fallback to an exact permutation test:

```text
stratum rows:        113  (S = 63, U = 50)
sanity_check:        PASS  (observed N, S, U match locked counts)

primary E audit:
  contingency (Q_E, S/U within branch):
    Q1:  N=0    (Q1 entirely outside the stratum -- tightest
                 catalog orbits all sit in other branches)
    Q2:  N=6    S=2,  U=4   S_frac=0.333  expected_S=3.35  expected_U=2.65
    Q3:  N=42   S=18, U=24  S_frac=0.429  expected_S=23.42 expected_U=18.58
    Q4:  N=65   S=43, U=22  S_frac=0.662  expected_S=36.24 expected_U=28.76

  chi^2:                     6.904228
  occupied_bin_count:        3
  min_occupied_bin_count:    6   (>= 2, fallback test branch active)
  min_expected_cell:         2.655   (< 5  -> permutation test branch fires)
  test_branch_taken:         permutation
  permutation_seed:          20260523
  n_permutations:            10000
  permutation_p_value:       0.0292
  threshold:                 p <= 0.01
  verdict:                   within_branch_energy_fails_audit
```

The within-branch direction is monotone in the same direction as the
catalog-wide v0.6a finding (Q2: 33% S -> Q3: 43% S -> Q4: 66% S),
but the permutation p-value of 0.0292 does not clear the registered
p <= 0.01 floor. The audit is a clean fail at the pre-registered
threshold.

The |L| within-branch sidecar (report-only, no verdict claim):

```text
sidecar |L| audit:
  contingency (Q_|L|, S/U within branch):
    Q1:  N=66   S=41, U=25  S_frac=0.621  expected_S=36.80 expected_U=29.20
    Q2:  N=45   S=22, U=23  S_frac=0.489  expected_S=25.09 expected_U=19.91
    Q3:  N=2    S=0,  U=2   S_frac=0.000  expected_S=1.12  expected_U=0.88
    Q4:  N=0    (Q4 entirely outside the stratum)

  chi^2:                     4.464524
  test_branch_taken:         permutation
  permutation_p_value:       0.0741
  threshold:                 p <= 0.01  (would be needed to trigger
                              a fresh |L| form lock as v0.6c)
  status:                    report_only; |L| does NOT meet the
                              loud-signal threshold for a fresh
                              registration.
```

Structural reading:

> Within the v0.5a stable branch, the within-branch energy quartile
> distribution is dominated by Q4 (m_3 = 0.4, 52 of 65 Q4 rows)
> and Q3 (mid-m_3 = {0.5..0.8}); Q1 is empty within the branch
> (Q1 collects the tightest catalog orbits, all in other branches).
> Energy is essentially a 1-to-1 label for m_3 sub-bin inside this
> stratum, and the residual within-branch stratification of S/U is
> below the registered chi-squared / permutation floor at p = 0.01.
> The v0.6a in-sample positive (chi^2 = 33.70) was, as the alignment
> warning anticipated, dominated by branch-shadow content. Energy
> does not carry stability information beyond what the v0.5a branch
> hash already provides on this catalog.

Per-other-branch diagnostic audits (3 non-stratum branches, all
report-only) and the within-branch Q_E x m_3 and Q_E x Q_|L| joint
tables are emitted on the receipt.

Receipts:

```text
results/isotrophy/k-facet-v06b-within-branch-energy/manifest.json
results/isotrophy/k-facet-v06b-within-branch-energy/per_row_table.csv
results/isotrophy/k-facet-v06b-within-branch-energy/contingency_table_within_branch_E.csv
results/isotrophy/k-facet-v06b-within-branch-energy/contingency_table_within_branch_L.csv
results/isotrophy/k-facet-v06b-within-branch-energy/within_branch_Q_E_by_m3.csv
results/isotrophy/k-facet-v06b-within-branch-energy/within_branch_Q_E_by_Q_L.csv
scripts/v06b_within_branch_energy_audit.py
```

## What v0.6b Is

v0.6b is a **catalog-side audit**, NOT a predictor (continues v0.5a /
v0.6a audit-not-predictor framing):

```text
Stratum:      v0.5a (m_3 < 1, z_0 < 0.3) branch only (113 rows).
Test:         chi-squared independence of Q_E vs S/U within the stratum.
Default df:   occupied_Q_E_bins - 1.
Fallback:     exact permutation test if any expected occupied cell < 5;
              within_branch_energy_inconclusive_sparse verdict if any
              occupied bin has < 2 rows.
Critical:     p <= 0.01 (chi-squared or permutation, depending on which
              fires).

Pass:         within_branch_energy_passes_audit
              -> energy carries within-branch stability information;
              license v0.6c as held-out predictor.

Fail:         within_branch_energy_fails_audit
              -> the v0.6a signal was bin-locality dressed in energy
              clothing; energy-shadow-as-mechanism sub-question closes.

Sparse:       within_branch_energy_inconclusive_sparse
              -> the within-branch Q_E partition is too thin for an
              honest test; v0.6c may register a power-conservative
              alternative (e.g., median-split E within branch).
```

A pass licenses v0.6c (separately-registered held-out predictor with
appropriate alignment-broken partition). A fail closes the
energy-shadow-as-mechanism question. A sparse-inconclusive verdict
licenses neither and may motivate a re-registered v0.6b' with a
coarser binning.

## Why Within-Branch on (m_3<1, z_0<0.3)

The v0.5a branch contingency:

```text
branch_label                  N    S    U   S_fraction
m_3 >= 1, z_0 >= 0.3          87   18   69   0.2069
m_3 >= 1, z_0 <  0.3          17    5   12   0.2941
m_3 <  1, z_0 >= 0.3          56   11   45   0.1964
m_3 <  1, z_0 <  0.3         113   63   50   0.5575    <-- v0.6b stratum
```

The stable branch (m_3 < 1, z_0 < 0.3) has:

- the largest N (113 rows),
- the only S majority on the catalog (S_fraction = 0.5575),
- the highest alignment with v0.6a's Q4 (95.6%).

Conditioning on this branch is the alignment-break with the highest
statistical power to ask the substantive within-branch question. The
other three branches have:

- (m_3 >= 1, z_0 >= 0.3): N = 87, S_frac = 0.2069 (U-skewed, 18 S /
  69 U, may have low S cell counts in some Q_E quartiles);
- (m_3 >= 1, z_0 < 0.3): N = 17 (too small for a four-bin chi-squared);
- (m_3 < 1, z_0 >= 0.3): N = 56, S_frac = 0.1964 (U-skewed).

Per-other-branch audits are recorded as report-only diagnostics on the
v0.6b receipt but are NOT gating. If the post-receipt residuals motivate
a separately-registered within-branch audit on (m_3 >= 1, z_0 >= 0.3)
under a coarser binning, that gets its own form lock.

## The Locked Form

```text
stratum:       rows r with branch_label(r) == "m_3 < 1, z_0 < 0.3"
               (113 rows from v0.5a per_row_table.csv).

feature:       Q_E(r) per v0.6a operational definition, using the
               GLOBAL supp-B quartile cutpoints from v0.6a's receipt:
                 q25_E = -1.6446242186674882
                 q50_E = -1.2315294472828655
                 q75_E = -0.9947713598607784
               No re-binning within branch. This is critical: v0.6b
               preserves v0.6a's feature definition; the within-branch
               Q_E distribution is whatever it is under the global
               cutpoints.

stability:     S/U per supp-B catalog (independent of Q_E).

contingency:   n_occupied_Q_E x 2 table over (Q_E, S/U) for the 113
               stratum rows.

primary test:  chi-squared statistic
                 chi^2 = sum over cells of (observed - expected)^2 /
                         expected
               where expected_{kj} = (row_k_total * col_j_total) / 113.

default df:    occupied_Q_E_bins - 1.

threshold:     p <= 0.01 (one-sided in the standard chi-squared
               upper-tail sense).
```

## Sparse-Cell Fallback Logic

Pre-registered, deterministic. Evaluated BEFORE the verdict is read.

```text
1. Compute expected cell counts:
     E_{kj} = (N_k * C_j) / 113
     where N_k = within-branch row count in Q_E bin k,
           C_j = within-branch row count in stability class j.

2. Compute min_occupied_bin_count:
     min over occupied Q_E bins of N_k.

3. Branch:
   a) If min_occupied_bin_count < 2:
        verdict = within_branch_energy_inconclusive_sparse
        do NOT report a p-value (the contingency is degenerate).

   b) Elif min over occupied cells of E_{kj} < 5:
        use exact permutation test:
          - permutation_seed = 20260523                  (locked)
          - n_permutations   = 10000                     (locked)
          - statistic        = chi-squared as computed above
          - p_value          = (# permutations with stat >= observed)
                               / n_permutations
        verdict at p_value <= 0.01.

   c) Else:
        use the asymptotic chi-squared(df) test:
          - df = occupied_Q_E_bins - 1
          - critical at p = 0.01:
              df = 1 -> 6.63
              df = 2 -> 9.21
              df = 3 -> 11.34
        verdict at chi^2 > critical(df).
```

Both the chi-squared and the permutation test compute the same test
statistic (chi-squared). The permutation test resamples S/U labels
across the 113 stratum rows uniformly without replacement; Q_E
assignments are fixed. This is a labels-permutation test of
independence under the null `Q_E independent of S/U within stratum`.

The locked permutation seed (`20260523`) is the same seed registered
for v0.5b's sidecar B random-half split. Re-using it preserves
audit-traceability across the v0.5/v0.6 chain.

## Pre-Audit Sanity Surface

Before v0.6b runs:

```text
1. v0.6a verdict must be on file:
     verdict in {energy_quartile_passes_audit,
                 energy_quartile_passes_audit_alignment_warning}.
   If v0.6a's verdict is energy_quartile_fails_audit, v0.6b is NOT
   licensed under this form lock; the chapter closes on E.

2. v0.5a per_row_table.csv must be present and readable.

3. Stratum row count must equal 113.
   If different, block and inspect (could indicate a parser or
   branch-assignment drift).

4. Stratum S/U counts must equal (63, 50).
   Same gate as #3.

5. Q_E values for the 113 stratum rows must be reproducible from the
   v0.6a runner under the locked global cutpoints. Per-row Q_E
   residual must be 0 (exact match) against the v0.6a per_row_table.
```

Sanity surface receipts are emitted regardless of verdict outcome.

## Non-Circularity Audit

```text
Within-branch conditioning uses the v0.5a branch_label only as a
stratification variable. The stratum membership is determined by
catalog parameters (m_3, z_0) alone; it does not inspect Q_E or S/U.

Q_E(r) is a function of:
  - the published mass triple (m_1, m_2, m_3),
  - the published initial conditions (r_i, v_i for i = 1..3),
  - the three-body Hamiltonian,
  - the v0.6a global quartile cutpoints (computed from the supp-B
    catalog E distribution BEFORE any S/U inspection).

Q_E(r) is NOT a function of:
  - the row's stability label,
  - the row's branch_label (Q_E depends on E only, not on
    (m_3, z_0) bit indicators).

The S/U label is supplied as the independent test column.

Conditional independence under the null:
  H_0: P(S | Q_E, branch=(m_3<1, z_0<0.3)) = P(S | branch=(m_3<1, z_0<0.3))
  H_1: P(S | Q_E, branch=(m_3<1, z_0<0.3)) != P(S | branch=...).

The non-circularity provenance is the same as v0.6a: catalog-only
features paired with the held-back stability label.
```

The within-branch test does NOT re-use S/U information beyond what
v0.6a already used. The branch conditioning is a stratification choice
that uses (m_3, z_0) only.

## Free Parameter Count

```text
Number of free parameters fitted to the data: 0
```

The stratum definition (v0.5a (m_3<1, z_0<0.3) branch), the feature
(Q_E with v0.6a global cutpoints), the test statistic (chi-squared),
the df formula (occupied_bins - 1), the fallback logic, the
permutation seed, the permutation count, and the critical p-value are
all locked pre-data.

The global Q_E cutpoints were determined in v0.6a from the full supp-B
E distribution, independently of the within-branch sub-table.

## Sidecar B: Within-Branch |L| Audit (Report-Only)

Under the same form shape, the v0.6b runner ALSO emits a within-branch
|L| audit:

```text
feature:        Q_|L|(r) per v0.6a operational definition, using the
                GLOBAL supp-B quartile cutpoints from v0.6a's receipt:
                  q25_L = 0.11747388134200563
                  q50_L = 0.2645346632690697
                  q75_L = 0.4190703676328425

stratum:        same 113-row v0.5a (m_3<1, z_0<0.3) branch.

contingency:    n_occupied_Q_|L| x 2 over (Q_|L|, S/U).

test:           same as primary (chi-squared with sparse-cell fallback
                to permutation or inconclusive_sparse).

REPORT ONLY:    NO VERDICT IS CLAIMED FROM THE SIDECAR.
                The sidecar's test statistic, p-value, and contingency
                are recorded on the receipt for diagnostic and motivation
                purposes only.

re-registration: if v0.6b primary fails (within_branch_energy_fails_audit)
                but the within-branch |L| sidecar shows a strong test
                statistic at p <= 0.01, the within-branch |L| audit MUST
                be separately registered as a fresh v0.6b' (or v0.6c)
                form lock with its own paper-side document, signed off
                before any further compute on |L|.
```

## Reserved Diagnostics (Not Gating)

Beyond the primary and the sidecar, the v0.6b runner emits per-row
and aggregated diagnostics for v0.6c form-lock design only:

```text
per-other-branch audits (m_3 >= 1, z_0 >= 0.3 and others):
  - chi-squared statistic and contingency
  - sparse-cell status
  - REPORT-ONLY, no verdict claim.

Within-branch Q_E x m_3 joint table:
  - tells whether within-branch Q_E is dominated by a single m_3 sub-bin.
  - if it is, v0.6c's partition design must account for it.

Within-branch Q_E x Q_|L| joint table:
  - tells whether E and |L| are tightly coupled within the stratum.
  - input to a possible joint (E, |L|) within-branch registration.

Per-row table emitted under v0.6b:
  (label, m_3, z_0, E, |L|, Q_E, Q_|L|, branch_label, in_v06b_stratum,
   stability)
```

## Edge Cases (Pre-Registered)

1. **Stratum count drift (N != 113, S != 63, U != 50)**: block v0.6b
   runner; inspect for parser/branch-assignment drift before re-running.

2. **All 4 Q_E quartiles occupied within branch but one has count <
   2**: triggers `within_branch_energy_inconclusive_sparse`. The
   pre-registered re-registration response is a v0.6b' with median-split
   (2-bin) Q_E or a constant-Q_E vs other binning that ensures all
   occupied bins have >= 2 rows.

3. **Two occupied Q_E quartiles within branch (Q_E distribution
   collapsed)**: chi-squared with df = 1, critical 6.63. Run normally.
   Receipt records the bin collapse.

4. **One occupied Q_E quartile within branch**: degenerate. Verdict
   `within_branch_energy_inconclusive_sparse`. Cannot distinguish E
   stratification within the branch under this binning.

5. **All sparse-cell expected counts < 5 but all occupied bins have
   >= 2 rows**: triggers the permutation test branch. The exact
   permutation p-value is the verdict-bearing statistic.

6. **Permutation test produces p <= 1 / n_permutations**: report
   p_value as "<= 1 / n_permutations" (a left-bound on the tail).
   This is rare given n_permutations = 10000; if it occurs, the
   audit passes at the registered threshold.

7. **The v0.6a verdict was energy_quartile_fails_audit**: v0.6b is
   NOT licensed under this form lock. The chapter closes on E.

## Pre-Mortem Expectation

The within-branch Q_E distribution has NOT been computed before
lock-in. The audit runs blind.

Expected published statements:

> v0.6b tests whether the v0.5a (m_3<1, z_0<0.3) branch's energy
> quartile carries stability information after conditioning on branch.
> The verdict is one of `within_branch_energy_passes_audit`,
> `within_branch_energy_fails_audit`, or
> `within_branch_energy_inconclusive_sparse` per the locked threshold
> and sparse-cell fallback.

Pre-mortem note: the v0.6a alignment-tightness reading of 0.956 makes
the most likely outcome `within_branch_energy_fails_audit` —
energy may not separate S/U once the branch is held fixed. If that
lands, the chapter closes on a sharp methodological statement:
*the energy shadow stratifies the catalog by labeling the v0.5a
branch, not by carrying independent within-branch stability content.*
A pass would be the more interesting outcome and would license v0.6c.

## Lock-In Statement

This audit form is committed before:

- the within-branch Q_E distribution has been computed,
- the within-branch S/U cell counts have been computed,
- the within-branch |L| sidecar test statistic has been computed,
- the per-other-branch diagnostic tables have been computed.

Any change to the stratum definition, feature, cutpoints, df formula,
sparse-cell threshold, permutation seed, permutation count, or
critical p-value after the v0.6b runner is executed is a
re-registration, not a refinement.

Implementation may proceed against this form lock using:

```text
results/isotrophy/k-facet-v06a-energy-quartile-audit/manifest.json
  (per-row table with E, |L|, Q_E, Q_|L|, branch_label, stability)
results/isotrophy/k-facet-v05a-branch-map/per_row_table.csv
  (branch_label provenance for the stratum definition)
```

as the sole inputs. No new dynamical compute is required.

## Receipt Schema (Planned)

```text
results/isotrophy/k-facet-v06b-within-branch-energy/
  manifest.json
    - mode = "v0.6b-within-branch-energy-audit"
    - form_lock = "internal/anniversary/kfacet_v06b_within_branch_energy_audit_form.md"
    - input_v06a_manifest, input_v05a_per_row_table
    - stratum = "v0.5a (m_3 < 1, z_0 < 0.3) branch"
    - stratum_row_count
    - stratum_S_count, stratum_U_count
    - sanity_checks: {N, S, U, Q_E reproducibility}
    - quartile_cutpoints_used_E (copied from v0.6a)
    - quartile_cutpoints_used_L (copied from v0.6a)
    - within_branch_contingency_E:    n_occupied_Q_E x 2
    - within_branch_min_bin_count
    - within_branch_min_expected_cell
    - test_branch_taken:               "chi_squared" |
                                        "permutation" |
                                        "inconclusive_sparse"
    - chi_squared_E:                   test statistic (if applicable)
    - df_E:                            occupied_bins - 1 (if chi-squared)
    - critical_E:                      chi-squared critical (if applicable)
    - permutation_seed:                20260523 (if permutation)
    - n_permutations:                  10000 (if permutation)
    - p_value_E:                       chi-squared upper-tail p OR
                                        permutation p
    - verdict:                          within_branch_energy_passes_audit |
                                        within_branch_energy_fails_audit |
                                        within_branch_energy_inconclusive_sparse
    - sidecar_L_contingency:           same shape (report-only)
    - sidecar_L_test_branch_taken
    - sidecar_L_chi_squared_or_permutation_p
    - sidecar_L_status:                "report_only"
    - diagnostic_per_other_branch:     dict per non-stratum branch
    - diagnostic_within_branch_Q_E_x_m3
    - diagnostic_within_branch_Q_E_x_Q_L
  contingency_table_within_branch_E.csv
  contingency_table_within_branch_L.csv  (sidecar)
  per_row_table.csv
    (label, m_3, z_0, E, |L|, Q_E, Q_|L|, branch_label,
     in_v06b_stratum, stability)
```

## Implementation Plan

Suggested script:

```text
scripts/v06b_within_branch_energy_audit.py
```

Suggested command:

```powershell
python scripts\v06b_within_branch_energy_audit.py `
  --v06a-manifest results\isotrophy\k-facet-v06a-energy-quartile-audit\manifest.json `
  --v06a-per-row results\isotrophy\k-facet-v06a-energy-quartile-audit\per_row_table.csv `
  --out          results\isotrophy\k-facet-v06b-within-branch-energy
```

Expected runtime: seconds (chi-squared branch) or under a minute
(permutation branch with n_permutations = 10000 over a 113-row
stratum).

## Interpretation

If v0.6b passes (`within_branch_energy_passes_audit`):

> The v0.5a stable branch's energy quartile stratifies S/U beyond
> what the branch label alone explains. Energy carries within-branch
> stability information, and the v0.6a alignment-warning verdict
> reflects necessary-but-not-sufficient bin-locality. This licenses
> v0.6c: a separately-registered held-out predictor with a partition
> design informed by the v0.6b per-branch diagnostics.

If v0.6b fails (`within_branch_energy_fails_audit`):

> The energy quartile carries no stability information beyond labeling
> the v0.5a stable branch. The v0.6a in-sample chi-squared was bin-
> locality dressed in energy clothing. The energy-shadow-as-mechanism
> sub-question closes; the chapter records a clean
> conditional-independence result.

If v0.6b is sparse-inconclusive (`within_branch_energy_inconclusive_sparse`):

> The within-branch Q_E partition is too thin under the locked
> binning for an honest test. v0.6b' may register a coarser within-
> branch binning (median-split or quantile=2) before re-running.

## Doc Trail

- `kfacet_v06a_energy_quartile_audit_form.md` -- v0.6a parent with
  alignment-warning verdict.
- `kfacet_v06_mechanism_preregistration.md` -- v0.6 parent
  registration.
- `kfacet_v05a_branch_map_form.md` -- v0.5a stratification axis.
- `kfacet_v05b_branch_predictor_form.md` -- inheritance source for
  sparse-cell discipline and McNemar conventions.
- v0.6a receipt: `results/isotrophy/k-facet-v06a-energy-quartile-audit/manifest.json`.

---

Alignment-breaking within-branch audit. Chi-squared by default, exact
permutation at sparse cells, honest sparse-inconclusive verdict if
the within-branch partition is degenerate. |L| sidecar report-only.
