# v0.7a' Velocity-Fraction Audit, Restricted-Scope (Pre-Close Confirmation)

Status: **VERDICT LANDED 2026-05-24**. Verdict:
**`velocity_fraction_restricted_passes_audit`** [PASS]. The
restricted-scope audit on the 250 analyzable rows produces
`chi^2 = 16.425` vs critical `11.34` (df = 3, `p = 9.28e-4`) with
`alignment_tightness = 0.698`, below the 0.8 warning threshold. The
velocity-fraction shadow stratifies S/U on the analyzable sub-catalog
**independently of the v0.5a branch hash**. The z-fraction sidecar
shows the same U-shape direction at `p = 0.021`, below the loud-signal
threshold (no fresh circularity audit triggered).

This document locks the v0.7a' minimal-scoped chi-squared audit on the
restricted catalog (the 250 rows analyzable under v0.7a's R2.A
discipline) with **explicit Pass / Partial / Fail criteria tied to the
v0.5a branch explanation**. It is the single pre-registered
confirmation run that licenses the v0.7 chapter close.

## Verdict (Landed, 2026-05-24)

Receipt:
`results/isotrophy/k-facet-v07a-prime-restricted-scope/manifest.json`.

```text
verdict:                  velocity_fraction_restricted_passes_audit  [PASS]
scope:                    250 analyzable rows (v0.7a integration_blocked == False)
S_count / U_count:        87 / 163

attrition disclosure (permanent on this receipt):
  v0.7a full catalog rows:               273
  v0.7a integration_blocked_count:        23  (8.42%)
  v0.7a additional sanity_failed_count:   11
  v0.7a total data-integrity:             34 / 273 = 12.5%
  v0.7a' restricted scope:               250 rows
```

Primary vf quartile contingency (re-binned within the 250-row subset):

```text
Q_vf  N    S    U    S_fraction  chi2_contrib
1     63   31   32   0.4921       5.76   <-- gamma_1 mostly positional
2     62   11   51   0.1774       7.95   <-- mixed, slightly velocity-leaning
3     62   18   44   0.2903       0.91
4     63   27   36   0.4286       1.80   <-- gamma_1 mostly velocity
                                  -----
                       chi^2 =   16.43
```

```text
chi^2:                    16.425218
test_branch_taken:        chi_squared  (min_expected = 21.6, well above
                                        the asymptotic floor 5)
df:                       3
critical (p=0.01):        11.34
p_value (chi-squared(3)): 9.276e-4
alignment_tightness:      0.6984  (<= 0.8 warning threshold)
```

z-fraction sidecar (D1+A; report-only):

```text
Q_z  N    S    U    S_fraction
1    63   26   37   0.4127       1.16
2    62   12   50   0.1935       6.52   <-- dominant chi^2 contribution
3    62   27   35   0.4355       2.09
4    63   22   41   0.3492       0.00
                                 -----
                       chi^2 =   9.77
```

```text
chi^2:                    9.772511
p_value (chi-squared(3)): 2.060e-2
alignment_tightness:      0.5000
sidecar_status:           report_only
indicative_verdict_if_promoted: velocity_fraction_restricted_fails_audit
                          (does NOT meet the loud-signal threshold at
                           p <= 0.01; no fresh circularity audit
                           triggered.)
```

**Structural reading:**

> The Floquet velocity-fraction direction shadow stratifies S/U on
> the 250 analyzable supp-B rows in a **non-monotone U-shape**:
> Q1 (gamma_1 mostly positional, vf <= q25) holds 49.2% S; Q2
> (mixed, slightly velocity-leaning) holds 17.7% S; Q3 holds 29.0%
> S; Q4 (gamma_1 mostly velocity, vf > q75) holds 42.9% S. The
> "pure" direction ends (Q1 and Q4) both concentrate stable orbits;
> the mixed/slightly-velocity-leaning Q2 concentrates unstable
> orbits. Alignment-tightness against the v0.5a branch hash is
> 0.698, below the 0.8 warning threshold -- the signal is NOT a
> branch-shadow re-projection. This is the FIRST low-dimensional
> projection in the four-chapter v0.4 + v0.5 + v0.6 + v0.7 audit
> chain to produce a stability signal that is statistically
> significant AND not bin-locality-aligned with prior chapters'
> branch-hash content.

**Catalog-wide caveat:** the v0.7a catalog-wide audit was
integration-attrited at 8.42% (23 blocked + 11 sanity-failed = 34/273).
The restricted-scope PASS holds on the 250 analyzable rows ONLY. The
attrited 23 rows cluster at long-period high-m_3 (m_3 = 1.5/1.6/1.7),
where the variational equation at rtol = 1e-12 hits the DOP853
step-size floor. Generalizing this result to the full supp-B catalog
requires either (a) a substrate-level precision relaxation pre-
registration (a separate audit family), or (b) a different orbit
representation that avoids the variational-precision wall. Neither
is licensed under this form lock.

Per the locked outcome interpretation, this verdict closes the v0.7
chapter as:

> The Floquet velocity-fraction mechanism produced a positive signal
> on the analyzable sub-catalog (`chi^2 = 16.43`, `p = 9.3e-4`,
> alignment 0.70), but cannot be evaluated on the full 273-row
> catalog at the locked variational precision. The attrited 23 rows
> (high-m_3 long-period regime) limit the result's catalog-wide
> generalizability. The chapter closes with a **qualified positive**
> in the orbit-dynamics shadow family: the first such positive in
> the v0.4-v0.5-v0.6-v0.7 chain, and the first to be branch-
> independent.

Audience: v0.7a' runner; v0.7 chapter-close author; paper-side reviewer
of the integration-attrited close.

Companions:

- `kfacet_v07a_velocity_fraction_audit_form.md` -- v0.7a parent form
  + verdict `velocity_fraction_blocked_integration_attrition`. The
  R1 + R2.A amendments and the locked non-circularity sentence carry
  forward unchanged.
- `kfacet_v05a_branch_map_form.md` -- v0.5a branch hash (the
  alignment-tightness comparison axis).
- `kfacet_v06a_energy_quartile_audit_form.md` -- precedent for the
  alignment-tightness threshold tree (0.8 warning, 0.95 severe).

Frame: v0.7a's 273-row audit was integration-attrited at 8.42%
(`integration_blocked_count = 23 > 14`). The catalog cannot be
honestly evaluated at variational precision 1e-12, but the 250
analyzable rows DO carry computed velocity_fraction values with sd =
0.144 (well above the constant-feature retirement floor). v0.7a' is
the single minimum-scope pre-registered confirmation that asks:
**on the analyzable subset, does the velocity-fraction shadow stratify
S/U at the registered floor, and does it do so independently of the
v0.5a branch hash?**

The Pass / Partial / Fail outcome tree below is locked **before** the
chi-squared, alignment-tightness, or sidecar p-value are read. All
three outcomes close the v0.7 chapter; the verdict's specific shape
determines how the chapter close is worded.

## What v0.7a' Is

v0.7a' is a **catalog-side audit on a restricted scope**, NOT a
predictor:

```text
Scope:        the 250 analyzable rows from v0.7a's per_row_table.csv
              (rows with integration_blocked == False).

Excluded:     23 integration-blocked rows + 11 R1-sanity-failed
              rows (still in per_row_table but flagged). The
              attrition is RECORDED on every receipt; it is NOT
              re-evaluated against the 5% gate (v0.7a's gate already
              fired and is the reason this scope is restricted).

Feature:      velocity_fraction(row), already computed in v0.7a per
              the D5 + B operational definition. No new variational
              compute. Existing per_row_table.csv is the input.

Binning:      RE-COMPUTED quartiles within the 250-row subset using
              numpy.quantile(method='linear'). This is necessary
              because v0.7a's quartile assignments were never written
              (the original runner blocked at the verdict step). The
              re-binning is mechanical and uses the same quartile
              method.

Contingency:  4 x 2 over (Q_vf, S/U), df = 3.

Critical:     chi-squared(3) at p = 0.01 = 11.34.

Alignment
guard:        max over Q_vf of (fraction-in-any-single-v0.5a-branch_
              label-bucket). Thresholds inherited from v0.6a/v0.7a:
              0.8 (warning) and 0.95 (severe).

Sparse-cell
fallback:     v0.6b inheritance verbatim (seed = 20260523,
              n_permutations = 10000). For quartile binning on 250
              rows with 87 S / 163 U, the chi-squared branch is
              overwhelmingly likely; fallback documented for
              completeness.

Sidecar:      z_fraction (D1+A direction-projection feature),
              quartile-binned within the 250-row subset.
              REPORT-ONLY (no verdict claim).
```

## Pass / Partial / Fail Outcome Tree (Locked Pre-Data)

The verdict is determined by the joint state of `chi^2` and the
alignment-tightness scalar against the v0.5a branch_label. The tree
is locked before either quantity is computed:

```text
verdict tree:

  if min_occupied_bin_count < 2:
      verdict = velocity_fraction_restricted_inconclusive_sparse
      (sparse contingency on the 250-row subset; structurally
       improbable for quartile binning but recorded for completeness)

  elif chi^2_vf > 11.34:
      if alignment_tightness <= 0.8:
          verdict = velocity_fraction_restricted_passes_audit
          [PASS]
      elif alignment_tightness <= 0.95:
          verdict = velocity_fraction_restricted_passes_audit_alignment_warning
          [PARTIAL]
      else:
          verdict = velocity_fraction_restricted_passes_audit_severe_alignment
          [PARTIAL-severe]

  else:
      verdict = velocity_fraction_restricted_fails_audit
      [FAIL]
```

### Outcome interpretation (locked)

```text
[PASS] velocity_fraction_restricted_passes_audit:
  On the 250-row analyzable subset, vf chi-squared exceeds 11.34
  AND vf is not bin-aligned with the v0.5a branch hash (alignment
  <= 0.8). The velocity-fraction shadow carries stability information
  independent of the v0.5a branch hash on the analyzable subset.
  The v0.7 chapter closes as: "the Floquet velocity-fraction
  mechanism produced a positive signal on the analyzable
  sub-catalog, but cannot be evaluated on the full 273-row catalog
  at the locked variational precision. The attrited 23 rows
  (high-m_3 long-period regime) limit the result's catalog-wide
  generalizability."

[PARTIAL] velocity_fraction_restricted_passes_audit_alignment_warning
         (alignment in (0.8, 0.95])
or
[PARTIAL-severe] velocity_fraction_restricted_passes_audit_severe_alignment
                 (alignment > 0.95):
  vf stratifies in-sample on the analyzable subset, but the
  stratification tracks the v0.5a branch hash tightly. This is the
  same bin-locality phenomenon v0.6a exposed for the conserved-
  quantity shadow. The v0.7 chapter closes as: "the Floquet
  velocity-fraction shadow does not carry stability information
  beyond labeling the v0.5a branch on the analyzable sub-catalog;
  the in-sample positive is branch-shadow content re-projected onto
  orbit-dynamics coordinates."

[FAIL] velocity_fraction_restricted_fails_audit:
  vf chi-squared does not exceed 11.34 on the analyzable sub-catalog.
  The v0.7 chapter closes as: "the Floquet velocity-fraction shadow
  does not stratify stability on the analyzable sub-catalog; the
  joint v0.4 + v0.5 + v0.6 + v0.7 envelope is sealed against
  catalog-coordinate and orbit-dynamics low-dimensional shadows on
  this catalog."
```

**All three outcomes close the v0.7 chapter and the isotrophy program.**
The Pass outcome adds the qualifier that the result is restricted to
the analyzable subset; the Partial and Fail outcomes are sharper
methodological statements.

## Attrition Disclosure (Recorded Permanently)

Every v0.7a' receipt records:

```text
v0.7a_full_catalog_row_count:            273
v0.7a_analyzable_row_count:              250
v0.7a_integration_blocked_count:          23
v0.7a_sanity_failed_count:                11
v0.7a_attrition_fraction:               0.0842
v0.7a_total_data_integrity_fraction:    0.125

v0.7a' restricted scope used:           250 rows (analyzable subset only)
v0.7a' S_count / U_count:                87 / 163
```

The attrition is permanent on the v0.7a' verdict. No amount of
chi-squared signal on the restricted subset can recover the
catalog-wide claim that v0.7a was registered to make. v0.7a' is
explicitly the **confirmation read** before chapter close, not a
re-attempt at the original catalog-wide audit.

## Why a Single Restricted-Scope Audit, Not v0.7a'' or v0.7b

The isotrophy program's pre-mortem discipline says: when an audit
attrites, the chapter closes; the program does not chase incremental
re-registrations down a sequence of progressively-relaxed scopes.

v0.7a' is registered as a SINGLE confirmation audit, not a sequence.
The verdict goes to one of {PASS, PARTIAL, PARTIAL-severe, FAIL,
INCONCLUSIVE_SPARSE}. Each verdict closes the v0.7 chapter. The
isotrophy program retires at end-of-v0.7 with the four-chapter
envelope (v0.4 / v0.5 / v0.6 / v0.7) as the publishable methodology
contribution.

No v0.7b held-out predictor will be registered under this scope.
No further v0.7 sub-registrations will be authored.

## Non-Circularity Audit

```text
The 250-row analyzable subset is the v0.7a output, filtered by
integration_blocked == False. The filter uses ONLY the integrator's
runtime status (Did compute_monodromy_vectorized raise?), which is
a function of the row's variational equation, not of the stability
label or any feature. The filter is information-disjoint from S/U.

The velocity_fraction values are the v0.7a output, computed under
the locked D5 + B operational definition + R1-amended sanity gates.
The non-circularity sentence and firewall from v0.7a carry forward
unchanged.

The re-binning to quartiles within the 250-row subset uses
numpy.quantile on the velocity_fraction column only. The S/U label
is the held-back test column, joined at the chi-squared step.
```

The v0.7a' alignment-tightness scalar against v0.5a branch_label is
the same comparison the v0.6a / v0.7a form locks used. No new
circularity risks.

## Free Parameter Count

```text
Number of free parameters fitted to the data: 0
```

The scope (analyzable subset), the binning rule (quartiles within
subset), the quantile method (numpy linear interpolation), the bin
assignment convention (right-closed lower), the df formula
(occupied_bins - 1), the critical value (11.34), the alignment-
tightness thresholds (0.8, 0.95), the sparse-cell fallback parameters,
and the Pass / Partial / Fail outcome tree are all locked pre-data.

## Implementation Plan

```text
Suggested script: scripts/v07a_prime_restricted_scope_audit.py

Inputs:
  results/isotrophy/k-facet-v07a-velocity-fraction-audit/per_row_table.csv
  results/isotrophy/k-facet-v05a-branch-map/per_row_table.csv
  (the v0.5a per_row_table is already joined into v0.7a's per_row_table
  via the branch_label column; reading v0.5a directly is optional.)

Output:
  results/isotrophy/k-facet-v07a-prime-restricted-scope/manifest.json
  results/isotrophy/k-facet-v07a-prime-restricted-scope/per_row_table.csv
  results/isotrophy/k-facet-v07a-prime-restricted-scope/contingency_table_vf.csv
  results/isotrophy/k-facet-v07a-prime-restricted-scope/contingency_table_z.csv

Expected runtime: seconds. No new variational compute.
```

## Lock-In Statement

This audit form is committed before the chi-squared, alignment-
tightness, sidecar chi-squared, or sidecar alignment-tightness has
been read for the 250-row analyzable subset.

Any change to the scope, feature, binning, df, critical value,
alignment-tightness thresholds, Pass/Partial/Fail outcome tree, or
sparse-cell fallback parameters after the v0.7a' runner is executed
is a re-registration, not a refinement.

This form lock is the LAST audit registration in the isotrophy
program. Any outcome closes the v0.7 chapter and licenses the
program retirement document.

## Doc Trail

- `kfacet_v07a_velocity_fraction_audit_form.md` -- v0.7a parent
  audit with attrition verdict.
- `kfacet_v07_mechanism_preregistration.md` -- v0.7 parent
  registration.
- `kfacet_v05a_branch_map_form.md` -- v0.5a branch hash (alignment
  comparison axis).
- `kfacet_v06a_energy_quartile_audit_form.md` -- alignment-tightness
  pattern.
- `kfacet_v07_writeup.md` -- v0.7 chapter close (qualified-positive
  on restricted domain; landed 2026-05-24).
- `kfacet_v08_mechanism_preregistration.md` -- v0.8 parent
  registration on Floquet direction-purity (opens v0.8).

---

Restricted-scope confirmation audit. Pass / Partial / Fail tree
locked. PASS landed at chi^2 = 16.43, p = 9.3e-4, alignment 0.698.
v0.7 chapter closed as qualified-positive on restricted domain;
v0.8 opened on Floquet direction-purity.
