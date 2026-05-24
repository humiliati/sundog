# v0.6a Energy-Quartile Audit Form Lock (Univariate E)

Status: **VERDICT LANDED 2026-05-23**. The v0.6a univariate-energy-
quartile audit passes the registered chi-squared threshold but triggers
the registered alignment warning. The |L| quartile audit under the same
shape is recorded as a **report-only sidecar** and emits no independent
verdict.

Audience: v0.6a runner; v0.6b form-lock author; paper-side reviewer of
the v0.6 transition from parent registration to first audit.

Companions:

- `kfacet_v06_mechanism_preregistration.md` -- v0.6 parent registration
  (this form is option A of the parent's audit-form candidate list).
- `kfacet_v05_writeup.md` -- v0.5 chapter close (projection-limit) that
  opened v0.6.
- `kfacet_v05a_branch_map_form.md` -- v0.5a audit pattern (4-bucket
  contingency, chi-squared(df) falsifier, audit-not-predictor framing).
- v0.3 cross-m_3 receipts / invariant implementation -- E and |L|
  sentinel sanity-check provenance.

Frame: v0.6 asks whether a higher-dimensional catalog-coordinate
projection — orbit-level conserved quantities — carries held-out
stability. v0.6a starts with the **cleanest first conserved-quantity
audit**: total energy E quartiles. Per the parent registration's
candidate-form list, this is option A; B (|L| quartiles), C (joint
(E, |L|)), D (energy-ratio K/|V|), and E (angular-momentum-scaled) are
held back for separate registrations.

## Verdict (Landed, 2026-05-23)

Receipt:
`results/isotrophy/k-facet-v06a-energy-quartile-audit/manifest.json`.

The registered runner completed all sanity gates:

```text
rows:                  273  (S = 97, U = 176)
sanity_v03_sentinels:  PASS  (max |Delta E| = 0, max |Delta |L|| = 0)
bound_orbit_check:     PASS  (E < 0 for all 273 rows)
```

The primary E quartile audit:

```text
E cutpoints:      q25 = -1.6446242186674882
                  q50 = -1.2315294472828655
                  q75 = -0.9947713598607784

Q_E table:        Q1  N=69  S=21  U=48  S_frac=0.3043
                  Q2  N=68  S=12  U=56  S_frac=0.1765
                  Q3  N=68  S=21  U=47  S_frac=0.3088
                  Q4  N=68  S=43  U=25  S_frac=0.6324

chi^2_E:          33.70315814531264
p_value_df3:      2.2886715282626394e-07
critical:         11.34
alignment_E:      0.9558823529411765  (> 0.8 warning threshold)

verdict:          energy_quartile_passes_audit_alignment_warning
```

Interpretation: the energy quartile shadow is strongly associated with
S/U in-sample, but the association is not cleanly independent of the
v0.5a `(m_3, z_0)` branch shadow. In particular Q4 is dominated by the
`m3_lt_1__z0_lt_0p3` branch at 65/68 rows. Therefore v0.6b may NOT
inherit the default leave-one-m_3-bin-out partition; it must be
re-registered with an alignment-breaking partition (for example
leave-one-E-quartile-out, or a constant-m_3 subcatalog design).

The |L| sidecar is also loud but is not a verdict:

```text
|L| cutpoints:    q25 = 0.11747388134200563
                  q50 = 0.2645346632690697
                  q75 = 0.4190703676328425

Q_|L| table:      Q1  N=69  S=41  U=28  S_frac=0.5942
                  Q2  N=68  S=24  U=44  S_frac=0.3529
                  Q3  N=68  S=21  U=47  S_frac=0.3088
                  Q4  N=68  S=11  U=57  S_frac=0.1618

chi^2_|L|:        28.954252352605597
p_value_df3:      2.2895491439323422e-06
alignment_|L|:    0.9565217391304348
sidecar_status:   report_only
```

Implementation provenance: v0.4a/v0.5a receipts provide row identity,
stability, and branch metadata, but they do not carry all published IC
velocity fields required for E and |L|. The runner therefore re-reads
`docs/isotrophy/supplementary-B_piano-init-condit-3d.txt` for the
catalog ICs and joins the receipt metadata. This remains catalog-only
and non-circular. The registered v0.3 sentinel sanity check is recorded
against the pre-existing `scripts.isotrophy_workbench.invariant_record`
implementation because the cross_m_3 gate receipts present in this
workspace do not carry scalar E/|L| fields.

## Why Univariate E Over Other Audit Forms

Per the parent registration:

```text
A. univariate E quartiles:        SELECTED (this form lock).
   - Maximally non-circular: E is conserved orbit-level scalar from IC.
   - Paper-simple: 4 x 2 contingency, df = 3, critical = 11.34.
   - No reference scale required for quartiles.

B. univariate |L| quartiles:      first sidecar, report-only.
   - Also clean, but |L| carries more coordinate/gauge interpretive
     baggage than E. Held back as the first sidecar; if A fails but
     the B sidecar is loud, B is registered as a separate v0.6b.

C. joint (E, |L|) audit:           held back.
   - 16 bins / df ~= 15. Too easy to dilute on the first audit and
     too easy to explain post hoc. Promotion requires either A or B
     to show signal first.

D. energy-ratio K/|V| audit:       held back (v0.6b/v0.6c material).
   - Requires pinned virial-normalization reference scale.

E. angular-momentum-scaled audit:  held back (v0.6b/v0.6c material).
   - Requires pinned moment-of-inertia I_tot reference scale.
```

The locked form prioritizes paper-simplicity and reference-scale-free
non-circularity over discrimination power. If A fails and the B
sidecar shows no signal either, that is itself the chapter result on
energy and angular-momentum univariate shadows.

## What v0.6a Is

v0.6a is a **catalog-side audit**, NOT a predictor (inherits v0.5a's
audit-not-predictor framing):

```text
Test:         chi-squared independence of energy quartile Q_E vs S/U
df:           3
Critical:     chi-squared(3) at p = 0.01 = 11.34
Pass:         chi^2 >  11.34 and alignment <= 0.8
                -> energy_quartile_passes_audit
Warn pass:    chi^2 >  11.34 and alignment > 0.8
                -> energy_quartile_passes_audit_alignment_warning
Fail:         chi^2 <= 11.34 -> energy_quartile_fails_audit
```

A pass with alignment-tightness `<= 0.8` licenses v0.6b under the
default held-out discipline. A pass with alignment-tightness `> 0.8`
licenses only an alignment-breaking v0.6b re-registration. A fail closes
the energy-shadow sub-question; the |L| sidecar may motivate a separate
v0.6b on |L| under a fresh form lock.

## The Locked Form

```text
feature:      E(row) = total energy at the row's published initial
                       conditions.

E(row) = sum_{i=1..3} 0.5 * m_i * |v_i|^2
       - sum_{1 <= i < j <= 3} (G * m_i * m_j) / |r_i - r_j|

              with G = 1 (Li-Liao normalization) and (m_1, m_2, m_3) =
              (1, 1, m_3_row). Initial conditions read from the same
              supp-B parser output used by v0.4a and v0.5a.

binning:      quartiles over the supp-B catalog's 273-row E distribution.

cutpoints:    q25, q50, q75 of the supp-B E array, computed via
              numpy.quantile(E, [0.25, 0.50, 0.75], method='linear').

bin
assignment:   bin k for row r is the smallest k in {1, 2, 3, 4} such that
                E(r) <= cutpoint_{k+1}      (k = 1, 2, 3)
              or k = 4 if E(r) > q75.

              Equivalently:
                Q_E(r) = 1  iff  E(r) <= q25
                Q_E(r) = 2  iff  q25 < E(r) <= q50
                Q_E(r) = 3  iff  q50 < E(r) <= q75
                Q_E(r) = 4  iff  E(r) >  q75

              Right-closed lower bins; ties at exact cutpoints (numerically
              impossible at double precision but pre-registered for
              completeness) go to the lower bin.

contingency:  4 x 2 table over (Q_E, stability) with stability in {S, U}.

test:         chi-squared statistic
                chi^2 = sum over cells of (observed - expected)^2 / expected
              where expected_{kj} = (row_k_total * col_j_total) / N.

df:           3.

critical:     11.34 (chi-squared(3) at p = 0.01).

verdict:      energy_quartile_passes_audit
                iff chi^2 > 11.34 AND alignment-tightness <= 0.8
              energy_quartile_passes_audit_alignment_warning
                iff chi^2 > 11.34 AND alignment-tightness > 0.8
              energy_quartile_fails_audit
                iff chi^2 <= 11.34
```

Physical orientation: supp-B is all bound (E < 0 for all 273 rows;
sign(E) RETIRED on supp-B per parent registration's constant-feature
rule). Q_E = 1 corresponds to the **most-negative** (tightest, most
deeply bound) E quartile; Q_E = 4 corresponds to the **least-negative**
(loosest) E quartile.

## Pre-Audit Sanity Surface

Before v0.6a runs:

```text
1. Per-row E and |L| computation against the v0.4a manifest's 273 rows.
   Catalog-only; no orbit integration.

2. Sanity check against the 7 v0.3 cross-m_3 sentinel rows
   (O_50, O_62, O_67, O_434 at m_3=0.4;
    O_242, O_282, O_284 at m_3=1.0).
   Per-row residual must agree to < 1e-6 against the v0.3 cross-m_3
   receipt E and |L| values. Block v0.6a if this fails.
   If the local cross_m_3 receipts do not carry scalar E/|L| fields,
   the runner may use the pre-existing v0.3 invariant implementation
   as the reference, but MUST record that provenance on the receipt.

3. Bound-orbit check: E < 0 for all 273 rows. If any row has E >= 0,
   record on receipt and pause v0.6a for inspection. (Periodic
   orbits are bound by definition; an E >= 0 reading would indicate
   a parser or operational-definition bug.)
```

Sanity surface receipts are emitted regardless of audit outcome.

## Non-Circularity Audit

```text
E(row) is a function of:
  - the published mass triple (m_1, m_2, m_3),
  - the published initial conditions (r_i, v_i for i = 1..3),
  - the three-body Hamiltonian (kinetic + pairwise gravitational).

E(row) is NOT a function of:
  - the row's Floquet spectrum or monodromy operator,
  - the row's stability label S/U,
  - any v0.4a Pass 1 / Pass 2 classification,
  - the v0.5a branch hash output,
  - the K_fib tangent decomposition.

The supp-B stability label S/U is supplied as the independent test
column. The quartile cutpoints are computed from the supp-B E
distribution alone, BEFORE the S/U column is joined.

This is the same non-circularity provenance as v0.5a's branch hash:
catalog-only features paired with an independent stability label.
```

The cutpoints DO depend on the supp-B distribution (because they are
supp-B quartiles), but they are independent of the stability label.
This is the assumption-light alternative to physically-motivated
cutpoints, which would risk disguised tuning.

## Free Parameter Count

```text
Number of free parameters fitted to the data: 0
```

The feature (E), the binning rule (quartiles), the cutpoint method
(numpy linear-interpolation quantile), the bin-assignment convention
(right-closed lower bins, ties to lower bin), the df (3), and the
critical value (11.34) are all locked pre-data. The quartile cutpoints
are computed deterministically from the supp-B E distribution; they
are not fitted to the S/U column.

## Chi-Squared Degrees of Freedom

```text
df = (number of bins - 1) * (number of classes - 1)
   = (4 - 1) * (2 - 1)
   = 3
```

Reference distribution `chi-squared(3)`. Critical value `11.34` at
`p = 0.01`. Test statistic `chi^2 > 11.34` triggers
`energy_quartile_passes_audit`.

Wording note (inherited from v0.5a discipline): the pass condition is
`energy_quartile_passes_audit`, NOT "energy-shadow theorem confirmed".
A pass licenses a v0.6b predictor registration; it does not on its own
establish a predictive mechanism.

## Sidecar A: |L| Quartile Audit (Report-Only)

Under the same form shape, the v0.6a runner ALSO emits a |L| quartile
audit:

```text
feature:        |L|(row) = euclidean norm of the row's total angular
                  momentum vector at IC.

|L|(row) = ||sum_{i=1..3} m_i * (r_i x v_i)||

binning:        quartiles over the supp-B catalog's 273-row |L| array,
                same quantile method as the primary E form.

contingency:    4 x 2 (Q_|L|, stability).

test:           chi-squared statistic, df = 3.

REPORT ONLY:    NO VERDICT IS CLAIMED FROM THE SIDECAR.
                The sidecar's chi^2 and contingency are recorded on the
                receipt for diagnostic and motivation purposes only.

re-registration: if A fails (energy_quartile_fails_audit) but the |L|
                sidecar shows a chi^2 substantially above 11.34, the |L|
                quartile audit MUST be separately registered as a fresh
                v0.6b (or v0.6a') form lock with its own paper-side
                document, signed off before any further compute on the
                |L| projection.
```

The sidecar's purpose is to surface a "loud |L| signal that wasn't in
E" finding cheaply, without licensing it as a verdict under the
current registration. Any subsequent claim on |L| requires its own
lock-in statement.

## Reserved Diagnostics (Not Gating, Not Verdict-Bearing)

Beyond the locked primary and the sidecar, the v0.6a runner emits
per-row and aggregated diagnostics for v0.6b form-lock design only:

```text
sign(E)             [expected constant negative on supp-B; retired
                     per parent registration if confirmed.]
|E|                 [magnitude; useful for log-scale visualization.]
L_z = z-component of L
sign(L_z)           [chirality diagnostic; not gating.]
|L_z| / |L|         [pre-mortem flagged in parent registration as
                     diagnostic only.]
E quartile         x  m_3 axis joint table
E quartile         x  z_0 axis joint table
E quartile         x  v0.5a branch_label joint table
|L| quartile       x  v0.5a branch_label joint table
```

The last three tables address the parent registration's Risk 1
(alignment with v0.5b's (m_3, z_0) buckets). If the E or |L| quartiles
are tightly aligned with the v0.5a branch label, the audit pass does
NOT license confidence in held-out generalization, and v0.6b's
partition must be re-registered as something other than the default
leave-one-m_3-bin-out.

Specifically, the receipt MUST report:

```text
max over E quartile bin of (fraction of bin in any single (m_3, z_0)
                            branch_label bucket)
```

This single number is the alignment-tightness scalar. The parent
registration requires it on every v0.6 receipt; v0.6a is the first
form lock to enforce it.

## Edge Cases (Pre-Registered)

1. **Tied E values at cutpoints**: numerically impossible at double
   precision for distinct rows. Pre-registered convention: ties go to
   the lower bin (right-closed lower intervals).

2. **Quartile bin size variation**: with 273 rows divided into 4
   quartiles, bin sizes will be approximately {68 or 69 each}. The
   small imbalance is absorbed by the chi-squared expected-count
   calculation; no Yates' correction is applied.

3. **Expected count < 5 in any cell**: with 97 S / 176 U over 273 rows
   and 4 bins, expected S count per bin is ~24 and expected U count
   per bin is ~44. All cells well above the chi-squared assumption
   threshold. No correction needed.

4. **Failed sanity check on v0.3 cross-m_3 sentinels**: block v0.6a;
   record the discrepancy and fix the operational-definition
   implementation before the audit runs. This is a hard gate on the
   non-circularity provenance (we cannot claim E was independently
   computable from IC if the implementation disagrees with v0.3).

5. **Bound-orbit check failure (E >= 0 for any row)**: same gate as
   #4. Periodic orbits are bound; an E >= 0 reading indicates a bug.

6. **Alignment-tightness scalar > 0.8**: pass v0.6a with a flagged
   verdict `energy_quartile_passes_audit_alignment_warning`. v0.6b's
   partition MUST be re-registered with an alternative that breaks
   the alignment (leave-one-E-quartile-out, or a sub-catalog at
   constant m_3).

## Pre-Mortem Expectation (Pre-Run)

At lock-in, the per-quartile S/U counts had NOT been computed. The
audit ran blind.

The expected published statement, regardless of outcome:

> v0.6a tests whether the supp-B catalog's energy quartile Q_E
> stratifies stability. The verdict is reported as
> `energy_quartile_passes_audit` (chi^2 > 11.34),
> `energy_quartile_passes_audit_alignment_warning` (chi^2 > 11.34 AND
> max-bin-alignment > 0.8), or `energy_quartile_fails_audit` (chi^2
> <= 11.34) under the locked chi-squared(3) threshold at p = 0.01.

Risk-1 pre-mortem note from the parent registration: E and |L| may
track (m_3, z_0) on supp-B. If the alignment-tightness scalar
exceeds 0.8, the audit pass is bin-locality-suspect and v0.6b's
default partition does not inherit cleanly.

## Lock-In Statement

This audit form is committed before:

- per-row E or |L| has been computed for the 273 supp-B rows,
- the supp-B quartile cutpoints have been calculated,
- the per-bin S/U distribution has been examined,
- the |L| sidecar's chi-squared has been computed.

Any change to the feature, binning, cutpoint method, bin-assignment
convention, df, critical value, alignment-tightness threshold, or
sidecar shape after the v0.6a runner is executed is a re-registration,
not a refinement.

Implementation proceeded against this form lock using:

```text
results/isotrophy/k-facet-v04a-domain-map/manifest.json
  (row identity, m_3, stability/provenance metadata)
results/isotrophy/k-facet-v05a-branch-map/per_row_table.csv
  (per-row branch_label for the alignment check)
docs/isotrophy/supplementary-B_piano-init-condit-3d.txt
  (published z_0, v_x, v_y, v_z, T, stability IC fields)
```

as catalog-only inputs. No new dynamical compute is required; per-row E
and |L| computation is catalog-only.

## Receipt Schema (Landed)

```text
results/isotrophy/k-facet-v06a-energy-quartile-audit/
  manifest.json
    - mode = "v0.6a-energy-quartile-audit"
    - form_lock = "internal/anniversary/kfacet_v06a_energy_quartile_audit_form.md"
    - input_manifest_v04a, input_per_row_table_v05a
    - input_catalog_supplementary_b
    - quartile_method = "numpy_quantile_linear"
    - cutpoints_E:  {q25, q50, q75}
    - cutpoints_L:  {q25, q50, q75}                        (sidecar)
    - bin_assignment_convention = "right_closed_lower_ties_to_lower"
    - sanity_check_v03_sentinels:
        - per_row_residual_E_max
        - per_row_residual_L_max
        - pass_threshold = 1e-6
        - status = pass | fail
    - bound_orbit_check:
        - rows_with_E_ge_0
        - status = pass | fail
    - contingency_E:    4 x 2 (Q_E, S/U)
    - chi_squared_E:    test statistic
    - df_E:             3
    - critical:         11.34
    - verdict:          energy_quartile_passes_audit |
                        energy_quartile_passes_audit_alignment_warning |
                        energy_quartile_fails_audit
    - alignment_tightness_scalar_E:
        max over Q_E of (fraction in any single branch_label bucket)
    - alignment_tightness_scalar_L:                        (sidecar)
    - contingency_L:    4 x 2 (Q_|L|, S/U)                 (sidecar)
    - chi_squared_L:    test statistic                     (sidecar)
    - df_L:             3                                  (sidecar)
    - sidecar_status:   report_only
    - diagnostic_tables_emitted: [E_vs_m3, E_vs_z0, E_vs_branch,
                                  L_vs_branch]
  contingency_table_E.csv
  contingency_table_L.csv                                  (sidecar)
  per_row_table.csv
    (label, m_3, z_0, E, |L|, Q_E, Q_|L|, branch_label, stability)
```

## Implementation / Command

Suggested script:

```text
scripts/v06a_energy_quartile_audit.py
```

Suggested command:

```powershell
python scripts\v06a_energy_quartile_audit.py `
  --manifest results\isotrophy\k-facet-v04a-domain-map\manifest.json `
  --v05a-table results\isotrophy\k-facet-v05a-branch-map\per_row_table.csv `
  --out results\isotrophy\k-facet-v06a-energy-quartile-audit
```

Observed runtime: seconds. No staged compute or operator interaction was
required.

## Interpretation

If v0.6a passes (chi^2 > 11.34, alignment-tightness <= 0.8):

> The supp-B catalog's energy quartile stratifies stability under a
> pre-registered univariate audit. The energy shadow carries
> in-sample stability information that the v0.5a 2-bit branch hash did
> not fully capture, AND the energy quartiles are not tightly aligned
> with v0.5b's bin-locality failure mode. This licenses v0.6b: a
> separately-registered held-out energy-quartile predictor with the
> v0.5b discipline.

If v0.6a passes with alignment warning (chi^2 > 11.34,
alignment-tightness > 0.8):

> Pass-but-suspect. The energy quartiles track (m_3, z_0) tightly on
> supp-B, so the in-sample signal is plausibly the same bin-locality
> phenomenon v0.5b exposed. v0.6b's partition must be re-registered to
> break the alignment.

If v0.6a fails (chi^2 <= 11.34):

> The energy-shadow does not stratify stability on supp-B under the
> univariate quartile audit. The chapter sub-question on E closes; the
> |L| sidecar's chi-squared and the v0.6 mechanism family's remaining
> projections (|L| univariate, joint, energy-ratio, angular-momentum-
> scaled) are open for separate registration if any sidecar signal
> warrants pursuit.

## Doc Trail

- `kfacet_v06_mechanism_preregistration.md` -- parent registration.
- `kfacet_v06b_within_branch_energy_audit_form.md` -- v0.6b
  alignment-breaking child (within-branch audit on the v0.5a
  (m_3<1, z_0<0.3) stratum, licensed by this form's
  `energy_quartile_passes_audit_alignment_warning` verdict).
  **Verdict landed 2026-05-24: `within_branch_energy_fails_audit`**
  (chi^2 = 6.90, permutation p = 0.029); confirms the v0.6a alignment
  warning was load-bearing.
- `kfacet_v05_writeup.md` -- v0.5 chapter close (predecessor).
- `kfacet_v05a_branch_map_form.md` -- v0.5a audit pattern (form template).
- v0.3 cross-m_3 receipts / invariant implementation -- E, |L|
  sentinel sanity-check provenance.

---

Audit floor, not predictor. Univariate energy quartile, df = 3, critical
11.34. |L| quartile sidecar report-only. Alignment-tightness scalar
guards against v0.5b's bin-locality failure mode. Verdict landed
2026-05-23 as `energy_quartile_passes_audit_alignment_warning`,
licensing v0.6b as an alignment-breaking within-branch audit. v0.6b
landed 2026-05-24 as `within_branch_energy_fails_audit`, confirming
the alignment warning was load-bearing: the v0.6a in-sample chi^2 was
dominated by branch-shadow content.
