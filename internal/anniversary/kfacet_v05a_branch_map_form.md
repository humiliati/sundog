# v0.5a Branch-Map Audit Registration (Branch-Shadow Audit)

Status: **VERDICT LANDED 2026-05-23**. Registered and computed against the
v0.4a manifest. Verdict: `branch_hash_passes_audit` with `chi^2 = 34.986`
vs critical `11.34` (`df = 3`, p ~= `1.23e-7`).
Audience: v0.5a runner; v0.5b reader of the verdict; paper-side reviewer of
the projection-limit framing.
Companion: `kfacet_v04_writeup.md` (chapter close that opened v0.5),
`kfacet_v04b_gamma3prime_form.md` (prior falsified baseline for non-circularity
provenance), `kfacet_v04a_domain_map_preregistration.md` (catalog source).
Frame: v0.5 stops asking what symmetry the row has and starts asking which
branch the row belongs to.

## Verdict (Landed, 2026-05-23)

The locked branch hash passes the audit:

```text
total rows:       273
observed S/U:     97 S / 176 U
active bits:      (m_3 < 1, z_0 < 0.3)
occupied buckets: 4
df:               3
chi^2:            34.986253
critical:         11.34   (chi-squared(3), p = 0.01)
p-value:          1.23e-7
verdict:          branch_hash_passes_audit
```

4x2 contingency table:

```text
branch_label                  N    S    U    S_fraction  chi2_contrib
m_3 >= 1, z_0 >= 0.3          87   18   69   0.2069       8.366
m_3 >= 1, z_0 <  0.3          17    5   12   0.2941       0.278
m_3 <  1, z_0 >= 0.3          56   11   45   0.1964       6.171
m_3 <  1, z_0 <  0.3         113   63   50   0.5575      20.171
```

Structural reading:

> The catalog-only branch shadow `(m_3 < 1, z_0 < 0.3)` carries stability
> information on supp-B. This is not yet a predictor; it is the v0.5a audit
> pass that licenses a separate v0.5b held-out prediction registration.

Receipts:

```text
results/isotrophy/k-facet-v05a-branch-map/manifest.json
results/isotrophy/k-facet-v05a-branch-map/contingency_table.csv
results/isotrophy/k-facet-v05a-branch-map/per_row_table.csv
scripts/v05a_branch_map_audit.py
```

## What v0.5a Is

v0.5a is a **catalog-side audit**, NOT a predictor:

```text
Test:         chi-squared independence of branch_label vs S/U on supp-B
df:           occupied_branch_count - 1
Critical:     chi-squared(df) at p = 0.01
Pass:         chi^2 >  critical -> branch hash carries stability information
Fail:         chi^2 <= critical -> branch shadow does not stratify S/U
```

A pass licenses v0.5b: a separately-registered predictor with held-out test.
A fail is the third structural-negative in the isotrophy program — recorded,
published, and the chain moves to the next mechanism family without contorting
the rule.

## The Locked Form (4-bit candidate, with constant-bit retirement)

```text
branch_label(row) = tuple of active nonconstant bits drawn from:
  b1 = (m_3 < 1)
  b2 = (z_0 < 0.3)
  b3 = (abs(v_z) < 1e-6)
  b4 = (m_3 * z_0^2 < 2)

retirement rule (deterministic, pre-data):
  any bit that is constant (all True or all False) on the catalog
  under test is RECORDED on the receipt but RETIRED from the active
  signature and from df.
```

The 4-bit candidate is locked before the audit runs. The retirement rule
is locked before the catalog degeneracy check (below). Active bit
selection is therefore mechanical, not curated.

## Catalog Degeneracy Check (Pre-Audit, Landed)

Per-bit truth counts over the v0.4a manifest's 273 supp-B rows:

```text
bit                          true / 273   status
b1 = (m_3 < 1)              169 / 273    ACTIVE
b2 = (z_0 < 0.3)            130 / 273    ACTIVE
b3 = (abs(v_z) < 1e-6)        0 / 273    RETIRED (constant FALSE)
b4 = (m_3 * z_0^2 < 2)      273 / 273    RETIRED (constant TRUE)
```

Active signature on supp-B: `(b1, b2)`. Constant bits b3, b4 are preserved
on the receipt for cross-substrate comparability (mesa, geometry, or
freeze-A choreographies may light up the retired bits and require their
own registration).

## Active Branch Buckets (Pre-Audit, Landed)

```text
bucket                              N
(m_3 >= 1, z_0 >= 0.3)             87
(m_3 >= 1, z_0 <  0.3)             17
(m_3 <  1, z_0 >= 0.3)             56
(m_3 <  1, z_0 <  0.3)            113
                                  ---
                                  273
```

All 4 buckets occupied -> occupied_branch_count = 4 -> df = 3.

## Primary Test

```text
H_0: branch_label and stability label are independent on supp-B.
H_1: branch_label and stability label are not independent.

Test statistic:    chi^2 over 4-bucket x 2-class contingency table
df:                occupied_branch_count - 1 = 3
Critical value:    chi-squared(3) at p = 0.01 = 11.34
verdicts (deterministic):
  chi^2 >  11.34:  branch_hash_passes_audit
  chi^2 <= 11.34:  branch_hash_fails_audit
```

## Interpretation

> *The branch hash is a coarse low-dimensional shadow of catalog parameters
> (m_3, z_0) only. The audit asks whether this pre-registered shadow
> stratifies the supp-B catalog's S/U distribution. A pass means the body's
> stability structure is partly visible in these branch coordinates; a fail
> joins the v0.4 Z_2-shadow result as another projection-limit negative.*

## Non-Circularity Audit

```text
b1, b2 are functions of catalog parameters (m_3, z_0) only:
  - m_3 is the published Li-Liao mass ratio (catalog input).
  - z_0 is the published initial-condition coordinate (catalog input).
  - NEITHER is derived from the Floquet spectrum.
  - NEITHER inspects the stability label.

b3, b4 (retired here, recorded for cross-substrate) likewise use catalog
geometry only (v_z is initial velocity; m_3 * z_0^2 is a dimensionful
geometric quantity).

The S/U label is supplied as the independent test column.
Assigning branch from catalog is information-disjoint from stability
labeling: the audit uses no stability information to construct the branch
hash.

This is the same non-circularity provenance as gamma_3'_orbit_pass2.
```

## Free Parameter Count

```text
Number of free parameters fitted to the data: 0
```

The 4-bit signature, the thresholds (`< 1`, `< 0.3`, `< 1e-6`, `< 2`),
the retirement rule (constant -> retire), and the df formula
(occupied_branch_count - 1) are all locked pre-data. The active-bit
selection is mechanical and audit-traceable. The 4-bucket
cross-classification follows from the active bits without further choice.

## Chi-Squared Degrees of Freedom

```text
df = occupied_branch_count - 1
   = 4 - 1                    [on supp-B; future catalogs may differ]
   = 3
```

Reference distribution `chi-squared(3)`. Critical value `11.34` at
`p = 0.01`. Test statistic `chi^2 > 11.34` triggers
`branch_hash_passes_audit`.

Wording note: the pass condition is `branch_hash_passes_audit`, NOT
"branch-shadow theorem confirmed". A pass licenses a v0.5b predictor
registration; it does not on its own establish a predictive mechanism.

## Reserved Diagnostics (Not Gating)

The following per-row continuous quantities are emitted on the receipt
but are NOT used in v0.5a's hash or test:

```text
T (period)                    -- catalog input, no normalization fixed yet
abs(v_z)                      -- magnitude only
sign(v_z)                     -- reserved for chirality diagnostic only
E, |L|                        -- orbit-level conserved quantities (from v0.3
                                 cross-m_3 receipts when available)
m_3, z_0 raw values           -- the underlying catalog coordinates
```

Per the v0.5 sign-off direction: *define T_kepler exactly and mu_eff
exactly before using continuous ratios beyond diagnostics*. Quantities
such as `T / T_kepler`, `log(T)`, `log(z_0)`, and `T * mu_eff^(1/2)` are
DEMOTED to diagnostics until their reference scales are pinned in a
separate registration. They do not enter the v0.5a hash.

## Edge Cases (Pre-Registered)

1. **Empty bucket on supp-B**: the pre-audit catalog check above shows
   all 4 buckets occupied; this case is inactive on supp-B but the df
   formula `occupied_branch_count - 1` would absorb empties on another
   catalog automatically.

2. **Cross-substrate bit re-activation**: if mesa, geometry, or
   freeze-A choreographies make b3 (`abs(v_z) < 1e-6`) or b4
   (`m_3 * z_0^2 < 2`) non-constant, that catalog gets its OWN
   registration. supp-B is locked to the active `(b1, b2)` signature.

3. **Pass with mass-only structure**: if v0.5a passes
   (`chi^2 > 11.34`) but the residual structure is dominated by b1
   (m_3 axis) and b2 (z_0 axis) is uninformative, v0.5b must
   re-register the predictor with explicit `b1`-only thresholds and a
   held-out test. ANY post-audit narrowing of the signature is a new
   registration, not a refinement.

4. **Marginal pass (`11.34 < chi^2 < ~16`)**: report as
   `branch_hash_passes_audit_marginal`. v0.5b registration must include
   leave-one-m_3-bin-out cross-validation as a gating step, not
   diagnostic.

## Pre-Mortem Expectation

The per-bucket S/U counts have NOT been computed before lock-in. The
audit runs blind to the actual chi-squared value.

The expected published statement, regardless of outcome:

> v0.5a tests whether the supp-B catalog's branch hash — formed by
> retiring degenerate bits and using the active `(m_3 < 1, z_0 < 0.3)`
> signature — stratifies stability. The verdict is reported as
> `branch_hash_passes_audit` (chi^2 > 11.34) or
> `branch_hash_fails_audit` (chi^2 <= 11.34) under the locked
> chi-squared(3) threshold at p = 0.01.

## Why v0.5a Is An Audit, Not A Predictor

A 4-bucket signature on 273 rows is too coarse to be honestly called a
predictor on this catalog. It tests the **projection limit**: does this
pre-registered low-dimensional branch shadow carry stability information?

```text
v0.4a  domain-shadow audit:    well-defined (PASS) but not predictive
v0.4b  Z_2-shadow predictor:   FALSIFIED (chi^2 = 1202 vs 26.22)
v0.5a  branch-shadow audit:    catalog projection-limit test
v0.5b  branch-shadow predictor: registered ONLY if v0.5a passes
```

We get the Z_2 buddha in the posture, not the statistic: this is a
projection-limit test. If the branch hash fails, we record that this
low-dimensional branch shadow also does not carry stability and move
to the next mechanism family without contorting the rule.

## Lock-In Statement

This audit registration is committed before v0.5a's chi-squared
statistic is computed against the v0.4a manifest's per-row catalog
parameters. The 4-bit candidate was specified before the constant-bit
degeneracy check; the active `(b1, b2)` signature follows from the
recorded retirement rule.

Any change to the bit candidates, thresholds, retirement rule, df
formula, or critical value after the chi-squared statistic is computed
is a **re-registration**, not a refinement.

Implementation may proceed using the v0.4a manifest at
`results/isotrophy/k-facet-v04a-domain-map/manifest.json` as the sole
input. No new dynamical compute is required: the four bits are
catalog-only.

## Receipts (planned)

```text
internal/anniversary/kfacet_v05a_branch_map_form.md          (this document)
results/isotrophy/k-facet-v05a-branch-map/manifest.json      (after audit)
  - bit_truth_counts:      per-bit true/false counts and retirement status
  - bucket_counts:         per-bucket N
  - contingency_table:     4 x 2 (bucket x S/U)
  - chi_squared:           test statistic
  - df:                    occupied_branch_count - 1
  - critical:              chi-squared(df) at p = 0.01
  - verdict:               branch_hash_passes_audit |
                           branch_hash_passes_audit_marginal |
                           branch_hash_fails_audit
  - per_row_table:         row_id, m_3, z_0, abs(v_z), m_3*z_0^2,
                           branch_label, stability
```

## Doc Trail

- `kfacet_v04_writeup.md` -- v0.4 structural-negative chapter close (opens v0.5)
- `kfacet_v04b_gamma3prime_form.md` -- prior baseline (falsified Z_2 shadow)
- `kfacet_v04a_domain_map_preregistration.md` -- catalog source manifest
- `docs/threebody/CROSS_SUBSTRATE_NOTES.md` sections 6-7 -- projection-language
  framing (body / shadow / projection-limit vocabulary)

---

Branch-shadow audit. Projection-limit floor, not predictor. Three buckets
of falsification at the threshold; a pass licenses v0.5b registration.
