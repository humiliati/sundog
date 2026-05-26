# v0.9a Signed vf Three-Zone Audit Form Lock

Status: **VERDICT LANDED 2026-05-24**. Verdict:
**`signed_vf_three_zone_fails_audit_chi2`** at chi^2 = 7.42 vs
critical 9.21 (df=2, p = 0.0245). The U-shape pattern check ALSO
fails (mixed S_fraction 0.34 is NOT less than positional 0.11;
mixed < velocity 0.44 holds but the bilateral pattern is broken).
The observed pattern is **monotone increasing** in vf:
positional-dominant 10.5% S, mixed 33.9% S, velocity-heavy 43.9%
S. **This is methodologically informative**: the v0.7a' U-shape
was an artifact of the quartile boundary at vf ~ 0.297, where v0.7a
Q1's 49% S-fraction was concentrated; under the physical cutpoint
at vf = 0.25, the bottom-most-vf zone shrinks from 63 to 19 rows
and its S-fraction collapses to 10.5%. The v0.7a' chi^2 = 16.43
positive is NOT invalidated -- it remains a real catalog
stratification finding -- but the U-shape mechanism hypothesis is.

This document locks the v0.9a three-zone chi-squared audit on the
v0.7 analyzable 250-row subset under the parent registration's
candidate slot **C** (three-zone audit with physically-motivated
cutpoints). Candidate forms A (ordered quartile) and B (binary
threshold) are REJECTED pre-compute under the v0.9 parent's
anti-circular discipline. Candidate D (Jonckheere-Terpstra trend
test) is held back.

## Verdict (Landed, 2026-05-24)

Receipt:
`results/isotrophy/k-facet-v09a-signed-vf-three-zone/manifest.json`.

```text
verdict:                signed_vf_three_zone_fails_audit_chi2
domain:                 250 analyzable rows (87 S / 163 U)

three-zone contingency:
  zone                       N    S    U    S_fraction  chi2_contrib
  positional-dominant (vf<0.25)  19    2   17   0.1053       4.93
  mixed (0.25 <= vf < 0.5)      165   56  109   0.3394       0.05
  velocity-heavy (vf >= 0.5)     66   29   37   0.4394       2.43
                                                              -----
                                                  chi^2 =     7.42

test_branch_taken:      chi_squared (asymptotic; min_expected =
                                     19 * 87 / 250 = 6.61, above
                                     the 5.0 floor)
df:                     2
critical (p=0.01):      9.21
p_value (chi-sq(2)):    0.0245
alignment_tightness:    0.5697  (well below 0.8 warning threshold)

pattern check:
  S_fraction(positional)   = 0.1053
  S_fraction(mixed)        = 0.3394
  S_fraction(velocity-h)   = 0.4394
  mixed < positional?      FALSE  (0.34 > 0.11; U-shape direction
                                   broken on the lower end)
  mixed < velocity-heavy?  TRUE   (0.34 < 0.44; this side of the
                                   U-shape holds)
  u_shape_confirmed?       FALSE  (requires BOTH sides; the
                                   physical-zone U-shape does NOT
                                   exist)
```

**Structural reading:**

> Under the physical cutpoint at vf = 0.25 (the 3:1 mass-weighted
> ratio between position and velocity components of gamma_1), the
> "positional-dominant" zone has only 19 rows in the analyzable
> subset, with 10.5% S-fraction. This is a sharp contrast with
> v0.7a' Q1 (vf in [0, 0.297]), which had 63 rows at 49.2% S. The
> mass of v0.7a Q1's high-S rows is concentrated near vf = 0.25 to
> 0.30, NOT at the deepest positional regime (vf < 0.25).
>
> The v0.7a' U-shape was an artifact of the QUARTILE BOUNDARY at
> vf ~ 0.297: the v0.7a' Q1 bin combined some genuinely low-vf
> rows (mostly unstable) with vf ~ 0.25-0.30 rows (heavily stable)
> in a way that the boundary placement made monotone-broken. Under
> a physical cutpoint, the boundary moves and the apparent U-shape
> disappears.
>
> The actual physical pattern is **monotone increasing in vf**:
> low vf (positional-dominant) is mostly unstable; mid vf (mixed)
> is intermediate; high vf (velocity-heavy) is most stable. This
> matches the broader thesis that velocity-dominated directions
> correlate with stability. But chi^2 = 7.42 does not reach the
> registered floor at p = 0.01 (critical = 9.21), so this signal
> is NOT licensed as a v0.9a pass.

**Important meta-finding**: the v0.7a' chi^2 = 16.43 is NOT
invalidated by this verdict. The v0.7a' result remains a real
catalog-coordinate stratification finding under quartile binning.
What v0.9a invalidates is the **U-shape MECHANISM hypothesis** --
i.e., the proposal that the pattern is symmetric U-enrichment of
the mixed zone. The pattern is actually monotone-increasing, and
the physical-zone test does not reach the p = 0.01 floor for the
3 x 2 contingency.

**Receipts:**

```text
docs/isotrophy/kfacet/kfacet_v09a_signed_vf_three_zone_form.md (this document)
results/isotrophy/k-facet-v09a-signed-vf-three-zone/manifest.json
results/isotrophy/k-facet-v09a-signed-vf-three-zone/per_row_table.csv
results/isotrophy/k-facet-v09a-signed-vf-three-zone/contingency_table_three_zone.csv
scripts/v09a_signed_vf_three_zone_audit.py
```

Per the locked outcome interpretation:

> "If v0.9a Chi-squared Fails: The physical-zone binning does not
> reach the chi-squared gate; the v0.7a' signal lives in a finer
> binning than the physically-motivated zones detect. Chapter
> closes on the v0.9a hypothesis."

v0.9b (held-out predictor) is NOT licensed under the fails verdict.

Audience: v0.9a runner; v0.9b form-lock author (conditional on
v0.9a passing); paper-side reviewer of the signed direction-
composition mechanism.

Companions:

- `kfacet_v09_mechanism_preregistration.md` -- v0.9 parent
  registration. Locks the anti-circular framing discipline that
  this form must respect.
- `kfacet_v08_writeup.md` -- v0.8 chapter close (structural-negative
  on unsigned purity) that opened v0.9.
- `kfacet_v07a_prime_restricted_scope_form.md` -- v0.7a' PASS at
  chi^2 = 16.43; the source observation v0.9 must NOT silently
  re-test under the same statistic.
- `kfacet_v07a_velocity_fraction_audit_form.md` -- v0.7a parent
  (vf operational definition; non-circularity sentence carries
  forward).

Frame: v0.8a confirmed the v0.7a' signal is signed, not symmetric in
unsigned purity. The vf distribution is density-asymmetric (184/250
rows below vf = 0.5; max vf = 0.87, no rows near vf = 1). v0.9a
tests a **specific U-shape hypothesis** on three physical zones:
that the **mixed zone is U-enriched relative to BOTH the
positional-dominant and velocity-heavy zones**. This is a
hypothesis-specific test, not a re-run of chi^2 of independence on
vf-ordering.

## Why Three-Zone (Form C) Over Alternatives

```text
A. Ordered quartile audit.         REJECTED by v0.9 parent
                                   (re-tests v0.7a' chi^2 by
                                    linear-transform invariance).

B. Binary vf < 0.5 threshold.      REJECTED by v0.9 parent
                                   (binary collapse of v0.7a'
                                    Q1+Q2 vs Q3+Q4).

C. Three-zone audit with           SELECTED. Physical cutpoints
   physical cutpoints.              {0.25, 0.50}, hypothesis-specific
                                    verdict tree (chi^2 > critical
                                    AND mixed < both direction
                                    zones).

D. Jonckheere-Terpstra             held back. Different test
   monotone-trend test.             statistic (one-sided trend test)
                                    rather than chi^2-of-independence.
                                    Cleaner non-circular shape but
                                    different test family.
```

C is selected because (a) the cutpoints {0.25, 0.50} are physically
motivated (see Physical Motivation section below), NOT data-driven
from v0.7a' quartile cutpoints; (b) the verdict tree requires a
hypothesis-specific S-fraction pattern (mixed-zone-U-enrichment)
in addition to the chi-squared threshold; (c) the result is
interpretable in physical units (positional vs mixed vs velocity
domination) rather than as catalog quartiles.

## Anti-Circular Argument (Load-Bearing)

The v0.9 parent's anti-circular discipline requires that any v0.9a
form lock break at least one of (feature derivation, test statistic,
evaluation domain) relative to v0.7a's chi^2-of-independence test
on vf quartiles.

**v0.9a breaks condition #1 (feature derivation)** via the physical
cutpoints {0.25, 0.50}, which are NOT v0.7a's quartile cutpoints
(approximately {0.297, 0.419, 0.504} for q25, q50, q75 of the 250-row
analyzable subset). The three-zone bin boundaries are pre-registered
from physical reasoning, not from the data distribution.

**v0.9a additionally breaks condition #2 (test statistic)** via the
augmented verdict tree, which requires chi^2 > critical AND a
hypothesis-specific S-fraction pattern. The basic chi^2 of
independence is the gate, but the pattern check converts the test
into a hypothesis-specific one. A v0.9a contingency that produced
chi^2 > critical but with the WRONG pattern (e.g., mixed has higher
S-fraction than direction-dominant zones) would NOT pass the
verdict.

Both conditions broken. v0.9a is honestly non-circular.

## What v0.9a Is

v0.9a is a **catalog-side audit on the v0.7 analyzable 250-row
subset**, NOT a predictor (continues the audit-not-predictor
framing):

```text
Domain:           250 analyzable rows from v0.7a per_row_table.csv
                  (rows with integration_blocked == False).
                  Inherited from v0.8 verbatim.

Feature:          vf(row) directly (NOT transformed). v0.7a
                  operational definition + non-circularity sentence
                  carry forward unchanged.

Zones:            three physically-motivated zones over [0, 1]:
                    positional-dominant:  vf  in [0, 0.25)
                    mixed:                 vf  in [0.25, 0.50)
                    velocity-heavy:       vf  in [0.50, 1]
                  cutpoints {0.25, 0.50} pre-registered with
                  physical motivation (below). Right-closed lower
                  intervals (ties to upper bin under standard
                  Python comparison; deterministic).

Test:             3 x 2 chi-squared (zone x S/U), df = occupied_zones
                  - 1 (usually 2).

Critical:         chi-squared(df, p = 0.01).
                  df = 2: critical = 9.21.

Primary
hypothesis:       S_fraction(mixed) < S_fraction(positional)
                  AND S_fraction(mixed) < S_fraction(velocity-heavy)
                  (U-shape with the mixed zone U-enriched relative
                  to BOTH direction-dominant zones).

Alignment guard:  max over zones of (fraction in any single
                  v0.5a branch_label bucket). Thresholds 0.8
                  (warning) / 0.95 (severe).

Sparse fallback:  v0.6b discipline (seed 20260523, n_permutations
                  10000, min_occupied_bin_count >= 2,
                  expected_cell >= 5).

Verdict tree:
  if min_occupied_bin_count < 2:
      verdict = signed_vf_three_zone_inconclusive_sparse
  elif chi^2 > critical
       AND S_fraction(mixed) < S_fraction(positional)
       AND S_fraction(mixed) < S_fraction(velocity-heavy):
      if alignment <= 0.8:
          verdict = signed_vf_three_zone_passes_audit
      elif alignment <= 0.95:
          verdict = signed_vf_three_zone_passes_audit_alignment_warning
      else:
          verdict = signed_vf_three_zone_passes_audit_severe_alignment
  elif chi^2 > critical:
      verdict = signed_vf_three_zone_fails_audit_pattern
      (chi^2 reaches threshold but the U-shape direction is wrong)
  else:
      verdict = signed_vf_three_zone_fails_audit_chi2
      (chi^2 does not reach threshold)
```

A clean Pass licenses v0.9b: a separately-registered held-out
signed-vf predictor with the v0.5b discipline. A Pattern Fail
indicates the catalog has structure in vf vs S/U but NOT the
predicted U-shape (would be a clean structural-negative on the
v0.7a'-inspired hypothesis). A Chi-squared Fail indicates the
catalog does not even reach the chi-squared gate under the
physical zone binning.

## Physical Motivation for Cutpoints {0.25, 0.50}

The cutpoints are chosen from physical reasoning about Hamiltonian
flow tangent dynamics, NOT from v0.7a' quartile boundaries:

```text
vf = 0.50:  the maximally-mixed direction in phase space. A vector
            with equal mass-weighted norm in position and velocity
            sub-blocks is at vf = 0.5. This is the natural
            boundary between "direction-dominated" (position OR
            velocity carries the majority of mass-weighted norm)
            and "mixed" (neither subspace dominates).

vf = 0.25:  the 3:1 boundary in mass-weighted norm. A vector with
            ||delta_q||^2 = 3 ||delta_v||^2 (i.e., position carries
            75% of mass-weighted norm) is at vf = 0.25. This is
            the boundary between "positional-dominant"
            (clearly position-dominated) and "mixed" (less than 3:1
            ratio favoring position).

asymmetric cutpoints:
  No symmetric upper cutpoint at vf = 0.75 is locked because:
  - the v0.8a vf distribution diagnostic shows max vf = 0.87
    (no rows near vf = 1);
  - the catalog has only 66/250 rows with vf >= 0.5;
  - splitting the velocity-heavy zone further would produce
    sparse bins (Risk 1 in the v0.9 parent's pre-mortem).
  The asymmetric structure (two cutpoints below 0.5, one zone
  above 0.5) reflects the catalog's actual density distribution,
  recorded in v0.8a's diagnostic. The cutpoints are still PHYSICAL
  (the 0.50 boundary is the natural mixing scale; the 0.25 boundary
  is the 3:1 mass-weighted ratio), not v0.7a'-quartile-derived.
```

This physical motivation is pre-registered. Any subsequent change
to the cutpoints after v0.9a runs is a re-registration, not a
refinement.

## Pre-Audit Sanity Surface

Before v0.9a runs:

```text
1. v0.7a per_row_table.csv must be present and readable.
2. Filter to analyzable rows (integration_blocked == False);
   expected count = 250.
3. v0.5a branch_label must be present per row (already joined in
   v0.7a per_row_table).
4. Verify vf values are in [0, 1] for all 250 rows.
5. Compute zone(r) for each row under the locked cutpoints.
6. Per-zone counts MUST be emitted on the receipt before the
   chi-squared statistic is computed; thin zones (< 5 rows) MUST
   fire the sparse-cell fallback.
```

## Non-Circularity Audit

```text
Feature derivation:
  vf is inherited unchanged from v0.7a (same operational definition,
  same selection rule, same tie-break cascade, same reference frame).

Binning derivation:
  zones use cutpoints {0.25, 0.50} from physical reasoning, NOT
  from v0.7a' quartile cutpoints (which are approximately
  {0.297, 0.419, 0.504}). The cutpoints are different by design.

Test derivation:
  primary statistic is chi-squared of independence (same family as
  v0.7a'), but the verdict tree REQUIRES the additional pattern
  check (mixed < both direction-dominant zones). A contingency
  that produces chi^2 > critical with the WRONG pattern (e.g.,
  positional zone is U-enriched) would be a Pattern Fail, not a
  Pass. This is a hypothesis-specific test.

Domain:
  same 250-row analyzable subset as v0.7a'. Domain is NOT broken
  in this form; the non-circularity rests on (feature) and (test)
  conditions.

The locked non-circularity sentences (v0.7a + v0.9 anti-circular)
carry forward verbatim:

> "v0.9 uses Floquet eigenvectors only as geometric directions; it
> does not use eigenvalue magnitude, spectral radius, unit-circle
> status, unstable-pair count, or any threshold that defines the
> published S/U label."

> "v0.9 does not test chi^2 of independence on vf-ordering against
> S/U on the v0.7a analyzable subset as the sole verdict criterion;
> v0.9a's verdict requires both chi^2 > critical AND a
> hypothesis-specific S-fraction pattern (mixed < both direction-
> dominant zones)."
```

## Free Parameter Count

```text
Number of free parameters fitted to the data: 0
```

The feature (vf), the zone cutpoints (physical {0.25, 0.50}), the
zone-label conventions (positional-dominant / mixed / velocity-heavy),
the test statistic (chi-squared), the df formula (occupied_zones -
1), the critical value (chi-squared(df, 0.01)), the
primary-hypothesis pattern check (mixed < both direction-dominant
zones), the alignment-tightness thresholds (0.8 / 0.95), the
sparse-cell fallback parameters, and the verdict tree are all
locked pre-data.

The cutpoints {0.25, 0.50} were chosen from physical reasoning
BEFORE the v0.9a runner is executed; they are NOT derived from
v0.7a' quartile cutpoints.

## Sparse-Cell Fallback Logic (Inherited from v0.6b)

```text
1. Compute expected cell counts E_{kj} = N_k * C_j / 250.
2. Compute min_occupied_bin_count = min over occupied zones of N_k.
3. Branch:
   a) If min_occupied_bin_count < 2:
        verdict = signed_vf_three_zone_inconclusive_sparse
        no p-value emitted.
   b) Elif min over occupied cells of E_{kj} < 5:
        permutation test (seed = 20260523, n_permutations = 10000);
        chi^2 statistic computed per permutation;
        p_value = empirical tail.
   c) Else:
        asymptotic chi-squared(df = occupied_zones - 1) at p = 0.01.
```

For zones (vf < 0.25, 0.25-0.5, >= 0.5) on 250 rows with 87 S / 163
U, expected counts likely all > 5 (zone counts probably ~50, ~135,
~65 based on v0.8a distribution; expected S per zone ~17-47;
expected U per zone ~33-88). Asymptotic chi-squared expected to
fire.

## Alignment-Tightness Guard (Inherited from v0.6a / v0.7a)

```text
For each zone in {positional-dominant, mixed, velocity-heavy}:
  - identify the v0.5a branch_label that dominates the zone.
  - compute dominant_fraction = (rows in zone in dominant branch)
                                / (rows in zone).

alignment_tightness_scalar = max over zones of dominant_fraction.

Thresholds:    0.8 (warning), 0.95 (severe).
```

## Edge Cases (Pre-Registered)

1. **Empty zone**: structurally unlikely given 250 rows with vf
   spanning [0.04, 0.87]; positional-dominant (vf < 0.25) has roughly
   the bottom ~22% of the distribution per v0.8a, velocity-heavy
   (vf >= 0.5) has roughly the top ~26%. If any zone has 0 rows,
   df adjusts to occupied_zones - 1; receipt flags the empty zone.

2. **Pattern Fail**: chi^2 > critical but the U-shape direction is
   wrong (e.g., positional-dominant has 18% S while mixed has 35%
   S). The receipt records the actual S-fraction ordering; this
   is a clean structural-negative on the U-shape hypothesis.

3. **Chi-squared Fail**: chi^2 <= critical. The catalog does not
   reach the 3-zone chi-squared gate under the physical zone
   binning. The receipt records the pattern direction for
   transparency, but the verdict is fail at chi^2.

4. **Sparse cells**: structurally unlikely but the fallback handles
   it via permutation test or inconclusive verdict.

5. **Alignment > 0.8**: pattern check still applies; alignment-
   warning verdict augments the standard Pass with a flag.
   v0.9b's partition design must address the alignment.

## Pre-Mortem Expectation

The per-row vf distribution on the 250-row subset is known from
v0.8a; the per-zone counts are NOT yet computed, but rough
estimates from the v0.8a vf distribution:

```text
positional-dominant (vf < 0.25):    ~55 rows  (v0.8a vf min ~0.04,
                                                q25 = 0.297, so
                                                rows with vf < 0.25
                                                are below q25)
mixed (0.25 <= vf < 0.50):           ~130 rows (most of q25-q50 +
                                                lower part of q50-q75)
velocity-heavy (vf >= 0.50):         ~65 rows  (above v0.7a' Q3-Q4
                                                boundary at vf = 0.504)
```

Expected per-zone S-fractions (rough estimate from v0.7a'
contingency):

```text
positional-dominant:  ~50% S  (the v0.7a' Q1-low-vf subset is here)
mixed:                ~25% S  (v0.7a' Q2 18% + Q3 29%)
velocity-heavy:       ~43% S  (v0.7a' Q4)
```

If these estimates hold, the U-shape pattern is satisfied
(mixed 25% < positional 50% AND mixed 25% < velocity-heavy 43%),
and chi^2 should be substantial. Expected v0.9a verdict: PASS
under the locked tree.

If the data deviates (e.g., the v0.8a unsigned purity fail had
revealed a different shape), the verdict tree handles it:
Pattern Fail or Chi-squared Fail.

Pre-mortem note: the U-shape predicted from v0.7a' is a strong
hypothesis; if v0.9a PASSES, this is the first explicitly
hypothesis-specific pre-registered positive in the chain (not just
chi^2-of-independence, but pattern-confirmed). If it FAILS, the
v0.7a' U-shape was misleading and the chain has a clear
structural-negative on the proposed pattern.

## Stop Conditions

1. **Anti-circular discipline failure**: NOT triggered. v0.9a
   breaks two of the three conditions (feature derivation via
   physical cutpoints; test derivation via pattern-specific verdict
   tree).

2. **Constant-feature retirement**: vf has sd = 0.144 on the
   250-row subset; retirement is structurally impossible.

3. **Alignment-tightness > 0.95**: v0.9b must use an alignment-
   breaking partition.

4. **Pattern Fail**: chapter closes on the U-shape hypothesis. v0.9
   may register an alternative pattern hypothesis if motivated
   physically, BUT the bandwidth tradeoff against Phase 15+
   should be weighed.

5. **Chi-squared Fail**: same as Pattern Fail, but the catalog
   does not even reach the chi-squared gate under the physical
   zone binning.

## Lock-In Statement

This audit form is committed before:

- per-row vf values are read into the v0.9a runner,
- per-zone counts are computed,
- the chi-squared statistic is computed,
- the alignment-tightness scalar is computed,
- the per-zone S-fractions are inspected.

Any change to the cutpoints, zone labels, df formula, critical
value, primary-hypothesis pattern, alignment-tightness thresholds,
constant-feature retirement threshold, sparse-cell fallback
parameters, or verdict tree after the v0.9a runner is executed
is a re-registration, not a refinement.

Implementation may proceed against
`results/isotrophy/k-facet-v07a-velocity-fraction-audit/per_row_table.csv`
as the sole input. No new variational compute is required.

## Receipt Schema (Planned)

```text
results/isotrophy/k-facet-v09a-signed-vf-three-zone/
  manifest.json
    - mode = "v0.9a-signed-vf-three-zone-audit"
    - form_lock = "docs/isotrophy/kfacet/kfacet_v09a_signed_vf_three_zone_form.md"
    - input_v07a_per_row_table
    - domain_row_count = 250
    - domain_S_count, domain_U_count
    - zone_cutpoints: [0.25, 0.50]
    - zone_labels: [positional-dominant, mixed, velocity-heavy]
    - non_circular_conditions_broken: ["feature_derivation_via_physical_cutpoints",
                                        "test_derivation_via_pattern_specific_verdict"]
    - contingency_zone:    n_zones x 2 (zone, S/U)
    - chi_squared:         test statistic
    - test_branch_taken:   "chi_squared" | "permutation" | "inconclusive_sparse"
    - df:                  occupied_zones - 1
    - critical:            chi-squared(df, 0.01)
    - p_value:             chi-squared tail OR permutation tail
    - alignment_tightness_scalar
    - alignment_records:   per-zone dominant branch + counts
    - pattern_check:
        - S_fraction_positional
        - S_fraction_mixed
        - S_fraction_velocity_heavy
        - mixed_lt_positional: bool
        - mixed_lt_velocity_heavy: bool
        - u_shape_confirmed: bool (mixed_lt_positional AND mixed_lt_velocity)
    - verdict:             signed_vf_three_zone_{passes_audit |
                                                  passes_audit_alignment_warning |
                                                  passes_audit_severe_alignment |
                                                  fails_audit_pattern |
                                                  fails_audit_chi2 |
                                                  inconclusive_sparse}
  per_row_table.csv  (with zone column)
  contingency_table_three_zone.csv
```

## Implementation Plan

Suggested script: `scripts/v09a_signed_vf_three_zone_audit.py`.

Suggested command:

```powershell
python scripts\v09a_signed_vf_three_zone_audit.py `
  --v07a-per-row results\isotrophy\k-facet-v07a-velocity-fraction-audit\per_row_table.csv `
  --out          results\isotrophy\k-facet-v09a-signed-vf-three-zone
```

Expected runtime: seconds. No new variational compute.

## Interpretation

If v0.9a passes (chi^2 > critical, U-shape confirmed, alignment
<= 0.8):

> The signed Floquet direction-composition shadow stratifies
> stability on the 250-row analyzable supp-B subset under a
> pre-registered hypothesis-specific U-shape pattern test
> (mixed zone U-enriched relative to both direction-dominant
> zones). This is the **first hypothesis-specific pre-registered
> positive** in the v0.4-v0.9 isotrophy chain (v0.7a' was a chi^2-
> of-independence positive). v0.9b is licensed: a held-out signed-
> vf predictor with the v0.5b discipline.

If v0.9a passes with alignment warning: pass-but-suspect; v0.9b's
partition must be re-registered alignment-breaking.

If v0.9a Pattern Fails (chi^2 > critical, U-shape NOT confirmed):

> The catalog has structure in vf vs S/U under the physical zone
> binning, but it is NOT the predicted U-shape (mixed zone is not
> U-enriched relative to both direction-dominant zones). The
> v0.7a' U-shape may have been an artifact of the quartile-binning
> boundaries; the physical-zone version does not reproduce it.
> Chapter closes as a structural-negative on the U-shape hypothesis.

If v0.9a Chi-squared Fails:

> The physical-zone binning does not reach the chi-squared gate;
> the v0.7a' signal lives in a finer binning than the
> physically-motivated zones detect. Chapter closes on the v0.9a
> hypothesis.

## Doc Trail

- `kfacet_v09_mechanism_preregistration.md` -- parent registration
  (anti-circular discipline locked).
- `kfacet_v08_writeup.md` -- v0.8 close (predecessor).
- `kfacet_v07a_prime_restricted_scope_form.md` -- v0.7a' PASS
  source of the U-shape hypothesis.
- `kfacet_v06b_within_branch_energy_audit_form.md` -- sparse-cell
  fallback tree (inheritance source).
- `kfacet_v06a_energy_quartile_audit_form.md` -- alignment-tightness
  guard (inheritance source).

---

Signed vf three-zone audit on the 250-row analyzable subset. Physical
cutpoints {0.25, 0.50}. Hypothesis-specific verdict tree (chi^2 >
critical AND mixed < both direction-dominant zones). Two conditions
of the anti-circular discipline broken (feature derivation, test
derivation). Inherits v0.6b sparse-cell fallback + v0.6a alignment-
tightness guard.
