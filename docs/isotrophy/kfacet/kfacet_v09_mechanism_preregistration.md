# v0.9 Mechanism Preregistration: Signed Floquet Direction-Composition

Status: **REGISTERED 2026-05-24. v0.9a child verdict landed
2026-05-24: `signed_vf_three_zone_fails_audit_chi2`** at chi^2 =
7.42 vs critical 9.21 (df=2, p = 0.0245). The U-shape pattern check
also fails (mixed > positional). Substantive finding: the v0.7a'
U-shape was a quartile-boundary artifact at vf ~ 0.297; under
physical cutpoint at vf = 0.25 the pattern is monotone-increasing
(positional 11% S < mixed 34% S < velocity-heavy 44% S), but the
3 x 2 chi^2 does not reach the p = 0.01 floor. The v0.7a' chi^2 =
16.43 positive is NOT invalidated; the U-shape MECHANISM hypothesis
is. See `kfacet_v09a_signed_vf_three_zone_form.md` for the full
verdict. v0.9b (held-out predictor) is NOT licensed under the
fails verdict.

This document locks the v0.9 body / projection / observable and the
**anti-circular framing discipline** for any future child form lock
that tests vf-ordering hypotheses after v0.7a' already produced a
chi^2 = 16.43 positive on a vf-ordering chi-squared of independence.

Audience: v0.9a form-lock author; paper-side reviewer of the
v0.8 -> v0.9 transition (purity-shadow structural-negative + signed
direction-composition open).

Companions:

- `kfacet_v08_writeup.md` -- v0.8 chapter close (structural-negative
  on unsigned direction-purity) that opened v0.9.
- `kfacet_v08a_purity_quartile_audit_form.md` -- v0.8a fails verdict
  + the density-asymmetry diagnostic that licensed signed-direction
  as the next hypothesis.
- `kfacet_v07a_prime_restricted_scope_form.md` -- v0.7a' PASS at
  chi^2 = 16.43; the source observation v0.9 must NOT silently
  re-test.
- `kfacet_v07a_velocity_fraction_audit_form.md` -- v0.7a parent
  audit (vf operational definition; non-circularity sentence
  carries forward).

Frame: v0.7a' produced a chi^2 = 16.43 (p = 9.3e-4) signal on
quartile-binned velocity-fraction vs S/U with alignment 0.698
against the v0.5a branch hash. v0.8a tested whether this signal is
captured by unsigned direction-purity `abs(vf - 0.5)` and failed
(chi^2 = 4.94, p = 0.176). The v0.8a diagnostic purity_signed
contingency reproduced v0.7a' chi^2 = 16.43 exactly, confirming the
signal is real but lives in the **signed** vf direction. v0.9
promotes signed direction-composition to the primary mechanism and
asks: **what specific pattern in vf vs S/U is real, beyond the
already-known fact that there is some pattern?**

## What v0.9 Is Not

v0.9 is NOT:

- **A re-test of the v0.7a' positive under a different framing.**
  Any chi^2 of independence test on vf-ordering (raw vf quartiles,
  signed quartiles, binary vf-side, log-vf, any monotone transform)
  reproduces v0.7a's chi^2 = 16.43 by linear-transform invariance.
  Such a test would silently re-confirm v0.7a' under a new name --
  the kind of mechanism-laundering the pre-mortem discipline was
  built to prevent.

- **A v0.7c held-out vf predictor.** v0.7a' was attrition-blocked
  on the catalog-wide audit; the chapter closed on a qualified-
  positive on restricted domain. Promoting v0.7a' to a held-out
  predictor under a new name (v0.9b vf-quartile predictor) would
  not be honest -- v0.7b was explicitly NOT licensed under the
  attrition verdict.

- **A re-projection of vf onto a different feature axis.** vf
  itself is the load-bearing scalar; v0.9 asks what SPECIFIC
  STRUCTURE in vf vs S/U is testable beyond chi^2 of independence.

## Body / Projection / Observable

```text
Body:
  the 250-row analyzable supp-B subset from v0.7a (rows where
  compute_monodromy_vectorized completed at R1-amended sanity
  gates). Inherited from v0.8 verbatim.

Projection:
  orbit-dynamics signed direction-composition shadow
    row -> vf(row) = velocity-fraction of gamma_1
  Same operational definition as v0.7a' (locked selection rule,
  tie-break cascade, CoM reduction, mass-weighted norm).

Observable:
  per-row (vf, stability) on 250 rows joined with v0.5a
  branch_label for alignment-tightness diagnostics.

Per-row computational cost:
  zero. vf values are derived from v0.7a per_row_table.csv. No
  new variational integration is needed.
```

The projection axis is **vf as a signed scalar in [0, 1]**, not
folded. v0.9 inherits v0.7a/v0.7a's vf with no transform.

## Anti-Circular Framing Discipline (Load-Bearing)

The central methodological challenge: v0.7a' has already established
chi^2 of independence = 16.43 on quartile-binned vf vs S/U on this
same 250-row subset. Any v0.9 test that:

- Uses vf-ordering as the feature, AND
- Uses chi^2 of independence as the test statistic, AND
- Tests on the same 250-row subset,

**reproduces v0.7a' by construction** (linear-transform invariance
of chi^2 of independence under monotone reordering of feature bins).
Such a test is not a fresh hypothesis test; it is a re-statement of
v0.7a'.

For v0.9 to be honestly non-circular, **at least one of** the three
above conditions must be broken:

```text
1. Different feature derivation:
   Use a non-monotone transform of vf (e.g., partition into three
   physically-motivated zones with pre-registered cutpoints from
   physical reasoning, NOT from v0.7a' quartile cutpoints).

2. Different test statistic:
   Use a hypothesis-specific test that is NOT chi^2 of independence.
   Examples: Jonckheere-Terpstra monotone-trend test (asks "is the
   relationship monotone?"); piecewise-binomial U-shape goodness-of-
   fit test (asks "does the U-shape fit a specific predicted
   pattern?"); contingency-shape Cochran-Armitage trend test
   ("is there a monotone trend?").

3. Different evaluation domain:
   Test on a HELD-OUT subset of the 250-row analyzable rows where
   the held-out partition is pre-registered (e.g., leave-one-m_3-
   bin-out, with the v0.7a' quartile cutpoints frozen as features
   from training folds).

Combinations are allowed. The v0.9a form lock must SPECIFY which
condition(s) it breaks and argue the non-circularity explicitly.
```

This anti-circular discipline is **load-bearing** for v0.9. A v0.9a
form lock that does not break at least one of the three conditions
is **rejected pre-compute** as a re-test of v0.7a'.

## Operational Definition

```text
vf is inherited unchanged from v0.7a:

  vf(row) = ||delta_v||^2 / (||delta_q||^2 + ||delta_v||^2)

  where (delta_q, delta_v) is the largest-real-part Floquet
  eigenvector under CoM reduction + mass-weighted norm. Locked
  selection rule, tie-break cascade, and reference frame from
  v0.7a apply.

vf is in [0, 1] for all 250 analyzable rows. No new transform is
applied at the parent level; the v0.9a form lock will specify
how vf is used (zones, trend test, held-out, etc.).
```

## Non-Circularity Audit

The v0.7a locked non-circularity sentence is **re-asserted verbatim**
for v0.9:

> v0.9 uses Floquet eigenvectors only as geometric directions; it
> does not use eigenvalue magnitude, spectral radius, unit-circle
> status, unstable-pair count, or any threshold that defines the
> published S/U label.

v0.9-specific anti-circular sentence (load-bearing):

> v0.9 does not test chi^2 of independence on vf-ordering against
> S/U on the v0.7a analyzable subset; that test has already been
> performed in v0.7a' and any v0.9 repetition under different
> binning labels reproduces v0.7a' by linear-transform invariance.
> v0.9 tests a specific pattern hypothesis (e.g., monotone-trend
> rejection, three-zone S-fraction prediction, held-out fold
> classification) that v0.7a' did NOT pre-register.

The v0.9a form lock must include both sentences explicitly.

## Disallowed Features (Inheritance + v0.9-Specific)

The v0.9 form locks may NOT use:

```text
v0.4b / v0.5 / v0.6 / v0.7 / v0.8 inheritance:
  - stability label S/U,
  - any function of M_i eigenvalue magnitudes (spectral radius,
    unit-circle count, max |lambda|, Krein signature, etc.),
  - v0.4a Pass classification,
  - v0.5a branch hash output (as feature; alignment-tightness
    diagnostic only),
  - Q_E / Q_|L| / any v0.6a quartile,
  - E / |L| as scalars (retired in v0.6),
  - unsigned purity = abs(vf - 0.5) as primary (v0.8a closed it).

v0.9-specific:
  - chi^2 of independence on vf-ordering vs S/U as the PRIMARY
    test statistic on the 250-row subset (would re-test v0.7a';
    see anti-circular discipline above).
  - v0.7a' quartile cutpoints as v0.9 binning (would inherit
    v0.7a's data-driven partition).
  - the v0.7a' Q_vf labels as features (would silently re-test).
  - any hypothesis registered AFTER reading the v0.7a' contingency
    that "predicts" the shape v0.7a' already observed (would be
    post-hoc, not pre-registered).
```

## Inheritance Discipline From v0.5 / v0.6 / v0.7 / v0.8

v0.9 inherits all prior register-shaping disciplines AND adds the
anti-circular framing as a new permanent discipline:

1. **Audit-then-predictor separation.** v0.9a registers an audit
   under one of the three anti-circular forms; v0.9b (if licensed
   by v0.9a passing) registers a held-out predictor.

2. **Asymmetric falsifier for the predictor.** v0.9b requires BOTH
   `McNemar p <= 0.01` AND positive accuracy delta against the
   always-U baseline.

3. **Held-out partition discipline.** v0.9b defaults to
   leave-one-m_3-bin-out on the 250-row analyzable subset.

4. **Tie rule + absent-bin fallback.** Inherited from v0.5b.

5. **Alignment-tightness guard.** v0.9a MUST pre-register an
   alignment-tightness scalar against the v0.5a branch hash
   (thresholds 0.8 warning / 0.95 severe).

6. **Sparse-cell fallback tree.** Inherited from v0.6b (seed
   20260523, n_permutations 10000).

7. **Constant-feature retirement.** vf_sd on the 250 rows was 0.144
   in v0.7a; retirement at sd < 0.01 is structurally unlikely.

8. **Anti-circular framing (v0.9-new).** Any v0.9a form lock must
   break at least one of (feature derivation, test statistic,
   evaluation domain) relative to v0.7a's chi^2-of-independence-
   on-vf-quartiles test.

## Candidate v0.9a Audit Forms (Sketches)

The user-specified candidates:

```text
A. Ordered quartile audit (Q1/Q2/Q3/Q4 labels preserved).
   STATUS: REJECTED under anti-circular discipline.
   Reason: chi^2 of independence on ordered vf quartiles vs S/U
   reproduces v0.7a's chi^2 = 16.43 by construction. The "ordered"
   label adds no new test; the statistic does not change under
   re-labeling.

B. Binary threshold audit at vf < 0.5 vs vf >= 0.5.
   STATUS: REJECTED under anti-circular discipline.
   Reason: collapses the v0.7a' Q1+Q2 vs Q3+Q4 contingency. The
   binary chi^2 (df = 1) is determined by the same sufficient
   statistics that produced v0.7a's chi^2 = 16.43. Re-test under
   coarser binning.

C. Three-zone audit (positional-pure / mixed / velocity-heavy).
   STATUS: LIVE CANDIDATE if zone boundaries are pre-registered
   with PHYSICAL MOTIVATION, NOT data-driven from v0.7a's quartile
   cutpoints.
   Example: vf in [0, 0.2) (positional-dominant), [0.2, 0.6) (mixed),
   [0.6, 1] (velocity-heavy). Cutpoints (0.2, 0.6) would need
   physical justification (e.g., a Hamiltonian-flow-specific
   threshold or a Floquet-theory-derived boundary).
   3 x 2 contingency, chi^2(2) at p = 0.01, critical 9.21.
   STILL TESTS CHI^2 OF INDEPENDENCE; the non-circularity rests
   ENTIRELY on the cutpoints being non-data-driven.

D. Ordinal trend + nonmonotonicity test.
   STATUS: LIVE CANDIDATE (cleanest non-circular shape).
   Use a Jonckheere-Terpstra test of "is vf vs S/U monotone?" --
   the v0.7a' U-shape predicts FAIL-TO-FIT-MONOTONE. This is a
   hypothesis-specific test, NOT chi^2 of independence. Verdict:
   pass if Jonckheere-Terpstra fails to reject H_0 (vf and S/U
   are not monotonically related); fail if monotone fit succeeds
   (would contradict the U-shape).
   df / critical: pre-registered from the test's null distribution.

Recommendation: D is the cleanest anti-circular path. C is allowed
ONLY if the zone boundaries are explicitly justified physically.
A and B are rejected.
```

## Open Design Questions (Resolved in v0.9a Form Lock)

1. **v0.9a form choice (C/D).** D is the cleanest anti-circular
   form. C is allowed with explicit physical motivation for zone
   boundaries.

2. **If D: which monotone-trend test?** Candidates: Jonckheere-
   Terpstra, Cochran-Armitage trend test on ordered categories,
   Mann-Kendall on the per-row (vf, S) pairs. Each has different
   power characteristics.

3. **If C: physical motivation for zone boundaries?** Must be
   pre-registered with explicit physical/Hamiltonian rationale, NOT
   data-driven from v0.7a' quartile cutpoints.

4. **Predictor licensing.** A pass at v0.9a licenses v0.9b
   (held-out predictor on the 250-row subset). The held-out
   discipline is straightforward (mirrors v0.5b); the audit shape
   is what makes v0.9 honest.

## Pre-Mortem (Recorded Now)

```text
Risk 1: v0.9a passes by repeating v0.7a' under a different name.
        Mitigation: anti-circular discipline rejects forms A, B
        outright. The v0.9a form lock must include both
        non-circularity sentences explicitly. Receipts must record
        which of (feature, statistic, domain) was broken.

Risk 2: v0.9a fails because the U-shape is real but our specific
        pattern hypothesis (e.g., "vf < 0.2 has 50% S") doesn't
        match the actual data distribution.
        Mitigation: this would be a clean structural-negative
        result; the chapter closes on the hypothesis-specific
        test, not on the U-shape itself. v0.9c could register a
        different specific hypothesis.

Risk 3: The U-shape is real but is bin-locality-driven (some m_3
        sub-bin produces the high-S Q1 + high-S Q4 pattern by
        chance). v0.7a' alignment was 0.698 against branch_label
        (below 0.8), but the alignment was on (m_3, z_0) buckets,
        not on m_3 alone.
        Mitigation: v0.9a must report a per-m_3 vf-pattern
        diagnostic (S_fraction by vf-zone, stratified by m_3)
        before any verdict is licensed.

Risk 4: The 250-row subset is biased toward shorter-period orbits
        (attrition concentrated at high-m_3 long-period). The U-
        shape may be specific to the short-period regime.
        Mitigation: per-m_3 analyzable counts are on the v0.8a
        receipt; v0.9b's held-out partition will see thin folds
        at high m_3.
```

## Stop Conditions

1. **v0.9a form lock fails anti-circular audit.** Form rejected
   pre-compute; chapter closes immediately on the v0.9 hypothesis.

2. **v0.9a constant-feature retirement.** Structurally unlikely
   given vf_sd = 0.144 on the 250-row subset.

3. **v0.9a alignment-tightness > 0.8.** v0.9b cannot inherit the
   default leave-one-m_3-bin-out partition.

4. **v0.9a fails the hypothesis-specific test.** Chapter closes as
   a structural-negative on the specific pattern hypothesis. v0.9c
   (a different specific hypothesis) may be registered separately,
   but the user should weigh bandwidth tradeoff against Phase 15+.

5. **v0.9b fails the held-out gate.** Chapter closes as a
   projection-limit (mirroring v0.5) on the signed direction-
   composition mechanism.

## Lock-In Statement

This parent registration is committed before:

- any v0.9a form lock has been authored,
- any specific pattern hypothesis has been pre-registered,
- the per-m_3 vf-pattern diagnostics have been examined.

The body / projection / observable, the operational definition of
vf (inherited from v0.7a), the anti-circular framing discipline,
the disallowed-feature list, and the inheritance discipline are
locked. Any change after the v0.9a form lock is a re-registration
of the parent, not a refinement of a child form.

The anti-circular framing is **permanent discipline** for any
future v0.9-family registration.

## Doc Trail

- `kfacet_v08_writeup.md` -- v0.8 structural-negative chapter close
  (opens v0.9).
- `kfacet_v08a_purity_quartile_audit_form.md` -- v0.8a fails verdict
  + density-asymmetry diagnostic.
- `kfacet_v07a_prime_restricted_scope_form.md` -- v0.7a' PASS
  (the load-bearing observation v0.9 must NOT silently re-test).
- `kfacet_v07a_velocity_fraction_audit_form.md` -- v0.7a parent
  (vf operational definition).
- `kfacet_v05a_branch_map_form.md` -- v0.5a branch hash (alignment-
  tightness comparison axis).

---

Parent registration locked. v0.9a form lock pending. The anti-circular
framing is the load-bearing discipline: any v0.9a form that does NOT
break at least one of (feature derivation, test statistic, evaluation
domain) relative to v0.7a's chi^2-of-independence-on-vf-quartiles is
rejected pre-compute.
