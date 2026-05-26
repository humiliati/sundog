# K_facet v0.8 Methodology + Result Handoff

Status: closed structural-negative (direction-purity does not capture
the v0.7a' signal), 2026-05-24.
Audience: collaborators, paper-side writers, future coding agents.
Canonical sources: `kfacet_v08_mechanism_preregistration.md`,
`kfacet_v08a_purity_quartile_audit_form.md`.
Companion (closed v0.7 chapter): `kfacet_v07_writeup.md`.

## One-Line Read

v0.8 closes as a **structural-negative on the unsigned direction-
purity hypothesis**. The v0.7a' U-shape signature (chi^2 = 16.43,
p = 9.3e-4 on the 250-row analyzable subset) was hypothesized to be
captured by `purity = abs(vf - 0.5)`; v0.8a tested this with the
same quartile chi-squared pipeline that produced v0.7a's positive
and **failed at chi^2 = 4.94, p = 0.176**. The diagnostic
`purity_signed` contingency reproduced v0.7a's chi^2 = 16.43
exactly (linear transform of vf; same ordering), confirming the
v0.7a' signal is real but **does not live in the unsigned distance
from 0.5**. The vf distribution is density-asymmetric (184/250 rows
below vf = 0.5, max vf = 0.87, no rows near vf = 1), violating the
purity transform's symmetry assumption. **The signal lives in the
signed vf direction**, and the v0.8 chapter closes on the purity
shadow as a structural-negative. v0.9 opens on signed
direction-composition.

## What v0.8 Tested

v0.8 opened after v0.7's qualified-positive close. v0.7a' produced
a non-monotone U-shape on the velocity-fraction quartiles: Q1 49% S,
Q2 18% S, Q3 29% S, Q4 43% S. The U-shape suggested **direction-
purity** -- the distance of gamma_1's velocity-fraction from the
maximally-mixed point at vf = 0.5 -- as the load-bearing axis. v0.8
tested this with a pre-registered chi-squared quartile audit on the
purity scalar.

Two pre-registered sub-chapters:

```text
v0.8a: purity-quartile audit
       Domain:      250 analyzable supp-B rows (v0.7a integration_
                    blocked == False).
       Feature:     purity(row) = abs(vf(row) - 0.5).
       Binning:     quartiles within the 250-row subset.
       Test:        4 x 2 chi-squared (Q_purity vs S/U), df = 3,
                    critical 11.34 at p = 0.01.
       Result:      purity_quartile_fails_audit
                    (chi^2 = 4.94, p = 0.176, alignment 0.587).

v0.8b: held-out purity predictor (NOT licensed under v0.8a fails).
```

The audit form was pre-registered as candidate A of the v0.8 parent
registration's four candidate forms (purity-quartile, monotone
purity-threshold, Spearman correlation, median-split). A was
selected for inheritance fidelity with v0.7a' (same quartile
chi-squared pipeline). The diagnostic purity_signed contingency was
required on the receipt as the asymmetric-U-shape diagnostic.

## Audit Chain Methodology

The v0.8 chain inherited v0.7's discipline (audit-then-predictor
separation, alignment-tightness guard at 0.8/0.95, sparse-cell
fallback tree with seed 20260523, constant-feature retirement at
sd < 0.01) and added one new register-shaping discipline:

**Domain restriction as permanent.** The v0.7a integration-attrition
verdict (23 blocked + 11 sanity-failed = 34/273 catalog rows
unevaluable at variational precision 1e-12) is carried as a
permanent domain restriction on v0.8. The v0.8 receipts explicitly
record the v0.7a attrition counts; v0.8's claims apply only to the
250-row analyzable subset. This avoids the failure mode of treating
attrition as a missing-data problem to be imputed past.

Two stages, both paper-side first:

1. **v0.8 parent registration**
   (`kfacet_v08_mechanism_preregistration.md`). Locked body /
   projection / observable; operational definition of purity =
   `abs(vf - 0.5)`; v0.7a non-circularity sentence re-asserted
   verbatim; disallowed-feature list extended (vf itself,
   purity_signed as primary, v0.7a' Q_vf quartile assignments);
   five circularity-risk locks (eigenvalue-choice, well-definedness,
   feature-extraction, branch-shadow conflation, v0.7a'-conflation);
   sketched four candidate audit forms A-D and three candidate
   predictor forms E-G.

2. **v0.8a child form lock + verdict**
   (`kfacet_v08a_purity_quartile_audit_form.md`). Selected A
   (purity-quartile audit) over B/C/D for inheritance fidelity.
   Verdict landed: `purity_quartile_fails_audit`. The diagnostic
   purity_signed contingency reproduced v0.7a's chi^2 = 16.43
   exactly, surfacing the density-asymmetry explanation.

## Result And Structural-Negative Verdict

```text
v0.8a verdict:   purity_quartile_fails_audit
  domain:           250 analyzable rows (87 S / 163 U).
  chi^2_purity:     4.943205
  p_value:          0.1760
  df:               3
  critical:         11.34
  alignment:        0.587  (well below 0.8 warning threshold)
  sd(purity):       0.0961 (above 0.01 retirement floor; feature
                            has real variation, just isn't captured
                            by this quartile chi-squared).
  test_branch:      chi_squared (min_expected = 21.6, asymptotic
                                 conditions met).

  primary contingency (Q_purity, S/U):
    Q1   N=63   S=21   33.3%   vf ~ [0.44, 0.56]    mid
    Q2   N=62   S=18   29.0%   vf ~ [0.39, 0.44] U [0.56, 0.61]
    Q3   N=62   S=19   30.6%   vf ~ [0.28, 0.39] U [0.61, 0.72]
    Q4   N=63   S=29   46.0%   vf ~ [0, 0.28]    U [0.72, 0.87]

    Q4 (highest purity) elevated at 46%; Q1-Q3 flat ~30%.
    Pattern not strong enough to reach chi^2(3) > 11.34.

  diagnostic purity_signed contingency (signed direction):
    Q1   N=63   S=31   49.2%   vf in [0, 0.30]     positional pure
    Q2   N=62   S=11   17.7%   vf in [0.30, 0.42]
    Q3   N=62   S=18   29.0%   vf in [0.42, 0.50]
    Q4   N=63   S=27   42.9%   vf in [0.50, 0.87]  velocity pure

    chi^2 = 16.43 (reproduces v0.7a' exactly; linear transform of vf
                   preserves quartile ordering).
    asymmetry: |S(Q1) - S(Q4)| = 0.064  (< 0.2 flag threshold)

  vf distribution diagnostic:
    vf < 0.5:   184 / 250 (73.6%)
    vf > 0.5:    66 / 250 (26.4%)
    vf max:      0.8653  (no rows near vf = 1)
```

The publishable v0.8 statement:

> v0.8 tested the hypothesis that v0.7a's U-shaped velocity-fraction
> signal is captured by an unsigned direction-purity scalar
> `purity = abs(vf - 0.5)`. The hypothesis fails: chi-squared(3) =
> 4.94 vs critical 11.34, p = 0.176, on the 250-row analyzable
> subset. The diagnostic purity_signed contingency reproduces
> v0.7a's chi-squared = 16.43 exactly, confirming the v0.7a'
> positive is real but lives in the **signed** vf direction, not
> in unsigned distance from 0.5. The failure is explained by the
> vf distribution's density asymmetry: 74% of rows lie below
> vf = 0.5 and max vf = 0.87, so the "vf near 1" pure end of the
> "purity" interpretation is structurally under-populated by the
> catalog. The unsigned purity transform treats vf = 0 and vf = 1
> as equivalent, but the catalog does not. The v0.8 chapter closes
> on the purity shadow; v0.9 opens on signed direction-composition.

Three structural sub-results preserved for v0.9:

1. **The v0.7a' signal is signed, not symmetric.** The chi^2 = 16.43
   on the 250-row subset is real, branch-independent (alignment
   0.698), and reproducible. But it is NOT captured by the unsigned
   purity transform. The signal lives in some signed-direction
   function of vf, not in |vf - 0.5|.

2. **The vf distribution is density-asymmetric.** 74% of analyzable
   rows have vf < 0.5; max vf = 0.87. Any v0.9 hypothesis that
   treats the two "pure ends" of vf as equivalent will continue
   to fail. The catalog's actual distribution is the constraint.

3. **The Q1-vs-Q4 S-fraction split is roughly symmetric.** Q1_signed
   (vf in [0, 0.30]) and Q4_signed (vf in [0.50, 0.87]) both
   concentrate stable orbits (49% and 43% S respectively). The
   "purity-of-direction correlates with stability" interpretation
   is correct in **outcome** but not in **density**. v0.9 needs to
   test a hypothesis that respects both.

Joint v0.4 + v0.5 + v0.6 + v0.7 + v0.8 envelope:

> Five sequential pre-registered low-dimensional projections of the
> supp-B body have produced five distinct chapter-close types:
> structural-negative on Z_2 symmetry (v0.4); projection-limit on
> 2-bit catalog branch hash (v0.5); conditional-independence on
> continuous catalog-coordinate (E, |L|) shadow (v0.6); integration-
> attrition + qualified-positive on restricted domain for Floquet
> velocity-fraction (v0.7); and structural-negative on Floquet
> direction-purity (v0.8). The v0.7a' positive on the 250-row
> analyzable subset is the only statistically-significant
> non-branch-aligned signal in the chain, and v0.8a confirms it
> is signed (not symmetric in vf). v0.9 opens on the signed
> direction-composition hypothesis with explicit anti-circular
> framing.

## Reproducibility Surface

```bash
# v0.8a purity-quartile audit (catalog-only, seconds):
python scripts/v08a_purity_audit.py
```

Key receipt directory:

```text
results/isotrophy/k-facet-v08a-purity-audit/manifest.json
results/isotrophy/k-facet-v08a-purity-audit/per_row_table.csv
results/isotrophy/k-facet-v08a-purity-audit/contingency_table_purity.csv
results/isotrophy/k-facet-v08a-purity-audit/contingency_table_purity_signed.csv
results/isotrophy/k-facet-v08a-purity-audit/per_m3_analyzable_counts.csv
```

Load-bearing constants live in the form-lock documents:

```text
kfacet_v08_mechanism_preregistration.md
  body / projection / observable; operational definition of purity;
  five circularity-risk locks; disallowed-feature list (vf itself,
  purity_signed as primary, v0.7a' Q_vf assignments); inheritance
  discipline.

kfacet_v08a_purity_quartile_audit_form.md
  Selected candidate A; quartile binning method (numpy linear);
  alignment-tightness thresholds 0.8/0.95; sparse-cell fallback;
  asymmetry-diagnostic threshold 0.2; verdict landed
  `purity_quartile_fails_audit`.
```

## Where v0.9 Opens

The v0.8 structural-negative confirms the v0.7a' signal is signed,
not symmetric. v0.9 opens on the **signed Floquet direction-
composition** mechanism (codex direction, 2026-05-24). Parent
registration at `kfacet_v09_mechanism_preregistration.md`.

The v0.9 parent's load-bearing methodological challenge is
**anti-circular framing**: any chi-squared-of-independence test on
vf-ordering (whether expressed as raw vf quartiles, signed quartiles,
or binary vf-side threshold) will reproduce v0.7a's chi^2 = 16.43
by the linear-transform invariance. v0.9 cannot honestly use chi^2
of independence as the test statistic; it must test a **specific
pattern hypothesis** that v0.7a' did not pre-register.

The parent registration's four candidate v0.9a forms:

```text
A. Ordered quartile audit (Q1/Q2/Q3/Q4 labels preserved):
   CIRCULAR RISK -- reproduces v0.7a' chi^2 exactly.
B. Threshold audit at vf < 0.5 vs vf >= 0.5:
   CIRCULAR RISK -- binary collapse of the same Q ordering.
C. Three-zone audit (positional-pure / mixed / velocity-heavy):
   Non-circular IF zone boundaries are pre-registered with
   physical motivation, NOT data-driven from v0.7a's quartile
   cutpoints.
D. Ordinal trend + nonmonotonicity test:
   Non-circular. Tests "is vf vs S/U monotone?" -- the
   v0.7a' U-shape predicts FAIL-TO-FIT-MONOTONE. This is a
   hypothesis-specific test, not chi^2 of independence.
```

Forms A and B are excluded under the anti-circular discipline.
Forms C and D are the live candidates; D is the cleanest
hypothesis-specific test of the U-shape pattern.

## Doc Trail

- `kfacet_v08_mechanism_preregistration.md` -- v0.8 parent
  registration.
- `kfacet_v08a_purity_quartile_audit_form.md` -- v0.8a child form
  lock + structural-negative verdict.
- `kfacet_v07_writeup.md` -- v0.7 qualified-positive chapter close
  (companion).
- `kfacet_v07a_velocity_fraction_audit_form.md`,
  `kfacet_v07a_prime_restricted_scope_form.md` -- v0.7 form locks
  (predecessors).
- `kfacet_v09_mechanism_preregistration.md` -- v0.9 parent
  registration (signed Floquet direction-composition; opens v0.9).

The chapter closes cleanly. v0.7a's positive is preserved
unchanged. v0.8a's structural-negative on unsigned purity is
preserved as a methodological finding: the v0.7a' U-shape is
real but NOT capturable by symmetric distance-from-mixed under
the catalog's actual vf distribution. The chain has now produced
six sequential pre-registered chapters with five distinct
chapter-close types; v0.9 extends the chain with an explicitly
anti-circular framing.
