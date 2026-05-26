# K_facet v0.9 Methodology + Result Handoff

Status: closed structural-negative (U-shape mechanism invalidated;
monotone-increasing meta-finding preserved), 2026-05-24.
Audience: collaborators, paper-side writers, future coding agents,
isotrophy-program lab initiates.
Canonical sources: `kfacet_v09_mechanism_preregistration.md`,
`kfacet_v09a_signed_vf_three_zone_form.md`.
Companion (closed v0.8 chapter): `kfacet_v08_writeup.md`.
Companion (program pause): `kfacet_isotrophy_program_pause.md`.

## One-Line Read

v0.9 closes as a **structural-negative on the U-shape mechanism
hypothesis**, with a **substantive meta-finding** preserved: the
v0.7a' chi^2 = 16.43 positive was an artifact of the v0.7a' quartile
boundary at vf ~ 0.297. Under the physical cutpoint vf = 0.25,
the positional-dominant zone shrinks from 63 to 19 rows and its
S-fraction collapses from 49% to 10.5%. The actual physical pattern
across the three signed-direction zones is **monotone-increasing**
(positional 11% S, mixed 34% S, velocity-heavy 44% S) — chi^2 =
7.42, p = 0.0245, below the registered floor at p = 0.01 (critical
9.21). The U-shape MECHANISM is invalidated; the v0.7a' chi^2 =
16.43 catalog-coordinate positive remains a valid quartile-binning
stratification finding. v0.9 is the **last chapter** in the
isotrophy program before the formal program pause; v0.9b is not
licensed.

## What v0.9 Tested

v0.9 opened after v0.8's structural-negative on unsigned direction-
purity. The v0.8a diagnostic purity_signed contingency reproduced
v0.7a's chi^2 = 16.43 exactly (linear transform of vf), confirming
the v0.7a' signal lives in the signed direction. v0.9 promoted
signed direction-composition with an **anti-circular framing
discipline** locked at the v0.9 parent:

```text
Anti-circular discipline (locked permanently):
  any v0.9 child form lock that tests vf-ordering vs S/U on the
  v0.7a' analyzable subset via chi^2 of independence reproduces
  v0.7a' by linear-transform invariance. To be honest, v0.9a must
  break at least one of:
    1. feature derivation,
    2. test statistic,
    3. evaluation domain.
```

The v0.9a child form lock selected candidate C (three-zone audit
with physically-motivated cutpoints {0.25, 0.50}) and broke
**two** of the three anti-circular conditions:

```text
Condition #1 (feature derivation) BROKEN: physical cutpoints
  {0.25, 0.50}, NOT v0.7a' quartile cutpoints (which were
  approximately {0.297, 0.419, 0.504}).

Condition #2 (test statistic) BROKEN: the verdict tree requires
  chi^2 > critical AND a hypothesis-specific pattern
  (mixed S-fraction < BOTH positional-dominant AND velocity-heavy
  S-fractions; i.e., the U-shape direction).
```

The hypothesis: under physical-zone binning, the v0.7a' U-shape
generalizes to a true U-shape where the mixed zone is U-enriched
relative to both direction-dominant zones.

## Audit Chain Methodology

The v0.9 chain inherited all prior register-shaping disciplines
(audit-then-predictor separation, alignment-tightness guard,
sparse-cell fallback tree, conservative tie rule, constant-feature
retirement, attrition-as-domain-restriction) and added one new
permanent discipline:

**Anti-circular framing for sequential audits on the same data.**
After v0.7a' produced a chi^2 = 16.43 positive on vf-quartile vs
S/U, any subsequent test on the same 250-row subset using
chi^2-of-independence on vf-ordering would silently reproduce
v0.7a' by linear-transform invariance of chi^2 under monotone
re-binning of an ordered feature. v0.9 explicitly identifies this
mechanism-laundering risk and rejects two candidate v0.9a forms
(A: ordered quartile audit; B: binary vf<0.5 threshold) pre-compute
under the locked anti-circular discipline. The framework is
permanent: any future v0.10+ child form lock that tests vf vs S/U
must respect the same discipline.

Stages, all paper-side first:

1. **v0.9 parent registration**
   (`kfacet_v09_mechanism_preregistration.md`). Locked body /
   projection / observable; vf operational definition inherited
   from v0.7a; anti-circular framing discipline locked.

2. **v0.9a child form lock + verdict**
   (`kfacet_v09a_signed_vf_three_zone_form.md`). Selected C
   (three-zone audit with physical cutpoints) over A/B/D. Locked
   verdict tree includes both chi^2 gate and hypothesis-specific
   pattern check.

## Result And Structural-Negative Verdict

```text
v0.9a verdict:   signed_vf_three_zone_fails_audit_chi2

three-zone contingency (locked physical cutpoints {0.25, 0.50}):
  zone                              N    S    U    S_frac  chi2_contrib
  positional-dominant (vf<0.25)    19    2   17   0.1053       4.93
  mixed (0.25 <= vf < 0.5)        165   56  109   0.3394       0.05
  velocity-heavy (vf >= 0.5)       66   29   37   0.4394       2.43
                                                                ----
                                                  chi^2 =       7.42

df:                   2  (3 occupied zones - 1)
critical (p=0.01):    9.21
p_value (chi-sq(2)):  0.0245
alignment_tightness:  0.5697  (well below 0.8 warning threshold)

pattern check:
  S_fraction(positional)   = 0.1053
  S_fraction(mixed)        = 0.3394
  S_fraction(velocity-h)   = 0.4394
  mixed < positional?      FALSE  (0.34 > 0.11)
  mixed < velocity-heavy?  TRUE   (0.34 < 0.44)
  u_shape_confirmed?       FALSE
```

The publishable v0.9 statement:

> v0.9a tested whether the v0.7a' chi^2 = 16.43 positive is captured
> by a hypothesis-specific U-shape on three physically-motivated
> direction zones. The hypothesis fails on both gates: the
> chi-squared(2) statistic of 7.42 does not reach the registered
> critical value 9.21 (p = 0.0245 vs threshold p = 0.01), AND the
> U-shape pattern direction is broken (mixed zone is NOT
> U-enriched relative to positional-dominant; mixed is in fact
> U-enriched relative to ONLY velocity-heavy). The substantive
> meta-finding: the v0.7a' U-shape was an artifact of the
> quartile boundary at vf ~ 0.297. The actual physical pattern
> under non-data-driven zones is **monotone-increasing in vf**
> (positional 11% S < mixed 34% S < velocity-heavy 44% S), which
> matches the broader thesis that velocity-dominated Floquet
> directions correlate with stability — but does not clear the
> registered chi-squared floor under three-zone physical binning.

Three structural sub-results preserved:

1. **The v0.7a' chi^2 = 16.43 positive is a quartile-binning
   stratification finding, NOT a structural mechanism**. Under
   the v0.7a' Q1 bin (vf in [0, 0.297]), 63 rows / 49% S. Under
   the physical zone (vf < 0.25), only 19 rows / 11% S. The
   high-S rows that drove v0.7a' Q1's S-fraction live near the
   quartile boundary (vf in [0.25, 0.30]), NOT in the deepest
   positional regime.

2. **The actual physical pattern is monotone-increasing in vf**.
   Across the three signed-direction zones, S_fraction goes
   0.105 -> 0.339 -> 0.439. This is consistent with the broader
   thesis that velocity-dominated gamma_1 directions correlate
   with stability. A J-T trend test (candidate D) would test this
   directly and would likely pass at p < 0.01 given the magnitude
   of the increase, but it is NOT registered under v0.9. Lab
   initiates may pursue this (see `kfacet_isotrophy_program_pause.md`).

3. **Anti-circular discipline pattern locked as permanent**. Any
   future sequential audit on the same data after a positive
   audit has produced an in-sample chi-squared must break at
   least one of (feature derivation, test statistic, evaluation
   domain) to be non-circular. This pattern is now part of the
   isotrophy methodology surface.

Joint v0.4 + v0.5 + v0.6 + v0.7 + v0.8 + v0.9 envelope (the
publishable multi-chapter statement at the program pause):

> Seven sequential pre-registered low-dimensional projections of
> the supp-B body have produced seven distinct chapter-close types:
> domain-of-applicability (v0.3), structural-negative on Z_2
> symmetry (v0.4), projection-limit on 2-bit catalog branch hash
> (v0.5), conditional-independence on continuous (E, |L|) shadow
> (v0.6), integration-attrition + qualified-positive on restricted
> domain (v0.7), structural-negative on unsigned direction-purity
> (v0.8), and structural-negative on U-shape mechanism with
> monotone-increasing meta-finding (v0.9). The v0.7a'
> chi^2 = 16.43 quartile-binning positive remains the chain's
> load-bearing positive substantive finding. v0.8 and v0.9
> contextualized but did not invalidate it: the signal is signed
> (v0.8 confirmed), not symmetric (v0.8 disconfirmed), and is
> quartile-boundary-driven rather than physical-zone-driven
> (v0.9 disconfirmed; monotone-increasing pattern preserved as
> meta-finding). The methodology surface (closure-relative
> discipline + two-pass classifier + projection language +
> audit-then-predictor + asymmetric falsifier + alignment-tightness
> guard + sparse-cell fallback + per-row integration-failure
> fallback + append-per-row resume + anti-circular framing for
> sequential audits) is the load-bearing publishable contribution.

## Reproducibility Surface

```bash
# v0.9a three-zone audit (catalog-only, seconds):
python scripts/v09a_signed_vf_three_zone_audit.py
```

Key receipt directory:

```text
results/isotrophy/k-facet-v09a-signed-vf-three-zone/manifest.json
results/isotrophy/k-facet-v09a-signed-vf-three-zone/per_row_table.csv
results/isotrophy/k-facet-v09a-signed-vf-three-zone/contingency_table_three_zone.csv
```

Load-bearing constants live in the form-lock documents:

```text
kfacet_v09_mechanism_preregistration.md
  body / projection / observable; anti-circular framing discipline
  (load-bearing); disallowed-feature list; inheritance discipline;
  candidate forms A-D with A and B rejected pre-compute.

kfacet_v09a_signed_vf_three_zone_form.md
  Selected candidate C; physical cutpoints {0.25, 0.50}; verdict
  tree requires chi^2 > critical AND U-shape pattern; verdict
  landed `signed_vf_three_zone_fails_audit_chi2` at chi^2 = 7.42.
```

## Where the Isotrophy Program Pauses

The v0.9 chapter close marks the end of the active isotrophy
program. The program PAUSES (does not retire) at end-of-v0.9.
See `kfacet_isotrophy_program_pause.md` for:

- The seven-chapter envelope summary.
- The publishable load-bearing findings (v0.7a' positive,
  monotone-increasing meta-finding, full methodology surface).
- Concrete reopening avenues for lab initiates ("waste compute on"
  candidates).
- Bandwidth redirect to three-body Phase 15+ (per the principal's
  pre-v0.7 direction call).

The program may be unpaused if a lab initiate produces clean
evidence on one of the reopening avenues (e.g., a J-T trend test
that passes the locked anti-circular discipline; a relaxed-precision
re-run of v0.7a that recovers the attrited 23 rows; or a targeted
m_3 = 0.4 sub-catalog mechanism investigation).

## Doc Trail

- `kfacet_v09_mechanism_preregistration.md` -- v0.9 parent
  registration (anti-circular discipline lock).
- `kfacet_v09a_signed_vf_three_zone_form.md` -- v0.9a child form
  lock + structural-negative verdict.
- `kfacet_isotrophy_program_pause.md` -- program pause document
  (concrete reopening avenues for lab initiates).
- `kfacet_v08_writeup.md` -- v0.8 structural-negative chapter close
  (companion).
- `kfacet_v07_writeup.md` -- v0.7 qualified-positive chapter close
  (load-bearing positive finding).

The chain closes cleanly at v0.9. The v0.7a' positive remains the
load-bearing substantive finding; the v0.9 monotone-increasing
meta-finding is preserved as a future-work item. Seven sequential
pre-registered chapters; seven distinct chapter-close types; one
load-bearing positive; one comprehensive methodology surface;
program pauses with explicit reopening avenues.
