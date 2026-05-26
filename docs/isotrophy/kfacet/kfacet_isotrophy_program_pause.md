# K_facet Isotrophy Program Pause (2026-05-24)

Status: **PAUSED 2026-05-24** at end-of-v0.9. The chain ran from
v0.3 (December 2025 / January 2026, pre-anniversary) to v0.9
(May 2026), producing seven sequential pre-registered chapters
with seven distinct chapter-close types and one load-bearing
substantive positive (v0.7a' chi^2 = 16.43 on signed velocity-
fraction quartiles, restricted to 250 analyzable rows).

The program is **paused**, not retired. Reopening avenues are
recorded below for lab initiates with bandwidth to spare.

Audience: paper-side writers, future coding agents, isotrophy-
program lab initiates, anyone who wants to know what's left.

## What the Isotrophy Program Did

Sundog v. Isotrophy tested whether a "sundog theorem" makes
non-tautological predictions about Li & Liao 2025's three-body
catalog (specifically the supplementary-B piano-trio catalog of
273 m_3-parametrized periodic orbits, half stable and half
unstable). The program ran as seven sequential pre-registered
chapters testing progressively richer low-dimensional projections
of the catalog body:

```text
v0.3:  Gamma_i rank gate (D_3-equivariant; m_3 = 1 strict G.2).
       Outcome: domain-of-applicability. 20/21 m_3 = 1 strict-G.2
       rows resolve as structural zeros; O_617 is a clean
       opposite-strict row with a bridge direction outside the
       valid D3 representation. Daughter catalog (supp-B) is
       Z_2-or-smaller and OUT of D_3-domain.

v0.4:  Z_2 shadow on supp-B (tangent-isotypic + orbit-gauge-
       rigidity).
       Outcome: structural-negative. Both registered Z_2-shadow
       predictors fail. gamma_3 retired pre-sweep (F_beta does
       not preserve K_fib on supp-B). gamma_3'_orbit_pass2
       falsified at chi^2 = 1202 (vs critical 26.22).

v0.5:  2-bit catalog branch shadow on (m_3 < 1, z_0 < 0.3).
       Outcome: projection-limit (audit-passes / predictor-fails).
       v0.5a audit passes chi^2 = 34.99, p = 1.23e-7. v0.5b
       held-out predictor fails: accuracy delta -0.019, McNemar
       p = 1.0. Branch hash describes but does not predict.

v0.6:  Continuous catalog-coordinate (E, |L|) shadow.
       Outcome: conditional-independence. v0.6a catalog audit
       passes chi^2 = 33.70 but alignment-tightness 0.956 trips
       the warning guard. v0.6b within-branch audit (alignment-
       breaking) fails permutation p = 0.029. Energy quartile is
       a 1-to-1 label for m_3 sub-bin within the stable branch.

v0.7:  Floquet velocity-fraction direction shadow.
       Outcome: integration-attrition + qualified-positive on
       restricted domain. v0.7a catalog-wide audit attrited at
       8.4% (variational integrator step-size wall at high m_3).
       v0.7a' restricted-domain confirmation on 250 analyzable
       rows PASSES at chi^2 = 16.43, p = 9.3e-4, alignment 0.698.
       FIRST non-branch-aligned positive in the chain.

v0.8:  Unsigned direction-purity = abs(vf - 0.5).
       Outcome: structural-negative. v0.8a fails at chi^2 = 4.94.
       Diagnostic confirms the v0.7a' signal is signed (linear-
       transform invariance of vf preserves the quartile chi^2).
       Density-asymmetric vf distribution (74% below 0.5)
       violates the purity transform's symmetry assumption.

v0.9:  Signed three-zone direction-composition under physical
       cutpoints {0.25, 0.50}.
       Outcome: structural-negative on U-shape hypothesis +
       monotone-increasing meta-finding. v0.9a fails at
       chi^2 = 7.42 (vs critical 9.21); U-shape pattern check
       also fails. Under physical zones, the actual pattern is
       monotone-increasing in vf (positional 11% S, mixed 34% S,
       velocity-heavy 44% S). The v0.7a' U-shape was a quartile-
       boundary artifact at vf ~ 0.297.
```

## The Load-Bearing Findings

After seven chapters, three findings carry the program's substantive
output:

1. **The v0.7a' chi^2 = 16.43 positive**. On the 250-row analyzable
   supp-B subset (post v0.7a integration-attrition), the velocity-
   fraction quartile contingency is statistically significant
   (p = 9.3e-4) AND branch-independent (alignment 0.70 against the
   v0.5a branch hash). This is the FIRST non-branch-aligned
   positive in the chain and remains uncontradicted by v0.8 and
   v0.9 (both of which contextualized it without invalidating it).

2. **The monotone-increasing meta-finding (v0.9a contingency)**.
   Under physical cutpoints {0.25, 0.50}, S_fraction is monotone
   in vf: positional 11%, mixed 34%, velocity-heavy 44%. This
   matches the broader thesis that velocity-dominated Floquet
   directions correlate with stability. The chi-squared on three
   physical zones (7.42) does not clear the registered p = 0.01
   floor (critical 9.21), but the trend is real and would likely
   clear a J-T trend test (untested).

3. **The methodology surface**. The chain produced and locked
   a comprehensive set of pre-registration discipline tools that
   are reusable across mechanism-search programs:

   - Closure-relative gate discipline (v0.3 origin).
   - Two-pass classifier with empirically-calibrated tolerances
     (v0.4a).
   - Projection-language vocabulary (body / shadow / projection-
     limit, from CROSS_SUBSTRATE_NOTES.md).
   - Audit-then-predictor separation (v0.5).
   - Asymmetric McNemar + positive-delta falsifier for held-out
     predictors (v0.5b).
   - Alignment-tightness guard against prior-chapter feature
     shadows (v0.6a, with thresholds 0.8 warning / 0.95 severe).
   - Sparse-cell fallback tree (v0.6b: chi-squared / exact
     permutation / inconclusive_sparse, with locked permutation
     seed 20260523 and n_permutations = 10000).
   - Non-circularity sentence template for Floquet-derived
     features (v0.7a: "use eigenvectors as geometric directions
     only").
   - Per-row integration-failure fallback with attrition gate
     at 5% of catalog (v0.7a R2.A).
   - Append-per-row resume mode for crash resilience (v0.7a R2.C).
   - Anti-circular framing discipline for sequential audits on
     the same data (v0.9 parent).

## Concrete Reopening Avenues for Lab Initiates

Things that could be done if a lab initiate has bandwidth to "waste
compute on." Each is a real research direction that the principal
deemed not worth the bandwidth tradeoff against three-body
Phase 15+, but which would yield publishable substance if
followed up.

### 1. Jonckheere-Terpstra trend test on v0.9a three-zone data (LOW EFFORT, HIGH SUBSTANCE)

**Status**: registered as candidate D in `kfacet_v09_mechanism_preregistration.md`,
never run.

**Hypothesis**: the three-zone contingency `S_fraction = 0.105 /
0.339 / 0.439` shows a monotone-increasing trend in vf that would
pass at p < 0.01 under a J-T trend test (which has more power than
chi-squared for ordered alternatives).

**Implementation**: ~30 minutes paper-side + seconds runner. Read
`results/isotrophy/k-facet-v09a-signed-vf-three-zone/per_row_table.csv`,
apply J-T trend test on zone-ordered S/U using scipy.stats.
Pre-register the one-sided test against H_0 = no monotone trend.

**Non-circularity**: this breaks condition #2 (test statistic)
of the v0.9 parent's anti-circular discipline; J-T is NOT chi-
squared of independence. Allowed under the locked framework.

**Why deferred**: another chapter's worth of paper-side work after
seven chapters had already been written; bandwidth shifted to
Phase 15+.

### 2. v0.7a' held-out predictor on the analyzable subset (LOW-MEDIUM EFFORT)

**Status**: parent registration mentions v0.7b as a candidate;
NEVER licensed under the v0.7a attrition verdict; could be
licensed under v0.7a' restricted-scope PASS retrospectively if
re-registered.

**Hypothesis**: the v0.7a' chi^2 = 16.43 in-sample positive is
held-out-predictive at p <= 0.01 under leave-one-m_3-bin-out on
the 250 analyzable rows.

**Implementation**: re-register `kfacet_v07b_*` paper-side as a
held-out predictor on the restricted scope (v0.7a' domain). Use
v0.5b's discipline verbatim (asymmetric McNemar, positive-delta
gate, leave-one-m_3-bin-out partition). Per-fold accuracy + pooled
McNemar at p <= 0.01.

**Risk**: m_3 = 0.4 is the load-bearing fold; held-out it may flip
the bucket-majority direction (same failure mode v0.5b exhibited).
Lab initiate should run v0.5b-style diagnostic per fold.

**Non-circularity**: domain is the v0.7a' subset (already used in
v0.7a' for in-sample chi-squared); held-out partition breaks the
in-sample-vs-held-out condition cleanly.

### 3. v0.7a relaxed-precision re-run (MEDIUM-HIGH EFFORT)

**Status**: never registered. The v0.7a R1 amendment locked
symplecticity gate at 1e-4 (relaxed from 1e-6) on the variational
precision 1e-12 substrate; the integration attrition came from
DOP853 step-adapter failures, not sanity-gate failures.

**Hypothesis**: at variational precision 1e-10 or 1e-9 (matching
v0.4a Pass 2), the DOP853 step adapter clears the 23 attrited
rows, allowing a catalog-wide v0.7a verdict.

**Implementation**: register a v0.7c paper-side at relaxed
precision. Re-run the full audit (~3-5 hours expected). The
restricted-domain v0.7a' verdict would be either confirmed (the
attrited rows match the analyzable subset's pattern) or contested
(the attrited rows have systematically different vf values).

**Risk**: precision change affects every per-row vf value; the
catalog-wide chi-squared at the new precision is its own audit and
cannot be compared to v0.7a' directly without explicit precision
disclosure on every receipt.

**Non-circularity**: catalog-wide claim under the new precision is
a fresh audit, not a re-test of v0.7a'.

### 4. m_3 = 0.4 sub-catalog targeted mechanism investigation (MEDIUM EFFORT, HIGH SUBSTANCE)

**Status**: never registered. The m_3 = 0.4 cluster has been the
load-bearing positive throughout the chain (v0.7a Q1 49% S; v0.5a
(m_3<1, z_0<0.3) branch 56% S).

**Hypothesis**: the m_3 = 0.4 sub-catalog (55 rows / 35 S / 20 U)
has a substrate-specific mechanism that the catalog-wide audits
average out. A within-m_3-bin audit on this single mass ratio
would reveal what makes the bin special.

**Implementation**: register a v0.10 chapter on the m_3 = 0.4
sub-catalog. Use any of the prior mechanism families (vf, |L|,
joint, gamma_1 direction) but restrict the domain to 55 rows.
Apply v0.6b-style within-stratum discipline.

**Risk**: low N (55 rows) limits chi-squared power; sparse-cell
fallback will likely fire frequently. The substantive return is
high if the mechanism is clean; the statistical return is
power-limited.

**Non-circularity**: the m_3 = 0.4 sub-catalog is a domain
restriction; conditioning on m_3 = 0.4 uses catalog parameter
only, not the stability label.

### 5. Joint (vf, Q_E) audit on the 250 analyzable rows (MEDIUM EFFORT)

**Status**: never registered. v0.6a (E quartile, chi^2 = 33.70)
and v0.7a' (vf quartile, chi^2 = 16.43) both passed in-sample;
the relationship between the two has not been characterized.

**Hypothesis**: vf and Q_E carry overlapping but partially
independent stability information. A joint 4x4x2 contingency
(Q_vf x Q_E x S/U) would reveal whether one is a proxy for the
other or whether they explain different variance in S/U.

**Implementation**: register a v0.10 chapter on joint
(vf, Q_E) stratification on the 250 analyzable rows. Compute the
4x4x2 contingency; run a stratified chi-squared (Cochran-Mantel-
Haenszel) to test conditional independence of Q_vf and S/U given
Q_E.

**Risk**: joint contingency is high-df; many cells will be sparse;
sparse-cell fallback fires.

**Non-circularity**: both Q_vf and Q_E are inherited from prior
chapters; the joint test is a NEW hypothesis (conditional
independence) that v0.6a and v0.7a' did not pre-register.

### 6. Cross-substrate transfer test (HIGH EFFORT)

**Status**: cross-substrate notes exist at
`docs/threebody/CROSS_SUBSTRATE_NOTES.md` sections 6-7.

**Hypothesis**: the v0.7a' velocity-fraction shadow is substrate-
specific to supp-B; applying the same mechanism to a different
three-body sub-catalog (e.g., free-fall, choreography, or mesa
analog) would test whether the signal is supp-B-specific or
generalizes.

**Implementation**: identify a different three-body sub-catalog
with comparable structure (stability labels + periodic-orbit ICs).
Apply the v0.7a operational definition + the v0.9a three-zone
audit. Compare verdicts.

**Risk**: identifying a comparable catalog is non-trivial;
re-implementation of variational integration substrate is real
work. High effort, but high publishable substance if it works.

**Non-circularity**: cross-substrate transfer is by design a fresh
test on new data.

### 7. Action-angle / KAM-style decomposition (HIGH EFFORT, THEORETICAL)

**Status**: mentioned as a candidate v0.7 mechanism family before
v0.7 picked gamma_1 direction-of-instability; never registered.

**Hypothesis**: the supp-B orbits have action-angle representations
that stratify stability via integrability proximity. KAM theory
says nearly-integrable orbits are stable; computing action-angle
variables per row would test this directly.

**Implementation**: heavy theoretical machinery. Per-row
action-angle decomposition requires extended variational
integration over multiple periods (not just one) plus
diagonalization in canonical coordinates. Compute cost: weeks.

**Non-circularity**: action-angle variables are new geometric
quantities, not derived from prior-chapter features.

### 8. Full-catalog audit at variational precision 1e-10 with v0.7a R2.A discipline

**Status**: candidate alternative to avenue 3. If attrition at
1e-10 is below 5%, the catalog-wide v0.7a verdict could be
recovered without re-registering precision.

**Implementation**: re-run `scripts/v07a_velocity_fraction_audit.py`
with rtol = atol = 1e-10 (modify the locked constants paper-side
first). Compare attrition rates and pattern stability against
the 1e-12 result.

**Risk**: precision change is a re-registration; needs paper-side
discipline.

## Reusable Methodology Surface (For Programs Beyond Isotrophy)

The discipline tools listed under "load-bearing findings" item 3
are not isotrophy-specific. They apply to any mechanism-search
program where:

- Multiple candidate mechanisms compete on the same data,
- Sequential audits risk silent re-testing under different labels,
- Catalog-coordinate vs orbit-dynamics axes need to be distinguished
  for non-circularity,
- Integration-precision walls may limit catalog-wide claims,
- Held-out generalization is the publishable predictor gate.

Future mesa, geometry, or three-body Phase 15+ work that follows
the same pattern is invited to inherit these disciplines as
appropriate.

## Bandwidth Redirect

Per the principal's pre-v0.7 direction (which the v0.7a' surprise
positive temporarily overrode), bandwidth redirects to:

- **three-body Phase 15+** (survival pocket investigation, the
  current high-ROI path).
- **Coarse-graining proof trunk** (P0/P1 priority per
  `docs/TODO.md`).
- **Public surface updates** including `threebody.html` (flagged
  as seriously stale during v0.7 close).

The isotrophy section's pause does NOT release the program's
discipline; any future re-opening must respect the locked
methodology surface AND the anti-circular framing discipline for
sequential audits on the v0.7a' analyzable subset.

## Doc Trail

- `kfacet_v03h_writeup.md` -- v0.3 chapter close.
- `kfacet_v04_writeup.md` -- v0.4 chapter close.
- `kfacet_v05_writeup.md` -- v0.5 chapter close.
- `kfacet_v06_writeup.md` -- v0.6 chapter close.
- `kfacet_v07_writeup.md` -- v0.7 chapter close (load-bearing
  qualified-positive).
- `kfacet_v08_writeup.md` -- v0.8 chapter close.
- `kfacet_v09_writeup.md` -- v0.9 chapter close (this chapter's
  companion).
- `docs/threebody/CROSS_SUBSTRATE_NOTES.md` -- cross-substrate
  framing (avenue 6).
- `docs/isotrophy/sundog_v_isotrophy.md` -- full chapter-by-chapter dated log.

---

Isotrophy program paused at end-of-v0.9. Seven chapters, seven
close types, one load-bearing positive, one comprehensive
methodology surface, eight concrete reopening avenues for lab
initiates. Reopening is invited; mechanism-laundering is not.
