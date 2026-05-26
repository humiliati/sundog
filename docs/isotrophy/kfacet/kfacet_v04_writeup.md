# K_facet v0.4 Methodology + Result Handoff

Status: closed structural-negative, 2026-05-23.
Audience: collaborators, paper-side writers, future coding agents.
Canonical sources: `kfacet_v04a_domain_map_preregistration.md`,
`kfacet_v04b_gamma3_form.md` (retired baseline),
`kfacet_v04b_gamma3prime_form.md` (falsified baseline),
`kfacet_v04b_mechanism_preregistration.md`, and
[`../../threebody/CROSS_SUBSTRATE_NOTES.md`](../../threebody/CROSS_SUBSTRATE_NOTES.md)
sections 6-7 (projection-language framing for the v0.3 -> v0.4 transition).
Companion (closed v0.3 chapter): `kfacet_v03h_writeup.md`.

## One-Line Read

v0.4 closes as a **structural-negative on the Z_2 shadow as a stability
projector** for the Li-Liao supplementary-B piano-trio catalog. The Z_2
orbit-level domain map is well-defined (v0.4a, 273/273 in Z2_clean after
the two-pass classifier), but every registered Z_2-shadow stability
predictor either has its precondition fail (gamma_3 tangent-isotypic,
retired) or is empirically falsified against the per-m_3 stability
distribution (gamma_3'_orbit_pass2, chi^2 = 1202 vs critical 26.22).
Stability information on this catalog is not carried by the Z_2 projection
at either the tangent-isotypic or orbit-gauge-rigidity granularity.

## What v0.4 Tested

v0.4 opened after v0.3's domain-of-applicability finding: v0.3's `Gamma_i`
rank gate is `D_3`-equivariant and structurally inapplicable to
supplementary-B piano-trios, which carry only the `Z_2 = (12)`-swap
symmetry. v0.4 re-framed the supp-B catalog as a primary `Z_2` body
(not a daughter of strict G.2) and tested whether the `Z_2` shadow
contains stability-predictive structure.

Two pre-registered chapters:

```text
v0.4a: domain-of-applicability audit
       Body:        supp-B piano-trio orbit as primary Z_2 object
       Projection:  orbit -> Z_2 symmetry-class shadow
       Observable:  table[m_3][stability][class] over 273 rows
       Result:      well-definedness PASS (all 273 land Z2_clean
                    after the two-pass gauge classifier)

v0.4b: mechanism-test audit (independent predictor)
       Body:        same primary Z_2 body
       Predictor:   non-circular gamma_3 family on Z_2 shadow features
       Observable:  per-m_3 stability distribution
       Result:      gamma_3 (tangent-isotypic) retired pre-sweep;
                    gamma_3'_orbit_pass2 (orbit-gauge-rigidity) FALSIFIED
                    by chi^2 = 1202 vs critical 26.22
```

## Audit Chain Methodology

The v0.4 chain inherited and extended the v0.3h discipline:
closure-relative gates, pre-registered constants, deterministic outcome
categories. Three stages:

1. **v0.4a two-pass gauge classifier**. Pass 1 over all 273 rows at
   default tolerances (`identity_rotation_tolerance = 1e-6`,
   `phase_grid = 73`). Pass 2 tight rerun on every row not landing in
   `Z2_clean` at Pass 1 (`1e-9`, `phase_grid = 361`). Four-band
   classifier: `Z2_clean / marginal_Z2 / smaller_symmetry / undefined`.
   No per-row knobs. The two-pass design encodes the O_434 anatomy
   probe's methodological lesson, which found that default tolerances
   misclassified one sentinel row by six orders of magnitude.

2. **v0.4b paper-side form lock**. Two registered baseline predictors,
   both with zero free parameters and pre-registered chi-squared(12)
   falsifiers:

   - `gamma_3` (tangent-isotypic): predict S iff `F_beta_even_dim >=
     F_beta_odd_dim` on `K_fib` (neutral-quotiented kernel).
   - `gamma_3'_orbit_pass2` (orbit gauge-rigidity): predict S iff the
     row required Pass 2 rescue in the v0.4a classifier.

3. **Sanity probes between registration and sweep**. A 7-row sample on
   existing v0.3 cross-m_3 receipts surfaced two findings that
   foreclosed the tangent path before any 2-hour sweep was committed:

   - **K_fib leakage**: F_beta does not preserve K_fib at tangent level
     on supp-B (leakage 0.10 -- 0.77 across all 7 rows). The gamma_3
     baseline's precondition fails uniformly. Retired pre-sweep with
     verdict `form_precondition_failed`.
   - **Anti-commutation cocycle**: typed `(R_i, phi_i) = (I, 0)` for all
     7 rows; the at-anchor `F_beta` already IS the typed reversing
     operator. No cocycle correction is available. Cocycle-rescue
     hypothesis (gamma_3_anticomm) ruled out.

The discipline successfully prevented two hours of compute being spent
against precondition-broken or rescue-foreclosed predictors. The probe
chain itself is a methodology contribution: cheap pre-sweep checks of
predictor preconditions are a load-bearing discipline that v0.4
implemented for the first time.

## Result And Structural-Negative Verdict

```text
v0.4a verdict:                    outcome_A_all_Z2_clean
  273 / 273 rows in Z2_clean after the two-pass classifier.
  marginal_Z2 = 0, smaller_symmetry = 0, undefined = 0.
  Pass 2 rescued 24 rows (8.8% of catalog) at the long-period
  high-m_3 regime; the two-pass design is validated.

v0.4b verdict (gamma_3 tangent baseline):  form_precondition_failed
  F_beta does not preserve K_fib at tangent precision on supp-B.
  7/7 sanity rows showed K_fib leakage 0.10 -- 0.77 (median 0.45).
  Projector decomposition is not an isotypic split; rule retired
  pre-sweep.

v0.4b' verdict (gamma_3'_orbit_pass2 baseline): falsified
  chi^2 = 1202.32 vs critical 26.22  (chi-squared(12), p = 0.01).
  Rule accuracy 63.74% is slightly below always-U baseline 64.47%.
  Gauge-rigidity is statistically independent of stability on supp-B.
```

The publishable v0.4 statement:

> v0.4 establishes the Li-Liao supplementary-B piano-trio catalog as a
> primary `Z_2`-symmetric body with a well-defined orbit-level Z_2
> domain map (v0.4a). Two non-circular Z_2-shadow stability predictors
> were registered before compute: gamma_3 on the tangent-isotypic
> projection, and gamma_3'_orbit_pass2 on the orbit-gauge-rigidity
> projection. The tangent predictor's precondition failed (F_beta does
> not preserve the neutral-quotiented kernel on supp-B). The orbit
> predictor was empirically falsified against the per-m_3 stability
> distribution at chi^2 = 1202 vs critical 26.22. **Stability
> information on this catalog is not carried by the Z_2 shadow** at
> either tested granularity. Any predictor with stability content must
> look beyond the Z_2 projection.

Three structural sub-results preserved in receipts and worth carrying
into v0.5:

1. **Orbit-tangent gap on supp-B**: F_beta is exactly an orbit-level
   symmetry of supp-B piano-trios but does not lift to a clean
   tangent-level symmetry of K_fib at this precision. The cocycle that
   would conceivably mediate the lift is identity for all tested rows;
   no transport-correction rescue is available.
2. **Long-period high-m_3 gauge stress**: 24 of 273 catalog rows
   require tight gauge minimization (`identity_rotation_tolerance = 1e-9`,
   `phase_grid = 361`) to land cleanly in `Z2_clean`. The pattern
   clusters at `m_3 in {1.4, 1.5, 1.6, 1.7}` and period `T >= ~170`.
   Methodology note carried into the v0.4a runner spec.
3. **Gauge-rigidity vs stability orthogonality**: empirically established
   via gamma_3'_orbit_pass2 chi^2 = 1202. These two orbit-level features
   are statistically independent.

## Reproducibility Surface

Primary scripts (workbench subcommands + one-shot aggregators):

```bash
npm run isotrophy:parse:b                         # supp-B parse, 273 rows

# v0.4a (operator-staged, ~85-120 min compute):
#   pass1: 15 commands, one per m_3, default tolerances
#   aggregator: identify Pass 2 candidates
#   pass2: per-flagged-row tight rerun
#   aggregator final: emit verdict + table[m_3][stability][class]
#   see kfacet_v04a_domain_map_preregistration.md for exact PowerShell.

# v0.4b (no compute, inline from v0.4a manifest):
python scripts/v04b_aggregator.py                # if implementing the
                                                  # row-z2-sweep path
# (Path retired pre-sweep; gamma_3'_orbit_pass2 was instead computed
# directly from the v0.4a manifest, no separate sweep.)
```

Key receipt directories:

```text
results/isotrophy/k-facet-v04a-domain-map/manifest.json
results/isotrophy/k-facet-v04a-domain-map/pass{1,2}/...

results/isotrophy/k-facet-v04b-gamma3prime-orbit-pass2/manifest.json

results/isotrophy/k-facet-v04a0-o434-anatomy/anatomy_receipt.json
  (the row anatomy that motivated the two-pass design)
```

Load-bearing constants live in the form-lock documents:
`kfacet_v04a_domain_map_preregistration.md` for the gauge thresholds
(`1e-6` / `1e-9` / `phase_grid 73` / `361`), classifier bands
(`Z2_clean: 1e-4`, `marginal_Z2: 1e-2`, `smaller_symmetry: 1`); and
`kfacet_v04b_gamma3prime_form.md` for the chi-squared falsifier
(`df = 12`, `critical = 26.22` at `p = 0.01`).

## Where v0.5 Opens

The v0.4 structural-negative is a falsifier-clean close: both registered
predictors fired and produced their pre-registered verdicts. The
catalog-level claim is "Z_2 shadow does not carry stability"; the
follow-on question is what does.

Three candidate v0.5 directions, all paper-side first:

1. **Richer-symmetry projection.** The piano-trio catalog's `Z_2`
   shadow is too coarse. What ADDITIONAL structural invariants could
   plausibly enrich it? Candidate: orbit-level conserved quantities
   `(E, |L|)` already verified tangent to v_bridge directions; their
   per-row joint distribution may stratify supp-B in a stability-relevant
   way without invoking new groups.

2. **Bifurcation-track projection.** Across the m_3 axis,
   supplementary-B has a continuous parameter family with discrete
   stability transitions visible in the per-m_3 S/U distribution
   (e.g., m_3 = 1.1 has 1S/22U, m_3 = 1.6 has 7S/1U). A v0.5 mechanism
   could track stability transitions as the body crosses `m_3`
   bifurcation thresholds, projecting the catalog onto a transition
   structure rather than an equivariant subspace.

3. **Closing without v0.5.** The v0.3 + v0.4 epilogues together
   establish a clean methodology surface (closure-relative discipline +
   two-pass classifier + pre-registration cocycle audit + structural
   sub-results) and two structural-negative results. If the next round
   targets a different substrate (mesa, geometry) rather than another
   K_facet refinement, the isotrophy program retires at end-of-v0.4
   with publishable methodology + two negative findings.

Codex direction will decide which path opens. No runner work proceeds
until the v0.5 mechanism family is registered paper-side.

**Update 2026-05-23:** v0.5 has opened with a **branch-shadow audit**
that LANDED with verdict `branch_hash_passes_audit`. The catalog-only
branch hash on `(m_3 < 1, z_0 < 0.3)` carries stability information on
supp-B: `chi^2 = 34.986` vs critical `11.34` (`chi-squared(3)`,
p ~= `1.23e-7`). The audit-dominant bucket is `(m_3 < 1, z_0 < 0.3)`
with 113 rows at 55.75% stable (catalog mean 35.5%); the other three
buckets sit at 20-29%. See `kfacet_v05a_branch_map_form.md`.

**v0.5b LANDED 2026-05-23 (predictor fails held-out).** The
leave-one-m_3-bin-out fold-trained branch-majority predictor was
tested against the always-U baseline on the 263-row gating subset
(12 bins with N >= 5). Verdict: **`branch_predictor_fails_heldout`**.
Model accuracy `0.6198` vs always-U `0.6388`; accuracy delta `-0.019`
(predictor LOSES to always-U). McNemar discordance `win=28, loss=33`,
`n_disc=61`, `p=1.0`. The load-bearing fold is m_3 = 0.4 (as
pre-mortem flagged): when held out, the low-mass/low-z_0 branch trains
as U on the residual five `m_3 < 1` bins (`28 S / 33 U`), so the
predictor cannot capture the m_3 = 0.4 stable cluster on its own
held-out fold. In other low-mass folds, the branch predicts S but
wins and false positives cancel.

**Joint v0.5 reading.** The catalog branch shadow is associated with
stability in-sample (`chi^2 = 34.99`, p ~= `1.23e-7`) but does not
predict held-out stability across mass bins (`accuracy_delta = -0.019`).
This is a clean **projection-limit result**: the branch hash describes
the supp-B distribution but does not lift to a mechanism. Combined
with v0.4: stability on this catalog is not in the Z_2 shadow at
either tangent or orbit-gauge-rigidity granularity AND is not
out-of-sample predictable from the catalog-coordinate branch shadow.
The branch hash is a descriptive catalog partition, not a predictive
mechanism. v0.5c remains open if a continuous feature (with explicit
`T_kepler` / `mu_eff` references) is to be promoted; otherwise the
v0.5 chapter closes as an audit-passes-predictor-fails projection
limit.

## Doc Trail

- `kfacet_v04a_domain_map_preregistration.md` -- two-pass classifier
  registration + verdict.
- `kfacet_v04a0_o434_anatomy.md` -- methodological pre-mortem (placed
  the two-pass design under the right precondition).
- `kfacet_v04b_gamma3_form.md` -- tangent-isotypic baseline, retired.
- `kfacet_v04b_gamma3prime_form.md` -- orbit-pass2 baseline, falsified.
- `kfacet_v04b_mechanism_preregistration.md` -- parent registration.
- `kfacet_v05a_branch_map_form.md` -- v0.5a branch-shadow audit (opens v0.5).
- `kfacet_v05b_branch_predictor_form.md` -- v0.5b held-out predictor.
- `kfacet_v05_writeup.md` -- v0.5 chapter close (projection-limit; companion).
- `kfacet_v03h_writeup.md` -- v0.3 chapter close (grand-companion).
- `kfacet_v03_freeze_b_comparison.md`,
  `kfacet_v03_gamma_crossm3_preregistration.md` -- the v0.3 alpha and
  II registrations that opened v0.4.

The chapter is closed cleanly. Receipts are durable. Pre-mortem
discipline saved ~4 hours of compute against broken preconditions.
