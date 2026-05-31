# v0.13a liao2021 Adapter + Leakage Preflight Form Draft

Status: **DRAFTED 2026-05-31; pending operator lock review.** No adapter, parser,
or preflight runner has been written; no liao2021 row has been integrated; no
`velocity_fraction` has been computed. This document locks the two contracts that
must pass BEFORE v0.13 may profile or rate-probe the Li/Liao 2021 non-hierarchical
table: (1) an **expansion-only D5 adapter** and (2) a **cross-ansatz leakage bound**.
Tolerances marked `[PROPOSED]` are the only operator decisions; everything else is
inherited from v0.7/v0.12/v0.13.

## Frame

v0.13's signal-blind discovery returned an **independent-target landscape negative**:
every genuinely independent (Tier 3) external catalog fails a hard gate (equal-mass
-> conditioning-strata; restricted substrate; or too small), and the only
schema-viable path is **Tier 2 — Li/Liao 2021 non-hierarchical unequal-mass**
(135,445 rows, source S/U linear-stability labels, full Newtonian planar). Snapshot:
`docs/isotrophy/kfacet/kfacet_v13_target_inventory_snapshot.md`.

That table is attractive but carries two integrity hazards a naive run would miss:

```text
H1  it is NOT the supp-A/B mirrored ansatz, so the frozen D5 expand_initial_state
    does not apply -- a new source-row -> 18D-state map is required.
H2  cross-ansatz physical duplicates between liao2021 and supp-A/B would NOT match
    by raw IC delta (different coordinates), so the v0.13 IC-overlap quarantine
    could silently miss real leakage.
```

v0.13a is a **preflight**: it does not test the velocity-fraction signal and does not
lock a target. It only decides whether liao2021 is admissible for v0.13 profile +
rate-probe at all. Cheapest gate first (the v0.12 lesson): no integration sweep until
the adapter is proven measurement-frozen and the leakage is proven bounded.

## Integrity Caveat

v0.13a stays inside the v0.13 firewall:

```text
forbidden in v0.13a:
  computing or recording velocity_fraction, zone_index, or any S/U-vs-feature stat
  tabulating adapter / leakage results by stability label
  relaxing any inherited D5 numerical control
  changing the v0.11 cutpoints

allowed in v0.13a:
  a new source-row -> 18D-state expansion (the adapter; H1)
  E, |L|, T, and mass-tuple computation used ONLY as overlap/identity invariants
    (H2) -- these are catalog-coordinate invariants, not the vf stability feature
    (same allowance v0.6 used for (E,|L|) and the v0.13 form used for the weaker
    invariant identity key); they never enter a stability statistic
```

The preflight computes NO `velocity_fraction`. (It need not: see the parity check,
which is monodromy/gate-only.)

## Locked Contract 1 -- Expansion-Only Adapter

**What is new (the only new physics-facing code):**

```text
parse_liao2021(path)            non-mirrored planar parser for the 2021 table
expand_liao2021_state(row)      source row -> (masses, x[3,3], v[3,3]) for the
                                liao2021 planar ansatz:
                                  r1=(x1,0,0)  r2=(1,0,0)  r3=(0,0,0)
                                  v1=(0,v1,0)  v2=(0,v2,0)
                                  v3=(0,-(m1 v1 + m2 v2)/m3, 0)
                                  masses=(m1,m2,m3)
                                CoM-centered identically to expand_initial_state
                                (center_com=True), z=vz=0 (planar).
```

**What is FROZEN, inherited byte-for-byte (imported, never redefined) from
`scripts.v07a_velocity_fraction_audit` / v0.12:**

```text
the DOP853 solve_ivp orbit + variational path  (integrate_orbit solver body)
compute_monodromy_vectorized                    (324-dim variational monodromy)
select_gamma_1                                  (largest-real-part + tie-break cascade)
velocity_fraction_and_z_fraction                (vf definition; CoM + mass-weighted norm)
symplecticity_residual, reciprocal_pair_residual
RTOL = ATOL = 1e-12,  MAX_STEP_FRACTION = 0.02
SYMPLECTICITY_GATE = RECIPROCAL_PAIR_GATE = 1e-4
```

The adapter changes ONLY the IC expansion. The integrator, monodromy, gamma
selection, vf definition, and gates are the same objects v0.12 ran on supp-A.

**Parity verification (pre-registered; firewall-clean, no vf recorded):**

```text
P1  code-inheritance identity assert: the adapter module's references to the seven
    frozen symbols above are `is`-identical to the v07a objects (no shadowing,
    no re-implementation). Fails -> ABORT.
P2  frame-invariance: for K_PARITY = 6 liao2021 probe rows (deterministic uniform,
    seed 20260523), apply each pre-registered isometry to the initial configuration
    and run the full pipeline. Assert the monodromy eigenvalue multiset, the
    symplecticity residual, and the reciprocal-pair residual are invariant:
      rotations  SO(2) by {0, 37, 90, 211} degrees in the orbit plane
      translation by a fixed offset (absorbed by CoM-centering)
    tolerance  tol_parity = 1e-8 relative   [PROPOSED]
    (vf is a rotation/translation-invariant scalar by construction once P1 holds,
     so it is NOT computed here -- the preflight stays vf-free.)
```

P1 + P2 together certify the adapter is expansion-only: only the coordinates fed
into the frozen pipeline changed, not the measurement.

## Locked Contract 2 -- Cross-Ansatz Leakage Bound

Raw IC matching is insufficient (H2). Leakage is bounded by a **canonical-invariant**
overlap against supp-A/B, dominated by a mass-tuple disjointness argument.

**Canonical normalization (fixes all three scaling freedoms so invariants are
comparable across catalogs):** rescale every orbit (liao2021 and supp-A/B) to

```text
G = 1,   total mass  sum(m_i) = 1,   moment of inertia  I0 = sum m_i |r_i - R_com|^2 = 1
```

(G=1 + sum m = 1 fixes the mass scale; I0 = 1 fixes the length scale; both together
fix the time scale.) In these canonical units E, |L|, and T are pure numbers.

**Identity key (canonical units):**

```text
(sorted mass-ratio triple,  E*,  |L|*,  T*)
```

**Dominant bound -- mass-tuple disjointness.** supp-A/B occupy ONLY the two-equal-mass
family (1,1,m3). A liao2021 row can be a supp-A/B duplicate only if its canonical
mass-ratio triple has two equal entries whose third matches a supp-A/B m3 within
`tol_mass`. The 2021 table is non-hierarchical UNEQUAL-mass, so this equal-pair slice
is expected to be near-empty; the leakage bound is dominated by its size.

**Backstop -- invariant match on the slice.** For each liao2021 row in the equal-pair
slice, match against supp-A/B rows of the same mass-tuple on (E*, |L|*, T*) within
tolerances. A liao2021 row is a leak iff mass-tuple AND all three invariants match.

```text
tol_mass = 1e-6  (relative, mass-equality + m3-grid match)   [PROPOSED]
tol_E    = 1e-4  (relative, canonical energy)                [PROPOSED]
tol_L    = 1e-4  (relative, canonical |L|)                   [PROPOSED]
tol_T    = 1e-4  (relative, canonical period)               [PROPOSED]

leakage_fraction = (#liao2021 rows that leak) / (#liao2021 rows after eligibility)
leakage gate     = leakage_fraction <= 0.05   (inherited v0.13 overlap gate)
```

Reflection-image leakage `(z0,vx,vy,vz)->(-z0,vx,vy,-vz)` is moot here (liao2021 is
planar, z0=vz=0) but is still reported as `reflection_leak = 0` for the receipt.

**Bound-quality guard (the honest escape hatch).** If the canonical normalization
cannot be computed for either catalog (missing field, parser ambiguity, or the unit
conventions cannot be reconciled to the registered canonical frame), leakage is
declared **unbounded**, and liao2021 is report-only regardless of how attractive the
schema is. A wide leakage estimate is not a pass.

## Preflight Verdict Tree

```text
if adapter parity P1 or P2 fails:
    verdict = adapter_not_expansion_only           -> ABORT (not the frozen measurement)
elif leakage is unbounded (canonical reconciliation fails):
    verdict = leakage_unbounded_report_only        -> liao2021 report-only
elif leakage_fraction > 0.05:
    verdict = leakage_blocked_report_only          -> liao2021 report-only
else:
    verdict = preflight_passed_probe_authorized    -> v0.13 may profile + run a
                                                       12-row rate probe on liao2021
                                                       (full 100-row probe still
                                                       gated by --authorize-full-probe)
```

A report-only outcome closes v0.13 as **`no_viable_external_target_found`** with the
field result preserved: an independent D5-tractable external target does not exist
in the surveyed landscape, and the sole schema-viable Tier-2 candidate is
leakage-blocked. That is a legitimate, publishable negative, not a failure of the
velocity-fraction hypothesis.

## Frozen Inputs

```text
liao2021 table   docs/isotrophy/external_targets/_staging/liao2021_nonhierarchical.txt
                 fetched from the snapshot's raw URL; download SHA-256 recorded;
                 promoted from _staging only after parse + this preflight
overlap sources  docs/isotrophy/supplementary-A_periodic-3d_mirror.txt
                 docs/isotrophy/supplementary-B_piano-init-condit-3d.txt
inherited D5     scripts/v07a_velocity_fraction_audit.py  (the seven frozen symbols)
inventory        results/isotrophy/k-facet-v13-external-target-search/target_inventory.csv
                 (snapshot: kfacet_v13_target_inventory_snapshot.md)
```

No relaxation of the inherited D5 controls is authorized. The preflight does not run
the full 135,445-row sweep; parity uses 6 rows, leakage uses invariants only (no
integration beyond the 6 parity rows).

## Output Contract

```text
results/isotrophy/k-facet-v13a-liao2021-preflight/
  manifest.json
    - schema = "sundog.isotrophy.v0.13a-liao2021-preflight.v1"
    - download_sha256, row_count, mass_variation_summary
    - adapter_parity = { code_inheritance_ok, frame_invariance_max_residual, passed }
    - leakage = { canonical_norm_ok, equal_pair_slice_count, leaked_rows,
                  leakage_fraction, reflection_leak, bounded, tolerances }
    - verdict
  parity_rows.csv          (6 rows: monodromy eigenvalue stats + gate residuals per isometry; NO vf)
  leakage_audit.csv        (equal-pair-slice rows + invariant match results)
```

No receipt in this chapter contains `velocity_fraction`, `zone_index`, or a per-row
stability-conditioned feasibility field.

## Claim Boundary

```text
preflight pass  = permission for v0.13 to profile + rate-probe liao2021. It does NOT
                  test, confirm, or estimate the velocity-fraction signal; it does NOT
                  lock a transfer; it does NOT upgrade independence beyond Tier 2.
preflight fail  = liao2021 report-only; v0.13 closes no_viable_external_target_found,
                  preserving the independent-target landscape negative.
in all cases    = no v0.10b / v0.11 revision, no v0.3h K_facet revision, not
                  theorem-facing, and the supp-B conditional positive is unchanged.
```

Even a fully successful downstream liao2021 transfer would be a **Tier-2,
same-Li/Liao-lineage** external result (different paper / orbit construction, same
author family) -- "near-the-source external," to be stated honestly, not an
arm's-length independent confirmation.

## Lock-In Statement

This form is committed before any adapter, parser, or preflight code is written.
After lock, implementation may add only `parse_liao2021` + `expand_liao2021_state` +
the preflight runner, importing the seven frozen D5 symbols unchanged. Any change to
the adapter's frozen-symbol set, the canonical normalization, the identity key, the
tolerances, the leakage gate, or the verdict tree after runner execution is a
re-registration, not a refinement.
