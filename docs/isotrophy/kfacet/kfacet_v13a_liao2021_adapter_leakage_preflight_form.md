# v0.13a liao2021 Adapter + Leakage Preflight Form Lock

Status: **OPERATOR LOCK 2026-05-31; AMENDED R1 2026-05-31** (P2 re-registered from
monodromy-eigenvalue-multiset invariance to vf-invariance after the original invariant
was found numerically unachievable -- see Amendment R1). This document locks the two
contracts that must pass BEFORE v0.13 may profile or rate-probe the Li/Liao 2021
non-hierarchical table: (1) an **expansion-only D5 adapter** and (2) a **cross-ansatz
leakage bound**.

Reviewed for self-consistency, non-circularity, and integrity:

- **Expansion-only boundary is locked.** The only new source-specific code is the
  row parser, the row-to-state expansion, and a thin explicit-state integration
  wrapper whose sole job is to feed that state to the inherited Newtonian RHS and
  variational machinery. No equation, solver, monodromy, gamma, vf definition, or
  gate logic may be reimplemented.
- **Parity check is vf-free.** The frame-invariance check compares only monodromy
  eigenvalue multisets and gate residuals. It must not call
  `velocity_fraction_and_z_fraction`, and no `velocity_fraction` / `zone_index`
  field may appear in the preflight receipt.
- **Canonical leakage comparison is implementable.** The scale normalization now
  gives explicit formulas for mass-normalized length, time, energy, angular
  momentum, and period. Near-zero comparisons use a symmetric relative tolerance
  denominator, so `|L| ~= 0` cannot make the leakage test numerically undefined.
- **All proposed tolerances are locked as written.** `tol_parity = 1e-8`,
  `tol_mass = 1e-6`, and `tol_E = tol_L = tol_T = 1e-4`; the inherited leakage
  gate remains `0.05`.

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
  recording velocity_fraction or zone_index in any receipt, or any S/U-vs-feature stat
  tabulating adapter / leakage results by stability label
  relaxing any inherited D5 numerical control
  changing the v0.11 cutpoints

allowed in v0.13a:
  a new source-row -> 18D-state expansion (the adapter; H1)
  E, |L|, T, and mass-tuple computation used ONLY as overlap/identity invariants
    (H2) -- these are catalog-coordinate invariants, not the vf stability feature
    (same allowance v0.6 used for (E,|L|) and the v0.13 form used for the weaker
    invariant identity key); they never enter a stability statistic
  computing velocity_fraction SOLELY as a discarded frame-invariance assertion inside
    the P2 parity check (Amendment R1) -- the value is never recorded in a receipt and
    never associated with a stability label; only the invariance residual |dvf| is kept
```

Under Amendment R1 the preflight computes `velocity_fraction` ONLY inside the P2
frame-invariance assertion (the original eigenvalue-multiset invariant was found
numerically unachievable -- see Amendment R1). The vf values are discarded; only the
invariance residual is kept. No `velocity_fraction` field appears in any receipt, and
nothing in v0.13a is conditioned on a stability label.

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
compute_monodromy_vectorized                    (324-dim variational monodromy)
select_gamma_1                                  (largest-real-part + tie-break cascade)
velocity_fraction_and_z_fraction                (vf definition; CoM + mass-weighted norm)
symplecticity_residual, reciprocal_pair_residual
RTOL = ATOL = 1e-12,  MAX_STEP_FRACTION = 0.02
SYMPLECTICITY_GATE = RECIPROCAL_PAIR_GATE = 1e-4
```

The adapter changes ONLY the IC expansion. Because the v0.7 `integrate_orbit(row)`
function calls the mirrored-ansatz `expand_initial_state(row)`, liao2021 may use a
thin explicit-state wrapper. That wrapper may only:

```text
accept (masses, x0, v0, period) from expand_liao2021_state
CoM-center exactly as registered
call the same Newtonian rhs_factory / DOP853 solve_ivp path
emit an IntegratedOrbit-compatible object for compute_monodromy_vectorized
```

It may not modify the equations, method, tolerances, max-step convention,
variational monodromy, gamma selection, vf definition, or gates.

**Parity verification (pre-registered; no vf recorded; P2 = vf-invariance per R1):**

```text
P1  code-inheritance identity assert: the adapter module's references to the frozen
    D5 symbols above are `is`-identical to the v07a objects (no shadowing, no
    re-implementation). The explicit-state wrapper is separately audited by P2.
    Fails -> ABORT.
P2  frame-invariance via vf-invariance (Amendment R1): for K_PARITY = 6 liao2021 probe
    rows (deterministic uniform, seed 20260523), apply each pre-registered isometry to
    the initial configuration, run the full integration + monodromy + gamma-selection +
    vf path, and assert velocity_fraction is invariant:
      rotations  SO(2) by {0, 37, 90, 211} degrees in the orbit plane
      translation by a fixed offset (absorbed by CoM-centering)
    tolerance  tol_parity = 1e-8 relative (symmetric near-zero comparator)
    The vf values are a DISCARDED code-correctness assertion -- never recorded, never
    stability-associated; only max|dvf| over the isometries is kept. vf is the correct
    invariant: it IS the measurement (so its frame-invariance directly proves
    expansion-only), and it is well-conditioned -- a bounded ratio of norms of the
    DOMINANT (best-conditioned) Floquet eigenvector -- unlike the monodromy eigenvalue
    multiset, whose tiny reciprocal eigenvalues are numerical garbage for unstable
    orbits (see Amendment R1). Integration-feasibility failure of an isometry on an
    extremely unstable orbit is REPORTED but does not by itself fail P2 (it is a
    feasibility matter the later rate-probe measures); vf-invariance is asserted over
    the isometries that integrate, requiring at least the base plus one isometry per row.
```

P1 + P2 together certify the adapter is expansion-only: only the coordinates fed
into the frozen pipeline changed, and the measurement (vf) is unchanged.

## Amendment R1 (2026-05-31) -- P2 re-registered to vf-invariance

The original P2 asserted invariance of the **monodromy eigenvalue multiset** (plus
gate residuals) at `tol_parity = 1e-8`. Implementation on the staged table found this
invariant **physically unachievable** for three-body monodromy: the residual scales
with the eigenvalue condition number (cond 1.0 stable orbit -> 5.8e-3; cond 7600 ->
~1.0), because an unstable orbit's reciprocal Floquet eigenvalue (1/lambda, |lambda|
large) is computed as numerical noise by `np.linalg.eigvals`, and even the degenerate
unit-eigenvalue block of a stable orbit cannot be multiset-matched to 1e-8. This is a
property of the FROZEN `compute_monodromy_vectorized`, not of the adapter (P1 passes;
the residual-vs-conditioning correlation proves the rotation is a genuine symmetry
defeated only by float precision). P2 is therefore re-registered to assert invariance
of `velocity_fraction` -- the measurement itself, which is well-conditioned (it depends
only on the dominant Floquet eigenvector) and whose invariance IS the expansion-only
claim. `tol_parity = 1e-8` is retained; vf is computed only as a discarded assertion.
P1, Contract 2 (leakage), all tolerances, the verdict tree, and the claim boundary are
unchanged. Evidence: `results/isotrophy/k-facet-v13a-liao2021-preflight/parity_rows.csv`
(pre-amendment eigenvalue-multiset residuals + |lambda| condition numbers).

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

Implementation formulas:

```text
M        = sum_i m_i
mu_i     = m_i / M
R_com    = sum_i mu_i r_i
I_mu     = sum_i mu_i |r_i - R_com|^2
ell      = sqrt(I_mu)
tau      = sqrt(ell^3 / M)                 # original time units per canonical time

mass*    = sort(mu_i)
E*       = E * ell / M^2
|L|*     = |L| / (M^(3/2) * sqrt(ell))
T*       = T / tau = T * sqrt(M / ell^3)
```

If `M <= 0`, `I_mu <= 0`, a period is missing, or any invariant is non-finite, the
canonical reconciliation fails and leakage is declared unbounded.

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
tol_mass = 1e-6  (relative, mass-equality + m3-grid match)
tol_E    = 1e-4  (relative, canonical energy)
tol_L    = 1e-4  (relative, canonical |L|)
tol_T    = 1e-4  (relative, canonical period)

relative comparator:
  abs(a - b) <= tol * max(1, abs(a), abs(b))

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
stability-conditioned feasibility field. `parity_rows.csv` may identify row indices,
mass keys, periods, isometry names, eigenvalue summary residuals, and gate residuals;
it may not include source stability labels.

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
