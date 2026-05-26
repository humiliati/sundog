# Shadow Faraday Phase 7 Spec: Source / Topology Boundary Audit

**Status**: Opened 2026-05-26; spec registered, execution not started.
**Parent ledger**: [`SHADOW_FARADAY.md`](SHADOW_FARADAY.md)
**Roadmap**: [`SUNDOG_V_FARADAY.md`](SUNDOG_V_FARADAY.md)
**Purpose**: Extend the closed Branch A receipt without laundering it. Phase 7
asks which nearby non-clean cases still close Faraday locally, and which ones
trip the pre-registered Branch B quarantine hooks by name.

## Phase 7 Thesis

Phase 3 proved a clean structural zero on the registered classical-vacuum,
contractible, magnetically-source-free domain. Phase 7 does **not** ask for a
new universal claim. It audits the first boundary around that result.

The load-bearing correction for this phase:

- Ordinary electric charge/current sources do **not** break Faraday closure.
  They enter `d * F = J`, not `dF = 0`.
- Magnetic charge/current, singular Dirac-string style defects, and
  non-contractible topology are the relevant Branch B cases for the Faraday
  Bianchi identity.
- Any Ampere-Maxwell / Gauss-law shadow story is a different extension and
  must not be silently folded into this phase.

## Registered Question

Given the Phase 2 two-tier plaquette-holonomy operator `P_shadow`, can the
closed Phase 3 result be sharpened into a boundary rule:

> magnetically clean, contractible patches preserve the Faraday residual zero;
> magnetic-source or topological obstructions trip named Branch B residuals.

## In Scope

1. **Electric-source, magnetically clean control.**
   A smooth classical EM patch with nonzero electric `J_mu` but `dF = 0`.
   Expected landing: Faraday residual remains `0`. This is a source-domain
   clarification for Faraday only, not a Maxwell-system claim.
2. **Magnetic-source quarantine.**
   A minimal smooth or regularized magnetic-source insertion with registered
   `dF = K_m != 0`. Expected landing: Branch B, named magnetic-source
   residual equal to the registered source flux through the audited 3-volume.
3. **Topological quarantine.**
   A solenoid / Aharonov-Bohm-style non-contractible loop. Expected landing:
   Branch B, named topology survivor. The residual is not allowed to masquerade
   as clean-domain evidence.

## Out Of Scope

- Deriving Ampere-Maxwell, Gauss's law, or the full Maxwell system from shadow
  primitives.
- Quantum EM, lattice QED, plasma/matter response, or curved-spacetime
  transport.
- Treating distributional singularities as clean smooth patches.
- Changing the Phase 2 operator, adding Floquet/twist rescue terms, or
  inventing new quarantine classes after calculation.

## Locked Inputs

Phase 7 inherits:

- Phase 1 sign convention: mostly-plus metric, `c = 1`, `F_{0i} = -E_i`,
  `F_{ij} = epsilon_{ijk} B_k`.
- Phase 2 operator: `P_shadow^stencil[A] = oint_partialomega A` and
  `P_shadow^point[A] = F_mu_nu`.
- Phase 3 success predicate and residual table shape.
- Phase 4 support style: tiny, explicit cases; no heavy simulation.
- Phase 5 limitation list and quarantine vocabulary.

## Forbidden Moves

Phase 7 must not:

1. Use sourced or topological cases to broaden the already-closed Phase 3
   Branch A claim.
2. Call an electric-current example a Faraday failure unless `dF != 0` is
   actually present.
3. Treat a gauge choice or global potential reconstruction as evidence.
4. Add an unregistered residual class after seeing the algebra.
5. Use a numerical spot-check as the proof.

## Work Order

1. Fill a three-row case table for the in-scope controls above.
2. For each case, state the domain condition before computing:
   `dF = 0`, `dF = K_m`, or non-contractible loop.
3. Compute the Faraday residual / holonomy survivor by hand first.
4. Add an optional tiny script only if it is a seconds-scale receipt that
   mirrors the hand calculation.
5. File the result in a future `docs/faraday/FARADAY_PHASE7_BOUNDARY.md` receipt.
6. Update `SHADOW_FARADAY.md`, `SUNDOG_V_FARADAY.md`, and `faraday.html` only
   after the Phase 7 receipt lands.

## Outcome Branches

| Branch | Meaning | Public posture |
| --- | --- | --- |
| A7-clean-control | Electric-source / magnetically clean case keeps `R_F = 0` | Faraday closure is source-tolerant only in the narrow `dF = 0` sense. No full-Maxwell claim. |
| B7-magnetic-source | Registered `dF = K_m != 0` produces named residual | Boundary sharpened; Branch A remains scoped. |
| B7-topology | Non-contractible loop leaves a named holonomy / phase survivor | Boundary sharpened; topology is quarantine, not failure. |
| C7-bounded-failure | Unregistered residual appears in a magnetically clean, contractible case | Re-open the relevant prior phase; public claim narrows. |

## Exit Criteria

Phase 7 exits only when:

- the three in-scope rows are filled;
- electric-source Faraday closure is clearly separated from full-Maxwell
  sourced dynamics;
- magnetic-source and topological survivors, if present, are named before
  public copy changes;
- any support script is reproducible and linked from the receipt; and
- the parent roadmap records the branch selected.
