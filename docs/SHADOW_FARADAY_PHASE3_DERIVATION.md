# Shadow Faraday Phase 3 Derivation Receipt

Status: 2026-05-25, takeoff cleared; derivation not yet computed.

This file is the Phase 3 receipt for
[`SHADOW_FARADAY.md`](SHADOW_FARADAY.md). It must not change the Phase 1 sign
convention, the Phase 2 operator, the admissibility rule, or the branch
taxonomy. If any of those need to change, stop and re-open the relevant phase
in the ledger before continuing here.

## Locked Inputs

Use only the Phase 3 Takeoff Gate in
[`SHADOW_FARADAY.md`](SHADOW_FARADAY.md#phase-3-takeoff-gate):

- `A_mu = (-Phi, A_x, A_y, A_z)`
- `F_{mu nu} = partial_mu A_nu - partial_nu A_mu`
- `F_{0i} = -E_i`
- `F_{ij} = epsilon_{ijk} B_k`
- smooth, source-free, contractible, static-surface domain
- `P_shadow^point[A] = F`
- `P_shadow^stencil[A] = oint_{partial omega} A`

Forbidden shortcuts:

- Do not invoke Faraday's law as a premise.
- Do not invoke the full Maxwell system as a premise.
- Do not use a global reconstruction of `A`.
- Do not add a new quarantine class after observing the algebra.

## Work Order

1. Point-limit reduction.
2. Homogeneous identity `dF = d(dA) = 0`.
3. Faraday component extraction.
4. Integral closure on registered `(S, partial S)`.
5. Finite-stencil locality receipt.
6. Lorentz invariant receipt.
7. Branch classification.

## Derivation

Pending.

## Closure Residual Table

| Receipt | Registered expression | Expected clean-domain value | Observed algebraic value | Branch impact |
| --- | --- | --- | --- | --- |
| Point-limit Faraday residual | `R_F^0(S)` | `0` | pending | pending |
| Finite-stencil residual | `R_F^epsilon(S)` | `O(epsilon)` or named truncation before limit; `0` after limit | pending | pending |
| Gauge invariance | `delta_lambda R_F^0(S)` | `0` | pending | pending |
| Lorentz invariant 1 | `I_1_from_shadow - I_1_from_F` | `0` | pending | pending |
| Lorentz invariant 2 | `I_2_from_shadow - I_2_from_F` | `0` | pending | pending |

## Branch Classification

Pending. Must be exactly one:

- Branch A - clean structural zero
- Branch B - named quarantine
- Branch C - bounded failure

## Supporting Checks

Optional only. Hand/exterior-calculus derivation decides the branch. SymPy or a
tiny Python check may verify signs and examples after the derivation is written.
