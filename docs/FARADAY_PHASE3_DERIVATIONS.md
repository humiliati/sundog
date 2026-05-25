# Shadow Faraday Phase 3 Derivation Receipt

Status: 2026-05-25, derivation computed and signed after proof-hygiene audit.

This file is the Phase 3 receipt for
[`SHADOW_FARADAY.md`](SHADOW_FARADAY.md). It does not change the Phase 1 sign
convention, the Phase 2 operator, the admissibility rule, or the branch
taxonomy.

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

### 1. Point-Limit Reduction

The registered point-limit operator is

```text
P_shadow^point[A]_{mu nu}(x)
  := lim_{epsilon -> 0+} epsilon^-2
       P_shadow^stencil[A]_{mu nu}(x, epsilon)
  = F_{mu nu}(x).
```

This recovers the Faraday tensor pointwise on differentiable `A` by definition
of the exterior derivative. No global `A` is reconstructed.

### 2. Homogeneous Identity

By the Phase 1 ledger, on smooth forms in the registered patch `U`:

```text
dF = d(dA) = 0.
```

In components this is the Bianchi identity:

```text
partial_alpha F_{beta gamma}
  + partial_beta F_{gamma alpha}
  + partial_gamma F_{alpha beta}
  = 0.
```

Equivalently, Stokes' theorem on any oriented three-volume `V subset U` gives
`int_V dF = oint_{partial V} F = 0`. The Phase 3 component derivation uses the
smooth case registered in the takeoff gate; lower regularity belongs to the
pre-registered regularity quarantine.

### 3. Faraday Component Extraction

Apply the registered sign convention:

```text
F_{0i} = -E_i
F_{ij} = epsilon_{ijk} B_k
```

Take the `(0, i, j)` component of `dF = 0`:

```text
partial_t F_{ij} + partial_i F_{j0} + partial_j F_{0i} = 0.
```

Using `F_{ij} = epsilon_{ijk} B_k`, `F_{j0} = E_j`, and `F_{0i} = -E_i`,
this is:

```text
epsilon_{ijk} partial_t B_k + partial_i E_j - partial_j E_i = 0,
```

which is exactly:

```text
curl E + partial_t B = 0.
```

This is extracted from `d(dA)=0`; Faraday's law is not invoked as a premise.

### 4. Integral Closure On Registered `(S, partial S)`

Choose any admissible oriented static surface-loop pair `(S, partial S)` inside
`U`. Integrating the component identity over `S` and applying Stokes' theorem
to the spatial curl term gives:

```text
oint_{partial S} E dot dl + d/dt int_S B dot dA = 0.
```

Because `P_shadow^point[A] = F`, the `E` and `B` in this identity are exactly
the registered point-limit shadow readouts inherited through the Phase 1 sign
convention. Therefore:

```text
R_F^0(S)
  = oint_{partial S} P_shadow^point(E) dot dl
    + d/dt int_S P_shadow^point(B) dot dA
  = 0.
```

All global `A` reconstruction terms cancel before this step, via `d(dA)=0`.

### 5. Finite-Stencil Locality Receipt

Let the normalized finite-stencil readout be:

```text
P_hat_shadow^epsilon[A]_{mu nu}
  = epsilon^-2 P_shadow^stencil[A]_{mu nu}.
```

For `C^2` fields:

```text
P_hat_shadow^epsilon[A]_{mu nu} = F_{mu nu} + O(epsilon),
```

equivalently the raw plaquette holonomy has per-plaquette error
`O(epsilon^3)`. A regular mesh of a fixed registered surface therefore has:

```text
R_F^epsilon(S) = O(epsilon),
lim_{epsilon -> 0+} R_F^epsilon(S) = R_F^0(S) = 0.
```

The fixed-`epsilon` term is a discretisation receipt, not a clean-domain
failure. No nonlocal data survives the registered limit.

### 6. Lorentz Invariant Receipt

The two invariants are:

```text
I_1 = F_{mu nu} F^{mu nu} = 2(|B|^2 - |E|^2)
I_2 = F_{mu nu} tilde F^{mu nu} = -4 E dot B
```

Since `P_shadow^point[A] = F` exactly in the registered limit, both invariants
computed from shadow data match the true invariants:

```text
I_1_from_shadow - I_1_from_F = 0
I_2_from_shadow - I_2_from_F = 0
```

Gauge invariance follows immediately: under `A -> A + d lambda`,
`F -> dA + d^2 lambda = F`, and the finite-stencil holonomy changes by
`oint_{partial omega} d lambda = 0` on every registered plaquette.

### 7. Branch Classification

All registered clean-domain residuals evaluate to exact structural zeros. The
derivation lands in **Branch A**.

## Closure Residual Table

| Receipt | Registered expression | Expected clean-domain value | Observed algebraic value | Branch impact |
| --- | --- | --- | --- | --- |
| Point-limit Faraday residual | `R_F^0(S)` | `0` | `0` | A |
| Finite-stencil residual | `R_F^epsilon(S)` | `O(epsilon)` or named truncation before limit; `0` after limit | `O(epsilon)` before limit; `0` after limit | A |
| Gauge invariance | `delta_lambda R_F^0(S)` | `0` | `0` | A |
| Lorentz invariant 1 | `I_1_from_shadow - I_1_from_F` | `0` | `0` | A |
| Lorentz invariant 2 | `I_2_from_shadow - I_2_from_F` | `0` | `0` | A |

## Branch Classification

**Branch A - clean structural zero.** Local shadow data suffices for Faraday
induction in the registered classical vacuum domain.

## Supporting Checks

The hand exterior-calculus derivation above is decisive. A tiny SymPy or Python
spot-check on a smooth source-free plane wave, or on any `C^2` compact-patch
test potential, may confirm signs and the finite-stencil scaling. No SymPy
notebook is required for the algebraic receipt itself.
