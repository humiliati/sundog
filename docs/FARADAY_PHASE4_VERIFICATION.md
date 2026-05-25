# Shadow Faraday Phase 4 Verification Receipt

Status: 2026-05-25, verification and minimal falsification battery complete.

This file is the Phase 4 receipt for
[`SHADOW_FARADAY.md`](SHADOW_FARADAY.md). It supports the Branch A Phase 3
derivation in [`FARADAY_PHASE3_DERIVATIONS.md`](FARADAY_PHASE3_DERIVATIONS.md)
without changing the Phase 1 sign convention, the Phase 2 operator, the
admissibility rule, or the branch taxonomy.

## Support Command

```powershell
npm run faraday:phase4
```

Receipt artifacts:

- `scripts/faraday-phase4-battery.mjs`
- `results/faraday/phase4-battery/manifest.json`
- `results/faraday/phase4-battery/cases.csv`
- `results/faraday/phase4-battery/finite-stencil.csv`

Run summary:

```text
Faraday Phase 4 battery: 5/5 predicates passed
```

Thresholds were locked in the support script before interpretation:

```text
tolerance = 1e-9
residualFloor = 1e-3
```

Clean-domain checks pass only when residuals are at or below `tolerance`.
Falsifiers pass only when the expected residual or named quarantine is observed
at or above `residualFloor`.

## Verification And Falsification Table

| Case | Type | Observed | Expected | Disposition |
| --- | --- | --- | --- | --- |
| `constant_b_control` | Clean verification | `maxFaradayResidual=0; I1=12.5; I2=0` | Structural zero | Pass. Confirms Branch A on the trivial registered clean-domain control. |
| `source_free_plane_wave` | Clean verification | `maxFaradayResidual=0; I1=0; I2=0` | Structural zero | Pass. Confirms Branch A on a nontrivial registered clean-domain candidate. |
| `nonlocal_projection_falsifier` | Falsifier | `delta=0.23; maxFaradayResidual=0.787734891504` | Named residual | Pass. Violating locality produces a residual; this cannot be used as a clean-domain rescue. |
| `artificial_monopole_quarantine` | Falsifier | `dF_xyz=3` | Named monopole quarantine | Pass. The source insertion is detected outside the registered clean domain. |
| `gauge_after_projection` | Invariance check | `epsilon=0.08; absDelta=0` | Invariant | Pass. The plaquette holonomy is unchanged under `A -> A + d lambda`. |

## Clean-Domain Controls

### Uniform Constant `B`

The trivial control uses:

```text
E = (0, 0, 0)
B = (0, 0, B0)
```

Therefore:

```text
curl E + partial_t B = 0
I1 = 2 |B|^2 = 12.5
I2 = -4 E dot B = 0
```

This is a low-value mathematical case but a useful sign and invariant control.

### Source-Free Plane Wave

The nontrivial clean-domain check uses:

```text
A_x = (A / k) sin(k(z - t))
E_x = A cos(k(z - t))
B_y = A cos(k(z - t))
```

with `A = 1.2`, `k = 1.7`, sampled at:

```text
(t, x, y, z) = (0.37, -0.21, 0.13, 0.19)
```

The support script evaluates:

```text
curl E = (0, -k A sin(k(z - t)), 0)
partial_t B = (0,  k A sin(k(z - t)), 0)
```

so the Faraday residual is exactly zero at the sampled point, and the
invariants satisfy `I1 = 0`, `I2 = 0`.

## Finite-Stencil Locality Receipt

The support script also checks the Phase 3 finite-stencil scaling on the
plane-wave `x-z` plaquette:

```text
P_shadow^stencil[A]_{xz} = oint_{partial omega_xz} A
epsilon^-2 P_shadow^stencil[A]_{xz} -> F_xz
```

| epsilon | normalized holonomy | target `F_xz` | absolute error | error / epsilon |
| --- | --- | --- | --- | --- |
| `0.2` | `-1.183201134947` | `-1.144255419023` | `0.038945715924` | `0.194728579619` |
| `0.1` | `-1.16940513161` | `-1.144255419023` | `0.025149712587` | `0.251497125866` |
| `0.05` | `-1.158232384991` | `-1.144255419023` | `0.013976965968` | `0.279539319362` |
| `0.025` | `-1.151591620605` | `-1.144255419023` | `0.007336201582` | `0.293448063271` |

The absolute error decreases with `epsilon`, consistent with the registered
`O(epsilon)` normalized finite-stencil residual. The point-limit residual
remains the Phase 3 gate.

## Falsifier Receipts

### Nonlocal Projection

The nonlocal falsifier deliberately evaluates the electric curl at `z + delta`
while leaving the magnetic time derivative at `z`:

```text
delta = 0.23
maxFaradayResidual = 0.787734891504
```

This violates the Phase 2 locality rule by construction. Its residual is a
successful falsifier, not evidence against Branch A on the registered local
operator.

### Artificial Monopole

The monopole quarantine uses:

```text
B = (x, y, z)
div B = 3
```

Equivalently, the spatial Bianchi component is:

```text
dF_xyz = 3
```

This trips the pre-registered monopole quarantine. It is outside the
source-free clean domain and therefore lands in Branch B as a named exclusion,
not Branch C.

### Gauge After Projection

The gauge check uses:

```text
lambda = 0.31 t x - 0.17 y z + 0.07 x z
```

On the plane-wave `x-z` plaquette at `epsilon = 0.08`, the support script
finds:

```text
abs(oint A - oint (A + d lambda)) = 0
```

This supports the Phase 2 gauge audit in the same finite-stencil language used
by Phase 3.

## Closure Residual Update

| Receipt | Phase 3 value | Phase 4 support | Disposition |
| --- | --- | --- | --- |
| Point-limit Faraday residual | `R_F^0(S) = 0` | Constant `B` and plane wave both pass | Supported |
| Finite-stencil residual | `R_F^epsilon(S) = O(epsilon)`, limit `0` | Plane-wave holonomy error decreases with `epsilon` | Supported |
| Gauge invariance | `delta_lambda R_F^0(S) = 0` | Finite plaquette gauge delta `0` | Supported |
| Lorentz invariant 1 | `I1_from_shadow - I1_from_F = 0` | Constant `B` and plane wave reconstruct expected `I1` | Supported |
| Lorentz invariant 2 | `I2_from_shadow - I2_from_F = 0` | Constant `B` and plane wave reconstruct expected `I2` | Supported |
| Quarantine machinery | Named Branch B hooks only | Monopole insertion trips `dF_xyz=3` | Supported |
| Locality boundary | Nonlocal readout forbidden | Nonlocal projection residual `0.787734891504` | Supported |

## Disposition

Phase 4 exits **pass**. The verification/falsification battery is complete,
and Branch A remains supported on the registered classical vacuum domain.

This does not close the chapter. Phase 5 still owes the chapter-close note,
limitations/handoff summary, and final fidelity audit before `/faraday` can
move toward public-share readiness.
