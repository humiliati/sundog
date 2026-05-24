# v0.4b gamma_3 Functional Form Lock

Status: locked paper-side, 2026-05-22. Threshold-rule baseline signed off
by Codex.
Audience: v0.4b implementation; v0.4c reader of the verdict.
Companion: `kfacet_v04b_mechanism_preregistration.md` (the registration
this form lock binds to).

## The Locked Form

`gamma_3` is a **zero-parameter threshold rule** on Z_2 tangent-structure
invariants:

```text
gamma_3(features) =
  +1.0  (predict S)  iff  F_beta_even_dim >= F_beta_odd_dim
   0.0  (predict U)  otherwise
```

Where:
- `F_beta_even_dim` = dim of the `F_beta = +I` isotypic of the
  **neutral-quotiented kernel** `K_fib = ker(M_i - I) / N_C`, where
  `N_C` is the 8-dim Hamiltonian neutral block
  (`X_H(y_i(0))`, `u_E`, 3 translation directions, 3 momentum-boost
  Jordan partners) computed by `compute_neutral_basis`. The
  `K_fib_basis` itself is built by `compute_k_fib_basis` at the row's
  pre-registered closure floor (default `1e-7`; tight-rerun via the
  v0.4a two-pass classifier if needed).
- `F_beta_odd_dim` = dim of the `F_beta = -I` isotypic of the same
  `K_fib`.

Both dims are computed by:

```text
1. Build K_fib_basis (18 x K_fib_dim) for the row.
2. Restrict F_beta to K_fib:  F_beta_K = K_fib_basis^T @ F_beta @ K_fib_basis.
3. Form projectors:           P_+ = (I_K + F_beta_K)/2,  P_- = (I_K - F_beta_K)/2.
4. Count orthonormal columns of each projector above projector floor 1e-3.
```

**Implementation guardrail (signed off by Codex):** every row receipt
emitted by `kfacet-row-z2-sweep` MUST state explicitly that the
projection target is `K_fib` (the neutral-quotiented kernel), not raw
`ker(M_i - I)` and not the bridge-admitted kernel. A `projection_target`
field at the row-receipt top level carries the literal string `"K_fib"`.
Future re-registrations against raw kernel or bridge-admitted kernel
require an explicit, separately-named functional form.

## Interpretation

> *Orbits where the trivial Z_2 isotypic (`F_beta = +I`) dominates the
> kernel of `(M_i - I)` are predicted stable; orbits where the sign Z_2
> isotypic (`F_beta = -I`) dominates are predicted unstable.*

The threshold rule encodes a structurally-clean claim: that Z_2 trivial
dominance of the orbit's neutral block (after Hamiltonian quotient)
correlates with stability of the full Floquet spectrum. The Z_2
trivial sector carries `F_beta`-symmetric tangent directions; the
Z_2 sign sector carries `F_beta`-antisymmetric directions. The rule
predicts that symmetric-dominance is stabilizing.

## Non-Circularity Audit

Cross-checked against the v0.4b pre-registration's disallowed feature
list:

| Disallowed feature | In gamma_3? |
| :--- | :--- |
| "has off-unit-circle Floquet pair" | NO |
| spectral radius of `M_i` | NO |
| unit-circle count / number of eigenvalues with `|lambda| > 1` | NO |
| max `|lambda_i|` over Floquet spectrum | NO |
| Krein signature | NO |
| any function of `M_i` eigenvalues | NO -- gamma_3 uses kernel ISOTYPIC DIM, which is a count, not a spectral function |
| the stability label, or anything derivable from it | NO |

`F_beta_even_dim` and `F_beta_odd_dim` are computed from
`ker(M_i - I)` via SVD of `(M_i - I)` (a unit-circle-band cluster
identification) followed by projection onto `F_beta` isotypics. The
SVD identifies the kernel cluster; it does not inspect off-unit-circle
behavior. The isotypic projection is a geometric `F_beta`-decomposition
of a tangent subspace, independent of how the rest of the Floquet
spectrum behaves.

In particular: `(F_beta_even_dim, F_beta_odd_dim)` is a **tangent
geometric invariant** of the orbit's neutral block, not a stability
signature. The threshold rule maps one geometric invariant to a
binary prediction.

## Free Parameter Count

```text
Number of free parameters fitted to the data: 0
```

The threshold rule has no tunable constants. The comparison
`F_beta_even_dim >= F_beta_odd_dim` is exact and discrete. No bin
boundaries, no slope coefficients, no regression weights.

## Chi-Squared Degrees of Freedom

```text
df = (number of gating bins with N >= 5) - (number of free parameters)
   = 12 - 0 = 12
```

Reference distribution for the falsifier is `chi-squared(12)`. Critical
value at `p = 0.01` is `26.22`. Test statistic `chi^2 > 26.22` -->
**falsifies this baseline rule**.

Wording note (signed off by Codex): the failure condition is recorded as
"falsifies this baseline threshold rule", **not** "falsifies gamma_3
globally". This preserves the possibility of a separately-registered
gamma_3' (e.g., a ratio rule or non-trivial threshold) following a
clean diagnostic-residual readout, without sounding like a retreat
from the gamma_3 family.

## Bin-Level Predicted Stable Fraction

For each m_3 bin, `gamma_3` emits

```text
P_S_predicted(m_3) = (count of rows in bin where F_beta_even_dim >= F_beta_odd_dim)
                     / N(m_3)
```

This is a deterministic function of the per-row kernel structure; no
additional fitting.

## Edge Cases (Pre-Registered)

1. **Ties (`F_beta_even_dim == F_beta_odd_dim`):** count as S (predicted
   stable). The `>=` direction is part of the rule lock; do NOT switch
   to `>` after seeing the data.
2. **Zero kernel dimensions on both sides (`F_beta_even_dim ==
   F_beta_odd_dim == 0`):** would imply ker(M_i - I) is empty modulo
   the neutral quotient. The rule still emits S (per #1). Flag the row
   in the diagnostic readout but do not exclude.
3. **Row whose two-pass v0.4a verdict was Pass 2 rescue:** use the
   tight-tolerance F_beta projection. The v0.4a manifest already
   records Pass 2 reclassification provenance; v0.4b reuses the same
   kernel basis convention.

## What the Threshold Rule Is Designed to Catch

The rule's binary structure means it can only detect **first-order**
correlation between Z_2 isotypic dominance and stability. If the
underlying mechanism is:

- **Z_2-trivial-dominance maps to S, Z_2-sign-dominance maps to U**:
  expected pass with chi-squared substantially below the critical
  value.
- **No correlation between isotypic dominance and stability** (the
  Z_2 shadow doesn't carry stability info): expected fail, or
  retired_no_variation if the dims are nearly-constant.
- **Inverse correlation** (trivial-dominance maps to U): also fail,
  with the diagnostic per-bin residuals showing systematic-but-
  inverted sign.
- **Some finer-grained mechanism** (e.g., even/odd ratio matters, or
  bridge-band count is the predictor): fail of gamma_3, but the
  diagnostic residuals can suggest gamma_3' for a follow-on
  registration.

## What the Threshold Rule Is NOT Designed to Catch

- Continuous variation in stability fraction inside a single isotypic
  ratio.
- Interactions with z_0 or period at finer-than-m_3 resolution.
- Direction-of-instability information (carried by the gamma_1
  sidecar, not by gamma_3).

If gamma_3 passes, the v0.4 mechanism is a real but coarse correlation
between Z_2 shadow geometry and stability. If gamma_3 fails, a
refined gamma_3' can use ratios or thresholds at non-trivial values
(e.g., predict S iff `F_beta_even_dim / kernel_dim > 0.4`); the
threshold-rule baseline establishes the cleanest non-circular floor.

## Lock-In Statement

This functional form is committed before any per-row `(F_beta_even_dim,
F_beta_odd_dim)` pair has been computed for the 273 supplementary-B
rows. Any change to the form after the per-row receipts exist is a
re-registration, not a refinement of the locked rule.

Implementation may proceed against this form lock.
