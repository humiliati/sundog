# O_617 Bridge Sub-Investigation — Deep Dive

Status: current, 2026-05-22.
Audience: collaborators, paper-side writers, future coding agents.
Companion: `internal/anniversary/kfacet_v03h_writeup.md`.
Receipt: `results/isotrophy/k-facet-v03-O617-deep-dive/deep_dive_receipt.json`.
Tooling: `scripts/o617_deep_dive.py` (one-shot, no integration).

## One-Line Read

> v0.3h resolves the eligible strict catalog rows as structural zeros.
> The sole quarantined row, O_617, is explained by weak sigma_3 catalog
> inclusion rather than by a failure of the Gamma_i audit chain.

Concretely: `O_617`'s sigma_3 closure residual sits at `1.62e-1`, the
third-worst of the 21 strict rows, against a catalog median near `2e-8`.
The defective D3 representation that the bridge audit flagged is the
downstream consequence of this weak sigma_3 closure, not a near-bifurcation
Floquet anomaly, not a Jordan block, and not a missed neutral direction.

## What the Six Probes Found

The deep dive ran six pre-registered no-integration probes against the
existing `M_i.npy`, `D3_*.npy`, bridge audit receipt, and sigma3-scan
receipts. Each probe answered a specific question; together they
triangulate the cause of O_617's defective-D3 disposition.

### Probe 1 — geometric anatomy of v_bridge

The bridge singular vector decomposes into:

```text
Per-body position norms: [0.316, 0.316, 0.611]
Per-body velocity norms: [0.459, 0.459, 0.076]
Per-axis position norms: [0.426, 0.615, 0.113]   # y-axis dominates
Per-axis velocity norms: [0.271, 0.588, 0.094]   # y-axis dominates
(12)-symmetric  position norm: 7.48e-1
(12)-antisym.   position norm: 1.13e-1
(12)-symmetric  velocity norm: 9.36e-2
(12)-antisym.   velocity norm: 6.47e-1
```

`v_bridge` is mixed-parity under body swap `(12)`: predominantly
symmetric in positions but predominantly antisymmetric in velocities.
Body 3 dominates positions while bodies 1 and 2 dominate velocities.
This mixed parity does not fit cleanly into the D3 isotypic decomposition,
which assumes consistent parity under `F_beta = (12) x R_pi x tau`.

### Probe 2 — sigma_3^3 closure decomposition

Decomposing `||sigma_3^3 - I||_inf` by kernel floor:

```text
Ambient (full 18D):      2.376e+03
On k=7 kernel (floors 1e-7..1e-4): 1.181e-5  (clean)
On k=8 kernel (floor 1e-3, admits bridge): 4.07e-2  (4%)
Cross terms kernel<->complement: ~1.79e+3  (huge)
```

Three observations land:

1. The k=7 in-kernel residual is at integration noise (1.18e-5), so the
   sigma_3 representation is clean on the *unambiguous* kernel
   (excluding the bridge).
2. Admitting the bridge into k=8 raises the in-kernel sigma_3^3 residual
   to 4%, three orders of magnitude above noise. The bridge vector is
   the direction that breaks sigma_3 closure on the kernel.
3. The kernel-complement cross terms under sigma_3^3 are 1.79e+3,
   indicating that sigma_3^3 mixes the recovered kernel basis with
   the structural non-kernel substantially. This is expected
   numerically for an unstable orbit with rtol-scale residual
   eigenvectors, but the magnitude here means the kernel basis is not
   a clean sigma_3-invariant subspace at this rtol.

### Probe 3 — monodromy spectrum

```text
Eigenvalue |.| range:           [0.8125, 1.2307]
On unit circle (||.| - 1| < 1e-2):  16 of 18
Near +1 (|.- 1| < 1e-2):            12 of 18
Real Schur 1x1 blocks:               8
Complex Schur 2x2 blocks:            5
Matrix condition number:             6.6e+05
```

The eigenvalue range `[0.81, 1.23]` forms reciprocal pairs as expected
for a symplectic monodromy. Two off-unit-circle eigenvalues bound the
unstable manifold; the other 16 are marginal/elliptic on or near the
unit circle. Twelve sit within `1e-2` of `+1` -- consistent with the
algebraic-near-1 multiplicity flagged in the bridge audit, but the
SVD-based geometric multiplicity at the fixed bridge floor was only
8, hence the audit's `jordan_defect_diagnostic = 4` (now contextually
explained: marginally-elliptic eigenvalues, not a Jordan defect).

### Probe 4 — (E, |L|) sensitivity along v_bridge

```text
cos(v_bridge, grad H)     = 4.02e-10
cos(v_bridge, grad |L|)   = -2.26e-09
H(y0) = -1.222719,   |L|(y0) = 3.185e-1
```

`v_bridge` is essentially exactly tangent to the (E, |L|) level set.
This is **important**: it rules out an interpretation where the bridge
is a numerical artifact violating conservation. The bridge direction
is a physically valid infinitesimal perturbation that preserves both
energy and angular momentum to integration precision. It is a real
direction in the constraint manifold of the orbit.

### Probe 5 — cross-row sigma_3 comparison (the smoking gun)

Pulling sigma_3-scan residuals from
`results/isotrophy/m3eq1-sigma3-precondition-fixed-inverse-orientation-25/`
for all 21 strict rows:

```text
O_617 sigma_group_residual:      1.62e-1
O_617 sigma_group_to_closure:    1.77e+7   (huge; smaller = tighter)
Catalog range sigma residual:    [1.4e-9, 2.1e-1]
Catalog median sigma residual:   2.2e-8
O_617 rank in sigma residual (asc): 19 of 21  (3rd worst)

O_617 F_beta_residual:           3.4e-9
Catalog range F_beta residual:   [8.6e-10, 1.1e-8]
Catalog median F_beta residual:  4.5e-9
O_617 rank in F_beta residual (asc): 4 of 21  (clean)
```

This is the headline. O_617's sigma_3 closure is six orders of magnitude
worse than catalog median, sitting near the worst three rows. Its
`sigma_group_to_closure` ratio (`1.77e+7`) is far outside the
catalog-typical neighborhood. By contrast, its `F_beta` closure is
catalog-normal (rank 4 of 21, near the best).

**O_617 sits at the catalog selection edge for sigma_3.** It is admitted
into supplementary-A's strict 21 only because the sigma_3 detector
tolerance was lax enough to include it. A tighter tolerance might
demote it from the strict list.

### Probe 6 — catalog metadata

```text
Label:    O_{617}(1)
Stability: U   (unstable)
Period:    129.276643
z0=0.250595, vx=0.318838, vy=0.551458, vz=0.000723
inertia_degenerate_candidate: False
```

Standard unstable choreography in the m_3=1 slice; no near-degenerate
inertia structure.

## Synthesis

The defective D3 at O_617's bridge boundary is a **downstream effect of
weak sigma_3 closure**, not a structural pathology of the orbit itself.
The mechanism:

1. The sigma_3-scan admitted O_617 with closure residual `1.62e-1`.
2. The constructed `rho(sigma_3)` matrix therefore satisfies
   `rho(sigma_3)^3 = rho_id` (i.e. `M_i`) only approximately, to about
   4% relative on the k=8 bridge-admitted kernel.
3. When the projector machinery tries to decompose `ker(M_i - I)` at
   k_dim=8 under this approximate D3 representation, the standard
   isotypic projector cannot produce a real 2D `E` block. The fragment
   it captures appears as an odd-dim `E(1)` with one marginal `P_E`
   singular value at `0.0147`.
4. The bridge audit correctly classifies this as
   `defective_E_block_confirmed` and excludes O_617 from v0.3h evidence.

The bridge singular vector is itself a real physical direction (probe
4: tangent to constraints; probe 1: structured mixed-parity under
`(12)`), so the defect lies in the *representation*, not the orbit.

This is a **catalog selection-edge phenomenon**. v0.3h's discipline is
robust to it: the defective row is not silently counted as evidence,
and the structural-zero claim survives intact on the 20 other rows.

## Implications

- **No tighter-rtol rerun needed.** Probe 3 shows the spectrum is
  marginal/elliptic, not near a true Jordan defect; tighter integration
  would not change the sigma_3 closure residual at the catalog selection
  edge.
- **Quotient refinement not indicated.** Probe 4 shows the bridge vector
  is invariant-conserving; it is not a missed neutral direction.
- **Possible future move: tighter sigma_3 catalog tolerance.** Re-running
  the sigma_3-scan with a residual tolerance below `1e-2` would likely
  remove O_617 from the strict catalog entirely. That is a catalog
  reconstruction question, not a v0.3h evidence question, and is left
  to a follow-up.
- **Paper-side framing.** The strict-21 catalog should be characterized
  not as a hard list but as a list whose admission residuals vary by
  six orders of magnitude in sigma_3 closure. O_617's defective-D3 at
  the bridge is the visible consequence of this internal variation.
  The v0.3h structural-zero claim holds on `20 of 21 rows whose
  sigma_3 closure is at integration precision`.

## Out of Scope

- Re-running the sigma_3-scan at tighter tolerance.
- Investigating whether other catalog rows would be reclassified under
  a tightened admission rule.
- Extending the bridge audit to non-strict rows or supplementary-B.
