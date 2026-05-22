# O_617 Bridge Sub-Investigation - Deep Dive

Status: current, corrected 2026-05-22.
Audience: collaborators, paper-side writers, future coding agents.
Companion: `internal/anniversary/kfacet_v03h_writeup.md`.
Receipt: `results/isotrophy/k-facet-v03-O617-deep-dive/deep_dive_receipt.json`.
Tooling: `scripts/o617_deep_dive.py` (one-shot, no integration).

## One-Line Read

> `O_617` is a clean opposite-strict catalog row. Its quarantine is not caused
> by an admission weakness; it is caused by a real bridge direction
> that falls outside the orbit's valid `D3` representation at the admitted
> kernel boundary.

This corrects the first deep-dive attribution. Sorting by canonical sigma3
residual made every opposite-strict row look weak. The orientation-aware
admission residual for `O_617` is `1.01e-8` (`to_closure = 1.105`), which is
catalog-normal. The old `1.62e-1` value is the canonical residual and is only a
diagnostic for an opposite-strict row; it is not the row's admission residual.

The bridge audit still stands: at `k_dim = 8`, admitting the bridge vector
makes the restricted representation defective (`sigma3^3 - I ~= 3.96e-2` and
an odd `E(1)` residual). The cause is not upstream catalog admission. It is a
structural boundary direction that the row's `D3` action does not carry as a
valid real standard irrep.

## Probe Summary

### Probe 1 - Geometric Anatomy Of `v_bridge`

The bridge singular vector has mixed `(12)` parity:

```text
(12)-symmetric  position norm: 7.48e-1
(12)-antisym.   position norm: 1.13e-1
(12)-symmetric  velocity norm: 9.36e-2
(12)-antisym.   velocity norm: 6.47e-1
```

It is mostly symmetric in positions and antisymmetric in velocities. That
mixed parity is a warning sign for the `D3` projection, which expects coherent
behavior under `F_beta = (12) x R_pi x tau`.

### Probe 2 - `sigma3^3` Closure At Kernel Floors

```text
On k=7 kernel: 1.181e-5  (clean)
On k=8 kernel: 4.07e-2   (defective bridge admitted)
```

The unambiguous k=7 kernel is clean. The order-3 failure appears only when the
bridge vector is admitted into the kernel.

### Probe 3 - Monodromy Spectrum

```text
Eigenvalue |.| range:              [0.8125, 1.2307]
Near unit circle (||.| - 1|<1e-2): 16 of 18
Near +1 (|lambda-1|<1e-2):         12 of 18
```

The spectrum is marginal/elliptic with reciprocal pairing as expected for a
symplectic monodromy. The bridge is not explained by a visible Jordan defect.

### Probe 4 - Constraint Tangency

```text
cos(v_bridge, grad H)   = 4.02e-10
cos(v_bridge, grad |L|) = -2.26e-9
```

The bridge vector is tangent to the `(E, |L|)` level set. It is physically
valid as an infinitesimal direction; the defect lives in the representation,
not in a conservation-law violation.

### Probe 5 - Orientation-Aware Sigma3 Admission

The strict-21 receipt splits as:

```text
13 canonical-strict rows
 8 opposite-strict rows
```

`O_617` is opposite-strict:

```text
O_617 admission orientation: opposite
O_617 admission residual:    1.010e-8
O_617 admission to closure:  1.105
O_617 canonical residual:    1.616e-1   (diagnostic only)
```

This is the correction. The row is not weakly admitted. It belongs normally to
the opposite-orientation strict catalog.

### Probe 6 - Catalog Metadata

```text
Label:     O_{617}(1)
Stability: U
Period:    129.276643
z0=0.250595, vx=0.318838, vy=0.551458, vz=0.000723
```

Standard unstable row in the `m_3 = 1` slice; no inertia-degeneracy flag.

## Synthesis

`O_617` is a clean opposite-strict choreography whose bridge-admitted kernel
contains a real near-kernel direction that the orbit's `D3` representation
does not act on irreducibly. The mechanism is:

1. The row is admitted normally under the opposite sigma3 orientation.
2. At the adaptive floor, `k_dim = 7` is clean and gives no standard sector.
3. The bridge singular value (`7.8359e-4`) is physically meaningful and
   tangent to the conserved-invariant surface.
4. Admitting that vector at `k_dim = 8` breaks the restricted order-3 relation
   at about 4% and yields `T(2)+S(6)+E(1)`.
5. `E(1)` is not a valid real standard `D3` block, so the bridge audit
   correctly classifies the row as `defective_E_block_confirmed`.

The honest label is therefore **structural bridge outside the valid D3
representation**, not admission weakness.

## WHY-Dive Addendum

The follow-up WHY-dive refines the bridge diagnosis again. The bridge direction
is not a loose quasi-kernel vector. It is a near-sign-isotypic direction:

```text
Rayleigh lambda at v_bridge:         0.999999
||(M-I)v|| / ||v||:                  7.84e-4
sigma3 v projection onto v_bridge:   0.9998
sigma3 v in k=8 fraction:            1.0000
<v, F_beta v>:                       -0.9999997
||F_beta v + v|| / ||v||:            7.79e-4
F_beta^2 v - v:                      0.0
sigma3^3 v - v:                      7.06e-2
```

So `v_bridge` tries to live in the sign `S` sector, not the trivial `T`
sector. `F_beta` acts as `-I` on the bridge direction, `F_beta^2` closes
exactly, and `sigma3` sends it almost exactly back to itself, but with a
sub-clean scalar drift. That drift accumulates over the three sigma3 actions
and leaves `sigma3^3 v` outside the `1e-3` relation floor. The character
projector cannot keep the approximate direction purely in `S`; the
bookkeeping residue appears as the defective odd `E(1)` block.

The corrected WHY outcome is:

```text
bridge_approx_sign_isotypic
```

This is still a quarantine. It is not a valid standard `D3` sector and not
countable evidence for `Gamma_i`. But the causal label is now sharper:
**O_617 sits at the edge of the sign sector, not outside D3 in a featureless
way.**

## Catalog-Wide Separator Addendum

The catalog-wide isotypic-edge separator (`scripts/catalog_near_t_separator.py`)
checks the same signed isotypic diagnostic across all 21 strict G.2 rows from
existing receipts. It confirms the WHY-dive is row-unique:

```text
clean_T:        42
clean_S:       116
clean_E:         0
edge_T:          0
edge_S:          2   (O_617 bridge contamination)
edge_E_as_T:     0
edge_E_as_S:     0
edge_E_other:    1   (O_617 bridge contamination)
```

Receipt: `results/isotrophy/k-facet-v03-near-T-separator/separator_manifest.json`.

The load-bearing parts are `edge_S_rows = [617]`, `edge_E_other_rows = [617]`,
and `clean_E_rows = []`. All 20 other rows are pure clean `T(2)+S(5 or 6)`
with no edge class of any kind. The `O_617` projector readout overcounts one
physical bridge direction (`T+S+E = 9` while `k_dim = 8`) because the
underlying group action on the bridge-admitted kernel is only approximate:
the bridge is caught mostly by `S` (near `F_beta = -I`) and weakly by
`E` as contamination. The separator nevertheless gives an independent check
that no catalog direction exhibits standard `E`-rotation behavior and that
`O_617` is the only row with a structural edge direction.

## Implications

- The v0.3h catalog result remains: 20 clean structural zeros plus one named
  quarantine.
- `O_617` is not counterevidence against the `Gamma_i` audit chain.
- A tighter sigma3 catalog-reconstruction sweep is no longer the natural next
  test for this row; its admission residual is already at integration scale.
- The WHY-dive's `bridge_approx_sign_isotypic` diagnosis is row-unique
  across the strict catalog, not a generic phenomenon.
- The open research question is what symmetry, near-symmetry, or boundary
  mechanism the bridge direction actually carries.

## Out Of Scope

- Promoting this script to a general `kfacet-row-anatomy` subcommand.
- Extending the bridge audit to non-strict rows or supplementary-B.
- Naming the bridge mechanism beyond the present defective-D3 diagnosis.
