# K_facet v0.3 Freeze + Supplementary-B Comparison

Status: current, 2026-05-22.
Audience: collaborators, paper-side writers, future coding agents.
Receipt: `results/isotrophy/k-facet-v03-freeze-supplementary-b-comparison/manifest.json`.

## Frozen Prediction

The v0.3h `Gamma_i` audit chain resolves 20 of the 21 strict G.2
single-curve rows as structural zeros:

```text
20 rows: c_i = d_i = 0
 1 row: O_617 quarantined, excluded from evidence
```

The frozen resolved prediction is therefore:

```text
K_facet_v0.3h(resolved Gamma mechanism) = 0
```

This is a mechanism-level prediction, not a claim that piano-trios do not
exist. `O_617` remains excluded and does not supply a hidden positive count.

## Supplementary-B Inventory

`docs/isotrophy/supplementary-B_piano-init-condit-3d.txt` parses cleanly as the
local piano-trio catalog:

```text
supplementary-B rows:       273
m3 = 1 rows:                 38
m3 != 1 rows:               235
stability:                  97 S, 176 U
strict-index label overlaps: 62, 231, 617
```

The label overlaps are inventory only. They are not continuation matches and
do not by themselves identify daughters of strict G.2 seeds.

## Comparison Verdict

The comparison is now decisive:

```text
v0.3h Gamma mechanism predicts: 0 piano-trio daughters
supplementary-B observes:       273 piano-trio rows
```

So v0.3 closes as a **structural-null mechanism** against supplementary-B. The
audit chain succeeded at measuring its own prediction, and the prediction does
not explain the published piano-trio catalog.

The careful sentence is:

> v0.3h predicts zero resolved piano-trio daughters from the strict G.2 seeds
> via the `Gamma_i` standard-sector mechanism; supplementary-B is nonempty, so
> v0.3h does not explain those piano-trios.

Do not shorten this to "no piano-trios" or "Sundog disproves supplementary-B."
The negative is about the proposed v0.3 mechanism.

## Domain-of-Applicability Addendum (II symmetry probe, 2026-05-22)

The cross-m_3 sentinel sweep (`kfacet_v03_gamma_crossm3_preregistration.md`)
extended the audit chain to seven supplementary-B sentinels: four at
`m_3 = 0.4` and three at `m_3 = 1.0`. The verdict reads (Q1.D, Q2.D):
**gate pathology on both axes**, with kernel-projected `D3` residuals at
`10^3 -- 10^5`, six to eight orders of magnitude above the relation floor.

A direct `sigma_3-scan` probe on the same seven rows establishes the cause
at the catalog-style closure level, separate from the variational-kernel
`D3` gate:

```text
sigma_3 closure residuals (catalog-style, after SO(3) gauge minimization):

  Strict G.2 21 (m_3 = 1):     ~1e-9 to 3e-8    (admission threshold)
  Piano-trio sentinels:         0.60 to 0.79     (orbit-scale; ~7-9 orders above)

  sigma_any_strict_single_curve_candidate_count: 0 / 7 piano-trio rows

F_beta closure residuals (same probe):
  Strict G.2 21:                ~1e-9
  Piano-trio sentinels (6 of 7): 8e-9 to 4e-8    (clean, comparable to G.2)
  Outlier O_434(0.4):            0.25            (also broken)
```

The structural reading: **the tested supplementary-B piano-trio sentinels do
NOT carry the `sigma_3 = (123)`-cycle symmetry.** Six of seven carry the
`F_beta = (12)`-swap symmetry at integration precision; `O_434(0.4)` is the
outlier that also breaks `F_beta`. The tested set is therefore
`Z_2`-or-smaller, not `D_3`. The v0.3 `Gamma_i` mechanism is defined on
`D_3`-symmetric orbits, so it cannot be evaluated on this catalog with the
current ansatz.

The careful synthesis is now:

> v0.3 `Gamma_i` is a `D_3`-equivariant rank-zero prediction on the strict
> G.2 21. Supplementary-B's piano-trio sentinels live in `Z_2`-or-smaller
> symmetry classes, disjoint from the prediction's domain of applicability.
> Predicted 0 from G.2 (verified). Observed 273 piano-trios (verified).
> The mismatch is a **domain-of-applicability** finding: v0.3 cannot
> predict piano-trios by construction, because they do not carry the
> assumed symmetry. Any v0.4 prediction of piano-trios must start from
> `Z_2`-equivariant mechanisms, with smaller-symmetry outliers tracked
> explicitly.

`O_434(0.4)` is a separate sub-investigation candidate: it breaks
`F_beta` closure too (residual 0.25), suggesting an even smaller
symmetry class than the other six piano-trio sentinels.

## Reproducibility Surface

Inventory command:

```bash
python scripts/isotrophy_workbench.py parse --source B --path docs/isotrophy/supplementary-B_piano-init-condit-3d.txt
```

Receipts:

```text
results/isotrophy/k-facet-v03-freeze-supplementary-b-comparison/manifest.json
results/isotrophy/k-facet-v03-gamma-crossm3/m3eq{0.4, 1.0-suppB}/
results/isotrophy/k-facet-v03-piano-symmetry-probe/m3eq{0.4, 1.0-suppB}/
```
