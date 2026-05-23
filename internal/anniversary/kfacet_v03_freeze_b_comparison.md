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

## Reproducibility Surface

Inventory command:

```bash
python scripts/isotrophy_workbench.py parse --source B --path docs/isotrophy/supplementary-B_piano-init-condit-3d.txt
```

Receipt:

```text
results/isotrophy/k-facet-v03-freeze-supplementary-b-comparison/manifest.json
```

