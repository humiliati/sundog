# v0.13 External Target Inventory Snapshot

Status: populated 2026-05-31 from a signal-blind literature/data sweep. This is
a source inventory only: no profile receipt, no D5 probe, no target selection, no
`velocity_fraction` computation, and no S/U-vs-feature statistic.

Runner inventory path:

```text
results/isotrophy/k-facet-v13-external-target-search/target_inventory.csv
```

That path is gitignored with the rest of `results/isotrophy/`, so this document is
the tracked snapshot of the candidate ledger.

## Candidate Ledger

| slug | tier | row count | substrate | label convention | current reading |
| --- | ---: | ---: | --- | --- | --- |
| `liao2021_nonhierarchical_unequal_mass_135445` | 2 | 135,445 | full Newtonian planar | source S/U linear stability | Best schema candidate; needs a non-mirrored planar parser and general-state D5 adapter before profile/probe. |
| `li_liao2018_pasj_unequal_mass_1349` | 2 | 1,349 | full Newtonian planar | S/M/U, marginal not locked | Large and relevant, but blocked until a marginal-row policy is pre-registered. |
| `hristov2024_freefall_equal_mass_24582` | 3 | 24,582 | full Newtonian free-fall | downloadable monodromy eigenvalues | Independent and large, but likely fails the conditioning-strata gate because all rows are equal-mass. |
| `hristov2024_eulerian_equal_mass_12431` | 3 | 12,431 ICs | full Newtonian planar | paper reports linear stability; only 7 stable | Independent and large, but likely fails conditioning-strata and sparse-stable gates. |
| `jpl_poincare_cr3bp_api` | 3 | API-dependent | circular restricted three-body | stability index, not supp-B S/U | Excellent external catalog, but wrong substrate for frozen full-Newtonian D5 transfer. |
| `suvakov_dmitrasinovic2013_prl_13_orbits` | 3 | 15 ICs / 13 distinct | full Newtonian planar | no large source S/U table | Historical anchor; fails row-count and conditioning-strata gates. |
| `dmitrasinovic_hudomal_shibayama_sugita2018_stability` | 3 | several hundred | full Newtonian planar | monodromy/Floquet linear stability | Strong independent stability reference, but likely one mass stratum and still needs a preserved machine-readable IC source. |
| `li_liao2019_freefall_different_mass_316` | 2 | 316 | full Newtonian free-fall | no source S/U table found in sweep | Meets row-count floor and may have mass variation, but fails label-convention gate unless a stability table is found. |

## Best Next Fork

The strongest actionable fork is the Li/Liao 2021 non-hierarchical table. It has
the right scale, substrate, mass variation, and S/U labels, but it is not the
mirrored ansatz:

```text
r1(0) = (x1, 0)
r2(0) = (1, 0)
r3(0) = (0, 0)
v1(0) = (0, v1)
v2(0) = (0, v2)
v3(0) = (0, -(m1*v1 + m2*v2)/m3)
```

A follow-up implementation should add a general-state D5 feasibility adapter that
keeps the v0.13 firewall: no `velocity_fraction`, no `zone_index`, and no
stability labels in per-row feasibility receipts.

## Sources

- Li/Liao GitHub repository: <https://github.com/sjtu-liao/three-body>
- Li/Liao 2021 non-hierarchical raw table: <https://raw.githubusercontent.com/sjtu-liao/three-body/main/non-hierarchical-3b-supplementary_data.txt>
- Li/Jing/Liao 2018 PASJ paper and supplement: <https://academic.oup.com/pasj/article/70/4/64/4999993>
- Hristov/Hristova/Dmitrasinovic/Tanikawa free-fall data page: <https://db2.fmi.uni-sofia.bg/3bodyfree/>
- Hristov/Hristova Eulerian-configuration catalog paper: <https://arxiv.org/abs/2404.16526>
- NASA/JPL periodic-orbits API: <https://ssd-api.jpl.nasa.gov/doc/periodic_orbits.html>
- Suvakov/Dmitrasinovic 2013: <https://arxiv.org/abs/1303.0181>
- Dmitrasinovic/Hudomal/Shibayama/Sugita 2018: <https://arxiv.org/abs/1705.03728>
