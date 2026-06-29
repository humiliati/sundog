# Kakeya Phase 3C - Shadow Collision Audit, q=11 + Scaling Law

- Artifact id: `KAK-PHASE3C-SHADOW-COLLISION-Q11`
- Date: 2026-06-29
- Status: internal measurement receipt; extends [`PHASE3B_SHADOW_COLLISION_AUDIT.md`](PHASE3B_SHADOW_COLLISION_AUDIT.md) to the largest supported field.
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Slate hook: [`../HODGE_KAKEYA_HYPOTHESES_SLATE.md`](../HODGE_KAKEYA_HYPOTHESES_SLATE.md)
- Script: [`../../scripts/kakeya-shadow-collision-audit.mjs`](../../scripts/kakeya-shadow-collision-audit.mjs)
- `q=11` manifest: [`../../results/kakeya/shadow-collision-audit-q11/manifest.json`](../../results/kakeya/shadow-collision-audit-q11/manifest.json)

## Verdict

**The q=11 rung confirms PHASE3B and exposes an exact scaling law.** The registered
direction shadow is many-to-one at `q=11` (largest nonempty collision class `1221` over `12`
structured signatures), the guard witnesses pass, and `KAK_SHADOW_REENCODING_EMPIRICAL` does
not fire. Across all three supported fields the largest nonempty collision class is exactly

```
collision(q) = q * (q^2 - q + 1)   ->   q=5: 105,  q=7: 301,  q=11: 1221
```

so the shadow's lossiness grows **cubically (~q^3)** in the field size. This remains a
projection-discipline receipt: no extremal-set search, no Euclidean Kakeya, no public claim.

## Commands Run

```powershell
node scripts/kakeya-workbench-tests.mjs            # KAKEYA_WORKBENCH_TESTS q={5,7,11} pass=39 fail=0
node scripts/kakeya-shadow-collision-audit.mjs --q 11
```

The `q=11` audit reported (and re-ran reproducibly):

```text
KAK_SHADOW_COLLISION_AUDIT q=11 states=300000 signatures=1 collisions=1 max_collision=300000 max_nonempty_collision=0 structured_states=14652 structured_signatures=12 structured_max_nonempty_collision=1221 guard=pass falsifier=clear out=results\kakeya\shadow-collision-audit-q11
```

## Receipt Fields (q = 11)

| Field | Value |
| --- | --- |
| `q` | `11` |
| point count | `121` |
| direction count | `12` (`0..10, inf`) |
| Dvir floor `q(q+1)/2` | `66` |
| bounded enumerated states | `300000` (truncated) |
| bounded signature count | `1` (all-zero shadow) |
| bounded max nonempty collision | `0` |
| structured line-extension states | `14652` |
| structured signature count | `12` |
| structured max nonempty collision | `1221`, shadow `100000000000` |
| guard witnesses | pass |
| falsifier `KAK_SHADOW_REENCODING_EMPIRICAL` | clear |

## Why the bounded enumeration is empty at q>=7 (and the law's derivation)

A full line in `F_q^2` has `q` points, so with `--max-size 6` no enumerated body of size
`<= 6` can contain a line. Hence for `q in {7, 11}` every bounded-enumeration body has the
all-zero direction shadow: one signature, zero nonempty collisions. The bounded brute force
is informative only at `q=5` (where `max-size 6 >= 5 = q`); for larger fields it is line-free
by construction, and the **structured line-extension family** ("each line, plus each line with
one outside point added") is the carrier.

That family makes the law exact. The largest nonempty class is the single-direction shadow
`10...0`: it contains the `q` slope-0 lines, plus, for each slope-0 line, the `q^2 - q`
bodies formed by adding one outside point (a line plus one point cannot complete a second
full line, so the shadow stays `10...0`). That is

```
q * (1 + (q^2 - q)) = q * (q^2 - q + 1)  =  105, 301, 1221  for q = 5, 7, 11,
```

matching the measured counts exactly. The witnesses are a bare line (size `q`) and the same
line plus one point (size `q+1`) sharing one direction bit - the H-K3 teaching point that the
direction shadow forgets body information, now with a closed-form multiplicity.

## Interpretation Boundary

Supports only:

> In the finite-field workbench, the registered direction-coverage bitset is a lossy
> projection whose collision multiplicity grows as `q(q^2 - q + 1)`; many distinct point
> bodies cast the same direction shadow, increasingly so as `q` grows.

It does not support any statement about Euclidean Kakeya, optimal finite-field Kakeya sets,
maximal-function estimates, or new incidence geometry. `q=11` is the largest field registered
in `kakeya-core`; the bounded enumeration is structurally vacuous for `q>=7`, so this
structured-family measurement is the ceiling of the current approach. The original
`KAK-SHADOW-REENCODING` guard still fires if a future surface promotes witness lines or point
membership as the primary signature.
