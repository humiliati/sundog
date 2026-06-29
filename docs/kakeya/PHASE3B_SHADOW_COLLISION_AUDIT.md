# Kakeya Phase 3B - Shadow Collision Audit

- Artifact id: `KAK-PHASE3B-SHADOW-COLLISION-AUDIT`
- Date: 2026-06-29
- Status: internal measurement receipt for the registered direction shadow.
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Slate hook:
  [`../HODGE_KAKEYA_HYPOTHESES_SLATE.md`](../HODGE_KAKEYA_HYPOTHESES_SLATE.md)
- Script: [`../../scripts/kakeya-shadow-collision-audit.mjs`](../../scripts/kakeya-shadow-collision-audit.mjs)
- `q=5` output manifest:
  [`../../results/kakeya/shadow-collision-audit/manifest.json`](../../results/kakeya/shadow-collision-audit/manifest.json)
- `q=7` output manifest:
  [`../../results/kakeya/shadow-collision-audit-q7/manifest.json`](../../results/kakeya/shadow-collision-audit-q7/manifest.json)

## Verdict

**H-K3 landed as a report-only audit and now has a `q=7` extension.** The
registered finite-field Kakeya direction shadow is many-to-one on the reproduced
default bounded `q = 5` state space, on a deterministic line-extension family
for `q in {5, 7}`, and on the two Phase-2 guard witnesses. The empirical
falsifier `KAK_SHADOW_REENCODING_EMPIRICAL` did not fire.

This is a projection-discipline receipt, not a Kakeya result. It does not search
for extremal sets, does not touch Euclidean Kakeya, and does not license public
claim copy.

## Command Run

```powershell
npm run kakeya:test
npm run kakeya:shadow-collision:all
```

The updated regression suite reported:

```text
KAKEYA_WORKBENCH_TESTS q={5,7,11} pass=39 fail=0
```

The `q=5` audit command:

```powershell
npm run kakeya:shadow-collision
```

reported:

```text
KAK_SHADOW_COLLISION_AUDIT q=5 states=245506 signatures=7 collisions=7 max_collision=244876 max_nonempty_collision=105 structured_states=630 structured_signatures=6 structured_max_nonempty_collision=105 guard=pass falsifier=clear out=results\kakeya\shadow-collision-audit
```

The `q=7` audit command:

```powershell
npm run kakeya:shadow-collision:q7
```

reported:

```text
KAK_SHADOW_COLLISION_AUDIT q=7 states=300000 signatures=1 collisions=1 max_collision=300000 max_nonempty_collision=0 structured_states=2408 structured_signatures=8 structured_max_nonempty_collision=301 guard=pass falsifier=clear out=results\kakeya\shadow-collision-audit-q7
```

Equivalent explicit command recorded by the manifest:

```powershell
node scripts/kakeya-shadow-collision-audit.mjs --q 5 --max-size 6 --max-states 300000
node scripts/kakeya-shadow-collision-audit.mjs --q 7 --max-size 6 --max-states 300000
```

## Receipt Fields

### `q = 5`

| Field | Value |
| --- | --- |
| `q` | `5` |
| point count | `25` |
| direction order | `0, 1, 2, 3, 4, inf` |
| max body size enumerated | `6` |
| enumerated states | `245506` |
| truncated | `false` |
| signature count | `7` |
| collision signature count | `7` |
| different-size collision signatures | `7` |
| max collision class | `244876` bodies with empty shadow `000000` |
| max nonempty collision class | `105` bodies with one-direction shadows |
| structured line-extension states | `630` |
| structured signatures | `6` |
| structured max nonempty collision class | `105` |

### `q = 7`

The bounded enumeration cap intentionally does not pretend to cover all
size-7 bodies in `F_7^2`. With body size `<= 6`, no nonempty shadow can appear
because a full line has `7` points. The nonempty `q=7` receipt therefore comes
from the deterministic line-extension family: each line, plus each same line
with one outside point added.

| Field | Value |
| --- | --- |
| `q` | `7` |
| point count | `49` |
| direction order | `0, 1, 2, 3, 4, 5, 6, inf` |
| max body size enumerated | `6` |
| enumerated states | `300000` |
| truncated | `true` |
| signature count | `1` |
| bounded max nonempty collision class | `0` |
| structured line-extension states | `2408` |
| structured signatures | `8` |
| structured collision signatures | `8` |
| structured different-size collision signatures | `8` |
| structured max nonempty collision class | `301` |

## Guard Witnesses

The script checks the two Phase-2 `KAK-SHADOW-REENCODING` guard witnesses before
the enumeration and structured family.

1. **Single-direction parallel-line collision.** The slope-0 intercept-0 line
   and the slope-0 intercept-1 line are different bodies of size `5` with the
   same primary shadow `100000`.
2. **Complete-shadow collision.** The whole plane and the whole plane minus
   point `0` are different bodies of sizes `25` and `24` with the same complete
   primary shadow `111111`.

Both guard witnesses passed for `q=5` and `q=7`.

## Enumeration Witness

### `q = 5`

The largest nonempty collision class has shadow `100000` and count `105`.

Two compact witnesses:

| Body | Size | Point indices | Shadow |
| --- | ---: | --- | --- |
| slope-0 line | `5` | `0 1 2 3 4` | `100000` |
| same line plus one point | `6` | `0 1 2 3 4 5` | `100000` |

The extra point does not add a full line in a new direction, so the body changes
while the direction shadow does not. This is the H-K3 teaching receipt: the
primary shadow is useful exactly because it forgets body information.

### `q = 7`

The structured line-extension family's largest nonempty collision class has
shadow `10000000` and count `301`.

Two compact witnesses:

| Body | Size | Point indices | Shadow |
| --- | ---: | --- | --- |
| slope-0 line | `7` | `0 1 2 3 4 5 6` | `10000000` |
| same line plus one point | `8` | `0 1 2 3 4 5 6 7` | `10000000` |

Again, the extra point changes the body but does not add a full line in a new
direction.

## Regression Pin

`scripts/kakeya-workbench-tests.mjs` now includes two additional projection
tests for every supported `q in {5, 7, 11}`:

1. A full line and the same line plus one off-line point have the same nonempty
   direction shadow but different body sizes.
2. The whole plane and the whole plane minus one point have the same complete
   shadow but different body sizes.

These tests pin the load-bearing claim that the registered shadow is a lossy
projection rather than a disguised body encoding.

## Output Files

- `results/kakeya/shadow-collision-audit/manifest.json`
- `results/kakeya/shadow-collision-audit/signature-summary.csv`
- `results/kakeya/shadow-collision-audit/structured-line-extension-summary.csv`
- `results/kakeya/shadow-collision-audit/witnesses.json`
- `results/kakeya/shadow-collision-audit/operator-commands.md`
- `results/kakeya/shadow-collision-audit-q7/manifest.json`
- `results/kakeya/shadow-collision-audit-q7/signature-summary.csv`
- `results/kakeya/shadow-collision-audit-q7/structured-line-extension-summary.csv`
- `results/kakeya/shadow-collision-audit-q7/witnesses.json`
- `results/kakeya/shadow-collision-audit-q7/operator-commands.md`

## Interpretation Boundary

The audit supports only this narrow sentence:

> In the finite-field workbench, the registered direction-coverage bitset is a
> lossy projection: many distinct point bodies cast the same direction shadow.

It does not support any sentence about Euclidean Kakeya, optimal finite-field
Kakeya sets, maximal-function estimates, or new incidence geometry. If a future
surface promotes all witness lines or point membership as the primary signature,
the original `KAK-SHADOW-REENCODING` guard still fires.
