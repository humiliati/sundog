# Kakeya Phase 3E - Direction Activation Spectrum

- Artifact id: `KAK-PHASE3E-DIRECTION-ACTIVATION-SPECTRUM`
- Date: 2026-07-06
- Status: internal workbench diagnostic receipt; turns the PHASE3D lemma's
  single threshold into an exact per-direction metric for any body.
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Slate hook:
  [`../HODGE_KAKEYA_HYPOTHESES_SLATE.md`](../HODGE_KAKEYA_HYPOTHESES_SLATE.md)
- Prior receipt:
  [`PHASE3D_SHADOW_COLLISION_LEMMA.md`](PHASE3D_SHADOW_COLLISION_LEMMA.md)
- Script:
  [`../../scripts/kakeya-direction-activation-spectrum.mjs`](../../scripts/kakeya-direction-activation-spectrum.mjs)
- Manifest:
  [`../../results/kakeya/direction-activation-spectrum/manifest.json`](../../results/kakeya/direction-activation-spectrum/manifest.json)

## Verdict

**Every direction now has an exact price tag.** For any body `K` in `F_q^2`
and any direction `d`, the minimal number of added points that lights `d` in
the registered shadow is

```text
activation(d, K) = min over the q intercept lines M in direction d
                   of (q - |M intersect K|)
```

with the argmin line as an explicit witness. This is exact, cheap
(`(q+1) * q` line overlaps per body), and works for any body. The PHASE3D
lemma becomes one row of the table: the spectrum of a bare line is
`[0, q-1, ..., q-1]`.

## Exactness

Lower bound: any enlarged body that lights `d` contains some full
direction-`d` line `M`, which requires adding at least `q - |M intersect K|`
points, hence at least the min. Achievability: add exactly the witness line's
missing points. The shadow is monotone under adding points, so incidental
lighting of other directions never invalidates the count for `d`.

## Anchor Laws (all machine-checked at q in {5, 7, 11})

| body | spectrum law |
| --- | --- |
| empty | every direction costs `q` |
| bare line | own `0`, every other direction exactly `q - 1` (PHASE3D sharpness as a metric) |
| line + 1 outside point | every other direction exactly `q - 2` (the line through that point reuses it plus one `L`-crossing) |
| threshold cross `L union M` | both lit, every third direction exactly `q - 2` |
| `k` concurrent lines, `k <= q - 1` | every missing direction exactly `q - k` |
| `k = q` star (pencil minus one direction) | **all zero** - see below |
| whole plane / greedy cover | all zero |

The `k`-star law receipts the **marginal-cost ladder**: the `(k+1)`-th
direction bit costs exactly `q - k` from a `k`-star, telescoping
`q + (q-1) + ... + 1 + 0 = q(q+1)/2` = the Dvir floor as a *greedy lower-bound
schedule*. The floor itself remains only a bound - the spectrum's per-direction
costs do **not** add up to a joint lighting cost, because one added point can
serve several directions (the greedy cover reaches all-zero at size 17/31/77
for `q = 5/7/11`, far below `q * (q-1) + q`).

**The star-closure surprise (`k = q`):** the pencil through a point minus one
direction is already a complete Kakeya set of size `q^2 - q + 1`. Every point
off the pivot lies on exactly one pencil line; a missing-direction line
avoiding the pivot is disjoint from the pivot's missing-direction line
(parallel), so all `q` of its points lie on star lines. The `q - 1` off-pivot
lines of the "removed" direction are inside the star for free. The spectrum
sees it instantly: `star-k(q-1)` has two directions at cost `1`, `star-kq` is
all-zero.

## Measured Panel (q = 5 shown; 13 bodies per field, 39 total)

```text
empty            size=0   costs=[5,5,5,5,5,5]
single-line      size=5   costs=[0,4,4,4,4,4]
line-plus-1      size=6   costs=[0,3,3,3,3,3]
line-plus-safe   size=8   costs=[0,3,3,3,3,3]
threshold-cross  size=9   costs=[0,0,3,3,3,3]
star-k2          size=9   costs=[0,0,3,3,3,3]
star-k3          size=13  costs=[0,0,0,2,2,2]
star-k4          size=17  costs=[0,0,0,0,1,1]
pencil-minus-one size=21  costs=[0,0,0,0,0,0]
whole-plane      size=25  costs=[0,0,0,0,0,0]
greedy-cover     size=17  costs=[0,0,0,0,0,0]
random-third     size=8   costs=[3,2,2,2,2,2]
random-half      size=12  costs=[1,2,2,2,1,0]   (organically lit: contains a full vertical line)
```

`line-plus-safe` shows why the lemma's `>= 1` fence is not tight body-by-body:
the prefix points happen to be collinear in the body's own direction, so other
directions still cost `q - 2`. Random bodies are seeded (`mulberry32`),
deterministic, no API.

## Executable Receipt

Command:

```powershell
npm run kakeya:activation-spectrum
```

Output:

```text
KAK_DIRECTION_ACTIVATION_SPECTRUM q={5,7,11} bodies=39 anchors=pass witness=pass bruteforce_q5=pass falsifier=clear out=results\kakeya\direction-activation-spectrum
```

Checks per body:

- **coherence** - cost `0` iff the registered `shadowBitset` bit is already
  set;
- **witness** - adding the witness points lights the direction, at exactly the
  claimed cost;
- **anchor** - the law column above, where declared;
- **brute force (q = 5 only)** - independent exhaustive search: for every
  direction with claimed cost `c >= 1`, *no* `(c-1)`-point addition lights it
  (all subsets of the complement enumerated through `Core.shadowBitset`;
  12650 subsets for the empty body alone, ~14k across the panel).

Falsifier `ACTIVATION_SPECTRUM_MISMATCH`: fires on any coherence, witness,
anchor, or brute-force disagreement. **Clear.**

## Regression Pin

`scripts/kakeya-workbench-tests.mjs` gains two pins for every supported `q`
(suite now 54/54):

1. `T8g bare-line-activation-q-minus-1`: every non-base direction of a bare
   line is exactly `q - 1` points away.
2. `T8h star-ladder-activation`: `k`-stars (`k = 2` and `k = q - 1`) price
   every missing direction at exactly `q - k`, and the `k = q` star covers all
   `q + 1` directions.

## Interpretation Boundary

Supports only:

> In the finite-field workbench, each unlit direction of the registered shadow
> has an exact, witnessed minimal activation cost, computable per body; bare
> lines, stars, and pencils obey closed-form spectra.

Per-direction costs are marginals, not a joint budget. The spectrum and its
witness fields are report-only diagnostics in `results/`; the public shadow
export (spec §5.3) is untouched and the `T10` export guard still passes. No
Euclidean Kakeya claim, no extremal-set claim, no new incidence geometry.

Next-rung option if pursued: the **joint-vs-marginal gap** - the exact minimal
addition lighting a *set* of directions versus the sum of its marginals
(interaction structure of shared points).
