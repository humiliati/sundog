# Kakeya Phase 3F - Joint-vs-Marginal Activation Gap

- Artifact id: `KAK-PHASE3F-JOINT-ACTIVATION-GAP`
- Date: 2026-07-06
- Status: internal workbench diagnostic receipt; the PHASE3E next-rung option
  (interaction structure of shared added points), executed.
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Slate hook:
  [`../HODGE_KAKEYA_HYPOTHESES_SLATE.md`](../HODGE_KAKEYA_HYPOTHESES_SLATE.md)
- Prior receipt:
  [`PHASE3E_DIRECTION_ACTIVATION_SPECTRUM.md`](PHASE3E_DIRECTION_ACTIVATION_SPECTRUM.md)
- Script:
  [`../../scripts/kakeya-joint-activation-gap.mjs`](../../scripts/kakeya-joint-activation-gap.mjs)
- Manifest:
  [`../../results/kakeya/joint-activation-gap/manifest.json`](../../results/kakeya/joint-activation-gap/manifest.json)

## Verdict

**The gap between marginal and joint activation is a sandwich with an exactly
located tax.** For a set `D` of unlit directions:

```text
joint(D, K)  = min over per-direction line choices of |union(M_d \ K)|
gap(D, K)    = sum of marginals - joint(D, K)
0 <= gap <= C(|D|, 2)          (proven, any body)
deficit      = C(|D|, 2) - gap  (the forced concurrence tax)
```

Measured across 4364 enumerated direction sets: **from the empty body the
deficit is zero everywhere except the full pencil**, which pays exactly
`(q - 1) / 2`. The full-pencil joint cost is the minimal complete Kakeya set:
machine-verified by exhaustive enumeration to be `q(q+1)/2 + (q-1)/2` at
`q in {5, 7}` (17, 31) - the Blokhuis-Mazzocca planar minimum - and witnessed
(`<= 71`) by an explicit parabola construction at `q = 11`, where it **beats
the Phase-3E greedy cover (71 < 77)**.

## Exactness and the Sandwich

Any enlarged body lighting all of `D` contains one full line per target
direction, so the optimum ranges over per-direction line choices; the solver
enumerates them exactly (branch-and-bound) inside pre-registered caps.

- `joint <= sum of marginals`: the union of the per-direction argmin witnesses
  is one valid choice.
- `joint >= sum - C(k, 2)` (so `gap <= C(k, 2)`): inclusion-exclusion - lines
  of distinct directions share at most one point, so a choice of `k` lines
  saves at most `C(k, 2)` points, and each chosen line adds at least its
  direction's marginal.
- Tightness on finite slopes (odd `q`): the tangents of `y = x^2` at
  `t` have slope `2t` (a bijection onto the finite slopes) and no three are
  concurrent, so any all-finite-slope `D` from empty achieves
  `gap = C(k, 2)` exactly.

## Measured Structure (4364 rows, all checks pass)

**From empty, deficit-0 is the rule; the pencil is the only exception:**

| `q` | multi-direction sets enumerated | deficit-0 | exceptions |
| ---: | ---: | ---: | --- |
| `5` | `57` (all `k <= 6`) | `56` | full pencil: deficit `2 = (q-1)/2` |
| `7` | `247` (all `k <= 8`) | `246` | full pencil: deficit `3 = (q-1)/2` |
| `11` | `781` (`k <= 4`) | `781` | none in cap (pencil not enumerated) |

Perfect pairwise sharing survives even for `inf`-containing sets of every
enumerated size; only completing the *entire* pencil forces concurrences (a
`(q+2)`-arc of lines would be needed, and dual arcs stop at `q + 1`).

**Structured bodies complete to exactly the minimum.** Lighting all missing
directions costs `12` from a bare line at `q = 5` (`5 + 12 = 17`) and `24` at
`q = 7` (`7 + 24 = 31`); star-k2 lands on `17`/`31` the same way - these
bodies embed in minimal complete sets, and their full-missing-set deficit is
again exactly `(q - 1)/2`. The near-complete `star-k(q-1)` does **not** embed:
its two missing cost-1 directions cannot share a point (joint `2`, deficit
`1`), so it completes at `q^2 - q + 3`, above the minimum it already exceeds.

**Random bodies invert the regime.** Dense bodies have small marginals, so
`gap` is capped by the marginal sum long before `C(k, 2)`: deficits up to `20`
(`q = 7`, full missing set: marginals `23`, joint `15`), and tight rows are
rare (`6/57`, `9/247`, `41/781`). Full pairwise sharing is a structured-body
phenomenon, not a generic one.

## The Full Pencil and the Imported Anchor

The joint cost of the full pencil from empty *is* the minimal complete Kakeya
set size (any complete set contains a line per direction; the union of those
lines is complete). The workbench:

- **`q = 5`**: exhaustive over `5^6` choices -> `17`; greedy cover also `17`.
- **`q = 7`**: exhaustive over `7^8` choices -> `31`; greedy cover also `31`.
- **`q = 11`**: enumeration out of cap; the parabola witness (tangents at all
  `t`, plus the vertical `x = 0`, whose `q` tangent-intersections pair up as
  `t <-> 2c - t` leaving `(q-1)/2` new points) gives a complete set of size
  `71` - strictly below the greedy cover's `77`.

`q(q+1)/2 + (q-1)/2` as the exact odd-`q` planar minimum is the
**Blokhuis-Mazzocca theorem, imported**: machine-verified as an exhaustive
instance at `q in {5, 7}`, cited (not derived) at `q = 11`. Bibliographic pin
landed 2026-07-06 in [`../KAKEYA_LITPASS_MEMO.md`](../KAKEYA_LITPASS_MEMO.md)
(Addendum: Blokhuis-Mazzocca, Building Bridges, Bolyai Soc. Math. Studies 19,
Springer 2008, pp. 205-218; arXiv:0911.4370). Note the
Dvir floor `q(q+1)/2` is strictly below the true minimum by `(q-1)/2` - the
same quantity as the pencil's concurrence tax.

## Executable Receipt

Command:

```powershell
npm run kakeya:joint-gap
```

Output:

```text
KAK_JOINT_ACTIVATION_GAP q={5,7,11} rows=4364 sandwich=pass anchors=pass bruteforce_q5=pass q5_full_pencil=17/greedy=17 q7_full_pencil=31/greedy=31 q11_full_pencil=<=71/greedy=77 falsifier=clear out=results\kakeya\joint-activation-gap
```

Checks per row: sandwich (`max marginal <= joint <= sum`), gap bound
(`0 <= gap <= C(k,2)`), singleton agreement with the PHASE3E metric,
constructive witness (applying the argmin intercepts lights all of `D` at
exactly the claimed cost). Independent `q = 5` brute force for rows with
`joint <= 3`: no smaller point-addition lights the set (1255 complement
subsets exhausted). Pre-registered caps: all subsets at `q in {5, 7}`
(`(q+1)^{q+1}`-bounded enumeration), `|D| <= 4` at `q = 11`. Panel = the six
PHASE3E body families, same seeds. Deterministic, no API.

Falsifier `JOINT_GAP_MISMATCH`: fires on any sandwich/bound/witness/anchor
failure or a brute-force undercut. **Clear.**

## Regression Pin

`scripts/kakeya-workbench-tests.mjs` gains two pins per supported `q` (suite
now 60/60):

1. `T8i joint-pair-2q-minus-1`: from empty, any two directions (finite-finite
   and finite-inf) jointly cost exactly `2q - 1` - one shared point, gap `1`.
2. `T8j parabola-minimal-kakeya`: the tangent-plus-vertical witness has size
   exactly `q(q+1)/2 + (q-1)/2` and covers all `q + 1` directions.

## Interpretation Boundary

Supports only:

> In the finite-field workbench, joint activation of a direction set is an
> exactly solvable line-choice optimization; its gap to the marginal sum is
> sandwiched in `[0, C(k,2)]`, tight from structured bodies for every proper
> direction subset, with the full pencil paying a concurrence tax of exactly
> `(q-1)/2`.

The Blokhuis-Mazzocca equality at `q = 11` is imported literature, not a
workbench derivation. Enumeration caps are pre-registered budget limits, not
mathematical boundaries. Report-only diagnostic in `results/`; the public
shadow export is untouched. No Euclidean Kakeya claim, no maximal-function
claim, no new incidence geometry.

Next-rung options if pursued: (i) deficit-onset theory for `inf`-containing
sets via dual-arc extension (measured deficit-0 up to the caps; proven only
for finite slopes); (ii) joint activation from *nonempty* structured bodies as
an embeddability probe (which bodies sit inside minimal complete sets - the
line and 2-star do, the `(q-1)`-star does not).
