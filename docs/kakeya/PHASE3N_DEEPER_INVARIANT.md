# Kakeya Phase 3N - Deeper-Invariant Thread: Resolved (Null)

- Artifact id: `KAK-PHASE3N-DEEPER-INVARIANT-PROBE`
- Date: 2026-07-08
- Status: internal receipt. **The deeper-invariant hypothesis is a NULL:** no
  arithmetic invariant beyond the quadratic-character signature governs the
  4-star level. `sig+type` is the classifier, with `q=11` harmonic the unique
  small-field exception across the whole dataset.
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Prior receipt:
  [`PHASE3M_INVARIANT_HUNT_AND_EXTENSION.md`](PHASE3M_INVARIANT_HUNT_AND_EXTENSION.md)
- Scripts:
  [`../../scripts/kakeya-orbit-invariant-hunt.mjs`](../../scripts/kakeya-orbit-invariant-hunt.mjs)
  (`npm run kakeya:invariant-hunt`, now reports per-rule collisions),
  [`../../scripts/kakeya-deeper-invariant-probe.mjs`](../../scripts/kakeya-deeper-invariant-probe.mjs)
  (`npm run kakeya:deeper-invariant`)
- Results: `results/kakeya/deeper-invariant-probe/`

## Step 1 - The reduction (collision analysis)

Re-running the invariant hunt with the q=19 solves promoted to EXACT and
per-rule collision reporting shows `sig+type` has **exactly one collision**
across all 33 known field-orbits:

```text
sig+type consistent=false  collisions: 11/harmonic(LOW) vs 19/harmonic(HIGH) @ -1-1-1|harmonic
```

Every other orbit at every field `q in {5..37}` is classified correctly by the
quadratic-character signature refined by orbit type. The harmonic signature is
`{chi(2), chi(-1), chi(2)}`, so `sig = -1-1-1` iff `chi(2) = chi(-1) = -1` iff
**`q = 3 (mod 8)`**. Since the harmonic orbit is unique per field with
`j = 1728` fixed, its level is a pure function of `q`. The entire
deeper-invariant question therefore reduces to one residue class:

> For `q = 3 (mod 8)`, is the harmonic 4-star LOW or HIGH?
> Anchors: `q=11` LOW (exact), `q=19` HIGH (exact).

## Step 2 - Why the natural invariant is unavailable

`j = 1728` is the CM-by-`i` value; its automorphism group has order 4, so the
arithmetic that would govern it is the **quartic** residue character. Quartic
characters exist over `F_q` only when `q = 1 (mod 4)` - but the whole anomaly
class is `q = 3 (mod 8) subset q = 3 (mod 4)`, where `F_q` has no primitive
4th root of unity and the quartic character degenerates. The anomaly lives
exactly where the obvious invariant collapses. The probe therefore tests
whatever finer data could survive: `q mod {16, 24, 3, 5, 7}`, `chi(3)`,
`chi(5)`, `chi(q-2)`, and higher-power residues where defined.

## Step 3 - Extension by construction (the decisive data)

The descent-augmented construction (validated exact at 6/6 harmonic fields
`q <= 19`, incl. the B&B-confirmed `q=19`) generates the harmonic level across
the `q = 3 (mod 8)` class. Controls at `q in {5,7,13,17,29,37}` reproduce every
known harmonic level (falsifier clear). Result:

| `q (=3 mod 8)` | 11 | 19 | 43 | 59 | 67 | 83 | 107 | 131 |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| level | **LOW** | HIGH | HIGH | HIGH | HIGH | HIGH | HIGH | HIGH |
| epistemics | DEF-LOW (exact) | exact | prob | prob | prob | prob | prob | prob |

**`q=11` is the only LOW in the entire class.** (`DEF-LOW` = verified
completion at `(q-3)/2`, hard; `prob` = descent could not beat `(q-1)/2`, a
verified upper bound - see caveat.)

## Step 4 - Battery verdict: no arithmetic separator

Every candidate invariant either fails to separate `q=11` or is vacuous:

- `q mod 16`, `q mod 24`, `q mod {3,5,7}`, `chi(3)`, `chi(5)`, `chi(q-2)`:
  **inconsistent** - each puts `q=11(LOW)` in the same bucket as some
  `HIGH` member.
- `res4(2)`, `res8(2)`, `chi3(-1)`: "consistent" but **defined on 0 points** -
  quartic/octic characters do not exist in this class (Step 2). Vacuous.
- `res3(2)`, `cls_2_cubefree`: defined only on the `q = 1 (mod 3)` members
  (all HIGH); never applies to `q=11` (`11 = 2 mod 3`). Vacuous.
- `smallfield(q <= 11)`: the planted null marker; "consistent" precisely
  because it encodes the finding - `q=11` is the lone exception.

**The only separating rule is "is `q` small".** There is no deeper arithmetic
invariant. The 4-star level is governed by the quadratic-character signature
plus orbit type, and `q=11` harmonic is a small-field boundary anomaly - the
unique misclassification in the whole `q = 5..131` dataset.

## Resolution

The PHASE3M open thread closes as a **null on "deeper invariant"** and a
**positive on the classifier**:

> The 4-star embeddability level is `sig+type` (quadratic-character signature
> of the cross-ratio orbit, refined by orbit type). This is exact-correct at
> every measured field-orbit except the single small-field anomaly
> `q=11` harmonic; the `q = 3 (mod 8)` harmonic class is HIGH for all
> `19 <= q <= 131` and LOW only at `q=11`.

## Caveat (epistemics)

The anchors `q=11` (LOW) and `q=19` (HIGH) are exact; `q in {43..131}` are
construction upper bounds (PROB-HIGH). A construction HIGH could in principle
mask a true LOW (a family gap) - though the descent has a perfect harmonic
record (6/6 exact `q <= 19`, gap only ever at `q=11` pure, closed by descent),
and 120 pure seeds + 80 random restarts never found a sub-`(q-1)/2` completion
at any of the six large fields. A fully exact confirmation at `q=43` is
infeasible (depth-40 B&B). So the resolution is construction-strong, formally
contingent on the descent being exact at `q >= 43` harmonic. The hard core -
`q=11` is the unique DEF-LOW and no arithmetic invariant separates it - stands
regardless.

## Interpretation Boundary

Supports only the workbench-internal classifier statement above. Construction
levels beyond `q=19` are upper bounds. `ex` imports the pinned
Blokhuis-Mazzocca minimum. Register untouched, no pins (sidecar fields). No
Euclidean claim, no incidence-geometry novelty claim.

With this the PHASE3E-3N arc has no open threads: the 4-star excess is
two-level (`{(q-3)/2, (q-1)/2}`, exact through `q=19`, UB to `q=37`), the
level is `sig+type` with one small-field exception, and the deeper-invariant
hypothesis is retired. Optional future levers (owner-gated): an exact `q=43`
harmonic solve if ever affordable; the geometric triple-concurrence mechanism
(`q-5` vs `q-4` triples) as a proof route rather than a measurement.
