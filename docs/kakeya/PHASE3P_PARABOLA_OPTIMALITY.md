# Kakeya Phase 3P - Parabola-Optimality: Reduction to Relative Segre

- Artifact id: `KAK-PHASE3P-PARABOLA-OPTIMALITY`
- Date: 2026-07-09
- Status: internal proof-development receipt. **Two pieces PROVEN, one
  REDUCED.** The concurrency identity and the dual reformulation are proved
  self-contained; parabola-optimality is reduced to a relative Segre /
  Blokhuis-Mazzocca statement, cited but **not closed here** (honestly open).
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Prior receipt:
  [`PHASE3O_TRIPLE_CONCURRENCE_MECHANISM.md`](PHASE3O_TRIPLE_CONCURRENCE_MECHANISM.md)
- Litpass anchor:
  [`../KAKEYA_LITPASS_MEMO.md`](../KAKEYA_LITPASS_MEMO.md) (Segre 1955 added
  this pass)
- Support script:
  [`../../scripts/kakeya-dual-arc-verify.mjs`](../../scripts/kakeya-dual-arc-verify.mjs)
  (`npm run kakeya:dual-arc`)
- Results: `results/kakeya/dual-arc-verify/`

## Setup

A **completion** of a 4-star `K0` (4 lines through the pivot `O`, directions
`d1..d4`) adds one line in each of the `q-3` remaining directions, giving
`q+1` lines, one per direction of `PG(1,q)`. Write `m_p` for the number of
these lines through a plane point `p`, `|K|` for the number of distinct points
covered, and `sacrifice = sum_p (m_p-1)(m_p-2)/2`. The excess over the pinned
Blokhuis-Mazzocca minimum is `ex = sacrifice - (q-1)/2`.

## Proposition 1 (concurrency identity) - PROVED

> For every completion, `sacrifice = |K| - q(q+1)/2`.

*Proof.* The `q+1` lines have pairwise distinct directions, so every pair
meets in exactly one (finite) point; hence `sum_p C(m_p, 2) = C(q+1, 2)`.
Also `sum_p m_p = (q+1)q` (each line carries `q` points) and
`sum_{p: m_p>=1} 1 = |K|`. Using `(m-1)(m-2)/2 = C(m,2) - (m-1)`,

```text
sacrifice = sum C(m_p,2) - sum (m_p - 1)
          = C(q+1,2) - [ (q+1)q - |K| ]
          = |K| - q(q+1)/2.    ∎
```

So **minimizing sacrifice = minimizing `|K|`** (the completion covering the
fewest points). This is unconditional - no profile assumption.

## Proposition 2 (optimal profile) - PROVED given pivot exactly 4

The four star lines are concurrent at `O`, so `m_O >= 4`. A completion line
through `O` would raise `m_O`; since `(m-1)(m-2)/2` is strictly convex, moving
that line off `O` (feasible: `q-3 >= 1` free directions) does not increase
sacrifice, so some minimizer has `m_O = 4`. Fixing `m_O = 4` and again by
convexity, replacing any `m >= 4` point off `O` by lower multiplicities does
not increase sacrifice; the minimizer's off-pivot points have `m_p <= 3`.
Then, with a single mult-4 point (`O`, cost 3) and `T` triples (cost 1 each),

```text
sacrifice = 3 + T,    T = number of triple points,
LOW <=> T = q-5,   HIGH <=> T = q-4.
```

(The realizability of a given profile is a separate question; this is the
*shape* of an optimum, matching G1-G3 of
[PHASE3O](PHASE3O_TRIPLE_CONCURRENCE_MECHANISM.md).)

## Proposition 3 (dual reformulation) - PROVED (D1) + VERIFIED (D2-D3)

Dualize each completion line to a point of `PG(2,q)*`: finite slope `m`,
intercept `b` -> `[m : -1 : b]`; vertical `x=b` -> `[1 : 0 : -b]`. A plane
point of multiplicity `m` (an `m`-fold concurrence of completion lines)
dualizes to an **`m`-rich line** (a line meeting the `q+1` dual points in
exactly `m`). Hence:

- **D1 (proved).** The 4 star lines pass through `O`, so their intercept is
  `0` and every star dual has third coordinate `0`; the four star duals lie on
  the dual line `{Z=0} = O*`. The pivot is the unique forced rich line.
- **D2 (verified, exact fields).** The optimal completion's `q+1` dual points
  have exactly one 4-rich line (`O*`), `T` 3-rich lines (`T = sacrifice - 3`),
  and no `>= 5`-rich line.
- **D3 (verified, exact fields).** The `q-3` non-star dual points lie on a
  single conic - i.e. the completion lines are tangent to a conic.

`npm run kakeya:dual-arc` -> `D1+D2+D3` on all of `q in {5..19}`, every orbit;
falsifier `DUAL_ARC_MISMATCH` clear.

So in the dual, **minimizing sacrifice = minimizing the number of 3-secants of
`q+1` points, four of which are forced onto the line `O*`.** Zero 3-secants
off `O*` would make the free points a `(q-3)`-cap extending to an arc; the
extremal configuration is an arc, i.e. (Segre) a conic.

## The reduction, and why parabola-optimality is exactly relative Segre

**Parabola-optimality** ("the min-`|K|` completion has its `q-3` free lines
tangent to a conic") is, by Proposition 3, equivalent to:

> Among `q+1` points of `PG(2,q)` (`q` odd) with 4 forced on a fixed line and
> one point in each remaining direction-class, a configuration minimizing
> 3-secants has its free points on a conic.

This is a **relative Segre statement**. The absolute backbone:

- **Segre (1955):** every `(q+1)`-arc in `PG(2,q)`, `q` odd, is a conic.
- **Blokhuis-Mazzocca (2008):** the minimal Kakeya set in `AG(2,q)`, `q` odd,
  is the conic construction, size `q(q+1)/2 + (q-1)/2`, and every extremal
  example is of that form.

**It is not a corollary of BM.** BM classifies completions that reach the
*absolute* minimum. But a 4-star with `ex > 0` (PHASE3G: most of them) never
reaches it - its minimal completion is strictly larger than the BM set - so
BM's classification does not apply to the completion. Parabola-optimality is a
genuinely *relative* extremal statement about near-minimal, pivot-constrained
configurations. That is the open content.

**Why any proof must use `q >= 13`.** Parabola-optimality *fails* at `q = 11`
harmonic (PHASE3O: the optimum is a non-parabola completion). Segre holds for
all odd `q`, so a pure-Segre argument cannot distinguish `q = 11`; the true
proof must exploit the interaction of the forced pivot line with small field
size. This rules out the easy routes and locates the difficulty precisely.

## Status (honest)

| claim | status |
| --- | --- |
| Prop 1 concurrency identity `sacrifice = |K| - q(q+1)/2` | **PROVED** |
| Prop 2 optimal profile `sacrifice = 3 + T` (given pivot 4) | **PROVED** (shape) |
| Prop 3 D1 star duals collinear | **PROVED** |
| Prop 3 D2/D3 rich-profile + conic | **VERIFIED** (`q<=19`, all orbits) |
| Parabola-optimality (`q >= 13`) | **REDUCED to relative Segre; OPEN** |

Parabola-optimality is **not proved here.** It is reduced to a precise,
classical-flavored relative-Segre lemma with fully verified scaffolding, the
`q=11` exception explained as the arc-classification boundary, and the two
absolute theorems (Segre, BM) pinned. Closing the relative statement is the
remaining theory work.

## Interpretation Boundary

Supports only the workbench-internal reduction. Propositions 1-3(D1) are
proved for the finite-field workbench; D2/D3 are machine-verified at exact
fields; parabola-optimality remains conjectural (reduced, not closed). `ex`
imports the pinned BM minimum. Register untouched, no pins. No Euclidean
claim, no incidence-geometry novelty claim; Segre and BM are imported, not
reproved.

Next: the relative-Segre lemma (owner-gated theory) and lever-2's companion
pole-incidence count. With Prop 1-3 in hand both are now statements about a
conic's tangents meeting a 4-line pencil, not about the solver.
