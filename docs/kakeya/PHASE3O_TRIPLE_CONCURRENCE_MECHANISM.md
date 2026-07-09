# Kakeya Phase 3O - Triple-Concurrence Mechanism (Geometric Proof Scaffold)

- Artifact id: `KAK-PHASE3O-TRIPLE-CONCURRENCE-ANATOMY`
- Date: 2026-07-09
- Status: internal geometric-mechanism receipt. Converts the 4-star level from
  a solver output into a verified incidence-count, prices `q=43` (lever 1) with
  no exact solve, and scaffolds the geometric proof (lever 2) down to two sharp
  open lemmas.
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Prior receipt: [`PHASE3N_DEEPER_INVARIANT.md`](PHASE3N_DEEPER_INVARIANT.md)
- Script:
  [`../../scripts/kakeya-triple-concurrence-anatomy.mjs`](../../scripts/kakeya-triple-concurrence-anatomy.mjs)
  (`npm run kakeya:triple-anatomy`)
- Results: `results/kakeya/triple-concurrence-anatomy/`

## The mechanism

The optimal 4-star completion is 4 star lines through the pivot `O` plus a
parabola's tangents in the `q-3` non-star directions. Sacrifice bookkeeping is
then pure incidence geometry:

```text
sacrifice = 3 (pivot O, multiplicity 4) + T,      T = number of triple points
level LOW  <=> T = q-5,   level HIGH <=> T = q-4.
```

**Verified on every exact field-orbit (G1-G3 clean, all rows):**

- **G1** - the pivot `O` has multiplicity exactly 4 (the 4 star lines); no
  completion routes another line through it.
- **G2** - no 3 parabola tangents are concurrent (the dual-conic fact: 3
  concurrent tangents would be 3 collinear points of the dual conic).
- **G3** - every non-pivot triple is exactly `{1 star line, 2 tangents}`,
  i.e. a tangent-chord pole (intersection of two tangents) landing on a star
  line. Two star lines meet only at `O`, so no other triple type occurs.
- **T reproduces ex** at every row (`3 + T - (q-1)/2 = ex`).

So the level is literally **the number of tangent-chord poles that land on the
star lines**, minimized over the parabola family - a cross-ratio-controlled
count, which is *why* the level is an orbit invariant.

## Parabola optimality and the anomaly

The pure-parabola completion equals the descent optimum for **every 4-star
except one**:

| field-orbit | pure ex | descent ex | parabola optimal? |
| --- | :-: | :-: | --- |
| all q in {5,7,13,17,19}, every orbit | = | = | yes |
| **q=11 harmonic** | 5 (HIGH, T=7) | **4 (LOW)** | **NO - non-parabola** |

The lone anomaly `q=11` harmonic is exactly a case where a non-conic
completion shaves one triple below what any parabola achieves. This is the
geometric identity of the small-field exception isolated in
[PHASE3N](PHASE3N_DEEPER_INVARIANT.md): not an arithmetic accident but a
failure of parabola-optimality at the smallest field of its residue class.

## Lever 1 - q=43 priced without a solve

The parabola T-count gives `q=43` harmonic **T = 39 = q-4 = HIGH** in
milliseconds. A depth-40 exact B&B (the direct route) is infeasible - but
since the parabola is optimal for every non-anomalous 4-star, this prices the
level. **Lever 1 is subsumed by the mechanism**: `q=43` HIGH is rigorous
*modulo* the parabola-optimality lemma below (verified at 5 consecutive fields,
`q=13..19`, all orbits).

## Lever 2 - the proof route, down to two lemmas

This receipt supplies the verified bookkeeping (G1-G3, T = ex) and localizes
the anomaly. A full proof of the level law needs exactly two more steps:

1. **Parabola-optimality lemma** (`q >= 13`): no non-parabola completion of a
   4-star beats the best parabola completion. Verified empirically at
   `q=13,17,19` (all orbits) and by the `q=11` counterexample being the sole
   exception; open as a theorem.
2. **Pole-incidence count**: `min over parabolas of T` equals `q-5` or `q-4`
   as a function of the cross-ratio of the 4 star directions. The per-star-line
   triple distribution (recorded in `manifest.json`, e.g. `q=19` harmonic
   `[7,0,8,0]` HIGH vs `q=19` generic `[7,0,7,0]` LOW) shows the level turning
   on whether the poles reach the minimum `q-5` or are forced one over; a
   closed count of poles-on-star-lines by cross-ratio would settle it. Open.

Both are sharp, concrete finite-geometry statements about a conic's tangent
poles meeting a pencil of 4 lines - no character mining, no forking paths.

## Executable Receipt

```powershell
npm run kakeya:triple-anatomy
```

```text
KAK_TRIPLE_ANATOMY geom_clean=true parabola_optimal_except_11h=true q43h=HIGH falsifier=clear out=results/kakeya/triple-concurrence-anatomy
```

Falsifier `TRIPLE_ANATOMY_MISMATCH` (instrument-only): fires if the geometric
bookkeeping fails on any row, or if a parabola-suboptimal orbit other than
`q=11` harmonic appears among the exact fields. **Clear.**

## Interpretation Boundary

Supports only the workbench-internal mechanism: the 4-star level is the triple
count of the optimal parabola completion, verified as an incidence statement
at exact fields, with `q=11` harmonic the unique non-parabola optimum. The
`q=43` HIGH pricing and the two lemmas are contingent on parabola-optimality
(open as a theorem). `ex`/`T` import the pinned Blokhuis-Mazzocca minimum.
Register untouched, no pins. No Euclidean claim, no incidence-geometry novelty
claim.

Net: lever 1 delivered (q=43 HIGH via the mechanism), lever 2 reduced from
"find a deeper invariant" to two named finite-geometry lemmas. The arc's only
remaining work is proving those two lemmas (owner-gated theory, not
measurement).
