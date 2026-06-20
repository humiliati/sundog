# BoxSEL Phase 2 — Single-Box Realizability Probe

**Date:** 2026-06-20  
**Status:** Probe landed (not a phase clearance). Separated, as planned, from the exact-oracle
semantic alignment — this is the geometric realizability question, the seed of the
**representation gap** `I_box ⊊ I*`.

## The question

The exact oracle ranges over **all** nonnegative type-volume models (`I*`). BoxSEL's actual body
is narrower: each atomic concept is a **single axis-parallel box**, and a query probability is a
box-volume ratio. So: *which tiny type-volume models are realizable by one axis-parallel box per
atom?* The gap between "any type-volume model" and "single-box-realizable" is exactly the
representation gap.

## The obstruction — axis-parallel Helly (number 2)

> If three axis-parallel boxes have all three **pairwise** intersections of positive volume,
> their **triple** intersection has positive volume too.

Per axis this is 1-D Helly for positive-length overlaps: a box overlaps another with positive
volume iff their intervals overlap with positive length on every axis, and for three intervals the
triple-overlap length is `min(highs) − max(lows)` = the overlap of the largest-low interval with
the smallest-high interval, itself a pairwise overlap, hence positive. The product over axes is
positive. The fact is **dimension-independent** (Helly number 2 for axis-parallel boxes).

## The representation-gap seed

Consider the type-volume model with **positive** mass on the pairwise cells `{A,B}`, `{B,C}`,
`{A,C}` and **zero** mass on the triple cell `{A,B,C}`:

- It is a **perfectly valid SEL type-volume model** — nonnegative weights, positive total; the
  exact oracle admits it as a finite/volume model (verified: `WeightedTypeModel` constructs and its
  `measure(A⊓B)` matches).
- It is **not realizable by any single-box embedding.** An exclusive pairwise cell is contained in
  the corresponding meet, so all three pairwise cells positive ⇒ all three meets positive ⇒ (Helly-2)
  the triple meet (= the triple cell) is positive — contradicting the zero triple cell.

Two flavors, both confirmed by exact random batteries (deterministic, `Fraction`s):

- **2-D:** some embeddings *do* stage all three pairwise cells positive, but **none** has a zero
  triple cell (`tested=11, violations=0`). Helly-2 forces the triple.
- **1-D (stronger):** **no** interval embedding even achieves all three pairwise cells positive
  (`tested=0`) — a single interval `B` covering parts on both sides of `A∩C` must contain `A∩C`,
  erasing the `{A,C}` cell.

The meet-level Helly-2 law is also checked directly (`tested=1391` in 1-D, `482` in 2-D pairwise-
positive triples; `0` violations each), plus an explicit 2-D triple and an exact 1-D Venn hand case.

## Artifacts & verification

- `scripts/boxsel_single_box.py` — exact box geometry (`box_meet`, `type_volumes` by axis
  partitioning), the Helly-2 witness, and the two random batteries.
- `scripts/test_boxsel_single_box.py` — **11/11 pass**, `python scripts/test_boxsel_single_box.py` → exit 0.

## Boundary & next work

- This is **realizability only**. It exhibits a valid SEL model no single box can produce; it does
  **not** yet exhibit a query `P(D|C)` whose box-attainable interval `I_box` is strictly inside the
  exact `I*`. That **query-interval manifestation is the Phase-4 target** (extremal query
  optimization), which this probe seeds: build an ontology that forces the pairwise co-occurrences,
  then read off a query sensitive to the triple cell.
- **Update 2026-06-20:** the first one-dimensional query-interval seed is now landed in
  [`PHASE4_HELLY_INTERVAL_GAP_SEED.md`](PHASE4_HELLY_INTERVAL_GAP_SEED.md): `I*=[0,1]` vs
  `I_box^1=[1/2,1]`. Higher-dimensional `I_box^n` remains future work.
- The probe uses axis-parallel boxes with no learned roles; BoxSEL's affine role maps are out of
  scope here (role-free fragment, consistent with the Phase-2 oracle).

---

*Sundog Research Lab — BoxSEL single-box realizability probe. Internal; a representation-gap seed,
not a phase clearance.*
