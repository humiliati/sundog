# Kakeya Phase 3U - Classification Route: Existence Map + Parabola Blind Spot

- Artifact id: `KAK-PHASE3U-CLASSIFICATION-ROUTE`
- Date: 2026-07-09
- Status: internal classification-route receipt. **The uniqueness theorem is
  not attempted here; the necessary prior - the existence leg - is mapped:**
  extremal permutations constructed at `q in {13, 19}`, the parabola blind
  spot proved at `q = 11`, existence at `q in {17, 23}` isolated as a new
  open sub-question that reshapes the banked conjecture.
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Prior receipt:
  [`PHASE3T_PERMUTATION_CENSUS.md`](PHASE3T_PERMUTATION_CENSUS.md)
- Scripts:
  [`../../scripts/kakeya-canonical-extremal.mjs`](../../scripts/kakeya-canonical-extremal.mjs)
  (`npm run kakeya:canonical-extremal`),
  [`../../scripts/kakeya-perm-existence.mjs`](../../scripts/kakeya-perm-existence.mjs)
  (`npm run kakeya:perm-existence`)
- Results: `results/kakeya/canonical-extremal/`,
  `results/kakeya/perm-existence/`

## Where the classification stood

Banked conjecture (PHASE3T): *the floor-extremal permutation is unique up to
`G+` for all odd `q >= 7`* - exhaustively true at `q in {7, 11}`. Attacking
uniqueness at general `q` first requires canonical models at general `q`,
i.e. an existence construction. That is what this receipt maps.

## Step 1 - Extraction instrument (dualize, find a bijective direction, shear)

A completion's `q+1` dual representations are parametrized by the choice of
"infinite" line; a representation yields a permutation iff some translate is
bijective (`N_s = q`), after a shear. Findings:

- The **first-found** parabola optimum usually has NO bijective direction
  (only `q=7`-harmonic and `q=13`-generic extracted on the first try).
- The `q=7` extraction produced an extremal with cycle type `[5,2]`, `0`
  fixed points - same unique `G`-orbit as the `[3-cycle + 4 fixed]` model,
  **definitively killing cycle type as an orbit invariant** (as suspected:
  two-sided affine composition does not preserve cycle structure).

## Step 2 - Existence probe (E1 exhaustive-in-family; E2 hillclimb)

**E1**: ALL parabola optima per LOW orbit, ALL `q+1` directions each:

| q / orbit | parabola optima | permutation reps | witness class |
| --- | :-: | :-: | :-: |
| 11 generic | 8 | **0** | - |
| 13 equianharmonic | 12 | 0 | - |
| 13 generic | 4 | **4 (verified)** | generic |
| 17 harmonic / gen-a / gen-b | 8 / 4 / 4 | 0 / 0 / 0 | - |
| 19 generic-a | 8 | **4 (verified)** | generic |
| 19 generic-b | 8 | 0 | - |
| 23 harmonic / gen-a / gen-b | 8 / 8 / 8 | 0 / 0 / 0 | - |

**E2** (transposition hillclimb, frozen fixed-point 4-fiber): the `q=11`
control found a **verified witness at sum-sigma = 9** (the instrument works
where extremals are dense); it missed everywhere at `13..23` *including*
`q=13` where E1 proves existence - so E2 misses at `17/23` are weak evidence,
not nonexistence.

## Findings

1. **Existence proved constructively at `q in {13, 19}`** (verified
   witnesses; both generic 4-fiber class, extending PHASE3T's generic-only
   law to every constructed witness so far). With the exhaustive `q in
   {7, 11}`, extremal permutations exist at `q in {7, 11, 13, 19}`.
2. **The parabola blind spot is proved at `q = 11`**: 24,200 extremal
   permutations exist (census), yet ALL 8 parabola optima of the generic
   orbit carry none - the extremal permutations dualize exclusively to
   **non-parabola optimal completions**. (Consistent with PHASE3Q: parabola-
   optimality is about the optimal *value*; non-parabola optima coexist.)
3. **Existence at `q in {17, 23}` is OPEN** - the parabola family provably
   yields none there (exhaustive within family), and the weak hillclimb found
   none. `q = 17` is the only tested field with all orbits LOW and the only
   `1 (mod 8)` field in range; whether that is relevant is unknown (one data
   point).
4. **The banked conjecture reshapes.** "Unique up to `G+` for all odd
   `q >= 7`" presupposed existence. The corrected form is two questions:
   *(a) for which `q` do floor-extremal permutations exist?* (known: yes at
   7, 11, 13, 19; open at 17, 23; no pattern yet) and *(b) where they exist,
   are they unique up to `G+`?* (known: yes at 7, 11; open elsewhere).

## Falsifiers

`CANONICAL_EXTREMAL_MISMATCH`, `PERM_EXISTENCE_MISMATCH` (instrument-only):
every extracted or hillclimbed candidate must verify exactly (permutation +
`sum-sigma = q-2` + exactly one 4-fiber), and nothing may fall below proved
floors. **Both clear.** All published witnesses are verified.

## Honest status

The classification theorem (uniqueness) was not advanced this turn beyond its
prerequisites; what was delivered is the existence map that any correct
statement of the theorem needs, one proved blind spot, one dead fingerprint
(cycle type), and two verified new canonical models. Next levers if pursued:
(i) decide existence at `q = 17` (stronger search: simulated annealing with
sideways moves / exact ILP-style over the 4-fiber-frozen permutation space,
or a structural argument for why `17` differs); (ii) enumerate the
NON-parabola optima at `q = 11`-generic that carry the permutations, as the
canonical-model source the parabola family misses.

## Interpretation Boundary

Workbench-internal. Existence results are verified witnesses; absences are
exhaustive only within the parabola family (E1) or weak-search (E2); the
floor for general `q` and the uniqueness theorem remain open. No Euclidean
claim.
