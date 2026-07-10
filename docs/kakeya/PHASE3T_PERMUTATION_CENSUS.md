# Kakeya Phase 3T - Permutation-Slice Census + Extremal Orbit Theorem

- Artifact id: `KAK-PHASE3T-PERMUTATION-CENSUS`
- Date: 2026-07-09
- Status: internal extremal-structure receipt. **Exhausted the permutation
  sub-conjecture at q in {7, 11} (and sampled q=13); found the floor tight in
  the permutation slice with a UNIQUE extremal up to symmetry, a universal
  extremal anatomy, and a generic-only 4-fiber law.** No new bounds (for
  q <= 17 the sub-conjecture is a corollary of the proved floor); the yield is
  structure for the analytic attack.
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Prior receipt:
  [`PHASE3S_VALUESET_REDUCTION.md`](PHASE3S_VALUESET_REDUCTION.md)
- Scripts:
  [`../../scripts/kakeya-permutation-census.mjs`](../../scripts/kakeya-permutation-census.mjs)
  (`npm run kakeya:perm-census`),
  [`../../scripts/kakeya-extremal-orbits.mjs`](../../scripts/kakeya-extremal-orbits.mjs)
  (`npm run kakeya:extremal-orbits`)
- Results: `results/kakeya/permutation-census/`,
  `results/kakeya/extremal-orbits/`

## Scope note (what this can and cannot show)

The permutation form - `f` a permutation of `F_q` with a 4-point fiber in some
translate `f - s*id` implies `sum_s sigma_s >= q-2` - is a **corollary of the
proved floor for q in {5,7,11,13,17}** (exhaustive at q<=7; all-orbit exact
B&B + PGL transitivity at 11/13/17). At `q = 19` the floor itself remains open
for the generic orbits (only harmonic/equianharmonic were solved exactly).
This census therefore measures tightness and extremal structure; it proves no
new bound.

## Census results (exhaustive; enumerator self-verified)

| q | permutations | with a 4-fiber | min sum-sigma | target q-2 | tight | extremals |
| :-: | --: | --: | :-: | :-: | :-: | --: |
| 7 | 5,040 (all) | 3,276 | 5 | 5 | **yes** | 1,764 |
| 11 | 39,916,800 (all) | 38,769,720 | 9 | 9 | **yes** | 24,200 |
| 13 | 2,000,000 (sampled) | 1,989,778 | 11 | 11 | **yes** | - |

- **No slack anywhere**: the permutation case is exactly as tight as the
  general floor - no weak-tool shortcut exists for this regime.
- q=7 sigma spectrum among 4-fiber permutations: `{5, 6, 7}` then a gap to
  `15` - a spectrum gap inside the permutation slice (the >= 15 tail is the
  near-linear degenerates).
- Degree fingerprint: **null result** - stored extremals all have Lagrange
  degree `q-2`, the generic permutation-polynomial degree; degree does not
  discriminate extremals.

## The extremal orbit theorem (exhaustive, prediction recorded pre-run)

Under the symmetry group `G = {f -> alpha f(beta x + gamma) + delta}`
(order `q^2 (q-1)^2`; preserves permutations, fibers, and sum-sigma):

- **q = 7: the 1,764 extremals form exactly ONE free G-orbit.**
  Canonical model: `f = (0 2 1)-cycle + identity on {3,4,5,6}` - a 3-cycle
  perturbation of a linear map whose **fixed-point set is the 4-fiber**.
- **q = 11: the 24,200 extremals form exactly TWO free G-orbits (12,100
  each), which MERGE under graph transposition** `f -> f^{-1}` (verified:
  each rep's inverse lands in the other orbit). So under the extended group
  `G+` the extremal is **unique up to symmetry at both fields**.
- Verified with zero escapes (every G-image of an extremal is extremal) and
  exact partition; all stabilizers trivial.

**Universal anatomy (verified over all 24,200 at q=11 and all 1,764 at
q=7):** every extremal has *exactly one 4-fiber* (`sigma = 3`) *plus exactly
`q-5` triples on `q-5` distinct slopes* (`sigma = 1` each) - forced
collisions spread maximally thin, one per slope. The remaining slopes are
maximally paired (all fibers <= 2, `N_s = (q+1)/2`-type profiles).

**Generic-only law:** at q=11, the 4-fiber positions of **all 24,200**
extremals lie in the generic cross-ratio class `{3,4,5,7,8,9}` - none
harmonic. Combined with the exact B&B fact that harmonic 4-stars are LOW at
q=11, this shows **harmonic floor-achievers exist only in non-bijective
regimes** (`N_d < q`): a sharp refinement of the regime census, which had
already shown (q=7, all functions) floor-achievers spread over every
`N_d in {4..7}`.

## Implications for the analytic route (honest)

1. **The naive two-step dies**: floor-achievers exist in every `N_d` regime,
   so "prove the permutation case, then strict inequality elsewhere" cannot
   work as stated.
2. **A classification route opens**: extremal uniqueness-mod-symmetry is an
   exact, Segre-flavored statement - *any 4-fiber permutation with
   `sum sigma = q-2` is `G+`-equivalent to the canonical model* - now with
   explicit canonical models and a universal anatomy to target. Proving the
   floor with uniqueness would be the relative-BM analogue of BM's own
   extremal classification.
3. **The spread-thin profile is the quantitative handle**: the floor is
   equivalent to "you cannot beat one collision-triple per slope across `q-5`
   slopes" - a per-slope statement that the lacunary/power-sum machinery can
   plausibly address slope-by-slope.

## Gotcha (instrument)

The v1 census used an iterative Heap's algorithm with the even/odd swap
convention **reversed** (`swap(A[c[i]], A[i])` for even `i` instead of
`swap(A[0], A[i])`), silently enumerating a multiset instead of the
permutation group - caught only because duplicate "extremal" samples are
impossible under a correct enumerator. Fixed; the run now self-verifies
`visited = distinct = q!` at `q <= 8`, and the unit check passes at
`q in {4..7}`. All published numbers are from the corrected enumerator.

## Falsifiers

`PERMUTATION_CENSUS_MISMATCH` and `EXTREMAL_ORBIT_MISMATCH`
(instrument-only): any 4-fiber permutation below `q-2` at these fields would
contradict proved results (= a bug); any non-extremal G-image or partition
failure likewise. **Both clear.**

## Interpretation Boundary

Workbench-internal extremal-structure census. No new bounds; the floor for
`q >= 19`-generic and general `q` remains open. The orbit/uniqueness facts are
exhaustive finite theorems at `q in {7, 11}` only; their generalization is a
banked conjecture ("the floor-extremal permutation is unique up to `G+` for
all odd `q >= 7`"). No Euclidean claim. Next lever if pursued: prove the
per-slope spread-thin bound, or the `G+`-uniqueness, via
Redei/power-sum identities - now with canonical models to test every step
against.
