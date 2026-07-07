# Kakeya Phase 3L - Star-Completion Anatomy + Parabola Construction (Track 1)

- Artifact id: `KAK-PHASE3L-STAR-PARABOLA-CONSTRUCTION`
- Date: 2026-07-07
- Status: internal construction-search receipt; Track 1 of the post-3K scope
  (construction instead of brute force). Extends the excess map from 5 fields
  to **10 fields (q = 5 .. 37)** in upper-bound form for seconds of compute.
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Prior receipt: [`PHASE3K_TWOLEVEL_PROBE.md`](PHASE3K_TWOLEVEL_PROBE.md)
- Scripts:
  [`../../scripts/kakeya-star-completion-anatomy.mjs`](../../scripts/kakeya-star-completion-anatomy.mjs)
  (`npm run kakeya:star-anatomy`),
  [`../../scripts/kakeya-star-parabola-construction.mjs`](../../scripts/kakeya-star-parabola-construction.mjs)
  (`npm run kakeya:star-construction`)
- Results: `results/kakeya/star-completion-anatomy/`,
  `results/kakeya/star-parabola-construction/`

## Step 1 - Anatomy of Optimal Completions (exact fields)

Recording the actual optimal completions (witness intercepts) at
`q in {5, 7, 11, 13}`, one representative 4-star per orbit; ALL optima
enumerated exhaustively at `q in {5, 7}` (16 / 96 / 480 optima):

- **The pivot always has multiplicity exactly 4** - no optimal completion
  routes a chosen line through the pivot (universal across all 592 enumerated
  optima and every witness).
- **Every other concurrency is a bare triple**: optimal sacrifice = 3 (pivot)
  + T triples with `T = q - 5` (low) / `q - 4` (high) exactly.
- **Conic signal**: at `q = 13`, the harmonic and generic optima have all 10
  chosen lines - and 12 of all 14 lines - tangent to a single **parabola**
  (max-incidence dual-conic fit). The `q = 11` and `q = 13`-equianharmonic
  witnesses read partial/non-parabola (6/8, 8/10), but optimum populations
  are large and the construction below shows parabola-form optima exist at
  13-equianharmonic.

## Step 2 - The Construction Family

> Choose an axis `a` among the 4 star directions; choose a parabola with axis
> direction `a` (the 3-parameter family `y = alpha x^2 + beta x + gamma`
> mapped by the PHASE3H axis map); complete the star with the parabola's
> tangents in the `q - 3` non-star directions (a parabola has exactly one
> tangent per non-axis direction - which is why the axis must be a star
> direction). `ex = sacrifice - (q-1)/2`, size-identity and completeness
> independently verified per winner.

Search space per orbit: `4 * q^2 (q-1)` candidates, `O(q)` sacrifice count
each - seconds per field through `q = 37`.

## Validation (solver-certified fields): 10/11 MATCH

The family achieves the exact optimum at every known field-orbit **except
q=11 harmonic** (construction 5 vs exact 4): `5:2 | 7:2,3 | 11:GAP,4 |
13:6,5,5 | 17:7,7,7`. The gap is real (exhaustive over the family) - the
q=11-harmonic optimum is not of pure parabola-tangent form, consistent with
its non-parabola anatomy. **Family-gap caveat: construction HIGH readings are
unreliable within +1; construction LOW readings are definitive refutations of
"high"** (they are verified completions at `(q-3)/2`).

## Evaluation Map (upper bounds, q = 19 .. 37)

| `q` | harmonic | equianharmonic | generic orbits |
| ---: | :---: | :---: | :--- |
| `19` | 9 H? | 9 H? | 8, 8 (L) |
| `23` | 10 (L) | - | 10, 10 (L), **11 H?** |
| `29` | 14 H? | - | 13, 13, 13, 13 (L) |
| `31` | 14 (L) | 15 H? | 14, 14, 14 (L), **15 H?** |
| `37` | 18 H? | **17 (L)** | 17 x5 (L) |

All 37 construction values across 10 fields lie in
`{(q-3)/2, (q-1)/2}` - the **two-level law extends to q = 37** in
upper-bound form (no third level ever appears).

## Pattern Scoring

- **Mod-8 harmonic rule: alive.** Definitively low at `q in {23, 31}`
  (7 mod 8) as predicted; high-consistent at `q in {29, 37}` (5 mod 8). The
  one tension - harmonic reads HIGH at `q = 19` (3 mod 8, predicted low) -
  coincides exactly with the family's only validation gap (`q = 11` harmonic,
  also 3 mod 8). Refined hypothesis: **the parabola family systematically
  misses harmonic-low at `q = 3 (mod 8)`**, where the true value is low.
- **Chi-equianharmonic pattern: strengthened, one definitive point.** Both
  roots of `l^2 - l + 1` QR -> low: `q = 13` (exact) and **q = 37
  (construction achieves 17 = low, certified)**. Roots non-QR ->
  high-consistent: `q in {7 (exact), 19, 31}`. **EQ-3 at q = 19 is NOT
  refuted** - the family fails to achieve eq-low there, consistent with
  (not proof of) high.
- **Generic uniformity: BROKEN in upper bounds.** `generic-c` at `q = 23` and
  `generic-b` at `q = 31` read high while sibling generics read low - the
  first generic splits observed (H-unreliability caveat applies; each could
  be a family gap). The generic-orbit invariant behind the split is open.

## Track 2 Sharpening

The q=19 exact B&B now has only **two** load-bearing solves: harmonic
(8 vs 9 - discriminates the mod-8 rule against the family-gap hypothesis)
and equianharmonic (8 vs 9 - settles EQ-3). The generics are certified low
from above. Staged command unchanged
([PHASE3K](PHASE3K_TWOLEVEL_PROBE.md)); ~2 solves x 5-24h with the
symmetric solver, or sharded.

## Falsifier

`CONSTRUCTION_INSTRUMENT_MISMATCH` (instrument-only): fires if any winning
candidate fails the size identity, completeness, or independent rebuild, or
if a construction value beats a solver-certified exact. **Clear** (37/37
verified). Validation gaps and evaluation outcomes are measurements.

## Interpretation Boundary

Supports only:

> In the finite-field workbench plus out-of-register sidecars (odd primes
> 5-37), a parabola-tangent family with axis in the star directions achieves
> the exact 4-star completion optimum at 10 of 11 solver-certified
> field-orbits and yields verified upper bounds elsewhere; all values lie on
> the two levels; low-side pattern confirmations are definitive,
> high-side readings are bounded by the known family gap.

Constructions cannot certify "high" - that direction needs lower bounds
(Track 2). The mod-8 and chi patterns remain post-hoc observations
(now with definitive low-side support at 23/31/37) pending the two staged
q=19 solves. `ex` values import the pinned Blokhuis-Mazzocca minimum at
`q >= 11`. Register untouched; no pins added (evaluation fields are
out-of-register). No Euclidean claim, no incidence-geometry novelty claim.
