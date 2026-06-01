# Kakeya Phase 2 - Tiny Finite-Field Workbench Spec

- Artifact id: `KAK-PHASE2-TINY-FINITE-FIELD-WORKBENCH-SPEC`
- Date: 2026-06-01
- Status: pre-implementation spec. No public page, executable probe, or
  `site-pages.json` entry is live.
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Front-A source:
  [`../KAKEYA_FINITE_FIELD_READER.md`](../KAKEYA_FINITE_FIELD_READER.md)
- Lit-pass: [`../KAKEYA_LITPASS_MEMO.md`](../KAKEYA_LITPASS_MEMO.md)

> Click the points. The direction shadow tells you what is missing.

## 1. Purpose

This spec freezes the tiny finite-field workbench before implementation. The
workbench is an educational, finite-field-only toy for the known Dvir theorem.
It must make direction-completeness visible without pretending to discover,
test, or support any open Kakeya result.

The workbench's useful question is narrow:

> Given a selected point set `K` in a small prime plane `F_q^2`, what exact
> direction-coverage shadow does it cast?

The workbench does not ask whether the Euclidean Kakeya conjecture is true. It
does not estimate Hausdorff or Minkowski dimension. It does not touch the Kakeya
maximal-function conjecture. It does not make a regime-2/control-sufficiency
claim.

## 2. Domain Lock

Primary domain:

- Field: prime field `F_q = Z/qZ`.
- Default field size: `q = 7`.
- Allowed UI field sizes in Phase 2 implementation: `q in {5, 7, 11}`.
- Dimension: `n = 2` only.
- Body: a point subset `K subset F_q^2`.
- Known theorem register: Dvir finite-field Kakeya; for this workbench, use the
  degree `<= q - 1` proof floor `binom(q + 1, 2)` in the plane as the displayed
  Dvir lower-bound card.

Out of scope:

- prime powers `q = p^k` with `k > 1`;
- `F_q^3` and higher-dimensional views;
- Euclidean tube, pixel, needle, or maximal-function views;
- stochastic or empirical claims about the theorem;
- any optimized search for extremal Kakeya sets.

Any expansion beyond `F_q^2` prime fields requires a new spec amendment.

## 3. Direction And Line Convention

Directions are projective directions in `F_q^2`, represented canonically as:

- finite slopes: `m in F_q`, represented by vector `(1, m)`;
- vertical direction: `inf`, represented by vector `(0, 1)`.

There are exactly `q + 1` directions.

Lines are represented by direction plus intercept:

- finite slope `m`, intercept `b`: `L(m, b) = {(x, m*x + b) : x in F_q}`;
- vertical intercept `b`: `L(inf, b) = {(b, y) : y in F_q}`.

Each direction has exactly `q` parallel lines, each line has exactly `q` points,
and the `q` lines in one direction partition `F_q^2`.

Canonical point order for bitsets and exports:

```text
point_index(x, y) = y*q + x
```

where `x, y in {0, ..., q - 1}`.

Canonical direction order:

```text
0, 1, ..., q - 1, inf
```

## 4. Registered Shadow

The primary displayed shadow is the direction-coverage bitset:

```text
S(K)[d] = 1 iff there exists an intercept b such that L(d, b) subset K.
```

Displayed primary fields:

- `q`;
- `|K|`;
- `body_fraction = |K| / q^2`;
- `directions_covered = sum_d S(K)[d]`;
- `coverage_fraction = directions_covered / (q + 1)`;
- missing direction labels;
- Dvir plane lower-bound card: `|K| >= binom(q + 1, 2)` if all directions are
  covered;
- verdict:
  - `complete finite-field Kakeya set` if all `q + 1` directions are covered;
  - `near miss` if at least one but not all directions are covered;
  - `empty shadow` if none are covered.

Displayed secondary teaching fields, allowed only behind deliberate UI toggles:

- one witness line for one selected covered direction;
- line-count histogram by direction: number of full lines in `K` for each
  direction.

Forbidden as primary shadow:

- full selected point list;
- all covered line intercepts by default;
- a generated minimal witness union advertised as "the body";
- Euclidean-looking area, dimension, tube thickness, or visual density metrics.

## 5. `KAK-SHADOW-REENCODING` Guard

The workbench passes the shadow-reencoding guard only if the primary shadow is
many-to-one.

Required implementation checks:

1. **Single-direction collision.** For any `q >= 5`, two distinct parallel lines
   of the same direction must produce the same primary shadow bitset: only that
   direction is covered.
2. **Complete-shadow collision.** `F_q^2` and `F_q^2` minus one point must both
   produce the complete primary shadow for `q >= 5`.
3. **Export discipline.** A "shadow export" may include only `q`, direction
   order, coverage bitset, and aggregate metrics. It must not include selected
   points, witness intercepts, or all line masks.
4. **Witness discipline.** A witness-line reveal may show one covered line for a
   user-selected direction, but the UI must not auto-reveal witnesses for every
   direction in the main view.

If any implementation uses all witness lines or point membership as the
registered signature, file `KAK-SHADOW-REENCODING` and do not promote.

## 6. Baselines

The workbench should include baselines as teaching controls, not as research
claims.

### Required Baselines

1. **Empty set.**
   - Body size: `0`.
   - Expected shadow: no covered directions.

2. **Single line.**
   - Body size: `q`.
   - Expected shadow: exactly one covered direction.

3. **Whole plane.**
   - Body size: `q^2`.
   - Expected shadow: all directions covered.

4. **Whole plane minus one point.**
   - Body size: `q^2 - 1`.
   - Expected shadow: all directions covered for `q >= 3`.
   - Purpose: demonstrate that the complete direction-shadow does not
     reconstruct the body.

5. **Random point subset.**
   - Parameter: body size `s`, sampled uniformly from subsets of size `s`.
   - Display: average coverage over a small deterministic seed set.
   - Purpose: show that point count alone is not the same as direction
     completeness.

6. **Random line-cover construction.**
   - Choose one intercept independently for each direction and take the union of
     those `q + 1` lines.
   - Expected shadow: all directions covered by construction.
   - Purpose: show a simple complete body without claiming minimality.

7. **Greedy line-cover construction.**
   - Process directions in canonical order.
   - For the current direction, add the intercept line that minimizes the number
     of newly added points; tie-break by smallest intercept.
   - Expected shadow: all directions covered by construction.
   - Purpose: provide a deterministic "try to overlap lines" comparison without
     claiming an extremal Kakeya construction.

### Forbidden Baseline Language

Do not label random or greedy line-cover outputs as:

- "near optimal";
- "smallest";
- "best known";
- "evidence for Dvir's lower bound";
- "Euclidean analogue".

They are visual controls only.

## 7. UI Contract

The Phase 2 implementation target is a single internal workbench surface, not a
public page.

Required controls:

- field-size selector: `q = 5`, `7`, `11`;
- point toggle grid for `F_q^2`;
- clear, fill, random subset, and baseline preset commands;
- direction list in canonical order with covered/missing state;
- Dvir lower-bound card shown only as known theorem context;
- one-direction witness toggle, off by default;
- shadow export button for the primary shadow only.

Required copy constraints:

- The surface must say "finite field" wherever a theorem claim appears.
- The surface must say the workbench is a toy around known mathematics.
- The surface must separate "selected points" from "direction shadow".
- The surface must avoid "dimension", "area", "measure", and "Euclidean" in
  metric labels unless in the boundary warning.

Required boundary warning:

> This is a finite-field toy around known mathematics. It is not Euclidean
> Kakeya evidence, not a maximal-function result, and not a regime-2 claim.

## 8. Exact Algorithms

All computations must be exact modular arithmetic over integers modulo prime
`q`.

Precompute line masks:

```text
for m in 0..q-1:
  for b in 0..q-1:
    L(m,b) = {(x, (m*x + b) mod q) : x in 0..q-1}

for b in 0..q-1:
  L(inf,b) = {(b, y) : y in 0..q-1}
```

Coverage:

```text
covered(d, K) = any line_mask(d,b) subset K for b in 0..q-1
S(K) = [covered(d,K) for d in canonical_direction_order]
```

Randomness:

- Use a fixed, local deterministic PRNG if random subsets or random line covers
  are exposed.
- Display the seed.
- Random outputs are examples, not evidence.

Complexity:

- `q <= 11`, `q^2 <= 121`, directions `<= 12`, line masks `<= 132`.
- Brute-force subset checks are acceptable and preferred for clarity.

## 9. Acceptance Tests

Before Phase 3 implementation can be promoted, the workbench must pass these
tests:

1. **Line cardinality.** Every precomputed line has exactly `q` points.
2. **Parallel partition.** For each direction, the `q` intercept lines are
   disjoint and their union is `F_q^2`.
3. **Nonparallel intersection.** Any two lines with different directions
   intersect in exactly one point.
4. **Empty set.** Empty set covers zero directions.
5. **Single-line set.** A single line covers exactly its own direction.
6. **Whole plane.** Whole plane covers all `q + 1` directions.
7. **Whole plane minus one.** Whole plane minus any one point covers all
   directions for `q in {5, 7, 11}`.
8. **Shadow collision.** At least two distinct bodies produce the same primary
   shadow for each supported `q`.
9. **Greedy baseline completeness.** Greedy line-cover construction covers all
   directions for each supported `q`.
10. **Export guard.** Shadow export contains no point-membership list and no
    witness intercept list.

Failure of tests 8 or 10 is `KAK-SHADOW-REENCODING`.

## 10. Falsifiers And Named Nulls

- `KAK-SHADOW-REENCODING`: the displayed or exported shadow reconstructs, or
  nearly reconstructs, the selected point set.
- `KAK-WORKBENCH-MISCALIBRATED`: the UI teaches pixel density, area, or visual
  overlap as if it were a finite-field theorem or Euclidean dimension claim.
- `KAK-FINITE-FIELD-LAUNDERING`: the workbench implies finite-field behavior is
  evidence for Euclidean Kakeya.
- `KAK-3D-LAUNDERING`: the workbench implies Wang-Zahl's `R^3` theorem supports
  open `n >= 4`.
- `KAK-BRIDGE-OVERCLAIM`: the workbench advertises Kakeya as
  regime-2/control-sufficiency rather than body-resistance only.
- `KAK-EXTREMALITY-OVERCLAIM`: random or greedy presets are described as
  optimal or near-optimal without a cited theorem and a separate proof note.

Named nulls are acceptable. If the workbench is a clear teaching toy but adds no
research instrument, file that as the result and keep the surface educational.

## 11. Phase 3 Exit Criteria

The future Phase 3 implementation may begin only after this spec is linked from
the Kakeya ledger and reader.

Phase 3 may be considered internally complete only when:

1. all acceptance tests pass for `q in {5, 7, 11}`;
2. the primary shadow export passes the no-reencoding guard;
3. the UI includes the required finite-field boundary warning;
4. baseline labels avoid extremality and Euclidean language;
5. a screenshot or local QA note records that the grid, direction list, theorem
   card, and warning are visible without overlap at desktop and mobile widths.

Public promotion remains separately blocked on:

- external incidence/combinatorics sanity review;
- `kakeya.html` page copy audit against the reader fences;
- Bucket 1 SEO/social readiness if a page is added to `site-pages.json`.

An unlinked live `kakeya.html` may be used as a review surface before this gate
clears only if it carries a visible `NOT PEER REVIEWED` banner and has no obvious
public inbound links.

## 12. Implementation Notes

Preferred implementation shape, if promoted:

- static browser module, no server dependency;
- precomputed masks generated in JavaScript at page load;
- no external finite-field package for prime fields;
- no persistent storage beyond optional local UI state;
- no network calls.

Suggested internal file targets, not authorized by this spec alone:

- `kakeya.html`;
- `public/js/kakeya-workbench.js`;
- optional local test script under `scripts/` if acceptance tests are easier
  outside the browser.

This spec authorizes the next design step. It does not authorize public launch.
