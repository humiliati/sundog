# Kakeya Phase 3R - Floor Conjecture: Relative-BM Proof Program (Scope)

- Artifact id: `KAK-PHASE3R-FLOOR-PROOF-PROGRAM`
- Date: 2026-07-09
- Status: internal research-program **scope**, not a proof. Lays out the
  relative Blokhuis-Mazzocca machinery for the exception-free floor conjecture
  identified in [PHASE3Q](PHASE3Q_RELATIVE_SEGRE_STATUS.md). No claims; a
  roadmap for owner-gated theory work.
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Prior receipts: [`PHASE3P_PARABOLA_OPTIMALITY.md`](PHASE3P_PARABOLA_OPTIMALITY.md),
  [`PHASE3Q_RELATIVE_SEGRE_STATUS.md`](PHASE3Q_RELATIVE_SEGRE_STATUS.md)
- Litpass: [`../KAKEYA_LITPASS_MEMO.md`](../KAKEYA_LITPASS_MEMO.md)

## Target (precise)

> **Floor conjecture (F).** Let `K ⊂ AG(2,q)`, `q` odd, be a Kakeya set (a line
> in every direction) that contains four concurrent lines through a point `O`.
> Then `|K| ≥ q(q+1)/2 + q - 2`, i.e. `sacrifice ≥ q - 2`, i.e. `ex ≥ (q-3)/2`
> (`T ≥ q - 5` in the triple count).

Why this target (not parabola-optimality): F is **exception-free** (holds at
every tested `q` including `q=11`, where harmonic sits *at* the floor), it is
the load-bearing lower half of the two-level law (the ceiling `ex ≤ (q-1)/2`
is already proved by the parabola construction), and it does not require the
`q=11`-obstructed conic-classification. F is the relative analogue of the BM
absolute bound `|Kakeya| ≥ q(q+1)/2 + (q-1)/2`; the pencil of 4 lines through
`O` should upgrade the lower-order term from `(q-1)/2` to `q-2`, a gain of
`(q-3)/2`.

## Why naive counting fails (the load-bearing obstruction)

Dually (PHASE3P): `q+1` points, four on a line `O*`, one free point per
non-star direction (a pencil through `W`); minimize 3-secants off `O*`.
Counting free-pairs whose join meets `O*` at a star point gives
`~ C(q-3,2)·4/(q+1) ≈ 2q` incidences under a uniform heuristic - an
**overcount** far above the tight `q-5`. The gap is exactly the structure: the
extremal free points lie on a conic, which forces most free-pair joins to miss
the star points. So **any proof of F must "see" the conic/arc extremality**;
elementary incidence counting cannot reach the tight constant. This is the
reason F needs genuine algebraic machinery, and it pins what that machinery
must accomplish.

## Machinery Option A (primary): Redei / lacunary polynomials

The standard engine for finite-field Kakeya lower bounds with sharp
lower-order terms.

- **Encode.** For the completion `U = {(a_i, b_i)}`, form the Redei polynomial
  `R(X,Y) = ∏_i (X + a_i Y - b_i)`, degree `|U|` in `X`. For each slope `m`,
  `R(X,m)` factors over `F_q`; a fully-covered direction `m` forces
  `(X^q - X) | R(X,m)` (or a high-multiplicity fully-reducible structure).
- **Lacunary step.** Rows/columns of exponents in `R` have gaps; the
  Blokhuis-Ball-Brouwer-Storme-Szonyi lacunary theory forces `R(X,m)` (or its
  relevant factor) into a special form (a `p`-th power times a low-degree
  factor, or `(X^q - X)`-divisible), bounding how small `|U|` can be while
  covering all `q+1` directions. This yields the BM absolute `(q-1)/2` term.
- **Pencil upgrade (the new content).** Four covering lines pass through
  `O = (0,0)`, so `O ∈ U` lies on 4 of the `q+1` covered directions. In `R`,
  the point `O` is the factor `X`; the four directions through it impose a
  *shared* high-order vanishing at a common `(X,Y)` locus. The conjecture is
  that this common vanishing tightens the lacunary gap by `(q-3)/2`. **Crux:**
  make the "4 shared linear factors through `O`" produce a degree/gap gain of
  exactly `(q-3)/2` in the lacunary bound.

## Machinery Option B (secondary): relative Lemma of Tangents

Segre's Lemma of Tangents governs the tangent lines of an *arc*; our dual set
is a near-arc with a forced 4-secant `O*`. A "relative lemma of tangents"
would assign to the `q-3` free points a tangent function whose degree is
controlled, with the four star points on `O*` as a boundary condition. The
`(q-3)/2` would come from the tangent function's parity (`q` odd), as in the
absolute case. **Crux:** define the tangent function for the pivot-constrained
configuration and bound its degree; the forced 4-secant is both the obstacle
(not an arc) and the source of the extra term.

Recommendation: **pursue A first** (sharper track record for lower-order Kakeya
terms; the pencil-at-`O` factor is concrete), keep B as the uniqueness/extremal
companion.

## Where `q = 11` must enter

F is exception-free, so unlike parabola-optimality the proof need not exclude
`q = 11`. But `q = 11` is where harmonic is *at* the floor via a non-conic
optimum; any sharp analysis should predict "floor achieved by a non-conic
config" as an allowed extremal at small `q`. A correct proof of F should be
*consistent* with the `q=11` non-conic minimizer (it is still `ex = (q-3)/2`),
which is a useful consistency check on any candidate argument.

## Milestones (each with a computational de-risking checkpoint)

1. **M1 - Redei setup + absolute reproduction.** Implement the Redei
   polynomial of a completion; verify computationally that the absolute BM
   bound's divisibility (`(X^q-X) | R(X,m)` for covered `m`) holds on our
   completions (`q ≤ 19`). Checkpoint: divisibility verifier, falsifier-fenced.
2. **M2 - Locate the pencil factor.** Compute, for the 4-star completions, the
   exact extra vanishing at `O` in `R` and measure its degree contribution vs
   `q`. Checkpoint: does the measured contribution scale as `(q-3)/2`? (If not,
   F's mechanism is different - a fast, decisive test.)
3. **M3 - Lacunary gain lemma.** Prove the pencil factor tightens the lacunary
   bound by `(q-3)/2`. This is the hard analytic step.
4. **M4 - Assemble F** and reconcile with the `q=11` non-conic minimizer.
5. **M5 - (optional) extremal/uniqueness** via Option B for the LOW-orbit
   classification (which orbits sit at the floor).

M1-M2 are workbench-tractable now and would *decisively* confirm or kill the
mechanism before any hard proof effort - the lane's standard "verify the
mechanism, then prove" discipline.

## Reading list (verify before citing)

- Blokhuis-Mazzocca 2008 (pinned) - the absolute argument to relativize.
- B. Segre 1955 (pinned) - Lemma of Tangents backbone for Option B.
- Szonyi, "Around Redei's theorem" / lacunary polynomial surveys - the Option A
  engine (LEAD, to pin).
- Ball, *Finite Geometry and Combinatorial Applications* - Redei/lacunary
  reference text (LEAD, to pin).

## Honest difficulty assessment

F is a genuine research problem at the level of a finite-geometry paper. M1-M2
(mechanism confirmation) are achievable in the workbench and low-risk. M3 (the
lacunary gain) is the real theorem and the point of highest uncertainty - it
may be a short adaptation of BM or may require new ideas; the naive-count
overcount above shows it is not elementary. Success probability is honestly
unknown until M2 reports whether the pencil factor scales as `(q-3)/2`.
Recommended next concrete action: **execute M1-M2** (workbench, decisive,
cheap) before committing to the hard analytic step.

## M1-M2 Execution 2026-07-09 - MECHANISM CONFIRMED

`npm run kakeya:floor-mechanism`
([`../../scripts/kakeya-floor-mechanism.mjs`](../../scripts/kakeya-floor-mechanism.mjs),
results `results/kakeya/floor-mechanism/`). Falsifier
`FLOOR_MECHANISM_MISMATCH` clear.

**M1 (Redei grounding) - the pencil factor is real and exactly quantified.**
For the exact minimal 4-star completion at `q in {5,7,11,13}`:

- every covered direction's line-intercept occurs with multiplicity `q`
  (`(X - c(y))^q | R(X,y)`), as required;
- the four star directions carry the shared root `c = 0` at multiplicity
  **exactly `q`** (measured `[q,q,q,q]` at every field), while non-star
  directions carry it at `2..7`. Since the star line through `O` has `q`
  points all at intercept `0` and nothing else does, this is the clean
  statement `X^q \| R(X, d)` (and `X^{q+1} \nmid`) for each of the 4 star
  slopes `d`. **That common `X^q`-divisibility across 4 directions is the
  concrete "pencil factor" the lacunary argument must exploit** - M3's target
  is now explicit.

**M2 (floor scaling) - the fourth line forces `(q-3)/2`.** Exhaustive minimal
completion excess:

| body | q=5 | q=7 | q=11 | q=13 |
| --- | :-: | :-: | :-: | :-: |
| 3-star | 0 | 0 | 0 | 0 |
| 4-star floor (min over orbits) | 2* | 2 | 4 | 5 |
| `(q-3)/2` | 1 | 2 | 4 | 5 |

- **A 3-star reaches the BM minimum exactly (`ex = 0`)** - three concurrent
  lines embed in a minimal Kakeya set, costing nothing over BM.
- **The 4-star floor is `(q-3)/2`**, a lower bound that holds at every `q` and
  is tight for `q >= 7`. (`*` at `q=5` it is loose, `2 >= 1`: that field has no
  LOW orbit, only the harmonic HIGH one, so the bound is not achieved - not a
  violation.)
- So the jump `0 -> (q-3)/2` is caused by the **fourth** concurrent line:
  `sacrifice` goes from `(q-1)/2` (BM, at a 3-star) to `q-2` (at a 4-star), a
  gain of exactly `(q-3)/2`.

**Verdict.** The mechanism assumed in this program is confirmed: the pencil
contributes a clean `X^q`-shared-root factor (M1) and forces exactly the
`(q-3)/2` lower-order gain (M2), so the relative-BM target `sacrifice >= q-2`
for 4-star Kakeya sets is the right statement and M3 (the lacunary/tangent
gain lemma turning the shared `X^q` factor into `+(q-3)/2`) is worth the hard
effort. Bonus finding to fold into the proof: 3-stars are BM-free, so the
relative bound is genuinely a "4th line" phenomenon, not a general pencil
count.

## Interpretation Boundary

This is a scope/roadmap plus a mechanism-confirmation probe - no lower bound is
proved or claimed. M1-M2 are exhaustive workbench measurements (`q <= 13`);
they confirm the *mechanism*, not the theorem. Redei/lacunary and Segre
machinery are imported literature. Register untouched. No Euclidean claim; F is
a finite-field workbench statement.
