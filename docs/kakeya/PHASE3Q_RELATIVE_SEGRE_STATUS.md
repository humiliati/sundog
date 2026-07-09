# Kakeya Phase 3Q - Relative-Segre Lemma: Honest Status (Not Closed)

- Artifact id: `KAK-PHASE3Q-PARABOLA-OPTIMALITY-VERIFY`
- Date: 2026-07-09
- Status: internal receipt. **The relative-Segre / parabola-optimality lemma
  does NOT close as a clean theorem.** Proven exhaustively for finite cases;
  FALSE at `q=11` harmonic; general `q>=13` genuinely OPEN (not in the
  literature searched, Segre route obstructed). This is the honest outcome of
  the attempt, not a proof.
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Prior receipt:
  [`PHASE3P_PARABOLA_OPTIMALITY.md`](PHASE3P_PARABOLA_OPTIMALITY.md)
- Script:
  [`../../scripts/kakeya-parabola-optimality-verify.mjs`](../../scripts/kakeya-parabola-optimality-verify.mjs)
  (`npm run kakeya:parabola-opt`)
- Results: `results/kakeya/parabola-optimality-verify/`

## What was attempted

Close the lemma from [PHASE3P](PHASE3P_PARABOLA_OPTIMALITY.md): *the
minimum-size completion of a 4-star is a conic (parabola) completion, for
`q >= 13`.* The honest result is a mix of one proof, one disproof, and one
open problem.

## Result 1 - Finite theorem (PROVED, exhaustive)

For `q in {5,7,11,13}` both quantities below are exact minima - `exactMin` by
full branch-and-bound over ALL completions, `parabMin` by exhaustive search of
the parabola family - recomputed from scratch in this receipt:

| `q` | orbit | exact ex | parabola ex | parabola-optimal? |
| ---: | --- | :-: | :-: | --- |
| 5 | harmonic | 2 | 2 | yes |
| 7 | harmonic / equianharmonic | 2 / 3 | 2 / 3 | yes |
| 11 | generic | 4 | 4 | yes |
| 11 | **harmonic** | **4** | **5** | **NO** |
| 13 | harmonic / generic / equianharmonic | 6 / 5 / 5 | = | yes |

Prior exact B&B (PHASE3K/3M) extends "yes" to all orbits at `q=17` and to
harmonic + equianharmonic at `q=19`. So parabola-optimality is a **rigorous
theorem for every exactly-solved 4-star with `q <= 19`, with the single
exception `q=11` harmonic.**

## Result 2 - The universal form is FALSE (disproof)

At `q=11` harmonic the exact minimum completion has `ex = 4` (LOW) but the best
conic completion only reaches `ex = 5` (HIGH): a non-conic completion is
strictly smaller. So "the optimum is always a conic completion" is **false as
stated**. Any correct statement must exclude `q=11` harmonic, i.e. must use a
hypothesis (like `q >= 13`) that a Segre-type argument does not naturally
provide (Segre holds for all odd `q`). This both explains the anomaly and
rules out the easy proof routes.

## Result 3 - General `q >= 13`: OPEN

- **Not in the literature searched.** The pencil/relative case is not the
  Ball-Blokhuis-Domenzain "finite Kakeya" result (arXiv:1503.06639), which is a
  different generalization (`N`-rich lines, `n >= 4`). Segre (1955) and
  Blokhuis-Mazzocca (2008) are the *absolute* backbones; neither covers the
  relative near-minimal, pivot-constrained statement (a 4-star with `ex > 0`
  never reaches the absolute BM minimum, so BM's classification does not
  apply).
- **Segre's Lemma of Tangents is obstructed.** That lemma is the engine of
  "extremal arc = conic", but it applies to *ovals/arcs* (no 3 collinear). Our
  dual configuration has `O*` as a **4-secant** (4 star points collinear) - it
  is deliberately not an arc - so the lemma does not apply off the shelf. The
  relative version with a forced 4-secant is the genuine open content.
- **Truth beyond `q=19` is untested.** Exact data stops at `q=19`; `q >= 23`
  exact B&B is infeasible (depth `>= 20`). The `q=11` failure shows the
  pattern *can* break, so even the conjecture "holds for all `q >= 13`" is
  supported only by 5 exact fields (`13,17,19` fully; `5,7` trivially) and is
  not obviously true.

## The cleaner target (reframing)

Parabola-optimality is the *messy* statement (one exception, open general
truth). The **exception-free companion** is the two-level floor:

> **Floor conjecture.** Every 4-star completion has `ex >= (q-3)/2`
> (equivalently `T >= q-5`; a Kakeya set containing 4 concurrent lines has
> `>= q(q+1)/2 + q - 2` points).

This holds at **every** tested `q` including `q=11` (where harmonic sits *at*
the floor, `ex = (q-3)/2 = 4`). It is a relative Blokhuis-Mazzocca lower bound
- also open, but clean (no exception) and the genuinely useful half of the
level law. Recommended as the target for future theory work in preference to
the exceptional parabola-optimality statement.

## Falsifier

`PARABOLA_OPT_FINITE_MISMATCH` (instrument): fires if any orbit other than
`q=11` harmonic is parabola-suboptimal on the exhaustively-solved fields, or if
a parabola beats the exact minimum. **Clear.**

## Honest verdict

The relative-Segre lemma is **not closed.** Delivered instead: a rigorous
finite theorem (`q <= 19` exact, one exception), a disproof of the universal
form (`q=11` harmonic), a precise localization of why the general case resists
(non-arc configuration, forced 4-secant, absent from the literature), and a
cleaner exception-free target (the floor conjecture). No proof of the general
lemma is claimed; Segre and BM are imported, not extended.

## Interpretation Boundary

Supports only: parabola-optimality is an exhaustively-verified theorem for
`q <= 19` 4-stars except `q=11` harmonic, and open in general. `ex` imports the
pinned BM minimum. Register untouched, no pins. No Euclidean claim, no
incidence-geometry novelty claim, no reproving of Segre/BM.
