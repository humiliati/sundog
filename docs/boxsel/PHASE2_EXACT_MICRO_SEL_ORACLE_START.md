# BoxSEL Phase 2 - Exact Micro-SEL Oracle Start

**Date:** 2026-06-20  
**Status:** Phase 2 started, not cleared. This note records the first executable
oracle/corpus/alignment slice and the gates it satisfies.

## Scope

This slice implements a tiny role-free SEL exact oracle:

- concepts are conjunctions of atomic names;
- the oracle enumerates Boolean logical types over those atoms;
- each type receives a nonnegative weight;
- each SEL conditional `(D | C)[l, u]` becomes the linear constraint
  `l * weight(C) <= weight(D and C) <= u * weight(C)`;
- a query `P(D | C)` is optimized by scaling to `weight(C) = 1` and solving two
  LPs (an exact rational two-phase simplex — no floating point) for the lower and
  upper endpoints.

This is a local testbed, not a scalable SEL reasoner.

## Corpus Generator

The generator builds tiny satisfiable role-free cases from a source type model:

- choose a small atom vocabulary;
- enumerate Boolean types;
- sample nonnegative integer counts over those types;
- select conjunction conditionals with nonempty denominators;
- emit interval axioms around the source-model proportions;
- hold out a query and compute its exact interval with the LP oracle.

The frozen smoke uses `case_count=6`, `seed=240711821`, three atoms, four axioms
per case, and conjunctions up to size two. For each case, the test verifies that
the source model satisfies every generated axiom and that the exact interval
contains the source query value.

## Semantic Alignment

The finite-counting vs geometric-volume check means:

> Integer type counts and rational volumes assigned to disjoint Boolean type
> cells induce the same conditional proportions and therefore the same linear
> SEL constraints.

It is tested as **scale-invariance**: for each case the integer count model `w`,
its normalized disjoint type-volume model `w / sum(w)` (volumes summing to 1), and
an arbitrary positive rescaling `k·w` must agree on every axiom's conditional, on
satisfaction, and on the query value. The volume and rescaled models carry
**different weight tuples** than the counts, and the smoke asserts that at least one
case genuinely differs (`normalized volumes != counts`) so the check cannot pass
vacuously.

This is the alignment needed for the micro-oracle's type-enumeration semantics. It
is **not** a claim that every such type-volume model is representable as single
BoxSEL boxes — single-box realizability belongs to the later `I_box` /
representation-gap phase.

## Artifacts

- `scripts/boxsel_exact_oracle.py`
- `scripts/test_boxsel_exact_oracle.py`

## Verification

Command:

```text
python scripts/test_boxsel_exact_oracle.py
```

Result: 41/41 checks pass, exit 0.

## Smoke Gates

The Phase 2 smoke test checks:

1. agreement with the Phase 1 PMP hand cases;
2. reproduction of the anchor toy exact interval `[0.16, 0.96]`;
3. the interval-premise sharp upper example, where the exact oracle returns
   `0.75` while the paper-form PMP upper is the sound but loose `1.0`;
4. a bounded finite-model search for the as-printed Algorithm 2 composition;
5. deterministic tiny-corpus generation;
6. source-model satisfaction + source query containment for each generated case;
7. finite-counting/type-volume semantic alignment for each generated case.

The shipped Algorithm 2 target now has an explicit 3-point counterexample:

```text
U  = {1, 2, 3}
Q1 = {1, 2}
A  = {1, 3}
Q2 = {1, 2}

P(A | Q1)          = 1/2
P(Q2 | A)          = 1/2
P(Q2 | A and Q1)   = 1
true P(Q2 | Q1)    = 1
shipped upper      = q1*q2 + 1 - q2 = 3/4
```

So both printed Algorithm 2 artifacts can survive together and put the reported
upper below the true query value on a finite model. This is still not a claim
about the paper's reported metrics; that blast radius remains gated on the
authors' construction/evaluation code.

## Known Boundaries

- **Float-LP ground truth — RESOLVED 2026-06-20.** `exact_interval` now solves with an
  in-house exact rational two-phase simplex (`_lp_min_exact`, Bland's rule) over
  `Fraction`s; numpy/scipy are removed. Endpoints are certified exact —
  `interval_exact()` returns `Fraction`s (e.g. the toy is exactly `[4/25, 24/25]`, a
  thirds case `[2/9, 8/9]`), so `I*` is trustworthy ground truth with no float snapping.

## Next Work

- Add a persisted corpus manifest once the case schema is stable.
- Add explicit single-box realizability probes for very small fragments, separated
  from the type-volume alignment check.
- Package Phase 2 as a prereg/result pair before using generated held-out cases
  for downstream sampler or delineator work.

---

*Sundog Research Lab - BoxSEL Phase 2 start note. Internal executable-start
artifact; not a phase-clearance report.*
