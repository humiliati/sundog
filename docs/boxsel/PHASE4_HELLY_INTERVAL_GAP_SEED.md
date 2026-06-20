# BoxSEL Phase 4 - Helly Interval-Gap Seed

**Date:** 2026-06-20  
**Status:** Phase-4 seed landed, not a full extremal-optimizer phase clearance.

## Purpose

Convert the Phase-2 single-box realizability obstruction into a measured query
interval gap:

```text
I_box^1 strictly inside I*
```

This is deliberately one-dimensional and analytic. It is the seed for the later
extremal-query optimizer, not the optimizer itself.

## Ontology

Atoms: `A, B, C`. Query:

```text
q = P(C | A and B)
```

Ontology:

```text
P(A | TOP) = 1/2
P(B | TOP) = 1/2
P(C | TOP) = 1/2
P(A and B | TOP) >= 1/4
P(A and C | TOP) >= 1/4
P(B and C | TOP) >= 1/4
```

## Exact Oracle Result

The exact type-volume oracle gives:

```text
I* = [0, 1]
```

Lower witness:

```text
outside = 1/4
AB-only = 1/4
AC-only = 1/4
BC-only = 1/4
ABC = 0
```

Upper witness:

```text
outside = 1/2
ABC = 1/2
```

## Single-Box Result

For one-dimensional single boxes, each atom is an interval of length `1/2`.
The pairwise-overlap constraints imply each pair of interval centers is within
`1/4`. Therefore the three centers have diameter at most `1/4`, and the triple
overlap has length at least:

```text
1/2 - 1/4 = 1/4
```

Since `|A and B| <= |A| = 1/2`,

```text
P(C | A and B) = |A and B and C| / |A and B| >= (1/4) / (1/2) = 1/2
```

The lower bound is tight:

```text
A = [0, 1/2]
B = [0, 1/2]
C = [1/4, 3/4]
q = 1/2
```

The upper endpoint is also tight:

```text
A = B = C = [0, 1/2]
q = 1
```

So:

```text
I_box^1 = [1/2, 1]
I*      = [0, 1]
```

The `q >= 1/2` infimum is both proven analytically (above) and corroborated by an exact
battery over all satisfying 1-D embeddings (65 configs): the analytic bound is verified
sound on each, every `q >= 1/2`, and the minimum reached is exactly `1/2` (tight at the
lower witness). The `I*` lower-bound model (pairwise `1/4`, triple `0`) is precisely the
Phase-2 Helly-forbidden Venn — which is *why* one-dimensional boxes cannot reach `0`.

## Boundary

This is a measured query-interval gap for **one-dimensional single boxes**:

```text
I_box^1 lower - I* lower = 1/2
```

It does not claim the same quantitative lower bound for arbitrary embedding
dimension. Higher-dimensional `I_box^n` needs the actual extremal-query optimizer
or a separate analytic bound.

## Higher-Dimensional Seed

The 1-D lower endpoint does **not** persist unchanged. A certified 2-D rational
box construction satisfies the same ontology with:

```text
q = P(C | A and B) = 513/1250 < 1/2
```

The construction:

```text
x = 25/41
z = 32/41
y0 = 1 - 1/(2x) = 9/50

A = [1-x, 1] x [y0, 1]
B = [0, 1]   x [1/2, 1]
C = [0, z]   x [y0, y0 + 1/(2z)]
```

All three atom areas are `1/2`, and the overlaps are:

```text
|A and B|       = 25/82
|A and C|       = 1/4
|B and C|       = 513/2050
|A and B and C| = 513/4100
q               = 513/1250
```

Appending full `[0,1]` axes embeds the same witness into every `n >= 2`
without changing any volume ratio. Therefore:

```text
inf I_box^n lower <= 513/1250 for every n >= 2
```

This is a certified upper bound on the higher-dimensional box lower endpoint,
not a proof of the exact infimum. The next optimizer/proof target is to determine
whether `inf I_box^n` equals this value, falls further, or tends toward the exact
oracle lower endpoint `0` as dimension grows.

## Artifacts

- `scripts/boxsel_phase4_interval_gap.py`
- `scripts/test_boxsel_phase4_interval_gap.py`

Verification:

```text
python scripts/test_boxsel_phase4_interval_gap.py
```

Result: 17/17 checks pass, exit 0.

After the higher-dimensional seed extension:

```text
Result: 25/25 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-4 interval-gap seed. Internal seed artifact;
not a phase-clearance report.*
