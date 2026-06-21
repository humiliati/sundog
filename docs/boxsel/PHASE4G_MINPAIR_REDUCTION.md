# BoxSEL Phase 4g - Min-Pair Reduction

**Date:** 2026-06-20  
**Status:** Exact n=2 lower-bound reduction. Global lower bound still open.

**Update after Phase 4h:** the 35 one-help mixed-side orbits below are now exact-closed. The n=2
frontier is **10 two-help mixed** orbits. See [`PHASE4H_ONE_HELP_CLOSURE.md`](PHASE4H_ONE_HELP_CLOSURE.md).

**Update after Phase 4j:** the final 10 two-help mixed orbits are exact-closed. The n=2 atlas is
now **123/123 closed**. See
[`PHASE4J_MIXED_ENVELOPE_CLOSURE.md`](PHASE4J_MIXED_ENVELOPE_CLOSURE.md).

**Update after Phase 4k:** `n >= 3` compression is now closed, so the arbitrary-dimension lower
endpoint is exact. See [`PHASE4K_DIMENSION_COMPRESSION.md`](PHASE4K_DIMENSION_COMPRESSION.md).

## Purpose

Phase 4f mapped the n=2 endpoint-order atlas:

```text
123 total n=2 orbits
31 zero-help orbits eliminated exactly
92 live orbits remained
```

Phase 4g closes a large exact subset of those live orbits by replacing endpoint-order detail with a
smaller invariant: which pairwise overlap is minimal on each axis.

## Key Identity

For three pairwise-overlapping intervals on one axis:

```text
|A and B and C| = min(|A and B|, |A and C|, |B and C|).
```

Therefore each axis contributes one of:

```text
q_k = 1                  if |A and B| is minimal
q_k = |A and C|/|A and B| if |A and C| is minimal
q_k = |B and C|/|A and B| if |B and C| is minimal
```

with exact side choices when minima are tied.

## Exact Closure Rule

If every axis can be assigned to the same side `S in {AC, BC}` so that:

```text
q_k = |S_k| / |A_k and B_k|,
```

then:

```text
q = prod_k q_k = |S| / |A and B| >= (1/4)/(1/2) = 1/2.
```

Since:

```text
1/2 > q_KKT = (9 + sqrt 17)/32,
```

every same-side orbit is exact-closed.

## Result

The n=2 atlas now splits:

```text
31  zero-help closed       (Phase 4f, q=1)
47  same-side live closed  (Phase 4g, q>=1/2)
35  one-help mixed open
10  two-help AC-vs-BC mixed open
```

So:

```text
78 / 123 n=2 orbits exact-closed
45 / 123 n=2 orbits remain open
```

The open n=2 cases are exactly the mixed-side cases. The KKT candidate lives in the two-help
mixed family.

## Boundary

This is an exact per-orbit lower certificate for 47 formerly-live orbits. It is **not** the global
lower-bound closure:

```text
1/4 <= inf I_box^n <= (9 + sqrt 17)/32        (UNCHANGED)
```

Remaining after Phase 4g:

- exact lower certificate for the 35 one-help mixed-side n=2 orbits;
- exact lower certificate for the 10 two-help AC-vs-BC mixed n=2 orbits;
- `n >= 3` compression/no-improvement proof.

Phase 4h closes the first of these remaining items.

## Artifacts

- `scripts/boxsel_phase4g_minpair_reduction.py`
- `scripts/test_boxsel_phase4g_minpair_reduction.py`

Verification:

```text
python scripts/test_boxsel_phase4g_minpair_reduction.py
```

Result:

```text
13/13 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-4g min-pair reduction. Internal; 78/123 n=2 orbits exact-closed,
45 mixed-side n=2 orbits remained open here; later phases close n=2 and arbitrary dimension.*
