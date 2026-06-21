# BoxSEL Phase 4k - Dimension Compression

**Date:** 2026-06-21  
**Status:** Arbitrary-dimension lower bound closed. The Phase-4 Helly-seed box infimum is exact.

## Result

Phase 4j proved:

```text
inf I_box^2 = (9 + sqrt 17)/32.
```

Phase 4k proves that no `n >= 3` box embedding can improve on that value. Since the Phase-4d
witness can be padded by full axes, the exact arbitrary-dimension lower endpoint is:

```text
inf I_box^n = (9 + sqrt 17)/32       for every n >= 2.
```

The representation gap for the Helly-seed query is therefore exact:

```text
I* lower      = 0
I_box^1 lower = 1/2
I_box^n lower = (9 + sqrt 17)/32     for n >= 2.
```

## Compression Proof

For each axis, interval Helly/min-pair gives:

```text
|A_k and B_k and C_k| = min(|A_k and B_k|, |A_k and C_k|, |B_k and C_k|).
```

So every axis is:

- `AC`-help,
- `BC`-help,
- or neutral `AB`-min.

If all helping axes are assignable to one side, the same-side product rule gives:

```text
q >= |S| / |A and B| >= (1/4)/(1/2) = 1/2 > q_KKT.
```

So only mixed `AC`/`BC` cases matter.

Partition arbitrary axes into:

```text
H = AC-help axes
K = BC-help axes
N = neutral AB-min axes.
```

Let:

```text
H: x=|AB_H|, y=|AC_H|=|ABC_H|, z=|BC_H|
K: X=|AB_K|, Y=|AC_K|, Z=|BC_K|=|ABC_K|
N: p=|AB_N|=|ABC_N|, u=|AC_N|, v=|BC_N|.
```

The neutral factor cancels from the query:

```text
q = yZ/(xX).
```

Use the `A`/`B` symmetry to orient the neutral group so:

```text
rho = v/u <= 1.
```

If `q >= 1/2`, we are done. Otherwise the mixed case satisfies the same Phase-4i triangle:

```text
s = y/x >= 1/2
r = z/y
t = 4zZv

1 <= t <= r <= 2.
```

The neutral-adjusted product `P = xXv` obeys:

```text
P <= 1 / (2(1 + (r-1)s))
P <= rho(1/2 - 1/(4s)) + t/(4rs).
```

Because `s >= 1/2` and `rho <= 1`, the second neutral-adjusted bound is no larger than the pure
Phase-4i mixed bound:

```text
P <= 1/2 - 1/(4s) + t/(4rs).
```

Therefore every arbitrary-dimensional mixed case compresses to the Phase-4i envelope:

```text
rP/t <= P_phase4i(r,t).
```

Phase 4j already proved:

```text
P_phase4i(r,t) <= X_OPT = (9 - sqrt 17)/8.
```

So:

```text
q = t/(4rP) >= 1/(4X_OPT) = (9 + sqrt 17)/32.
```

## What Is Now Closed

The Phase-4 extremal query lower endpoint is no longer a sandwich:

```text
inf I_box^n = (9 + sqrt 17)/32       for every n >= 2.
```

The only remaining Phase-4 distinction is dimensional:

```text
n = 1  : lower endpoint = 1/2
n >= 2 : lower endpoint = (9 + sqrt 17)/32.
```

The lane can now move from extremal optimization into the intended Sundog question: taxonomy and
trace-based false-closure detection.

## Artifacts

- `scripts/boxsel_phase4k_dimension_compression.py`
- `scripts/test_boxsel_phase4k_dimension_compression.py`

Verification:

```text
python scripts/test_boxsel_phase4k_dimension_compression.py
```

Result:

```text
17/17 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-4k dimension compression. Internal; arbitrary-dimension lower
endpoint closed for the Helly-seed query.*
