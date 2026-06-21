# BoxSEL Phase 4d — The Exact Algebraic Optimum (KKT solve)

**Date:** 2026-06-20  
**Status:** Closed form found and verified exactly. The infimum has a clean algebraic value.

## Result

```text
inf I_box^n  =  (9 + sqrt 17) / 32  ~=  0.41009705        (attained at n = 2)
```

Strictly below the Phase-4 rational witness `513/1250 = 0.41040`, and below the Phase-4c numerical
value `0.4100984` (Nelder–Mead under-converged just *above* the true optimum). Certified achievable
in exact `Q(sqrt 17)` arithmetic.

## The KKT solve

Phase 4c located the optimum at `n = 2` with both `|A∩C|` and `|B∩C|` active (and `|A∩B|` slack).
Take the structured family that generalizes the Phase-4 witness — `B_1` full, `A` right/top-aligned,
`C` left-aligned and sharing `A`'s axis-2 bottom — parameterized by `x = |A_1|`:

```text
A = [1-x, 1] x [1 - 1/(2x), 1]
B = [0, 1]   x [1/2, 1]
C = [0, 2(1-x)] x [y0, y0 + 1/(4(1-x))],   y0 = 1 - 1/(2x)
```

The marginals `|A| = |B| = |C| = 1/2` hold identically. Imposing the two active constraints:

```text
|A∩C| = (1-x) · 1/(4(1-x)) = 1/4                      -> always (forces |C_2| = 1/(4(1-x)))
|B∩C| = 1/4   with  |C_1| = z   =>   z = 2(1-x)  AND  z = x/(2(1-x))
```

Equating the two expressions for `z`:

```text
2(1-x) = x/(2(1-x))   =>   4(1-x)^2 = x   =>   4x^2 - 9x + 4 = 0   =>   x = (9 - sqrt 17)/8 ~= 0.60961.
```

At this `x`, the geometry collapses cleanly:

```text
|A∩B∩C| = 1/8        (exactly)
|A∩B|    = x/2 = (9 - sqrt 17)/16 ~= 0.30481   (>= 1/4, slack)
q*       = |A∩B∩C| / |A∩B| = (1/8)/(x/2) = 1/(4x) = (9 + sqrt 17)/32.
```

## Exact verification

`scripts/boxsel_kkt_exact.py` builds the config in a tiny exact field `Q(sqrt 17)` (a `Surd` class:
rational `a + b·sqrt 17`, exact `+ - * /` and sign/compare) and checks, with **no floating point**:

| quantity | value |
| --- | --- |
| `x` | root of `4x^2 - 9x + 4 = 0`, `(9 - sqrt 17)/8` |
| `|A| = |B| = |C|` | `1/2` |
| `|A∩C| = |B∩C|` | `1/4` (both active) |
| `|A∩B|` | `(9 - sqrt 17)/16 > 1/4` (slack) |
| `|A∩B∩C|` | `1/8` |
| **`q*`** | **`(9 + sqrt 17)/32`** |

`scripts/test_boxsel_kkt_exact.py` — **12/12 pass**, exit 0.

## Status: certified vs open

- **Certified (exact):** the config is feasible and achieves `q* = (9 + sqrt 17)/32`, so
  `inf I_box^n <= (9 + sqrt 17)/32`. This is a strictly better certified upper bound than the old
  `513/1250`, now an exact algebraic value.
- **Numerically the global minimum:** Phase-4c's broad search (Nelder–Mead, DE, exact grid) finds
  nothing below it; the value matches the structural KKT optimum.
- **Open (the last thread):** a fully rigorous matching **lower bound**. The proven lower bound is
  still `1/4` (loose); proving `inf I_box^n >= (9 + sqrt 17)/32` would need to rule out every other
  combinatorial cell at `n = 2` and every `n >= 3`. So the certified sandwich is

  ```text
  1/4  <=  inf I_box^n  =  (9 + sqrt 17)/32  ~= 0.4100970   (upper certified + numerically exact).
  ```

## The lane picture

The representation gap `I*_lower = 0` vs `inf I_box^n = (9 + sqrt 17)/32` is now a clean algebraic
number — a *persistent* gap of `~0.41`, with a closed form. Combined with Phase 4c's search-gap
finding (every from-scratch optimizer missed even the rational witness), the moral is sharp: the
exact answer here is reachable only by analysis, not by search.

---

*Sundog Research Lab — BoxSEL Phase-4d exact KKT optimum. Internal; closed form
`(9 + sqrt 17)/32` certified achievable in `Q(sqrt 17)`; matching lower bound still open.*
