# BoxSEL Phase 4e - Lower-Bound Closure Start

**Date:** 2026-06-20  
**Status:** Started. Global lower-bound closure is **not** complete.

## Purpose

Phase 4d gave an exact achievable value:

```text
q_KKT = (9 + sqrt 17) / 32 ~= 0.41009705.
```

Before Phase 4e, the certified global statement is:

```text
1/4 <= inf I_box^n <= q_KKT      for n >= 2.
```

The lower-bound closure task is to prove the missing inequality:

```text
inf I_box^n >= q_KKT.
```

This note starts that proof by closing one local loophole: inside the structured 2-D family that
generated the KKT witness, `q_KKT` is not merely stationary. It is the exact minimum.

## Structured-Family Certificate

Use the Phase-4D normal form:

```text
A = [1-x, 1] x [1 - 1/(2x), 1]
B = [0, 1]   x [1/2, 1]
C = [0, z]   x [1 - 1/(2x), 1 - 1/(2x) + 1/(2z)]
```

The marginals `|A|=|B|=|C|=1/2` hold by construction. The active seed constraints reduce to:

```text
z >= 2(1-x)          from |A and C| >= 1/4
z <= x/(2(1-x))     from |B and C| >= 1/4
```

So feasibility requires:

```text
2(1-x) <= x/(2(1-x))
4x^2 - 9x + 4 <= 0
x >= x* = (9 - sqrt 17)/8.
```

The query in this family is:

```text
q(x,z) = 2(x+z-1)(1/2 - 1/(2x) + 1/(2z)) / x.
```

For fixed `x`, `q(x,z)` is concave in `z`:

```text
d^2q/dz^2 = -2(1-x)/(x z^3) < 0.
```

Therefore its minimum over the feasible `z` interval lies at an endpoint. Both active-constraint
endpoints give the same boundary value:

```text
q0(x) = (-2x^2 + 5x - 2) / (2x^2).
```

And:

```text
q0'(x) = (4 - 5x)/(2x^3),
```

so `q0` has no interior minimum below its endpoints on the feasible interval. Since:

```text
q0(x*) = (9 + sqrt 17)/32
q0(1)  = 1/2,
```

the structured family satisfies:

```text
q(x,z) >= (9 + sqrt 17)/32.
```

Equality occurs exactly at the Phase-4D KKT witness, where both `|A and C|` and `|B and C|` are
active.

## What This Proves

Certified:

- The KKT value is a true minimum in the structured 2-D normal form, not just a local stationary
  point.
- The claim boundary is repaired: the global theorem is still a sandwich, not an equality.
- The exact implementation checks the algebra in `Q(sqrt 17)` and grid-smokes rational structured
  samples.

Still open:

- **2-D endpoint-order atlas:** rule out every non-structured interval cell order.
- **Dimension compression:** prove `n >= 3` cannot beat the 2-D candidate, or find a counterexample.

The falsifier remains simple: a feasible box embedding in any dimension with
`q < (9 + sqrt 17)/32` breaks the candidate and resets the target.

## Artifacts

- `scripts/boxsel_phase4e_lower_bound.py`
- `scripts/test_boxsel_phase4e_lower_bound.py`

Verification:

```text
python scripts/test_boxsel_phase4e_lower_bound.py
```

Result:

```text
17/17 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-4e lower-bound closure start. Internal; restricted-family
certificate only, global lower bound still open.*
