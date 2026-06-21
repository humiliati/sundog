# BoxSEL Phase 4i - Two-Help Mixed Core

**Date:** 2026-06-20  
**Status:** Final 10 n=2 orbits reduced to one analytic envelope. Global lower bound still open.

**Update after Phase 4j:** the two-parameter envelope maximum below is now exact-closed. The n=2
atlas is fully closed with `inf I_box^2 = (9 + sqrt 17)/32`.

**Update after Phase 4k:** the separate `n >= 3` compression proof is now banked, so
`inf I_box^n = (9 + sqrt 17)/32` for every `n >= 2`.

## Purpose

After Phase 4h:

```text
113 / 123 n=2 orbits exact-closed
10  / 123 n=2 orbits remain open
```

The remaining cases are exactly the two-help `AC`-vs-`BC` mixed orbits. Phase 4i reduces all 10 to
one two-parameter relaxation.

## Mixed-Core Normal Form

WLOG axis 1 is `AC`-minimum and axis 2 is `BC`-minimum. Let:

```text
x = |A_1 and B_1|,  y = |A_1 and C_1|,  z = |B_1 and C_1|
X = |A_2 and B_2|,  Y = |A_2 and C_2|,  Z = |B_2 and C_2|.
```

Then:

```text
q = yZ/(xX).
```

Set:

```text
r = z/y
t = 4zZ.
```

The global constraints and the mixed-cell geometry imply:

```text
1 <= t <= r <= 2
q >= t/(4 r x X).
```

The interval geometry gives two upper bounds on `X`:

```text
X <= 1 / (2(x + (r-1)y))
X <= 1/(2x) - (r-t)/(4 r y).
```

Balancing those two bounds gives the relaxation envelope:

```text
P(r,t) = max_y r x X / t
       = 2r^2 / (t(r^2 - rt + 3r + t
           + sqrt((r-1)(r-t)(r^2 - rt + 7r + t)))).
```

If:

```text
P(r,t) <= X_OPT = (9 - sqrt 17)/8
```

for all `1 <= t <= r <= 2`, then:

```text
q >= 1/(4 X_OPT) = (9 + sqrt 17)/32 = q_KKT.
```

## What Is Certified

- The 10 remaining n=2 reps are exactly `("AC", "BC")` mixed patterns.
- The endpoint identity is exact:

```text
P(2,1) = (9 - sqrt 17)/8
1/(4 P(2,1)) = (9 + sqrt 17)/32.
```

- The `t=1` edge is monotone increasing in `r`; the derivative sign reduces to the exact squared
  margin `16r^3 > 0`.
- A rational-grid guard over the two-parameter envelope finds no value above `X_OPT`.

## What Remains

This phase does **not** close the global lower bound. The remaining proof obligation is now much
smaller and explicit:

```text
prove P(r,t) <= P(2,1) for all 1 <= t <= r <= 2.
```

After that, the n=2 lower bound is closed. The separate `n >= 3` compression/no-improvement proof
would still remain.

## Artifacts

- `scripts/boxsel_phase4i_two_help_mixed_core.py`
- `scripts/test_boxsel_phase4i_two_help_mixed_core.py`

Verification:

```text
python scripts/test_boxsel_phase4i_two_help_mixed_core.py
```

Result:

```text
13/13 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-4i two-help mixed core. Internal; final 10 n=2 orbits reduced to
one envelope; envelope maximum proof and n>=3 compression remain open.*
