# BoxSEL Phase 4j - Mixed Envelope Closure

**Date:** 2026-06-20  
**Status:** The final 10 n=2 two-help mixed orbits are exact-closed. The n=2 infimum is now proved.

## Result

Phase 4i reduced the remaining n=2 frontier to one envelope:

```text
P(r,t) = 2r^2 / (t(B + sqrt(D))),     1 <= t <= r <= 2
```

where:

```text
B = r^2 - rt + 3r + t
D = (r-1)(r-t)(r^2 - rt + 7r + t).
```

Phase 4j proves:

```text
P(r,t) <= P(2,1) = (9 - sqrt 17)/8.
```

Therefore every one of the final 10 mixed n=2 orbits has:

```text
q >= 1/(4 P(2,1)) = (9 + sqrt 17)/32 = q_KKT.
```

Combined with Phases 4f-4h:

```text
31  zero-help closed          (Phase 4f)
47  same-side live closed     (Phase 4g)
35  one-help mixed closed     (Phase 4h)
10  two-help mixed closed     (Phase 4j)
---
123 / 123 n=2 orbits exact-closed
```

So:

```text
inf I_box^2 = (9 + sqrt 17)/32.
```

## Certificate

Use shifted variables:

```text
a = r - 1
b = t - 1
0 <= b <= a <= 1.
```

Let `alpha = X_OPT = (9 - sqrt 17)/8` and:

```text
L = 2r^2/(alpha t) - B.
```

If `L <= 0`, the envelope comparison is immediate. Because:

```text
2/alpha = (9 + sqrt 17)/4,
```

the sign of `L` is the sign of an exact `Q(sqrt17)` polynomial `M(a,b)`.

When `M > 0`, squaring is legal. The squared residual factors exactly:

```text
D - L^2 = (a+1)^2 F(a,b) / (8(b+1)^2).
```

So it remains to prove:

```text
M(a,b) > 0  =>  F(a,b) >= 0.
```

For fixed `a`:

- `M` is strictly decreasing in `b` on `[0,a]`.
- `M(a,a) < 0`, so the `M>0` region is an initial interval `[0,beta)`.
- `F` is concave in `b`.
- `M(a,0)` is strictly increasing; `F(a,0) >= 0` whenever `M(a,0)>0`. The `F(a,0)`
  threshold is `a = sqrt17 - 4`, and `M(sqrt17 - 4,0) = -12(sqrt17 - 4) < 0`, so the live branch
  starts after that threshold.
- On the other endpoint, `M=0`, the resultant

```text
Res_b(F,M) =
-32 a (a+1)^4 (135a sqrt17 + 1247a + 448sqrt17 + 6080)
```

has a fixed nonzero sign for `0<a<=1`, so `F` cannot cross zero along the live `M=0` branch. It
is positive at the branch start, hence positive throughout.

By concavity, `F>=0` on the full `M>0` interval. This proves the envelope maximum and closes the
ten orbits.

## Claim Boundary

This closes the n=2 endpoint-order atlas only. The arbitrary-dimensional lower bound still needs a
separate compression/no-improvement proof:

```text
1/4 <= inf I_box^n <= (9 + sqrt 17)/32.
```

The exact n=2 statement is stronger:

```text
inf I_box^2 = (9 + sqrt 17)/32.
```

## Artifacts

- `scripts/boxsel_phase4j_mixed_envelope_closure.py`
- `scripts/test_boxsel_phase4j_mixed_envelope_closure.py`

Verification:

```text
python scripts/test_boxsel_phase4j_mixed_envelope_closure.py
```

Result:

```text
21/21 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-4j mixed-envelope closure. Internal; n=2 lower bound closed,
arbitrary-dimension compression remains open.*
