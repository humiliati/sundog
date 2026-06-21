# BoxSEL Phase 4h - One-Help Closure

**Date:** 2026-06-20  
**Status:** Exact n=2 closure for every one-help orbit. Global lower bound still open.

## Purpose

After Phase 4g:

```text
78 / 123 n=2 orbits exact-closed
45 / 123 n=2 orbits open
```

The open set split into:

```text
35 one-help mixed-side orbits
10 two-help AC-vs-BC mixed orbits
```

Phase 4h closes all 35 one-help mixed-side orbits exactly.

## Proof

It is enough to prove the `AC` case; `BC` is symmetric.

Assume axis 1 is the only helping axis and:

```text
q = |A_1 and C_1| / |A_1 and B_1| < 1/2.
```

Then:

```text
|A_1 and C_1| < |A_1 and B_1| / 2.
```

Since the ontology requires:

```text
|A and C| = |A_1 and C_1| |A_2 and C_2| >= 1/4,
```

we would need:

```text
|A_2 and C_2| > 1 / (2 |A_1 and B_1|).
```

But:

```text
|A_2 and C_2| <= |A_2|
|A_1 and B_1| <= |A_1|
|A_1| |A_2| = 1/2
```

so:

```text
|A_2| = 1/(2|A_1|) <= 1/(2|A_1 and B_1|),
```

a contradiction. Therefore every one-help orbit has:

```text
q >= 1/2 > q_KKT.
```

## Result

The n=2 atlas now stands at:

```text
31  zero-help closed          (Phase 4f)
47  same-side live closed     (Phase 4g)
35  one-help mixed closed     (Phase 4h)
10  two-help mixed open
```

So:

```text
113 / 123 n=2 orbits exact-closed
10  / 123 n=2 orbits remain open
```

The remaining n=2 cases are exactly the two-help `AC`-vs-`BC` mixed orbits; this is where the KKT
candidate lives.

## Boundary

The global sandwich is unchanged until the final 10 n=2 mixed orbits and the `n >= 3` compression
are closed:

```text
1/4 <= inf I_box^n <= (9 + sqrt 17)/32.
```

## Artifacts

- `scripts/boxsel_phase4h_one_help_closure.py`
- `scripts/test_boxsel_phase4h_one_help_closure.py`

Verification:

```text
python scripts/test_boxsel_phase4h_one_help_closure.py
```

Result:

```text
12/12 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-4h one-help closure. Internal; 113/123 n=2 orbits exact-closed,
10 two-help mixed n=2 orbits remain open; global lower bound still open.*
