# BoxSEL Phase 4b — `inf I_box^n` Trend RESOLVED

**Date:** 2026-06-20  
**Status:** Trend resolved with a **certified sandwich**. The exact infimum within the bracket is
left open (a harder optimization); the *trend question* — does it vanish? — is answered.

## Question

For the Helly-seed ontology (atoms sized `1/2`, pairwise co-occurrence `≥ 1/4`) and query
`q = P(C | A∩B)`: does `inf I_box^n` equal `513/1250`, fall lower, or tend toward the oracle's
`I*` lower endpoint `0` as embedding dimension grows?

## Answer

```text
1/4  ≤  inf I_box^n  ≤  513/1250        for every n ≥ 2
```

So the box lower endpoint does **NOT** tend to `0`. The representation gap (`I*` lower `= 0` vs
`I_box^n` lower) **persists, bounded below by `1/4`** — it does not vanish with dimension. (This
corrects an earlier off-hand guess that the gap might vanish in high dimensions.)

## Why: three certified ingredients

**1. Factorization (exact).** Axis-parallel box volumes and overlaps factor over axes, so

```text
q = |A∩B∩C| / |A∩B| = ∏_k |A_k∩B_k∩C_k| / ∏_k |A_k∩B_k| = ∏_k q_k,   q_k = P(C_k | A_k∩B_k) ∈ (0,1].
```

This is exactly why 1-D is stuck (one factor, `inf = 1/2`) while higher dims can drop (a product
of sub-unit factors): the certified 2-D witness factors as `q = (16/25)·(513/800) = 513/1250`.

**2. Per-axis lemma (proven; exhaustively grid-verified).** For three intervals with `|A∩B| > 0`,

```text
|A∩B∩C| · |A| · |B|  ≥  |A∩B| · |A∩C| · |B∩C|        i.e.   q_k ≥ P(C|A_k) · P(C|B_k).
```

*Proof.* Let `J = A∩B`. If one of `A,B` contains the other, the inequality reduces to `|A| ≥ |A∩C|`
(trivially true). Otherwise the tails `A\B` and `B\A` lie on **opposite** sides of `J`. Since `C`
is an interval, if it reaches both opposite tails it must cover all of `J`, so `|C∩J| = |J|`; then
`(|A\B|+|J|)(|J|+|B\A|) ≥ (|C∩(A\B)|+|J|)(|J|+|C∩(B\A)|)` because each tail term can only shrink.
If `C` misses a tail, that tail's term is `0` and the bound is immediate. ∎  
Exhaustively re-verified in exact rationals on all interval triples over the `N=6` (6321 triples)
and `N=8` (31536 triples) grids — **0 violations** — and on 26M random float triples.

**3. Global chain (exact).** Both `|A∩C|` and `|A|` factor over axes, so
`∏_k P(C|A_k) = |A∩C|/|A| ≥ (1/4)/(1/2) = 1/2`, and likewise `|B∩C|/|B| ≥ 1/2`. Multiplying the
per-axis lemma over all axes:

```text
q = ∏_k q_k ≥ ∏_k P(C|A_k)P(C|B_k) = (|A∩C|/|A|)(|B∩C|/|B|) ≥ (1/2)(1/2) = 1/4.
```

**Upper bound.** The certified Phase-4 witness gives `q = 513/1250 < 1/2`, extended to every `n ≥ 2`
by full-`[0,1]`-axis padding.

## What is certified vs open

- **Certified:** the factorization; the per-axis lemma (proof + exact grids); `q ≥ 1/4` for **all**
  `n`; `inf I_box^n ≤ 513/1250` for `n ≥ 2`; hence the `[1/4, 513/1250]` sandwich and the
  non-vanishing of the gap.
- **Open:** the *exact* `inf` within `[1/4, 513/1250]`, and whether `n ≥ 3` improves on the `n = 2`
  value `513/1250`. That needs the actual extremal optimizer (a harder, non-smooth program).

## Search-gap aside (NUMERICAL, illustrative — not certified)

A naive random search returns minima of `≈ 0.500 (n=1)`, `0.437 (n=2)`, `0.440 (n=3)`, `0.460 (n=4)`,
`0.468 (n=5)`. Note it **could not even reach** the certified `513/1250 ≈ 0.410` witness at `n = 2`,
and its apparent *rise* in higher dimensions is **search failure**, not the true infimum — a live
instance of the lane's own **search gap**. The analytic bound, not the search, resolves the trend.

## Artifacts & verification

- `scripts/boxsel_inf_trend.py` — factorization, exact per-axis-lemma checker + grid verifier,
  global chain, certified sandwich.
- `scripts/test_boxsel_inf_trend.py` — **17/17 pass**, `python scripts/test_boxsel_inf_trend.py` → exit 0.

---

*Sundog Research Lab — BoxSEL Phase-4b inf-trend resolution. Internal; certified sandwich, exact
infimum left open.*
