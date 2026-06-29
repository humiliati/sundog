# BoxSEL Phase 4l — The Search-Order Filtration (C1)

**Date:** 2026-06-29  
**Status:** `search_resist_confirmed` (bounded-positive). C1 of the cross-lane conjecture slate
([BOXSEL_CONJECTURE_SLATE.md](BOXSEL_CONJECTURE_SLATE.md)) — the one residue the census left live.
Toy / n=2 fragment / one filtration; frozen-as-portfolio.

## What this is

The census ([slate](BOXSEL_CONJECTURE_SLATE.md) C3 verification) collapsed the "bold unifier" into
the already-banked **sufficient-statistic-order schema** — *except* for one genuinely new axis. The
σ-order schema's filtrations are all about what a **statistic determines**. BoxSEL's search gap is
about what a **bounded process reaches**. This phase makes that axis a measured order parameter and
shows it carries the same determine/resist shape — on the one lane with closed-form ground truth.

It turns Phase-4c's qualitative finding ("every from-scratch search misses the optimum") into a
number.

## The filtration

```text
F_g  = configs reachable by exact rational search at resolution 1/g,
       nested along a divisibility chain 6 | 12 | 24  (F_6 ⊆ F_12 ⊆ F_24, since i/6 = 4i/24)
       ⟹ grid_min(g) is MONOTONE non-increasing  ⟹ a proper filtration.

σ_search(target, ε) = least g in the chain with grid_min(g) ≤ target + ε
                      (the search order at which bounded exact search first reaches within ε of target).
```

## The measured determine/resist split

```text
grid_min(6)  = 1/2   = 0.500000
grid_min(12) = 4/9   = 0.444444
grid_min(24) = 4/9   = 0.444444     reachable floor = 4/9
optimum      = (9 + √17)/32 ≈ 0.410097          (Phase-4d closed form, the ground truth)

DETERMINE  σ_search(1/2, 0)        = 6      finite  → reachable
RESIST     σ_search(optimum, 0)    = None   ∞ order → never reached (irrational; no finite rung)
RESIST     σ_search(optimum, +0.02)= None   ∞ order → not even APPROACHED

RESIST MARGIN = floor − optimum = 4/9 − (9+√17)/32 = (47 − 9√17)/288 ≈ 0.034347   (measured, exact)
```

Inside one filtration: targets **above** the reachable floor (½) are finite-order (determine);
the closed-form optimum sits **below** the floor and is infinite-order over the feasible chain
(resist). The optimum is reachable only *down to* the floor (σ_search(optimum, margin) = 12), never
closer.

## The pre-registered vacuity guard (why this isn't a pretty essay)

"Irrational, ergo not on any finite grid" is trivially true and proves nothing — the Phase-7 lesson.
C1 is `search_resist_confirmed` only because all three hold, measured:

- **meter LIVE** — a determine-side target (½) is reached at finite order 6, so the optimum's
  unreachability is a property of the *target*, not a dead probe (cf. the H9-strong order-meter
  control). The floor 4/9 is itself finite-order (g = 12).
- **measured plateau** — grid_min stays ≥ optimum + 0.02 for every rung; bounded search does not even
  approach the truth, it stalls 0.034 above it.
- **monotone** — proper nested filtration.

If grid_min had descended within 0.02 of the optimum at any feasible g, C1 would be `search_resist_null`.

## Honest boundary

The resist is **relative to bounded *unstructured* search** (the denominator filtration). Phase-4c
already showed Nelder–Mead **seeded from the analytic witness's symmetry orbit** *does* reach the
optimum. That is the determine/resist content, not a contradiction: the optimum lies outside the
reachable set of bounded unstructured search but is reachable by **analysis** — "reachable only by
analysis, not by naive search," now with an order parameter. Not a claim that *no* procedure reaches
it; not beyond the n=2 fragment; toy.

## Files

- `scripts/boxsel_phase4l_search_order.py` — the filtration, `σ_search`, the report + verdict
  (reuses Phase-4c `exact_grid_min` and Phase-4d `Q_STAR`).
- `scripts/test_boxsel_phase4l_search_order.py` — frozen test (locks the floors, the split, the exact
  margin, the meter-live guard, the verdict): **16/16 PASS, exit 0**.

---

*Sundog Research Lab — BoxSEL Phase-4l. C1 of the cross-lane conjecture slate: search-reachability is
a determine/resist filtration distinct from the σ-order determination filtrations; bounded exact
search resists the closed-form optimum by a measured 4/9 − (9+√17)/32 ≈ 0.0343. Internal;
frozen-as-portfolio.*
