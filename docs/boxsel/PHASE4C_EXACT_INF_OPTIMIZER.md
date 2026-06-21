# BoxSEL Phase 4c — The Exact-Infimum Optimizer

**Date:** 2026-06-20  
**Status:** Trend pinned to a finite-dimensional minimum; exact algebraic value left open. The
headline is a **severe search-gap demonstration**.

## Question

Pin the exact value of `inf I_box^n` inside the Phase-4b bracket `[1/4, 513/1250]`, for the seed
ontology (atoms `1/2`, pairwise `≥ 1/4`) and query `q = P(C | A∩B)`.

## Answer

- **The infimum is ATTAINED AT `n = 2`** — a finite-dimensional minimum. `n=1` gives exactly `1/2`;
  `n ≥ 3` do **not** improve on `n = 2`. It does not vanish and does not keep decreasing. (Large `n`
  is forced toward near-full intervals — `∏_k |A_k| = 1/2` — which pushes each factor `q_k → 1`, so
  the value plateaus/rises back toward `1/2`; consistent with every optimizer below.)
- **The `n=2` optimum is `≈ 0.41010`**, with **both** `|A∩C| = |B∩C| = 1/4` active. The Phase-4
  witness had only `|A∩C|` active, which is why it sits slightly higher at `513/1250 = 0.41040`. So
  the witness is **near-optimal but not exact** — about `0.0003` above the true value.
- **Certified:** `1/4 ≤ inf I_box^n ≤ 513/1250` (Phase 4b). **Numerical:** attained at `n=2`,
  `≈ 0.41010`. **Open:** the exact algebraic value (KKT with two active overlap constraints), and a
  proof that no `n ≥ 3` beats `n = 2`.

## The headline: a severe search gap

Every **from-scratch** optimizer fails to reach even the rational witness `513/1250`. Only local
search **seeded from the analytic witness** finds the optimum:

| method | best feasible `q` reached | verdict |
| --- | --- | --- |
| random search (numpy, ~3.2M feasible samples) | `0.437` (n=2) | misses |
| SLSQP (60 multistarts) | `0.500` (n=2), `1.000` (n≥3) | fails outright |
| differential evolution (popsize 25, 400 iters) | `0.430` (n=2), `0.432` (n=3), `0.443` (n=4) | misses; rises with n |
| **exact rational grid** (g ≤ 24, deterministic) | `≥ 4/9 ≈ 0.444` | misses; **non-monotone in g** (g=12→4/9, g=16→1/2) |
| **Nelder–Mead seeded from the analytic witness** | `0.41010` (n=2), `0.41040` (n≥3) | **reaches the optimum** |

The optimum needs endpoint denominators (`41`, `1600`, …) no practical grid contains, and the
landscape defeats both gradient (SLSQP) and global gradient-free (DE) search. So on this fragment
the **search gap `I_box \ I_sample` is severe**: without the analytic handle, a search returns a
badly-wrong infimum (off by `0.03`–`0.09`). That is the lane's own thesis — search underperforms
the representable truth — biting the *meta*-optimization of `I_box` itself. It also re-confirms the
two independent optimizer trends (DE, NM-padded) that `n ≥ 3` does not beat `n = 2`.

## What is certified vs numerical

- **Certified (exact, in the frozen test):** the `[1/4, 513/1250]` bracket; the witness is a valid
  feasible upper bound; `q ≥ 1/4`; the exact-grid search-gap values (`1/2`, `4/9`, …) — all `>` the
  witness.
- **Numerical (provenance constants, reproduced by the session's exploration):** the infimum is
  attained at `n = 2`, `≈ 0.41010`, with both C-overlaps active; the four float-optimizer figures.

## Artifacts & verification

- `scripts/boxsel_exact_inf_optimizer.py` — certified bracket, exact rational grid search-gap probe,
  documented optimizer-result provenance.
- `scripts/test_boxsel_exact_inf_optimizer.py` — **17/17 pass**,
  `python scripts/test_boxsel_exact_inf_optimizer.py` → exit 0.

## Next (open)

- The **exact algebraic optimum** at `n=2`: solve the KKT with `|A∩C| = |B∩C| = 1/4` active (and
  `|A∩B|` slack) — yields the precise infimum and, ideally, a tighter certified rational upper bound
  below `513/1250`.
- A **proof** that `n ≥ 3` cannot beat `n = 2` (the small-deficit argument is the seed).

---

*Sundog Research Lab — BoxSEL Phase-4c exact-infimum optimizer. Internal; finite-dimensional
minimum located, exact algebraic value open; primary result is the search-gap demonstration.*
