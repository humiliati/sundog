# PDE C1 — kNN Convergence Check (Pre-Registration)

> Pre-registration of the scale-dependence test that adjudicates the
> provisional `PDE-C1-NEG-A` from the v4-regime kNN run
> (`results/proof/c1-kolmogorov-v4-knn/`, `incompat_fraction = 0.0716`).
> Filed 2026-05-28, **before** the convergence run is read. Purpose:
> distinguish a *genuine* fiber-incompatibility from a *finite-radius
> boundary-straddling artifact*. Classification thresholds below are
> fixed here and not tuned post-hoc (post-hoc change → `PDE-C1-NEG-B`).

## 1. The question

The v4 kNN run solved coverage (`fidelity_coverage = 1.0`) and fired a
mechanical `PDE-C1-NEG-A`: 7.16% of fidelity-passing neighbourhoods
have local minority fraction above `delta_action = 0.10`. Two competing
explanations:

- **Genuine fiber-incompatibility (true NEG-A).** Distinct microstates
  with the same `Phi_K` need different proxy actions — expected here,
  since the proxy label depends on the full-state `E_K` future while
  `Phi_K` sees only the low modes, so high modes vary freely within a
  fiber and drive different labels. Predicts `incompat_fraction` is
  **constant as the neighbourhood radius `r_k` shrinks** (shrinking the
  signature-ball never constrains the high modes).
- **Finite-radius boundary-straddling (true POSITIVE).** The proxy is
  locally a function of `Phi_K` (control-sufficient), with a clean
  decision surface; the 7.16% is just the attractor measure within
  `r_k` of that surface. Predicts `incompat_fraction → 0` as `r_k → 0`,
  scaling ~linearly with `r_k` (shell of thickness `r_k` around a
  codimension-1 surface).

This is exactly Reading 2's regime-2 (control-sufficient) vs. regime-3
(control-insufficient) distinction, made empirical.

## 2. The test

Re-run the v4 regime (`--preset lock_v4 --adjudicator knn-sweep`).
Query the `BallTree` once at `k = 100`; for each
`k ∈ {10, 20, 30, 50, 100}` sub-slice that query and compute:

- `r_k_median` (over all samples) — the neighbourhood scale;
- `fidelity_coverage` = fraction with `r_k ≤ epsilon_K`;
- `incompat_fraction` = fraction of fidelity-passing samples with local
  minority fraction `> delta_action`.

Smaller `k` → smaller `r_k`. The shape of `incompat_fraction(r_k)` is
the discriminator. Cost: one ~26-min integration; the sweep itself is a
single `k=100` query plus sub-slicing (negligible).

## 3. Pre-registered classification

Ordinary-least-squares fit of `incompat_fraction` on `r_k_median`
across the five swept `k`; let `a` be the intercept (extrapolation to
`r_k → 0`) and `min_incompat` the smallest `incompat_fraction` over the
sweep:

- **`PDE-C1-NEG-A` confirmed (PLATEAU_NONZERO)** iff `a > 0.02` **and**
  `min_incompat > delta_incompat` (= 0.01). The incompatibility
  survives extrapolation to zero radius → genuine.
- **`STRICTNESS_WITNESS_POSITIVE` (DECAYS_TO_ZERO)** iff `a < 0.01`.
  The incompatibility extrapolates to zero → finite-radius boundary
  artifact; the provisional NEG-A is overturned and the proxy is
  control-sufficient on fibers at this cell.
- **`INCONCLUSIVE_CONVERGENCE`** otherwise (`0.01 ≤ a ≤ 0.02`). Neither
  decisively; a larger `N` or a wider `k` range is needed. Non-verdict.

The vacuity gate (global `damp_fraction ∈ [delta_proxy_min,
1 - delta_proxy_min]`) still applies and takes precedence; v4's
`damp_fraction = 0.30` passes it.

## 4. Caveats recorded before the read

- **Small-`k` grain.** At `k = 10`, `minority > 0.10` means `≥ 2/10`
  disagree (effective threshold 0.20); at `k = 100`, `≥ 11/100` (0.11).
  The threshold grain differs across `k` — a known confound. The raw
  `incompat_fraction(k)` table is reported alongside the mechanical
  classification so the trend is visible irrespective of the OLS rule.
- **Linear extrapolation is first-order.** A smooth codim-1 boundary
  gives `incompat ∝ r_k` (linear through the origin); the OLS intercept
  is a first-order discriminator, not a proof. A clearly-positive or
  clearly-zero intercept is trustworthy; a marginal one files
  `INCONCLUSIVE_CONVERGENCE`, not a forced call.
- **High-`k` fidelity.** At `k = 100`, `r_k` grows and
  `fidelity_coverage` may fall below 1; `incompat_fraction` is always
  computed over the fidelity-passing set at that `k`.

## 5. Cross-references

- [`PDE_C1_KNN_ADJUDICATION_DESIGN.md`](PDE_C1_KNN_ADJUDICATION_DESIGN.md)
  — the adjudicator this check stress-tests; §8 flagged the
  boundary-straddling risk this resolves.
- [`PDE_C1_FIBER_PROTOCOL.md`](PDE_C1_FIBER_PROTOCOL.md) §5b — the kNN
  adjudication method.
- [`PDE_C1_LOCK_EXECUTION_SYNTHESIS.md`](PDE_C1_LOCK_EXECUTION_SYNTHESIS.md)
  — the v0–v5 obstruction this campaign is working through.
- `results/proof/c1-kolmogorov-v4-knn/` — the provisional NEG-A this
  check adjudicates.
