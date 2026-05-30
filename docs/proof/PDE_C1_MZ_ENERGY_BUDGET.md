# PDE C1 — Mori–Zwanzig Energy-Budget Diagnostic (pre-registration)

> Mechanism-lane (B), post framing-first. **Purpose:** ground the *known*
> Mori–Zwanzig / closure mechanism in this specific cell and **explain the
> measured `D_witness` boundary layer** — NOT to claim a closure discovery
> (the recon, [`PDE_C1_MECHANISM_RECON.md`](PDE_C1_MECHANISM_RECON.md), found
> the phenomenon is MZ/AIM folklore). Status: pre-registered, not built, not
> run. Finite-Galerkin, sampled-support, numerical.

## 1. What we measure and why

The control-sufficiency result (kNN POSITIVE + paired fiber-constancy) says
the safety decision is `Phi_K`-measurable a.e. This diagnostic asks the
*mechanistic* question behind it: **how much of the low-band energy tendency
is determined by the low modes alone, vs. carried by the unresolved high
modes?** If the unresolved coupling `R` is a minority of the budget and its
lookahead integral exceeds the decision margin only on a small `mu`-set,
that both grounds the mechanism here and explains the ~3.7% `D_witness`
residual as the coupling boundary layer.

## 2. The exact decomposition

Integrator (from `step()`): `d ω̂(k)/dt = −N̂(k) − ν|k|² ω̂(k) + f̂(k)`,
with `N̂ = nonlinear_hat` (advection `u·∇ω`). Differentiating the harness
low-band energy `E_low = Σ_low |ω̂(k)/scale|²/|k|²` along the flow:

```
dE_low/dt = D_low + F_low + T_low
  D_low = −2ν Σ_low |ω̂(k)|²/scale²            (low dissipation, low-determined)
  F_low = Σ_low (2/|k|²) Re[ω̂*(k) f̂(k)]/scale² (forcing input, low-determined)
  T_low = −Σ_low (2/|k|²) Re[ω̂*(k) N̂(k)]/scale² (nonlinear transfer in)
```

Mori–Zwanzig split of the only unresolved-dependent term, by re-evaluating
the nonlinear term on the **low-passed** field `ω̂_L` (all modes outside the
9 signature modes + conjugates zeroed):

```
T_LLL = −Σ_low (2/|k|²) Re[ω̂*(k) N̂(ω̂_L)(k)]/scale²   (band-closed; Φ_K only)
R     = T_low − T_LLL                                   (≥1 high mode; the MZ coupling)
g(Φ_K) = D_low + F_low + T_LLL                          (resolved / Markovian part)
dE_low/dt = g(Φ_K) + R
```

`g` depends only on the low modes; `R` is the orthogonal-dynamics coupling.
Two nonlinear evaluations per state (full, low-passed). **No closure model
is fit** — `R` is measured, not modeled.

## 3. Validation (smoke gates, must pass before the headline)

1. **Budget closure.** `g + R` matches the finite-difference tendency
   `(E_low(step(ω̂)) − E_low(ω̂))/dt` to `O(dt)` (semi-implicit Euler, so a
   small discretization residual is expected and bounded, not zero).
2. **`R → 0` on a low state.** For a low-passed input `ω̂_L`, `|R| / |dE_low/dt|`
   is at machine-noise level (the full and band-closed transfers coincide
   when there are no high modes).
3. **Sign sanity.** `F_low >= 0` (forcing injects into the band),
   `D_low <= 0` (dissipation removes).

Failing any gate blocks interpretation — the decomposition would be mis-wired.

## 4. Metrics

**Level 1 — instantaneous coupling (cheap; primary).** At each of the
`50000` adjudication samples (on `supp mu`), decompose `dE_low/dt = g + R`.

**Headline (amended 2026-05-29 after validation): population RMS ratio.**
The per-sample `|R| / |dE_low/dt|` is ill-conditioned where `dE_low/dt -> 0`
(near-laminar states; the validation smoke showed it blowing up to ~10^3).
So the robust primary is the population-level

```
rms(R) / rms(dE_low/dt)   over the 50000 samples   (also rms(R)/rms(g), rms(R)/rms(T_low))
```

plus the bounded per-sample `rho = |R|/(|g|+|R|)` (median / p90 / mean) and
the fraction of samples with `|R| > |g|` as secondaries. *Reading:* a small
`rms(R)/rms(dE_low/dt)` (a minority) means the low-band energy tendency is
mostly low-determined — the mechanism, measured here. Caveat: instantaneous,
not yet decision-tied. Structural validation gate carried into the run:
`D_low_max <= 0` (else `MZ_BUDGET_INVALID_DISSIPATION_SIGN`).

**Level 2 — lookahead-integrated coupling (run only if Level 1 encouraging;
~2x cost).** Along each `tau`-step lookahead, accumulate `I_R = ∫ R dt` and
the realized excursion. *Reading:* the decision flips on `R` only where
`|I_R|` exceeds the margin to `E_max`; that set should be ~`D_witness`
measure and concentrated near the decision boundary — explaining the
residual. Pair-level sharpening (do disagreeing witness pairs have larger
`|ΔI_R|`?) is the strongest tie and is the registered follow-up within
Level 2.

## 5. Pre-registered interpretation (no goalpost-moving)

- This **measures and grounds a known mechanism** (cite Mori–Zwanzig /
  closure) in this cell and **explains** the `D_witness` boundary layer. It
  is **not** a closure result and not novel mechanism — the novelty is the
  observability framing (separation statement §7).
- **No pass/fail verdict on C1.** Unlike the cell-set adjudicators, this is
  explanatory, not promotion-bearing. It cannot make C1 stronger or weaker
  as a *claim*; it can only explain the existing measured numbers. Honest
  outcomes: (a) small `rho` + boundary-concentrated `I_R` → mechanism cleanly
  grounded; (b) large `rho` but control still holds → the sufficiency comes
  from cancellation/averaging over `tau`, a subtler and still-interesting
  story; (c) decomposition fails validation → fix wiring, do not interpret.
- C1 promotion status is **unchanged** by this diagnostic.

## 6. Build + run plan

- Add an `mz-budget` diagnostic to the harness: the decomposition (§2) +
  the three validation gates (§3) as a `--self-test` path, Level-1
  accumulation at sample points, Level-2 accumulation along the lookahead
  behind a flag. Deterministic; reuses `nonlinear_hat` / `low_energy`.
- Smoke + validation gates first. Then:

```
# Level 1 (primary, ~22 min)
python scripts/pde_c1_kolmogorov_cell.py --preset lock_v5     --adjudicator mz-budget --out results/proof/c1-mz-budget-g200
python scripts/pde_c1_kolmogorov_cell.py --preset lock_v7_g300 --adjudicator mz-budget --out results/proof/c1-mz-budget-g300
# Level 2 (only if Level 1 encouraging, ~44 min) — same presets, --mz-lookahead
```

## 7. Cross-references

- [`PDE_C1_MECHANISM_RECON.md`](PDE_C1_MECHANISM_RECON.md) — why this is grounding, not discovery.
- [`PDE_C1_SEPARATION_STATEMENT.md`](PDE_C1_SEPARATION_STATEMENT.md) §7 — the observability framing the mechanism supports.
- [`PDE_C1_PAIRED_FIBER_CONSTANCY.md`](PDE_C1_PAIRED_FIBER_CONSTANCY.md) — the `D_witness` boundary layer this explains.
- Mori–Zwanzig closure: Parish–Duraisamy, Phys. Rev. Fluids 2017 (arXiv 1611.03311).
