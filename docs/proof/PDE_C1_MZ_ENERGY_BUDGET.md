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

## 8. Level-1 v1 result + the energy-conservation finding (2026-05-29)

The first Level-1 run (G=200, `results/proof/c1-mz-budget-g200/`) returned a
result that **verification caught before interpretation**: the band-closed
transfer `T_LLL` is **identically zero** (`8e-17`, machine precision), not
small. This is a conservation law, not a bug — the budget-closure and
`R->0` validation gates all passed. The self-advection of the low-passed
field conserves the low-passed field's energy, so its net transfer summed
over the band is exactly zero:

```
T_LLL = -Σ_low (2/(scale²k²)) Re[ω̂_L*·N̂(ω̂_L)] = -(2/scale²) Σ_all (1/k²) Re[ω̂_L*·N̂(ω̂_L)] = 0.
```

**Physical finding (real, worth keeping):** *the low band cannot
self-determine its own energy* — **all** net nonlinear transfer into it is
out-of-band (high-mode) mediated. So `g = D_low + F_low` (dissipation +
forcing only) and `R = T_low` (the entire inter-scale transfer). Measured
at G=200: `rms(R)/rms(dE/dt) ≈ 14`, `rms(R)/rms(g) ≈ 1`, `corr(g,R) ≈ -0.69`,
`rms(dE/dt)/rms(g) ≈ 0.24` — i.e. a large transfer `R` quasi-balanced
against dissipation+forcing, leaving a small net tendency.

**Consequence:** "is `R` small?" is the wrong question — `R` is the whole
transfer, by conservation. Control-sufficiency cannot mean `R` is small; it
must mean `R` is **predictable from `Φ_K`**. This reframes Level 1.

## 9. Level-1 v2 (redesign): is `R` predictable from `Φ_K`? (held-out R²)

The corrected mechanistic test — does the signature pin the net high-mode
coupling `R = T_low`?

**First estimator tried and rejected (kNN conditional variance).** The
natural disintegration `eta_R = E_i[Var_local(R)]/Var(R)` over kNN signature-
neighbours was implemented and **failed its own calibration**: the positive
control `eta_g` (for `g`, an *exact* function of `Φ_K`) came out `≈ 0.42`,
not `≈ 0`. Cause: kNN conditional variance is confounded by (i) the target's
steepness (`g` is quadratic in `Φ_K`, so it varies a lot inside any ball) and
(ii) 18-dim neighbourhood width (curse of dimensionality — kNN balls are
never truly local). Both inflate the estimate independent of predictability.
Recorded, not rescued.

**Estimator used (held-out regression R²).** The statistically correct,
steepness-agnostic, unbiased test of "is `R` a function of `Φ_K`":

```
slaving_index = R²( R predicted from Φ_K ), held out on a block split
              (first 70% train / last 30% test — no temporal leakage)
```

A flexible regressor (`HistGradientBoostingRegressor`) is fit on the train
block and scored on the held-out block. `R² -> 1` means `R` is (an
approximate) function of `Φ_K` — **the high modes' net energy-transfer into
the band is signature-determined even though the high modes themselves roam
(twin states)**: slaving of the *relevant functional*, not the state. Unifies
with the kNN action result (`Var(action|Φ_K)->0`).

**Built-in calibration controls (known-answer, same data, same regressor):**

- `R²(g)` — **positive control.** `g = D_low + F_low` is an exact function of
  `Φ_K`, so `R²(g) ≈ 1` (sets the regressor's achievable ceiling). Validated
  inline at **0.997**.
- `R²(permuted R)` — **negative control.** Permuted `R` has no signature
  dependence, so `R² ≈ 0` (negative = worse than the mean). Validated inline
  at **−0.20** (confirms no block-split leakage).

**Pre-registered reading + validation gates:**

- Validation: `R²(g) > 0.90` AND `R²(perm) < 0.10` AND `D_low_max <= 0`;
  else `MZ_COUPLING_ESTIMATOR_INVALID`, do not interpret.
- `slaving_index >= 0.70` → `COUPLING_SIGNATURE_SLAVED` (`R` is `Φ_K`-pinned).
- `0.30 <= slaving_index < 0.70` → `COUPLING_PARTIALLY_SLAVED`.
- `slaving_index < 0.30` → `COUPLING_NOT_SLAVED` (sufficiency is
  integrated/averaged → Level 2).
- Still **explanatory, non-promotion**; C1 status unchanged either way.

**Inline pre-run validation (G=200, 4000-sample probe):** `R²(g)=0.997`,
`R²(perm)=-0.20` (controls pass), and the measurement `R²(R|Φ_K)=0.998` —
`R` is predictable from `Φ_K` *at the exact-function ceiling*. The full
50000-sample headline runs confirm this on the registered samples.

Re-runs: `results/proof/c1-mz-coupling-g200/`, `…-g300/` (new dirs; the v1
`c1-mz-budget-*` dirs stand as the conservation-finding record).

## 10. Level-1 v2 result (2026-05-29) — COUPLING_SIGNATURE_SLAVED, both regimes

The corrected diagnostic ran on the full registered 50000-sample sets and
returned `COUPLING_SIGNATURE_SLAVED` at both Grashof regimes, with both
calibration controls clean.

| quantity | G=200 (`lock_v5`) | G=300 (`lock_v7_g300`) |
| --- | --- | --- |
| verdict | `COUPLING_SIGNATURE_SLAVED` | `COUPLING_SIGNATURE_SLAVED` |
| **`slaving_index = R²(R\|Φ_K)`** | **0.998** | **0.990** |
| `R²(g)` positive control (ceiling) | 0.999 | 0.980 |
| `R²(permuted R)` negative control | −0.001 | −0.0005 |
| `corr(g,R)` | −0.85 | −0.93 |
| `rms(R)/rms(dE/dt)` | 14.1 | 2.9 |
| `T_LLL` check `rms(R)/rms(T_low)` | 1.0 | 1.0 |
| train / test (held out) | 35000 / 15000 | 35000 / 15000 |

Results: `results/proof/c1-mz-coupling-g200/`, `…-g300/`.

**Reading.** The net high-mode energy-transfer `R = T_low` into the low band
is **~99% predictable from the signature `Φ_K`** at both regimes — *at the
exact-function ceiling* set by the `g`-control — while the negative control
(permuted `R`) sits at ~0 on 15000 held-out points, ruling out block-split
leakage. So `R` is, empirically, an approximate **function of `Φ_K`**, even
though the high modes themselves are certified **not** reconstructable from
`Φ_K` (twin-state non-injectivity). Equivalently the low-band energy
*tendency* is ~99% signature-determined (`R` and `dE/dt` differ only by the
exact function `g`).

**The mechanism, measured.** The state roams; its decision-relevant aggregate
is slaved to the signature. This is the local, approximate **energy-budget
closure** `R ≈ R(Φ_K)` — the Mori–Zwanzig orthogonal/memory part is only
~1% here — and it holds *precisely where state reconstruction fails*. That is
why the safety decision is `Φ_K`-measurable (control-sufficiency): not because
the coupling is small (it is the whole transfer, by conservation), but
because it is signature-determined. Closes the loop with the observability
framing (separation statement §7): functional observability of the decision
works because the energy-budget closure survives the unresolved state.

**Scope / honesty.** Explanatory, not promotion-bearing — it explains *why*
the measured control-sufficiency holds; it does not change C1's status.
Instantaneous-budget predictability (the lookahead-integrated tie, Level 2,
remains a registered option but is now less necessary — the instantaneous
coupling is already ~99% slaved). Finite-Galerkin, sampled-support, two
Grashof points at `k_f=2`. The `c1-mz-budget-*` dirs (superseded magnitude
diagnostic) stand as the `T_LLL≡0` conservation-finding record. **C1 stays
PROVISIONAL, UNPROMOTED.**

## 7. Cross-references

- [`PDE_C1_MECHANISM_RECON.md`](PDE_C1_MECHANISM_RECON.md) — why this is grounding, not discovery.
- [`PDE_C1_SEPARATION_STATEMENT.md`](PDE_C1_SEPARATION_STATEMENT.md) §7 — the observability framing the mechanism supports.
- [`PDE_C1_PAIRED_FIBER_CONSTANCY.md`](PDE_C1_PAIRED_FIBER_CONSTANCY.md) — the `D_witness` boundary layer this explains.
- Mori–Zwanzig closure: Parish–Duraisamy, Phys. Rev. Fluids 2017 (arXiv 1611.03311).
