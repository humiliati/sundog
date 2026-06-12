# HS10 PREREG — LATTICE-FRAG: the jittered-lattice recovery horizon obeys the 1/ε Debye-Waller law

**Frozen:** 2026-06-11, before any HS10 code exists. Lane: charFun determine/resist law (red team,
lattice clause). Internal slate ref: HS10 of slate 2026-06-10 (gitignored internal document).

**Standing discipline (binds this prereg):** pre-registered KILL criterion — a clean null is a
SUCCESS; forward-generate only (the prediction is the analytic charFun pushed through the frozen
estimator — no fitting to data); deterministic seeded runs + frozen tests; cheap headless (numpy +
the frozen sklearn apparatus, no GPU, no external data); name the nearest prior, state the delta.

**Honest scope, stated up front:** the *qualitative* wash of the jittered lattice is a theorem —
charFun of an independent sum is the product, so ξ = ±1 + ε·N(0,1) has
Re φ_ε(s) = cos(s)·e^(−ε²s²/2), absolutely continuous for ε > 0, hence eventually washes by the
proved `absCont_resists` / `resistance_general`. **None of that is claimed as new.** All
falsifiable content lives in the **quantitative bridge** from the frozen apparatus's readout to the
DW envelope — exactly the bridge any future "this wash means resist" interpretation will lean on.

---

## §1 Claim

On the frozen S0 v2 band-pass apparatus (`pvnp_phase5_lossiness_crossover.gen_s0` conventions via
`shadow_charfun_populations`), with the averaging population jittered to ξ = ±1 + ε·N(0,1):

1. **(DW tracking)** The band fringe-amplitude **ratio** R(λ,ε) = â(λ,ε)/â(λ,0) — a paired,
   common-random-numbers estimate in which the lattice recurrence factor cos(2πλt) and the
   apparatus's own band dephasing cancel — tracks the analytic prediction (the charFun product law
   pushed through the *same* estimator pipeline) within **RMS ≤ 0.10** across the masked validity
   window, pooled over the main ε set.
2. **(1/ε horizon)** The empirical ratio-horizon λ*(ε) — first downward crossing of R(·,ε) below
   θ = 0.5 inside the masked window — scales as a power law in ε with fitted log-log slope
   **−1 ± 0.25** over the main ε set (discriminates 1/ε from 1/ε² and from
   stretched-exponential/saturation alternatives).

Converting the law's one SURVIVE clause from exact-atomicity-only (measure-zero physically) into a
quantitative atomicity-tolerance statement: *how much* analog jitter the lattice's survival
tolerates before the apparatus's recovery horizon closes, and with what scaling.

## §2 First leg (the only leg)

**(a) Population (added to `scripts/shadow_charfun_populations.py`, strictly additive):**
`pop='lattice_jitter'` with parameter ε: draws `lat = rng.choice([-1,1], shape)` then
`gz = rng.standard_normal(shape)`, returns `lat + ε·gz` — **in that fixed order**, so at fixed λ
the (xc, xd, lat, gz, obs-noise) streams are identical for every ε including ε = 0, and ε enters
only as a multiplier (paired/common-random-numbers design; the ratio is deterministic given seeds).
Analytic charFun: `Re φ_ε(s) = cos(s)·exp(−ε²s²/2)` added to `charfun_re` under the same pop key.
Existing pops untouched (G2 regression-pins them).

**(b) Driver `scripts/shadow_lattice_jitter_dw.py`:**

- **Amplitude estimator (frozen):** per sample i with template `T_i(t) = cos(2π·xc_i·t)·env_f(t)`:
  `â_i = Σ_t sig_i(t)·T_i(t) / Σ_t T_i(t)²`; `â(λ,ε) = mean_i â_i` (least-squares amplitude of the
  sample's own band template; A = 1 so â ≈ 1 at λ = 0).
- **Analytic prediction through the same pipeline (forward-generated):** for the SAME draws
  (xc_i, xd_i), the ξ- and noise-expectation of the generator
  `sig_pred_i = D·bump + A·cos(2πxc_i t)·Re φ_ε(2πλt)·env_f + C·xd_i·parity·env_g`;
  `pred(λ,ε) = mean_i ⟨sig_pred_i, T_i⟩/⟨T_i,T_i⟩`. The estimator is linear in the signal and the
  per-sample K-average is unbiased for Re φ, so E[â] = pred exactly — residuals are pure
  finite-(n,K)+noise variance, which is what the RMS kill budgets.
- **Ratios:** `R_emp = â(λ,ε)/â(λ,0)`, `R_pred = pred(λ,ε)/pred(λ,0)`, same paired draws.
- **Grids:** main ε ∈ {0.15, 0.2, 0.3, 0.5}; sanity ε ∈ {0.01, 0.03} (no-early-collapse checks
  only — their horizons sit beyond the validity window by design; the slate's original use of them
  as primary points was refuted as unobservable decor). Ratio λ-grid: 0.05 to λ_max(ε) in steps of
  0.05, λ_max(ε) = min(8.75/(1+4ε), 5.5). n = 600 per cell (the `sweep_pop` default convention);
  per-λ seed = `20260608 + int(round(λ·1000)) + 7` (the frozen sweep convention), shared across ε.
- **Validity mask (pre-registered):** λ ≤ λ_max(ε) (Nyquist: max instantaneous fringe frequency
  xc_hi + λ(1+4ε) = 7 + λ(1+4ε) must stay below the t-grid Nyquist 63/4 = 15.75; 4σ jitter bound)
  AND |pred(λ,0)| ≥ 0.10 (denominator floor — excludes the lattice recurrence nulls, e.g. the
  banked λ = 0.5 band-null, where the ratio is undefined-noisy).
- **Horizon:** λ*(ε) = first downward crossing of R_emp below θ = 0.5, linear interpolation between
  adjacent masked grid points. If no crossing exists inside the masked window for a main ε, K2
  FIRES (saturation/flattening is exactly what the exponent kill is for). Fit: OLS of log λ* on
  log ε over the 4 main ε; report R² alongside.
- **Secondary, reported, NO kill:** the unchanged frozen recovery sweep (`sweep_pop`, frozen
  LAMBDAS, n = 600) for each main ε — cont(λ) and disc(λ) curves; disc is predicted ≈ banked
  (jitter keeps a finite centered mean — determination is a theorem); recovery half-life λ*_c(ε)
  reported (censored where beyond the frozen grid).

**(c) Frozen test `scripts/test_shadow_lattice_jitter_dw.py`** (runs before the verdict run; pins
gates, not outcomes): G2 regression baseline, pairing determinism, R(λ→0) ≈ 1, estimator
self-consistency (â on a noiseless, expectation-substituted signal equals pred to machine
precision), mask logic. Headline numbers are byte-pinned in the banking commit after the run.

**Predicted magnitudes (guidance, NOT a kill surface):** the analytic ratio crosses θ = 0.5 near
λ*·ε ≈ 0.36–0.40 (band centered at |t| = 0.50), i.e. λ* ≈ 2.5 / 1.9 / 1.25 / 0.75 at the four main
ε — all inside their validity windows (5.47 / 4.86 / 3.97 / 2.92) with margin. The slate's quoted
3.8/2.9/1.9/1.1 correspond to a lower-threshold convention (θ ≈ 0.2); the threshold choice moves
the constant, never the exponent — the kill is on the exponent.

## §3 KILL criteria (either firing is informative; both passing = clean confirmatory success)

- **K1 (bridge RMS):** RMS(R_emp − R_pred) over the masked grid, pooled over main ε, **> 0.10** —
  the apparatus's readout does not reduce to the DW envelope; some unmodeled channel (aliasing,
  estimator bias, contamination) dominates the bridge, and every future wash-interpretation
  inherits that caveat.
- **K2 (horizon exponent):** fitted log-log slope of λ*(ε) outside **[−1.25, −0.75]**, OR any main
  ε lacking a crossing inside its masked window — the recurrence-destruction mechanism is not pure
  Debye-Waller at the readout level.
- **Sanity (reported with the kills, fires K1-style only if gross):** ε ∈ {0.01, 0.03} must show
  NO early collapse: min R_emp over the masked window ≥ 0.8.

**What CANNOT kill:** the recovery-sweep secondary leg (corollary of proved theorems); G-gate
failures (§4 — apparatus bugs, abort and fix, not results); the predicted-magnitude guidance above.

## §4 Gates (abort = bug/redesign, not result)

- **G1 (ε = 0 denominator control, the slate-mandated one):** on the extended λ-grid, the
  *unnormalized* â(λ,0) tracks pred(λ,0) with RMS ≤ 0.10 — verifies the apparatus reproduces its
  own band-dephasing structure (the banked cont(λ=2) = 0.655-vs-envelope-1.000 gap lives here),
  i.e. that the ratio cancels what we claim it cancels.
- **G2 (regression):** existing pops byte-unchanged — `sweep_pop('lattice', n=400, lams=[0,0.5,1,2])`
  cont = [0.7289, 0.4391, 0.5373, 0.566] and `'gaussian'` = [0.6939, 0.5059, 0.0, 0.0]
  (re-verified on this machine 2026-06-11, 12/12 frozen test).
- **G3 (pairing determinism):** a repeated (λ,ε) cell reproduces â byte-identically; |R_emp − 1| ≤
  0.02 at λ = 0.05 for every ε.
- **G4 (power):** split-half SE of R_emp near each main-ε crossing ≤ 0.03; if violated, n is a
  pre-authorized power knob (raise and rerun); kill thresholds and grids stay frozen.

## §5 Nearest priors (named), and the delta

**Banked:** the lattice SURVIVE clause itself (`shadow_charfun_populations.py` + the Lean
`ShadowDecayLattice.twoPoint_shadow_survives` / `twoPoint_charFun` — exact ±1 atoms only); the
charFun sharpening receipt (`docs/atlas/SHADOW_CHARFUN_DETERMINE_RESIST_LAW.md`) which already
prints analytic envelopes next to empirical cont. **External:** Debye 1913 / Waller 1923 (the DW
factor IS the textbook answer to "what does Gaussian jitter do to a lattice's coherent peaks");
Lukacs, *Characteristic Functions* (convolution ⇒ product). **The delta:** first
stability/perturbation analysis of any clause of the law — binary survive → an ε-resolved horizon
scaling law, with the falsifiable content placed honestly in the apparatus-to-envelope bridge.
("Jittered lattice eventually washes" is a corollary of proved theorems and is NOT claimed.)

**Lean follow-on (owner-gated, not this run):** charFun-of-convolution = product as the tower's
next theorem, which would make the jittered-lattice envelope itself machine-checked.

## §6 Receipt plan

`docs/atlas/SHADOW_LATTICE_JITTER_DW_RESULT.md`: outcome (CONFIRMED / K1 / K2), the R_emp-vs-R_pred
RMS table, the λ*(ε) table + fitted slope ± R², gate results, the secondary recovery curves,
deviations (if any) with reasons, and the frozen-test pin list. Memory + slate-status update after
banking.
