# Phase 5 — Cross-Substrate Sameness: the Shadow-Invertibility Law (candidate operator)

> **STATUS: CANDIDATE OPERATOR — SYNTHETIC LEG MEASURED + PHYSICAL LEG (S2) PARTIAL, 2026-06-07. NOT the
> Phase-5 Exit yet.** S2 ran as a `partial physical leg` (§3.13): discrete-determines CONFIRMED on real
> halo physics (ice-phase + handedness, `disc=1.000` flat); continuous-physical PARTIAL (size resists
> 0.97→0.45 — magnitude scale-leak). Full physical discharge (clean continuous washout + measured-sky
> polarimetry) still owed; handedness is a predicted/unobserved observable. The coarse-graining
> roadmap's Phase 5 (`../COARSE_GRAINING_PROOF_ROADMAP.md`
> §"Phase 5 — Cross-substrate sameness") requires a **measured** cross-substrate operator-identity
> table on ≥2 substrates to dissolve the equivocation attack. This file supplies the **candidate
> operator**, the **measurable test design** (conjecture + falsifier), and now the **synthetic measured
> table** (§3.11, `operator_confirmed_synthetic` on S0+S1 — two structurally-different *synthetic*
> substrates, frozen + determinism-receipted). The **physical** table (S2, the halo instantiation) is
> still owed; **do not cite this as Phase-5-complete or public-eligible.** It also **reframes** the
> Phase-5 target: from "the same
> sufficient statistic `Φ` recurs" to "the same *invertibility split* recurs," which directly answers
> the resistance/inversion question the founding theorem cared about (and which the proof trunk had
> drifted away from — see `pvnp/SUNDOG_CERTIFICATE_PROBLEM.md` and the 2026-06-05 lacuna review).

## 0. The claim (near-theorem)

> **The Shadow-Invertibility Law.** Let a hidden state decompose as `x = (x_c, x_d)` — a
> continuous-magnitude part `x_c` (values in a manifold) and a discrete/topological part `x_d` (values
> in a discrete set / a structurally-stable invariant). Let `σ = H(x)` be a shadow that is a **lossy
> projection averaged over an ensemble/population `P`**. Then:
> - **`x_d` is determined by `σ`** (invertible) to the extent `H` is `x_d`-sensitive and `x_d` is
>   `P`-structurally-stable (the discrete classes do not overlap under `P`'s perturbations);
> - **`x_c` resists** `σ` (non-invertible) to the extent the average over `P` smears `H`'s
>   `x_c`-signature.
> - **The lossiness is essential:** if `H` is injective (no averaging), both `x_c` and `x_d` are
>   determined. Resistance is a property of *lossy/averaged* shadows, not of shadows as such.

Symptom: **continuous-resists / discrete-inverts.** Cause: **structural stability under lossy
averaging.**

## 1. The mechanism (why it generalizes)

A determining-shadow is almost always an **average over a population** (the crystal cloud, the
training distribution, the measurement ensemble). Ask what survives the average:

- A **continuous magnitude** sends nearby values to overlapping shadows; under the population spread
  they smear and the average loses them — structurally *unstable* → washed out → **resists**.
- A **discrete/topological invariant** sends distinct values to **disjoint** classes (a sign, a winding
  number, a homotopy/coset class) that cannot overlap under small perturbation — structurally *stable*
  → averaging **reinforces** the class → **inverts**.

This is Thom's **structural stability** (the catastrophe *type* is stable, the precise position is
not), Nye's **wave dislocations** (the topological skeleton of a field is its robust part), and
Blackwell **sufficiency** at once: `σ` is sufficient for `x_d` and insufficient for `x_c`. The two
control knobs are **(i) the lossiness of `H`** (how much averaging) and **(ii) the kind of `x`**
(magnitude vs. invariant); together they predict a **marginal↔exact spectrum**.

## 2. Cross-substrate instantiation table (the operator, read across the portfolio)

| substrate | lossy/averaged shadow | `x_d` (discrete → determined) | `x_c` (continuous → resists) | regime |
| --- | --- | --- | --- | --- |
| **Aharonov–Bohm** (`CROSS_SUBSTRATE_NOTES.md` §6.3, l.1012–1092) | loop holonomy `∮A=Φ` | flux quantum / AB phase (topological) — **exact** | interior field `B(x)` — exact resist | **EXACT** (the program's only non-marginal regime-2) |
| **Syndrome certificate** (`pvnp/SUNDOG_CERTIFICATE_PROBLEM.md`) | parity-check `z=Hy` | the coset / algebraic class — determined by algebra | the secret `s` (`qᵏ`-to-one) — one-way | **EXACT** (by algebra) |
| **Halo polarization** (`../SUNDOG_V_ATLAS.md` §1.2, Shadow 3) | ensemble-averaged Stokes `V` | handedness / `Z₂` parity — invertible | crystal size, position — resist | **clean** (discrete invertible) |
| **Optical vortices** (atlas medium-tower) | scintillation field | wave-dislocation index (Nye, integer) — invertible | turbulence strength `C_n²` — resists | **clean** |
| **Ice phase** (atlas material-tower) | halo radius (28° vs 22°) | crystal system (hex/cubic/pyramidal) — discrete, determined | exact `n(λ)` — resists | **clean** |
| **Mesa / NSE-C1 / Sabra** (§6.3 three-for-three) | near-injective control shadow (FVE 0.97–0.99, PR≈2) | — (no genuine discrete invariant probed) | the body — **nearly invertible** | **MARGINAL** = *insufficient lossiness* |
| **Gate-0 trained body** (`pvnp/DIRECTIONB_GATE0_NOTE.md`) | single smooth map, no ensemble | — | `z` fully exposed (control-sufficiency) | **NULL** = *injective → no resistance* |

The marginal and null rows are the law's **own negative side**, not exceptions: mesa/NS/Sabra were
*near-injective* (not lossy enough), so `x_c` was nearly recoverable and the separation collapsed
toward vacuity — `CROSS_SUBSTRATE_NOTES.md` states this verbatim ("where the projection is
near-invertible … the separation collapses toward vacuity"). Gate-0 is a single smooth map with no
averaging, so nothing resists. The exact rows (AB, syndrome) are exact **because** the determined
variable is discrete/topological *and* the shadow is genuinely lossy.

## 3. The frozen lossiness-crossover slate (the falsifier that would land Phase 5)

> **STATUS: SLATE DRAFT.** §3.1–§3.6 and §3.8–§3.10 (question, substrates, metrics, thresholds, void
> gates, verdict tree, and anti-p-hack discipline) **freeze now**. §3.7 numeric constants are
> **smoke-calibrated on throwaway seeds, then
> frozen with the prediction** (the syndrome-lane discipline — never tune a frozen constant after
> seeing the frozen-seed result). Anti-p-hack §3.10.

### 3.1 The single question
As a shadow's **lossiness** (ensemble spread) increases, does recovery of a **continuous** hidden
variable decay to chance while recovery of a **discrete** hidden variable stays exact — and does the
**same crossover** appear across ≥2 structurally-different substrates? A yes is the measured shared
operator (the Phase-5 anti-equivocation content: one operator, not one word).

`x = (x_c, x_d)`: `x_c` continuous, `x_d ∈ {±1}` discrete. Shadow `σ` = ensemble-average over `K`
sub-units, each carrying its own `x_c,i = x_c* + λ·ξ_i` (`ξ_i ~ N(0,1)`); **`x_d` is shared** across
the population (a structural property of the cloud). `λ` is the lossiness knob.

### 3.2 The two FROZEN substrates (S0, S1) + the staged physical one (S2)
- **S0 — 1-D caustic toy (INLINE; v2 = band-pass fringe).** `σ(t)`, `t` on a `T`-point grid over
  `[−1,1]`, `=` `D·bump(t; t0, w_b)` (scale-free **central** geometric halo, carries nothing) `+`
  `A·cos(2π·x_c,i·t)·env_f(t)` (continuous: fringe **frequency** = `x_c,i`, carried by a **band-pass**
  envelope `env_f(t)=exp(−(|t|−t0_f)²/2w_f²)` that **vanishes at `t=0`** so the size lives *only* in the
  off-centre fringes; additive average → `cos(2π x_c* t)·exp(−2π²λ²t²)` on that off-centre band →
  Debye–Waller-damps to nothing, frequency fully unrecoverable, **no central survivor**) `+`
  `x_d·C·sin(2π f_p·t)·env_g(t)` (discrete: parity channel on the central Gaussian
  `env_g(t)=exp(−t²/2w²)`, fixed `f_p` outside `[x_c]` range, the **sign factors out of the average**).
  Features = the `T` samples of `σ̄(t)` (+ noise). `x_c*` = fringe frequency (continuous);
  `x_d ∈ {±1}` = parity.
  **v2 rationale (calibration-caught, pre-registered amendment).** v1 used the *central* Gaussian `env`
  for the fringe; the frequency leaked through the surviving `t≈0` region (`cont` floored at ~0.68 and
  **no obs-noise threaded `cont(0)≥0.70` AND `cont(λ_max)≤0.10`** — the window was empty). The band-pass
  `env_f` removes the central leak so the washout completes, and it is *more faithful to the halo*: in
  real optics the **size** lives in the off-centre supernumerary fringes while the central peak is the
  **scale-free geometric halo** — exactly the Shadow-1 (scale-free geometry) vs Shadow-2 (off-centre,
  size-bearing) split. This is a frozen-generator **amendment (slate v2)**, not a power tweak; it is
  re-calibrated on the throwaway seed and re-predicted before any frozen run.
- **S1 — 2-D vector-field substrate (INLINE; the cross-substrate leg — different in dimensionality,
  field type, and discrete-invariant type).** On a `G×G` grid over `[−1,1]²`, sub-unit `i` field
  `V_i(p) = A·cos(2π·f0·r(p) + x_c,i)·r̂(p) + x_d·B·τ̂(p)/(r(p)+ε)` where `r=|p−center|`, `r̂` radial
  unit, `τ̂` tangential unit. **Continuous** `x_c` = the **phase offset** of a fixed-wavenumber radial
  texture (washes out by additive population mixing via amplitude attenuation,
  `→ cos(2π f0 r + x_c*)·exp(−½λ²)·r̂`, not S0-style spatial frequency cancellation and not centroid-
  preserved). **Discrete** `x_d ∈ {±1}` = the **circulation sign / winding orientation** (the sign of
  `∮V·dl`, with magnitude regularized by `ε`; conserved under same-sign superposition → survives any
  `λ`). Features = the `2·G²` components `(Vx, Vy)` of `V̄(p)` (+ noise). `x_c*` = radial phase offset;
  `x_d` = winding sign.
- **S2 — halo physical instantiation (APPARATUS-GATED; pre-registered slate in §3.12).** `x_c` =
  crystal/droplet **size**, `x_d` = **ice phase** (halo radius, robust) or **handedness** (Stokes-`V`
  sign, *predicted/novel*). **Lit-pass outcome (2026-06-07, `docs/atlas/S2_LITPASS_E_G.md`):** HaloSim
  (closed ray-tracer) cannot host the apparatus → a **standalone real-physics forward model** is
  required; and "size off the refraction-halo fringe" is physically **dead** (Berry 1994, zero-contrast
  step edge) → the size shadow lives on the **corona** (Airy `[2J₁(x)/x]²`, `θ∝λ/a`). S2 therefore runs
  as **two halo/diffraction-family legs** (corona = continuous-resists; refraction halo =
  discrete-determines) — see §3.12.

### 3.3 Metrics (frozen)
- `cont(λ) = max(0, R²_cv)` — cross-validated coefficient of determination of the continuous probe
  predicting `x_c*` from `σ`, floored at 0 (negative `R²` = worse than the mean baseline = 0).
- `disc(λ) = (acc_cv − maj) / (1 − maj)` — recovery determinant of the discrete probe predicting `x_d`
  (`maj` = majority-class frequency); 0 at chance, 1 perfect (the JEPA-0D / Gate-0 metric).
- **Probe families:** linear (`LinearRegression` / `LogisticRegression`) **and** one-hidden-layer MLP.
  **Verdict policy: BEST-OF the two for each metric** — the strongest claim on both sides (continuous
  *resisting even the better probe* is strong resistance; discrete *recovered by the better probe* is
  strong determination). Both families are reported separately for transparency; verdicts read best-of.

### 3.4 The crossover statistic + cross-substrate identity (frozen, numerical)
Per substrate, per `λ`: half-life `λ*_c` = smallest grid `λ` with `cont(λ) ≤ ½·cont(0)`; `λ*_d` =
smallest grid `λ` with `disc(λ) ≤ ½·disc(0)`, else **censored** (`> λ_max`). Crossover statistic =
`λ*_d / λ*_c` (predicted `→ ∞`, reported `> λ_max/λ*_c` when `λ*_d` censored). **"Same qualitative
crossover" is defined numerically** (§3.5): a substrate "shows the crossover" iff it passes the
continuous-resists gate **and** the discrete-determines gate; **cross-substrate identity holds iff S0
AND S1 both show it.**

### 3.5 FROZEN thresholds (gates)
- **Probe-power preflight (`λ=0`):** `cont(0) ≥ 0.70` AND `disc(0) ≥ 0.95` (else `void_underpowered`).
- **Continuous-resists gate:** `cont(λ_max) ≤ 0.10` AND `λ*_c` is **in-grid** (a finite half-life).
- **Discrete-determines gate:** `min_λ disc(λ) ≥ 0.95` across the **whole** grid (⇒ `λ*_d` censored).
- **Pre-registered prediction (set at freeze, after calibration):** `cont(λ)` monotone-decreasing
  (within tol) from `cont(0) ≥ 0.70` to `cont(λ_max) ≤ 0.10` with `λ*_c` in-grid; `disc(λ) ≥ 0.95` flat
  across the grid; **the same on S0 and S1**. S0 analytic anchor: `λ*_c` follows from the
  `exp(−2π²λ²t²)` fringe damping and the `env`/noise scale; S1 analytic anchor: the radial texture
  amplitude decays as `exp(−½λ²)` against the fixed noise scale; `disc` is `λ`-independent by
  construction on both substrates.

> **FROZEN PREDICTION — locked from seed-999 calibration (2026-06-07; S0 v2, `t0_f=0.50, w_f=0.18`,
> `σ_n=0.30`).** Committed *before* the `data_seed=20260605` run. On the `data_seed`, n=2000:
> - **S0:** `cont(0) ≈ 0.75` (plateau through `λ≤0.15`), monotone decay to `cont(λ=2.0) = 0`, `λ*_c =
>   0.75` (in-grid); `disc(λ)` flat `= 1.00` across the whole grid, `λ*_d` censored.
> - **S1:** `cont(0) ≈ 0.98`, monotone decay to `cont(λ=2.0) = 0`, `λ*_c = 1.5` (in-grid); `disc(λ)`
>   flat `= 1.00`, `λ*_d` censored.
> - **Cross-substrate identity = True** ⇒ expected verdict `operator_confirmed_synthetic` (closes the
>   *synthetic* anti-equivocation lacuna; **NOT Phase-5-complete, NOT public-eligible** — S2 still gated).
>
> Any deviation is read against §3.8 as-is — a falsified prediction is the cheap, valuable negative,
> not a thing to re-tune. Calibration is now CLOSED; no constant changes past this line.

### 3.6 Void / preflight gates (checked BEFORE the verdict tree)
| void branch | trip condition |
| --- | --- |
| `void_not_frozen` | any §3.7 constant unset, or the constants block not finalized |
| `void_label_leak` | `x_c*`, `x_d`, or `λ` appears in the feature vector (assert: features are exactly the `σ` components — no label/λ columns) |
| `void_underpowered` | `cont(0) < 0.70` OR `disc(0) < 0.95` (harness too weak to even read the variables at zero lossiness) |
| `void_class_imbalance` | `x_d` majority fraction ∉ `[0.45, 0.55]` (the `disc` baseline must be balanced) |
| `void_probe_nonconverge` | BOTH probe families fail to converge at `λ=0` (record convergence; a single non-converged MLP still scores) |
| `void_nondeterministic` | a re-run at the same seeds does not reproduce the curves within float tol |

### 3.7 FROZEN constants (smoke-calibrate on throwaway seed, then freeze)
Starting values (calibration targets — tune ONLY on the throwaway seed to hit §3.5's preflight + gates,
then freeze the final values with the prediction):
- **S0 (v2):** `T=64`, `t∈[−1,1]`, `t0=0`, `w=0.5` (parity `env_g`), `w_b=0.1`; **band-pass fringe
  `t0_f=0.50`, `w_f=0.18`** (size-bearing fringe off-centre, `≈0` at `t=0`, ~1.8 cycles in-band for
  preflight headroom); `A=1.0`, `C=1.0`, `D=0.5`, `f_p=8`; `x_c* ~ U[3,7]`; `K=64`; obs-noise
  `σ_n = 0.30` (calibrated).
- **S1:** `G=16`, `[−1,1]²`, `ε=0.10`; `A=1.0`, `B=1.0`, `f0=3.0`;
  `x_c* ~ U[−1.0,1.0]` radians; `K=64`; obs-noise (start `0.30`).
- **Shared:** `λ` grid `{0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00}` (calibrate `λ_max` so
  `cont(λ_max) ≤ 0.10`); `n = 2000` samples per `(substrate, λ)`; **CV = 4-fold**;
  `data_seed = 20260605`, `probe_seed = 0`; **throwaway calibration seed = 999, n=500**.
- **Probes (frozen):** linear = `LinearRegression` / `LogisticRegression(max_iter=2000)`;
  MLP = `MLP{Regressor,Classifier}(hidden_layer_sizes=(64,), max_iter=500, random_state=0)`;
  non-convergence is recorded, not fatal (a single non-converged MLP still contributes its score).
- **Calibration protocol:** on seed 999 only, adjust `{σ_n, amplitudes, λ_max, x_c* range, K, n}` until
  the preflight (`cont(0)≥0.70`, `disc(0)≥0.95`) and the gates land in-grid on BOTH S0 and S1; freeze
  the final constants + the §3.5 prediction; then run the frozen `data_seed`. Calibration may change
  scale/power only — not generator equations, thresholds, probe families, void gates, or verdict
  branches. **Never tune on `data_seed`.**

### 3.8 Verdict tree (after void gates; first match wins)
| branch | condition | reading |
| --- | --- | --- |
| `law_falsified_discrete_decays` | `disc(λ)` also decays (`min_λ disc < 0.95` with a finite `λ*_d`) on a passing-preflight substrate | the discrete variable is NOT structurally stable — **the law is wrong** (major banked negative) |
| `law_falsified_continuous_survives` | `cont(λ_max) > 0.10` (continuous not washed across the grid) | insufficient lossiness — re-examine the knob (NOT a confirmation) |
| `operator_partial` | gates pass on S0 but not reproduced on S1 | crossover real but not yet substrate-invariant |
| `operator_confirmed_synthetic` | continuous-resists AND discrete-determines gates pass on **S0 AND S1** | **measured shared operator on ≥2 structurally-different synthetic substrates** — closes the *synthetic* anti-equivocation lacuna. **NOT Phase-5-complete; NOT public-eligible.** |
| `operator_confirmed_physical` | S2 (halo) reproduces it under the wave/Stokes apparatus | the full measured operator-identity — **the Phase-5 Exit; public-eligible after evidence-tier review** |

### 3.9 Synthetic vs. physical status (load-bearing)
`operator_confirmed_synthetic` (S0+S1) is **distinct from full Phase-5 completion.** It discharges the
*synthetic* anti-equivocation question (the same operator on two structurally-different in-silico
substrates) — a real, banked result. **Full Phase-5 (the roadmap Exit) and any public claim require
S2**, the halo physical instantiation, which stays apparatus-gated behind the wave/Stokes HaloSim
tooling (not yet built). Do not promote `operator_confirmed_synthetic` to "Phase-5 complete."

### 3.10 Anti-p-hack discipline
Labels (`x_c*`, `x_d`) are **scoring-only**; the generator/regime is **not** tuned to produce the
crossover (only the throwaway-seed calibration tunes power, §3.7). Pre-register the prediction and the
verdict branches before the frozen run; report `law_falsified_*` plainly (the law is a conjecture —
the cheap negative is the valuable outcome). Deterministic (seed-pinned, byte-reproducible).
**S0/S1 inline-runnable** (numpy + sklearn, the JEPA-0D / Gate-0 preflight pattern); **S2
operator-staged.**

### 3.11 FROZEN RESULT (2026-06-07) — `operator_confirmed_synthetic`
Run: `scripts/pvnp_phase5_lossiness_crossover.py --frozen`, `data_seed=20260605`, `n=2000`, `K=64`,
`CV=4`, `σ_n=0.30`, S0 v2 (`t0_f=0.50, w_f=0.18`). Artifact:
`results/pvnp/phase5-lossiness-crossover/frozen.json`.

| substrate | `cont(0)` | `cont(λ=2.0)` | `λ*_c` | `min_λ disc` | `λ*_d` | gates |
| --- | --- | --- | --- | --- | --- | --- |
| **S0** | 0.885 | **0.000** | 0.75 (in-grid) | **1.000** | censored | resists ✓ determines ✓ |
| **S1** | 0.992 | **0.053** | 2.0 (in-grid) | **1.000** | censored | resists ✓ determines ✓ |

- **Cross-substrate identity = True.** `disc(λ)=1.000` at every λ on both substrates (the
  discrete/topological variable recovered with no held-out error — at the metric ceiling, all λ; *not*
  algebra-exactness); `cont(λ)` decays monotonically (within CV tol — sub-0.001 upticks at small λ) to
  ≤0.10 in-grid on both (the continuous variable is washed out by the lossy average).
  **Verdict: `operator_confirmed_synthetic`** — measured shared operator on two structurally-different
  synthetic substrates. Closes the *synthetic* anti-equivocation lacuna.
- **Prediction deviations (reported per §3.10, not re-tuned):** (1) S0 `cont(0)=0.885` vs predicted
  ≈0.75 — more probe power at n=2000, gate passes with more margin (favorable). (2) S1 `λ*_c=2.0` vs
  predicted 1.5 — the continuous variable resisted one grid-step longer; `cont` = 0.669 (λ=1.5) → 0.053
  (λ=2.0). The `cont(λ_max)≤0.10` gate passes (0.053), but the S1 washout is **boundary-tight**: it
  completes only at the last grid point. The pre-freeze λ-grid extension to 2.0 is load-bearing for the
  S1 leg. The *qualitative* law (continuous-resists / discrete-determines) held on both; the fine
  half-life estimate on S1 was off by one step.
- **Determinism (receipt, 2026-06-07).** All stochastic components are seeded (`default_rng(seed+·)`,
  `KFold/StratifiedKFold random_state=probe_seed`, `MLP random_state=0`). Two independent `--frozen`
  runs (`…/phase5-lossiness-crossover/frozen.json` and `…-repro/frozen.json`) are **byte-identical** on
  `cont`, `disc`, `maj`, `λ*_c`, `λ*_d`, the gate dict, and `cross_substrate_identity` (only wall-time
  differs: 256.8s vs 225.1s). Byte-reproducible — receipt confirmed, not merely structural.
- **STATUS GATE (load-bearing, §3.9):** this is **NOT Phase-5-complete and NOT public-eligible.** It
  discharges the *synthetic* question only. Full Phase-5 (the roadmap Exit) and any public claim require
  **S2** — the halo physical instantiation behind the wave/Stokes HaloSim layer (not yet built).

### 3.12 S2 — physical slate (PRE-REGISTERED, 2026-06-07; apparatus not yet run)
Structural pre-registration of the halo physical leg, written **before** the forward model is built or
calibrated. Grounded in `docs/atlas/S2_LITPASS_E_G.md` (Stage-0 lit-pass). Numeric constants + the
frozen prediction are set later, on throwaway seed 999, and frozen before the `data_seed` run (§3.7
discipline carries verbatim). **Binding rule:** the forward-model *physics equations* (corona Airy,
min-deviation halo radius, Fresnel×retarder Mueller chain) are **fixed by cited literature and may NOT
be tuned**; only power knobs `{obs-noise, n, K, λ_max, population-width, x_c range}` calibrate.

**3.12.1 Architecture — two halo/diffraction-family legs (honest relaxation).** The legible *continuous*
and *discrete* observables are physically distinct phenomena, so S2 does **not** reproduce S0/S1's
single-shadow simultaneity (both variables on one shadow). It demonstrates **each half of the law on
its appropriate physical substrate**, under population-spread lossiness:
- **S2c — fold-Airy parhelion supernumerary (continuous-resists leg; size PHASE-encoded).** `x_c` =
  crystal size `d*`, carried by the supernumerary fringe scale `κ = c_κ·d` of a fold caustic dressed
  by the Airy function, `Ī(θ) = mean_i Ai(−κ_i·(θ−θ_edge))²`, subunit sizes `d_i = d*·(1+λ·ξ_i)`
  (λ = relative size-spread = lossiness knob). The fold **edge `θ_edge` is size-independent**, so size
  lives only in the fringes and washes by destructive interference as λ grows (Berry & Upstill; Berry
  1994 parhelion supernumeraries, faint, contrast ~0.178). **Calibration finding (the graded-resistance
  refinement):** the fringes wash fully, but `cont` **plateaus at ~0.31** rather than → 0, because size
  is a *scale/magnitude* and the Airy envelope's decay-scale near the edge (`∝ 1/√d`) leaks a *rough*
  size that ensemble-averaging cannot erase. Pushing below the 0.10 gate requires a contrivance (a
  near-floor size range so the spread piles at the Mishchenko–Macke existence floor) — **rejected as
  engineered**; the honest reading is *partial* resistance. (The corona is identical in this respect —
  both diffraction size-readouts plateau at ~0.31.)
- **S2h — refraction halo (discrete-determines leg).** `x_d ∈ {±1}` **shared** across the population.
  Primary: **ice phase** (`+1` = hexagonal, 22° radius; `−1` = pyramidal/odd-radius, ~28°), read off
  the halo radius via min-deviation `δ_min(n,A)` — pure geometry, robust. Secondary (flagged
  *predicted/novel*): **handedness** = sign of population Stokes-`V` from the Fresnel×birefringent-
  retarder Mueller chain (`V/I ∝ sin2φ·sinδ`, `δ=2πΔn·L/λ_light`, ice `Δn=+0.0014`, `n≈1.31`). Shadow
  `σ̄` = ensemble-averaged halo radial profile (ice-phase) or stacked `(Ī,Q̄,Ū,V̄)` (handedness) over
  `K` subunits sharing `x_d`, with orientation/size spread λ. The shared discrete sign survives the
  average (the S0/S1 mechanism), so `disc(λ)` must stay flat.

**3.12.2 Metrics + features.** Metrics unchanged (§3.3): `cont = max(0,R²_cv)`, `disc =
(acc−maj)/(1−maj)`, best-of {linear, MLP}. Features `σ̄` are exactly the shadow-profile samples — **no
size value, no parity sign, no λ column** (`void_label_leak`, §3.6). Probes read `x_d` off V's **sign
pattern** / the radius class, `x_c` off the supernumerary fringe structure.

**3.12.3 Gates (re-spec for the split — the one deviation from §3.5's single-shadow form).** Because the
two halves live on different legs, the preflight + gates apply **per leg**:
- **S2c:** preflight `cont(0) ≥ 0.70`; continuous-resists `cont(λ_max) ≤ 0.10` AND `λ*_c` in-grid.
- **S2h:** preflight `disc(0) ≥ 0.95`; discrete-determines `min_λ disc(λ) ≥ 0.95` (`λ*_d` censored).
- Void gates (§3.6) carry per leg; add **`void_size_floor`** — every size-leg subunit must satisfy
  the halo/diffraction existence floor `2π a/λ_light ≳ 100` ⇒ `a ≳ 10 µm` (Mishchenko–Macke 1999),
  else the leg is physically meaningless.

**3.12.4 Frozen prediction (shape; numbers locked at freeze, post-seed-999).** `cont_S2c(λ)` monotone
(within tol) from `cont(0) ≈ 0.96` **decaying to a plateau `≈ 0.30` at `λ_max`** (NOT ≤0.10 — the
magnitude scale-leak; the strict continuous-resists gate is *predicted to FAIL*), `λ*_c ≈ 1.5`;
`disc_S2hp(λ)` and `disc_S2hh(λ)` flat `= 1.00` (radius class / V-sign λ-independent by physics).
**Graded-resistance refinement (pre-registered reading):** phase-*offset* continuous variables (S1)
wash to ~0; *scale/magnitude* variables (size, here) resist only partially. Expected verdict =
**partial physical leg** (continuous-physical partial, discrete-physical confirmed).

**3.12.5 Verdict (extends §3.8).** `operator_confirmed_physical` requires **S2c passes continuous-
resists AND S2h passes discrete-determines.** Sub-cases, reported honestly:
- ice-phase S2h passes ⇒ the **discrete physical anchor** §4 leans on is banked (robust, geometric).
- handedness S2h passes ⇒ a **predicted, not previously observed** physical effect is exhibited (own
  the novelty; no halo-Stokes-V observation exists in the literature — §3.12.6).
- S2c passes preflight but `cont(λ_max) > 0.10` (washes only to the magnitude-scale-leak plateau ~0.31)
  ⇒ **continuous-physical PARTIAL** ⇒ S2 is a **partial physical leg** (discrete-physical confirmed,
  continuous-physical *partial — strong but not full resistance*); do **not** promote to full
  `operator_confirmed_physical`. This is the **pre-registered expected outcome** (§3.12.4).
- S2c fails preflight (size not legible even monodisperse) ⇒ `void_underpowered` on the continuous leg
  ⇒ partial physical leg (continuous-physical *owed*).

**3.12.6 Honest status (load-bearing).** (a) S2 is a **two-leg** demonstration, weaker than S0/S1's
single-shadow crossover — the relaxation is forced by physics and is disclosed, not hidden. (b) The
**handedness leg is a novel predicted observable**: there is *no* published measurement of nonzero
Stokes-`V` in a visible ice halo, and "net-`V` = population handedness" is an uncited Sundog framing
(`SYNTHESIS`); ice-phase is the robust primary, handedness the flagged deep target. (c) S2 remains a
**forward-model simulation** of real optics (not photographs) — `operator_confirmed_physical` via S2 is
the apparatus tier; measured-sky polarimetry would be a further, higher tier. (d) Until run, S2 is
unstarted; nothing here promotes Phase-5 past §3.9's status gate.

### 3.13 S2 FROZEN RESULT (2026-06-07) — `partial physical leg`
Run: `scripts/pvnp_phase5_lossiness_crossover.py --frozen --s2`, `data_seed=20260605`, `n=2000`, `K=64`,
`CV=4`. Forward model: `scripts/s2_optics.py` (physics fixed, unit-tested). Artifact:
`results/pvnp/phase5-lossiness-crossover/frozen_s2.json`.

| leg | metric(0) | metric(λ=2.0) | `λ*` | gate |
| --- | --- | --- | --- | --- |
| **S2c_fold** (continuous, size) | `cont0 = 0.968` | `cont = 0.450` | `λ*_c = 2.0` | resists **✗ (partial)** |
| **S2hp_phase** (discrete, ice-phase) | `disc0 = 1.000` | `min disc = 1.000` | `λ*_d` censored | determines **✓** |
| **S2hh_hand** (discrete, handedness) | `disc0 = 1.000` | `min disc = 1.000` | `λ*_d` censored | determines **✓** |

- **Verdict: `partial physical leg`** (the §3.12.5 pre-registered expected outcome). **Discrete-physical
  CONFIRMED** — `disc(λ)=1.000` flat across the whole grid on BOTH the ice-phase (robust geometry) and
  the handedness (Stokes-`V`) legs: the discrete/topological invariant is exactly determined from the
  lossy halo shadow, regardless of population spread. This is the **physical anchor §4 leans on**, now
  on real halo optics (min-deviation geometry + Fresnel×birefringent Mueller chain). **Continuous-
  physical PARTIAL** — size `cont` decays 0.968 → 0.450 (monotone, `λ*_c=2.0`) but does **not** reach the
  `≤0.10` continuous-resists gate.
- **Graded-resistance refinement (the banked finding):** *phase-offset* continuous variables wash to ~0
  (S1: cont 0.992→0.053); *scale/magnitude* variables resist only **partially** (size: cont 0.968→0.450,
  residual scale-leak in the diffraction envelope `∝1/√d`). The law's "continuous resists" is **graded
  by encoding**, not absolute. Confirmed identically on the corona and the fold-Airy substrates.
- **Prediction deviations (reported per §3.10, not re-tuned):** the continuous plateau is **0.45 vs the
  predicted ~0.30** (it resisted *less*; the linear probe reads the residual envelope better at n=2000),
  and `λ*_c=2.0 vs predicted 1.5`. The *qualitative* prediction (continuous partial, discrete flat,
  verdict = partial physical leg) held exactly.
- **Receipts.** (1) **Determinism:** two independent `--frozen --s2` runs
  (`…/frozen_s2.json` and `…-s2repro/frozen_s2.json`) are **byte-identical** on all `cont`/`disc`/`maj`/
  `λ*` curves + the verdict (only wall-time differs, 471.8 vs 495.2 s). (2) **Shuffled-parity null**
  (`scripts/s2_null_check.py`): on both discrete legs at λ=0 and λ=2, `disc(true)=1.000` while
  `disc(shuffled)≈0` (0.045, −0.031, −0.005, 0.031) — the `disc=1.000` is a **real** x_d↔shadow
  correlation, not a label leak.
- **STATUS GATE (load-bearing).** S2 is a **PARTIAL** physical leg, **NOT** full
  `operator_confirmed_physical`, **NOT** Phase-5-complete, **NOT** public-eligible. It banks: (i) the
  discrete-determines half on real halo physics (the §4 anchor); (ii) the graded-resistance refinement.
  It does **not** bank a full continuous washout (owed, or accepted as physically partial for magnitude
  variables), and the handedness leg remains a **predicted/unobserved** observable (no halo-Stokes-`V`
  measurement exists). Forward-model tier; measured-sky polarimetry is a further tier.

## 4. Dig-in: where alignment sits relative to the law (the founding-theorem correction)

This is the payoff that makes the law load-bearing, not just elegant — stated as the law's
**prediction**, not a closed verdict. The **founding Sundog Alignment Theorem bet on `x_c`** — that the
*continuous, embodied body* would *resist* reward-hacking. The law **predicts** the continuous body is
the **resistant** side; the three-for-three marginal body-resistance results (mesa PR≈2 / FVE 0.97–0.99,
NS-C1 FVE~0.99, Sabra eff-rank 1.7/30) are **consistent** with the near-injective reading — a
near-injective body is a low-lossiness shadow, where `x_c` has not yet washed out and stays readable.
But the causal step — that near-injectivity *caused* the marginality — is **not yet tested.** It needs
an intervention on the actual bodies: **registered prediction** — artificially raise the ensemble
spread (lossiness) on the mesa / NS / Sabra control shadow and the continuous-resists / discrete-
determines separation should *sharpen*; absent that intervention this is a consistency argument, not a
demonstrated cause. So, on the law as currently conjectured (synthetic receipt §3.11, physical S2
owed): **the founding theorem appears to have bet on the resistant side of its own law** — a reframing
the law predicts and the marginal results fit, pending the intervention test and S2.

The mature, earned property is **far more cleanly carried by the other** side: the alignment-relevant
object is a **discrete/algebraic invariant** — a certificate, a coset, a parity — the **determinable**
side. ("Cleanly carried by," not "must live on": the law itself forces the hedge via §5's own
falsifier — a continuous parameter read *through* a discrete encoding can survive averaging, so the
discrete/continuous boundary is encoding-dependent.) Two senses of *resist* are also in play and are
**not** the same property: the law's sense is **resistance to epistemic recovery from a lossy shadow**;
the founding theorem's sense was **resistance to reward-hacking.** The bridge from one to the other is
the **certificate-verifier framing** — a discrete invariant an attacker cannot forge *is* a
hacking-robust target — not the law alone. The certificate lane's measured positive (the syndrome's
spoof-resistance, the capacity-relative one-way threshold) is exactly such a discrete-algebraic
invariant that the lossy shadow **determines** and that an attacker cannot forge. So:

> **Founding bet:** alignment via continuous body-resistance (the `x_c` / resistant side) → marginal.
> **Corrected claim:** alignment via discrete-invariant determination (the `x_d` / determinable side —
> the certificate, the parity, the topological invariant) → the side with the lab's only *exact*
> results (AB, syndrome) and its only clean physical demo (halo handedness).

The halo handedness layer is the *photographable proof of concept* that discrete invariants are
cleanly determinable from a lossy shadow — the optical witness for the corrected alignment claim.

**The law's mechanism now has a falsifiable synthetic receipt.** §3.11's frozen run exhibits the
dichotomy as a measurement that *could have failed and did not* (the `law_falsified_*` branches were
live): across the lossiness grid, on two structurally-different substrates, `disc(λ)=1.000` — the
discrete invariant recovered with **no held-out error** (n=2000, all λ — at the metric ceiling, *not*
the algebra-exactness of the AB/syndrome rows) — while `cont(λ)` decays to chance. Two cautions on what
this does and does **not** show:
> 1. The generators were **built** to instantiate the split (the discrete sign factors out of the
>    average; the continuous part Debye–Waller / amplitude-decays). A clean crossover therefore confirms
>    the **law's mechanism is internally coherent on a constructed example** — it contains **no trained
>    body** and cannot, by itself, evidence that real trained bodies are near-injective or that the
>    founding theorem's bet was wrong. That is the separate, still-untested intervention claim above.
> 2. The measured `x_d` is a parity / winding sign — a generic topological toy. Whether a specific
>    **alignment** invariant (certificate / coset) is `P`-structurally-stable *in the same sense* is a
>    separate assumption, anchored by the AB/syndrome exact results, **not** demonstrated by these toys
>    (and the syndrome's exactness is a crypto-trapdoor mechanism, distinct from structural-stability-
>    under-averaging).

So the receipt anchors the *logic* of the correction — the split is real and sharp where the mechanism
is present — and nothing more. It is **synthetic** (S0 + S1), excludes the physical halo leg (S2), and
even on S1 the continuous-resists gate cleared only at the last grid point (`cont=0.053` at λ=2.0 —
boundary-tight; §3.11). The AB/syndrome exact results and the halo-handedness demo are the correction's
*physical* anchors.

**S2 update (2026-06-07, §3.13): a partial physical leg, and it lands on the correction's load-bearing
side.** The halo forward model ran as a `partial physical leg`: the **discrete-determines half is now
anchored on real halo physics** — `disc(λ)=1.000` flat on both the ice-phase (min-deviation geometry)
and handedness (Fresnel×birefringent Mueller, Stokes-`V`) legs, shuffled-parity-null-verified and
determinism-receipted. That is precisely the side §4 says alignment-relevant structure lives on, now
measured (in forward-model simulation) on a third, physical substrate. The **continuous side resisted
only partially** (size `cont` 0.97→0.45, magnitude scale-leak) — which *refines* rather than weakens the
correction: "continuous resists" is **graded by encoding** (phase-offsets wash to ~0; magnitudes resist
partially, their mean surviving averaging), and the determinable discrete invariant is the clean,
robust side on every substrate tried. Still owed for a *full* physical discharge: a clean continuous-
physical washout and measured-sky (not forward-model) polarimetry; the handedness leg also remains a
*predicted, unobserved* observable.

## 5. Falsification surface

- **A discrete `x_d` the lossy shadow determines but that is NOT structurally stable under `P`** (the
  classes overlap at the ensemble's support) — would break the determination half.
- **A continuous `x_c` that survives averaging** via a hidden topological encoding (a continuous
  parameter read through a discrete invariant) — would blur the dichotomy; the law would need the
  "kind" defined post-encoding.
- **No measured crossover** (continuous and discrete decay together, or neither) — kills the operator
  on that substrate. *This falsifier did not fire on the synthetic substrates (§3.11, both passed);
  it remains live for S2.*
- **Lossiness without resistance / injective with resistance** — would refute the essential-lossiness
  clause.

**What the synthetic slate actually exercised:** only the third falsifier (no measured crossover) — and
it did not fire (§3.11). Falsifiers 1, 2, and 4 are **not testable on generators built to embody the
dichotomy** (the toy's `x_d` is shared/structurally-stable by construction, so `law_falsified_discrete_
decays` *cannot* fire; no hidden topological encoding of `x_c` was instantiated; lossiness and
resistance co-vary by design). They remain **entirely open**, and probing them is part of what S2 — and
deliberately adversarial substrate design — must do. The synthetic pass is not a pass of the surface.

## 6. Honest boundary

The *pieces* are borrowed and known — Thom (structural stability), Nye (wave dislocations), Blackwell
(sufficiency), lossy-trapdoor one-wayness (crypto). The **claim** is the *synthesis*: that one
operator, with one mechanism (structural stability under lossy averaging) and two knobs (lossiness,
kind), governs the invertibility split across the whole portfolio, and that the marginal/exact
spectrum the program measured *is that operator seen from different lossiness*. This is a **conjecture
with a falsifier**, not a closed theorem. The measured cross-substrate crossover is now **delivered on
the synthetic substrates** (§3.11, `operator_confirmed_synthetic` — two structurally-different
substrates, not a shared word) and **still owed on the physical one** (S2). The physical crossover
remains required before any Phase-5-complete or public claim. This REFRAMES, and does not yet
discharge, the roadmap's Phase-5 Exit.

## 7. Cross-references

- `../COARSE_GRAINING_PROOF_ROADMAP.md` §Phase 5 — the slot this fills (as a candidate) and reframes.
- `../CROSS_SUBSTRATE_NOTES.md` §6.3 — the bridging table + the three-for-three + the AB exact case;
  this law is the candidate *content* of the "shared operator, not a shared word."
- `../SUNDOG_V_ATLAS.md` §1.2–1.3 — the determining-shadow tower; the cleanest test bed (§3).
- `pvnp/SUNDOG_CERTIFICATE_PROBLEM.md` + `pvnp/DIRECTIONB_GATE0_NOTE.md` — the syndrome (exact),
  the gradient-barrier null, and the alignment-side correction (§4).

---

*Sundog Research Lab — Phase-5 candidate operator. The shadow-invertibility law: lossy averaged
shadows determine the structurally-stable (discrete/topological) part of a state and resist the
continuous part; lossiness is essential. Synthetic cross-substrate crossover delivered (§3.11, S0+S1,
2026-06-07); physical (S2) crossover still owed. Conjecture with a falsifier; not Phase-5-complete; no
public claim until the physical leg is measured.*
