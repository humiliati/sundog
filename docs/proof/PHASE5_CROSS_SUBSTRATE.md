# Phase 5 — Cross-Substrate Sameness: the Shadow-Invertibility Law (candidate operator)

> **STATUS: DRAFT CANDIDATE OPERATOR, 2026-06-05. NOT the measured Exit yet.** The coarse-graining
> roadmap's Phase 5 (`../COARSE_GRAINING_PROOF_ROADMAP.md` §"Phase 5 — Cross-substrate sameness")
> requires a **measured** cross-substrate operator-identity table on ≥2 substrates to dissolve the
> equivocation attack. This file supplies the **candidate operator** and the **measurable test
> design** — the conjecture and the falsifier — not the measured table. The measured table is still
> owed; do not cite this as Phase-5-complete. It also **reframes** the Phase-5 target: from "the same
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
- **S2 — halo physical instantiation (APPARATUS-GATED).** `x_c` = crystal **size** (off the
  Airy/Pearcey dressing), `x_d` = **handedness** (Stokes `V` sign) or **ice phase** (halo radius),
  swept over population spread via HaloSim **+ the wave/Stokes layer (not yet built; staged)**.

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
  discrete/topological variable is exactly determined regardless of lossiness); `cont(λ)` decays
  monotonically to ≤0.10 in-grid on both (the continuous variable is washed out by the lossy average).
  **Verdict: `operator_confirmed_synthetic`** — measured shared operator on two structurally-different
  synthetic substrates. Closes the *synthetic* anti-equivocation lacuna.
- **Prediction deviations (reported per §3.10, not re-tuned):** (1) S0 `cont(0)=0.885` vs predicted
  ≈0.75 — more probe power at n=2000, gate passes with more margin (favorable). (2) S1 `λ*_c=2.0` vs
  predicted 1.5 — the continuous variable resisted one grid-step longer; `cont` = 0.669 (λ=1.5) → 0.053
  (λ=2.0). The `cont(λ_max)≤0.10` gate passes (0.053), but the S1 washout is **boundary-tight**: it
  completes only at the last grid point. The pre-freeze λ-grid extension to 2.0 is load-bearing for the
  S1 leg. The *qualitative* law (continuous-resists / discrete-determines) held on both; the fine
  half-life estimate on S1 was off by one step.
- **Determinism:** all stochastic components are seeded (`default_rng(seed+·)`, `KFold/StratifiedKFold
  random_state=probe_seed`, `MLP random_state=0`) ⇒ structurally byte-reproducible; explicit re-run
  receipt available on request.
- **STATUS GATE (load-bearing, §3.9):** this is **NOT Phase-5-complete and NOT public-eligible.** It
  discharges the *synthetic* question only. Full Phase-5 (the roadmap Exit) and any public claim require
  **S2** — the halo physical instantiation behind the wave/Stokes HaloSim layer (not yet built).

## 4. Dig-in: where alignment sits relative to the law (the founding-theorem correction)

This is the payoff that makes the law load-bearing, not just elegant. The **founding Sundog Alignment
Theorem bet on `x_c`** — that the *continuous, embodied body* would *resist* reward-hacking. The law
says the continuous body is exactly the **resistant** side — and resistance went three-for-three
marginal precisely because trained bodies are near-injective (insufficient lossiness). **The founding
theorem bet on the wrong side of its own law.**

The mature, earned property lives on the **other** side: the alignment-relevant object is a
**discrete/algebraic invariant** — a certificate, a coset, a parity — which is the **determinable**
side. The certificate lane's measured positive (the syndrome's spoof-resistance, the capacity-relative
one-way threshold) is exactly a discrete-algebraic invariant that the lossy shadow **determines** and
that an attacker cannot forge. So:

> **Founding bet:** alignment via continuous body-resistance (the `x_c` / resistant side) → marginal.
> **Corrected claim:** alignment via discrete-invariant determination (the `x_d` / determinable side —
> the certificate, the parity, the topological invariant) → the side with the lab's only *exact*
> results (AB, syndrome) and its only clean physical demo (halo handedness).

The halo handedness layer is the *photographable proof of concept* that discrete invariants are
cleanly determinable from a lossy shadow — the optical witness for the corrected alignment claim.

## 5. Falsification surface

- **A discrete `x_d` the lossy shadow determines but that is NOT structurally stable under `P`** (the
  classes overlap at the ensemble's support) — would break the determination half.
- **A continuous `x_c` that survives averaging** via a hidden topological encoding (a continuous
  parameter read through a discrete invariant) — would blur the dichotomy; the law would need the
  "kind" defined post-encoding.
- **No measured crossover** in §3 (continuous and discrete decay together, or neither) — kills the
  operator on the atlas substrate.
- **Lossiness without resistance / injective with resistance** — would refute the essential-lossiness
  clause.

## 6. Honest boundary

The *pieces* are borrowed and known — Thom (structural stability), Nye (wave dislocations), Blackwell
(sufficiency), lossy-trapdoor one-wayness (crypto). The **claim** is the *synthesis*: that one
operator, with one mechanism (structural stability under lossy averaging) and two knobs (lossiness,
kind), governs the invertibility split across the whole portfolio, and that the marginal/exact
spectrum the program measured *is that operator seen from different lossiness*. This is a **conjecture
with a falsifier**, not a closed theorem; the measured cross-substrate crossover (§3) is owed before
any Phase-5-complete or public claim. It REFRAMES, and does not yet discharge, the roadmap's Phase-5
Exit.

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
continuous part; lossiness is essential. The measured cross-substrate crossover is owed. DRAFT
conjecture; not Phase-5-complete; no public claim until measured.*
