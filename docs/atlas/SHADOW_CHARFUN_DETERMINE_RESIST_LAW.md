# The determine/resist dichotomy = the characteristic-function spectrum of the averaging kernel

> **STATUS: BOUNDED-POSITIVE, 2026-06-08.** A mechanistic *sharpening* of the Shadow-Invertibility
> Law (`ATLAS_PHASE5_CROSS_SUBSTRATE.md`): it names **what in the population `P` governs the
> resist/determine split**. Two legs, both clean. Empirical leg = internal (frozen-as-portfolio,
> NOT public-eligible). Lean leg = a generic-math **candidate 5th public pillar** for `sundogcert`,
> built + axiom-clean locally, **pending owner review before any push**. Hypothesis #2 of the
> fresh-slate (`ww6koomb1`). Attribution: Debye (1913) / Waller (1923); the Riemann–Lebesgue lemma;
> Lukacs, *Characteristic Functions*; the Cauchy/Poisson-kernel charFun.

## 0. The claim

The Shadow-Invertibility Law says a lossy *averaged* shadow **determines** a discrete/topological
hidden variable and the continuous-magnitude one **resists**, and that *the lossiness is essential*.
It did not say **which property of the averaging population** does the work. This result does:

> **The charFun-spectrum law.** Averaging the continuous fringe `cos(2π(c + λ x) t)` over a population
> `x ∼ μ` factors, for **every** probability measure `μ`, through `μ`'s **characteristic function**
> `φ_μ = charFun μ`:
>
> &nbsp;&nbsp;&nbsp;&nbsp; `∫ cos(2π(c + λ x) t) dμ = Re[ exp(2π i c t) · φ_μ(2π λ t) ]`.
>
> From this single identity the two halves split into **two different spectral conditions**:
> - **RESIST** (the continuous `c` washes out as `λ → ∞`) ⟺ **`‖φ_μ(s)‖ → 0` as `|s| → ∞`** — the
>   Riemann–Lebesgue property of any absolutely-continuous `μ`. *Governed by the charFun tail, NOT
>   the variance.*
> - **DETERMINE** (a shared label survives the average) ⟺ **`μ` has a finite centered mean**
>   `∫ x dμ = 0`, integrable. *Governed by the first moment.*
>
> These are **independent**: neither condition implies the other. The **Cauchy** population is the
> witness — it *resists* (`φ = exp(-γ|s|) → 0`) yet *cannot determine* (no mean; its sample average is
> itself Cauchy and never concentrates). The Gaussian's Debye–Waller damping `exp(-2π²λ²t²)` is just
> `Re φ_μ` for `μ = N(0,1)`; the law is the same shape for every `μ`.

The Gaussian special case was already proved (`Sundogcert.ShadowDecay`, `pvnp_phase5_…` S0). This
generalizes both the **mechanism** (charFun) and the **empirical sweep** (population swap).

## 1. Empirical leg (the falsifiable core) — `scripts/shadow_charfun_populations.py`

Swap the per-subunit averaging population `ξ` in the frozen S0 lossiness-crossover (`gen_s0`) from
Gaussian to {uniform, Cauchy, lattice ±1}, all unit-scale, and rerun the **unchanged** cont/disc
recovery apparatus across the frozen `LAMBDAS` grid. `gen_s0_pop(pop='gaussian')` reproduces the
frozen `gen_s0` code path exactly (the control).

### Pre-registration (kill criteria, set before running)
- AC populations {gaussian, uniform, **Cauchy**}: `φ_μ → 0` ⟹ continuous recovery **washes**
  (`cont(λ=2) ≤ 0.10`, finite half-life in grid).
- LATTICE (`φ = cos`, recurs to ±1): continuous recovery **survives** (`cont(λ=2) > 0.30`, half-life
  censored — never washes).
- **KILLED IF** the lattice cont collapses to ~0 like Gaussian (⟹ charFun does *not* govern — the
  mechanism would be raw variance), **OR** an AC finite-variance population (uniform) fails to resist
  (⟹ contradicts Riemann–Lebesgue).

### Result — both kill criteria avoided (BOUNDED-POSITIVE)

| population | `φ_μ(s)` | cont(λ=2) | half-life λ\*_c | verdict |
|---|---|---|---|---|
| gaussian | `exp(-s²/2)` | 0.000 | 0.75 | RESISTS |
| uniform | `sinc(√3 s)` | 0.000 | 0.75 | RESISTS |
| **cauchy** (∞ variance) | `exp(-\|s\|)` | 0.000 | 0.5 | **RESISTS** |
| **lattice** ±1 | `cos(s)` | **0.655** | **None** (censored) | **SURVIVES** |

- **Cauchy resists despite infinite variance** — the surprise that isolates *charFun decay* from
  *variance* as the governing quantity.
- The analytic `‖φ_μ‖` envelope at the fringe-peak probe **tracks** the empirical cont curve; the
  lattice cont visibly **dips at λ=0.5** (where `cos(π/2)=0`) then **recovers at λ=0.75→1.0** (where
  the cosine recurs to 1) — the resonant-recurrence signature, exactly as pre-registered.

### The DETERMINE half — concentration probe (law of large numbers holds ⟺ finite mean)

Median `|avg − d|` of a shared label `d=1` plus averaged per-unit noise, over growing `K`:

| population | K=16 | K=64 | K=256 | K=1024 | ratio | verdict |
|---|---|---|---|---|---|---|
| gaussian | 0.174 | 0.083 | 0.042 | 0.022 | 8.1× | DETERMINES (LLN) |
| uniform | 0.167 | 0.085 | 0.042 | 0.021 | 8.1× | DETERMINES (LLN) |
| lattice | 0.125 | 0.094 | 0.039 | 0.021 | 5.8× | DETERMINES (LLN) |
| **cauchy** | 0.985 | 0.973 | 1.001 | 0.984 | **1.0×** | **BREAKS** (no mean) |

Cauchy's K-average is *itself* Cauchy (a stable law) — it never concentrates, no matter how many units
are averaged. **The separator is concrete: Cauchy resists yet breaks determine.**

Frozen test `scripts/test_shadow_charfun_populations.py` (12/12 PASS) locks: AC-resist, Cauchy-resist-
despite-∞-variance, lattice-survive, lattice resonant-recurrence, finite-mean-determines,
Cauchy-breaks-determine, analytic-envelope match.

## 2. Lean leg (the deductive law) — `sundogcert/Sundogcert/ShadowDecayGeneral.lean`

Generalizes the published `ShadowDecay.lean` (Gaussian-only) to an abstract probability measure `μ`.
Built locally on Lean 4.30.0 + mathlib v4.30.0; **full `lake build` clean (2884 jobs, 0 warnings),
every theorem axiom-clean** (`[propext, Classical.choice, Quot.sound]`, no `sorryAx`).

| theorem | statement |
|---|---|
| `shadow_decay_general` / `shadow_decay_charFun` | the abstract identity `∫ cos(2π(c+λx)t) dμ = Re[exp(2πi c t)·charFun μ (2πλt)]`, ANY `μ` |
| `general_recovers_debye_waller` | consistency lock: specialized to `N(0,1)` it reproduces the proven Gaussian Debye–Waller RHS (kernel-caught drift guard) |
| `resistance_general` | from the **named** hypothesis `‖charFun μ‖ → 0`: the shadow `→ 0` as `λ → ∞` (`t>0`). Bound `\|shadow\| ≤ ‖charFun μ (2πλt)‖` (unit-modulus phasor) |
| `gaussian_charFun_tendsto_zero` + `gaussian_resists` | the Gaussian **discharges** resist (`‖charFun‖ = exp(-s²/2) → 0`), recovering `ShadowDecay.resistance` |
| `determination_general` | from the **named, SEPARATE** hypotheses `Integrable id μ ∧ ∫ x dμ = 0`: `∫ (d+λcx) dμ = d` |
| `gaussian_determines` | the Gaussian **discharges** determine, recovering `ShadowDecay.determination` |
| `gaussian_resist_and_determine` | the dichotomy capstone: the Gaussian discharges **both** spectral conditions (stated as separate hypotheses precisely because they are different facts about `μ`) |

**What is NOT formalized (named, honest boundary):** the **Cauchy instance**. mathlib has
`ProbabilityTheory.cauchyMeasure` but **neither** its charFun (`exp(-γ|s|)`, a from-scratch
Poisson-kernel computation — mathlib has the Poisson kernel but no `charFun_cauchy`) **nor** the
non-integrability of `id` against it. So the Cauchy is the *motivating* separator — established
**empirically** (§1) and named as a candidate next sub-pillar — not asserted as a theorem. Same
discipline as `ShadowDecay`'s imported wall: the mechanism is proved, the un-built instance is named.

## 3. What this buys, and what it does not

- **Buys:** the Shadow-Invertibility Law's resist/determine split is no longer a qualitative "lossy
  averaging smears the continuous" — it is the **decay of the population's characteristic function**
  (resist) vs its **finite mean** (determine), two independent spectral conditions, with a concrete
  separator (Cauchy). The Gaussian was a single point; this is the law on the whole space of
  populations, proved abstractly and swept empirically.
- **Does not buy:** a new physical claim. The S0 substrate is a synthetic fringe toy; the physical
  (S2 halo) leg of Phase 5 is still owed (`ATLAS_PHASE5_CROSS_SUBSTRATE.md`). The Cauchy charFun and
  its non-integrability remain future Lean work. The empirical leg stays internal/frozen-as-portfolio;
  the Lean leg is public-eligible **pending owner review** (mirrors the `ShadowDecay`/`HaloGeometry`
  de-Phase-5 publication discipline — the file already uses generic Debye–Waller / charFun language).

## Files
- `scripts/shadow_charfun_populations.py` — population-swap lossiness sweep + analytic envelope + the
  determine-half concentration probe.
- `scripts/test_shadow_charfun_populations.py` — frozen test (12/12 PASS).
- `sundogcert/Sundogcert/ShadowDecayGeneral.lean` — the abstract charFun law (7 theorems, axiom-clean,
  wired into the `Sundogcert` root).
