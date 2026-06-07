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

> **STATUS: SLATE DRAFT — freeze §3.2–§3.8 before any run.** Anti-p-hack per the lane (freeze regime
> + prediction + verdict tree before execution; labels scoring-only; report the null plainly).

### 3.1 The single question
As a shadow's **lossiness** (ensemble spread) increases, does recovery of a **continuous** hidden
variable decay to chance while recovery of a **discrete** hidden variable stays exact — and does the
**same crossover** appear across ≥2 structurally-different substrates? A yes is the measured shared
operator (the Phase-5 Exit); the *cross-substrate sameness of the crossover* is the anti-equivocation
content (one operator, not one word).

### 3.2 Hidden state + two probes (frozen)
`x = (x_c, x_d)`: `x_c` continuous (a "size"-like magnitude), `x_d ∈ {±1}` discrete (a
"handedness"-like `Z₂` parity). Shadow `σ` = ensemble-average over a population of `K` sub-units (each
sub-unit carries its own `x_c,i`; `x_d` is **shared** — a structural property of the population). Two
**model-free** probes (linear + one-hidden-layer MLP, the preflight discipline): a continuous-recovery
probe (`σ → x_c`) and a discrete-recovery probe (`σ → x_d`), each scored against its own
majority/chance baseline.

### 3.3 The lossiness knob (frozen)
`λ` = the population spread of `x_c`: `x_c,i = x_c* + λ·ξ_i`, `ξ_i ~ N(0,1)`; `x_d` shared. Swept on a
frozen grid `λ ∈ {0, …, λ_max}`. Lossiness = how much the average smears the continuous signature.

### 3.4 Three substrates, staged by cost
- **S0 — synthetic caustic toy (INLINE-runnable, model-free, ~minutes; the operator's core test).**
  `σ(t)` = a scale-free gross feature (a fixed bump — carries *nothing*, models the geometric shadow)
  **+** a fringe dressing `cos(x_c,i·(t−t0))·env(t)` (the size signature; averaging gives
  `cos(x_c*·(t−t0))·exp(−½λ²(t−t0)²)` — the fringe **damps with λ**, Debye–Waller-like) **+** a parity
  channel `x_d·sin(ω0(t−t0))·env(t)` (handedness; the **sign survives averaging**). Continuous recovery
  rides the washing-out fringe; discrete rides the surviving parity. The faithful controlled instance.
- **S1 — a second synthetic substrate of different mechanism (INLINE-runnable; the cross-substrate
  leg).** A structurally-different generator — e.g. a "field" substrate where `x_d` is the **sign of a
  circulation / a winding number** and the lossy channel is **additive mixing**, not fringe-damping. If
  the crossover *shape* matches S0 despite the different mechanics, the operator is substrate-invariant
  — the anti-equivocation content.
- **S2 — the halo physical instantiation (APPARATUS-GATED; the photographable capstone).** The
  two-tower halo version: `x_c` = crystal **size** (Shadow 2, off the Airy/Pearcey dressing), `x_d` =
  **handedness** or **ice phase** (Shadow 3, off Stokes `V` / halo radius), swept over population
  spread, generated by HaloSim **+ the wave/Stokes layer** (the apparatus extension — NOT yet built;
  staged). The physical witness that makes it more than a toy.

### 3.5 The crossover statistic (frozen)
Per substrate, per `λ`: `cont(λ)` and `disc(λ)` (probe recovery det above baseline). Define the
**half-life** `λ*_c` = the `λ` where `cont` falls to half its `λ=0` value, and `λ*_d` likewise.
**Crossover statistic = `λ*_d / λ*_c`**, predicted `≫ 1` (ideally `λ*_d` unreached on the grid). Report
both full curves + the ratio + the analytic S0 prediction.

### 3.6 Pre-registered prediction (freeze before running)
- `cont(λ)`: high at `λ=0`, **monotone decay to chance** by a finite `λ*_c` within the grid.
- `disc(λ)`: `≈ exact (≥ 0.95)` across the **whole** grid; `λ*_d` unreached.
- **Same qualitative crossover on S0 AND S1.**
- S0 quantitative: with the fringe damping `exp(−½λ²(t−t0)²)`, `λ*_c` is analytic from the `env` scale;
  `disc` is `λ`-independent by construction (the sign factors out of the average).

### 3.7 Verdict tree (first match wins)
| branch | condition | reading |
| --- | --- | --- |
| `void` | not frozen / labels leaked into probe inputs / nondeterministic | no result |
| `law_falsified_discrete_decays` | `disc(λ)` also decays to chance | the discrete variable is NOT structurally stable as claimed — **the law is wrong** (a major banked negative) |
| `law_falsified_continuous_survives` | `cont(λ)` does NOT decay across the grid | insufficient lossiness / the continuous signature isn't washing — re-examine the knob (NOT a confirmation) |
| `operator_partial` | clean crossover on S0 but not reproduced on S1 | not yet substrate-invariant |
| `operator_confirmed_synthetic` | crossover on S0 **and** S1 (cont→chance, disc→exact, same shape) | **measured shared operator on ≥2 substrates** — Phase-5 anti-equivocation met *synthetically*; the physical S2 still owed |
| `operator_confirmed_physical` | S2 (halo) reproduces it | the full measured operator-identity — Phase-5 Exit, public-eligible after evidence-tier review |

### 3.8 Anti-p-hack discipline
Freeze §3.2–§3.7 before any run. Labels (`x_c*`, `x_d`) are **scoring-only** — the probes are
supervised, but the generator/regime is **not** tuned to produce the crossover. Pre-register the
prediction and the verdict branches; report `law_falsified_*` plainly if it appears (the law is a
conjecture — the negative is the valuable outcome, and the cheapest one). Deterministic (seed-pinned).
**S0/S1 are inline-runnable** (numpy + sklearn, the JEPA-0D / Gate-0 preflight pattern); **S2 is
operator-staged** behind the wave/Stokes apparatus extension.

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
