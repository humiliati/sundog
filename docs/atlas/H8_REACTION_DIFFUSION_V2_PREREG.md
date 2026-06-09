# H8-v2 pre-registration — RD determine/resist done right: phase resists, chirality determines (substrate S3-CGL)

> **DESIGN LOCKED 2026-06-09, before any v2 run.** Successor to the v1 NULL
> (`H8_REACTION_DIFFUSION_RESULT.md`), which the adversarial review killed because the chosen continuous
> latent (wavelength via diffusion-scale) had a **finite mean** → the charFun law predicts it DETERMINES
> (concentrates by LLN), and the determine half was a trivial single-threshold separator. v2 fixes BOTH
> failure modes by construction: the continuous latent enters through a **charFun-decaying channel (a
> phase)**, and the discrete latent is **topological chirality** — invisible to every marginal/power
> statistic. This mirrors the proven **S1** (2-D vector field: phase resists, winding determines) on a
> genuine reaction–diffusion body. NOT public-eligible; frozen-as-portfolio; a clean NULL is a success;
> nothing committed without owner sign-off. Attribution: Turing (1952); Kuramoto; **Aranson & Kramer
> (2002, RMP 74:99, the Complex Ginzburg–Landau review)**; Cross & Hohenberg (1993); the Shadow-
> Invertibility / charFun-spectrum laws (`SHADOW_CHARFUN_DETERMINE_RESIST_LAW.md`); Debye/Waller; the S1
> leg of `pvnp_phase5_lossiness_crossover.py`.

## What v1 taught (the binding constraints on v2)
1. **RESIST ⟺ charFun decay (a PHASE), not a finite-mean scale.** v1's wavelength concentrated under
   averaging (half-life GREW with K — LLN). v2's continuous latent must enter as `e^{iφ}` so averaging
   damps it by `|charFun(λ)|→0`, **K-invariantly** (the genuine resist signature S0/S1/S2 satisfy).
2. **DETERMINE must be non-trivial.** v1's spots-vs-stripes was a non-overlapping component-count
   threshold averaging could never break. v2's discrete latent (chirality) is **invisible to |A|, the
   power spectrum, and the intensity histogram** — CW and CCW spirals are mirror images with identical
   marginals — so it can only be read from the **phase winding**, and a phase-randomized surrogate loses
   it (the null finally bites).
3. **The decisive gate is the K-dependence of the resist half-life** — the v1 falsifier becomes the v2
   pass condition (K-invariant = charFun; growing-with-K = LLN/determine-type = KILL).

## The body — Complex Ginzburg–Landau (substrate S3-CGL)
`∂A/∂t = A + (1+ib)∇²A − (1+ic)|A|²A`, `A(x,y,t) ∈ ℂ`, periodic BCs, spectral/ETD or semi-implicit step.
CGL is the **canonical amplitude (envelope) equation of oscillatory reaction–diffusion media** (BZ-type),
universally derived near a Hopf bifurcation — a legitimate RD model whose **complex field makes the phase
and the topological winding native and exact.** Seeded with a charge-`q` core `A₀ = tanh(r)·e^{iqθ}`
(θ = atan2(y−y₀, x−x₀)) it relaxes to a rigidly-rotating **spiral wave** of chirality `q = ±1` rotating at
frequency ω. *(Parameters b, c in the stable-spiral regime; fixed in calibration.)*

## The two latents (both native CGL dynamical quantities)
- **continuous `c` (must RESIST):** the spiral's **rotation phase** — equivalently the **observation
  time** `t_obs` at which the snapshot is taken (a rigidly rotating spiral satisfies `A(·,t)≈A(·,0)·e^{−iωt}`
  for the global phase). Drawn `c ~ U[0, 2π)`. Enters as a genuine `e^{ic}` factor → charFun-decays.
- **discrete `d` (must DETERMINE):** the spiral **chirality / topological charge** `q ∈ {+1, −1}` (the
  winding of `arg A` around the core). A topological invariant of the dynamics; **invisible to every
  marginal/power feature.**

## The shadow operator — ensemble observation-phase jitter (Debye–Waller / charFun)
Each realization carries a nominal phase `c` and chirality `d`; its **shadow** is the mean over `K`
subunits whose observation phase is jittered by `λ`: snapshot the spiral trajectory at `t_obs,i` giving
phase `c + λ·ξ_i`, `ξ_i ~ N(0,1)`. Subunits **share the spatial spiral structure**, differ only in global
phase (the faithful S0/S1 construction). Mean of `K` phase-sensitive features = the shadow. Mechanism:
`mean_i [A·e^{i(c+λξ_i)}] = A·e^{ic}·⟨e^{iλξ}⟩ = A·e^{ic}·charFun(λ)` → the **global phase washes by charFun
decay, K-invariantly** (more subunits reduce sampling noise around the envelope, they do NOT move it — the
v1 LLN failure cannot recur, by construction). The spatial structure `A(x)` (hence the chirality winding)
is preserved (scaled by `charFun(λ)`).

## Features
- **phase-sensitive (for the resist):** real+imag parts of the low-order complex Fourier coefficients of
  `A` (carry the global phase `c`).
- **chirality (for the determine):** a **phase-winding / curl statistic** — e.g. the spatially-averaged
  signed phase-gradient curl `⟨Im(conj(A)·(∂ₓ+i∂ᵧ)A)⟩`-type scalar (flips sign with `q`, invariant to
  global phase). Chirality is NOT in `|A|`, the radial power spectrum, or the intensity histogram.
- (amplitude `|A|` radial profile retained as a phase-insensitive reference channel.)
Exact feature set fixed in calibration; the chirality feature must be demonstrably global-phase-invariant.

## The two legs (mirroring S1/S2/v1)
- **S3'c — continuous-resists.** `c` = rotation phase (vary), `d` = dummy (one chirality). Tests whether
  the ensemble shadow washes the phase **K-invariantly**.
- **S3'd — discrete-determines.** `d` = chirality ±1 (vary, balanced), `c` = dummy phase (common range →
  chirality ⊥ phase). Tests whether the ensemble shadow keeps the chirality.
- **Crossover** = S3'c resists (charFun) AND S3'd determines.

## Pre-registered gates
Reuse the apparatus thresholds + the **new decisive K-invariance gate**. `LAMBDAS` as v1; verdict
best-of {linear, MLP}.

| Gate | Condition |
|---|---|
| G1 preflight-c | S3'c `cont[0] ≥ 0.70` (λ=0 phase recoverable) |
| G2 preflight-d | S3'd `disc[0] ≥ 0.95` (λ=0 chirality recoverable) |
| G3 continuous-resists | S3'c `cont[-1] ≤ 0.10` AND in-grid half-life |
| **G3★ resist is charFun (THE decisive gate)** | the S3'c half-life is **K-INVARIANT** across `K ∈ {1,4,16,64,256}` (saturates) — NOT growing with K. *(v1 failed exactly this.)* |
| G4 discrete-determines | S3'd `min(disc) ≥ 0.95` AND `λ*_d` censored |
| **G5 CROSSOVER** | G3 AND G3★ AND G4 AND C5 |

## Controls (the v1 holes, closed)
- **C0 — λ=0 injectivity:** a single (K=1, λ=0) snapshot recovers phase (S3'c) and chirality (S3'd).
- **C1 = G3★ — K-invariance (the load-bearing one):** the resist half-life must be flat in K (charFun),
  the explicit fix for the v1 LLN artifact. Coded as a hard gate, not a printout.
- **C2 — chirality is non-trivial / the null bites:** a **phase-randomized** surrogate (preserve `|A|` and
  the power spectrum, randomize Fourier phases) must drop chirality-recovery to **chance**, AND an
  **intensity-CDF-matched** surrogate likewise — proving the determine uses the phase winding, not any
  marginal. *(If chirality is recoverable from |A|/power/histogram alone, KILL-4.)*
- **C3 — chirality ⊥ phase:** dummy `c` drawn from the common range; verify chirality-recovery is
  invariant to `c`.
- **C4 — label-permutation:** shuffling `c`/`d` → recoveries to chance.
- **C5 — class balance:** S3'd majority ∈ [0.45, 0.55] (balanced construction).

## Honest prior (pre-committed)
- **Resist half:** now **correct by construction** — the phase enters via `e^{ic}` so it MUST charFun-wash
  K-invariantly. The genuine risk is whether the *finite spatial-structure leak* (the chirality/amplitude
  channel) contaminates phase-recovery, or whether a residual phase reference survives — calibration
  decides. Prior ~70% clean charFun resist.
- **Determine half:** the real question. Chirality is invisible to marginals (good — non-trivial), but it
  must (a) be reliably READABLE from the curl feature at λ=0, and (b) SURVIVE averaging as `charFun(λ)→0`
  shrinks the signal amplitude (the determine could degrade at large λ as the field washes toward 0).
  Prior ~50% the chirality survives to large λ; if it degrades, that is an honest bounded result (the
  determine and resist share the same vanishing amplitude).
- **Overall:** genuinely uncertain — a real test, not a foregone positive. **A NULL here is informative**
  (it would say the topological determine cannot outlast the phase resist on a shared amplitude), and a
  POSITIVE would be the first genuine determine/resist crossover on a nonlinear RD body.

## Kill criteria (a kill is bankable)
- **KILL-1 (resist not charFun):** S3'c half-life GROWS with K (v1 redux) → not a charFun resist.
- **KILL-2 (phase doesn't wash):** S3'c `cont[-1] > 0.10` → the phase leaks (a residual reference survives).
- **KILL-3 (determine washes too):** S3'd `min(disc) < 0.95` → chirality does not outlast the phase resist
  (the shared-amplitude bounded null).
- **KILL-4 (determine vacuous):** chirality recoverable from |A|/power/histogram, or the phase-randomized /
  CDF-matched surrogate is NOT separated → the determine is a marginal artifact, not topology.

## Honest boundaries (pre-committed)
- **CGL is the amplitude REDUCTION** of oscillatory RD, not the full 2-species PDE. Faithful and canonical,
  but the full-2-species (FHN/Barkley spiral) version is a named **stretch** — flagged, not claimed.
- The continuous latent (rotation phase) is a **geometric/dynamical phase**, as in S1 — not a "kinetics"
  parameter. The Shadow law is about determine/resist of latents; this is a legitimate latent, stated
  plainly (no "resists the kinetics" language).
- Forward-only; bands not points; deterministic re-run (byte-identical features); seeds recorded.
- The Atlas catastrophe/jet machinery stays OUT as a read-out (the H7/v1 lesson).

## Power knobs (calibratable on a throwaway seed, then FROZEN)
CGL `b, c` (stable-spiral regime), grid, integration time + ω, the feature set (which Fourier modes; the
exact chirality-curl statistic), `K`, `n`, obs-noise, the phase-jitter scaling, `λ`-grid resolution.
NOT calibratable: the gate thresholds, G3★ K-invariance, the honest prior, the kill criteria.

## Files (to be produced)
- `scripts/reaction_diffusion_v2_phase.py` — the S3-CGL probe (CGL spiral generator; phase + chirality
  features; S3'c/S3'd legs; the K-invariance test; phase-randomized + CDF-matched nulls; `--calibrate`/
  `--frozen`).
- `scripts/test_reaction_diffusion_v2_phase.py` — frozen test (the crossover IF it holds, or the locked
  NULL; the K-invariance signature; chirality-invisible-to-marginals; determinism).
- `results/atlas/h8v2/` + `docs/atlas/H8_REACTION_DIFFUSION_V2_RESULT.md` — the receipt (post-run).
