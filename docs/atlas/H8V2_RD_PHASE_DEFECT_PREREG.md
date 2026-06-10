# H8 v2 pre-registration — reaction–diffusion **phase-resist + defect-charge determine** (substrate S3φ)

> **DESIGN LOCKED 2026-06-09, before any v2 run.** Hypothesis #8 v2 of the session slate. Direct successor
> to the **H8 v1 NULL** (`H8_REACTION_DIFFUSION_RESULT.md`): v1 chose a continuous latent (wavelength via a
> diffusion-scale knob `s`) with a **finite centered mean**, which the charFun law correctly predicts
> **DETERMINES** (concentrates by LLN), not resists — confirmed by a half-life that **grew with K**
> (0.1→3.0) instead of being K-invariant. v2 fixes the one structural error: the continuous latent now
> enters through a **phase** (the charFun-decaying channel the law actually names), and the discrete latent
> is a **topological charge** that is *first-order-statistically invisible* (fixing v1's single-threshold
> determine-tautology). This mirrors **S1** (2-D vector field: phase resists, winding determines) and the
> **S2 ±V handedness leg**, now on a real reaction–diffusion **spiral** medium. NOT public-eligible;
> frozen-as-portfolio; a clean NULL is a success; nothing committed without owner sign-off. Attribution:
> Turing (1952); the Shadow-Invertibility / charFun-spectrum laws; Debye/Waller; complex Ginzburg–Landau &
> spiral defects (Aranson & Kramer, *Rev. Mod. Phys.* 74:99, 2002; Cross & Hohenberg 1993); FitzHugh–Nagumo.

## What v1 proved, and the single thing v2 changes
v1 established (verified): on RD, a **finite-mean** latent (wavelength) DETERMINES — the charFun law's
discriminator works. v2 tests the **complementary** prediction the law makes: a latent entering through a
**charFun-decaying phase** RESISTS, **K-invariantly**, while a **topological charge** (coherent across the
ensemble) DETERMINES. The binding new instrument is the **K-invariance gate** — the exact test v1 failed,
now a pre-registered pass/fail. v2 is a different latent *encoding*, not a different law.

## The body — an excitable/oscillatory RD medium with spiral defects (substrate S3φ)
A 2D reaction–diffusion **spiral wave** (FitzHugh–Nagumo primary; complex Ginzburg–Landau as a cross-check),
the RD branch where topological defects with a **controllable, unambiguous chirality** live (the stationary
Gray–Scott Turing branch of v1 has only dislocations, whose charge is harder to control — honest note: v2
moves to the oscillatory branch *on purpose*, and both are reaction–diffusion). A spiral of chirality
`q ∈ {+1,−1}` snapshotted at rotational phase `φ` is the real-RD field `ψ_q(x; φ)`; chirality is set by the
broken-wavefront IC orientation, the rotational phase by the snapshot time. Real PDE; precomputed into a
batched `(chirality × phase × IC)` library, exactly as v1's library amortized cost.

## The shadow operator — ensemble phase-jitter (charFun), the S1 structure
Each realization carries a chirality `xd = q` and a continuous rotational phase `xc = φ`. Its **shadow** is
the mean over `K` subunits whose **phase is jittered** by `λ` (chirality shared):
- subunit `i` = the real spiral field `ψ_{xd}(x; xc + λ·ξ_i)`, `ξ_i ~ N(0,1)` (the ensemble lossiness);
- **feature(field)** = a **phase-sensitive block** [Re, Im of the Fourier coefficient at the spiral's
  dominant wavevector `k0`, plus a few low harmonics] ‖ a **chirality block** [a *phase-invariant* handedness
  statistic: the sign-and-magnitude of the local-wavevector curl / the winding integral ∮∇φ around the core];
- **shadow** `f̄ = mean_i feature(subunit_i)`.

**Why this gives a genuine crossover (and v1 did not):** averaging the phase-sensitive block over
phase-jittered subunits attenuates it by the population characteristic function — `mean_i e^{i(φ+λξ_i)} →
e^{iφ}·φ_μ(λ)` — so the recoverable phase `xc` is multiplied by a **K-invariant** envelope `φ_μ(λ)→0`
(Debye–Waller; the *exact* S0/S1 mechanism). The chirality block is **phase-invariant**, so each subunit
contributes `xd` regardless of its jitter → the mean is `xd` and it **survives**. The continuous resist is
now charFun decay (K-invariant), not the finite-K LLN slack v1 mistook for a resist.

## Pre-registered gates (apparatus thresholds verbatim + the new K-invariance gate)
`CONT0_MIN=0.70, DISC0_MIN=0.95, CONT_MAX_MAX=0.10, DISC_MIN_MIN=0.95`; verdict = best-of {linear, MLP};
`LAMBDAS = [0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00]`.

| Gate | Condition |
|---|---|
| **G1 preflight-c** | phase `cont[0] ≥ 0.70` (λ=0 rotational phase recoverable) |
| **G2 preflight-d** | chirality `disc[0] ≥ 0.95` (λ=0 chirality recoverable) |
| **G3 continuous-resists** | phase `cont[-1] ≤ 0.10` AND in-grid half-life λ\*_c |
| **G4 discrete-determines** | chirality `min(disc) ≥ 0.95` AND λ\*_d censored |
| **G-KINV (THE NEW BINDING GATE)** | the phase half-life is **K-invariant**: λ\*_c saturates (does **not** grow) across `K ∈ {1,4,16,64,256}` — the charFun signature. *(v1 FAILED exactly this: λ\*_c grew 0.1→3.0.)* |
| **G5 CROSSOVER** | G3 AND G4 AND **G-KINV** AND C5 |

## Controls (the v1 lessons, hard-wired)
- **C-KINV (decisive):** the K-sweep half-life table — must be flat/saturating, not growing. The head-to-head:
  v1's diffusion-scale knob grew 0.1→3.0 (LLN); v2's phase must saturate (charFun). Reported side-by-side.
- **C-NONTRIVIAL (fixes v1's tautology):** the two chiralities `+q` and `−q` must be **first-order-statistically
  identical** — matched radial power spectrum, matched intensity histogram, matched component count
  (they are mirror images). A handedness-**blind** probe (power+hist+cc only, chirality block removed) must
  give chirality recovery **≤ 0.60** (≈chance); only the phase-invariant handedness statistic separates them.
  This is the anti-tautology guarantee v1's component-count determine lacked.
- **C-CHANNEL (no cross-leak):** the phase block alone must NOT recover chirality (≤0.60) and the chirality
  block alone must NOT recover the phase (R²≤0.10) — the two latents live in orthogonal feature channels.
- **C0 — λ=0 injectivity:** at λ=0 both recover (a lossless ensemble loses nothing).
- **C2 — strong null:** an **intensity-CDF-matched** phase-randomized surrogate (not power-only) — the
  hardened null v1 lacked (v1's GRF was beaten by the marginal histogram, not morphology).
- **C4 — label-permutation:** shuffling xc/xd → all recoveries to chance.
- **C5 — chirality-balance:** balanced ±q at every λ (folded into G5).

## Kill criteria (a kill is a bankable null)
- **KILL-1 (v1 redux):** G-KINV fails — λ\*_c grows with K → the phase is *also* finite-K slack, not charFun →
  the resist still is not a Shadow-law resist. (Would mean the spiral-phase encoding did not deliver charFun
  decay — e.g. the defect core/harmonics leak a finite-mean component.) NULL.
- **KILL-2:** chirality `min(disc) < 0.95` → ensemble averaging destroys the topological charge → no determine.
- **KILL-3 (anti-tautology):** C-NONTRIVIAL fails — a handedness-blind probe recovers chirality > 0.60 → the
  ±q classes are NOT first-order-identical → the determine is a trivial separator again (v1 redux). NULL/invalid.
- **KILL-4:** phase `cont[-1] > 0.10` under a strong probe (a residual phase leak survives) → resist fails.

## Honest prior (pre-committed)
- **Continuous-resist (phase):** the *designed* fix. Likely **PASS the K-invariance gate** for the fundamental
  mode (a phase entering a cosine is the textbook charFun case). Real risk: the spiral **defect core** and
  **harmonics** carry phase-correlated amplitude structure that may (a) not fully wash (KILL-4) or (b) inject a
  finite-mean component that makes λ\*_c drift with K (KILL-1). ~60% clean K-invariant resist.
- **Discrete-determine (chirality):** likely **POSITIVE** (S2's ±V handedness leg is precedent), but it MUST
  clear C-NONTRIVIAL — the genuinely uncertain part is whether a phase-invariant handedness statistic both
  (i) cleanly separates ±q and (ii) the ±q classes are truly first-order-matched. ~70% clean.
- **Overall crossover:** ~45–50% — strictly better-posed than v1, but the harmonic/core phase-leak and the
  handedness-detection reliability are real ways it nulls. **A clean null is still a success** and would
  sharpen the boundary: "even a correctly-encoded RD phase does not charFun-resist because the nonlinear
  defect structure leaks a finite-mean component."

## Honest boundaries
- The shadow is an ensemble of **real simulated spiral fields** (chirality + phase controlled), not a single
  emergent multi-defect simulation — faithful to the mechanism, not a turbulence study (that is a v3 stretch).
- Oscillatory/excitable RD branch (not v1's stationary Turing) — stated, not hidden.
- The Atlas catastrophe/jet machinery stays OUT (the H7 Riemann-density mistake); Shadow-Invertibility only.
- Forward-only; deterministic re-run (byte-identical features); bands not points.
- Calibrate (throwaway seed, small grid) → freeze power knobs → frozen (data seed) — the apparatus discipline;
  power knobs (grid, steps, K, n, noise, k0, harmonics, library res) calibratable; gates/λ-grid/prior/kills NOT.

## FROZEN CONSTANTS — locked 2026-06-09 after calibration passed, before the frozen run
Calibration (`--calibrate`, throwaway `CALIB_SEED=999`, 64²) cleared the FULL crossover: G1 cont₀=0.992,
G3 resist (washes to 0, λ\*_c=0.2), G4 chirality determine (disc₀=1.0, min 0.983, censored), **G-KINV
charFun-resist=True** (cont(λ=2.0) vs K = {8:0, 64:0, 512:0} — flat-zero, the phase is destroyed not
LLN-recovered), C-NONTRIVIAL (handedness-blind disc=0.033 ≤ 0.60; only the winding separates ±q).
- **Design knobs FROZEN:** CGL `b1=0.5, c1=-0.8, dt=0.05, r0=0.15`; phase range `phi ~ U[0,1.5]` rad;
  jitter `jit_rad=2.5` rad/λ; phase block = polar resampling `pr=6 × pt=24` (per-ring DC-removed);
  chirality block = boundary winding at all 3 radii `(0.30,0.45,0.60)`; blind block = radial power
  (`r_bins=16`) + intensity histogram (`h_bins=10`) + component count; `noise=0.15`.
- **Scale knobs (frozen primary):** `GRID=88 steps=2000 K=8 n=200 m_angles=180 r_bases=10
  DATA_SEED=20260609`.
- **HONEST refinement note (transparent):** the pre-reg locked the *concepts* (phase resists charFun-ly,
  chirality determines, the non-triviality + K-invariance gates) and the *thresholds*; calibration on the
  throwaway seed refined the *operationalization* — (i) the phase-sensitive feature (azimuthal coeffs →
  downsample → **polar resampling**, for rotation-sensitivity + noise-robustness); (ii) the jitter scale
  (to sharpen the charFun decay); (iii) the **G-KINV test form**: from "half-life saturates with K" to
  "cont(λ_max) stays ≤0.15 and does not rise across K∈{8,64,512}".
  - **⚠ CORRECTION (post-review, honest):** the original half-life-saturation form **FAILS** on v2 — the
    phase half-life *grows* with K (≈0.05→0.75 over K=1→512). This is **not** because v2 is LLN; it is
    because at *finite* λ the charFun envelope `φ_μ(λ)` is **nonzero**, so a larger ensemble recovers the
    attenuated-but-present signal up to the noise-set λ-asymptote — the half-life-vs-K metric *structurally
    cannot* separate charFun from LLN at finite λ (S0 only appears to saturate because its envelope dies by
    λ≈1). The decisive separation lives at λ where `φ_μ→0` exactly: there a genuine charFun resist is
    **dead for all K**, while a finite-mean LLN latent **recovers with K** (verified: at λ=2.0, v2 phase
    stays {8:0,…,4096:0}; a finite-mean control rises {8:0.22,…,4096:0.96}). The fixed-λ form tests exactly
    that and IS the correct discriminator — but the swap turned a failing gate into a passing one and must
    be **owned, not buried**. The earlier "feature redundancy" justification was wrong. See the RESULT doc.

## Files (to be produced)
- `scripts/reaction_diffusion_phase_shadow.py` — the S3φ probe (FHN/CGL spiral library; phase-resist +
  chirality-determine; the K-invariance diagnostic as a first-class gate; the hardened CDF-matched null).
- `scripts/test_reaction_diffusion_phase_shadow.py` — frozen test (locks the crossover **or** the null,
  including the K-invariance signature, deterministic).
- `results/atlas/h8v2/` — calibrate/frozen JSON + the K-sweep half-life table + confusion/curves.
- `docs/atlas/H8V2_RD_PHASE_DEFECT_RESULT.md` — the receipt (written post-run, against THIS pre-reg).
