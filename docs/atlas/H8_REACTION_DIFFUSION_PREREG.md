# H8 pre-registration — reaction–diffusion as a determine/resist shadow (substrate **S3**)

> **DESIGN LOCKED 2026-06-09, before the frozen run.** Hypothesis #8 of the session slate. Extends the
> **Shadow-Invertibility Law** to a brand-new substrate — a 2D **reaction–diffusion** field (Gray–Scott),
> the first genuinely *nonlinear, pattern-forming PDE body* the law has touched (prior substrates: S0
> 1-D caustic toy, S1 2-D vector field, S2 halo optics). The law: a lossy ensemble shadow **DETERMINES**
> a discrete/topological latent and **RESISTS** a continuous one, with the split sharpening as ensemble
> lossiness `λ` grows. NOT public-eligible; frozen-as-portfolio; a clean NULL is a success; nothing
> committed without owner sign-off. Attribution: Turing (1952); Gray & Scott (1983/84) + Pearson (1993,
> *Science* 261:189) for the (F,k) phase diagram; the Shadow-Invertibility Law (`ATLAS_PHASE5_CROSS_SUBSTRATE.md`);
> Debye (1913)/Waller (1923); the charFun-spectrum law (`SHADOW_CHARFUN_DETERMINE_RESIST_LAW.md`);
> Liu et al. (2013, *PNAS* 110:11905, mussel beds = Cahn–Hilliard ≠ Turing) for the look-alike panel.

## The honest prior (pre-committed)
- **Discrete-determine half:** likely **POSITIVE** — H3/S2 show the determine half is robust, and pattern
  *class* (spots vs stripes) is a genuine topological invariant that survives ensemble averaging. NULL
  probability ~15%.
- **Continuous-resist half:** the **real test, genuinely uncertain**. The faithful mechanism (the exact
  characteristic wavelength λ\*, carried by the spectral peak, washing as the kinetics-jitter spread `λ`
  grows) could (a) cleanly resist, (b) leave a **central leak** (the averaged-peak centroid still encodes
  λ\* → cont fails to reach ≤0.10 — the failure mode S0-v2 fixed with an off-centre band-pass feature),
  or (c) be **trivial single-frame blur** rather than ensemble decoherence. We pre-register the
  separators that distinguish a genuine resist from each. Rough split: ~55% clean resist, ~25%
  central-leak/needs-the-fallback, ~20% no separation.
- **Is-it-RD (B-panel) half:** moderately strong that mechanism *class* {RD, Cahn–Hilliard, matched-GRF,
  wave} is determinable from a lossy shadow; the **RD-vs-matched-GRF margin is the tightest** and is the
  honest risk.

This is a **substrate extension of an established law, not a new law.** No overclaiming: a positive S3 is
"the law's mechanism reaches a nonlinear RD body," a null is "the law's continuous-resist does not survive
the RD nonlinearity (bounded)."

---

## The body — 2D Gray–Scott reaction–diffusion (substrate S3)
`∂u/∂t = Du ∇²u − u v² + F(1−u)`,  `∂v/∂t = Dv ∇²v + u v² − (F+k) v`, periodic BCs, 5-point Laplacian
(`np.roll`), explicit Euler `dt=1.0`, `Du=0.16, Dv=0.08`. Random IC per field (seeded: `u≈1`, `v` a few
seeded Gaussian blobs/noise). The developed `v`-field after `T_STEPS` is the pattern. The morphology
**class** is set by the (F,k) basin; the characteristic **wavelength** λ\* by the exact (F,k).
*(Pearson basin anchors, calibratable: spots ≈ (0.030,0.063), stripes ≈ (0.030,0.057), labyrinth ≈
(0.046,0.063), gaps/holes ≈ (0.026,0.055). Exact values + the in-class wavelength-sweep direction are
**power knobs** fixed in `--calibrate` before the frozen run.)*

## The shadow operator — ensemble-jitter (Debye–Waller), NOT single-frame blur
Each realization carries a central kinetics value; its **shadow** is the mean over `K` **subunits** whose
kinetics are jittered by `λ` around the central value:
- subunit kinetics `θ_i = θ* + λ·ξ_i`, `ξ_i ~ N(0,1)` (the **ensemble lossiness knob**, exactly S0/S2);
- each subunit → a Gray–Scott field → a **translation-invariant feature vector** `f_i` =
  [radial power spectrum `S(k)` (R_BINS bins) ‖ intensity histogram (H_BINS bins) ‖ connected-component /
  defect count];
- **shadow** `f̄ = mean_i f_i`.

At `λ=0` the `K` subunits are identical → kinetics recoverable (injectivity). As `λ` grows the subunits
**decohere** → the wavelength-bearing spectral peak Debye–Waller-broadens and washes, while class-bearing
morphology survives. **Cost:** features are precomputed once over a batched **kinetics→field library**
(real Gray–Scott PDE), then shadows are formed by averaging library features — faithful to the
decoherence mechanism, affordable. This is single-author honest: the shadow is an ensemble average of
**real RD fields**, not a hand-built profile.

## The two legs (mirroring S2; the binding A-gate = the crossover)
- **S3c — continuous-resists.** One fixed class (e.g. stripes/labyrinth). `xc` = characteristic
  wavelength λ\* (set by an in-class kinetics knob, drawn `U[xc_lo,xc_hi]`); `xd` = dummy. Tests whether
  the ensemble shadow **washes** the wavelength.
- **S3d — discrete-determines.** `xd` = class ∈ {spots +1, stripes −1} (isolated-component vs
  connected-stripe topology); `xc` = dummy; shadow uses IC/orientation jitter `λ`. Tests whether the
  ensemble shadow **keeps** the class.
- **Crossover (the headline)** = S3c **resists** AND S3d **determines**.

## The B-panel (secondary readout — the "is-it-RD" confusion matrix)
A 4-way mechanism classifier on the same shadow features: {RD-Turing (Gray–Scott), **Cahn–Hilliard**
(phase separation, multiple coarsening times), **matched-spectrum phase-randomized GRF**, **FitzHugh–
Nagumo** excitable waves}. Reported as a confusion matrix at the frozen `λ`-grid. The matched-GRF class
doubles as the **non-vacuity null** for S3d (see controls). Secondary, non-binding.

---

## Pre-registered gates (reuse the frozen apparatus thresholds verbatim)
`CONT0_MIN=0.70, DISC0_MIN=0.95, CONT_MAX_MAX=0.10, DISC_MIN_MIN=0.95` (from
`pvnp_phase5_lossiness_crossover.py`). Verdict policy: **best-of {linear, MLP}** per metric (both
reported). `λ`-grid `LAMBDAS = [0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00]`.

| Gate | Condition |
|---|---|
| **G1 preflight-c** (injectivity) | S3c `cont[0] ≥ 0.70` (λ=0 wavelength recoverable — the info IS there) |
| **G2 preflight-d** | S3d `disc[0] ≥ 0.95` (λ=0 class recoverable) |
| **G3 continuous-resists** | S3c `cont[-1] ≤ 0.10` **AND** in-grid half-life `λ*_c ≠ None` |
| **G4 discrete-determines** | S3d `min(disc) ≥ 0.95` **AND** `λ*_d` censored (never washes) |
| **G5 CROSSOVER (headline)** | G3 **AND** G4 |
| **G6 is-it-RD (B-panel)** | 4-way balanced accuracy ≥ 0.90 on the shadow (secondary) |

## Controls (non-triviality — the load-bearing anti-confound battery)
- **C0 — λ=0 injectivity:** a single (`K=1`, λ=0) field recovers wavelength (S3c) and class (S3d). Proves
  the info exists pre-shadow; the resist is *averaging*, not absence.
- **C1 — ensemble-vs-blur separator (the non-triviality of resist):** a **single-frame Gaussian blur**
  matched to wash the peak is run alongside the ensemble shadow. The resist must have an **in-grid
  half-life in `λ` (the ensemble-jitter axis)** and must **not** be reproduced by the static blur of a
  single field — i.e., it is decoherence, not band-limiting. *(If the only thing that washes the
  wavelength is static blur, G3's "ensemble" claim is downgraded to trivial — pre-registered as a
  downgrade, not a pass.)*
- **C2 — matched-power phase-randomized GRF null:** a GRF matched to the RD radial S(k). S3d's class
  discriminator must separate RD from this GRF **above chance** → it uses morphology/harmonics/phase, not
  just the spectral peak. (If RD-vs-GRF ≤ 60% balanced acc, the discrete-determine claim is vacuous.)
- **C3 — class ⊥ wavelength decorrelation:** in S3c (resist leg) the class is fixed (dummy `xd`); in S3d
  (determine leg) the wavelength is a dummy drawn from a **common range across classes** so class is
  decorrelated from λ\* (the H3-v1 confound, designed out by the two-leg split).
- **C4 — label-permutation:** shuffling `xc`/`xd` independently drives all recoveries to chance.
- **C5 — class-balance:** S3d majority ∈ [0.45, 0.55] at every λ.

## Kill criteria (explicit; a kill is a bankable null)
- **KILL-1 (resist fails):** S3c `cont[-1] > 0.10` under a strong probe (central leak survives) → the RD
  wavelength is NOT washed by ensemble averaging → continuous-resist does not reach the RD substrate.
  *(If it's specifically a central-leak, the pre-registered S0-v2 fallback — an off-centre/high-pass
  spectral feature that fully decoheres — may be applied AND RE-RUN under a clearly-labeled amended
  pre-reg; the primary verdict stands on the faithful feature.)*
- **KILL-2 (determine fails):** S3d `min(disc) < 0.95` → ensemble averaging destroys the class → no
  determine half on RD.
- **KILL-3 (not lossiness):** resist is `λ`-independent (flat) or has no in-grid half-life → not an
  ensemble-decoherence effect.
- **KILL-4 (vacuous determine):** C2 fails (RD-vs-matched-GRF ≤ 60%) → the class survives only as a
  spectral peak a GRF also has → determine is vacuous.

## Power knobs — calibratable on the throwaway seed, then FROZEN
Tuned ONLY in `--calibrate` (throwaway `CALIB_SEED`, 64² smoke) to clear G1/G2 + show the crossover, then
**frozen** into this doc before the `--frozen` (data-seed, 128² primary) run: grid size `GRID`, integration
steps `T_STEPS`, subunit count `K`, sample count `n`, obs-noise `NOISE`, feature bin counts
`(R_BINS, H_BINS)`, the per-class (F,k) anchors, the in-class wavelength-sweep direction, the library
resolution `(M_kinetics, R_ic)`, and the `xc` range. **NOT** calibratable: the gate thresholds, the
`λ`-grid, the honest prior, the kill criteria.

> **FROZEN CONSTANTS — locked 2026-06-09 after calibration passed, before the frozen run.**
> Calibration (`--calibrate`, throwaway `CALIB_SEED=999`, 64²/`steps=3500`/`K=6`/`n=64`) cleared
> **preflight + CROSSOVER** on the *faithful* feature (no off-centre fallback needed): S3c `cont0=0.726`
> → washes to `0.000` by λ=0.5 (half-life λ\*_c=0.5); S3d `disc=1.000` at every λ (λ\*_d censored).
> **Design knobs FROZEN** (passed calibration, unchanged for the frozen run): `R_BINS=24 H_BINS=12
> thr=0.20`; diffusion-scale wavelength knob `s ~ U[0.70, 1.30]`, jitter `sig_s=0.35`, clip `[0.45, 1.50]`;
> basins `spots=(0.030,0.062) stripes=(0.030,0.055)` [+ `labyrinth=(0.046,0.063) gaps=(0.026,0.055)` for the
> B-panel]; `Du=0.16 Dv=0.08`; library `(M_kinetics=40, R_ic=8)` per class; float32 integration.
> **Scale knobs for the frozen primary:** `GRID=128 T_STEPS=4000 K=8 n=160 NOISE=0.04 DATA_SEED=20260609`
> (B-panel `panel_n=60 panel_k=6`). The frozen run uses exactly these; `T_STEPS` trimmed 6000→4000 (a
> compute knob; patterns develop well before 3500 — calibration confirmed) and library left at the
> calibration resolution (40×8) for tractability given the measured per-field cost.

## Honest boundaries (pre-committed)
- **Synthetic-only Milestone-1.** Real chemistry/biology (zebrafish, CIMA, dryland imagery) is a named
  **stretch**, never the headline; the literature warns simulation-match ≠ proof of RD (Economou & Green
  2014). "In the wild" stays "Turing-*like*."
- **The Atlas catastrophe/jet machinery is OUT of scope as a read-out** — pointing the `c₃` jet
  discriminant at a scalar RD density is the H7 Riemann-density mistake (reads curvature, not a caustic).
  The applicable machinery is Shadow-Invertibility only. *(The one speculative RD caustic — Cross–Newell
  phase-surface disclinations — is explicitly deferred, not built.)*
- **The single-snapshot wall:** on a frozen snapshot, arrested/pinned Cahn–Hilliard can mimic stationary
  Turing; the cleanest is-it-RD discriminator (fixed-λ\* vs coarsening) wants a time series. The B-panel
  is reported with this caveat.
- **Forward-only / no inversion.** Bands not points (report seeds; deterministic re-run within 1e-6).

## The charFun leg (stretch — the H2 analog on RD)
Swap the kinetics-jitter population in S3c {gaussian, uniform, Cauchy, lattice} and test that resist
tracks **charFun decay, not variance** (reusing `draw_pop`/`charfun_re`). Conceptual payoff: **Cahn–
Hilliard is the "Cauchy of patterns"** — coarsening sends spectral mass to k→0, no selected scale, the
charFun never concentrates. Reported only if clean; not part of the binding gate.

## Files (to be produced)
- `scripts/reaction_diffusion_shadow.py` — the S3 probe (Gray–Scott library, shadow, S3c/S3d legs,
  B-panel, copied probe fns + gates, `--calibrate`/`--frozen` modes).
- `scripts/test_reaction_diffusion_shadow.py` — frozen test (locks the gates, deterministic, fast).
- `results/atlas/h8/` — `calibrate.json` / `frozen.json` + the confusion matrix + the λ-curves.
- `docs/atlas/H8_REACTION_DIFFUSION_RESULT.md` — the receipt (written post-run, against THIS pre-reg).
