# H8 capstone — why no load-bearing charFun-resist exists (a snapshot/window no-go)

> **2026-06-09, rev.2 (post-adversarial-review `wxywgz2f3`).** The formal closure of the H8 arc (v1–v5,
> five empirical nulls, two substrate families). The contentful claim is **not** the trivial "recoverable ⟹
> geometric" (true of any recoverable quantity); it is that **the charFun-RESIST mechanism itself forces the
> latent into a form that a finite-order static template reproduces**, and that the only structures escaping
> static templates (trajectory-irreducible invariants) are **determine-type, not resist**. So both escape
> routes are foreclosed *for resist latents specifically*. NOT public-eligible; a proof **sketch** with
> explicit hypotheses + an honest open gap, not a Lean-checked theorem. Attribution: the Shadow / charFun
> laws; the v1–v5 receipts; the review (counterexample hunt that found the low-pass strengthening).

## Setup
- **Latent** `φ ∈ Φ` (interval); **body** producing per-φ an observation `O ~ μ_φ` on a feature space `𝓕`
  (deterministic+fixed-IC ⇒ point mass; chaotic/stochastic ⇒ a distribution). **Lossy ensemble shadow** at
  `λ`: latent jittered `φ↦φ+λξ`, shadow = mean feature over the jitter population.
- **charFun-RESIST**: the φ-dependence of the mean feature is through **Fourier modes**,
  `E_{μ_φ}[feature] = Σ_m c_m e^{imφ}`, so ensemble jitter washes mode `m` by the jitter charFun:
  `E_ξ[e^{im(φ+λξ)}] = e^{imφ}·\hatμ(mλ) → 0`. **This is forced**: a feature-dependence that washes under
  charFun jitter *must* be (almost-)periodic in φ — i.e. **a phase**. (A non-periodic, e.g. finite-mean,
  dependence does **not** wash this way — it DETERMINES. So "resist ⟺ phase" is not a choice, it is the
  mechanism.)
- **Surrogate class `𝒮_k`** (the "natural / geometric" foils the experiments used): static, dynamics-free
  families matched on a **finite-order statistic set** — symmetry transforms, superpositions, matched power
  spectrum / amplitude histogram (the bare vortex of v2, SUP2 of v4, matched-spectrum/IAAFT of v5).
- **GEOMETRIC w.r.t. `𝒮_k`** = φ recoverable (comparable own-R²) from some `{ν_φ}∈𝒮_k` → **not load-bearing**.
  **LOAD-BEARING w.r.t. `𝒮_k`** = recoverable from `{μ_φ}` but from no `{ν_φ}∈𝒮_k`. *(Quantifiers pinned: `𝒮_k`
  is FIXED before the test; the empirics always used a fixed natural `𝒮_k`.)*

## The result (snapshot/window no-go for charFun-resist)
**Claim.** For an observation-based (snapshot or finite-window) shadow, **no latent is simultaneously
charFun-RESISTING, recoverable, and load-bearing against a finite-order natural surrogate class.** Both routes
to load-bearing are foreclosed *by the resist property itself*:

**(R1) Resist low-pass-filters the mode index ⟹ the recoverable signal is finite-order.** Under jitter, mode
`m` carries weight `\hatμ(mλ)`, which **decays in `m`** (for Gaussian jitter `\hatμ(mλ)=e^{-(mλ)²/2}`:
`m=1:0.61, m=2:0.14, m=3:0.011, m≥4:<3e-4` at `λ=1`). So the φ-information surviving any `λ>0` sits in a
**bounded band of low modes** `m ≤ m*(λ)`, a finite-order statistic — reproduced by a finite-order
matched-statistics template in `𝒮_k`. A latent "requiring arbitrarily-high-order structure" **cannot
simultaneously charFun-resist and stay recoverable** (the resist washes exactly the high-order content). The
escape route "no finite-order sufficient statistic" is therefore **closed for resist latents.** *(Verified for
Gaussian/sub-Gaussian jitter; general AC jitter plausible — `\hatμ` decays by Riemann–Lebesgue — but not
fully checked.)*

**(R2) Trajectory-irreducible invariants are determine-type, not resist.** The only quantities that genuinely
escape *every* static template are functionals of the whole trajectory (ergodic time-averages, Koopman
rotation numbers, braiding/worldline invariants). But these **concentrate (finite mean) → they DETERMINE**
(LLN, half-life grows with K — the v1/v3 determine signature), they do **not** charFun-resist. So the
trajectory-shadow escape, even when it exists, yields a **determine** latent, never a load-bearing **resist**.

Together: a charFun-resist latent is (R1) finite-order ⟹ finite-order-template-reproducible, and (R2) not
trajectory-irreducible. **No load-bearing charFun-resist.** This **cuts**: it is *false* for determine-type
latents (which need not be periodic and can be trajectory-irreducible), so it is not a universal tautology.

## What it does NOT say (the honest tautology footnote)
The *unrestricted* statement "recoverable ⟹ geometric vs SOME static family" is **trivially true and
content-free**: given any decoder `D` with identifying statistic `T(φ)=E_{μ_φ}[D]`, a static distribution
`ν_φ` *matched on that mean* (NOT a point mass — a point mass fails for finite-range decoders, e.g. a
Bernoulli feature) reproduces the recovery. So "load-bearing vs ALL static families" is empty by construction;
the only meaningful arena is a **fixed natural `𝒮_k`**, which is what the result above and all five
experiments use. We lead with the restricted claim and relegate the tautology here, precisely so it is not
mistaken for the content.

## Empirical anchor (corrected to the receipts)
The five nulls are the result's instances — *honestly*, not tidied:
- **v1** wavelength — finite-mean → DETERMINE (not a resist; the determine side).
- **v2** rotation phase — a resist whose mode-1-dominated feature-curve *was* an SO(2) orbit → finite-order
  template (bare vortex) reproduced it.
- **v3** temporal phase — resist **partially failed** (a determine-type, K-growing tail), and the recoverable
  part was orientation (mode-1) → template-reproduced. *(Evidence for both sides, not a clean resist.)*
- **v4** relative phase — a resist; the (mode-1) interference curve reproduced by a static superposition (SUP2).
- **v5** slow phase — recoverable from **matched static statistics** (power spectrum 0.93 + amplitude
  histogram 0.92), i.e. a finite-order template — *not* via a single periodic curve; the (R1) finite-order
  mechanism, on a non-CGL substrate.
**Caveat:** "charFun-resist ⟹ the feature-curve is a *symmetry-orbit / interference* coordinate" is an
**empirical** fact about these instances (each was single-mode-dominated), **not** a theorem — a periodic
curve in feature space is a closed loop, not in general a group orbit. Only "reproduced by *some* periodic /
finite-order static template" follows.

## What is new here vs the v5 conjecture
- **NEW (deductive):** (R1) the resist mechanism is a low-pass filter in `m`, closing the high-order escape;
  (R2) trajectory-irreducible invariants are determine-type, closing the trajectory escape **for resist
  latents**; the precise **snapshot/window vs trajectory** framing; the pinned restricted-`𝒮_k` quantifiers.
- **NOT new:** the empirical chain (resist ⟹ phase ⟹ static-statistic ⟹ geometric) — stated in the v4/v5
  receipts.
- **STILL OPEN:** a fully general (all AC jitter laws) proof of (R1); and whether a **determine-type** latent
  can be load-bearing on a genuine **trajectory** shadow (outside the resist program — the natural next target
  if H8 is ever reopened). A formal proof would need "snapshot ⟹ finite-order identifying statistic" made
  rigorous.

## Status & boundaries
- Proof **sketch**, hypotheses explicit (Gaussian/sub-Gaussian jitter for R1; fixed natural `𝒮_k`).
- Refutes the *absolute* load-bearing notion as ill-posed; establishes the *restricted, resist-specific*
  no-go modulo the open generalizations above.
- Empirically: v1–v5 are exactly its predicted instances; the adversarial counterexample hunt (four
  constructions, two substrate families) failed to break it for the structural reasons (R1)+(R2).

## The H8 record
Receipts + pre-regs: v1 `H8_REACTION_DIFFUSION_RESULT.md`, v2 `H8V2_RD_PHASE_DEFECT_RESULT.md`, v3
`H8V3_RD_LOADBEARING_RESULT.md`, v4 `H8V4_RD_RELATIVE_PHASE_RESULT.md`, v5
`H8V5_SUBSTRATE_GENERALITY_RESULT.md`. Methodological asset: the **own-R²-within-distribution** load-bearing
test (never train-real/test-surrogate transfer-R²); read the dissection, not the headline.
