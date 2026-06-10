# H8 v2 Result — phase-resist + chirality-determine (substrate S3φ): **SCOPED bounded-positive**

> **2026-06-09.** Against `docs/atlas/H8V2_RD_PHASE_DEFECT_PREREG.md`. Successor to the **H8 v1 NULL**.
> **The charFun determine/resist mechanism is genuinely confirmed on a spiral phase/chirality construction
> — and v1's two specific errors are fixed — BUT the reaction–diffusion dynamics are NOT load-bearing, so
> this does NOT earn "the Shadow law extends to an RD substrate." It is, honestly, S1 re-derived with the
> vortex sourced from a (decorative) PDE.** Caught and scoped by a 5-skeptic adversarial review + my own
> verification, before any claim. NOT public-eligible; frozen-as-portfolio. Attribution: Aranson & Kramer
> (CGL spirals); the Shadow-Invertibility / charFun laws; S1/S2 (the phase-resist + handedness precedents).

## Headline — what is real, and what is not
**Real (verified):**
- **The phase resists by genuine charFun information-destruction** — not v1's finite-K LLN slack. At
  λ=2.0 (where the charFun envelope ≈0), recovery stays **{K=8:0, 64:0, 512:0, 4096:0}**, while a
  finite-mean LLN control **recovers {8:0.22, 64:0.74, 512:0.93, 4096:0.96}**. v1's exact failure mode is
  fixed.
- **The determine tautology is fixed** — the discrete latent is the spiral **chirality** (topological
  winding ±1), and ±q are **exact mirror images** (radial-power L1 diff 5e-18, histogram diff 0.0,
  component-count identical), so a handedness-**blind** probe is at chance (disc 0.03) while only the
  winding separates them. v1's determine was a first-order-visible single threshold; this is not.
- **Code is clean:** CGL integrator stable, mirror convention correct, byte-reproducible, no stale receipt,
  the v1 histogram leak gone; the G-KINV gate has teeth (it fires on an injected finite-mean leak).
- The gates pass on **fresh seeds + larger scale** (frozen seed 20260609, 88², n=200; +3 more seeds) —
  not throwaway-seed overfitting.

**NOT real (the scope-down, verified by ablation):**
- **The RD dynamics are decorative.** Replacing the 1500–2000-step CGL PDE with a **bare analytic vortex
  `tanh(r/r₀)·e^{iqθ}` (zero RD)** reproduces **every** gate identically:
  | | cont₀ | cont(λ=2) | disc | G-KINV cont-vs-K | charFun |
  |---|---|---|---|---|---|
  | CGL PDE | 0.994 | 0.0 | 1.0 | {8:0,64:0,512:0} | True |
  | bare vortex (no RD) | 0.996 | 0.0 | 1.0 | {8:0,64:0,512:0} | True |
  So the crossover is a property of **[mean over jittered ROTATIONS of an azimuthal-phase field] + [its
  exact mirror]** — i.e. **S1's construction** (synthetic vortex + winding), with the vortex sourced from
  a PDE that changes no result. **This is not a reaction-diffusion substrate extension.**

## Integrity disclosures (owned, not buried)
- **G-KINV gate was substituted.** The *originally pre-registered* form ("phase half-life saturates with
  K") **FAILS** — the half-life *grows* ≈0.05→0.75 over K=1→512. I replaced it with the fixed-λ test it
  passes. The fixed-λ test **is** the correct discriminator (the half-life metric structurally cannot
  separate charFun from LLN at *finite* λ, because the charFun envelope is nonzero there; the separation
  only appears where the envelope →0, which the fixed-λ test probes — and the finite-mean control above
  confirms v2 is genuinely charFun there). But a failing gate became a passing one by substitution; that
  is disclosed here and in the pre-reg correction. The earlier "feature redundancy" reason was wrong.
- **C-CHANNEL fails; "orthogonal channels" is false.** The pre-reg claimed the phase block alone must NOT
  recover chirality (≤0.60); in fact it recovers it at **disc=1.0** (the spiral arm-twist is
  handedness-correlated). The determine is still legitimately carried by the phase-*invariant* winding and
  the non-triviality runs through the *blind* block — but the receipt does **not** assert orthogonal
  channels.
- **The determine is orthogonal to the lossiness, not "surviving" it.** The winding is phase-invariant, so
  the phase-jitter operator never challenges it (disc(chir) flat 1.0 across λ). This is **inherited from
  S1** (S1's winding is also phase-jitter-orthogonal), not a new defect — but unlike S2's ±V (where
  lossiness acts on |V| yet sign survives), here the operator simply does not touch the discrete latent.
  Not over-claimed as "survives the shadow."

## Pre-registered scorecard (as gated, with the honest annotations)
| Gate | Result |
|---|---|
| G1 preflight-c | PASS (cont₀=0.990, frozen) |
| G2 preflight-d | PASS (disc₀=1.0) |
| G3 continuous-resists | PASS (washes to 0, λ\*_c=0.15) |
| G4 discrete-determines | PASS (min disc 0.99, censored) — but lossiness-orthogonal (see disclosure) |
| **G-KINV** (fixed-λ form) | PASS (cont(λ=2) vs K = {8:0,64:0,512:0}) — **original half-life form FAILS** (substituted) |
| C-NONTRIVIAL | PASS (blind 0.03, chir 1.0) — genuine (±q first-order-identical by construction) |
| C-CHANNEL | **FAIL** (phase block leaks chirality at 1.0; not orthogonal) |
| Faithfulness (RD load-bearing) | **FAIL** (bare-vortex ablation identical) |
| **Overall** | **SCOPED bounded-positive:** charFun mechanism confirmed + v1's errors fixed; **≈ S1, not an RD extension** |

## The deeper lesson (the real contribution)
**The determine/resist crossover is substrate-agnostic at the level of [a phase] + [a topological charge].**
That is *why* a bare vortex suffices and the RD is decorative — the charFun law operates on the
phase/winding *structure*, independent of the dynamics that produced it. Making the crossover genuinely
RD-*specific* requires the **dynamics to be load-bearing**: the continuous latent must be the spiral's
*natural temporal phase* (the real CGL time-evolution — verified here NOT to be a rigid rotation, residual
~0.71), and the discrete latent an *emergent, interacting* defect charge (a turbulent multi-defect CGL
field), not an imposed rotation + a mirror pair. That is the v3 boundary.

## What v3 would need (scoped, owner-gated, NOT built)
- Continuous latent = the spiral's **dynamical** phase (ensemble of real CGL snapshots across time), so the
  resist is tested on the field's own evolution, not `ndimage.rotate`.
- Discrete latent = an **emergent** topological charge in a multi-defect CGL field (charges nucleate /
  annihilate / interact), with a lossiness mode that **can** corrupt the winding (so the determine genuinely
  *survives* the shadow, S2-grade, rather than being orthogonal to it).
- Wire **C-CHANNEL** as a hard gate; report **both** G-KINV forms; keep the bare-vortex ablation as the
  load-bearing-ness check.

## Honest boundaries
- Synthetic; the rotational phase is an **imposed spatial operation**, not the field's dynamical phase.
- The substrate is a small symmetry-orbit of ~one spiral family (the ±q mirror pair × `r_bases` near-identical
  bases) — acceptable for a *mechanism* demo (S0/S1 use a fixed template), NOT for a substrate claim.
- Forward-only; deterministic re-run (byte-identical features).

## Files
- `scripts/reaction_diffusion_phase_shadow.py` — the v2 probe (CGL spiral library; phase-resist +
  chirality-determine; the fixed-λ G-KINV test).
- `scripts/test_reaction_diffusion_phase_shadow.py` — frozen test locking the HONEST picture (genuine
  charFun via the finite-mean control; the bare-vortex ablation = RD not load-bearing; C-CHANNEL leak).
- `results/atlas/h8v2/{calibrate,frozen}.json`.
- `docs/atlas/H8V2_RD_PHASE_DEFECT_PREREG.md` — the locked pre-reg (with the post-review correction).
