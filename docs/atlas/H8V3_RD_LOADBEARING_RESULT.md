# H8 v3 Result — load-bearing temporal-phase resist (substrate S3τ): **NULL** (obstacle confirmed, not escaped)

> **2026-06-09.** Against `docs/atlas/H8V3_RD_LOADBEARING_PREREG.md`. Milestone-1 (the load-bearing resist).
> **v3 does NOT achieve a load-bearing charFun resist.** My load-bearing discriminator (the ablation
> cross-test) turned out to be **vacuous** (an out-of-distribution detector, not a mechanism test), and the
> temporal phase τ is read off **spiral orientation = the v2 SO(2) symmetry coordinate** — it is
> determine-type (G-KINV fails). A 3-skeptic adversarial review + my own verification caught this before any
> claim — the **third** such catch in the H8 saga. NOT public-eligible; frozen-as-portfolio. Attribution:
> Aranson & Kramer (CGL); the Shadow-Invertibility / charFun laws; S0–S2.

## Headline — NULL, and WHY (owned plainly)
The pre-registered KILL-LB and KILL-PERIODIC both fire. The probe correctly banks `LOADBEARING_RESIST=False`
(G-KINV fails). The deeper finding is that the **ablation battery v3 was built around does not work** — it
cannot tell "the RD is load-bearing" from "the features look different." So v3 is hollow in the same family
as v2, *and* its centerpiece discriminator is invalid.

## The two killers (verified in the actual apparatus)
- **The cross-test is VACUOUS.** It trains a recoverer on REAL frames and tests it on SURROGATE frames; a
  low transfer-R² was read as "the surrogate lacks the dynamics." But a **real-RD field with byte-identical
  dynamics and merely column-PERMUTED features** "fails" it too:
  | cross-test | R² |
  |---|---|
  | real → real (2nd draw, positive control) | **0.866** |
  | real → real **with shuffled feature columns** | **0.000** |
  | within the shuffled distribution (own recovery) | **0.864** (τ is right there) |
  The cross-failure is 100% feature-layout shift, 0% "RD is load-bearing." So A1/A2/A3 "failures" prove
  nothing about the dynamics. The anti-vacuity guard (own-preflight-pass ⇒ cross-fail is meaningful) is
  logically broken — the real-RD shuffle control passes its own preflight (own R²=0.86) and still cross-fails.
- **The v2 SO(2) trap is NOT escaped.** τ is recoverable from rotation-only frames (rotation own cont₀=0.887),
  and **rotation → real cross-R² = 0.809** — rotation *does* encode τ. My A2 "fail" (real→rotation=0.000) was
  the cherry-picked direction. And the entire τ signal lives in the rotation-sensitive downsampled-field
  block; the radial "breathing" channel — which the docstring claimed "a pure rotation cannot fake" — recovers
  **exactly 0** (a dead channel, not a clean resist). τ is the spiral orientation, the same symmetry-orbit
  coordinate v2 identified.

## Pre-registered scorecard
| Gate | Result |
|---|---|
| G1 preflight (τ recoverable) | PASS at K=8 (cont₀≈0.85–0.90) — but the channel is orientation (SO(2)) |
| G3 resists (washes at fixed K=8) | washes to 0 — **but this is finite-K decoherence of a symmetry coordinate, not the law's resist** |
| **G-KINV (charFun)** | **FAIL** — cont(λ=2) vs K = {8:0, 64:0.17, 512:0.53} *rises* with K (LLN); half-life-vs-K also marches outward {8:0.75, 64:1.0, 512:2.0} — the v1 determine-type signature. The finite-mean channel **is the bulk rotation** (a rotation surrogate reproduces the field-block rise). |
| **LB-cont (load-bearing)** | **INVALID** — the cross-test discriminator is vacuous; retracted. Rotation encodes τ (rotation→real=0.81), so the dynamics are not shown necessary. |
| **NON-RIGIDITY preflight (KILL-RIGID)** | Never computed in code (only quoted). Against the *best* rigid rotation the equilibrated residual is **0.21–0.61** (mostly below the 0.4 anchor); the "0.7–1.2" was vs fixed-ω (angle-offset masquerading as non-rigidity). The spiral is **mostly rigidly rotating**. |
| **Overall** | **NULL** — KILL-LB (vacuous discriminator + rotation encodes τ) and KILL-PERIODIC (determine-type) both fire. |

## What is genuinely true (the bankable boundary — a CONJECTURE, not a law)
Across the three H8 results, on the **CGL-spiral RD family**, *every* latent channel exhibited is either a
**finite-mean symmetry coordinate (determine-type)** or **dead**; **no genuinely load-bearing charFun-resisting
channel of an RD field has been shown.**
- **v1:** the RD-emergent **wavelength** is finite-mean → DETERMINES (the charFun law predicted it).
- **v2:** the **rotation phase** charFun-resists but is the SO(2) symmetry coordinate → **not load-bearing**
  (a bare vortex reproduced every gate).
- **v3:** the **temporal phase** is *also* read off spiral orientation (SO(2)) → its finite-mean piece breaks
  G-KINV (determine-type), and its "load-bearing-ness" was a vacuous OOD artifact.
The recurring shape: **load-bearing ⟺ a finite-mean / determine-type dynamical channel; charFun-resist ⟺ a
symmetry-orbit coordinate (not load-bearing).** Stated as a *conjecture on one substrate family* (3 points),
not a general law. *Internal tension to keep honest:* in v3 the same SO(2) channel is both the "ablation-fail"
evidence and the "determine-type" evidence — which is itself why the dichotomy can't be claimed cleanly.

## Methodological lesson (reusable)
A load-bearing test must separate "**needs the dynamics**" from "**the features look different**." Train-on-real
/ test-on-surrogate transfer-R² conflates them (a real-RD permuted-column control fails it). A valid test must
train+evaluate each surrogate **within its own distribution** and ask whether it can produce a feature→latent
map at all (it can), or hold the feature pipeline fixed and remove the reaction term. The v2 head-to-head
(matched surrogate must FAIL on the *same* pipeline) is sounder than the cross-test.

## Integrity disclosures (owned)
- The cross-test is **retracted** as a load-bearing discriminator.
- KILL-RIGID was not computed in code (best-rigid residual is 0.21–0.61 — partly below threshold).
- G-KINV shipped only the fixed-λ form (the pre-reg demanded BOTH forms + a finite-mean control "no
  substitution this time"); the half-life-vs-K form + controls are reported here post-hoc and **agree**
  (determine-type), so no hidden pass — but it is a pre-reg-compliance deviation.
- No discrete/chirality (LB-disc) leg was shipped despite "carried"; A3's calibrate margin (0.325) is
  seed-fragile; "real" mode linearly interpolates frames (synthetic intermediates).
- The settle=1500 G-KINV numbers are ~0.13 transient-inflated; the equilibrated residual (settle≥8000)
  plateaus at ~0.48 (full) / ~0.35 (field) — determine-type survives, but the headline number was inflated.

## Files
- `scripts/reaction_diffusion_temporal_shadow.py` — the v3 probe (resist works at K=8; the cross-test is
  retracted; correctly banks `LOADBEARING_RESIST=False` via G-KINV).
- `scripts/test_reaction_diffusion_temporal_shadow.py` — frozen test locking the NULL (cross-test vacuity,
  the v2-trap rotation→real, determine-type G-KINV).
- `results/atlas/h8v3/{calibrate,frozen}.json`.
- `docs/atlas/H8V3_RD_LOADBEARING_PREREG.md` — the locked pre-reg (with the post-review correction).
