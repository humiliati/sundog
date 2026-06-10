# H8 v4 pre-registration — a SYMMETRY-INVARIANT charFun-resist latent (substrate S3Δ)

> **DESIGN LOCKED 2026-06-09, before any v4 run.** The "one more shot" at escaping the H8 obstacle. The
> trilogy (v1 NULL, v2 SCOPED, v3 NULL) all hit the same wall: every charFun-resist latent tried was a
> **symmetry-orbit coordinate** (v2 rotation φ, v3 temporal τ — both the single-vortex SO(2) phase), so a
> geometric transform of a static template reproduced it → not load-bearing; while RD-emergent scalars (v1
> wavelength) are finite-mean → determine. v4 tests whether the conjectured equivalence (**load-bearing ⟺
> determine-type; charFun-resist ⟺ symmetry coordinate**) is a law or CGL-single-spiral-specific, by a
> latent that is **a phase AND symmetry-invariant** — the exact conjunction the obstacle forbids. NOT
> public-eligible; honest NULL is a success (and the most likely outcome). Attribution: Aranson & Kramer
> (CGL vortex interactions); the Shadow-Invertibility / charFun laws; S0–S2.

## The escape latent — Δθ, the relative phase of an INTERACTING vortex pair (substrate S3Δ)
`xc = Δθ = θ₁ − θ₂`, the relative azimuthal phase of **two close, interacting CGL vortices** (separation a
few core radii — coupled but not immediately annihilating). Body: integrate the v2 CGL field `A=u+iw` from a
two-vortex IC, settle onto the coupled (co-rotating / orbiting) attractor; library-amortized like v2 (one
trajectory per `(base, Δθ)` tile). **Shadow** = mean over `K` subunits at **jittered relative phase**
`Δθ + λ·ξ`. **Read-out = SO(2)-INVARIANT relative features only:** register each frame on the pair's
center-of-mass axis (quotient out the global orientation), then read the inter-core **connector** (the
`|A|`-depletion bridge along the core–core line), the relative arm-linkage angle, and SO(2)-invariant
cross-features between the two core neighborhoods — depends on Δθ, blind to a global rotation.
- **Why it escapes the trap:** a global rotation adds the same constant to θ₁,θ₂ → Δθ **invariant** → Δθ is
  NOT a coordinate on any single symmetry orbit, so `field(Δθ) ≠ g·field(0)` for any group element `g` — the
  v2/v3 mechanism cannot apply. Yet Δθ is a phase (the connector geometry is periodic in Δθ → charFun-resist).
  For an **interacting** pair the connector is a nonlinear functional of the coupled PDE → candidate
  load-bearing.

## THE VALID LOAD-BEARING TEST (the v3 fix — never the vacuous cross-test)
v3's cross-test (train-on-real / test-on-surrogate transfer-R²) was a **vacuous OOD detector** — a real-RD
field with byte-identical dynamics but column-shuffled features "failed" it. **v4 NEVER cross-applies a model
across distributions.** Each candidate (the interacting pair; every surrogate) is **trained AND tested
ENTIRELY WITHIN ITS OWN distribution**, identical pipeline / n / seed / noise / probe / CV / thresholds. The
discriminator is the **own-R² gap**:
- **LB := real-pair own-R²(Δθ) ≥ 0.70  AND  SUP2 own-R²(Δθ) ≤ 0.30  AND  A-DIFFONLY own-R²(Δθ) ≤ 0.30**, at
  matched everything. (A surrogate that *reproduces the mechanism* recovers its OWN Δθ; only one *missing*
  the mechanism fails on its own data.)

### Surrogates (ablation battery, each evaluated within its own distribution)
| Surrogate | What it is | Role |
|---|---|---|
| **SUP2 (decisive)** | two NON-interacting superposed static vortices `T₁·V(x−c₁;θ₁)+T₂·V(x−c₂;θ₂)` at the same geometric Δθ, no PDE coupling | if SUP2 own-R²(Δθ) is HIGH → Δθ is geometric → **KILL-GEOMETRIC** (obstacle confirmed 4th time) |
| A-STATIC | one frozen interacting-pair frame, no dynamics | own-R²(Δθ)→0 (only one Δθ) — sanity |
| A-GLOBALROT | one interacting-pair frame rigidly rotated | own-R²(Δθ)→0 (global rotation can't change Δθ) — the **trivial-FAIL control** |
| A-DIFFONLY | diffusion-only (no reaction term) → no coupled limit cycle | must fail LB (proves the **reaction** coupling is load-bearing) |

### Corrected anti-vacuity guard (a positive-control PAIR — what v3 lacked)
The test is valid only if BOTH land: **(i) trivial-PASS:** a feature that DOES carry Δθ (the registered
connector, or Δθ written in as a clean linear feature) gives own-R² HIGH on the interacting pair — so a
"fail" is a real absence, not a dead probe. **(ii) trivial-FAIL / NULL:** Δθ randomized per-frame
(decoupled from the label) → own-R² collapses to ~0 — so a "pass" is real recovery, not label leakage.
(Column-shuffle cannot masquerade here: shuffling is within-distribution-preserving → own-R² unchanged.)

## G-KINV (BOTH forms + finite-mean control, NO substitution — the v2/v3 integrity fix)
Jitter population `Δθ + λ·ξ`. Report together: **FORM-1 half-life-vs-K** `λ*(K)`, `K∈{8,64,512,4096}` —
charFun-resist ⟺ `λ*` **saturates/flat** with K; finite-mean ⟺ `λ*` marches outward. **FORM-2 fixed-λ-vs-K**
at `λ_test` past the crossover — must **stay ≤0.15 and not rise** with K. **CONTROL** (side-by-side): a
finite-mean LLN scalar `d+λ_c·ξ` on the SAME pipeline MUST show the determine signature (λ* grows, cont
rises) — proving the apparatus discriminates. Both forms must agree RESIST; if either shows determine,
**KILL-PERIODIC** fires.
- **Broadband-charFun PREREQUISITE:** confirm the Δθ→feature spectrum is **broadband** (multiple azimuthal
  `m`). If single-tone `cos(mΔθ)`, charFun **recurs** (lattice-survive) → Δθ DETERMINES not resists →
  KILL-PERIODIC. (The exact v1/v3 backwards-K trap, here on the Δθ axis.)

## Crux go/no-go (BEFORE the frozen run — the decisive de-risk)
1. **Stable bound pair:** a CGL `(b1,c1,separation)` regime where two vortices co-orbit without annihilating
   over the recording window. If unfindable/unstable → v4 scopes to the synthetic-Δθ mechanism only (disclosed).
2. **The decisive SUP2 check:** does SUP2 recover Δθ from registered SO(2)-invariant features? If SUP2
   own-R² is HIGH (≈ the interacting pair), Δθ is geometric → **fast NULL** (obstacle confirmed), no full build.
   Only if SUP2 own-R² is LOW while the interacting pair is HIGH do we proceed to the full frozen run.

## Honest prior (pre-committed)
- **~55% NULL (KILL-GEOMETRIC):** SUP2 recovers Δθ — the relative angle of two superposed structures is
  geometric → Δθ is symmetry-invariant but STILL geometric/substrate-agnostic. Banks the sharpest boundary
  yet: *even relative phases between superposable RD structures are geometric coordinates; no charFun-resisting
  latent of an RD field is load-bearing.*
- **~30% load-bearing RESIST (the genuine escape):** SUP2 fails, the interacting pair passes, both G-KINV
  forms show charFun, the controls land — the conjecture is BROKEN (load-bearing charFun-resist exists on RD).
- **~15% the bound-pair regime is unfindable/unstable** → scope to synthetic-Δθ; partial/inconclusive.

## Kill criteria (each bankable)
- **KILL-GEOMETRIC:** SUP2 own-R²(Δθ) > 0.30 → Δθ geometric → NULL (obstacle confirmed).
- **KILL-PERIODIC:** Δθ determines (G-KINV either form shows finite-mean / single-tone spectrum) → NULL.
- **KILL-VACUITY-GUARD:** either anti-vacuity control fails (trivial-PASS low or trivial-FAIL high) → the test
  is invalid, do not report a verdict (fix the probe).
- **KILL-UNSTABLE:** no stable bound-pair regime → scope to synthetic-Δθ only, disclose.

## Honest boundaries
- The Δθ read-out requires SO(2)-invariant registration (core detection + axis alignment); registration
  error is a real risk and must be reported (a positive-control: registered-vs-unregistered own-R²).
- Library-amortized like v2 for the resist sweep; the surrogates need their own libraries.
- Forward-only; deterministic re-run; the load-bearing test uses own-R² gaps (never cross-R²).

## Files (to be produced)
- `scripts/reaction_diffusion_relphase_shadow.py` — the S3Δ probe (interacting-pair CGL library; Δθ
  registration + read-out; the within-distribution head-to-head battery; both G-KINV forms + controls).
- `scripts/test_reaction_diffusion_relphase_shadow.py` — frozen test locking the outcome + the valid-test
  controls.
- `results/atlas/h8v4/` + `docs/atlas/H8V4_RD_RELATIVE_PHASE_RESULT.md` — the receipt.
