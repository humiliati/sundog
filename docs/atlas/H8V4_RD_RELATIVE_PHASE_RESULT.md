# H8 v4 Result — symmetry-invariant relative-phase resist (substrate S3Δ): **NULL** (obstacle sharpened)

> **2026-06-09.** Against `docs/atlas/H8V4_RD_RELATIVE_PHASE_PREREG.md`. The "one more shot" at escaping the
> H8 obstacle with a latent that is a phase **and** symmetry-invariant. **Killed at the pre-registered crux
> go/no-go (KILL-GEOMETRIC):** the relative phase Δθ is recoverable from a **static** superposition of two
> vortices → it is a **geometric** quantity → not load-bearing. The escape from the symmetry-orbit trap was
> real, but Δθ fell into a **broader geometric trap**. NOT public-eligible; frozen-as-portfolio. A clean,
> fast, pre-registered NULL — the most-likely (~55%) outcome, and the sharpest boundary in the saga.

## Headline — KILL-GEOMETRIC at the crux
The decisive surrogate **SUP2** (two **non-interacting** superposed vortices at the same geometric Δθ, no PDE
coupling) recovers Δθ from its own SO(2)-invariant registered features **better** than the interacting pair:
| candidate (own-R² of Δθ, within-distribution — the *valid* test) | own-R² |
|---|---|
| interacting evolved CGL pair | 0.828 |
| **SUP2 (static superposition, zero dynamics)** | **0.929** |
By the pre-registered gate (`SUP2 own-R² ≤ 0.30` for load-bearing), **0.929 ≫ 0.30 → KILL-GEOMETRIC** → Δθ is
geometric/substrate-agnostic → **NULL**. The interaction is not needed (it only adds noise); a static
superposition reproduces the relative-phase signal. No full build was warranted — the crux settled it.

## Why it escaped one trap but not the broader one
- **Escaped the symmetry-orbit trap (genuinely):** Δθ = θ₁−θ₂ is invariant under the global SO(2) (a rigid
  rotation shifts both cores equally), so it is **not** a coordinate on any single symmetry orbit — unlike
  v2's rotation φ and v3's temporal τ. This was the real, novel property v4 set out to exploit.
- **Fell into the geometric trap:** Δθ is an **interference phase** of two superposable structures, and the
  inter-core interference pattern at a given Δθ is reproduced by a static superposition (own-R²=0.929 with
  zero dynamics). So "symmetry-invariant" was **necessary but not sufficient** — the latent must also be
  **non-geometric** (not reproducible by any static template *or superposition*), which the relative phase is
  not.

## The methodological win (the v3 fix worked)
The crux used the **valid** load-bearing test — **own-R² within each distribution** (does the surrogate
recover Δθ from *its own* frames?) — NOT v3's vacuous train-real/test-surrogate cross-test. SUP2's high
own-R²=0.929 is a genuine "the static superposition has the mechanism," not an out-of-distribution artifact.
The v3 lesson held: a load-bearing test must ask whether the surrogate can do the task *within its own data*.

## The H8 obstacle — now triangulated FOUR ways (sharpened, still a conjecture)
On the RD substrate, **every charFun-resisting latent exhibited is GEOMETRIC** (reproducible by a static
template or superposition — whether a symmetry-orbit coordinate or a symmetry-invariant interference phase),
hence **NOT load-bearing**; while the genuinely RD-dynamical quantities are **finite-mean → DETERMINE**.
- **v1** wavelength — finite-mean → determines.
- **v2** rotation phase — SO(2) symmetry-orbit coordinate → geometric → not load-bearing.
- **v3** temporal phase — also spiral orientation (SO(2)) → geometric → not load-bearing (+ determine-type tail).
- **v4** relative phase — symmetry-INVARIANT but a geometric interference phase → a static superposition
  recovers it → not load-bearing.
**Sharpened conjecture:** `charFun-resist ⟺ geometric (static-template/superposition-reproducible) ⟹ not
load-bearing`; `load-bearing ⟹ finite-mean ⟹ determine-type`. Four serious, differently-designed attempts,
each over-claim caught by the lab's adversarial review (v1 false prize, v2 hollow load-bearing, v3 vacuous
discriminator) and verified. Stated as a conjecture on the CGL-spiral RD family, not a proven law.

## Honest boundaries
- Crux-level result (the pre-registered go/no-go), not the full frozen apparatus — but the gate it fires
  (KILL-GEOMETRIC) is decisive and the own-R² test is the valid one.
- The bug found en route is itself instructive: a *product* two-vortex ansatz makes Δθ cancel; the relative
  phase only lives in a *superposition*, which is exactly why it is geometric.
- Δθ ∈ [0,2] rad (bounded, no wrap); SO(2)-invariant registration (core-detection + axis alignment).

## Files
- `scripts/reaction_diffusion_relphase_shadow.py` — the v4 go/no-go probe (interacting pair vs SUP2, Δθ via
  SO(2)-invariant registration, own-R² head-to-head).
- `docs/atlas/H8V4_RD_RELATIVE_PHASE_PREREG.md` — the locked pre-reg (KILL-GEOMETRIC fired as pre-committed).
