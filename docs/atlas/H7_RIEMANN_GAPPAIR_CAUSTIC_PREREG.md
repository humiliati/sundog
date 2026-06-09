# H7 pre-registration — the Riemann gap-pair caustic (empirical reconnaissance)

> **LOCKED 2026-06-08, before running.** A fresh hypothesis (post-slate). Points the lab's validated
> caustic machinery (the H4/H5 jet classifier; catastrophe optics, Berry/Nye lineage) at the **spectrum**
> instead of physical optics: do the consecutive **gap-pairs** of the Riemann zeros carry a **caustic** —
> a fold/cusp/non-analyticity — and does it match GUE/random-matrix universality or **deviate** (an
> arithmetic fingerprint)? NOT public-eligible; the rigorous Riemann lane stays frozen — this is an
> **empirical probe**, not a claim about RH. Attribution: Montgomery (pair correlation); Odlyzko (numerics);
> Berry & Keating (the zeros as a spectrum; spectral caustics; form-factor arithmetic).

## The honest prior (pre-committed, so we don't fool ourselves)
The unfolded zeros obey **GUE statistics to high precision** (Montgomery–Odlyzko). So the *most likely
outcome is a NULL on an "arithmetic caustic"* — the zeros' gap structure is RMT-universal. **A clean null
is the expected, bankable result.** The prize (low probability) is any caustic/deviation present in the
zeros but NOT in GUE. We design to (a) find the caustic if it exists, (b) cleanly rule it out if not, and
(c) separate "GUE-type structure" (real but expected) from "arithmetic deviation" (the prize).

## Object
The first `N` nontrivial zeros `γ_n` (target `N ~ 10⁴`; `mpmath.zetazero`, or fewer / Odlyzko tables if
compute-bound). **Unfold** with the Riemann–von Mangoldt smooth count `⟨N(t)⟩ = (t/2π)log(t/2π) − t/2π +
7/8`: `w_n = ⟨N(γ_n)⟩` (so mean spacing 1), gaps `δ_n = w_{n+1} − w_n`.

## Controls (the discriminators)
- **CUE/GUE** — eigenphases of random unitary matrices (size `m`), unfolded to unit mean spacing (bulk =
  GUE universality). The "the zeros are RMT" reference.
- **Poisson** — independent gaps (a random point process). The "no correlation" reference.
Every test is run on zeros vs CUE vs Poisson; a result counts only if it **separates** them.

## What "caustic" means here — two faces, both pre-registered

### Face 1 — the literal gap-pair caustic (the jet-classifier-native version)
The 2D density `ρ(u,v)` of consecutive pairs `(δ_n, δ_{n+1})` (KDE on a grid). "Caustic" = a fold/cusp in
its structure, detected by the **H4/H5-validated jet classifier** applied to the **gradient map**
`∇ρ : (u,v) ↦ (ρ_u, ρ_v)` (a 2D→2D map whose caustics = the parabolic/ridge singularities of `ρ`), and to
the conditional-mean curve `m(u) = E[δ_{n+1} | δ_n = u]` (folds = multivaluedness).
- **Pre-registered, honest:** the GUE gap-pair density is expected **smooth (no literal singularity)** — so
  Face 1 likely returns **no caustic**, cleanly ruling out the literal gap-pair caustic. *But* level
  rigidity makes consecutive GUE gaps **anti-correlated** (a large gap follows a small one), so `ρ` should
  show an **anti-correlation ridge** absent in Poisson — a classifiable structure even without a
  singularity. We report both: caustic (yes/no) and the GUE-vs-Poisson ridge.

### Face 2 — the form-factor caustic (where a real singularity + the arithmetic actually live)
The spectral form factor `K(τ) = |Σ_n e^{2πi w_n τ}|² / N` (equivalently the transform of the 2-level /
pair correlation). GUE has a **genuine non-analyticity at τ=1**: `K(τ) = min(|τ|, 1)` — a fold/kink, the
universality edge. Beyond `τ=1` the **primes contribute** (Bogomolny–Keating): the zeros' form factor
deviates from the flat GUE plateau. This is the real "gap-pair caustic," properly understood (gap-pair ↔
pair correlation ↔ form factor ↔ τ=1 caustic).
- **Pre-registered:** (P-F1) the zeros' `K(τ)` shows the **τ=1 caustic** (GUE universality); (P-F2) does it
  **deviate** from `min(|τ|,1)` near/beyond τ=1 — the arithmetic fingerprint — beyond the finite-`N` noise
  floor set by the CUE control?

## Pre-registered outcomes
- **NULL-A (expected):** no literal caustic in the gap-pair density; zeros' form factor = GUE (τ=1 kink,
  flat beyond) within the CUE noise floor → "the gap-pair caustic, literally, isn't there; the zeros are
  RMT-universal." Clean, bounded, bankable.
- **STRUCTURE (likely):** the gap-pair density carries the GUE **anti-correlation ridge**, classifiably
  distinct from Poisson; the form factor shows the τ=1 caustic. The zeros' gap-pairs carry RMT structure —
  real, if expected.
- **THE PRIZE (low prob):** a caustic / form-factor deviation present in the **zeros but not CUE** —
  arithmetic. Face 1 cusp not in GUE, or Face 2 deviation beyond τ=1 above the noise floor.

## Kill criteria
- KILL (no power) if zeros, CUE, and Poisson are **all indistinguishable** on every test → compute more
  zeros / wider window before interpreting.
- The arithmetic-caustic claim is KILLED (→ NULL-A) if the zeros track CUE within its finite-`N` noise
  floor on both faces.

## What the jet classifier must see to count (Face 1)
A genuine caustic: `∇ρ` has a **caustic curve** (det of its Jacobian sign-changes) with a **cusp** (`c2=0`,
`|c3|` bounded — the H4 template) that is **present in the zeros and absent (or different) in CUE**. A
smooth blob (no det-Jacobian sign change) = no caustic = NULL-A on Face 1.

## Honest boundaries
- Rigorous Riemann lane frozen; this is empirical reconnaissance, NOT a number-theory result or an RH claim.
- Finite-`N` statistics: every "deviation" must clear the CUE noise floor (same `N`, same estimator).
- Forward/measure-only; the catastrophe labels are *computed* (jet classifier), not asserted.

## Files (to be produced)
- `scripts/riemann_gappair.py` (zeros + unfold + gaps; CUE/Poisson controls; Face-1 density + jet
  classifier; Face-2 form factor).
- `scripts/test_riemann_gappair.py` (frozen).
- `docs/atlas/H7_RIEMANN_GAPPAIR_CAUSTIC_RESULT.md` (receipt, after, against this pre-reg).
