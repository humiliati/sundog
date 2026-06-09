# H5 pre-registration — the mirage (Δn, s) ladder as a classified caustic / cusp diagram

> **LOCKED 2026-06-08, before running.** Hypothesis #5 of slate `ww6koomb1`. The mirage analog of the
> halo Atlas: mirages are **caustics of the atmospheric-refraction ray map**, so mirage image-multiplicity
> should form a **classified ladder** (1 → 3 → 5 images: simple refraction → superior/inferior mirage →
> Fata Morgana) in a refraction control plane (Δn, s), with the transitions being **fold/cusp
> catastrophes** — which the H4-validated jet classifier (`atlas_jet_classify.py`) can label directly.
> NOT public-eligible. Attribution: standard mirage ray theory (Lehn; Können; Bruton); the
> catastrophe-optics view of natural focusing (Berry; Nye, *Natural Focusing and the Structure of Light*).

## The forward model (paraxial, flat-Earth)
Horizontally-stratified atmosphere n(h). Back-trace a ray from the observer (h_obs, range 0) at elevation
angle θ to the object range R; record the height it reaches, `h_target(θ)`. Paraxial ray ODE (n≈1, small
angles): `dh/dx = ψ`, `dψ/dx = n'(h)` (rays curve toward higher n). Refraction profile with a localized
inversion layer:

&nbsp;&nbsp; `n(h) = n0 − (Δn/2)·tanh((h − h0)/s)`  ⟹  `n'(h) = −(Δn/2s)·sech²((h−h0)/s)`

- **Δn** = index contrast across the inversion (∝ temperature-inversion strength; the "lens strength").
- **s** = inversion-layer thickness (sharp thin layer = small s).
The relevant order field is the **ray-transfer curve** `h_target(θ; Δn, s)`. An object at height H is seen
in every direction θ with `h_target(θ) = H`; **#images = #solutions**. Caustics (image merge/birth) are the
folds `dh_target/dθ = 0`.

## Why this is a cusp (the classifier connection, exact)
Chart `F(θ, Δn) = (Δn, h_target(θ; Δn, s))` (axis0=θ, axis1=Δn). Then `det DF = −∂h_target/∂θ`, so:
- **caustic** `φ=det DF=0` ⟺ `∂h_target/∂θ=0` = an image-merge fold;
- **cusp** `φ=0 ∧ c2=∂φ/∂θ=0` ⟺ `∂h_target/∂θ = ∂²h_target/∂θ² = 0` = the **1→3-image onset** (the
  transfer curve develops its S-shape — two folds born at a cusp as Δn crosses threshold);
- `c3 = −∂³h_target/∂θ³` bounded ⟹ **A₃ cusp** (exactly the H4 Stage-1 structure, but from real ray
  physics instead of a synthetic cubic).

## Pre-registered predictions
- **P1 (model validity):** a uniform gradient gives a MONOTONIC transfer (1 displaced image, no mirage); a
  localized inversion (tanh) gives a NON-monotonic S-shaped transfer → a 3-image band. Sign conventions
  reproduce inferior (n increasing upward → inverted image below) vs superior (inversion → image above).
- **P2 (the ladder):** as (Δn, s) vary, #images climbs in a DISCRETE ladder 1 → 3 → 5; the boundaries are
  smooth curves in the (Δn, s) plane (the bifurcation set).
- **P3 (cusp at the 1→3 onset):** the jet classifier reads the 1→3 transition as an **A₃ cusp** (a cusp
  present, |c3| bounded away from 0, corank-1 — matched to the H4 cusp template), NOT a fold-only and NOT
  a D₄.
- **P4 (higher rung):** a DOUBLE inversion (stacked layers → Fata Morgana) yields a 5-image band whose
  caustic structure is higher than a single cusp (a swallowtail / two cusps), classifier-distinguishable.

## Kill criteria
- KILL if the transfer is monotonic for ALL physical (Δn, s) (no fold → no mirage multiplicity → the model
  or framing is wrong).
- KILL the cusp claim if the 1→3 onset classifies as fold-only / |c3|→0 (higher) / corank-2.
- KILL the ladder claim if #images does not change in discrete steps with (Δn, s) (no bifurcation set).
- A NULL (e.g. only folds, no genuine cusp) is a legitimate bounded result.

## Validation anchors (the falsifiable grounding)
- Standard atmospheric refraction `dn/dh ≈ −2.3×10⁻⁸ /m`; a mirage needs the layer gradient strong enough
  to make the transfer non-monotonic over R — the model's mirage-onset Δn/s must be physically sensible.
- Image counts must match known phenomenology: inferior/superior ≤ 3 images (cusp), Fata Morgana 5+
  (higher). A nonsensical count (e.g. even-only, or unbounded) fails.

## Files
- `scripts/mirage_ladder.py` (Stage 1: ray-tracer + h_target + phenomenology validation).
- `scripts/mirage_ladder_sweep.py` or extend (Stage 2: (Δn,s) ladder + jet classification).
- `scripts/test_mirage_ladder.py` (frozen).
- `docs/atlas/H5_MIRAGE_LADDER_RESULT.md` (receipt, after, against this pre-reg).
