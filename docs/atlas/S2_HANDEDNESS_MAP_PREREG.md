# S2 deepening — the per-feature ±V(φ) handedness map (PRE-REGISTRATION)

> **2026-06-09.** Closes the one **in-house** deliverable owed by `S2_MEASURED_SKY_SCOPE.md`: the
> pre-registered ±V sky profile that the scope names ("per-feature ±V ~1% antisymmetric around the
> ring, integrating to ~0") **but never computed**. Forward-model only (NO inversion). NOT
> public-eligible. Attribution: Fresnel-rhomb TIR phase (Born & Wolf §1.5.4 / Hecht §4.7); Können &
> Tinbergen 1991 (the measured 22° linear-pol + U=0 cancellation we anchor the V-null to);
> Mueller–Stokes formalism (Lukacs-style algebra); ice birefringence Warren & Brandt 2008.

## The gap this closes
The existing `s2_optics` Mueller chain is `mueller_fresnel` (transmission diattenuator, **returns
`None` on TIR**) × `mueller_retarder` (ice birefringence) × `mueller_fresnel`. So `ray_stokes`
returns **identically zero for any TIR ray** — the chain is *blind to the TIR phase retardance*, the
**primary** linear→circular mechanism on exactly the TIR-rich features (parhelic circle, subhelic /
46° grazing) where the scope predicts per-feature ±V. Stage 1 adds it; Stage 2 computes the owed map.

## Stage 1 — the TIR-phase retarder (must-pass mechanism gate)
`tir_retardance(θ, n1, n2)` = the Fresnel total-internal-reflection s–p phase. **Kill if it fails the
analytic anchors** (then the mechanism is mis-implemented and the whole deepening is void):
- ice (1.31→1): `δ_max = 30.56°` at `θ = 59.1°` (closed form `tan(δ_max/2)=cos²θc/(2 sinθc)`).
- glass (1.51→1): `δ_max = 45.9°`, and `δ(θ)` crosses **45° at two angles** bracketing `θ_max≈51.6°`
  (the textbook Fresnel-rhomb pair) — the canonical linear→circular demonstration.
- `δ = 0` at the critical angle and `δ → 0` at grazing (`θ → 90°`); pure retarder (energy-conserving).

## Stage 2 — the per-feature ±V(φ) forward model
Forward-model `V(φ)/I` around a TIR-rich feature (a single-TIR ray path: entry-refract → TIR bounce →
exit-refract) over a **mirror-symmetric crystal ensemble**, binning the exit Stokes vector by sky
azimuth φ. The full Mueller chain now = entry-Fresnel × birefringent-retarder × **TIR-retarder** ×
exit-Fresnel. Two falsifiable claims, **scored separately**:

| Claim | Statement | PASS criterion | Kill / falsify criterion |
|---|---|---|---|
| **A** (per-feature V real) | the TIR-phase + birefringence chain genuinely makes circular pol | peak `\|V/I\|` ≥ 1% somewhere on the feature | if peak `\|V/I\|` ≈ 0 (< 0.1%) even with the TIR phase → mechanism produces no V, Claim A falsified |
| **B** (net-V null) | `V(φ)` is azimuthally **antisymmetric**, `∮V dφ ≈ 0` | antisymmetry residual `‖V(φ)+V(−φ)‖ / ‖V(φ)‖ < 5%` **and** net `∮V/∮\|V\| < 5%` | if net `∮V/∮\|V\|` is large (> 20%) → a real **net population handedness** (surprising POSITIVE, against the disfavored-net-V prior) |

**The antisymmetry must EMERGE from the forward model** — it is *not* imposed. The mechanism: under
the principal-plane mirror (φ→−φ) the orientation distribution is invariant but each ray's retarder
fast-axis azimuth flips sign, flipping the `sinδ` (V-generating) terms → `V(−φ) = −V(φ)`. This is the
**exact V-analog of Können's measured `U = 0`** (U is likewise odd under the same mirror), so Claim B's
null is a *structural* statement, not an accident — and that is the scope's stated "ideal honest
result: a ±V map that does both [shows per-feature V AND nets to ~0]."

## Honest boundaries (carried into the receipt)
- Forward-model tier. Single-TIR schematic path with a parametrized orientation ensemble — it
  demonstrates the *mechanism and its symmetry*, it is **not** a full per-habit halo raytracer and not
  a measured-sky detection (Stage C stays external/collaboration-gated).
- Claim A = **defensible** (per-feature ±V from TIR+birefringence, rainbow-TIR precedent). Claim B's
  net-null is the **expected/honest** outcome; a Claim-B *failure* would be the only "positive,"
  and is disfavored on the population-handedness prior.
- The §0.2 ray-optics / size-floor caveats and the "V stays forward-model, linear pol is the
  observed-tier anchor" framing from Stage A/B travel unchanged.
