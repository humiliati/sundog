# S2 follow-on — full per-habit polarized halo raytracer (PRE-REGISTRATION)

> **2026-06-09.** Removes the one disclosed boundary of `S2_HANDEDNESS_MAP_RESULT.md`: that map imposed
> a **schematic fixed face-sequence** (enter side → TIR basal → exit side) and never ray-marched the
> crystal. Here one genuine ray-MARCHED polyhedron engine forward-models the halos AND their
> polarization, so the 22°/46° rings, the Können-validated linear pol, and the ±V circular map all
> emerge together. Forward-model / ray-optics tier; NO inversion; NOT public-eligible; deterministic
> (seeded). Attribution: Wendling 1979 / Macke 1996 / Greenler / Tape (halo raytracing); Können &
> Tinbergen 1991 (polarimetry); Mishchenko / Mueller–Stokes; ice n=1.31 Warren & Brandt, Δn=+0.0014.

## What it adds over the schematic map
A convex hexagonal-ice **polyhedron** (8 half-spaces) + vector ray-marching (`exit_face` = argmin
positive t over outward faces) + a **polarized ray-tree** (entry-Fresnel → per-segment birefringence →
inter-interface frame rotation → TIR-phase retarder / **partial-reflection diattenuator** → exit, with
energy in Stokes I) + per-habit orientation samplers (random SO(3), plate, column) + sky binning. The
new physics piece is `s2_optics.mueller_fresnel_reflect` (the partial Fresnel reflection the
transmission-only chain lacked).

## Stage gates (PASS / kill)
| Stage | Gate | PASS | Kill / falsify |
|---|---|---|---|
| 1 | `mueller_fresnel_reflect` | energy `R+T=1`; normal-inc `R=0.0184`; **one partial reflection makes V=0**; Brewster `R_p→0`; **continuity to `mueller_tir` at θc**; physicality | fails any analytic anchor → mechanism wrong |
| 2 | geometry + frame bridge | convex `exit_face` correct; the interface composition **reproduces the validated `s2_handedness_map.trace_ray` (V/I + sky az) to 1e-9**; `I≥√(Q²+U²+V²)` across the ensemble | bridge mismatch → a frame-rotation sign bug (corrupts V, not I) |
| 3a | **GEOMETRY** | random ensemble peaks at **21.84°±0.7°** and **45.73°±1.2°** (`halo_min_deviation`) | no 22° halo → geometry bug |
| 3b | **LINEAR POL (Können)** | 22°-ring DoP at the **Fresnel floor ~3.7%** (`[2.5,5.5]%`), **U/I≈0** (mirror symmetry), pol essentially pure Q | DoP ≠ floor or U≠0 → Mueller/frame bug |
| 3c | **CIRCULAR V (payoff)** | per-feature **|V/I|%-level** (>1%); the achiral display **nets to ~0** (`<3%` = V-analog of Können U=0); `V(az)` **antisymmetric** | per-feature V≈0 → no mechanism; large net V → real population handedness (surprise, against prior) |

## Honest boundaries (carried to the receipt)
- **Ray-optics / forward-model tier.** Geometric ray-tree, no diffraction dressing, no size
  distribution; "a clean null is a success".
- **The ~9% Können *peak* linear DoP is NOT targeted here.** That peak comes from the birefringent
  **two-image angular split** (the o- and e-rays exiting at slightly different angles, ~0.11°, each
  highly polarized) — an effect requiring o/e ray-splitting at two indices. This engine models
  birefringence as a **phase retarder** (→ the circular V of Gate 3c), so its linear-pol anchor is the
  **Fresnel floor (~3.7%) + U=0**; the two-image split stays with the analytic `s2_konnen_validate.py`.
- **V stays forward-model**; the linear pol is the observed-tier anchor (Stage A). Milestone = single
  internal reflection (`K=1`): 22°/46° + sundogs + the single-TIR V mechanism. Multi-bounce features
  (parhelic circle) are the `K≥2` stretch.
- Habits: random / plate / column (Parry/pyramidal/Lowitz out of scope here).
