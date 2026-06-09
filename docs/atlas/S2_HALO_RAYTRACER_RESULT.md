# S2 follow-on — full per-habit polarized halo raytracer (RESULT)

> **2026-06-09.** One genuine ray-MARCHED polyhedron engine removes the schematic fixed-face boundary of
> `S2_HANDEDNESS_MAP_RESULT.md`: the 22°/46° halos, the Können-validated linear polarization, and the
> per-feature ±V circular handedness map now all EMERGE from the same forward model. Pre-registration:
> `S2_HALO_RAYTRACER_PREREG.md`. Forward-model / ray-optics tier; deterministic (seeded); NOT
> public-eligible. Attribution: Wendling 1979 / Macke 1996 / Greenler / Tape (halo raytracing); Können &
> Tinbergen 1991 (polarimetry); Mishchenko / Mueller–Stokes; ice n=1.31 Warren & Brandt, Δn=+0.0014.

## Headline — all three gates pass from one engine
A convex hexagonal-ice **polyhedron** + vector ray-march (`exit_face`) + a **polarized ray-tree**
(entry-Fresnel → birefringence → inter-interface frame rotation → TIR-phase retarder / **partial
Fresnel reflection** → exit, energy in Stokes I) + per-habit orientation samplers reproduces the halos
AND their polarization, with the ±V handedness result preserved **without the schematic face boundary**.

## Stage 1 — `mueller_fresnel_reflect` (the partial reflection the chain lacked)
`R_s=r_s², R_p=r_p²` (energy-correct, no geometric factor); lower 2×2 `= sign(r_s·r_p)·√(R_sR_p)` —
sub-critical reflection is real (`sinδ=0`) so a single partial reflection makes **no circular pol**
(the physics guarantee that V comes only from TIR phase + birefringence). Validated: energy `R+T=1`;
normal-inc `R=0.0180`; `V=0`; Brewster `R_p→0`; **continuity to `mueller_tir` at θc** (lower-2×2 → 0.999
→ 1); physicality. All pass.

## Stage 2 — geometry + the frame-bookkeeping bridge
- Convex `exit_face` correct (centre +x → prism face 0 at t=a; +z → top basal).
- **Bridge: the raytracer's interface composition reproduces the validated `s2_handedness_map.trace_ray`
  (V/I + sky azimuth) to `max|err| = 0.0` over 240 plate paths** — the multi-bounce frame-rotation /
  Mueller convention is exactly the one already validated against the Fresnel-rhomb anchors.
- **Physical realizability `I ≥ √(Q²+U²+V²)` holds for every deposit** across a ray-marched ensemble (a
  frame-sign bug would violate this) — the independent check that `trace_tree`'s actual loop is sound.

## Stage 3 — the three pre-registered gates (e_sun = 20°, K=1)
| Gate | Result | Anchor |
|---|---|---|
| **1 — geometry** | 22° halo peak **22.25°**; 46° (broad) halo peak **47.75°** (edge 45.73°) | `halo_min_deviation` 21.84° / 45.73° ✓ |
| **2 — linear pol (Können)** | 22°-ring DoP **3.74%**; **U/I = +0.016% ≈ 0**; `\|U/Q\| = 0.004` (pure radial Q) | Können Fresnel floor ~3.7%, U=0 ✓ |
| **3 — circular V (payoff)** | per-ray peak **\|V/I\| = 5.36%** (TIR-only); **net `\|∮V\|/∮\|V\|` = 0.87%**; V(az) antisymmetric (residual 9.4%) | per-feature V real + V-analog of Können U=0 ✓ |

So the full raytracer **independently reproduces** the schematic map's two findings — per-feature ±V is
real (few-% per ray) and the achiral-ice display nets to ~0 (antisymmetric about the principal plane) —
now with the path geometry *derived* by ray-marching rather than imposed.

## Stage 5 — the K≥2 stretch: the parhelic circle (multi-bounce)
Two additions push the engine to multi-bounce features: (1) the **external-reflection glint** at the
entry face (`include_external` — a *vertical* crystal face is a vertical mirror, so it preserves the
ray's elevation), and (2) `K=2` internal reflections. On oriented (plate / column) crystals these
produce the **parhelic circle** — a horizontal white circle through the sun.

| feature (plate, e_sun=20°, K=2) | result |
|---|---|
| **parhelic circle** | a feature at **constant elevation = sun elevation (21°)** spanning the **full azimuth circle** (all 24 of 24 15°-bins) |
| **120° parhelia** | the recognizable **two-internal-reflection (K=2)** bright spots at az ≈ ±120° (`I(120°) > I(gap)`, robust across seeds) |
| **PHC linear pol** | net DoP **5.9%**, **U/I ≈ 0** (radial, mirror-symmetric) |
| **PHC circular V** | per-feature ±V from the multi-bounce TIR paths, **net `\|∮V\|/∮\|V\|` → ~0** (0.3–2.7%) — the achiral net-zero extends to the parhelic circle |

So the same engine, at `K=2`, generates the parhelic circle and its 120° parhelia, and the **net-zero
±V handedness law extends to this multi-bounce feature** — the V-analog of Können's U=0 is not special to
the single-TIR arc.

## Pre-registered scorecard
| Gate | Result |
|---|---|
| Stage 1 reflection Mueller (energy / no-V / Brewster / θc-continuity / physicality) | **PASS** |
| Stage 2 polyhedron + frame bridge (== trace_ray to 1e-9) + realizability | **PASS** |
| GATE 1 geometry — 22°/46° at the analytic radii | **PASS** |
| GATE 2 linear pol — Fresnel-floor DoP ~3.7%, U=0 | **PASS** |
| GATE 3 circular V — per-feature %-level, net → 0 | **PASS** |
| GATE 4 parhelic circle (K=2) — const elevation, 120° parhelia, net-zero V | **PASS** |
| kill: no 22° halo / linear pol ≠ Können / V not antisymmetric | not triggered |

## Honest boundaries
- **Ray-optics / forward-model tier**; geometric ray-tree, no diffraction dressing or size distribution.
- **The ~9% Können *peak* linear DoP is NOT modeled here** — it arises from the birefringent **two-image
  angular split** (o/e rays exiting ~0.11° apart, each highly polarized), which needs o/e ray-splitting
  at two indices. This engine models birefringence as a **phase retarder** (→ the circular V of Gate 3),
  so its linear anchor is the **Fresnel floor (3.74%) + U=0**; the two-image split stays with the
  analytic `s2_konnen_validate.py`. The two engines are complementary, not redundant.
- The bridge is a **convention lock** (the interface ops compose identically to `trace_ray`); the
  *independent* validation of the ray-marched loop is the realizability + the U=0 / DoP / net→0 gates.
- **V stays forward-model**; the linear pol is the observed-tier anchor (Stage A). `K=1` milestone =
  22°/46° + sundogs + single-TIR V; the **`K≥2` stretch (DONE) = the parhelic circle + 120° parhelia**.
  Further multi-bounce features (subhelic / anthelic / Liljequist arcs, `K≥3`) and habits beyond
  random/plate/column (Parry, pyramidal, Lowitz) remain open. The external-reflection glint
  (`include_external`) is the white-PHC mechanism; it is off by default so the `K=1` gates are unchanged.

## Files
- `scripts/s2_optics.py` — `mueller_fresnel_reflect` added (the partial-reflection diattenuator).
- `scripts/s2_halo_raytracer.py` — the polyhedron + polarized ray-tree engine + samplers + gates.
- `scripts/test_s2_halo_raytracer.py` — frozen test (Stage 1/2/3 + bridge + realizability), deterministic.
- `docs/atlas/S2_HALO_RAYTRACER_PREREG.md` — pre-registration.
- `scripts/s2_handedness_map.py` — `trace_ray` = the bridge reference; `s2_konnen_validate.py` — Gate-2 anchor.
