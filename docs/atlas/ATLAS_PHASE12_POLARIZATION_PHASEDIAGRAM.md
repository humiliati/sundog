# Atlas Phase 12 — the forward-generated POLARIZATION phase diagram

> **2026-06-09.** Closes the Phase-7 forward-model gaps with the session's polarized halo raytracer and
> adds a **polarization layer** to the classified (elevation × habit) phase diagram. The Atlas now
> carries not just *where* each feature is (the bifurcation-set fold radii) but *what a polarimeter
> sees* at it. Builds on `S2_HALO_RAYTRACER_RESULT.md` + `ATLAS_HALO_POLARIZATION_OBSERVABLE.md`. NOT
> public-eligible (frozen-as-portfolio). Attribution: Greenler/Tape/Cowley (geometry); Können &
> Tinbergen 1991 (the 22° polarization mechanism); the law DoP(R) is a synthesis, not new physics.

## What Phase 12 closes
Phase 7 flagged honest forward-model gaps: *"No random-orientation sky_grid model"*, *"No plate model
beyond the parhelion"*, *"No Parry wedge"*. The session's raytracer (`s2_halo_raytracer.py`) has the
**random / plate / column / Parry / pyramidal** habits and forward-generates the actual halos +
reflection/multi-path features — so it **is** the generator those cells lacked. It also reproduces, by
ray-marching, the one occlusion the old model only asserted:

**Horizon-clip occlusion (the random-orientation gap).** A ring of radius R around the sun (elevation
e) has its lowest point at sky-elevation `e − R`, so it clears the horizon (full ring) at `e ≥ R`. The
raytracer confirms it: the 22° ring's below-horizon flux fraction falls **36% (e < R) → 0% (e > R)** as
the sun crosses `e = R = 21.84°` — exactly the Phase-7 occlusion boundary. (Same for 9/18/20/24/35/46°
at e = their radii.)

## The polarization layer (the new content)
Every classified Atlas feature now carries a polarization signature, split by mechanism:

| feature | R | linear DoP(R) | pol signature |
|---|---|---|---|
| 9° pyramidal ring | 8.96° | **0.61%** | radial (U=0), no V |
| 18° pyramidal ring | 18.0° | 2.48% | radial (U=0), no V |
| 20° pyramidal ring | 20.0° | 3.06% | radial (U=0), no V |
| 22° halo / sundogs / tangent arcs / circumscribed | 21.84° | 3.65% | radial (U=0), no V |
| 23–24° pyramidal arcs | 23.82° | 4.35% | radial (U=0), no V |
| 35° pyramidal ring | 35.0° | **9.45%** | radial (U=0), no V |
| 46° halo / supralateral / infralateral / CZA / CHA | 45.73° | **16.23%** | radial (U=0), no V |
| parhelic circle (colored / internal-reflection parts) | — | — | **±V antisymmetric, net→0** (+ linear) |
| 120° parhelia | — | — | ±V antisymmetric, net→0 |
| subhelic / anthelic / Tricker arcs | — | — | ±V antisymmetric, net→0 |
| sub-horizon basal-TIR arcs | — | — | ±V antisymmetric, net→0 |

- **Refraction folds** (the A₂-fold rings/arcs) → the radial **DoP(R) ladder** `(1−cos⁴(R/2))/(1+cos⁴(R/2))`,
  U=0, no circular V — habit- and n-independent given R.
- **TIR / internal-reflection features** → the per-feature **±V handedness** (azimuthally antisymmetric,
  net integral → 0 = the V-analog of Können's U=0), with linear pol on top.

So the same bifurcation point now reads as a *geometry × polarization* cell: a refraction fold is a
radius **and** a radial DoP; a TIR feature is a locus **and** a net-zero ±V handedness map.

## Honest boundary (what stays with the Phase-7 machinery)
The **orientation-driven derived boundary elevations** — 60.74° plate-parhelia off, 29.71° column
UTA+LTA merge, 32.20°/57.80° CZA/CHA — remain the authority of the Phase-7 caustic-map / admissibility
machinery (`atlas_caustic_map`, `atlas_forward_sweep`). The raytracer **forward-generates** those
features, but does not cleanly re-derive the exact boundary *elevation* by simple flux-windowing (the
plate parhelion's disappearance needs a targeted Können-style feature fit, not a scattering-angle
window). Phase 12 therefore *closes the generator/occlusion gaps and adds polarization*; it does not
supersede the derived B-walls. The two are complementary.

## Files
- `scripts/atlas_phase12_polarized_phasediagram.py` — the polarization layer + the horizon-clip
  validation.
- `scripts/test_atlas_phase12_polarized_phasediagram.py` — frozen test (6/6 PASS).
- `scripts/s2_halo_raytracer.py` — the forward generator; `s2_optics.halo_pol_dop` — the DoP(R) law.
- `docs/atlas/ATLAS_PHASE7_PHASE_DIAGRAM.md` — the geometric phase diagram this layers onto.
