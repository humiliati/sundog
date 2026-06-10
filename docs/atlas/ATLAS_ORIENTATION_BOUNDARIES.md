# Atlas orientation-boundary forward fits — the raytracer's reach (a nuanced result)

> **2026-06-09.** Tests whether the full polarized ray-marcher (`s2_halo_raytracer.py`) can
> INDEPENDENTLY re-derive the Phase-7 *orientation-driven* boundary elevations that Phase 12 left to the
> caustic-map / admissibility machinery. The honest answer is **partial**: it re-derives the two **TIR
> admissibility walls** but not the **topological merge** or the **wobble-masked parhelion-off** wall.
> NOT public-eligible. Builds on `ATLAS_PHASE12_POLARIZATION_PHASEDIAGRAM.md` + `ATLAS_PHASE7_PHASE_DIAGRAM.md`.

## CONFIRMED — the two TIR admissibility walls (a feature TIR-vanishes / appears)
These are the *clean* kind of boundary: a refraction path crosses the critical angle, so a feature
switches off (or on) sharply. The raytracer reproduces both, by the collapse/appearance of the
**90°-wedge (46-family) flux** at the relevant sky region:

| wall | derived | raytracer signature | result |
|---|---|---|---|
| **CZA off** | **32.196°** (`arccos√(n²−1)`) | near-**zenith** 90°-wedge flux: 2643 (e=20) → 2866 (28) → 1181 (31) → **378 (32) → 91 (33) → 0 (35)** | collapses through **31–35°**, brackets 32.196° ✓ |
| **CHA on** | **57.804°** (`arcsin√(n²−1)`) | near-**horizon** 90°-wedge flux: **0 (≤55) → 631 (57) → 2358 (58)** → 12075 (63) | appears through **55–58°**, brackets 57.804° ✓ |

The two are exact complements (`CZA + CHA = 90.000°`, both TIR onsets on the same 90° basal/side wedge).
The **full ray-marcher independently confirms the Phase-7 component-B walls** — the near-zenith CZA
flux switches off, and the near-horizon CHA flux switches on, right at the derived elevations (within
the §0.2 ~1–2° chromatic/disk smear band).

## NOT cleanly fittable (confirming the Phase-12 honest boundary for these)
| wall | derived | why the raytracer can't extract it |
|---|---|---|
| column UTA+LTA **merge** | 29.71° | a **topological A₃-metamorphosis**: the circumscribed loop's connecting **sides** (at the sun's elevation) are intrinsically faint — three detectors (side-fill at el≈e, lateral-az fraction, around-ring-angle side fraction) all stay **flat ~0–1%** across the merge. The features form; the *transition* doesn't extract from MC flux. |
| plate **parhelia off** | 60.74° | the parhelion's disappearance is **masked by wobble-broadened 22°-ring background** at the sun's elevation: the band flux persists (127 at e=55 → 177 at e=67, *past* the wall) instead of vanishing. The 60°-vertical-face admissibility wall isn't isolated from the halo background. |

## The lesson
The raytracer re-derives **admissibility walls** (TIR onsets — a sharp feature switch) cleanly, but not
**topological merges** (faint connecting geometry) or **wobble-masked admissibility walls** (where the
feature's own halo background swamps the disappearance). So Phase 12's honest boundary is **refined, not
removed**: 2 of the 4 orientation boundaries (the CZA/CHA TIR pair) are now **raytracer-confirmed**; the
merge + parhelion-off remain the authority of `atlas_caustic_map` / `atlas_forward_sweep`. A clean,
honest partial result — exactly where the forward instrument's reach ends.

## Files
- `scripts/atlas_orientation_boundaries.py` — the CZA/CHA confirm detectors + the merge/parhelion null
  detectors + the report.
- `scripts/test_atlas_orientation_boundaries.py` — frozen test (CZA/CHA brackets + the two nulls).
- `scripts/s2_halo_raytracer.py` — the engine; `docs/atlas/ATLAS_PHASE7_PHASE_DIAGRAM.md` — the derived walls.
