# S2 deepening — the per-feature ±V(φ) handedness map (RESULT: a bounded-positive + structural null)

> **2026-06-09.** Closes the one **in-house** deliverable owed by `S2_MEASURED_SKY_SCOPE.md`: the
> pre-registered ±V sky profile the scope *names* — "per-feature ±V ~1% antisymmetric around the ring,
> integrating to ~0" — **but never computed**. Pre-registration: `S2_HANDEDNESS_MAP_PREREG.md`.
> Forward-model only (NO inversion). NOT public-eligible. Attribution: Fresnel-rhomb TIR phase
> (Born & Wolf §1.5.4 / Hecht §4.7); Können & Tinbergen 1991 (the measured 22° linear-pol + U=0
> cancellation we anchor the V-null to); Mueller–Stokes formalism; ice birefringence Warren & Brandt 2008.

## Headline — both pre-registered claims confirmed
1. **The chain was blind to the primary mechanism, now fixed.** The transmission-only Mueller chain
   (`mueller_fresnel` returns `None` on TIR) returned **identically zero V for any TIR ray** — it never
   modeled the **TIR phase retardance** (the Fresnel-rhomb s–p phase), the *primary* linear→circular
   mechanism on exactly the TIR-rich features the scope flagged. `tir_retardance()` adds it.
2. **Claim A — per-feature ±V is REAL (bounded-positive).** The TIR phase **alone** produces %-level
   circular polarization; with ice birefringence it is larger.
3. **Claim B — the achiral-ice display nets to ~0 (structural null).** `V(φ)` is azimuthally
   **antisymmetric** and `∮V dφ → 0` — the **V-analog of Können's measured `U = 0`**, forced by ice's
   σ_v mirror symmetry — while a single *chiral sub-path* keeps full net handedness, proving the
   cancellation is mirror-partner physics, **not** per-ray averaging.

## Stage 1 — the TIR-phase retarder vs analytic Fresnel-rhomb anchors (mechanism gate)
`tir_retardance(θ, n1, n2) = 2·arctan( cosθ·√(sin²θ − (n2/n1)²) / sin²θ )` on TIR; 0 otherwise.
`mueller_tir(...)` wraps it as a **pure retarder** (no diattenuation, energy-conserving).

| anchor | model | reference |
|---|---|---|
| ice (1.31→1) peak retardance | **δ_max = 30.57° @ θ = 59.10°** | analytic 30.56° @ 59.1° ✓ |
| glass (1.51→1) peak | **δ_max = 45.94°** | analytic 45.9° ✓ |
| glass 45° crossings (the rhomb pair) | **48.6° and 54.6°** bracketing the peak | textbook Fresnel-rhomb angles ✓ |
| below crit / at grazing | **δ = 0** (both) | required ✓ |
| Mueller matrix | **pure retarder** (M₀₀=1, no row/col-0 coupling) | energy-conserving ✓ |

## Stage 2 — the forward ±V(φ) map (genuine 3D polarization ray trace)
Hexagonal ice **plate** (basal horizontal, optic axis vertical) spun uniformly about the vertical;
path = entry-refract → **TIR off the basal face** → exit-refract; the full Mueller chain is
entry-Fresnel × birefringent-retarder × **TIR-retarder** × exit-Fresnel, propagated with real vector
Snell/TIR and proper inter-interface frame rotations. The feature is the sub-horizon basal-TIR arc at
elevation `−e_sun`. **Achirality is honored by summing the full physical exit set {faces 1..5}** — a
mirror-closed set — so the antisymmetry **emerges from the enumeration**, not a hand-picked mirror pair.

Result (sun elevation 20°; the structure is robust across 10–35°, with per-ray |V/I| growing
5.8%→19.6% with elevation, all other metrics unchanged):

| quantity | TIR-only (`dn=0`) | full chain (`dn=DN_ICE`) | meaning |
|---|---|---|---|
| **Claim A** per-ray peak `\|V/I\|` | **3.78%** | **11.75%** | the mechanism's per-ray circular-pol strength |
| **Claim A** flux-avg feature `\|V/I\|` | 0.83% | **2.56%** | the observable a polarimeter would see |
| chiral sub-path net `\|∮V\|/∮\|V\|` | **100.0%** | **100.0%** | a one-handed sub-ensemble does NOT self-cancel |
| **Claim B** achiral TOTAL net `\|∮V\|/∮\|V\|` | **0.00%** | **0.00%** | net population handedness cancels (→ U=0 analog) |
| **Claim B** antisymmetry residual | **0.00%** | **0.00%** | `V(φ)` odd about the principal plane |
| self-mirror exit face (3) net V | **2.3e-18** (per-ray `\|V/I\|`=0.96%) | — | rays carry V but +/- pairs cancel exactly |

So a single chiral ray-path carries a definite, one-signed handedness (~few % to ~12% per ray); the
mirror-partner paths that achiral ice *must* supply with equal weight produce the opposite handedness at
the mirror azimuth, so the **whole display is antisymmetric and nets to zero**. That is precisely the
scope's stated "ideal honest result: a ±V map that does both."

## Pre-registered scorecard
| Gate | Result |
|---|---|
| Stage-1 TIR-phase reproduces ice δ_max + the glass rhomb pair | **PASS** (exact) |
| Claim A — per-feature `\|V/I\|` ≥ 1% (TIR phase makes circular pol) | **PASS** (per-ray 3.8–11.8%; full-chain observable 2.56%) |
| Claim B — `V(φ)` antisymmetric, `∮V → 0` | **PASS** (net 0.00%, residual 0.00%) |
| anti-tautology — a chiral sub-path keeps net handedness | **PASS** (chiral net 100%) |
| kill: peak `\|V/I\|` ≈ 0 even with TIR phase | not triggered |
| kill/positive: large net `∮V` (real population handedness) | not triggered (net → 0, as disfavored-prior expected) |

## Honest boundaries
- **Forward-model tier.** Genuine 3D polarization ray trace with the validated Mueller pieces, but a
  **schematic fixed face-sequence** (we do not ray-march the polyhedron to confirm the ray geometrically
  stays inside / hits those faces in order); all incidence angles are physical (Snell). It demonstrates
  the *mechanism and its symmetry*, **not** a per-habit halo raytracer and **not** a measured-sky
  detection (Stage C stays external/collaboration-gated).
- The feature here is a **sub-horizon basal-TIR arc** (the cleanest guaranteed-TIR plate path); the
  TIR-rich named arcs (parhelic circle, subhelic, 46° grazing) share the same mechanism and the same σ_v
  net-zero, which is the transferable result.
- Claim A is **defensible** (per-feature ±V from TIR + birefringence; rainbow-TIR precedent). Claim B's
  net-null is the **expected/honest** outcome and the V-analog of Können's `U=0`. **V stays
  forward-model**; the linear polarization remains the observed-tier anchor (Stage A). The §0.2
  ray-optics / size-floor caveats travel unchanged.

## Files
- `scripts/s2_optics.py` — `tir_retardance` / `mueller_tir` added (the missing TIR-phase mechanism).
- `scripts/s2_handedness_map.py` — the 3D polarization ray trace + the ±V(φ) sweep + claim metrics.
- `scripts/test_s2_handedness_map.py` — frozen test (16/16 PASS): Fresnel-rhomb anchors + Claims A/B +
  the anti-tautology and self-mirror-symmetry checks. Deterministic (no RNG).
- `docs/atlas/S2_HANDEDNESS_MAP_PREREG.md` — the pre-registration (claims + kill criteria).
- `docs/atlas/S2_MEASURED_SKY_SCOPE.md` — the parent roadmap (this closes its in-house deliverable).
