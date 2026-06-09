# Atlas predicted observable — the linear polarization of the classified halos

> **2026-06-09.** Connects the polarized halo raytracer (`S2_HALO_RAYTRACER_RESULT.md`) back to the
> Atlas (`SUNDOG_V_ATLAS.md`): the Atlas classifies *where* the halos are (the bifurcation-set fold
> radii R); this adds *what they look like to a polarimeter* — a **smooth observable on the classified
> set**. Forward (geometry → observable, NO inversion); falsifiable by sky polarimetry; NOT
> public-eligible. Attribution: Können & Tinbergen 1991 (the 22° mechanism); Greenler / Tape / Cowley
> (halo geometry); the Atlas classification (this repo). The generalization to arbitrary radius is a
> **synthesis**, not new physics.

## The law — one parameter (the halo radius) fixes the polarization
For a minimum-deviation **refraction** halo of angular radius `R`:

> **DoP(R) = (1 − cos⁴(R/2)) / (1 + cos⁴(R/2))** — radially oriented (E in the scattering plane),
> with **U = 0** (mirror symmetry) and **no net circular V** (refraction halo).

**Derivation.** At minimum deviation the internal/external incidence difference equals half the
deviation, `θᵢ − θₜ = R/2`. The Fresnel transmittance ratio through one face is `T_p/T_s =
1/cos²(θᵢ − θₜ)`, so through the symmetric two-face path `T_s/T_p = cos⁴(R/2)`, and
`DoP = (T_p − T_s)/(T_p + T_s) = (1 − cos⁴(R/2))/(1 + cos⁴(R/2))`. The refractive index and the wedge
apex **cancel via Snell** — they enter only by setting `R`. So **every refraction halo's polarization
is determined by its radius alone**, regardless of habit or which wedge made it. This is exactly Können
& Tinbergen's 22° Fresnel floor (`R = 21.84° → 3.7%`) lifted to the whole Atlas. (`s2_optics.halo_pol_dop`.)

## The Atlas observable table
| halo | R (deg) | kind | predicted DoP |
|---|---|---|---|
| 9° pyramidal | 8.96 | pyramidal | **0.62%** |
| 18° pyramidal | 18.0 | pyramidal | 2.48% |
| 20° pyramidal | 20.0 | pyramidal | 3.06% |
| 22° (prism 60°) | 21.84 | regular | 3.65% |
| 23° pyramidal | 23.0 | pyramidal | 4.00% |
| 24° pyramidal | 23.82 | pyramidal | 4.42% |
| 35° pyramidal | 35.0 | pyramidal | 9.45% |
| 46° (prism+basal 90°) | 45.73 | regular | 16.42% |

The **pyramidal odd-radius (Galle) family** is the headline: a monotone DoP ladder from **0.6% (9°) to
9.5% (35°)** — a barely-studied, fully forward, falsifiable signature of the Atlas's pyramidal stratum.

## Raytracer confirmation (forward Monte-Carlo, `dn=0`)
| halo | raytracer DoP | law DoP | U/I |
|---|---|---|---|
| 9° pyramidal | **0.68%** | 0.61% | −0.03% |
| 22° | **3.75%** | 3.65% | +0.01% |
| 46° | 13.0% | 16.4% | +0.16% |

The cleanly-isolable 9° and 22° halos match the law to ~0.1%, with `U/I ≈ 0` (radial pol). The **broad,
blended 46° ring** runs below the pure min-deviation law (ring-average over a diffuse, supralateral-arc-
contaminated region) — a disclosed boundary: the law is the **min-deviation (inner-edge) prediction**;
ring-averages on broad/overlapping halos read lower.

## Why this matters for the Atlas
- The Atlas's classified halos (fold radii `R`) now each carry a **predicted polarization observable** —
  the bifurcation diagram gains a measurable, falsifiable scalar field `DoP(R)` on top of the geometry.
- It is **habit- and n-independent given R**: two different crystals/wedges producing a halo at the same
  radius produce the **same** polarization — a clean, testable invariant.
- The pyramidal-family ladder (0.6% → 9.5%) is a concrete prediction a polarimeter could check, tying
  the geometric pyramidal stratum to an observable.

## Honest boundaries
- **Refraction halos only.** Reflection/TIR features (parhelic circle, subhelic) carry the per-feature
  ±V handedness instead (`S2_HANDEDNESS_MAP_RESULT.md`); `DoP(R)` is for the refraction folds.
- Min-deviation (inner-edge) prediction; broad/blended rings deviate (the 46° case). Ray-optics tier;
  birefringence slightly lowers the linear DoP (it shifts a little flux into V on TIR-rich paths, ~0 here).
- Forward-model / NOT public-eligible; the generalization is synthesis on Können's mechanism.

## Files
- `scripts/s2_optics.py` — `halo_pol_dop(R)` (the law).
- `scripts/atlas_halo_polarization.py` — the Atlas observable table + raytracer confirmation.
- `scripts/test_atlas_halo_polarization.py` — frozen test (law + raytracer cross-check).
- `scripts/s2_halo_raytracer.py` — the engine that confirms it; `s2_konnen_validate.py` — the 22° anchor.
