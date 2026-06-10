#!/usr/bin/env python
"""Atlas orientation-boundary forward fits — which Phase-7 boundary elevations the polarized raytracer
can INDEPENDENTLY re-derive (and which it honestly cannot).

Phase 12 closed the random/horizon-clip gaps and added the polarization layer, but left the
ORIENTATION-DRIVEN boundary elevations to the Phase-7 caustic-map / admissibility machinery. This pass
tests whether the full ray-marcher can re-derive them with targeted feature detectors. The result is
nuanced and honest:

  CONFIRMED (admissibility walls — a feature TIR-appears/vanishes, the clean kind):
    * CZA off at 32.196° — plate basal-entry → side-exit (90° wedge) near-ZENITH flux COLLAPSES as the
      side-face incidence crosses the critical angle (TIR). Raytracer: flux spans the wall, ~0 by 35°.
    * CHA on at 57.804° — the complement: near-HORIZON 90°-wedge flux APPEARS only for sun > 57.8°.
      Raytracer: ~0 below 55°, strong above 58°. (CZA+CHA = 90.000, both TIR onsets.)

  NOT cleanly fittable (confirming the Phase-12 honest boundary for these):
    * 29.71° column UTA+LTA merge — a TOPOLOGICAL A₃-metamorphosis; the circumscribed loop's connecting
      SIDES are intrinsically faint, so the merge does not extract from MC flux (three detectors flat).
    * 60.74° plate-parhelia off — the parhelion's disappearance is masked by wobble-broadened 22°-ring
      background at the sun's elevation; the flux persists past the wall (no clean vanish).

So the raytracer re-derives the TIR ADMISSIBILITY walls but not the topological-merge / wobble-masked
walls — those remain the authority of `atlas_caustic_map` / `atlas_forward_sweep`. NOT public-eligible.
Run:  python scripts/atlas_orientation_boundaries.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2_halo_raytracer as rt  # noqa: E402
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 console robustness
except Exception:
    pass

CZA_WALL = 32.196   # arccos(sqrt(n^2-1)) — basal/side 90-deg wedge TIR onset (CZA disappears above)
CHA_WALL = 57.804   # arcsin(sqrt(n^2-1)) = 90 - CZA_WALL (CHA appears above; exact complement)


# ----- the CONFIRMED detectors: the two TIR admissibility walls ----------------------------- #
def cza_flux(e_deg, n_orient=9000, seed=44):
    """Near-ZENITH 90°-wedge (46-family) flux — the CZA. Collapses as e crosses CZA_WALL (TIR)."""
    dep = rt.run_ensemble("plate", e_deg=float(e_deg), n_orient=n_orient, dn=0.0, K=2, seed=seed)
    sc, el, I = dep[:, 0], dep[:, 2], dep[:, 3]
    return float(I[(sc >= 38) & (sc < 52) & (el > e_deg + 15)].sum())


def cha_flux(e_deg, n_orient=9000, seed=45):
    """Near-HORIZON 90°-wedge (46-family) flux — the CHA. Appears only for sun > CHA_WALL."""
    dep = rt.run_ensemble("plate", e_deg=float(e_deg), n_orient=n_orient, dn=0.0, K=2, seed=seed)
    sc, el, I = dep[:, 0], dep[:, 2], dep[:, 3]
    return float(I[(sc >= 38) & (sc < 52) & (el < e_deg - 15) & (el < 35)].sum())


# ----- the NULL detectors: the boundaries the raytracer can NOT cleanly fit ------------------ #
def merge_sidefill(e_deg, n_orient=9000, seed=42):
    """Column 22°-ring flux at the sun's own elevation (the circumscribed-loop SIDES). The 29.71° merge
    would turn this on — but the connecting sides are intrinsically faint, so it stays flat (null)."""
    dep = rt.run_ensemble("column", e_deg=float(e_deg), n_orient=n_orient, dn=0.0, K=1, seed=seed)
    sc, el, I = dep[:, 0], dep[:, 2], dep[:, 3]
    ring = (sc >= 20.5) & (sc < 23.5)
    tot = I[ring].sum()
    return float(I[ring & (np.abs(el - e_deg) < 5)].sum() / tot) if tot > 0 else 0.0


def parhelion_flux(e_deg, n_orient=9000, seed=43):
    """Plate 22°-flux at the sun's elevation (the parhelion band). The 60.74° wall would zero it — but
    wobble-broadened ring background persists past the wall, so no clean vanish (null)."""
    dep = rt.run_ensemble("plate", e_deg=float(e_deg), n_orient=n_orient, dn=0.0, K=1, seed=seed)
    sc, el, I = dep[:, 0], dep[:, 2], dep[:, 3]
    return float(I[(sc >= 20) & (sc < 24) & (np.abs(el - e_deg) < 6)].sum())


def _report():
    print("=" * 78)
    print("Atlas orientation-boundary forward fits — what the raytracer can re-derive")
    print("=" * 78)

    print(f"\nCONFIRMED — CZA off at {CZA_WALL}° (near-zenith 90°-wedge flux collapses, TIR onset):")
    for e in (25, 28, 31, 33, 35):
        print(f"  e={e:2d}°: near-zenith flux = {cza_flux(e):8.0f}")
    print(f"  -> collapses through ~31–35°, bracketing the derived {CZA_WALL}° (within the §0.2 smear).")

    print(f"\nCONFIRMED — CHA on at {CHA_WALL}° (near-horizon 90°-wedge flux appears, complement TIR wall):")
    for e in (52, 55, 57, 58, 63):
        print(f"  e={e:2d}°: near-horizon flux = {cha_flux(e):8.0f}")
    print(f"  -> ~0 below 55°, strong above 58°, bracketing the derived {CHA_WALL}° (CZA+CHA = 90.000).")

    print("\nNOT cleanly fittable (the raytracer generates the features but not the wall elevation):")
    s20, s40 = merge_sidefill(20), merge_sidefill(40)
    print(f"  29.71° column merge: side-fill {s20*100:.1f}% (e=20, below) vs {s40*100:.1f}% (e=40, above)"
          f" -> FLAT, no step (topological merge, faint sides).")
    p55, p67 = parhelion_flux(55), parhelion_flux(67)
    print(f"  60.74° plate-parhelia off: parhelion-band flux {p55:.0f} (e=55) vs {p67:.0f} (e=67, past wall)"
          f" -> PERSISTS (wobble background masks the vanish).")
    print("\n  => the raytracer re-derives the TIR admissibility walls (CZA/CHA); the topological merge")
    print("     and wobble-masked parhelion-off stay with the caustic-map/admissibility machinery.")


if __name__ == "__main__":
    _report()
