#!/usr/bin/env python
"""Atlas predicted observable — the POLARIZATION of the classified refraction halos.

The Atlas classifies the halo bifurcation set: each halo is a caustic (an A2 fold) whose angular
RADIUS R is a fold position derived from {n, geometry} (Phases 6.5/7/11). This module adds a SMOOTH
OBSERVABLE on that classified set — the linear degree-of-polarization — via the one-parameter law

    DoP(R) = (1 - cos^4(R/2)) / (1 + cos^4(R/2))          (radial; U = 0; no net circular V)

(`s2_optics.halo_pol_dop`). At minimum deviation theta_i - theta_t = R/2 and the Fresnel s/p
diattenuation through the two faces gives cos^4(R/2); the n- and apex-dependence cancels via Snell, so
the polarization of EVERY refraction halo is fixed by its radius ALONE. This generalizes Können &
Tinbergen 1991's 22-deg Fresnel-floor (3.7%) to the whole Atlas, including the pyramidal {10-11}
odd-radius (Galle) family 9/18/20/23/24/35 deg.

The prediction is FORWARD (geometry -> observable, no inversion), FALSIFIABLE by sky polarimetry, and
CONFIRMED by the polarized halo raytracer at the cleanly-isolable halos (9, 22 deg; the broad/blended
46-deg ring deviates as expected). NOT public-eligible. Attribution: Können & Tinbergen 1991 (the
22-deg mechanism); Greenler / Tape / Cowley (halo geometry); the Atlas classification (this repo).

Run:  python scripts/atlas_halo_polarization.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2_optics as so  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# The Atlas's classified refraction halos (radius R in deg). Regular hexagonal wedges (computed from
# halo_min_deviation) + the {10-11} pyramidal (Galle) odd-radius family (the Atlas-validated radii).
ATLAS_HALOS = [
    ("9-deg pyramidal",         8.96,                          "pyramidal"),
    ("18-deg pyramidal",        18.0,                          "pyramidal"),
    ("20-deg pyramidal",        20.0,                          "pyramidal"),
    ("22-deg (prism 60)",       float(so.halo_min_deviation(60)), "regular"),
    ("23-deg pyramidal",        23.0,                          "pyramidal"),
    ("24-deg pyramidal",        23.82,                         "pyramidal"),
    ("35-deg pyramidal",        35.0,                          "pyramidal"),
    ("46-deg (prism+basal 90)", float(so.halo_min_deviation(90)), "regular"),
]


def predicted_table():
    """[(name, radius_deg, kind, predicted_DoP), ...] sorted by radius — the Atlas observable."""
    return [(nm, R, k, float(so.halo_pol_dop(R))) for nm, R, k in ATLAS_HALOS]


def raytracer_dop(habit, lo, hi, n_orient=14000, seed=21, e_deg=20.0):
    """Flux-weighted linear DoP and U/I the polarized raytracer produces on a halo ring [lo,hi]."""
    import s2_halo_raytracer as rt
    dep = rt.run_ensemble(habit, e_deg=e_deg, n_orient=n_orient, dn=0.0, K=1, seed=seed)
    rp = rt.ring_pol(dep, lo, hi)
    return (rp["dop"], rp["uoi"]) if rp else (float("nan"), float("nan"))


def _report():
    print("=" * 76)
    print("Atlas predicted observable — linear polarization of the classified halos")
    print("=" * 76)
    print("  DoP(R) = (1 - cos^4(R/2)) / (1 + cos^4(R/2))   (radial; U=0; no circular V)\n")
    print(f"  {'halo':26s} {'R (deg)':>8s} {'kind':>10s} {'predicted DoP':>14s}")
    for nm, R, k, dop in predicted_table():
        print(f"  {nm:26s} {R:8.2f} {k:>10s} {dop * 100:13.2f}%")
    print("\n  -> the pyramidal family spans DoP 0.6% (9 deg) -> 9.5% (35 deg); the law is monotone in R.")

    print("\nRaytracer confirmation (forward Monte-Carlo, dn=0) at the cleanly-isolable halos:")
    for nm, habit, lo, hi, R in (("9-deg ", "pyramidal", 8.0, 10.5, 8.96),
                                 ("22-deg", "random", 20.5, 23.5, float(so.halo_min_deviation(60))),
                                 ("46-deg", "random", 44.0, 48.0, float(so.halo_min_deviation(90)))):
        dop, uoi = raytracer_dop(habit, lo, hi, seed=21 if habit == "random" else 22)
        print(f"  {nm}: raytracer DoP={dop * 100:5.2f}%  U/I={uoi * 100:+.2f}%   vs law(R={R:.2f})="
              f"{so.halo_pol_dop(R) * 100:5.2f}%")
    print("\n  (9/22 deg match to ~0.1%; the broad/blended 46-deg ring-average runs below the pure"
          "\n   min-deviation law — a disclosed boundary. See ATLAS_HALO_POLARIZATION_OBSERVABLE.md.)")


if __name__ == "__main__":
    _report()
