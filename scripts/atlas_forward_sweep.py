#!/usr/bin/env python
"""Atlas Phase 7 — the forward sweep: COMPUTE the master bifurcation set of the halo phase diagram.

Assembles the derived phase-boundary elevations of the (sun-elevation x crystal-habit) phase diagram by
RUNNING the repo machinery (the §6 armchair-catastrophe gate: derive, never assert). Each boundary is
classified component-A (caustic catastrophe) or component-B (ray-admissibility wall); the random rings are
a third kind (horizon-clip occlusion of an always-present A2 fold, NOT a bifurcation). §0.2: every boundary
is a ~1-2° smeared band. NOT public-eligible. Companion doc: docs/atlas/ATLAS_PHASE7_PHASE_DIAGRAM.md.

Reproduces the Phase-7 workflow synthesis (run wsktxxxg9), including its TWO refuted/removed transitions
(the Wegener-reaches-anthelion and the pyramidal-lower-branch-at-20° fabrications), kept here as explicit
negative receipts so the collapsed cells are auditable.
"""
import math
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_caustic_map as cm
import atlas_strata_map as sm
from s2_optics import halo_min_deviation
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 console robustness
except Exception:
    pass

N = cm.N_ICE
SQ = math.sqrt(N * N - 1.0)


# ---- component-B admissibility walls from closed form + the forward models ----------------------- #
def cza_wall():
    return math.degrees(math.acos(SQ))                     # CZA disappears / 90°-wedge off-sky


def cha_wall():
    return math.degrees(math.asin(SQ))                     # CHA appears = 90 - CZA wall (complement)


def _bisect_admissible(present, lo, hi, tol=0.05):
    """Smallest elevation in [lo,hi] at which `present(h)` flips from True to False (a disappear wall)."""
    if not present(lo) or present(hi):
        return None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if present(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def wedge_present(wedge, frac=2e-3, ngrid=240):
    return lambda h: float(cm.sky_grid(h, ngrid=ngrid, wedge=wedge)[3].mean()) > frac


# ---- a minimal PLATE forward model (the one feature with no existing repo model) ----------------- #
def plate_parhelion_present(h_deg, nbeta=1440):
    """Plate (c vertical): vertical prism side faces (normals horizontal at azimuth β); the 60° parhelion
    wedge = two faces 120° apart. Present at elevation h if ANY roll β gives an admissible entry+exit
    refraction (no TIR) with the apparent source on-sky. Returns True/False."""
    su = cm.sun_dir(h_deg)
    d0 = -su
    b = np.linspace(0.0, 2 * math.pi, nbeta, endpoint=False)
    n1 = np.stack([np.cos(b), np.sin(b), np.zeros_like(b)], axis=-1)
    n2 = np.stack([np.cos(b + cm.FACE_SEP), np.sin(b + cm.FACE_SEP), np.zeros_like(b)], axis=-1)
    M = b.shape[0]
    d0b = np.broadcast_to(d0, (M, 3))
    entry = (n1 @ d0) < 0
    d1, v1 = cm._refract_vec(d0b, n1, 1.0 / N)
    exitok = np.sum(d1 * n2, axis=-1) > 0
    d2, v2 = cm._refract_vec(d1, n2, N)
    src = -d2                                               # apparent source
    onsky = src[:, 2] > -math.sin(math.radians(0.5))       # above (within the sun-disk of) the horizon
    return bool(np.any(entry & v1 & exitok & v2 & onsky))


def plate_parhelia_limit():
    return _bisect_admissible(plate_parhelion_present, 30.0, 75.0)


# ---- Lowitz A3-lips + reabsorption from the cusp field ------------------------------------------- #
def _lowitz_interior(h, ng=320, bd=5.0):
    cm.LOWITZ_ALPHA0 = math.radians(60.0)
    cusp, detJ, g, good, X, Y, su = sm.cusp_field(h, wedge="lowitz60", ngrid=ng)
    dist = ndimage.distance_transform_edt(good)
    lab, n = ndimage.label(cusp, structure=np.ones((3, 3)))
    return sum(1 for i in range(1, n + 1) if np.mean(dist[tuple(np.where(lab == i))]) >= bd)


def lowitz_lips_birth():
    return _bisect_admissible(lambda h: _lowitz_interior(h) < 4, 12.0, 22.0, tol=0.1)   # 2 -> 4


# ---- the two FABRICATED transitions the verify pass refuted (kept as negative receipts) ---------- #
def wegener_reaches_anthelion_FALSE():
    """REFUTED: the Wegener apparent source never approaches the antisolar point. Returns the min
    ray-to-antisolar angle over a wide elevation sweep (flat ~44° — never → 0; the '22.13°' was fabricated)."""
    best = 180.0
    for h in range(5, 60, 3):
        G, A, sky, ok, su = cm.sky_grid(float(h), ngrid=200, wedge="wegener")
        anti = -su
        s = sky[ok]
        if len(s):
            best = min(best, float(np.degrees(np.arccos(np.clip(s @ anti, -1, 1))).min()))
    return best


def pyramidal_lower_branch_flat():
    """REFUTED: the 23.8° oriented-pyramidal arc's min-deviation is flat at all elevations (no h≈20°
    'lower-branch appear' event). Returns (min, max) of the caustic min-deviation over h."""
    cm.PYR_DPHI = math.radians(120.0)
    ds = []
    for h in range(6, 31, 4):
        G, A, sky, ok, su = cm.sky_grid(float(h), ngrid=240, wedge="pyrcol")
        s = sky[ok]
        if len(s):
            ds.append(float(np.degrees(np.arccos(np.clip(s @ su, -1, 1))).min()))
    return (min(ds), max(ds))


def main():
    print("Atlas Phase 7 — the master bifurcation set of the halo phase diagram (all DERIVED)\n")
    print("  A2 fold-ring radii (elevation-independent; horizon-clip occlusion only, NOT bifurcations):")
    for apex, name in ((60, "22° halo"), (90, "46° halo"), (28.0, "9°"), (52.4, "18°"),
                       (56.0, "20°"), (63.8, "24°"), (80.2, "35°")):
        print(f"    {name:>9}: halo_min_deviation({apex}) = {halo_min_deviation(apex, N):.3f}°")

    print("\n  Phase-boundary elevations (component A = catastrophe, B = admissibility wall):")
    rows = [
        ("column UTA+LTA → circumscribed merge", "A", cm.merge_elevation()),
        ("plate CZA off / column+rosette supralateral off (90°-wedge)", "B", cza_wall()),
        ("plate CHA appears (= 90 − CZA, complement)", "B", cha_wall()),
        ("supralateral off-sky (basal90 forward model)", "B", _bisect_admissible(wedge_present("basal90"), 25.0, 40.0)),
        ("plate parhelia disappear (plate forward model)", "B", plate_parhelia_limit()),
        ("Lowitz A3-lips cusp-pair birth (cusp_field)", "A", lowitz_lips_birth()),
        ("Lowitz family off-sky (lowitz60 forward model)", "B", _bisect_admissible(wedge_present("lowitz60"), 55.0, 70.0)),
        ("Wegener arc off-sky (wegener forward model; SOFT)", "B", _bisect_admissible(wedge_present("wegener"), 45.0, 60.0)),
    ]
    for name, comp, val in rows:
        v = f"{val:6.2f}°" if val is not None else "  —   "
        print(f"    [{comp}] {v}  {name}")

    print("\n  NEGATIVE RECEIPTS — the two transitions the Phase-7 verify pass REFUTED (kept auditable):")
    wmin = wegener_reaches_anthelion_FALSE()
    print(f"    Wegener 'reaches anthelion at 22.13°': REFUTED — min ray-to-antisolar angle = {wmin:.1f}° "
          f"(flat, never → 0; fabricated). Anthelic cells collapse to one.")
    plo, phi = pyramidal_lower_branch_flat()
    print(f"    Pyramidal 'lower-branch appears at 20°': REFUTED — caustic min-dev flat {plo:.2f}–{phi:.2f}° "
          f"over h (no discrete event; fabricated). Composite 0–20/20–29.7 boundary collapses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
