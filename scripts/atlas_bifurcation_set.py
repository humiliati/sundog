#!/usr/bin/env python
"""Atlas Phase 6.5-A — the bifurcation set, COMPONENT B (ray-admissibility walls).

Derives the halo-atlas transition elevations from first principles {refractive index n, crystal-face
geometry, sun elevation} so they fall OUT of the computation — replacing the hardcoded magic-number
thresholds in public/js/parhelion-geometry.mjs (`czaVisible h<=32`, `TANGENT_ARC_CIRCUMSCRIBED_H=29`),
which trip the atlas's own §6 "armchair catastrophe" gate. The numbers 22/32/46/58 are OUTPUTS here,
never inputs.

The two-component wall taxonomy (SUNDOG_V_ATLAS.md §1.1):
  COMPONENT A — caustic catastrophes (caustics coalescing / changing type). The halo EDGES are A₂
    folds (minimum-deviation, ∂δ/∂θ=0); the 29° UTA+LTA→circumscribed-halo MERGE is an A₃ cusp
    (∂δ=∂²δ=0). The 29° cusp needs the 2-D orientation→sky caustic map and is Phase 6.5-B (NOT here).
  COMPONENT B — ray-admissibility walls (a face-pair ray path appears/disappears via TIR / grazing):
    - CZA disappearance (≈32°): horizontal plate, light enters the TOP face, exits a SIDE face
      (90° effective wedge). The arc reaches the zenith / the path goes off-sky when the discriminant
      n²−cos²h > 1, i.e. cos h < √(n²−1), i.e.  h > arccos(√(n²−1)).
    - CHA appearance (≈58°): the SAME 90° plate prism, opposite faces — light enters a SIDE face and
      exits the BOTTOM face. The horizontal↔vertical face swap maps cos h → sin h, so the path is
      admissible when  h > arcsin(√(n²−1)) = 90° − arccos(√(n²−1))  (the exact complement of the CZA
      wall — one derivation, two mutually-validating walls).
    - TIR critical angle  θ_c = arcsin(1/n)  is the underlying admissibility primitive.

SCOPE (atlas §0.2): these are ray-optics edges. The real transition is the SMOOTHED image under the
sun's ~0.5° angular disk, crystal-tilt spread, and chromatic dispersion n(λ) — reported as a BAND, not
a point. Physics is fixed by Snell/TIR + Tape geometry; nothing is tuned. NOT public-eligible (the
lit-pass attribution gate, Phase 0.5, precedes any outward claim).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cza_formula as cza                       # reuse the verified CZA derivation
from s2_optics import halo_min_deviation        # reuse the A₂ fold-radius kernel (+ NaN-past-TIR guard)

N_ICE = cza.ICE_REFRACTIVE_INDEX                 # 1.31, visible centroid
N_VIOLET, N_RED = 1.317, 1.306                   # ice n across the visible (cza_formula module note)
TOL_DEG = 1.0                                    # PRE-REGISTERED PASS/FAIL tolerance (plan)


def tir_critical_angle_deg(n=N_ICE):
    """Total-internal-reflection critical angle θ_c = arcsin(1/n) — the admissibility primitive."""
    return math.degrees(math.asin(1.0 / n))


def cza_disappearance_deg(n=N_ICE):
    """Sun elevation above which the CZA (plate top→side, 90° wedge) goes off-sky: arccos(√(n²−1))."""
    return math.degrees(math.acos(math.sqrt(n ** 2 - 1.0)))


def cha_appearance_deg(n=N_ICE):
    """Sun elevation above which the CHA (plate side→bottom, 90° wedge) becomes admissible:
    arcsin(√(n²−1)) = 90° − (CZA wall). The cos h→sin h horizontal↔vertical face swap."""
    return math.degrees(math.asin(math.sqrt(n ** 2 - 1.0)))


def fold_radius_deg(apex_deg, n=N_ICE):
    """A₂ fold-caustic radius (the halo edge) = minimum deviation through the wedge (s2_optics)."""
    return float(halo_min_deviation(apex_deg, n))


def chromatic_band(fn, *args):
    """(violet, red) endpoints of a derivation across ice's visible dispersion → the smearing band."""
    v = fn(*args, n=N_VIOLET)
    r = fn(*args, n=N_RED)
    return (min(v, r), max(v, r))


# (label, kind, component, derive() -> deg, documented_deg, mechanism)
ROWS = [
    ("22deg halo edge", "A2 fold", "A (primitive)", lambda: fold_radius_deg(60), 22.0,
     "60deg prism min-deviation (dδ/dθ=0)"),
    ("46deg halo edge", "A2 fold", "A (primitive)", lambda: fold_radius_deg(90), 46.0,
     "90deg prism min-deviation (dδ/dθ=0)"),
    ("CZA disappears", "admissibility wall", "B", cza_disappearance_deg, 32.0,
     "plate top->side 90deg; cos h < sqrt(n^2-1) -> off-sky/TIR"),
    ("CHA appears", "admissibility wall", "B", cha_appearance_deg, 58.0,
     "plate side->bottom 90deg; h > arcsin(sqrt(n^2-1)) (complement of CZA)"),
]


def results():
    """Structured rows: derived value, documented value, residual, chromatic band, PASS within tol."""
    out = []
    for label, kind, comp, fn, doc, mech in ROWS:
        val = fn()
        # chromatic band: folds use fold_radius_deg(apex,n); walls use their own fn(n=)
        if label.startswith("22"):
            band = chromatic_band(fold_radius_deg, 60)
        elif label.startswith("46"):
            band = chromatic_band(fold_radius_deg, 90)
        elif "CZA" in label:
            band = chromatic_band(cza_disappearance_deg)
        else:
            band = chromatic_band(cha_appearance_deg)
        out.append({
            "label": label, "kind": kind, "component": comp, "mechanism": mech,
            "derived_deg": round(val, 3), "documented_deg": doc,
            "residual_deg": round(abs(val - doc), 3),
            "chromatic_band_deg": (round(band[0], 2), round(band[1], 2)),
            "pass": abs(val - doc) <= TOL_DEG,
        })
    return out


def main():
    print("Atlas Phase 6.5-A — bifurcation set, component B (admissibility walls) + A2 fold primitives")
    print(f"  n_ice={N_ICE} (visible centroid; band {N_RED}red..{N_VIOLET}violet);  "
          f"TIR critical angle θ_c = arcsin(1/n) = {tir_critical_angle_deg():.2f} deg")
    print(f"  PRE-REGISTERED tolerance: |derived - documented| <= {TOL_DEG} deg\n")
    print(f"  {'feature':<16}{'comp':<14}{'derived':>9}{'doc':>6}{'resid':>7}  {'chromatic band':<16}{'PASS'}")
    print("  " + "-" * 78)
    rows = results()
    for r in rows:
        b = r["chromatic_band_deg"]
        print(f"  {r['label']:<16}{r['component']:<14}{r['derived_deg']:>9.3f}{r['documented_deg']:>6.0f}"
              f"{r['residual_deg']:>7.3f}  {f'{b[0]:.2f}-{b[1]:.2f}':<16}{'PASS' if r['pass'] else 'FAIL'}")
    allpass = all(r["pass"] for r in rows)
    print("\n  mechanisms:")
    for r in rows:
        print(f"    {r['label']:<16} [{r['kind']}] {r['mechanism']}")
    print(f"\n  29deg UTA+LTA->circumscribed merge = A3 CUSP (component A) -> Phase 6.5-B "
          "(2-D orientation->sky caustic map; NOT derived here).")
    print(f"  SCOPE (§0.2): ray-optics edges; observed transitions are the smoothed image (sun 0.5deg "
          "disk + tilt + chromatic band).")
    print(f"\n  ALL DERIVED VALUES WITHIN ±{TOL_DEG} deg OF DOCUMENTED: {allpass}")
    return allpass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
