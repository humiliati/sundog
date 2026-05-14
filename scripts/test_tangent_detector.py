"""
test_tangent_detector.py — Pass C2 regression test for tangent_detector.

Runs the wing-azimuth-offset Lab b* ridge detector against the post-C1
tangent-route eligibility set (p2, p13, p27) and prints per-photo
verdicts. Spec sanity check (geometry-only, no image load) is in
test_spec_geometry().

Usage:
    python scripts/test_tangent_detector.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tangent_detector import (
    WING_AZIMUTH_INNER_DEG,
    WING_AZIMUTH_OUTER_DEG,
    WING_SAMPLE_COUNT,
    detect,
    predicted_wing_point,
    summarize,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_DIR = os.path.join(REPO_ROOT, "docs", "calibration")

PHOTOS = {
    "p2":  ("2.Photometeor-jeff_mod_red.jpg",                  18.6),
    "p13": ("13.480859565_17934474635991868_323320248088719839_n.jpg",  6.83),
    "p27": ("27.Zzhau2Uiospj37zhwHEr4D-1200-80.jpg.webp",        0.5),
}


def load_anchor(name):
    path = os.path.join(CALIB_DIR, f"{name}-anchor.json")
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def test_spec_geometry():
    """Sanity check: predicted wing points should sit above the sun and
    flank the apex symmetrically. No image load required."""
    print("=== Spec geometry check ===")
    for name in ("p2", "p13", "p27"):
        anchor = load_anchor(name)
        sun_x, sun_y = anchor["sun"]
        r22 = anchor["r22"]
        apex_y = sun_y - r22

        l_inner = predicted_wing_point(sun_x, sun_y, r22, WING_AZIMUTH_INNER_DEG, "left")
        r_inner = predicted_wing_point(sun_x, sun_y, r22, WING_AZIMUTH_INNER_DEG, "right")
        l_outer = predicted_wing_point(sun_x, sun_y, r22, WING_AZIMUTH_OUTER_DEG, "left")
        r_outer = predicted_wing_point(sun_x, sun_y, r22, WING_AZIMUTH_OUTER_DEG, "right")

        assert l_inner[0] < sun_x < r_inner[0], "inner wings not flanking sun"
        assert l_outer[0] < l_inner[0] < sun_x, "left wing not monotone outward"
        assert sun_x < r_inner[0] < r_outer[0], "right wing not monotone outward"
        assert all(p[1] < sun_y for p in (l_inner, r_inner, l_outer, r_outer)), \
            "wing samples not above sun"

        print(
            f"  {name}: sun=({sun_x},{sun_y}) R22={r22} apex_y={apex_y}; "
            f"left wing x range [{l_outer[0]:.0f},{l_inner[0]:.0f}]; "
            f"right wing x range [{r_inner[0]:.0f},{r_outer[0]:.0f}]"
        )
    print(f"  Wing azimuth window: [{WING_AZIMUTH_INNER_DEG}°, {WING_AZIMUTH_OUTER_DEG}°], "
          f"{WING_SAMPLE_COUNT} samples per wing.")
    print("  PASS")
    print()


def test_detect_on_eligible_photos():
    print("=== Detector run on post-C1 eligibility set ===")
    results = {}
    for name, (photo_file, h_deg) in PHOTOS.items():
        anchor = load_anchor(name)
        sun = tuple(anchor["sun"])
        r22 = anchor["r22"]
        photo_path = os.path.join(CALIB_DIR, photo_file)
        if not os.path.exists(photo_path):
            print(f"  {name}: MISSING photo at {photo_path}")
            continue
        print(f"\n--- {name} (h={h_deg}°) ---")
        result = detect(photo_path, sun, r22)
        results[name] = result
        print(summarize(result))
    print()
    print("=== Per-photo verdict roll-up ===")
    for name, r in results.items():
        print(f"  {name}: {r['verdict']}")
    return results


def main():
    test_spec_geometry()
    test_detect_on_eligible_photos()


if __name__ == "__main__":
    main()
