"""
test_tangent_curvature.py — Pass C4 regression runner.

Runs the wing-slope geometric curvature detector against the post-C1
tangent-route eligibility set (p2 / p13 / p27) and prints per-photo
verdicts. Pre-registered gates live in tangent_curvature; this script
exercises them.

Usage:
    python scripts/test_tangent_curvature.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tangent_curvature import (
    GRADIENT_MIN_AMPLITUDE,
    RIDGE_OFFSET_TOLERANCE_PX,
    MIN_EDGE_POINTS,
    FIT_RMS_TOLERANCE_PX,
    RATIO_MIN,
    RATIO_MAX,
    detect,
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


def main():
    print("=== Pass C4 -- wing-slope geometric curvature detector ===")
    print(
        f"  Gates: gradient >= {GRADIENT_MIN_AMPLITUDE} L*; "
        f"offset <= +/-{RIDGE_OFFSET_TOLERANCE_PX} px; "
        f">= {MIN_EDGE_POINTS} candidates; RMS <= {FIT_RMS_TOLERANCE_PX} px; "
        f"R_uta/R22 in [{RATIO_MIN}, {RATIO_MAX}]"
    )
    print()

    results = {}
    for name, (photo_file, h_deg) in PHOTOS.items():
        anchor = load_anchor(name)
        sun = tuple(anchor["sun"])
        r22 = anchor["r22"]
        photo_path = os.path.join(CALIB_DIR, photo_file)
        if not os.path.exists(photo_path):
            print(f"--- {name}: MISSING photo at {photo_path} ---")
            continue
        print(f"--- {name} (h={h_deg} deg) ---")
        result = detect(photo_path, sun, r22)
        results[name] = result
        print(summarize(result))
        print()

    print("=== Per-photo verdict roll-up ===")
    for name, r in results.items():
        print(f"  {name}: {r['verdict']}")
    return results


if __name__ == "__main__":
    main()
