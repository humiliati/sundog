"""
test_tangent_matched_filter.py -- Pass C6 regression runner.

Runs the matched-filter detector against the post-C1 tangent-route
eligibility set (p2 / p13 / p27) and prints per-photo verdicts plus
the full R_uta-vs-correlation scan curve.

Usage:
    python scripts/test_tangent_matched_filter.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tangent_matched_filter import (
    detect,
    summarize,
    R_UTA_SCAN_STEP_PX,
    MIN_PEAK_CORR,
    MIN_PROMINENCE_RATIO,
    PRE_REGISTERED_RATIO_MIN,
    PRE_REGISTERED_RATIO_MAX,
    TEMPLATE_WIDTH_SIGMA_PX,
    ARC_MAX_AZIMUTH_DEG,
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
    print("=== Pass C6 -- matched-filter tangent-arc detection ===")
    print(
        f"  Template: Gaussian arc, sigma={TEMPLATE_WIDTH_SIGMA_PX} px, "
        f"azimuth window [0, {ARC_MAX_AZIMUTH_DEG}] deg"
    )
    print(
        f"  Gates: R_uta/R22 in [{PRE_REGISTERED_RATIO_MIN}, {PRE_REGISTERED_RATIO_MAX}]; "
        f"peak corr >= {MIN_PEAK_CORR}; prominence >= {MIN_PROMINENCE_RATIO}x baseline"
    )
    print()

    results = {}
    for name, (photo_file, h_deg) in PHOTOS.items():
        anchor = load_anchor(name)
        sun = tuple(anchor["sun"])
        r22 = anchor["r22"]
        photo_path = os.path.join(CALIB_DIR, photo_file)
        if not os.path.exists(photo_path):
            print(f"--- {name}: MISSING photo at {photo_path}")
            continue
        print(f"--- {name} (h={h_deg} deg) ---")
        result = detect(photo_path, sun, r22)
        results[name] = result
        print(summarize(result))
        # Brief scan curve
        scan_vals = result["scan_r_uta_values"]
        scan_scores = result["scan_scores"]
        # Show every 4th value to keep output compact
        compact = []
        for i in range(0, len(scan_vals), 4):
            compact.append(f"R={scan_vals[i]}:{scan_scores[i]:+.3f}")
        print(f"  scan (every 4th): {', '.join(compact)}")
        print()

    print("=== Per-photo verdict roll-up ===")
    for name, r in results.items():
        print(f"  {name}: {r['verdict']}")
    return results


if __name__ == "__main__":
    main()
