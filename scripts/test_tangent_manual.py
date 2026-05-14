"""
test_tangent_manual.py -- Pass C5 of the Phase 10 attack campaign.

Manual sample selection from visual crops -- the literature-standard
alternative to gradient-based edge detection that Persona 1 section 5
of PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md explicitly named:

    "Real measurement needs either gradient-based edge detection (the
    spine is at a brightness or chromaticity transition, not a peak) or
    manual sample selection from visual crops."

Pass C4 tested gradient-based edge detection (not-recovered). Pass C5
tests the manual-sample-selection branch: load hand-anchored
upper-tangent points from each anchor JSON, run the linearized
circle-fit from tangent_curvature.fit_circle, and report whether the
fitted R_uta matches the predicted R_uta (= R22 by the locally-circular
approximation).

Pre-registered gates (chosen to fit hand-anchored data, not detector
output):

    >= 5 surviving manual points per photo
    circle-fit RMS <= 10 px
    R_uta_obs / R22 in [0.7, 1.3]

A photo is "manual-recovered" if all gates pass.

Usage:
    python scripts/test_tangent_manual.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tangent_curvature import fit_circle


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_DIR = os.path.join(REPO_ROOT, "docs", "calibration")

PHOTOS = ("p2", "p13", "p27")

MIN_MANUAL_POINTS = 5
FIT_RMS_TOLERANCE_PX = 10.0
RATIO_MIN = 0.7
RATIO_MAX = 1.3


def load_anchor(name):
    path = os.path.join(CALIB_DIR, f"{name}-anchor.json")
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def evaluate_photo(name):
    anchor = load_anchor(name)
    sun_x, sun_y = anchor["sun"]
    r22 = anchor["r22"]

    block = anchor.get("upper_tangent_manual_samples")
    if not block:
        return {
            "photo": name,
            "verdict": "no_manual_samples_block",
            "n_points": 0,
        }

    points = block.get("points", [])
    n = len(points)
    pt_xy = [(p["x"], p["y"]) for p in points]

    fit = None
    if n >= 3:
        fit = fit_circle(pt_xy)

    result = {
        "photo": name,
        "sun": (sun_x, sun_y),
        "r22": r22,
        "n_points": n,
        "anchor_status": block.get("status"),
        "method_note": block.get("method", "").split(".")[0] + "." if block.get("method") else None,
    }

    if fit is None:
        result["verdict"] = "not_recovered_too_few_or_singular"
        result["fit"] = None
    else:
        cx, cy, R_obs, rms = fit
        ratio = R_obs / r22
        result["fit"] = {
            "center_x": cx,
            "center_y": cy,
            "R_uta_obs": R_obs,
            "R_uta_predicted": r22,
            "ratio_obs_over_predicted": ratio,
            "fit_rms_px": rms,
        }
        if (
            n >= MIN_MANUAL_POINTS
            and rms <= FIT_RMS_TOLERANCE_PX
            and RATIO_MIN <= ratio <= RATIO_MAX
        ):
            result["verdict"] = "manual_recovered"
        else:
            result["verdict"] = "not_recovered"
    return result


def summarize(result):
    lines = []
    name = result["photo"]
    lines.append(f"--- {name} ---")
    if result["n_points"] == 0:
        lines.append(f"  status: {result.get('anchor_status', 'no manual samples')}")
        lines.append(f"  verdict: {result['verdict']}")
        return "\n".join(lines)

    lines.append(f"  sun={result['sun']}  R22={result['r22']}  n_points={result['n_points']}")
    if result.get("anchor_status"):
        lines.append(f"  anchor status: {result['anchor_status']}")
    if result.get("fit") is None:
        lines.append("  circle fit: insufficient points or singular")
    else:
        f = result["fit"]
        lines.append(
            f"  fit center: ({f['center_x']:.1f}, {f['center_y']:.1f})  "
            f"R_uta_obs={f['R_uta_obs']:.1f}  R_uta_pred={f['R_uta_predicted']}  "
            f"ratio={f['ratio_obs_over_predicted']:.3f}  RMS={f['fit_rms_px']:.2f}px"
        )
    lines.append(f"  VERDICT: {result['verdict']}")
    return "\n".join(lines)


def main():
    print("=== Pass C5 -- manual sample selection from visual crops ===")
    print(
        f"  Gates: >= {MIN_MANUAL_POINTS} manual points; RMS <= {FIT_RMS_TOLERANCE_PX} px; "
        f"R_uta_obs / R22 in [{RATIO_MIN}, {RATIO_MAX}]"
    )
    print()

    results = {}
    for name in PHOTOS:
        result = evaluate_photo(name)
        results[name] = result
        print(summarize(result))
        print()

    print("=== Per-photo verdict roll-up ===")
    for name, r in results.items():
        print(f"  {name}: {r['verdict']}")

    eligible = [r for r in results.values() if r["verdict"] == "manual_recovered"]
    print()
    print(f"Recovered photos: {len(eligible)} / {len(PHOTOS)}")
    if len(eligible) >= 2:
        print("Coverage gate: PASS (>= 2 eligible photos)")
    else:
        print("Coverage gate: FAIL (< 2 eligible photos for route promotion)")
    return results


if __name__ == "__main__":
    main()
