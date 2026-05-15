"""Pass C7 Step 2: compare HaloSim-predicted upper-tangent-arc opening angle
against p2's existing C5 hand-anchor points, under the canonical literature
inverse handle (opening angle / arc extent) rather than the project-original
circle-fit curvature inverse handle.

Wave-2 W4 origin: Persona A flagged that the literature standard tangent-arc
inverse handle is opening angle / arc extent (Tape Ch 6 / Cowley tangent-arcs
page), not circle-fit curvature. Pass C5 fit a circle to 5 hand-anchor points;
Pass C7 reuses those points under the opening-angle metric and compares to
HaloSim's literature-canonical simulation at the same sun elevation.

Step 1 (this script): geometric reduction of the existing C5 hand-anchor
points to an "opening-angle" measurement. Reads p<NN>-anchor.json's
upper_tangent_manual_samples.points block, computes azimuth-from-zenith and
radial-distance-from-sun for each anchor relative to (sun, R22) from the same
anchor file, and reports the wing extent.

Step 2 (separate): cross-check vs HaloSim image at the same sun elevation;
see docs/calibration/HALOSIM_C7_RUN_INSTRUCTIONS.md and the matching outputs
under docs/calibration/halosim_outputs/.

Run from repo root: python scripts/tangent_opening_angle.py
"""

from __future__ import annotations
import json
import math
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
ANCHOR_DIR = os.path.join(REPO_ROOT, "docs", "calibration")

# Photos with sun_altitude where Pass C5 attempted manual anchoring.
ELIGIBLE = [
    ("p2", 18.6),
    ("p13", 6.83),
    ("p27", 0.5),
]


def load_anchor(name: str) -> dict:
    with open(os.path.join(ANCHOR_DIR, f"{name}-anchor.json"), encoding="utf-8") as f:
        return json.load(f)


def opening_angle_reduction(name: str, h_deg: float) -> dict | None:
    """Reduce a photo's upper_tangent_manual_samples to opening-angle metrics.

    Returns dict with per-point and aggregate angular positions, or None if
    the anchor file has no upper_tangent_manual_samples block.
    """
    anchor = load_anchor(name)
    block = anchor.get("upper_tangent_manual_samples")
    if not block or not block.get("points"):
        return None
    sun_x, sun_y = anchor["sun"]
    r22 = anchor["r22"]
    px_per_deg = r22 / 22.0  # 22-degree halo as the angular ruler

    rows = []
    for pt in block["points"]:
        dx = pt["x"] - sun_x
        dy = pt["y"] - sun_y  # image-y increases downward; up = negative dy
        radial_px = math.hypot(dx, dy)
        radial_deg = radial_px / px_per_deg
        # Azimuth from zenith axis (straight up from sun), in image-plane terms.
        # +azimuth = right, -azimuth = left.
        # dy is negative when above the sun.
        # arctan2(dx, -dy) gives azimuth measured clockwise from up.
        azimuth_deg = math.degrees(math.atan2(dx, -dy))
        rows.append(
            {
                "label": pt["label"],
                "image_xy": (pt["x"], pt["y"]),
                "offset_from_sun_px": (dx, dy),
                "radial_deg_from_sun": radial_deg,
                "azimuth_deg_from_zenith": azimuth_deg,
            }
        )

    # Aggregate metrics.
    left = [r for r in rows if r["azimuth_deg_from_zenith"] < -1.0]
    right = [r for r in rows if r["azimuth_deg_from_zenith"] > 1.0]
    apex = [r for r in rows if abs(r["azimuth_deg_from_zenith"]) <= 1.0]

    summary = {
        "photo": name,
        "h_deg": h_deg,
        "r22_px": r22,
        "px_per_deg": px_per_deg,
        "n_points": len(rows),
        "apex_radial_deg": apex[0]["radial_deg_from_sun"] if apex else None,
        "left_outer_az_deg": (
            min(r["azimuth_deg_from_zenith"] for r in left) if left else None
        ),
        "right_outer_az_deg": (
            max(r["azimuth_deg_from_zenith"] for r in right) if right else None
        ),
        "left_outer_radial_deg": (
            max(r["radial_deg_from_sun"] for r in left) if left else None
        ),
        "right_outer_radial_deg": (
            max(r["radial_deg_from_sun"] for r in right) if right else None
        ),
        "full_opening_az_deg": None,  # filled below if both wings present
        "rows": rows,
    }
    if summary["left_outer_az_deg"] is not None and summary["right_outer_az_deg"] is not None:
        summary["full_opening_az_deg"] = (
            summary["right_outer_az_deg"] - summary["left_outer_az_deg"]
        )
    return summary


def print_summary(s: dict) -> None:
    print(f"\n=== {s['photo']} (h = {s['h_deg']:.2f}°) ===")
    print(f"  r22 = {s['r22_px']} px  ->  {s['px_per_deg']:.3f} px/deg")
    print(f"  hand-anchor points: {s['n_points']}")
    if s["apex_radial_deg"] is not None:
        # Sanity check: apex should be at ~22.0° (sitting on the 22° halo).
        delta = s["apex_radial_deg"] - 22.0
        ok = "  (within 0.5° of halo crown ✓)" if abs(delta) < 0.5 else "  (off halo crown ✗)"
        print(f"  apex radial from sun: {s['apex_radial_deg']:.2f}°  Δ={delta:+.2f}°{ok}")
    if s["full_opening_az_deg"] is not None:
        half_l = s["left_outer_az_deg"]
        half_r = s["right_outer_az_deg"]
        print(f"  outer-wing azimuth from zenith: left {half_l:+.2f}°, right {half_r:+.2f}°")
        print(f"  full opening angle (right - left): {s['full_opening_az_deg']:.2f}°")
        print(f"  outer-wing radial from sun: left {s['left_outer_radial_deg']:.2f}°, right {s['right_outer_radial_deg']:.2f}°")

    print("  per-point detail:")
    for r in s["rows"]:
        print(
            f"    {r['label']:<22}  image=({r['image_xy'][0]:>4},{r['image_xy'][1]:>4})  "
            f"radial={r['radial_deg_from_sun']:>5.2f}°  az_from_zenith={r['azimuth_deg_from_zenith']:>+6.2f}°"
        )


def main() -> int:
    print("Pass C7 Step 2 — Upper tangent arc opening-angle reduction")
    print("=" * 70)
    print("Reduces existing C5 hand-anchor points to opening-angle metrics.")
    print("Uses the per-photo (sun, R22) anchor as the angular reference.")
    print()
    for name, h in ELIGIBLE:
        try:
            s = opening_angle_reduction(name, h)
        except FileNotFoundError:
            print(f"\n=== {name} (h = {h:.2f}°) ===  [anchor file not found]")
            continue
        if s is None:
            print(f"\n=== {name} (h = {h:.2f}°) ===  [no upper_tangent_manual_samples block]")
            continue
        print_summary(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
