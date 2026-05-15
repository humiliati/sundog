"""Pass C7 Step 2 — HaloSim scale lock via horizon-line detection.

HaloSim's autosave .sim files declare: "The projection is a 'Camera View'.
The horizon is the white horizontal line." Use that as our angular ruler:
vertical distance from sun to horizon = h° at HaloSim's pixel scale.

Approach:
  1. Load grayscale image.
  2. Sun is the brightest pixel near the image-center column (HaloSim
     centers the view on the sun horizontally).
  3. Horizon detection: apply a Sobel-y filter (vertical gradient → strong
     where there's a sharp horizontal line). Sum |Sobel-y| across each
     row to get a horizontal-line strength profile. Smooth lightly. The
     strongest peak below sun_y (with width ≥ ~70% of image width when
     thresholded) is the horizon.
  4. Scale: px_per_deg = (horizon_y - sun_y) / h_deg.
  5. R22 in HaloSim pixels = 22 * px_per_deg.
  6. Cross-validate scale across multiple renders (h = 1°, 6.83°, 18.6°,
     20°). Renders at h ≥ ~30° won't have horizon on screen.

Then sample brightness along an arc just outside the 22° halo to find the
upper-tangent-arc wing extent.

Run: PYTHONIOENCODING=utf-8 python scripts/tangent_halosim_horizon.py
"""

from __future__ import annotations
import math
import os
import sys
import json

import numpy as np
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
HALOSIM_DIR = os.path.join(REPO_ROOT, "docs", "calibration", "halosim_outputs")

# Files known to contain the horizon (h is small enough for horizon to land in-frame).
HORIZON_RUNS = [
    ("halosim_tangent_p13_h1_10000mr.png", 1.0, "h=1° (max-open)"),
    ("halosim_tangent_p13_h6.83_25000mr.png", 6.83, "p13"),
    ("halosim_tangent_p2_h18.6_25000mr.png", 18.6, "p2"),
    ("halosim_tangent_p2_h20_25000mr.png", 20.0, "p2-near"),
]

# Files where horizon may be off-screen (high sun); used for opening-angle
# measurement once scale is locked.
HIGH_SUN_RUNS = [
    ("halosim_tangent_p13_h46_100000mr.png", 46.0, "h=46° (circumscribed)"),
]


def load_grayscale(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def find_sun(gray: np.ndarray) -> tuple[int, int]:
    """Sun = global brightest pixel. HaloSim sun is saturated white (255)."""
    h, w = gray.shape
    # Restrict to roughly the upper half + center to avoid horizon/subhorizon
    # bright stripes interfering at very low sun altitudes.
    sun_idx = np.unravel_index(np.argmax(gray), gray.shape)
    return int(sun_idx[1]), int(sun_idx[0])  # (x, y)


def sobel_y(gray: np.ndarray) -> np.ndarray:
    """Vertical gradient (detects horizontal lines/edges)."""
    # 3x3 Sobel-y kernel: row [-1, -2, -1] / [0, 0, 0] / [1, 2, 1]
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    h, w = gray.shape
    out = np.zeros_like(gray)
    # numpy-only 2D convolution via slicing
    for i in range(3):
        for j in range(3):
            out[1:-1, 1:-1] += kernel[i, j] * gray[i:h - 2 + i, j:w - 2 + j]
    return out


def find_horizon(gray: np.ndarray, sun_y: int, search_below_sun_min_px: int = 30) -> int | None:
    """Detect horizon as the strongest horizontal-line row below sun_y.

    Returns None if no horizon-like row is found.
    """
    h, w = gray.shape
    sy = sobel_y(gray)
    # Strong horizontal line ⇒ many columns with consistent sign of sy at this row.
    # Use absolute value to get edge magnitude.
    row_strength = np.abs(sy).sum(axis=1)
    # Mask out rows above (sun_y + min_below).
    cutoff = sun_y + search_below_sun_min_px
    if cutoff >= h - 1:
        return None
    masked = row_strength.copy()
    masked[:cutoff] = 0
    # Light smoothing to suppress single-row jitter.
    k = 5
    ker = np.ones(k) / k
    sm = np.convolve(masked, ker, mode="same")
    peak_idx = int(np.argmax(sm))
    # Require the peak to be substantially above background.
    bg = np.percentile(sm[cutoff:], 60)
    if sm[peak_idx] < 2.5 * bg:
        return None
    return peak_idx


def measure_scale_from_horizon(filename: str, h_deg: float, label: str) -> dict:
    path = os.path.join(HALOSIM_DIR, filename)
    if not os.path.exists(path):
        return {"label": label, "error": "file not found"}
    gray = load_grayscale(path)
    sun_x, sun_y = find_sun(gray)
    horizon_y = find_horizon(gray, sun_y)
    if horizon_y is None:
        return {
            "label": label,
            "file": filename,
            "image_size": gray.shape[::-1],
            "sun_xy": (sun_x, sun_y),
            "horizon_y": None,
            "px_per_deg": None,
            "r22_px": None,
            "note": "no horizon detected (high sun or image misconfigured)",
        }
    delta_y = horizon_y - sun_y
    px_per_deg = delta_y / h_deg if h_deg > 0 else None
    r22_px = 22.0 * px_per_deg if px_per_deg else None
    return {
        "label": label,
        "file": filename,
        "h_deg": h_deg,
        "image_size": gray.shape[::-1],
        "sun_xy": (sun_x, sun_y),
        "horizon_y": horizon_y,
        "delta_y": delta_y,
        "px_per_deg": px_per_deg,
        "r22_px": r22_px,
    }


def measure_wing_extent(
    gray: np.ndarray,
    sun_x: int,
    sun_y: int,
    r22_px: float,
    sample_radius_factor: float = 1.15,
) -> dict:
    """Measure upper-tangent-arc wing azimuth extent at a given scale.

    Samples brightness along an arc at radius = sample_radius_factor * r22
    (just outside the 22° halo), over azimuths -90° to +90° from zenith.
    """
    H, W = gray.shape
    r_sample = sample_radius_factor * r22_px
    n_az = 361
    azimuths = np.linspace(-np.pi / 2, np.pi / 2, n_az)
    sx = sun_x + r_sample * np.sin(azimuths)
    sy = sun_y - r_sample * np.cos(azimuths)  # negative cos because y up = -y

    # Reject samples that fall off-image
    valid = (sx >= 1) & (sx <= W - 2) & (sy >= 1) & (sy <= H - 2)
    profile = np.full(n_az, np.nan)
    sx_v = sx[valid]
    sy_v = sy[valid]
    # Bilinear sample
    x0 = np.floor(sx_v).astype(int)
    y0 = np.floor(sy_v).astype(int)
    fx = sx_v - x0
    fy = sy_v - y0
    vals = (
        gray[y0, x0] * (1 - fx) * (1 - fy)
        + gray[y0, x0 + 1] * fx * (1 - fy)
        + gray[y0 + 1, x0] * (1 - fx) * fy
        + gray[y0 + 1, x0 + 1] * fx * fy
    )
    profile[valid] = vals

    # Smooth lightly
    finite = np.where(np.isnan(profile), np.nanmean(profile), profile)
    k = 7
    ker = np.ones(k) / k
    sm = np.convolve(finite, ker, mode="same")

    # Background = median of profile over valid range
    bg = float(np.nanmedian(profile))
    peak = float(np.nanmax(profile))
    threshold = bg + 0.4 * (peak - bg)  # 40% of the way from bg to peak

    zenith_bin = n_az // 2
    right_idx = zenith_bin
    while right_idx + 1 < n_az and sm[right_idx + 1] > threshold:
        right_idx += 1
    left_idx = zenith_bin
    while left_idx - 1 >= 0 and sm[left_idx - 1] > threshold:
        left_idx -= 1

    return {
        "r_sample_px": r_sample,
        "r_sample_deg": r_sample * 22 / r22_px,
        "profile_min": float(np.nanmin(profile)),
        "profile_max": peak,
        "profile_bg": bg,
        "threshold": threshold,
        "left_az_deg": math.degrees(azimuths[left_idx]),
        "right_az_deg": math.degrees(azimuths[right_idx]),
        "full_opening_az_deg": math.degrees(azimuths[right_idx] - azimuths[left_idx]),
    }


def main() -> int:
    print("Pass C7 Step 2 — HaloSim scale lock via horizon detection")
    print("=" * 70)
    print()

    scales = []
    print("--- Scale calibration from low-sun renders (horizon visible) ---")
    for fname, h, label in HORIZON_RUNS:
        r = measure_scale_from_horizon(fname, h, label)
        scales.append(r)
        if "error" in r:
            print(f"  {label:<20}  {r['error']}")
            continue
        ppd = r.get("px_per_deg")
        r22 = r.get("r22_px")
        if ppd is None:
            print(
                f"  {label:<20}  sun=({r['sun_xy'][0]:>4}, {r['sun_xy'][1]:>3})  horizon=NOT DETECTED  ({r['note']})"
            )
        else:
            print(
                f"  {label:<20}  sun=({r['sun_xy'][0]:>4}, {r['sun_xy'][1]:>3})  horizon_y={r['horizon_y']:>3}  "
                f"Δy={r['delta_y']:>3}  →  {ppd:.2f} px/° (R22={r22:.1f} px)"
            )
    print()

    # Aggregate scale (mean of valid)
    valid_scales = [s for s in scales if s.get("px_per_deg") is not None]
    if not valid_scales:
        print("ERROR: No scale calibration succeeded. Cannot proceed.")
        return 1
    avg_ppd = float(np.mean([s["px_per_deg"] for s in valid_scales]))
    std_ppd = float(np.std([s["px_per_deg"] for s in valid_scales]))
    avg_r22 = avg_ppd * 22.0
    print(f"Aggregate scale: {avg_ppd:.2f} ± {std_ppd:.2f} px/° (n={len(valid_scales)})")
    print(f"Aggregate R22:   {avg_r22:.1f} px")
    print()

    print("--- Tangent-arc opening-angle measurement ---")
    for fname, h, label in HORIZON_RUNS:
        path = os.path.join(HALOSIM_DIR, fname)
        if not os.path.exists(path):
            continue
        gray = load_grayscale(path)
        sun_x, sun_y = find_sun(gray)
        wings = measure_wing_extent(gray, sun_x, sun_y, avg_r22)
        print(
            f"  {label:<20}  L={wings['left_az_deg']:+6.2f}°  R={wings['right_az_deg']:+6.2f}°  "
            f"full opening={wings['full_opening_az_deg']:5.2f}°  "
            f"(peak={wings['profile_max']:.1f}, bg={wings['profile_bg']:.1f}, thr={wings['threshold']:.1f})"
        )
    print()

    # Compare to p2 hand-anchor
    print("--- Pass C7 verdict: HaloSim vs C5 hand-anchor at h=18.6° ---")
    with open(os.path.join(REPO_ROOT, "docs", "calibration", "p2-anchor.json"), encoding="utf-8") as f:
        anchor = json.load(f)
    sx_ph, sy_ph = anchor["sun"]
    pts = anchor["upper_tangent_manual_samples"]["points"]
    outer_l = pts[2]  # left outer
    outer_r = pts[4]  # right outer
    az_l = math.degrees(math.atan2(outer_l["x"] - sx_ph, -(outer_l["y"] - sy_ph)))
    az_r = math.degrees(math.atan2(outer_r["x"] - sx_ph, -(outer_r["y"] - sy_ph)))
    print(f"  C5 hand-anchor (p2):  left {az_l:+6.2f}°, right {az_r:+6.2f}°, full opening {az_r-az_l:.2f}°")

    p2_halosim = next((s for s in scales if s.get("label") == "p2"), None)
    if p2_halosim and p2_halosim.get("px_per_deg"):
        gray = load_grayscale(os.path.join(HALOSIM_DIR, p2_halosim["file"]))
        wings = measure_wing_extent(gray, *p2_halosim["sun_xy"], p2_halosim["r22_px"])
        print(
            f"  HaloSim p2 (per-image scale):  left {wings['left_az_deg']:+6.2f}°, "
            f"right {wings['right_az_deg']:+6.2f}°, full opening {wings['full_opening_az_deg']:.2f}°"
        )
        delta_full = wings["full_opening_az_deg"] - (az_r - az_l)
        print(f"  Δ full opening (HaloSim - C5): {delta_full:+.2f}°")

    return 0


if __name__ == "__main__":
    sys.exit(main())
