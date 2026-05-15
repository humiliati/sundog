"""Pass C7 Step 2 — measure upper-tangent-arc wing extent from HaloSim images.

Reads the HaloSim PNG renders under docs/calibration/halosim_outputs/,
locates the 22-degree halo (used as the angular ruler), identifies the
upper-tangent-arc brightness on a ring slightly outside the halo, and
extracts the azimuth at which the wings fall to background. The result
is the HaloSim-predicted wing azimuth at the rendered sun altitude.

Comparison vs the C5 hand-anchor opening-angle reduction (computed by
scripts/tangent_opening_angle.py) is the Pass C7 verdict: does the
canonical literature inverse handle agree with the C5 hand-anchor
points, or do they diverge?

Run: python scripts/tangent_halosim_measure.py
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

# Renders to measure. Each (filename, h_deg, label).
RUNS = [
    ("halosim_tangent_p2_h18.6_25000mr.png", 18.6, "p2"),
    ("halosim_tangent_p13_h6.83_25000mr.png", 6.83, "p13"),
    ("halosim_tangent_p13_h1_10000mr.png", 1.0, "low-h sanity"),
    ("halosim_tangent_p2_h20_25000mr.png", 20.0, "p2-near"),
]


def load_grayscale(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32)


def find_22deg_halo(gray: np.ndarray) -> tuple[float, float, float]:
    """Estimate (sun_x, sun_y, r22_px) for a HaloSim render.

    HaloSim's view is symmetric about the sun-meridian (vertical line) and
    the sun sits at image-center horizontally. The vertical sun position
    varies with the rendered sun-elevation. The 22-degree halo is a
    high-brightness annulus centered on the sun.

    Approach:
      1. sun_x = image center column (HaloSim centers the view on the sun).
      2. sun_y = column-wise brightness maximum along sun_x (the sun core).
      3. r22_px = radial distance from (sun_x, sun_y) at which the radial
         brightness profile peaks (the 22-degree halo ring).
    """
    h, w = gray.shape
    sun_x = w / 2.0

    # Find sun_y: the y-coordinate of peak brightness along the central column.
    central_col = gray[:, int(sun_x)]
    sun_y = float(np.argmax(central_col))

    # Build a radial brightness profile.
    yy, xx = np.indices(gray.shape)
    rr = np.hypot(xx - sun_x, yy - sun_y)

    # Bin into integer-radius bins out to a generous max.
    max_r = int(min(sun_x, w - sun_x, sun_y, h - sun_y))
    r_int = np.clip(rr.astype(int), 0, max_r)

    # Mean brightness per radius bin.
    sums = np.bincount(r_int.ravel(), weights=gray.ravel(), minlength=max_r + 1)
    cnts = np.bincount(r_int.ravel(), minlength=max_r + 1)
    cnts = np.where(cnts == 0, 1, cnts)
    profile = sums / cnts

    # The 22-degree halo is the strongest ring after the central sun blob.
    # Mask out the inner ~50 px (sun core + immediate surround) and find
    # the brightest local maximum.
    inner_mask = 50
    search = profile[inner_mask:].copy()
    # Smooth with a moving average.
    kernel_n = 5
    kernel = np.ones(kernel_n) / kernel_n
    smoothed = np.convolve(search, kernel, mode="same")
    peak_idx = int(np.argmax(smoothed))
    r22_px = peak_idx + inner_mask

    return sun_x, sun_y, float(r22_px)


def measure_wing_extent(
    gray: np.ndarray,
    sun_x: float,
    sun_y: float,
    r22_px: float,
    background_pct: float = 50.0,
) -> dict:
    """Measure the upper-tangent-arc wing azimuth extent.

    Strategy: the upper tangent arc sits *outside* the 22-degree halo, with
    its apex on the halo crown directly above the sun. Sample brightness
    along an arc at radius = 1.3 * r22 (well outside the halo, where the
    halo intensity has fallen but the tangent-arc wings are still bright)
    over azimuths from -90 to +90 degrees relative to zenith. Identify the
    apex peak at azimuth 0 and trace outward until brightness drops to the
    azimuth-mean background level.
    """
    h, w = gray.shape

    # Sample radius: well outside the halo, capturing the wing locus.
    r_sample = 1.35 * r22_px

    # Sample 360 azimuth bins covering 0..pi (upper hemisphere only).
    n_az = 361
    azimuths = np.linspace(-np.pi / 2, np.pi / 2, n_az)
    # For each azimuth, sample the brightness at (sun_x + r*sin(az), sun_y - r*cos(az)).
    sample_x = sun_x + r_sample * np.sin(azimuths)
    sample_y = sun_y - r_sample * np.cos(azimuths)

    # Bilinear interpolation over the image.
    def bilinear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x0 = np.clip(np.floor(x).astype(int), 0, w - 2)
        y0 = np.clip(np.floor(y).astype(int), 0, h - 2)
        fx = x - x0
        fy = y - y0
        return (
            gray[y0, x0] * (1 - fx) * (1 - fy)
            + gray[y0, x0 + 1] * fx * (1 - fy)
            + gray[y0 + 1, x0] * (1 - fx) * fy
            + gray[y0 + 1, x0 + 1] * fx * fy
        )

    profile = bilinear(sample_x, sample_y)

    # Define background as a low percentile of the upper-hemisphere profile.
    bg = float(np.percentile(profile, background_pct))
    peak = float(profile.max())

    # Threshold midway between background and peak.
    threshold = bg + 0.5 * (peak - bg)

    # Find zenith bin and trace outward in both directions until profile falls
    # below threshold.
    zenith_bin = n_az // 2
    az_step_deg = 180.0 / (n_az - 1)

    # Right (positive azimuth) extent.
    right_idx = zenith_bin
    while right_idx + 1 < n_az and profile[right_idx + 1] > threshold:
        right_idx += 1
    right_az_deg = math.degrees(azimuths[right_idx])

    # Left (negative azimuth) extent.
    left_idx = zenith_bin
    while left_idx - 1 >= 0 and profile[left_idx - 1] > threshold:
        left_idx -= 1
    left_az_deg = math.degrees(azimuths[left_idx])

    return {
        "r22_px": r22_px,
        "px_per_deg": r22_px / 22.0,
        "r_sample_px": r_sample,
        "r_sample_deg": r_sample / (r22_px / 22.0),
        "n_azimuth_bins": n_az,
        "az_step_deg": az_step_deg,
        "profile_min": float(profile.min()),
        "profile_max": peak,
        "background_pct_value": bg,
        "threshold": threshold,
        "left_az_deg": left_az_deg,
        "right_az_deg": right_az_deg,
        "full_opening_az_deg": right_az_deg - left_az_deg,
    }


def process(filename: str, h_deg: float, label: str) -> dict:
    path = os.path.join(HALOSIM_DIR, filename)
    if not os.path.exists(path):
        return {"file": filename, "label": label, "error": "file not found"}
    gray = load_grayscale(path)
    sun_x, sun_y, r22_px = find_22deg_halo(gray)
    wings = measure_wing_extent(gray, sun_x, sun_y, r22_px)

    return {
        "file": filename,
        "label": label,
        "h_deg": h_deg,
        "image_size": gray.shape[::-1],  # (w, h)
        "sun_xy": (sun_x, sun_y),
        "r22_px": r22_px,
        "wings": wings,
    }


def main() -> int:
    print("Pass C7 Step 2 — HaloSim opening-angle measurement")
    print("=" * 70)
    print()

    results = []
    for fname, h, label in RUNS:
        r = process(fname, h, label)
        results.append(r)

        print(f"--- {label}: {fname} ---")
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        print(f"  h = {r['h_deg']}°")
        print(f"  image size: {r['image_size']}")
        print(f"  sun = ({r['sun_xy'][0]:.0f}, {r['sun_xy'][1]:.0f})")
        print(f"  r22 = {r['r22_px']:.1f} px  ->  {r['wings']['px_per_deg']:.2f} px/deg")
        print(f"  sample radius = {r['wings']['r_sample_px']:.1f} px  ({r['wings']['r_sample_deg']:.1f}° from sun)")
        print(f"  profile: min={r['wings']['profile_min']:.1f}, peak={r['wings']['profile_max']:.1f}, bg(50%)={r['wings']['background_pct_value']:.1f}")
        print(f"  threshold (midway peak/bg): {r['wings']['threshold']:.1f}")
        print(f"  wing extent: left {r['wings']['left_az_deg']:+.2f}°, right {r['wings']['right_az_deg']:+.2f}°")
        print(f"  full opening angle: {r['wings']['full_opening_az_deg']:.2f}°")
        print()

    # Compare to hand-anchor reductions where applicable.
    print("=" * 70)
    print("Cross-check vs C5 hand-anchor opening-angle reduction (p2 only)")
    print("=" * 70)
    p2_result = next((r for r in results if r["label"] == "p2"), None)
    if p2_result and "wings" in p2_result:
        # Hand-anchor measurement from p2-anchor.json.
        with open(os.path.join(REPO_ROOT, "docs", "calibration", "p2-anchor.json"), encoding="utf-8") as f:
            anchor = json.load(f)
        sun_x_ph, sun_y_ph = anchor["sun"]
        r22_ph = anchor["r22"]
        ppd_ph = r22_ph / 22.0
        outer_left = anchor["upper_tangent_manual_samples"]["points"][2]  # left outer
        outer_right = anchor["upper_tangent_manual_samples"]["points"][4]  # right outer
        dx_l = outer_left["x"] - sun_x_ph
        dy_l = outer_left["y"] - sun_y_ph
        dx_r = outer_right["x"] - sun_x_ph
        dy_r = outer_right["y"] - sun_y_ph
        az_l = math.degrees(math.atan2(dx_l, -dy_l))
        az_r = math.degrees(math.atan2(dx_r, -dy_r))
        print(f"  p2 hand-anchor (C5): left {az_l:+.2f}°, right {az_r:+.2f}°, full opening {az_r-az_l:.2f}°")
        print(f"  p2 HaloSim (C7):    left {p2_result['wings']['left_az_deg']:+.2f}°, right {p2_result['wings']['right_az_deg']:+.2f}°, full opening {p2_result['wings']['full_opening_az_deg']:.2f}°")
        delta = (p2_result["wings"]["full_opening_az_deg"]) - (az_r - az_l)
        print(f"  Δ full-opening: {delta:+.2f}°")

    return 0


if __name__ == "__main__":
    sys.exit(main())
