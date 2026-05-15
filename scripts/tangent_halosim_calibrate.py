"""Pass C7 Step 2 — Calibrate HaloSim pixel-to-degree scale from a clean
22°-halo-only render.

Once the user produces a HaloSim render of just the 22° halo (from
22deg halo.sim, with whatever sun elevation gives a clean ring), this
script:

  1. Finds the sun (brightest central column pixel; HaloSim centers the
     view horizontally on the sun).
  2. Measures the 22° halo radius in HaloSim pixels by sampling the
     radial brightness profile and locating the strongest ring peak
     outside the sun core.
  3. Reports R22 in pixels and the implied 1°/px scale, which can then
     be applied to the other renders (assuming HaloSim's projection
     keeps FOV/scale constant across renders — verifiable by checking
     R22 in pixels stays consistent across multiple renders).

Usage: drop the calibration BMP into docs/calibration/halosim_outputs/
(it'll be auto-converted to PNG if it ends in .bmp). Then run:

  PYTHONIOENCODING=utf-8 python scripts/tangent_halosim_calibrate.py [FILENAME]

If FILENAME omitted, looks for any *_22halo_only*.png|bmp or *_h25*.png|bmp
or any file matching 'calibration' in the halosim_outputs directory.
"""

from __future__ import annotations
import glob
import os
import sys

import numpy as np
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
HALOSIM_DIR = os.path.join(REPO_ROOT, "docs", "calibration", "halosim_outputs")


def find_calibration_file(arg: str | None) -> str | None:
    if arg:
        # Try as full path, then as basename in HALOSIM_DIR
        for candidate in (arg, os.path.join(HALOSIM_DIR, arg)):
            if os.path.exists(candidate):
                return candidate
        return None
    # Auto-discover
    patterns = [
        "*calibration*.png",
        "*calibration*.bmp",
        "*22halo_only*.png",
        "*22halo_only*.bmp",
        "*_22deg*.png",
        "*_22deg*.bmp",
        "*_h0_*.png",
        "*_h0_*.bmp",
        "*_h25_*.png",
        "*_h25_*.bmp",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(HALOSIM_DIR, pat)))
        if hits:
            return hits[-1]  # newest
    return None


def ensure_png(path: str) -> str:
    """Convert BMP to PNG if needed; return the PNG path."""
    if path.lower().endswith(".png"):
        return path
    if not path.lower().endswith(".bmp"):
        return path
    png_path = path[:-4] + ".png"
    if not os.path.exists(png_path):
        Image.open(path).save(png_path, "PNG", optimize=True)
        print(f"  Converted: {os.path.basename(path)} → {os.path.basename(png_path)}")
    return png_path


def load_grayscale(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def find_sun_central(gray: np.ndarray) -> tuple[int, int]:
    """Sun = brightest pixel on central column (HaloSim centers horizontally on sun)."""
    H, W = gray.shape
    sx = W // 2
    col = gray[:, sx]
    sy = int(np.argmax(col))
    # Refine: take brightest pixel in a small box around the central-column peak
    box = 30
    y0, y1 = max(0, sy - box), min(H, sy + box + 1)
    x0, x1 = max(0, sx - box), min(W, sx + box + 1)
    sub = gray[y0:y1, x0:x1]
    rel = np.unravel_index(np.argmax(sub), sub.shape)
    return int(x0 + rel[1]), int(y0 + rel[0])


def measure_r22(gray: np.ndarray, sun_x: int, sun_y: int) -> dict:
    """Find the strongest ring peak around the sun = the 22° halo."""
    H, W = gray.shape
    yy, xx = np.indices(gray.shape)
    rr = np.hypot(xx - sun_x, yy - sun_y)
    max_r = int(min(sun_x, W - sun_x, sun_y, H - sun_y))
    r_int = np.clip(rr.astype(int), 0, max_r)
    sums = np.bincount(r_int.ravel(), weights=gray.ravel(), minlength=max_r + 1)
    cnts = np.bincount(r_int.ravel(), minlength=max_r + 1)
    cnts = np.where(cnts == 0, 1, cnts)
    profile = sums / cnts

    # Smooth
    k = 7
    ker = np.ones(k) / k
    sm = np.convolve(profile, ker, mode="same")

    # Mask out sun core (inner 80 px) - the 22° halo radius will be larger
    # than the sun blob in any reasonable HaloSim FOV.
    inner_mask = 80
    sm_search = sm.copy()
    sm_search[:inner_mask] = 0

    # Find all local maxima in the smoothed profile above a threshold.
    bg = float(np.percentile(sm_search[inner_mask:], 30))
    peak_val = float(sm_search.max())
    threshold = bg + 0.15 * (peak_val - bg)

    candidates = []
    for i in range(inner_mask + 5, len(sm_search) - 5):
        if (
            sm_search[i] > sm_search[i - 1]
            and sm_search[i] > sm_search[i + 1]
            and sm_search[i] > threshold
        ):
            candidates.append((i, float(sm_search[i])))

    return {
        "profile_min": float(profile.min()),
        "profile_max": float(profile.max()),
        "profile_bg_30pct": bg,
        "threshold": threshold,
        "max_r": max_r,
        "candidates": candidates[:15],  # first 15 candidate rings
    }


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    path = find_calibration_file(arg)
    if path is None:
        print("ERROR: No calibration file found.")
        print(f"  Searched directory: {HALOSIM_DIR}")
        print(f"  Patterns tried: *calibration*, *22halo_only*, *_22deg*, *_h0_*, *_h25_*")
        print("  Provide a filename or path as an argument, or save the calibration render")
        print("  with a name matching one of those patterns.")
        return 1

    path = ensure_png(path)
    print(f"Calibration file: {os.path.basename(path)}")
    gray = load_grayscale(path)
    H, W = gray.shape
    print(f"  Image: {W} x {H}")

    sun_x, sun_y = find_sun_central(gray)
    print(f"  Sun:   ({sun_x}, {sun_y})")

    r = measure_r22(gray, sun_x, sun_y)
    print(f"  Profile: min={r['profile_min']:.1f}, max={r['profile_max']:.1f}, bg(30%)={r['profile_bg_30pct']:.1f}")
    print(f"  Threshold for ring candidates: {r['threshold']:.1f}")
    print(f"  Top ring-radius candidates (radius_px, ring-mean-brightness):")
    for i, (rad, val) in enumerate(r["candidates"]):
        deg_if_22halo = 22.0
        ppd = rad / deg_if_22halo
        print(f"    {i+1:>2}. r = {rad:>3} px, brightness = {val:6.1f}   (if this is the 22° halo: 1° = {ppd:.2f} px)")
    print()
    print("Identify which radius corresponds to the visible 22° halo ring.")
    print("(In a clean 22-halo-only render, it should be the strongest candidate by a wide margin.)")
    print("Once you confirm the right radius, that's R22 in HaloSim pixels — the scale lock.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
