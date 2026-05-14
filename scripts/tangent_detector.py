"""
tangent_detector.py — Pass C2 of the Phase 10 attack campaign.

Wing-azimuth-offset Lab b* ridge detector for the upper tangent arc, built
to test whether the post-C1 tangent-route detection-gate failure is
tooling-conditional or class-level.

Background
----------
The Phase 10 closeout single-handle verdict noted that the upper-tangent-arc
inversion route fails detection across the post-C1 sampled set (p2 h=18.6°,
p13 h=6.83°, p27 h=0.5°) under a column-peak-on-sun-meridian intensity
detector. The synthetic optical audit
(PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md Persona 1 §3-§5) read each failure
mode as a *protocol artifact* rather than a class-level negative:

- p2: apex is locally flat by tangency definition → column-peak gives no
  localizable signal at the apex but the *wings* are detectable.
- p13: forward-scatter haze washes the intensity signal; the chromatic
  signal (yellow ridge against bluer sky) survives.
- p27: sun-bloom flare contaminates the sun meridian; off-meridian samples
  avoid the artifact.

The audit recommended a Lab b* ridge detector on a 22°-halo-subtracted
residual image, sampled off the sun meridian.

Algorithm
---------
1. Load the photo as an RGB array.
2. Convert RGB → CIE Lab via sRGB-gamma and the D65 white-point matrix.
3. For each wing (left, right) and each azimuth in
   [WING_AZIMUTH_INNER_DEG, WING_AZIMUTH_OUTER_DEG]:
   a. Compute the predicted tangent-arc point (xc, yc) on a locally-circular
      approximation tangent to the 22° halo crown above the sun.
   b. Sample b* values along a short radial profile through (xc, yc), running
      outward from the sun. The profile length is RADIAL_HALF_PX on either
      side of the predicted ridge point.
   c. Compute the *residual* b* profile by subtracting the radial baseline
      (median of the two profile endpoints) — this is the per-sample
      22°-halo-subtraction step.
   d. Find the residual-b* peak. Record peak offset from the predicted
      tangent radial coordinate and peak residual amplitude.
4. Aggregate per wing: a wing is "ridge-detected" if at least
   MIN_WING_AGREE samples have residual-b* peak ≥ RIDGE_AMPLITUDE_THRESHOLD
   and median absolute radial offset ≤ RIDGE_OFFSET_TOLERANCE_PX.

Outputs
-------
For each photo, the detector returns a dict with:
  - per-wing sample tables (azimuth_deg, predicted_radial_px, peak_offset_px,
    peak_b_residual)
  - per-wing verdict: "ridge-detected", "ridge-absent", "ambiguous"
  - per-photo verdict: "recovered", "not-recovered", "ambiguous"

The receipt language in PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md §4.8 caps
the claim at: "column-peak-on-sun-meridian fails on all four photos; a
literature-standard wing-based or Lab b*-channel ridge detector
recovers / does not recover the route on {photos}." This module produces
the {photos} list.

References
----------
- atoptics.co.uk/blog/tangent-arcs/ (Cowley, *Tangent Arcs*)
- dewbow.co.uk/haloes/utan1.html (Fleet, *High Sun Tangent Arcs*)
- PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md Persona 1 §3-§5
- PHASE10_ATTACK_ROADMAP.md Pass C2
"""

import math
import os

import numpy as np


WING_AZIMUTH_INNER_DEG = 15.0
WING_AZIMUTH_OUTER_DEG = 75.0
WING_SAMPLE_COUNT = 24

RADIAL_HALF_PX = 18
RADIAL_SAMPLE_COUNT = 37

RIDGE_AMPLITUDE_THRESHOLD = 3.0
RIDGE_OFFSET_TOLERANCE_PX = 10
MIN_WING_AGREE = 8


def srgb_to_linear(rgb_u8):
    rgb = rgb_u8.astype(np.float64) / 255.0
    low = rgb <= 0.04045
    out = np.where(low, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return out


def linear_rgb_to_xyz(rgb_lin):
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    return rgb_lin @ M.T


def xyz_to_lab(xyz):
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    x = xyz[..., 0] / xn
    y = xyz[..., 1] / yn
    z = xyz[..., 2] / zn

    delta = 6.0 / 29.0
    delta3 = delta ** 3

    def f(t):
        return np.where(t > delta3, np.cbrt(t), t / (3 * delta * delta) + 4.0 / 29.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def rgb_to_lab(rgb_u8):
    return xyz_to_lab(linear_rgb_to_xyz(srgb_to_linear(rgb_u8)))


def load_image_lab(photo_path):
    from PIL import Image

    img = Image.open(photo_path).convert("RGB")
    arr = np.array(img)
    lab = rgb_to_lab(arr)
    return arr, lab


def compute_halo_b_radial_profile(lab, sun_x, sun_y):
    """Median b* by integer pixel radius from sun.

    The 22° halo is azimuthally symmetric, so its chromatic ridge appears
    as a peak in the radial median of Lab b*. Subtracting this radial
    median from the b* image leaves the azimuthally-asymmetric residual,
    which is where the upper tangent arc (concentrated above sun) lives.

    This is the "22°-halo-subtracted residual image" recommended in
    PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md Persona 1 §4.
    """
    H, W, _ = lab.shape
    b_channel = lab[..., 2]
    ys, xs = np.indices((H, W))
    r = np.hypot(xs - sun_x, ys - sun_y).astype(int)
    max_r = int(r.max())
    profile = np.zeros(max_r + 1, dtype=np.float64)
    # vectorize via np.bincount-like grouped median is not native; fall back
    # to a per-radius median, which is O(max_r) Python loop over numpy slices.
    flat_r = r.ravel()
    flat_b = b_channel.ravel()
    order = np.argsort(flat_r, kind="stable")
    sorted_r = flat_r[order]
    sorted_b = flat_b[order]
    bounds = np.searchsorted(sorted_r, np.arange(max_r + 2))
    for rr in range(max_r + 1):
        start, end = bounds[rr], bounds[rr + 1]
        if end > start:
            profile[rr] = np.median(sorted_b[start:end])
    return profile


def halo_subtracted_b(lab, sun_x, sun_y, halo_profile):
    H, W, _ = lab.shape
    ys, xs = np.indices((H, W))
    r = np.hypot(xs - sun_x, ys - sun_y).astype(int)
    r = np.clip(r, 0, len(halo_profile) - 1)
    return lab[..., 2] - halo_profile[r]


def predicted_wing_point(sun_x, sun_y, r22_px, azimuth_deg, wing):
    apex_y = sun_y - r22_px
    R_uta = r22_px
    center_y = apex_y - R_uta
    sign = -1.0 if wing == "left" else 1.0
    theta = math.radians(azimuth_deg)
    x = sun_x + sign * R_uta * math.sin(theta)
    y = center_y + R_uta * math.cos(theta)
    return x, y


def sun_to_point_direction(sun_x, sun_y, x, y):
    dx, dy = x - sun_x, y - sun_y
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return 0.0, -1.0
    return dx / norm, dy / norm


def sample_b_along_radial(b_field, sun_x, sun_y, center_x, center_y):
    ux, uy = sun_to_point_direction(sun_x, sun_y, center_x, center_y)
    H, W = b_field.shape

    ts = np.linspace(-RADIAL_HALF_PX, RADIAL_HALF_PX, RADIAL_SAMPLE_COUNT)
    xs = center_x + ts * ux
    ys = center_y + ts * uy

    xi = np.round(xs).astype(int)
    yi = np.round(ys).astype(int)
    in_bounds = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    if not np.all(in_bounds):
        return None

    b = b_field[yi, xi]
    return ts, b


def find_ridge_peak(ts, b_profile):
    idx = int(np.argmax(b_profile))
    return float(ts[idx]), float(b_profile[idx]), b_profile


def detect_wing(b_residual, sun_x, sun_y, r22_px, wing):
    samples = []
    out_of_frame = 0
    for i in range(WING_SAMPLE_COUNT):
        t = i / (WING_SAMPLE_COUNT - 1)
        az_deg = WING_AZIMUTH_INNER_DEG + t * (
            WING_AZIMUTH_OUTER_DEG - WING_AZIMUTH_INNER_DEG
        )
        cx, cy = predicted_wing_point(sun_x, sun_y, r22_px, az_deg, wing)
        result = sample_b_along_radial(b_residual, sun_x, sun_y, cx, cy)
        if result is None:
            out_of_frame += 1
            samples.append(
                {
                    "azimuth_deg": az_deg,
                    "predicted_x": cx,
                    "predicted_y": cy,
                    "in_frame": False,
                }
            )
            continue
        ts, b_profile = result
        offset, amplitude, _residual = find_ridge_peak(ts, b_profile)
        samples.append(
            {
                "azimuth_deg": az_deg,
                "predicted_x": cx,
                "predicted_y": cy,
                "in_frame": True,
                "ridge_offset_px": offset,
                "ridge_amplitude_b": amplitude,
            }
        )

    in_frame_samples = [s for s in samples if s["in_frame"]]
    coherent = [
        s
        for s in in_frame_samples
        if s["ridge_amplitude_b"] >= RIDGE_AMPLITUDE_THRESHOLD
        and abs(s["ridge_offset_px"]) <= RIDGE_OFFSET_TOLERANCE_PX
    ]
    if not in_frame_samples:
        verdict = "out-of-frame"
    elif len(coherent) >= MIN_WING_AGREE:
        verdict = "ridge-detected"
    elif len(coherent) >= MIN_WING_AGREE // 2:
        verdict = "ambiguous"
    else:
        verdict = "ridge-absent"

    return {
        "wing": wing,
        "samples": samples,
        "in_frame_count": len(in_frame_samples),
        "coherent_count": len(coherent),
        "verdict": verdict,
        "out_of_frame_count": out_of_frame,
    }


def detect(photo_path, sun, r22_px):
    sun_x, sun_y = sun
    _arr, lab = load_image_lab(photo_path)
    halo_profile = compute_halo_b_radial_profile(lab, sun_x, sun_y)
    b_residual = halo_subtracted_b(lab, sun_x, sun_y, halo_profile)
    left = detect_wing(b_residual, sun_x, sun_y, r22_px, "left")
    right = detect_wing(b_residual, sun_x, sun_y, r22_px, "right")

    wings = [left, right]
    detected = sum(1 for w in wings if w["verdict"] == "ridge-detected")
    absent = sum(1 for w in wings if w["verdict"] == "ridge-absent")
    if detected == 2:
        photo_verdict = "recovered"
    elif absent == 2:
        photo_verdict = "not-recovered"
    else:
        photo_verdict = "ambiguous"

    return {
        "photo": os.path.basename(photo_path),
        "sun": [sun_x, sun_y],
        "r22_px": r22_px,
        "left": left,
        "right": right,
        "verdict": photo_verdict,
    }


def summarize(result):
    lines = []
    lines.append(f"{result['photo']}  sun={tuple(result['sun'])}  R22={result['r22_px']}")
    for wing_key in ("left", "right"):
        w = result[wing_key]
        lines.append(
            f"  {wing_key:5s} wing: verdict={w['verdict']:13s}  "
            f"coherent={w['coherent_count']}/{w['in_frame_count']}  "
            f"(out-of-frame={w['out_of_frame_count']})"
        )
        if w["in_frame_count"]:
            amps = [s["ridge_amplitude_b"] for s in w["samples"] if s["in_frame"]]
            offs = [s["ridge_offset_px"] for s in w["samples"] if s["in_frame"]]
            lines.append(
                f"           amplitude(b*) median={np.median(amps):+.2f}  max={np.max(amps):+.2f}"
            )
            lines.append(
                f"           offset(px)    median={np.median(offs):+.2f}  "
                f"abs-median={np.median(np.abs(offs)):.2f}"
            )
    lines.append(f"  PHOTO VERDICT: {result['verdict']}")
    return "\n".join(lines)
