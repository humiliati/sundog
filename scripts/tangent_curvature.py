"""
tangent_curvature.py — Pass C4 of the Phase 10 attack campaign.

Wing-slope geometric curvature detector for the upper tangent arc.

Background
----------
Pass C2 (`scripts/tangent_detector.py`) returned not-recovered on the post-C1
sampled set (p2 / p13 / p27) using a wing-azimuth-offset Lab b\* ridge
detector with 22°-halo-radial-profile subtraction. The C2 receipt narrowed
the tangent-route open question to non-literature-standard detector
designs. Pass C4 tests the first of those — the wing-slope geometric
curvature detector that Persona 1 §3 (p2 entry) specifically recommended:

> "For p2, the apex is locally flat by tangency definition; a curvature
> detector applied to the wings would work, and the wings are detectable."

And Persona 1 §5 (atlas-model implication, broader audit memo §4.4):

> "Real measurement needs either gradient-based edge detection (the spine
> is at a brightness or chromaticity transition, not a peak) or manual
> sample selection from visual crops."

C2 measured chromatic *ridge* (peaks in Lab b\* residual). C4 measures
luminance-gradient *edges* (transitions in CIE L\*) and fits a circle to
the wing-edge locus. The fitted circle's radius is the observed wing
curvature; the test is whether observed curvature matches the locally-
circular upper-tangent prediction R_uta = R22.

Algorithm
---------
1. RGB → CIE Lab (reuses `tangent_detector.rgb_to_lab`).

2. For each wing (left, right) and each azimuth in [WING_AZIMUTH_INNER_DEG,
   WING_AZIMUTH_OUTER_DEG], sample the L\* (luminance) profile along the
   sun-radial direction through the predicted wing point. Profile window:
   ±RADIAL_HALF_PX, RADIAL_SAMPLE_COUNT samples.

3. Compute the radial gradient ∂L\*/∂r as a central-difference along the
   profile. The wing-arc edge (the arc's brightness transition) shows up
   as a gradient *peak*; this is the signal Persona 1 §5 named.

4. For each sample point, the edge candidate is at the offset of maximum
   |gradient|. Sign of the gradient distinguishes inside-edge (negative
   gradient = brighter sunward) from outside-edge (positive gradient =
   brighter outward); both are valid wing-edge candidates.

5. Collect the resulting (x, y) edge-candidate points across both wings.
   Filter to those whose gradient magnitude exceeds GRADIENT_MIN_AMPLITUDE
   AND whose radial offset is within RIDGE_OFFSET_TOLERANCE_PX of the
   predicted tangent locus.

6. Fit a circle to the surviving edge points using the linearized
   formulation x² + y² + Dx + Ey + F = 0 (least squares on [D, E, F]).
   The fitted center is (-D/2, -E/2); fitted radius is
   √(D²/4 + E²/4 − F).

7. Compare fitted R_uta_obs to predicted R_uta = R22 (the locally-circular
   approximation; matches `overlay_calibrate.py:462` in spirit).

Pre-registered gates
--------------------
A photo is "curvature-recovered" if all of:
    - ≥ MIN_EDGE_POINTS edge candidates surviving filters
    - circle-fit residual RMS ≤ FIT_RMS_TOLERANCE_PX
    - R_uta_obs / R22 in [RATIO_MIN, RATIO_MAX] (radius matches prediction)

A photo is "curvature-not-recovered" if any gate fails or the fit is
singular.

Outputs
-------
Per photo: edge-candidate count per wing, fitted R_uta_obs, fitted center
offset from (sun_x, apex_y), circle-fit residual RMS, and verdict.

The detector exists alongside C2's chromatic detector; the two test
different hypotheses on the same calibration set.

References
----------
- atoptics.co.uk/blog/tangent-arcs/ (Cowley, *Tangent Arcs*)
- PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md Persona 1 §3 p2 entry, §5
- PHASE10_ATTACK_ROADMAP.md Pass C4 (this pass)
"""

import math
import os

import numpy as np

from tangent_detector import (
    WING_AZIMUTH_INNER_DEG,
    WING_AZIMUTH_OUTER_DEG,
    WING_SAMPLE_COUNT,
    RADIAL_HALF_PX,
    RADIAL_SAMPLE_COUNT,
    load_image_lab,
    predicted_wing_point,
    sun_to_point_direction,
)


GRADIENT_MIN_AMPLITUDE = 1.5
RIDGE_OFFSET_TOLERANCE_PX = 12
MIN_EDGE_POINTS = 12
FIT_RMS_TOLERANCE_PX = 8.0
RATIO_MIN = 0.7
RATIO_MAX = 1.3


def sample_l_along_radial(lab, sun_x, sun_y, center_x, center_y):
    """Sample CIE L* values along a sun-radial profile through (cx, cy)."""
    ux, uy = sun_to_point_direction(sun_x, sun_y, center_x, center_y)
    H, W, _ = lab.shape

    ts = np.linspace(-RADIAL_HALF_PX, RADIAL_HALF_PX, RADIAL_SAMPLE_COUNT)
    xs = center_x + ts * ux
    ys = center_y + ts * uy

    xi = np.round(xs).astype(int)
    yi = np.round(ys).astype(int)
    in_bounds = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    if not np.all(in_bounds):
        return None

    return ts, lab[yi, xi, 0], xs, ys


def find_gradient_peak(ts, l_profile, xs, ys):
    """Find the radial offset of the strongest L* gradient on the profile.

    Returns (offset_px, signed_gradient, peak_x, peak_y) for the strongest
    edge candidate. The gradient is computed as a central-difference."""
    grad = np.gradient(l_profile)
    idx = int(np.argmax(np.abs(grad)))
    return float(ts[idx]), float(grad[idx]), float(xs[idx]), float(ys[idx])


def collect_edge_candidates(lab, sun_x, sun_y, r22_px):
    """Per-azimuth gradient-peak collection across both wings."""
    candidates = []
    out_of_frame = 0
    weak = 0
    far = 0
    for wing in ("left", "right"):
        for i in range(WING_SAMPLE_COUNT):
            t = i / (WING_SAMPLE_COUNT - 1)
            az_deg = WING_AZIMUTH_INNER_DEG + t * (
                WING_AZIMUTH_OUTER_DEG - WING_AZIMUTH_INNER_DEG
            )
            cx, cy = predicted_wing_point(sun_x, sun_y, r22_px, az_deg, wing)
            sampled = sample_l_along_radial(lab, sun_x, sun_y, cx, cy)
            if sampled is None:
                out_of_frame += 1
                continue
            ts, l_profile, xs, ys = sampled
            offset, signed_grad, px, py = find_gradient_peak(ts, l_profile, xs, ys)
            if abs(signed_grad) < GRADIENT_MIN_AMPLITUDE:
                weak += 1
                continue
            if abs(offset) > RIDGE_OFFSET_TOLERANCE_PX:
                far += 1
                continue
            candidates.append(
                {
                    "wing": wing,
                    "azimuth_deg": az_deg,
                    "predicted_x": cx,
                    "predicted_y": cy,
                    "edge_x": px,
                    "edge_y": py,
                    "offset_px": offset,
                    "signed_gradient": signed_grad,
                }
            )
    return candidates, {
        "out_of_frame": out_of_frame,
        "weak": weak,
        "far": far,
        "total_candidates": len(candidates),
    }


def fit_circle(points):
    """Linear least-squares circle fit.

    Solves x² + y² + Dx + Ey + F = 0 for [D, E, F]. Returns (cx, cy, R,
    rms_residual_px) where the residual is the RMS distance from each
    point to the fitted circle."""
    pts = np.array(points, dtype=np.float64)
    if pts.shape[0] < 3:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = coeffs
    cx = -D / 2
    cy = -E / 2
    radicand = (D ** 2 + E ** 2) / 4 - F
    if radicand <= 0:
        return None
    R = math.sqrt(radicand)
    dists = np.hypot(x - cx, y - cy)
    rms = float(np.sqrt(np.mean((dists - R) ** 2)))
    return cx, cy, R, rms


def detect(photo_path, sun, r22_px):
    sun_x, sun_y = sun
    _arr, lab = load_image_lab(photo_path)
    candidates, counts = collect_edge_candidates(lab, sun_x, sun_y, r22_px)

    points = [(c["edge_x"], c["edge_y"]) for c in candidates]
    fit = fit_circle(points) if len(points) >= MIN_EDGE_POINTS else None

    R_uta_predicted = r22_px

    if fit is None:
        verdict = "not-recovered"
        cx, cy, R_uta_obs, rms = (None, None, None, None)
        ratio = None
    else:
        cx, cy, R_uta_obs, rms = fit
        ratio = R_uta_obs / R_uta_predicted
        if (
            len(points) >= MIN_EDGE_POINTS
            and rms <= FIT_RMS_TOLERANCE_PX
            and RATIO_MIN <= ratio <= RATIO_MAX
        ):
            verdict = "curvature-recovered"
        else:
            verdict = "not-recovered"

    return {
        "photo": os.path.basename(photo_path),
        "sun": [sun_x, sun_y],
        "r22_px": r22_px,
        "R_uta_predicted": R_uta_predicted,
        "candidates_kept": counts["total_candidates"],
        "candidates_out_of_frame": counts["out_of_frame"],
        "candidates_weak_gradient": counts["weak"],
        "candidates_far_from_locus": counts["far"],
        "fit_cx": cx,
        "fit_cy": cy,
        "R_uta_obs": R_uta_obs,
        "ratio_obs_over_predicted": ratio,
        "fit_residual_rms_px": rms,
        "verdict": verdict,
    }


def summarize(result):
    lines = []
    lines.append(
        f"{result['photo']}  sun={tuple(result['sun'])}  R22={result['r22_px']}  "
        f"R_uta_pred={result['R_uta_predicted']}"
    )
    lines.append(
        f"  edge candidates kept: {result['candidates_kept']}  "
        f"(out-of-frame={result['candidates_out_of_frame']}, "
        f"weak={result['candidates_weak_gradient']}, "
        f"far={result['candidates_far_from_locus']})"
    )
    if result["R_uta_obs"] is None:
        lines.append("  circle fit: SINGULAR or too few points")
    else:
        lines.append(
            f"  fit center: ({result['fit_cx']:.1f}, {result['fit_cy']:.1f})  "
            f"R_uta_obs={result['R_uta_obs']:.1f}  "
            f"ratio={result['ratio_obs_over_predicted']:.3f}  "
            f"RMS={result['fit_residual_rms_px']:.2f}px"
        )
    lines.append(f"  VERDICT: {result['verdict']}")
    return "\n".join(lines)
