"""
tangent_matched_filter.py -- Pass C6 of the Phase 10 attack campaign.

Matched-filter detection against a parameterized upper-tangent-arc model.

Background
----------
Pass C5 manual sample selection recovered the upper tangent arc on p2
(R_uta_obs / R22 = 0.824, RMS = 1.23 px). The receipt named matched-filter
detection as the highest-leverage next detector candidate because:

  C2 (wing-radial Lab b* ridge) and C4 (wing-slope luminance-gradient
  curvature) both sampled per-point features and missed p2's signal. C5
  manual recovered at "visual chromatic-dome" resolution. Matched-filter
  tests the whole-arc-shape hypothesis directly: if a synthetic
  upper-tangent template correlates strongly with the b* residual image
  at the predicted curvature, then automated detection IS possible -
  just not via per-sample picks.

C6 closes this question. If matched-filter recovers p2, the route's
detection-tooling story is: "the arc IS automatable, just needs a
whole-shape detector instead of per-sample." If matched-filter ALSO
misses p2, the signal is at a deeper "visual gestalt" level than any
template-based detector can reach on the current calibration set.

Algorithm
---------
1. Load image, compute halo-subtracted Lab b* residual (reuses
   `tangent_detector.compute_halo_b_radial_profile` and
   `tangent_detector.halo_subtracted_b` -- same substrate Pass C2 used).

2. For each candidate R_uta in [R_UTA_SCAN_FRAC_MIN * R22,
   R_UTA_SCAN_FRAC_MAX * R22] (step R_UTA_SCAN_STEP_PX):

   a. Generate a synthetic 2D template: pixels close to the predicted
      arc locus (locally-circular tangent to the 22 deg halo crown,
      centered at (sun_x, sun_y - R22 - R_uta)) get a Gaussian weight
      with width TEMPLATE_WIDTH_SIGMA_PX; pixels far from the arc get
      0. The template is restricted to azimuth in [0, ARC_MAX_AZIMUTH_DEG]
      from the apex direction so we include the apex (different from
      C2/C4 which excluded the inner strip).

   b. Compute Pearson correlation between the template (where non-zero)
      and the b* residual at the same pixels.

3. Find the R_uta with peak correlation. Report:
   - R_uta_obs
   - peak correlation score
   - peak prominence = peak_score / mean(scores at other R_uta values)

4. Pre-registered gates:
   - R_uta_obs / R22 in [0.7, 1.3]
   - peak correlation >= MIN_PEAK_CORR (above zero by a margin)
   - peak prominence >= MIN_PROMINENCE_RATIO (above baseline)

Pre-registered before the run: see the Pass C6 entry in
PHASE10_ATTACK_ROADMAP.md.
"""

import math
import os

import numpy as np

from tangent_detector import (
    load_image_lab,
    compute_halo_b_radial_profile,
    halo_subtracted_b,
)


R_UTA_SCAN_FRAC_MIN = 0.5
R_UTA_SCAN_FRAC_MAX = 1.5
R_UTA_SCAN_STEP_PX = 4

TEMPLATE_WIDTH_SIGMA_PX = 4.0
ARC_MAX_AZIMUTH_DEG = 75.0

PRE_REGISTERED_RATIO_MIN = 0.7
PRE_REGISTERED_RATIO_MAX = 1.3
MIN_PEAK_CORR = 0.10
MIN_PROMINENCE_RATIO = 1.5


def generate_template(image_shape, sun_x, sun_y, r22_px, r_uta_px,
                     width_sigma=TEMPLATE_WIDTH_SIGMA_PX,
                     max_azimuth_deg=ARC_MAX_AZIMUTH_DEG):
    """2D template encoding the predicted upper-tangent arc locus.

    Apex at (sun_x, sun_y - r22_px); circle center at
    (sun_x, sun_y - r22_px - r_uta_px); arc is the lower portion of
    the circle in image coords. Pixels within `width_sigma` of the arc
    locus get an exponential weight; pixels outside the wing azimuth
    range get 0.
    """
    H, W = image_shape
    ys, xs = np.indices((H, W))

    apex_y = sun_y - r22_px
    center_x = sun_x
    center_y = apex_y - r_uta_px

    dx = xs - center_x
    dy = ys - center_y

    # Restrict to "below center" half (arc lives there)
    # and within the wing azimuth range from the straight-down direction.
    angle_from_down_deg = np.degrees(np.arctan2(np.abs(dx), dy))
    in_arc_range = (dy > 0) & (angle_from_down_deg <= max_azimuth_deg)

    r_from_center = np.hypot(dx, dy)
    d_arc = np.abs(r_from_center - r_uta_px)

    template = np.where(
        in_arc_range,
        np.exp(-(d_arc ** 2) / (2 * width_sigma ** 2)),
        0.0,
    )
    return template


def pearson_correlation(template, b_residual, support_threshold=0.05):
    """Pearson correlation between template and image, computed over
    the template's non-trivial support."""
    mask = template > support_threshold
    n_support = int(mask.sum())
    if n_support < 100:
        return 0.0, n_support

    t = template[mask].astype(np.float64)
    b = b_residual[mask].astype(np.float64)

    t_centered = t - t.mean()
    b_centered = b - b.mean()
    t_norm = np.sqrt(np.sum(t_centered ** 2))
    b_norm = np.sqrt(np.sum(b_centered ** 2))
    if t_norm < 1e-9 or b_norm < 1e-9:
        return 0.0, n_support
    return float(np.dot(t_centered, b_centered) / (t_norm * b_norm)), n_support


def scan_r_uta(b_residual, sun_x, sun_y, r22_px):
    r_min = int(round(r22_px * R_UTA_SCAN_FRAC_MIN))
    r_max = int(round(r22_px * R_UTA_SCAN_FRAC_MAX))
    r_uta_values = list(range(r_min, r_max + 1, R_UTA_SCAN_STEP_PX))
    scores = []
    supports = []
    for r in r_uta_values:
        tpl = generate_template(b_residual.shape, sun_x, sun_y, r22_px, float(r))
        score, n_support = pearson_correlation(tpl, b_residual)
        scores.append(score)
        supports.append(n_support)
    return r_uta_values, scores, supports


def detect(photo_path, sun, r22_px):
    sun_x, sun_y = sun
    _arr, lab = load_image_lab(photo_path)
    halo_profile = compute_halo_b_radial_profile(lab, sun_x, sun_y)
    b_residual = halo_subtracted_b(lab, sun_x, sun_y, halo_profile)

    r_uta_values, scores, supports = scan_r_uta(b_residual, sun_x, sun_y, r22_px)
    scores_arr = np.array(scores)

    peak_idx = int(np.argmax(scores_arr))
    peak_score = float(scores_arr[peak_idx])
    peak_r_uta = int(r_uta_values[peak_idx])
    ratio = peak_r_uta / r22_px

    other_scores = np.delete(scores_arr, peak_idx)
    baseline_mean = float(other_scores.mean()) if other_scores.size else 0.0
    prominence = peak_score / baseline_mean if baseline_mean > 1e-9 else (
        float("inf") if peak_score > 0 else 0.0
    )

    pass_ratio = PRE_REGISTERED_RATIO_MIN <= ratio <= PRE_REGISTERED_RATIO_MAX
    pass_score = peak_score >= MIN_PEAK_CORR
    pass_prominence = prominence >= MIN_PROMINENCE_RATIO

    if pass_ratio and pass_score and pass_prominence:
        verdict = "matched-filter-recovered"
    elif pass_ratio and pass_score:
        verdict = "ratio-and-score-pass-prominence-fails"
    elif pass_ratio:
        verdict = "ratio-pass-score-fails"
    else:
        verdict = "not-recovered"

    return {
        "photo": os.path.basename(photo_path),
        "sun": [sun_x, sun_y],
        "r22_px": r22_px,
        "r_uta_scan_min": r_uta_values[0],
        "r_uta_scan_max": r_uta_values[-1],
        "r_uta_scan_step": R_UTA_SCAN_STEP_PX,
        "peak_r_uta": peak_r_uta,
        "peak_score": peak_score,
        "ratio_obs_over_predicted": ratio,
        "baseline_mean_score": baseline_mean,
        "prominence_ratio": prominence,
        "scan_r_uta_values": list(r_uta_values),
        "scan_scores": [round(s, 4) for s in scores],
        "verdict": verdict,
    }


def summarize(result):
    lines = []
    lines.append(
        f"{result['photo']}  sun={tuple(result['sun'])}  R22={result['r22_px']}"
    )
    lines.append(
        f"  R_uta scan: [{result['r_uta_scan_min']}, {result['r_uta_scan_max']}] "
        f"step {result['r_uta_scan_step']} px"
    )
    lines.append(
        f"  peak: R_uta={result['peak_r_uta']} px  "
        f"score={result['peak_score']:.4f}  "
        f"ratio={result['ratio_obs_over_predicted']:.3f}"
    )
    lines.append(
        f"  baseline mean: {result['baseline_mean_score']:.4f}  "
        f"prominence: {result['prominence_ratio']:.2f}x"
    )
    lines.append(f"  VERDICT: {result['verdict']}")
    return "\n".join(lines)
