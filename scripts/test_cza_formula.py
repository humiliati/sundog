"""Pass A1a regression test: literature CZA formula vs. legacy hardcode.

Reads p2 and p27 anchor JSONs and measures predicted-vs-observed CZA apex
y-position under (a) the legacy `WB_R46`-based hardcode and (b) the
literature formula in `cza_formula.py`. Prints both deltas plus the
qualitative direction.

Pass A1a exit criteria (per docs/PHASE10_ATTACK_ROADMAP.md sec. 3, Pass A1a):
- Sanity check: at h = 22 deg, the literature formula round-trips to ~46 deg
  above sun. The legacy WB_R46 = 440 in workbench coordinates corresponds
  to 44 deg (since WB_R22 = 220 at 10 px/deg), so there is a ~1.7 deg gap
  even at the legacy operating point. (This is the WB_R46 = 460 issue
  the audit memo flags as the second compounding bug.)
- p2 (h = 18.6 deg): legacy residual is the audit memo's filed -19.3 px;
  literature should collapse it close to zero.
- p27 (h = 0.5 deg): literature CZA apex should be off-frame (y < 0),
  meaning the visible "CZA" in the photo is mis-identified.

This test does not assert specific px values; it asserts the qualitative
direction. The actual numerical deltas printed here are recorded as the
A1a result note in `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`.

Exits non-zero only if the qualitative direction is wrong (legacy worse
than literature on p2, or literature CZA apex is in-frame at p27). If
the qualitative direction is correct, A1b is cleared to proceed; if not,
A1b is blocked and Pass A1 is reopened with a new hypothesis.
"""

import json
import math
import os
import sys

# Allow running from anywhere as long as cza_formula.py is alongside.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cza_formula import (
    cza_above_sun_deg,
    cza_apex_y_above_sun_px,
    cza_apex_y_in_image,
    CZA_DISAPPEARANCE_ALTITUDE_DEG,
)

# Project paths.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
ANCHOR_DIR = os.path.join(REPO_ROOT, "docs", "calibration")

# Workbench legacy constants (mirrors overlay_calibrate.py:64-66).
WB_R22 = 220
WB_R46 = 440  # legacy; geometrically 2 * WB_R22 = 44 deg in workbench coords.

# Eligible photos for this regression test (per audit memo: p2 has the
# clean -19.3 px legacy residual, p27 is the off-frame off-by-everything
# case; both have anchored JSON files).
ELIGIBLE_PHOTOS = ["p2", "p27"]


def legacy_cza_y_above_sun_px(r22_px):
    """Legacy atlas: CZA y-offset above sun, in photo px.

    The atlas computes the CZA apex in workbench coords as
    `WB_SUN[1] - WB_R46`. This is then transformed to photo coords by
    a uniform scaling that preserves the WB_R46 / WB_R22 ratio of 2.
    So in photo coords the legacy puts the CZA at `2 * r22_px` above
    the sun. (Equivalently 44 deg above sun, which is wrong by ~2 deg
    even at h = 22 deg, and increasingly wrong below it.)
    """
    return 2.0 * r22_px


def load_anchor(name):
    """Load a p{NN}-anchor.json by short name (e.g. 'p2')."""
    path = os.path.join(ANCHOR_DIR, f"{name}-anchor.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # 1. Sanity check: formula reduces correctly at h = 22 deg.
    print("=" * 78)
    print("Pass A1a regression: CZA literature formula vs. legacy hardcode")
    print("=" * 78)
    print()
    print("Sanity check at h = 22 deg (the legacy operating point):")
    val22 = cza_above_sun_deg(22.0)
    legacy_at_22 = WB_R46 / (WB_R22 / 22.0)  # = WB_R46 deg-equivalent
    print(f"  literature CZA-above-sun: {val22:.3f} deg")
    print(f"  legacy (WB_R46/WB_R22 = 2.0 -> 44 deg): {legacy_at_22:.3f} deg")
    print(f"  delta:                    {val22 - legacy_at_22:+.3f} deg")
    print(f"  -> WB_R46 = 440 should be {round(2.091 * WB_R22)} (~{(2.091 * WB_R22 - WB_R46)/WB_R22 * 100:.1f}% high)")
    print()
    print(f"CZA disappearance altitude: {CZA_DISAPPEARANCE_ALTITUDE_DEG:.3f} deg")
    print()

    # 2. Per-photo predicted-vs-observed.
    print("Per-photo residuals (observed_apex_y - predicted_apex_y):")
    print()
    fmt_header = "{:<5} {:>5}  {:>4}  {:>5}  {:>5}  {:>10}  {:>7}  {:>10}  {:>10}  {:>7}  {:>10}"
    fmt_row    = "{:<5} {:>5.1f}  {:>4}  {:>5}  {:>5}  {:>9.1f}  {:>7.1f}  {:>+10.1f}  {:>9.1f}  {:>7.1f}  {:>+10.1f}"
    print(fmt_header.format(
        "photo", "h", "r22", "sun_y", "obs_y",
        "lit_above", "lit_y", "lit_resid",
        "leg_above", "leg_y", "leg_resid",
    ))
    print("-" * 110)

    qualitative_direction_ok = True
    rows = []
    for name in ELIGIBLE_PHOTOS:
        anchor = load_anchor(name)
        h = anchor.get("sun_altitude")
        if h is None:
            # p2 doesn't carry sun_altitude directly; recover from r22 and parhelion.
            # Use the inferred h from the residual table: p2 = 18.6, p27 = 0.5.
            inferred = {"p2": 18.6, "p27": 0.5}.get(name)
            if inferred is None:
                raise RuntimeError(f"no sun_altitude for {name} and no inferred fallback")
            h = inferred

        r22 = anchor["r22"]
        sun_y = anchor["sun"][1]

        # Observed apex y: read from anchor's cza_apex if present.
        # Pass A2 (2026-05-13): p27's cza_apex field was moved to
        # _disputed.cza_apex_legacy_2026_05_13.value because the visible
        # feature was re-classified as 46° halo top / supralateral merger.
        # The regression here still uses the original observed pixel
        # position for the geometric check (literature CZA at h=0.5° is
        # off-frame, which is what A2's re-classification *rests on*).
        cza_apex = anchor.get("cza_apex")
        if cza_apex is None:
            disputed = anchor.get("_disputed", {})
            legacy = disputed.get("cza_apex_legacy_2026_05_13")
            if legacy and isinstance(legacy.get("value"), list):
                cza_apex = legacy["value"]
            else:
                raise RuntimeError(f"{name} anchor has no cza_apex (top-level or _disputed)")
        obs_y = cza_apex[1]

        lit_above = cza_apex_y_above_sun_px(h, r22)
        leg_above = legacy_cza_y_above_sun_px(r22)
        lit_y = sun_y - lit_above if lit_above is not None else None
        leg_y = sun_y - leg_above
        lit_resid = (obs_y - lit_y) if lit_y is not None else None
        leg_resid = obs_y - leg_y

        if lit_y is None:
            # CZA disappeared.
            print(f"{name:<5} {h:>5.1f}  {r22:>4}  {sun_y:>5}  {obs_y:>5}  CZA disappeared at this altitude")
        else:
            print(fmt_row.format(
                name, h, r22, sun_y, obs_y,
                lit_above, lit_y, lit_resid,
                leg_above, leg_y, leg_resid,
            ))
        rows.append({
            "name": name, "h": h, "r22": r22, "sun_y": sun_y, "obs_y": obs_y,
            "lit_above": lit_above, "lit_y": lit_y, "lit_resid": lit_resid,
            "leg_above": leg_above, "leg_y": leg_y, "leg_resid": leg_resid,
        })

    print()
    print("Qualitative direction check:")
    print()

    # p2: literature should beat legacy substantially.
    p2 = next(r for r in rows if r["name"] == "p2")
    if p2["lit_resid"] is not None and abs(p2["lit_resid"]) < abs(p2["leg_resid"]):
        improvement = abs(p2["leg_resid"]) - abs(p2["lit_resid"])
        print(f"  p2:  literature residual ({p2['lit_resid']:+.1f} px) is closer to zero than legacy ({p2['leg_resid']:+.1f} px) by {improvement:.1f} px. CONFIRMED memo direction.")
    else:
        print(f"  p2:  literature residual NOT closer to zero than legacy. CONTRADICTS memo direction.")
        qualitative_direction_ok = False

    # p27: literature CZA apex should be at y < 0 (above frame top, off-screen).
    p27 = next(r for r in rows if r["name"] == "p27")
    if p27["lit_y"] is not None and p27["lit_y"] < 0:
        print(f"  p27: literature CZA apex y = {p27['lit_y']:+.1f} (above the top of the frame). Visible 'apex' at y = {p27['obs_y']} is NOT CZA. CONFIRMED memo direction.")
    else:
        lit_y_str = f"{p27['lit_y']:+.1f}" if p27['lit_y'] is not None else "DISAPPEARED"
        print(f"  p27: literature CZA apex y = {lit_y_str} is in-frame. Visible apex at y = {p27['obs_y']} could be CZA after all. CONTRADICTS memo direction.")
        qualitative_direction_ok = False

    print()
    if qualitative_direction_ok:
        print("RESULT: A1a CONFIRMS the audit memo's qualitative finding.")
        print("        A1b is CLEARED to proceed (with the verified formula in cza_formula.py).")
        sys.exit(0)
    else:
        print("RESULT: A1a CONTRADICTS the audit memo's qualitative finding.")
        print("        A1b is BLOCKED. Reopen Pass A1 with a new hypothesis.")
        sys.exit(1)


if __name__ == "__main__":
    main()
