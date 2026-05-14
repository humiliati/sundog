"""CZA (Circumzenithal Arc) literature-formula module.

Pass A1a deliverable from `docs/PHASE10_ATTACK_ROADMAP.md`.

Provides the literature CZA-apex position as a function of sun altitude
and the photo's measured 22-degree halo radius. Replaces the legacy
`sun_y - WB_R46` hardcode in `scripts/overlay_calibrate.py:381` (which is
geometrically correct only at h ~= 22° and even there mispredicts by
~1.7° because `WB_R46 = 440` was set to `2 * WB_R22` rather than
`2.091 * WB_R22`).

The formula here derives from the standard atmospheric-optics derivation
of the CZA arc altitude:

    CZA_arc_altitude_above_horizon = arcsin(sqrt(n^2 - cos^2 h))

where n = 1.31 is the refractive index of ice for the dominant CZA
wavelength and h is the sun altitude above the horizon. The CZA position
above the sun is therefore:

    CZA_above_sun_deg(h) = arcsin(sqrt(n^2 - cos^2 h)) - h

The CZA disappears (geometrically) when n^2 - cos^2 h > 1, i.e. when
cos h < sqrt(n^2 - 1) ~ 0.8462, i.e. when h > ~32.2 deg. The function
returns None above this threshold so callers can render the CZA layer
as "not applicable" rather than crash or produce a bogus position.

References:
- atoptics.co.uk/blog/circumzenithal-arc/
- iapetus.jb.man.ac.uk/cza/CZA.html
- audit memo: docs/calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md
  (sec. 2 items 1-3, 4.2; sec. 5 item 1)

Note on the audit memo's stated formula. The memo writes the formula as
`90 - h - arcsin(sqrt(n^2 - cos^2 h))` (sec. 2 item 2) but the memo's own
numerical predictions ("~57 deg above sun at h = 0.5 deg", "matches
WB_R46 at h ~ 22 deg") only hold for `arcsin(sqrt(n^2 - cos^2 h)) - h`.
Pass A1a's verify-gate caught the memo's stated formula was wrong before
it leaked into the atlas patch. The qualitative finding (formula bug
exists, p27 CZA off-frame at h = 0.5 deg) survives; the implementation
uses the verified formula.
"""

import math

# Refractive index of ice for the dominant CZA wavelength (sodium-D line).
# Per atoptics.co.uk and Bravais's classical derivation, the CZA forms
# from refraction through the top face and out the side face of a
# horizontally-oriented hexagonal plate ice crystal; the dominant
# refractive index for the visible-band centroid is 1.31. Slightly
# wavelength-dependent in the third decimal; not load-bearing here.
ICE_REFRACTIVE_INDEX = 1.31

# Sun altitude above which the CZA disappears geometrically. Solved from
# n^2 - cos^2 h = 1 -> cos h = sqrt(n^2 - 1). At n = 1.31 this is
# h_disappearance = arccos(sqrt(0.7161)) ~ 32.183 deg.
CZA_DISAPPEARANCE_ALTITUDE_DEG = math.degrees(math.acos(math.sqrt(ICE_REFRACTIVE_INDEX**2 - 1.0)))


def cza_above_sun_deg(h_deg, n=ICE_REFRACTIVE_INDEX):
    """Return the angular distance from the sun to the CZA apex, in degrees.

    Returns None if h_deg is above the CZA disappearance threshold
    (~32.2 deg at n = 1.31), where the CZA exits the sky.

    Inputs:
      h_deg: sun altitude above horizon, in degrees. Must be in [0, 90].
      n:     refractive index. Defaults to ice (1.31).

    Returns:
      float (degrees) or None.
    """
    if h_deg < 0.0 or h_deg > 90.0:
        raise ValueError(f"h_deg must be in [0, 90]; got {h_deg}")
    h_rad = math.radians(h_deg)
    discriminant = n**2 - math.cos(h_rad) ** 2
    if discriminant > 1.0:
        # CZA above the zenith / off-sky; geometrically disappears.
        return None
    if discriminant < 0.0:
        # Should not occur for n >= 1; included for completeness.
        return None
    return math.degrees(math.asin(math.sqrt(discriminant))) - h_deg


def cza_apex_y_above_sun_px(h_deg, r22_px, n=ICE_REFRACTIVE_INDEX):
    """Return the predicted CZA apex offset above the sun, in pixels.

    Scales the angular distance by the photo's measured R22 (the pixel
    radius of the 22-degree halo). The conversion is `r22_px / 22 deg`
    pixels per degree.

    Returns None if the CZA disappears at this altitude.

    Inputs:
      h_deg:  sun altitude above horizon, in degrees.
      r22_px: measured pixel radius of the 22-degree halo for this photo.
      n:      refractive index. Defaults to ice (1.31).

    Returns:
      float (pixels) or None. Positive value means the CZA apex sits
      that many pixels ABOVE the sun in the image (which usually means
      a smaller image y-coordinate, depending on the image convention).
    """
    deg = cza_above_sun_deg(h_deg, n=n)
    if deg is None:
        return None
    return deg * (r22_px / 22.0)


def cza_apex_y_in_image(h_deg, r22_px, sun_y_px, n=ICE_REFRACTIVE_INDEX):
    """Return the predicted CZA apex y-coordinate in image space.

    Convention: image y increases downward (top of frame is y = 0). The
    CZA apex is above the sun, so its image y is `sun_y - offset_above`.
    May return a negative value if the CZA apex sits above the top of
    the photo (off-frame); callers should treat that as "not applicable
    at this altitude on this image."

    Returns None if the CZA disappears at this altitude (h above ~32.2 deg).
    """
    above_px = cza_apex_y_above_sun_px(h_deg, r22_px, n=n)
    if above_px is None:
        return None
    return sun_y_px - above_px


if __name__ == "__main__":
    # Self-test: print a small altitude table.
    print(f"CZA disappearance altitude: {CZA_DISAPPEARANCE_ALTITUDE_DEG:.3f} deg")
    print()
    print(f"{'h (deg)':>7}  {'CZA above sun (deg)':>20}")
    print("-" * 30)
    for h in [0.0, 0.5, 5.0, 10.0, 18.6, 22.0, 30.0, 32.0, CZA_DISAPPEARANCE_ALTITUDE_DEG, 35.0]:
        v = cza_above_sun_deg(h)
        v_str = f"{v:>15.3f}" if v is not None else "        DISAPPEARED"
        print(f"  {h:>5.2f}  {v_str}")
