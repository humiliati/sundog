"""
overlay_calibrate.py — render the Halo Atlas predictions on top of a sundog
photograph and emit a residuals report.

Usage:
    python scripts/overlay_calibrate.py <photo> --sun X,Y --r22 PX [options]

Required:
    --sun X,Y      Sun pixel center (e.g. --sun 400,356)
    --r22 PX       Observed 22° halo radius in photo pixels (e.g. --r22 145)

Optional (atlas pose; default = canonical-halo-atlas):
    --sun-altitude DEG          (default: derived from --parhelion-offset if
                                 given, else 25)
    --parhelion-offset PX       Observed parhelion offset from sun in photo px;
                                if given, sun-altitude is inverse-inferred
                                from h = arccos(R_22_obs / offset).
    --parhelic-curvature C      (default: 0.05)
    --cza-curvature C           (default: 0.85)
    --supralateral S            (default: 0.40)
    --upper-tangent U           (default: 0.0)
    --out PATH                  Output PNG (default: <photo>_overlay.png)

Observed-feature overrides (green markers; optional):
    --parhelion-left X          Observed left parhelion x (photo px)
    --parhelion-right X         Observed right parhelion x (photo px)
    --cza-apex X,Y              Observed CZA apex (photo px)

The script does NOT need the workbench running — it re-implements the atlas
math from the geometry module, transforms to photo coords via the 22°-halo
anchor, and overlays.
"""

import argparse
import math
import os
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("ERROR: pip install Pillow --break-system-packages", file=sys.stderr)
    sys.exit(1)


# Workbench constants (must mirror parhelion-geometry.mjs)
WB_SUN = (500, 500)
WB_R22 = 220
WB_R46 = 440  # holds at 440 — see SUNDOG_V_GEOMETRY.md "R_46 note"


def parse_pair(s, name):
    try:
        a, b = s.split(",")
        return float(a), float(b)
    except Exception:
        sys.exit(f"ERROR: --{name} expects two comma-separated numbers (got {s!r})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo", type=str)
    ap.add_argument("--sun", required=True, type=str)
    ap.add_argument("--r22", required=True, type=float)
    ap.add_argument("--sun-altitude", type=float, default=None)
    ap.add_argument("--parhelion-offset", type=float, default=None)
    ap.add_argument("--parhelic-curvature", type=float, default=0.05)
    ap.add_argument("--cza-curvature", type=float, default=0.85)
    ap.add_argument("--supralateral", type=float, default=0.40)
    ap.add_argument("--upper-tangent", type=float, default=0.0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--parhelion-left", type=float, default=None)
    ap.add_argument("--parhelion-right", type=float, default=None)
    ap.add_argument("--cza-apex", type=str, default=None)
    args = ap.parse_args()

    photo_path = Path(args.photo)
    if not photo_path.is_file():
        sys.exit(f"ERROR: photo not found: {photo_path}")

    sx, sy = parse_pair(args.sun, "sun")
    r22_obs = args.r22

    # Inverse-infer sun altitude from parhelion offset if available.
    if args.sun_altitude is not None:
        h_deg = args.sun_altitude
    elif args.parhelion_offset is not None:
        if args.parhelion_offset <= r22_obs:
            sys.exit("ERROR: parhelion-offset must be > r22 (parhelia sit outside the 22° halo for h>0)")
        h_deg = math.degrees(math.acos(r22_obs / args.parhelion_offset))
    else:
        h_deg = 25.0
    h_rad = math.radians(h_deg)

    scale = r22_obs / WB_R22

    def w2p(x, y):
        return (sx + (x - WB_SUN[0]) * scale, sy + (y - WB_SUN[1]) * scale)

    photo = Image.open(photo_path).convert("RGB").copy()
    draw = ImageDraw.Draw(photo, "RGBA")

    # 22° halo (anchor)
    draw.ellipse([sx - r22_obs, sy - r22_obs, sx + r22_obs, sy + r22_obs],
                 outline=(255, 80, 80, 220), width=2)
    # 46° halo
    r46_p = WB_R46 * scale
    draw.ellipse([sx - r46_p, sy - r46_p, sx + r46_p, sy + r46_p],
                 outline=(80, 180, 255, 200), width=2)

    # Daggers — altitude-bound
    offset_w = WB_R22 / math.cos(h_rad)
    dagger_p = []
    for sign in (-1, +1):
        px, py = w2p(WB_SUN[0] + sign * offset_w, WB_SUN[1])
        dagger_p.append((px, py))
        draw.line([px - 7, py, px + 7, py], fill=(255, 255, 80, 255), width=2)
        draw.line([px, py - 9, px, py + 9], fill=(255, 255, 80, 255), width=2)

    # Parhelic arc — smile direction (lower arc of circle whose center is above daggers)
    d_apex = 200 * max(0.0, min(1.0, args.parhelic_curvature))
    halfChord = offset_w
    prev = None
    if d_apex < 0.5:
        draw.line([dagger_p[0], dagger_p[1]], fill=(255, 150, 50, 220), width=2)
    else:
        u_ = (halfChord ** 2 - d_apex ** 2) / (2 * d_apex)
        cy_w = WB_SUN[1] - u_
        r_w = math.hypot(halfChord, u_)
        for i in range(161):
            uu = -1 + 2 * i / 160
            x_w = WB_SUN[0] + uu * r_w
            if x_w < 0 or x_w > 1000:
                continue
            inside = 1 - uu * uu
            if inside < 0:
                continue
            y_w = cy_w + r_w * math.sqrt(inside)
            xp, yp = w2p(x_w, y_w)
            if prev:
                draw.line([prev, (xp, yp)], fill=(255, 150, 50, 220), width=2)
            prev = (xp, yp)

    # CZA — anchored to 46° halo top
    def cza_apex(c):
        anchored = WB_SUN[1] - WB_R46  # 60
        return anchored + (0.85 - max(0.4, min(1.4, c))) * 200

    def circle_thru_apex(apexY, endpointY, halfW):
        cy = (halfW * halfW + endpointY * endpointY - apexY * apexY) / (2 * (endpointY - apexY))
        r = abs(apexY - cy)
        return cy, r

    apexY = cza_apex(args.cza_curvature)
    cy_w, r_w = circle_thru_apex(apexY, 240, 300)
    prev = None
    for i in range(161):
        u = -1 + 2 * i / 160
        x_w = WB_SUN[0] + u * r_w
        if x_w < 0 or x_w > 1000:
            continue
        inside = 1 - u * u
        if inside < 0:
            continue
        y_w = cy_w - r_w * math.sqrt(inside)
        if y_w > 240:
            continue
        xp, yp = w2p(x_w, y_w)
        if prev:
            draw.line([prev, (xp, yp)], fill=(170, 80, 255, 220), width=2)
        prev = (xp, yp)

    # Supralateral arc — tangent to 46° halo top
    if args.supralateral > 0.001:
        tangentY = WB_SUN[1] - WB_R46
        R_supra = 400
        cy_supra = tangentY - R_supra
        prev = None
        for i in range(161):
            u = -1 + 2 * i / 160
            x_w = WB_SUN[0] + u * R_supra
            if x_w < 0 or x_w > 1000:
                continue
            inside = 1 - u * u
            if inside < 0:
                continue
            y_w = cy_supra + R_supra * math.sqrt(inside)
            if y_w < 0 or y_w > tangentY + 5:
                continue
            xp, yp = w2p(x_w, y_w)
            if prev:
                draw.line([prev, (xp, yp)], fill=(255, 200, 255, 200), width=2)
            prev = (xp, yp)

    # Upper tangent arc — tangent to 22° halo top
    if args.upper_tangent > 0.001:
        tangentY = WB_SUN[1] - WB_R22
        R_uta = 200
        cy_uta = tangentY - R_uta
        prev = None
        for i in range(161):
            u = -1 + 2 * i / 160
            x_w = WB_SUN[0] + u * R_uta
            if x_w < 0 or x_w > 1000:
                continue
            inside = 1 - u * u
            if inside < 0:
                continue
            y_w = cy_uta + R_uta * math.sqrt(inside)
            if y_w < 0 or y_w > tangentY + 5:
                continue
            xp, yp = w2p(x_w, y_w)
            if prev:
                draw.line([prev, (xp, yp)], fill=(255, 255, 255, 255), width=2)
            prev = (xp, yp)

    # Observed-feature markers (green crosses)
    def green_cross(x, y):
        draw.line([x - 10, y - 2, x + 10, y - 2], fill=(0, 255, 0, 255), width=2)
        draw.line([x - 10, y + 2, x + 10, y + 2], fill=(0, 255, 0, 255), width=2)

    if args.parhelion_left is not None:
        green_cross(args.parhelion_left, sy)
    if args.parhelion_right is not None:
        green_cross(args.parhelion_right, sy)
    if args.cza_apex is not None:
        cx, cy_obs = parse_pair(args.cza_apex, "cza-apex")
        draw.line([cx - 10, cy_obs, cx + 10, cy_obs], fill=(0, 255, 0, 255), width=2)
        draw.line([cx, cy_obs - 10, cx, cy_obs + 10], fill=(0, 255, 0, 255), width=2)

    # Legend
    y = 8
    items = [
        ((255, 80, 80, 255), "22° halo (anchor)"),
        ((80, 180, 255, 255), "46° halo"),
        ((255, 255, 80, 255), f"daggers (h={h_deg:.1f}°)"),
        ((255, 150, 50, 255), f"parhelic c={args.parhelic_curvature:.2f}"),
        ((170, 80, 255, 255), "CZA tangent-46° top"),
        ((255, 200, 255, 255), "supralateral"),
        ((255, 255, 255, 255), "upper tangent"),
        ((0, 255, 0, 255), "observed"),
    ]
    for color, label in items:
        draw.line([6, y + 6, 30, y + 6], fill=color, width=3)
        draw.text((34, y), label, fill=(255, 255, 255, 255))
        y += 14

    out_path = Path(args.out) if args.out else photo_path.with_name(photo_path.stem + "_overlay.png")
    photo.save(out_path)

    print(f"saved {out_path}")
    print(f"\nCalibration:")
    print(f"  photo:               {photo_path}")
    print(f"  sun pixel:           ({sx:.1f}, {sy:.1f})")
    print(f"  R_22 observed:       {r22_obs:.1f} px")
    print(f"  scale:               1 workbench unit = {scale:.4f} photo px")
    print(f"  R_46 predicted:      {r46_p:.1f} px")
    print(f"  sun altitude:        {h_deg:.2f}°" + ("  (inverse-inferred from --parhelion-offset)" if args.parhelion_offset is not None and args.sun_altitude is None else ""))
    print(f"  dagger offset:       {offset_w * scale:.1f} px (= R_22 / cos h)")
    print(f"  dagger positions:    ({dagger_p[0][0]:.1f}, {dagger_p[0][1]:.1f}) and ({dagger_p[1][0]:.1f}, {dagger_p[1][1]:.1f})")

    cza_apex_p = w2p(WB_SUN[0], apexY)
    print(f"  CZA apex predicted:  ({cza_apex_p[0]:.1f}, {cza_apex_p[1]:.1f})")

    if args.parhelion_left is not None and args.parhelion_right is not None:
        lr = (dagger_p[0][0] - args.parhelion_left, dagger_p[1][0] - args.parhelion_right)
        print(f"\nResiduals (predicted - observed):")
        print(f"  left dagger:   {lr[0]:+.1f} px")
        print(f"  right dagger:  {lr[1]:+.1f} px")
    if args.cza_apex is not None:
        cx, cy_obs = parse_pair(args.cza_apex, "cza-apex")
        print(f"  CZA apex:      ({cza_apex_p[0] - cx:+.1f}, {cza_apex_p[1] - cy_obs:+.1f}) px")


if __name__ == "__main__":
    main()
