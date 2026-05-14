"""
overlay_calibrate.py — render the Halo Atlas predictions on top of a sundog
photograph and emit a residuals report.

Usage:
    python scripts/overlay_calibrate.py <photo> --sun X,Y --r22 PX [options]
    python scripts/overlay_calibrate.py <photo> --anchors anchors.json [options]

Required:
    --sun X,Y      Sun pixel center (e.g. --sun 400,356)
    --r22 PX       Observed 22° halo radius in photo pixels (e.g. --r22 145)
    OR
    --anchors PATH JSON anchor bundle. CLI flags override matching JSON fields.

Optional (atlas pose; default = canonical-halo-atlas):
    --sun-altitude DEG          (default: derived from --parhelion-offset if
                                 given, else 25)
    --parhelion-offset PX       Observed parhelion offset from sun in photo px;
                                if given, sun-altitude is inverse-inferred
                                from h = arccos(R_22_obs / offset).
    --parhelic-y-offset-r22 K   Vertical belt offset as a fraction of R22.
                                Negative moves the parhelic circle / daggers
                                upward in image coordinates (default: -0.05).
    --parhelic-curvature C      (default: 0.05)
    --cza-curvature C           (default: 0.85)
    --supralateral S            (default: 0.40)
    --upper-tangent U           (default: 0.0)
    --lower-tangent L           (default: 0.0)
    --suncave-parry P           (default: 0.0)
    --parry-supralateral P      (default: 0.0)
    --infralateral I            (default: 0.0)
    --out PATH                  Output PNG (default: <photo>_overlay.png)

Observed-feature overrides (green markers; optional):
    --parhelion-left X          Observed left parhelion x (photo px)
    --parhelion-right X         Observed right parhelion x (photo px)
    --parhelion-y Y             Observed parhelion/parhelic-belt y (photo px)
    --cza-apex X,Y              Observed CZA apex (photo px)
    --tangent-samples SPEC      Observed tangent-arc samples as
                                "X1,Y1;X2,Y2;X3,Y3[;...]"
    --tangent-kind upper|lower  Which tangent primitive samples refer to
                                (default: upper)

The script does NOT need the workbench running — it re-implements the atlas
math from the geometry module, transforms to photo coords via the 22°-halo
anchor, and overlays.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("ERROR: pip install Pillow --break-system-packages", file=sys.stderr)
    sys.exit(1)

# Pass A1b (2026-05-13): use the literature CZA formula instead of the
# WB_R46-anchored hardcode. See scripts/cza_formula.py and
# docs/PHASE10_ATTACK_ROADMAP.md Pass A1a/A1b.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cza_formula import cza_apex_y_above_sun_px


# Workbench constants (must mirror parhelion-geometry.mjs)
WB_SUN = (500, 500)
WB_R22 = 220
# Pass A1b (2026-05-13): WB_R46 corrected from the legacy 440 to
# round(2.091 * WB_R22) = 460. The literature angular ratio is 46/22
# ~ 2.091, not 2.0; the legacy 440 encoded 44 deg in workbench coords
# (since 220 px = 22 deg), undersizing the 46 deg halo by ~4.5% angularly.
# Affects the 46 deg halo radius (line ~344) and the supralateral apex
# base (line ~412); the CZA inner function at line ~383 no longer reads
# WB_R46 (it now uses the literature formula via cza_formula). See
# docs/PHASE10_ATTACK_ROADMAP.md Pass A1b for the per-consumer rationale.
WB_R46 = round(2.091 * WB_R22)


def parse_pair(s, name):
    try:
        a, b = s.split(",")
        return float(a), float(b)
    except Exception:
        sys.exit(f"ERROR: --{name} expects two comma-separated numbers (got {s!r})")


def parse_samples(s, name):
    samples = []
    try:
        for chunk in s.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            samples.append(parse_pair(chunk, name))
    except Exception:
        sys.exit(f"ERROR: --{name} expects semicolon-separated X,Y pairs (got {s!r})")
    if len(samples) < 3:
        sys.exit(f"ERROR: --{name} needs at least three X,Y samples")
    return samples


def load_anchor_json(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception as exc:
        sys.exit(f"ERROR: could not read --anchors {path!r}: {exc}")
    if not isinstance(data, dict):
        sys.exit("ERROR: --anchors JSON must be an object")
    return data.get("anchors", data)


def first_value(data, *keys):
    for key in keys:
        cur = data
        ok = True
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return None


def pair_from_anchor(data, *keys):
    value = first_value(data, *keys)
    if value is None:
        return None
    if isinstance(value, str):
        return parse_pair(value, keys[0])
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    if isinstance(value, dict):
        x = first_value(value, "x", "px.x", "0")
        y = first_value(value, "y", "px.y", "1")
        if x is not None and y is not None:
            return float(x), float(y)
    sys.exit(f"ERROR: anchor field {keys[0]!r} must be X,Y or [X,Y]")


def samples_from_anchor(data, *keys):
    value = first_value(data, *keys)
    if value is None:
        return None
    if isinstance(value, str):
        return parse_samples(value, keys[0])
    if isinstance(value, list):
        samples = []
        for item in value:
            if isinstance(item, str):
                samples.append(parse_pair(item, keys[0]))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                samples.append((float(item[0]), float(item[1])))
            elif isinstance(item, dict) and "x" in item and "y" in item:
                samples.append((float(item["x"]), float(item["y"])))
            else:
                sys.exit(f"ERROR: tangent sample {item!r} must be X,Y, [X,Y], or {{x,y}}")
        if len(samples) < 3:
            sys.exit("ERROR: tangent_samples needs at least three samples")
        return samples
    sys.exit(f"ERROR: anchor field {keys[0]!r} must be sample list or string")


def number_from_anchor(data, *keys):
    value = first_value(data, *keys)
    return None if value is None else float(value)


def circle_fit(points):
    # Least-squares circle fit: x^2 + y^2 + A*x + B*y + C = 0.
    sums = {
        "x": 0.0, "y": 0.0, "xx": 0.0, "yy": 0.0, "xy": 0.0,
        "xxx": 0.0, "yyy": 0.0, "xxy": 0.0, "xyy": 0.0,
    }
    n = len(points)
    for x, y in points:
        xx = x * x
        yy = y * y
        sums["x"] += x
        sums["y"] += y
        sums["xx"] += xx
        sums["yy"] += yy
        sums["xy"] += x * y
        sums["xxx"] += xx * x
        sums["yyy"] += yy * y
        sums["xxy"] += xx * y
        sums["xyy"] += x * yy

    matrix = [
        [sums["xx"], sums["xy"], sums["x"], -(sums["xxx"] + sums["xyy"])],
        [sums["xy"], sums["yy"], sums["y"], -(sums["xxy"] + sums["yyy"])],
        [sums["x"], sums["y"], float(n), -(sums["xx"] + sums["yy"])],
    ]

    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(matrix[row][col]))
        if abs(matrix[pivot][col]) < 1e-9:
            sys.exit("ERROR: tangent samples are degenerate; cannot fit a circle")
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        div = matrix[col][col]
        matrix[col] = [v / div for v in matrix[col]]
        for row in range(3):
            if row == col:
                continue
            factor = matrix[row][col]
            matrix[row] = [matrix[row][i] - factor * matrix[col][i] for i in range(4)]

    a, b, c = matrix[0][3], matrix[1][3], matrix[2][3]
    cx = -a / 2
    cy = -b / 2
    r_sq = cx * cx + cy * cy - c
    if r_sq <= 0:
        sys.exit("ERROR: tangent samples produced invalid fitted radius")
    r = math.sqrt(r_sq)
    residuals = [math.hypot(x - cx, y - cy) - r for x, y in points]
    rms = math.sqrt(sum(v * v for v in residuals) / len(residuals))
    return cx, cy, r, rms


def draw_sample_cross(draw, x, y, color):
    draw.line([x - 7, y, x + 7, y], fill=color, width=2)
    draw.line([x, y - 7, x, y + 7], fill=color, width=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo", type=str)
    ap.add_argument("--anchors", type=str, default=None)
    ap.add_argument("--sun", type=str, default=None)
    ap.add_argument("--r22", type=float, default=None)
    ap.add_argument("--sun-altitude", type=float, default=None)
    ap.add_argument("--parhelion-offset", type=float, default=None)
    ap.add_argument("--parhelic-y-offset-r22", type=float, default=-0.05)
    ap.add_argument("--parhelic-curvature", type=float, default=0.05)
    ap.add_argument("--cza-curvature", type=float, default=0.85)
    ap.add_argument("--supralateral", type=float, default=0.40)
    ap.add_argument("--upper-tangent", type=float, default=0.0)
    ap.add_argument("--lower-tangent", type=float, default=0.0)
    ap.add_argument("--suncave-parry", type=float, default=0.0)
    ap.add_argument("--parry-supralateral", type=float, default=0.0)
    ap.add_argument("--infralateral", type=float, default=0.0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--parhelion-left", type=float, default=None)
    ap.add_argument("--parhelion-right", type=float, default=None)
    ap.add_argument("--parhelion-y", type=float, default=None)
    ap.add_argument("--cza-apex", type=str, default=None)
    ap.add_argument("--tangent-samples", type=str, default=None)
    ap.add_argument("--tangent-kind", choices=("upper", "lower"), default=None)
    args = ap.parse_args()

    photo_path = Path(args.photo)
    if not photo_path.is_file():
        sys.exit(f"ERROR: photo not found: {photo_path}")

    anchors = load_anchor_json(args.anchors) if args.anchors else {}

    sun_anchor = parse_pair(args.sun, "sun") if args.sun else pair_from_anchor(
        anchors, "sun", "sun_px", "sun.pixel", "sun_pixel"
    )
    if sun_anchor is None:
        sys.exit("ERROR: --sun X,Y is required unless --anchors supplies sun/sun_px")
    sx, sy = sun_anchor

    r22_obs = args.r22 if args.r22 is not None else number_from_anchor(
        anchors, "r22", "r22_px", "halo_22_radius_px", "anchor_22_halo_radius_px"
    )
    if r22_obs is None:
        sys.exit("ERROR: --r22 PX is required unless --anchors supplies r22/r22_px")

    if args.sun_altitude is None:
        args.sun_altitude = number_from_anchor(anchors, "sun_altitude", "sun_altitude_deg")
    if args.parhelion_offset is None:
        args.parhelion_offset = number_from_anchor(anchors, "parhelion_offset", "parhelion.offset_px")
    if args.parhelion_left is None:
        args.parhelion_left = number_from_anchor(anchors, "parhelion_left", "parhelion.left_x", "parhelion.left_px")
    if args.parhelion_right is None:
        args.parhelion_right = number_from_anchor(anchors, "parhelion_right", "parhelion.right_x", "parhelion.right_px")
    if args.parhelion_y is None:
        args.parhelion_y = number_from_anchor(anchors, "parhelion_y", "parhelion.y", "parhelic_belt_y")
    if args.parhelion_offset is None and args.parhelion_left is not None and args.parhelion_right is not None:
        args.parhelion_offset = ((sx - args.parhelion_left) + (args.parhelion_right - sx)) / 2
    if args.cza_apex is None:
        cza_anchor = pair_from_anchor(anchors, "cza_apex", "cza.apex", "cza_apex_px")
        if cza_anchor is not None:
            args.cza_apex = f"{cza_anchor[0]},{cza_anchor[1]}"
    tangent_samples = parse_samples(args.tangent_samples, "tangent-samples") if args.tangent_samples else samples_from_anchor(
        anchors, "tangent_samples", "upper_tangent.samples", "lower_tangent.samples", "tangent.samples"
    )
    tangent_kind = args.tangent_kind or first_value(anchors, "tangent_kind", "tangent.kind", "upper_tangent.kind") or "upper"
    if tangent_kind not in ("upper", "lower"):
        sys.exit("ERROR: tangent_kind must be 'upper' or 'lower'")

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
    belt_offset_w = args.parhelic_y_offset_r22 * WB_R22
    belt_y_w = WB_SUN[1] + belt_offset_w

    def w2p(x, y):
        return (sx + (x - WB_SUN[0]) * scale, sy + (y - WB_SUN[1]) * scale)

    def draw_wb_points(points, color, width=2):
        prev = None
        for x_w, y_w in points:
            xp, yp = w2p(x_w, y_w)
            if prev:
                draw.line([prev, (xp, yp)], fill=color, width=width)
            prev = (xp, yp)

    def circle_branch_points(cx, cy, r, x_min, x_max, y_min, y_max, branch="upper", steps=160):
        points = []
        for i in range(steps + 1):
            x_w = x_min + (x_max - x_min) * i / steps
            u = (x_w - cx) / r
            inside = 1 - u * u
            if inside < 0:
                continue
            root = r * math.sqrt(inside)
            y_w = cy + root if branch == "lower" else cy - root
            if y_w < y_min or y_w > y_max:
                continue
            points.append((x_w, y_w))
        return points

    def polar_arc_points(radius, start_deg, end_deg, y_min, y_max, steps=80):
        points = []
        for i in range(steps + 1):
            theta = math.radians(start_deg + (end_deg - start_deg) * i / steps)
            x_w = WB_SUN[0] + radius * math.cos(theta)
            y_w = WB_SUN[1] + radius * math.sin(theta)
            if 0 <= x_w <= 1000 and y_min <= y_w <= y_max:
                points.append((x_w, y_w))
        return points

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
        px, py = w2p(WB_SUN[0] + sign * offset_w, belt_y_w)
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
        cy_w = belt_y_w - u_
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

    # CZA — Pass A1b (2026-05-13): anchored at the literature CZA position
    # `arcsin(sqrt(n^2 - cos^2 h)) - h`, scaled by WB_R22 / 22 deg. The legacy
    # `anchored = WB_SUN[1] - WB_R46` was correct only at h ~ 22 deg; at other
    # altitudes the literature formula gives a different position. See
    # scripts/cza_formula.py and docs/PHASE10_ATTACK_ROADMAP.md Pass A1a/A1b.
    # `h_deg` is closed over from the enclosing function (set at line ~287).
    def cza_apex(c):
        cza_above_sun_w = cza_apex_y_above_sun_px(h_deg, WB_R22)
        if cza_above_sun_w is None:
            # CZA disappears geometrically at h > ~32.2 deg. Fall back to the
            # legacy WB_R46 anchor so the curve renders at *some* position
            # (callers / vocabulary classifications mark CZA as "not
            # applicable" at high h, e.g. p7 in the anchor table).
            anchored = WB_SUN[1] - WB_R46
        else:
            anchored = WB_SUN[1] - cza_above_sun_w
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

    # Lower tangent arc -- mirror of upper tangent at the 22° halo bottom
    if args.lower_tangent > 0.001:
        tangentY = WB_SUN[1] + WB_R22
        R_lta = 200
        cy_lta = tangentY + R_lta
        points = circle_branch_points(
            WB_SUN[0],
            cy_lta,
            R_lta,
            WB_SUN[0] - R_lta,
            WB_SUN[0] + R_lta,
            tangentY - 5,
            800,
            "upper",
        )
        draw_wb_points(points, (255, 255, 255, 220), width=2)

    # Suncave Parry arc -- Parry-orientation cap whose shoulders bow sunward
    if args.suncave_parry > 0.001:
        apexY = WB_SUN[1] - WB_R22 - 36
        endpointY = WB_SUN[1] - WB_R22 + 24
        halfWidth = 210
        cy_parry, r_parry = circle_thru_apex(apexY, endpointY, halfWidth)
        points = circle_branch_points(
            WB_SUN[0],
            cy_parry,
            r_parry,
            WB_SUN[0] - halfWidth,
            WB_SUN[0] + halfWidth,
            0,
            endpointY + 8,
            "upper",
            steps=120,
        )
        draw_wb_points(points, (255, 245, 210, 230), width=2)

    # Parry supralateral shoulders -- rare upper-lateral Parry-family accents
    if args.parry_supralateral > 0.001:
        left = polar_arc_points(500, 206, 248, 0, WB_SUN[1] - 10)
        right = polar_arc_points(500, 292, 334, 0, WB_SUN[1] - 10)
        draw_wb_points(left, (255, 210, 255, 220), width=2)
        draw_wb_points(right, (255, 210, 255, 220), width=2)

    # Infralateral arcs -- paired lower-lateral arcs outside the 46° halo
    if args.infralateral > 0.001:
        left = polar_arc_points(500, 142, 160, WB_SUN[1] + 5, 800)
        right = polar_arc_points(500, 20, 38, WB_SUN[1] + 5, 800)
        draw_wb_points(left, (120, 255, 220, 220), width=3)
        draw_wb_points(right, (120, 255, 220, 220), width=3)

    # Observed-feature markers (green crosses)
    def green_cross(x, y):
        draw.line([x - 10, y - 2, x + 10, y - 2], fill=(0, 255, 0, 255), width=2)
        draw.line([x - 10, y + 2, x + 10, y + 2], fill=(0, 255, 0, 255), width=2)

    if args.parhelion_left is not None:
        green_cross(args.parhelion_left, args.parhelion_y if args.parhelion_y is not None else dagger_p[0][1])
    if args.parhelion_right is not None:
        green_cross(args.parhelion_right, args.parhelion_y if args.parhelion_y is not None else dagger_p[1][1])
    if args.cza_apex is not None:
        cx, cy_obs = parse_pair(args.cza_apex, "cza-apex")
        draw.line([cx - 10, cy_obs, cx + 10, cy_obs], fill=(0, 255, 0, 255), width=2)
        draw.line([cx, cy_obs - 10, cx, cy_obs + 10], fill=(0, 255, 0, 255), width=2)
    tangent_fit = None
    if tangent_samples:
        for x, y_obs in tangent_samples:
            draw_sample_cross(draw, x, y_obs, (80, 255, 180, 255))
        tangent_fit = circle_fit(tangent_samples)
        cx_fit, cy_fit, r_fit, _rms = tangent_fit
        draw.ellipse([cx_fit - r_fit, cy_fit - r_fit, cx_fit + r_fit, cy_fit + r_fit],
                     outline=(80, 255, 180, 180), width=1)

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
        ((255, 245, 210, 255), "suncave Parry"),
        ((255, 210, 255, 255), "Parry supralateral"),
        ((120, 255, 220, 255), "infralateral arcs"),
        ((0, 255, 0, 255), "observed"),
        ((80, 255, 180, 255), f"{tangent_kind} tangent samples"),
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
    print(f"  parhelic y offset:   {args.parhelic_y_offset_r22:+.3f} R_22 = {belt_offset_w * scale:+.1f} px")
    print(f"  dagger offset:       {offset_w * scale:.1f} px (= R_22 / cos h)")
    print(f"  dagger positions:    ({dagger_p[0][0]:.1f}, {dagger_p[0][1]:.1f}) and ({dagger_p[1][0]:.1f}, {dagger_p[1][1]:.1f})")

    cza_apex_p = w2p(WB_SUN[0], apexY)
    print(f"  CZA apex predicted:  ({cza_apex_p[0]:.1f}, {cza_apex_p[1]:.1f})")

    if args.parhelion_left is not None and args.parhelion_right is not None:
        lr = (dagger_p[0][0] - args.parhelion_left, dagger_p[1][0] - args.parhelion_right)
        print(f"\nResiduals (predicted - observed):")
        print(f"  left dagger:   {lr[0]:+.1f} px")
        print(f"  right dagger:  {lr[1]:+.1f} px")
        if args.parhelion_y is not None:
            print(f"  dagger belt y: {dagger_p[0][1] - args.parhelion_y:+.1f} px")
    if args.cza_apex is not None:
        cx, cy_obs = parse_pair(args.cza_apex, "cza-apex")
        print(f"  CZA apex:      ({cza_apex_p[0] - cx:+.1f}, {cza_apex_p[1] - cy_obs:+.1f}) px")
    if tangent_fit is not None:
        cx_fit, cy_fit, r_fit, rms_fit = tangent_fit
        predicted_center_w = (WB_SUN[0], WB_SUN[1] - WB_R22 - 200) if tangent_kind == "upper" else (WB_SUN[0], WB_SUN[1] + WB_R22 + 200)
        pred_cx, pred_cy = w2p(*predicted_center_w)
        pred_r = 200 * scale
        obs_k = 1 / r_fit
        pred_k = 1 / pred_r
        print(f"  {tangent_kind} tangent fit:")
        print(f"    observed center: ({cx_fit:.1f}, {cy_fit:.1f}), r={r_fit:.1f}px, RMS={rms_fit:.2f}px")
        print(f"    predicted center:({pred_cx:.1f}, {pred_cy:.1f}), r={pred_r:.1f}px")
        print(f"    center residual: ({pred_cx - cx_fit:+.1f}, {pred_cy - cy_fit:+.1f}) px")
        print(f"    radius residual: {pred_r - r_fit:+.1f} px")
        print(f"    curvature residual: {pred_k - obs_k:+.6f} 1/px")


if __name__ == "__main__":
    main()
