"""Phase 15 #1' — read pyramidal ring radii in DEGREES off HaloSim's
own stamped angular Scale (Tools -> Scale, FIX).

Why this is the genuinely-new method: every prior attempt failed on the
px<->degree scale-lock + ring-centre-finding on split-sky "Sun centered
Plan" renders (the projection is non-linear; 4 centring methods all
defeated by the plot-square/split-sky edges). HaloSim's native Scale
instrument dissolves all of that at once:

  * its ORIGIN marks the exact sun centre (no centre-finding),
  * its TICKS encode HaloSim's own px->degree mapping for the active
    projection (no scale-lock, no linearity assumption — the non-uniform
    tick spacing IS the projection function), read straight off the
    instrument the HaloSim authors built to "measure distances from the
    sun" (help corpus h10/h1/h5).

The FIXed scale is stamped into autosave.bmp as a pure SATURATED colour
(magenta on a blue render, green on a B&W/white render — FIX picks for
contrast), so it is detected by saturation+structure, never a fixed hue.

Pipeline per scale-stamped PNG:
  1. scale mask = highly-saturated pixels (render is grey/blue, far less
     saturated than the pure-colour ruler).
  2. ruler axis via PCA of the mask; ORIGIN = the axis end nearer the
     image/render centre (the sun); TIP = far end.
  3. ticks = perpendicular spurs off the axis; ordered by axial distance
     from origin. h10: "the fine gradations are degrees" -> the k-th tick
     = k degrees (taller 5th/10th ticks are a counting sanity-check).
     This yields a monotone (px_along_axis -> degree) LUT that already
     embeds the projection non-linearity.
  4. ring radius (px) from the origin, sampled in the half WITHOUT the
     ruler so the ruler does not contaminate the radial profile.
  5. ring_deg = interp(ring_px, LUT). Anchors: w22 (3->5) must read
     ~21.8 deg and w46 (1->5) ~45.7 deg (HaloSim's own ordinary 22/46
     halos) — internal consistency before any odd-radius row is trusted.

NO fabrication: a ring is reported only if it is a clear radial peak AND
the scale LUT validates on the 22/46 anchors; otherwise the frame is
reported unreadable.

Usage: python scripts/pyramidal_scale_read.py [pyr_w*_scale.png ...]
       (no args -> all phase15_pyrfilter/pyr_w*_scale.png)
"""
from __future__ import annotations
import glob
import os
import sys
import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE15 = os.path.join(REPO, "docs", "calibration", "halosim_outputs",
                       "phase15_pyrfilter")

# Tape AH-CH10 p6 (faces -> halo radius, deg). Wedge tag -> (entry,exit,pred).
WEDGE = {
    "w9":   ("3", "26", 9.0),
    "w18":  ("13", "25", 18.3),
    "w20":  ("23", "26", 19.9),
    "w22":  ("3", "5", 21.8),    # ordinary 22 deg halo — anchor
    "w23a": ("1", "25", 22.9),
    "w23b": ("3", "25", 23.8),
    "w35":  ("23", "25", 34.9),
    "w46":  ("1", "5", 45.7),    # ordinary 46 deg halo — linearity anchor
}


def scale_mask(a: np.ndarray) -> np.ndarray:
    """Pure-colour ruler = high saturation & bright; render is grey
    (sat~0) or muted blue. Hue-agnostic."""
    R, G, B = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    mx = a.max(2)
    mn = a.min(2)
    sat = mx - mn
    return (sat > 110) & (mx > 140)


def ruler_axis(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) < 50:
        return None
    cx, cy = xs.mean(), ys.mean()
    # principal axis of the ruler pixel cloud
    u = np.vstack([xs - cx, ys - cy]).astype(float)
    cov = u @ u.T / u.shape[1]
    w, V = np.linalg.eigh(cov)
    ax = V[:, np.argmax(w)]                    # unit axis vector
    t = (xs - cx) * ax[0] + (ys - cy) * ax[1]  # axial coord
    perp = -(xs - cx) * ax[1] + (ys - cy) * ax[0]
    tlo, thi = t.min(), t.max()
    end_lo = np.array([cx + ax[0] * tlo, cy + ax[1] * tlo])
    end_hi = np.array([cx + ax[0] * thi, cy + ax[1] * thi])
    H, W = mask.shape
    ctr = np.array([W / 2.0, H / 2.0])
    # ORIGIN = ruler end nearer the render centre (the sun); TIP = far end
    if np.hypot(*(end_lo - ctr)) <= np.hypot(*(end_hi - ctr)):
        origin, sgn = end_lo, 1.0
    else:
        origin, sgn = end_hi, -1.0
    return dict(origin=origin, ax=ax * sgn, cx=cx, cy=cy,
                t=sgn * t, perp=perp, xs=xs, ys=ys)


def tick_degrees(R):
    """Tick = local maximum of |perpendicular extent| along the axis.
    Ordered ticks from the origin are 1,2,3,... degrees (h10)."""
    t = R["t"]
    perp = np.abs(R["perp"])
    order = np.argsort(t)
    t = t[order]
    perp = perp[order]
    tmax = int(t.max())
    # per-axial-bin max perpendicular extent (tick height profile)
    nb = tmax + 1
    prof = np.zeros(nb)
    for ti, pi in zip(t.astype(int), perp):
        if 0 <= ti < nb and pi > prof[ti]:
            prof[ti] = pi
    sm = np.convolve(prof, np.ones(3) / 3, mode="same")
    base = np.median(sm[sm > 0]) if np.any(sm > 0) else 0.0
    ticks = []
    for i in range(2, nb - 2):
        if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1] and sm[i] > base * 1.15:
            ticks.append(i)
    # merge ticks closer than 2 px (anti-double-count)
    merged = []
    for p in ticks:
        if merged and p - merged[-1] < 2:
            continue
        merged.append(p)
    return np.array(merged, dtype=float)  # px-distance of tick k (k=deg)


def radial_ring(a: np.ndarray, origin, axis, mask) -> list:
    """Brightness/contrast radial profile from the scale origin, sampled
    in the half-plane AWAY from the ruler (so the ruler is not measured
    as a feature). Returns candidate ring radii in px."""
    L = (0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2])
    H, W = L.shape
    yy, xx = np.indices(L.shape)
    dx, dy = xx - origin[0], yy - origin[1]
    rr = np.hypot(dx, dy)
    # half-plane opposite the ruler axis (dot < 0), exclude the scale px
    side = (dx * axis[0] + dy * axis[1]) < 0
    use = side & (~mask) & (rr > 12)
    rmax = int(rr[use].max()) if use.any() else 0
    if rmax < 40:
        return [], None
    ri = rr[use].astype(int)
    val = L[use]
    # background-agnostic contrast vs local-radial baseline
    s = np.bincount(ri, val, minlength=rmax + 1)
    c = np.bincount(ri, minlength=rmax + 1).astype(float)
    c[c == 0] = 1
    prof = s / c
    sm = np.convolve(prof, np.ones(5) / 5, mode="same")
    base = np.array([np.median(sm[max(0, i - 25):i + 26])
                     for i in range(len(sm))])
    res = sm - base
    nz = np.std(res[30:rmax - 10]) or 1.0
    snr = res / nz
    peaks = []
    for i in range(20, rmax - 5):
        if snr[i] > snr[i - 1] and snr[i] >= snr[i + 1] and snr[i] > 3.0:
            peaks.append((i, round(float(snr[i]), 1)))
    return peaks, snr


def px_to_deg(px, ticks):
    """Tick k (0-indexed j) is (j+1) degrees. Interpolate ring px."""
    if len(ticks) < 5:
        return None
    deg = np.arange(1, len(ticks) + 1, dtype=float)
    if px <= ticks[0]:
        return float(px / ticks[0]) if ticks[0] > 0 else None
    if px >= ticks[-1]:
        return None  # beyond calibrated range — do not extrapolate
    return float(np.interp(px, ticks, deg))


def analyze(path: str) -> dict:
    tag = next((k for k in WEDGE if os.path.basename(path).startswith("pyr_" + k + "_")), None)
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    m = scale_mask(a)
    R = ruler_axis(m)
    if R is None:
        return dict(tag=tag, ok=False, why="no scale ruler detected")
    ticks = tick_degrees(R)
    peaks, _ = radial_ring(a, R["origin"], R["ax"], m)
    out = dict(tag=tag, ok=True, origin=tuple(np.round(R["origin"], 1)),
               n_ticks=len(ticks),
               tick_px=[round(float(x), 1) for x in ticks[:8]],
               peaks_px=peaks[:10])
    if tag:
        out["entry"], out["exit"], out["pred_deg"] = WEDGE[tag]
    rings_deg = []
    for rpx, snr in peaks:
        d = px_to_deg(rpx, ticks)
        if d is not None:
            rings_deg.append((round(d, 2), rpx, snr))
    out["rings_deg"] = rings_deg
    return out


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("-")]
    if args:
        paths = [x if os.path.isabs(x) else os.path.join(PHASE15, x) for x in args]
    else:
        paths = sorted(glob.glob(os.path.join(PHASE15, "pyr_w*_scale.png")))
    if not paths:
        print("no scale PNGs found")
        return 1
    rows = []
    for p in paths:
        r = analyze(p)
        rows.append(r)
        if not r["ok"]:
            print(f"{os.path.basename(p)}: UNREADABLE ({r['why']})")
            continue
        print(f"{os.path.basename(p)}  tag={r['tag']} "
              f"origin={r['origin']} ticks={r['n_ticks']} "
              f"tick_px[0:6]={r['tick_px'][:6]}")
        print(f"  ring peaks px={r['peaks_px']}")
        print(f"  rings_deg(deg,px,snr)={r['rings_deg']}  "
              f"pred={r.get('pred_deg')}")
    # anchor check
    def near(tag, target, tol=2.0):
        rr = next((x for x in rows if x.get("tag") == tag and x["ok"]), None)
        if not rr or not rr["rings_deg"]:
            return None
        best = min(rr["rings_deg"], key=lambda z: abs(z[0] - target))
        return best[0], abs(best[0] - target) <= tol
    a22, a46 = near("w22", 21.8), near("w46", 45.7)
    print(f"\nANCHORS: w22->{a22} (expect ~21.8)  w46->{a46} (expect ~45.7)")
    print("(anchors must validate before any odd-radius residual is trusted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
