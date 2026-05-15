"""Phase 15 #1' — read pyramidal ring radii in DEGREES off HaloSim's
own stamped angular Scale (Tools -> Scale, FIX).

Why this method: every prior attempt failed on the px<->degree
scale-lock + ring-centre-finding on split-sky "Sun centered Plan"
renders. HaloSim's native Scale instrument removes both: it is drawn
FROM the sun, so its row gives the exact sun-y, and its degree ticks
encode HaloSim's own (non-linear) px->degree mapping for the active
projection — read straight off the instrument the HaloSim authors built
to "measure distances from the sun" (help h10/h1/h5).

Established reality (8 ray-filtered renders): the face-pair filters do
NOT isolate single rings (6-fold crystal symmetry) — every wedge shows
the same dense odd-radius family. So a per-wedge residual table is
structurally impossible. The honest, defensible analysis is a SET-MATCH:
measure the full set of ring radii (deg, off HaloSim's scale) and compare
to the Tape AH-CH10 p6 predicted set. Anchors: the strong ordinary-halo
ring must read ~21.8 deg in w22 (3->5) and ~45.7 deg in w46 (1->5)
(HaloSim's own 22/46 halos) — if the anchors fail, the read is declared
not-valid and NOTHING is tabulated (no fabrication).

Geometry locks, in order of robustness:
  * sun-Y  = median row of the green ruler pixels (the ruler is a
    horizontal line drawn from the sun).
  * sun-X  = 1-D search (Y fixed) maximising detrended ring-peak SNR in
    the LEFT (filtered, non-ruler) half — well-constrained because Y is
    nailed; this is NOT the failed 2-D centre search.
  * px->deg = interpolation on the ruler's ordered tick px-positions;
    h10 "the fine gradations are degrees" -> tick k = k deg. The two
    physical anchors (w22, w46) validate this absolute assignment.

Usage: python scripts/pyramidal_scale_read.py            (all 8)
       python scripts/pyramidal_scale_read.py pyr_w22_e3_x5_scale.png
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

WEDGE = {  # tag -> (entry, exit, Tape AH-CH10 p6 predicted halo radius deg)
    "w9":   ("3", "26", 9.0),
    "w18":  ("13", "25", 18.3),
    "w20":  ("23", "26", 19.9),
    "w22":  ("3", "5", 21.8),     # ordinary 22 deg halo  — ANCHOR
    "w23a": ("1", "25", 22.9),
    "w23b": ("3", "25", 23.8),
    "w35":  ("23", "25", 34.9),
    "w46":  ("1", "5", 45.7),     # ordinary 46 deg halo  — ANCHOR
}
TAPE_SET = sorted({v[2] for v in WEDGE.values()})  # 9,18.3,19.9,21.8,22.9,23.8,34.9,45.7


def smooth(p, k=5):
    return np.convolve(p, np.ones(k) / k, mode="same")


def green_ruler(a):
    """Green FIXed ruler on the white/grey B&W renders (7/8). Returns
    mask + sun-Y (median ruler row)."""
    R, G, B = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    m = (G - R > 25) & (G - B > 25)
    if m.sum() < 80:
        # one early frame (w9) is magenta on a blue render
        m = (R - G > 55) & (B - G > 35)
        if m.sum() < 80:
            return None, None
    ys = np.where(m)[1] if False else np.where(m)[0]
    return m, float(np.median(ys))


def tick_px(mask, suny):
    """Tick = vertical spur off the horizontal ruler. tick-height(x) =
    max|y-suny| of ruler px in column x; local maxima = tick centres.
    Returned in ascending x (image px)."""
    ys, xs = np.where(mask)
    W = mask.shape[1]
    h = np.zeros(W)
    for x, y in zip(xs, ys):
        d = abs(y - suny)
        if d > h[x]:
            h[x] = d
    sm = smooth(h, 3)
    xs_on = np.where(sm > 0)[0]
    if xs_on.size < 10:
        return np.array([])
    x0, x1 = xs_on.min(), xs_on.max()
    base = np.median(sm[x0:x1 + 1][sm[x0:x1 + 1] > 0])
    ticks = []
    for i in range(x0 + 1, x1):
        if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1] and sm[i] > base * 1.05:
            if ticks and i - ticks[-1] < 2:
                if sm[i] > sm[ticks[-1]]:
                    ticks[-1] = i
                continue
            ticks.append(i)
    return np.array(ticks, dtype=float)


def left_radial(a, sun_x, suny, mask):
    """Detrended ring-SNR radial profile in the LEFT half (filtered
    rings; excludes the ruler which is on the right)."""
    L = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    H, W = L.shape
    yy, xx = np.indices(L.shape)
    rr = np.hypot(xx - sun_x, yy - suny)
    use = (xx < sun_x - 6) & (~mask) & (rr > 12)
    if not use.any():
        return None, 0
    rmax = int(rr[use].max())
    ri = rr[use].astype(int)
    s = np.bincount(ri, L[use], minlength=rmax + 1)
    c = np.bincount(ri, minlength=rmax + 1).astype(float)
    c[c == 0] = 1
    prof = smooth(s / c, 5)
    bl = np.array([np.median(prof[max(0, i - 25):i + 26])
                   for i in range(len(prof))])
    res = prof - bl
    nz = np.std(res[25:rmax - 10]) or 1.0
    return res / nz, rmax


def find_sun_x(a, suny, mask):
    """1-D search (Y locked by the ruler): the sun-x that makes the
    left-half rings stack into the sharpest detrended peaks."""
    best = (-1.0, None)
    for sx in range(820, 985, 3):
        snr, rmax = left_radial(a, sx, suny, mask)
        if snr is None or rmax < 80:
            continue
        score = float(np.clip(snr[25:rmax - 10], 0, None).sum())
        if score > best[0]:
            best = (score, sx)
    return best[1]


def ring_peaks(snr, rmax):
    pk = []
    for i in range(20, rmax - 5):
        if snr[i] > snr[i - 1] and snr[i] >= snr[i + 1] and snr[i] > 3.0:
            if pk and i - pk[-1][0] < 5:
                if snr[i] > pk[-1][1]:
                    pk[-1] = (i, float(snr[i]))
                continue
            pk.append((i, round(float(snr[i]), 1)))
    return pk


def px_to_deg(r_px, sun_x, ticks):
    """tick j (0-idx) is (j+1) deg from origin (h10). Tick px-distance
    from sun_x: t = ticks - sun_x (drop non-positive)."""
    t = ticks - sun_x
    t = t[t > 1]
    if len(t) < 6:
        return None
    deg = np.arange(1, len(t) + 1, dtype=float)
    if r_px < t[0] or r_px > t[-1]:
        return None  # outside HaloSim's calibrated range — never extrapolate
    return float(np.interp(r_px, t, deg))


def analyze(path):
    tag = next((k for k in WEDGE if os.path.basename(path).startswith("pyr_" + k + "_")), None)
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    mask, suny = green_ruler(a)
    if mask is None:
        return dict(tag=tag, ok=False, why="no ruler")
    sun_x = find_sun_x(a, suny, mask)
    if sun_x is None:
        return dict(tag=tag, ok=False, why="sun_x search failed")
    snr, rmax = left_radial(a, sun_x, suny, mask)
    peaks = ring_peaks(snr, rmax)
    ticks = tick_px(mask, suny)
    rings = []
    for rpx, s in peaks:
        d = px_to_deg(rpx, sun_x, ticks)
        if d is not None:
            rings.append((round(d, 1), rpx, s))
    return dict(tag=tag, ok=True, sun=(int(sun_x), int(suny)),
                n_tick=int((ticks - sun_x > 1).sum()),
                peaks_px=[p[0] for p in peaks], rings_deg=rings,
                pred=WEDGE[tag][2] if tag else None)


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("-")]
    paths = ([x if os.path.isabs(x) else os.path.join(PHASE15, x) for x in args]
             if args else sorted(glob.glob(os.path.join(PHASE15, "pyr_w*_scale.png"))))
    rows = [analyze(p) for p in paths]
    for p, r in zip(paths, rows):
        b = os.path.basename(p)
        if not r["ok"]:
            print(f"{b}: UNREADABLE ({r['why']})")
            continue
        print(f"{b} tag={r['tag']} sun={r['sun']} ticks={r['n_tick']} "
              f"pred={r['pred']}")
        print(f"  ring peaks px={r['peaks_px']}")
        print(f"  rings_deg(deg,px,snr)={r['rings_deg']}")

    def strongest_deg(tag):
        rr = next((x for x in rows if x.get("tag") == tag and x["ok"]), None)
        if not rr or not rr["rings_deg"]:
            return None
        return max(rr["rings_deg"], key=lambda z: z[2])[0]  # highest-SNR ring

    a22, a46 = strongest_deg("w22"), strongest_deg("w46")
    ok22 = a22 is not None and abs(a22 - 21.8) <= 2.5
    ok46 = a46 is not None and abs(a46 - 45.7) <= 3.0
    print(f"\nANCHORS  w22 strongest -> {a22} deg (expect ~21.8, "
          f"{'OK' if ok22 else 'FAIL'});  "
          f"w46 strongest -> {a46} deg (expect ~45.7, "
          f"{'OK' if ok46 else 'FAIL'})")
    if not (ok22 and ok46):
        print("VERDICT: anchors FAIL -> the scale read is NOT validated; "
              "no measured-vs-Tape set-match is tabulated (no fabrication). "
              "Pyramidal stays P2.")
        return 2
    # Anchors valid -> honest SET-MATCH of all measured rings vs Tape set
    alld = sorted({d for r in rows if r["ok"] for d, _, _ in r["rings_deg"]})
    print(f"\nSET-MATCH (anchors validated): measured ring degrees "
          f"(union over 8 frames) = {alld}")
    print(f"Tape AH-CH10 p6 predicted set = {TAPE_SET}")
    print("  Tape  | nearest measured | resid")
    for t in TAPE_SET:
        if alld:
            m = min(alld, key=lambda z: abs(z - t))
            print(f"  {t:5.1f} | {m:6.1f}          | {m - t:+.1f}")
        else:
            print(f"  {t:5.1f} |   (none)         |   -")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
