"""Phase 15 follow-up #1 -- pyramidal odd-radius ring residual.

Quantitative probe of the pyramidal P2 candidate
(`SPECULATIVE_HALO_PROOFS.md`): measure the concentric ring radii in the
14E HaloSim receipt and compare them to the closed-form Tape AH-CH10 p6
wedge -> minimum-deviation table.

Method (per HALOSIM_VALIDATION_PROTOCOL.md, generalized to many rings,
with an honesty gate):

  1. Mask to the blue plot (grey app margin is BRIGHTER than rings -- must
     be excluded); R channel as ring signal. Auto-detect the ring centre
     (the "Sun centered Plan" projection puts the sun at the centre) from
     plot-bbox + bright-core priors, refined by radial-profile
     peak-sharpness over a generous (cx,cy) grid.
  2. Azimuthally-averaged radial profile (in-mask bincount; the proven
     technique from tangent_halosim_calibrate.py).
  3. RESOLVABILITY GATE: detrend the monotonic core-glow decay (wide
     median baseline) and keep only >3-sigma ring bumps. Require >=3
     distinct rings OUTSIDE the inner core to proceed.
  4. If NOT resolvable: report the honest negative + the radial-profile
     evidence file, and recommend the higher-fidelity render needed. **Do
     not** scale-lock or tabulate residuals off a smooth profile --
     fabricating a table would be dishonest.
  5. If resolvable (future higher-ray / ray-filter-isolated receipt):
     scale-lock on the ordinary 22 deg halo (3-5 prism, 21.8 deg), do the
     46 deg linearity cross-check (equidistant projection test), tabulate
     predicted-vs-measured residuals, and draw the predicted-circle
     overlay.

Exit codes: 0 resolvable+tabulated, 2 attempted-but-not-resolvable,
1 unmeasurable. This is a reproduction-strengthening measurement, NOT a
public result; see the gate in SPECULATIVE_HALO_PROOFS.md.

Usage:  python scripts/pyramidal_ring_residual.py [receipt.png]
        (receipt arg is abs or relative to phase14e/; default = the 1M
        14E receipt. Overlay/evidence names derive from the receipt stem,
        so a higher-ray re-run never clobbers the 1M artifact.)
"""
from __future__ import annotations
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(THIS_DIR)
PHASE14E = os.path.join(REPO, "docs", "calibration", "halosim_outputs", "phase14e")
DEFAULT_RECEIPT = os.path.join(PHASE14E, "pyramidal_h18_1M_bw.png")


def resolve_paths(argv: list[str]) -> tuple[str, str, str]:
    """Receipt path = argv[1] (abs, or relative to phase14e/), else the
    1M 14E default. Overlay / evidence names are derived from the receipt
    stem so a higher-ray re-run does not clobber the 1M artifact."""
    rcpt = DEFAULT_RECEIPT
    if len(argv) > 1:
        cand = argv[1]
        rcpt = cand if os.path.isabs(cand) else os.path.join(PHASE14E, cand)
    stem = os.path.splitext(os.path.basename(rcpt))[0]
    overlay = os.path.join(PHASE14E, stem + "_residual_overlay.png")
    proftxt = os.path.join(PHASE14E, stem + "_radialprofile.txt")
    return rcpt, overlay, proftxt

# Tape AH-CH10 p6 (Fig 10-8), n = 1.31. (faces, wedge_deg, halo_radius_deg).
# The ordinary 22 deg (3-5) and 46 deg (1-5) rings are the scale anchors;
# the rest are the odd-radius family.
TAPE = [
    ("3-26", 28.0, 9.0),
    ("13-25", 52.4, 18.3),
    ("23-26", 56.0, 19.9),
    ("3-5 (prism, 22 halo)", 62.0, 21.8),
    ("1-25", 60.0, 22.9),
    ("3-25", 63.8, 23.8),
    ("23-25", 80.2, 34.9),
    ("1-5 (basal+prism, 46 halo)", 90.0, 45.7),
]
R22 = 21.8   # ordinary-halo scale anchor (deg)
R46 = 45.7   # linearity cross-check anchor (deg)


def plot_mask_and_signal(img: Image.Image):
    """The receipt = grey app margin (R==G==B, L~224, BRIGHTER than rings)
    around a blue 'Sun centered Plan' disc. Mask to the blue plot; use the
    R channel as the ring signal (whitish rings -> high R; blue background
    -> low R; the flat blue level is suppressed and the grey margin is
    excluded by the mask)."""
    a = np.asarray(img, dtype=np.float32)
    R, G, B = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    sat = a.max(2) - a.min(2)
    mask = ((B - R) > 18.0) & (sat > 12.0)
    signal = np.where(mask, R, 0.0)
    return mask, signal


def radial_profile(signal: np.ndarray, mask: np.ndarray,
                    cx: float, cy: float, rmax: int) -> np.ndarray:
    """Azimuthally-averaged R-channel profile, averaging over IN-MASK
    pixels only (the plot is not a full disc -- split-sky dims one half --
    so divide by the in-mask count per radius, not all pixels)."""
    yy, xx = np.indices(signal.shape)
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    rr = np.clip(rr, 0, rmax)
    m = mask.ravel()
    rrm = rr.ravel()[m]
    sums = np.bincount(rrm, weights=signal.ravel()[m], minlength=rmax + 1)
    cnts = np.bincount(rrm, minlength=rmax + 1).astype(float)
    cnts[cnts == 0] = 1.0
    return sums / cnts


def smooth(p: np.ndarray, k: int = 7) -> np.ndarray:
    return np.convolve(p, np.ones(k) / k, mode="same")


def peak_sharpness(prof: np.ndarray, inner: int) -> float:
    """Sum of positive second-difference magnitude = how 'ringy' a profile
    is. Maximized when the centre is correct (rings stack coherently)."""
    s = smooth(prof)
    d2 = np.abs(np.diff(s, 2))
    return float(d2[inner:].sum())


def _rmax_for(mask: np.ndarray, cx: float, cy: float) -> int:
    ys, xs = np.where(mask)
    return int(min(cx - xs.min(), xs.max() - cx,
                   cy - ys.min(), ys.max() - cy)) - 2


def find_center(mask: np.ndarray, signal: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    # Prior 1: plot-mask bbox centre. Prior 2: bright-ring-core centroid
    # (top 0.5% R inside the mask). Search a generous box spanning both.
    bx, by = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    sv = signal[mask]
    thr = np.percentile(sv, 99.5)
    cys, cxs = np.where((signal >= thr) & mask)
    kx, ky = cxs.mean(), cys.mean()
    cx0, cy0 = (bx + kx) / 2.0, (by + ky) / 2.0
    best, best_s = (cx0, cy0), -1.0
    # Coarse +/-90 px (the bbox-centre vs core-centroid gap is ~90 px).
    for dy in range(-90, 91, 10):
        for dx in range(-90, 91, 10):
            cx, cy = cx0 + dx, cy0 + dy
            rmax = _rmax_for(mask, cx, cy)
            if rmax < 50:
                continue
            s = peak_sharpness(radial_profile(signal, mask, cx, cy, rmax), 28)
            if s > best_s:
                best_s, best = s, (cx, cy)
    cx0, cy0 = best
    for dy in range(-9, 10):
        for dx in range(-9, 10):
            cx, cy = cx0 + dx, cy0 + dy
            rmax = _rmax_for(mask, cx, cy)
            s = peak_sharpness(radial_profile(signal, mask, cx, cy, rmax), 28)
            if s > best_s:
                best_s, best = s, (cx, cy)
    return best


def detrended_rings(prof: np.ndarray, rmax: int, inner: int = 40):
    """Detrend the monotonic core-glow decay (wide median baseline) and
    return >3-sigma ring bumps outside the core. This is the resolvability
    gate: if the odd-radius rings are present above Monte-Carlo noise it
    finds them; if they are not, it honestly returns nothing rather than
    forcing matches to a smooth profile."""
    s = smooth(prof, 5)
    w = 51
    base = np.array([
        np.median(s[max(0, i - w // 2):min(len(s), i + w // 2 + 1)])
        for i in range(len(s))
    ])
    resid = s - base
    noise = float(np.std(resid[inner:rmax - 10])) or 1.0
    snr = resid / noise
    bumps = []
    for i in range(inner + 3, rmax - 5):
        if snr[i] > snr[i - 1] and snr[i] >= snr[i + 1] and snr[i] > 3.0:
            bumps.append((i, round(float(snr[i]), 1)))
    return bumps, noise, snr


def main() -> int:
    RECEIPT, OVERLAY_OUT, PROF_TXT = resolve_paths(sys.argv)
    if not os.path.isfile(RECEIPT):
        print(f"ERROR: receipt not found: {RECEIPT}")
        return 1
    img = Image.open(RECEIPT).convert("RGB")
    W, H = img.size
    mask, signal = plot_mask_and_signal(img)
    cx, cy = find_center(mask, signal)
    rmax = _rmax_for(mask, cx, cy)
    prof = radial_profile(signal, mask, cx, cy, rmax)
    bumps, noise, snr = detrended_rings(prof, rmax)

    print(f"receipt : {os.path.relpath(RECEIPT, REPO)}")
    print(f"image   : {W}x{H}   centre=({cx:.0f},{cy:.0f})   rmax={rmax}px")
    print(f"detrend : noise={noise:.2f}  >3sigma ring bumps (r_px,snr)="
          f"{bumps if bumps else 'NONE'}")

    # Resolvability gate. The odd-radius family needs >=3 distinct rings
    # OUTSIDE the inner core (r > ~70 px) to support a residual table; a
    # single near-core feature is the 9 deg ring blended with the sun and
    # is not enough to scale-lock + tabulate.
    core_cut = 70
    outer = [b for b in bumps if b[0] > core_cut]
    resolvable = len(outer) >= 3

    if not resolvable:
        near_core = [b for b in bumps if b[0] <= core_cut]
        print()
        print("RESULT: rings NOT resolvable from this receipt.")
        print(f"  Only the inner bright feature is significant "
              f"(r<={core_cut}px bumps: {near_core or 'none'} -- the ~9 deg "
              f"ring blended with the sun core). No discrete 18-35 deg ring "
              f"loci exceed 3 sigma over the diffuse glow.")
        print("  This is a DATA limit (1M rays is too few for the "
              "intrinsically faint pyramidal paths in an all-sky plan "
              "projection), not a method limit: the detrend+3sigma test "
              "would surface rings if present.")
        print("  => No predicted-vs-measured residual table is produced "
              "(fabricating one off a smooth profile would be dishonest).")
        print("  Refined next step: a dedicated higher-fidelity render -- "
              ">=4-10M rays and/or per-ring HaloSim ray-filter isolation "
              "(filter by entry/exit faces per the Tape wedge table) -- then "
              "re-run this script (the resolvability gate will pass and emit "
              "the residual table + linearity cross-check).")
        # Evidence artifact: detrended SNR profile (NOT a circle overlay --
        # drawing predicted circles needs a scale lock we do not have).
        with open(PROF_TXT, "w", encoding="utf-8") as fh:
            fh.write("# pyramidal receipt radial analysis -- "
                     "rings NOT resolvable\n")
            fh.write(f"# receipt={os.path.basename(RECEIPT)} "
                     f"centre=({cx:.0f},{cy:.0f}) rmax={rmax} "
                     f"detrend_noise={noise:.3f}\n")
            fh.write("# r_px\tprofile\tdetrend_snr\n")
            for r in range(0, rmax):
                fh.write(f"{r}\t{prof[r]:.2f}\t{snr[r]:.2f}\n")
        print(f"  evidence: {os.path.relpath(PROF_TXT, REPO)} "
              f"(radial profile + detrended SNR)")
        return 2  # 2 = attempted, not resolvable (distinct from 0/1)

    # --- Resolvable path (future higher-ray receipt): scale-lock on the
    # 22 deg halo, linearity cross-check via 46 deg, residual table. ---
    peaks = [(r, float(prof[r])) for r, _ in outer]
    band = [(r, v) for r, v in peaks if 0.30 * rmax <= r <= 0.78 * rmax]
    anchor = max(band or peaks, key=lambda t: t[1])[0]
    k = anchor / R22
    pred46 = k * R46
    near46 = min(peaks, key=lambda t: abs(t[0] - pred46))
    err46 = near46[0] / k - R46
    linear_ok = abs(err46) <= 1.5
    print(f"scale   : 22 deg ring={anchor}px -> k={k:.3f} px/deg")
    print(f"linchk  : 46 deg err {err46:+.2f} deg -> "
          f"{'linear (valid)' if linear_ok else 'NON-LINEAR (suspect)'}")
    print("\n  faces                         wedge  pred  meas  resid  match")
    rows = []
    for faces, wedge, pdeg in TAPE:
        rp, _ = min(peaks, key=lambda t: abs(t[0] - k * pdeg))
        mdeg = rp / k
        resid = mdeg - pdeg
        ok = abs(resid) <= 1.0
        rows.append((faces, wedge, pdeg, mdeg, resid, ok))
        print(f"  {faces:<28} {wedge:5.1f} {pdeg:5.1f} {mdeg:5.1f} "
              f"{resid:+5.1f}  {'YES' if ok else 'no'}")
    matched = sum(1 for *_, ok in rows if ok)
    print(f"\nsummary : {matched}/{len(rows)} matched +/-1.0 deg; "
          f"linearity {'OK' if linear_ok else 'FAILED'}.")
    ov = img.copy()
    d = ImageDraw.Draw(ov)
    for faces, wedge, pdeg, *_ in rows:
        rpx = k * pdeg
        d.ellipse([cx - rpx, cy - rpx, cx + rpx, cy + rpx],
                  outline=(255, 80, 80), width=1)
    d.line([cx - 8, cy, cx + 8, cy], fill=(255, 255, 0), width=1)
    d.line([cx, cy - 8, cx, cy + 8], fill=(255, 255, 0), width=1)
    ov.save(OVERLAY_OUT, "PNG", optimize=True)
    print(f"overlay : {os.path.relpath(OVERLAY_OUT, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
