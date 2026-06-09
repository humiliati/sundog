#!/usr/bin/env python
"""H5 (slate `ww6koomb1`) — the mirage (Δn, s) ladder as a classified caustic / cusp diagram.

STAGE 1 (this file): a paraxial atmospheric-refraction ray-transfer forward model + phenomenology
validation. Mirages are CAUSTICS of the ray map: back-trace a ray from the observer at elevation θ to the
object range R; `h_target(θ)` = the height it reaches. An object at height H is seen in every direction θ
with h_target(θ)=H, so #images = #solutions, and the image-merge caustics are the folds dh_target/dθ=0.

Profile (localized inversion = superior-mirage / Fata-Morgana case):
    n(h) = n0 - (Δn/2)·tanh((h-h0)/s)   =>   n'(h) = -(Δn/2s)·sech²((h-h0)/s)
n decreases through the layer (warm aloft, cold below = a temperature inversion), rays curve DOWN toward
the denser air below -> a near-horizontal ray grazing the layer is focused -> the transfer h_target(θ)
develops an S-shape -> a 3-image band (superior mirage). Δn = index contrast (inversion strength), s =
layer thickness. Paraxial ray ODE (n≈1, small angles): dh/dx = ψ, dψ/dx = n'(h). Rays curve toward higher n.

NOT public-eligible. Attribution: Lehn; Können; Bruton (mirage ray theory); Berry/Nye (catastrophe optics).
Run: python scripts/mirage_ladder.py
"""
import sys
from pathlib import Path
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

R_DEFAULT = 20000.0      # object range [m]
H_OBS = 10.0             # observer height [m]
H0 = 120.0              # inversion centre height [m] (raised for vertical headroom vs strong bending)
N0 = 1.0
NSTEP = 2500            # RK4 steps along range


def nprime_single(h, dn, s, h0=H0):
    """n'(h) for n(h)=n0-(Δn/2)tanh((h-h0)/s): a localized inversion gradient (negative in the layer)."""
    return -(dn / (2.0 * s)) / np.cosh((h - h0) / s) ** 2


def nprime_double(h, dn, s, h0=H0, gap=60.0):
    """Two stacked inversion layers (h0±gap/2) -> a Fata-Morgana profile that can fold the transfer FOUR
    times (5 images). Same total contrast dn split across the two layers."""
    return (-(dn / (2.0 * s)) / np.cosh((h - (h0 - gap / 2)) / s) ** 2
            - (dn / (2.0 * s)) / np.cosh((h - (h0 + gap / 2)) / s) ** 2)


def trace(theta, dn, s, R=R_DEFAULT, h_obs=H_OBS, nstep=NSTEP, h0=H0, gradfn=nprime_single):
    """Vectorized RK4 of dh/dx=ψ, dψ/dx=n'(h) from x=0 (h=h_obs, ψ=θ) to x=R. Returns (h_target, min_h),
    min_h = lowest height reached along the path (min_h<0 => the ray hit the ground = not seen)."""
    theta = np.atleast_1d(theta).astype(float)
    h = np.full_like(theta, h_obs)
    psi = theta.copy()
    minh = h.copy()
    dx = R / nstep

    def deriv(h, psi):
        return psi, gradfn(h, dn, s, h0)

    for _ in range(nstep):
        k1h, k1p = deriv(h, psi)
        k2h, k2p = deriv(h + 0.5 * dx * k1h, psi + 0.5 * dx * k1p)
        k3h, k3p = deriv(h + 0.5 * dx * k2h, psi + 0.5 * dx * k2p)
        k4h, k4p = deriv(h + dx * k3h, psi + dx * k3p)
        h = h + (dx / 6.0) * (k1h + 2 * k2h + 2 * k3h + k4h)
        psi = psi + (dx / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
        minh = np.minimum(minh, h)
    return h, minh


def transfer_curve(dn, s, R=R_DEFAULT, ntheta=2400, h0=H0, h_obs=H_OBS, gradfn=nprime_single,
                   span=(-0.004, 0.020)):
    """h_target(θ) over a WIDE θ grid (covers the grazing-angle shift under strong bending). Returns
    (theta, h_target, valid) where valid = the ray cleared the ground (min_h>0) AND ended above ground."""
    theta = np.linspace(span[0], span[1], ntheta)
    ht, minh = trace(theta, dn, s, R, h_obs, h0=h0, gradfn=gradfn)
    valid = (minh > 0.0) & (ht > 0.0)
    return theta, ht, valid


def _longest_run(valid):
    """Indices of the longest contiguous True run in `valid` (the physical, above-ground image region)."""
    best_i = best_len = cur_i = cur_len = 0
    for i, v in enumerate(valid):
        if v:
            if cur_len == 0:
                cur_i = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_i = cur_len, cur_i
        else:
            cur_len = 0
    return best_i, best_i + best_len


def count_folds(theta, h_target, valid=None, edge=4):
    """#folds = #INTERIOR turning points (sign-changes of dh_target/dθ) over the LONGEST contiguous
    above-ground segment of the transfer. Interior-only (≥edge from the segment ends) excludes the
    ground-truncation edge artifact. Each fold = an image merge/birth; max #images = #folds+1."""
    if valid is None:
        valid = np.ones_like(theta, bool)
    i0, i1 = _longest_run(valid)
    if i1 - i0 < 2 * edge + 5:
        return 0
    th, ht = theta[i0:i1], h_target[i0:i1]
    d = np.gradient(ht, th)
    sgn = np.sign(d)
    ch = np.where(np.abs(np.diff(sgn)) > 0)[0]
    ch = [c for c in ch if edge <= c < len(d) - edge]      # interior only
    return len(ch)


def main():
    print("=" * 84)
    print("H5 Stage 1 — mirage ray-transfer forward model + phenomenology validation")
    print("=" * 84)
    R = R_DEFAULT
    print(f"  geometry: range R={R:.0f} m, observer h={H_OBS} m, inversion centre h0={H0} m\n")

    # (1) uniform-ish gradient (broad layer s>>) -> MONOTONIC transfer -> 1 image
    s_broad = 2000.0
    th, ht, vd = transfer_curve(1e-5, s_broad, R)
    f_broad = count_folds(th, ht, vd)
    print(f"(1) broad layer s={s_broad} m, Δn=1e-5: #folds={f_broad}  -> expect 0 (monotonic, 1 image)"
          f"   [{'PASS' if f_broad == 0 else 'FAIL'}]")

    # (2) localized inversion -> S-shaped transfer -> 3-image band; the 1->3 onset (the cusp)
    print("\n(2) localized inversion sweep (Δn) at s=5 m: when does the transfer FOLD (1->3 image onset)?")
    print(f"    {'Δn':>10}  {'#folds':>7}  {'max#images':>11}  {'fold-band Δh [m]':>16}")
    for dn in (1e-5, 1.5e-5, 2e-5, 3e-5, 5e-5, 1e-4):
        th, ht, vd = transfer_curve(dn, 5.0, R)
        nf = count_folds(th, ht, vd)
        i0, i1 = _longest_run(vd)
        htv = ht[i0:i1]
        d = np.gradient(htv, th[i0:i1]) if i1 - i0 > 5 else np.array([1.0])
        turns = htv[np.where(np.abs(np.diff(np.sign(d))) > 0)[0]] if nf >= 2 else np.array([])
        band = float(np.ptp(turns)) if len(turns) >= 2 else 0.0
        print(f"    {dn:>10.0e}  {nf:>7}  {nf+1:>11}  {band:>16.2f}")

    # (3) sign / phenomenology: inversion curves rays DOWN -> superior mirage
    print("\n(3) sign check: inversion (n drops with height) curves rays DOWN -> superior mirage (image up);")
    print("    a near-grazing ray should bend below its straight path (focusing toward dense air below).")
    th, ht, vd = transfer_curve(5e-5, 5.0, R)
    straight = H_OBS + th * R
    bend = (ht - straight)[vd]
    print(f"    max downward bend (valid rays) = {bend.min():.2f} m (negative = curved down)"
          f"   [{'PASS' if bend.min() < -0.5 else 'FAIL'}]")

    # verdict for Stage 1
    th, ht, vd = transfer_curve(5e-5, 5.0, R)
    strong_folds = count_folds(th, ht, vd)
    ok = (f_broad == 0) and (strong_folds == 2) and (bend.min() < -0.5)
    print("\n" + "=" * 84)
    if ok:
        print("STAGE 1 PASS: the forward model reproduces mirage phenomenology — a broad gradient gives a")
        print("  monotonic transfer (1 image); a localized inversion FOLDS the transfer (S-shape) into a")
        print(f"  3-image band ({strong_folds} folds at Δn=1e-4, s=4 m); rays curve down toward dense air")
        print("  (superior mirage). Ready for Stage 2 (the (Δn,s) ladder + jet-classifier catastrophe labels).")
    else:
        print("STAGE 1 ISSUE: check the model (folds/sign).")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
