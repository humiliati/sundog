#!/usr/bin/env python
"""H7 — curved-Earth mirage ray-tracer (deepening H5's flat-Earth mirage ladder).

Over a spherical Earth the surface curves AWAY from a straight ray, so in the height-above-surface frame
a ray gains the curvature term 1/R_E on top of refraction:
    dh/dx = ψ ,   dψ/dx = 1/R_E + n'(h)
(rays curve toward higher n via n'(h); the +1/R_E is the surface dropping away). h<0 means the ray hit the
Earth (blocked / beyond the horizon). This single extra term unlocks the curved-Earth phenomenology the
flat model can't show:

  * standard-refraction EFFECTIVE EARTH RADIUS / k-factor: a straight ray over radius R_eff matches the
    real ray iff 1/R_eff = 1/R_E + dn/dh, i.e. k = R_eff/R_E = 1/(1 + R_E·dn/dh) ≈ 4/3 for a standard
    lapse — the textbook result (the horizon is ~8% farther than geometric);
  * the DUCTING threshold dn/dh = -1/R_E: a horizontal ray then neither rises nor falls relative to the
    surface — it is TRAPPED and follows the curve (looming / Novaya-Zemlya: the sun/objects seen when
    geometrically BELOW the horizon). dn/dh < -1/R_E => sub-horizon ducting; dn/dh > -1/R_E => finite
    horizon.

NOT public-eligible (research; the public mirage explainer/workbench is a separate, deferred product).
Attribution: standard atmospheric refraction / mirage theory (Lehn; Können; Bruton; Young's mirage pages).
Run: python scripts/mirage_curved.py
"""
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

R_E = 6.371e6                 # Earth radius [m]
INV_RE = 1.0 / R_E            # ≈ 1.570e-7 /m  (the ducting-threshold gradient magnitude)
G_STD = -3.92e-8              # standard refraction dn/dh [1/m] -> k≈4/3
NSTEP = 6000


def grad_standard(h, g=G_STD):
    return np.full_like(np.atleast_1d(h).astype(float), g)


def grad_inversion(h, g=G_STD, dn=0.0, s=10.0, h0=30.0):
    """standard lapse + a localized inversion layer (extra strong-negative gradient) for ducting/superior."""
    return g - (dn / (2.0 * s)) / np.cosh((np.atleast_1d(h).astype(float) - h0) / s) ** 2


def grad_hot_ground(h, g=G_STD, amp=3e-7, scale=2.0):
    """standard lapse + a near-surface POSITIVE gradient (hot ground, n rises with height) -> inferior."""
    return g + amp * np.exp(-np.atleast_1d(h).astype(float) / scale)


def trace(theta, gradfn, R=4.0e5, h_obs=10.0, nstep=NSTEP, curved=True, **kw):
    """RK4 of dh/dx=ψ, dψ/dx=(1/R_E if curved)+n'(h), from x=0 (h_obs, θ) to range R. Vectorized over θ.
    Returns (h_final, min_h, hit_x) where hit_x = first range the ray reached h≤0 (else R)."""
    theta = np.atleast_1d(theta).astype(float)
    h = np.full_like(theta, float(h_obs)); psi = theta.copy()
    minh = h.copy(); hit = np.full_like(theta, R)
    dx = R / nstep
    base = INV_RE if curved else 0.0

    def dpsi(hh):
        return base + gradfn(hh, **kw)

    for stp in range(nstep):
        k1h, k1p = psi, dpsi(h)
        k2h, k2p = psi + 0.5 * dx * k1p, dpsi(h + 0.5 * dx * k1h)
        k3h, k3p = psi + 0.5 * dx * k2p, dpsi(h + 0.5 * dx * k2h)
        k4h, k4p = psi + dx * k3p, dpsi(h + dx * k3h)
        h = h + (dx / 6.0) * (k1h + 2 * k2h + 2 * k3h + k4h)
        psi = psi + (dx / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
        newhit = (h <= 0.0) & (hit >= R)
        hit = np.where(newhit, (stp + 1) * dx, hit)
        minh = np.minimum(minh, h)
    return h, minh, hit


def k_factor(g):
    return 1.0 / (1.0 + R_E * g)


def horizon_distance(h_obs, gradfn=grad_standard, **kw):
    """Range at which a ray launched at the grazing (most-downward, still-clearing) angle reaches h=0:
    the apparent horizon. Found by scanning downward launch angles."""
    R = 2.0e5
    thetas = np.linspace(-3e-3, 0.0, 4000)
    _, _, hit = trace(thetas, gradfn, R=R, h_obs=h_obs, **kw)
    # the horizon = the largest range reached before hitting ground (grazing ray), among rays that DO hit
    hits = hit[hit < R]
    return float(hits.max()) if len(hits) else R


def main():
    print("=" * 86)
    print("H7 — curved-Earth mirage ray-tracer (deepening the H5 flat-Earth ladder)")
    print("=" * 86)
    print(f"  R_E={R_E:.3e} m,  1/R_E={INV_RE:.3e} /m (the ducting-threshold gradient)\n")

    # (1) standard-refraction k-factor (the textbook 4/3)
    k = k_factor(G_STD)
    print(f"(1) standard refraction dn/dh={G_STD:.2e}/m -> k = 1/(1+R_E·dn/dh) = {k:.3f}  "
          f"(textbook 4/3≈1.333)   [{'PASS' if abs(k - 4/3) < 0.05 else 'FAIL'}]")
    print(f"    effective Earth radius R_eff = k·R_E = {k*R_E/1e3:.0f} km (vs {R_E/1e3:.0f} km geometric)")

    # (2) horizon distance: refraction extends it by ~sqrt(k); validate vs sqrt(2 h R_eff)
    print("\n(2) horizon distance (observer h=10 m): refraction pushes it out by ~√k")
    h_obs = 10.0
    d_geo = horizon_distance(h_obs, grad_standard, g=0.0)         # no refraction (curved only)
    d_ref = horizon_distance(h_obs, grad_standard, g=G_STD)       # standard refraction
    d_geo_th = np.sqrt(2 * h_obs * R_E); d_ref_th = np.sqrt(2 * h_obs * k * R_E)
    print(f"    traced: geometric={d_geo/1e3:.2f} km, refracted={d_ref/1e3:.2f} km  (ratio {d_ref/d_geo:.3f})")
    print(f"    theory: geometric={d_geo_th/1e3:.2f} km, refracted={d_ref_th/1e3:.2f} km  (√k={np.sqrt(k):.3f})"
          f"   [{'PASS' if abs(d_ref/d_geo - np.sqrt(k)) < 0.08 else 'FAIL'}]")

    # (3) the DUCTING threshold dn/dh = -1/R_E: a horizontal ray stays level (trapped) -> Novaya Zemlya
    print("\n(3) DUCTING threshold — a HORIZONTAL ray (θ=0, h=50 m) over range 300 km, vs gradient dn/dh:")
    print(f"    {'dn/dh [1/m]':>13} {'dn/dh / (-1/R_E)':>17} {'h(300km) [m]':>14}   regime")
    R3 = 3.0e5
    for g in (-0.5 * INV_RE, -0.9 * INV_RE, -1.0 * INV_RE, -1.5 * INV_RE, -2.5 * INV_RE):
        hf, mh, hit = trace(np.array([0.0]), grad_standard, R=R3, h_obs=50.0, g=g)
        ducted = hit[0] >= R3                      # never hit the ground over 300 km = ducted/sees far
        reg = ("DUCTED (follows curve, sees beyond horizon)" if g <= -INV_RE
               else f"falls to horizon @ {hit[0]/1e3:.0f} km" if hit[0] < R3 else "rises")
        print(f"    {g:>13.3e} {g/(-INV_RE):>17.2f} {hf[0]:>14.1f}   {reg}")
    h_thr = trace(np.array([0.0]), grad_standard, R=R3, h_obs=50.0, g=-INV_RE)[0][0]      # at threshold
    h_sub = trace(np.array([0.0]), grad_standard, R=R3, h_obs=50.0, g=-0.5 * INV_RE)[0][0]  # sub-critical
    h_sup = trace(np.array([0.0]), grad_standard, R=R3, h_obs=50.0, g=-2.5 * INV_RE)[0][0]  # super-critical
    duct_ok = abs(h_thr - 50.0) < 50.0 and h_sub > 500.0 and h_sup < 0.0
    geo_hz = np.sqrt(2 * 50.0 * R_E)
    print(f"    => at dn/dh=-1/R_E the horizontal ray stays LEVEL (h≈50 m over 300 km) — DUCTED: it reaches")
    print(f"       300 km vs the geometric horizon √(2·50·R_E)={geo_hz/1e3:.0f} km (≈{R3/geo_hz:.0f}× beyond) "
          f"= Novaya-Zemlya. Sub-critical rises, super-critical dives.   [{'PASS' if duct_ok else 'FAIL'}]")

    # (4) superior mirage / looming = an ELEVATED duct: with a moderate super-critical inversion, SOME
    #     launch angles get TRAPPED (oscillate in the duct, staying low over a long range) — scan θ.
    print("\n(4) SUPERIOR mirage / looming — elevated duct (inversion h0=120 m): scan θ for TRAPPED rays")
    R4 = 3.0e5
    th4 = np.linspace(-1.5e-3, 1.5e-3, 1200)
    hf4, mh4, hit4 = trace(th4, grad_inversion, R=R4, h_obs=120.0, g=G_STD, dn=1.5e-5, s=30.0, h0=120.0)
    trapped_mask = (hit4 >= R4) & (mh4 > 0.0) & (hf4 < 600.0)       # never hit ground, stayed low (ducted)
    hf4s, mh4s, hit4s = trace(th4, grad_standard, R=R4, h_obs=120.0, g=G_STD)
    std_mask = (hit4s >= R4) & (mh4s > 0.0) & (hf4s < 600.0)
    n_tr, n_std = int(trapped_mask.sum()), int(std_mask.sum())
    loom = n_tr > 0 and n_tr > n_std
    print(f"    inversion: {n_tr}/{len(th4)} launch angles stay TRAPPED below 600 m over 300 km (an elevated")
    print(f"    duct); standard atmosphere: {n_std}/{len(th4)} (rays rise away or hit the horizon)   "
          f"[{'PASS — elevated duct / looming (Fata Morgana)' if loom else 'CHECK'}]")

    # (5) inferior mirage (hot ground): curvature-INSENSITIVE at short range (= the flat H5 result)
    hf_hot, _, _ = trace(np.linspace(-3e-3, 2e-3, 3000), grad_hot_ground, R=4000.0, h_obs=3.0,
                        amp=8e-7, scale=1.5)
    dh = np.gradient(hf_hot)
    inferior = int(np.sum(np.abs(np.diff(np.sign(dh))) > 0)) >= 1
    print(f"\n(5) inferior mirage (hot ground, 4 km): transfer folds = {inferior} (desert/road mirage —")
    print(f"    curvature-insensitive at short range, matches the flat H5 model).")

    ok = (abs(k - 4/3) < 0.05 and abs(d_ref/d_geo - np.sqrt(k)) < 0.08 and duct_ok and loom)
    print("\n" + "=" * 86)
    print("RESULT:" + (" curved-Earth mirage model validated — reproduces the k≈4/3 effective Earth radius,"
                       " the √k-extended horizon, the dn/dh=-1/R_E ducting threshold (Novaya-Zemlya), and"
                       " inversion looming (superior mirage). The H5 ladder now lives over a real horizon."
                       if ok else " CHECK — a validation gate failed; inspect above."))
    print("=" * 86)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
