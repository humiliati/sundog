#!/usr/bin/env python
"""H8-RF (Slate 3, S3-A2-H8RF) — is the regularization-removal of MODEL-WISE double descent a Whitney A3
cusp in the Mei-Montanari closed-form RANDOM-FEATURES ridge risk? The owed substrate upgrade named by the
banked isotropic null (docs/atlas/H8_DOUBLE_DESCENT_CUSP_RESULT.md): the RF model structurally possesses
the classical-regime min the isotropic model provably lacked, so merge-vs-escape is genuinely open here.

Pre-registration: docs/atlas/H8_RF_CUSP_PREREG.md (FROZEN before the first full-grid run — all slices,
grids, windows, thresholds, gates, and the kill lattice are pinned there; this file implements it).

Closed form (arXiv:1908.05355, Definition 1 + Theorem 2, linear target):
    R_RF(psi1; psi2, lbar, tau2) = F1^2 * B(zeta,psi1,psi2,lbar) + tau2 * V(zeta,psi1,psi2,lbar)
with lbar = lambda/mu_star^2, zeta = mu1/mu_star (Assumption 1 Eq. 9 PINNED — the intro Eq. 4 prints
zeta=mu1^2/mu_star^2, a named paper-internal discrepancy; the MC gate arbitrates), and B=E1/E0, V=E2/E0
polynomials in chi = nu1*nu2 at xi = i*sqrt(psi1*psi2*lbar). On the imaginary axis the C+ branch reduces
to nu_j = i*b_j with b_j > 0 real and chi = -b1*b2 < 0 — a positive damped fixed point (de-fiddled).

NOT public-eligible. Attribution: Mei-Montanari 1908.05355 (closed form); Nakkiran et al. 2003.01897
(regularization removal); Thom/Whitney (cusp); the Atlas jet classifier (atlas_jet_classify.py).
Run: python scripts/double_descent_cusp_rf.py
"""
import sys
from pathlib import Path
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---- pinned activation constants (ReLU; prereg section 2) ----------------------------------------- #
MU0 = 1.0 / np.sqrt(2.0 * np.pi)
MU1 = 0.5
MUS2 = 0.25 - 1.0 / (2.0 * np.pi)          # mu_star^2
Z2 = MU1 ** 2 / MUS2                       # zeta^2  (Eq. 9 pinned: zeta = mu1/mu_star)

# ---- pinned grids / windows (prereg section 3) ----------------------------------------------------- #
PSI2 = 10.0
SLICES = (0.4, 0.8, 1.2, 1.6, 2.4)         # tau^2 locus slices
PRIMARY = 0.8                              # headline slice
LBAR_FLOOR = 0.02                          # pole-recast classification floor
LBAR_C_MIN = 0.0286                        # slice validity: approach window must stay >= floor
CEN_LO, CEN_HI, CEN_N = 0.25, 60.0, 12000  # census grid (one-shot 4x tie-break: hi -> 240)
NG = 260                                   # chart grid (escalation: 520)
CHART_FACTORS = (0.4, 2.5, 0.5, 1.6)       # psi1 in [f0*psi1c, f1*psi1c]; lbar in [max(floor,f2*lc), f3*lc]
RESID_TOL = 1e-10                          # nu fixed-point residual (ABORT-B if exceeded)


def nu_chi(psi1, psi2, lbar, z2=Z2, tol=1e-14, maxit=200000):
    """chi = nu1*nu2 at xi = i*sqrt(psi1*psi2*lbar), C+ branch: nu_j = i*b_j, b_j>0, chi=-b1*b2.
    Vectorized over psi1 (array ok). Returns (chi, max_residual)."""
    psi1 = np.asarray(psi1, float)
    s = np.sqrt(psi1 * psi2 * lbar)
    b1, b2 = psi1 / s, np.full_like(psi1, psi2) / s
    for _ in range(maxit):
        den = 1.0 + z2 * b1 * b2
        nb1 = psi1 / (s + b2 + z2 * b2 / den)
        nb2 = psi2 / (s + b1 + z2 * b1 / den)
        if (np.all(np.abs(nb1 - b1) <= tol * np.maximum(1.0, np.abs(nb1)))
                and np.all(np.abs(nb2 - b2) <= tol * np.maximum(1.0, np.abs(nb2)))):
            b1, b2 = nb1, nb2
            break
        b1, b2 = 0.5 * (b1 + nb1), 0.5 * (b2 + nb2)
    den = 1.0 + z2 * b1 * b2
    r1 = np.abs(b1 * (s + b2 + z2 * b2 / den) - psi1) / np.maximum(psi1, 1e-30)
    r2 = np.abs(b2 * (s + b1 + z2 * b1 / den) - psi2) / psi2
    return -(b1 * b2), float(max(np.max(r1), np.max(r2)))


def risk_rf(psi1, psi2, lbar, tau2, F12=1.0, z2=Z2):
    """Closed-form RF excess test risk (Theorem 2, F_star=0). Vectorized over psi1."""
    psi1 = np.asarray(psi1, float)
    chi, resid = nu_chi(psi1, psi2, lbar, z2)
    z4, z6 = z2 * z2, z2 * z2 * z2
    E0 = (-chi ** 5 * z6 + 3 * chi ** 4 * z4 + (psi1 * psi2 - psi2 - psi1 + 1) * chi ** 3 * z6
          - 2 * chi ** 3 * z4 - 3 * chi ** 3 * z2
          + (psi1 + psi2 - 3 * psi1 * psi2 + 1) * chi ** 2 * z4
          + 2 * chi ** 2 * z2 + chi ** 2 + 3 * psi1 * psi2 * chi * z2 - psi1 * psi2)
    E1 = psi2 * chi ** 3 * z4 - psi2 * chi ** 2 * z2 + psi1 * psi2 * chi * z2 - psi1 * psi2
    E2 = (chi ** 5 * z6 - 3 * chi ** 4 * z4 + (psi1 - 1) * chi ** 3 * z6 + 2 * chi ** 3 * z4
          + 3 * chi ** 3 * z2 + (-psi1 - 1) * chi ** 2 * z4 - 2 * chi ** 2 * z2 - chi ** 2)
    return F12 * (E1 / E0) + tau2 * (E2 / E0), resid


def census(psi2, tau2, lbar, lo=CEN_LO, hi=CEN_HI, n=CEN_N):
    """Interior critical points of R(psi1) at fixed (psi2, tau2, lbar): sign changes of the finite-
    difference dR/dpsi1, edges excluded 3 cells. Returns (positions, types, max_residual)."""
    ps = np.linspace(lo, hi, n)
    R, resid = risk_rf(ps, psi2, lbar, tau2)
    dR = np.gradient(R, ps)
    sc = np.where(np.abs(np.diff(np.sign(dR))) > 0)[0]
    sc = sc[(sc > 3) & (sc < n - 4)]
    return [float(ps[i]) for i in sc], ["min" if dR[i] < 0 else "max" for i in sc], resid


# ---- the generic fold-pair tracker (SHARED code path: headline + tol_cal control; prereg s7) ------- #
def bisect_drop(count_at, c_lo, c_hi, count_low, rel=1e-8):
    """Bisect the control value where the critical-point count drops from count_low to count_low-2.
    count_at(c) -> int. Requires count_at(c_lo) >= count_low and count_at(c_hi) <= count_low-2."""
    lo, hi = float(c_lo), float(c_hi)
    while (hi - lo) > rel * hi:
        mid = 0.5 * (lo + hi)
        if count_at(mid) >= count_low:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def pair_separation(crit, count_low):
    """Separation of the annihilating pair: count_low==2 -> the only pair; else the closest adjacent pair."""
    xs = sorted(crit)
    if len(xs) < 2:
        return float("nan")
    if count_low == 2:
        return xs[-1] - xs[0]
    return float(min(np.diff(xs)))


def scaling_fit(crit_at, c_c, count_low, n_pts=12, eps_lo=0.02, eps_hi=0.3):
    """log-log fit of pair separation vs eps=(c_c - c)/c_c over the pinned window. Returns (slope, R2, pts)."""
    eps = np.geomspace(eps_lo, eps_hi, n_pts)
    seps = []
    for e in eps:
        crit = crit_at(c_c * (1.0 - e))
        seps.append(pair_separation(crit, count_low))
    seps = np.array(seps)
    okm = np.isfinite(seps) & (seps > 0)
    x, y = np.log(eps[okm]), np.log(seps[okm])
    if okm.sum() < n_pts:
        return float("nan"), 0.0, int(okm.sum())
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(coef[0]), 1.0 - ss_res / max(ss_tot, 1e-30), int(okm.sum())


def quartic_control():
    """tol_cal calibration family V(x;h)=x^4+h*x^2+0.15*x (exact 1/2 law), run through the SAME census
    (sign changes of dV/dx on a 12000-grid) + SAME bisect_drop + SAME scaling_fit code path."""
    xs = np.linspace(-2.0, 2.0, 12000)

    def crit_at(h):
        d = 4 * xs ** 3 + 2 * h * xs + 0.15
        sc = np.where(np.abs(np.diff(np.sign(d))) > 0)[0]
        sc = sc[(sc > 3) & (sc < len(xs) - 4)]
        return [float(xs[i]) for i in sc]

    h_c = bisect_drop(lambda h: len(crit_at(h)), -2.0, -0.05, 3)
    # control approaches its h_c from BELOW (3 critical points side): eps measured toward more-negative h
    def crit_at_signed(c):
        return crit_at(c)
    # map: c = h_c*(1-eps) walks AWAY from merge for h_c<0 ... use the same scaling_fit with c_c=h_c and
    # crit_at on c values h = h_c - |h_c|*eps  (the 3-root side), via a wrapper preserving the code path:
    def crit_wrap(c):
        # scaling_fit calls crit_at(c_c*(1-e)); with c_c=|h_c| we map u=|h_c|(1-e) -> h = -u... NO:
        # keep it literal: h = h_c*(1+e) is the 3-root side for h_c<0. Implemented by passing c_c=-|h_c|
        # and crit_at(h) directly: c_c*(1-e) = h_c*(1-e) is the 1-root side. So wrap with reflection:
        return crit_at(2 * h_c - c)          # reflect around h_c: h_c*(1-e) -> h_c*(1+e)
    slope, r2, npts = scaling_fit(crit_wrap, h_c, 3)
    return h_c, slope, r2, npts


# ---- chart + classification (banked construction; SHARED normalization, prereg s3) ----------------- #
def rf_chart(psi2, tau2, p_span, l_span, norm, ng=NG):
    """F(psi1,lbar)=(lbar_norm, R_norm) on an ng x ng grid (axis0=psi1, axis1=lbar), normalized with the
    SHARED constants norm=(m_ref, s_ref, lo_ref, hi_ref). Mirrors double_descent_cusp.cusp_chart."""
    import atlas_jet_classify as jc
    pp = np.linspace(*p_span, ng); ll = np.linspace(*l_span, ng)
    R = np.empty((ng, ng)); worst = 0.0
    for j, lb in enumerate(ll):
        R[:, j], resid = risk_rf(pp, psi2, lb, tau2)
        worst = max(worst, resid)
    m_ref, s_ref, lo_ref, hi_ref = norm
    L = np.broadcast_to(ll, (ng, ng))
    Xn = (L - lo_ref) / (hi_ref - lo_ref)
    Yn = (R - m_ref) / s_ref
    d = 1.0 / (ng - 1)
    phi, c2, c3 = jc.jet_from_chart(Xn, Yn, d, d)
    cusps = jc.cusp_c3(phi, c2, c3, edge=4)
    rank = jc.corank_from_chart(Xn, Yn, d, d, edge=4)
    has_caustic = bool(np.any(np.abs(np.diff(np.sign(phi), axis=0)) > 0))
    return {"caustic": has_caustic, "n_cusps": len(cusps), "c3": [round(v, 4) for v in cusps[:5]],
            "corank": (rank or {}).get("corank"), "s1_min_rel": round((rank or {}).get("s1_min_rel", float("nan")), 4),
            "resid": worst, "R_mean": float(R.mean()), "R_std": float(R.std())}


def slice_windows(tau2, lbar_lo=LBAR_FLOOR, lbar_hi=12.0):
    """Per-slice measurement: lbar_c (bisected), merge psi1*, chart spans per the pinned rules."""
    def count_at(lb):
        crit, _, _ = census(PSI2, tau2, lb)
        return len(crit)
    if count_at(lbar_lo) < 2:
        return None                                            # no pair at floor: excluded-with-report
    lbar_c = bisect_drop(count_at, lbar_lo, lbar_hi, 2)
    crit, _, _ = census(PSI2, tau2, lbar_c * (1.0 - 1e-6))
    psi1c = float(np.mean(crit)) if len(crit) >= 2 else float("nan")
    f0, f1, f2, f3 = CHART_FACTORS
    p_span = (f0 * psi1c, f1 * psi1c)
    l_span = (max(LBAR_FLOOR, f2 * lbar_c), f3 * lbar_c)
    return {"tau2": tau2, "lbar_c": lbar_c, "psi1c": psi1c, "p_span": p_span, "l_span": l_span}


# ---- Monte-Carlo gate (Eq. 2 estimator, exact normalization; prereg s4) ---------------------------- #
def sphere(rng, m, d):
    g = rng.standard_normal((m, d))
    return g * (np.sqrt(d) / np.linalg.norm(g, axis=1, keepdims=True))


def risk_mc(psi1, psi2, lbar, tau2, d, reps=24, n_test=4000, seed=2026):
    """MC test error of RF ridge (Eq. 2: (1/n)||y-Za||^2 + (N*lam/d)||a||^2, lam = lbar*mu_star^2)."""
    n = int(round(psi2 * d)); N = max(1, int(round(psi1 * d)))
    lam = lbar * MUS2
    acc = 0.0
    for r in range(reps):
        rng = np.random.default_rng(seed + 7919 * r)
        beta = rng.standard_normal(d); beta /= np.linalg.norm(beta)        # F1 = 1
        X = sphere(rng, n, d); TH = sphere(rng, N, d)
        Z = np.maximum(X @ TH.T / np.sqrt(d), 0.0)
        y = X @ beta + rng.standard_normal(n) * np.sqrt(tau2)
        c = N * lam / d
        if N <= n:
            a = np.linalg.solve(Z.T @ Z / n + c * np.eye(N), Z.T @ y / n)
        else:
            a = Z.T @ np.linalg.solve(Z @ Z.T / n + c * np.eye(n), y) / n
        Xt = sphere(rng, n_test, d)
        Zt = np.maximum(Xt @ TH.T / np.sqrt(d), 0.0)
        acc += float(np.mean((Xt @ beta - Zt @ a) ** 2))
    return acc / reps


MC_CELLS = (
    # (psi2, d, tau2, psi1, lbar)  — 6 published-geometry cells + 4 primary-slice cells (prereg s4)
    (3.0, 100, 0.5, 0.5, 0.11), (3.0, 100, 0.5, 0.5, 1.1),
    (3.0, 100, 0.5, 6.0, 0.11), (3.0, 100, 0.5, 6.0, 1.1),
    (3.0, 100, 0.5, 1.0, 0.55), (3.0, 100, 0.5, 12.0, 0.55),
    (10.0, 60, 0.8, 3.0, 0.06), (10.0, 60, 0.8, 3.0, 0.10),
    (10.0, 60, 0.8, 25.0, 0.06), (10.0, 60, 0.8, 25.0, 0.10),
)


def main():
    import atlas_jet_classify as jc
    verdict_token = None
    print("=" * 100)
    print("H8-RF — double-descent removal in the Mei-Montanari RF closed form: Whitney A3 cusp or second null?")
    print(f"  prereg: docs/atlas/H8_RF_CUSP_PREREG.md | zeta^2={Z2:.6f} (Eq.9 pinned), mu_star^2={MUS2:.7f}")
    print("=" * 100)

    # ---------------- GATE 1: ABORT-A-APPARATUS — MC validation ---------------------------------- #
    print("\n[GATE 1] MC validation (Eq. 2 estimator; 24 instances/cell; threshold rel err < 12%):")
    print(f"    {'psi2':>5} {'d':>4} {'tau2':>5} {'psi1':>6} {'lbar':>6} {'R closed':>10} {'R MC':>10} {'rel':>8}")
    mc_ok = True
    mc_rows = []
    for (p2, d, t2, p1, lb) in MC_CELLS:
        rc = float(risk_rf(np.array([p1]), p2, lb, t2)[0][0])
        rm = risk_mc(p1, p2, lb, t2, d)
        rel = abs(rc - rm) / max(rm, 1e-12)
        mc_ok &= rel < 0.12
        mc_rows.append((p1, lb, rc, rm, rel))
        print(f"    {p2:>5.0f} {d:>4d} {t2:>5.2f} {p1:>6.2f} {lb:>6.3f} {rc:>10.4f} {rm:>10.4f} {rel:>8.2%}"
              + ("" if rel < 0.12 else "  <-- FAIL"))

    # ---------------- GATE 1b: reference-curve invariants (Fig 3-left + Fig 1) -------------------- #
    print("\n[GATE 1b] published reference-curve invariants:")
    c_small, t_small, _ = census(10.0, 0.2, 0.02)
    maxes = [p for p, t in zip(c_small, t_small) if t == "max"]
    g_a = any(2.0 < p < 20.0 for p in maxes)
    ps = np.linspace(CEN_LO, CEN_HI, CEN_N)
    R3, _ = risk_rf(ps, 10.0, 3.0, 0.2)
    g_b = (len(census(10.0, 0.2, 3.0)[0]) == 0) and bool(np.all(np.gradient(R3, ps) < 0))
    g_c = abs(float(risk_rf(np.array([0.01]), 10.0, 0.1, 0.2)[0][0]) - 1.0) < 0.02
    c_f1, t_f1, _ = census(3.0, 0.0, 0.011, lo=0.25, hi=12.0, n=6000)
    g_d = any(1.0 < p < 6.0 for p, t in zip(c_f1, t_f1) if t == "max")
    for nm, g in (("Fig3: interior max at lbar=0.02 (tau2=0.2, psi1 in (2,20))", g_a),
                  ("Fig3: monotone-decreasing, 0 critical pts at lbar=3", g_b),
                  ("R(psi1->0) -> F1^2 within 2%", g_c),
                  ("Fig1: interior max near psi1~psi2=3 (tau2=0, lbar~0.011; apparatus tier)", g_d)):
        print(f"    [{'PASS' if g else 'FAIL'}] {nm}")
    ref_ok = g_a and g_b and g_c and g_d
    if not (mc_ok and ref_ok):
        print("\nVERDICT: ABORT-A-APPARATUS — closed-form transcription or solver fails its external gates;")
        print("         fix the apparatus (transcription amendment logged in the prereg) before ANY result claim.")
        return 2

    # ---------------- GATE 2: ABORT-B — in-context instrument controls --------------------------- #
    print("\n[GATE 2] in-context controls (matched grid; any out-of-band => ABORT-B, no verdict):")
    a4_gen = jc.cusp_c3(*jc.synthetic_swallowtail(-0.40))[0]
    a4_near_l = jc.cusp_c3(*jc.synthetic_swallowtail(-0.02))
    a4_near = a4_near_l[0] if a4_near_l else float("nan")
    a4_ratio = a4_near / a4_gen
    col = jc.cusp_c3(*jc.halo_chart(30.0, "prism60")[:3], jc.halo_chart(30.0, "prism60")[3])
    col_top = col[0] if col else float("nan")
    Xu, Yu, du = jc.synthetic_umbilic(0.0); ru = jc.corank_from_chart(Xu, Yu, du, du)
    Xs, Ys, ds = jc.synthetic_swallowtail_chart(0.0); rs = jc.corank_from_chart(Xs, Ys, ds, ds)
    import double_descent_cusp as ddi
    iso = ddi.cusp_chart((1.05, 4.0), (0.05, 0.6))
    ctl = {
        "Morin A4 dive (near/generic < 0.25)": a4_ratio < 0.25,
        "column A3 bounded (top |c3| > 2.0)": col_top > 2.0,
        "synthetic D4 umbilic w=0 -> corank-2": ru["corank"] == 2,
        "A4 swallowtail chart h=0 -> corank-1": rs["corank"] == 1,
        "banked isotropic chart -> caustic, 0 cusps": iso["caustic"] and iso["n_cusps"] == 0,
    }
    for nm, g in ctl.items():
        print(f"    [{'PASS' if g else 'FAIL'}] {nm}")
    print(f"      (A4 near/gen = {a4_near:.3f}/{a4_gen:.3f} = {a4_ratio:.3f}; column top|c3| = {col_top:.3f};"
          f" umbilic s1_min_rel={ru['s1_min_rel']:.4f}; iso n_cusps={iso['n_cusps']})")
    h_c_ctl, slope_ctl, r2_ctl, np_ctl = quartic_control()
    tol_cal = max(0.05, 3.0 * abs(slope_ctl - 0.5))
    ctl_scale_ok = abs(slope_ctl - 0.5) <= 0.05 and r2_ctl >= 0.99
    print(f"    [{'PASS' if ctl_scale_ok else 'FAIL'}] scaling-control (quartic exact 1/2 law, SAME code path):"
          f" h_c={h_c_ctl:.6f} slope={slope_ctl:.4f} R2={r2_ctl:.5f} ({np_ctl} pts) -> tol_cal={tol_cal:.3f}")
    if not (all(ctl.values()) and ctl_scale_ok):
        print("\nVERDICT: ABORT-B — an in-context control is out of its frozen band; no verdict may be reported.")
        return 2

    # ---------------- STAGE 3: per-slice census, lbar_c locus, K1 diagnostics -------------------- #
    print("\n[STAGE 3] slice measurements (pair-at-floor, lbar_c bisection, merge point):")
    slices = []
    for t2 in SLICES:
        w = slice_windows(t2)
        if w is None:
            print(f"    tau2={t2}: NO pair at the lbar={LBAR_FLOOR} floor -> excluded-with-report")
            continue
        valid = w["lbar_c"] >= LBAR_C_MIN
        print(f"    tau2={t2}: lbar_c={w['lbar_c']:.6f}  psi1*={w['psi1c']:.4f}"
              f"  chart psi1 {w['p_span'][0]:.3f}..{w['p_span'][1]:.3f}, lbar {w['l_span'][0]:.4f}..{w['l_span'][1]:.4f}"
              + ("" if valid else f"  <-- lbar_c < {LBAR_C_MIN}: floor-clipped, excluded"))
        if valid:
            slices.append(w)
    prim = next((w for w in slices if w["tau2"] == PRIMARY), None)
    if prim is None:
        print("\nVERDICT: ABORT-A-RESULT -> K1 — the primary slice has no classifiable interior pair on the")
        print("         pre-registered grid (fallback PROHIBITED on this branch by the prereg). Clean null.")
        return 1

    # K1 diagnostics on the primary slice (approach window; grid-resolution guard eps>=0.005)
    print(f"\n[STAGE 3b] K1 diagnostics, primary slice tau2={PRIMARY} (approach window eps in [0.005,0.3]):")
    lc = prim["lbar_c"]
    eps_w = np.geomspace(0.005, 0.3, 24)
    k1_fired, k1_why = False, ""
    pmax_first = pmax_last = None
    dpsi = (CEN_HI - CEN_LO) / (CEN_N - 1)
    for e in eps_w[::-1]:                       # far -> near
        crit, types, _ = census(PSI2, PRIMARY, lc * (1.0 - e))
        if len(crit) != 2:
            k1_fired, k1_why = True, f"count={len(crit)} at eps={e:.4f} (K1c)"
            break
        if crit[0] < CEN_LO + 3 * dpsi or crit[-1] > CEN_HI - 3 * dpsi:
            k1_fired, k1_why = True, f"boundary exit at eps={e:.4f} (K1a)"
            break
        pmax = crit[types.index("max")] if "max" in types else float("nan")
        pmax_last = pmax
        if pmax_first is None:
            pmax_first = pmax
    if not k1_fired and pmax_first and pmax_last and pmax_last / pmax_first >= 2.0:
        # one-shot 4x domain-extension tie-break (prereg: the only allowed re-measurement)
        crit4, _, _ = census(PSI2, PRIMARY, lc * (1.0 - eps_w[0]), hi=240.0, n=CEN_N)
        if len(crit4) != 2:
            k1_fired, k1_why = True, "escape confirmed on the 4x extension (K1b)"
        else:
            print("      escape suspicion resolved by the one-shot 4x extension (pair interior)")
    print(f"      peak-psi1 across window: {pmax_first if pmax_first else float('nan'):.4f} -> "
          f"{pmax_last if pmax_last else float('nan'):.4f}; merge psi1*={prim['psi1c']:.4f} (interior)")
    if k1_fired:
        print(f"\nVERDICT: K1 — no finite-interior-point 2->0 annihilation ({k1_why}). The cusp claim is DEAD;")
        print("         banked as the SECOND mechanism-class null (isotropic = peak-escape; RF = this).")
        return 1
    print("      K1 does NOT fire: bracketed interior max+min pair annihilates 2->0 at finite "
          f"(psi1*, lbar_c) = ({prim['psi1c']:.4f}, {lc:.6f})")

    # ---------------- STAGE 4: chart classification + K4 stability + K2 -------------------------- #
    print("\n[STAGE 4] jet classification (shared normalization from the primary slice):")
    base = rf_chart(PSI2, PRIMARY, prim["p_span"], prim["l_span"], (0, 1, prim["l_span"][0], prim["l_span"][1]))
    norm = (base["R_mean"], base["R_std"], prim["l_span"][0], prim["l_span"][1])
    print(f"    shared normalization constants: m_ref={norm[0]:.6f} s_ref={norm[1]:.6f} "
          f"lbar window [{norm[2]:.4f}, {norm[3]:.4f}]")

    def classify_all(ng):
        out = []
        for w in slices:
            r = rf_chart(PSI2, w["tau2"], w["p_span"], w["l_span"], norm, ng=ng)
            out.append((w, r))
        return out

    res260 = classify_all(NG)
    for w, r in res260:
        print(f"    tau2={w['tau2']}: caustic={r['caustic']} #cusps={r['n_cusps']} |c3|={r['c3']} "
              f"corank={r['corank']} (s1_min_rel={r['s1_min_rel']}) resid={r['resid']:.1e}")
    if any(r["resid"] >= RESID_TOL for _, r in res260):
        print("\nVERDICT: ABORT-B — nu fixed-point residual out of tolerance on a chart grid.")
        return 2

    prim_r = next(r for w, r in res260 if w["tau2"] == PRIMARY)
    # K4: grid doubling + edge perturbations on the primary slice
    print("\n[STAGE 4b] K4 stability (primary slice): ng-doubling + 8 edge perturbations (+-10% per factor):")
    stable = True
    r520 = rf_chart(PSI2, PRIMARY, prim["p_span"], prim["l_span"], norm, ng=2 * NG)
    okd = r520["n_cusps"] >= 1 and r520["corank"] == 1 if prim_r["n_cusps"] >= 1 else r520["n_cusps"] == 0
    stable &= okd
    print(f"    ng=520: #cusps={r520['n_cusps']} corank={r520['corank']} [{'PASS' if okd else 'FAIL'}]")
    f = list(CHART_FACTORS)
    for i in range(4):
        for sgn in (+1, -1):
            ff = list(f); ff[i] = ff[i] * (1 + 0.1 * sgn)
            p_span = (ff[0] * prim["psi1c"], ff[1] * prim["psi1c"])
            l_span = (max(LBAR_FLOOR, ff[2] * prim["lbar_c"]), ff[3] * prim["lbar_c"])
            rp = rf_chart(PSI2, PRIMARY, p_span, l_span, norm, ng=NG)
            ok = (rp["n_cusps"] >= 1) == (prim_r["n_cusps"] >= 1)
            stable &= ok
            print(f"    factor[{i}] {'+' if sgn > 0 else '-'}10%: #cusps={rp['n_cusps']} [{'PASS' if ok else 'FAIL'}]")
    if not stable:
        print("\nVERDICT: K4 — the cusp call is not grid/edge-stable; VOID (no germ claim from this chart).")
        return 1

    # K2: cusp present? corank? |c3| locus adjudication
    if prim_r["n_cusps"] == 0:
        print("\nVERDICT: K2 — finite interior merge but NO cusp read grid-stably at it (wrong germ as claimed).")
        return 1
    if prim_r["corank"] != 1:
        print("\nVERDICT: K2 — corank-2 at the merge (umbilic class, not A3).")
        return 1
    valid_locus = [(w, r) for w, r in res260 if r["n_cusps"] >= 1]
    if len(valid_locus) >= 3:
        tops = sorted((r["c3"][0] for _, r in valid_locus), reverse=True)
        med = float(np.median([r["c3"][0] for _, r in valid_locus]))
        ratios = {w["tau2"]: r["c3"][0] / med for w, r in valid_locus}
        print(f"\n[STAGE 4c] |c3| locus adjudication: median={med:.4f}; ratios={ {k: round(v,3) for k,v in ratios.items()} }")
        rmin = min(ratios.values())
        if rmin < 0.5:
            print(f"    min ratio {rmin:.3f} < 0.5 -> mandated ng-doubling escalation on ALL slices:")
            res520 = classify_all(2 * NG)
            vl = [(w, r) for w, r in res520 if r["n_cusps"] >= 1]
            med2 = float(np.median([r["c3"][0] for _, r in vl])) if len(vl) >= 3 else float("nan")
            ratios = {w["tau2"]: r["c3"][0] / med2 for w, r in vl} if len(vl) >= 3 else {}
            rmin = min(ratios.values()) if ratios else float("nan")
            print(f"    after escalation: median={med2:.4f}; ratios={ {k: round(v,3) for k,v in ratios.items()} }")
            if not ratios or rmin < 0.5:
                lab = "A4-dive" if (ratios and rmin < 0.25) else "germ-indeterminate band"
                print(f"\nVERDICT: K2 — |c3| locus ratio {rmin if ratios else float('nan'):.3f} ({lab}); the A3")
                print("         claim is dead as stated; banked as 'finite merge of unresolved germ class'.")
                return 1
        c3_verdict = "locus-consistent (all ratios >= 0.5)"
    else:
        c3_verdict = "VOID (<3 locus slices) — A3 call rests on corank-1 + K3 alone; |c3| descriptive"
        print(f"\n[STAGE 4c] |c3| locus adjudication: {c3_verdict}")

    # ---------------- STAGE 5: K3 scaling battery (shared code path) ----------------------------- #
    print("\n[STAGE 5] K3 fold-pair scaling battery (primary slice; slope 0.5 +- "
          f"{tol_cal:.3f}, R2 >= 0.99, >=12 pts, eps in [0.02, 0.3]):")

    def crit_at(lb):
        return census(PSI2, PRIMARY, lb)[0]
    slope, r2, npts = scaling_fit(crit_at, lc, 2)
    k3_ok = abs(slope - 0.5) <= tol_cal and r2 >= 0.99 and npts >= 12
    print(f"    slope={slope:.4f}  R2={r2:.5f}  pts={npts}  [{'PASS' if k3_ok else 'FAIL'}]")
    if not k3_ok:
        print("\nVERDICT: K3 — the merge fails the A3 normal-form scaling; any K2-passing cusp call is")
        print("         RETRACTED (verdict: cusp-like merge failing the normal-form scaling).")
        return 1

    # ---------------- HEADLINE ------------------------------------------------------------------- #
    print("\n" + "=" * 100)
    print("VERDICT: A3-CONFIRMED (bounded-positive) — in the Mei-Montanari RF closed form the")
    print("  regularization-removal of model-wise double descent is a FOLD-PAIR ANNIHILATION terminating")
    print(f"  at finite interior (psi1*, lbar_c) = ({prim['psi1c']:.4f}, {lc:.6f}) [tau2={PRIMARY}], read as a")
    print(f"  corank-1 cusp by the calibrated jet classifier (|c3| {c3_verdict}), with the fold-pair")
    print(f"  separation obeying the A3 1/2-law (slope {slope:.4f}, R2 {r2:.5f}).")
    print("  Germ table: isotropic ridge = peak-escape NULL (banked 2026-06-08); RF = A3 + sqrt-law.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
