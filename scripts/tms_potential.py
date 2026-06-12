#!/usr/bin/env python
"""S3-A5 — the TMS potential (arXiv:2310.06301 v1, Lemma 3.1) + importance deformation + gates.

PREREG (binding, frozen 2026-06-12 BEFORE this file first ran): docs/atlas/TMS_KGON_GERM_PREREG.md.
H0 harvest receipt: docs/atlas/TMS_H0_HARVEST.md (closed-form transcription; source archived at
internal/harvests/slttms_2310.06301_v1.tex).

CONTENTS
  - H_cart(W, b, I): the published closed form H (Lemma 3.1 / eq. (4)), torch.float64 so autograd
    supplies analytic gradients/Hessians for the classification machinery. L = H/(3c).
    RULE-2 DEVIATION (declared in the prereg header): the importance weight I is LAB-DERIVED
    (Elhage et al. 2022's concept applied to the paper's eq.-(2) one-hot distribution); H_I is the
    exact linear reweighting sum_i I_i * H_i of the PUBLISHED per-output pieces; I=1 ==> H exactly.
  - loss_quadrature(W, b, I): the INDEPENDENT reference path — direct integration of the raw ReLU
    integrand of L(w) under eq. (2), kink-split per piece (piecewise-quadratic integrand =>
    Gauss-Legendre exact per piece). Never shares code with H_cart.
  - polar_to_cart / kgon: the paper's (l, theta, b) parametrization (O(2) quotient) + standard
    k-gon and k^{sigma+} constructors (b+ = 1/(2c)).
  - kgon_solve(k): the published polynomial system (Appendix E.1) for (l*, b*); root-finding by
    dense sign-scan + Newton polish (deterministic, multi-start-free).
  - main(): the K1 / V0 / V1 gates per prereg section 2. Exit nonzero on any gate failure.

Run: python scripts/tms_potential.py
"""
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 console robustness

import torch

torch.set_default_dtype(torch.float64)

C_DEFAULT = 6
R = 2

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


# ===== the published closed form (Lemma 3.1 / eq. (4)) — torch path ========== #
def H_pieces(W, b):
    """Per-output-feature pieces H_i (i = output coordinate), shape (c,). W: (2,c) tensor; b: (c,).
    Transcribed from Lemma 3.1: H = sum_i [ d(b_i<=0) Hm_i + d(b_i>0) Hp_i ],
      Hm_i = sum_{j!=i} d(P_ij) (a_ij+b_i)^3/a_ij  +  d(P_i)[b^3/l^4 + b^3/l^2 + N_i] + (1-d(P_i)),
      Hp_i = sum_{j!=i} [ d(Q_ij)(-b_i^3/a_ij) + (1-d(Q_ij))(a_ij^2 + 3 a_ij b_i + 3 b_i^2) ] + N_i,
      N_i  = (1-l_i^2)^2 - 3(1-l_i^2) b_i + 3 b_i^2,
    with a_ij = W_i . W_j, l_i^2 = ||W_i||^2,
      P_ij = {a_ij > 0  and  -a_ij <= b_i <= 0},   P_i = {l_i^2 > 0 and -l_i^2 <= b_i <= 0},
      Q_ij = {-a_ij > b_i > 0}."""
    A = W.T @ W                                   # (c,c); a_ij = A[i,j]; l_i^2 = A[i,i]
    c = A.shape[0]
    l2 = torch.diagonal(A)
    bi = b.unsqueeze(1)                           # b_i broadcast over j
    off = ~torch.eye(c, dtype=torch.bool)

    # ---- b_i <= 0 branch ---- #
    P_ij = (A > 0) & (bi >= -A) & (bi <= 0) & off
    a_safe = torch.where(P_ij, A, torch.ones_like(A))
    cross_m = torch.where(P_ij, (A + bi) ** 3 / a_safe, torch.zeros_like(A))
    sum_cross_m = cross_m.sum(dim=1)
    P_i = (l2 > 0) & (b >= -l2) & (b <= 0)
    l2_safe = torch.where(P_i, l2, torch.ones_like(l2))
    N = (1 - l2) ** 2 - 3 * (1 - l2) * b + 3 * b ** 2
    self_m = torch.where(P_i, b ** 3 / l2_safe ** 2 + b ** 3 / l2_safe + N,
                         torch.ones_like(b))
    Hm = sum_cross_m + self_m

    # ---- b_i > 0 branch ---- #
    Q_ij = (bi > 0) & (-A > bi) & off             # bi>0 INSIDE the mask: keeps the discarded-branch
                                                  # forward finite (A=0 rows) so autograd stays NaN-free
    aq_safe = torch.where(Q_ij, A, -torch.ones_like(A))
    cross_p = torch.where(Q_ij, -bi ** 3 / aq_safe,
                          A ** 2 + 3 * A * bi + 3 * bi ** 2)
    cross_p = torch.where(off, cross_p, torch.zeros_like(cross_p))
    Hp = cross_p.sum(dim=1) + N

    return torch.where(b <= 0, Hm, Hp)


def H_cart(W, b, I=None):
    """The TMS potential H (published; L = H/(3c)). With importance I (c,): H_I = sum_i I_i H_i —
    the rule-2 lab deformation (exact at I=1)."""
    pieces = H_pieces(W, b)
    if I is None:
        return pieces.sum()
    return (torch.as_tensor(I, dtype=torch.float64) * pieces).sum()


def L_cart(W, b, I=None):
    c = W.shape[1]
    return H_cart(W, b, I) / (3 * c)


# ===== the independent quadrature reference (V0/V1) ========================== #
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)     # exact for deg<=15 per smooth piece


def _int_piece(f2_coeffs, lo, hi):
    """Integral over [lo,hi] of the quadratic with coeffs (c0,c1,c2) in mu."""
    if hi <= lo:
        return 0.0
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    mu = mid + half * _GL_X
    c0, c1, c2 = f2_coeffs
    return float(np.sum(_GL_W * (c0 + c1 * mu + c2 * mu * mu)) * half)


def loss_quadrature(W, b, I=None):
    """Direct integration of L(w) = (1/c) sum_i int_0^1 sum_j I_j (mu d_ij - ReLU(a_ji mu + b_j))^2 dmu
    under eq. (2) (one-hot input mu e_i). Kink-split: each term is piecewise quadratic in mu with a
    single kink at mu* = -b_j / a_ji. NEVER calls the closed form."""
    W = np.asarray(W, float)
    b = np.asarray(b, float)
    c = W.shape[1]
    Iv = np.ones(c) if I is None else np.asarray(I, float)
    A = W.T @ W
    total = 0.0
    for i in range(c):                                # input feature
        for j in range(c):                            # output coordinate
            a = A[j, i]
            bj = b[j]
            d = 1.0 if i == j else 0.0
            # regions where a*mu + bj >= 0 on [0,1]
            if a == 0.0:
                segs = [((0.0, 1.0), bj >= 0)]
            else:
                mu0 = -bj / a
                if a > 0:
                    segs = [((0.0, min(max(mu0, 0.0), 1.0)), False),
                            ((min(max(mu0, 0.0), 1.0), 1.0), True)]
                else:
                    segs = [((0.0, min(max(mu0, 0.0), 1.0)), True),
                            ((min(max(mu0, 0.0), 1.0), 1.0), False)]
            for (lo, hi), active in segs:
                if hi <= lo:
                    continue
                if active:
                    # (d*mu - (a*mu+bj))^2 = ((d-a)mu - bj)^2
                    p, q = d - a, -bj
                    coeffs = (q * q, 2 * p * q, p * p)
                else:
                    coeffs = (0.0, 0.0, d)            # (d*mu)^2 = d*mu^2 (d in {0,1})
                total += Iv[j] * _int_piece(coeffs, lo, hi)
    return total / c


# ===== polar parametrization + k-gon constructors ============================ #
def polar_to_cart(l, theta, b):
    """W columns at cumulative angles phi_i = sum_{m<i} theta_m (phi_1 = 0; O(2) pinned).
    l, theta, b: torch tensors (c,), (c,), (c,); theta sums to 2*pi (enforced by caller)."""
    phi = torch.cumsum(theta, 0) - theta          # exclusive cumsum: phi_i = sum_{m<i} theta_m
    W = torch.stack([l * torch.cos(phi), l * torch.sin(phi)])
    return W, b


def kgon(k, c, l_star, b_star, sigma=0, b_dead=-0.5):
    """Standard k-gon (k live features at angles 2*pi/k, dead features l=0) with `sigma` of the dead
    biases at the published optimal +1/(2c) (the k^{sigma+}-gon)."""
    l = np.zeros(c)
    l[:k] = l_star
    theta = np.zeros(c)
    theta[:k - 1] = 2 * np.pi / k
    theta[k - 1:] = (2 * np.pi - (k - 1) * (2 * np.pi / k)) / (c - k + 1)
    b = np.full(c, float(b_dead))
    b[:k] = b_star
    for s in range(sigma):
        b[k + s] = 1.0 / (2 * c)
    return (torch.tensor(l), torch.tensor(theta), torch.tensor(b))


def H_polar(l, theta, b, I=None):
    W, bb = polar_to_cart(l, theta, b)
    return H_cart(W, bb, I)


# ===== the published polynomial system (Appendix E.1) ======================== #
def _s_of_k(k):
    """The unique integer in [k/4 - 1, k/4)."""
    import math
    lo = k / 4.0 - 1.0
    s = math.ceil(lo)
    if s == lo:                                   # half-open: s in [lo, k/4)
        pass
    if not (lo <= s < k / 4.0):
        s += 1
    return int(s)


def _FGHM(k):
    s = _s_of_k(k)
    al = 2 * np.pi / k
    j = np.arange(1, s + 1)
    F = 1 + 2 * np.sum(np.cos(j * al) ** 2)
    G = 1 + 2 * np.sum(np.cos(j * al))
    Hc = 1 + 2 * np.sum(1.0 / np.cos(j * al))
    M = 1 + 2 * s
    return F, G, Hc, M, s, al


def _poly_system(xy, k):
    x, y = xy
    F, G, Hc, M, _, _ = _FGHM(k)
    e1 = x ** 6 * G + x ** 2 * y ** 2 * Hc + 2 * x ** 4 * y * M + y ** 2 - x ** 4
    e2 = 2 * x ** 8 * F + 3 * x ** 6 * y * G - x ** 2 * y ** 3 * Hc - 2 * x ** 6 - 2 * y ** 3
    return np.array([e1, e2])


def kgon_solve(k, xy_window=((0.05, 2.0), (-2.0, -1e-9)), ngrid=400, apply_constraint=True):
    """All roots of the published system in x>0, y<0 (with the gon-constraint -x^2 cos(s*al) < y
    applied iff apply_constraint — the paper LISTS both 12-gon system roots although the second
    violates the constraint; the solver gate reproduces the published numbers constraint-free and
    reports constraint status separately). Dense sign-scan + Newton polish; deterministic."""
    from scipy import optimize
    F, G, Hc, M, s, al = _FGHM(k)
    xs = np.linspace(*xy_window[0], ngrid)
    ys = np.linspace(*xy_window[1], ngrid)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    E1 = X ** 6 * G + X ** 2 * Y ** 2 * Hc + 2 * X ** 4 * Y * M + Y ** 2 - X ** 4
    E2 = 2 * X ** 8 * F + 3 * X ** 6 * Y * G - X ** 2 * Y ** 3 * Hc - 2 * X ** 6 - 2 * Y ** 3
    cand = []
    sc1 = (np.diff(np.sign(E1), axis=0) != 0)[:, :-1] | (np.diff(np.sign(E1), axis=1) != 0)[:-1, :]
    sc2 = (np.diff(np.sign(E2), axis=0) != 0)[:, :-1] | (np.diff(np.sign(E2), axis=1) != 0)[:-1, :]
    hits = np.argwhere(sc1 & sc2)
    for ix, iy in hits:
        sol = optimize.fsolve(_poly_system, [xs[ix], ys[iy]], args=(k,), full_output=True)
        (xr, yr), info, ier, _ = sol
        if ier != 1 or xr <= 0 or yr >= 0:
            continue
        if apply_constraint and not (-xr ** 2 * np.cos(s * al) < yr):
            continue
        if np.max(np.abs(_poly_system((xr, yr), k))) > 1e-9:
            continue
        if not any(abs(xr - u) < 1e-6 and abs(yr - v) < 1e-6 for u, v in cand):
            cand.append((float(xr), float(yr)))
    return sorted(cand)


# 4-gon: l*=1, b*=0 (published; global min at c=4) — the polynomial system targets k>=5.
KGON_PUBLISHED = {4: (1.0, 0.0), 5: (1.17046, -0.28230), 6: (1.32053, -0.61814),
                  7: (1.44839, -0.96691), 8: (1.55045, -1.29119)}
KGON_8_ALT = (1.55041, -1.29122)                  # the paper's OTHER stated 8-gon value (4e-5 apart)
KGON_12_PUBLISHED = [(1.03322, -0.46654), (1.24975, -0.85483)]
H_K = {5: 0.23738, 6: 0.86746, 7: 1.74870, 8: 2.77311}   # L(k,c) = (h_k + c - k)/(3c)


# ============================== gates ======================================== #
def main():
    print("=" * 78)
    print("S3-A5 tms_potential — K1 / V0 / V1 gates (prereg frozen 2026-06-12)")
    print("=" * 78)

    print("\nK1a — published polynomial system: (l*, b*) reproduction (tol 1e-4):")
    solved = {}
    for k in (5, 6, 7, 8):
        roots = kgon_solve(k)
        ok = [r for r in roots if abs(r[0] - KGON_PUBLISHED[k][0]) < 1e-4
              and abs(r[1] - KGON_PUBLISHED[k][1]) < 1e-4]
        check(f"k={k}: root matches published {KGON_PUBLISHED[k]}", len(ok) == 1,
              f"roots={[(round(x,5), round(y,5)) for x, y in roots]}")
        if roots:
            solved[k] = ok[0] if ok else roots[0]
    if 8 in solved:
        x8, y8 = solved[8]
        d_main = max(abs(x8 - KGON_PUBLISHED[8][0]), abs(y8 - KGON_PUBLISHED[8][1]))
        d_alt = max(abs(x8 - KGON_8_ALT[0]), abs(y8 - KGON_8_ALT[1]))
        winner = "(1.55045, -1.29119) [table]" if d_main < d_alt else "(1.55041, -1.29122) [E.2 text]"
        print(f"  [info] 8-gon paper-inconsistency RESOLUTION: high-precision root "
              f"({x8:.7f}, {y8:.7f}) matches {winner}  (d_table={d_main:.2e}, d_text={d_alt:.2e})")

    print("\nK1b — 12-gon BOTH roots + non-existence c in {9,10,11,13}:")
    r12 = kgon_solve(12, apply_constraint=False)
    hit12 = sum(1 for u, v in KGON_12_PUBLISHED
                if any(abs(u - x) < 1e-4 and abs(v - y) < 1e-4 for x, y in r12))
    check("12-gon: both published SYSTEM roots found (constraint-free)", hit12 == 2,
          f"found={[(round(x,5), round(y,5)) for x, y in r12]}")
    s12, al12 = _FGHM(12)[4], _FGHM(12)[5]
    for x, y in r12:
        ok_con = -x ** 2 * np.cos(s12 * al12) < y
        print(f"  [info] 12-gon root ({x:.5f}, {y:.5f}): gon-constraint "
              f"-x^2 cos(s*al) < y is {'SATISFIED' if ok_con else 'VIOLATED'}"
              f"{'' if ok_con else '  (paper lists it anyway — noted, not silently corrected)'}")
    for k in (9, 10, 11, 13):
        check(f"c={k}: NO root (published non-existence)", len(kgon_solve(k)) == 0,
              f"found={kgon_solve(k)}")

    print("\nK1c — loss table via the implemented H (tol 1e-5):  L(k,c) = (h_k + c - k)/(3c)")
    cases = [(5, 6, 0, 0.06874), (6, 6, 0, 0.04819), (4, 5, 0, 0.06667), (5, 5, 0, 0.01583),
             (4, 4, 0, 0.0), (5, 6, 1, 0.06180), (4, 5, 1, 0.05667), (4, 6, 1, 0.10417)]
    for k, c, sigma, L_pub in cases:
        ls, bs = KGON_PUBLISHED[k] if k in solved or k == 4 else solved[k]
        if k in solved:
            ls, bs = solved[k]
        l, th, b = kgon(k, c, ls, bs, sigma=sigma)
        L = float(H_polar(l, th, b)) / (3 * c)
        tag = f"{k}-gon" + ("+" * sigma) + f" c={c}"
        check(f"L({tag}) = {L_pub:.5f}", abs(L - L_pub) < 1e-5, f"got {L:.5f}")
    # formula-level cross-check + the exact 1/144 sigma+ gap at c=6
    for k in (5, 6, 7, 8):
        ls, bs = solved.get(k, KGON_PUBLISHED[k])
        l, th, b = kgon(k, 8, ls, bs)
        Hval = float(H_polar(l, th, b))
        check(f"H(k={k}, c=8) = h_k + (8-k) (Cor. D.2 dead-feature constant)",
              abs(Hval - (H_K[k] + (8 - k))) < 3e-4, f"got {Hval:.5f} vs {H_K[k] + 8 - k:.5f}")
    l5, th5, b5 = kgon(5, 6, *solved.get(5, KGON_PUBLISHED[5]), sigma=0)
    l5p, th5p, b5p = kgon(5, 6, *solved.get(5, KGON_PUBLISHED[5]), sigma=1)
    gap = (float(H_polar(l5, th5, b5)) - float(H_polar(l5p, th5p, b5p))) / 18.0
    check("L(5) - L(5+) = 1/144 exactly (b+ = 1/(2c) mechanism)", abs(gap - 1.0 / 144) < 1e-9,
          f"gap={gap:.9f} vs {1/144:.9f}")

    print("\nV0 — closed form vs INDEPENDENT kink-split quadrature (200 seeded w, rel tol 1e-6):")
    rng = np.random.default_rng(20260612)
    worst = 0.0
    for _ in range(200):
        c = int(rng.integers(4, 8))
        W = rng.normal(0, 0.9, (2, c))
        b = rng.normal(-0.2, 0.6, c)
        Lq = loss_quadrature(W, b)
        Lc = float(L_cart(torch.tensor(W), torch.tensor(b)))
        rel = abs(Lc - Lq) / max(abs(Lq), 1e-12)
        worst = max(worst, rel)
    check("V0 worst rel err <= 1e-6", worst <= 1e-6, f"worst={worst:.2e}")

    print("\nV1 — importance deformation vs weighted quadrature (rule-2 validation):")
    worst = 0.0
    for u in (0.5, 1.5, 2.0):
        for _ in range(50):
            c = 6
            W = rng.normal(0, 0.9, (2, c))
            b = rng.normal(-0.2, 0.6, c)
            I = np.ones(c); I[-1] = u
            Lq = loss_quadrature(W, b, I)
            Lc = float(L_cart(torch.tensor(W), torch.tensor(b), I))
            worst = max(worst, abs(Lc - Lq) / max(abs(Lq), 1e-12))
    check("V1 worst rel err <= 1e-6 (u in {0.5, 1.5, 2.0})", worst <= 1e-6, f"worst={worst:.2e}")
    Wt = torch.tensor(rng.normal(0, 0.9, (2, 6)))
    bt = torch.tensor(rng.normal(-0.2, 0.6, 6))
    ident = abs(float(H_cart(Wt, bt, np.ones(6))) - float(H_cart(Wt, bt)))
    check("H_I(I=1) == H to machine precision", ident < 1e-14, f"|diff|={ident:.1e}")

    print("\nautograd smoke — analytic gradient available for the classification machinery:")
    l, th, b = kgon(5, 6, *solved.get(5, KGON_PUBLISHED[5]))
    l = l.clone().requires_grad_(True)
    H = H_polar(l, th, b)
    H.backward()
    g_live = float(l.grad[:5].abs().max())
    check("|grad_l H| at the 5-gon (live coords) ~ 0 (critical point)", g_live < 5e-4,
          f"max|g|={g_live:.2e}")

    print(f"\n{'ALL GATES PASS' if fail == 0 else str(fail) + ' GATE FAILURES'} — "
          f"{'classification machinery unblocked' if fail == 0 else 'STOP per prereg (fix/withdraw)'}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
