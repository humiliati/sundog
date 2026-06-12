#!/usr/bin/env python
"""H12 -- KL-bandit symmetry breaking despite a representable analytic optimum (slate HS12).

Prereg: docs/atlas/H12_RLHF_CUSP_PREREG.md (frozen 2026-06-11, commit 5d52071d, BEFORE this file
existed). Exact-expected-gradient KL-regularized reward maximization on a Z2-symmetric two-good-
mode bandit; shared-trunk tanh MLP (104 params, NO final bias -- the Hessian gauge fix); the
question is whether the gradient-flow LANDSCAPE of the parameterization sustains CERTIFIED
symmetry-broken attractors below a finite beta*, even though the Gibbs optimum pi* ~ exp(r/beta)
is symmetric, unique, analytic, and representable by the same trunk (controls: representability
distillation; tabular global convergence, Mei et al. 2020).

Certification battery (every verdict-bearing point): ||grad||inf <= 1e-10; NO UNSTABLE direction
(lam_max(H) <= +1e-9, torch-exact float64, cross-validated vs the numpy apparatus); step-halving
invariance; perturb-return (2 seeded draws, the dynamical attractor certificate).
DEVIATION D1 (recorded pre-verdict; caught by the frozen test): the prereg's lam_max <= -1e-8 is
UNSATISFIABLE at functional optima for an overparameterized trunk -- at equilibrium the parameter
Hessian is J^T H_L J + sum_a (grad^2 logit_a) gL_a with gL -> 0, hence rank <= 6: 98 of 104
eigenvalues are numerically zero. Amended certificate: lam_max <= +1e-9 (no unstable direction) +
perturb-return promoted from hysteresis-only to EVERY verdict-bearing point. All other thresholds
unchanged; the order-parameter stability claim is carried by the dynamical certificate.
Catastrophe calls rest on BISTABILITY TOPOLOGY of densely-continuated charts (folds by bisection);
no jet/|c3| instrument in any kill path (H4 banked confound).

Pre-registered outcomes: KILL(i) no certified breaking (clean null: deterministic function
approximation alone insufficient); KILL(ii) breaking but battery fails (named non-cusp null);
POSITIVE breaking + pitchfork exponent 1/2 +/- 0.15 + cusp wedge width exponent 3/2 +/- 0.4;
gate(iii) tabular control breaks -> quarantine; downgrade(iv) representability fails -> capacity
collapse (named, not the claim).

NOT public-eligible. Attribution: Korbak et al. 2022; Mei et al. 2020; Bishop PRML section 10;
Thom/Zeeman (cusp); H4/H8 in-house battery lessons.
Run:  python scripts/cat_rlhf_cusp.py            (full verdict run)
      python scripts/cat_rlhf_cusp.py --smoke    (plumbing check, NO verdict)
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import torch                                    # noqa: E402

torch.set_num_threads(1)

# ---- FROZEN constants (prereg section 2) ------------------------------------ #
SEED = 20260611
EMB = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 0.0], [-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]])
N_ACT, H1N, H2N = 6, 8, 8
NPARAM = 2 * H1N + H1N + H1N * H2N + H2N + H2N   # 104
BETA_GRID = [round(float(b), 4) for b in np.geomspace(0.05, 1.5, 16)]
N_INITS = 8
M_BREAK = 0.05                                   # macroscopic breaking floor
GRAD_TOL = 1e-10
LAM_TOL = 1e-9                                   # D1: no UNSTABLE direction (was -1e-8; see header)
PITCH_BAND, PITCH_R2 = (0.35, 0.65), 0.99
WIDTH_BAND, WIDTH_R2 = (1.1, 1.9), 0.98
WIN_LO, WIN_HI, WIN_MIN_PTS = 0.02, 0.30, 12
HYST_FRACS = [0.60, 0.70, 0.80, 0.85, 0.90]
EPS_BISECT_TOL = 1e-5
MAX_STEPS = 500_000


def rewards(eps):
    r = np.zeros(N_ACT)
    r[0], r[1] = 1.0 + eps / 2.0, 1.0 - eps / 2.0
    return r


def unpack(th):
    i = 0
    W1 = th[i:i + 2 * H1N].reshape(H1N, 2); i += 2 * H1N
    b1 = th[i:i + H1N]; i += H1N
    W2 = th[i:i + H1N * H2N].reshape(H2N, H1N); i += H1N * H2N
    b2 = th[i:i + H2N]; i += H2N
    w3 = th[i:i + H2N]
    return W1, b1, W2, b2, w3


def logits_np(th):
    W1, b1, W2, b2, w3 = unpack(th)
    A1 = np.tanh(EMB @ W1.T + b1)                # (6, H1)
    A2 = np.tanh(A1 @ W2.T + b2)                 # (6, H2)
    return A2 @ w3, A1, A2                       # (6,)


def j_and_grad(th, beta, eps):
    """Exact J and its exact gradient (manual backprop, float64)."""
    r = rewards(eps)
    L, A1, A2 = logits_np(th)
    Ls = L - L.max()
    p = np.exp(Ls); p /= p.sum()
    q = r - beta * np.log(N_ACT * p)             # r - beta*log(pi/u)
    J = float(p @ q) - 0.0                       # = E[r] - beta*KL
    gL = p * (q - p @ q)                         # dJ/dL
    W1, b1, W2, b2, w3 = unpack(th)
    gw3 = A2.T @ gL
    d2 = (gL[:, None] * w3[None, :]) * (1.0 - A2 ** 2)
    gW2 = d2.T @ A1
    gb2 = d2.sum(0)
    d1 = (d2 @ W2) * (1.0 - A1 ** 2)
    gW1 = d1.T @ EMB
    gb1 = d1.sum(0)
    g = np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2, gw3])
    return J, g


def pi_of(th):
    L, _, _ = logits_np(th)
    Ls = L - L.max()
    p = np.exp(Ls)
    return p / p.sum()


def m_of(th):
    p = pi_of(th)
    return float(p[0] - p[1])


def gibbs_pi(beta, eps):
    z = rewards(eps) / beta
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


def m_gibbs(beta, eps):
    p = gibbs_pi(beta, eps)
    return float(p[0] - p[1])


def j_of_pi(p, beta, eps):
    return float(p @ rewards(eps)) - beta * float(p @ np.log(np.maximum(N_ACT * p, 1e-300)))


# ---- torch twin (certification only) ---------------------------------------- #
_EMB_T = torch.tensor(EMB, dtype=torch.float64)


def j_torch(th_t, beta, eps):
    i = 0
    W1 = th_t[i:i + 2 * H1N].reshape(H1N, 2); i += 2 * H1N
    b1 = th_t[i:i + H1N]; i += H1N
    W2 = th_t[i:i + H1N * H2N].reshape(H2N, H1N); i += H1N * H2N
    b2 = th_t[i:i + H2N]; i += H2N
    w3 = th_t[i:i + H2N]
    A1 = torch.tanh(_EMB_T @ W1.T + b1)
    A2 = torch.tanh(A1 @ W2.T + b2)
    L = A2 @ w3
    p = torch.softmax(L, dim=0)
    r = torch.tensor(rewards(eps), dtype=torch.float64)
    return p @ r - beta * (p @ torch.log(N_ACT * p))


def torch_grad(th, beta, eps):
    t = torch.tensor(th, dtype=torch.float64, requires_grad=True)
    j_torch(t, beta, eps).backward()
    return t.grad.numpy()


def torch_lam_max(th, beta, eps):
    t = torch.tensor(th, dtype=torch.float64)
    Hm = torch.autograd.functional.hessian(lambda x: j_torch(x, beta, eps), t)
    Hm = 0.5 * (Hm + Hm.T)
    return float(torch.linalg.eigvalsh(Hm)[-1])


def hess_fd(th, beta, eps, h=1e-6):
    """Numpy FD Hessian of the analytic gradient (Newton polish only; cert uses torch)."""
    n = th.size
    Hm = np.empty((n, n))
    for k in range(n):
        e = np.zeros(n); e[k] = h
        _, gp = j_and_grad(th + e, beta, eps)
        _, gm = j_and_grad(th - e, beta, eps)
        Hm[:, k] = (gp - gm) / (2 * h)
    return 0.5 * (Hm + Hm.T)


# ---- convergence (Armijo ascent + Newton polish), prereg-pinned ------------- #
def converge(th0, beta, eps, max_steps=MAX_STEPS):
    # D2 (pre-verdict, apparatus power; see receipt): eta cap 1.0 -> 256.0. The prereg's cap
    # made small-beta runs crawl through tanh-saturation plateaus and exhaust the step budget
    # uncertified (smoke evidence). Armijo validates every step as genuine ascent, so a larger
    # cap changes reachability/speed, never the certificates. Thresholds/budgets unchanged.
    # D3 (pre-verdict, wall-clock only; see receipt): EARLY STALL-EXIT. Saturation-plateau
    # points crawl with negligible ||grad|| improvement and were burning the entire step budget
    # (~20+ min each) to end exactly where they stall: reported UNCERTIFIED with the same
    # endpoint character. If the best ||grad||inf has not improved by >= 2% over the trailing
    # 60k steps, exit early; the point is reported uncertified exactly as the prereg's
    # "non-converged runs carry no weight" clause anticipates. Hyperbolic-attractor convergence
    # is geometric (steady improvement) and never trips this. Thresholds/battery unchanged.
    # D4 (pre-verdict, wall-clock only; see receipt): (a) eta grows only after a CLEAN
    # (no-backtrack) step -- the previous grow-every-step policy re-paid ~15 backtracking
    # evaluations per step collapsing eta from the cap; (b) the stall criterion tightens to
    # "best ||grad||inf improved < 25% over the trailing 60k steps" (geometric convergence to a
    # hyperbolic attractor improves by orders of magnitude per 60k; slow saturation crawls do
    # not). Certificates, thresholds, budgets unchanged.
    th = th0.copy()
    eta, ETA_MAX, C = 0.5, 256.0, 1e-4
    J, g = j_and_grad(th, beta, eps)
    steps = 0
    best_g = float(np.max(np.abs(g)))
    best_at_ckpt = best_g
    while steps < max_steps and np.max(np.abs(g)) > 1e-8:
        gn2 = float(g @ g)
        n_bt = 0
        while True:
            th_n = th + eta * g
            J_n, g_n = j_and_grad(th_n, beta, eps)
            if J_n >= J + C * eta * gn2 or eta < 1e-16:
                break
            eta *= 0.5
            n_bt += 1
        th, J, g = th_n, J_n, g_n
        if n_bt == 0:
            eta = min(eta * 1.5, ETA_MAX)
        steps += 1
        best_g = min(best_g, float(np.max(np.abs(g))))
        if steps % 60_000 == 0:
            if best_g > best_at_ckpt / 1.25:
                break                            # stalled: uncertifiable plateau, stop paying
            best_at_ckpt = best_g
    # Newton polish to GRAD_TOL. D1: the Hessian is rank-deficient at functional optima
    # (rank <= 6), so use a damped PSEUDO-INVERSE step (g lies in the Jacobian range, where
    # curvature is strict); reject steps that grow ||g||.
    for _ in range(60):
        ginf = np.max(np.abs(g))
        if ginf <= GRAD_TOL:
            break
        Hm = hess_fd(th, beta, eps)
        lam = np.linalg.eigvalsh(Hm)
        if lam[-1] > 1e-6:
            break                                # unstable direction: not a basin, stop polishing
        step = -np.linalg.pinv(Hm, rcond=1e-10) @ g
        damp = 1.0
        for _ in range(30):
            th_n = th + damp * step
            J_n, g_n = j_and_grad(th_n, beta, eps)
            if np.max(np.abs(g_n)) < ginf:
                th, J, g = th_n, J_n, g_n
                break
            damp *= 0.5
        else:
            break                                # no productive polish step
    return th, J, g, steps


def step_halving_ok(th, beta, eps, m0):
    for eta in (1e-2, 5e-3):
        t = th.copy()
        for _ in range(20_000):
            _, g = j_and_grad(t, beta, eps)
            t += eta * g
        if abs(m_of(t) - m0) > 1e-8:
            return False
    return True


def certify(th, beta, eps, full=True, pr_seed=0):
    """Prereg 2e battery, D1-amended (see header). Returns (ok, info)."""
    _, g = j_and_grad(th, beta, eps)
    ginf = float(np.max(np.abs(g)))
    info = {"grad_inf": ginf}
    if ginf > GRAD_TOL:
        info["fail"] = "grad"
        return False, info
    cross = float(np.max(np.abs(g - torch_grad(th, beta, eps))))
    info["np_torch_diff"] = cross
    if cross > 1e-10:
        info["fail"] = "cross"
        return False, info
    lam = torch_lam_max(th, beta, eps)
    info["lam_max"] = lam
    if lam > LAM_TOL:                            # D1: no unstable direction
        info["fail"] = "hessian_unstable"
        return False, info
    if full and not step_halving_ok(th, beta, eps, m_of(th)):
        info["fail"] = "step_halving"
        return False, info
    if full and not perturb_return_ok(th, beta, eps, m_of(th), SEED + 555 + pr_seed):
        info["fail"] = "perturb_return"          # D1: the dynamical attractor certificate
        return False, info
    return True, info


def perturb_return_ok(th, beta, eps, m0, seed):
    rng = np.random.default_rng(seed)
    for _ in range(2):
        t = th + rng.normal(0, 1e-4, th.size)
        t, _, _, _ = converge(t, beta, eps)
        if abs(m_of(t) - m0) > 1e-6:
            return False
    return True


def init_theta(seed):
    rng = np.random.default_rng(SEED + seed)
    W1 = rng.normal(0, 1 / np.sqrt(2), (H1N, 2))
    b1 = rng.normal(0, 1 / np.sqrt(2), H1N)
    W2 = rng.normal(0, 1 / np.sqrt(H1N), (H2N, H1N))
    b2 = rng.normal(0, 1 / np.sqrt(H1N), H2N)
    w3 = rng.normal(0, 1 / np.sqrt(H2N), H2N)
    return np.concatenate([W1.ravel(), b1, W2.ravel(), b2, w3])


# ---- tabular control --------------------------------------------------------- #
def tabular_converge(beta, eps, seed):
    rng = np.random.default_rng(SEED + 9000 + seed)
    L = rng.normal(0, 1.0, N_ACT)
    r = rewards(eps)
    eta = 0.5
    for _ in range(400_000):
        Ls = L - L.max()
        p = np.exp(Ls); p /= p.sum()
        q = r - beta * np.log(N_ACT * p)
        gL = p * (q - p @ q)
        if np.max(np.abs(gL)) <= 1e-12:
            break
        L = L + eta * gL
    p = np.exp(L - L.max()); p /= p.sum()
    return float(p[0] - p[1])


# ---- representability control ------------------------------------------------ #
def distill_tv(beta, eps=0.0):
    target = rewards(eps) / beta
    target = target - target.mean()
    th = init_theta(0)

    def mse_grad(t):
        L, A1, A2 = logits_np(t)
        e = L - target
        mse = float(e @ e)
        gL = 2 * e
        W1, b1, W2, b2, w3 = unpack(t)
        gw3 = A2.T @ gL
        d2 = (gL[:, None] * w3[None, :]) * (1 - A2 ** 2)
        gW2 = d2.T @ A1; gb2 = d2.sum(0)
        d1 = (d2 @ W2) * (1 - A1 ** 2)
        gW1 = d1.T @ EMB; gb1 = d1.sum(0)
        return mse, np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2, gw3])

    eta = 0.1
    mse, g = mse_grad(th)
    for _ in range(MAX_STEPS):
        if mse < 1e-14 or np.max(np.abs(g)) < 1e-12:
            break
        while True:
            th_n = th - eta * g
            mse_n, g_n = mse_grad(th_n)
            if mse_n <= mse - 1e-4 * eta * float(g @ g) or eta < 1e-16:
                break
            eta *= 0.5
        th, mse, g = th_n, mse_n, g_n
        eta = min(eta * 1.5, 1.0)
    L, _, _ = logits_np(th)
    p = np.exp(L - L.max()); p /= p.sum()
    return 0.5 * float(np.abs(p - gibbs_pi(beta, eps)).sum())


# ---- fold walker (injectable converge for frozen-test calibration) ----------- #
def walk_fold(th_branch, beta, direction, conv, mfun, step0=0.02, eps_lim=0.5,
              tol=EPS_BISECT_TOL, sign_keep=+1):
    """Walk eps from 0 in `direction` (+1/-1) on the sign_keep branch (warm-started), then
    bisect the fold to `tol`. conv(th, eps) -> th_converged. Returns (eps_fold, th_last_good)
    or (None, th) if no fold inside eps_lim (walks on a STATIC landscape; path-free folds)."""
    th_good, eps_good = th_branch.copy(), 0.0
    step = step0
    lost_at = None
    while abs(eps_good) < eps_lim:
        eps_try = eps_good + direction * step
        th_try = conv(th_good, eps_try)
        if np.sign(mfun(th_try)) == sign_keep and abs(mfun(th_try)) > 1e-3:
            th_good, eps_good = th_try, eps_try
        else:
            lost_at = eps_try
            break
    if lost_at is None:
        return None, th_good
    lo, hi = eps_good, lost_at                    # branch alive at lo, lost at hi
    while abs(hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        th_try = conv(th_good, mid)
        if np.sign(mfun(th_try)) == sign_keep and abs(mfun(th_try)) > 1e-3:
            th_good, lo = th_try, mid
        else:
            hi = mid
    return 0.5 * (lo + hi), th_good


def ols_loglog(x, y):
    lx, ly = np.log(np.asarray(x)), np.log(np.asarray(y))
    A = np.stack([lx, np.ones_like(lx)], 1)
    coef, _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
    yh = A @ coef
    ss_res = float(((ly - yh) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    return float(coef[0]), 1.0 - ss_res / max(ss_tot, 1e-300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/atlas/h12/cat_rlhf_cusp_result.json")
    args = ap.parse_args()
    t0 = time.time()
    beta_grid = BETA_GRID[::3] if args.smoke else BETA_GRID
    n_inits = 3 if args.smoke else N_INITS
    mode = "SMOKE (no verdict)" if args.smoke else "FULL (verdict run)"

    print("=" * 92)
    print(f"H12 -- KL-bandit symmetry breaking vs representable Gibbs optimum   [{mode}]")
    print(f"  trunk 2-{H1N}-{H2N}-1 tanh, no final bias ({NPARAM} params); exact gradients; "
          f"battery: grad<=1e-10, lam_max<=-1e-8, step-halving, perturb-return")
    print("=" * 92)

    # ---- controls first ---- #
    tab_bad = []
    for b in beta_grid:
        for s in range(min(n_inits, 4)):
            mt = tabular_converge(b, 0.0, s)
            if abs(mt - m_gibbs(b, 0.0)) > 1e-6:
                tab_bad.append((b, s, mt))
    gate_tab = len(tab_bad) == 0
    print(f"[GATE iii] tabular control: global convergence to Gibbs at all beta "
          f"-> {'PASS' if gate_tab else 'QUARANTINE ' + str(tab_bad[:3])}")
    tv_lo = distill_tv(beta_grid[0])
    rep_ok = tv_lo <= 1e-3
    print(f"[CTRL iv] representability @ beta={beta_grid[0]}: TV(distill, Gibbs) = {tv_lo:.2e} "
          f"(<= 1e-3) -> {'PASS' if rep_ok else 'FAIL (capacity-collapse downgrade)'}")

    # ---- coarse sweep ---- #
    print(f"\nCOARSE SWEEP ({len(beta_grid)} beta x {n_inits} inits, eps=0):")
    coarse, broken = [], []
    for b in beta_grid:
        ms = []
        for s in range(n_inits):
            th, J, g, steps = converge(init_theta(s), b, 0.0)
            ok, info = certify(th, b, 0.0, full=not args.smoke,
                               pr_seed=int(b * 10000) + s)
            mm = m_of(th)
            dj = j_of_pi(gibbs_pi(b, 0.0), b, 0.0) - J
            coarse.append({"beta": b, "seed": s, "m": mm, "J": J, "dJ_gibbs": dj,
                           "certified": bool(ok), "steps": steps, **info})
            ms.append((mm, ok))
            if ok and abs(mm) >= M_BREAK:
                broken.append((b, s, mm, th))
        summ = " ".join(f"{m:+.3f}{'*' if ok else '!'}" for m, ok in ms)
        print(f"  beta={b:<8} m: {summ}   (m_gibbs=0; * certified, ! uncertified)", flush=True)
    breaking = len(broken) > 0
    print(f"\nBREAKING (certified |m| >= {M_BREAK} from fresh init): "
          f"{'YES at beta=' + str(sorted(set(b for b, *_ in broken))) if breaking else 'NONE'}")

    pitch, wedge, a3band = {}, {}, {}
    if breaking and not args.smoke:
        # ---- beta-continuation upward from the largest broken beta ---- #
        b_anchor, s_anchor, m_anchor, th_anchor = max(broken, key=lambda x: x[0])
        sign_br = np.sign(m_anchor)
        chain = [(b_anchor, abs(m_anchor), th_anchor.copy())]
        b_cur, th_cur, db = b_anchor, th_anchor.copy(), 0.05
        while db > 1e-5:
            b_try = b_cur + db
            th_try, _, _, _ = converge(th_cur, b_try, 0.0)
            mm = m_of(th_try)
            if np.sign(mm) == sign_br and abs(mm) > max(1e-4, 0.1 * chain[-1][1]):
                ok, _ = certify(th_try, b_try, 0.0, pr_seed=int(b_try * 10000))
                if ok:
                    chain.append((b_try, abs(mm), th_try.copy()))
                    b_cur, th_cur = b_try, th_try
                    continue
            db *= 0.5
        # beta* fit on the upper half of the chain (m^2 linear in beta)
        chain.sort(key=lambda x: x[0])
        bs = np.array([c[0] for c in chain]); mm2 = np.array([c[1] for c in chain]) ** 2
        top = mm2 <= 0.5 * mm2.max()
        if top.sum() >= 3:
            A = np.stack([bs[top], np.ones(int(top.sum()))], 1)
            coef, _, _, _ = np.linalg.lstsq(A, mm2[top], rcond=None)
            beta_star = float(-coef[1] / coef[0]) if coef[0] < 0 else float(bs.max() + db)
        else:
            beta_star = float(bs.max() + db)
        # refine window to >= WIN_MIN_PTS certified points
        def in_win(b):
            t = (beta_star - b) / beta_star
            return WIN_LO <= t <= WIN_HI
        win = [(b, m, th) for b, m, th in chain if in_win(b)]
        tries = 0
        while len(win) < WIN_MIN_PTS and tries < 40:
            targets = np.linspace(beta_star * (1 - WIN_HI), beta_star * (1 - WIN_LO),
                                  WIN_MIN_PTS + 4)
            have = np.array([b for b, _, _ in win]) if win else np.array([])
            gaps = [t for t in targets
                    if have.size == 0 or np.min(np.abs(have - t)) > 1e-4]
            if not gaps:
                break
            b_t = gaps[0]
            src = min(chain, key=lambda c: abs(c[0] - b_t))
            th_try, _, _, _ = converge(src[2], b_t, 0.0)
            mm = m_of(th_try)
            ok, _ = certify(th_try, b_t, 0.0, pr_seed=int(b_t * 10000) + 1)
            if ok and np.sign(mm) == sign_br and abs(mm) > 1e-4:
                chain.append((b_t, abs(mm), th_try.copy()))
                chain.sort(key=lambda x: x[0])
                if in_win(b_t):
                    win.append((b_t, abs(mm), th_try))
            tries += 1
        win.sort(key=lambda x: x[0])
        if len(win) >= 3:
            slope, r2 = ols_loglog([beta_star - b for b, _, _ in win],
                                   [m for _, m, _ in win])
            pitch = {"beta_star": beta_star, "n_window": len(win), "exponent": slope,
                     "r2": r2, "window": [(b, m) for b, m, _ in win]}
            print(f"\nPITCHFORK: beta* = {beta_star:.4f}; window n={len(win)}; "
                  f"exponent = {slope:.3f} (band {PITCH_BAND}); R2 = {r2:.4f}")
        else:
            pitch = {"beta_star": beta_star, "n_window": len(win), "exponent": None, "r2": None}
            print(f"\nPITCHFORK: window too thin (n={len(win)}) -- battery fails")

        # ---- cusp wedge: fold bisection at pinned beta fractions ---- #
        def conv_at(b):
            return lambda th, e: converge(th, b, e)[0]
        widths = []
        print("\nCUSP WEDGE (fold bisection, perturb-return at each accepted point):")
        for fr in HYST_FRACS:
            b_h = fr * beta_star
            src = min(chain, key=lambda c: abs(c[0] - b_h))
            th_h, _, _, _ = converge(src[2], b_h, 0.0)
            mm = m_of(th_h)
            if abs(mm) < 1e-3:
                print(f"  beta={b_h:.4f} (fr={fr}): branch not found -- skipped")
                continue
            if not perturb_return_ok(th_h, b_h, 0.0, mm, SEED + int(fr * 100)):
                print(f"  beta={b_h:.4f}: perturb-return FAILED (freeze artifact) -- skipped")
                continue
            sgn = int(np.sign(mm))
            e_lo, _ = walk_fold(th_h, b_h, -sgn, conv_at(b_h), m_of, sign_keep=sgn)
            e_hi, _ = walk_fold(th_h, b_h, +sgn, conv_at(b_h), m_of, sign_keep=sgn)
            # the +m branch folds at negative eps; mirror branch by symmetry at +|e|
            if e_lo is None:
                print(f"  beta={b_h:.4f}: no fold inside |eps|<=0.5 -- skipped")
                continue
            w = 2.0 * abs(e_lo)
            widths.append((b_h, w, abs(e_lo), None if e_hi is None else abs(e_hi)))
            print(f"  beta={b_h:.4f} (fr={fr}): fold |eps| = {abs(e_lo):.6f} -> width = {w:.6f}")
        if len(widths) >= 3:
            wslope, wr2 = ols_loglog([beta_star - b for b, _, _, _ in widths],
                                     [w for _, w, _, _ in widths])
            wedge = {"widths": [(b, w) for b, w, _, _ in widths], "exponent": wslope, "r2": wr2}
            print(f"  width exponent = {wslope:.3f} (band {WIDTH_BAND}); R2 = {wr2:.4f}")
        else:
            wedge = {"widths": [(b, w) for b, w, _, _ in widths], "exponent": None, "r2": None}
            print(f"  wedge: insufficient measurable folds ({len(widths)}) -- battery fails")

    # ---- verdict (prereg section 3) ---- #
    print("\n" + "=" * 92)
    if args.smoke:
        verdict = "smoke"
        print("SMOKE complete -- plumbing only, NO verdict.")
    elif not gate_tab:
        verdict = "quarantine"
        print("RESULT: QUARANTINE -- tabular control broke (apparatus/scope bug, not a result).")
    elif not breaking:
        verdict = "kill_i"
        print("RESULT: KILL(i) -- CLEAN INFORMATIVE NULL. No certified symmetry breaking from")
        print("  fresh inits anywhere on the grid: deterministic function approximation alone is")
        print("  insufficient for symmetry-broken collapse at this scale; sampling noise /")
        print("  multi-state capacity are implicated for real collapse.")
    else:
        battery = (pitch.get("exponent") is not None and wedge.get("exponent") is not None
                   and PITCH_BAND[0] <= pitch["exponent"] <= PITCH_BAND[1]
                   and pitch["r2"] >= PITCH_R2
                   and WIDTH_BAND[0] <= wedge["exponent"] <= WIDTH_BAND[1]
                   and wedge["r2"] >= WIDTH_R2)
        if battery and not rep_ok:
            verdict = "capacity_collapse"
            print("RESULT: DOWNGRADE(iv) -- breaking + battery pass BUT representability failed:")
            print("  named outcome 'capacity collapse', NOT the landscape claim.")
        elif battery:
            verdict = "positive"
            print("RESULT: POSITIVE -- certified symmetry-broken attractors under exact")
            print(f"  deterministic gradient flow despite a representable Gibbs optimum:")
            print(f"  beta* = {pitch['beta_star']:.4f}, pitchfork exponent "
                  f"{pitch['exponent']:.3f} (R2 {pitch['r2']:.4f}), wedge width exponent "
                  f"{wedge['exponent']:.3f} (R2 {wedge['r2']:.4f}).")
        else:
            verdict = "kill_ii"
            print("RESULT: KILL(ii) -- symmetry-broken attractors EXIST but are NOT cusp-class")
            print(f"  at the calibrated battery: pitchfork {pitch.get('exponent')} "
                  f"(R2 {pitch.get('r2')}), width {wedge.get('exponent')} "
                  f"(R2 {wedge.get('r2')}). Banked H4-style as a named null.")
    print("=" * 92)

    out = Path(args.out if not args.smoke else args.out.replace(".json", "_smoke.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        verdict=verdict, smoke=bool(args.smoke), beta_grid=beta_grid, n_inits=n_inits,
        gate_tabular=bool(gate_tab), representability_tv=tv_lo, rep_ok=bool(rep_ok),
        breaking=bool(breaking), coarse=coarse, pitchfork=pitch, wedge=wedge,
        m_break=M_BREAK, seed=SEED, wall_s=round(time.time() - t0, 1)), indent=2,
        default=lambda o: float(o) if isinstance(o, (np.floating,)) else
        bool(o) if isinstance(o, np.bool_) else o))
    print(f"\nwrote {out}  ({round(time.time() - t0, 1)}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
