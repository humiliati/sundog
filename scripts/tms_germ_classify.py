#!/usr/bin/env python
"""S3-A5 — TMS k-gon germ classification machinery. PREREG: docs/atlas/TMS_KGON_GERM_PREREG.md
(frozen 2026-06-12). Substrate gates: scripts/tms_potential.py (K1/V0/V1 ALL PASS 2026-06-12).

STAGE A (this file, --stage e0): inventory continuation over the rule-2 control axis u = I_6 in
[0.5, 2.0] + the E0 EXISTENCE GATE — the window must contain >= 1 transition event
(global-minimizer identity change OR critical-point inventory change) BEFORE any germ
classification unblinds. Widen-once rule: [0.25, 3.0]; still empty => E0 ABORT.

Members continued (c=6, the paper's 18-family inventory restricted to the loss-relevant ladder):
4, 4+, 4++, 5, 5+, 6. Dead-feature flat directions (b_dead < 0 cones) are PINNED and AUDITED
(H exactly constant along them), never counted as degeneracy. The 5-gon/4-gon live parts are
u-independent (only their dead-feature penalty scales with I_6 — checked); the 6-gon genuinely
deforms. Lift-off probe: warm-started full-space polish from 5+(l6 bumped) detects birth/death of
an l6 > 0 basin (critical-point inventory change).

Deterministic: fixed starts, Newton/Levenberg with autograd gradients+Hessians, no RNG.
Run: python scripts/tms_germ_classify.py --stage e0
"""
import argparse
import sys
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, "scripts")
import tms_potential as tp

torch.set_default_dtype(torch.float64)
C = 6
B_DEAD = -0.5                       # pinned dead bias (flat direction; audited)
BPLUS = 1.0 / (2 * C)               # the published sigma+ optimal positive bias (u-invariant argmin)


# ===== member configurations and active-coordinate packing ==================== #
class Member:
    """A continued critical-orbit family. Active coords: live l (k), free live angles (k-1),
    live b (k), plus sigma+ dead biases (sigma). Dead l pinned 0; remaining dead b pinned B_DEAD;
    dead angular positions pinned (flat; W=0)."""

    def __init__(self, name, k, sigma):
        self.name, self.k, self.sigma = name, k, sigma
        ls, bs = tp.KGON_PUBLISHED[k]
        self.v0 = np.concatenate([np.full(k, ls), np.full(k - 1, 2 * np.pi / k),
                                  np.full(k, bs), np.full(sigma, BPLUS)])

    def unpack(self, v):
        k, sig = self.k, self.sigma
        v = torch.as_tensor(v)
        l_live = v[:k]
        th_live = v[k:2 * k - 1]
        b_live = v[2 * k - 1:3 * k - 1]
        b_plus = v[3 * k - 1:3 * k - 1 + sig]
        l = torch.zeros(C, dtype=v.dtype)
        l = torch.cat([l_live, torch.zeros(C - k, dtype=v.dtype)])
        # angles: live gaps th_1..th_{k-1}; closing live gap (2pi - sum) split equally across the
        # dead features sitting in it (l=0 there: H independent of the split — audited flat).
        closing = 2 * np.pi - th_live.sum()
        th_rest = torch.full((C - k + 1,), 1.0, dtype=v.dtype) * closing / (C - k + 1)
        th = torch.cat([th_live, th_rest])
        b_dead = torch.full((C - k - sig,), B_DEAD, dtype=v.dtype)
        b = torch.cat([b_live, b_plus, b_dead])
        return l, th, b

    def H(self, v, u):
        l, th, b = self.unpack(v)
        I = torch.ones(C, dtype=l.dtype)
        I[C - 1] = u                # the deformed feature is the LAST (a dead one for k<6 members)
        return tp.H_polar(l, th, b, I)


def H_active(member, v_np, u):
    v = torch.tensor(v_np, requires_grad=True)
    H = member.H(v, u)
    g = torch.autograd.grad(H, v)[0]
    return float(H.detach()), g.detach().numpy()


def hess_active(member, v_np, u):
    v = torch.tensor(v_np)
    return torch.autograd.functional.hessian(lambda x: member.H(x, u), v).numpy()


def polish(member, v_np, u, iters=60, tol=1e-11):
    """Levenberg-damped Newton to a critical point (works for minima/saddles; deterministic)."""
    v = np.array(v_np, float)
    lam = 1e-8
    for _ in range(iters):
        Hval, g = H_active(member, v, u)
        if np.linalg.norm(g, np.inf) < tol:
            return v, Hval, True
        Hm = hess_active(member, v, u)
        try:
            step = np.linalg.solve(Hm + lam * np.eye(len(v)), -g)
        except np.linalg.LinAlgError:
            lam *= 10
            continue
        v2 = v + step
        _, g2 = H_active(member, v2, u)
        if np.linalg.norm(g2) <= np.linalg.norm(g):
            v, lam = v2, max(lam * 0.3, 1e-12)
        else:
            lam *= 10
            if lam > 1e6:
                break
    Hval, g = H_active(member, v, u)
    return v, Hval, np.linalg.norm(g, np.inf) < 1e-7


# ===== the lift-off probe (full live space, feature 6 inserted mid-gap) ======= #
class LiftOff(Member):
    """Full 6-feature active space, warm-started from the 5+(-gon) with l6 bumped: detects an
    l6 > 0 critical point (the '5->6 transition state / asymmetric hexagon' basin)."""

    def __init__(self, l6_start=0.20):
        self.name, self.k, self.sigma = f"lift(l6={l6_start})", 6, 0
        ls, bs = tp.KGON_PUBLISHED[5]
        l = np.array([ls] * 5 + [l6_start])
        th = np.array([2 * np.pi / 5] * 4 + [np.pi / 5])        # feature 6 mid-gap (36/36 split)
        b = np.array([bs] * 5 + [BPLUS])
        self.v0 = np.concatenate([l, th, b])


def member_lambda_min(member, v, u, flat_tol=1e-10):
    """Min Hessian eigenvalue over active coords, relative to median |eig| (theta_M units)."""
    Hm = hess_active(member, v, u)
    ev = np.linalg.eigvalsh(Hm)
    scale = np.median(np.abs(ev)) + 1e-300
    return ev[0] / scale, ev / scale


# ===== STAGE A: continuation + E0 ============================================ #
def stage_e0(u_lo=0.5, u_hi=2.0, nu=31, widened=False):
    print("=" * 78)
    print(f"S3-A5 STAGE A — inventory continuation + E0 gate, u = I_6 in [{u_lo}, {u_hi}] ({nu} pts)")
    print("=" * 78)
    members = [Member("4-gon", 4, 0), Member("4+", 4, 1), Member("4++", 4, 2),
               Member("5-gon", 5, 0), Member("5+", 5, 1), Member("6-gon", 6, 0)]
    lift = LiftOff()
    us = np.linspace(u_lo, u_hi, nu)
    warm = {m.name: m.v0.copy() for m in members}
    warm[lift.name] = lift.v0.copy()
    rows, ident, liftstate = [], [], []
    for u in us:
        Ls = {}
        for m in members:
            v, Hval, ok = polish(m, warm[m.name], u)
            if ok:
                warm[m.name] = v
            Ls[m.name] = Hval / (3 * C) if ok else np.nan
        vL, HL, okL = polish(lift, warm[lift.name], u)
        l6 = abs(vL[5])
        if okL:
            warm[lift.name] = vL
        alive = okL and l6 > 0.05
        liftstate.append((u, alive, l6 if okL else np.nan, HL / 18 if okL else np.nan))
        gm = min(Ls, key=lambda n: Ls[n])
        ident.append((u, gm))
        rows.append((u, Ls, gm, alive))
        lifttag = f"lift-off basin l6={l6:.3f} L={HL/18:.5f}" if alive else "lift collapses -> 5-ish"
        print(f"  u={u:.3f}: " + "  ".join(f"{n}={Ls[n]:.5f}" for n in Ls)
              + f"   GLOBAL-MIN={gm}   [{lifttag}]")

    print("\nE0 adjudication:")
    events = []
    for a, b in zip(ident, ident[1:]):
        if a[1] != b[1]:
            events.append(("global-min identity change", a[0], b[0], f"{a[1]} -> {b[1]}"))
    for a, b in zip(liftstate, liftstate[1:]):
        if a[1] != b[1]:
            events.append(("critical-point inventory change (lift-off basin)",
                           a[0], b[0], f"alive {a[1]} -> {b[1]}"))
    for kind, ua, ub, what in events:
        print(f"  EVENT: {kind} in u in ({ua:.3f}, {ub:.3f}): {what}")
    if events:
        print(f"\n  E0 PASS: {len(events)} event(s) in the window — classification UNBLOCKED.")
        return 0, events
    if not widened:
        print("\n  E0: zero events — applying the pre-registered widen-once enlargement [0.25, 3.0].")
        return stage_e0(0.25, 3.0, 56, widened=True)
    print("\n  E0 ABORT: zero events after widen-once — banked as 'no transition in the feasible "
          "closed-form window' (lesser receipt; prereg section 2).")
    return 1, []


# ===== STAGE B: adjudication of the E0 event (prereg section 4 partition) ===== #
THETA_M = 1e-3                      # frozen (prereg section 4)
A4_DIVE = 0.25
CORANK2_REL = 0.05


def find_ustar(m_a, m_b, lo=0.6, hi=0.75, iters=60):
    """Bisection on L_a(u) - L_b(u) = 0 (the global-min identity crossing), warm-started."""
    va, vb = m_a.v0.copy(), m_b.v0.copy()

    def gap(u):
        nonlocal va, vb
        va, Ha, _ = polish(m_a, va, u)
        vb, Hb, _ = polish(m_b, vb, u)
        return Ha - Hb
    glo, ghi = gap(lo), gap(hi)
    assert glo * ghi < 0, f"no sign change in [{lo},{hi}]: {glo} {ghi}"
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        g = gap(mid)
        if glo * g <= 0:
            hi = mid
        else:
            lo, glo = mid, g
    return 0.5 * (lo + hi), va, vb


def essential_corank(member, v, u):
    rel, ev = member_lambda_min(member, v, u)
    return int(np.sum(np.abs(ev) < THETA_M)), rel, ev


def stage_classify():
    print("=" * 78)
    print("S3-A5 STAGE B — adjudication of the E0 event (O-partition, prereg section 4)")
    print("=" * 78)
    import atlas_jet_classify as jc
    okall = True

    print("\n(1) u* localization (5+ vs 6-gon global-min crossing):")
    m5p, m6 = Member("5+", 5, 1), Member("6-gon", 6, 0)
    ustar, v5, v6 = find_ustar(m5p, m6)
    print(f"    u* = {ustar:.9f}  (E0 bracket was (0.650, 0.700))")

    print("\n(2) theta_M nondegeneracy certificates through the crossing (both members,")
    print(f"    lambda_min/scale >= theta_M = {THETA_M}; essential corank must be 0):")
    for u in (0.5, ustar - 0.05, ustar, ustar + 0.05, 1.0, 2.0):
        for m, v0 in ((m5p, v5), (m6, v6)):
            v, Hval, ok = polish(m, v0.copy(), u)
            cor, rel, _ = essential_corank(m, v, u)
            good = ok and cor == 0 and rel >= THETA_M
            okall &= good
            print(f"    u={u:.4f} {m.name:6s}: lambda_min/scale={rel:+.5f}  essential corank={cor}"
                  f"  L={Hval/18:.6f}  [{'OK' if good else 'FAIL'}]")

    print("\n(3) K2 control battery (pinned protocol; thresholds frozen on controls):")
    gen = jc.cusp_c3(*jc.synthetic_swallowtail(-0.40))[0]
    dive = jc.cusp_c3(*jc.synthetic_swallowtail(0.0))
    r_dive = (dive[0] if dive else 0.0) / gen
    c1 = r_dive < A4_DIVE
    print(f"    A4 control dives: ratio(h=0)={r_dive:.3f} < {A4_DIVE}  [{'OK' if c1 else 'FAIL'}]")
    r_a3 = min((jc.cusp_c3(*jc.synthetic_swallowtail(h)) or [0])[0] / gen for h in (-0.20, -0.10))
    c2 = r_a3 >= A4_DIVE
    print(f"    A3 members bounded: min ratio(h=-0.2,-0.1)={r_a3:.3f} >= {A4_DIVE}  "
          f"[{'OK' if c2 else 'FAIL'}]")
    X, Y, d = jc.synthetic_umbilic(0.0)
    rD = jc.corank_from_chart(X, Y, d, d)
    c3 = rD["corank"] == 2 and rD["s1_min_rel"] < CORANK2_REL
    print(f"    D4 control fires corank-2: s1_min_rel={rD['s1_min_rel']:.4f}  [{'OK' if c3 else 'FAIL'}]")
    X, Y, d = jc.synthetic_swallowtail_chart(0.0)
    rA = jc.corank_from_chart(X, Y, d, d)
    c4 = rA["corank"] == 1
    print(f"    A4 does NOT fire corank-2: s1_min_rel={rA['s1_min_rel']:.4f}  [{'OK' if c4 else 'FAIL'}]")
    # theta_M clearance on the control family's nondegenerate critical points (1-D reduced):
    h, g = -0.40, 0.30
    al = np.roots([4.0, 0.0, 2 * h, g])            # V'(a)=a^4+h a^2+g a -> V''=4a^3+2ha+g roots? no:
    # V(a) = a^5/5 + h a^3/3 + g a^2/2; V'(a) = a^4 + h a^2 + g a = a(a^3 + h a + g)
    crit = [0.0] + [float(r.real) for r in np.roots([1.0, 0.0, h, g]) if abs(r.imag) < 1e-12]
    vpp = [abs(4 * a ** 3 + 2 * h * a + g) for a in crit]
    clear = min(vpp) / (np.median(vpp) + 1e-300)
    c5 = clear >= 10 * THETA_M
    print(f"    theta_M clearance on control criticals: min|V''|/median={clear:.3f} >= "
          f"{10*THETA_M}  [{'OK' if c5 else 'FAIL'}]")
    okall &= c1 and c2 and c3 and c4 and c5

    print("\n(4) chart certificate status:")
    try:
        z = np.load("scripts/tms_liftoff_chart.npz")
        phi_min = float(z["phi_smooth_min"])
        curl = float(z["curl_worst"])
        asym = float(z["asym_worst"])
        smooth = z["smooth"]
        sgn = np.sign(z["phi"])
        cross = np.zeros_like(smooth)
        cross[:-1, :] |= (np.diff(sgn, axis=0) != 0)
        cross[:, :-1] |= (np.diff(sgn, axis=1) != 0)
        n_caustic = int((cross & smooth).sum())
        # O1 candidates live ONLY where the caustic meets the critical-point locus Y = 0
        # (V'=0 AND V''=0 = a fold of actual critical points; elsewhere = tilt-family context)
        sgY = np.sign(z["Y"])
        crY = np.zeros_like(smooth)
        crY[:-1, :] |= (np.diff(sgY, axis=0) != 0)
        crY[:, :-1] |= (np.diff(sgY, axis=1) != 0)
        n_o1 = int((cross & crY & smooth).sum())
        cc = curl <= 1e-7 and asym <= 1e-6
        print(f"    lift-off chart: min phi on SMOOTH cells = {phi_min:.5f}; smooth caustic "
              f"crossings = {n_caustic}; Y=0-COINCIDENT (O1 candidate) cells = {n_o1}; kink "
              f"cells (O4-routed) = {int(z['n_kink'])}; curl_worst(smooth) = {curl:.2e}; "
              f"asym_worst = {asym:.2e}  [{'OK' if cc else 'FAIL'}]")
        okall &= cc
        if n_o1 > 0:
            print("    !! O1 CANDIDATE cells found — classification of those loci required before")
            print("       any O2 verdict (cusp_c3 on the chart + the slope battery).")
            okall = False
        if n_caustic == 0 and phi_min > 0:
            print("    -> EMPTY CAUSTIC: no fold/cusp anywhere on the pinned chart window —")
            print("       confirms the analytic rigid-scaling argument (I_6 scales the entire")
            print("       output-6 piece; the 5+ live config is u-independent, so the lift-off")
            print("       curvature = I_6 x const > 0: NO degeneracy is reachable on this axis).")
    except FileNotFoundError:
        print("    [pending] run --stage chart first (background; writes tms_liftoff_chart.npz)")

    print("\n(5) VERDICT (prereg section 4 partition):")
    if okall:
        print("    O2 — MAXWELL LEVEL-CROSSING: global-minimizer identity changes at u* with BOTH")
        print("    competing critical points present, essential-corank 0, and lambda_min/scale >=")
        print("    theta_M on both sides and AT the crossing. Headline (O1) does NOT fire: the")
        print("    5+ -> 6-gon transition is NOT an elementary catastrophe along the importance")
        print("    axis — it is a level-crossing between coexisting nondegenerate critical points.")
        print("    (O2 is a pre-registered SUCCESS branch; clean null = success.)")
    else:
        print("    ADJUDICATION INCOMPLETE/FAILED — see FAIL lines above (K2/certificate kills).")
    return 0 if okall else 1


# ===== the lift-off chart (slow; run as --stage chart, writes npz) ============ #
def _sym_W(vs, x):
    """Z2-symmetric pentagon + feature 6 pinned on the mirror axis (interior signed coordinate x).
    vs = [lC, lB, lA, beta, gamma, bC, bB, bA, b6]; feature order [C, B+, B-, A+, A-, 6]."""
    lC, lB, lA, beta, gamma, bC, bB, bA, b6 = [vs[i] for i in range(9)]
    zero = vs[0] * 0
    cols = [torch.stack([lC, zero]),
            torch.stack([lB * torch.cos(beta), lB * torch.sin(beta)]),
            torch.stack([lB * torch.cos(beta), -lB * torch.sin(beta)]),
            torch.stack([lA * torch.cos(gamma), lA * torch.sin(gamma)]),
            torch.stack([lA * torch.cos(gamma), -lA * torch.sin(gamma)]),
            torch.stack([-x, zero])]
    W = torch.stack(cols, dim=1)
    b = torch.stack([bC, bB, bB, bA, bA, b6])
    return W, b


def _H_sym(vs, x, u):
    W, b = _sym_W(vs, x)
    I = torch.ones(6, dtype=W.dtype)
    I[5] = u
    return tp.H_cart(W, b, I)


def _signature(W, b):
    """Active-indicator signature of the closed form at (W, b): which delta-regions are live.
    Signature changes between neighboring chart cells = a chamber-boundary (kink) crossing —
    the prereg's smoothness audit; kink cells route to O4, never to the smooth certificate."""
    A = (W.T @ W).detach().numpy()
    bb = b.detach().numpy()
    l2 = np.diag(A)
    bits = []
    bits.append(bb <= 0)
    bits.append((l2 > 0) & (bb >= -l2) & (bb <= 0))
    for i in range(len(bb)):
        for j in range(len(bb)):
            if i != j:
                bits.append(np.array([(A[i, j] > 0) and (-A[i, j] <= bb[i] <= 0),
                                      (bb[i] > 0) and (-A[i, j] > bb[i])]))
    return hash(np.concatenate([np.atleast_1d(x) for x in bits]).tobytes())


def _vred(vs0, x, u):
    """Partial MINIMIZATION over the 9 symmetric coords at pinned (x, u): L-BFGS-B with autograd
    gradients (true minimizer — never a saddle), then envelope-theorem dV_red/dx + signature."""
    from scipy import optimize

    def f_and_g(v):
        t = torch.tensor(v, requires_grad=True)
        Hv = _H_sym(t, torch.tensor(float(x)), u)
        g = torch.autograd.grad(Hv, t)[0].numpy()
        return float(Hv.detach()), g

    res = optimize.minimize(f_and_g, np.array(vs0, float), jac=True, method="L-BFGS-B",
                            options={"maxiter": 300, "ftol": 1e-16, "gtol": 1e-11})
    vs = res.x
    t = torch.tensor(vs)
    xt = torch.tensor(float(x), requires_grad=True)
    Hv = _H_sym(t, xt, u)
    dHdx = float(torch.autograd.grad(Hv, xt)[0])      # envelope theorem: dV_red/dx
    W, b = _sym_W(t, torch.tensor(float(x)))
    return vs, float(Hv.detach()), dHdx, _signature(W, b)


def stage_chart(ng=420, x_lim=0.45, u_lo=0.5, u_hi=2.0):
    """The pinned-protocol lift-off chart F(x,u) = (u, dV_red/dx): V_red by symmetric-subspace
    partial minimization (asymmetry-audited), envelope-theorem Y, curl certificate, phi = dY/dx."""
    print(f"STAGE B chart: ng={ng}, x in [-{x_lim},{x_lim}], u in [{u_lo},{u_hi}] (pinned protocol)")
    ls, bs = tp.KGON_PUBLISHED[5]
    vs0 = np.array([ls, ls, ls, 2 * np.pi / 5, 4 * np.pi / 5, bs, bs, bs, BPLUS])
    xs = np.linspace(-x_lim, x_lim, ng)
    us = np.linspace(u_lo, u_hi, ng)
    V = np.zeros((ng, ng))
    Yenv = np.zeros((ng, ng))
    SIG = np.zeros((ng, ng), dtype=np.int64)
    col_warm = {}
    i0 = int(np.argmin(np.abs(xs)))                   # start from x ~ 0 outward (warm chains)
    order = list(range(i0, ng)) + list(range(i0 - 1, -1, -1))
    for j, u in enumerate(us):
        warm = col_warm.get(j - 1, {})
        mywarm = {}
        for i in order:
            x = xs[i]
            seed = mywarm.get(i - 1 if i > i0 else i + 1, warm.get(i, vs0))
            vs, Vv, dv, sig = _vred(seed, x, u)
            V[i, j], Yenv[i, j], SIG[i, j] = Vv, dv, sig
            mywarm[i] = vs
        col_warm[j] = mywarm
        if j % 40 == 0:
            print(f"  u row {j}/{ng} done (u={u:.3f})")
    # kink audit (prereg smoothness audit): signature changes between neighbors; exclude +-2 cells
    kink = np.zeros((ng, ng), bool)
    kink[:-1, :] |= SIG[:-1, :] != SIG[1:, :]
    kink[1:, :] |= SIG[:-1, :] != SIG[1:, :]
    kink[:, :-1] |= SIG[:, :-1] != SIG[:, 1:]
    kink[:, 1:] |= SIG[:, :-1] != SIG[:, 1:]
    from scipy import ndimage
    smooth = ~ndimage.binary_dilation(kink, iterations=2)
    # curl certificate ON SMOOTH CELLS: envelope Y vs 4th-order FD of V along x
    dx = xs[1] - xs[0]
    fd = (V[:-4, :] - 8 * V[1:-3, :] + 8 * V[3:-1, :] - V[4:, :]) / (12 * dx)
    diff = np.abs(fd - Yenv[2:-2, :]) / max(np.median(np.abs(Yenv)), 1e-12)
    sm_in = smooth[2:-2, :]
    curl_worst = float(diff[sm_in].max()) if sm_in.any() else np.nan
    # asymmetry audit: 16-var ASYMMETRIC minimize at pinned x (5x5 subsample), antisym seed noise
    def H_asym(v, x, u):
        lC, lB1, lB2, lA1, lA2 = v[0], v[1], v[2], v[3], v[4]
        p1, p2, p3, p4 = v[5], v[6], v[7], v[8]       # absolute angles of B+,B-,A+,A- (C at p0)
        p0 = v[9]
        bC, bB1, bB2, bA1, bA2, b6 = v[10], v[11], v[12], v[13], v[14], v[15]
        zero = v[0] * 0
        cols = [torch.stack([lC * torch.cos(p0), lC * torch.sin(p0)]),
                torch.stack([lB1 * torch.cos(p1), lB1 * torch.sin(p1)]),
                torch.stack([lB2 * torch.cos(p2), lB2 * torch.sin(p2)]),
                torch.stack([lA1 * torch.cos(p3), lA1 * torch.sin(p3)]),
                torch.stack([lA2 * torch.cos(p4), lA2 * torch.sin(p4)]),
                torch.stack([-x + zero, zero])]
        W = torch.stack(cols, dim=1)
        b = torch.stack([bC, bB1, bB2, bA1, bA2, b6])
        I = torch.ones(6, dtype=W.dtype)
        I[5] = u
        return tp.H_cart(W, b, I)

    from scipy import optimize as opt
    asym_worst = 0.0
    for iu in np.linspace(0, ng - 1, 5, dtype=int):
        for ix in np.linspace(0, ng - 1, 5, dtype=int):
            vs = col_warm[int(iu)][int(ix)]
            seed = np.array([vs[0], vs[1], vs[1], vs[2], vs[2],
                             vs[3], -vs[3], vs[4], -vs[4], 0.0,
                             vs[5], vs[6], vs[6], vs[7], vs[7], vs[8]])
            seed += 2e-3 * np.sin(np.arange(16) * 2.3)          # deterministic antisym noise

            def fg(v, _x=float(xs[ix]), _u=float(us[iu])):
                t = torch.tensor(v, requires_grad=True)
                Hv = H_asym(t, torch.tensor(_x), _u)
                return float(Hv.detach()), torch.autograd.grad(Hv, t)[0].numpy()

            r = opt.minimize(fg, seed, jac=True, method="L-BFGS-B",
                             options={"maxiter": 300, "gtol": 1e-11})
            # ENERGY-based audit: the symmetric V_red is wrong only if the asymmetric minimizer
            # finds strictly LOWER H (coordinate drift along near-flat modes is benign).
            gain = (V[int(ix), int(iu)] - r.fun) / max(abs(V[int(ix), int(iu)]), 1e-12)
            asym_worst = max(asym_worst, gain)
    phi = np.gradient(Yenv, dx, axis=0)               # phi = d2 V_red / dx2 = det DF of (u, Y)
    phi_smooth_min = float(phi[smooth].min()) if smooth.any() else np.nan
    np.savez("scripts/tms_liftoff_chart.npz", V=V, Y=Yenv, phi=phi, xs=xs, us=us, kink=kink,
             smooth=smooth, curl_worst=curl_worst, asym_worst=asym_worst,
             phi_smooth_min=phi_smooth_min, n_kink=int(kink.sum()))
    print(f"chart done: min phi(all)={phi.min():.5f}  min phi(SMOOTH)={phi_smooth_min:.5f}  "
          f"kink cells={int(kink.sum())}/{ng*ng}  curl_worst(smooth)={curl_worst:.2e}  "
          f"asym_worst={asym_worst:.2e}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="e0", choices=["e0", "classify", "chart"])
    ap.add_argument("--ng", type=int, default=420)
    args = ap.parse_args()
    if args.stage == "e0":
        code, _ = stage_e0()
        return code
    if args.stage == "chart":
        return stage_chart(ng=args.ng)
    if args.stage == "classify":
        return stage_classify()
    return 2


if __name__ == "__main__":
    sys.exit(main())
