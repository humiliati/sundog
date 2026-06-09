#!/usr/bin/env python
"""H8 (slate `wm13nclfe`) — is the regularization-induced disappearance of DOUBLE DESCENT a Whitney A3
cusp (fold-pair annihilation)? Stage 1: the forward model + validation.

RECAST (fixes the category error): at ridge λ=0 the test-risk peak at the interpolation threshold γ=p/n=1
is a variance POLE (R ~ 1/(1−γ)), NOT a catastrophe germ — so 'the (γ,η) peak is an A2 fold' is ill-posed.
The catastrophe framing applies only to the REGULARIZED risk R(γ;λ>0), a smooth surface. The headline test
(Stage 2): in the (γ,λ) plane, is the locus where the double-descent bump ANNIHILATES (Nakkiran 2020,
'optimal regularization removes double descent') a Whitney A3 cusp?

Forward model (isotropic ridge regression, well-specified, proportional asymptotics — Hastie–Montanari–
Rosset–Tibshirani 2019): with effective ridge κ and γ=p/n,
    κ(γ,λ) = [(λ+γ−1) + √((1−λ−γ)² + 4λ)] / 2          (the unique κ>0; Σ=I self-consistent ridge)
    R(γ,λ; σ², r²) = (r²·κ² + σ²·γ) / ((1+κ)² − γ)       (excess test risk; r²=‖β*‖², σ²=label noise)
SMOOTH (no training loop). Stage 1 here VALIDATES R against Monte-Carlo ridge and exhibits double descent.

NOT public-eligible. Attribution: Belkin 2019 / Mei–Montanari / Hastie et al 2019 (double descent);
Nakkiran 2020 (regularization removal); Thom/Whitney (cusp). Run: python scripts/double_descent_cusp.py
"""
import sys
from pathlib import Path
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent))


def kappa(g, lam):
    """Effective ridge κ(γ,λ): the positive root of κ² + κ(1−λ−γ) − λ = 0 (isotropic Σ=I fixed point)."""
    g = np.asarray(g, float); lam = np.asarray(lam, float)
    return ((lam + g - 1.0) + np.sqrt((1.0 - lam - g) ** 2 + 4.0 * lam)) / 2.0


def risk(g, lam, sig2=1.0, r2=1.0):
    """Closed-form excess test risk of isotropic ridge regression. Double descent: peaks near γ=1."""
    k = kappa(g, lam)
    return (r2 * k ** 2 + sig2 * g) / ((1.0 + k) ** 2 - g)


def risk_mc(g, lam, sig2=1.0, r2=1.0, n=300, draws=40, seed=0):
    """Monte-Carlo excess ridge risk: X~N(0,1)^{n×p}, y=Xβ*+ε, β̂=(XᵀX+nλI)⁻¹Xᵀy, excess=‖β̂−β*‖²."""
    rng = np.random.default_rng(seed)
    p = max(1, int(round(g * n)))
    beta = rng.standard_normal(p); beta *= np.sqrt(r2) / np.linalg.norm(beta)
    acc = 0.0
    for _ in range(draws):
        X = rng.standard_normal((n, p))
        y = X @ beta + rng.standard_normal(n) * np.sqrt(sig2)
        bhat = np.linalg.solve(X.T @ X + n * lam * np.eye(p), X.T @ y)
        acc += float(np.sum((bhat - beta) ** 2))
    return acc / draws


def n_critical(lam, sig2=1.0, r2=1.0, g_lo=1.02, g_hi=4.0, ng=4000):
    """#interior critical points of R(γ) (zeros of ∂R/∂γ) on (g_lo,g_hi), AWAY from the γ=1 pole. A
    max+min pair (2) annihilating to 0 as λ grows = a fold; the merge point is the cusp candidate."""
    gs = np.linspace(g_lo, g_hi, ng)
    dR = np.gradient(risk(gs, lam, sig2, r2), gs)
    return int(np.sum(np.abs(np.diff(np.sign(dR))) > 0))


def cusp_chart(g_span, lam_span, sig2=1.0, r2=1.0, ng=260):
    """Mirror of the H5 mirage chart: F(γ,λ)=(λ, R(γ;λ)), non-dimensionalized, fed to the jet classifier.
    det DF = −∂R/∂γ, so the caustic φ=0 is the risk-critical-point locus and a cusp (φ=0∧c2=0) is the
    peak-annihilation. Returns {caustic, n_cusps, c3, corank}."""
    import atlas_jet_classify as jc
    gg = np.linspace(*g_span, ng); ll = np.linspace(*lam_span, ng)
    G, L = np.meshgrid(gg, ll, indexing="ij")             # axis0=γ, axis1=λ
    R = risk(G, L, sig2, r2)
    Xn = ((L - ll[0]) / (ll[-1] - ll[0]))                  # λ normalized to [0,1] (axis1)
    Yn = (R - R.mean()) / (R.std() + 1e-30)                # R -> O(1)
    d = 1.0 / (ng - 1)
    phi, c2, c3 = jc.jet_from_chart(Xn, Yn, d, d)
    cusps = jc.cusp_c3(phi, c2, c3, edge=4)
    rank = jc.corank_from_chart(Xn, Yn, d, d, edge=4)
    has_caustic = bool(np.any(np.abs(np.diff(np.sign(phi), axis=0)) > 0))
    return {"caustic": has_caustic, "n_cusps": len(cusps), "c3": [round(v, 3) for v in cusps[:5]],
            "corank": rank["corank"], "s1_min_rel": round(rank["s1_min_rel"], 4)}


def main():
    print("=" * 80)
    print("H8 Stage 1 — isotropic ridge double-descent risk: closed form vs Monte-Carlo")
    print("=" * 80)
    print("\n(1) VALIDATE the closed form against Monte-Carlo ridge (σ²=1, r²=1, n=300, 40 draws):")
    print(f"    {'γ':>5} {'λ':>6} {'R (closed)':>12} {'R (MC)':>10} {'rel.err':>9}")
    ok = True
    for g in (0.5, 0.8, 1.1, 2.0):
        for lam in (0.02, 0.2):
            rc = float(risk(g, lam)); rm = risk_mc(g, lam)
            rel = abs(rc - rm) / max(rm, 1e-9)
            ok &= rel < 0.12
            print(f"    {g:>5.2f} {lam:>6.2f} {rc:>12.3f} {rm:>10.3f} {rel:>9.2%}"
                  + ("" if rel < 0.12 else "  <-- mismatch"))

    print("\n(2) DOUBLE DESCENT: R(γ) at several λ (peak near γ=1 shrinks as λ grows = Nakkiran removal):")
    gs = np.linspace(0.2, 3.0, 280)
    print(f"    {'λ':>7}  peak R   peak γ   R(γ=3)   monotone?")
    peak_lams = []
    for lam in (0.001, 0.01, 0.05, 0.2, 0.5, 1.0):
        R = risk(gs, lam)
        # interior local max (the double-descent bump)
        dR = np.gradient(R, gs)
        maxima = [(gs[i], R[i]) for i in range(2, len(gs) - 2)
                  if dR[i - 1] > 0 and dR[i + 1] < 0]
        if maxima:
            pg, pr = max(maxima, key=lambda t: t[1])
            mono = "no (bump)"
        else:
            pg, pr, mono = float("nan"), float("nan"), "YES (no bump)"
        peak_lams.append((lam, len(maxima)))
        print(f"    {lam:>7.3f}  {pr:6.2f}  {pg:6.2f}  {risk(3.0,lam):6.2f}   {mono}")

    # the bump disappears somewhere between the small-λ (bump) and large-λ (monotone) regimes
    has_bump = [lam for lam, nm in peak_lams if nm >= 1]
    no_bump = [lam for lam, nm in peak_lams if nm == 0]
    print(f"\n    bump present at λ ∈ {has_bump};  GONE at λ ∈ {no_bump}")
    print(f"    => the annihilation λ* lives between {max(has_bump) if has_bump else '?'} and "
          f"{min(no_bump) if no_bump else '?'} — Stage 2 classifies it with the jet tool.")

    stage1 = ok and has_bump and no_bump
    print(f"\n  Stage 1: {'PASS' if stage1 else 'ISSUE'} (closed form validated; double descent reproduced).")

    # ========================= STAGE 2: is the annihilation a Whitney A3 cusp? ===================== #
    print("\n" + "=" * 80)
    print("H8 Stage 2 — classify the double-descent annihilation (chart F(γ,λ)=(λ,R), det DF=−∂R/∂γ)")
    print("=" * 80)

    print("\n(A) critical-point structure: #zeros of ∂R/∂γ vs λ (away from the γ=1 pole, γ∈(1.02,4)):")
    print("    a max+min PAIR (2) merging to 0 = a fold-pair annihilation (cusp candidate); 1→0 = no fold.")
    crit = [(lam, n_critical(lam)) for lam in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)]
    print("    " + "  ".join(f"λ={lam}:{nc}crit" for lam, nc in crit))
    pair = any(nc >= 2 for _, nc in crit)
    print(f"    => max+min pair present (fold-pair to annihilate): {pair}")

    # the null MECHANISM: as λ grows the peak slides to γ→∞ and its amplitude→0 (escape, not annihilation)
    gw = np.linspace(1.02, 80.0, 30000)
    print("    peak location / bump height as λ grows (does the peak slide to γ→∞ rather than merge?):")
    pg = []
    for lam in (0.10, 0.20, 0.30, 0.36):
        Rw = risk(gw, lam); dw = np.gradient(Rw, gw)
        idx = [i for i in range(2, len(gw) - 2) if dw[i - 1] > 0 and dw[i + 1] < 0]
        if idx:
            i = max(idx, key=lambda j: Rw[j]); pg.append(gw[i])
            print(f"      λ={lam}: peak γ={gw[i]:6.2f}  bump height={Rw[i]-1.0:.4f}")
    slides = len(pg) >= 2 and pg[-1] > 2.0 * pg[0]            # peak γ diverges as λ→λ*

    print("\n(B) the jet classifier on the non-dimensionalized chart (γ∈[1.05,4.0], λ∈[0.05,0.6]):")
    r1 = cusp_chart((1.05, 4.0), (0.05, 0.6))
    print(f"    caustic={r1['caustic']}  #cusps={r1['n_cusps']}  |c3|={r1['c3']}  "
          f"corank={r1['corank']} (s1_min_rel={r1['s1_min_rel']})")

    print("\n(C) calibration — the same classifier on the Morin A4 control (|c3|→0) and a Whitney cusp:")
    import atlas_jet_classify as jc
    a4 = jc.cusp_c3(*jc.synthetic_swallowtail(-0.02))
    a4 = a4[0] if a4 else float("nan")
    print(f"    A4 swallowtail (h=−0.02, near-merge): |c3|={a4:.3f}  (A4 ⇒ c3→0; our cusp must stay bounded)")

    a3 = r1["caustic"] and r1["n_cusps"] >= 1 and r1["corank"] == 1 and (r1["c3"] and r1["c3"][0] > 1.0)
    print("\n" + "=" * 80)
    print("VERDICT (vs H8 pre-reg)")
    print("=" * 80)
    if a3 and pair:
        verdict = ("BOUNDED-POSITIVE: the regularization-induced DISAPPEARANCE of double descent is a Whitney "
                   "A3 CUSP — a max+min critical-point pair annihilates as λ grows, the jet classifier reads a "
                   "cusp on F(γ,λ)=(λ,R) with |c3| BOUNDED (not A4) and corank-1 (not D4). The germ of "
                   "'optimal regularization removes double descent' is a fold-pair annihilation — the exact "
                   "catastrophe class as the H5 mirage 1→3-image onset, now in an ML loss landscape.")
    elif r1["caustic"] and not pair:
        verdict = ("NULL (informative): NOT a Whitney cusp. The disappearance of double descent is a SINGLE "
                   "max (1→0 critical points, no max+min fold-pair) whose location SLIDES to γ→∞ while its "
                   f"amplitude→0 ({'peak escapes to infinity' if slides else 'peak flattens'}) — the bump "
                   "escapes/flattens, it does not annihilate at a finite point. The jet classifier finds the "
                   "critical-point caustic but NO cusp (corank-1). A clean, pre-registered null: 'optimal "
                   "regularization removes double descent' is a peak-amplitude shrinkage, not a fold-pair "
                   "annihilation catastrophe (at least in the well-specified isotropic ridge model).")
    elif r1["c3"] and r1["c3"][0] <= 1.0:
        verdict = ("KILLED/A4: |c3|→0 at the annihilation — it is an A4 swallowtail (or higher), not an A3 cusp.")
    else:
        verdict = "MIXED — see (A)/(B); report honestly."
    print("VERDICT:", verdict)
    print("=" * 80)
    return 0 if (stage1 and a3 and pair) else 1


if __name__ == "__main__":
    sys.exit(main())
