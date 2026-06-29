#!/usr/bin/env python3
"""The Resistant-Body Test (docs/RESISTANT_BODY_TEST_PREREG.md): does regime-2
sharpness track LOW RECOVERABILITY (the shadow can't rebuild the body) or HIGH
DIMENSION? Deconfounds the two — flat-Gaussian body, rank-r linear shadow, control
functional kept inside the shadow. CONSTRUCTED/toy; not an NSE result. Run:
  python scripts/resistant_body_test.py
"""
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

N = 4000
DS = [8, 32, 128]
RHOS = [0.25, 0.50, 0.75, 0.90]
rng = np.random.default_rng(20260629)


def cv_r2(X, y):
    kf = KFold(4, shuffle=True, random_state=0)
    return float(cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2").mean())


def recon_fve(S, X):
    """mean CV R2 of reconstructing each body coord from the shadow (clip<0 to 0)."""
    kf = KFold(4, shuffle=True, random_state=0)
    scores = []
    for j in range(X.shape[1]):
        scores.append(max(0.0, cross_val_score(LinearRegression(), S, X[:, j], cv=kf, scoring="r2").mean()))
    return float(np.mean(scores))


def recon_fve_mlp(S, X, ncols=6):
    """compute-can't-cross check: a 2x-ish nonlinear reconstructor on a few dropped coords."""
    kf = KFold(4, shuffle=True, random_state=0)
    cols = list(range(X.shape[1]))[-ncols:]  # the dropped/back coords
    scores = []
    for j in cols:
        s = cross_val_score(MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=400, random_state=0),
                            S, X[:, j], cv=kf, scoring="r2").mean()
        scores.append(max(0.0, s))
    return float(np.mean(scores))


def run_cell(D, rho):
    r = max(2, round(rho * D))
    X = rng.standard_normal((N, D))
    S = X[:, :r]                                  # rank-r linear shadow = first r coords
    y = X[:, 0] + X[:, 1] + 0.1 * rng.standard_normal(N)   # control functional, inside S
    ctrl = cv_r2(S, y)
    rec = recon_fve(S, X)
    sharp = ctrl * (1 - rec)
    return r, ctrl, rec, sharp


def main():
    print(f"RESISTANT_BODY_TEST   N={N}   grid D x rho_rec   (sharpness = control_suff * (1 - recon_FVE))\n")
    print("   D    rho_rec   r    control_suff   recon_FVE(=r/D?)   sharpness")
    grid = {}
    for D in DS:
        for rho in RHOS:
            r, ctrl, rec, sharp = run_cell(D, rho)
            grid[(D, rho)] = (ctrl, rec, sharp)
            print(f"  {D:>3}    {rho:.2f}    {r:>3}     {ctrl:.3f}          {rec:.3f} (r/D={r/D:.3f})     {sharp:.3f}")

    # (3) flatness in D at matched rho_rec
    print("\n  D-flatness at matched rho_rec (sharpness across D; should be ~constant):")
    for rho in RHOS:
        vals = [grid[(D, rho)][2] for D in DS]
        print(f"    rho_rec={rho:.2f}: sharpness over D{DS} = {[round(v,3) for v in vals]}  spread={max(vals)-min(vals):.3f}")

    # (2) compute-can't-cross: bigger reconstructor can't beat the info floor on dropped coords
    D, rho = 32, 0.50
    r = max(2, round(rho * D))
    X = rng.standard_normal((N, D)); S = X[:, :r]
    mlp_back = recon_fve_mlp(S, X)
    print(f"\n  compute-can't-cross (D={D}, rho={rho}, dropped coords): "
          f"MLP recon R2 on dropped coords = {mlp_back:.3f}  (info floor = 0 ; >~0 would falsify)")

    # (4) decisive cells
    netlike = grid[(128, 0.90)]
    ablike = grid[(8, 0.25)]
    print("\n  DECISIVE:")
    print(f"    high-D recoverable (D=128, rho=0.90, 'net.7' analogue): sharpness={netlike[2]:.3f}  "
          f"(control={netlike[0]:.3f}, recon_FVE={netlike[1]:.3f})  -> MARGINAL")
    print(f"    low-D resistant   (D=8,  rho=0.25, 'AB' analogue):      sharpness={ablike[2]:.3f}  "
          f"(control={ablike[0]:.3f}, recon_FVE={ablike[1]:.3f})  -> SHARP")

    # verdict against the frozen prediction
    flat = all((max(grid[(D, rho)][2] for D in DS) - min(grid[(D, rho)][2] for D in DS)) < 0.08 for rho in RHOS)
    rises = all(grid[(DS[1], 0.25)][2] > grid[(DS[1], 0.90)][2] for _ in [0])
    ctrl_ok = all(grid[k][0] > 0.8 for k in grid)
    floor_ok = mlp_back < 0.1
    print("\n  CHECKS vs frozen prediction:")
    print(f"    control_suff ~1 across grid: {ctrl_ok}")
    print(f"    recon_FVE info floor not beaten by MLP: {floor_ok}")
    print(f"    sharpness flat in D at matched rho_rec: {flat}")
    print(f"    sharpness rises as rho_rec falls (fixed D): {rises}")
    ok = ctrl_ok and floor_ok and flat and rises and netlike[2] < 0.2 < ablike[2]
    print(f"\n  VERDICT: {'PREDICTION HELD — recoverability is the axis, dimension is NOT' if ok else 'see numbers (prediction did not cleanly hold)'}")


if __name__ == "__main__":
    main()
