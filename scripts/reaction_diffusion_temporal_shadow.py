#!/usr/bin/env python
"""H8 v3 / substrate S3tau — LOAD-BEARING temporal-phase resist (Milestone-1).

Successor to the H8 v2 SCOPED result (the RD was decorative: a bare vortex reproduced every gate, because
the latent was a rotational phase = the SO(2) symmetry-orbit coordinate). v3 fixes it: the continuous
RESIST latent is the spiral's TEMPORAL phase tau (snapshot integration time) read from REAL CGL frames; the
lossiness is jitter-in-TIME. Load-bearing-ness is a PRE-REGISTERED ablation battery (docs/atlas/
H8V3_RD_LOADBEARING_PREREG.md): a static / rotation / time-warp surrogate must FAIL to reproduce the resist
(measured by the non-rigidity residual + a cross-application test). Go/no-go crux confirmed the equilibrated
CGL spiral is GENUINELY non-rigid (residual 0.7-1.2 even after rotation+translation), so tau is NOT a
symmetry coordinate. Milestone-1 = the resist + battery; the chirality determine is carried (LB-disc
reported honestly). NOT public-eligible. Attribution: Aranson & Kramer (CGL); Shadow/charFun laws; S0-S2.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.random import default_rng
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
CV, PROBE_SEED = 4, 0
DATA_SEED, CALIB_SEED = 20260609, 999
CONT0_MIN, CONT_MAX_MAX = 0.70, 0.10

CFG = dict(
    grid=64, b1=0.0, c1=-1.4, dt=0.05, r0=0.15,        # BF-stable but NON-RIGID regime (crux: residual 0.73)
    settle=1500, nframes=50, stride=5,                 # trajectory: settle then record nframes at stride
    K=8, n=140, noise=0.10, down=10, r_amp=20,         # shadow / feature
    t_lo=0.0, t_hi=12.0, jit=5.0,                      # continuous latent = frame-time (SUB-period window; no phase wrap)
    r_bases=10,
)

GRID = None
LIB = {}          # base -> dict(frames_u, frames_w [nframes,G,G], omega)
RADIDX = None
NFEAT = None


# ============================ CGL trajectory ==================================================== #
def _lap(a):
    return np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4.0 * a


def cgl_traj(b1, c1, settle, nframes, stride, grid, rng):
    g = np.linspace(-1, 1, grid); X, Y = np.meshgrid(g, g)
    r = np.sqrt(X ** 2 + Y ** 2); th = np.arctan2(Y, X)
    A = (np.tanh(r / CFG["r0"]) * np.exp(1j * (th + 0.3 * rng.standard_normal()))).astype(np.complex128)
    dt = CFG["dt"]
    for _ in range(settle):
        A += dt * (A + (1 + 1j * b1) * _lap(A) - (1 + 1j * c1) * np.abs(A) ** 2 * A)
    fu = np.empty((nframes, grid, grid), np.float32); fw = np.empty((nframes, grid, grid), np.float32)
    for f in range(nframes):
        for _ in range(stride):
            A += dt * (A + (1 + 1j * b1) * _lap(A) - (1 + 1j * c1) * np.abs(A) ** 2 * A)
        fu[f] = A.real; fw[f] = A.imag
    return fu, fw


def _omega(fu, fw):
    """rotation rate (deg/frame) from the m=1 azimuthal phase shift between consecutive frames."""
    c = GRID // 2; yy, xx = np.mgrid[0:GRID, 0:GRID]
    rr = np.sqrt((xx - c) ** 2 + (yy - c) ** 2); th = np.arctan2(yy - c, xx - c)
    ring = (rr > 0.25 * c) & (rr < 0.55 * c)
    ph = [np.angle(np.sum((fu[f][ring] + 1j * fw[f][ring]) * np.exp(-1j * th[ring]))) for f in range(len(fu))]
    d = np.diff(np.unwrap(ph))
    return float(np.degrees(np.median(d)))


# ============================ features ========================================================== #
def _setup(grid):
    global RADIDX, NFEAT
    c = grid // 2; yy, xx = np.mgrid[0:grid, 0:grid]
    rr = np.clip((np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / c * CFG["r_amp"]).astype(int), 0, CFG["r_amp"] - 1)
    RADIDX = rr
    NFEAT = CFG["down"] ** 2 + CFG["r_amp"]


def feature(u, w):
    """downsampled field (rotation+structure-sensitive) ‖ radial |A|(r) profile (rotation-INVARIANT;
    captures the non-rigid breathing/wave-field amplitude dynamics a pure rotation cannot fake)."""
    D = CFG["down"]; g = u.shape[0]
    ds = u[:(g // D) * D, :(g // D) * D].reshape(D, g // D, D, g // D).mean((1, 3))
    amp = np.sqrt(u ** 2 + w ** 2)
    prof = np.bincount(RADIDX.ravel(), amp.ravel(), minlength=CFG["r_amp"]) / \
        (np.bincount(RADIDX.ravel(), minlength=CFG["r_amp"]) + 1e-9)
    return np.concatenate([(ds - ds.mean()).ravel(), prof])


# ============================ library ========================================================== #
def build_library(grid, seed):
    global GRID, LIB
    GRID = grid; _setup(grid)
    LIB = {}
    for b in range(CFG["r_bases"]):
        fu, fw = cgl_traj(CFG["b1"], CFG["c1"], CFG["settle"], CFG["nframes"], CFG["stride"], grid,
                          default_rng(seed + 7 * b))
        LIB[b] = dict(fu=fu, fw=fw, omega=_omega(fu, fw))
    print(f"    [lib] {CFG['r_bases']} CGL trajectories x {CFG['nframes']} frames; "
          f"<omega>={np.mean([LIB[b]['omega'] for b in LIB]):.1f} deg/frame", flush=True)


def _frame_feat(lib, t, mode="real", ref=None):
    """feature of the field at frame-time t under a given SOURCE mode (the ablation battery):
       real     = the genuine CGL snapshot (interp to fractional frame);
       static   = frame 0 frozen (no dynamics);
       rotation = frame 0 rigidly ROTATED by omega*t (the v2 SO(2) surrogate);
       timewarp = a single reference trajectory resampled at warped time (reparametrization)."""
    nf = CFG["nframes"]
    if mode == "static":
        return feature(lib["fu"][0], lib["fw"][0])
    if mode == "rotation":
        deg = lib["omega"] * t
        u = ndimage.rotate(lib["fu"][0], deg, reshape=False, mode="grid-wrap", order=1)
        w = ndimage.rotate(lib["fw"][0], deg, reshape=False, mode="grid-wrap", order=1)
        return feature(u, w)
    src = ref if mode == "timewarp" else lib                  # timewarp reads ONE reference trajectory
    tt = (t * 1.37) % (nf - 1) if mode == "timewarp" else t    # nonlinear time reparametrization
    i = int(np.clip(tt, 0, nf - 1.001)); a = tt - i
    u = (1 - a) * src["fu"][i] + a * src["fu"][i + 1]
    w = (1 - a) * src["fw"][i] + a * src["fw"][i + 1]
    return feature(u, w)


def gen_c(n, lam, rng, noise, mode="real", ref_base=None):
    """temporal-phase RESIST: xc = frame-time t0; shadow = mean over K subunits at jittered TIMES."""
    K = CFG["K"]
    xc = rng.uniform(CFG["t_lo"], CFG["t_hi"], n)
    feats = np.empty((n, NFEAT))
    ref = LIB[ref_base if ref_base is not None else 0] if mode == "timewarp" else None
    for i in range(n):
        lib = LIB[int(rng.integers(0, CFG["r_bases"]))]
        ts = np.clip(xc[i] + lam * CFG["jit"] * rng.standard_normal(K), 0, CFG["nframes"] - 1.001)
        feats[i] = np.mean([_frame_feat(lib, t, mode, ref) for t in ts], axis=0)
    return feats + rng.normal(0, noise, feats.shape), xc


# ============================ probe ============================================================ #
def cont_recovery(X, y):
    Xs = StandardScaler().fit_transform(X); kf = KFold(CV, shuffle=True, random_state=PROBE_SEED)
    lin = float(cross_val_score(LinearRegression(), Xs, y, cv=kf, scoring="r2").mean())
    mlp = float(cross_val_score(MLPRegressor(hidden_layer_sizes=(64,), max_iter=600, random_state=0),
                                Xs, y, cv=kf, scoring="r2").mean())
    return max(0.0, lin, mlp)


def cross_test(Xtrain, ytrain, Xtest, ytest):
    """train cont on real features, predict t0 on a SURROGATE's features.
    *** RETRACTED as a load-bearing discriminator (see H8V3_RD_LOADBEARING_RESULT.md): this measures
    feature-distribution OVERLAP, not mechanism necessity -- a real-RD field with merely column-shuffled
    features cross-scores ~0 while tau is fully recoverable within that distribution. v3 is a NULL. ***"""
    sc = StandardScaler().fit(Xtrain)
    m = MLPRegressor(hidden_layer_sizes=(64,), max_iter=800, random_state=0).fit(sc.transform(Xtrain), ytrain)
    pred = m.predict(sc.transform(Xtest))
    ss_res = ((ytest - pred) ** 2).sum(); ss_tot = ((ytest - ytest.mean()) ** 2).sum()
    return float(max(0.0, 1 - ss_res / (ss_tot + 1e-9)))


def half_life(curve):
    b = curve[0]
    if b <= 0:
        return None
    for lam, v in zip(LAMBDAS, curve):
        if v <= 0.5 * b:
            return lam
    return None


def k_invariance(seed, noise, lam_test=2.0, ks=(8, 64, 512)):
    base = CFG["K"]; out = {}
    for kk in ks:
        CFG["K"] = kk
        out[str(kk)] = round(cont_recovery(*gen_c(CFG["n"], lam_test, default_rng(seed + kk + 31), noise)), 4)
    CFG["K"] = base
    vals = list(out.values())
    return {"cont_vs_K": out, "is_charfun_resist": bool(max(vals) <= 0.15 and vals[-1] - vals[0] <= 0.10)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true"); ap.add_argument("--frozen", action="store_true")
    ap.add_argument("--out", default="results/atlas/h8v3")
    args = ap.parse_args()
    if args.frozen:
        CFG.update(grid=80, n=200, r_bases=14); seed, mode = DATA_SEED, "frozen"
    else:
        seed, mode = CALIB_SEED, "calibrate"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    print(f"[cfg-S3tau] mode={mode} seed={seed} grid={CFG['grid']} n={CFG['n']} K={CFG['K']} "
          f"noise={CFG['noise']} regime=({CFG['b1']},{CFG['c1']})", flush=True)
    build_library(CFG["grid"], seed)

    # resist sweep (REAL dynamics)
    print("S3tau-c (temporal-phase RESIST, real CGL):", flush=True)
    cont = []
    for lam in LAMBDAS:
        c = cont_recovery(*gen_c(CFG["n"], lam, default_rng(seed + int(lam * 1000) + 7), CFG["noise"]))
        cont.append(round(c, 4)); print(f"  [real lam={lam:<4}] cont={c:.3f}", flush=True)
    lc = half_life(cont)
    kinv = k_invariance(seed, CFG["noise"])

    # LOAD-BEARING ablation battery
    print("LOAD-BEARING ablations (each MUST fail if the RD dynamics are load-bearing):", flush=True)
    Xr, yr = gen_c(CFG["n"], 0.0, default_rng(seed + 1), CFG["noise"])           # real, lam=0
    c0_static = cont_recovery(*gen_c(CFG["n"], 0.0, default_rng(seed + 2), CFG["noise"], mode="static"))
    Xrot, yrot = gen_c(CFG["n"], 0.0, default_rng(seed + 3), CFG["noise"], mode="rotation")
    c0_rot = cont_recovery(Xrot, yrot)
    cross_rot = cross_test(Xr, yr, Xrot, yrot)                                   # real-trained -> rotation frames
    Xtw, ytw = gen_c(CFG["n"], 0.0, default_rng(seed + 4), CFG["noise"], mode="timewarp")
    cross_tw = cross_test(Xr, yr, Xtw, ytw)
    print(f"  A1 static cont0={c0_static:.3f} (load-bearing: <0.30)", flush=True)
    print(f"  A2 rotation: own cont0={c0_rot:.3f}; CROSS (real-trained on rotation frames) R2={cross_rot:.3f} "
          f"(load-bearing: cross<0.40)", flush=True)
    print(f"  A3 time-warp: CROSS R2={cross_tw:.3f} (load-bearing: cross<0.40)", flush=True)

    g = {
        "G1_preflight": cont[0] >= CONT0_MIN, "G3_resists": cont[-1] <= CONT_MAX_MAX and lc is not None,
        "G_KINV_charfun": kinv["is_charfun_resist"],
        "LB_static_fails": c0_static < 0.30, "LB_rotation_fails": cross_rot < 0.40,
        "LB_timewarp_fails": cross_tw < 0.40,
    }
    g["LB_cont"] = g["LB_static_fails"] and g["LB_rotation_fails"] and g["LB_timewarp_fails"]
    g["LOADBEARING_RESIST"] = (g["G1_preflight"] and g["G3_resists"] and g["G_KINV_charfun"] and g["LB_cont"])
    print(f"\n== {mode.upper()} S3tau ==")
    print(f"  resist: cont0={cont[0]} contMax={cont[-1]} lam*={lc} charFun={g['G_KINV_charfun']} (cont_vs_K={kinv['cont_vs_K']})")
    print(f"  LB-cont: static_fails={g['LB_static_fails']} rotation_fails={g['LB_rotation_fails']} timewarp_fails={g['LB_timewarp_fails']}")
    print(f"  ** LOAD-BEARING RESIST = {g['LOADBEARING_RESIST']} **")
    if mode == "calibrate":
        print(f"  CALIBRATION {'PASSES -> freeze + frozen' if g['LOADBEARING_RESIST'] else 'NULL/SCOPED (see which gate fails)'}")
    (out / f"{mode}.json").write_text(json.dumps(
        {"mode": mode, "seed": seed, "cfg": CFG, "cont": cont, "lambda_star": lc, "k_invariance": kinv,
         "ablations": {"static_cont0": c0_static, "rotation_own_cont0": c0_rot, "rotation_cross_r2": cross_rot,
                       "timewarp_cross_r2": cross_tw}, "gates": g, "wall_s": round(time.time() - t0, 1)}, indent=2,
        default=lambda o: bool(o) if isinstance(o, np.bool_) else o))
    print(f"  wrote {out/(mode+'.json')}  ({round(time.time()-t0,1)}s)", flush=True)


if __name__ == "__main__":
    main()
