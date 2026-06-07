#!/usr/bin/env python
"""Phase-5 lossiness-crossover slate — S0/S1 harness.

Implements docs/proof/PHASE5_CROSS_SUBSTRATE.md §3. Model-free (numpy + sklearn). Tests the candidate
Shadow-Invertibility operator: as ensemble lossiness lambda grows, does CONTINUOUS recovery decay to
chance while DISCRETE recovery stays exact, on BOTH a 1-D caustic toy (S0) and a 2-D vector field (S1)?

Modes:
  --calibrate : throwaway seed 999, n=500 — the pre-freeze power check (tune scale/power only).
  --frozen    : frozen data_seed 20260605, n=2000 — the real run (only after freeze sign-off).

Metrics (frozen): cont = max(0, R2_cv) vs mean baseline; disc = (acc_cv - maj)/(1 - maj).
Verdict policy: BEST-OF {linear, MLP} for each metric (both reported).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from numpy.random import default_rng
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ---- FROZEN constants (§3.7) ------------------------------------------------ #
LAMBDAS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00]
K = 64
CV = 4
DATA_SEED, PROBE_SEED, CALIB_SEED = 20260605, 0, 999
# S0 (v2: band-pass fringe envelope — t0_f/w_f place the size-bearing fringe OFF-centre, ~0 at t=0)
S0 = dict(T=64, w=0.5, w_b=0.1, t0=0.0, t0_f=0.50, w_f=0.18, A=1.0, C=1.0, D=0.5,
          f_p=8.0, xc_lo=3.0, xc_hi=7.0)
# S1
S1 = dict(G=16, eps=0.10, A=1.0, B=1.0, f0=3.0, xc_lo=-1.0, xc_hi=1.0)
NOISE = 0.30                       # obs-noise (calibration target)
# Gates (§3.5)
CONT0_MIN, DISC0_MIN = 0.70, 0.95
CONT_MAX_MAX, DISC_MIN_MIN = 0.10, 0.95


def _std(X):
    return StandardScaler().fit_transform(X)


# --------------------------------------------------------------------------- #
def gen_s0(n, lam, rng, noise):
    c = S0
    t = np.linspace(-1, 1, c["T"])
    env_g = np.exp(-t ** 2 / (2 * c["w"] ** 2))                       # central Gaussian (parity channel)
    # v2: band-pass fringe envelope — vanishes at t=0, peaks at |t|=t0_f, so the size-bearing
    # fringe lives ONLY in the off-centre band and Debye-Waller-damps fully (no central leak).
    env_f = np.exp(-(np.abs(t) - c["t0_f"]) ** 2 / (2 * c["w_f"] ** 2))      # (T,)
    bump = np.exp(-(t - c["t0"]) ** 2 / (2 * c["w_b"] ** 2))
    xc = rng.uniform(c["xc_lo"], c["xc_hi"], n)                       # continuous label (fringe freq)
    xd = rng.choice([-1.0, 1.0], n)                                  # discrete label (parity sign)
    xi = rng.standard_normal((n, K))
    xci = xc[:, None] + lam * xi                                     # (n,K) per-subunit
    # fringe averaged over K subunits: mean_i cos(2*pi*xci*t)
    fringe = np.cos(2 * np.pi * xci[:, :, None] * t[None, None, :]).mean(1)   # (n,T)
    parity = (xd[:, None] * np.sin(2 * np.pi * c["f_p"] * t)[None, :])        # (n,T)
    sig = (c["D"] * bump[None, :]                                     # scale-free central geometric halo
           + c["A"] * fringe * env_f[None, :]                        # off-centre size fringe (v2 band-pass)
           + c["C"] * parity * env_g[None, :])                       # central parity channel
    sig = sig + rng.normal(0, noise, sig.shape)
    return sig, xc, xd                                              # features (n,T)


def gen_s1(n, lam, rng, noise):
    c = S1
    g = np.linspace(-1, 1, c["G"])
    px, py = np.meshgrid(g, g, indexing="xy")
    p = np.stack([px.ravel(), py.ravel()], 1)                       # (P,2), P=G*G
    r = np.linalg.norm(p, axis=1)
    rs = np.maximum(r, 1e-6)
    rhat = p / rs[:, None]                                          # radial unit (P,2)
    that = np.stack([-p[:, 1], p[:, 0]], 1) / rs[:, None]           # tangential unit (P,2)
    xc = rng.uniform(c["xc_lo"], c["xc_hi"], n)                     # continuous label (phase)
    xd = rng.choice([-1.0, 1.0], n)                                # discrete label (winding sign)
    xi = rng.standard_normal((n, K))
    xci = xc[:, None] + lam * xi                                    # (n,K)
    # radial texture averaged over subunits: mean_i cos(2*pi*f0*r + xci)
    tex = np.cos(2 * np.pi * c["f0"] * r[None, None, :] + xci[:, :, None]).mean(1)   # (n,P)
    Vx = c["A"] * tex * rhat[None, :, 0] + xd[:, None] * c["B"] * that[None, :, 0] / (r[None, :] + c["eps"])
    Vy = c["A"] * tex * rhat[None, :, 1] + xd[:, None] * c["B"] * that[None, :, 1] / (r[None, :] + c["eps"])
    feat = np.concatenate([Vx, Vy], 1)                             # (n, 2P)
    feat = feat + rng.normal(0, noise, feat.shape)
    return feat, xc, xd


# --------------------------------------------------------------------------- #
def cont_recovery(X, y):
    Xs = _std(X)
    kf = KFold(CV, shuffle=True, random_state=PROBE_SEED)
    lin = float(cross_val_score(LinearRegression(), Xs, y, cv=kf, scoring="r2").mean())
    mlp = float(cross_val_score(MLPRegressor(hidden_layer_sizes=(64,), max_iter=500,
                random_state=0), Xs, y, cv=kf, scoring="r2").mean())
    return {"lin": max(0.0, lin), "mlp": max(0.0, mlp), "best": max(0.0, lin, mlp)}


def disc_recovery(X, y):
    Xs = _std(X)
    yb = (y > 0).astype(int)
    maj = max(yb.mean(), 1 - yb.mean())
    skf = StratifiedKFold(CV, shuffle=True, random_state=PROBE_SEED)
    lin = float(cross_val_score(LogisticRegression(max_iter=2000), Xs, yb, cv=skf, scoring="accuracy").mean())
    mlp = float(cross_val_score(MLPClassifier(hidden_layer_sizes=(64,), max_iter=500,
                random_state=0), Xs, yb, cv=skf, scoring="accuracy").mean())
    det = lambda a: (a - maj) / max(1 - maj, 1e-9)
    return {"lin": det(lin), "mlp": det(mlp), "best": det(max(lin, mlp)), "maj": maj}


def half_life(curve, lams):
    """smallest lam where curve <= 0.5*curve(0); None if never (censored)."""
    base = curve[0]
    if base <= 0:
        return None
    for lam, v in zip(lams, curve):
        if v <= 0.5 * base:
            return lam
    return None


def sweep(gen, n, seed, noise, tag):
    rng_master = default_rng(seed)
    cont, disc, imb = [], [], []
    for lam in LAMBDAS:
        # per-lambda independent draw (seed-derived, deterministic)
        rng = default_rng(seed + int(round(lam * 1000)) + 7)
        X, yc, yd = gen(n, lam, rng, noise)
        c = cont_recovery(X, yc); d = disc_recovery(X, yd)
        cont.append(round(c["best"], 4)); disc.append(round(d["best"], 4)); imb.append(round(d["maj"], 3))
        print(f"  [{tag} lam={lam:<4}] cont={c['best']:.3f} (lin {c['lin']:.2f}/mlp {c['mlp']:.2f})  "
              f"disc={d['best']:.3f} (lin {d['lin']:.2f}/mlp {d['mlp']:.2f})  maj={d['maj']:.2f}", flush=True)
    lc = half_life(cont, LAMBDAS); ld = half_life(disc, LAMBDAS)
    return {"cont": cont, "disc": disc, "maj": imb, "lambda_star_c": lc, "lambda_star_d": ld}


def gates(res):
    cont, disc = res["cont"], res["disc"]
    g = {
        "cont0_ge_0.70": cont[0] >= CONT0_MIN,
        "disc0_ge_0.95": disc[0] >= DISC0_MIN,
        "cont_max_le_0.10": cont[-1] <= CONT_MAX_MAX,
        "lambda_star_c_in_grid": res["lambda_star_c"] is not None,
        "min_disc_ge_0.95": min(disc) >= DISC_MIN_MIN,
        "lambda_star_d_censored": res["lambda_star_d"] is None,
        "class_balanced": all(0.45 <= m <= 0.55 for m in res["maj"]),
    }
    g["continuous_resists"] = g["cont_max_le_0.10"] and g["lambda_star_c_in_grid"]
    g["discrete_determines"] = g["min_disc_ge_0.95"]
    g["preflight_ok"] = g["cont0_ge_0.70"] and g["disc0_ge_0.95"]
    g["shows_crossover"] = g["continuous_resists"] and g["discrete_determines"]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--frozen", action="store_true")
    ap.add_argument("--noise", type=float, default=NOISE)
    ap.add_argument("--noise-s0", type=float, default=None)
    ap.add_argument("--noise-s1", type=float, default=None)
    ap.add_argument("--out", default="results/pvnp/phase5-lossiness-crossover")
    args = ap.parse_args()
    n0 = args.noise_s0 if args.noise_s0 is not None else args.noise
    n1 = args.noise_s1 if args.noise_s1 is not None else args.noise
    if args.frozen:
        seed, n, mode = DATA_SEED, 2000, "frozen"
    else:
        seed, n, mode = CALIB_SEED, 500, "calibrate"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[cfg] mode={mode} seed={seed} n={n} noise_s0={n0} noise_s1={n1} K={K} CV={CV} "
          f"lambdas={LAMBDAS}", flush=True)

    print("S0 (1-D caustic toy):", flush=True)
    r0 = sweep(gen_s0, n, seed, n0, "S0")
    print("S1 (2-D vector field):", flush=True)
    r1 = sweep(gen_s1, n, seed, n1, "S1")
    g0, g1 = gates(r0), gates(r1)

    cross_id = g0["shows_crossover"] and g1["shows_crossover"]
    print(f"\n== {mode.upper()} ==")
    for tag, r, g in [("S0", r0, g0), ("S1", r1, g1)]:
        print(f"  {tag}: preflight={g['preflight_ok']}  cont0={r['cont'][0]} contMax={r['cont'][-1]} "
              f"lam*_c={r['lambda_star_c']}  minDisc={min(r['disc'])} lam*_d={r['lambda_star_d']}  "
              f"resists={g['continuous_resists']} determines={g['discrete_determines']} "
              f"-> shows_crossover={g['shows_crossover']}")
    print(f"  cross-substrate identity (S0 AND S1 show it) = {cross_id}")
    if mode == "calibrate":
        ok = g0["preflight_ok"] and g1["preflight_ok"] and cross_id
        print(f"  CALIBRATION {'PASSES — ready to freeze constants + run --frozen' if ok else 'NEEDS TUNING (scale/power only, §3.7)'}")

    (out / f"{mode}.json").write_text(json.dumps(
        {"mode": mode, "seed": seed, "n": n, "noise_s0": n0, "noise_s1": n1, "lambdas": LAMBDAS,
         "S0": {**r0, "gates": g0}, "S1": {**r1, "gates": g1},
         "cross_substrate_identity": bool(cross_id), "wall_s": round(time.time() - t0, 1)}, indent=2,
        default=lambda o: bool(o) if isinstance(o, np.bool_) else o))
    print(f"  wrote {out/(mode+'.json')}  ({round(time.time()-t0,1)}s)", flush=True)


if __name__ == "__main__":
    main()
