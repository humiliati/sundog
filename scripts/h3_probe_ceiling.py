#!/usr/bin/env python
"""H3-PC — PROBE-CEILING: is the banked clf_d c-suppression (H3 v2, lam=2.0, ridge c-R2=0.0063)
informational DESTRUCTION or PROBE-RELATIVE CONCEALMENT?

Runs the FROZEN battery of docs/atlas/H3_PROBE_CEILING_PREREG.md (HS4 of slate 2026-06-10) against
deterministically re-derived clf_d pooled reps on new disjoint draws. Injection-calibrated per-member
detection floors; once-touched frozen test split; KSG MI leg with rank-rule nulls. Outcomes V/a/b/c per
the prereg's precedence — outcome (b), the kill branch, is the clean-null SUCCESS.

R2 is UNCLIPPED everywhere. NOT public-eligible. Attribution: H3 v2 (shadow_pooled_synthetic_v2.py);
amnesic probing (Elazar & Goldberg); V-information (Xu et al. 2020); LEACE (Belrose et al. 2023);
KSG MI estimator (Kraskov, Stoegbauer & Grassberger 2004).
Run: python scripts/h3_probe_ceiling.py   (CPU, deterministic; ~1-2 h full size)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")     # binding: pin BLAS/OpenMP BEFORE numeric imports
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

sys.path.insert(0, "scripts")
import shadow_pooled_synthetic_v2 as v2            # noqa: E402  (torch single-thread inside)

from sklearn.linear_model import Ridge             # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor    # noqa: E402
from sklearn.kernel_approximation import Nystroem  # noqa: E402
from sklearn.pipeline import make_pipeline         # noqa: E402
from sklearn.preprocessing import StandardScaler   # noqa: E402
from sklearn.decomposition import PCA              # noqa: E402
from sklearn.model_selection import KFold, cross_val_score  # noqa: E402

SEED = v2.SEED                                     # 1234
N_POOL, N_TEST = 20000, 10000
POOL_SEED, TEST_SEED = SEED + 50001, SEED + 60001  # 51235 / 61235 (disjoint from banked, see prereg §6)
VDIR_SEED, NYS_SEED = SEED + 555, SEED + 70001     # 1789 / 71235
SHUF_SEED, MI_SUB_SEED, LC_SEED = SEED + 80001, SEED + 85001, SEED + 90001  # 81235 / 86235 / 91235
LAM = 2.0
INJ_LEVELS = [0.10, 0.20]
PCA_KS = [2, 4, 8, 16, 32]
N_SHUF, MI_SUB = 99, 5000


def cv_r2(est, X, y):
    """Pool-CV R2, UNCLIPPED, banked CV convention (scale once per dataset, 5-fold KFold rs=0)."""
    Xs = StandardScaler().fit_transform(X)
    kf = KFold(5, shuffle=True, random_state=0)
    return float(cross_val_score(est, Xs, y, cv=kf, scoring="r2").mean())


def split_r2(est, Xpool, ypool, Xtest, ytest):
    """Fit on the full pool, score ONCE on the frozen split (unclipped)."""
    sc = StandardScaler().fit(Xpool)
    est.fit(sc.transform(Xpool), ypool)
    return float(est.score(sc.transform(Xtest), ytest))


# ---- the FROZEN battery (prereg section 3; grids row-major; ties -> first in grid) ---- #
def battery():
    fams = {}
    fams["P1_ridge"] = [("alpha=1.0", lambda: Ridge(alpha=1.0))]
    fams["P2_mlp"] = [("(128,64)", lambda: MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=600,
                                                        random_state=0))]
    fams["P3_knn"] = [(f"k={k}", lambda k=k: KNeighborsRegressor(k)) for k in (5, 10, 20, 50, 100)]
    fams["P4_nystroem"] = [(f"g={g},a={a}", lambda g=g, a=a: make_pipeline(
        Nystroem(gamma=g, n_components=2000, random_state=NYS_SEED), Ridge(alpha=a)))
        for g in (0.001, 0.01, 0.1, 1.0) for a in (0.1, 1.0, 10.0)]
    fams["P5_gbt"] = [(f"lr={lr},it={it}", lambda lr=lr, it=it: HistGradientBoostingRegressor(
        random_state=0, learning_rate=lr, max_iter=it)) for lr in (0.05, 0.1) for it in (200, 500)]
    return fams


def select(fam, X, y):
    """argmax pool-CV within a family; ties -> first in grid order."""
    best = None
    for name, mk in fam:
        r = cv_r2(mk(), X, y)
        if best is None or r > best[2] + 1e-12:
            best = (name, mk, r)
    return best                                    # (config_name, maker, cv)


def ksg_mi(X, y, k=5):
    """KSG estimator I (Kraskov 2004, alg.1), max-norm, joint (X standardized PCA scores, y std c)."""
    n = X.shape[0]
    Z = np.hstack([X, y[:, None]])
    tree = cKDTree(Z)
    # distance to k-th neighbor (excluding self) in max-norm
    dk = tree.query(Z, k=k + 1, p=np.inf)[0][:, k]
    eps = np.nextafter(dk, 0)                      # strictly-less ball
    tx, ty = cKDTree(X), cKDTree(y[:, None])
    nx = np.array(tx.query_ball_point(X, eps, p=np.inf, return_length=True)) - 1
    ny = np.array(ty.query_ball_point(y[:, None], eps, p=np.inf, return_length=True)) - 1
    return float(digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)))


def main():
    print("=" * 96)
    print("H3-PC — probe-ceiling audit of the banked clf_d c-suppression (frozen prereg, battery+MI)")
    print("=" * 96)

    # bodies: deterministic retrain on the banked train draw
    u_tr, c_tr, d_tr = v2.gen(v2.N_TRAIN, v2.TRAIN_LAM, SEED + 1)
    clf, _ = v2.train_body("clf_d", u_tr, c_tr, d_tr)
    reg, _ = v2.train_body("reg_c", u_tr, c_tr, d_tr)

    # new disjoint draws
    u_pool, c_pool, _ = v2.gen(N_POOL, LAM, POOL_SEED)
    u_test, c_test, _ = v2.gen(N_TEST, LAM, TEST_SEED)
    z_pool, z_test = v2.phi_pool(clf, u_pool), v2.phi_pool(clf, u_test)
    zr_pool = v2.phi_pool(reg, u_pool)

    gates = {}
    # C0 continuity (apparatus): raw mean still washes c on the new pool
    c0 = cv_r2(Ridge(alpha=1.0), u_pool.mean(axis=1), c_pool)
    gates["C0_raw_pool_ridge"] = c0
    print(f"  C0 continuity: raw-mean ridge c-R2 on the new pool = {c0:+.4f}  (gate <= 0.05)")

    # positive control (apparatus): reg_c reps carry c under at least one member
    pos = cv_r2(Ridge(alpha=1.0), zr_pool, c_pool)
    gates["positive_control_regc_ridge"] = pos
    print(f"  positive control: reg_c pool ridge c-R2 = {pos:+.4f}  (gate >= 0.45)")

    # ---- injection calibration (prereg section 4) ---- #
    vdir = np.random.default_rng(VDIR_SEED).standard_normal(v2.H)
    vdir /= np.linalg.norm(vdir)
    g_pool = (c_pool - c_pool.mean()) / c_pool.std()

    def inject(alpha):
        return z_pool + alpha * g_pool[:, None] * vdir[None, :]

    def bisect(target):
        lo, hi = 0.0, 3.0
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            if cv_r2(Ridge(alpha=1.0), inject(mid), c_pool) < target:
                lo = mid
            else:
                hi = mid
        a = 0.5 * (lo + hi)
        return a, cv_r2(Ridge(alpha=1.0), inject(a), c_pool)

    alphas, calib_ok = {}, True
    for lvl in INJ_LEVELS:
        a, r = bisect(lvl)
        alphas[lvl] = a
        ok = abs(r - lvl) <= 0.01
        calib_ok &= ok
        gates[f"calibration_{lvl}"] = {"alpha": a, "ridge_cv": r, "ok": ok}
        print(f"  calibration {lvl}: alpha={a:.4f} ridge-CV={r:+.4f}  [{'OK' if ok else 'FAIL'}]")

    VOID = (c0 > 0.05) or (pos < 0.45) or (not calib_ok)
    if VOID:
        verdict = "V"
        print("\nVERDICT: V (VOID — apparatus gate failed; fix and re-run; not a result)")
    z_inj = {lvl: inject(alphas[lvl]) for lvl in INJ_LEVELS}

    # ---- per-member liveness + floors (any grid config >= 0.05 on the L-injection) ---- #
    fams = battery()
    liveness, floors = {}, {}
    for fam_name, fam in fams.items():
        if fam_name == "P1_ridge":
            floors[fam_name] = 0.10                # by construction
            liveness[fam_name] = {str(l): True for l in INJ_LEVELS}
            continue
        liveness[fam_name] = {}
        for lvl in INJ_LEVELS:
            live = False
            for cfg_name, mk in fam:
                if cv_r2(mk(), z_inj[lvl], c_pool) >= 0.05:
                    live = True
                    break
            liveness[fam_name][str(lvl)] = live
        floors[fam_name] = (0.10 if liveness[fam_name]["0.1"] else
                            (0.20 if liveness[fam_name]["0.2"] else None))
        tag = "MEMBER-BLIND" if floors[fam_name] is None else f"floor={floors[fam_name]}"
        print(f"  liveness {fam_name:13s}: 0.10={liveness[fam_name]['0.1']}  "
              f"0.20={liveness[fam_name]['0.2']}  -> {tag}")

    # ---- the battery on the REAL reps: select on pool, replicate once on the frozen split ---- #
    print("\n  battery on the REAL clf_d reps (pool-CV -> once-touched frozen split):")
    readout = {}
    for fam_name, fam in fams.items():
        cfg, mk, cv = select(fam, z_pool, c_pool)
        sp = split_r2(mk(), z_pool, c_pool, z_test, c_test)
        readout[fam_name] = {"config": cfg, "pool_cv": cv, "split": sp,
                             "member_blind": floors[fam_name] is None}
        print(f"    {fam_name:13s} [{cfg:12s}]  pool-CV={cv:+.4f}  split={sp:+.4f}"
              f"{'   (MEMBER-BLIND: no certificate weight)' if floors[fam_name] is None else ''}")

    # ---- MI leg (pool-only; seeded 5000-subsample; rank-rule nulls) ---- #
    sub = np.random.default_rng(MI_SUB_SEED).choice(N_POOL, MI_SUB, replace=False)
    c_sub = c_pool[sub]
    c_std = (c_sub - c_sub.mean()) / c_sub.std()
    shuf_seeds = np.random.default_rng(SHUF_SEED).integers(0, 2**31, N_SHUF)

    def pca_scores(Z, k):
        Zs = StandardScaler().fit_transform(Z[sub])
        S = PCA(n_components=k, random_state=0).fit_transform(Zs)
        return StandardScaler().fit_transform(S)

    def null_max(S):
        mis = []
        for s in shuf_seeds:
            perm = np.random.default_rng(int(s)).permutation(MI_SUB)
            mis.append(ksg_mi(S, c_std[perm]))
        return float(np.max(mis))

    mi = {"per_k": {}, "leg": "void"}
    passing = []
    for k in PCA_KS:
        S_inj = pca_scores(z_inj[0.10], k)
        mi_inj = ksg_mi(S_inj, c_std)
        nmax_inj = null_max(S_inj)
        live = mi_inj > nmax_inj
        mi["per_k"][k] = {"mi_inj010": mi_inj, "null_max_inj": nmax_inj, "inj_live": live}
        if live:
            passing.append(k)
        print(f"  MI liveness PCA-k={k:2d}: MI(0.10-inj)={mi_inj:+.4f} vs null-max={nmax_inj:+.4f}"
              f"  [{'live' if live else 'blind'}]")
    if passing and 32 in passing:
        mi["leg"] = "live"
        mi["real_ok"] = True
        for k in passing:
            S_real = pca_scores(z_pool, k)
            mi_real = ksg_mi(S_real, c_std)
            nmax_real = null_max(S_real)
            ok = mi_real <= nmax_real
            mi["per_k"][k].update({"mi_real": mi_real, "null_max_real": nmax_real, "real_in_null": ok})
            mi["real_ok"] &= ok
            print(f"  MI real PCA-k={k:2d}: MI(real)={mi_real:+.4f} vs null-max={nmax_real:+.4f}"
                  f"  [{'in-null' if ok else 'SIGNAL'}]")
    else:
        print("  MI leg VOID (no PCA-k passed injection-liveness incl. k=32) -> probe-battery-only")
    mi["passing_ks"] = passing

    # ---- learning curve (REPORTED, non-gating): best member + P1 ---- #
    best_fam = max(readout, key=lambda f: readout[f]["pool_cv"])
    best_mk = dict(fams[best_fam])[readout[best_fam]["config"]]   # the pool-selected config, reused
    lc_rng = np.random.default_rng(LC_SEED)
    lc = {}
    for n in (2000, 5000, 10000, 20000):
        idx = lc_rng.choice(N_POOL, n, replace=False) if n < N_POOL else np.arange(N_POOL)
        lc[n] = {"best": cv_r2(best_mk(), z_pool[idx], c_pool[idx]),
                 "ridge": cv_r2(Ridge(alpha=1.0), z_pool[idx], c_pool[idx])}
    print("\n  learning curve (best member + ridge, reported): " +
          "  ".join(f"n={n}: {lc[n]['best']:+.3f}/{lc[n]['ridge']:+.3f}" for n in lc))

    # ---- verdict (prereg section 5; precedence V > a > b > c) ---- #
    sub_outcomes = []
    if not VOID:
        a_hit = [f for f, r in readout.items() if r["pool_cv"] >= 0.30 and r["split"] >= 0.24]
        unreplicated = [f for f, r in readout.items() if r["pool_cv"] >= 0.30 and r["split"] < 0.24]
        live_members = [f for f in readout if not readout[f]["member_blind"]]
        b_battery = all(readout[f]["pool_cv"] <= 0.05 and readout[f]["split"] <= 0.05
                        for f in live_members)
        b_mi = (mi["leg"] == "void") or mi.get("real_ok", False)
        if a_hit:
            verdict = "a"
        elif b_battery and b_mi and not unreplicated:
            verdict = "b"
        else:
            verdict = "c"
            counted = {f: readout[f]["split"] for f in readout if readout[f]["split"] >= 0.05}
            if counted:
                ceiling = max(counted.values())
            else:
                ceiling = 0.0
                sub_outcomes.append("SPLIT-ONLY-FLUCTUATION" if any(
                    readout[f]["split"] >= 0.05 for f in readout) else "BAND-EMPTY")
            for f in unreplicated:
                sub_outcomes.append(f"UNREPLICATED-POSITIVE:{f}")
    print("\n" + "=" * 96)
    desc = {"V": "VOID — apparatus gate failed; not a result.",
            "a": "CONCEALMENT COUNTEREXAMPLE — the banked 'probe-robust suppression' clause BREAKS; "
                 "correction owed to H3_POOLED_SHADOW_RESULT.md (battery-relative suppression).",
            "b": "CERTIFIED PROBE-CEILING (clean-null SUCCESS): suppression certified informational "
                 "down to each live member's per-member floor, battery- and linear-direction-relative"
                 + (" [probe-battery-only: MI leg void]" if mi["leg"] == "void" else ""),
            "c": "BOUNDED-PARTIAL CONCEALMENT — named middle outcome; see counted set + sub-outcomes."}
    print(f"VERDICT: ({verdict}) {desc[verdict]}")
    if verdict == "c":
        print(f"  ceiling = {ceiling:+.4f}   sub-outcomes: {sub_outcomes or 'none'}")
    print("=" * 96)

    out = {"prereg": "docs/atlas/H3_PROBE_CEILING_PREREG.md",
           "params": {"lam": LAM, "n_pool": N_POOL, "n_test": N_TEST, "pool_seed": POOL_SEED,
                      "test_seed": TEST_SEED, "vdir_seed": VDIR_SEED, "nys_seed": NYS_SEED,
                      "shuf_seed": SHUF_SEED, "mi_sub_seed": MI_SUB_SEED, "lc_seed": LC_SEED,
                      "inj_levels": INJ_LEVELS, "pca_ks": PCA_KS, "n_shuf": N_SHUF, "mi_sub": MI_SUB},
           "gates": gates, "void": bool(VOID), "liveness": liveness, "floors": floors,
           "readout": readout, "mi_leg": mi["leg"], "mi": mi, "learning_curve": lc,
           "verdict": verdict, "sub_outcomes": sub_outcomes,
           "ceiling": (ceiling if verdict == "c" else None)}
    p = Path("results/atlas/h3/probe_ceiling_result.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=lambda o: float(o) if isinstance(o, np.floating)
                            else int(o) if isinstance(o, np.integer)
                            else bool(o) if isinstance(o, np.bool_) else o))
    print(f"\nwrote {p}")
    return 0 if verdict in ("a", "b", "c") else 1


if __name__ == "__main__":
    sys.exit(main())
