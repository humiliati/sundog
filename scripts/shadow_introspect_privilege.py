"""H3-PL (HS1) RUN — privilege located AT the pooling stage + seed-nulled back-action.

Executes the FROZEN prereg (docs/atlas/H3_PRIVILEGE_LOCALIZATION_PREREG.md v2, commit 815ea529).
All thresholds/seeds and BOTH verdict classifiers come from hs1_privilege_config (the frozen,
9/9-tested spec); the substrate machinery from shadow_pooled_synthetic_v2. SIGMA_D is forced to the
pinned 9.0 before any regen. The k=20 retrain null is checkpointed per-retrain (resumable).

Emits results/atlas/h3/privilege_result.json.
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
import hs1_privilege_config as cfg          # noqa: E402
import shadow_pooled_synthetic_v2 as sub    # noqa: E402

sub.SIGMA_D = cfg.SIGMA_D                    # PIN the de-saturation (not the default 1.5)
OUT = os.path.join("results", "atlas", "h3", "privilege_result.json")
CKPT = os.path.join("results", "atlas", "h3", "privilege_null_ckpt.json")
DETECT_FLOOR = 0.05                         # a member "detects" injected c if CV-R2 >= this


def phi_perunit(phi, units):
    with torch.no_grad():
        return phi.net(torch.tensor(units)).numpy()      # (n, K, H) pre-pool


def cv_r2(name, X, y, seed):
    Xs = StandardScaler().fit_transform(X)
    kf = KFold(5, shuffle=True, random_state=0)
    return float(max(0.0, cross_val_score(_model(name, seed), Xs, y, cv=kf, scoring="r2").mean()))


def _model(name, seed):
    if name == "P1_ridge":
        return Ridge(alpha=1.0)
    if name == "P4_nystroem":
        return make_pipeline(Nystroem(gamma=1.0, n_components=cfg.NYSTROEM_N, random_state=seed),
                             Ridge(alpha=0.1))
    return HistGradientBoostingRegressor(max_iter=150, random_state=seed)


def split_r2(name, Xtr, ytr, Xsp, ysp, seed):
    sc = StandardScaler().fit(Xtr)
    m = _model(name, seed).fit(sc.transform(Xtr), ytr)
    return float(max(0.0, r2_score(ysp, m.predict(sc.transform(Xsp)))))


def reduce_cols(Xtr, Xsp, k, seed):
    if Xtr.shape[1] <= k:
        return Xtr, Xsp
    p = PCA(n_components=k, random_state=seed).fit(Xtr)   # fit on TRAIN only
    return p.transform(Xtr), p.transform(Xsp)


def inject(X, c, s, rng):
    z = (c - c.mean()) / (c.std() + 1e-9)
    d = rng.standard_normal(X.shape[1]); d /= np.linalg.norm(d)
    return StandardScaler().fit_transform(X) + s * z[:, None] * d[None, :]


def ceiling(arm, Xtr_raw, Xsp_raw, ctr, csp, rng):
    """Matched-dim ceiling for one arm: PCA->k_col, per-member injection liveness on TRAIN, then the
    max split-R2 over live members (best-of-3 fits; split scored once per arm = threshold-anchored)."""
    Xtr, Xsp = reduce_cols(Xtr_raw, Xsp_raw, cfg.K_COL, cfg.SEEDS["battery"])
    members, live = {}, {}
    for name in cfg.COUNTED_BATTERY:
        floor = None
        for s in cfg.INJ_LEVELS:
            if cv_r2(name, inject(Xtr, ctr, s, rng), ctr, cfg.SEEDS["battery"]) >= DETECT_FLOOR:
                floor = s; break
        live[name] = floor is not None
        cvbest = max(cv_r2(name, StandardScaler().fit_transform(Xtr), ctr, cfg.SEEDS["battery"] + j)
                     for j in range(3)) if live[name] else 0.0     # best-of-3 on CV (train only)
        members[name] = {"live": live[name], "floor": floor, "cv_r2": round(cvbest, 4)}
    live_members = [n for n in cfg.COUNTED_BATTERY if live[n]]
    if not live_members:
        return {"arm": arm, "ceiling": 0.0, "members": members, "winner": None, "all_blind": True}
    winner = max(live_members, key=lambda n: members[n]["cv_r2"])   # select on CV
    ceil = split_r2(winner, Xtr, ctr, Xsp, csp, cfg.SEEDS["battery"])  # score split ONCE
    return {"arm": arm, "ceiling": round(ceil, 4), "members": members, "winner": winner,
            "all_blind": False}


def linear_cka(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    hsic = np.linalg.norm(Xc.T @ Yc, "fro") ** 2
    return float(hsic / (np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro") + 1e-12))


def train_joint(frozen_phi, units, c, d, beta, seed):
    torch.manual_seed(seed)
    phi = sub.Phi()
    phi.load_state_dict(frozen_phi.state_dict())          # init at the frozen clf_d encoder
    d_head, c_head = torch.nn.Linear(sub.H, 2), torch.nn.Linear(sub.H, 1)
    U = torch.tensor(units); dt = torch.tensor((d > 0).astype(np.int64)); ct = torch.tensor(c).float()
    opt = torch.optim.Adam(list(phi.parameters()) + list(d_head.parameters()) +
                           list(c_head.parameters()), lr=1e-3)
    ce, mse = torch.nn.CrossEntropyLoss(), torch.nn.MSELoss()
    n, bs = U.shape[0], 256
    for _ in range(120):
        perm = torch.randperm(n)
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            opt.zero_grad()
            rep = phi(U[idx])
            loss = ce(d_head(rep), dt[idx]) + beta * mse(c_head(rep)[:, 0], ct[idx])
            loss.backward(); opt.step()
    phi.eval()
    return phi


def main():
    rng = np.random.default_rng(cfg.SEEDS["battery"])
    res = {"hypothesis": "H3-PL", "prereg_commit": "815ea529", "sigma_d": cfg.SIGMA_D}

    # ---- body (train at TRAIN_LAM; measure at lambda=2.0) ---- #
    u_tr, c_tr, d_tr = sub.gen(cfg.N_TRAIN, sub.TRAIN_LAM, cfg.SEEDS["body"])
    body, dfit = sub.train_body("clf_d", u_tr, c_tr, d_tr)
    u_pr, c_pr, d_pr = sub.gen(cfg.N_TRAIN, cfg.LAMBDA, cfg.SEEDS["cv_split"])          # probe-fit
    u_sp, c_sp, d_sp = sub.gen(cfg.N_SPLIT, cfg.LAMBDA, cfg.SEEDS["cv_split"] + 1)      # once-touched

    # ---- gates (hard abort) ---- #
    u_l0, c_l0, _ = sub.gen(cfg.N_SPLIT, 0.0, cfg.SEEDS["cv_split"] + 2)
    g_c0 = sub.c_r2(u_pr.mean(axis=1), c_pr)
    g_c1 = sub.c_r2(u_l0[:, 0, :], c_l0)
    g_dacc = sub.d_acc(sub.phi_pool(body, u_pr), d_pr)
    res["gates"] = {"C0_raw_mean_c_r2": round(g_c0, 4), "C1_single_unit_c_r2": round(g_c1, 4),
                    "pooled_d_acc": round(g_dacc, 4), "train_fit": round(dfit, 4)}
    if not (g_c0 <= cfg.C0_MAX and g_c1 >= cfg.C1_MIN and cfg.DESAT_BAND[0] <= g_dacc <= cfg.DESAT_BAND[1]):
        res["verdict_access"] = "ABORT_gate_failed"
        _write(res); print("GATE ABORT:", res["gates"]); return

    # ---- three matched-dim ACCESS ceilings ---- #
    raw_tr = u_pr.reshape(len(u_pr), -1);  raw_sp = u_sp.reshape(len(u_sp), -1)
    tp_tr = phi_perunit(body, u_pr).reshape(len(u_pr), -1)
    tp_sp = phi_perunit(body, u_sp).reshape(len(u_sp), -1)
    pl_tr = sub.phi_pool(body, u_pr);  pl_sp = sub.phi_pool(body, u_sp)
    arms = {
        "raw_perunit":     ceiling("raw_perunit",     raw_tr, raw_sp, c_pr, c_sp, rng),
        "trained_perunit": ceiling("trained_perunit", tp_tr,  tp_sp,  c_pr, c_sp, rng),
        "pooled":          ceiling("pooled",          pl_tr,  pl_sp,  c_pr, c_sp, rng),
    }
    cr, ct_, cp = arms["raw_perunit"]["ceiling"], arms["trained_perunit"]["ceiling"], arms["pooled"]["ceiling"]
    res["access"] = {"arms": arms, "m_enc": round(ct_ - cr, 4), "m_pool": round(ct_ - cp, 4),
                     "verdict": cfg.classify_access(cr, ct_, cp)}
    _write(res)

    # ---- back-action: k=20 retrain null (checkpointed) ---- #
    base_pool = sub.phi_pool(body, u_pr)
    base_dacc = sub.d_acc(base_pool, d_pr)
    ck = json.load(open(CKPT)) if os.path.exists(CKPT) else {"cka": [], "drift": []}
    for i in range(len(ck["cka"]), cfg.K_NULL):
        s = cfg.SEEDS["null"][i]
        ur, ccr, ddr = sub.gen(cfg.N_TRAIN, sub.TRAIN_LAM, s)
        rb, _ = sub.train_body("clf_d", ur, ccr, ddr)
        rp = sub.phi_pool(rb, u_pr)
        ck["cka"].append(linear_cka(rp, base_pool))
        ck["drift"].append(abs(sub.d_acc(rp, d_pr) - base_dacc))
        json.dump(ck, open(CKPT, "w"))
        print(f"  null {i+1}/{cfg.K_NULL}: cka={ck['cka'][-1]:.4f} drift={ck['drift'][-1]:.4f}")
    cka_thr = float(np.percentile(ck["cka"], 100 - cfg.NULL_PCTILE))   # LOW cka = more restructuring
    drift_thr = float(np.percentile(ck["drift"], cfg.NULL_PCTILE))

    # ---- back-action: joint training over betas ---- #
    joint = {}
    for beta in cfg.BETAS:
        jphi = train_joint(body, u_tr, c_tr, d_tr, beta, cfg.SEEDS["joint"])
        jpool_pr = sub.phi_pool(jphi, u_pr)
        rec = ceiling("joint_pooled", jpool_pr, sub.phi_pool(jphi, u_sp), c_pr, c_sp, rng)["ceiling"]
        jd = sub.d_acc(jpool_pr, d_pr)
        jcka = linear_cka(jpool_pr, base_pool)
        jdrift = abs(jd - base_dacc)
        joint[str(beta)] = {"rec_joint": round(rec, 4), "d_acc": round(jd, 4),
                            "cka": round(jcka, 4), "drift": round(jdrift, 4),
                            "cka_outside_null": jcka < cka_thr, "drift_outside_null": jdrift > drift_thr}
    vbeta = cfg.select_verdict_beta({b: joint[str(b)]["d_acc"] for b in cfg.BETAS})
    if vbeta is None:
        ba_verdict = "objective_abandoned"
    else:
        j = joint[str(vbeta)]
        ba_verdict = cfg.classify_backaction(j["rec_joint"], j["d_acc"], cp,
                                             j["cka_outside_null"], j["drift_outside_null"])
    res["back_action"] = {"null_cka_thr": round(cka_thr, 4), "null_drift_thr": round(drift_thr, 4),
                          "joint": joint, "verdict_beta": vbeta, "verdict": ba_verdict}
    _write(res)
    print("\nACCESS:", res["access"]["verdict"], "(m_enc", res["access"]["m_enc"],
          "m_pool", res["access"]["m_pool"], ")")
    print("BACK-ACTION:", ba_verdict, "(beta", vbeta, ")")


def _write(res):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
