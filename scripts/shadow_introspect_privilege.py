"""H3-PL v3 RUN — privilege located AT the pooling stage + seed-nulled back-action.

Executes the FROZEN v3 prereg (docs/atlas/H3_PRIVILEGE_LOCALIZATION_V3_PREREG.md, freeze commit
fb851f53). Thresholds/seeds/classifiers from hs1_privilege_config (11/11 frozen-tested); substrate
from shadow_pooled_synthetic_v2 at the pinned SIGMA_D=9.0; PerUnitReadout (P6) + calibrate_injection
imported from the dry-run module. v3 deltas vs the voided v2 run: P6 demodulating member on every
arm (its per-unit arms' liveness = the raw-arm native control; pooled-style arms via calibrated
injection); linear-member liveness via the fixed-R2-target calibrated injection; back-action gate =
band floor. P6 fit budget: 3 seeds x (<=8k subsample, 80 epochs), selected on an internal 75/25
holdout (CV side only); verdict split scored exactly once per arm. Null loop checkpointed.

Emits results/atlas/h3/privilege_v3_result.json.
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
from hs1_v3_dryrun import PerUnitReadout, calibrate_injection  # noqa: E402

sub.SIGMA_D = cfg.SIGMA_D
OUT = os.path.join("results", "atlas", "h3", "privilege_v3_result.json")
CKPT = os.path.join("results", "atlas", "h3", "privilege_v3_null_ckpt.json")
P6_SUB, P6_EPOCHS, P6_SEEDS = 8000, 80, 3


def _linmodel(name, seed):
    if name == "P1_ridge":
        return Ridge(alpha=1.0)
    if name == "P4_nystroem":
        return make_pipeline(Nystroem(gamma=1.0, n_components=cfg.NYSTROEM_N, random_state=seed),
                             Ridge(alpha=0.1))
    return HistGradientBoostingRegressor(max_iter=150, random_state=seed)


def cv_r2(name, X, y, seed):
    Xs = StandardScaler().fit_transform(X)
    kf = KFold(5, shuffle=True, random_state=0)
    return float(max(0.0, cross_val_score(_linmodel(name, seed), Xs, y, cv=kf, scoring="r2").mean()))


def split_r2(name, Xtr, ytr, Xsp, ysp, seed):
    sc = StandardScaler().fit(Xtr)
    m = _linmodel(name, seed).fit(sc.transform(Xtr), ytr)
    return float(max(0.0, r2_score(ysp, m.predict(sc.transform(Xsp)))))


def reduce_cols(Xtr, Xsp, k, seed):
    if Xtr.shape[1] <= k:
        return Xtr, Xsp
    p = PCA(n_components=k, random_state=seed).fit(Xtr)
    return p.transform(Xtr), p.transform(Xsp)


def p6_fit(units_tr, c_tr, seed):
    """One P6 fit on a subsample; returns the trained module."""
    n = min(P6_SUB, len(units_tr))
    idx = np.random.default_rng(seed).choice(len(units_tr), n, replace=False)
    U = torch.tensor(units_tr[idx]); C = torch.tensor(c_tr[idx]).float()
    torch.manual_seed(seed)
    m = PerUnitReadout(units_tr.shape[2])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    mse = torch.nn.MSELoss()
    bs = 256
    for _ in range(P6_EPOCHS):
        perm = torch.randperm(n)
        for b in range(0, n, bs):
            j = perm[b:b + bs]
            opt.zero_grad()
            loss = mse(m(U[j]), C[j])
            loss.backward(); opt.step()
    m.eval()
    return m


def p6_eval(m, units, c):
    with torch.no_grad():
        pred = m(torch.tensor(units)).numpy()
    return float(max(0.0, 1 - np.var(c - pred) / np.var(c)))


def p6_best(units_tr, c_tr, base_seed):
    """Best-of-3 P6 on CV side: internal 75/25 holdout selects; returns (module, holdout_r2)."""
    cut = int(0.75 * len(units_tr))
    best, best_r2 = None, -1.0
    for j in range(P6_SEEDS):
        m = p6_fit(units_tr[:cut], c_tr[:cut], base_seed + j)
        r2 = p6_eval(m, units_tr[cut:], c_tr[cut:])
        if r2 > best_r2:
            best, best_r2 = m, r2
    return best, best_r2


def ceiling(arm, per_unit_tr, per_unit_sp, flat_tr, flat_sp, ctr, csp, rng, p6_live_gate):
    """One arm's ceiling. Linear members on PCA-k_col features with calibrated-injection liveness;
    P6 on the un-reduced per-unit tensor (pooled-style arms pass (n,1,H)). All selection on CV;
    the verdict split is scored exactly once (by the CV-winning member)."""
    Xtr, Xsp = reduce_cols(flat_tr, flat_sp, cfg.K_COL, cfg.SEEDS["battery"])
    s_inj, ach, conv = calibrate_injection(flat_tr, ctr, rng, k_col=cfg.K_COL)
    z = (ctr - ctr.mean()) / (ctr.std() + 1e-9)
    dvec = rng.standard_normal(Xtr.shape[1]); dvec /= np.linalg.norm(dvec)
    Xinj = StandardScaler().fit_transform(Xtr) + s_inj * z[:, None] * dvec[None, :]
    members = {}
    for name in ("P1_ridge", "P4_nystroem", "P5_gbt"):
        live = conv and cv_r2(name, Xinj, ctr, cfg.SEEDS["battery"]) >= cfg.DETECT_FLOOR
        cvbest = max(cv_r2(name, Xtr, ctr, cfg.SEEDS["battery"] + j) for j in range(3)) if live else 0.0
        members[name] = {"live": bool(live), "cv_r2": round(cvbest, 4)}
    p6_mod, p6_cv = p6_best(per_unit_tr, ctr, cfg.SEEDS["battery"])
    members["P6_demod"] = {"live": bool(p6_live_gate), "cv_r2": round(p6_cv, 4)}
    live_names = [n for n, m in members.items() if m["live"]]
    if not live_names:
        return {"arm": arm, "ceiling": 0.0, "members": members, "winner": None, "all_blind": True,
                "inj": {"strength": round(s_inj, 4), "achieved": round(ach, 4), "converged": conv}}
    winner = max(live_names, key=lambda n: members[n]["cv_r2"])
    if winner == "P6_demod":
        ceil = p6_eval(p6_mod, per_unit_sp, csp)
    else:
        ceil = split_r2(winner, Xtr, ctr, Xsp, csp, cfg.SEEDS["battery"])
    return {"arm": arm, "ceiling": round(ceil, 4), "members": members, "winner": winner,
            "all_blind": False,
            "inj": {"strength": round(s_inj, 4), "achieved": round(ach, 4), "converged": conv}}


def linear_cka(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    hsic = np.linalg.norm(Xc.T @ Yc, "fro") ** 2
    return float(hsic / (np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro") + 1e-12))


def phi_perunit(phi, units):
    with torch.no_grad():
        return phi.net(torch.tensor(units)).numpy()


def train_joint(frozen_phi, units, c, d, beta, seed):
    torch.manual_seed(seed)
    phi = sub.Phi(); phi.load_state_dict(frozen_phi.state_dict())
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
    res = {"hypothesis": "H3-PL-v3", "prereg_commit": "fb851f53", "sigma_d": cfg.SIGMA_D}

    u_tr, c_tr, d_tr = sub.gen(cfg.N_TRAIN, sub.TRAIN_LAM, cfg.SEEDS["body"])
    body, dfit = sub.train_body("clf_d", u_tr, c_tr, d_tr)
    u_pr, c_pr, d_pr = sub.gen(cfg.N_TRAIN, cfg.LAMBDA, cfg.SEEDS["cv_split"])
    u_sp, c_sp, d_sp = sub.gen(cfg.N_SPLIT, cfg.LAMBDA, cfg.SEEDS["cv_split"] + 1)

    # ---- gates ---- #
    u_l0, c_l0, _ = sub.gen(cfg.N_SPLIT, 0.0, cfg.SEEDS["cv_split"] + 2)
    g_c0 = sub.c_r2(u_pr.mean(axis=1), c_pr)
    g_c1 = sub.c_r2(u_l0[:, 0, :], c_l0)
    g_dacc = sub.d_acc(sub.phi_pool(body, u_pr), d_pr)
    res["gates"] = {"C0": round(g_c0, 4), "C1": round(g_c1, 4),
                    "pooled_d_acc": round(g_dacc, 4), "train_fit": round(dfit, 4)}
    if not (g_c0 <= cfg.C0_MAX and g_c1 >= cfg.C1_MIN
            and cfg.DESAT_BAND[0] <= g_dacc <= cfg.DESAT_BAND[1]):
        res["verdict_access"] = "ABORT_gate_failed"; _write(res); print("GATE ABORT"); return
    print("gates OK:", res["gates"])

    # ---- ACCESS: three arms; P6 raw fit doubles as its in-run liveness control ---- #
    tp_pr = phi_perunit(body, u_pr); tp_sp = phi_perunit(body, u_sp)
    pl_pr = sub.phi_pool(body, u_pr); pl_sp = sub.phi_pool(body, u_sp)
    raw_arm = ceiling("raw_perunit", u_pr, u_sp, u_pr.reshape(len(u_pr), -1),
                      u_sp.reshape(len(u_sp), -1), c_pr, c_sp, rng, p6_live_gate=True)
    p6_raw_ok = raw_arm["members"]["P6_demod"]["cv_r2"] >= cfg.P6_CONTROL_MIN
    res["p6_raw_control"] = {"cv_r2": raw_arm["members"]["P6_demod"]["cv_r2"], "pass": bool(p6_raw_ok)}
    print("raw arm:", raw_arm["ceiling"], "p6_cv:", res["p6_raw_control"])
    tr_arm = ceiling("trained_perunit", tp_pr, tp_sp, tp_pr.reshape(len(tp_pr), -1),
                     tp_sp.reshape(len(tp_sp), -1), c_pr, c_sp, rng, p6_live_gate=p6_raw_ok)
    print("trained arm:", tr_arm["ceiling"])
    pl_arm = ceiling("pooled", pl_pr[:, None, :], pl_sp[:, None, :], pl_pr, pl_sp,
                     c_pr, c_sp, rng, p6_live_gate=False)  # pooled P6 = MLP-on-pooled, injection-gated
    pl_arm["members"]["P6_demod"]["live"] = bool(
        pl_arm["inj"]["converged"] and pl_arm["members"]["P6_demod"]["cv_r2"] >= cfg.DETECT_FLOOR)
    print("pooled arm:", pl_arm["ceiling"])
    cr, ct_, cp = raw_arm["ceiling"], tr_arm["ceiling"], pl_arm["ceiling"]
    res["access"] = {"arms": {"raw_perunit": raw_arm, "trained_perunit": tr_arm, "pooled": pl_arm},
                     "m_enc": round(ct_ - cr, 4), "m_pool": round(ct_ - cp, 4),
                     "verdict": cfg.classify_access(cr, ct_, cp)}
    _write(res)
    print("ACCESS:", res["access"]["verdict"], "m_enc", res["access"]["m_enc"],
          "m_pool", res["access"]["m_pool"])

    # ---- back-action: fresh k=20 null (checkpointed) ---- #
    base_pool = pl_pr; base_dacc = g_dacc
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
    cka_thr = float(np.percentile(ck["cka"], 100 - cfg.NULL_PCTILE))
    drift_thr = float(np.percentile(ck["drift"], cfg.NULL_PCTILE))

    # ---- joint arms ---- #
    joint = {}
    for beta in cfg.BETAS:
        jphi = train_joint(body, u_tr, c_tr, d_tr, beta, cfg.SEEDS["joint"])
        jp_pr = sub.phi_pool(jphi, u_pr); jp_sp = sub.phi_pool(jphi, u_sp)
        jarm = ceiling("joint_pooled", jp_pr[:, None, :], jp_sp[:, None, :], jp_pr, jp_sp,
                       c_pr, c_sp, rng, p6_live_gate=False)
        jarm["members"]["P6_demod"]["live"] = bool(
            jarm["inj"]["converged"] and jarm["members"]["P6_demod"]["cv_r2"] >= cfg.DETECT_FLOOR)
        jd = sub.d_acc(jp_pr, d_pr)
        jcka = linear_cka(jp_pr, base_pool); jdrift = abs(jd - base_dacc)
        joint[str(beta)] = {"rec_joint": jarm["ceiling"], "d_acc": round(jd, 4),
                            "cka": round(jcka, 4), "drift": round(jdrift, 4),
                            "cka_outside_null": bool(jcka < cka_thr),
                            "drift_outside_null": bool(jdrift > drift_thr), "arm": jarm}
        print(f"  joint beta={beta}: rec={jarm['ceiling']} d_acc={jd:.4f}")
    vbeta = cfg.select_verdict_beta({b: joint[str(b)]["d_acc"] for b in cfg.BETAS})
    if vbeta is None:
        ba = "objective_abandoned"
    else:
        j = joint[str(vbeta)]
        ba = cfg.classify_backaction(j["rec_joint"], j["d_acc"], cp,
                                     j["cka_outside_null"], j["drift_outside_null"])
    res["back_action"] = {"null_cka_thr": round(cka_thr, 4), "null_drift_thr": round(drift_thr, 4),
                          "joint": joint, "verdict_beta": vbeta, "verdict": ba}
    _write(res)
    print("\nACCESS:", res["access"]["verdict"], "| BACK-ACTION:", ba, "(beta", vbeta, ")")


def _write(res):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
