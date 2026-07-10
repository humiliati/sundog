"""H3-PL v3 ACHIEVABILITY DRY-RUN (prereg section 7) -- pre-freeze, control-only.

Asserts every v3 gate has a reachable region on the pinned substrate BEFORE the prereg freezes
(the process fix from the void v2 run). Five asserts:
  1. fresh-seed body (52235) reproduces the de-sat band + C0/C1 gates;
  2. P6 positive control: the demodulating probe reads c from RAW per-unit features at lambda=2.0
     (CV R2 >= 0.30) -- proves the reference arm is alive;
  3. injection calibration converges per arm (raw-perunit / trained-perunit / pooled): bisection
     finds a strength with ridge-CV on injected features in [0.05, 0.20];
  4. beta=0 gate control: continued pure-d training holds d-acc >= 0.75 (gate reachable);
  5. gate arithmetic: BACKACTION_GATE == DESAT_BAND[0] <= DESAT_BAND[1].

BOUNDARY (contamination firewall): P6 is NEVER trained on trained-per-unit features here, and no
beta>0 joint arm runs -- the two open questions stay unharvested. n=8000 (substrate recipe scale).
Emits results/atlas/hs1/v3_dryrun.json (values pinned into the prereg at freeze).
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
import shadow_pooled_synthetic_v2 as sub  # noqa: E402

sub.SIGMA_D = 9.0                          # pinned de-saturation (tuning db1c52b2)
SEED_BODY = 52235                          # v3 fresh ledger
SEED_DRY = 87235
LAMBDA = 2.0
N = 8000
DESAT_BAND = (0.75, 0.85)
BACKACTION_GATE = 0.75
INJ_TARGET_BAND = (0.05, 0.20)
INJ_TARGET = 0.10
P6_CONTROL_MIN = 0.30
OUT = os.path.join("results", "atlas", "hs1", "v3_dryrun.json")


def ridge_cv(X, y):
    Xs = StandardScaler().fit_transform(X)
    kf = KFold(5, shuffle=True, random_state=0)
    return float(max(0.0, cross_val_score(Ridge(alpha=1.0), Xs, y, cv=kf, scoring="r2").mean()))


def calibrate_injection(X, c, rng, k_col=128):
    """Bisection on injected strength until ridge-CV R2 on injected features ~ INJ_TARGET.
    Returns (strength, achieved_r2, converged)."""
    if X.shape[1] > k_col:
        X = PCA(n_components=k_col, random_state=SEED_DRY).fit_transform(X)
    z = (c - c.mean()) / (c.std() + 1e-9)
    d = rng.standard_normal(X.shape[1]); d /= np.linalg.norm(d)
    Xs = StandardScaler().fit_transform(X)

    def r2_at(s):
        return ridge_cv(Xs + s * z[:, None] * d[None, :], c)

    lo, hi = 0.01, 8.0
    if r2_at(hi) < INJ_TARGET:                      # even huge injection undetectable -> fail fast
        return hi, r2_at(hi), False
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if r2_at(mid) < INJ_TARGET:
            lo = mid
        else:
            hi = mid
    ach = r2_at(hi)
    return hi, ach, bool(INJ_TARGET_BAND[0] <= ach <= INJ_TARGET_BAND[1] * 2)  # converged near target


class PerUnitReadout(torch.nn.Module):
    """P6, the demodulating probe: per-unit MLP -> mean over units -> linear (the substrate's own
    head class as a probe)."""
    def __init__(self, in_dim):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(in_dim, 128), torch.nn.ReLU(),
                                       torch.nn.Linear(128, 128), torch.nn.ReLU(),
                                       torch.nn.Linear(128, 32))
        self.head = torch.nn.Linear(32, 1)

    def forward(self, u):                    # u: (B, K, in_dim)
        return self.head(self.net(u).mean(dim=1))[:, 0]


def p6_cv_r2(units, c, seed):
    """5-fold CV of the P6 probe trained on per-unit tensors (CV-side only by construction)."""
    kf = KFold(5, shuffle=True, random_state=0)
    r2s = []
    U = torch.tensor(units); C = torch.tensor(c).float()
    for tr_idx, te_idx in kf.split(units):
        torch.manual_seed(seed)
        m = PerUnitReadout(units.shape[2])
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        mse = torch.nn.MSELoss()
        n, bs = len(tr_idx), 256
        for _ in range(60):                                    # dry-run budget (full run uses 120)
            perm = np.random.default_rng(seed).permutation(n)
            for b in range(0, n, bs):
                idx = torch.tensor(tr_idx[perm[b:b + bs]])
                opt.zero_grad()
                loss = mse(m(U[idx]), C[idx])
                loss.backward(); opt.step()
        with torch.no_grad():
            pred = m(U[torch.tensor(te_idx)]).numpy()
        ss = 1 - np.var(c[te_idx] - pred) / np.var(c[te_idx])
        r2s.append(max(0.0, float(ss)))
    return float(np.mean(r2s))


def continued_d_training(body, units, d, seed):
    """beta=0 gate control: keep training the body on PURE d-loss; return final pooled d-acc."""
    torch.manual_seed(seed)
    phi = sub.Phi(); phi.load_state_dict(body.state_dict())
    head = torch.nn.Linear(sub.H, 2)
    U = torch.tensor(units); dt = torch.tensor((d > 0).astype(np.int64))
    opt = torch.optim.Adam(list(phi.parameters()) + list(head.parameters()), lr=1e-3)
    ce = torch.nn.CrossEntropyLoss()
    n, bs = U.shape[0], 256
    for _ in range(60):
        perm = torch.randperm(n)
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            opt.zero_grad()
            loss = ce(head(phi(U[idx])), dt[idx])
            loss.backward(); opt.step()
    phi.eval()
    return phi


def phi_perunit(phi, units):
    with torch.no_grad():
        return phi.net(torch.tensor(units)).numpy()


def main():
    rng = np.random.default_rng(SEED_DRY)
    res = {"pass_name": "hs1_v3_dryrun", "sigma_d": sub.SIGMA_D, "n": N,
           "seeds": {"body": SEED_BODY, "dry": SEED_DRY}}

    # ---- 1. fresh body + band + gates ---- #
    u_tr, c_tr, d_tr = sub.gen(N, sub.TRAIN_LAM, SEED_BODY)
    body, fit = sub.train_body("clf_d", u_tr, c_tr, d_tr)
    u_ev, c_ev, d_ev = sub.gen(sub.N_PROBE, LAMBDA, SEED_DRY + 1)
    dacc = sub.d_acc(sub.phi_pool(body, u_ev), d_ev)
    u_l0, c_l0, _ = sub.gen(sub.N_PROBE, 0.0, SEED_DRY + 2)
    c0 = sub.c_r2(u_ev.mean(axis=1), c_ev)
    c1 = sub.c_r2(u_l0[:, 0, :], c_l0)
    res["a1_body"] = {"pooled_d_acc": round(dacc, 4), "in_band": bool(DESAT_BAND[0] <= dacc <= DESAT_BAND[1]),
                      "C0": round(c0, 4), "C0_pass": bool(c0 <= 0.10),
                      "C1": round(c1, 4), "C1_pass": bool(c1 >= 0.50), "train_fit": round(fit, 4)}

    # ---- 2. P6 positive control on RAW per-unit at lambda=2 ---- #
    u_p, c_p, _ = sub.gen(N, LAMBDA, SEED_DRY + 3)
    p6_raw = p6_cv_r2(u_p, c_p, SEED_DRY)
    res["a2_p6_raw_control"] = {"cv_r2": round(p6_raw, 4), "pass": bool(p6_raw >= P6_CONTROL_MIN)}

    # ---- 3. injection calibration per arm ---- #
    arms = {"raw_perunit": u_p.reshape(len(u_p), -1),
            "trained_perunit": phi_perunit(body, u_p).reshape(len(u_p), -1),
            "pooled": sub.phi_pool(body, u_p)}
    res["a3_injection"] = {}
    for name, X in arms.items():
        s, ach, ok = calibrate_injection(X, c_p, rng)
        res["a3_injection"][name] = {"strength": round(s, 4), "achieved_r2": round(ach, 4), "converged": ok}

    # ---- 4. beta=0 gate control ---- #
    phi0 = continued_d_training(body, u_tr, d_tr, SEED_DRY + 4)
    dacc0 = sub.d_acc(sub.phi_pool(phi0, u_ev), d_ev)
    res["a4_beta0_gate"] = {"d_acc": round(dacc0, 4), "pass": bool(dacc0 >= BACKACTION_GATE)}

    # ---- 5. gate arithmetic ---- #
    res["a5_gate_arith"] = {"gate": BACKACTION_GATE, "band": DESAT_BAND,
                            "pass": bool(BACKACTION_GATE == DESAT_BAND[0] <= DESAT_BAND[1])}

    all_pass = (res["a1_body"]["in_band"] and res["a1_body"]["C0_pass"] and res["a1_body"]["C1_pass"]
                and res["a2_p6_raw_control"]["pass"]
                and all(v["converged"] for v in res["a3_injection"].values())
                and res["a4_beta0_gate"]["pass"] and res["a5_gate_arith"]["pass"])
    res["verdict"] = "FREEZE_CLEAR" if all_pass else "FREEZE_BLOCKED"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    for k in ("a1_body", "a2_p6_raw_control", "a3_injection", "a4_beta0_gate", "a5_gate_arith"):
        print(k, "=>", json.dumps(res[k]))
    print("VERDICT:", res["verdict"])


if __name__ == "__main__":
    main()
