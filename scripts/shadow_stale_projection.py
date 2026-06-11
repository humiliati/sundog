#!/usr/bin/env python
"""H3-SP — STALE-STRATEGY PROJECTION (HS2 of slate 2026-06-10): train a c-reporter where c is
accessible (per-sample lam ~ U[0,0.5]), FREEZE it, deploy at lam=2.0 (washed). What structure does the
stale output have — a lock on the determinable survivors (d, gain/mag), a graded lock, isotropic
residue, graceful collapse to mean-reporting, or transfer of the demodulator (stale-generalizes)?

Runs the FROZEN prereg docs/atlas/H3_STALE_PROJECTION_PREREG.md. Post-review design: projection
readouts computed for ALL structured cells; {g,mag} grouped as one family; PS-d vs PS-g split (the
open question is the d-partial; a gain lock is partially foreordained off-manifold); g-counterfactual
banked. R2 unclipped everywhere. The "report" is a regression head, NOT a self-report — no
introspection/confabulation claims. NOT public-eligible. Attribution: H3 v2; Nisbett & Wilson 1977;
Turpin et al. 2023; HS4 probe conventions.
Run: python scripts/shadow_stale_projection.py   (CPU deterministic, ~40-60 min)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np

sys.path.insert(0, "scripts")
import shadow_pooled_synthetic_v2 as v2            # noqa: E402  (import-only; torch single-thread)
import torch                                       # noqa: E402
import torch.nn as nn                              # noqa: E402

from sklearn.linear_model import Ridge, LogisticRegression          # noqa: E402
from sklearn.neural_network import MLPRegressor                     # noqa: E402
from sklearn.kernel_approximation import Nystroem                   # noqa: E402
from sklearn.pipeline import make_pipeline                          # noqa: E402
from sklearn.preprocessing import StandardScaler                    # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score  # noqa: E402

SEED = v2.SEED
TRAIN_SEED, DEPLOY_SEED, CONTROL_SEED = SEED + 100001, SEED + 110001, SEED + 120001
INIT_SEEDS = [SEED + 130001 + k for k in range(5)]            # 131235..131239
N_TRAIN, N_DEPLOY, N_CONTROL = 8000, 10000, 4000
NYS_SEED = SEED + 70001                                        # HS4's live-member config, continuity


def gen_hs2(n, lam_spec, seed):
    """Binding rng order: c, d, g, lam(only if 'train'), xi, eta; units = g*(fringe|disc) THEN +obs."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(v2.C_LO, v2.C_HI, n)
    d = rng.choice([-1.0, 1.0], n)
    g = rng.uniform(0.5, 1.5, n)
    lam = rng.uniform(0.0, 0.5, n) if lam_spec == "train" else np.full(n, float(lam_spec))
    xi = rng.standard_normal((n, v2.K))
    c_i = c[:, None] + lam[:, None] * xi
    fringe = np.cos(v2.W_RFF[None, None, :] * c_i[:, :, None] + v2.PSI[None, None, :])
    eta = rng.standard_normal((n, v2.K, v2.D)) * v2.SIGMA_D
    disc = d[:, None, None] * v2.A_DISC[None, None, :] + eta
    units = g[:, None, None] * np.concatenate([fringe, disc], axis=2)
    units = units + rng.standard_normal(units.shape) * v2.OBS_NOISE
    return units.astype(np.float32), c.astype(np.float32), d.astype(np.float32), g.astype(np.float32)


def cv_r2(X, y, est=None):
    """Banked CV convention, unclipped."""
    Xs = StandardScaler().fit_transform(X)
    kf = KFold(5, shuffle=True, random_state=0)
    return float(cross_val_score(est or Ridge(alpha=1.0), Xs, y, cv=kf, scoring="r2").mean())


def make_reporter(head_kind, init_seed):
    """Binding init rule: seed set immediately before constructing EACH reporter; Phi first."""
    torch.manual_seed(init_seed)
    np.random.seed(init_seed % (2**31))
    phi = v2.Phi()
    head = (nn.Linear(v2.H, 1) if head_kind == "lin"
            else nn.Sequential(nn.Linear(v2.H, 64), nn.ReLU(), nn.Linear(64, 1)))
    return phi, head


def train_reporter(head_kind, init_seed, units, c):
    phi, head = make_reporter(head_kind, init_seed)
    U, target = torch.tensor(units), torch.tensor(c)[:, None]
    opt = torch.optim.Adam(list(phi.parameters()) + list(head.parameters()), lr=1e-3)
    crit = nn.MSELoss()
    n = U.shape[0]
    for epoch in range(120):
        perm = torch.randperm(n)
        for b in range(0, n, 256):
            idx = perm[b:b + 256]
            opt.zero_grad()
            loss = crit(head(phi(U[idx])), target[idx])
            loss.backward()
            opt.step()
    phi.eval(); head.eval()
    with torch.no_grad():
        pred = head(phi(U))[:, 0].numpy()
    fit = float(1 - np.var(c - pred) / np.var(c))
    return phi, head, fit


def report(phi, head, units):
    with torch.no_grad():
        return head(phi(torch.tensor(units)))[:, 0].numpy()


def partial_r2(rep, proxy, c):
    """Squared Pearson corr of the c-residualized report and c-residualized proxy (linear)."""
    def resid(y):
        A = np.vstack([c, np.ones_like(c)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ coef
    r1, r2_ = resid(rep), resid(proxy.astype(np.float64))
    if r1.std() < 1e-12 or r2_.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(r1, r2_)[0, 1] ** 2)


def ols_r2(y, X):
    A = np.hstack([X, np.ones((len(y), 1))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    return float(1 - np.var(y - pred) / np.var(y))


def main():
    print("=" * 96)
    print("H3-SP — stale-strategy projection: frozen accessible-regime reporter deployed at lam=2.0")
    print("=" * 96)

    u_tr, c_tr, d_tr, g_tr = gen_hs2(N_TRAIN, "train", TRAIN_SEED)
    u_dep, c_dep, d_dep, g_dep = gen_hs2(N_DEPLOY, 2.0, DEPLOY_SEED)
    u_ctl, c_ctl, d_ctl, g_ctl = gen_hs2(N_CONTROL, 0.0, CONTROL_SEED)
    raw_dep = u_dep.mean(axis=1)
    mag_dep = np.linalg.norm(raw_dep, axis=1)
    raw_ctl = u_ctl.mean(axis=1)
    mag_ctl = np.linalg.norm(raw_ctl, axis=1)

    gates = {}
    # C0 wash gate (modified generator -> re-gated)
    gates["C0_raw_dep_ridge"] = cv_r2(raw_dep, c_dep)
    # MI gate: proxies must carry no c (linear AND strong probe); d~c at chance
    mlp_probe = lambda: MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=600, random_state=0)
    gates["MI_g_ridge"] = cv_r2(c_dep[:, None], g_dep)
    gates["MI_g_mlp"] = cv_r2(c_dep[:, None], g_dep, mlp_probe())
    gates["MI_mag_ridge"] = cv_r2(c_dep[:, None], mag_dep)
    gates["MI_mag_mlp"] = cv_r2(c_dep[:, None], mag_dep, mlp_probe())
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    gates["MI_d_balacc"] = float(cross_val_score(
        LogisticRegression(max_iter=2000), StandardScaler().fit_transform(c_dep[:, None]),
        (d_dep > 0).astype(int), cv=skf, scoring="balanced_accuracy").mean())
    print(f"  C0 wash: {gates['C0_raw_dep_ridge']:+.4f} (<=0.05)   "
          f"MI g: {gates['MI_g_ridge']:+.4f}/{gates['MI_g_mlp']:+.4f}  "
          f"mag: {gates['MI_mag_ridge']:+.4f}/{gates['MI_mag_mlp']:+.4f} (<=0.01)  "
          f"d: {gates['MI_d_balacc']:.4f} (<=0.52)")
    apparatus_ok = (gates["C0_raw_dep_ridge"] <= 0.05
                    and max(gates["MI_g_ridge"], gates["MI_g_mlp"],
                            gates["MI_mag_ridge"], gates["MI_mag_mlp"]) <= 0.01
                    and gates["MI_d_balacc"] <= 0.52)

    var_c_dep = float(np.var(c_dep))
    cells = []
    for kind in ("lin", "mlp"):
        for init in INIT_SEEDS:
            phi, head, fit = train_reporter(kind, init, u_tr, c_tr)
            rep_dep = report(phi, head, u_dep)
            rep_ctl = report(phi, head, u_ctl)
            # control + train-fit gates (per cell)
            ctl_r2 = cv_r2(rep_ctl[:, None], c_ctl)
            # coupling band (max of three probes on the 1-D report)
            coup = max(
                cv_r2(rep_dep[:, None], c_dep),
                cv_r2(rep_dep[:, None], c_dep, mlp_probe()),
                cv_r2(rep_dep[:, None], c_dep, make_pipeline(
                    Nystroem(gamma=1.0, n_components=2000, random_state=NYS_SEED),
                    Ridge(alpha=0.1))))
            var_ratio = float(np.var(rep_dep) / var_c_dep)
            var_ratio_ctl = float(np.var(rep_ctl) / np.var(c_ctl))
            # projection partials (deploy) + lam=0 baselines
            p_d = partial_r2(rep_dep, d_dep, c_dep)
            p_g = partial_r2(rep_dep, g_dep, c_dep)
            p_m = partial_r2(rep_dep, mag_dep, c_dep)
            b_d = partial_r2(rep_ctl, d_ctl, c_ctl)
            b_g = partial_r2(rep_ctl, g_ctl, c_ctl)
            b_m = partial_r2(rep_ctl, mag_ctl, c_ctl)
            fam_d, fam_g = p_d, max(p_g, p_m)
            bas_d, bas_g = b_d, max(b_g, b_m)
            # g-counterfactual (banked, non-gating)
            rep_cf = report(phi, head, (u_dep / g_dep[:, None, None]).astype(np.float32))
            r2_cf = ols_r2(rep_dep, rep_cf[:, None])
            r2_cfg = ols_r2(rep_dep, np.column_stack([rep_cf, g_dep]))
            # cell verdict (precedence SG > CO > PS-d > PS-g > PP > IS)
            if coup > 0.3:
                cv_ = "SG"
            elif var_ratio < 0.25:
                cv_ = "CO"
            elif fam_d >= 0.5 and fam_d >= 5 * max(bas_d, 1e-9):
                cv_ = "PS-d"
            elif fam_g >= 0.5 and fam_g >= 5 * max(bas_g, 1e-9):
                cv_ = "PS-g"
            elif max(fam_d, fam_g) >= 0.1:
                cv_ = "PP"
            else:
                cv_ = "IS"
            cells.append({"head": kind, "init": init, "train_fit": fit, "ctl_r2": ctl_r2,
                          "coupling": coup, "var_ratio": var_ratio, "var_ratio_ctl": var_ratio_ctl,
                          "p_d": p_d, "p_g": p_g, "p_mag": p_m,
                          "b_d": b_d, "b_g": b_g, "b_mag": b_m,
                          "fam_d": fam_d, "fam_gmag": fam_g,
                          "gcf_r2": r2_cf, "gcf_scale_share": r2_cfg - r2_cf, "cell": cv_})
            print(f"  {kind}/{init}: fit={fit:.3f} ctl={ctl_r2:+.3f} coup={coup:+.3f} "
                  f"var={var_ratio:.3f} | d={p_d:.3f} g={p_g:.3f} mag={p_m:.3f} "
                  f"(bas {bas_d:.3f}/{bas_g:.3f}) gcf-share={r2_cfg - r2_cf:+.3f} -> {cv_}")

    trainfit_ok = all(c_["train_fit"] >= 0.7 for c_ in cells)
    control_ok = all(c_["ctl_r2"] >= 0.9 for c_ in cells)
    VOID = not (apparatus_ok and trainfit_ok and control_ok)

    # run verdict
    def head_modal(kind):
        vs = [c_["cell"] for c_ in cells if c_["head"] == kind]
        for v in set(vs):
            if vs.count(v) >= 4:
                return v
        return None

    m_lin, m_mlp = head_modal("lin"), head_modal("mlp")
    if VOID:
        verdict = "V"
    elif m_lin is not None and m_lin == m_mlp:
        verdict = {"PS-d": "A-d", "PS-g": "A-g", "PP": "P", "CO": "K1",
                   "IS": "K2", "SG": "G"}[m_lin]
    elif m_lin is not None and m_mlp is not None:
        verdict = "D"
    else:
        verdict = "M"

    readings = {
        "V": "VOID — apparatus gate failed (C0/MI/train-fit/lam0-control); fix and re-run.",
        "A-d": "SIGNATURE (determinable-subspace): the stale report locks onto the discrete survivor "
               "— a proxy-correlation audit could flag stale reports in this substrate.",
        "A-g": "SCALE-CHANNEL PROJECTION: gain-family lock — pre-named partially foreordained "
               "(multiplicative gain off-manifold); weaker reading; counterfactual share banked.",
        "P": "PARTIAL-PROJECTION: graded determinable-subspace lock; magnitudes banked.",
        "K1": "GRACEFUL-COLLAPSE (clean null = SUCCESS): stale strategies degrade toward "
              "mean-reporting; report-audits cannot flag what collapses to the prior.",
        "K2": "ISOTROPIC (clean null = SUCCESS): variance above floor but latent-decoupled "
              "(consistent with propagated input noise); no determinable-subspace signature.",
        "G": "STALE-GENERALIZES: the accessible-regime demodulator transfers at this deploy lam — "
             "the deploy-shift cannot manufacture decoupling for this strategy class.",
        "D": "ARCHITECTURE-DEPENDENT: the heads' modal verdicts differ — itself the pre-registered "
             "finding.",
        "M": "MIXED: no >=4/5 modal verdict in at least one head; full per-cell table banked.",
    }
    print("\n" + "=" * 96)
    print(f"VERDICT: ({verdict}) {readings[verdict]}")
    print(f"  head modal verdicts: lin={m_lin} mlp={m_mlp}")
    print("=" * 96)

    out = {"prereg": "docs/atlas/H3_STALE_PROJECTION_PREREG.md",
           "params": {"train_seed": TRAIN_SEED, "deploy_seed": DEPLOY_SEED,
                      "control_seed": CONTROL_SEED, "init_seeds": INIT_SEEDS,
                      "n": [N_TRAIN, N_DEPLOY, N_CONTROL], "nys_seed": NYS_SEED},
           "gates": gates, "apparatus_ok": bool(apparatus_ok),
           "trainfit_ok": bool(trainfit_ok), "control_ok": bool(control_ok),
           "cells": cells, "head_modal": {"lin": m_lin, "mlp": m_mlp},
           "verdict": verdict, "void": bool(VOID)}
    p = Path("results/atlas/h3/stale_projection_result.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=lambda o: float(o) if isinstance(o, np.floating)
                            else int(o) if isinstance(o, np.integer)
                            else bool(o) if isinstance(o, np.bool_) else o))
    print(f"\nwrote {p}")
    return 0 if verdict != "V" else 1


if __name__ == "__main__":
    sys.exit(main())
