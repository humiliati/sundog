#!/usr/bin/env python3
"""
Yang-Mills cheap Category-A controls (Q1 + Q5), in-house, on the registered
SU(2) 3D Phase-2 data (12^3, beta {2.0, 2.4, 2.8}, 32 configs/beta).

Q5 (lattice sanity): is the observed 1x1 plaquette <W11> in family with standard
SU(2) 2+1D values? Benchmark = the single-plaquette / leading mean-field value
I_2(beta)/I_1(beta) (modified Bessel ratio), which the on-lattice plaquette
should sit slightly ABOVE (neighbouring plaquettes correlate and raise <P>).

Q1 (signature/target disjointness) — and, going deeper than the packet asked,
TARGET VALIDITY: is gamma_held (the held-out LS-slope target the 3 gamma probes
tried to predict) (a) independent of the signature [the leakage question], and
(b) carrying any genuine signal, or is it dominated by an epsilon-floor CLAMP on
the 3x3 Wilson loop W33, which at this (beta, volume) is a noise-level observable?

This is a bounded-null diagnostic, not a Yang-Mills, confinement, or mass-gap
claim. It interprets WHY the Phase-2 null landed at chance; it does not promote
anything.
"""
import csv
import json
import os
import numpy as np
from scipy import special

# Resolve repo-relative paths from this script's location (sundog/scripts/..).
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BETAS = [2.0, 2.4, 2.8]
ROOT = "results/yang-mills/phase2/SU2_3D"
ENS = {
    2.0: "2026-05-29_su2_3d_beta2.0_ensemble_v0",
    2.4: "2026-05-29_su2_3d_beta2.4_ensemble_v0",
    2.8: "2026-05-29_su2_3d_beta2.8_ensemble_v0",
}
SIG_COLS = ["W11_mean", "W11_var", "W12_mean", "W12_var",
            "W13_mean", "W13_var", "W22_mean", "W22_var"]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_beta(beta):
    base = os.path.join(ROOT, ENS[beta])
    sig = read_csv(os.path.join(base, "signatures", "signature_vectors.csv"))
    hel = read_csv(os.path.join(base, "heldout", "heldout_summary.csv"))
    sig = {int(r["config_idx"]): r for r in sig}
    hel = {int(r["config_idx"]): r for r in hel}
    idx = sorted(set(sig) & set(hel))
    X = np.array([[float(sig[i][c]) for c in SIG_COLS] for i in idx])
    gamma = np.array([float(hel[i]["gamma_held"]) for i in idx])
    clamp = np.array([int(hel[i]["clamped"]) for i in idx])
    W33 = np.array([float(hel[i]["W33_mean"]) for i in idx])
    W14 = np.array([float(hel[i]["W14_mean"]) for i in idx])
    W23 = np.array([float(hel[i]["W23_mean"]) for i in idx])
    return X, gamma, clamp, W33, W14, W23


def cv_r2(X, y, folds=5, seed=0):
    """Out-of-sample R^2 of OLS y~X via k-fold CV (honest small-n leakage test)."""
    n = len(y)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    parts = np.array_split(order, folds)
    sse, sst = 0.0, float(np.sum((y - y.mean()) ** 2))
    for k in range(folds):
        te = parts[k]
        tr = np.concatenate([parts[j] for j in range(folds) if j != k])
        A = np.column_stack([np.ones(len(tr)), X[tr]])
        coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred = np.column_stack([np.ones(len(te)), X[te]]) @ coef
        sse += float(np.sum((y[te] - pred) ** 2))
    return 1.0 - sse / sst


def adj_r2(X, y):
    n, p = X.shape
    A = np.column_stack([np.ones(n), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    r2 = 1.0 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)


def r2_on(feat, y):
    feat = feat.reshape(-1, 1).astype(float)
    A = np.column_stack([np.ones(len(y)), feat])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return 1.0 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)


def tertile_labels(y):
    q1, q2 = np.quantile(y, [1 / 3, 2 / 3])
    return np.where(y <= q1, 0, np.where(y <= q2, 1, 2))


out = {"Q5_plaquette_in_family": {}, "Q1_disjointness_and_target_validity": {}}

# ---- Q5: <W11> vs SU(2) single-plaquette Bessel benchmark ----
for beta in BETAS:
    X, *_ = load_beta(beta)
    w11 = float(X[:, 0].mean())  # ensemble-mean 1x1 plaquette
    bessel = float(special.iv(2, beta) / special.iv(1, beta))
    out["Q5_plaquette_in_family"][f"beta_{beta}"] = {
        "observed_W11": round(w11, 5),
        "bessel_benchmark_I2_over_I1": round(bessel, 5),
        "observed_minus_benchmark": round(w11 - bessel, 5),
        "in_family": bool(abs(w11 - bessel) < 0.05 and w11 > bessel - 0.01),
    }

# ---- Q1: disjointness (leakage) + target validity (clamp/W33 noise) ----
poolX, poolG, poolC = [], [], []
for beta in BETAS:
    X, gamma, clamp, W33, W14, W23 = load_beta(beta)
    n = len(gamma)
    # leakage: can the signature predict gamma_held?
    maxabscorr = float(np.max([abs(np.corrcoef(X[:, j], gamma)[0, 1]) for j in range(X.shape[1])]))
    leak_cv = cv_r2(X, gamma)
    leak_adj = adj_r2(X, gamma)
    # target validity: is gamma_held just the clamp / W33?
    r2_clamp = r2_on(clamp.astype(float), gamma)
    r2_W33 = r2_on(W33, gamma)
    corr_gW33 = float(np.corrcoef(W33, gamma)[0, 1])
    # W33 signal-to-noise across configs (is the underlying loop just noise?)
    w33_mean, w33_sd = float(W33.mean()), float(W33.std(ddof=1))
    w33_tstat = float(w33_mean / (w33_sd / np.sqrt(n)))     # vs zero (ensemble signal)
    w33_per_config_snr = float(w33_mean / w33_sd)           # per-config S/N
    # tertile composition: is the top gamma tertile == the clamp set (= noise)?
    tert = tertile_labels(gamma)
    clamp_by_tertile = {int(t): round(float(clamp[tert == t].mean()), 3) for t in (0, 1, 2)}
    out["Q1_disjointness_and_target_validity"][f"beta_{beta}"] = {
        "n": n,
        "clamp_fraction": round(float(clamp.mean()), 3),
        "gamma_clamped_mean": round(float(gamma[clamp == 1].mean()) if clamp.any() else float("nan"), 3),
        "gamma_unclamped_mean": round(float(gamma[clamp == 0].mean()) if (clamp == 0).any() else float("nan"), 3),
        "LEAKAGE_max_abs_corr_sig_vs_gamma": round(maxabscorr, 3),
        "LEAKAGE_cv_r2_gamma_on_signature": round(leak_cv, 3),
        "LEAKAGE_adj_r2_gamma_on_signature": round(leak_adj, 3),
        "VALIDITY_r2_gamma_on_clamp_indicator": round(r2_clamp, 3),
        "VALIDITY_r2_gamma_on_W33": round(r2_W33, 3),
        "VALIDITY_corr_gamma_W33": round(corr_gW33, 3),
        "W33_ensemble_mean": round(w33_mean, 5),
        "W33_ensemble_sd": round(w33_sd, 5),
        "W33_signal_tstat_vs_zero": round(w33_tstat, 2),
        "W33_per_config_SNR": round(w33_per_config_snr, 3),
        "clamp_fraction_by_gamma_tertile": clamp_by_tertile,
    }
    poolX.append(X - X.mean(0)); poolG.append(gamma - gamma.mean()); poolC.append(clamp)

# pooled within-beta-demeaned leakage (more power, beta confound removed)
PX, PG = np.vstack(poolX), np.concatenate(poolG)
out["Q1_disjointness_and_target_validity"]["pooled_within_beta_demeaned"] = {
    "n": len(PG),
    "LEAKAGE_cv_r2_gamma_on_signature": round(cv_r2(PX, PG), 3),
    "LEAKAGE_adj_r2_gamma_on_signature": round(adj_r2(PX, PG), 3),
}

os.makedirs("results/yang-mills/phase2/SU2_3D/cheap_a_controls", exist_ok=True)
with open("results/yang-mills/phase2/SU2_3D/cheap_a_controls/q1q5_summary.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
