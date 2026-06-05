"""Role A1 - de-confound stress. Try to BREAK the claim that raw input cannot
linearly read u_t/e_t above the 0.10 det bar. Distinguish LINEAR leak (critical)
from NONLINEAR read (expected/healthy)."""
import sys, numpy as np
from numpy.random import default_rng
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from jepa_0d_accumulator_preflight import AccCfg, gen_accumulator, parity_encode, parity_decode, PSI

EPS = 1e-9

def cv_acc(X, y, cv=4, clf=None):
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return float("nan"), 1.0
    maj = float(counts.max()/counts.sum())
    folds = int(min(cv, counts.min()))
    if folds < 2:
        return float("nan"), maj
    if clf is None:
        clf = LogisticRegression(max_iter=2000, C=1.0)
    acc = float(cross_val_score(clf, X, y, cv=folds, scoring="accuracy").mean())
    return acc, maj

def det(acc, maj):
    if np.isnan(acc): return float("nan")
    return (acc - maj)/max(1.0-maj, EPS)

def smed(xs):
    a = np.asarray([v for v in xs if not (isinstance(v,float) and np.isnan(v))], float)
    return float(np.median(a)) if a.size else float("nan")

def run(seed=0, n=2000, p_noise=0.10, verbose=True):
    cfg = AccCfg(n=n, seed=seed, p_noise=p_noise)
    rng = default_rng(seed)
    d = gen_accumulator(cfg, rng)
    X = d["X"].astype(np.float64)
    u, e = d["u"], d["e"]
    T = cfg.T
    res = {}

    # ---- (A) baseline linear (reproduce committed probe) ----
    u_lin = [det(*cv_acc(X, u[:,t])) for t in range(T)]
    e_lin = [det(*cv_acc(X, e[:,t])) for t in range(T)]
    res["lin_u_max"] = float(np.nanmax(u_lin)); res["lin_u_med"] = smed(u_lin)
    res["lin_e_max"] = float(np.nanmax(e_lin)); res["lin_e_med"] = smed(e_lin)

    # ---- (B) MLP (nonlinear) on raw tokens -> u_t, e_t ----
    mlp = lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, alpha=1e-3, random_state=0)
    u_mlp = [det(*cv_acc(X, u[:,t], clf=mlp())) for t in range(T)]
    e_mlp = [det(*cv_acc(X, e[:,t], clf=mlp())) for t in range(T)]
    res["mlp_u_max"] = float(np.nanmax(u_mlp)); res["mlp_u_med"] = smed(u_mlp)
    res["mlp_e_max"] = float(np.nanmax(e_mlp)); res["mlp_e_med"] = smed(e_mlp)
    res["mlp_u_perT"] = [round(x,3) for x in u_mlp]

    if verbose:
        print(f"[seed{seed} n{n} pn{p_noise}] LINEAR: u_max={res['lin_u_max']:.3f} u_med={res['lin_u_med']:.3f} e_max={res['lin_e_max']:.3f}")
        print(f"           NONLIN MLP: u_max={res['mlp_u_max']:.3f} u_med={res['mlp_u_med']:.3f} e_max={res['mlp_e_max']:.3f}")
        print(f"           MLP u_det perT: {res['mlp_u_perT']}")
    return res, d, cfg

if __name__ == "__main__":
    run(seed=0, n=2000)
