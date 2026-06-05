"""ROLE A3 scratch: metric & support audit of the JEPA-0D preflight.

We do NOT modify the committed script. We import its generator and re-implement
the det metric to test ordinal vs categorical, early-tick instability, position-only
neutralisation, and flip-read support.
"""
import numpy as np
from numpy.random import default_rng
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold

from jepa_0d_accumulator_preflight import AccCfg, gen_accumulator, PSI

EPS = 1e-9


def cv_predict(X, y, cv=4):
    """Return per-row CV predictions (multiclass) plus majority baseline."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return None, 1.0
    maj = float(counts.max() / counts.sum())
    folds = int(min(cv, counts.min()))
    if folds < 2:
        return None, maj
    clf = LogisticRegression(max_iter=2000, C=1.0)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    yhat = cross_val_predict(clf, X, y, cv=skf)
    return yhat, maj


def det(acc, maj):
    return (acc - maj) / max(1.0 - maj, EPS)


def main():
    cfg = AccCfg()
    cfg.n = 2000  # keep <=2000 for speed per instructions
    rng = default_rng(0)
    data = gen_accumulator(cfg, rng)
    X = data["X"].astype(np.float64)
    u = data["u"]
    T = cfg.T

    print(f"L={data['L']} n={cfg.n}")
    print("\n=== PER-TICK: accuracy-det vs within-1 (adjacent) det vs MAE ===")
    print(f"{'tick':>4} {'maj':>6} {'acc':>6} {'acc_det':>8} "
          f"{'w1_acc':>7} {'w1_maj':>7} {'w1_det':>7} "
          f"{'mae':>6} {'mae_maj':>7} {'mae_red':>7} {'Emaj':>5}")
    acc_dets, w1_dets = [], []
    for t in range(T):
        y = u[:, t]
        yhat, maj = cv_predict(X, y)
        if yhat is None:
            print(f"{t+1:>4} (single class)")
            continue
        acc = float((yhat == y).mean())
        # within-1 (ordinal-tolerant) accuracy
        w1_acc = float((np.abs(yhat - y) <= 1).mean())
        # within-1 baseline: best constant predictor under within-1 tolerance.
        # for a constant c, P(|c - y|<=1). pick best c.
        vals = np.arange(y.max() + 1)
        w1_base = max(float((np.abs(c - y) <= 1).mean()) for c in vals)
        # MAE of model vs MAE of the optimal constant under L1 (the median)
        mae = float(np.abs(yhat - y).mean())
        med = int(np.median(y))
        mae_maj = float(np.abs(med - y).mean())
        mae_red = (mae_maj - mae) / max(mae_maj, EPS)  # fractional MAE reduction
        Emaj = "Y" if maj > 0.90 else ""
        ad = det(acc, maj)
        w1d = det(w1_acc, w1_base)
        acc_dets.append(ad)
        w1_dets.append(w1d)
        print(f"{t+1:>4} {maj:>6.3f} {acc:>6.3f} {ad:>8.4f} "
              f"{w1_acc:>7.3f} {w1_base:>7.3f} {w1d:>7.4f} "
              f"{mae:>6.3f} {mae_maj:>7.3f} {mae_red:>7.4f} {Emaj:>5}")

    print(f"\nacc_det   median={np.median(acc_dets):.4f}  max={np.max(acc_dets):.4f}")
    print(f"within1_det median={np.median(w1_dets):.4f}  max={np.max(w1_dets):.4f}")

    # Ticks where majority is so high det is near-meaningless (small denominator OR small minority)
    print("\n=== majority profile (denominator 1-maj) ===")
    for t in range(T):
        vals, cnts = np.unique(u[:, t], return_counts=True)
        maj = cnts.max() / cnts.sum()
        print(f"tick {t+1:>2}: maj={maj:.3f}  1-maj={1-maj:.3f}  "
              f"n_minority={int((1-maj)*cfg.n)}  n_classes={len(vals)}")


if __name__ == "__main__":
    main()
