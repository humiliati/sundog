"""ROLE A3 scratch part B:
1. Reproduce the committed script's EXACT det path (cross_val_score, no shuffle) per tick,
   to find ticks where det is numerically unstable / dominated by rounding.
2. The CORE downstream question: a TinyGPT body has POSITION embeddings. At read time the
   body read is per-position (a fixed checkpoint tick). Can a position-only predictor get
   u_det credit? The per-position-majority baseline is supposed to neutralise E[u_t|t].
   But the body read at gate-4/5 is on the LATENT, and the question is whether 'majority per
   position' is the right null when the read is multiclass logistic on a body that KNOWS t.
   Simulate: give the probe ONLY the position (constant feature) -> it must score det~0.
   Then give it position PLUS a tiny noisy hint and see how det behaves.
3. Tick-6 single-class anomaly check.
"""
import numpy as np
from numpy.random import default_rng
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from jepa_0d_accumulator_preflight import AccCfg, gen_accumulator, _cv_acc, _det

EPS = 1e-9


def main():
    cfg = AccCfg()
    cfg.n = 2000
    rng = default_rng(0)
    data = gen_accumulator(cfg, rng)
    X = data["X"].astype(np.float64)
    u = data["u"]
    T = cfg.T

    print("=== EXACT committed-script det path per tick (cross_val_score, folds=min(4,minclass)) ===")
    for t in range(T):
        acc, maj = _cv_acc(X, u[:, t], multiclass=True)
        d = _det(acc, maj)
        vals, cnts = np.unique(u[:, t], return_counts=True)
        minclass = cnts.min()
        folds = min(4, minclass)
        print(f"tick {t+1:>2}: acc={acc:.4f} maj={maj:.4f} det={d:.4f} "
              f"minclass={minclass} folds={folds} 1-maj={1-maj:.4f}")

    print("\n=== POSITION-ONLY null: does a probe that only knows position score det~0? ===")
    # Build a 'body' read that is JUST the position info. For a single tick the position is
    # constant across rows, so a per-tick probe with constant X is degenerate. The realistic
    # downstream scenario: a body read where the feature carries E[u_t|t] (position prior) but
    # NOT u_t itself. Simulate a body latent = one-hot-ish encoding of t plus pure noise.
    # If the per-position-majority baseline neutralises position, det should be ~0.
    for t in [4, 8, 12]:
        y = u[:, t-1]
        # latent that knows position perfectly but nothing about this row's u: pure noise feature
        feat = rng.normal(size=(cfg.n, 16))  # noise only
        acc, maj = _cv_acc(feat, y, multiclass=True)
        print(f"tick {t}: NOISE-only feat -> acc={acc:.4f} maj={maj:.4f} det={_det(acc,maj):.4f}")
        # latent that = the position prior probabilities replicated as features (still no per-row u)
        # i.e. feature = E[u|t] broadcast -> constant -> logistic should predict majority -> det~0
        prior_feat = np.tile(np.bincount(y, minlength=7) / cfg.n, (cfg.n, 1))
        acc2, maj2 = _cv_acc(prior_feat, y, multiclass=True)
        print(f"tick {t}: PRIOR-const feat -> acc={acc2:.4f} maj={maj2:.4f} det={_det(acc2,maj2):.4f}")

    print("\n=== Tick-6 anomaly: did StratifiedKFold collapse it earlier? check raw class counts ===")
    for t in [5, 6, 7]:
        vals, cnts = np.unique(u[:, t-1], return_counts=True)
        print(f"tick {t}: classes={vals.tolist()} counts={cnts.tolist()}")


if __name__ == "__main__":
    main()
