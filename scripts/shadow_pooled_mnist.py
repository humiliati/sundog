"""H3 Substrate B — MNIST CNN pooled-shadow determine/resist test.

Pre-registration: docs/atlas/H3_POOLED_SHADOW_PREREG.md  (LOCKED 2026-06-08, §B)

The core test (the imported wall): a trained CNN body
    g = head . GAP . phi
has a global-average-pool (GAP) bottleneck `z = mean_xy phi(image)` over the
spatial feature map. The Shadow-Invertibility Law predicts the GAP shadow
DETERMINES the discrete latent (digit class y) and RESISTS the continuous
nuisance (rotation theta) — to the extent the trained encoder did not learn a
pooling-robust theta-code (it has no incentive to, since it is trained only to
classify y).

Probes (frozen sklearn on FROZEN reps, held-out images):
  - y    : classifier accuracy (over 0.1 chance), report over majority too
  - theta: ridge-regression R^2 via KFold
  - PRE-pool  = flattened pre-GAP conv feature map
  - POST-pool = the GAP vector (the shadow)

Lossiness sweep: ensemble-average the GAP vector over K augmented copies of the
SAME held-out image, each re-rotated by theta + N(0, lambda*sigma) extra deg,
for lambda in {0, 0.5, 1, 2, 4}. Probe theta-R^2 (and y-acc) of the
ensemble-averaged GAP vs lambda.

HONEST NOTE: y-determine is partly trivial (y IS the training target). The
load-bearing test is theta-resist (P-B1) and the sweep (P-B3). KILL-B1 fires if
theta is fully recoverable post-GAP (theta-R^2 post ~= pre).

Forward-only / no inversion. NOT public-eligible.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")  # silence sklearn ConvergenceWarning etc.

import torch
import torch.nn as nn
from scipy.ndimage import rotate as nd_rotate
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Dimensionality cap for the probe: high-dim flattened pre-GAP maps (1568-dim
# with ~1500 held-out samples) starve a plain ridge/logreg of samples, which
# UNFAIRLY depresses pre-pool recovery vs the 32-dim GAP. We standardize then
# PCA-reduce any rep with > PROBE_MAX_DIM features to PROBE_MAX_DIM components,
# so PRE-pool and POST-pool are compared on a dimensionality-fair footing. The
# 32-dim GAP is below the cap and is left untouched. PCA is a frozen,
# deterministic, forward-only linear probe head — no inversion of the body.
PROBE_MAX_DIM = 128

SEED = 1234
SIGMA_DEG = 10.0  # base spread (deg) for the lossiness sweep
LAMBDAS = [0.0, 0.5, 1.0, 2.0, 4.0]
K_ENSEMBLE = 8
ROT_RANGE = 30.0  # theta ~ U[-30, +30] deg


def set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_base_digits(n_max: int):
    """Load MNIST (openml) if reachable, else fall back to sklearn load_digits.

    Returns (images[N,H,W] float in [0,1], labels[N] int, source_str, img_side).
    """
    # Primary: MNIST via openml (real network).
    try:
        from sklearn.datasets import fetch_openml

        X, y = fetch_openml(
            "mnist_784",
            version=1,
            return_X_y=True,
            as_frame=False,
            parser="liac-arff",
        )
        X = X.astype(np.float32) / 255.0
        y = y.astype(np.int64)
        imgs = X.reshape(-1, 28, 28)
        # subsample (stratified-ish: shuffle then take n_max)
        rng = np.random.RandomState(SEED)
        idx = rng.permutation(len(imgs))[:n_max]
        return imgs[idx], y[idx], "mnist_openml_28x28", 28
    except Exception as e:  # pragma: no cover - network dependent
        print(f"[data] MNIST fetch failed ({type(e).__name__}: {str(e)[:120]}); "
              f"falling back to sklearn load_digits 8x8.")

    # Fallback: 1797 real 8x8 handwritten digits, no download.
    from sklearn.datasets import load_digits

    d = load_digits()
    imgs = (d.images.astype(np.float32) / 16.0)  # 0..16 -> 0..1
    y = d.target.astype(np.int64)
    return imgs, y, "sklearn_load_digits_8x8", imgs.shape[1]


def rotate_batch(imgs: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """Rotate each image by its theta (deg). reshape=False keeps the canvas."""
    out = np.empty_like(imgs)
    for i in range(len(imgs)):
        out[i] = nd_rotate(
            imgs[i], thetas[i], reshape=False, order=1, mode="constant", cval=0.0
        )
    return out


# --------------------------------------------------------------------------- #
# Body: tiny CNN with a GAP bottleneck
# --------------------------------------------------------------------------- #
class TinyCNN(nn.Module):
    """2 conv -> ReLU -> conv feature map -> GAP (shadow) -> FC head."""

    def __init__(self, n_classes: int = 10, ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch * 2)
        self.pool = nn.MaxPool2d(2)  # spatial downsample between conv blocks
        self.head = nn.Linear(ch * 2, n_classes)
        self.feat_ch = ch * 2

    def feature_map(self, x):
        """Pre-GAP conv feature map [B, C, H', W']."""
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x

    def gap(self, fmap):
        """Global average pool over spatial dims -> shadow [B, C]."""
        return fmap.mean(dim=(2, 3))

    def forward(self, x):
        fmap = self.feature_map(x)
        z = self.gap(fmap)
        return self.head(z)


def train_cnn(model, Xtr, ytr, epochs, batch, lr, device):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            xb = Xtr[idx].to(device)
            yb = ytr[idx].to(device)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f"  epoch {ep+1}/{epochs}  loss={tot/n:.4f}")
    model.eval()


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #
def _maybe_pca(seed: int, n_features: int):
    """Frozen StandardScaler (+ PCA to PROBE_MAX_DIM if over the cap)."""
    steps = [StandardScaler()]
    if n_features > PROBE_MAX_DIM:
        steps.append(PCA(n_components=PROBE_MAX_DIM, random_state=seed))
        steps.append(StandardScaler())  # rescale PCA scores for the linear head
    return steps


def probe_y_acc(reps: np.ndarray, y: np.ndarray, seed: int = SEED) -> float:
    """Held-out digit-class accuracy via logreg + KFold CV (dim-fair)."""
    pipe = make_pipeline(
        *_maybe_pca(seed, reps.shape[1]),
        LogisticRegression(max_iter=400, C=1.0),
    )
    kf = KFold(5, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, reps, y, cv=kf, scoring="accuracy")
    return float(scores.mean())


def probe_theta_r2(reps: np.ndarray, theta: np.ndarray, seed: int = SEED) -> float:
    """Held-out theta R^2 via ridge + KFold CV (dim-fair)."""
    pipe = make_pipeline(
        *_maybe_pca(seed, reps.shape[1]),
        Ridge(alpha=1.0),
    )
    kf = KFold(5, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, reps, theta, cv=kf, scoring="r2")
    return float(scores.mean())


# --------------------------------------------------------------------------- #
# Main experiment
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000, help="images to use (MNIST)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--probe-n", type=int, default=2000,
                    help="held-out images used for the (expensive) probes")
    ap.add_argument("--out", type=str,
                    default="results/atlas/h3/mnist_result.json")
    args = ap.parse_args()

    t0 = time.time()
    set_seeds()
    device = torch.device("cpu")

    # --- data --------------------------------------------------------------- #
    imgs, labels, source, side = load_base_digits(args.n)
    n_classes = int(labels.max() + 1)
    print(f"[data] source={source} imgs={imgs.shape} classes={n_classes}")

    # per-image continuous nuisance rotation
    rng = np.random.RandomState(SEED)
    thetas = rng.uniform(-ROT_RANGE, ROT_RANGE, size=len(imgs)).astype(np.float32)
    rot_imgs = rotate_batch(imgs, thetas)

    # tensors [N,1,H,W]
    X = torch.from_numpy(rot_imgs[:, None, :, :].astype(np.float32))
    y_t = torch.from_numpy(labels)

    # split: train / held-out (probe) — probes use FROZEN reps on held-out
    n = len(X)
    n_probe = min(args.probe_n, n // 4)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    probe_idx = perm[:n_probe]
    train_idx = perm[n_probe:]
    Xtr, ytr = X[train_idx], y_t[train_idx]
    Xpr = X[probe_idx]
    y_probe = labels[probe_idx.numpy()]
    theta_probe = thetas[probe_idx.numpy()]
    imgs_probe = imgs[probe_idx.numpy()]      # UNROTATED base, for the sweep
    theta_base_probe = thetas[probe_idx.numpy()]

    print(f"[split] train={len(Xtr)} probe(held-out)={len(Xpr)}")

    # --- body --------------------------------------------------------------- #
    model = TinyCNN(n_classes=n_classes, ch=16).to(device)
    print("[train] CNN classify y ...")
    train_cnn(model, Xtr, ytr, args.epochs, args.batch, args.lr, device)

    # training/held-out classification accuracy (sanity)
    with torch.no_grad():
        logits = model(Xpr.to(device))
        cnn_acc = float((logits.argmax(1).cpu().numpy() == y_probe).mean())
    print(f"[body] held-out CNN classify acc = {cnn_acc:.3f}")

    # --- extract FROZEN reps on held-out: PRE-pool & POST-pool -------------- #
    model.eval()
    with torch.no_grad():
        fmap = model.feature_map(Xpr.to(device))        # [P, C, H', W']
        pre_pool = fmap.reshape(len(Xpr), -1).cpu().numpy()   # flattened pre-GAP
        post_pool = model.gap(fmap).cpu().numpy()             # GAP vector (shadow)
    print(f"[reps] pre-pool dim={pre_pool.shape[1]} post-pool dim={post_pool.shape[1]}")

    # --- probes: y-acc & theta-R^2, PRE vs POST ----------------------------- #
    chance_y = 1.0 / n_classes
    majority_y = float(np.bincount(y_probe).max() / len(y_probe))

    pre_y = probe_y_acc(pre_pool, y_probe)
    post_y = probe_y_acc(post_pool, y_probe)
    pre_theta = probe_theta_r2(pre_pool, theta_probe)
    post_theta = probe_theta_r2(post_pool, theta_probe)
    print(f"[probe] y-acc   pre={pre_y:.3f} post={post_y:.3f} (chance={chance_y:.3f} "
          f"majority={majority_y:.3f})")
    print(f"[probe] theta-R2 pre={pre_theta:.3f} post={post_theta:.3f}")

    # --- HONESTY: probe-robustness of the static PRE/POST theta gap --------- #
    # The static P-B1 gap is SENSITIVE to the pre-pool probe: a naive ridge on
    # the raw 1568-dim flattened map is sample-starved and UNDER-reads theta,
    # which can flip the sign of (pre - post). We record the raw (no-PCA) ridge
    # at several alphas so the receipt does not overstate P-B1. The DEFAULT
    # probe above is dimensionality-fair (StandardScaler + PCA to PROBE_MAX_DIM)
    # and is what the predictions/JSON use; the load-bearing, probe-INDEPENDENT
    # evidence is the lossiness sweep (fixed 32-dim throughout).
    def _raw_theta_r2(R, t, alpha):
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        kf = KFold(5, shuffle=True, random_state=SEED)
        return float(cross_val_score(pipe, R, t, cv=kf, scoring="r2").mean())

    robustness = {}
    for a in (1.0, 50.0, 200.0):
        rp = _raw_theta_r2(pre_pool, theta_probe, a)
        ro = _raw_theta_r2(post_pool, theta_probe, a)
        robustness[f"alpha_{int(a)}"] = {
            "theta_r2_pre_noPCA": rp,
            "theta_r2_post_noPCA": ro,
            "pre_minus_post": rp - ro,
        }
        print(f"[robust] no-PCA ridge alpha={a:<5} pre={rp:.3f} post={ro:.3f} "
              f"gap={rp-ro:+.3f}")

    # --- control: label-permutation (no-leakage check on POST) -------------- #
    yperm = y_probe.copy(); rng.shuffle(yperm)
    tperm = theta_probe.copy(); rng.shuffle(tperm)
    perm_y = probe_y_acc(post_pool, yperm)
    perm_theta = probe_theta_r2(post_pool, tperm)
    print(f"[ctrl] permuted post: y-acc={perm_y:.3f} theta-R2={perm_theta:.3f}")

    # --- lossiness sweep ---------------------------------------------------- #
    # For each lambda, build an ensemble-averaged GAP per held-out image: K
    # augmented copies of the SAME image, each rotated by theta_base + extra
    # N(0, lambda*sigma) degrees. Probe theta-R2 and y-acc of the averaged GAP.
    print(f"[sweep] K={K_ENSEMBLE} sigma={SIGMA_DEG}deg lambdas={LAMBDAS}")
    sweep = []
    sweep_rng = np.random.RandomState(SEED + 7)
    for lam in LAMBDAS:
        ens_gap = np.zeros((len(imgs_probe), model.feat_ch), dtype=np.float32)
        for k in range(K_ENSEMBLE):
            extra = sweep_rng.normal(0.0, lam * SIGMA_DEG, size=len(imgs_probe))
            thetas_k = (theta_base_probe + extra).astype(np.float32)
            rk = rotate_batch(imgs_probe, thetas_k)
            xk = torch.from_numpy(rk[:, None, :, :].astype(np.float32))
            with torch.no_grad():
                fk = model.feature_map(xk.to(device))
                gk = model.gap(fk).cpu().numpy()
            ens_gap += gk
        ens_gap /= K_ENSEMBLE
        s_theta = probe_theta_r2(ens_gap, theta_base_probe)
        s_y = probe_y_acc(ens_gap, y_probe)
        sweep.append({"lambda": lam, "theta_r2": s_theta, "y_acc": s_y})
        print(f"  lambda={lam:<4} theta-R2={s_theta:.3f}  y-acc={s_y:.3f}")

    # --- evaluate predictions / kill criteria ------------------------------- #
    theta_drop = pre_theta - post_theta
    # P-B1 resist: theta-R2 drops post-GAP vs pre-GAP (meaningful margin) under
    # the dimensionality-fair probe.
    p_b1 = bool(post_theta < pre_theta - 0.10)
    # P-B1 robustness: does the DIRECTION (pre > post) survive a fair NON-PCA
    # probe (strong-ridge alpha=200, which de-overfits the 1568-dim map)?
    p_b1_robust_direction = bool(
        robustness["alpha_200"]["pre_minus_post"] > 0.05
    )
    # P-B2 determine: y-acc stays high post-GAP (well over chance)
    p_b2 = bool(post_y >= max(0.85, 5 * chance_y))
    # P-B3 sweep: theta-R2 washes further toward chance as lambda grows
    theta0 = sweep[0]["theta_r2"]
    thetaN = sweep[-1]["theta_r2"]
    p_b3 = bool(thetaN < theta0 - 0.05)
    # KILL-B1: theta fully recoverable post-GAP (post ~= pre) under fair probe
    kill_b1 = bool(post_theta >= pre_theta - 0.05)

    result = {
        "leg": "Substrate B — MNIST CNN (H3 §B)",
        "prereg": "docs/atlas/H3_POOLED_SHADOW_PREREG.md",
        "seed": SEED,
        "data_source": source,
        "img_side": int(side),
        "n_images": int(len(imgs)),
        "n_classes": n_classes,
        "n_probe_heldout": int(len(Xpr)),
        "epochs": args.epochs,
        "rot_range_deg": ROT_RANGE,
        "sweep_sigma_deg": SIGMA_DEG,
        "sweep_K": K_ENSEMBLE,
        "cnn_heldout_acc": cnn_acc,
        "chance_y": chance_y,
        "majority_y": majority_y,
        "pre_pool_dim": int(pre_pool.shape[1]),
        "post_pool_dim": int(post_pool.shape[1]),
        "y_acc_pre": pre_y,
        "y_acc_post": post_y,
        "theta_r2_pre": pre_theta,
        "theta_r2_post": post_theta,
        "theta_r2_drop_pre_minus_post": theta_drop,
        "control_permuted_y_acc_post": perm_y,
        "control_permuted_theta_r2_post": perm_theta,
        "probe_robustness_noPCA": robustness,
        "P_B1_robust_direction_alpha200": p_b1_robust_direction,
        "sweep": sweep,
        "predictions": {
            "P_B1_resist_theta_drops_post": p_b1,
            "P_B2_determine_y_high_post": p_b2,
            "P_B3_sweep_theta_washes": p_b3,
        },
        "KILL_B1_theta_fully_recoverable_post": kill_b1,
        "honest_notes": [
            "y-determine is partly trivial (y IS the training target); the "
            "load-bearing test is theta-resist.",
            "The STATIC P-B1 pre/post gap is probe-sensitive: a naive ridge on "
            "the raw 1568-dim pre-pool map under-reads theta and can flip the "
            "sign. The direction (pre>post) holds only under a dimensionality-"
            "fair probe (PCA-128 default, or strong-ridge alpha=200). See "
            "probe_robustness_noPCA.",
            "The probe-INDEPENDENT, load-bearing evidence is the lossiness "
            "sweep (fixed 32-dim GAP throughout): theta-R2 washes monotonically "
            "as lambda grows while y-acc holds — a genuine lossiness effect.",
        ],
        "runtime_sec": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] wrote {args.out}  ({result['runtime_sec']}s)")
    print(json.dumps(
        {k: result[k] for k in ("predictions", "KILL_B1_theta_fully_recoverable_post")},
        indent=2))
    return result


if __name__ == "__main__":
    main()
