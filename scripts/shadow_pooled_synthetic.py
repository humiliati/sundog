#!/usr/bin/env python
"""H3 Substrate A — synthetic-latent mean-pool net (the imported-wall test).

Pre-registration: docs/atlas/H3_POOLED_SHADOW_PREREG.md  (LOCKED 2026-06-08, slate ww6koomb1).

Tests the Shadow-Invertibility Law's IMPORTED WALL on a *trained* body:
a body g = head o pool o phi has a mean-pool bottleneck z = mean_i phi(u_i) over K units.
The Law predicts the DISCRETE latent d (shared, structurally stable) is DETERMINED from z,
and the CONTINUOUS latent c (carried per-unit with ensemble spread lambda) RESISTS (is lost
from z) -- to the extent the trained encoder phi did NOT learn an averaging-robust c-code.
The split is objective-dependent: clf_d/recon should wash c; reg_c may defeat the resist.

We train ONCE at train-lambda=1.0 (primary, per pre-reg), freeze phi, then probe frozen reps
across the eval-lambda grid. PRE-pool feature = concat of all K per-unit reps {phi(u_i)};
POST-pool feature = z = mean_i phi(u_i). Probe c (R2 via KFold) and d (balanced acc).

NOT public-eligible. Forward-only / no inversion. A clean null is a success.
Run:  python scripts/shadow_pooled_synthetic.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Hyper-parameters (pre-reg §A)
# ---------------------------------------------------------------------------
K = 64                 # units per sample
N_TRAIN = 4000         # training samples
N_PROBE = 1500         # held-out probe samples (per lambda)
F = 32                 # raw unit-feature dim   (24-40 per pre-reg)
H = 32                 # phi output dim
HID = 64               # phi hidden width
N_FREQ = 6             # sinusoidal-embedding frequencies
C_LO, C_HI = 3.0, 7.0  # continuous-latent range
OBS_NOISE = 0.05       # raw-feature observation noise
TRAIN_LAMBDA = 1.0     # train-once lambda (primary)
EPOCHS = 60
BATCH = 256
LR = 2e-3
LAMBDA_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]

DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# FIXED random mixing that ENTANGLES c and d (so a trained phi is genuinely needed)
# ---------------------------------------------------------------------------
_rng = np.random.RandomState(7)
EMB_DIM = 2 * N_FREQ                      # sin+cos per frequency
FREQS = np.linspace(0.5, 4.0, N_FREQ)    # several frequencies
IN_DIM = EMB_DIM + 1                      # [e(c_i) ; d*b]   (b is a scalar gain here -> 1 dim)
B_GAIN = 1.3                              # the fixed d-channel gain "b"
W1 = _rng.randn(48, IN_DIM) / np.sqrt(IN_DIM)   # first mixing layer (fixed)
W2 = _rng.randn(F, 48) / np.sqrt(48)            # second mixing layer (fixed)


def sinusoidal_embed(c_vals):
    """c_vals: (...,) -> (..., 2*N_FREQ) sinusoidal embedding over several frequencies."""
    c_vals = np.asarray(c_vals)
    ang = c_vals[..., None] * FREQS[None, :]           # (..., N_FREQ)
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)


def generate(n, lam, rng):
    """Generate n samples, K units each.

    Returns u (n,K,F) raw unit features, c (n,) continuous latent, d (n,) in {-1,+1}.
    Per-unit spread: c_i = c + lam * xi_i,  xi_i ~ N(0,1).
    u_i = W2 @ tanh(W1 @ [e(c_i); d*b]) + obs_noise.
    """
    c = rng.uniform(C_LO, C_HI, size=n)                 # (n,)
    d = rng.choice([-1.0, 1.0], size=n)                 # balanced in expectation
    xi = rng.randn(n, K)                                # (n,K)
    c_i = c[:, None] + lam * xi                         # (n,K) per-unit continuous
    emb = sinusoidal_embed(c_i)                         # (n,K,EMB_DIM)
    d_chan = (d[:, None] * B_GAIN)[..., None] * np.ones((1, K, 1))  # (n,K,1)
    inp = np.concatenate([emb, d_chan], axis=-1)        # (n,K,IN_DIM)
    h = np.tanh(inp @ W1.T)                             # (n,K,48)
    u = h @ W2.T                                        # (n,K,F)
    u = u + OBS_NOISE * rng.randn(*u.shape)
    return u.astype(np.float32), c.astype(np.float32), d.astype(np.float32)


# ---------------------------------------------------------------------------
# Bodies: phi (2-layer MLP F->HID->H, ReLU); z = mean_i phi(u_i); head.
# ---------------------------------------------------------------------------
class Body(nn.Module):
    def __init__(self, objective):
        super().__init__()
        self.objective = objective
        self.phi = nn.Sequential(
            nn.Linear(F, HID), nn.ReLU(),
            nn.Linear(HID, H), nn.ReLU(),
        )
        if objective == "clf_d":
            self.head = nn.Linear(H, 2)
        elif objective == "reg_c":
            self.head = nn.Linear(H, 1)
        elif objective == "recon":
            self.head = nn.Linear(H, F)
        else:
            raise ValueError(objective)

    def encode(self, u):
        """u (B,K,F) -> per-unit reps (B,K,H)."""
        B, k, f = u.shape
        return self.phi(u.reshape(B * k, f)).reshape(B, k, H)

    def pool(self, u):
        return self.encode(u).mean(dim=1)               # z (B,H)

    def forward(self, u):
        z = self.pool(u)
        return self.head(z)


def train_body(objective, u, c, d):
    """Train one body to convergence at the fixed train-lambda. Returns the trained Body."""
    torch.manual_seed(SEED + hash(objective) % 1000)
    net = Body(objective).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    u_t = torch.from_numpy(u)
    c_t = torch.from_numpy(c)
    d_t = torch.from_numpy(((d + 1) / 2).astype(np.int64))   # {0,1}
    umean_t = torch.from_numpy(u.mean(axis=1))               # (N,F) pooled raw target for recon
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    n = u.shape[0]
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            ub = u_t[idx]
            opt.zero_grad()
            out = net(ub)
            if objective == "clf_d":
                loss = ce(out, d_t[idx])
            elif objective == "reg_c":
                loss = mse(out.squeeze(-1), c_t[idx])
            else:  # recon
                loss = mse(out, umean_t[idx])
            loss.backward()
            opt.step()
    net.eval()
    return net


# ---------------------------------------------------------------------------
# Probes (frozen sklearn on FROZEN reps)
# ---------------------------------------------------------------------------
def probe_c_r2(X, c):
    """5-fold CV R2 of a Ridge regressor for the continuous latent c."""
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    scores = cross_val_score(pipe, X, c, cv=5, scoring="r2")
    return float(np.mean(scores))


def probe_d_acc(X, d):
    """5-fold stratified CV balanced-accuracy of a logistic probe for the discrete latent d."""
    y = ((d + 1) / 2).astype(int)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipe, X, y, cv=skf, scoring="balanced_accuracy")
    return float(np.mean(scores))


@torch.no_grad()
def features(net, u):
    """Return (pre, post): per-unit reps vs the mean-pool shadow.

    POST-pool feature = z = mean_i phi(u_i)  (N,H) -- the shadow under test.

    PRE-pool feature = a SINGLE per-unit rep phi(u_0)  (N,H).  The pre-reg offers
    "mean-of-probe-over-units, OR a single unit" for the pre-pool probe; we use one
    unit because the full concat (N, K*H = 2048) is rank-deficient against N_probe=1500
    and makes Ridge R2 diverge (an ill-posed probe, not real pre-pool information).
    The single-unit rep is the honest "what one un-pooled unit carries about c" baseline
    and keeps the PRE probe in the SAME H-dim space as POST -> a fair pre/post gap.
    """
    u_t = torch.from_numpy(u)
    reps = net.encode(u_t).cpu().numpy()                # (N,K,H)
    post = reps.mean(axis=1)                            # (N,H)  shadow
    pre = reps[:, 0, :]                                 # (N,H)  single un-pooled unit
    return pre, post


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    train_rng = np.random.RandomState(SEED)
    u_tr, c_tr, d_tr = generate(N_TRAIN, TRAIN_LAMBDA, train_rng)

    bodies = {}
    for obj in ["clf_d", "reg_c", "recon"]:
        bodies[obj] = train_body(obj, u_tr, c_tr, d_tr)

    # Fixed probe RNG so every lambda re-draws an independent held-out probe set.
    probe_rng = np.random.RandomState(SEED + 999)

    results = {obj: {"pre": {}, "post": {}} for obj in bodies}
    controls = {}

    # Majority baseline for d (balanced) is 0.5 by construction.
    controls["majority_d_balanced_acc"] = 0.5

    # Label-permutation control: at train-lambda probe set, shuffle c and d -> chance.
    perm_rng = np.random.RandomState(SEED + 31)
    u_perm, c_perm, d_perm = generate(N_PROBE, TRAIN_LAMBDA, perm_rng)
    _, post_perm = features(bodies["clf_d"], u_perm)
    c_shuf = c_perm.copy(); perm_rng.shuffle(c_shuf)
    d_shuf = d_perm.copy(); perm_rng.shuffle(d_shuf)
    controls["label_perm_post_c_r2"] = probe_c_r2(post_perm, c_shuf)
    controls["label_perm_post_d_acc"] = probe_d_acc(post_perm, d_shuf)

    for lam in LAMBDA_GRID:
        u_p, c_p, d_p = generate(N_PROBE, lam, probe_rng)
        for obj, net in bodies.items():
            pre, post = features(net, u_p)
            results[obj]["pre"][str(lam)] = {
                "c_r2": probe_c_r2(pre, c_p),
                "d_acc": probe_d_acc(pre, d_p),
            }
            results[obj]["post"][str(lam)] = {
                "c_r2": probe_c_r2(post, c_p),
                "d_acc": probe_d_acc(post, d_p),
            }

    # -------------------------------------------------------------------
    # Evaluate pre-registered gates P-A1..P-A5 and KILL-A1..A3
    # -------------------------------------------------------------------
    def post_c(obj, lam):  return results[obj]["post"][str(lam)]["c_r2"]
    def post_d(obj, lam):  return results[obj]["post"][str(lam)]["d_acc"]
    def pre_c(obj, lam):   return results[obj]["pre"][str(lam)]["c_r2"]

    gates = {}

    # P-A1 lossiness-essential: lambda=0 -> post-pool c-R2 high (>=0.6) ALL bodies.
    p_a1_vals = {obj: post_c(obj, 0.0) for obj in bodies}
    gates["P-A1"] = {
        "pass": all(v >= 0.6 for v in p_a1_vals.values()),
        "detail": "lambda=0 post-pool c-R2 (>=0.6 all): " +
                  ", ".join(f"{o}={v:.3f}" for o, v in p_a1_vals.items()),
    }

    # P-A2 resist clf_d/recon: lambda->2 post-pool c-R2 low (<=0.15) for clf_d (and largely recon).
    a2_clf = post_c("clf_d", 2.0)
    a2_rec = post_c("recon", 2.0)
    gates["P-A2"] = {
        "pass": a2_clf <= 0.15,
        "detail": f"lambda=2 post-pool c-R2: clf_d={a2_clf:.3f} (<=0.15 gate), recon={a2_rec:.3f}",
    }

    # P-A3 determine: post-pool d-acc stays high (>=0.85) across lambda, all bodies.
    a3_min = {obj: min(post_d(obj, lam) for lam in LAMBDA_GRID) for obj in bodies}
    gates["P-A3"] = {
        "pass": all(v >= 0.85 for v in a3_min.values()),
        "detail": "min over lambda of post-pool d-acc (>=0.85 all): " +
                  ", ".join(f"{o}={v:.3f}" for o, v in a3_min.items()),
    }

    # P-A4 pre/post gap: at lambda=2, PRE-pool c-R2 - POST-pool c-R2 >= 0.4 for clf_d.
    a4_gap = pre_c("clf_d", 2.0) - post_c("clf_d", 2.0)
    gates["P-A4"] = {
        "pass": a4_gap >= 0.4,
        "detail": f"lambda=2 clf_d PRE c-R2={pre_c('clf_d',2.0):.3f} - POST c-R2={post_c('clf_d',2.0):.3f} "
                  f"= gap {a4_gap:.3f} (>=0.4 gate)",
    }

    # P-A5 objective-dependence (headline): at lambda=2, post-pool c-R2(reg_c) > c-R2(clf_d) by >=0.2.
    a5_diff = post_c("reg_c", 2.0) - post_c("clf_d", 2.0)
    gates["P-A5"] = {
        "pass": a5_diff >= 0.2,
        "detail": f"lambda=2 post-pool c-R2: reg_c={post_c('reg_c',2.0):.3f} - clf_d={post_c('clf_d',2.0):.3f} "
                  f"= diff {a5_diff:.3f} (>=0.2 gate)",
    }

    # Kill criteria
    kills = []
    if post_c("clf_d", 2.0) > 0.5:
        kills.append("KILL-A1")  # clf_d keeps continuous through pooling -> imported wall fails
    if any(post_d(obj, lam) < 0.6 for obj in bodies for lam in LAMBDA_GRID):
        # "drops to chance" -> near 0.5 balanced; use 0.6 as a generous chance band
        kills.append("KILL-A2")
    # KILL-A3: c-resist is lambda-INDEPENDENT (flat) -> not a lossiness effect.
    clf_post_c_curve = [post_c("clf_d", lam) for lam in LAMBDA_GRID]
    c_resist_range = max(clf_post_c_curve) - min(clf_post_c_curve)
    if c_resist_range < 0.1:
        kills.append("KILL-A3")

    out = {
        "meta": {
            "seed": SEED, "K": K, "N_train": N_TRAIN, "N_probe": N_PROBE,
            "F": F, "H": H, "hidden": HID, "n_freq": N_FREQ, "c_range": [C_LO, C_HI],
            "obs_noise": OBS_NOISE, "train_lambda": TRAIN_LAMBDA, "epochs": EPOCHS,
            "lambda_grid": LAMBDA_GRID,
            "d_probe": "balanced-accuracy (majority/chance = 0.5)",
            "c_probe": "Ridge 5-fold CV R2",
        },
        "results": results,
        "controls": controls,
        "gates": gates,
        "kills_triggered": kills,
        "c_resist_lambda_range_clf_d": c_resist_range,
    }

    out_path = Path(__file__).resolve().parent.parent / "results" / "atlas" / "h3" / "synthetic_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    # Console summary
    print("=== H3 Substrate A — pooled-shadow synthetic ===")
    print(f"controls: {controls}")
    hdr = "lam   " + "  ".join(f"{l:>5}" for l in LAMBDA_GRID)
    for obj in bodies:
        print(f"\n[{obj}]")
        print(hdr)
        print("postC " + "  ".join(f"{post_c(obj,l):>5.2f}" for l in LAMBDA_GRID))
        print("postD " + "  ".join(f"{post_d(obj,l):>5.2f}" for l in LAMBDA_GRID))
        print("preC  " + "  ".join(f"{pre_c(obj,l):>5.2f}" for l in LAMBDA_GRID))
    print("\n=== GATES ===")
    for g, v in gates.items():
        print(f"{g}: {'PASS' if v['pass'] else 'FAIL'} — {v['detail']}")
    print(f"\nKILLS triggered: {kills if kills else 'none'}")
    print(f"wrote {out_path}")
    return out


if __name__ == "__main__":
    main()
