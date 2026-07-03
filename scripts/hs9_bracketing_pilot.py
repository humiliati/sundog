"""HS9 design-review Tier-B pilot: bracketing controls ONLY (the fold question is never touched).

Trains the alpha=0 trunk on the two-route base task, freezes it, then trains three heads on the
route-dominance target r(x): CEILING (full hidden state), FLOOR-1 (scalar pooled mean s1),
FLOOR-4 (4 group-means s4). Emits results/atlas/hs9/bracketing_pilot.json with per-seed metrics
and the pinned promote-criteria evaluation (criteria live in
docs/atlas/HS9_INTROSPECT_FOLD_DESIGN_REVIEW.md section 4, committed before this ran).

Deterministic: numpy only, fixed seeds, full-batch Adam.
"""
import json
import os
import numpy as np

SEEDS = [51235, 61235, 1789]
N_TRAIN, N_EVAL = 4096, 2048
M = 8              # dims per group
WIDTH = 32
HEAD_HIDDEN = 16
TRUNK_STEPS, HEAD_STEPS = 2000, 1500
LR = 0.01
OUT = os.path.join("results", "atlas", "hs9", "bracketing_pilot.json")


def make_data(rng, n):
    y = rng.choice([-1.0, 1.0], size=n)
    u_a = rng.standard_normal(M)
    u_a /= np.linalg.norm(u_a)
    u_b = rng.standard_normal(M)
    u_b /= np.linalg.norm(u_b)
    degraded_a = rng.random(n) < 0.5  # True -> group A degraded, B reliable
    xa = np.where(degraded_a[:, None], 0.2, 1.0)[:, 0:1] * y[:, None] * u_a[None, :] \
        + np.where(degraded_a[:, None], 1.2, 0.3) * rng.standard_normal((n, M))
    xb = np.where(degraded_a[:, None], 1.0, 0.2)[:, 0:1] * y[:, None] * u_b[None, :] \
        + np.where(degraded_a[:, None], 0.3, 1.2) * rng.standard_normal((n, M))
    return np.concatenate([xa, xb], axis=1), y


class Adam:
    def __init__(self, params, lr):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, 0.9, 0.999, 1e-8, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train_trunk(rng, x, y):
    w1 = rng.standard_normal((x.shape[1], WIDTH)) * 0.3
    b1 = np.zeros(WIDTH)
    w2 = rng.standard_normal(WIDTH) * 0.3
    b2 = np.zeros(1)
    params = [w1, b1, w2, b2]
    opt = Adam(params, LR)
    for _ in range(TRUNK_STEPS):
        z = x @ w1 + b1
        h = np.tanh(z)
        logit = h @ w2 + b2
        # logistic loss on labels ±1: log(1+exp(-y*logit))
        s = -y * logit
        p = 1.0 / (1.0 + np.exp(-s))          # sigmoid(s)
        dlogit = (-y * p) / len(y)
        dw2 = h.T @ dlogit
        db2 = np.array([dlogit.sum()])
        dh = np.outer(dlogit, w2)
        dz = dh * (1 - h * h)
        dw1 = x.T @ dz
        db1 = dz.sum(axis=0)
        opt.step(params, [dw1, db1, dw2, db2])
    return params


def trunk_forward(params, x):
    w1, b1, w2, b2 = params
    h = np.tanh(x @ w1 + b1)
    return h, h @ w2 + b2


def route_labels(params, x):
    """r(x) = [ |Delta_A| >= |Delta_B| ], Delta_G = logit(x) - logit(x with group G zeroed)."""
    _, logit = trunk_forward(params, x)
    xa0 = x.copy()
    xa0[:, :M] = 0.0
    _, la = trunk_forward(params, xa0)
    xb0 = x.copy()
    xb0[:, M:] = 0.0
    _, lb = trunk_forward(params, xb0)
    return (np.abs(logit - la) >= np.abs(logit - lb)).astype(float)


def train_head(rng, feats, r):
    w1 = rng.standard_normal((feats.shape[1], HEAD_HIDDEN)) * 0.5
    b1 = np.zeros(HEAD_HIDDEN)
    w2 = rng.standard_normal(HEAD_HIDDEN) * 0.5
    b2 = np.zeros(1)
    params = [w1, b1, w2, b2]
    opt = Adam(params, LR)
    t = 2.0 * r - 1.0  # ±1 targets
    for _ in range(HEAD_STEPS):
        h = np.tanh(feats @ w1 + b1)
        logit = h @ w2 + b2
        s = -t * logit
        p = 1.0 / (1.0 + np.exp(-s))
        dlogit = (-t * p) / len(t)
        dw2 = h.T @ dlogit
        db2 = np.array([dlogit.sum()])
        dh = np.outer(dlogit, w2)
        dz = dh * (1 - h * h)
        dw1 = feats.T @ dz
        db1 = dz.sum(axis=0)
        opt.step(params, [dw1, db1, dw2, db2])
    return params


def head_acc(params, feats, r):
    w1, b1, w2, b2 = params
    logit = np.tanh(feats @ w1 + b1) @ w2 + b2
    return float((((logit > 0).astype(float)) == r).mean())


def run_seed(seed):
    rng = np.random.default_rng(seed)
    x, y = make_data(rng, N_TRAIN + N_EVAL)
    xt, yt = x[:N_TRAIN], y[:N_TRAIN]
    xe, ye = x[N_TRAIN:], y[N_TRAIN:]
    trunk = train_trunk(rng, xt, yt)
    _, logit_e = trunk_forward(trunk, xe)
    base_acc = float(((logit_e > 0) == (ye > 0)).mean())
    rt, re = route_labels(trunk, xt), route_labels(trunk, xe)
    maj = float(max(re.mean(), 1 - re.mean()))
    ht, _ = trunk_forward(trunk, xt)
    he, _ = trunk_forward(trunk, xe)
    s1t, s1e = ht.mean(axis=1, keepdims=True), he.mean(axis=1, keepdims=True)
    s4t = ht.reshape(len(ht), 4, WIDTH // 4).mean(axis=2)
    s4e = he.reshape(len(he), 4, WIDTH // 4).mean(axis=2)
    ceiling = head_acc(train_head(rng, ht, rt), he, re)
    floor1 = head_acc(train_head(rng, s1t, rt), s1e, re)
    floor4 = head_acc(train_head(rng, s4t, rt), s4e, re)
    row = dict(seed=seed, base_acc=base_acc, route_minority=round(1 - maj, 4),
               majority_rate=maj, ceiling_acc=ceiling, floor1_acc=floor1, floor4_acc=floor4)
    row["criteria"] = dict(
        c1_base=base_acc >= 0.90,
        c2_balance=(1 - maj) >= 0.30,
        c3_ceiling=ceiling >= 0.90,
        c4_floor=floor1 <= maj + 0.05,
        c5_gap=(ceiling - floor1) >= 0.35,
    )
    row["all_pass"] = all(row["criteria"].values())
    return row


def main():
    rows = [run_seed(s) for s in SEEDS]
    verdict = "PROMOTE" if all(r["all_pass"] for r in rows) else "PARK"
    out = dict(pilot="hs9_bracketing", seeds=SEEDS, rows=rows, verdict=verdict,
               criteria_doc="docs/atlas/HS9_INTROSPECT_FOLD_DESIGN_REVIEW.md#4")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    for r in rows:
        print(f"seed {r['seed']}: base {r['base_acc']:.3f} minority {r['route_minority']:.3f} "
              f"ceiling {r['ceiling_acc']:.3f} floor1 {r['floor1_acc']:.3f} "
              f"floor4 {r['floor4_acc']:.3f} pass={r['all_pass']}")
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
