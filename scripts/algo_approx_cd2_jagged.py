"""
C-D2 v3 (decisive) — generalization-width threshold vs tropical complexity, with a target
whose epsilon-approximation complexity actually equals its exact piece count.

v1/v2 were inconclusive because both targets were CONVEX (smooth at coarse scale), so the
net could clear the variance bar with far fewer than k pieces. This v3 uses a NON-CONVEX,
jagged continuous PL with k pieces and random (non-monotone) slopes: dropping any breakpoint
incurs O(1) error, so eps-complexity ~ exact complexity ~ k. A 1-hidden-layer ReLU net needs
~ k-1 units (one per breakpoint). Prediction: threshold ~ k-1, i.e. LINEAR in k.

Falsifier GROKS_FAR_BELOW_COMPILED_SIZE: even here, generalization happens far below k.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)


def make_jagged(k, lo=-2.0, hi=2.0, seed=1234):
    """Continuous non-convex PL with k pieces (k-1 evenly spaced breakpoints), random slopes."""
    rng = np.random.default_rng(seed + 7 * k)
    zs = np.linspace(lo, hi, k + 1)[1:-1]                 # k-1 interior breakpoints
    slopes = rng.uniform(-2.0, 2.0, size=k)              # random, non-monotone -> jagged
    intercepts = np.zeros(k)
    for i in range(1, k):                                 # enforce continuity at each breakpoint
        intercepts[i] = intercepts[i - 1] + (slopes[i - 1] - slopes[i]) * zs[i - 1]
    def f(x):                                             # x: (N,1) -> (N,)
        xi = x[:, 0]
        idx = np.searchsorted(zs, xi)                    # interval 0..k-1
        return slopes[idx] * xi + intercepts[idx]
    return f


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x)


def train_eval(k, width, seed, n=400, epochs=4000, lr=0.01, lo=-2.5, hi=2.5):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    f = make_jagged(k)
    xtr = rng.uniform(lo, hi, (n, 1)).astype(np.float32)
    xte = rng.uniform(lo, hi, (n, 1)).astype(np.float32)
    ytr = torch.from_numpy(f(xtr).astype(np.float32)).unsqueeze(1)
    yte = torch.from_numpy(f(xte).astype(np.float32)).unsqueeze(1)
    xtr, xte = torch.from_numpy(xtr), torch.from_numpy(xte)
    model = MLP(width)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); lossf(model(xtr), ytr).backward(); opt.step()
    with torch.no_grad():
        return lossf(model(xte), yte).item() / max(yte.var(unbiased=False).item(), 1e-12)


def main():
    print("### C-D2 v3: JAGGED (non-convex) target -- threshold vs tropical complexity k ###")
    ks = (2, 3, 4, 6, 8)
    seeds = (0, 1, 2)
    bar = 0.01
    ths = []
    for k in ks:
        wmax = 2 * k + 3
        row, thr = [], None
        for w in range(1, wmax + 1):
            med = float(np.median([train_eval(k, w, s) for s in seeds]))
            row.append((w, med))
            if thr is None and med < bar:
                thr = w
        ths.append(thr)
        print(f"k={k:2d}  threshold(rel<{bar})= {thr}  (predict ~{k-1}) | "
              + "  ".join(f"w{w}:{m:.3f}" for w, m in row))
    print(f"\nk        = {list(ks)}")
    print(f"threshold= {ths}   (predicted ~ k-1 = {[k-1 for k in ks]})")
    if all(t is not None for t in ths):
        # tracking: monotone non-decreasing AND last threshold within [k-2, k] band
        tracks = all(ths[i] <= ths[i + 1] for i in range(len(ths) - 1)) and ths[-1] >= ks[-1] - 2
        far_below = ths[-1] <= max(2, ks[-1] // 3)
        if tracks:
            print("-> SUPPORT: threshold tracks tropical complexity k (~ k-1) for essential, "
                  "non-approximable pieces.")
        elif far_below:
            print("-> GROKS_FAR_BELOW_COMPILED_SIZE: generalizes far below k even for a jagged target.")
        else:
            print("-> PARTIAL: threshold rises with k but sub-linearly (between exact-size and far-below).")
    else:
        print("-> INCONCLUSIVE: some k never cleared the bar.")


if __name__ == "__main__":
    main()
