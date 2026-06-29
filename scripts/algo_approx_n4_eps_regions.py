"""
N-4 — the eps-essential region count as the learnable invariant (algorithmic-approximation slate 2).

C-D2 found that the *exact* tropical piece count k does NOT predict a ReLU net's
generalization-width threshold, for two confounded reasons (exact-vs-eps complexity;
existence-vs-trainability). N-4 tests the redemption: the **eps-essential** region count —
the minimum number of linear pieces needed to approximate the target within the same bar the
threshold uses — IS the predictor, and it moves with eps while exact k does not.

Two upgrades over C-D2:
  (1) eps-essential count via OPTIMAL piecewise-linear segmentation (DP), at each variance bar.
  (2) a fitting ORACLE: best-of-R restarts per width, so the measured threshold reflects
      representational capacity, not a single SGD run's luck (the C-D2 jagged confound).

Prediction (redemption): threshold(bar) ~ eps_essential(bar) - 1, tracking across all three
target families (smooth / essential / jagged) and across bars; exact k is the bar->0 limit.
Falsifier EPSILON_REGIONS_ALSO_FAIL: threshold tracks eps_essential no better than exact k.

Deterministic (seeded). CPU. Prints tables + a verdict line.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
LO, HI = -2.5, 2.5          # sampling range (matches C-D2)


# ----- target families (k-piece 1-D PL functions) -------------------------------------------
def target_fn(kind: str, k: int):
    """Return a callable f(x:(N,) ndarray)->(N,) for the chosen k-piece PL family."""
    if kind == "smooth":                       # tangent lines to 0.5 x^2; jumps shrink with k
        ts = np.linspace(-2.0, 2.0, k)
        s, b = ts, -0.5 * ts ** 2
        return lambda x: (x[:, None] * s[None, :] + b[None, :]).max(axis=1)
    if kind == "essential":                    # convex, fixed slope jumps (=1): every piece sharp
        zs = np.linspace(-2.0, 2.0, k + 1)[1:-1]
        slopes = np.arange(k) - (k - 1) / 2.0
        inter = np.zeros(k)
        for i in range(1, k):
            inter[i] = inter[i - 1] + (slopes[i - 1] - slopes[i]) * zs[i - 1]
        return lambda x: (x[:, None] * slopes[None, :] + inter[None, :]).max(axis=1)
    if kind == "jagged":                       # non-convex, random slopes: eps-complexity ~ k
        rng = np.random.default_rng(1234 + 7 * k)
        zs = np.linspace(-2.0, 2.0, k + 1)[1:-1]
        slopes = rng.uniform(-2.0, 2.0, size=k)
        inter = np.zeros(k)
        for i in range(1, k):
            inter[i] = inter[i - 1] + (slopes[i - 1] - slopes[i]) * zs[i - 1]
        def f(x):
            idx = np.searchsorted(zs, x)
            return slopes[idx] * x + inter[idx]
        return f
    raise ValueError(kind)


# ----- eps-essential count: min #segments for an optimal PL fit within the variance bar ------
def _seg_sse(ps_n, ps_x, ps_y, ps_xx, ps_xy, ps_yy, i, j):
    """Least-squares SSE of a single line through sample points i..j (inclusive)."""
    n = ps_n[j + 1] - ps_n[i]
    sx = ps_x[j + 1] - ps_x[i]; sy = ps_y[j + 1] - ps_y[i]
    sxx = ps_xx[j + 1] - ps_xx[i]; sxy = ps_xy[j + 1] - ps_xy[i]; syy = ps_yy[j + 1] - ps_yy[i]
    if n <= 1:
        return 0.0
    cxx = sxx - sx * sx / n              # centered moments
    cxy = sxy - sx * sy / n
    cyy = syy - sy * sy / n
    if cxx <= 1e-12:                     # all x equal -> best fit is the mean
        return max(cyy, 0.0)
    return max(cyy - cxy * cxy / cxx, 0.0)


def eps_essential_pieces(f, bar: float, ngrid: int = 201, mmax: int = 14) -> int:
    """Smallest m such that the optimal m-segment PL fit leaves frac-variance-unexplained < bar."""
    xs = np.linspace(LO, HI, ngrid)
    ys = f(xs)
    tot = float(((ys - ys.mean()) ** 2).sum())
    if tot <= 1e-12:
        return 1
    ps_n = np.arange(ngrid + 1, dtype=float)
    ps_x = np.concatenate([[0.0], np.cumsum(xs)])
    ps_y = np.concatenate([[0.0], np.cumsum(ys)])
    ps_xx = np.concatenate([[0.0], np.cumsum(xs * xs)])
    ps_xy = np.concatenate([[0.0], np.cumsum(xs * ys)])
    ps_yy = np.concatenate([[0.0], np.cumsum(ys * ys)])
    INF = float("inf")
    # dp[j] after m segments = min SSE fitting points 0..j with m segments
    dp = [_seg_sse(ps_n, ps_x, ps_y, ps_xx, ps_xy, ps_yy, 0, j) for j in range(ngrid)]
    if dp[ngrid - 1] / tot < bar:
        return 1
    for m in range(2, mmax + 1):
        ndp = [INF] * ngrid
        for j in range(ngrid):
            best = INF
            for i in range(0, j):
                cand = dp[i] + _seg_sse(ps_n, ps_x, ps_y, ps_xx, ps_xy, ps_yy, i + 1, j)
                if cand < best:
                    best = cand
            ndp[j] = best
        dp = ndp
        if dp[ngrid - 1] / tot < bar:
            return m
    return mmax + 1


# ----- net fitting oracle: best-of-R restarts per width --------------------------------------
class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x)


def _fit_once(f, width, seed, n=300, epochs=1500, lr=0.01):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xtr = rng.uniform(LO, HI, (n, 1)).astype(np.float32)
    xte = rng.uniform(LO, HI, (n, 1)).astype(np.float32)
    ytr = torch.from_numpy(f(xtr[:, 0]).astype(np.float32)).unsqueeze(1)
    yte = torch.from_numpy(f(xte[:, 0]).astype(np.float32)).unsqueeze(1)
    xtr_t, xte_t = torch.from_numpy(xtr), torch.from_numpy(xte)
    model = MLP(width)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); lossf(model(xtr_t), ytr).backward(); opt.step()
    with torch.no_grad():
        return lossf(model(xte_t), yte).item() / max(yte.var(unbiased=False).item(), 1e-12)


def oracle_fracvar_by_width(f, wmax, restarts=5):
    """best-of-restarts held-out frac-variance-unexplained for each width 1..wmax (the oracle)."""
    return {w: min(_fit_once(f, w, s) for s in range(restarts)) for w in range(1, wmax + 1)}


def threshold_at(fracvar_by_w, bar):
    for w in sorted(fracvar_by_w):
        if fracvar_by_w[w] < bar:
            return w
    return None


# ----- experiment ---------------------------------------------------------------------------
def main():
    families = ["smooth", "essential", "jagged"]
    ks = [3, 5, 8]
    bars = [0.10, 0.05, 0.02, 0.01]
    restarts = 6
    print("###### N-4: eps-essential region count as the learnable invariant ######")
    print("legend: th=oracle net threshold(width); ee=eps-essential pieces; predict th ~ ee-1\n")

    cells = []   # (family, k, bar, threshold, eps_ess)
    for kind in families:
        print(f"=== family: {kind} ===")
        for k in ks:
            f = target_fn(kind, k)
            wmax = 2 * k + 3
            fv = oracle_fracvar_by_width(f, wmax, restarts=restarts)
            parts = []
            for bar in bars:
                th = threshold_at(fv, bar)
                ee = eps_essential_pieces(f, bar)
                cells.append((kind, k, bar, th, ee))
                parts.append(f"bar={bar:>4}: th={str(th):>4} ee={ee:>2}(->{ee-1})")
            print(f"  k={k}  exact_k={k} (k-1={k-1})  | " + "  | ".join(parts))
        print()

    # ---- scoring: does threshold track (eps_essential - 1) better than exact (k - 1)? ----
    ok = [(th, ee, k) for (_, k, _, th, ee) in cells if th is not None]
    hit_ee = sum(1 for th, ee, k in ok if abs(th - (ee - 1)) <= 1)
    hit_k = sum(1 for th, ee, k in ok if abs(th - (k - 1)) <= 1)
    n = len(ok)
    # does eps-essential MOVE with the bar (vs constant exact k)? average spread per (family,k)
    spreads = []
    for kind in families:
        for k in ks:
            ees = [ee for (fk, kk, _, _, ee) in cells if fk == kind and kk == k]
            spreads.append(max(ees) - min(ees))
    avg_spread = float(np.mean(spreads))

    print("--- VERDICT (N-4) ---")
    print(f"  cells with a threshold: {n}/{len(cells)}")
    print(f"  threshold within +/-1 of (eps_essential - 1): {hit_ee}/{n} = {hit_ee/n:.2f}")
    print(f"  threshold within +/-1 of (exact k - 1):        {hit_k}/{n} = {hit_k/n:.2f}")
    print(f"  eps-essential spread across bars (per target, mean): {avg_spread:.2f}  "
          f"(exact k spread = 0 by definition)")
    per_fam = {}
    for kind in families:
        sub = [(th, ee) for (fk, _, _, th, ee) in cells if fk == kind and th is not None]
        if sub:
            per_fam[kind] = sum(1 for th, ee in sub if abs(th - (ee - 1)) <= 1) / len(sub)
    print(f"  per-family eps-tracking rate: " + ", ".join(f"{k}:{v:.2f}" for k, v in per_fam.items()))
    if hit_ee >= 0.7 * n and hit_ee > hit_k and avg_spread >= 1.0:
        print("  -> REDEEMS: the eps-essential region count tracks the threshold (and moves with")
        print("     eps), where the exact piece count does not. Region geometry is the learnable")
        print("     invariant once measured at the operative scale.")
    elif hit_ee > hit_k:
        print("  -> PARTIAL: eps-essential tracks better than exact k, but not cleanly across all")
        print("     families (likely a residual trainability gap on the non-convex family).")
    else:
        print("  -> EPSILON_REGIONS_ALSO_FAIL: eps-essential tracks no better than exact k; region")
        print("     geometry, smoothed or not, is not the learnable invariant.")


if __name__ == "__main__":
    main()
